import os
import time
import uuid
import logging
import json
from typing import List, Dict, Optional
import difflib
import asyncio
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .stt import transcribe_file, _get_model
from .llm import get_llm_response
from .tts import synthesize_speech


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, os.pardir, os.pardir))
print(PROJECT_ROOT)
DOWNLOAD_DIR = "/home/jetson/Avatar_react/backend/downloaded_media"
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
CONFIG_DIR = os.path.join(APP_ROOT, "config")
SCHEDULE_FILE = os.path.join(APP_ROOT, "schedules.json")


os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Edge Avatar Assistant", version="0.2.0")

# CORS - allow local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
app.mount("/media", StaticFiles(directory=DOWNLOAD_DIR), name="media")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

# Support multiple wake words via env var, fallback to single default
WAKE_WORDS = [w.strip().lower() for w in os.getenv("WAKE_WORDS", "agent").split(",") if w.strip()]

# Readiness flag
READY_STT = False
_AVATARS: Dict[str, Dict] = {}

schedule_lock = asyncio.Lock()
event_queues: Dict[str, asyncio.Queue] = {}
client_states: Dict[str, Dict] = {} # New client state management

class Schedule(BaseModel):
    id: str
    time: str
    prompt: str
    enabled: bool
    user_id: str # ----------> client_id
    avatar_id: str

async def schedule_checker():
    while True:
        await asyncio.sleep(20)
        now = datetime.now()
        current_time = now.strftime("%H:%M")

        if not os.path.exists(SCHEDULE_FILE):
            continue

        due_schedules = []
        remaining_schedules = []

        async with schedule_lock:
            try:
                with open(SCHEDULE_FILE, "r+") as f:
                    try:
                        schedules = json.load(f)
                    except json.JSONDecodeError:
                        schedules = []

                    for schedule in schedules:
                        if schedule.get("enabled") and schedule.get("time") == current_time:
                            due_schedules.append(schedule)
                        else:
                            remaining_schedules.append(schedule)

                    if due_schedules:
                        f.seek(0)
                        f.truncate()
                        json.dump(remaining_schedules, f, indent=4)
            except Exception as e:
                logging.error(f"Error processing schedules file: {e}")
                continue

        for schedule in due_schedules:
            client_id = schedule.get("user_id")

            # Check if schedules are enabled for this client
            if client_id in client_states and not client_states[client_id].get("schedules_enabled", True):
                logging.info(f"Schedules are disabled for client {client_id}. Skipping.")
                continue

            avatar_id = schedule.get("avatar_id")
            prompt = schedule.get("prompt")
            logging.info(f"Executing schedule for client {client_id} with prompt: {prompt}")

            avatars = _get_avatars()
            avatar = avatars.get(avatar_id.lower())
            if not avatar:
                logging.error(f"Avatar '{avatar_id}' for scheduled task not found.")
                continue

            # LLM and TTS
            reply = get_llm_response(prompt, avatar)
            if not reply:
                reply = "I couldn't think of a response."
            
            language = avatar.get("language", "en")
            speaker_wav = avatar.get("voice")
            
            tts_language = language
            if language == "fr":
                tts_language = "fr-fr"

            audio_path = synthesize_speech(reply, speaker_wav=speaker_wav, temp_dir=TEMP_DIR, language=tts_language)

            if not audio_path:
                logging.error(f"Failed to synthesize speech for scheduled task.")
                continue

            # Push event to the client's queue
            if client_id in event_queues:
                event_data = {
                    "type": "scheduled_event",
                    "audio_url": f"/temp/{os.path.basename(audio_path)}",
                    "avatar_id": avatar_id,
                    "response": reply,
                    "lip_sync_video": avatar.get("lip_sync_video")
                }
                await event_queues[client_id].put(event_data)
                logging.info(f"Pushed event to client {client_id}")


@app.get("/api/events/{client_id}")
async def event_stream(request: Request, client_id: str):
    logging.info(f"Client {client_id} connected to SSE stream.")
    queue = asyncio.Queue()
    event_queues[client_id] = queue

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    logging.info(f"Client {client_id} disconnected.")
                    break
                
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            if client_id in event_queues:
                del event_queues[client_id]
            logging.info(f"SSE stream for {client_id} closed.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _fuzzy_contains(text: str, target: str, threshold: float = 0.78) -> bool:
    """Return True if target appears in text allowing for small ASR errors.

    - Direct substring check
    - Compact (remove spaces) substring check
    - Fuzzy ratio of any sliding window of similar length
    """
    if not text or not target:
        return False
    text_norm = _normalize_text(text)
    target_norm = _normalize_text(target)

    # direct checks
    if target_norm in text_norm:
        return True
    if target_norm.replace(" ", "") in text_norm.replace(" ", ""):
        return True

    # fuzzy sliding window over words
    text_words = text_norm.split()
    target_words = target_norm.split()
    if not text_words or not target_words:
        return False

    window = max(1, len(target_words))
    for i in range(0, len(text_words) - window + 1):
        span = " ".join(text_words[i : i + window])
        ratio = difflib.SequenceMatcher(a=span, b=target_norm).ratio()
        if ratio >= threshold:
            return True
    return False


def _expand_variants(w: str) -> List[str]:
    """Common mishears for short wake words (tweak as needed)."""
    variants = {w}
    if w == "agent":
        variants.update({"agent", "hey agent", "asian", "agentt", "ag ent"})
    if w == "avatar":
        variants.update({"avatar", "a vatar", "avataa", "hey avatar"})
    return list(variants)


def _load_avatars_from_file() -> List[Dict]:
    """Loads avatar data from the JSON config file."""
    config_path = os.path.join(CONFIG_DIR, "avatars.json")
    if not os.path.exists(config_path):
        return []
    with open(config_path, "r") as f:
        data = json.load(f)
        avatars = data.get("avatars", [])
        for avatar in avatars:
            base_id = avatar.get("base_id")
            if base_id:
                avatar["image"] = f"/media/{base_id}.png"
                avatar["ideal_video"] = f"/media/{base_id}.mp4"
                avatar["lip_sync_video"] = f"/media/{base_id}_lip.mp4"
                avatar["voice"] = f"backend/downloaded_media/{base_id}.wav"
                avatar["media_type"] = "video"
        return avatars
def _get_avatars() -> Dict[str, Dict]:
    global _AVATARS
    if not _AVATARS:
        logging.info("Loading avatars from config file...")
        avatar_list = _load_avatars_from_file()
        _AVATARS = {a["id"]: a for a in avatar_list}
    return _AVATARS

@app.on_event("startup")
def _warmup_models():
    global READY_STT
    try:
        # Load Whisper into memory and run a tiny dummy pass to initialize kernels
        import numpy as _np
        model = _get_model()
        _ = model.transcribe(_np.zeros(int(0.25 * 16000), dtype=_np.float32))
        READY_STT = True
        logging.info("STT model warmed up and ready.")
        _ = _get_avatars() # Load avatars at startup
        logging.info("Avatars loaded at startup.")
        asyncio.create_task(schedule_checker())
        logging.info("Scheduler started.")
    except Exception as e:
        logging.exception("Failed to warm up STT: %s", e)
        READY_STT = False


@app.get("/api/ready")
def ready():
    return {"ready": READY_STT}


@app.get("/api/schedules")
async def get_schedules():
    if not os.path.exists(SCHEDULE_FILE):
        return []
    async with schedule_lock:
        with open(SCHEDULE_FILE, "r") as f:
            try:
                schedules = json.load(f)
            except json.JSONDecodeError:
                schedules = []
    return schedules

@app.post("/api/schedules")
async def create_schedule(schedule: Schedule):
    # Re-enable schedules for the client if they were previously disabled
    client_id = schedule.user_id
    if client_id not in client_states:
        client_states[client_id] = {}
    if not client_states[client_id].get("schedules_enabled", True):
        client_states[client_id]["schedules_enabled"] = True
        logging.info(f"Re-enabled schedules for client {client_id} due to new schedule creation.")

    async with schedule_lock:
        if not os.path.exists(SCHEDULE_FILE):
            with open(SCHEDULE_FILE, "w") as f:
                json.dump([], f)

        with open(SCHEDULE_FILE, "r+") as f:
            try:
                schedules = json.load(f)
            except json.JSONDecodeError:
                schedules = []
            schedules.append(schedule.dict())
            f.seek(0)
            f.truncate()
            json.dump(schedules, f, indent=4)

    return {"status": "ok", "message": "Schedule created"}

@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    if not os.path.exists(SCHEDULE_FILE):
        raise HTTPException(status_code=404, detail="Schedules file not found")

    async with schedule_lock:
        with open(SCHEDULE_FILE, "r+") as f:
            try:
                schedules = json.load(f)
            except json.JSONDecodeError:
                schedules = []

            new_schedules = [s for s in schedules if s.get("id") != schedule_id]

            if len(new_schedules) == len(schedules):
                raise HTTPException(status_code=404, detail="Schedule not found")

            f.seek(0)
            f.truncate()
            json.dump(new_schedules, f, indent=4)

    return {"status": "ok", "message": "Schedule deleted"}

@app.get("/api/avatars")
def list_avatars():
    return JSONResponse(content=list(_get_avatars().values()))


@app.get("/api/config")
def get_config():
    return {"wake_word": WAKE_WORDS[0] if WAKE_WORDS else "agent", "wake_words": WAKE_WORDS}


async def _stop_activity_for_client(client_id: str):
    """Helper to stop all client activity and disable schedules."""
    logging.info(f"Stopping activity for client {client_id}.")

    # Disable future schedules
    if client_id not in client_states:
        client_states[client_id] = {}
    client_states[client_id]["schedules_enabled"] = False
    logging.info(f"Disabled schedules for client {client_id}.")

    # Send stop event to client to halt current activity (e.g., lip sync)
    if client_id in event_queues:
        event_data = {"type": "stop_activity"}
        await event_queues[client_id].put(event_data)
        logging.info(f"Pushed stop_activity event to client {client_id}.")


@app.post("/api/wake")
async def wake_detect(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    file_ext = os.path.splitext(audio.filename)[1] or ".webm"
    input_path = os.path.join(TEMP_DIR, f"wake_{uuid.uuid4().hex}{file_ext}")
    with open(input_path, "wb") as f:
        f.write(await audio.read())
    try:
        text = transcribe_file(input_path)
        norm = _normalize_text(text or "")
        detected = False
        matched_word: Optional[str] = None
        for ww in WAKE_WORDS:
            candidates = _expand_variants(ww)
            for cand in candidates:
                if _fuzzy_contains(norm, cand):
                    detected = True
                    matched_word = ww
                    break
            if detected:
                break
        return {"detected": detected, "text": text, "matched": matched_word}
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass


class VadStoppedEvent(BaseModel):
    type: str
    reason: str
    timestamp: str
    sessionId: str

@app.post("/api/vad_stopped")
async def vad_stopped(event: VadStoppedEvent):
    """
    Endpoint to handle VAD stopped events from the frontend or other stop signals.

    This function triggers the server to halt all ongoing and future activities
    for a specific client session, including LLM, TTS, and lip-syncing, by
    sending a 'stop_activity' event and disabling scheduled tasks.
    """
    logging.info(f"Received VAD stopped event: {event.reason} for session {event.sessionId}")
    await _stop_activity_for_client(event.sessionId)
    return {"status": "ok", "message": "Stop signal processed."}



@app.post("/api/interact")
async def interact(avatar_id: str = Form(...), client_id: str = Form(...), audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    file_ext = os.path.splitext(audio.filename)[1] or ".webm"
    input_path = os.path.join(TEMP_DIR, f"user_{uuid.uuid4().hex}{file_ext}")
    with open(input_path, "wb") as f:
        f.write(await audio.read())

    try:
        # 1) STT
        avatars = _get_avatars()
        logging.info(f"Loaded avatars: {avatars.keys()}")
        avatar_id_lower = avatar_id.lower()
        avatar = avatars.get(avatar_id_lower)
        logging.info(f"Received avatar_id: {avatar_id}, Lowercase ID: {avatar_id_lower}, Found avatar: {avatar is not None}")
        if not avatar:
            raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found")

        language = avatar.get("language", "en")
        text = transcribe_file(input_path, language=language)
        if not text:
            raise HTTPException(status_code=422, detail="Could not transcribe audio")

        # Check for stop command
        normalized_text = _normalize_text(text)
        if normalized_text == "stop" or normalized_text == "please stop":
            logging.info(f"Received stop command from client {client_id}")
            await _stop_activity_for_client(client_id)
            return {"transcription": text, "response": "Stopping activity."}

        # Re-enable schedules on any other interaction
        if client_id not in client_states:
            client_states[client_id] = {}
        if not client_states[client_id].get("schedules_enabled", True):
             client_states[client_id]["schedules_enabled"] = True
             logging.info(f"Re-enabled schedules for client {client_id}")

        # 2) LLM
        reply = get_llm_response(text, avatar)
        if not reply:
            reply = "I'm here. How can I help you?"

        # 3) TTS (use avatar speaker ref if available)
        speaker_wav = avatar.get("voice")
        
        tts_language = language
        if language == "fr":
            tts_language = "fr-fr"

        audio_path = synthesize_speech(reply, speaker_wav=speaker_wav, temp_dir=TEMP_DIR, language=tts_language)
        if not audio_path:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")

        return {
            "transcription": text,
            "response": reply,
            "audio_url": f"/temp/{os.path.basename(audio_path)}",
        }
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8600, reload=False)