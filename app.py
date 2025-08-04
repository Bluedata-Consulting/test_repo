import os
import logging
import time
import traceback
import re
import multiprocessing as mp
from queue import Empty
from typing import Dict
import torch
import json
import asyncio
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pydub.exceptions import PydubException

# FastAPI imports
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import librosa
import soundfile as sf
from faster_whisper import WhisperModel
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableLambda
from langchain_core.tools import tool
from TTS.api import TTS

import uvicorn

# --- Local Imports ---
from workers import video_sync_worker
from auth import authenticate_user # Retained as requested

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI app initialization ---
app = FastAPI(title="Real-Time Avatar", version="1.0.0")

# --- Session Middleware ---
app.add_middleware(SessionMiddleware, secret_key="your-super-secret-key")

# --- Static Files and Templates ---
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory="downloaded_media"), name="media")

# --- Directories ---
TEMP_DIR = "./temp"
DOWNLOAD_DIR = "downloaded_media"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- Global In-Memory Session Storage ---
user_sessions: Dict[str, Dict] = {}

# --- Device Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# --- Define paths ---
INPUT_PATH = os.path.join(TEMP_DIR, "input.wav")
OUTPUT_PATH = os.path.join(TEMP_DIR, "output.wav")
# Fixed: Use relative path for default speaker WAV
DEFAULT_SPEAKER_WAV = os.path.join(DOWNLOAD_DIR, "1749534228.wav")

# Ensure default speaker WAV exists
if not os.path.exists(DEFAULT_SPEAKER_WAV):
    try:
        AudioSegment.silent(duration=1000).export(DEFAULT_SPEAKER_WAV, format="wav")
        logging.warning(f"Default speaker WAV not found. Created a dummy at: {DEFAULT_SPEAKER_WAV}")
    except Exception as e:
        logging.error(f"Failed to create default speaker WAV: {e}")

# --- Load Faster Whisper Model Globally ---
whisper_model = None
try:
    print("Loading Whisper model...")
    whisper_model = WhisperModel("tiny", device=device, compute_type="int8")
    logging.info(f"Faster Whisper model 'tiny' loaded successfully on {device}.")
except Exception as e:
    logging.error(f"Could not load Faster Whisper model: {e}", exc_info=True)
    whisper_model = None

# --- Load TTS Model Globally ---
tts_model = None
try:
    print("Loading TTS model...")
    tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
    logging.info("TTS model loaded successfully.")
except Exception as e:
    logging.error(f"Could not load TTS model: {e}", exc_info=True)
    tts_model = None

# --- STT function ---
def stt(input_audio_path):
    if not whisper_model:
        logging.error("Whisper model not loaded.")
        return None
    
    # Check if file exists
    if not os.path.exists(input_audio_path):
        logging.error(f"Audio file not found: {input_audio_path}")
        return None
    
    # Check file size
    file_size = os.path.getsize(input_audio_path)
    if file_size == 0:
        logging.error(f"Audio file is empty: {input_audio_path}")
        return None
    
    logging.info(f"Processing audio file: {input_audio_path}, size: {file_size} bytes")
    
    try:
        # Convert audio format if needed
        if input_audio_path.endswith('.webm'):
            # Convert webm to wav first
            wav_path = input_audio_path.replace('.webm', '.wav')
            try:
                audio = AudioSegment.from_file(input_audio_path)
                audio.export(wav_path, format="wav")
                input_audio_path = wav_path
                logging.info(f"Converted webm to wav: {wav_path}")
            except Exception as e:
                logging.error(f"Error converting webm to wav: {e}")
                return None
        
        # Load and process audio
        y, sr = librosa.load(input_audio_path, sr=16000)
        logging.info(f"Audio loaded: shape={y.shape}, sr={sr}")
        
        # Check if audio has content
        if len(y) == 0:
            logging.error("Audio array is empty after loading")
            return None
        
        # Check audio duration
        duration = len(y) / sr
        logging.info(f"Audio duration: {duration:.2f} seconds")
        
        if duration < 0.1:  # Less than 100ms
            logging.error("Audio too short for transcription")
            return None
        
        # Transcribe
        segments, info = whisper_model.transcribe(y)
        full_text = " ".join([segment.text for segment in segments])
        
        logging.info(f"Transcription result: '{full_text}'")
        
        # Clean up temporary wav file if created
        if input_audio_path.endswith('.wav') and 'wav_path' in locals():
            try:
                os.remove(wav_path)
            except:
                pass
        
        return full_text.strip() if full_text.strip() else None
        
    except Exception as e:
        logging.error(f"Error during audio transcription: {e}", exc_info=True)
        return None

# --- LLM and Langchain setup ---
llm_chain = None
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    print("Setting up Groq LLM...")
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
    
    # Prompt Template
    prompt_template = [
        ('system', 'You are a helpful assistant, your name is BlueAssistant. Keep your responses to a maximum of three sentences.'),
        ('user', '{input}')
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template)
    parser = StrOutputParser()
    
    # Define the runnable for STT
    runnable_stt = RunnableLambda(stt)

    # The full LLM chain: STT -> Prompt -> LLM -> Parser
    llm_chain = runnable_stt | prompt | llm | parser
    logging.info("Groq LLM and Langchain chain initialized successfully.")
except (ImportError, ValueError) as e:
    logging.error(f"Could not initialize LLM: {e}", exc_info=True)
    llm_chain = None

# --- TTS tool ---
@tool
def text_to_speech(text: str, speaker_wav_path: str = DEFAULT_SPEAKER_WAV) -> str:
    """Convert text to cloned speech using a reference speaker."""
    if not tts_model:
        logging.error("TTS model not loaded.")
        return "TTS model not available"
    
    try:
        output_filename = f"tts_output_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language="en",
            file_path=output_path
        )
        return output_path
    except Exception as e:
        logging.error(f"Error during TTS generation: {e}", exc_info=True)
        return None

# --- TTS Worker Process ---
def tts_worker_process(tts_queue: mp.Queue, audio_output_queue: mp.Queue, stop_event: mp.Event, speaker_wav_path: str):
    logging.info(f"TTS worker process started with speaker: {speaker_wav_path}")
    local_tts_model = None
    try:
        # Initialize TTS model within the worker process
        local_tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
        logging.info("TTS model loaded successfully in worker process.")
    except Exception as e:
        logging.error(f"Failed to load TTS model in worker: {e}", exc_info=True)
        stop_event.set()
        return

    while not stop_event.is_set():
        try:
            item = tts_queue.get(timeout=1)
            if item == "__END_OF_RESPONSE__":
                logging.info("TTS worker received end of response signal.")
                audio_output_queue.put("__END_OF_RESPONSE__")
                continue

            text_to_speak = item

            if not local_tts_model:
                logging.error("TTS model not initialized in worker. Skipping text.")
                continue

            try:
                # Generate a unique filename for each audio chunk
                output_audio_filename = f"tts_output_{int(time.time() * 1000)}_{os.getpid()}.wav"
                output_audio_path = os.path.join(TEMP_DIR, output_audio_filename)

                local_tts_model.tts_to_file(
                    text=text_to_speak,
                    speaker_wav=speaker_wav_path,
                    language="en",
                    file_path=output_audio_path
                )
                
                # Verify the file was created successfully
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    logging.info(f"Generated TTS audio: {output_audio_path}")
                    audio_output_queue.put(output_audio_path)
                else:
                    logging.error(f"TTS audio file not created or empty: {output_audio_path}")
                    
            except Exception as e:
                logging.error(f"Error during TTS generation in worker: {e}", exc_info=True)

        except Empty:
            continue
        except Exception as e:
            logging.error(f"Unexpected error in TTS worker: {e}", exc_info=True)
            break
    logging.info("TTS worker process stopped.")

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logging.info(f"WebSocket connection established for user_id: {user_id}")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        logging.info(f"WebSocket connection closed for user_id: {user_id}")

    async def send_json(self, data: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(data)
            except WebSocketDisconnect:
                self.disconnect(user_id)
            except Exception as e:
                logging.error(f"Error sending JSON to user {user_id}: {e}", exc_info=True)

manager = ConnectionManager()

# --- Dependency for checking authentication ---
async def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return user_id

# --- Avatar and Session Management ---
def get_avatars_from_local():
    avatars_list = []
    for filename in os.listdir(DOWNLOAD_DIR):
        if filename.endswith(('.mp4', '.mov', '.avi')):
            avatar_id = os.path.splitext(filename)[0]
            video_path = os.path.join(DOWNLOAD_DIR, filename)
            audio_path = os.path.join(DOWNLOAD_DIR, f"{avatar_id}.wav")
            image_path = os.path.join(DOWNLOAD_DIR, f"{avatar_id}.png")

            if os.path.exists(video_path) and os.path.exists(audio_path):
                avatars_list.append({
                    'id': avatar_id,
                    'video_url': f"/media/{filename}",
                    'speaker_wav_path': audio_path,
                    'image_url': f"/media/{avatar_id}.png" if os.path.exists(image_path) else "/static/default_avatar.png"
                })
    return avatars_list

def cleanup_session(user_id: str):
    if user_id not in user_sessions:
        return
    logging.info(f"Cleaning up session for user_id: {user_id}")
    session_data = user_sessions[user_id]
    stop_event = session_data.get('stop_event')
    if stop_event:
        stop_event.set()
    
    # Give workers a moment to process stop signal before terminating
    time.sleep(0.1)

    processes = session_data.get('processes', {})
    for name, process in processes.items():
        if process and process.is_alive():
            logging.info(f"Terminating worker process: {name}")
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                logging.warning(f"Worker process {name} did not terminate gracefully.")
    
    if session_data.get('monitor_task') and not session_data['monitor_task'].done():
        session_data['monitor_task'].cancel()
        logging.info(f"Cancelled monitor_task for user_id: {user_id}")

    del user_sessions[user_id]
    logging.info(f"Session for user_id: {user_id} cleaned up.")

def start_worker_processes(user_id: str, avatar_data: Dict):
    cleanup_session(user_id)

    tts_queue = mp.Queue()
    audio_queue = mp.Queue()
    video_queue = mp.Queue()
    stop_event = mp.Event()

    speaker_wav_path = avatar_data.get('speaker_wav_path')
    if not speaker_wav_path or not os.path.exists(speaker_wav_path):
        logging.error(f"Speaker WAV file not found: {speaker_wav_path}. Using default.")
        speaker_wav_path = DEFAULT_SPEAKER_WAV

    user_sessions[user_id] = {
        'tts_queue': tts_queue, 'audio_queue': audio_queue, 'video_queue': video_queue,
        'stop_event': stop_event, 'processes': {}, 'selected_avatar': avatar_data,
        'monitor_task': None
    }

    tts_process = mp.Process(target=tts_worker_process, args=(tts_queue, audio_queue, stop_event, speaker_wav_path))
    tts_process.daemon = True
    tts_process.start()
    user_sessions[user_id]['processes']['tts'] = tts_process
    logging.info(f"TTS worker process started for user {user_id}.")
    
    # Video Sync Worker Process - Fixed path handling
    video_file_path_for_worker = avatar_data['video_url'].replace("/media/", f"{DOWNLOAD_DIR}/")
    
    # Ensure the video file exists
    if not os.path.exists(video_file_path_for_worker):
        logging.error(f"Video file not found: {video_file_path_for_worker}")
        return False
    
    # Pass the temp directory to the worker so it knows where to create files
    video_process = mp.Process(
        target=video_sync_worker, 
        args=(audio_queue, video_queue, video_file_path_for_worker, stop_event, TEMP_DIR)
    )
    video_process.daemon = True
    video_process.start()
    user_sessions[user_id]['processes']['video'] = video_process
    logging.info(f"Video sync worker process started for user {user_id}.")
    
    # Monitor task for video queue
    user_sessions[user_id]['monitor_task'] = asyncio.create_task(monitor_video_queue(user_id))
    logging.info(f"Video queue monitor task started for user {user_id}.")
    return True

# --- Core Logic ---
async def process_llm_and_tts(user_id: str, audio_filepath: str):
    if not llm_chain:
        await manager.send_json({'type': 'error', 'message': 'LLM not configured.'}, user_id)
        return
    if user_id not in user_sessions or 'tts_queue' not in user_sessions[user_id]:
        await manager.send_json({'type': 'error', 'message': 'Session not ready for TTS.'}, user_id)
        return

    tts_queue = user_sessions[user_id]['tts_queue']
    logging.info(f"Processing audio for LLM and TTS for user {user_id}: '{audio_filepath}'")
    
    try:
        # First, get the transcription
        transcription = stt(audio_filepath)
        
        # Clean up the audio file after transcription
        try:
            if os.path.exists(audio_filepath):
                os.remove(audio_filepath)
                logging.info(f"Cleaned up input audio file: {audio_filepath}")
        except Exception as e:
            logging.warning(f"Could not clean up audio file {audio_filepath}: {e}")
        
        if not transcription:
            await manager.send_json({'type': 'error', 'message': 'Failed to transcribe audio. Please try speaking longer or check your microphone.'}, user_id)
            return
        
        await manager.send_json({'type': 'transcription', 'text': transcription}, user_id)
        
        full_response = ""
        sentence_enders = re.compile(r'([.!?])\s*')
        
        # Create a prompt with the transcription
        prompt_input = {"input": transcription}
        
        # Get LLM response (skip the STT part of the chain since we already have transcription)
        llm_only_chain = prompt | llm | parser
        
        # Now, invoke the LLM chain with the transcription
        async for chunk_text in llm_only_chain.astream(prompt_input):
            if chunk_text:
                full_response += chunk_text
                await manager.send_json({'type': 'llm_chunk', 'text': chunk_text}, user_id)
                
                match = sentence_enders.search(full_response)
                while match:
                    sentence_end_idx = match.end()
                    sentence_to_speak = full_response[:sentence_end_idx].strip()
                    if sentence_to_speak:
                        tts_queue.put(sentence_to_speak)
                        logging.debug(f"Sent to TTS queue: '{sentence_to_speak}'")
                    full_response = full_response[sentence_end_idx:]
                    match = sentence_enders.search(full_response)

        if full_response.strip():
            tts_queue.put(full_response.strip())
            logging.debug(f"Sent remaining to TTS queue: '{full_response.strip()}'")
        
        tts_queue.put("__END_OF_RESPONSE__")
        await manager.send_json({'type': 'llm_end'}, user_id)
        
    except Exception as e:
        logging.error(f"LLM/TTS processing error for user {user_id}: {e}", exc_info=True)
        await manager.send_json({'type': 'error', 'message': f'Processing error: {str(e)}'}, user_id)
        
        # Clean up the audio file in case of error
        try:
            if os.path.exists(audio_filepath):
                os.remove(audio_filepath)
        except:
            pass

async def monitor_video_queue(user_id: str):
    if user_id not in user_sessions: 
        return
    video_queue = user_sessions[user_id]['video_queue']
    audio_queue = user_sessions[user_id]['audio_queue']
    stop_event = user_sessions[user_id]['stop_event']
    
    while not stop_event.is_set():
        try:
            # First, check for new TTS audio files
            try:
                audio_path = audio_queue.get_nowait()
                if audio_path == "__END_OF_RESPONSE__":
                    logging.info(f"Received __END_OF_RESPONSE__ from TTS worker for user {user_id}")
                    await manager.send_json({'type': 'tts_end'}, user_id)
                    continue

                if audio_path and os.path.exists(audio_path):
                    filename = os.path.basename(audio_path)
                    temp_audio_url = f"/temp/{filename}"
                    await manager.send_json({'type': 'audio_chunk', 'url': temp_audio_url}, user_id)
                    logging.debug(f"Sent audio chunk URL to client: {temp_audio_url}")
                    
                    # Don't clean up immediately - let the client access it first
                    # Schedule cleanup after a delay
                    asyncio.create_task(delayed_cleanup(audio_path, 180))  # Increased delay to 180 seconds
                    
            except Empty:
                pass

            # Then, check for new video chunks
            try:
                video_path = video_queue.get_nowait()
                if video_path and os.path.exists(video_path):
                    filename = os.path.basename(video_path)
                    temp_video_url = f"/temp/{filename}"
                    await manager.send_json({'type': 'video_chunk', 'url': temp_video_url}, user_id)
                    logging.debug(f"Sent video chunk URL to client: {temp_video_url}")
                    
                    # Don't clean up immediately - let the client access it first
                    # Schedule cleanup after a delay
                    asyncio.create_task(delayed_cleanup(video_path, 180))  # Increased delay to 180 seconds
                    
            except Empty:
                await asyncio.sleep(0.05)
        except Exception as e:
            logging.error(f"Queue monitor error for user {user_id}: {e}", exc_info=True)
            break
    logging.info(f"Queue monitor stopped for user {user_id}.")

async def delayed_cleanup(file_path: str, delay: int):
    """Clean up a file after a delay to allow client access."""
    await asyncio.sleep(delay)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.debug(f"Cleaned up file after delay: {file_path}")
    except Exception as e:
        logging.warning(f"Failed to clean up file {file_path}: {e}")

# --- FastAPI Routes ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, user_id: str = Depends(get_current_user)):
    if not user_id:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("layout.html", {"request": request, "user_id": user_id})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_form(request: Request, username: str = Form(...), password: str = Form(...)):
    user_id = authenticate_user(username, password) # Using authenticate_user from auth.py
    if user_id:
        request.session["user_id"] = user_id
        return RedirectResponse(url="/", status_code=303)
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})

@app.get("/logout")
async def logout(request: Request):
    user_id = request.session.pop("user_id", None)
    if user_id:
        cleanup_session(str(user_id))
    return RedirectResponse(url="/login")

@app.get("/api/avatars", dependencies=[Depends(get_current_user)])
async def get_all_avatars():
    return JSONResponse(content=get_avatars_from_local())

# Mount temp directory for serving temporary files
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            message = await websocket.receive()
            if 'text' in message:
                data = json.loads(message['text'])
                event_type = data.get("type")
                if event_type == "select_avatar":
                    avatar_id = data.get("avatar_id")
                    avatars = get_avatars_from_local()
                    selected_avatar = next((av for av in avatars if av['id'] == avatar_id), None)
                    if selected_avatar:
                        if start_worker_processes(user_id, selected_avatar):
                           await manager.send_json({'type': 'avatar_selected', 'success': True, 'avatar': selected_avatar}, user_id)
                        else:
                           await manager.send_json({'type': 'error', 'message': 'Failed to start workers.'}, user_id)
            elif 'bytes' in message:
                audio_bytes = message['bytes']
                
                # Check if audio data is empty
                if not audio_bytes or len(audio_bytes) == 0:
                    logging.error("Received empty audio data")
                    await manager.send_json({'type': 'error', 'message': 'No audio data received'}, user_id)
                    continue
                
                filename = f"user_audio_{user_id}_{int(time.time())}.webm"
                filepath = os.path.join(TEMP_DIR, filename)

                try:
                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)
                    logging.info(f"Received audio chunk and saved to {filepath}")
                    
                    asyncio.create_task(process_llm_and_tts(user_id, filepath))
                    
                except Exception as e:
                    logging.error(f"Error processing audio bytes: {e}", exc_info=True)
                    await manager.send_json({'type': 'error', 'message': f'Error processing audio: {str(e)}'}, user_id)
                    
                    # Clean up file if it exists
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except:
                            pass

    except WebSocketDisconnect:
        logging.info(f"Client {user_id} disconnected.")
    except Exception as e:
        logging.error(f"WebSocket processing error for user {user_id}: {e}", exc_info=True)
        try:
            await manager.send_json({'type': 'error', 'message': f"Server error: {str(e)}"}, user_id)
        except:
            pass
    finally:
        cleanup_session(user_id)
        manager.disconnect(user_id)

# --- Main Execution ---
if __name__ == "__main__":
    # Clean up old temp files on startup
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, f))
        except OSError as e:
            logging.warning(f"Error removing old temp file {f}: {e}")
    
    # Set start method for multiprocessing (important for macOS/Windows)
    mp.set_start_method('spawn', force=True)
    uvicorn.run(app, host="0.0.0.0", port=8506)