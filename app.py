
import os
import logging
import time
import traceback
import re
import multiprocessing as mp
from queue import Empty
from typing import Dict, List
import torch
import json
import asyncio
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pydub.exceptions import PydubException
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

# FastAPI imports
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Form, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import librosa
import soundfile as sf
from faster_whisper import WhisperModel
from TTS.api import TTS

# Langchain imports with Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence, Literal
import operator
from langchain_core.runnables import RunnableConfig

# Langfuse integration
from langfuse.langchain import CallbackHandler
from langfuse import observe

# Gemini imports for image processing
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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


# --- Schedule Storage (Persistent) ---
schedule_storage_file = "schedules.json"

def load_schedules():
    """Load schedules from persistent storage"""
    try:
        if os.path.exists(schedule_storage_file):
            with open(schedule_storage_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading schedules: {e}")
    return {}

def save_schedules(schedules):
    """Save schedules to persistent storage"""
    try:
        with open(schedule_storage_file, 'w') as f:
            json.dump(schedules, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving schedules: {e}")

# Load schedules at startup
persistent_schedules = load_schedules()

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
    whisper_model = WhisperModel("tiny", device='cpu')
    logging.info(f"Faster Whisper model 'tiny' loaded successfully on {device}.")
except Exception as e:
    logging.error(f"Could not load Faster Whisper model: {e}")
    whisper_model = None

# --- Load TTS Model Globally ---
tts_model = None
try:
    print("Loading TTS model...")
    tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
    logging.info("TTS model loaded successfully.")
except Exception as e:
    logging.error(f"Could not load TTS model: {e}")
    tts_model = None

# --- Load Gemini Model for Image Processing ---
gemini_model = None
try:

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyBi1WtYCB63ZNas6bYPG36uwiwraPZOqkI"
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        logging.info("Gemini model loaded successfully for image processing.")
    else:
        logging.warning("GEMINI_API_KEY not set. Image processing features will be disabled.")
except Exception as e:
    logging.error(f"Could not load Gemini model: {e}")
    gemini_model = None

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
        logging.error(f"Error during audio transcription: {e}")
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
        logging.error(f"Failed to load TTS model in worker: {e}")
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
                logging.error(f"Error during TTS generation in worker: {e}")

        except Empty:
            continue
        except Exception as e:
            logging.error(f"Unexpected error in TTS worker: {e}")
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
                logging.error(f"Error sending JSON to user {user_id}: {e}")

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
    user_sessions[user_id]['monitor_task'] = asyncio.create_task(monitor_queues(user_id))
    logging.info(f"Queue monitor task started for user {user_id}.")
    return True

# --- New helper: ensure workers are running for scheduled tasks ---
def ensure_user_workers(user_id: str, avatar_data: Dict | None = None) -> bool:
    """
    Ensure the session and worker processes (TTS + video) exist and are alive for the given user.
    If they are missing or dead, start them using provided avatar_data or by selecting a local avatar.
    Returns True if workers are running after the call, False otherwise.
    """
    try:
        sess = user_sessions.get(user_id)
        if sess:
            processes = sess.get('processes', {})
            tts_alive = 'tts' in processes and processes['tts'] and processes['tts'].is_alive()
            video_alive = 'video' in processes and processes['video'] and processes['video'].is_alive()

            # If TTS is alive and either we don't need a video or video is alive, we're good
            if tts_alive and (not sess.get('selected_avatar') or video_alive):
                logging.debug(f"Workers already running for user {user_id} (tts_alive={tts_alive}, video_alive={video_alive})")
                return True

            avatar_to_use = sess.get('selected_avatar') or avatar_data
        else:
            avatar_to_use = avatar_data

        if not avatar_to_use:
            avatars = get_avatars_from_local()
            if not avatars:
                logging.warning(f"ensure_user_workers: no avatars available to start for user {user_id}")
                return False
            avatar_to_use = avatars[0]

        logging.info(f"ensure_user_workers: starting workers for user {user_id} using avatar {avatar_to_use.get('id')}")
        started = start_worker_processes(user_id, avatar_to_use)
        if started:
            user_sessions.setdefault(user_id, {})['selected_avatar'] = avatar_to_use
            return True
        return False

    except Exception as e:
        logging.error(f"ensure_user_workers error for user {user_id}: {e}")
        return False

# --- Initialize TTS for Vision Mode (or any non-avatar session) ---
def initialize_default_tts(user_id: str):
    """Initialize a default TTS worker for sessions without a selected avatar."""
    if user_id in user_sessions:
        # Clean up existing session if any
        cleanup_session(user_id)
    
    tts_queue = mp.Queue()
    audio_queue = mp.Queue()
    stop_event = mp.Event()

    user_sessions[user_id] = {
        'tts_queue': tts_queue,
        'audio_queue': audio_queue,
        'video_queue': mp.Queue(), # Dummy queue
        'stop_event': stop_event,
        'processes': {},
        'monitor_task': None,
        'selected_avatar': None
    }

    # Start TTS worker with default speaker
    tts_process = mp.Process(target=tts_worker_process, args=(tts_queue, audio_queue, stop_event, DEFAULT_SPEAKER_WAV))
    tts_process.daemon = True
    tts_process.start()
    user_sessions[user_id]['processes']['tts'] = tts_process
    logging.info(f"Default TTS worker process started for user {user_id}.")

    # Monitor task for queues
    user_sessions[user_id]['monitor_task'] = asyncio.create_task(monitor_queues(user_id))
    logging.info(f"Queue monitor task started for default session user {user_id}.")

# --- LLM and Langchain setup with Google Gemini ---
llm = None
prompt = None
parser = None
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyBi1WtYCB63ZNas6bYPG36uwiwraPZOqkI" # Use env or fallback
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    print("Setting up Google Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
    
    # Prompt Template
    prompt_template = [
        ('system', 'You are a helpful assistant, your name is BlueAssistant. Keep your responses to a maximum of three sentences.'),
        ('user', '{input}')
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template)
    parser = StrOutputParser()
    
    logging.info("Google Gemini LLM and Langchain components initialized successfully.")
except (ImportError, ValueError) as e:
    logging.error(f"Could not initialize LLM: {e}")
    llm = None
    prompt = None
    parser = None

# --- LangGraph State Definition ---
class AvatarState(TypedDict):
    user_id: str
    mode: Literal['avatar', 'vision', 'schedule']
    
    # Inputs
    audio_filepath: str | None
    image_data: bytes | None
    text_input: str | None

    # STT Output
    transcription: str
    
    # Control flow flags
    require_trigger: bool
    contains_trigger: bool

    # LLM Output
    llm_response: str
    
    # Final output paths
    tts_audio_paths: Annotated[List[str], operator.add]
    
    # Completion flag
    response_complete: bool

# --- LangGraph Node Functions ---
async def stt_node(state: AvatarState) -> dict:
    """Transcribe audio or use provided text. Check for trigger word if required."""
    logging.info(f"STT Node: Processing for user {state['user_id']} in mode {state['mode']}")
    
    transcription = ""
    
    # If text is provided directly (schedule mode), use it
    if state.get('text_input'):
        transcription = state['text_input']
        logging.info(f"STT Node: Using provided text input: '{transcription}'")
    
    # If audio is provided, transcribe it
    elif state.get('audio_filepath'):
        audio_filepath = state['audio_filepath']
        transcription = stt(audio_filepath)
        
        # Cleanup input audio file
        try:
            if os.path.exists(audio_filepath):
                os.remove(audio_filepath)
                logging.info(f"STT Node: Cleaned up input audio file: {audio_filepath}")
        except Exception as e:
            logging.warning(f"STT Node: Could not clean up audio file {audio_filepath}: {e}")
    
    if not transcription:
        raise Exception("Failed to get transcription from either audio or text input")

    # Check for trigger word if this mode requires it
    contains_trigger = False
    if state.get("require_trigger", False):
        contains_trigger = "agent" in transcription.lower()

    # Send transcription to client
    await manager.send_json(
        {'type': 'transcription', 'text': transcription}, 
        state['user_id']
    )
    
    # If trigger was required but not found, inform the user
    if state.get("require_trigger", False) and not contains_trigger:
        await manager.send_json(
            {'type': 'info', 'message': 'Trigger word "agent" not detected. Say "agent" to activate.'}, 
            state['user_id']
        )
        # Send end signals to maintain consistency
        await manager.send_json({'type': 'llm_end'}, state['user_id'])
        await manager.send_json({'type': 'tts_end'}, state['user_id'])
    
    return {
        "transcription": transcription,
        "contains_trigger": contains_trigger
    }

async def llm_node(state: AvatarState) -> dict:
    """Process transcription with LLM and stream response, handling different modes.
    For 'schedule' mode this returns exactly one short rephrased sentence
    (first sentence only) and queues only that single sentence to TTS.
    """
    logging.info(f"LLM Node: Processing for user {state['user_id']} in mode {state['mode']}")
    user_id = state['user_id']
    transcription = state['transcription']
    mode = state['mode']
    image_data = state.get('image_data')
    response_text = ""

    # helper: take first sentence up to punctuation
    def first_sentence_only(text: str) -> str:
        if not text:
            return ""
        # Split on first sentence terminator (.,!?), include terminator
        m = re.search(r'^(.*?[\.!\?])(\s|$)', text.strip())
        if m:
            return m.group(1).strip()
        # no punctuation — return the whole trimmed line
        return text.strip()

    # local fallback rephraser for schedule mode
    def local_single_rephrase(base_text: str) -> str:
        try:
            m = re.search(r"(\d{1,2}:\d{2}).*?for\s+(.+?)[\.\!]?$", base_text, re.I)
            if m:
                time_part = m.group(1)
                purpose = m.group(2)
                return f"It's {time_part}. It's a great time to focus on {purpose}."
            return base_text.strip().rstrip('.!')
        except Exception:
            return base_text

    async def send_to_client_and_tts(text: str):
        nonlocal response_text
        if not text:
            return
        # only one chunk for schedule mode, but reuse this for others if needed
        response_text += text + " "
        # send visible chunk once
        try:
            await manager.send_json({'type': 'llm_chunk', 'text': text}, user_id)
        except Exception:
            pass
        # push single sentence to tts queue (if available)
        if user_id in user_sessions and 'tts_queue' in user_sessions[user_id]:
            user_sessions[user_id]['tts_queue'].put(text)
            logging.debug(f"LLM Node: Sent single sentence to TTS queue: '{text}'")

    try:
        if mode == 'vision' and image_data:
            if not gemini_model:
                raise Exception("Gemini model not configured")
            logging.info("LLM Node: Using Gemini for multimodal processing.")
            content = [transcription, {"mime_type": "image/jpeg", "data": image_data}]
            try:
                gemini_response = gemini_model.generate_content(
                    content,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                txt = getattr(gemini_response, "text", None) or "I've processed your image but couldn't generate a response."
                # for vision keep behavior similar to before but only send once
                await send_to_client_and_tts(first_sentence_only(txt))
            except Exception as e:
                logging.error(f"Error with Gemini processing: {e}")
                await send_to_client_and_tts("Error processing your request.")
        else:
            # schedule mode -> get a single rephrase
            if mode == 'schedule':
                # Prefer LLM rephrase if available
                if llm and prompt and parser:
                    try:
                        schedule_prompt = ChatPromptTemplate.from_messages([
                            ('system', (
                                "You are concise. Rephrase the given short reminder into a single short spoken-style sentence suitable for TTS. "
                                "Return only that single sentence."
                            )),
                            ('user', '{input}')
                        ])
                        schedule_chain = schedule_prompt | llm | parser
                        out_text = schedule_chain.run({"input": transcription})
                        # extract first non-empty line and then the first sentence
                        lines = [ln.strip() for ln in (out_text or "").splitlines() if ln.strip()]
                        chosen = lines[0] if lines else out_text or ""
                        chosen = first_sentence_only(chosen)
                        if not chosen:
                            chosen = first_sentence_only(local_single_rephrase(transcription))
                        await send_to_client_and_tts(chosen)
                    except Exception as e:
                        logging.error(f"Schedule LLM run failed: {e} — falling back to local")
                        chosen = first_sentence_only(local_single_rephrase(transcription))
                        await send_to_client_and_tts(chosen)
                else:
                    logging.info("LLM not configured for schedule mode; using local rephrase.")
                    chosen = first_sentence_only(local_single_rephrase(transcription))
                    await send_to_client_and_tts(chosen)

            else:
                # interactive mode (keep streaming behavior)
                if not llm or not prompt or not parser:
                    raise Exception("LLM components not configured")
                logging.info("LLM Node: Using Langchain for text processing.")
                llm_only_chain = prompt | llm | parser
                # stream and split into sentence chunks as before
                buffer = ""
                sentence_splitter = re.compile(r'([.!?])')
                async for chunk_text in llm_only_chain.astream({"input": transcription}, config={"callbacks": [CallbackHandler()]}):
                    if not chunk_text:
                        continue
                    buffer += chunk_text
                    await manager.send_json({'type': 'llm_chunk', 'text': chunk_text}, user_id)
                    parts = sentence_splitter.split(buffer)
                    while len(parts) >= 3:
                        sentence = (parts[0] + parts[1]).strip()
                        if sentence and user_id in user_sessions and 'tts_queue' in user_sessions[user_id]:
                            user_sessions[user_id]['tts_queue'].put(sentence)
                            response_text += sentence + " "
                            logging.debug(f"LLM Node (interactive): Sent to TTS queue: '{sentence}'")
                        buffer = "".join(parts[2:])
                        parts = sentence_splitter.split(buffer)
                if buffer.strip():
                    if user_id in user_sessions and 'tts_queue' in user_sessions[user_id]:
                        user_sessions[user_id]['tts_queue'].put(buffer.strip())
                        response_text += buffer.strip()

        # signal end-of-response to TTS worker
        if user_id in user_sessions and 'tts_queue' in user_sessions[user_id]:
            user_sessions[user_id]['tts_queue'].put("__END_OF_RESPONSE__")

        await manager.send_json({'type': 'llm_end'}, user_id)
        logging.info(f"LLM Node: Completed for user {user_id}")

    except Exception as e:
        logging.error(f"LLM Node: Error during processing for user {user_id}: {e}")
        await manager.send_json({'type': 'error', 'message': f'LLM processing error: {str(e)}'}, user_id)
        raise e

    return {"llm_response": response_text, "response_complete": True}


async def tts_node(state: AvatarState) -> dict:
    """Placeholder for TTS node - actual TTS is handled by worker processes."""
    logging.info(f"TTS Node: TTS handled by worker for user {state['user_id']}")
    return {}

# --- LangGraph Conditional Edge Functions ---
def route_after_stt(state: AvatarState) -> Literal["llm", "__end__"]:
    """Route based on whether a trigger is required and detected."""
    if not state.get("require_trigger", False):
        return "llm"
    if state.get("contains_trigger", False):
        return "llm"
    return "__end__"

def route_after_llm(state: AvatarState) -> Literal["tts", "__end__"]:
    """Route based on whether LLM response is complete."""
    if state.get("response_complete", False):
        return "tts"
    return "__end__"

# --- LangGraph Workflow Setup ---
avatar_workflow = StateGraph(AvatarState)
avatar_workflow.add_node("stt", stt_node)
avatar_workflow.add_node("llm", llm_node)
avatar_workflow.add_node("tts", tts_node)
avatar_workflow.add_edge(START, "stt")
avatar_workflow.add_conditional_edges("stt", route_after_stt, {"llm": "llm", "__end__": END})
avatar_workflow.add_conditional_edges("llm", route_after_llm, {"tts": "tts", "__end__": END})
avatar_workflow.add_edge("tts", END)
avatar_app = avatar_workflow.compile()

# --- Helper: normalize user time strings to HH:MM ---
def normalize_time_string(time_str: str, default_tz: str = "Asia/Kolkata") -> str | None:
    """
    Normalize times like '1:30 PM', '13:30' to 'HH:MM' (24-hour).
    Returns None if parsing fails.
    """
    if not time_str or not isinstance(time_str, str):
        return None

    time_str = time_str.strip()
    fmts = ["%I:%M %p", "%H:%M", "%I:%M%p", "%I.%M %p", "%H:%M:%S"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.strftime("%H:%M")
        except Exception:
            continue

    cleaned = re.sub(r"[^0-9APMapm: ]", "", time_str)
    for fmt in fmts:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%H:%M")
        except Exception:
            continue

    return None

# --- Core Unified Processing Function ---
async def unified_process_interaction(
    user_id: str,
    mode: Literal['avatar', 'vision', 'schedule'],
    audio_filepath: str | None = None,
    image_data: bytes | None = None,
    text_input: str | None = None,
    require_trigger: bool = False
):
    """Process user interaction through the unified LangGraph workflow."""
    try:
        if user_id not in user_sessions:
            logging.warning(f"No active session for user {user_id}. Starting a default TTS-only session.")
            initialize_default_tts(user_id)

        initial_state = AvatarState(
            user_id=user_id, mode=mode, audio_filepath=audio_filepath,
            image_data=image_data, text_input=text_input,
            transcription="", require_trigger=require_trigger,
            contains_trigger=False, llm_response="",
            tts_audio_paths=[], response_complete=False
        )
        
        async for event in avatar_app.astream_events(initial_state, version="v1"):
            logging.debug(f"LangGraph event for user {user_id}: {event}")
            
        logging.info(f"Unified interaction completed for user {user_id}")
        
    except Exception as e:
        logging.error(f"Error in unified processing for user {user_id}: {e}")
        await manager.send_json({'type': 'error', 'message': f'Processing error: {str(e)}'}, user_id)
        try:
            if audio_filepath and os.path.exists(audio_filepath):
                os.remove(audio_filepath)
        except:
            pass

# --- Queue Monitoring ---
async def monitor_queues(user_id: str):
    if user_id not in user_sessions: return
    session = user_sessions[user_id]
    video_queue = session['video_queue']
    audio_queue = session['audio_queue']
    stop_event = session['stop_event']
    
    while not stop_event.is_set():
        try:
            # Check for new TTS audio files
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
                    asyncio.create_task(delayed_cleanup(audio_path, 180))
            except Empty:
                pass

            # Check for new video chunks (only if an avatar is selected)
            if session.get('selected_avatar'):
                try:
                    video_path = video_queue.get_nowait()
                    if video_path and os.path.exists(video_path):
                        filename = os.path.basename(video_path)
                        temp_video_url = f"/temp/{filename}"
                        await manager.send_json({'type': 'video_chunk', 'url': temp_video_url}, user_id)
                        asyncio.create_task(delayed_cleanup(video_path, 180))
                except Empty:
                    pass
            
            await asyncio.sleep(0.05)
        except Exception as e:
            logging.error(f"Queue monitor error for user {user_id}: {e}")
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
    user_id = authenticate_user(username, password)
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

app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

@app.post("/api/video/process")
async def process_video_file(
    request: Request,
    audio: UploadFile = File(...),
    image: UploadFile = File(None),
    user_id: str = Depends(get_current_user)
):
    if not user_id: raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        audio_filename = f"user_audio_{user_id}_{int(time.time())}.webm"
        audio_filepath = os.path.join(TEMP_DIR, audio_filename)
        with open(audio_filepath, "wb") as f: f.write(await audio.read())
        
        image_data = await image.read() if image and image.filename else None
        
        asyncio.create_task(unified_process_interaction(
            user_id=user_id, mode='vision', audio_filepath=audio_filepath,
            image_data=image_data, require_trigger=True
        ))
        
        return JSONResponse(content={"status": "processing", "message": "Your request is being processed"})
    except Exception as e:
        logging.error(f"Error processing video file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
                elif event_type == "enable_vision":
                    if user_id in user_sessions:
                        user_sessions[user_id]['vision_mode'] = True
                        logging.info(f"Vision mode enabled for user {user_id}")
                elif event_type == "disable_vision":
                    if user_id in user_sessions:
                        user_sessions[user_id]['vision_mode'] = False
                        logging.info(f"Vision mode disabled for user {user_id}")
                elif event_type == "schedule":
                    time_val = data.get("time")
                    purpose = data.get("purpose")
                    
                    normalized_time = normalize_time_string(time_val)
                    if not normalized_time:
                        await manager.send_json({'type': 'error', 'message': 'Invalid time format. Use HH:MM or H:MM AM/PM.'}, user_id)
                    else:
                        schedule_info = {'enabled': True, 'time': normalized_time, 'prompt': purpose}

                        # Ensure session exists
                        user_sessions.setdefault(user_id, {})
                        user_sessions[user_id].setdefault('schedules', []).append(schedule_info)
                        
                        # Persist the schedules
                        persistent_schedules.setdefault(user_id, []).append(schedule_info)
                        save_schedules(persistent_schedules)
                        
                        logging.info(f"Schedule added and saved for user {user_id}: Time={normalized_time}, Prompt='{purpose}'")
                        await manager.send_json({'type': 'schedule_set', 'success': True}, user_id)
                        
                        # Automatically select first available avatar if none is selected
                        if not user_sessions[user_id].get('selected_avatar'):
                            avatars = get_avatars_from_local()
                            if avatars:
                                first_avatar = avatars[0]
                                if start_worker_processes(user_id, first_avatar):
                                    user_sessions[user_id]['selected_avatar'] = first_avatar
                                    await manager.send_json({'type': 'avatar_selected', 'success': True, 'avatar': first_avatar}, user_id)
                                    logging.info(f"Automatically selected avatar {first_avatar['id']} for scheduled mode")
                                else:
                                    await manager.send_json({'type': 'error', 'message': 'Failed to start avatar workers for schedule.'}, user_id)
                            else:
                                await manager.send_json({'type': 'error', 'message': 'No avatars available for scheduled mode.'}, user_id)

            elif 'bytes' in message:
                audio_bytes = message['bytes']

                
                # Check if vision mode is active for this user
                is_vision_mode = user_sessions.get(user_id, {}).get('vision_mode', False)

                if is_vision_mode:
                    # Handle combined audio and image data for vision mode
                    if len(audio_bytes) < 4:
                        logging.warning(f"Received insufficient data for vision mode from user {user_id}. Skipping.")
                        continue
                    
                    audio_len = int.from_bytes(audio_bytes[:4], 'big')
                    if len(audio_bytes) < 4 + audio_len:
                        logging.warning(f"Received incomplete data packet for vision mode from user {user_id}. Skipping.")
                        continue
                    
                    audio_data = audio_bytes[4 : 4 + audio_len]
                    image_data = audio_bytes[4 + audio_len:] if len(audio_bytes) > 4 + audio_len else None
                    
                    if not audio_data:
                        logging.warning(f"No audio data in vision mode packet from user {user_id}. Skipping.")
                        continue

                    filename = f"user_audio_{user_id}_{int(time.time())}.webm"
                    filepath = os.path.join(TEMP_DIR, filename)
                    
                    try:
                        with open(filepath, "wb") as f: f.write(audio_data)
                        logging.info(f"Received audio/image for vision and saved audio to {filepath}")
                        await unified_process_interaction(
                            user_id=user_id, mode='vision',
                            audio_filepath=filepath, image_data=image_data,
                            require_trigger=False
                        )
                    except Exception as e:
                        logging.error(f"Error processing vision audio/image bytes: {e}")
                        await manager.send_json({'type': 'error', 'message': f'Error processing vision input: {str(e)}'}, user_id)
                        if os.path.exists(filepath):
                            try: os.remove(filepath)
                            except: pass

                else:
                    # Handle regular audio data for avatar mode
                    if not audio_bytes or len(audio_bytes) < 100:
                        logging.warning(f"Received empty or insufficient audio data from user {user_id} ({len(audio_bytes)} bytes). Skipping processing.")
                        continue
                    
                    filename = f"user_audio_{user_id}_{int(time.time())}.webm"
                    filepath = os.path.join(TEMP_DIR, filename)

                    try:
                        with open(filepath, "wb") as f: f.write(audio_bytes)
                        logging.info(f"Received audio chunk and saved to {filepath}")
                        await unified_process_interaction(
                            user_id=user_id, mode='avatar',
                            audio_filepath=filepath, require_trigger=True
                        )
                    except Exception as e:
                        logging.error(f"Error processing audio bytes: {e}")
                        await manager.send_json({'type': 'error', 'message': f'Error processing audio: {str(e)}'}, user_id)
                        if os.path.exists(filepath):
                            try: os.remove(filepath)
                            except: pass

    except WebSocketDisconnect:
        logging.info(f"Client {user_id} disconnected.")
    except Exception as e:
        logging.error(f"WebSocket processing error for user {user_id}: {e}")
    finally:
        cleanup_session(user_id)
        manager.disconnect(user_id)

# --- Scheduler Task (replaced with minute-aligned, timezone-aware and worker-ensuring version) ---
async def scheduled_avatar_task():
    logging.info("Scheduled Avatar Task started.")
    # Choose timezone (use TZ env if present)
    tz_name = os.getenv("TZ", "Asia/Kolkata")
    if ZoneInfo:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = None
            logging.warning(f"ZoneInfo couldn't load {tz_name}; falling back to local time.")
    else:
        tz = None

    last_checked_minute = None

    while True:
        try:
            now = datetime.now(tz) if tz else datetime.now()
            current_time_str = now.strftime('%H:%M')
            # Only run schedule-check logic once per minute (when minute changes)
            if current_time_str != last_checked_minute:
                last_checked_minute = current_time_str
                logging.debug(f"Checking schedules at {current_time_str}")

                schedules_to_remove = {}

                # shallow copy of mapping keys to avoid mutation during iteration
                schedules_copy = dict(persistent_schedules)

                for u_id, user_schedules in schedules_copy.items():
                    if u_id not in schedules_to_remove:
                        schedules_to_remove[u_id] = []
                    if not isinstance(user_schedules, list):
                        logging.warning(f"Invalid schedule format for user {u_id}: expected list, got {type(user_schedules)}")
                        continue

                    for schedule_info in list(user_schedules):
                        try:
                            if not isinstance(schedule_info, dict):
                                logging.warning(f"Invalid schedule item for user {u_id}: {schedule_info}")
                                continue

                            if not schedule_info.get('enabled', False):
                                continue

                            stored_time = schedule_info.get('time', '')
                            if not stored_time:
                                continue

                            # stored_time should already be 'HH:MM' from normalization above.
                            # Compare directly
                            if stored_time == current_time_str:
                                logging.info(f"Scheduled time triggered for user {u_id} at {current_time_str}")

                                # Ensure workers are running (will start them if missing)
                                ok = ensure_user_workers(u_id)
                                if not ok:
                                    logging.error(f"Failed to ensure workers for user {u_id} — skipping scheduled task")
                                    continue

                                synthesized_prompt = f"It's {current_time_str}, time for {schedule_info.get('prompt', 'your scheduled activity')}."
                                # Notify user (will only reach UI if websocket connected)
                                try:
                                    await manager.send_json({'type': 'schedule_triggered', 'message': f'Scheduled event triggered: {synthesized_prompt}'}, u_id)
                                except Exception as send_error:
                                    logging.debug(f"Could not send schedule notification (user may be offline): {send_error}")

                                # If LLM is configured, go through LLM pipeline; otherwise directly queue TTS text
                                try:
                                    if llm and prompt and parser:
                                        await unified_process_interaction(user_id=u_id, mode='schedule', text_input=synthesized_prompt, require_trigger=False)
                                        logging.info(f"Processed scheduled task via LLM for user {u_id}")
                                    else:
                                        # Fallback: send text directly to user's TTS queue so audio is generated
                                        sess = user_sessions.get(u_id)
                                        if sess and 'tts_queue' in sess:
                                            sess['tts_queue'].put(synthesized_prompt)
                                            sess['tts_queue'].put("__END_OF_RESPONSE__")
                                            logging.info(f"Queued fallback TTS for scheduled task for user {u_id}")
                                        else:
                                            logging.error(f"No tts_queue available to queue fallback TTS for user {u_id}")
                                except Exception as process_error:
                                    logging.error(f"Error handling scheduled task for user {u_id}: {process_error}")

                                schedules_to_remove[u_id].append(schedule_info)

                        except Exception as inner_err:
                            logging.error(f"Error while iterating schedules for user {u_id}: {inner_err}")

                # Remove executed schedules (one-time)
                schedule_changed = False
                for u_id, to_remove_list in schedules_to_remove.items():
                    if to_remove_list and u_id in persistent_schedules and isinstance(persistent_schedules[u_id], list):
                        persistent_schedules[u_id] = [s for s in persistent_schedules[u_id] if s not in to_remove_list]
                        if not persistent_schedules[u_id]:
                            del persistent_schedules[u_id]
                        schedule_changed = True

                if schedule_changed:
                    try:
                        save_schedules(persistent_schedules)
                        logging.info("Saved updated schedules after processing")
                    except Exception as save_error:
                        logging.error(f"Error saving schedules: {save_error}")

            # Sleep a short interval so we notice the minute change reliably
            await asyncio.sleep(5)

        except Exception as e:
            logging.error(f"Error in scheduled_avatar_task: {e}")
            await asyncio.sleep(5)

# --- Ensure the scheduled task is registered at startup (works with uvicorn) ---
@app.on_event("startup")
async def _start_background_tasks():
    logging.info("Registering scheduled_avatar_task on startup")
    asyncio.create_task(scheduled_avatar_task())

# --- Main Execution ---
if __name__ == "__main__":
    for f in os.listdir(TEMP_DIR):
        try: os.remove(os.path.join(TEMP_DIR, f))
        except OSError as e: logging.warning(f"Error removing old temp file {f}: {e}")
    
    mp.set_start_method('spawn', force=True)

    uvicorn.run(app, host="0.0.0.0", port=8506)
