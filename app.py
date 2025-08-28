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
    GEMINI_API_KEY = "AIzaSyDo4rsDiBFty_hv_STlaXr_u2NxG-5SLDo"
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
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
        logging.error(f"Error during TTS generation: {e}")
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
    user_sessions[user_id]['monitor_task'] = asyncio.create_task(monitor_video_queue(user_id))
    logging.info(f"Video queue monitor task started for user {user_id}.")
    return True

# --- Initialize TTS for Vision Mode ---
def initialize_vision_tts(user_id: str):
    """Initialize TTS worker for vision mode sessions"""
    if user_id in user_sessions:
        # Clean up existing session if any
        cleanup_session(user_id)
    
    tts_queue = mp.Queue()
    audio_queue = mp.Queue()
    stop_event = mp.Event()

    user_sessions[user_id] = {
        'tts_queue': tts_queue,
        'audio_queue': audio_queue,
        'stop_event': stop_event,
        'processes': {},
        'monitor_task': None
    }

    # Start TTS worker with default speaker
    tts_process = mp.Process(target=tts_worker_process, args=(tts_queue, audio_queue, stop_event, DEFAULT_SPEAKER_WAV))
    tts_process.daemon = True
    tts_process.start()
    user_sessions[user_id]['processes']['tts'] = tts_process
    logging.info(f"TTS worker process started for vision mode user {user_id}.")

    # Monitor task for audio queue
    user_sessions[user_id]['monitor_task'] = asyncio.create_task(monitor_vision_audio_queue(user_id))
    logging.info(f"Audio queue monitor task started for vision mode user {user_id}.")

# --- LLM and Langchain setup with Google Gemini ---
llm = None
prompt = None
parser = None
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyDo4rsDiBFty_hv_STlaXr_u2NxG-5SLDo" # Use env or fallback
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    print("Setting up Google Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    
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
    audio_filepath: str
    transcription: str
    contains_trigger: bool # New field for trigger word check
    llm_response: str
    tts_audio_paths: Annotated[List[str], operator.add]
    response_complete: bool

# --- LangGraph Node Functions ---
# Modified STT node to be async
async def stt_node(state: AvatarState) -> dict:
    """Transcribe audio to text and check for trigger word"""
    logging.info(f"STT Node: Processing audio for user {state['user_id']}")
    transcription = stt(state['audio_filepath'])
    
    # Cleanup input audio file
    try:
        if os.path.exists(state['audio_filepath']):
            os.remove(state['audio_filepath'])
            logging.info(f"STT Node: Cleaned up input audio file: {state['audio_filepath']}")
    except Exception as e:
        logging.warning(f"STT Node: Could not clean up audio file {state['audio_filepath']}: {e}")
    
    if not transcription:
        raise Exception("Failed to transcribe audio")
    
    # Check for trigger word "agent"
    contains_trigger = "agent" in transcription.lower()
    
    # Send transcription to client
    await manager.send_json(
        {'type': 'transcription', 'text': transcription}, 
        state['user_id']
    )
    
    # If no trigger word, send a message and end
    if not contains_trigger:
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

# Modified LLM node to be async
async def llm_node(state: AvatarState) -> dict:
    """Process transcription with LLM and stream response (only if trigger word present)"""
    # If no trigger word, skip LLM processing
    if not state.get("contains_trigger", False):
        logging.info(f"LLM Node: Skipping LLM processing for user {state['user_id']} (no trigger)")
        return {"llm_response": "", "response_complete": True}
    
    logging.info(f"LLM Node: Processing transcription for user {state['user_id']}")
    if not llm or not prompt or not parser:
        raise Exception("LLM components not configured")
    
    # Use only the LLM part of the chain (without STT)
    llm_only_chain = prompt | llm | parser
    
    response = ""
    
    # Create Langfuse callback handler for this user/session
    langfuse_handler = CallbackHandler()
    
    try:
        # Wrap LLM call with Langfuse observation
        @observe(name="Avatar LLM Interaction")
        async def get_llm_response():
            async for chunk_text in llm_only_chain.astream(
                {"input": state['transcription']}, 
                config={"callbacks": [langfuse_handler]}
            ):
                yield chunk_text
        
        buffer = ""
        sentence_splitter = re.compile(r'([.!?])')
        
        async for chunk_text in get_llm_response():
            if chunk_text:
                buffer += chunk_text
                # Send chunk to client
                await manager.send_json(
                    {'type': 'llm_chunk', 'text': chunk_text}, 
                    state['user_id']
                )

                # Process complete sentences
                parts = sentence_splitter.split(buffer)
                while len(parts) >= 3:
                    sentence = (parts[0] + parts[1]).strip()
                    if sentence:
                        # Add to response
                        response += sentence + " "
                        # Send to TTS queue if available
                        if state['user_id'] in user_sessions and 'tts_queue' in user_sessions[state['user_id']]:
                            user_sessions[state['user_id']]['tts_queue'].put(sentence)
                            logging.debug(f"LLM Node: Sent to TTS queue: '{sentence}'")
                    buffer = "".join(parts[2:])
                    parts = sentence_splitter.split(buffer)
        
        # Handle remaining buffer
        if buffer.strip():
            response += buffer.strip()
            if state['user_id'] in user_sessions and 'tts_queue' in user_sessions[state['user_id']]:
                user_sessions[state['user_id']]['tts_queue'].put(buffer.strip())
                logging.debug(f"LLM Node: Sent remaining to TTS queue: '{buffer.strip()}'")
        
        # Send end signal to TTS
        if state['user_id'] in user_sessions and 'tts_queue' in user_sessions[state['user_id']]:
            user_sessions[state['user_id']]['tts_queue'].put("__END_OF_RESPONSE__")
        
        # Send end signal to client
        await manager.send_json({'type': 'llm_end'}, state['user_id'])
        logging.info(f"LLM Node: Completed streaming for user {state['user_id']}")
        
    except Exception as e:
        logging.error(f"LLM Node: Error during streaming for user {state['user_id']}: {e}")
        raise e
    
    return {"llm_response": response, "response_complete": True}

# TTS node remains the same (placeholder)
async def tts_node(state: AvatarState) -> dict:
    """Placeholder for TTS node - TTS is handled by worker processes"""
    logging.info(f"TTS Node: TTS handled by worker for user {state['user_id']}")
    # TTS is handled by the worker process via the queue
    return {}

# --- LangGraph Conditional Edge Functions ---
def route_after_stt(state: AvatarState) -> Literal["llm", "__end__"]:
    """Route based on whether trigger word was detected"""
    if state.get("contains_trigger", False):
        return "llm"
    return "__end__"

def route_after_llm(state: AvatarState) -> Literal["tts", "__end__"]:
    """Route based on whether LLM response is complete"""
    if state.get("response_complete", False):
        return "tts"
    return "__end__"

# --- LangGraph Workflow Setup ---
avatar_workflow = StateGraph(AvatarState)

# Add nodes
avatar_workflow.add_node("stt", stt_node)
avatar_workflow.add_node("llm", llm_node)
avatar_workflow.add_node("tts", tts_node)

# Add edges
avatar_workflow.add_edge(START, "stt")
# Conditional edge after STT to check for trigger word
avatar_workflow.add_conditional_edges(
    "stt",
    route_after_stt,
    {
        "llm": "llm",
        "__end__": END
    }
)
# Conditional edge after LLM to handle streaming completion
avatar_workflow.add_conditional_edges(
    "llm",
    route_after_llm,
    {
        "tts": "tts",
        "__end__": END
    }
)
avatar_workflow.add_edge("tts", END)

# Compile the graph without checkpointer (simpler for this use case)
avatar_app = avatar_workflow.compile()

# --- Core Processing Function using LangGraph ---
# Modified to use async/await
async def process_avatar_interaction(user_id: str, audio_filepath: str):
    """Process user interaction through the LangGraph workflow"""
    try:
        # Initialize state
        initial_state = AvatarState(
            user_id=user_id,
            audio_filepath=audio_filepath,
            transcription="",
            contains_trigger=False,
            llm_response="",
            tts_audio_paths=[],
            response_complete=False
        )
        
        # Execute the workflow using astream_events for better async support
        async for event in avatar_app.astream_events(initial_state, version="v1"):
            logging.debug(f"LangGraph event for user {user_id}: {event}")
            # The actual processing is handled within the nodes
            
        logging.info(f"Avatar interaction completed for user {user_id}")
        
    except Exception as e:
        logging.error(f"Error in avatar processing for user {user_id}: {e}")
        await manager.send_json(
            {'type': 'error', 'message': f'Processing error: {str(e)}'}, 
            user_id
        )
        # Cleanup on error
        try:
            if os.path.exists(audio_filepath):
                os.remove(audio_filepath)
        except:
            pass

# --- Video Section Functions (using Gemini directly for multimodal) ---
async def process_gemini_interaction(user_id: str, audio_filepath: str, image_data = None):
    """Process user input through Gemini with optional image"""
    if not gemini_model:
        await manager.send_json({'type': 'error', 'message': 'Gemini model not configured.'}, user_id)
        return

    # Initialize TTS if not already done
    if user_id not in user_sessions or 'tts_queue' not in user_sessions[user_id]:
        initialize_vision_tts(user_id)
    
    try:
        # Transcribe audio
        transcription = stt(audio_filepath)
        
        # Cleanup input audio file
        try:
            if os.path.exists(audio_filepath):
                os.remove(audio_filepath)
                logging.info(f"Cleaned up input audio file: {audio_filepath}")
        except Exception as e:
            logging.warning(f"Could not clean up audio file {audio_filepath}: {e}")

        if not transcription:
            await manager.send_json(
                {'type': 'error', 'message': 'Failed to transcribe audio. Please try speaking longer or check your microphone.'},
                user_id
            )
            return

        await manager.send_json({'type': 'transcription', 'text': transcription}, user_id)

        # Prepare content for Gemini
        content = []
        
        # Add image if provided
        if image_data:
            # Add image to content
            content.append({
                "mime_type": "image/jpeg",
                "data": image_data
            })
            
        # Add text to content
        content.append(transcription if image_data else f"User: {transcription}")
        
        # Generate response from Gemini
        response = gemini_model.generate_content(
            content,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        if not response.text:
            await manager.send_json({'type': 'error', 'message': 'No response from Gemini.'}, user_id)
            return
            
        answer_text = response.text
        await manager.send_json({'type': 'gemini_response', 'text': answer_text}, user_id)
        
        # Convert response to speech using the existing TTS queue
        if user_id in user_sessions and 'tts_queue' in user_sessions[user_id]:
            tts_queue = user_sessions[user_id]['tts_queue']
            # Split response into sentences for streaming
            sentences = re.split(r'[.!?]+', answer_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    tts_queue.put(sentence)
                    await manager.send_json({'type': 'llm_chunk', 'text': sentence}, user_id)
            tts_queue.put("__END_OF_RESPONSE__")
            await manager.send_json({'type': 'llm_end'}, user_id)
        else:
            await manager.send_json({'type': 'error', 'message': 'Session not ready for TTS.'}, user_id)
            
    except Exception as e:
        logging.error(f"Gemini processing error for user {user_id}: {e}")
        await manager.send_json({'type': 'error', 'message': f'Processing error: {str(e)}'}, user_id)
        try:
            if os.path.exists(audio_filepath):
                os.remove(audio_filepath)
        except:
            pass

# --- Queue Monitoring ---
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
            logging.error(f"Queue monitor error for user {user_id}: {e}")
            break
    logging.info(f"Queue monitor stopped for user {user_id}.")

# --- Vision Mode Audio Queue Monitoring ---
async def monitor_vision_audio_queue(user_id: str):
    if user_id not in user_sessions: 
        return
    audio_queue = user_sessions[user_id]['audio_queue']
    stop_event = user_sessions[user_id]['stop_event']
    
    while not stop_event.is_set():
        try:
            # Check for new TTS audio files
            try:
                audio_path = audio_queue.get_nowait()
                if audio_path == "__END_OF_RESPONSE__":
                    logging.info(f"Received __END_OF_RESPONSE__ from TTS worker for vision user {user_id}")
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
                await asyncio.sleep(0.05)
        except Exception as e:
            logging.error(f"Vision audio queue monitor error for user {user_id}: {e}")
            break
    logging.info(f"Vision audio queue monitor stopped for user {user_id}.")

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

# --- Video Section Routes ---
@app.post("/api/video/process")
async def process_video_file(
    request: Request,
    audio: UploadFile = File(...),
    image: UploadFile = File(None),
    user_id: str = Depends(get_current_user)
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Save audio file
        audio_filename = f"user_audio_{user_id}_{int(time.time())}.webm"
        audio_filepath = os.path.join(TEMP_DIR, audio_filename)
        with open(audio_filepath, "wb") as f:
            f.write(await audio.read())
        
        # Save image if provided
        image_data = None
        if image and image.filename:
            image_data = await image.read()
        
        # Process in background
        asyncio.create_task(process_gemini_interaction(user_id, audio_filepath, image_data))
        
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
                    
                    # Use LangGraph for avatar responses with trigger word check
                    # Properly create async task within the event loop
                    asyncio.create_task(process_avatar_interaction(user_id, filepath))
                    
                except Exception as e:
                    logging.error(f"Error processing audio bytes: {e}")
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
        logging.error(f"WebSocket processing error for user {user_id}: {e}")
        try:
            await manager.send_json({'type': 'error', 'message': f"Server error: {str(e)}"}, user_id)
        except:
            pass
    finally:
        cleanup_session(user_id)
        manager.disconnect(user_id)

# --- Video WebSocket Endpoint ---
@app.websocket("/ws/video/{user_id}")
async def video_websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        # Initialize TTS for this vision session when connection is established
        initialize_vision_tts(user_id)
        
        while True:
            message = await websocket.receive()
            if 'bytes' in message:
                # Handle binary data (audio + optional image)
                data = message['bytes']
                
                # Parse the data - first 4 bytes are audio length
                if len(data) < 4:
                    await manager.send_json({'type': 'error', 'message': 'Invalid data format'}, user_id)
                    continue
                
                audio_len = int.from_bytes(data[:4], 'big')
                if len(data) < 4 + audio_len:
                    await manager.send_json({'type': 'error', 'message': 'Incomplete data'}, user_id)
                    continue
                
                audio_data = data[4:4+audio_len]
                image_data = data[4+audio_len:] if len(data) > 4+audio_len else None
                
                # Save audio
                audio_filename = f"user_audio_{user_id}_{int(time.time())}.webm"
                audio_filepath = os.path.join(TEMP_DIR, audio_filename)
                with open(audio_filepath, "wb") as f:
                    f.write(audio_data)
                
                # Process with Gemini
                asyncio.create_task(process_gemini_interaction(user_id, audio_filepath, image_data))
                
            elif 'text' in message:
                # Handle text messages
                data = json.loads(message['text'])
                if data.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong'})
                    
    except WebSocketDisconnect:
        logging.info(f"Video client {user_id} disconnected.")
    except Exception as e:
        logging.error(f"Video WebSocket error for user {user_id}: {e}")
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
