import torch
import os
import numpy as np
import torchaudio
import logging
from datetime import datetime
from queue import Empty
import io
import traceback
import ffmpeg
from contextlib import contextmanager
import time
import subprocess
import json
import soundfile as sf

logging.basicConfig(
    filename='useravatar.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@contextmanager
def gpu_memory_manager():
    try:
        torch.cuda.empty_cache()
        yield
    finally:
        # 'out' might not always be defined, handle gracefully
        try:
            del out
        except NameError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

from pathlib import Path

def video_sync_worker(audio_queue, video_queue, user_avatar_id, stop_event, temp_dir):
    logging.info(f"Video sync worker: Started with user_avatar_id={user_avatar_id}, temp_dir={temp_dir}")
    while not stop_event.is_set():
        try:
            audio_path = audio_queue.get(timeout=1)
            logging.info(f"Video sync worker: Received audio_path={audio_path}")

            if audio_path == "__END_OF_RESPONSE__":
                logging.info("Video sync worker: Received __END_OF_RESPONSE__, signaling end to video_queue.")
                video_queue.put("__END_OF_RESPONSE__")
                break  # Stop the worker loop

            if audio_path is None: 
                logging.info("Video sync worker: Received None audio path, waiting for next query.")
                continue

            # Determine video paths based on user avatar ID
            ideal_video_path = os.path.join( user_avatar_id)
            lip_sync_video_path = os.path.join( user_avatar_id.replace(".mp4", "_lip.mp4"))

            # Check if video files exist
            if not os.path.exists(ideal_video_path):
                logging.error(f"Ideal video file not found: {ideal_video_path}")
                continue

            if not os.path.exists(lip_sync_video_path):
                logging.error(f"Lip-sync video file not found: {lip_sync_video_path}")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = os.path.join(temp_dir, f"synced_video_{timestamp}.mp4")

            try:
                # Sync audio with lip-sync video
                synced_path = sync_video_with_audio(lip_sync_video_path, audio_path, output_path)
                if synced_path:
                    video_queue.put(synced_path)
                    logging.info(f"Video sync worker: Synced video at {synced_path}")
            except Exception as e:
                logging.error(f"Error syncing video: {str(e)}")
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Error in video sync worker loop: {str(e)}")

    logging.info("Video sync worker: Shutting down due to stop_event")

def sync_video_with_audio(video_path, audio_path, output_path):
    try:
        start_time = time.time()
        
        # Verify input files exist
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get exact duration of audio and video
        try:
            audio_duration = get_media_duration(audio_path)
            video_duration = get_media_duration(video_path)
            logging.info(f"Audio duration: {audio_duration}s, Video duration: {video_duration}s")
        except Exception as e:
            logging.error(f"Error getting media duration: {str(e)}")
            raise
        
        # Simple case: audio is shorter than or equal to video
        if audio_duration <= video_duration:
            logging.info("Audio is shorter than video - trimming video to exactly match audio duration")
            try:
                # Use explicit duration to match audio exactly (not using -shortest)
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', audio_path,
                    '-map', '0:v:0',  # Take video from first input
                    '-map', '1:a:0',  # Take audio from second input
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-t', str(audio_duration),  # Explicitly set duration to audio length
                    output_path
                ]
                run_ffmpeg_command(cmd)
            except Exception as e:
                logging.error(f"Error in simple case: {str(e)}")
                raise
        else:
            # Complex case: audio is longer than video, need to loop
            logging.info("Audio is longer than video - creating looped video with exact audio duration")
            temp_dir = os.path.dirname(output_path)
            looped_video_path = os.path.join(temp_dir, f"looped_{os.path.basename(output_path)}")
            
            # Calculate how many loops we need to cover the audio
            loops = int(np.ceil(audio_duration / video_duration))
            logging.info(f"Need to loop video {loops} times to cover audio duration")
            
            # Create a concat file for ffmpeg
            concat_file_path = os.path.join(temp_dir, f"concat_{os.path.basename(output_path)}.txt")
            with open(concat_file_path, 'w') as f:
                for _ in range(loops):
                    f.write(f"file '{os.path.abspath(video_path)}'\n")
            
            try:
                # Create looped video without audio
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file_path,
                    '-c', 'copy',
                    looped_video_path
                ]
                run_ffmpeg_command(cmd)
                
                # Add audio to looped video and trim to exact audio length
                cmd = [
                    'ffmpeg', '-y',
                    '-i', looped_video_path,
                    '-i', audio_path,
                    '-map', '0:v:0',  # Take video from first input
                    '-map', '1:a:0',  # Take audio from second input
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-t', str(audio_duration),  # Explicitly set duration to audio length
                    output_path
                ]
                run_ffmpeg_command(cmd)
            except Exception as e:
                logging.error(f"Error in complex case: {str(e)}")
                raise
            finally:
                # Clean up temporary files
                if os.path.exists(concat_file_path):
                    os.remove(concat_file_path)
                if os.path.exists(looped_video_path):
                    os.remove(looped_video_path)
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output video file not created: {output_path}")
        
        sync_time = time.time() - start_time
        logging.info(f"Video sync completed in {sync_time:.2f}s for {output_path}")
        return output_path
    
    except Exception as e:
        logging.error(f"Error in sync_video_with_audio: {str(e)}")
        # If output file was partially created, clean it up
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logging.info(f"Removed incomplete output file: {output_path}")
            except:
                pass
        raise

def get_media_duration(media_path):
    """Get the duration of a media file using ffprobe."""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        media_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except subprocess.CalledProcessError as e:
        error_msg = f"FFprobe error: {e.stderr}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

def run_ffmpeg_command(cmd):
    """Run an FFmpeg command with proper error handling and logging."""
    logging.info(f"Running FFmpeg command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)