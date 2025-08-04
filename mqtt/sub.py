import paho.mqtt.client as mqtt
import json
import logging
import requests
import os
import time # Import time for potential delays/retries
from datetime import datetime # Import datetime for user details filename

# --- Configuration ---
MQTT_BROKER_HOST = "broker.hivemq.com"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_AVATAR = "ai_avatar_studio/avatars"
MQTT_TOPIC_USER_DETAILS = "ai_avatar_studio/user_details"

# IMPORTANT:
# Set the base URL for your FastAPI server.
# Since your FastAPI app is running on localhost:8506, this is the correct URL.
FASTAPI_SERVER_BASE_URL = "http://localhost:8506" 

# Directory to save downloaded media files and user info
DOWNLOAD_DIR = "downloaded_media"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MQTT Callbacks ---

def on_connect(client, userdata, flags, rc):
    """Callback function for when the client connects to the MQTT broker."""
    logger.debug(f"Attempted connection with result code {rc}")
    if rc == 0:
        logger.info("Successfully connected to MQTT Broker!")
        # Subscribe to topics upon successful connection
        client.subscribe(MQTT_TOPIC_AVATAR)
        logger.info(f"Subscribed to topic: {MQTT_TOPIC_AVATAR}")
        client.subscribe(MQTT_TOPIC_USER_DETAILS)
        logger.info(f"Subscribed to topic: {MQTT_TOPIC_USER_DETAILS}")
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code {rc}. Check network, broker status, or firewall.")
        # You might want to implement a retry mechanism here for production systems
        # For now, the loop_forever() will attempt to reconnect automatically.

def on_disconnect(client, userdata, rc):
    """Callback function for when the client disconnects from the MQTT broker."""
    logger.warning(f"Disconnected with result code {rc}. Attempting to reconnect...")

def on_message(client, userdata, msg):
    """Callback function for when a message is received from the broker."""
    logger.info(f"Message received on topic: {msg.topic}")
    try:
        # Decode payload as UTF-8 and parse as JSON
        payload = json.loads(msg.payload.decode('utf-8'))
        logger.info(f"Decoded JSON Payload: {json.dumps(payload, indent=2)}")

        if msg.topic == MQTT_TOPIC_AVATAR:
            logger.info("--- Avatar Metadata Received ---")
            avatar_url_path = payload.get('url') # This is the path like /api/avatar-media/{id}
            filename = payload.get('filename')
            avatar_id = payload.get('id')
            # These are expected to be sent by app.py if you want to download originals
            original_audio_id = payload.get('original_audio_id') 
            original_media_id = payload.get('original_media_id')

            if avatar_url_path and filename:
                logger.info(f"Avatar URL Path: {avatar_url_path}")
                logger.info(f"Avatar Filename: {filename}")
                logger.info(f"Avatar ID: {avatar_id}")

                # Construct the full download URL for the avatar video
                full_avatar_download_url = f"{FASTAPI_SERVER_BASE_URL}{avatar_url_path}"
                output_avatar_path = os.path.join(DOWNLOAD_DIR, filename)

                logger.info(f"Attempting to download avatar from: {full_avatar_download_url}")
                try:
                    response = requests.get(full_avatar_download_url, stream=True, timeout=60) # Increased timeout
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    with open(output_avatar_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Successfully downloaded avatar to: {output_avatar_path}")
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"Connection error when downloading avatar from {full_avatar_download_url}. Is FastAPI server running and accessible? Error: {e}")
                except requests.exceptions.Timeout as e:
                    logger.error(f"Timeout error when downloading avatar from {full_avatar_download_url}. Server too slow or network issue. Error: {e}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading avatar from {full_avatar_download_url}: {e}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during avatar download: {e}")
            else:
                logger.warning("Avatar payload missing URL path or filename. Cannot download avatar.")

            # --- Download Original Uploads (Audio/Image) if their IDs are present ---
            # NOTE: For these to be downloaded, your FastAPI app (app.py) must be modified
            # to include 'original_audio_id' and 'original_media_id' in the MQTT payload
            # when it publishes to MQTT_TOPIC_AVATAR.
            
            if original_audio_id:
                audio_download_url = f"{FASTAPI_SERVER_BASE_URL}/api/upload-media/{original_audio_id}"
                # You might need to fetch the original filename from your FastAPI /api/upload-media/{file_id} endpoint
                # or include it in the MQTT payload. For now, using a generic name.
                audio_filename = f"original_audio_{original_audio_id}.wav" 
                output_audio_path = os.path.join(DOWNLOAD_DIR, audio_filename)
                logger.info(f"Attempting to download original audio from: {audio_download_url}")
                try:
                    response = requests.get(audio_download_url, stream=True, timeout=60)
                    response.raise_for_status()
                    with open(output_audio_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Successfully downloaded original audio to: {output_audio_path}")
                except Exception as e:
                    logger.error(f"Error downloading original audio {audio_download_url}: {e}")

            if original_media_id:
                media_download_url = f"{FASTAPI_SERVER_BASE_URL}/api/upload-media/{original_media_id}"
                # You might need to fetch the original filename from your FastAPI /api/upload-media/{file_id} endpoint
                # or include it in the MQTT payload. For now, using a generic name.
                media_filename = f"original_image_{original_media_id}.png" 
                output_media_path = os.path.join(DOWNLOAD_DIR, media_filename)
                logger.info(f"Attempting to download original image from: {media_download_url}")
                try:
                    response = requests.get(media_download_url, stream=True, timeout=60)
                    response.raise_for_status()
                    with open(output_media_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Successfully downloaded original image to: {output_media_path}")
                except Exception as e:
                    logger.error(f"Error downloading original image {media_download_url}: {e}")


        elif msg.topic == MQTT_TOPIC_USER_DETAILS:
            logger.info("--- User Details Received ---")
            # Extract all relevant user details from the payload
            user_id_from_payload = payload.get('user_id', 'N/A')
            nickname = payload.get('nickname', 'N/A')
            dob = payload.get('dob', 'N/A')
            hobbies = payload.get('hobbies', 'N/A')
            best_memory = payload.get('bestMemory', 'N/A')
            timestamp = payload.get('timestamp', 'N/A')
            
            logger.info(f"  User ID (from FastAPI): {user_id_from_payload}")
            logger.info(f"  Nickname: {nickname}")
            logger.info(f"  Date of Birth: {dob}")
            logger.info(f"  Hobbies: {hobbies}")
            logger.info(f"  Best Memory: {best_memory}")
            logger.info(f"  Timestamp: {timestamp}")

            # Save user details to a JSON file
            try:
                # Use the user_id from the payload to create a unique filename
                user_details_filename = f"user_details_{user_id_from_payload}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                user_details_path = os.path.join(DOWNLOAD_DIR, user_details_filename)
                
                with open(user_details_path, 'w') as f:
                    json.dump(payload, f, indent=4)
                logger.info(f"Successfully saved user details to: {user_details_path}")
            except Exception as e:
                logger.error(f"Error saving user details to file: {e}")

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON payload. Raw payload: {msg.payload.decode('utf-8', errors='ignore')}")
    except Exception as e:
        logger.error(f"An error occurred in on_message: {e}")

# --- Main Execution ---
def run_subscriber():
    client = mqtt.Client()
    # Assign callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect # Add disconnect callback

    # Optional: Set a client ID for easier debugging on the broker side
    # client.client_id = "my_edge_device_subscriber" 

    try:
        logger.info(f"Attempting to connect to MQTT broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60) # 60 seconds keepalive
        
        # Start the network loop in a blocking call. This will process
        # incoming messages and handle reconnections.
        logger.info("MQTT client loop started. Waiting for messages...")
        client.loop_forever() 
    except ConnectionRefusedError:
        logger.critical(f"Connection refused. Is the MQTT broker running at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT} and accessible from this device?")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during subscriber setup: {e}")
    finally:
        logger.info("Subscriber script finished.")

if __name__ == "__main__":
    run_subscriber()
