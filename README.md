# edge-device branch
## This branch will have the frontend / backend code for the edge-device realtime avatar interaction 

![image](https://github.com/user-attachments/assets/901f5974-ed54-4554-b4da-6711ef1050e2)


## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Directory Structure](#directory-structure)
4. [Usage](#usage)
5. [API Endpoints](#api-endpoints)
---
## Prerequisites

* Python 3.11+
* linux / ubuntu / wsl
* UV ⚡
* torch == 2.5.1 , torchvision==0.20.1  , torchaudio==2.5.1
* numpy==1.22.0

## Installation 
## [Manual way] :)

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/Avatar1_project-edge-device.git
   cd Avatar1_project-edge-device
   ```

2. **Create and activate a Python virtual environment using `uv`⚡**

* **Initialize a virtual environment**
  To create a default environment in `.venv`:

  ```bash
  uv venv edge
  ```

* **Activate the virtual environment**

  * **Linux**

    ```bash
    source edge/bin/activate
    ```

* **Verify and use**
  Once activated, you can install packages :

  ```bash
  uv pip install -r requirements.txt
  ```
---
* **or just simply run**
  ```bash
  uv sync
  ```
---
---

## Directory Structure

```
Avatar1_project-main/
├── downloaded_media/
│       ├── 1749534228.mp4
|       ├── 1749534228.wav 
|       └── 1749534228.png
|                         
├── log_dir/
|      └──  useravatar.log
|      
├── mgtt/
|      └──  sub.py
|   
├──  ssl_keys/
|      ├── cert.pem
|      ├── key.pem     
|      └── private_key.pem
|
├──  static/
|      ├── scripts.js
|      └── styles.css
|
├──  temp/
|      ├── synced_video_20250711_082925_511956.mp4
|      └── tts_output_1752222563579_15358.wav    
|
├──  templates/
|      ├── layout.html
|      ├── login.html     
|      └── index.html
|      
├── app.py                 
├── auth.py                   
├── workers.py
├── media_data.db
├── pyproject.toml
├── uv.lock              
└── requirements.txt   
```

---

## Usage

1. **Start the server**

   ```bash
   python app.py
   ```

2. **Open in browser**

   Navigate to `https://localhost:8506/`
---

## API Endpoints

| Endpoint                        | Method   | Description                                                                 |
|---------------------------------|----------|-----------------------------------------------------------------------------|
| `/login`                        | GET      | Serves the login page                                                       |
| `/login`                        | POST     | Authenticates user credentials and sets session. Redirects to / on success  |
| `/logout`                       | GET      | Clears the current user session (logout)                                    |
| `/api/avatars`                  | GET      | Lists all available avatars from local storage.                             |
| `/temp/{filename}`              | GET      | Serves temporary audio or video chunks.                                     |
| `/media/{filename}`             | GET      | Serves pre-recorded avatar media (video, audio, images).                    |
| `/ws/{user_id}`                 | WS       | WebSocket endpoint for real-time communication (audio input, LLM/TTS output)|
| `/` (root)                      | GET      | Serve frontend HTML file                                                    |

---

---
* **or just simply execute run.sh**
* ```bash
  ./run.sh
  ```
---

