# Edge Avatar Assistant (STT + LLM + TTS)

Backend: FastAPI (CPU-only Faster-Whisper, Ollama, Coqui TTS)
Frontend: Vanilla HTML/CSS/JS

## Requirements
- Python 3.10 (use your `edge` venv if available)
- Ollama running locally with a text model (recommended):
  - `ollama pull llama3.2:1b`
  - CPU-only: `export OLLAMA_NO_GPU=1 && ollama serve`
- Downloaded media in repository root `downloaded_media/` with files per avatar id:
  - `<id>.mp4` (idle)
  - `<id>_lip.mp4` (lip video)
  - `<id>.wav` (speaker reference for TTS)

## Install
```
/path/to/python -m pip install fastapi uvicorn pydub librosa soundfile faster-whisper TTS requests
```
## Run ollama
Ollama serve


## Run backend
```
cd voice_avatar_app
/path/to/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8600
```

## Serve frontend
Use any static server. Example with Python:
```
cd voice_avatar_app/voice_avatar react
npm install
npm run dev
```
Then open http://localhost:8700 and it will call the backend at http://localhost:8600

## Configuration
Environment variables (optional):
- `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default `llama3.2:1b`)

