import os
import logging
import numpy as np
import librosa
from pydub import AudioSegment
from faster_whisper import WhisperModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_whisper = None
device = 'cuda' if os.environ.get('USE_CUDA', '1') == '1' else 'cpu'

def _get_model():
    global _whisper
    if _whisper is None:
        logging.info("Loading Faster-Whisper 'tiny' on {device}")
        _whisper = WhisperModel("tiny", device=device , compute_type="int8")
    return _whisper


def _ensure_wav(path: str) -> str:
    if path.lower().endswith('.wav'):
        return path
    wav_path = os.path.splitext(path)[0] + '.wav'
    audio = AudioSegment.from_file(path)
    audio.export(wav_path, format='wav')
    return wav_path


def transcribe_file(audio_path: str, language: str = "en") -> str:
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return ""

    try:
        use_path = _ensure_wav(audio_path)
        y, sr = librosa.load(use_path, sr=16000, mono=True)
        # Trim leading/trailing silence to reduce hallucinations
        y, _ = librosa.effects.trim(y, top_db=25)
        if y is None or len(y) == 0:
            return ""
        duration = len(y) / float(sr)
        logging.info(f"Processing audio with duration {duration:05.3f}s")
        if duration < 0.1:
            return ""
        model = _get_model()
        segments, _info = model.transcribe(y, language=language)
        text = " ".join([seg.text for seg in segments]).strip()
        if use_path != audio_path:
            try:
                os.remove(use_path)
            except Exception:
                pass
        return text
    except Exception as e:
        logging.error(f"STT error: {e}")
        return ""


