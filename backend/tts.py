import os
import time
import logging
from typing import Optional
from TTS.api import TTS


_tts = None
device = 'cuda' if os.environ.get('USE_CUDA', '1') == '1' else 'cpu'

def _get_tts():
    global _tts
    if _tts is None:
        logging.info("Loading TTS 'your_tts' on {device}..")
        _tts = TTS("tts_models/multilingual/multi-dataset/your_tts" ).to(device)
    return _tts


def synthesize_speech(text: str, speaker_wav: Optional[str], temp_dir: str, language: str = "en") -> Optional[str]:
    if not text:
        return None
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, f"tts_{int(time.time()*1000)}.wav")
    try:
        tts = _get_tts()
        kwargs = {
            "text": text,
            "language": language,
            "file_path": out_path,
        }
        if speaker_wav and os.path.exists(speaker_wav):
            kwargs["speaker_wav"] = speaker_wav
        tts.tts_to_file(**kwargs)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        return None
    except Exception as e:
        logging.error(f"TTS error: {e}")
        return None


