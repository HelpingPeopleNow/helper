"""Whisper STT adapter using faster-whisper (CPU-optimized)."""

import io
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy-loaded model (loaded on first request to save startup time)
_model = None
_model_name = None


def _get_model():
    """Load the whisper model on first use."""
    global _model, _model_name
    if _model is not None:
        return _model

    from faster_whisper import WhisperModel

    _model_name = os.getenv("WHISPER_MODEL", "tiny")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "2"))

    logger.info("Loading whisper model: %s (compute=%s, threads=%d)", _model_name, compute_type, cpu_threads)
    _model = WhisperModel(_model_name, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
    logger.info("Whisper model loaded: %s", _model_name)
    return _model


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    """Transcribe audio bytes to text.

    Returns:
        {"text": "transcribed text", "language": "es"}
    """
    model = _get_model()

    # Write to temp file (faster-whisper needs a file path)
    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,
            language=None,  # auto-detect
        )
        text = " ".join(seg.text.strip() for seg in segments)
        language = info.language or "en"
        logger.info("Whisper transcribed: lang=%s, text_len=%d", language, len(text))
        return {"text": text, "language": language}
    except Exception:
        logger.exception("Whisper transcription failed")
        return {"text": "", "language": "en"}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
