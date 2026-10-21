"""
Step 1b: turn recorded audio into text, using faster-whisper.

faster-whisper runs completely on your own laptop. No internet, no API key,
no cost. The first run downloads the model (a few hundred megabytes), which
is why the first run is slow and later runs are fast.
"""

import numpy as np

from robot_voice_assistant import config
from robot_voice_assistant.logs import get_logger

log = get_logger("STT")

# The model is loaded once and reused. Loading it takes several seconds, so
# we must not do it for every question.
_model = None


class SpeechToTextError(Exception):
    """Speech could not be converted to text. The message explains why."""


def _get_model():
    global _model
    if _model is None:
        log.info("Loading Whisper model '%s' (first run downloads it)...", config.WHISPER_MODEL_SIZE)
        try:
            from faster_whisper import WhisperModel

            # int8 on the CPU: works on any laptop, no graphics card needed.
            _model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        except Exception as exc:
            raise SpeechToTextError(
                f"The Whisper model could not be loaded: {exc}. "
                "If this is the first run, check your internet connection -- "
                "the model has to be downloaded once."
            ) from exc
        log.info("Whisper model ready.")
    return _model


def transcribe(audio: np.ndarray) -> str:
    """
    Converts audio into text. Returns "" when nothing intelligible was said.

    Hindi, English and mixed Hinglish all work, because the language is
    detected automatically (config.WHISPER_LANGUAGE = "auto").
    """
    if audio is None or audio.size == 0:
        return ""

    # "auto" means: let Whisper decide. Anything else forces that language.
    language = None if config.WHISPER_LANGUAGE in ("", "auto") else config.WHISPER_LANGUAGE

    try:
        segments, info = _get_model().transcribe(audio, language=language)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except SpeechToTextError:
        raise
    except Exception as exc:
        raise SpeechToTextError(f"Whisper could not process the audio: {exc}") from exc

    log.info("Detected language: %s", getattr(info, "language", "?"))
    log.info("Transcript: %r", text)
    return text


def listen_and_transcribe(seconds: float = None) -> str:
    """Convenience: record from the microphone and return the text."""
    from robot_voice_assistant.voice import listen

    return transcribe(listen.record(seconds))
