import os
import sys
import threading
import time

import numpy as np

from agent import config

_SAMPLE_RATE = 16000
_model = None
_model_lock = threading.Lock()


def _best_device():
    """
    (device, compute_type) for this machine.

    A GPU is used when ctranslate2 can actually see one -- float16 there is
    several times faster than int8 on the CPU. Everything falls back to int8
    on the CPU, which runs on any laptop with no extra packages.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _get_model():
    """The one shared model. Loaded once, then reused for every question."""
    global _model
    if _model is None:
        # Locked: the web API and the console app can both ask for the model,
        # and loading it twice would waste a gigabyte and several seconds.
        with _model_lock:
            if _model is None:
                device, compute = _best_device()
                started = time.time()
                print(
                    f"Loading speech-to-text model ({config.WHISPER_MODEL_SIZE}, "
                    f"{device}/{compute})...", file=sys.stderr,
                )
                from faster_whisper import WhisperModel
                _model = WhisperModel(
                    config.WHISPER_MODEL_SIZE,
                    device=device,
                    compute_type=compute,
                    # All cores: transcription is the slowest step on a CPU.
                    cpu_threads=os.cpu_count() or 4,
                )
                print(f"Speech-to-text model ready in {time.time() - started:.1f}s",
                      file=sys.stderr)
    return _model


def preload() -> None:
    """
    Loads the model ahead of time, off the request path.

    Without this the FIRST question pays the whole load cost -- measured at
    5 seconds for 'base' and up to 50 seconds when the model still has to be
    downloaded, which makes the assistant look broken on the first try.
    """
    try:
        _get_model()
    except Exception as exc:  # never stop the server because a preload failed
        print(f"[STT] Preload failed ({type(exc).__name__}); "
              f"the model will load on first use.", file=sys.stderr)


def decode_options(language: str = None) -> dict:
    """
    Decoding settings shared by every caller, tuned for spoken English.

    Measured on six spoken questions (see the project notes):
      * language fixed          -- skips the language-detection pass
      * beam_size 1 (greedy)    -- no accuracy loss on clear speech
      * vad_filter              -- skips the silence a hands-free recording
                                   always has at the end
      * no previous-text conditioning -- each question is independent, and
                                   conditioning invites hallucinated context
    """
    if language is None:
        language = None if config.WHISPER_LANGUAGE in ("", "auto") else config.WHISPER_LANGUAGE
    return {
        "language": language,
        "beam_size": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }


def record_question() -> np.ndarray:
    """
    Records audio from the default microphone for a fixed window
    (config.MAX_RECORD_SECONDS). Push-to-talk: the caller should only call
    this right after the operator presses Enter and starts speaking.
    """
    import sounddevice as sd
    print(f"Listening for up to {config.MAX_RECORD_SECONDS:.0f} seconds... speak now.", file=sys.stderr)
    audio = sd.rec(
        int(config.MAX_RECORD_SECONDS * _SAMPLE_RATE),
        samplerate=_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def transcribe(audio, return_info: bool = False):
    """
    Converts audio into text using faster-whisper.

    When config.WHISPER_LANGUAGE is 'auto' (default), Whisper auto-detects the
    language. If Whisper detects anything other than 'en' or 'hi' (such as 'te'
    when an Indian English/Hindi speaker speaks), all_language_probs is checked
    to choose between 'hi' and 'en', and the audio is re-transcribed in the
    chosen fallback language.

    Empty audio or audio where VAD filters out all speech is handled safely,
    returning "".
    """
    if audio is None:
        return ("", None) if return_info else ""
    if isinstance(audio, np.ndarray) and audio.size == 0:
        return ("", None) if return_info else ""
    if isinstance(audio, str):
        if not os.path.exists(audio) or os.path.getsize(audio) == 0:
            return ("", None) if return_info else ""
    try:
        if len(audio) == 0:
            return ("", None) if return_info else ""
    except TypeError:
        pass

    model = _get_model()
    opts = decode_options()

    try:
        segments, info = model.transcribe(audio, **opts)
    except (ValueError, IndexError) as exc:
        # faster-whisper raises ValueError: max() iterable argument is empty
        # when VAD removes all audio frames (pure silence / empty speech).
        return ("", None) if return_info else ""
    except Exception as exc:
        print(f"[STT] Transcription error: {exc}", file=sys.stderr)
        return ("", None) if return_info else ""

    if getattr(info, "duration_after_vad", None) is not None and info.duration_after_vad <= 0:
        return ("", info) if return_info else ""

    detected_lang = getattr(info, "language", None)
    if config.WHISPER_LANGUAGE in ("", "auto") and detected_lang not in ("en", "hi"):
        if detected_lang == "ur":
            fallback_lang = "hi"
        else:
            probs_raw = getattr(info, "all_language_probs", None)
            probs = {}
            if isinstance(probs_raw, dict):
                probs = probs_raw
            elif isinstance(probs_raw, (list, tuple)):
                try:
                    probs = dict(probs_raw)
                except (ValueError, TypeError):
                    probs = {}

            hi_prob = probs.get("hi", 0.0)
            en_prob = probs.get("en", 0.0)
            fallback_lang = "hi" if hi_prob >= en_prob else "en"

        fallback_opts = decode_options(language=fallback_lang)
        try:
            fallback_segments, fallback_info = model.transcribe(audio, **fallback_opts)
            fallback_text = " ".join(segment.text.strip() for segment in fallback_segments).strip()
            return (fallback_text, fallback_info) if return_info else fallback_text
        except (ValueError, IndexError):
            return ("", info) if return_info else ""
        except Exception as exc:
            print(f"[STT] Fallback transcription error: {exc}", file=sys.stderr)
            return ("", info) if return_info else ""

    try:
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return (text, info) if return_info else text
    except (ValueError, IndexError):
        return ("", info) if return_info else ""
    except Exception as exc:
        print(f"[STT] Segment processing error: {exc}", file=sys.stderr)
        return ("", info) if return_info else ""


def listen_for_question() -> str:
    """Records and transcribes one question. Returns "" if nothing was heard."""
    return transcribe(record_question())
