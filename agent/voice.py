"""
Text-to-speech for Kelvin6K.

Two distinct capabilities, deliberately separated:

  * `speak(text)`      -- synthesise AND play on THIS machine's speaker.
                          Used by the console app (main.py), where the operator
                          is sitting at the machine running the code.
  * `synthesize(text)`  -- synthesise to a WAV/MP3 file and return the path,
                          playing nothing. Used by the web API so the audio can
                          be streamed to the browser and played on the
                          OPERATOR'S machine.

Every step logs under [TTS] so a failure is visible rather than silent, and
`speak()` can re-raise instead of swallowing errors.

Provider order: ElevenLabs when a key is configured, otherwise the offline
Windows SAPI5 voice via pyttsx3.
"""

import os
import sys
import tempfile
import threading
import time
from typing import Optional, Tuple

from agent import config
from agent.logging_setup import get_logger

log = get_logger("TTS")

# Both the monitoring loop and the voice assistant call speak() from their own
# threads; this lock keeps an alert and a Q&A answer from overlapping.
_speak_lock = threading.Lock()


class TTSError(Exception):
    """Raised when speech could not be produced or played."""


def provider_name() -> str:
    """Which provider will be used for the next call."""
    return "ElevenLabs" if config.ELEVENLABS_API_KEY else "pyttsx3"


# --- offline (Windows SAPI5 via pyttsx3) --------------------------------

def _new_pyttsx3_engine():
    """
    Fresh engine per call.

    pyttsx3 caches driver instances, and a reused engine whose `runAndWait`
    loop was interrupted raises "run loop already started" on the next call.
    """
    import pyttsx3

    engine = pyttsx3.init()
    # Ensure the engine is audible: a driver-level volume of 0 produces a
    # perfectly valid but completely silent waveform.
    try:
        if float(engine.getProperty("volume") or 0) <= 0.0:
            log.warning("pyttsx3 volume was 0 -- raising to 1.0")
            engine.setProperty("volume", 1.0)
    except Exception:
        pass
    return engine


def _pyttsx3_voices() -> list:
    try:
        engine = _new_pyttsx3_engine()
        return list(engine.getProperty("voices") or [])
    except Exception as exc:
        log.warning("Could not enumerate voices: %s", exc)
        return []


def _synthesize_pyttsx3(text: str, path: str) -> str:
    engine = _new_pyttsx3_engine()
    log.info("Generating audio (pyttsx3 / Windows SAPI5)")
    engine.save_to_file(text, path)
    engine.runAndWait()

    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        raise TTSError(
            "pyttsx3 produced no audio file. The Windows speech engine may be "
            "missing a voice -- check Settings > Time & Language > Speech."
        )
    log.info("Audio generated (%d bytes)", os.path.getsize(path))
    return path


def _play_pyttsx3(text: str) -> None:
    engine = _new_pyttsx3_engine()
    log.info("Playback started (pyttsx3 -> default output device)")
    engine.say(text)
    engine.runAndWait()


# --- ElevenLabs ---------------------------------------------------------

def _elevenlabs_call(text: str):
    """
    Returns audio bytes from ElevenLabs, handling BOTH SDK generations.

    0.2.x exposes module-level generate()/set_api_key(); 1.x+ replaced them with
    an ElevenLabs client class. The installed SDK is detected at call time so an
    upgrade doesn't silently break speech.
    """
    import elevenlabs

    if hasattr(elevenlabs, "ElevenLabs"):  # 1.x+
        log.info("Using ElevenLabs SDK 1.x client API")
        client = elevenlabs.ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        stream = client.text_to_speech.convert(
            voice_id=config.ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
        )
        return b"".join(stream)

    log.info("Using ElevenLabs SDK 0.2.x module API")
    elevenlabs.set_api_key(config.ELEVENLABS_API_KEY)
    return elevenlabs.generate(text=text, voice=config.ELEVENLABS_VOICE_ID)


def _synthesize_elevenlabs(text: str, path: str) -> str:
    log.info("Generating audio (ElevenLabs, voice=%s)", config.ELEVENLABS_VOICE_ID)
    audio = _elevenlabs_call(text)
    if not audio:
        raise TTSError("ElevenLabs returned no audio.")
    with open(path, "wb") as handle:
        handle.write(audio)
    log.info("Audio generated (%d bytes)", len(audio))
    return path


# --- public API ---------------------------------------------------------

def synthesize(text: str, out_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Synthesises speech to a file WITHOUT playing it.

    Returns (path, mime_type). Raises TTSError on failure -- the caller decides
    how to report it. Used by the web API to send audio to the browser.
    """
    text = (text or "").strip()
    if not text:
        raise TTSError("Nothing to speak: empty text.")

    log.info("Starting")
    provider = provider_name()
    log.info("Provider: %s", provider)

    directory = out_dir or tempfile.gettempdir()
    os.makedirs(directory, exist_ok=True)
    stamp = int(time.time() * 1000)

    if provider == "ElevenLabs":
        path = os.path.join(directory, f"kelvin6k_tts_{stamp}.mp3")
        try:
            _synthesize_elevenlabs(text, path)
            return path, "audio/mpeg"
        except Exception as exc:
            # Fall back rather than leaving the operator with silence.
            log.warning("ElevenLabs failed (%s) -- falling back to pyttsx3", exc)

    path = os.path.join(directory, f"kelvin6k_tts_{stamp}.wav")
    _synthesize_pyttsx3(text, path)
    return path, "audio/wav"


def speak(text: str, raise_on_error: bool = False) -> bool:
    """
    Synthesises AND plays on THIS machine's speaker.

    Returns True if playback ran without error. When `raise_on_error` is set the
    underlying TTSError propagates instead, so a caller that must not fail
    silently can react.
    """
    text = (text or "").strip()
    if not text:
        return False

    print(f"[VOICE] {text}", file=sys.stderr)

    with _speak_lock:
        log.info("Starting")
        provider = provider_name()
        log.info("Provider: %s", provider)

        try:
            if provider == "ElevenLabs":
                try:
                    import elevenlabs

                    audio = _elevenlabs_call(text)
                    log.info("Audio generated (%d bytes)", len(audio or b""))
                    log.info("Playing audio")
                    elevenlabs.play(audio)
                    log.info("Playback completed")
                    return True
                except Exception as exc:
                    log.warning("ElevenLabs playback failed (%s) -- using pyttsx3", exc)

            _play_pyttsx3(text)
            log.info("Playback completed")
            return True

        except Exception as exc:
            message = (
                f"{type(exc).__name__}: {exc}. "
                f"Voices installed: {len(_pyttsx3_voices())}. "
                "If audio is generated but inaudible, check the Windows default "
                "playback device and the Volume Mixer entry for python.exe."
            )
            log.error("[TTS ERROR] %s", message)
            if raise_on_error:
                raise TTSError(message) from exc
            return False


def diagnostics() -> dict:
    """Machine-readable TTS health, used by tts_test.py and the API."""
    voices = _pyttsx3_voices()
    info = {
        "provider": provider_name(),
        "elevenlabs_key_configured": bool(config.ELEVENLABS_API_KEY),
        "elevenlabs_voice_id": config.ELEVENLABS_VOICE_ID,
        "pyttsx3_voices": [getattr(v, "name", "?") for v in voices],
        "voice_count": len(voices),
    }
    try:
        engine = _new_pyttsx3_engine()
        info["volume"] = engine.getProperty("volume")
        info["rate"] = engine.getProperty("rate")
        current = engine.getProperty("voice")
        info["current_voice"] = str(current).split("\\")[-1] if current else None
    except Exception as exc:
        info["engine_error"] = f"{type(exc).__name__}: {exc}"
    return info
