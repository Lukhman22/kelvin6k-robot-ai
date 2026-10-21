"""
Step 1a: record sound from the microphone.

This file does ONE job: turn what you say into a block of numbers (audio).
It does not understand speech -- that is speech_to_text.py's job.

Keeping them separate means that if nothing works, you can find out whether
the microphone failed or the speech recognition failed.
"""

import numpy as np

from robot_voice_assistant import config
from robot_voice_assistant.logs import get_logger

log = get_logger("MIC")


class MicrophoneError(Exception):
    """The microphone could not be used. The message explains why."""


def list_microphones() -> list:
    """Returns the input devices Windows can see. Useful when nothing works."""
    try:
        import sounddevice as sd
    except Exception as exc:
        raise MicrophoneError(f"The sounddevice package could not load: {exc}") from exc

    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) > 0:
            devices.append({"index": index, "name": device.get("name", "?")})
    return devices


def audio_level(audio: np.ndarray) -> float:
    """
    How loud the recording is, as one number between 0.0 and 1.0.

    This is the RMS (root mean square) -- roughly the average loudness.
    Near 0.000 means the microphone recorded silence.
    """
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def record(seconds: float = None) -> np.ndarray:
    """
    Records from the default microphone and returns the audio.

    Raises MicrophoneError with a plain-English reason if anything goes wrong,
    so the robot can say "Sorry, mujhe microphone nahi mil raha hai" instead
    of crashing.
    """
    seconds = seconds or config.RECORD_SECONDS

    try:
        import sounddevice as sd
    except Exception as exc:
        raise MicrophoneError(f"The sounddevice package could not load: {exc}") from exc

    try:
        device_name = sd.query_devices(kind="input").get("name", "unknown")
    except Exception as exc:
        raise MicrophoneError(
            "No microphone was found. Plug one in, or check "
            "Windows Settings > Privacy & security > Microphone."
        ) from exc

    log.info("Device: %s", device_name)
    log.info("Recording for %.0f seconds -- speak now.", seconds)

    try:
        audio = sd.rec(
            int(seconds * config.SAMPLE_RATE),
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as exc:
        # PortAudioError covers "device in use", "permission denied", etc.
        raise MicrophoneError(
            f"The microphone could not record: {exc}. "
            "Another program may be using it, or microphone access is blocked "
            "in Windows Settings > Privacy & security > Microphone."
        ) from exc

    audio = audio.flatten()
    level = audio_level(audio)

    log.info("Recorded %d samples (%.1f seconds)", audio.size, audio.size / config.SAMPLE_RATE)
    log.info("Audio level: %.4f", level)

    if level < config.MIN_AUDIO_LEVEL:
        log.warning(
            "Audio level is almost zero -- the microphone recorded silence. "
            "Check that the correct input device is selected and not muted."
        )

    return audio
