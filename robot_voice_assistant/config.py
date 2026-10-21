"""
All settings for the robot voice assistant, in one place.

Everything is read from the .env file in the project root, so no API key or
password is ever written inside a Python file. Change a value in .env and
restart -- you never need to edit this file.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the project root


# --- Microphone / recording ---------------------------------------------

# How many seconds to record after you press Enter.
RECORD_SECONDS = float(os.getenv("RECORD_SECONDS", "5"))

# Whisper works at 16000 samples per second. Do not change this.
SAMPLE_RATE = 16000

# If the loudness of your recording is below this, it was effectively silence.
# Used to tell "the microphone heard nothing" apart from "Whisper failed",
# which are two completely different problems.
MIN_AUDIO_LEVEL = 0.005


# --- Speech-to-text (faster-whisper) ------------------------------------

# tiny | base | small | medium.  Bigger = more accurate but slower.
# Hindi and Hinglish need at least "small".
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")

# "auto" lets Whisper detect Hindi or English by itself, which is what you
# want for mixed sentences like "Abhi kaun printing kar raha hai?".
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto").strip().lower()


# --- Claude --------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "")


# --- Machine data ---------------------------------------------------------

# Stage 1: a simple JSON file you can edit by hand to test the assistant.
# Stage 2: this is swapped for a PLC / REST API / MQTT source without any
# change to the AI agent.
MACHINE_STATUS_FILE = os.getenv("MACHINE_STATUS_FILE", "data/machine_status.json")
