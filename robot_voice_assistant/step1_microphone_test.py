"""
STEP 1 TEST:  Microphone  ->  Speech-to-Text  ->  print the text

Run it with:

    python -m robot_voice_assistant.step1_microphone_test

Or just list your microphones without recording:

    python -m robot_voice_assistant.step1_microphone_test --devices

This step has NO Claude and NO speaker. If the text that gets printed matches
what you said, step 1 is finished and we move on to step 2.
"""

import argparse
import sys

from robot_voice_assistant import config
from robot_voice_assistant.voice import listen, speech_to_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the microphone and speech-to-text.")
    parser.add_argument("--devices", action="store_true", help="list microphones and exit")
    parser.add_argument("--seconds", type=float, default=config.RECORD_SECONDS,
                        help="how long to record")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: microphone -> speech-to-text")
    print("=" * 60)

    # --- list microphones ------------------------------------------------
    try:
        devices = listen.list_microphones()
    except listen.MicrophoneError as exc:
        print(f"\nFAILED: {exc}")
        return 1

    if not devices:
        print("\nFAILED: Windows reports no microphone at all.")
        print("Fix: plug in a microphone, then check")
        print("     Settings > Privacy & security > Microphone > 'Let desktop apps access'")
        return 1

    print(f"\nMicrophones found ({len(devices)}):")
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")

    if args.devices:
        return 0

    # --- record ----------------------------------------------------------
    print(f"\nPress Enter, then say something for {args.seconds:.0f} seconds.")
    print('Try:  "Abhi kaun printing kar raha hai?"')
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return 1

    try:
        audio = listen.record(args.seconds)
    except listen.MicrophoneError as exc:
        print(f"\nFAILED at recording: {exc}")
        return 1

    level = listen.audio_level(audio)

    # A silent recording is a microphone problem, not a Whisper problem.
    # Saying so here saves a lot of confusion later.
    if level < config.MIN_AUDIO_LEVEL:
        print(f"\nFAILED: the recording is silent (audio level {level:.4f}).")
        print("The microphone was opened but captured nothing.")
        print("Fix: check the correct input device is the Windows default, that it")
        print("     is not muted, and that its input volume is not at zero.")
        return 1

    # --- transcribe -------------------------------------------------------
    print(f"\nAudio level: {level:.4f}  (good -- real sound was captured)")
    print("Converting speech to text...")

    try:
        text = speech_to_text.transcribe(audio)
    except speech_to_text.SpeechToTextError as exc:
        print(f"\nFAILED at speech-to-text: {exc}")
        return 1

    print("\n" + "=" * 60)
    if text:
        print(f"YOU SAID: {text}")
        print("=" * 60)
        print("\nSTEP 1 PASSED. Tell me the text above and we go to step 2.")
        return 0

    print("NO TEXT was recognised.")
    print("=" * 60)
    print("\nSound was captured but Whisper found no words.")
    print("Fix: speak louder and closer to the microphone, or use a bigger model:")
    print("     set WHISPER_MODEL_SIZE=medium in your .env file")
    return 1


if __name__ == "__main__":
    sys.exit(main())
