"""
kelvin6k -- 3D printing robot monitoring agent.

Threads:
  main       : camera + vision + live dashboard (OpenCV GUI must own this thread)
  background : machine/process monitoring loop + rule engine + spoken alerts
  background : push-to-talk voice assistant (mic -> Whisper -> Gemini -> TTS)

All three share one PrinterState. If the camera is unavailable, the app keeps
running headless so monitoring and voice still work.

MONITORING AND ADVISORY ONLY -- this application never sends commands to the
printer, pump, accelerator, or robot.
"""

import sys
import threading

from agent import config, db
from agent.agent_loop import MonitoringAgent
from agent.logging_setup import configure_logging
from agent.rules import RuleEngine
from agent.services import machine_state as ms
from agent.state import PrinterState
from agent.vision_loop import VisionLoop
from agent.voice_assistant import VoiceAssistant


def main():
    configure_logging()

    # --assistant runs the tool-calling assistant console instead of the
    # camera pipeline (no camera or microphone required).
    if "--assistant" in sys.argv:
        from assistant_cli import main as assistant_main

        sys.argv = [arg for arg in sys.argv if arg != "--assistant"]
        sys.exit(assistant_main())

    db.init_db()

    state = PrinterState()
    # Let the assistant read the live in-process state directly.
    ms.bind_live_state(state)
    rules = RuleEngine()  # shared, so alert cooldowns cover every source

    monitor = MonitoringAgent(state=state, rules=rules)
    assistant = VoiceAssistant(state=state)
    vision_loop = VisionLoop(
        state=state,
        rules=rules,
        on_alerts=monitor.handle_alerts,
        # SPACE in the live video window asks a question, so the operator never
        # has to leave the camera view.
        on_voice_trigger=assistant.ask_once if config.VOICE_ENABLED else None,
    )

    threading.Thread(target=monitor.run, daemon=True).start()
    if config.VOICE_ENABLED:
        threading.Thread(target=assistant.run, daemon=True).start()
    else:
        print("Voice assistant disabled (VOICE_ENABLED=false).")

    try:
        camera_ran = vision_loop.run()
        if not camera_ran:
            # Headless fallback: keep monitoring + voice alive without the camera.
            print("Running without camera. Press Ctrl+C to stop.")
            while True:
                threading.Event().wait(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        vision_loop.stop()
        monitor.stop()
        assistant.stop()
        print("\nkelvin6k stopped.")


if __name__ == "__main__":
    main()
