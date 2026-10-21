
import sys
import time
from typing import Optional

from agent.assistant import Assistant

from agent.listener import listen_for_question
from agent.metrics import get_voice_telemetry_summary
from agent.state import PrinterState
from agent.voice import speak

WAKE_WORD = "okay kelvin"


class VoiceAssistant:
    """
    Alexa-style Voice Assistant for Kelvin6k:
        Background Loop -> Detects 'okay kelvin' -> Queries Live Metrics -> Speaks.
    """

    def __init__(self, state: PrinterState):
        self.state = state
        self.assistant = Assistant(state=state)
        self._running = False
        self._busy = False

    def stop(self) -> None:
        self._running = False

    @property
    def busy(self) -> bool:
        return self._busy

    def _resolve_direct_telemetry(self, query: str) -> Optional[str]:
        """
        Fast-path deterministic router for core industrial metrics.
        Bypasses heavy LLM latency to answer under 500ms.
        """
        q = query.lower()
        telemetry = get_voice_telemetry_summary(self.state)

        # 1. Who is printing?
        if any(w in q for w in ["who", "operator", "kaun", "kon"]):
            op = telemetry["operator"]
            return f"The current operator is {op}."

        # 2. Dimensions: Width, Breadth, Height
        if any(w in q for w in ["dimension", "height", "breadth", "width", "size"]):
            h = telemetry["layer_height_mm"]
            w = telemetry["layer_width_mm"]
            if h is None and w is None:
                return "Part dimensions are currently unavailable from the camera feed."
            h_str = f"{h} millimeters" if h is not None else "unavailable"
            w_str = f"{w} millimeters" if w is not None else "unavailable"
            return f"Measured layer height is {h_str}, and breadth is {w_str}."

        # 3. Layer Count: Finished, Remaining
        if any(w in q for w in ["layer", "kitni layer", "progress", "kitna print"]):
            curr = telemetry["current_layer"]
            tot = telemetry["total_layers"]
            rem = telemetry["remaining_layers"]
            pct = telemetry["progress_percent"]

            if curr is None or tot is None:
                return "Layer tracking data is not yet recorded for the active job."
            return (
                f"Currently on layer {curr} of {tot}. "
                f"{rem} layers remaining, progress is {pct} percent."
            )

        return None

    def handle_voice_query(self, query: str) -> None:
        if not query or self._busy:
            return
        self._busy = True
        try:
            print(f"\n[Operator]: {query}")

            # 1. Try fast local telemetry first
            answer = self._resolve_direct_telemetry(query)

            # 2. Fallback to existing assistant for troubleshooting/RAG questions
            if not answer:
                answer = self.assistant.ask(query)

            print(f"[Kelvin]: {answer}\n")
            if hasattr(self.state, "record_interaction"):
                self.state.record_interaction(query, answer)

            try:
                speak(answer)
            except Exception as exc:
                print(f"[TTS ERROR] Could not speak answer: {exc}", file=sys.stderr)
        finally:
            self._busy = False

    def ask_once(self) -> None:
        """Called by camera window SPACE bar fallback."""
        if self._busy:
            return
        try:
            question = listen_for_question()
            if question:
                self.handle_voice_query(question)
        except Exception as exc:
            print(f"[LISTEN ERROR] {exc}", file=sys.stderr)

    def run(self) -> None:
        """
        Continuous background listening loop.
        Triggers automatically on 'okay kelvin'.
        """
        print("\n=======================================================")
        print("  Kelvin6k Industrial Voice Assistant (Alexa Mode)    ")
        print(f"  Listening for trigger: '{WAKE_WORD.upper()}'         ")
        print("=======================================================\n")

        self._running = True
        while self._running:
            if self._busy:
                time.sleep(0.1)
                continue

            try:
                phrase = listen_for_question()
                if not phrase:
                    continue

                phrase_lower = phrase.lower().strip()

                if WAKE_WORD in phrase_lower:
                    print(f"[Trigger Detected]: {phrase}")
                    command = phrase_lower.replace(WAKE_WORD, "").strip()

                    if not command:
                        try:
                            speak("Yes, I am listening.")
                        except Exception:
                            pass
                        command = listen_for_question()

                    if command:
                        self.handle_voice_query(command)

            except (EOFError, KeyboardInterrupt):
                break
            except Exception as exc:
                print(f"[LOOP NOTICE] {exc}", file=sys.stderr)
                time.sleep(0.3)
if __name__ == "__main__":
    from agent.state import PrinterState

    state = PrinterState()
    assistant = VoiceAssistant(state=state)
    assistant.run()