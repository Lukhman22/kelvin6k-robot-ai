"""
Kelvin6K voice assistant -- interactive console.

    python assistant_cli.py                       # type questions (no mic needed)
    python assistant_cli.py --voice               # speak questions (press Enter, then talk)
    python assistant_cli.py --ask "kitni baki hai?"
    python assistant_cli.py --tools               # show which tool each question hits
    python assistant_cli.py --demo-job            # seed a demo print job to try it out

Also reachable as: python main.py --assistant
"""

import argparse
import json
import sys

from agent import config, db
from agent.assistant import Assistant, tool_debug
from agent.logging_setup import configure_logging
from agent.rag.retriever import index_summary
from agent.services import machine_state as ms

_SAMPLES = [
    "kitni wall print ho gayi hai?",
    "abhi kitni baki hai?",
    "printing chal rahi hai?",
    "current layer kya hai?",
    "layer width kitni hai?",
    "is there any problem?",
]


def seed_demo_job() -> None:
    """Creates a demo job + machine state so questions have real data to read."""
    db.init_db()
    ms.start_print_job("Demo Wall", total_length_mm=1000.0, material="concrete", total_layers=60)
    ms.update_progress(completed_length_mm=650.0, current_layer=42)
    db.record_machine_state(
        printing=True, status="PRINTING", current_layer=42,
        layer_width_mm=18.5, layer_width_px=462.5, confidence=0.94,
        motion_score=1.2, camera_connected=True,
    )
    db.record_layer_measurement(layer_number=42, width_px=462.5, width_mm=18.5, confidence=0.94)
    print("Seeded demo job: 1000 mm total, 650 mm completed, layer 42, width 18.5 mm.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kelvin6K voice assistant")
    parser.add_argument("--voice", action="store_true", help="use the microphone instead of typing")
    parser.add_argument("--ask", type=str, default=None, help="ask one question and exit")
    parser.add_argument("--tools", action="store_true", help="show tool routing for the sample questions")
    parser.add_argument("--demo-job", action="store_true", help="seed a demo print job first")
    parser.add_argument("--no-llm", action="store_true", help="force the offline deterministic router")
    args = parser.parse_args()

    configure_logging()
    db.init_db()

    if args.demo_job:
        seed_demo_job()

    if args.tools:
        for question in _SAMPLES:
            print(json.dumps(tool_debug(question), indent=2, default=str))
            print()
        return 0

    assistant = Assistant(use_llm=False if args.no_llm else None)

    print("=== Kelvin6K assistant ===")
    print(f"LLM: {'Gemini (' + config.GEMINI_MODEL + ')' if assistant.use_llm else 'offline router (no GEMINI_API_KEY)'}")
    rag = index_summary()
    print(f"RAG: {rag['chunks']} chunks from {len(rag['documents'])} document(s), mode={'semantic' if rag['semantic'] else 'lexical'}")
    job = db.get_active_job()
    print(f"Job: {job['job_name'] + ' (' + str(job['completed_length_mm']) + '/' + str(job['total_length_mm']) + ' mm)' if job else 'none active -- try --demo-job'}")
    print()

    if args.ask:
        print(f"You: {args.ask}")
        print(f"Assistant: {assistant.ask_and_speak(args.ask)}")
        return 0

    print("Sample questions:")
    for q in _SAMPLES:
        print(f"  - {q}")
    print("\nType a question (or 'quit'). ", end="")
    if args.voice:
        print("Voice mode: press Enter, then speak.")
    else:
        print()

    while True:
        try:
            if args.voice:
                input("\n[Enter to speak] ")
                from agent.listener import listen_for_question

                question = listen_for_question()
                if not question:
                    print("Didn't catch that.")
                    continue
                print(f"You: {question}")
            else:
                question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        print(f"Assistant: {assistant.ask_and_speak(question)}")

    print("\nBye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
