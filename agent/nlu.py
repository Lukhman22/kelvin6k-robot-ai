"""
agent/nlu.py - Semantic NLU Fallback for Kelvin6K.

Uses the project's configured Google Gemini model (config.GEMINI_MODEL)
to classify natural-language, incomplete, grammatically irregular, and
Hinglish queries that fail strict deterministic matching in query_router.py.

Guarantees:
- Uses ONLY the project's configured Gemini model via google-genai SDK.
- Returns ONLY structured intent, entities, confidence, ambiguous.
- Whitelist of READ/KNOWLEDGE intents ONLY; state mutations are strictly blocked.
- Never performs DB writes, actions, or state resolution.
- Deterministic router remains primary.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Dict, Optional

from agent import config
from agent.logging_setup import get_logger
from agent.query_router import Intent

log = get_logger("NLU")

# Small whitelist of strictly READ/KNOWLEDGE intents allowed in Phase 1
ALLOWED_READ_INTENTS: frozenset[Intent] = frozenset({
    Intent.GET_CURRENT_LAYER,
    Intent.GET_LAYER_WIDTH,
    Intent.GET_LAYER_HEIGHT,
    Intent.GET_WALL_DIMENSIONS,
    Intent.GET_PRINT_QUALITY,
    Intent.GET_MACHINE_STATUS,
    Intent.GET_MATERIAL_USAGE,
    Intent.GET_MATERIAL_REMAINING,
    Intent.GET_PRINT_RATE,
    Intent.GET_ETA,
    Intent.GET_DEADLINE_STATUS,
    Intent.GET_RECENT_PROBLEMS,
    Intent.GET_COMPLETED_PRODUCTION,
    Intent.GET_REMAINING_PRODUCTION,
    Intent.GET_OPERATOR,
    Intent.GET_SITE_PEOPLE_COUNT,
    Intent.KNOWLEDGE_QUERY,
})

# Forbidden action intents that NLU must NEVER produce or execute
FORBIDDEN_WRITE_INTENTS: frozenset[Intent] = frozenset({
    Intent.SHIFT_START,
    Intent.SHIFT_END,
    Intent.LOG_WALL_PRODUCTION,
    Intent.LOG_SHIFT_ISSUE,
    Intent.LOG_SHIFT_HANDOVER,
    Intent.RESOLVE_SHIFT_ISSUE,
})

CONFIDENCE_THRESHOLD = 0.70

_NLU_SYSTEM_PROMPT = """You are the Semantic NLU classifier for Kelvin6K, an industrial concrete 3D printer assistant.
Analyze the operator's spoken query (English, Hindi, or Hinglish, including incomplete phrases, unusual word orders, or minor speech-to-text slips).
Classify the user's intent into EXACTLY ONE allowed intent from the whitelist below, or null if unhandled or ambiguous.

ALLOWED INTENTS (READ & KNOWLEDGE ONLY):
- GET_CURRENT_LAYER: Inquiries about the current layer number, layer count, or present layer.
- GET_LAYER_WIDTH: Inquiries about the width, thickness, or lateral dimensions of the bead or layer.
- GET_PRINT_QUALITY: Inquiries about quality, tolerance compliance, surface finish, or visual standards.
- GET_MACHINE_STATUS: Inquiries about overall printer status, health, whether printing is running fine/operational.
- GET_MATERIAL_USAGE: Inquiries about how much concrete, material, or paste has been poured, used, or consumed.
- GET_PRINT_RATE: Inquiries about print speed, velocity, or deposition rate.
- GET_ETA: Inquiries about remaining time, completion time, or how long printing will take.
- GET_DEADLINE_STATUS: Inquiries about schedule adherence, delays, or whether the job is on track or late.
- GET_RECENT_PROBLEMS: Inquiries about recent errors, alarms, faults, warnings, or defects.
- GET_COMPLETED_PRODUCTION: Inquiries about total completed wall count or production finished.
- GET_REMAINING_PRODUCTION: Inquiries about remaining wall count or production left to print.
- GET_OPERATOR: Inquiries about who is currently operating or running the machine.
- GET_SITE_PEOPLE_COUNT: Inquiries about how many people, workers, or persons are currently on site or visible on camera.
- KNOWLEDGE_QUERY: Troubleshooting questions, defect causes (why layers are peeling/not sticking), remedies, technical guides, how-to procedures.

CRITICAL RULES:
1. READ ONLY: NEVER classify state-changing actions (start/end shift, log wall, resolve issue). If an action is requested, return intent: null and confidence: 0.0.
2. If the query is uninformative noise or gibberish, return intent: null with confidence: 0.0.
3. Extract any mentioned entities (wall_number as int, target_metric as string, date as string).
4. Set ambiguous: true if the query is equally likely to refer to two different intents.

OUTPUT FORMAT: Return a valid JSON object ONLY:
{
  "intent": "<INTENT_NAME>" or null,
  "entities": {
    "wall_number": null,
    "target_metric": null,
    "date": null
  },
  "confidence": 0.0 to 1.0,
  "ambiguous": false
}"""


@dataclass
class NLUResult:
    intent: Optional[Intent] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    ambiguous: bool = False
    source: str = "none"


# Test/runtime override hook for dependency injection
_model_override: Optional[Callable[[str], Optional[str]]] = None


def set_model_override(fn: Optional[Callable[[str], Optional[str]]]) -> None:
    """Allows test fixtures or local harnesses to supply a semantic model callable."""
    global _model_override
    _model_override = fn


def _call_semantic_model(question: str) -> Optional[str]:
    """
    Calls the project's configured Gemini model (config.GEMINI_MODEL)
    using the official google-genai SDK.
    """
    global _model_override
    if _model_override is not None:
        return _model_override(question)

    if not config.GEMINI_API_KEY:
        log.debug("[NLU] GEMINI_API_KEY not configured; skipping semantic NLU")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=f"Operator query: {question}",
            config=types.GenerateContentConfig(
                system_instruction=_NLU_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=250,
            ),
        )
        return (resp.text or "").strip()
    except Exception as exc:
        log.warning("[NLU] Gemini model (%s) call failed: %s", config.GEMINI_MODEL, exc)
        return None


def _parse_model_response(raw_text: str) -> NLUResult:
    """Parses JSON response and enforces whitelist and action restrictions."""
    if not raw_text:
        return NLUResult(intent=None, confidence=0.0)

    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        data = json.loads(cleaned.strip())

        intent_str = data.get("intent")
        entities = data.get("entities") or {}
        confidence = float(data.get("confidence") or 0.0)
        ambiguous = bool(data.get("ambiguous", False))

        intent = None
        if intent_str and intent_str in Intent._value2member_map_:
            cand_intent = Intent(intent_str)
            # Whitelist gate: Only allowed read intents may pass through
            if cand_intent in ALLOWED_READ_INTENTS:
                intent = cand_intent
            else:
                log.warning("[NLU SAFETY] Discarded non-whitelisted/action intent: %s", cand_intent)
                confidence = 0.0

        return NLUResult(
            intent=intent,
            entities=entities,
            confidence=confidence,
            ambiguous=ambiguous,
            source="gemini_semantic_model",
        )
    except Exception as exc:
        log.warning("[NLU] Failed to parse model response %r: %s", raw_text, exc)
        return NLUResult(intent=None, confidence=0.0)


def semantic_fallback(
    question: str, context: Optional[Dict[str, Any]] = None
) -> NLUResult:
    """
    Main entry point for Phase 1 Semantic NLU fallback.
    Invoked only when deterministic router yields UNKNOWN_QUERY or weak fallback.
    """
    if not question or not question.strip():
        return NLUResult(intent=None, confidence=0.0)

    # ATTENDANCE SAFETY GUARD: Attendance operations are strictly local and must NEVER call Gemini.
    q_low = question.lower()
    if any(w in q_low for w in ("shift", "attendance", "clock in", "clock out", "clocked in", "clocked out", "on shift", "working days", "kaam kar raha", "shift par", "shift pe")):
        log.info("[NLU SAFETY] Bypassing Gemini semantic fallback for attendance query: %r", question)
        return NLUResult(intent=None, confidence=0.0)

    raw_response = _call_semantic_model(question.strip())
    if not raw_response:
        return NLUResult(intent=None, confidence=0.0)

    return _parse_model_response(raw_response)
