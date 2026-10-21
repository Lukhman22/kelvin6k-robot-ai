"""
Conversational Q&A over the live machine/print state, powered by Google Gemini.

The model is given the full fused state as JSON and is instructed to answer ONLY
from it. Fields that could not be measured arrive as
{"status": "unavailable", "reason": "..."} so the correct answer to an
unmeasurable question is "I can't determine that", never an invented number.
"""

import json

from google import genai
from google.genai import errors, types

from agent import config
from agent.state import PrinterState

_SYSTEM_PROMPT = (
    "You are the voice assistant for the operator of a 3D printing robot. You "
    "receive the CURRENT machine and print state as JSON, plus the operator's "
    "spoken question (transcribed by speech-to-text, so it may contain minor "
    "errors).\n"
    "\n"
    "ABSOLUTE RULES:\n"
    "1. Answer ONLY from the JSON provided. Never invent, estimate, or infer a "
    "number that is not in it.\n"
    "2. If a field is {\"status\": \"unavailable\"}, say plainly that you cannot "
    "determine it, and give the short reason from the JSON. Do NOT substitute a "
    "guess or a typical value.\n"
    "3. If a value is marked SIMULATED, say it is a simulated value, not a real "
    "machine reading.\n"
    "4. 'visible_layer_bands' is how many layer bands are visible in the camera "
    "frame right now -- it is NOT the total number of layers printed. Never "
    "present it as a total.\n"
    "4b. For any question about whether printing is running, whether a layer is "
    "being printed, the current layer width, or whether there is a problem, use "
    "the 'print_status' block. It is debounced and confidence-checked and is the "
    "authoritative answer. If 'layer_width' is null but 'layer_width_px' has a "
    "value, say the width is not calibrated to millimetres yet -- never read the "
    "pixel number out as millimetres. If 'status' is ABNORMAL_LAYER, say the "
    "layer is outside the configured width limits.\n"
    "5. Wall count cannot be reliably determined from the camera view; say so "
    "when asked how many walls there are.\n"
    "6. You are a monitoring and advisory system only. You may recommend an "
    "action, but never claim to have changed any machine setting.\n"
    "\n"
    "STYLE: one or two short sentences, written to be spoken aloud. No markdown, "
    "no bullet points. Spell out units in words ('82 percent', '0.8 millimeters', "
    "'1200 R P M').\n"
    "\n"
    "LANGUAGE: the operator may ask in English, Hindi, or a mix ('Print chal raha "
    "hai kya?', 'Layer print ho rahi hai kya?', 'Current layer width kya hai?'). "
    "Reply in the SAME language the question was asked in, keeping technical terms "
    "the operator used. The same rules above apply regardless of language."
)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _describe_exception(exc: Exception):
    """
    Maps a Gemini SDK exception to (category, spoken_message).
    Never includes the API key.
    """
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        message = (getattr(exc, "message", "") or "").lower()

        if code in (401, 403) or "api key" in message or "permission" in message:
            return "invalid_api_key", (
                "My Gemini API key was rejected. Please check GEMINI_API_KEY in the dot env file."
            )
        if code == 404 or "not found" in message:
            return "invalid_model", (
                f"The configured model, {config.GEMINI_MODEL}, was not found or is not available "
                "for this API key. Please check GEMINI_MODEL in the dot env file."
            )
        if code == 429 or "quota" in message or "rate" in message:
            return "rate_limit", (
                "I've hit the Gemini free tier rate limit. Please wait a moment and ask again."
            )
        if isinstance(exc, errors.ServerError) or (code is not None and code >= 500):
            return "server_error", "The Gemini service is having a problem right now. Please try again."
        return "api_error", f"Sorry, the AI service rejected that request: {getattr(exc, 'message', exc)}"

    return "network", (
        "I couldn't reach the Gemini service. Please check the network connection."
    )


def answer_question(question: str, state: PrinterState) -> str:
    if not config.GEMINI_API_KEY:
        return (
            "Configuration error: no Gemini API key is set. "
            "Please add GEMINI_API_KEY to your dot env file and restart the agent."
        )

    user_message = (
        f"CURRENT MACHINE AND PRINT STATE (the only facts you may use):\n"
        f"{json.dumps(state.snapshot(), indent=2, default=str)}\n\n"
        f"Operator's question: {question}"
    )

    try:
        response = _get_client().models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=300,
                temperature=0.2,
            ),
        )
    except Exception as exc:  # SDK raises APIError subclasses and transport errors
        _, message = _describe_exception(exc)
        return message

    text = (response.text or "").strip()
    return text or "I didn't get a clear answer for that."


def test_connection() -> dict:
    """
    Minimal live request to verify the Gemini key and model work.
    Never raises, and never includes the API key in the result.
    """
    if not config.GEMINI_API_KEY:
        return {"success": False, "category": "no_key", "detail": "No GEMINI_API_KEY is configured."}

    try:
        response = _get_client().models.generate_content(
            model=config.GEMINI_MODEL,
            contents="Reply with the single word: ok",
            config=types.GenerateContentConfig(max_output_tokens=10),
        )
    except Exception as exc:
        category, detail = _describe_exception(exc)
        return {"success": False, "category": category, "detail": detail}

    return {
        "success": True,
        "category": "ok",
        "detail": f"The Gemini API key and model '{config.GEMINI_MODEL}' are working correctly.",
    }
