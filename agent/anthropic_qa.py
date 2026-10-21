import json

import anthropic
from anthropic import Anthropic

from agent import config
from agent.state import PrinterState

_SYSTEM_PROMPT = (
    "You are a voice assistant for the operator of a 3D printer. You are given "
    "the printer's current sensor readings and recent alerts as JSON, along "
    "with the operator's spoken question (transcribed by speech-to-text, so it "
    "may contain minor errors). Answer using ONLY the data provided. Keep the "
    "answer to one or two short sentences written to be spoken aloud: no "
    "markdown, no bullet points, and spell out units in words (\"82 percent\", "
    "\"210 degrees\", \"12.5 millimeters\") instead of symbols or abbreviations. "
    "If the question asks about something the data doesn't cover, say so "
    "plainly instead of guessing."
)

_client = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        kwargs = {}
        if config.ANTHROPIC_API_KEY:
            kwargs["api_key"] = config.ANTHROPIC_API_KEY
        if config.ANTHROPIC_WORKSPACE_ID:
            # The Anthropic-supported header for identity-linked keys that
            # must declare which workspace a request acts in.
            kwargs["default_headers"] = {"anthropic-workspace-id": config.ANTHROPIC_WORKSPACE_ID}
        _client = Anthropic(**kwargs)
    return _client


def _message_of(exc: Exception) -> str:
    return (getattr(exc, "message", "") or str(exc)).lower()


def _is_billing_error(exc: Exception) -> bool:
    message = _message_of(exc)
    return "credit" in message or "billing" in message or "balance" in message


def _is_missing_workspace_error(exc: Exception) -> bool:
    message = _message_of(exc)
    return "workspace" in message and ("required" in message or "send the id" in message)


def _is_invalid_workspace_error(exc: Exception) -> bool:
    message = _message_of(exc)
    return "workspace" in message and not _is_missing_workspace_error(exc)


def _describe_exception(exc: Exception):
    """
    Maps a caught Anthropic SDK exception to (category, spoken_message).
    Never includes the API key or workspace ID value itself.
    """
    if isinstance(exc, anthropic.AuthenticationError):
        return "invalid_api_key", "My Anthropic API key seems to be invalid. Please check the key in the dot env file."
    if isinstance(exc, anthropic.PermissionDeniedError):
        if _is_invalid_workspace_error(exc):
            return "invalid_workspace_id", (
                "My configured Anthropic workspace ID looks wrong, or this key doesn't have "
                "access to it. Please check ANTHROPIC_WORKSPACE_ID in the dot env file."
            )
        return "permission", "My Anthropic API key doesn't have permission for this request. Please check the account's permissions."
    if isinstance(exc, anthropic.NotFoundError):
        if _is_invalid_workspace_error(exc):
            return "invalid_workspace_id", (
                "My configured Anthropic workspace ID was not found. Please check "
                "ANTHROPIC_WORKSPACE_ID in the dot env file."
            )
        return "invalid_model", f"The configured model, {config.ANTHROPIC_MODEL}, was not found or is unavailable for this account."
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limit", "I'm being rate limited right now. Please try asking again in a moment."
    if isinstance(exc, anthropic.BadRequestError):
        if _is_missing_workspace_error(exc):
            return "missing_workspace_id", (
                "This Anthropic API key requires a workspace ID that isn't configured. "
                "Please set ANTHROPIC_WORKSPACE_ID in the dot env file."
            )
        if _is_invalid_workspace_error(exc):
            return "invalid_workspace_id", (
                "My configured Anthropic workspace ID looks wrong. Please check "
                "ANTHROPIC_WORKSPACE_ID in the dot env file."
            )
        if _is_billing_error(exc):
            return "billing", "There's a billing problem with the Anthropic account. Please check the account's credit balance."
        return "bad_request", f"Sorry, that request was invalid: {getattr(exc, 'message', exc)}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "network", "I couldn't reach the AI service. Please check the network connection."
    if isinstance(exc, anthropic.APIStatusError):
        return "api_error", f"Sorry, I ran into a problem answering that: {getattr(exc, 'message', exc)}"
    return "unknown", "Sorry, I ran into an unexpected problem answering that."


def answer_question(question: str, state: PrinterState) -> str:
    if not config.ANTHROPIC_API_KEY:
        return (
            "Configuration error: no Anthropic API key is set. "
            "Please add ANTHROPIC_API_KEY to your dot env file and restart the agent."
        )

    user_message = (
        f"Current printer data:\n{json.dumps(state.snapshot(), indent=2)}\n\n"
        f"Operator's question: {question}"
    )

    try:
        response = _get_client().messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=300,
            output_config={"effort": "low"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        _, message = _describe_exception(exc)
        return message

    if response.stop_reason == "refusal":
        return "I'm not able to answer that question."

    for block in response.content:
        if block.type == "text":
            return block.text.strip()
    return "I didn't get a clear answer for that."


def test_connection() -> dict:
    """
    Sends a minimal request to verify the configured API key, workspace ID,
    and model work end-to-end. Never raises and never includes the API key
    or workspace ID value in the result. Used by check_api_key.py.
    """
    if not config.ANTHROPIC_API_KEY:
        return {"success": False, "category": "no_key", "detail": "No ANTHROPIC_API_KEY is configured."}

    try:
        _get_client().messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return {"success": True, "category": "ok", "detail": "The API key, workspace, and model are working correctly."}
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        category, detail = _describe_exception(exc)
        return {"success": False, "category": category, "detail": detail}


def discover_workspaces():
    """
    Attempts to safely list available workspaces for this key via the
    Anthropic organization API (read-only, no secrets involved). Returns a
    list of {"id", "name"} dicts, or None if discovery isn't possible (e.g.
    this key isn't org-admin-scoped) -- never raises.
    """
    try:
        pages = _get_client().beta.organization.workspaces.list(limit=100)
        return [{"id": ws.id, "name": ws.name} for ws in pages]
    except Exception:
        return None
