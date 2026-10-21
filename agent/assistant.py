"""
Alexa-style voice assistant with LLM tool calling.

The model NEVER invents machine values. Every question about the current job is
answered by calling a Python tool that reads the real machine state (live
PrinterState or the SQLite machine_state table); documentation questions go
through RAG. If a tool says a value is unavailable, the assistant says so.

Works with text input (no microphone needed) and with speech.
"""

import datetime
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent import config, shift_store
from agent.logging_setup import get_logger
from agent.rag.retriever import search_documentation
from agent.services import machine_state as ms

log = get_logger("ASSISTANT")


# --- spoken date/time -------------------------------------------------------
# Attendance is announced aloud before it is written, so the operator can catch
# a wrong clock before a record exists. These format the SERVER's time; nothing
# here accepts a time from the browser or hard-codes one.

def _spoken_date(moment: datetime.datetime) -> str:
    """18 September 2026 -- no leading zero, which reads badly aloud."""
    return f"{moment.day} {moment.strftime('%B %Y')}"


def _spoken_time(moment: datetime.datetime) -> str:
    """8:05 AM -- 12-hour, no leading zero on the hour."""
    return f"{moment.strftime('%I').lstrip('0') or '12'}:{moment.strftime('%M %p')}"


def _spoken_line(line_id: str) -> str:
    """"line_1" -> "Line 1". Falls back to the raw id for anything unexpected."""
    match = re.fullmatch(r"line[_\- ]?(\w+)", (line_id or "").strip(), re.IGNORECASE)
    return f"Line {match.group(1)}" if match else (line_id or "this line")

SYSTEM_PROMPT = (
    "You are the voice assistant for the Kelvin6K printing machine, speaking to "
    "the machine operator.\n"
    "\n"
    "TOOLS ARE THE ONLY SOURCE OF TRUTH:\n"
    "1. For ANY question about the current job -- is printing running, how much "
    "is printed, how much remains, current layer, layer width, machine or camera "
    "status -- you MUST call the matching tool. Never answer these from memory, "
    "assumption, or an earlier turn.\n"
    "2. For questions about procedures, manuals, SOPs, specifications, error "
    "codes, maintenance or troubleshooting, call search_documentation.\n"
    "3. NEVER invent a number. If a tool returns available=false, tell the "
    "operator plainly that the value is not available and give the tool's reason. "
    "If search_documentation returns found=false, say the information was not "
    "found in the available documentation.\n"
    "4. If a tool result contains a 'note' about low confidence, mention that the "
    "measurement is uncertain.\n"
    "\n"
    "LANGUAGE: the operator speaks English, Hindi, or a mix (Hinglish), e.g. "
    "'kitni wall print ho gayi hai?', 'abhi kitni baki hai?', 'printing chal rahi "
    "hai?'. Reply in the SAME language and register they used, keeping technical "
    "terms as they said them. For Hinglish, reply in Hinglish.\n"
    "\n"
    "STYLE: short spoken sentences -- one or two. No markdown, no bullet points. "
    "Say units in words ('650 millimeter', '18.5 millimeter'). When citing "
    "documentation, name the source briefly.\n"
    "\n"
    "You are monitoring and advisory only: you may recommend an action but never "
    "claim to have changed a machine setting."
)


# --- tools exposed to the model ----------------------------------------
# Docstrings and type hints become the tool schema, so they are written for the
# model to read.

def get_current_print_status() -> dict:
    """Check whether the machine is currently printing. Use for questions like
    'is printing running', 'printing chal rahi hai', 'is the machine working'."""
    return ms.get_current_print_status()


def get_current_layer() -> dict:
    """Get the current layer number being printed. Use for 'current layer kya hai',
    'which layer is printing now'."""
    return ms.get_current_layer()


def get_current_layer_width() -> dict:
    """Get the measured width of the current printed layer in millimetres.
    Use for 'layer width kitni hai', 'what is the current layer width'."""
    return ms.get_current_layer_width()


def get_print_progress() -> dict:
    """Get print progress: total, completed and remaining wall length in
    millimetres plus percentage. Use for 'kitni wall print ho gayi hai',
    'abhi kitni baki hai', 'how much is printed', 'how much is remaining'."""
    return ms.get_print_progress()


def get_machine_status() -> dict:
    """Get overall machine and camera status, including camera connection,
    detection confidence and motion. Use for 'camera ka status kya hai',
    'is there any problem'."""
    return ms.get_machine_status()


def search_docs(query: str) -> dict:
    """Search the ingested machine manuals, SOPs, maintenance and troubleshooting
    documentation. Use ONLY for documented procedures/specifications, never for
    live machine values. Returns found=false when nothing relevant exists."""
    return search_documentation(query)


def get_active_shift() -> dict:
    """Get the active shift and operator currently running on the line."""
    active = shift_store.get_active_shift_for_line("line_1")
    if not active:
        return {"available": True, "active": False, "reason": "No active shift is currently running on Line 1."}
    op = shift_store.get_operator(active["operator_id"])
    return {
        "available": True,
        "active": True,
        "shift_id": active["shift_id"],
        "shift_type": active["shift_type"],
        "shift_date": active["shift_date"],
        "operator_id": active["operator_id"],
        "operator_name": op["name"] if op else active["operator_id"],
        "line_id": active["line_id"],
    }


def get_site_people_count() -> dict:
    """Get the current people count detected on site via dedicated Camera 2."""
    return ms.get_site_people_count()


def get_site_detections() -> dict:
    """Get the current structured object detections and segmentation entities from dedicated Camera 2."""
    return ms.get_site_detections()


TOOLS: List[Callable] = [
    get_current_print_status,
    get_current_layer,
    get_current_layer_width,
    get_print_progress,
    get_machine_status,
    get_active_shift,
    get_site_people_count,
    get_site_detections,
    search_docs,
]

_TOOLS_BY_NAME = {fn.__name__: fn for fn in TOOLS}


@dataclass
class PendingAction:
    session_id: str
    line_id: str
    intent: str
    params: Dict[str, Any]
    prompt_spoken: str
    created_at: float
    expires_at: float


class PendingActionManager:
    """Manages multi-terminal pending voice actions with 30s confirmation TTL."""

    def __init__(self, ttl_seconds: float = 30.0):
        self.ttl = ttl_seconds
        self._pending: Dict[Tuple[str, str], PendingAction] = {}
        self._lock = threading.Lock()

    def set(self, session_id: str, line_id: str, intent: str, params: Dict[str, Any], prompt: str) -> None:
        now = time.time()
        with self._lock:
            self._pending[(line_id, session_id)] = PendingAction(
                session_id=session_id,
                line_id=line_id,
                intent=intent,
                params=params,
                prompt_spoken=prompt,
                created_at=now,
                expires_at=now + self.ttl,
            )

    def get(self, session_id: str, line_id: str) -> Optional[PendingAction]:
        now = time.time()
        with self._lock:
            action = self._pending.get((line_id, session_id))
            if action and now <= action.expires_at:
                return action
            if action:
                self._pending.pop((line_id, session_id), None)
            return None

    def refresh(self, session_id: str, line_id: str) -> None:
        now = time.time()
        with self._lock:
            action = self._pending.get((line_id, session_id))
            if action:
                action.expires_at = now + self.ttl

    def clear(self, session_id: str, line_id: str) -> None:
        with self._lock:
            self._pending.pop((line_id, session_id), None)


@dataclass
class SessionContext:
    line_id: str = "default_line"
    session_id: str = "default_session"
    last_issue_id: Optional[int] = None
    last_rag_recommendation: Optional[str] = None
    awaiting_disambiguation: Optional[Dict[str, Any]] = None
    awaiting_outcome: Optional[Dict[str, Any]] = None
    awaiting_problem_details: bool = False
    pending_problem_info: Optional[Dict[str, Any]] = None
    awaiting_action_for_issue: bool = False
    last_updated_at: float = field(default_factory=time.time)
    current_operator_id: Optional[str] = None
    current_operator_name: Optional[str] = None
    current_shift_id: Optional[int] = None
    current_line_id: Optional[str] = None
    current_issue_id: Optional[int] = None


_context = SessionContext()


class SessionManager:
    """Manages lightweight in-memory session-scoped issue context with sliding TTL."""

    def __init__(self, ttl_seconds: float = 1800.0):
        self.ttl = ttl_seconds
        self._sessions: Dict[Tuple[str, str], SessionContext] = {}
        self._lock = threading.Lock()

    def _get_locked(self, session_id: str, line_id: str, now: float) -> SessionContext:
        key = (line_id, session_id)
        ctx = self._sessions.get(key)
        if ctx and (now - ctx.last_updated_at > self.ttl):
            ctx = SessionContext(line_id=line_id, session_id=session_id, last_updated_at=now)
            self._sessions[key] = ctx
        elif not ctx:
            ctx = SessionContext(line_id=line_id, session_id=session_id, last_updated_at=now)
            self._sessions[key] = ctx
        else:
            ctx.last_updated_at = now

        if not ctx.current_line_id:
            ctx.current_line_id = line_id

        if not ctx.current_shift_id:
            active = shift_store.get_active_shift_for_line(line_id)
            if active:
                ctx.current_shift_id = active["shift_id"]
                ctx.current_operator_id = active["operator_id"]
                op = shift_store.get_operator(active["operator_id"])
                ctx.current_operator_name = op["name"] if op else active["operator_id"]

        if ctx.current_issue_id is not None and ctx.last_issue_id is None:
            ctx.last_issue_id = ctx.current_issue_id
        elif ctx.last_issue_id is not None and ctx.current_issue_id is None:
            ctx.current_issue_id = ctx.last_issue_id

        return ctx

    def get(self, session_id: str, line_id: str) -> SessionContext:
        now = time.time()
        with self._lock:
            return self._get_locked(session_id, line_id, now)

    def update(self, session_id: str, line_id: str, **kwargs) -> SessionContext:
        now = time.time()
        with self._lock:
            ctx = self._get_locked(session_id, line_id, now)
            for k, v in kwargs.items():
                if hasattr(ctx, k):
                    setattr(ctx, k, v)
            if "last_issue_id" in kwargs and "current_issue_id" not in kwargs:
                ctx.current_issue_id = kwargs["last_issue_id"]
            elif "current_issue_id" in kwargs and "last_issue_id" not in kwargs:
                ctx.last_issue_id = kwargs["current_issue_id"]
            ctx.last_updated_at = now
            return ctx

    def clear(self, session_id: str, line_id: str) -> None:
        with self._lock:
            self._sessions.pop((line_id, session_id), None)


class Assistant:
    """
    Tool-calling assistant over Gemini.

    Falls back to a deterministic local router when no API key is configured, so
    the whole stack (tools, database, RAG) is testable offline.
    """

    def __init__(self, state=None, use_llm: Optional[bool] = None):
        if state is not None:
            ms.bind_live_state(state)
        self.use_llm = (bool(config.GEMINI_API_KEY) if use_llm is None else use_llm)
        self._client = None
        self._history: List[Any] = []
        # Latched off after a configuration failure so one bad key does not
        # add a failing network round trip to every single question.
        self._claude_disabled = False
        self._actions = PendingActionManager(ttl_seconds=30.0)
        self._session_ctx = SessionManager(ttl_seconds=1800.0)
        self._last_gemini_called: bool = False
        self._last_troubleshoot_result = None

    @staticmethod
    def _server_now() -> float:
        """
        The authoritative clock for attendance.

        One method so the time SPOKEN in the confirmation and the time WRITTEN
        by start_shift come from the same source, and so a test can freeze it.
        """
        return time.time()

    # --- LLM path ---

    def _client_or_none(self):
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=config.GEMINI_API_KEY)
            except Exception as exc:
                log.warning("Could not create Gemini client: %s", exc)
                return None
        return self._client

    def _ask_llm(self, question: str) -> Optional[str]:
        client = self._client_or_none()
        if client is None:
            return None
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=TOOLS,  # SDK runs these automatically and feeds results back
                    temperature=0.2,
                    max_output_tokens=400,
                ),
            )
            return (response.text or "").strip() or None
        except Exception as exc:
            log.warning("LLM call failed (%s); using local router.", exc)
            return None

    # --- offline deterministic router ---

    @staticmethod
    def route(question: str) -> str:
        """Maps a question to a tool name using keywords (English + Hinglish)."""
        q = question.lower()

        def has(*words: str) -> bool:
            return any(w in q for w in words)

        if has("width", "chaudai", "chaudayi", "mota"):
            return "get_current_layer_width"
        # Progress questions. "wall"/"printed"/"how many" matter: the machine
        # prints wall length, so "how many walls have been printed?" is a
        # progress question, not a machine-status one.
        if has("baki", "remaining", "bacha", "left", "progress", "kitni wall",
               "how much", "kitna print", "complete", "percent", "%",
               "wall", "how many", "printed", "print ho gay", "ho gayi", "ho gaya",
               "done", "target", "total"):
            return "get_print_progress"
        if has("layer"):
            # "layer width" handled above; plain layer questions -> layer number
            return "get_current_layer"
        if has("camera", "problem", "issue", "dikkat", "kharab", "status"):
            return "get_machine_status"
        if has("printing", "print ho", "chal rah", "chal rha", "running", "working"):
            return "get_current_print_status"
        if has("how do i", "procedure", "manual", "sop", "maintenance", "error code",
               "kaise", "troubleshoot"):
            return "search_docs"
        return "get_machine_status"

    @staticmethod
    def _speak_number(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return str(value)

    # Returned when the question could not be classified at all. This is
    # deliberately NOT a machine-status answer: previously an unrecognised
    # question fell through to a keyword tool whose default was
    # get_machine_status, so a misheard question produced a confident-sounding
    # "Camera is connected. Machine status is PRINTING." That hid the real
    # failure (bad transcript) behind a plausible answer.
    NOT_UNDERSTOOD = (
        "Sorry, I didn't understand that. Please try again. You can ask about "
        "production, layers, material, timing, machine or camera status, "
        "problems, or the documentation."
    )

    # Claude phrases facts; it never originates them. The facts come from the
    # query router, which reads the database and live machine state, and they
    # are passed in as JSON. The prompt forbids inventing or adjusting any
    # value, so the worst a model failure can do is produce a clumsy sentence
    # around correct numbers -- never a wrong number.
    _CLAUDE_SYSTEM = (
        "You are the voice of the Kelvin6K concrete 3D printing machine, speaking "
        "to its operator.\n"
        "You are given the operator\'s question and a JSON object of FACTS "
        "read from the machine\'s database and live state.\n"
        "RULES:\n"
        "1. Use ONLY the numbers in FACTS. Never invent, estimate, or infer a "
        "value that is not there.\n"
        "2. If FACTS says a value is unavailable, say plainly that it is not "
        "available and give the reason from FACTS.\n"
        "3. Reply in ENGLISH ONLY, even if the question is Hindi or Hinglish.\n"
        "4. One or two short spoken sentences. No markdown, no lists. Spell "
        "units in words: 650 millimeters, 65 percent, layer 42.\n"
        "5. You are read-only: never claim to have changed a machine setting."
    )

    def _phrase_with_claude(self, question: str, result) -> Optional[str]:
        """
        Turns router facts into a spoken English sentence using Claude.

        Returns None on ANY problem, so the caller falls back to the
        deterministic sentence. A voice assistant that goes silent because a
        network call failed is worse than one that sounds slightly robotic.
        """
        if not config.ANTHROPIC_API_KEY or self._claude_disabled:
            return None
        try:
            from agent.anthropic_qa import _get_client

            payload = {
                "intent": result.intent.value,
                "available": result.available,
                "facts": result.facts,
                "reason": result.reason,
                "documentation": result.knowledge,
            }
            response = _get_client().messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=200,
                timeout=8.0,
                system=self._CLAUDE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": (f"QUESTION: {question}\n\n"
                                f"FACTS: {json.dumps(payload, default=str)}"),
                }],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            ).strip()
            return text or None
        except Exception as exc:
            # A bad key or a missing workspace fails identically on every call,
            # so retrying it would add seconds of silence to each question. Stop
            # trying after the first configuration failure and say so once; the
            # deterministic wording carries the conversation.
            name = type(exc).__name__
            if name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError"):
                self._claude_disabled = True
                log.warning(
                    "Claude is disabled for this session: %s. The assistant will use its "
                    "own wording. Check ANTHROPIC_API_KEY / ANTHROPIC_WORKSPACE_ID.", name)
            else:
                log.warning("Claude phrasing unavailable (%s); using the local wording.", name)
            return None

    def _answer_locally(self, question: str, routed: Optional[Any] = None) -> str:
        """
        Deterministic answers without an LLM.

        Uses the central query router (agent/query_router.py), which classifies
        the question into one of the documented intents and resolves it against
        real data.

        Three distinct outcomes, never conflated:
          * understood + data present  -> the real answer
          * understood + data missing  -> why it is unavailable
          * not understood             -> say so; never guess an intent
        """
        from agent import query_router

        question = (question or "").strip()
        if not question:
            return "I didn't catch a question. Please try again."

        result = routed if routed is not None else query_router.route(question)
        if result.intent == query_router.Intent.UNKNOWN_QUERY:
            log.info("Unclassified question: %r", question)
            return self.NOT_UNDERSTOOD

        # Deterministic sentence first: it is always correct, and becomes the
        # fallback if Claude is unreachable or misconfigured.
        deterministic = query_router.summarise(result)
        # A wake-word acknowledgement is fixed wording, not a fact to phrase.
        if result.intent == query_router.Intent.WAKE_WORD:
            return deterministic
        spoken = self._phrase_with_claude(question, result)
        if spoken:
            log.info("[CLAUDE] %s", spoken)
            return spoken
        return deterministic

    def _answer_with_tools(self, question: str) -> str:
        """Legacy keyword->tool path, kept as the last-resort fallback."""
        tool = self.route(question)

        if tool == "search_docs":
            result = search_docs(question)
            if not result.get("found"):
                return result.get("message", "That was not found in the available documentation.")
            top = result["results"][0]
            return f"From {top['source']}: {top['text'][:300]}"

        result = _TOOLS_BY_NAME[tool]()

        if tool == "get_print_progress":
            if not result.get("available"):
                return result.get("reason", "I don't have a current machine status.")
            return (
                f"{self._speak_number(result['completed_mm'])} millimeter printed, "
                f"{self._speak_number(result['remaining_mm'])} millimeter remaining "
                f"out of {self._speak_number(result['total_mm'])} millimeter"
                + (f" ({result['percentage']} percent)." if result.get("percentage") is not None else ".")
            )

        if tool == "get_current_layer":
            if not result.get("available"):
                return result.get("reason", "I don't have a current layer number.")
            return f"Current layer is {result['current_layer']}."

        if tool == "get_current_layer_width":
            if not result.get("available"):
                return result.get("reason", "I don't have a current layer width.")
            note = f" {result['note']}" if result.get("note") else ""
            return f"Current layer width is {result['layer_width_mm']} millimeter.{note}"

        if tool == "get_current_print_status":
            if not result.get("available"):
                return result.get("reason", "I don't have a current machine status.")
            note = f" {result['note']}" if result.get("note") else ""
            return ("Yes, printing is currently running." if result.get("printing")
                    else "No, printing is not currently detected.") + note

        if not result.get("available"):
            return result.get("reason", "I don't have a current machine status.")
        camera = result.get("camera") or {}
        connected = "connected" if camera.get("connected") else "not connected"
        note = f" {result['note']}" if result.get("note") else ""
        return f"Camera is {connected}. Machine status is {result.get('status') or 'unknown'}.{note}"

    # --- operator resolution & action execution ---

    def _resolve_operator(
        self,
        operator_id: Optional[str],
        session_id: str,
        line_id: str,
        spoken_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolves the operator this action belongs to, most trustworthy first:

          1. an operator_id supplied by the terminal (a real credential)
          2. a name the operator spoke, matched against the operators table
          3. whoever already holds the active shift on this line

        NOTE ON TRUST: a spoken name is an ASSERTION, not authentication --
        anyone can say "Anand here". It is accepted because voice attendance
        was specified that way, and the confirmation step is the only check.
        If these records ever feed payroll or compliance, step 1 (a badge scan
        or PIN from the terminal) must become mandatory.
        """
        if operator_id:
            op = shift_store.get_operator(operator_id)
            if op:
                return op

        if spoken_name:
            clean_name = spoken_name.strip().lower()
            if clean_name in ("he", "him", "his", "she", "her", "my", "me"):
                ctx = self._session_ctx.get(session_id, line_id)
                if ctx and ctx.current_operator_id:
                    op = shift_store.get_operator(ctx.current_operator_id)
                    if op:
                        return op

            op = shift_store.find_operator_by_name(spoken_name, active_only=False)
            if op:
                return op

            from agent import query_router
            t_id, t_name = query_router.extract_operator_tuple_from_text(spoken_name)
            if t_id:
                op = shift_store.get_operator(t_id)
                if op:
                    return op
            return None

        # Fallback: check session context
        ctx = self._session_ctx.get(session_id, line_id)
        if ctx and ctx.current_operator_id:
            op = shift_store.get_operator(ctx.current_operator_id)
            if op:
                return op
        return None

    def _prepare_confirmation_prompt(self, intent_val: str, params: Dict[str, Any]) -> str:
        line_id = params.get("line_id", "line_1")
        if intent_val == "SHIFT_START":
            now = datetime.datetime.fromtimestamp(self._server_now())
            name = params.get("operator_name") or "Operator"
            if params.get("create_operator"):
                return (
                    f"New employee detected. I could not find an operator named {name}. "
                    f"Would you like to register as a new employee and start your shift?"
                )
            return f"Shall I start the shift for {name}?"

        if intent_val == "SHIFT_END":
            name = params.get("operator_name") or "Operator"
            return f"Shall I end the shift for {name}?"
        if intent_val == "LOG_WALL_PRODUCTION":
            wn = params.get("wall_number")
            st = params.get("status", "completed")
            active = shift_store.get_active_shift_for_line(line_id)
            op = shift_store.get_operator(active["operator_id"]) if active else None
            name_prefix = f"{op['name']}, " if op and op.get("name") else ""
            return f"{name_prefix}You are about to log Wall {wn} as {st} on {line_id}. Please confirm to proceed."
        if intent_val == "LOG_SHIFT_ISSUE":
            dt = params.get("downtime_minutes", 0)
            dt_str = f" with {dt:.0f} minutes downtime" if dt else ""
            action_taken = params.get("action_taken")
            desc = params.get("description")
            result = params.get("result")
            active = shift_store.get_active_shift_for_line(line_id)
            op = shift_store.get_operator(active["operator_id"]) if active else None
            name_prefix = f"{op['name']}, " if op and op.get("name") else ""
            if action_taken and desc:
                res_part = f", status '{result}'" if result else ""
                return f"{name_prefix}You are about to log an issue for '{desc}' with action '{action_taken}'{res_part} on {line_id}. Please confirm to proceed."
            return f"{name_prefix}You are about to log an issue on {line_id}{dt_str}. Please confirm to proceed."
        if intent_val == "LOG_SHIFT_HANDOVER":
            return f"You are about to initiate a shift handover on {line_id}. Please confirm to proceed."
        if intent_val == "RESOLVE_SHIFT_ISSUE":
            t_id = params.get("target_issue_id")
            action_taken = params.get("action_taken")
            cause = params.get("cause")
            result = params.get("result")
            id_str = f"Issue #{t_id}" if t_id else "the issue"
            details = []
            if cause:
                details.append(f"cause '{cause}'")
            if action_taken:
                details.append(f"action '{action_taken}'")
            if result:
                details.append(f"status '{result}'")
            det_str = f" with {', '.join(details)}" if details else ""
            active = shift_store.get_active_shift_for_line(line_id)
            op = shift_store.get_operator(active["operator_id"]) if active else None
            name_prefix = f"{op['name']}, " if op and op.get("name") else ""
            return f"{name_prefix}You are about to update {id_str}{det_str} on {line_id}. Please confirm to proceed."
        return "Please confirm this action to proceed."

    def _execute_action(self, action: PendingAction) -> str:
        line_id = action.params.get("line_id", "line_1")

        if action.intent == "SHIFT_START":
            operator_id = action.params.get("operator_id")
            op_name = action.params.get("operator_name", "Operator")
            if action.params.get("create_operator"):
                clean_code = re.sub(r"[^A-Za-z0-9]", "", op_name).upper()
                new_op_id = f"OP_{clean_code}"
                if shift_store.get_operator(new_op_id):
                    idx = 1
                    while shift_store.get_operator(f"{new_op_id}_{idx}"):
                        idx += 1
                    new_op_id = f"{new_op_id}_{idx}"
                shift_store.create_operator(
                    operator_id=new_op_id,
                    name=op_name,
                    role="operator",
                    active=1,
                )
                operator_id = new_op_id
                action.params["operator_id"] = new_op_id

            started_at = self._server_now()
            moment = datetime.datetime.fromtimestamp(started_at)
            shift_date = moment.date().isoformat()
            try:
                # 1. Create HR Attendance Record
                record_id = shift_store.create_attendance_record(
                    operator_id=operator_id,
                    employee_name=op_name,
                    attendance_date=shift_date
                )
                
                # 2. Create Machine Production Shift (for issue logging, metrology, etc)
                hour = moment.hour
                shift_type = "Morning" if 6 <= hour < 14 else "Afternoon" if 14 <= hour < 22 else "Night"
                shift_store.start_shift(
                    operator_id=operator_id,
                    shift_date=shift_date,
                    shift_type=shift_type,
                    line_id=line_id
                )
                self._session_ctx.update(
                    action.session_id,
                    line_id,
                    current_operator_id=operator_id,
                    current_operator_name=op_name,
                )
                return f"{op_name}, your shift has started."
            except Exception as exc:
                log.error("Failed to start attendance shift: %s", exc)
                return f"Failed to start shift: {exc}"

        if action.intent == "SHIFT_END":
            operator_id = action.params.get("operator_id")
            op_name = action.params.get("operator_name", "Operator")
            try:
                records = shift_store.list_attendance_records()
                active = None
                for r in records:
                    if r["status"] == "ACTIVE" and (r["operator_id"] == operator_id or (not operator_id and r["employee_name"] == op_name)):
                        active = r
                        break
                if not active:
                    return f"No active shift found for {op_name}."
                shift_store.checkout_attendance_record(
                    operator_id=operator_id,
                    employee_name=op_name
                )
                
                # End Machine Production Shift
                machine_shift = shift_store.get_active_shift_for_line(line_id)
                if machine_shift and (machine_shift["operator_id"] == operator_id or not operator_id):
                    shift_store.end_shift(machine_shift["shift_id"])
                start_dt = datetime.datetime.fromtimestamp(active["shift_start"]) if isinstance(active["shift_start"], (int, float)) else datetime.datetime.strptime(active["shift_start"], "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.datetime.now()
                diff = end_dt - start_dt
                hours = int(diff.total_seconds() // 3600)
                minutes = int((diff.total_seconds() % 3600) // 60)
                dur_str = f" {hours} hours {minutes} minutes" if hours > 0 else f" {minutes} minutes"
                self._session_ctx.clear(action.session_id, line_id)
                return f"{op_name}, your shift has ended. Duration:{dur_str}."
            except Exception as exc:
                log.error("Failed to end attendance shift: %s", exc)
                return f"Failed to end shift: {exc}"
        if action.intent == "LOG_WALL_PRODUCTION":
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"Cannot log wall production because no active shift is running on {line_id}."
            wall_num = action.params.get("wall_number")
            if not wall_num:
                return "No wall number specified to log production."
            try:
                shift_store.record_production(
                    shift_id=active["shift_id"],
                    wall_number=wall_num,
                    print_file=action.params.get("print_file") or "current_job.gcode",
                    wall_height=float(action.params.get("wall_height", 3000.0)),
                    segment_layers=int(action.params.get("segment_layers", 50)),
                    layers_completed=int(action.params.get("layers_completed", 50)),
                    status=action.params.get("status", "completed"),
                )
                op = shift_store.get_operator(active["operator_id"])
                name_prefix = f"{op['name']}, " if op and op.get("name") else ""
                return f"{name_prefix}Wall {wall_num} production recorded successfully."
            except Exception as exc:
                log.error("Failed to record production: %s", exc)
                return f"Failed to record production: {exc}"

        if action.intent == "LOG_SHIFT_ISSUE":
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"Cannot log issue because no active shift is running on {line_id}."
            try:
                new_id = shift_store.log_issue(
                    shift_id=active["shift_id"],
                    issue_type=action.params.get("issue_type", "operational_issue"),
                    description=action.params.get("description", "Operator reported issue"),
                    wall_number=action.params.get("wall_number"),
                    severity=action.params.get("severity", "warning"),
                    downtime_minutes=float(action.params.get("downtime_minutes", 0.0)),
                    rag_recommendation=action.params.get("rag_recommendation"),
                    rag_verified=int(action.params.get("rag_verified", 0)),
                    rag_source=action.params.get("rag_source"),
                    action_taken=action.params.get("action_taken"),
                    result=action.params.get("result"),
                    cause=action.params.get("cause"),
                )
                self._session_ctx.update(action.session_id, line_id, last_issue_id=new_id, current_issue_id=new_id)
                op = shift_store.get_operator(active["operator_id"])
                name_prefix = f"{op['name']}, " if op and op.get("name") else ""

                rag_rec = action.params.get("rag_recommendation")
                rag_ver = action.params.get("rag_verified", 0)
                guidance = ""
                if rag_rec:
                    if rag_ver == 1:
                        guidance = f" Recommendation: {rag_rec}"
                    else:
                        guidance = " Note: No verified solution is available for this issue."

                action_taken = action.params.get("action_taken")
                result = action.params.get("result")
                details = []
                if action_taken:
                    details.append(f"action '{action_taken}'")
                if result:
                    details.append(f"status '{result}'")
                det_str = f" with {', '.join(details)}" if details else ""
                return f"{name_prefix}Issue #{new_id} logged successfully on {line_id}{det_str}.{guidance}"
            except Exception as exc:
                log.error("Failed to log issue: %s", exc)
                return f"Failed to log issue: {exc}"

        if action.intent == "LOG_SHIFT_HANDOVER":
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"Cannot initiate handover because no active shift is running on {line_id}."
            try:
                shift_store.create_handover(
                    from_shift_id=active["shift_id"],
                    line_id=line_id,
                    completed_work=action.params.get("completed_work", "Shift completed work"),
                    pending_work=action.params.get("pending_work", "Shift pending work"),
                    open_issues=action.params.get("open_issues", "None"),
                )
                return f"Shift handover initiated successfully for {line_id}."
            except Exception as exc:
                log.error("Failed to create handover: %s", exc)
                return f"Failed to initiate handover: {exc}"

        if action.intent == "RESOLVE_SHIFT_ISSUE":
            issue_id = action.params.get("target_issue_id")
            if not issue_id:
                return "No target issue ID specified to resolve."
            cause = action.params.get("cause")
            action_taken = action.params.get("action_taken")
            result = action.params.get("result")
            try:
                ok = shift_store.update_issue(
                    issue_id=issue_id,
                    cause=cause,
                    action_taken=action_taken,
                    result=result,
                )
                if not ok:
                    return f"Failed to update Issue #{issue_id}."
                if result == "resolved":
                    sess = self._session_ctx.get(action.session_id, line_id)
                    if sess.last_issue_id == issue_id or sess.current_issue_id == issue_id:
                        self._session_ctx.update(action.session_id, line_id, last_issue_id=None, current_issue_id=None)
                res_str = f" as {result}" if result else ""
                active = shift_store.get_active_shift_for_line(line_id)
                op = shift_store.get_operator(active["operator_id"]) if active else None
                name_prefix = f"{op['name']}, " if op and op.get("name") else ""
                return f"{name_prefix}Issue #{issue_id} updated successfully{res_str}."
            except Exception as exc:
                log.error("Failed to resolve issue: %s", exc)
                return f"Failed to resolve issue: {exc}"

        return "Unknown action."

    # --- public ---

    def ask(
        self,
        question: str,
        session_id: str = "default_terminal",
        line_id: str = "line_1",
        operator_id: Optional[str] = None,
        routed: Optional[Any] = None,
        attachments: Optional[List[Any]] = None,
    ) -> str:
        """Answers one question or handles two-step confirmed actions."""
        from agent import query_router

        question = (question or "").strip()
        if not question and not attachments:
            return "I didn't catch a question."

        self._last_gemini_called = False
        self._last_troubleshoot_result = None

        # If attachments are present, proceed with RAG-first multimodal troubleshooting
        if attachments:
            from agent import troubleshooter
            res = troubleshooter.troubleshoot(question or "Analyze attached file.", attachments=attachments)
            self._last_troubleshoot_result = res
            self._last_gemini_called = bool(res.gemini_called)
            return res.answer

        # 1. Check for pending action confirmation
        pending = self._actions.get(session_id, line_id)
        if pending:
            clean_q = query_router.strip_wake_words(question).strip().lower()
            q_lower = clean_q or question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))

            # Strict confirmation:
            # Only explicit YES/HAAN/CONFIRM executes
            # Only NO/NAHI/NOT rejects
            # Only CANCEL/STOP cancels
            # Do not treat "Okay Kelvin", "Sure", "Kar do", "Start karo" or unrelated speech as confirmation.
            affirmative_words = {"yes", "haan", "ha", "haanji", "confirm", "confirmed"}
            negative_words = {"no", "nahi", "na", "not", "dont", "don't"}
            cancel_words = {"cancel", "cancelled", "stop", "stopped", "roko"}

            uncertain_phrases = ("not sure", "dont know", "don't know", "no idea", "not clear", "pata nahi", "nahi pata", "samajh nahi")
            uncertain_words = {"unsure", "confused", "maybe", "shyad", "shayad"}

            is_uncertain = any(p in q_lower for p in uncertain_phrases) or bool(words & uncertain_words)

            has_cancel = bool(words & cancel_words)
            has_no = bool(words & negative_words)
            has_yes = bool(words & affirmative_words)

            if not clean_q or is_uncertain:
                is_yes = False
                is_no = False
                is_cancel = False
            elif has_yes and (has_no or has_cancel):
                # Conflicting / contradictory
                is_yes = False
                is_no = False
                is_cancel = False
            elif has_yes:
                is_yes = True
                is_no = False
                is_cancel = False
            elif has_cancel:
                is_yes = False
                is_no = False
                is_cancel = True
            elif has_no:
                is_yes = False
                is_no = True
                is_cancel = False
            else:
                is_yes = False
                is_no = False
                is_cancel = False

            if is_yes and not (is_no or is_cancel):
                self._actions.clear(session_id, line_id)
                return self._execute_action(pending)
            elif is_cancel or is_no:
                self._actions.clear(session_id, line_id)
                return "Action cancelled. No changes were made."
            else:
                # Anything else / unclear: remain in confirmation mode and ask again.
                # Do NOT clear the pending confirmation state.
                self._actions.refresh(session_id, line_id)
                name = pending.params.get("operator_name")
                if not name:
                    active = shift_store.get_active_shift_for_line(line_id)
                    op = shift_store.get_operator(active["operator_id"]) if active else None
                    name = op["name"] if op else None
                if not name:
                    m_name = re.match(r"^([A-Z][a-zA-Z0-9_\s]+?),\s*", pending.prompt_spoken)
                    if m_name and m_name.group(1).lower() != "you are about to":
                        name = m_name.group(1)

                name_prefix = f"{name}, " if name else ""
                prompt = pending.prompt_spoken
                m_shall = re.search(r"\b(Shall I [^?]+\?)", prompt, re.IGNORECASE)
                if m_shall:
                    q_part = m_shall.group(1).strip()
                else:
                    stripped_prompt = re.sub(rf"^{re.escape(name or '')},\s*", "", prompt, flags=re.IGNORECASE).strip()
                    q_part = stripped_prompt

                please_part = "please answer Yes or No." if name_prefix else "Please answer Yes or No."
                return f"{name_prefix}{please_part} {q_part}"

        # 1.1 Check awaiting_disambiguation
        sess = self._session_ctx.get(session_id, line_id)
        if sess.awaiting_disambiguation:
            disambig = sess.awaiting_disambiguation
            candidates = disambig["candidate_issues"]
            cand_ids = disambig["candidate_issue_ids"]
            pending_params = disambig["pending_params"]
            q_lower = question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))

            cancel_terms = {"cancel", "stop", "roko"}
            if (words & cancel_terms) or "mat karo" in q_lower:
                self._session_ctx.update(session_id, line_id, awaiting_disambiguation=None)
                return "Action cancelled. No changes were made."

            if any(w in q_lower for w in ("both", "dono", "all", "sab", "dono ko")):
                return "Please resolve one issue at a time. Which issue number would you like to update?"

            selected_id = None
            m_id = query_router.extract_issue_id(question)
            if m_id and m_id in cand_ids:
                selected_id = m_id
            else:
                m_num = re.search(r"\b(\d+)\b", q_lower)
                if m_num and int(m_num.group(1)) in cand_ids:
                    selected_id = int(m_num.group(1))
                elif any(w in q_lower for w in ("first", "pehla", "pehli", "1st")):
                    selected_id = candidates[0]["issue_id"]
                elif len(candidates) > 1 and any(w in q_lower for w in ("second", "dusra", "dusri", "2nd")):
                    selected_id = candidates[1]["issue_id"]
                else:
                    matched = [c for c in candidates if any(w in c.get("description", "").lower() for w in q_lower.split())]
                    if len(matched) == 1:
                        selected_id = matched[0]["issue_id"]

            if not selected_id:
                cand_list = ", ".join(f"Issue #{c['issue_id']} ({c.get('description') or c.get('issue_type')})" for c in candidates)
                return f"Please specify which issue to update: {cand_list}."

            self._session_ctx.update(session_id, line_id, awaiting_disambiguation=None)
            pending_params["target_issue_id"] = selected_id
            action_taken = pending_params.get("action_taken")
            result = pending_params.get("result")
            cause = pending_params.get("cause")

            if action_taken and not result:
                self._session_ctx.update(
                    session_id,
                    line_id,
                    awaiting_outcome={
                        "target_issue_id": selected_id,
                        "action_taken": action_taken,
                        "cause": cause,
                        "params": pending_params,
                    },
                )
                return f"Recorded action '{action_taken}' for Issue #{selected_id}. Did this resolve the issue and resume printing?"

            if not result and not action_taken:
                return f"What would you like to do with Issue #{selected_id}?"

            prompt = self._prepare_confirmation_prompt("RESOLVE_SHIFT_ISSUE", pending_params)
            self._actions.set(session_id, line_id, "RESOLVE_SHIFT_ISSUE", pending_params, prompt)
            return prompt

        # 1.2 Check awaiting_outcome
        if sess.awaiting_outcome:
            out_info = sess.awaiting_outcome
            q_lower = question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))

            cancel_terms = {"cancel", "stop", "roko"}
            if (words & cancel_terms) or "mat karo" in q_lower:
                self._session_ctx.update(session_id, line_id, awaiting_outcome=None)
                return "Action cancelled. No changes were made."

            if any(w in q_lower for w in ("agar", "if ", "will", "hoga", "jayega", "dekhte", "check karte", "might")):
                self._session_ctx.update(session_id, line_id, awaiting_outcome=None)
                return "Understood. Let me know once the outcome is verified."

            affirms = {"yes", "haan", "ha", "confirm", "resolved", "solve", "theek", "thik", "resumed", "resume"}
            negatives = {"no", "nahi", "not resolved", "still stuck", "abhi bhi", "fail"}
            res_val = query_router.extract_resolution_result(question)
            if res_val == "resolved":
                is_yes = True
                is_no = False
            elif res_val == "not_resolved":
                is_yes = False
                is_no = True
            else:
                is_yes = bool(words & affirms) or any(w in q_lower for w in ("solve ho gaya", "thik ho gaya", "printing resume"))
                is_no = bool(words & negatives) or any(w in q_lower for w in ("not resolved", "abhi bhi dikkat", "thik nahi"))

            target_id = out_info["target_issue_id"]
            action_taken = out_info["action_taken"]
            cause = out_info.get("cause")
            params = out_info.get("params", {})
            params["target_issue_id"] = target_id
            params["action_taken"] = action_taken
            params["cause"] = cause
            self._session_ctx.update(session_id, line_id, awaiting_outcome=None)

            if is_yes and not is_no:
                params["result"] = "resolved"
            elif is_no:
                params["result"] = "not_resolved"
            else:
                return "Please indicate whether the issue was resolved or is still unresolved."

            prompt = self._prepare_confirmation_prompt("RESOLVE_SHIFT_ISSUE", params)
            self._actions.set(session_id, line_id, "RESOLVE_SHIFT_ISSUE", params, prompt)
            return prompt

        # 1.3 Check awaiting_problem_details
        if sess.awaiting_problem_details:
            clean_q = query_router.strip_wake_words(question).strip().lower()
            q_lower = clean_q or question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))
            cancel_terms = {"cancel", "stop", "roko"}
            if (words & cancel_terms) or "mat karo" in q_lower:
                self._session_ctx.update(
                    session_id,
                    line_id,
                    awaiting_problem_details=False,
                    pending_problem_info=None,
                    awaiting_action_for_issue=False,
                )
                return "Action cancelled. No changes were made."

            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                self._session_ctx.update(session_id, line_id, awaiting_problem_details=False)
                return f"Cannot report issue because no active shift is running on {line_id}."

            op = shift_store.get_operator(active["operator_id"]) if active else None
            name = op["name"] if op else (active["operator_id"] if active else "Operator")

            problem_desc = question.strip()
            k = search_documentation(problem_desc)
            rag_rec = None
            rag_src = None
            rag_ver = 0
            if k.get("found") and k.get("results"):
                for res in k["results"]:
                    if query_router._evaluate_rag_verified(res.get("text", "")) == 1:
                        rag_ver = 1
                        rag_src = res.get("source")
                        single_doc = dict(k)
                        single_doc["results"] = [res]
                        spoken = query_router.speak_knowledge(single_doc, problem_desc)
                        rag_rec = spoken or res.get("text", "")
                        break

            issue_type = "operational_issue"
            p_low = problem_desc.lower()
            if "material flow" in p_low or "material slope" in p_low:
                issue_type = "material_flow_issue"
            elif "nozzle" in p_low:
                issue_type = "nozzle_clog"
            elif "pump" in p_low:
                issue_type = "pump_pressure"
            elif "feeder" in p_low:
                issue_type = "feeder_jam"
            elif "sensor" in p_low:
                issue_type = "sensor_fault"

            problem_info = {
                "problem": problem_desc,
                "issue_type": issue_type,
                "rag_recommendation": rag_rec,
                "rag_verified": rag_ver,
                "rag_source": rag_src,
            }
            self._session_ctx.update(
                session_id,
                line_id,
                awaiting_problem_details=False,
                pending_problem_info=problem_info,
                awaiting_action_for_issue=True,
            )
            if rag_ver == 1 and rag_rec:
                return f"{name}, I found a site-verified solution: {rag_rec} Please tell me what action you took."
            else:
                return f"{name}, no site-verified solution was found for this problem. Please tell me what action you took."

        # 1.4 Check awaiting_action_for_issue
        if sess.awaiting_action_for_issue and sess.pending_problem_info:
            clean_q = query_router.strip_wake_words(question).strip().lower()
            q_lower = clean_q or question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))
            cancel_terms = {"cancel", "stop", "roko"}
            if (words & cancel_terms) or "mat karo" in q_lower:
                self._session_ctx.update(session_id, line_id, awaiting_action_for_issue=False, pending_problem_info=None)
                return "Action cancelled. No changes were made."

            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                self._session_ctx.update(session_id, line_id, awaiting_action_for_issue=False, pending_problem_info=None)
                return f"Cannot log issue because no active shift is running on {line_id}."

            op = shift_store.get_operator(active["operator_id"]) if active else None
            name = op["name"] if op else (active["operator_id"] if active else "Operator")

            action_val, _ = query_router.extract_operator_action(question)
            if not action_val:
                action_val = question.strip()
            res_val = query_router.extract_resolution_result(question)
            cause_val = query_router.extract_resolution_cause(question)

            info = sess.pending_problem_info
            problem_text = info["problem"]
            rag_ver = info.get("rag_verified", 0)
            rag_rec = info.get("rag_recommendation") if rag_ver == 1 else None
            rag_src = info.get("rag_source") if rag_ver == 1 else None

            params = {
                "action": "LOG_SHIFT_ISSUE",
                "line_id": line_id,
                "operator_id": active["operator_id"],
                "operator_name": name,
                "shift_id": active["shift_id"],
                "issue_type": info.get("issue_type", "operational_issue"),
                "description": problem_text,
                "action_taken": action_val,
                "result": res_val,
                "cause": cause_val,
                "rag_recommendation": rag_rec,
                "rag_verified": rag_ver,
                "rag_source": rag_src,
            }
            self._session_ctx.update(session_id, line_id, awaiting_action_for_issue=False, pending_problem_info=None)
            prompt = self._prepare_confirmation_prompt("LOG_SHIFT_ISSUE", params)
            self._actions.set(session_id, line_id, "LOG_SHIFT_ISSUE", params, prompt)
            return prompt

        # 1.5 Check for explicit issue ID alone (without resolution or action language)
        explicit_id = query_router.extract_issue_id(question)
        action_taken, rag_rec = query_router.extract_operator_action(question)
        cause = query_router.extract_resolution_cause(question)
        result = query_router.extract_resolution_result(question)
        has_res_action = bool(action_taken or rag_rec or result or any(w in question.lower() for w in ("resolve", "resolved", "solve", "theek", "thik", "fix", "close", "khatam", "band", "done")))

        if explicit_id is not None and not has_res_action:
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"Cannot access issue because no active shift is running on {line_id}."
            shift_issues = shift_store.list_shift_issues(shift_id=active["shift_id"], limit=50)
            target_issue = next((iss for iss in shift_issues if iss["issue_id"] == explicit_id), None)
            if not target_issue:
                all_issues = shift_store.list_shift_issues(limit=50)
                target_issue = next((iss for iss in all_issues if iss["issue_id"] == explicit_id), None)
                if not target_issue:
                    return f"Issue #{explicit_id} not found."
                if target_issue["shift_id"] != active["shift_id"]:
                    return f"Issue #{explicit_id} belongs to a different shift and cannot be modified."
            if target_issue.get("result") == "resolved":
                return f"Issue #{explicit_id} is already resolved."
            self._session_ctx.update(session_id, line_id, last_issue_id=explicit_id)
            return f"What would you like to do with Issue #{explicit_id}?"

        # 2. Check if the question is an action intent
        if routed is not None and hasattr(routed, "intent"):
            classified_intent = routed.intent
        elif explicit_id is not None and has_res_action:
            classified_intent = query_router.Intent.RESOLVE_SHIFT_ISSUE
        else:
            classified_intent, _ = query_router.classify(question)
        action_intents = {
            query_router.Intent.SHIFT_START,
            query_router.Intent.SHIFT_END,
            query_router.Intent.LOG_WALL_PRODUCTION,
            query_router.Intent.LOG_SHIFT_ISSUE,
            query_router.Intent.LOG_SHIFT_HANDOVER,
            query_router.Intent.RESOLVE_SHIFT_ISSUE,
        }

        # Route action statements and resolution statements to RESOLVE_SHIFT_ISSUE
        # when an active/current unresolved issue exists, without requiring an explicit issue ID
        if classified_intent not in action_intents and (action_taken or result or rag_rec):
            active_shift = shift_store.get_active_shift_for_line(line_id)
            if active_shift:
                unresolved = shift_store.list_shift_issues(shift_id=active_shift["shift_id"], unresolved_only=True)
                if unresolved or (sess.last_issue_id and any(iss["issue_id"] == sess.last_issue_id for iss in shift_store.list_shift_issues(shift_id=active_shift["shift_id"]))):
                    classified_intent = query_router.Intent.RESOLVE_SHIFT_ISSUE

        if classified_intent in action_intents:
            # The router extracts the name; resolving it is the assistant's job,
            # because only the assistant talks to the operators table.
            if routed is None or getattr(routed, "intent", None) != classified_intent:
                routed = query_router.resolve(classified_intent, question)

            if classified_intent == query_router.Intent.LOG_WALL_PRODUCTION:
                wn = routed.facts.get("wall_number") if getattr(routed, "facts", None) else None
                if wn is None:
                    return "Which wall number was completed? Please specify the wall number to record production."

            if classified_intent == query_router.Intent.LOG_SHIFT_ISSUE:
                clean_q = query_router.strip_wake_words(question).strip().lower().rstrip(".!?")
                if clean_q in ("i have a problem", "there is a problem", "we have a problem"):
                    active = shift_store.get_active_shift_for_line(line_id)
                    if not active:
                        return f"Cannot report issue because no active shift is running on {line_id}."
                    op = shift_store.get_operator(active["operator_id"]) if active else None
                    name = op["name"] if op else (active["operator_id"] if active else "Operator")
                    self._session_ctx.update(
                        session_id,
                        line_id,
                        awaiting_problem_details=True,
                        pending_problem_info=None,
                        awaiting_action_for_issue=False,
                    )
                    return f"{name}, what problem are you facing?"

            if classified_intent == query_router.Intent.RESOLVE_SHIFT_ISSUE:
                params = dict(routed.facts)
                target_id = params.get("target_issue_id") or explicit_id
                domain_kws = params.get("domain_keywords", []) or query_router.extract_domain_keywords(question)
                action_taken = params.get("action_taken") or action_taken
                rag_recommended = params.get("rag_recommended", False) or rag_rec
                cause = params.get("cause") or cause
                result = params.get("result") or result

                active = shift_store.get_active_shift_for_line(line_id)
                if not active:
                    return f"Cannot resolve issue because no active shift is running on {line_id}."

                unresolved = shift_store.list_shift_issues(shift_id=active["shift_id"], unresolved_only=True)

                selected_id = None
                # 1. Explicit issue ID from question (must exist and not be already resolved)
                if target_id:
                    shift_issues = shift_store.list_shift_issues(shift_id=active["shift_id"], limit=50)
                    target_issue = next((iss for iss in shift_issues if iss["issue_id"] == target_id), None)
                    if not target_issue:
                        all_issues = shift_store.list_shift_issues(limit=50)
                        target_issue = next((iss for iss in all_issues if iss["issue_id"] == target_id), None)
                        if not target_issue:
                            return f"Issue #{target_id} not found."
                        if target_issue["shift_id"] != active["shift_id"]:
                            return f"Issue #{target_id} belongs to a different shift and cannot be modified."

                    if target_issue.get("result") == "resolved":
                        return f"Issue #{target_id} is already resolved."
                    selected_id = target_id

                # 2. Session last_issue_id (only if on active shift and unresolved)
                if not selected_id and sess.last_issue_id:
                    if any(iss["issue_id"] == sess.last_issue_id for iss in unresolved):
                        selected_id = sess.last_issue_id

                # 3. Unique domain keyword match
                if not selected_id and domain_kws and unresolved:
                    matches = [iss for iss in unresolved if any(kw in iss.get("description", "").lower() for kw in domain_kws)]
                    if len(matches) == 1:
                        selected_id = matches[0]["issue_id"]

                # 4. Exactly one unresolved issue on active shift
                if not selected_id:
                    if len(unresolved) == 1:
                        selected_id = unresolved[0]["issue_id"]
                    elif len(unresolved) > 1:
                        cand_ids = [iss["issue_id"] for iss in unresolved]
                        sess.awaiting_disambiguation = {
                            "candidate_issues": unresolved,
                            "candidate_issue_ids": cand_ids,
                            "pending_params": params,
                        }
                        self._session_ctx.update(session_id, line_id, awaiting_disambiguation=sess.awaiting_disambiguation)
                        cand_strs = ", ".join(f"Issue #{iss['issue_id']} ({iss.get('description') or iss.get('issue_type')})" for iss in unresolved)
                        return f"Multiple unresolved issues found on this shift: {cand_strs}. Which issue would you like to update?"
                    else:
                        return f"No unresolved issues found on the current shift for {line_id}."

                params["target_issue_id"] = selected_id
                if rag_recommended:
                    if sess.last_rag_recommendation:
                        action_taken = f"Applied Kelvin recommendation: {sess.last_rag_recommendation}"
                    else:
                        action_taken = "Applied Kelvin recommendation"
                    params["action_taken"] = action_taken

                if action_taken and not result:
                    self._session_ctx.update(
                        session_id,
                        line_id,
                        awaiting_outcome={
                            "target_issue_id": selected_id,
                            "action_taken": action_taken,
                            "cause": cause,
                            "params": params,
                        },
                    )
                    return f"Recorded action '{action_taken}' for Issue #{selected_id}. Did this resolve the issue and resume printing?"

                if not result and not action_taken:
                    return f"What would you like to do with Issue #{selected_id}?"

                prompt = self._prepare_confirmation_prompt("RESOLVE_SHIFT_ISSUE", params)
                self._actions.set(session_id, line_id, "RESOLVE_SHIFT_ISSUE", params, prompt)
                return prompt
            spoken_name = (routed.facts or {}).get("spoken_name") or query_router.extract_operator_name(question)

            op = self._resolve_operator(operator_id, session_id, line_id, spoken_name=spoken_name)

            # Speech-to-text can split a name apart ("Anand here" arrived as
            # "On and Here"), leaving normal extraction with nothing. For a
            # shift start where the operator DID introduce themselves, try to
            # recover them from the name slot -- unique match only.
            name_offered = (
                not op
                and not spoken_name
                and classified_intent == query_router.Intent.SHIFT_START
                and bool(shift_store.name_slot_candidates(question))
            )
            if name_offered:
                op = shift_store.find_operator_in_text(question)
                if not op:
                    return "I didn't catch your name. Please say your name again and start your shift."

            if classified_intent == query_router.Intent.SHIFT_START:
                is_self_voice = bool(re.search(r"\b(?:my\s+shift|my\s+attendance|this\s+is|\bhere\b|main\b|hoon\b|hu\b|speaking\b)", question.lower()))
                if spoken_name:
                    matches = shift_store.find_matching_operators(spoken_name, active_only=False)
                    if len(matches) > 1:
                        return f"Multiple employees found matching '{spoken_name.title()}'. Please specify the full name or employee ID."
                    elif len(matches) == 0 and not op:
                        if is_self_voice:
                            params = dict(routed.facts or {})
                            params["operator_name"] = spoken_name.title()
                            params["create_operator"] = True
                            params["line_id"] = line_id
                            prompt = self._prepare_confirmation_prompt("SHIFT_START", params)
                            self._actions.set(session_id, line_id, "SHIFT_START", params, prompt)
                            return prompt
                        return f"I could not find an employee named {spoken_name.title()} in the database. No shift was started. Please register the employee first."
                    elif matches:
                        op = matches[0]

                if not op:
                    if not spoken_name:
                        return "No active operator is registered on this line. Please specify which employee you would like to start a shift for."
                    return "Please specify which employee you would like to start a shift for."
                if not op.get("active"):
                    if is_self_voice:
                        return f"I could not find an operator named {spoken_name.title()}."
                    return f"Employee {op['name']} is inactive and cannot start a shift."

                # Check for existing active shift to prevent duplicates
                existing = shift_store.get_active_attendance(operator_id=op["operator_id"])
                if existing:
                    started = shift_store.format_clock_time(existing.get("shift_start"))
                    return f"{op['name']} already has an active shift. Started at {started}."

                # Explicit unambiguous 3rd-person shift-start command executes directly; self voice requires confirmation
                is_explicit = not is_self_voice and (bool(spoken_name) or not (routed.facts or {}).get("requires_confirmation", True))
                if is_explicit:
                    started_at = self._server_now()
                    moment = datetime.datetime.fromtimestamp(started_at)
                    shift_date = moment.date().isoformat()
                    try:
                        rec = shift_store.create_attendance_record(
                            operator_id=op["operator_id"],
                            employee_name=op["name"],
                            attendance_date=shift_date,
                        )
                    except ValueError as val_err:
                        return str(val_err)
                    except Exception as exc:
                        log.error("Failed to create attendance record: %s", exc)
                        return f"Database error: Could not record shift start for {op['name']}."

                    # Verify persistence read-back
                    verified = shift_store.get_attendance_record(rec["id"])
                    if not verified or verified.get("status") != "ACTIVE":
                        return f"Database error: Failed to verify attendance persistence for {op['name']}."

                    hour = moment.hour
                    shift_type = "Morning" if 6 <= hour < 14 else "Afternoon" if 14 <= hour < 22 else "Night"
                    try:
                        shift_store.start_shift(
                            operator_id=op["operator_id"],
                            shift_date=shift_date,
                            shift_type=shift_type,
                            line_id=line_id,
                            start_time=started_at,
                        )
                    except Exception as exc:
                        log.warning("Could not start machine shift: %s", exc)

                    start_clock = shift_store.format_clock_time(rec["shift_start"])
                    self._session_ctx.update(
                        session_id,
                        line_id,
                        current_operator_id=op["operator_id"],
                        current_operator_name=op["name"],
                    )
                    return f"{op['name']}'s shift has started at {start_clock}."

                params = dict(routed.facts)
                params["operator_id"] = op["operator_id"]
                params["operator_name"] = op.get("name") or op["operator_id"]
                params["line_id"] = line_id
                prompt = self._prepare_confirmation_prompt(classified_intent.value, params)
                self._actions.set(session_id, line_id, classified_intent.value, params, prompt)
                return prompt

            if classified_intent == query_router.Intent.SHIFT_END:
                if spoken_name:
                    matches = shift_store.find_matching_operators(spoken_name, active_only=False)
                    if len(matches) > 1:
                        return f"Multiple employees found matching '{spoken_name.title()}'. Please specify the full name or employee ID."
                    elif len(matches) == 0 and not op:
                        return f"I could not find an employee named {spoken_name.title()} in the database."
                    elif matches:
                        op = matches[0]

                if not op:
                    return "Please specify which employee's shift you would like to end."

                active_rec = shift_store.get_active_attendance(operator_id=op["operator_id"])
                if not active_rec:
                    return f"No active shift found for {op['name']}."

                is_self_voice = bool(re.search(r"\b(?:my\s+shift|my\s+attendance|this\s+is|\bhere\b|main\b|hoon\b|hu\b|speaking\b)", question.lower()))
                has_pronoun = any(w in question.lower().split() for w in ("his", "her", "their", "he", "him"))
                is_explicit = not is_self_voice and (bool(spoken_name) or has_pronoun or not (routed.facts or {}).get("requires_confirmation", True))
                if is_explicit:
                    try:
                        completed_rec = shift_store.checkout_attendance_record(operator_id=op["operator_id"])
                    except Exception as exc:
                        log.error("Failed to checkout attendance record: %s", exc)
                        return f"Database error: Could not end shift for {op['name']}."

                    machine_shift = shift_store.get_active_shift_for_line(line_id)
                    if machine_shift and machine_shift.get("operator_id") == op["operator_id"]:
                        shift_store.end_shift(machine_shift["shift_id"])
                    end_clock = shift_store.format_clock_time(completed_rec.get("shift_end"))
                    dur_str = shift_store.format_worked_spoken(completed_rec.get("worked_seconds", 0))
                    self._session_ctx.update(
                        session_id,
                        line_id,
                        current_operator_id=op["operator_id"],
                        current_operator_name=op["name"],
                    )
                    return f"{op['name']}'s shift has ended at {end_clock}. Duration: {dur_str}."

                params = dict(routed.facts)
                params["operator_id"] = op["operator_id"]
                params["operator_name"] = op.get("name") or op["operator_id"]
                params["line_id"] = line_id
                prompt = self._prepare_confirmation_prompt(classified_intent.value, params)
                self._actions.set(session_id, line_id, classified_intent.value, params, prompt)
                return prompt

            if not op:
                if spoken_name:
                    return f"I could not find an operator named {spoken_name.title()}."
                return "No active operator is registered on this line. Please select your operator profile on the terminal."

            params = dict(routed.facts)
            params["operator_id"] = op["operator_id"]
            params["operator_name"] = op.get("name") or op["operator_id"]
            params["line_id"] = line_id
            prompt = self._prepare_confirmation_prompt(classified_intent.value, params)
            self._actions.set(session_id, line_id, classified_intent.value, params, prompt)
            return prompt

        # 3. Deterministic factual intents (resolved directly without Claude/Gemini)
        deterministic_factual_intents = {
            query_router.Intent.GET_COMPLETED_PRODUCTION,
            query_router.Intent.GET_REMAINING_PRODUCTION,
            query_router.Intent.GET_TOTAL_TARGET,
            query_router.Intent.GET_PRINT_RATE,
            query_router.Intent.GET_CURRENT_LAYER,
            query_router.Intent.GET_LAYER_WIDTH,
            query_router.Intent.GET_LAYER_HEIGHT,
            query_router.Intent.GET_WALL_DIMENSIONS,
            query_router.Intent.GET_MATERIAL_USAGE,
            query_router.Intent.GET_MATERIAL_REMAINING,
            query_router.Intent.GET_MATERIAL_RATE,
            query_router.Intent.GET_ELAPSED_TIME,
            query_router.Intent.GET_ETA,
            query_router.Intent.GET_DEADLINE_STATUS,
            query_router.Intent.GET_24_HOUR_PRODUCTION,
            query_router.Intent.GET_SHIFT_PRODUCTION,
            query_router.Intent.GET_OPERATOR,
            query_router.Intent.GET_MACHINE_STATUS,
            query_router.Intent.GET_ROBOT_STATUS,
            query_router.Intent.GET_CAMERA_STATUS,
            query_router.Intent.GET_COMMUNICATION_STATUS,
            query_router.Intent.GET_PRINT_QUALITY,
            query_router.Intent.GET_RECENT_PROBLEMS,
            query_router.Intent.GET_RECENT_EVENTS,
            query_router.Intent.WAKE_WORD,
            query_router.Intent.GET_ACTIVE_SHIFT,
            query_router.Intent.GET_PREVIOUS_SHIFT,
            query_router.Intent.GET_SHIFT_BY_DATE,
            query_router.Intent.GET_SHIFT_BY_OPERATOR,
            query_router.Intent.GET_WALL_HISTORY,
            query_router.Intent.GET_SHIFT_ISSUES,
            query_router.Intent.GET_HANDOVER_STATUS,
            query_router.Intent.GET_SITE_PEOPLE_COUNT,
            query_router.Intent.GET_SITE_DETECTIONS,
            query_router.Intent.GET_ATTENDANCE_HOURS,
            query_router.Intent.GET_ATTENDANCE_STATUS,
            query_router.Intent.GET_ATTENDANCE_OVERTIME,
            query_router.Intent.GET_PRINT_PROGRESS,
            query_router.Intent.GET_OPERATOR_WORKING_DAYS,
            query_router.Intent.RECORD_WALLS_COMPLETED,
            query_router.Intent.GET_WALL_PRODUCTION,
            query_router.Intent.CORRECT_WALL_PRODUCTION,
        }
        if classified_intent in deterministic_factual_intents:
            active = shift_store.get_active_shift_for_line(line_id)
            context = {
                "session_id": session_id,
                "line_id": line_id,
                "operator_id": operator_id or (active.get("operator_id") if active else None) or sess.current_operator_id,
                "shift_id": (active.get("shift_id") if active else None) or sess.current_shift_id,
                "last_issue_id": sess.last_issue_id or sess.current_issue_id,
            }
            if (
                routed is None
                or getattr(routed, "intent", None) != classified_intent
                or (getattr(routed, "facts", None) and routed.facts.get("needs_clarification") and routed.facts.get("clarification_type") == "operator")
            ):
                routed = query_router.route(question, context=context)
            return query_router.summarise(routed)

        # 4. Troubleshooting and knowledge queries (RAG-first decision flow with Gemini fallback)
        if classified_intent == query_router.Intent.KNOWLEDGE_QUERY or attachments:
            from agent import troubleshooter
            res = troubleshooter.troubleshoot(
                question,
                attachments=attachments,
                session_id=session_id,
                line_id=line_id,
                operator_id=operator_id,
            )
            self._last_troubleshoot_result = res
            self._last_gemini_called = bool(res.gemini_called)
            if res.rag_found:
                self._session_ctx.update(session_id, line_id, last_rag_recommendation=res.answer)
            return res.answer

        # 5. Standard queries (LLM or deterministic router)
        if self.use_llm:
            answer = self._ask_llm(question)
            if answer:
                self._last_gemini_called = True
                return answer
            log.info("Falling back to the local router for this question.")
        self._last_gemini_called = False
        answer = self._answer_locally(question, routed=routed)
        return answer

    def ask_and_speak(
        self,
        question: str,
        session_id: str = "default_terminal",
        line_id: str = "line_1",
        operator_id: Optional[str] = None,
        routed: Optional[Any] = None,
    ) -> str:
        """Answers and speaks the reply; speech failure never breaks the answer."""
        answer = self.ask(question, session_id=session_id, line_id=line_id, operator_id=operator_id, routed=routed)
        try:
            from agent.voice import speak

            speak(answer)
        except Exception as exc:
            log.warning("Text-to-speech unavailable: %s", exc)
        return answer


def tool_debug(question: str) -> Dict[str, Any]:
    """Which tool a question routes to, and what it returns. Used by tests/CLI."""
    tool = Assistant.route(question)
    payload = search_docs(question) if tool == "search_docs" else _TOOLS_BY_NAME[tool]()
    return {"question": question, "tool": tool, "result": payload}


if __name__ == "__main__":  # quick manual check
    print(json.dumps(tool_debug("kitni wall print ho gayi hai?"), indent=2, default=str))
