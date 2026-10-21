"""
Turns a retrieved documentation record into English a robot can speak.

The RAG index stores structured Markdown -- "## Issue: ...", "**Problem:** ...",
"**Solution / Corrective Action:** ..." -- because that shape is what keeps one
issue in one chunk and makes provenance auditable. None of it belongs in an
operator's ear. Before this module the router returned the raw chunk, so the
assistant said:

    "From operational_rules.md, Issue: Cleaning Intervals - Accelerator Hole,
     Pipe, Hopper and Pump Front: ## Issue: Cleaning Intervals ... **Problem:**
     Four different parts are cleaned on four different intervals, counted in
     different units. They sound simila"

...truncated mid-word, with the actual answer (every five layers, big Allen key,
never wire) discarded below the cut.

This module SELECTS and REORDERS fields that already exist. It never adds a
fact, never resolves a conflict, and never promotes a possible cause to a
confirmed one -- the qualifiers in the source labels are carried through into
the spoken sentence on purpose.

Records that carry no field labels (the hand-written manual and troubleshooting
guide) fall back to cleaned prose, which is what they already were.
"""

import re
from typing import Dict, List, Optional

# "**Solution / Corrective Action (CONFLICT - DO NOT ASSUME):** text"
#  -> name="Solution / Corrective Action", qualifier="CONFLICT - DO NOT ASSUME"
_FIELD_RE = re.compile(
    r"^\*\*(?P<name>[^*():]+?)(?:\s*\((?P<qualifier>[^)]*)\))?:\*\*\s*(?P<value>.*)$"
)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*(?:Issue:\s*)?(?P<title>.*?)\s*$")

# Fields never spoken. Source and Verification Status are provenance for an
# auditor, not an answer; the question lists exist only so retrieval can match
# an operator's phrasing.
_SILENT_FIELDS = {
    "source",
    "english questions",
    "hinglish questions",
    "agent follow-up",
    "important parameters",  # folded into the solution sentence when numeric
}

# Qualifiers that must survive into speech.
_UNCONFIRMED = (
    "POSSIBLE", "LIKELY", "DISPUTED", "NOT CONFIRMED",
    "OPERATOR REPORTED", "OPERATOR_REPORTED", "UNVERIFIED", "NEEDS VALIDATION", "OBSERVATION ONLY"
)
_CONFLICT = ("CONFLICT", "DO NOT ASSUME", "MUST CONFIRM", "CONFLICTING")
_TEMPORARY = ("TEMPORARY", "WORKAROUND")
_AUTHORIZATION = (
    "AUTHORIZATION REQUIRED", "SAFETY REVIEW REQUIRED", "ENGINEERING APPROVAL",
    "UNAPPROVED WORKAROUND", "UNAPPROVED", "REQUIRES AUTHORIZATION"
)

# The operator asked "why", so a cause is worth speaking. Otherwise a how-to
# answer should lead with the action, not with a hypothesis.
_CAUSE_WANTED = (
    "why", "cause", "reason", "kyun", "kyu", "kaise hua", "what causes",
    "what is causing", "problem kya", "kya wajah",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# How many sentences a spoken answer may run to. Bounded by SENTENCES, never by
# a character count, so an answer can never stop mid-word.
_MAX_SENTENCES = 5


def _strip_markdown(text: str) -> str:
    """Removes Markdown that would otherwise be read aloud verbatim."""
    text = re.sub(r"```[\s\S]*?```", " ", text)          # fenced code
    text = re.sub(r"`([^`]*)`", r"\1", text)             # inline code
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)  # headings
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)       # bold
    text = re.sub(r"(?<!\w)\*(?!\s)([^*]*?)(?<!\s)\*(?!\w)", r"\1", text)  # italics
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"[*#`]+", "", text)                   # any stragglers
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _clip(text: str, limit: int = _MAX_SENTENCES) -> str:
    """Keeps whole sentences only -- never cuts a word."""
    parts = _sentences(text)
    if not parts:
        return ""
    kept = parts[:limit]
    out = " ".join(kept)
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


def parse_record(text: str) -> Dict[str, Dict[str, str]]:
    """
    Splits a structured record into {field_name_lower: {value, qualifier}}.

    Returns {} for a record with no field labels, which is how the caller knows
    to fall back to cleaned prose.
    """
    fields: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None

    for line in text.splitlines():
        match = _FIELD_RE.match(line.strip())
        if match:
            current = match.group("name").strip().lower()
            fields[current] = {
                "value": match.group("value").strip(),
                "qualifier": (match.group("qualifier") or "").strip(),
            }
            continue
        # Continuation: the indented "  - ..." lines under a solution list.
        stripped = line.strip()
        if current and stripped and not stripped.startswith("#"):
            joined = (fields[current]["value"] + " " + re.sub(r"^[-*+]\s*", "", stripped)).strip()
            fields[current]["value"] = joined

    return fields


def record_title(text: str) -> str:
    """The issue name from the heading, with the 'Issue:' label removed."""
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            match = _HEADING_RE.match(line)
            if match:
                return _strip_markdown(match.group("title"))
    return ""


def _has(qualifier: str, needles) -> bool:
    upper = qualifier.upper()
    return any(n in upper for n in needles)


def speak_record(text: str, question: str = "") -> str:
    """
    One retrieved record, spoken.

    Order is operational: what to do, then the safety warning, then -- only if
    the operator asked why -- the cause with its confidence intact.
    """
    fields = parse_record(text)
    if not fields:
        # Label-free prose (the hand-written manual and troubleshooting guide).
        # Drop the heading line: spoken aloud, a title reads as a fragment.
        body = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("#")
            and not line.lstrip().lower().startswith(("sources:", "source:"))
        )
        cleaned = _strip_markdown(body)
        return _clip(cleaned) if cleaned else ""

    asked_why = any(term in (question or "").lower() for term in _CAUSE_WANTED)
    pieces: List[str] = []

    solution = fields.get("solution / corrective action") or fields.get("solution")
    safety = fields.get("safety / important notes")
    cause = next(
        (fields[k] for k in fields if k.startswith("cause")),
        None,
    )
    problem = fields.get("problem")
    status = (fields.get("verification status") or {}).get("value", "")
    params = (fields.get("important parameters") or {}).get("value", "")

    conflict = (
        (solution and _has(solution["qualifier"], _CONFLICT))
        or _has(status, _CONFLICT)
    )
    temporary = (
        (solution and _has(solution["qualifier"], _TEMPORARY))
        or _has(status, _TEMPORARY)
    )
    authorization = (
        (_has(safety["value"], _AUTHORIZATION) if safety else False)
        or (solution and _has(solution["qualifier"], _AUTHORIZATION))
        or _has(status, _AUTHORIZATION)
    )

    # A conflict is stated as a conflict, before anything that might read as a
    # recommendation. The source deliberately does not resolve these.
    if conflict:
        pieces.append("The available site data is conflicting on this.")

    def spoken_cause() -> str:
        if not (cause and cause["value"]):
            return ""
        body = _strip_markdown(cause["value"])
        if not _has(cause["qualifier"], _UNCONFIRMED):
            return body
        # The source often already opens with "Possible causes include ...";
        # prefixing again produced "A possible cause is possible causes ...".
        if body.lower().startswith(("possible", "a possible")):
            lead = body
        else:
            lead = f"A possible cause is {body[0].lower()}{body[1:]}"
        if "not confirmed" not in lead.lower():
            lead += " This cause is not confirmed."
        return lead

    if asked_why:
        told = spoken_cause()
        if told:
            pieces.append(told)

    if solution and solution["value"]:
        pieces.append(_strip_markdown(solution["value"]))
    elif problem and problem["value"]:
        pieces.append(_strip_markdown(problem["value"]))

    if temporary and "temporar" not in " ".join(pieces).lower():
        pieces.append("This is a temporary workaround, not a permanent fix.")

    if safety and safety["value"]:
        note = _strip_markdown(safety["value"])
        if note.lower() not in ("none specified in the source.", "none specified in the source"):
            pieces.append(note)

    if authorization and "authorization" not in " ".join(pieces).lower():
        pieces.append(
            "This is a machine-specific procedure and requires authorization "
            "before it is carried out."
        )

    # Numeric parameters are the operator's takeaway, so keep them when the
    # solution sentence did not already carry them.
    if params and params.lower().startswith(("none", "not applicable")) is False:
        cleaned = _strip_markdown(params)
        if cleaned and cleaned.split(".")[0] not in " ".join(pieces):
            pieces.append(cleaned)

    spoken = _clip(" ".join(p for p in pieces if p))
    return spoken or _clip(_strip_markdown(text))


def speak_knowledge(knowledge: Optional[dict], question: str = "") -> Optional[str]:
    """
    The top retrieved record, spoken. None when nothing was retrieved, so the
    caller keeps its own "not in the documentation" wording.
    """
    if not knowledge or not knowledge.get("found"):
        return None
    results = knowledge.get("results") or []
    if not results:
        return None
    for r in results:
        spoken = speak_record(r.get("text", ""), question)
        if spoken:
            return spoken
    return None
