"""
Central query router for the Kelvin6K voice assistant.

Classifies a natural-language question (English, Hindi or Hinglish) into an
intent, then RESOLVES that intent against the real system -- the SQLite
database, the live machine state, the vision pipeline and the RAG index.

The critical property: resolvers return FACTS, never prose. The language model
formats those facts into a sentence but can never originate a number. If a fact
is unavailable, the resolver says so explicitly and the assistant reports it as
unavailable rather than guessing.

Intent coverage matches the operator's vocabulary, including Hinglish
("kitni walls print ho gayi hain", "material kitna use hua", "camera chal raha
hai"), so exact phrasing is never required.
"""

import datetime
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent import db, metrics, shift_store
from agent.logging_setup import get_logger
from agent.rag.embeddings import _STOPWORDS, tokenize
from agent.rag.retriever import search_documentation
from agent.services import machine_state as ms
from agent.spoken import speak_knowledge, parse_record

log = get_logger("ROUTER")


class Intent(str, Enum):
    GET_COMPLETED_PRODUCTION = "GET_COMPLETED_PRODUCTION"
    GET_REMAINING_PRODUCTION = "GET_REMAINING_PRODUCTION"
    GET_TOTAL_TARGET = "GET_TOTAL_TARGET"
    GET_PRINT_RATE = "GET_PRINT_RATE"
    GET_CURRENT_LAYER = "GET_CURRENT_LAYER"
    GET_LAYER_WIDTH = "GET_LAYER_WIDTH"
    GET_LAYER_HEIGHT = "GET_LAYER_HEIGHT"
    GET_WALL_DIMENSIONS = "GET_WALL_DIMENSIONS"
    GET_MATERIAL_USAGE = "GET_MATERIAL_USAGE"
    GET_MATERIAL_REMAINING = "GET_MATERIAL_REMAINING"
    GET_MATERIAL_RATE = "GET_MATERIAL_RATE"
    GET_ELAPSED_TIME = "GET_ELAPSED_TIME"
    GET_ETA = "GET_ETA"
    GET_DEADLINE_STATUS = "GET_DEADLINE_STATUS"
    GET_24_HOUR_PRODUCTION = "GET_24_HOUR_PRODUCTION"
    GET_SHIFT_PRODUCTION = "GET_SHIFT_PRODUCTION"
    GET_OPERATOR = "GET_OPERATOR"
    GET_MACHINE_STATUS = "GET_MACHINE_STATUS"
    GET_ROBOT_STATUS = "GET_ROBOT_STATUS"
    GET_CAMERA_STATUS = "GET_CAMERA_STATUS"
    GET_COMMUNICATION_STATUS = "GET_COMMUNICATION_STATUS"
    GET_PRINT_QUALITY = "GET_PRINT_QUALITY"
    GET_RECENT_PROBLEMS = "GET_RECENT_PROBLEMS"
    GET_RECENT_EVENTS = "GET_RECENT_EVENTS"
    GET_SITE_PEOPLE_COUNT = "GET_SITE_PEOPLE_COUNT"
    GET_SITE_DETECTIONS = "GET_SITE_DETECTIONS"
    GET_ATTENDANCE_HOURS = "GET_ATTENDANCE_HOURS"
    GET_ATTENDANCE_STATUS = "GET_ATTENDANCE_STATUS"
    GET_ATTENDANCE_OVERTIME = "GET_ATTENDANCE_OVERTIME"
    GET_PRINT_PROGRESS = "GET_PRINT_PROGRESS"
    GET_OPERATOR_WORKING_DAYS = "GET_OPERATOR_WORKING_DAYS"
    WAKE_WORD = "WAKE_WORD"
    KNOWLEDGE_QUERY = "KNOWLEDGE_QUERY"
    COMBINED_QUERY = "COMBINED_QUERY"
    UNKNOWN_QUERY = "UNKNOWN_QUERY"

    # --- Phase 2C Action Intents (Write) ---
    SHIFT_START = "SHIFT_START"
    SHIFT_END = "SHIFT_END"
    LOG_WALL_PRODUCTION = "LOG_WALL_PRODUCTION"
    LOG_SHIFT_ISSUE = "LOG_SHIFT_ISSUE"
    LOG_SHIFT_HANDOVER = "LOG_SHIFT_HANDOVER"
    RESOLVE_SHIFT_ISSUE = "RESOLVE_SHIFT_ISSUE"

    # --- Phase 2C Historical Query Intents (Read) ---
    GET_ACTIVE_SHIFT = "GET_ACTIVE_SHIFT"
    GET_PREVIOUS_SHIFT = "GET_PREVIOUS_SHIFT"
    GET_SHIFT_BY_DATE = "GET_SHIFT_BY_DATE"
    GET_SHIFT_BY_OPERATOR = "GET_SHIFT_BY_OPERATOR"
    GET_WALL_HISTORY = "GET_WALL_HISTORY"
    GET_SHIFT_ISSUES = "GET_SHIFT_ISSUES"
    GET_HANDOVER_STATUS = "GET_HANDOVER_STATUS"

    # --- Employee Wall Production Tracking ---
    RECORD_WALLS_COMPLETED = "RECORD_WALLS_COMPLETED"
    GET_WALL_PRODUCTION = "GET_WALL_PRODUCTION"
    CORRECT_WALL_PRODUCTION = "CORRECT_WALL_PRODUCTION"


UNAVAILABLE_MESSAGE = "That live data is currently unavailable."


@dataclass
class RouterResult:
    intent: Intent
    facts: Dict[str, Any] = field(default_factory=dict)
    knowledge: Optional[Dict[str, Any]] = None
    available: bool = True
    reason: str = ""
    matched_terms: List[str] = field(default_factory=list)
    question: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "facts": self.facts,
            "knowledge": self.knowledge,
            "available": self.available,
            "reason": self.reason,
            "matched_terms": self.matched_terms,
        }


# ---------------------------------------------------------------------------
# Intent classification
#
# Each intent lists trigger phrases in English AND Hindi/Hinglish. Matching is
# substring-based on a normalised question, scored by how many (and how long)
# the matches are, so natural paraphrases work without exact wording.
# ---------------------------------------------------------------------------

# (intent, phrases, priority). Priority breaks ties between overlapping
# phrases: "current layer width" contains both "current layer" and "layer
# width", and the more specific intent must win regardless of phrase length.
# Higher priority = more specific.
_PATTERNS: List[Tuple[Intent, List[str], int]] = [
    # --- Phase 2C Action Intents (Write actions, highest priority) ---
    (Intent.GET_ATTENDANCE_HOURS, [
        "how many hours have i worked", "how many hours has", "how many hours did", "how many hours",
        "how long has he worked", "how long has she worked", "how long have i worked", "how long did he work", "how long did i work",
        "how long has", "how long did",
        "hours worked today", "hours worked", "what's my shift status", "whats my shift status",
        "when did i start today", "am i clocked in", "am i currently working", "my shift status",
        "did i start", "hours today", "kitne ghante kaam kiya", "kitni der kaam kiya", "worked how many hours"
    ], 7),
    (Intent.GET_ATTENDANCE_STATUS, [
        "who is currently on shift", "who is on shift", "who's on shift", "who's currently on shift",
        "who is currently working", "who is working right now", "who is working",
        "who is clocked in", "who is currently clocked in", "how many employees are working",
        "currently on shift", "currently working", "active on shift", "who is on duty",
        "kaun shift pe hai", "kaun kaun shift pe hai", "kaun kaam kar raha hai"
    ], 7),
    (Intent.GET_ATTENDANCE_OVERTIME, [
        "who has the most overtime", "who has overtime", "most overtime", "highest overtime",
        "sabse zyada overtime", "max overtime", "maximum overtime",
        "overtime today", "how much overtime", "extra hours", "overtime", "over time"
    ], 7),
    (Intent.GET_OPERATOR_WORKING_DAYS, [
        "how many working days has", "how many working days did", "how many working days have i completed",
        "how many working days", "working days completed", "number of working days",
        "working days has", "working days did", "working days",
        "kitne din kaam kiya", "kitne working days"
    ], 7),
    (Intent.GET_PRINT_PROGRESS, [
        "what is the current print progress", "what is the print progress", "current print progress",
        "print progress", "printing progress", "progress of print", "progress of printing",
        "progress kya hai", "print kitna progress hua", "printing ka progress",
        "which print job is running", "what print job is running", "which job is running",
        "current print job", "running print job"
    ], 7),
    (Intent.SHIFT_START, [
        "start shift for", "start my shift", "begin my shift", "my shift start",
        "start morning shift", "start afternoon shift", "start night shift",
        "start a shift", "start the shift", "start new shift", "mera shift start",
        "shift start karo", "shift shuru karo", "begin shift", "nayi shift start",
        "shift chalu karo", "shift shuru", "shuru karo", "clock me in", "mark me present",
        "clock in", "start his shift", "start her shift", "start employee shift",
        "i've started work", "ive started work", "i am starting work", "i'm starting work",
        "starting my shift", "start shift"
    ], 8),
    (Intent.SHIFT_END, [
        "end shift for", "end my shift", "close my shift", "stop my shift",
        "band karo mera shift", "meri shift end karo", "meri shift khatam karo",
        "end the shift", "end current shift", "end active shift",
        "shift end karo", "shift khatam karo", "shift close karo", "close shift",
        "close the shift", "shift band karo", "shift complete karo", "shift khatam",
        "clock me out", "clock out", "end his shift", "end her shift",
        "i'm done for today", "im done for today", "i am done for today", "i'm finishing my shift",
        "i've finished work", "ive finished work", "done for today", "end shift"
    ], 8),
    (Intent.LOG_WALL_PRODUCTION, [
        "log wall", "wall complete ho gayi", "record wall", "wall print ho chuki",
        "wall complete hui", "wall entry karo", "wall log karo", "wall complete",
        "log wall production", "record wall production",
        "wall print ho gayi", "wall print ho gai", "wall print kar di", "wall complete kar di",
        "i printed wall", "i completed wall",
        "i completed two walls", "i completed two wall", "i completed 2 walls",
        "i completed one wall", "i completed 1 wall",
        "i completed three walls", "i completed four walls", "i completed five walls",
        "completed two walls", "completed two wall", "completed 2 walls",
        "record wall 1", "record wall 2", "record wall 3", "record wall 4", "record wall 5",
        "log wall 1", "log wall 2", "log wall 3", "log wall 4", "log wall 5",
        "log 1 wall", "log 2 wall", "log 3 wall", "log 4 wall", "log 5 wall",
        "record 1 wall", "record 2 wall", "record 3 wall", "record 4 wall", "record 5 wall",
        "log one wall", "log two walls", "log two wall", "log three walls", "log four walls", "log five walls",
        "record one wall", "record two walls", "record two wall", "record three walls", "record four walls", "record five walls",
        "1 wall log karo", "1 wall record karo", "ek wall log karo", "ek wall record karo",
        "ek wall log", "ek wall record", "log ek wall", "record ek wall",
        "pehli wall log", "pehli wall record", "pehli wall log karo", "pehli wall record karo",
    ], 8),
    # --- Employee Wall Production Tracking ---
    (Intent.CORRECT_WALL_PRODUCTION, [
        "correct the previous record", "correct the last record", "correct previous record",
        "correct previous wall record", "correct last wall record", "correct wall record",
        "update hamadan's last wall entry", "update hamadans last wall entry", "update last wall entry",
        "correct hamadan's last entry", "correct hamadans last entry", "correct last entry",
        "correction: hamadan printed", "correction: it should be", "it should be 3 walls not 4",
        "it should be", "should be 3 walls not 4", "should be 3 walls", "should be 2 walls",
        "hamadan ki last wall entry correct karo", "last wall record correct karo", "wall record correct karo",
        "correction: 3 walls", "correct to 3 walls", "correct to 2 walls", "correct to 4 walls",
        "correct entry to", "correct the record", "wall production correct karo",
        "change last wall entry", "change last wall", "change wall entry",
        "it was 2 walls not 5", "it was 3 walls not 4",
    ], 9),
    (Intent.RECORD_WALLS_COMPLETED, [
        "completed 4 walls; save it", "completed 4 walls save it",
        "completed 3 walls; save it", "completed 3 walls save it",
        "completed 2 walls; save it", "completed 2 walls save it",
        "completed 1 wall; save it", "completed 1 wall save it",
        "completed walls; save it", "completed walls save it",
        "log 2 completed walls for operator", "log completed walls for operator",
        "log 2 completed walls", "log 3 completed walls", "log 4 completed walls", "log 1 completed wall",
        "record 3 walls printed by operator", "record 3 walls printed by", "record walls printed by operator",
        "record 4 walls completed by", "record 2 walls completed by", "record 3 walls completed by",
        "record completed walls", "save it", "completed 4 walls", "completed 3 walls", "completed 2 walls",
        "completed 1 wall", "completed 5 walls", "completed four walls", "completed three walls",
        "completed two walls", "completed one wall",
        "i completed 4 walls", "i completed 3 walls", "i completed 2 walls", "i completed 1 wall",
        "maine 4 wall complete kar di", "maine 3 wall complete kar di", "maine 2 wall complete kar di",
        "maine 1 wall complete kar di", "wall complete kar di, save karo", "wall complete kar di save karo",
        "save 4 walls", "save 3 walls", "save 2 walls", "save 1 wall",
        "record 4 walls", "record 3 walls", "record 2 walls", "record 1 wall",
        "printed 4 walls; save it", "printed 4 walls", "printed 3 walls", "printed 2 walls", "printed 1 wall",
    ], 8),
    (Intent.GET_WALL_PRODUCTION, [
        "how many walls did hamadan print today", "how many walls did hamadan print",
        "how many walls did he print today", "how many walls did she print today",
        "how many walls did he print", "how many walls did she print",
        "total walls completed this week", "total walls completed today", "total walls completed",
        "walls completed this week", "walls completed today", "walls completed for current shift",
        "what is hamadan's total wall count", "what is hamadans total wall count",
        "hamadan total wall count", "hamadan's total wall count", "hamadans total wall count", "hamadan wall count",
        "show wall production for current shift", "show wall production for shift", "show wall production",
        "wall production for current shift", "wall production for shift", "wall production this week", "wall production today",
        "how many walls were printed under job", "how many walls were printed for job", "walls printed under job",
        "how many walls were printed under", "how many walls printed under job",
        "kitni wall print ki hamadan ne aaj", "hamadan ne kitni wall print ki aaj", "hamadan ne kitni wall print ki",
        "is hafte kitni wall print hui", "aaj kitni wall print hui", "current shift wall production",
    ], 8),
    (Intent.LOG_SHIFT_ISSUE, [
        "log issue", "problem record karo", "issue record karo", "downtime record karo",
        "nozzle clog ho gaya", "downtime note karo", "dikkat note karo",
        "log an issue", "record issue", "record problem",
        # Natural problem statements (English and Hinglish)
        "i have a material flow problem", "there is a material flow problem",
        "material flow problem is happening", "we have a material flow problem",
        "material flow problem", "material flow issue",
        "i have a problem", "there is a problem", "we have a problem",
        "i have one problem", "i have a one problem",
        "there is one problem", "there is a one problem",
        "we have one problem", "we have a one problem",
        "problem is happening", "issue is happening",
        "i have an issue", "there is an issue", "we have an issue",
        "facing a problem", "facing an issue",
        "having a problem", "having an issue",
        "problem aa rahi hai", "dikkat aa rahi hai", "samasya aa rahi hai",
        "problem ho rahi hai", "dikkat ho rahi hai", "samasya ho rahi hai",
        "material flow mein dikkat", "material flow problem ho rahi",
        "material flow problem aa rahi",
    ], 8),
    (Intent.LOG_SHIFT_HANDOVER, [
        "create handover", "handover create karo", "shift handover note karo",
        "prepare handover", "handover record karo", "initiate handover",
        "initiate shift handover", "handover this shift", "shift handover karo",
    ], 8),
    (Intent.RESOLVE_SHIFT_ISSUE, [
        "issue resolve ho gaya", "issue resolved", "issue solve ho gaya",
        "issue solve ho gayi", "issue resolved ho gaya", "resolve issue",
        "problem solve ho gayi", "problem solve ho gaya", "problem solved",
        "problem theek ho gayi", "problem theek ho gaya", "problem thik ho gayi",
        "problem thik ho gaya", "nozzle clean kiya", "nozzle clean kar diya",
        "nozzle clean ki", "nozzle clean kar di", "downtime khatam",
        "issue close karo", "close issue", "resolve kar diya",
        "mark as resolved", "mark resolved", "recommendation follow",
        "recommendation follow kiya", "recommendation apply kar di",
        "recommendation apply kar diya",
        "recommend kiya tha woh apply kar diya",
        "recommend kiya tha woh apply",
        "recommend kiya tha apply kar diya",
        # Natural action statements (English and Hindi/Hinglish)
        "i cleaned the nozzle", "cleaned the nozzle", "cleaned nozzle",
        "i checked the pump", "checked the pump", "pump check kiya", "pump check kar diya",
        "i replaced the hose", "replaced the hose", "hose replace kiya", "hose replace kar diya",
        "hose badal diya", "hose badla", "pump check ki", "pump check kar di",
        # Natural resolution statements (English and Hindi/Hinglish)
        "the material flow is normal now", "material flow is normal now", "ab material flow normal hai",
        "the problem is resolved", "problem is resolved", "the issue is resolved", "issue is resolved",
        "problem resolve ho gayi", "problem resolve ho gaya",
        "the issue is fixed", "issue is fixed", "issue fixed",
        "the problem is fixed", "problem is fixed", "problem fixed",
        "issue fix ho gaya", "issue fix ho gayi", "problem fix ho gaya", "problem fix ho gayi",
    ], 8),

    # --- Phase 2C Historical Query Intents ---
    (Intent.GET_PREVIOUS_SHIFT, [
        "previous shift", "pichli shift", "last shift", "pichla shift",
        "shift before this", "previous shift mein kya hua",
        "shift timing kya thi", "shift timing", "previous shift timing",
    ], 7),
    (Intent.GET_SHIFT_BY_DATE, [
        "shift on", "yesterday's shift", "kal ki shift", "aaj subah ki shift",
        "shift date", "shifts on date",
        "who worked on", "who worked on date", "who was on shift on", "who worked",
        "worked on", "shifts on", "shifts on date", "shifts date",
    ], 7),
    (Intent.GET_SHIFT_BY_OPERATOR, [
        "how many hours did", "my shift duration", "shift duration", "hours did i work", "how many hours",
        "kitne ghante kaam kiya", "kitne ghante ki thi", "meri shift kitne ghante", "maine kitne ghante",
        "how many walls did", "how many wall did", "kitni walls print ki", "kitni wall print ki",
        "how many walls have i printed", "how many walls did i print", "how many walls have i completed",
        "how many walls did i complete", "kitni wall print ki maine", "maine kitni wall print ki",
        "how many wall have i printed", "how many wall did i print", "how many wall have i completed",
        "how many wall did i complete", "how many wall did rahul print",
        "how many walls did rahul print", "how many walls printed",
        "how many wall printed",
        "when did", "shift start time", "shift end time", "shift start", "shift end",
        "kab shuru hui", "kab khatam hui",
        "ne apni shift mein kya print", "apni shift mein kya print",
        "ne kya print kiya", "kya print kiya tha", "kya print kiya",
        "shifts by", "operator shifts", "kis shift mein kaam kiya",
        "shifts for operator", "walls print ki", "wall print ki",
        "ne kitni walls", "ne kitni wall", "ne kitna print", "ne kya print",
    ], 8),
    (Intent.GET_WALL_HISTORY, [
        "wall status", "history of wall", "wall history", "wall 1 status",
        "wall 2 status", "wall 3 status", "wall 4 status", "wall 5 status",
        "wall kitni print hui", "wall number", "status of wall",
        "wall history status", "history status", "wall ka status",
    ], 7),
    (Intent.GET_HANDOVER_STATUS, [
        "handover status", "handover notes", "last handover", "pending work from last shift",
        "pichle handover", "handover mein kya tha",
    ], 7),
    (Intent.GET_SHIFT_ISSUES, [
        "shift issues", "problems this shift", "is shift mein kya dikkat",
        "issues today", "downtime this shift", "downtime kitna hua",
        "what issues happened", "all issues",
        "what issues occurred", "issues occurred", "problems occurred",
        "issues during", "problems during",
        "what problem did", "what problems did", "what issue did", "what issues did",
        "problem did", "problems did", "issue did", "issues did",
        "what problem did rahul have", "what problems did rahul have", "what issue did rahul have",
        "ki shift mein kya dikkat", "shift mein kya dikkat", "kya dikkat hui",
        "kya samasya hui", "shift mein kya issue", "kya dikkat aayi",
        # Natural shift problem queries (Bug 2)
        "what problem occurred during my shift", "what problems occurred during my shift",
        "problem occurred during my shift", "problems occurred during my shift",
        "what problem occurred", "what problems occurred",
        "problem occurred",
        "what problem happened during my shift", "what problems happened during my shift",
        "what problem happened", "what problems happened",
        "problem happened during my shift", "problems happened during my shift",
        "problem during my shift", "problems during my shift",
        "problem in my shift", "problems in my shift",
        "what issue occurred during my shift", "what issues occurred during my shift",
        "issue occurred during my shift", "issues occurred during my shift",
        "issue during my shift", "issues during my shift",
        "issue in my shift", "issues in my shift",
        "what happened during my shift",
        "meri shift mein kya problem", "meri shift mein kya dikkat",
        "meri shift mein kya issue", "meri shift mein kya samasya",
        "hamari shift mein kya problem", "hamari shift mein kya dikkat",
        "is shift mein kya problem", "is shift mein kya issue",
        "what caused the issue", "what was the cause", "cause of the issue",
        "issue ki kya wajah thi", "issue ki wajah", "kya wajah thi", "dikkat ki wajah",
        "what action did",
        # Natural first-person action queries (Bug 1)
        "what action did i take", "what actions did i take",
        "what step did i take", "what steps did i take",
        "what did i do", "what action did i take on my shift",
        "maine kya action liya", "maine kya step liya",
        "maine kya kiya", "humne kya kiya", "hamne kya action liya",
        "ne kya action liya", "ne kya action",
        "kya action liya", "action liya tha", "kya step liya",
        "what solution did",
        "ne kya solution", "ne kya kiya", "ke issues", "ne kya fix",
        "verified rag solution", "verified rag", "was there a verified", "verified solution",
        "kya koi verified rag solution", "kya koi verified", "verified solution tha kya",
        "rag verified tha", "rag solution verified",
        "what did kelvin recommend", "kelvin ne kya recommend kiya",
        "kelvin recommendation", "kelvin's recommendation", "kelvin ka solution",
        "rag recommendation", "rag source", "kya recommend kiya tha", "kelvin ne kya recommend",
    ], 8),
    (Intent.GET_ACTIVE_SHIFT, [
        "what shift is currently active", "what shift is active", "which shift is active",
        "shift is currently active", "shift is active",
        "active shift", "kaun si shift chal rahi", "current shift status",
        "which shift is running", "shift details", "active shift details",
        "kiska shift", "kiska shift chal raha", "kaun si shift", "who is on shift",
        "current active shift", "shift chal rahi", "shift chal raha",
        "what is my current shift", "my current shift", "current shift",
        "who worked on line 1", "who worked on line", "who is on line 1", "who is on line",
        "who is working on line 1", "who is working on line",
        "line 1 par kaun hai", "line 1 pe kaun hai", "line 1 par kisne kaam kiya",
        "who was on line 1", "who worked on this line",
    ], 7),

    # --- time windows: most specific, they override production wording ---
    (Intent.GET_24_HOUR_PRODUCTION, [
        "24 hour", "24 hours", "last day", "today's production", "todays production",
        "24 ghante", "aaj kitna", "last 24", "aaj ka production", "today production",
    ], 6),
    (Intent.GET_SHIFT_PRODUCTION, [
        "current shift production", "how much did we print this shift",
        "shift production", "this shift", "shift mein kitna",
        "shift ka production", "shift mein",
    ], 7),
    (Intent.GET_RECENT_EVENTS, [
        "last 10 minutes", "last ten minutes", "recently happened", "what happened",
        "last hour", "recent events", "kya hua", "10 minute", "last few minutes",
    ], 6),

    # --- specific measurements ---
    (Intent.GET_LAYER_WIDTH, [
        "layer width", "width of the layer", "layer ki width", "layer width kitni",
        "chaudai", "bead width", "how wide is the layer", "layer kitni chaudi",
    ], 5),
    (Intent.GET_LAYER_HEIGHT, [
        "layer height", "height of the layer", "layer ki height", "layer height kitni",
        "layer thickness", "how thick is the layer",
    ], 5),
    (Intent.GET_WALL_DIMENSIONS, [
        "wall height", "wall width", "wall dimension", "wall ki height", "wall ka size",
        "dimensions of the wall", "wall thickness",
    ], 5),
    (Intent.GET_MATERIAL_REMAINING, [
        "material remaining", "material left", "material baki", "material bacha",
        "material is remaining", "how much material is left", "kitna material bacha",
        "material kitna bacha",
    ], 5),
    (Intent.GET_MATERIAL_RATE, [
        "material consumption", "consumption rate", "material rate",
        "material kitna use ho raha", "material per hour",
    ], 5),
    (Intent.GET_MATERIAL_USAGE, [
        "material used", "material usage", "material consumed", "material kitna use",
        "kitna material use", "kitna material", "material kitna laga",
        "kitna material laga", "how much material", "material kitna",
    ], 4),

    # --- timing ---
    (Intent.GET_DEADLINE_STATUS, [
        "on schedule", "deadline", "on track", "behind schedule", "schedule par",
        "time par", "late ho",
    ], 5),
    (Intent.GET_ETA, [
        "eta", "when will", "finish time", "kab khatam", "kab tak", "time remaining",
        "how long until", "kitna time lagega", "complete hoga", "kab complete",
    ], 5),
    (Intent.GET_ELAPSED_TIME, [
        "how long has", "elapsed", "running since", "kitni der se", "kab se chal",
        "print been running", "runtime",
    ], 5),
    (Intent.GET_PRINT_RATE, [
        "print rate", "printing rate", "how fast", "speed of printing", "rate kya",
        "printing speed", "kitni tezi",
    ], 5),

    # --- remaining production: must beat generic "how many walls" ---
    (Intent.GET_REMAINING_PRODUCTION, [
        "remaining", "how many are left", "walls left", "baki", "bacha hua",
        "kitni baki", "kitna baki", "still to print", "pending", "are remaining",
        "walls baki", "kitni walls baki",
    ], 4),

    # --- status / quality / problems ---
    (Intent.GET_PRINT_QUALITY, [
        "print quality", "quality", "quality good", "quality kaisi", "defect",
        "is the print good", "print theek",
    ], 4),
    (Intent.GET_CAMERA_STATUS, [
        "camera", "camera working", "camera status", "camera chal raha", "vision system",
    ], 4),
    (Intent.GET_ROBOT_STATUS, [
        "robot status", "robot online", "robot working", "robot ka status", "robot chal",
    ], 4),
    (Intent.GET_COMMUNICATION_STATUS, [
        "communication", "connection status", "comms", "network status",
    ], 4),
    (Intent.GET_RECENT_PROBLEMS, [
        "problem", "problems", "issue", "issues", "error", "alert", "warning",
        "dikkat", "kharabi", "fault", "koi problem", "what is wrong",
    ], 4),
    (Intent.GET_SITE_PEOPLE_COUNT, [
        # English
        "how many people are on site", "how many people are currently on site",
        "how many people are working on site", "how many people are there right now",
        "how many people on site", "how many people are there",
        "how many people", "how many persons", "people on site", "people count", "person count",
        "who is on site", "anyone on site", "count people", "workers on site", "workers visible",
        "how many workers", "how many workers on site", "site people",
        "people in the room", "people visible", "people visible on camera", "people on camera", "workers on camera",
        "how many people are visible on camera", "how many workers are on site",
        "how many persons are on site",
        # Hindi / Hinglish
        "abhi site pe kitne log hain", "abhi site par kitne log hain", "abhi site pe kitne log", "abhi site par kitne log",
        "site par kitne log hain", "site pe kitne log hain", "site par kitne log", "site pe kitne log",
        "kitne log hain", "kitne log dikh rahe hain", "kitne log dikh rahe", "kitne log present hain",
        "kitne log site par", "kitne log site pe", "site pe kaun kaun hai", "site par kaun hai",
        "kitne bande hain", "site par kitne bande",
        "site par kitne workers", "site pe kitne workers", "kitne workers",
        "site par kitne vyakti", "site pe kitne vyakti", "kitne vyakti",
        "camera me kitne log", "camera par kitne log", "camera me kitne",
        "site pe kaun kaun", "site par kaun",
    ], 5),
    (Intent.GET_SITE_DETECTIONS, [
        "what objects are visible", "what objects are detected", "what do you see on camera 2",
        "what is visible on camera 2", "what is detected on camera 2", "camera 2 detections",
        "detected objects on site", "objects on camera 2", "objects detected on camera",
        "objects on site", "what is on camera 2", "what is detected on site",
        "camera 2 pe kya dikh raha hai", "camera 2 me kya dikh raha", "site camera me kya dikh raha",
        "kya kya objects dikh rahe hain", "objects visible on camera", "objects detected",
        # General site camera / camera 2 detection patterns (model-agnostic, no hardcoded classes)
        "visible on camera 2", "detected on camera 2", "see on camera 2", "seen on camera 2",
        "on camera 2", "dikh raha hai camera 2", "camera 2 par", "camera 2 pe",
        "objects visible on camera 2", "objects detected on camera 2",
        "site camera detections", "camera detections", "what objects are on site",
    ], 5),
    (Intent.GET_OPERATOR, [
        "kaun printing", "kaun print", "kaun chala", "kaun operate", "kaun kar raha",
        "operator kaun", "kaun sa operator", "who is printing", "who is operating",
        "who is running", "current operator", "operator name", "operator kon",
    ], 5),
    (Intent.GET_CURRENT_LAYER, [
        "current layer", "which layer", "what layer", "layer number", "kaunsa layer",
        "kaun sa layer", "abhi layer", "layer chal raha", "layers completed",
        "how many layer", "how many layers", "kitne layer", "layer are we printing",
        # Hinglish forms for "which layer is running" and "how many layers".
        "kaunsi layer", "layer chal rahi", "layer chal", "total layers",
        "total kitni layer", "kitni layer", "layer hai", "layer kitni",
        "layer print ho", "layer printed", "layer have been",
    ], 3),
    (Intent.GET_MACHINE_STATUS, [
        "machine status", "machine state", "machine ka status", "machine chal",
        "is the machine", "printing chal rahi", "printing running", "is printing",
        "print ho raha", "status", "print chal", "printing chal", "machine chal rahi",
        "printing ka status", "printing status", "print raha", "printing hai",
    ], 2),
    (Intent.GET_TOTAL_TARGET, [
        "total target", "target", "total walls", "how many in total", "total kitna",
        "kul kitna",
    ], 2),

    # --- most generic: completed production ---
    (Intent.GET_COMPLETED_PRODUCTION, [
        "how many walls", "walls printed", "walls have been printed", "completed",
        "how much printed", "production", "kitni wall", "kitni walls", "print ho gay",
        "print ho chuki", "kitna print hua", "abhi tak kitna", "wall count",
        "printed so far", "how many have we printed", "how much is printed",
        "how much has been printed", "much printed", "is printed",
        # Common speech-to-text spellings of the Hinglish phrasing.
        "print ho gai", "print ho gayi", "walls print", "wall print",
        "kitne wall", "kitne walls", "keetni", "keetney",
        # "kitni printing complete ho gayi", "kitna print ho chuka hai",
        # "well kitni complete hui" -- previously unmatched.
        "printing complete", "complete ho gay", "complete hui", "complete ho chuki",
        "kitni printing", "print ho chuka", "ho chuka", "kitni complete",
        "kitni well", "well print", "kitni wall print", "print hua",
    ], 1),
]

# Knowledge questions: definitions, causes, procedures -- answered from RAG.
_KNOWLEDGE_TERMS = [
    "what is", "what does", "why does", "why can", "how can", "how does",
    "explain", "meaning", "kya hota", "kya hai matlab", "kaise hota", "kyun hota",
    "definition", "reason for", "causes",
]

# Technical topics that indicate a documentation lookup rather than live data.
#
# Deliberately EXCLUDES "layer width": it names a live measurement the router
# already resolves, and adding it sent four working queries ("layer width kitni
# hai", "what is the current layer width", ...) to COMBINED_QUERY. The live-hint
# guard in `classify` protects those anyway, but there is no reason to widen the
# vocabulary into a noun that has its own live intent.
_TECHNICAL_TOPICS = [
    "delamination", "layer shifting", "layer shift", "bonding", "adhesion",
    "warping", "spaghetti", "clog", "nozzle", "curing", "shrinkage", "crack",
    "porosity", "segregation", "slump", "maintenance", "procedure", "sop",
    "calibration", "calibrate",
    # Printing subjects documented in the ingested knowledge base. Without
    # these, a question naming any of them never reached RAG at all.
    "accelerator", "hose", "choking", "choke", "clogging", "layer thickness",
    "thickness", "gap", "void", "servo", "pump", "material flow", "fly ash",
    "hopper", "esp", "g code", "gcode", "wall crack", "printing defect",
    "printing problem", "sieve", "u joint", "z height", "delaminate",
    "pipe", "clean", "shift ho", "seal", "seals", "coupling", "interval",
    "allen key", "saaf", "schedule",
    # ATOM technical specifications and configuration topics
    "atom", "linear axis", "rover base", "robot only", "printer weight",
    "equipment dimensions", "safety fence", "safety-fence", "number of axis",
    "build envelop", "build reach", "axes", "repeatability", "kelvicrete",
    # Robot component reference library topics
    "reference photograph", "reference photographs", "robot reference", "component reference",
    "reference photo", "reference photos", "reference image", "reference images",
]

# Signals that the operator is reporting a PROBLEM or asking HOW TO do
# something, rather than asking for a current reading.
#
# This exists because `_EXPLANATION_TERMS` is only consulted when a live phrase
# also matched (see `classify`). A troubleshooting question that matches no live
# phrase used to lose its signal entirely and fall through to
# `_fallback_keywords`, where a bare "layer" became GET_CURRENT_LAYER -- so
# "why is there a gap in my printed layer?" was answered with "Current layer is
# 42 out of 60". These terms are checked BEFORE that fallback.
#
# Bare "problem" and "issue" are deliberately ABSENT. They also open the live
# "kya koi problem hai" query, which would then route to COMBINED_QUERY -- and
# `summarise` builds its combined sentence from only four fact keys
# (layer_width_mm, completed_mm, print_rate_mm_per_min, quality_percent), none
# of which GET_RECENT_PROBLEMS produces. The live problem list was therefore
# dropped from the spoken answer. Excluding these two words costs nothing: every
# troubleshooting question still matches on "defect", "not working", "fix",
# "troubleshoot", "error", "fault" or "blockage".
#
# The Hinglish entries are the operator utterances documented in the
# troubleshooting dataset ("layer patli aa rahi hai", "layers chipak nahi
# rahe"), which carry the defect meaning in words no English list would catch.
_PROBLEM_TERMS = [
    "what should i do", "what should i check", "how often", "how do i", "how to",
    "defect", "not working", "repeatedly", "troubleshoot",
    "kya karein", "kya karna", "kaise karein", "error", "fault", "blocked",
    "blockage", "stuck", "uneven", "too thick", "too thin", "correct", "fix",
    "patli", "patla", "moti", "mota", "kharab", "chipak nahi", "alag ho",
    "gir gay", "kam aa raha", "zyada spread", "phat", "last kab",
    "saaf", "badalna", "badalni", "kab clean", "kab saaf", "kab check",
]

# Signals that the operator wants an EXPLANATION alongside a value. Plain
# "what is" is excluded: "what is the current layer width" is a pure live query.
_EXPLANATION_TERMS = [
    "can happen", "what problems", "why", "explain", "mean", "cause", "how can",
    "is it normal", "normal?", "should i", "kyun", "kaise hota", "kya hota",
]

_LIVE_HINT_TERMS = [
    "current", "abhi", "now", "today", "aaj", "status", "how many", "kitni", "kitna",
]

# --- gate for consulting the documentation before the fallback keyword table ---
#
# `_fallback_keywords` maps a single bare word to a live intent, so a question
# it cannot really answer still got a confident live number: "How should the
# wall be lifted?" matched "wall" and was answered "650.0 millimeter printed out
# of 1000.0 millimeter". Asking retrieval first fixes that -- but only when
# retrieval genuinely has an answer, or the same hijack happens in reverse.
#
# Two independent conditions, because neither alone is sufficient:
#
# FALLBACK_RAG_MIN_SCORE is NOT RAG_MIN_SCORE and must not be confused with it.
# RAG_MIN_SCORE (0.15) decides whether a retrieved passage is worth showing at
# all; this decides whether retrieval should outrank a live intent, which needs
# a much higher bar. Measured on this corpus: "kitna material use hua hai"
# retrieves an irrelevant record at 0.219, so 0.15 would send a working live
# query to the documentation. 0.35 is an empirical separator fitted to the
# current 101-chunk index, not a principled constant -- re-measure it if the
# index changes materially. The tests pin the two live queries that sit closest
# to this boundary so a future change fails loudly instead of silently.
FALLBACK_RAG_MIN_SCORE = 0.35
#
# FALLBACK_RAG_MIN_TERMS protects single-word live queries. Lexical scores are
# normalised against what a query could achieve, so a one-informative-word
# question scores near the ceiling against any document containing that word:
# "who is the operator" retrieves an unrelated record at 0.734, which no
# threshold can screen out. Requiring two informative words does, because a
# genuine documentation question names both a subject and something about it.
FALLBACK_RAG_MIN_TERMS = 2


# Hinglish has no fixed spelling, and speech-to-text picks whichever form it
# hears. These are spelling variants of the SAME word, folded to one form so a
# single phrase in _PATTERNS covers all of them. Word-for-word only -- nothing
# here changes meaning, so "printing chal rhi h" and "printing chal rahi hai"
# become the same string.
_HINGLISH_FORMS = {
    "h": "hai", "hai": "hai", "hain": "hai", "he": "hai", "hy": "hai",
    "rhi": "rahi", "rahi": "rahi", "rhi?": "rahi",
    "rha": "raha", "rhaa": "raha", "raha": "raha",
    "rhe": "rahe", "rhy": "rahe",
    "gyi": "gayi", "gai": "gayi", "gyee": "gayi", "gayee": "gayi",
    "gya": "gaya", "gyaa": "gaya",
    "kr": "kar", "krr": "kar",
    "krta": "karta", "krna": "karna", "krke": "karke",
    "nhi": "nahi", "nai": "nahi",
    "chuka": "chuka", "chuki": "chuki", "chuke": "chuka",
    "kitni": "kitni", "kitnee": "kitni", "keetni": "kitni", "keetney": "kitni",
    "baaki": "baki", "bcha": "bacha", "bacha": "bacha",
    "kon": "kaun", "kaun": "kaun", "koun": "kaun",
    "konsi": "kaunsi", "kaunsi": "kaunsi", "kounsi": "kaunsi",
    "abhi": "abhi", "abi": "abhi",
    "wel": "well", "wal": "wall",
    "hva": "hua", "hva?": "hua", "huva": "hua", "hui": "hui",
    "layers": "layer", "walls": "wall", "wells": "well",
    "prnt": "print", "printng": "printing",
}

# Last resort. If no phrase matched but the question clearly IS about the
# machine, fall back to the intent that single word implies. This exists
# because speech-to-text mangles sentences ("it will all print away" was
# "kitni wall print ho gayi"): a recognisable machine word is better evidence
# than nothing, and answering the likely question beats refusing outright.
# Ordered most specific first.


# Whisper transcribes Hindi speech in DEVANAGARI SCRIPT, not romanised
# Hinglish: "printing chal rahi hai" comes back as "प्रिंटिंग चल रही है".
# Every pattern in this file is written in Latin text, so a Devanagari
# transcript previously matched nothing at all and became UNKNOWN_QUERY no
# matter how clear the question was. These are the machine words the operator
# actually uses, mapped to the spelling the patterns expect.
#
# Longest phrases first: "चल रही है" must be replaced before "है" alone.
_DEVANAGARI = [
    ("प्रिंटिंग", "printing"), ("प्रिन्टिंग", "printing"), ("प्रिंट", "print"),
    ("छपाई", "printing"),
    ("चल रही है", "chal rahi hai"), ("चल रहा है", "chal raha hai"),
    ("चल रही", "chal rahi"), ("चल रहा", "chal raha"),
    ("हो गई है", "ho gayi hai"), ("हो गयी है", "ho gayi hai"),
    ("हो गई", "ho gayi"), ("हो गयी", "ho gayi"), ("हो गया", "ho gaya"),
    ("हो चुकी", "ho chuki"), ("हो चुका", "ho chuka"),
    ("कितनी", "kitni"), ("कितना", "kitna"), ("कितने", "kitne"),
    ("दीवार", "wall"), ("वॉल", "wall"), ("वेल", "well"),
    ("लेयर", "layer"), ("परत", "layer"), ("लेयर्स", "layers"),
    ("बाकी", "baki"), ("बचा", "bacha"), ("शेष", "baki"),
    ("कौनसी", "kaunsi"), ("कौन सी", "kaunsi"), ("कौन", "kaun"),
    ("कुल", "total"), ("पूरा", "complete"), ("पूरी", "complete"),
    ("कम्पलीट", "complete"), ("कंप्लीट", "complete"),
    ("अभी", "abhi"), ("स्टेटस", "status"), ("स्थिति", "status"),
    ("मशीन", "machine"), ("कैमरा", "camera"), ("रोबोट", "robot"),
    ("चौड़ाई", "width"), ("ऊंचाई", "height"),
    ("मटेरियल", "material"), ("सामग्री", "material"),
    ("ऑपरेटर", "operator"), ("समस्या", "problem"), ("दिक्कत", "problem"),
    ("कब", "kab"), ("क्या", "kya"), ("है", "hai"), ("हैं", "hai"),
    ("रही", "rahi"), ("रहा", "raha"), ("गई", "gayi"), ("गयी", "gayi"),
]

# The wake phrase is not part of the question, and speech-to-text mangles it
# constantly ("hey robot" -> "high robot", "आइरोट", "a robot"). Stripping it
# stops a misheard wake word from burying a perfectly good question.
_WAKE_PREFIXES = [
    # "Okay Kelvin" and the spellings speech-to-text returns for it.
    "okay kelvin6k", "ok kelvin6k", "okay kelvin", "ok kelvin", "okey kelvin",
    "o k kelvin", "okay calvin", "ok calvin", "okay kevin", "ok kevin",
    "hey kelvin", "ok kelvin6", "okay kelvin6",
    # Older phrases, still stripped so previously recorded questions keep working.
    "hey robot", "hey rob", "high robot", "hi robot", "hello robot", "ok robot",
    "a robot", "ai robot", "eye robot", "hey robert",
    "ओके केल्विन", "ठीक है केल्विन", "आइरोट", "अरोबोट", "हे रोबोट",
]


def _strip_wake_word(text: str) -> str:
    """Removes a leading wake phrase, however badly it was transcribed."""
    # Strip addressing Kelvin directly (e.g. "Kelvin, ...", "Okay Kelvin, ...", "Hey Kelvin ...")
    stripped = re.sub(
        r"^(?:(?:okay|ok|hey|hi|hello)?\s*(?:kelvin|calvin|kevin)(?:6k)?)\b\s*[,.:;]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if stripped != text:
        text = stripped
    for phrase in _WAKE_PREFIXES:
        if text.startswith(phrase):
            return text[len(phrase):].lstrip(" ,.")
    return text


def strip_wake_words(text: str) -> str:
    """
    The question with any wake phrase removed. Empty means "only the wake word".

    Public because the assistant needs to tell a bare "Okay Kelvin" apart from a
    question it failed to understand: the first deserves "Yes, what would you
    like to know?", the second deserves an apology.
    """
    return _normalise(text)


def _normalise(text: str) -> str:
    """
    Lower-case, strip punctuation, and fold Hinglish spelling variants.

    Hyphens and underscores become spaces: speech-to-text sometimes returns
    hyphenated runs like "Kitni-Walls-Print-Ho-Gayi-Hain", which would otherwise
    match no phrase at all.

    The spelling fold is what makes casual speech work. Whisper transcribes the
    same spoken sentence as "printing chal rhi h" one time and "printing chal
    rahi hai" the next; without folding, only one of them matches.
    """
    text = text.lower().strip()
    text = _strip_wake_word(text)
    # Devanagari -> Latin BEFORE anything else, so a Hindi-script transcript
    # goes through exactly the same matching as a romanised one.
    for hindi, latin in _DEVANAGARI:
        if hindi in text:
            text = text.replace(hindi, latin)
    text = re.sub(r"[-_/]+", " ", text)
    text = text.replace("’", "'")
    text = re.sub(r"['’]s\b", " ", text)
    text = re.sub(r"[?!.,;:\"']", " ", text)
    text = re.sub(r"\bmaterial\s+slope\b", "material flow", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _strip_wake_word(text)
    return " ".join(_HINGLISH_FORMS.get(word, word) for word in text.split())


_FUTURE_OR_PLANNING_TERMS = [
    "karni hai", "karna hai", "karne hai",
    "plan", "target", "will", "chahiye", "karunga", "karenge",
]

_QUESTION_INDICATORS = [
    "kitni", "kitna", "kitne", "keetni", "keetney",
    "how many", "how much", "what is", "kya", "count",
    "so far", "abhi tak",
]

_WALL_COMPLETION_VERBS = [
    "print ho gayi", "print ho gai", "print ho chuki", "print hui",
    "complete ho gayi", "complete ho gai", "complete ho chuki", "complete hui",
    "print kar di", "print kar diya",
    "complete kar di", "complete kar diya",
    "i printed wall", "i completed wall",
    "log wall", "record wall", "wall entry karo", "wall log karo",
    "completed wall", "completed walls", "completed two walls", "completed two wall", "completed 2 walls",
]


def _is_future_or_planning(q: str) -> bool:
    return any(t in q for t in _FUTURE_OR_PLANNING_TERMS)


def _is_wall_production_question(q: str) -> bool:
    return any(t in q for t in _QUESTION_INDICATORS)


def _is_issue_question(q: str) -> bool:
    question_starters = [
        "how", "what", "why", "kaise", "kya", "kyun",
        "which", "who", "kisko", "kisne", "kahan", "where"
    ]
    return any(re.search(rf"\b{qs}\b", q) for qs in question_starters)


def _is_completed_wall_production(q: str) -> bool:
    if _is_future_or_planning(q) or _is_wall_production_question(q):
        return False
    if any(t in q for t in _PROBLEM_TERMS) or any(w in q for w in ("void", "crack", "gap", "defect", "choke", "choking", "problem", "issue", "dikkat", "samasya")):
        return False
    has_wall = bool(re.search(r"\b(?:walls?|wells?)\b", q))
    if not has_wall:
        return False
    if any(p in q for p in _WALL_COMPLETION_VERBS):
        return True
    return bool(re.search(r"\b(?:i\s+(?:have\s+)?(?:completed|printed)|completed)\s+(?:\d+|one|two|three|four|five|ek|do|teen|char|chaar|panch)\s+(?:walls?|wells?)\b", q))

_GENERIC_DETECTION_TERMS = {
    "object", "objects", "anything", "something", "item", "items",
    "what", "all", "detections", "everything", "classes", "what do you see",
}


def _clean_candidate_class(cand: Optional[str]) -> Optional[str]:
    if not cand:
        return None
    cand = cand.strip().lower()
    cand = re.sub(r"^(?:the|a|an|any|some|that|this)\s+", "", cand)
    cand = re.sub(r"\s+(?:on\s+camera(?:\s*2)?|on\s+site|via\s+camera(?:\s*2)?|right\s+now|currently|please)$", "", cand)
    cand = re.sub(r"^(?:on\s+camera(?:\s*2)?|on\s+site|via\s+camera(?:\s*2)?)\s*", "", cand)
    cand = cand.strip()
    if not cand or cand in _GENERIC_DETECTION_TERMS or cand.startswith("what"):
        return None
    return cand


def extract_requested_detection_class(q: str) -> Optional[str]:
    """
    Extracts the target object/class from a natural language detection question
    without hardcoding any class names.
    Returns None for general detection queries (e.g. 'what objects are visible?').
    """
    if not q:
        return None
    q_norm = q.lower().strip()
    if re.search(r"\bwhat\s+(?:objects?|do\s+you\s+see|is\s+visible|is\s+detected|is\s+on)\b", q_norm):
        return None
    patterns = [
        r"\b(?:is|are|was|were)\s+(?:there\s+)?(.+?)\s+(?:visible|detected|seen)\b",
        r"\b(?:do|can|could)\s+you\s+see\s+(.+?)(?:\s+(?:visible|detected|on\s+camera|on\s+site)|$|\?)",
        r"\bcheck\s+(?:if\s+(.+?)\s+is\s+(?:visible|detected)|for\s+(.+?)(?:\s+on\s+camera|\s+on\s+site|$))",
        r"\bcamera(?:\s*2)?\s+(?:me|par|pe)\s+(.+?)\s+dikh\s+rah",
        r"\b(?:kya\s+)?(.+?)\s+dikh\s+rah[ia]\s+(?:hai\s+)?(?:camera|site)",
        r"\b(.+?)\s+(?:is|are)?\s*(?:visible|detected)\s+on\s+(?:camera|site)\b",
        r"\b(.+?)\s+(?:visible|detected)\b",
    ]
    for p in patterns:
        m = re.search(p, q_norm)
        if m:
            cand = next((_clean_candidate_class(g) for g in m.groups() if g and _clean_candidate_class(g)), None)
            if cand:
                return cand
    return None


def _find_supported_class(candidate: str, supported_classes: List[str]) -> Optional[str]:
    """
    Dynamically checks if candidate matches any class in the loaded YOLO model's supported_classes.
    Handles case-insensitivity, singular/plural, and space/underscore variations.
    """
    if not candidate or not supported_classes:
        return None
    c_cand = candidate.strip().lower()
    norm_map: Dict[str, str] = {}
    for sc in supported_classes:
        sc_low = sc.lower()
        norm_map[sc_low] = sc
        norm_map[sc_low.replace("_", " ")] = sc
        norm_map[sc_low.replace(" ", "")] = sc
        norm_map[sc_low.replace("-", " ")] = sc

    if c_cand in norm_map:
        return norm_map[c_cand]
    if c_cand.endswith("es") and c_cand[:-2] in norm_map:
        return norm_map[c_cand[:-2]]
    if c_cand.endswith("s") and c_cand[:-1] in norm_map:
        return norm_map[c_cand[:-1]]
    if (c_cand + "s") in norm_map:
        return norm_map[c_cand + "s"]
    if (c_cand + "es") in norm_map:
        return norm_map[c_cand + "es"]
    return None


def _is_site_detection_query(q: str) -> bool:
    """
    Dynamically checks if a query is asking whether an object/class is visible
    or detected on Camera 2 / site camera, without hardcoding any classes.
    """
    if not q:
        return False
    if any(p in q for p in ("people", "person", "persons", "worker", "workers", "vyakti", "bande", "log ")) and any(k in q for k in ("count", "kitne", "how many", "who is", "anyone")):
        return False
    if any(w in q for w in ("quality", "pressure", "material", "temperature", "motor", "speed", "flow", "downtime", "hours", "started", "ended", "completed")):
        if not ("visible" in q or "detected" in q or "dikh rah" in q or "camera 2" in q):
            return False
    if any(p in q for p in (
        "what objects are visible", "what objects are detected", "what do you see on camera",
        "what is visible on camera", "what is detected on camera", "camera 2 detections",
        "detected objects on site", "objects on camera 2", "objects detected on camera",
        "objects on site", "what is on camera 2", "what is detected on site",
        "objects visible on camera", "objects detected", "visible on camera 2",
        "detected on camera 2", "see on camera 2", "seen on camera 2",
        "camera 2 me kya dikh raha", "camera 2 pe kya dikh raha",
    )):
        return True
    cand = extract_requested_detection_class(q)
    return cand is not None


def _is_maintenance_query(q: str) -> bool:
    """Detects inquiries about machine maintenance intervals and procedures."""
    has_maint_action = any(k in q for k in (
        "clean", "saaf", "saf", "cleaning", "maintenance", "servicing",
        "interval", "frequency", "schedule", "check", "replace", "badal",
        "allen key", "curing", "cure", "restart",
    ))
    if not has_maint_action:
        return False

    if "accelerator" in q and any(w in q for w in ("hole", "coupling", "clean", "saaf", "check", "daily", "roz", "layer", "wire", "rod", "key")):
        return True
    if "pipe" in q and any(w in q for w in ("clean", "saaf", "ghante", "hour", "hours", "interval", "operation")):
        return True
    if "hopper" in q and any(w in q for w in ("clean", "saaf", "wall", "well", "kab", "when")):
        return True
    if "pump" in q and any(w in q for w in ("front", "seal", "seals", "pause", "pauses", "shade", "clean", "saaf", "wall", "well", "check", "replace", "minute", "chalana")):
        return True
    if "coupling" in q and any(w in q for w in ("check", "daily", "roz", "inspect", "tight", "loose", "wear")):
        return True
    if "seal" in q and any(w in q for w in ("check", "weekly", "hafta", "hafte", "inspect", "replace", "wear")):
        return True
    if ("curing" in q or "cure" in q) and any(w in q for w in ("wall", "well", "day", "din", "baar", "times", "water", "hose", "often")):
        return True
    if "restart" in q and any(w in q for w in ("time", "window", "hour", "ghanta", "maximum", "max", "alignment", "der", "taking too long")):
        return True
    if "hose" in q and any(w in q for w in ("position", "shape", "bend", "loose", "tight", "routing", "connection", "connect", "lagana", "jodna")):
        return True
    if any(p in q for p in (
        "cleaning interval", "cleaning schedule", "parts need cleaning",
        "what cleaning is due", "maintenance interval", "maintenance schedule",
        "kitni layers ke baad", "kitni layer ke baad", "kitne ghante mein clean",
        "kitni wall ke baad", "do wall ke baad", "ek wall ke baad", "har do wall",
    )):
        return True

    return False


# Words that can sit where a name would and are never a name. Without this,
# "okay here start my shift" would offer "okay" as the operator.
_NOT_A_NAME = {
    "okay", "ok", "kelvin", "yes", "no", "right", "so", "and", "then", "now",
    "am", "is", "i", "im", "me", "my", "shift", "start", "end", "begin",
    "line", "today", "over", "from", "just", "still", "back", "who", "it",
    "main", "hoon", "hu", "bol", "raha", "rahi", "hai", "ji", "sir",
    "the", "a", "an", "this", "that", "his", "her", "employee", "new", "attendance",
    "ka", "ki", "ke", "ko", "karo", "for", "of", "in", "on", "duty", "work",
    "hi", "he", "him", "my", "me", "their", "them", "unka", "unki", "apna", "apni",
    "morning", "afternoon", "night", "general",
}


def _clean_candidate_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = re.sub(r"['’]s\b", "", name).strip(" ,.-'")
    parts = [p for p in name.split() if p.lower() not in _NOT_A_NAME]
    if not parts:
        return None
    return " ".join(parts).title()


def _is_shift_start_command(q: str, raw_question: str) -> Tuple[bool, Optional[str]]:
    """
    Detects imperative natural-language shift start / clock-in commands in English and Hinglish.
    Returns (is_start_cmd, extracted_candidate_name).
    """
    # 1. Guard against questions or timing inquiries
    if any(re.search(rf"\b{re.escape(w)}\b", q) for w in (
        "kab", "when did", "did", "timing", "status", "shuru hui kya",
        "start hui kya", "started yet", "active yet", "hours did",
        "kitne ghante", "how long", "working days"
    )):
        return False, None
    if re.search(r"\b(?:when|did|kya)\b.*?\b(?:shift|start|shuru|attendance)\b", q) and any(w in q for w in ("hui", "hua", "did", "when")):
        return False, None

    # 2. Check explicit clock-in commands
    m = re.search(r"^([a-z0-9_\- ]+?)\s+clock\s+in\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))
    m = re.search(r"^([a-z0-9_\- ]+?)\s+(?:ko\s+)?clock\s+in\s+karo\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))
    m = re.search(r"\bclock\s+in\s+(?:for\s+)?([a-z0-9_\- ]+)\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))

    # 3. Check start shift / attendance for <name>
    m = re.search(r"\b(?:start|begin|chalu|shuru)\s+(?:the\s+|a\s+|new\s+)?(?:shift|attendance)\s+(?:for|of)\s+([a-z0-9_\- ]+)\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))

    # 4. Check start <name>('s) shift / attendance
    m = re.search(r"\b(?:start|begin|chalu|shuru)\s+([a-z0-9_\- ]+?)(?:'s|s)?\s+(?:shift|attendance)\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))

    # 5. Check Hinglish <name> ka/ki shift start / shuru karo
    m = re.search(r"\b([a-z0-9_\- ]+?)\s+(?:ka|ki|ke|ko)\s+shift\s+(?:start|begin|shuru|chalu)(?:\s+karo)?\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))

    # 6. Check Hinglish <name> ka/ki attendance start / lagao / mark karo
    m = re.search(r"\b([a-z0-9_\- ]+?)\s+(?:ka|ki|ke|ko)\s+attendance\s+(?:start|shuru|chalu|lagao|mark)(?:\s+karo)?\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))

    # 7. Check self or generic start shift commands
    if re.search(r"\b(?:start|begin|shuru|chalu)\s+(?:my\s+|the\s+|a\s+|new\s+)?shift\b", q) or re.search(r"\b(?:clock\s+in|clock\s+me\s+in|mark\s+me\s+present)\b", q):
        return True, None

    return False, None


def _is_shift_end_command(q: str, raw_question: str) -> Tuple[bool, Optional[str]]:
    """
    Detects imperative natural-language shift end / clock-out commands in English and Hinglish.
    Returns (is_end_cmd, extracted_candidate_name).
    """
    if any(re.search(rf"\b{re.escape(w)}\b", q) for w in (
        "kab", "when did", "did", "timing", "status", "khatam hui kya",
        "ended yet", "hours did", "kitne ghante", "how long"
    )):
        return False, None
    if re.search(r"\b(?:when|did|kya)\b.*?\b(?:shift|end|khatam|band)\b", q) and any(w in q for w in ("hui", "hua", "did", "when")):
        return False, None

    # Clock out
    m = re.search(r"^([a-z0-9_\- ]+?)\s+clock\s+out\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))
    m = re.search(r"^([a-z0-9_\- ]+?)\s+(?:ko\s+)?clock\s+out\s+karo\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))
    m = re.search(r"\bclock\s+out\s+(?:for\s+)?([a-z0-9_\- ]+)\b", q)
    if m:
        return True, _clean_candidate_name(m.group(1))

    # Generic, self or pronoun end shift commands
    if re.search(r"\b(?:end|close|stop|khatam|band)\s+(?:my\s+|the\s+|a\s+|current\s+|his\s+|her\s+|their\s+)?(?:shift|attendance)\b", q) or re.search(r"\b(?:clock\s+out|clock\s+me\s+out|done\s+for\s+today)\b", q):
        return True, None

    # End shift for <name>
    m = re.search(r"\b(?:end|close|stop|khatam|band)\s+(?:the\s+|a\s+|current\s+)?(?:shift|attendance)\s+(?:for|of)\s+([a-z0-9_\- ]+)\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        if cand:
            return True, cand

    # End <name>('s) shift
    m = re.search(r"\b(?:end|close|stop|khatam|band)\s+([a-z0-9_\- ]+?)(?:'s|s)?\s+(?:shift|attendance)\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        if cand:
            return True, cand

    # Hinglish <name> ka/ki shift end / khatam / band karo
    m = re.search(r"\b([a-z0-9_\- ]+?)\s+(?:ka|ki|ke|ko)\s+shift\s+(?:end|close|stop|khatam|band)(?:\s+karo)?\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        if cand:
            return True, cand

    return False, None


def _is_attendance_status_query(q: str, raw_question: str) -> Tuple[bool, Optional[str]]:
    """
    Detects queries asking whether someone (or anyone) is on shift / currently working / clocked in.
    Returns (is_attendance_query, extracted_candidate_name).
    """
    # Guard against start / end commands and active machine shift queries
    if any(w in q for w in ("start", "begin", "shuru", "chalu", "end", "close", "khatam", "band", "clock in", "clock out", "what shift", "which shift", "shift is active", "shift is currently active", "active shift", "kaun si shift", "current shift")):
        return False, None

    # 1. "is <name> currently on shift", "is <name> on shift", "is <name> clocked in"
    m = re.search(r"\bis\s+([a-z0-9_\- ]+?)\s+(?:currently\s+)?(?:on\s+shift|clocked\s+in|working\s+on\s+shift|on\s+duty)\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        if cand and cand not in ("there", "anyone", "someone", "anybody", "he", "she", "it", "shift"):
            return True, cand
        return True, None

    # 2. "check if/whether <name> is on shift/clocked in"
    m = re.search(r"\b(?:check|see|tell\s+me)\s+(?:if|whether)\s+([a-z0-9_\- ]+?)\s+(?:is\s+)?(?:currently\s+)?(?:on\s+shift|clocked\s+in)\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        return True, cand

    # 3. Hinglish: "kya <name> abhi shift par/pe hai", "kya <name> shift par/pe hai", "<name> shift par/pe hai kya"
    m = re.search(r"\bkya\s+([a-z0-9_\- ]+?)\s+(?:abhi\s+)?(?:shift|kaam)\s+(?:par|pe|me|mein)\s+hai\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        return True, cand
    m = re.search(r"\b([a-z0-9_\- ]+?)\s+(?:abhi\s+)?(?:shift|kaam)\s+(?:par|pe|me|mein)\s+hai\s+kya\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        return True, cand

    # 4. "<name> shift status", "shift status for <name>"
    m = re.search(r"\b([a-z0-9_\- ]+?)(?:'s|s)?\s+shift\s+status\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        if cand not in ("what", "current", "my"):
            return True, cand
    m = re.search(r"\bshift\s+status\s+(?:for|of)\s+([a-z0-9_\- ]+)\b", q)
    if m:
        cand = _clean_candidate_name(m.group(1))
        return True, cand

    # 5. General attendance queries
    if any(p in q for p in (
        "who is currently on shift", "who is on shift", "who is clocked in",
        "who is currently working", "who is working right now", "who is on duty",
        "currently on shift", "currently working", "active on shift", "kaun shift pe hai",
        "kaun kaun shift pe hai", "kaun kaam kar raha hai", "anyone on shift",
        "is anyone on shift", "is anyone working", "are any employees working"
    )):
        return True, None

    return False, None


def _extract_wall_quantity(text: str) -> Optional[int]:
    """
    Extracts an integer quantity of walls from text.
    Supports digits and number words in English/Hinglish (e.g. '4 walls', 'two walls', 'chaar walls', 'ek wall').
    Preserves negative values (e.g. '-2') so validation can reject them properly.
    """
    if not text:
        return None
    t = _normalise(text)

    # 1. Negative numbers
    m_neg = re.search(r"-\s*(\d+)\s*(?:completed\s+|printed\s+)?(?:walls?|wells?)", t)
    if m_neg:
        return -int(m_neg.group(1))

    # 2. Positive digits before wall/walls
    m_dig = re.search(r"\b(\d+)\s*(?:completed\s+|printed\s+)?(?:walls?|wells?)", t)
    if m_dig:
        return int(m_dig.group(1))

    # 3. Digits after 'to' or 'it should be' or 'should be'
    m_to = re.search(r"\b(?:to|be|it\s+should\s+be|should\s+be)\s+(\d+)\s*(?:walls?|wells?)?", t)
    if m_to:
        return int(m_to.group(1))

    # 4. English / Hinglish word numbers before wall
    num_words = {
        "one": 1, "ek": 1,
        "two": 2, "do": 2,
        "three": 3, "teen": 3,
        "four": 4, "chaar": 4, "char": 4,
        "five": 5, "panch": 5, "paanch": 5,
        "six": 6, "chhe": 6, "che": 6,
        "seven": 7, "saat": 7,
        "eight": 8, "aath": 8,
        "nine": 9, "nau": 9,
        "ten": 10, "das": 10,
    }
    for w, val in num_words.items():
        if re.search(rf"\b{w}\s*(?:completed\s+|printed\s+)?(?:walls?|wells?)", t):
            return val
        if re.search(rf"\b(?:to|be|it\s+should\s+be|should\s+be)\s+{w}\b", t):
            return val

    # 5. Bare digit in a recording sentence like "completed 4 walls; save it" or "record 3 walls"
    m_any = re.search(r"\b(\d+)\b", t)
    if m_any:
        return int(m_any.group(1))

    return None


def _extract_correction_quantities(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extracts (new_quantity, old_quantity) from correction phrases.
    Examples:
      - 'it should be 3 walls, not 4' -> (3, 4)
      - 'should be 3 walls not 4' -> (3, 4)
      - 'update Hamadan's last wall entry to 2 walls' -> (2, None)
      - 'correction: Hamadan printed 3 walls, not 4' -> (3, 4)
      - 'correct previous wall record to 3' -> (3, None)
    """
    t = _normalise(text)
    word_to_num = {
        "one": 1, "ek": 1, "two": 2, "do": 2, "three": 3, "teen": 3,
        "four": 4, "chaar": 4, "char": 4, "five": 5, "panch": 5, "paanch": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    # Pattern: should be / was / it was X walls not Y
    m = re.search(r"(?:should\s+be|it\s+should\s+be|it\s+was|was|to)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|ek|do|teen|chaar|char|panch)\s*(?:walls?|wells?)?.*?(?:not|naki|instead\s+of)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|ek|do|teen|chaar|char|panch)", t)
    if m:
        g1 = m.group(1)
        g2 = m.group(2)
        new_qty = int(g1) if g1.isdigit() else word_to_num.get(g1)
        old_qty = int(g2) if g2.isdigit() else word_to_num.get(g2)
        return new_qty, old_qty

    # Pattern: to X walls / as X walls
    m_to = re.search(r"\b(?:to|as)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|ek|do|teen|chaar|char|panch)\s*(?:walls?|wells?)?", t)
    if m_to:
        g1 = m_to.group(1)
        new_qty = int(g1) if g1.isdigit() else word_to_num.get(g1)
        return new_qty, None

    new_qty = _extract_wall_quantity(text)
    return new_qty, None


def _extract_wall_production_scope(text: str) -> Tuple[str, Optional[str]]:
    """
    Extracts (scope, filter_value) for GET_WALL_PRODUCTION.
    Scopes: 'today', 'this_week', 'active_shift', 'job', 'all_time', 'date'.
    """
    t = _normalise(text)
    if any(w in t for w in ("this week", "week", "hafte", "hafta", "weekly")):
        return "this_week", None
    m_job = re.search(r"\bjob\s*(?:#|number|no\.?)?\s*([0-9a-z_\-]+)", t)
    if m_job:
        raw_job = m_job.group(1).lower().strip()
        job_id = f"job_{raw_job}" if not raw_job.startswith("job_") else raw_job
        return "job", job_id
    if any(w in t for w in ("current shift", "active shift", "this shift", "shift")):
        return "active_shift", None
    if any(w in t for w in ("total", "all time", "overall", "so far", "kul", "total wall count")):
        return "all_time", None
    if any(w in t for w in ("today", "aaj")):
        return "today", None
    # Check date
    dt = extract_date_from_text(text)
    if dt:
        return "date", dt
    return "today", None


def _is_wall_production_correction_command(q: str, raw_question: str) -> Tuple[bool, Optional[str], Optional[int], Optional[int]]:
    """
    Detects natural-language corrections for wall production records.
    Returns (is_corr, op_name, new_qty, old_qty).
    """
    has_corr_marker = any(w in q for w in (
        "correct", "correction", "update", "replace", "wrong", "galti",
        "theek karo", "sahi karo", "badal", "instead of", "should be",
        "change", "modify", "alter", "fix", "mistake",
    )) or bool(re.search(r"\b(?:not|instead of)\s+\d+\b", q)) or bool(re.search(r"\bit was \d+", q))
    if not has_corr_marker:
        return False, None, None, None

    if not any(w in q for w in ("wall", "walls", "record", "entry", "production")):
        return False, None, None, None

    new_qty, old_qty = _extract_correction_quantities(raw_question)
    op_name = extract_operator_name(raw_question)
    return True, op_name, new_qty, old_qty


def _is_wall_production_record_command(q: str, raw_question: str) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Detects natural-language commands to record completed walls by an operator.
    Returns (is_rec_cmd, op_name, quantity).
    """
    if any(w in q for w in ("how many", "kitni", "kitna", "what is", "count", "history", "status", "timing")):
        return False, None, None

    has_save_marker = any(w in q for w in ("save it", "save karo", "save", "completed", "printed by", "for operator", "printed", "built", "constructed", "banaya", "banayi", "log", "record"))
    has_walls = bool(re.search(r"\b(?:walls?|wells?)\b", q))
    if not has_walls:
        return False, None, None

    qty = _extract_wall_quantity(raw_question)

    if re.search(r"\b(?:completed|printed)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|ek|do|teen|chaar|char|panch)\s+walls?\b", q):
        op_name = extract_operator_name(raw_question)
        return True, op_name, qty

    if has_save_marker and qty is not None:
        op_name = extract_operator_name(raw_question)
        return True, op_name, qty

    if re.search(r"\b(?:log|record|save)\s+(?:\d+|one|two|three|four|five|ek|do|teen|chaar|char)\s+(?:completed\s+|printed\s+)?walls?\b", q):
        op_name = extract_operator_name(raw_question)
        return True, op_name, qty

    return False, None, None


def _is_wall_production_query(q: str, raw_question: str) -> Tuple[bool, Optional[str], str, Optional[str]]:
    """
    Detects queries asking for wall counts produced by an employee or for a job/shift/week.
    Returns (is_query, op_name, scope, filter_val).
    """
    if not any(w in q for w in ("how many", "kitni", "kitne", "count", "total", "production", "completed", "summary")):
        return False, None, "today", None
    if not any(w in q for w in ("wall", "walls", "well", "wells")):
        return False, None, "today", None

    if any(w in q for w in ("progress", "percent", "percentage", "millimeter", "mm", "layer width", "current layer", "save")):
        return False, None, "today", None

    scope, val = _extract_wall_production_scope(raw_question)
    op_name = extract_operator_name(raw_question)
    return True, op_name, scope, val



def _score(question: str, phrases: List[str]) -> Tuple[float, List[str]]:
    hits = [p for p in phrases if p in question]
    if not hits:
        return 0.0, []
    # Longer matches are more specific, so they outrank generic ones.
    return float(max(len(p) for p in hits)) + 0.1 * len(hits), hits


def classify(question: str) -> Tuple[Intent, List[str]]:
    """Best-matching intent plus the phrases that triggered it."""
    q = _normalise(question)
    if not q:
        # _normalise strips the wake phrase, so an empty result means the
        # operator said only "Okay Kelvin". That is understood perfectly --
        # reporting UNKNOWN_QUERY would wrongly mark it as a failure.
        return (Intent.WAKE_WORD, ["wake word"]) if question.strip() else (Intent.UNKNOWN_QUERY, [])

    # Maintenance interval and procedure inquiries -> KNOWLEDGE_QUERY
    if _is_maintenance_query(q):
        return Intent.KNOWLEDGE_QUERY, ["maintenance_rule"]

    # Fault codes, parameters, and historical operator knowledge queries -> KNOWLEDGE_QUERY
    if re.search(r"\b(?:e136(?:\.1)?|e137(?:\.1)?|h0d11)\b", q, re.IGNORECASE):
        return Intent.KNOWLEDGE_QUERY, ["fault_code_or_parameter"]
    if ("camera" in q or "vision" in q) and any(w in q for w in ("lighting", "shadow", "shadows", "false stop", "false stops")):
        return Intent.KNOWLEDGE_QUERY, ["camera_lighting_observation"]

    # Explicit employee wall production correction commands
    is_wall_corr, corr_op, corr_new_qty, corr_old_qty = _is_wall_production_correction_command(q, question)
    if is_wall_corr:
        op_id = extract_operator_from_text(question) or (extract_operator_tuple_from_text(corr_op)[0] if corr_op else None)
        return Intent.CORRECT_WALL_PRODUCTION, [f"correct_wall_production:{op_id or corr_op or 'prev'}"]

    # Explicit employee wall production recording commands
    is_wall_rec, rec_op, rec_qty = _is_wall_production_record_command(q, question)
    if is_wall_rec:
        op_id = extract_operator_from_text(question) or (extract_operator_tuple_from_text(rec_op)[0] if rec_op else None)
        return Intent.RECORD_WALLS_COMPLETED, [f"record_walls_completed:{op_id or rec_op or 'active'}:{rec_qty}"]

    # Explicit employee wall production queries
    is_wall_q, q_op, q_scope, q_val = _is_wall_production_query(q, question)
    if is_wall_q:
        op_id = extract_operator_from_text(question) or (extract_operator_tuple_from_text(q_op)[0] if q_op else None)
        return Intent.GET_WALL_PRODUCTION, [f"get_wall_production:{op_id or q_op or 'all'}:{q_scope}"]

    # Explicit shift start and shift end commands
    is_shift_start, start_cand = _is_shift_start_command(q, question)
    if is_shift_start:
        target_name = start_cand or extract_operator_name(question)
        op_id = extract_operator_from_text(question)
        if op_id:
            return Intent.SHIFT_START, [f"operator_shift_start:{op_id}"]
        if target_name:
            t_id, t_name = extract_operator_tuple_from_text(target_name)
            return Intent.SHIFT_START, [f"operator_shift_start:{t_id or target_name}"]
        return Intent.SHIFT_START, ["shift_start"]

    is_shift_end, end_cand = _is_shift_end_command(q, question)
    if is_shift_end:
        target_name = end_cand or extract_operator_name(question)
        op_id = extract_operator_from_text(question)
        if op_id:
            return Intent.SHIFT_END, [f"operator_shift_end:{op_id}"]
        if target_name:
            t_id, t_name = extract_operator_tuple_from_text(target_name)
            return Intent.SHIFT_END, [f"operator_shift_end:{t_id or target_name}"]
        return Intent.SHIFT_END, ["shift_end"]

    # Explicit attendance status / who is on shift queries
    is_att_status, att_cand = _is_attendance_status_query(q, question)
    if is_att_status:
        target_name = att_cand or extract_operator_name(question)
        op_id = extract_operator_from_text(question)
        if op_id:
            return Intent.GET_ATTENDANCE_STATUS, [f"attendance_status:{op_id}"]
        if target_name:
            t_id, t_name = extract_operator_tuple_from_text(target_name)
            return Intent.GET_ATTENDANCE_STATUS, [f"attendance_status:{t_id or target_name}"]
        return Intent.GET_ATTENDANCE_STATUS, ["attendance_status"]

    # Historical incident queries -> KNOWLEDGE_QUERY (resolved offline via incident_store)
    if re.search(r"\b(?:inc-\d{8}-[a-z0-9]+|similar\s+incident|last\s+incident|previous\s+incident|faced\s+this|seen\s+this\s+defect|verified\s+by\s+an\s+engineer)\b", q):
        return Intent.KNOWLEDGE_QUERY, ["historical_incident"]

    # Check for registered operator dynamically using existing lookup
    op_id = extract_operator_from_text(question)

    # Kelvin recommendation questions -> KNOWLEDGE_QUERY
    if "kelvin" in q and ("recommend" in q or "suggest" in q or "solution" in q):
        return Intent.KNOWLEDGE_QUERY, ["kelvin_recommendation"]

    # Dynamic registered operator query routing
    if op_id:
        # 1. Operator production query
        is_op_production = bool(
            re.search(r"\b(?:what\s+did\s+.+\s+print|how\s+many\s+walls?\s+did|walls?\s+did\s+.+\s+print|ne\s+kya\s+print|ne\s+kitn\w+\s+wall|ne\s+kitn\w+\s+print|ka\s+production|ne\s+apni\s+shift\s+mein\s+kya\s+print|kya\s+print\s+kiya)\b", q)
            or (("print" in q or "wall" in q or "walls" in q or "production" in q) and any(w in q for w in ("what did", "how many", "kitni", "kitna", "kya")))
        )
        if is_op_production and not _is_completed_wall_production(q):
            return Intent.GET_SHIFT_BY_OPERATOR, [f"operator_production:{op_id}"]

        # 2. Operator issues / problems / actions query
        is_op_issue = bool(
            re.search(r"\b(?:problem|problems|issue|issues|dikkat|samasya|action|solution|kya\s+kiya|kya\s+action|kya\s+solution|kya\s+fix|ke\s+issues|what\s+did\s+.+\s+do)\b", q)
        )
        if is_op_issue:
            return Intent.GET_SHIFT_ISSUES, [f"operator_issue:{op_id}"]

        # 3. Operator timing / duration query
        is_op_timing = bool(
            re.search(r"\b(?:when\s+did|hours\s+did|how\s+many\s+hours|shift\s+start|shift\s+end|shift\s+timing|duration|kitne\s+ghante|kab\s+shuru|kab\s+khatam|shuru\s+hui|khatam\s+hui)\b", q)
        )
        if is_op_timing:
            return Intent.GET_SHIFT_BY_OPERATOR, [f"operator_timing:{op_id}"]

    best_intent = Intent.UNKNOWN_QUERY
    best_score = 0.0
    best_terms: List[str] = []

    for intent, phrases, priority in _PATTERNS:
        if intent == Intent.LOG_WALL_PRODUCTION:
            if _is_future_or_planning(q) or _is_wall_production_question(q):
                continue
            if not re.search(r"\b(?:walls?|wells?)\b", q):
                continue
        if intent == Intent.LOG_SHIFT_ISSUE:
            if _is_future_or_planning(q) or _is_issue_question(q):
                continue
        if intent == Intent.RESOLVE_SHIFT_ISSUE:
            if _is_future_or_planning(q) or _is_issue_question(q):
                continue
        if intent == Intent.GET_ATTENDANCE_HOURS:
            if any(w in q for w in ("printing", "layer", "machine", "take", "until", "it been")):
                continue
        score, hits = _score(q, phrases)
        if score <= 0:
            continue
        weighted = priority * 1000 + score
        if weighted > best_score:
            best_intent, best_score, best_terms = intent, weighted, hits

    if _is_completed_wall_production(q) and best_intent != Intent.LOG_WALL_PRODUCTION:
        best_intent = Intent.LOG_WALL_PRODUCTION
        best_score = 8000.0
        best_terms = ["completed_wall_production"]

    if _is_site_detection_query(q) and best_intent not in (Intent.GET_SITE_DETECTIONS, Intent.GET_SITE_PEOPLE_COUNT):
        best_intent = Intent.GET_SITE_DETECTIONS
        best_score = 7500.0
        best_terms = ["site_detection_query"]

    shift_and_action_intents = {
        Intent.SHIFT_START,
        Intent.SHIFT_END,
        Intent.LOG_WALL_PRODUCTION,
        Intent.LOG_SHIFT_ISSUE,
        Intent.LOG_SHIFT_HANDOVER,
        Intent.RESOLVE_SHIFT_ISSUE,
        Intent.GET_ACTIVE_SHIFT,
        Intent.GET_PREVIOUS_SHIFT,
        Intent.GET_SHIFT_BY_DATE,
        Intent.GET_SHIFT_BY_OPERATOR,
        Intent.GET_WALL_HISTORY,
        Intent.GET_SHIFT_ISSUES,
        Intent.GET_HANDOVER_STATUS,
        Intent.GET_SITE_DETECTIONS,
        Intent.GET_ATTENDANCE_STATUS,
        Intent.GET_ATTENDANCE_HOURS,
        Intent.GET_ATTENDANCE_OVERTIME,
        Intent.GET_PRINT_PROGRESS,
        Intent.GET_OPERATOR_WORKING_DAYS,
        Intent.RECORD_WALLS_COMPLETED,
        Intent.GET_WALL_PRODUCTION,
        Intent.CORRECT_WALL_PRODUCTION,
    }

    # Protect write/action, site detections, and historical shift intents from being hijacked
    if best_score > 0 and best_intent in shift_and_action_intents:
        return best_intent, best_terms

    # Kelvin recommendation questions -> KNOWLEDGE_QUERY
    if "kelvin" in q and ("recommend" in q or "suggest" in q or "solution" in q):
        return Intent.KNOWLEDGE_QUERY, ["kelvin_recommendation"]

    is_knowledge = any(t in q for t in _KNOWLEDGE_TERMS)
    has_topic = any(t in q for t in _TECHNICAL_TOPICS)
    wants_explanation = any(t in q for t in _EXPLANATION_TERMS)
    reports_problem = any(t in q for t in _PROBLEM_TERMS)
    wants_live = any(t in q for t in _LIVE_HINT_TERMS)

    # A live intent, a live hint ("current", "abhi", "status", "kitni"), and no
    # sign of a problem -> the operator wants the reading, nothing else. This
    # runs FIRST so that widening _TECHNICAL_TOPICS cannot hijack a working live
    # query: "current layer width kitni hai" stays GET_LAYER_WIDTH.
    #
    # `is_knowledge` is deliberately NOT part of the problem signal here --
    # "what is" opens plenty of live questions, and counting it sent five live
    # queries to the wrong intent when measured.
    if best_score > 0 and wants_live and not (has_topic or reports_problem):
        return best_intent, best_terms

    # A live measurement PLUS a request to explain or a problem report ->
    # combined answer: resolve the real value, then add the documentation.
    if best_score > 0 and (wants_explanation or has_topic or reports_problem):
        return Intent.COMBINED_QUERY, best_terms + [t for t in _TECHNICAL_TOPICS if t in q]

    if has_topic or (is_knowledge and best_score == 0):
        return Intent.KNOWLEDGE_QUERY, [t for t in _TECHNICAL_TOPICS if t in q] or ["definition"]

    # A problem report that matched no live phrase. This must be answered from
    # the documentation, and it has to be decided HERE -- below this point the
    # fallback keyword table would map a bare "layer" or "print" to a live
    # intent and answer a defect question with a production number.
    if reports_problem and best_score == 0:
        return Intent.KNOWLEDGE_QUERY, [t for t in _PROBLEM_TERMS if t in q] or ["problem"]

    if best_score == 0:
        # Nothing matched a live phrase. Ask the documentation before the
        # fallback keyword table does, because that table only knows live
        # intents and will answer a documented question with a production
        # number. Gated on BOTH thresholds -- see their definitions above.
        if _documentation_can_answer(question, q):
            return Intent.KNOWLEDGE_QUERY, ["documented"]

        # Phase 1 Semantic NLU Fallback:
        # Attendance operations are strictly local and must NEVER call Gemini.
        is_attendance_term = any(w in q for w in ("shift", "attendance", "clock in", "clock out", "clocked in", "clocked out", "on shift", "kaam kar raha", "shift par", "shift pe"))
        if not is_attendance_term:
            try:
                from agent.nlu import semantic_fallback, ALLOWED_READ_INTENTS
                nlu_res = semantic_fallback(question)
                if (
                    nlu_res
                    and nlu_res.intent in ALLOWED_READ_INTENTS
                    and nlu_res.confidence >= 0.70
                    and not nlu_res.ambiguous
                ):
                    log.info(
                        "Semantic NLU rescued query %r -> %s (confidence %.2f)",
                        question, nlu_res.intent.value, nlu_res.confidence
                    )
                    return nlu_res.intent, ["semantic_nlu", nlu_res.intent.value.lower()]
            except Exception as exc:
                log.warning("Semantic NLU fallback failed (%s); proceeding to keyword fallback", exc)

        # Still nothing. Look for a single machine word -- a mangled
        # transcript still names its subject.
        for keyword, intent in _fallback_keywords():
            if keyword in q:
                log.info("No phrase matched; falling back on the keyword %r", keyword)
                return intent, [keyword]
        return Intent.UNKNOWN_QUERY, []

    return best_intent, best_terms


def _documentation_can_answer(question: str, normalised: str) -> bool:
    """
    Whether retrieval has a confident answer for a question that matched no
    live phrase.

    Deliberately conservative: a false positive here sends a working live query
    to the documentation, which is worse than leaving a documented question to
    the fallback table. Any failure is swallowed -- retrieval being unavailable
    must never change how a live question is routed.
    """
    try:
        result = search_documentation(question)
        if not result.get("found"):
            return False
        score = float(result["results"][0].get("score") or 0.0)
        if score < FALLBACK_RAG_MIN_SCORE:
            return False
        informative = [t for t in tokenize(normalised) if t not in _STOPWORDS]
        if len(informative) < FALLBACK_RAG_MIN_TERMS:
            return False
        log.info(
            "No live phrase matched; the documentation answers this (score %.3f)", score
        )
        return True
    except Exception as exc:  # retrieval is advisory here, never load-bearing
        log.warning("Could not check the documentation before fallback: %s", exc)
        return False


def _evaluate_rag_verified(top_text: str) -> int:
    """
    Determines whether a retrieved RAG document is SITE-VERIFIED (1) or unverified (0).
    Rules:
      - CONFLICT / DISPUTED / DO NOT ASSUME / MUST CONFIRM -> 0
      - UNVERIFIED / NOT CONFIRMED / UNCONFIRMED / POSSIBLE ONLY -> 0
      - DOCUMENTED -> 0
      - SITE-VERIFIED only (without unconfirmed, conflict, or documented qualifiers) -> 1
    Never marks verified on bare substring alone.
    """
    fields = parse_record(top_text)
    status_raw = ""
    if fields and "verification status" in fields:
        status_raw = fields["verification status"].get("value", "")
    else:
        for line in top_text.splitlines():
            if "verification status" in line.lower():
                status_raw = line.split(":", 1)[1].strip()
                break

    st = status_raw.upper()
    if not st:
        return 0
    if any(k in st for k in ("CONFLICT", "DISPUTED", "DO NOT ASSUME", "MUST CONFIRM")):
        return 0
    if any(k in st for k in ("UNVERIFIED", "NOT CONFIRMED", "UNCONFIRMED", "POSSIBLE ONLY")):
        return 0
    if "DOCUMENTED" in st:
        return 0
    if "SITE-VERIFIED" in st and not any(neg in st for neg in ("NOT SITE-VERIFIED", "PENDING")):
        return 1
    return 0


def _fallback_keywords() -> List[Tuple[str, Intent]]:
    """Single machine words, most specific first, for mangled transcripts."""
    return [
        ("baki", Intent.GET_REMAINING_PRODUCTION),
        ("remaining", Intent.GET_REMAINING_PRODUCTION),
        ("bacha", Intent.GET_REMAINING_PRODUCTION),
        ("operator", Intent.GET_OPERATOR),
        ("layer", Intent.GET_CURRENT_LAYER),
        ("width", Intent.GET_LAYER_WIDTH),
        ("material", Intent.GET_MATERIAL_USAGE),
        ("camera", Intent.GET_CAMERA_STATUS),
        ("quality", Intent.GET_PRINT_QUALITY),
        ("printing", Intent.GET_COMPLETED_PRODUCTION),
        ("print", Intent.GET_COMPLETED_PRODUCTION),
        ("production", Intent.GET_COMPLETED_PRODUCTION),
        ("wall", Intent.GET_COMPLETED_PRODUCTION),
        ("well", Intent.GET_COMPLETED_PRODUCTION),
    ]


# ---------------------------------------------------------------------------
# Resolvers -- each returns real facts or an explicit unavailability
# ---------------------------------------------------------------------------

def _unavailable(reason: str) -> RouterResult:
    return RouterResult(
        intent=Intent.UNKNOWN_QUERY, available=False, reason=reason or UNAVAILABLE_MESSAGE
    )


def _progress_facts() -> Tuple[bool, Dict[str, Any], str]:
    progress = ms.get_print_progress()
    if not progress.get("available"):
        return False, {}, progress.get("reason", UNAVAILABLE_MESSAGE)
    return True, progress, ""


def _job_and_timing() -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    job = db.get_active_job()
    return job, metrics.production_metrics(job)


def _unwrap(entry: Any) -> Any:
    """metrics fields are {value, provenance, unit, reason}; keep them intact."""
    return entry


def _spoken_line(line_id: str) -> str:
    m = re.fullmatch(r"line[_\- ]?(\w+)", (line_id or "").strip(), re.IGNORECASE)
    return f"Line {m.group(1)}" if m else (line_id or "this line")


def extract_date_from_text(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
        "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?\b", text)
    if m:
        day = int(m.group(1))
        m_name = m.group(2).lower()
        if m_name in months:
            y = int(m.group(3)) if m.group(3) else datetime.date.today().year
            return f"{y:04d}-{months[m_name]:02d}-{day:02d}"
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?\b", text)
    if m:
        m_name = m.group(1).lower()
        if m_name in months:
            day = int(m.group(2))
            y = int(m.group(3)) if m.group(3) else datetime.date.today().year
            return f"{y:04d}-{months[m_name]:02d}-{day:02d}"
    low = text.lower()
    if "yesterday" in low or "kal" in low:
        return (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    if "today" in low or "aaj" in low:
        return datetime.date.today().isoformat()
    return None


def extract_wall_number(text: str) -> Optional[int]:
    """
    Extracts a wall number (1, 2, 3...) from text.
    Supports:
      - Suffix digits: 'wall 1', 'wall 2', 'wall no. 3', 'wall number 4'
      - Prefix digits: '1 wall', '2 walls', '2 wall'
      - Ordinals: '1st wall', 'second wall', 'dusri wall', 'pehli wall', '3rd wall'
      - Cardinal words: 'one wall', 'two walls', 'ek wall', 'do wall', 'wall one', 'wall do'
    Returns None if no wall number or ordinal is specified.
    """
    if not text:
        return None
    t = _normalise(text)

    wall_re = r"(?:walls?|wells?)"

    # 1. Ordinals with digit suffix (e.g. '1st wall', '2nd wall', 'wall 1st', 'wall 2nd')
    m = re.search(rf"\b(\d+)(?:st|nd|rd|th)\s+{wall_re}\b", t)
    if m:
        return int(m.group(1))
    m = re.search(rf"\b{wall_re}\s*(?:number\s*|no\.?\s*)?(\d+)(?:st|nd|rd|th)\b", t)
    if m:
        return int(m.group(1))

    # 2. Plain digits after wall (e.g. 'wall 1', 'wall no 2', 'wall number 3')
    m = re.search(rf"\b{wall_re}\s*(?:number\s*|no\.?\s*)?(\d+)\b", t)
    if m:
        return int(m.group(1))

    # 3. Plain digits before wall (e.g. '1 wall', '2 walls', '2 wall')
    m = re.search(rf"\b(\d+)\s+{wall_re}\b", t)
    if m:
        return int(m.group(1))

    # 4. Ordinal words (before or after wall)
    ordinals = {
        "first": 1, "pehli": 1, "pehla": 1, "pahli": 1, "pahla": 1,
        "second": 2, "dusri": 2, "dusra": 2, "doosri": 2, "doosra": 2,
        "third": 3, "teesri": 3, "teesra": 3, "tisri": 3, "tisra": 3,
        "fourth": 4, "chauthi": 4, "chautha": 4,
        "fifth": 5, "panchvi": 5, "panchva": 5, "panchveen": 5, "panchwein": 5,
    }
    for w, num in ordinals.items():
        if re.search(rf"\b{w}\s+{wall_re}\b", t) or re.search(rf"\b{wall_re}\s+(?:number\s*|no\.?\s*)?{w}\b", t):
            return num

    # 5. Cardinal words (before or after wall)
    cardinals = {
        "ek": 1, "one": 1,
        "do": 2, "two": 2,
        "teen": 3, "three": 3,
        "char": 4, "chaar": 4, "four": 4,
        "panch": 5, "five": 5,
    }
    for w, num in cardinals.items():
        if re.search(rf"\b{w}\s+{wall_re}\b", t) or re.search(rf"\b{wall_re}\s+{w}\b", t):
            return num

    return None


def extract_downtime(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minute|minutes|minto|ghante|ghanta)", text.lower())
    if m:
        val = float(m.group(1))
        if "ghant" in text.lower():
            val *= 60.0
        return val
    return 0.0


# Words that can sit where a name would and are never a name. Without this,
# "okay here start my shift" would offer "okay" as the operator.
# "Anand here" / "this is Anand" / "main Anand hoon" / "Anand bol raha hai".
# One capture group per form, tried in order.
_NAME_PATTERNS = (
    re.compile(r"\b(?:this\s+is|it'?s)\s+([a-z][a-z'\-]{1,29})\b"),
    re.compile(r"\bmain\s+([a-z][a-z'\-]{1,29})\s+(?:hoon|hu|bol)"),
    re.compile(r"\b([a-z][a-z'\-]{1,29})\s+bol\s+(?:raha|rahi)\b"),
    re.compile(r"\b([a-z][a-z'\-]{1,29})\s+here\b"),
    re.compile(r"\b([a-z][a-z'\-]{1,29})\s+speaking\b"),
)


def extract_operator_name(text: str, db_path: Optional[str] = None) -> Optional[str]:
    """
    The name the operator said, or None.
    Handles self-introductions ("Anand here") and explicit commands ("start Hamadan's shift").
    """
    if not text:
        return None
    lowered = (text or "").lower()
    lowered = _strip_wake_word(re.sub(r"[^a-z0-9' \-]+", " ", lowered).strip())
    lowered = re.sub(r"\s+", " ", lowered).strip()

    # 1. Check self-introduction patterns first (returns spoken lowercase name)
    for pattern in _NAME_PATTERNS:
        match = pattern.search(lowered)
        if not match:
            continue
        raw_name = match.group(1).lower().strip()
        if raw_name and raw_name not in _NOT_A_NAME:
            return raw_name

    # 2. Check 3rd-person action and query patterns
    action_patterns = (
        re.compile(r"^([a-z0-9_\- ]+?)\s+clock\s+in\b"),
        re.compile(r"^([a-z0-9_\- ]+?)\s+(?:ko\s+)?clock\s+in\s+karo\b"),
        re.compile(r"^([a-z0-9_\- ]+?)\s+clock\s+out\b"),
        re.compile(r"^([a-z0-9_\- ]+?)\s+(?:ko\s+)?clock\s+out\s+karo\b"),
        re.compile(r"\bclock\s+(?:in|out)\s+(?:for\s+)?([a-z0-9_\- ]+)\b"),
        re.compile(r"\b(?:start|begin|end|close|stop)\s+(?:the\s+|a\s+|new\s+|current\s+)?(?:shift|attendance)\s+(?:for|of)\s+([a-z0-9_\- ]+)\b"),
        re.compile(r"\b(?:start|begin|end|close|stop)\s+([a-z0-9_\- ]+?)(?:'s|s)?\s+(?:shift|attendance)\b"),
        re.compile(r"\b([a-z0-9_\- ]+?)\s+(?:ka|ki|ke|ko)\s+(?:shift|attendance)\s+(?:start|begin|end|close|stop|shuru|chalu|lagao|mark|khatam|band)\b"),
        re.compile(r"\bwhen\s+did\s+([a-z0-9_\- ]+?)\s+start\b"),
        re.compile(r"\bhow\s+(?:many\s+hours|long)\s+has\s+([a-z0-9_\- ]+?)\s+worked\b"),
        re.compile(r"\bhow\s+many\s+working\s+days\s+has\s+([a-z0-9_\- ]+)\b"),
    )
    for p in action_patterns:
        match = p.search(lowered)
        if match:
            candidate = _clean_candidate_name(match.group(1))
            if candidate and candidate.lower() not in _NOT_A_NAME:
                return candidate.lower()

    # 3. Match registered operators in database
    op_id, op_name = extract_operator_tuple_from_text(lowered, db_path=db_path)
    if op_name:
        return op_name.lower()

    return None


def extract_shift_type(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ("night", "raat", "evening")):
        return "Night"
    if any(w in lower for w in ("afternoon", "dopahar", "second")):
        return "Afternoon"
    return "Morning"


def refresh_known_operators():
    """Refresh cached known operators (kept for test/caller compatibility)."""
    pass


def extract_operator_tuple_from_text(text: str, db_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Extracts (operator_id, operator_name) from natural text by matching registered operators.
    Matches operator_id, full name, or first name against word boundaries.
    """
    if not text:
        return None, None
    q_norm = re.sub(r"['’]s\b", "", text.lower())
    q_norm = q_norm.replace("'", " ").replace("’", " ")
    try:
        ops = shift_store.list_operators(db_path=db_path)
    except Exception:
        return None, None

    matched: List[Tuple[str, str]] = []
    for op in ops:
        op_id = op["operator_id"]
        op_name = op.get("name") or op_id
        # Check operator_id (e.g. OP001, OP_ANAND)
        if re.search(rf"\b{re.escape(op_id.lower())}\b", q_norm):
            matched.append((op_id, op_name))
            continue
        # Check employee code (e.g. 'code 13', 'operator code 13')
        emp_code = str(op.get("employee_code") or "").strip().lower()
        if emp_code and re.search(rf"\b(?:operator\s+code|employee\s+code|code)\s*:?\s*{re.escape(emp_code)}\b", q_norm):
            matched.append((op_id, op_name))
            continue
        # Check full name (e.g. Anand Sharma)
        full_norm = op_name.lower().strip()
        if full_norm and re.search(rf"\b{re.escape(full_norm)}\b", q_norm):
            matched.append((op_id, op_name))
            continue
        # Check first name (if >= 3 characters, e.g. Anand)
        parts = full_norm.split()
        if parts:
            first_name = parts[0]
            if len(first_name) >= 3 and re.search(rf"\b{re.escape(first_name)}\b", q_norm):
                matched.append((op_id, op_name))
                continue

    if len(matched) == 1:
        return matched[0]
    return None, None


def extract_operator_from_text(text: str, db_path: Optional[str] = None) -> Optional[str]:
    """Finds known operator ID from question text."""
    op_id, _ = extract_operator_tuple_from_text(text, db_path=db_path)
    return op_id


def extract_issue_id(text: str) -> Optional[int]:
    m = re.search(r"\b(?:issue|problem)\s*(?:number\s*|no\.?\s*|#\s*)?(\d+)\b", text.lower())
    if m:
        return int(m.group(1))
    return None


def extract_resolution_cause(text: str) -> Optional[str]:
    """
    Extracts operator-reported cause safely, preserving operator wording.
    Examples:
    - 'cause material buildup tha' -> 'material buildup'
    - 'reason was material blockage' -> 'material blockage'
    - 'cause tha material jam ho gaya tha' -> 'material jam ho gaya'
    - 'wajah material blockage thi' -> 'material blockage'
    - 'due to material blockage' -> 'material blockage'
    If question ('iska reason kya tha') or not provided, returns None.
    """
    low = text.strip()
    if re.search(r"\b(?:reason kya|cause kya|kya reason|kya cause|wajah kya|kya wajah)\b", low, re.IGNORECASE):
        return None
    if re.search(r"\b(?:kya|what|kaun|kaunsi|why|kyun)\b", low, re.IGNORECASE) and not re.search(r"\b(?:tha|thi|was|is)\b", low, re.IGNORECASE):
        return None

    m = re.search(r"\b(?:cause|reason|wajah)\s+(?:was|tha|thi|hai|is)?\s*:?\s*(.+?)(?:\s*(?:aur|action|result|and|,|\.|$))", low, re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        c = re.sub(r"\s+(?:tha|thi|hai)$", "", c, flags=re.IGNORECASE).strip()
        if c and not any(w in c.lower() for w in ["kya", "what", "pata nahi", "unknown"]):
            return c
    m = re.search(r"\bdue to\s+(.+?)(?:\s*(?:aur|action|result|and|,|\.|$))", low, re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        c = re.sub(r"\s+(?:tha|thi|hai)$", "", c, flags=re.IGNORECASE).strip()
        if c:
            return c
    m = re.search(r"(.+?)\s+ki wajah se\b", low, re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        if c:
            return c
    return None


def extract_operator_action(text: str) -> Tuple[Optional[str], bool]:
    """
    Returns (action_taken, is_rag_recommended).
    is_rag_recommended is True ONLY if the operator explicitly states they applied/followed
    the Kelvin recommendation.
    A vague reference like 'Kelvin ne bola tha' or 'Kelvin said' returns (None, False).
    """
    low = text.lower().strip()

    # Informational questions seeking advice/procedures are NOT actions
    question_indicators = [
        "how", "kaise", "what", "kya", "why", "kyun", "karein",
        "karna hai", "karna chahiye", "should i", "can i", "could i"
    ]
    if any(re.search(rf"\b{qi}\b", low) for qi in question_indicators):
        return None, False

    applied_terms = ["apply kiya", "apply kar diya", "applied", "followed", "follow kiya", "use kiya", "used", "implement kiya", "implemented"]
    kelvin_terms = ["kelvin", "recommendation", "suggestion"]

    has_kelvin = any(k in low for k in kelvin_terms)
    has_applied = any(a in low for a in applied_terms)

    if has_kelvin:
        if has_applied:
            return "Applied Kelvin recommendation", True
        else:
            return None, False

    action_verbs = [
        "kiya", "kar diya", "kara", "cleaned", "flushed", "restarted", "replaced",
        "checked", "inspected", "examined", "fixed", "repaired", "cleared",
        "adjusted", "changed", "swapped", "tightened", "reset", "unblocked"
    ]

    if any(v in low for v in action_verbs) or re.search(r"\b(?:action|kadam|step)\b", low):
        act = text.strip()
        m = re.search(r"\b(?:action taken|action|step taken)\s*[:=]?\s*(.+)", act, re.IGNORECASE)
        if m:
            act = m.group(1).strip()
        # Clean resolution suffixes from the action string
        res_suffix_patterns = [
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?(?:the\s+)?(?:material\s+flow\s+is\s+normal(?:\s+now)?|material\s+flow\s+normal(?:\s+now)?|ab\s+material\s+flow\s+normal\s+hai)\.?$",
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?(?:the\s+)?(?:issue|problem)\s*(?:is\s+)?(?:resolved?|fixed?|solved?)\.?$",
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?(?:the\s+)?(?:issue|problem)?\s*(?:thik|theek|solve|resolved?|fix)\s*(?:ho gaya|ho gayi|ho gya|kar diya)?\.?$",
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?(?:is\s+)?(?:resolved?|fixed?|solved?)\.?$",
            r"(?:,\s*|\s+aur\s+|\s+and\s+)?it\s+is\s+working\s+now\.?$",
        ]
        for pat in res_suffix_patterns:
            act = re.sub(pat, "", act, flags=re.IGNORECASE).strip()
        act = act.rstrip(".,;").strip()
        if act:
            return act, False

    return None, False


def extract_resolution_result(text: str) -> Optional[str]:
    low = text.lower().strip()

    conditional_markers = [
        "agar", "if ", "if,", "will", "hoga", "hogi", "jayega", "jayegi",
        "dekhte", "check karte", "might", "maybe", "shayad", "hope"
    ]
    if any(m in low for m in conditional_markers):
        return None

    negative_markers = [
        "not resolved", "still stuck", "abhi bhi", "thik nahi", "theek nahi",
        "solve nahi", "not fixed", "unresolved", "kam nahi kar raha", "fail", "nahi hua"
    ]
    if any(m in low for m in negative_markers):
        return "not_resolved"

    positive_markers = [
        "the material flow is normal now", "material flow is normal now",
        "the material flow is normal", "material flow is normal",
        "ab material flow normal hai", "material flow normal hai", "material flow normal",
        "the problem is resolved", "problem is resolved", "the issue is resolved", "issue is resolved",
        "problem resolve ho gayi", "problem resolve ho gaya",
        "the issue is fixed", "issue is fixed", "issue fixed",
        "the problem is fixed", "problem is fixed", "problem fixed",
        "issue fix ho gaya", "issue fix ho gayi", "problem fix ho gaya", "problem fix ho gayi",
        "resolved", "solve ho gaya", "solve ho gayi", "solve ho gya",
        "problem solved", "issue solved", "thik ho gaya", "theek ho gaya",
        "thik ho gya", "theek ho gya", "fix ho gaya", "fix ho gya",
        "resume ho gayi", "resumed", "chal pada", "chal rahi hai ab",
        "it is working now",
        "solved", "mark as resolved", "issue resolve ho gaya"
    ]
    if any(m in low for m in positive_markers):
        return "resolved"

    return None


def extract_domain_keywords(text: str) -> List[str]:
    keywords = [
        "nozzle", "hopper", "pump", "feeder", "temperature", "motor",
        "sensor", "heater", "material", "pressure", "hose", "valve",
        "bed", "gantry", "extruder", "robot"
    ]
    low = text.lower()
    return [kw for kw in keywords if kw in low]


def resolve(intent: Intent, question: str, context: Optional[Dict[str, Any]] = None) -> RouterResult:
    """Pulls the real data for an intent. Never fabricates."""

    # --- production ---
    if intent in (
        Intent.GET_COMPLETED_PRODUCTION,
        Intent.GET_REMAINING_PRODUCTION,
        Intent.GET_TOTAL_TARGET,
    ):
        ok, progress, reason = _progress_facts()
        if not ok:
            return RouterResult(intent, available=False, reason=reason)
        return RouterResult(intent, facts={
            "job_name": progress.get("job_name"),
            "completed_mm": progress["completed_mm"],
            "remaining_mm": progress["remaining_mm"],
            "total_mm": progress["total_mm"],
            "percentage": progress["percentage"],
            "unit": "mm",
            # The job tracks LENGTH. There is no wall/piece count column in the
            # database, so a "how many walls" question cannot be answered as a
            # count -- and must not be answered as though millimetres were walls.
            "measurement_basis": "length_mm",
            "wall_count": None,
            "wall_count_reason": (
                "Production is tracked by printed length, not by a wall or piece "
                "count. No wall-count field exists in the database."
            ),
        })

    if intent == Intent.GET_PRINT_RATE:
        _job, timing = _job_and_timing()
        entry = timing.get("print_rate_mm_per_min", {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={"print_rate_mm_per_min": entry["value"], "unit": "mm/min"})

    if intent == Intent.GET_ELAPSED_TIME:
        _job, timing = _job_and_timing()
        entry = timing.get("elapsed", {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={"elapsed": entry["value"]})

    if intent == Intent.GET_ETA:
        _job, timing = _job_and_timing()
        entry = timing.get("eta", {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={"eta": entry["value"]})

    if intent == Intent.GET_DEADLINE_STATUS:
        _job, timing = _job_and_timing()
        entry = timing.get("deadline_status", {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "deadline_status": entry["value"],
            "time_to_deadline": timing.get("time_to_deadline", {}).get("value"),
        })

    # --- layers / geometry ---
    if intent == Intent.GET_CURRENT_LAYER:
        layer = ms.get_current_layer()
        if not layer.get("available"):
            return RouterResult(intent, available=False, reason=layer.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "current_layer": layer["current_layer"],
            "total_layers": layer.get("total_layers"),
        })

    if intent == Intent.GET_LAYER_WIDTH:
        width = ms.get_current_layer_width()
        if not width.get("available"):
            return RouterResult(intent, available=False, reason=width.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "layer_width_mm": width["layer_width_mm"],
            "confidence": width.get("confidence"),
            "unit": "mm",
            "note": width.get("note"),
        })

    if intent in (Intent.GET_LAYER_HEIGHT, Intent.GET_WALL_DIMENSIONS):
        job = db.get_active_job()
        from agent.services.machine_state import _snapshot  # live vision snapshot

        vision = _snapshot() or {}
        geometry = metrics.geometry_metrics(job, {"layers": {
            "layer_height_mm": vision.get("layer_height_mm"),
            "layer_height_px": vision.get("layer_height_px"),
            "layer_width_mm": vision.get("layer_width"),
            "layer_width_px": vision.get("layer_width_px"),
        }})
        key = "layer_height_mm" if intent == Intent.GET_LAYER_HEIGHT else "wall_height_mm"
        entry = geometry.get(key, {})
        if entry.get("value") is None:
            extra = geometry.get("wall_width_mm", {}) if intent == Intent.GET_WALL_DIMENSIONS else {}
            if extra.get("value") is not None:
                return RouterResult(intent, facts={"wall_width_mm": extra["value"], "unit": "mm"})
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        facts = {key: entry["value"], "unit": "mm"}
        if intent == Intent.GET_WALL_DIMENSIONS:
            facts["wall_width_mm"] = geometry.get("wall_width_mm", {}).get("value")
        return RouterResult(intent, facts=facts)

    # --- material ---
    if intent in (Intent.GET_MATERIAL_USAGE, Intent.GET_MATERIAL_REMAINING, Intent.GET_MATERIAL_RATE):
        job, timing = _job_and_timing()
        elapsed = timing.get("elapsed_seconds", {}).get("value")
        material = metrics.material_metrics(job, elapsed)
        key = {
            Intent.GET_MATERIAL_USAGE: "material_used_kg",
            Intent.GET_MATERIAL_REMAINING: "material_remaining_kg",
            Intent.GET_MATERIAL_RATE: "consumption_rate_kg_per_hour",
        }[intent]
        entry = material.get(key, {})
        if entry.get("value") is None:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={key: entry["value"], "unit": entry.get("unit", "kg")})

    # --- history ---
    if intent in (Intent.GET_24_HOUR_PRODUCTION, Intent.GET_SHIFT_PRODUCTION, Intent.GET_RECENT_EVENTS):
        history = metrics.production_history()
        window = "24h" if intent == Intent.GET_24_HOUR_PRODUCTION else (
            "8h" if intent == Intent.GET_SHIFT_PRODUCTION else "10m")
        entry = history["windows"].get(window, {})
        events = db.get_recent_events(limit=5) if intent == Intent.GET_RECENT_EVENTS else None
        if entry.get("value") is None and not events:
            return RouterResult(intent, available=False, reason=entry.get("reason", UNAVAILABLE_MESSAGE))
        facts: Dict[str, Any] = {"window": window}
        if entry.get("value") is not None:
            facts["produced_mm"] = entry["value"]
            facts["unit"] = "mm"
        else:
            facts["produced_note"] = entry.get("reason", "")
        if events:
            facts["events"] = [
                {"type": e["event_type"], "message": e["message"],
                 "seconds_ago": round(__import__("time").time() - e["timestamp"], 1)}
                for e in events
            ]
        return RouterResult(intent, facts=facts)

    if intent == Intent.WAKE_WORD:
        return RouterResult(intent, facts={"acknowledged": True})

    # --- operator ---
    if intent == Intent.GET_OPERATOR:
        # The operator is a shop-floor fact, not a measurement: no camera or
        # sensor can identify a person. It comes from OPERATOR_NAME in .env,
        # and when that is not set the honest answer is that it is not recorded
        # -- never a guessed name.
        from agent import config as _config

        name = (_config.OPERATOR_NAME or "").strip()
        if not name:
            return RouterResult(
                intent, available=False,
                reason=("The current operator is not recorded. No operator name is "
                        "configured, and the machine has no way to identify who is "
                        "running it. Set OPERATOR_NAME in the .env file."),
            )
        return RouterResult(intent, facts={"operator": name, "source": "configured"})

    # --- status ---
    if intent == Intent.GET_MACHINE_STATUS:
        status = ms.get_current_print_status()
        machine = ms.get_machine_status()
        if not status.get("available") and not machine.get("available"):
            return RouterResult(intent, available=False, reason=status.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "printing": status.get("printing"),
            "status": status.get("status"),
            "confidence": machine.get("confidence"),
            "note": status.get("note") or machine.get("note"),
        })

    if intent in (Intent.GET_ROBOT_STATUS, Intent.GET_CAMERA_STATUS, Intent.GET_COMMUNICATION_STATUS):
        machine = ms.get_machine_status()
        camera = (machine.get("camera") or {}) if machine.get("available") else {}
        if intent == Intent.GET_CAMERA_STATUS:
            if not camera:
                return RouterResult(intent, available=False,
                                    reason="Camera status is not available from the pipeline.")
            return RouterResult(intent, facts={
                "camera_connected": camera.get("connected"),
                "camera_status": camera.get("status"),
                "fps": camera.get("fps"),
            })
        if not machine.get("available"):
            return RouterResult(intent, available=False, reason=machine.get("reason", UNAVAILABLE_MESSAGE))
        return RouterResult(intent, facts={
            "camera_connected": camera.get("connected"),
            "machine_status": machine.get("status"),
            "machine_link": "not connected",
            "machine_link_note": (
                "No machine communication link is connected; robot telemetry such as "
                "pump and accelerator is unavailable."
            ),
        })

    if intent == Intent.GET_SITE_PEOPLE_COUNT:
        site_data = ms.get_site_people_count()
        if not site_data.get("available"):
            return RouterResult(
                intent,
                available=False,
                reason=site_data.get("reason") or "Site camera (Camera 2) is currently unavailable.",
            )
        return RouterResult(
            intent,
            facts={
                "people_count": site_data.get("people_count", 0),
                "confidence": site_data.get("confidence", 1.0),
                "provenance": site_data.get("provenance", "REAL"),
                "detector": site_data.get("detector", "unknown"),
                "note": site_data.get("note", ""),
            },
        )

    if intent == Intent.GET_SITE_DETECTIONS:
        site_data = ms.get_site_detections()
        if not site_data.get("available"):
            return RouterResult(
                intent,
                available=False,
                reason=site_data.get("reason") or "Site camera (Camera 2) is currently unavailable.",
            )

        entities = site_data.get("entities", [])
        supported_classes = site_data.get("supported_classes", [])

        # Dynamically check target class against loaded model's supported_classes
        candidate = extract_requested_detection_class(question or "")
        unsupported_class = None
        target_class = None
        target_entities = []

        if candidate:
            matched = _find_supported_class(candidate, supported_classes)
            if matched:
                target_class = matched
                target_entities = [
                    e for e in entities
                    if (e.get("class_name") if isinstance(e, dict) else getattr(e, "class_name", "")).lower() == matched.lower()
                ]
            else:
                unsupported_class = candidate

        return RouterResult(
            intent,
            facts={
                "entities": entities,
                "people_count": site_data.get("people_count", 0),
                "supported_classes": supported_classes,
                "target_class": target_class,
                "target_entities": target_entities,
                "unsupported_class": unsupported_class,
                "confidence": site_data.get("confidence", 1.0),
                "provenance": site_data.get("provenance", "REAL"),
                "detector": site_data.get("detector", "unknown"),
            },
        )

    if intent == Intent.GET_PRINT_QUALITY:
        from agent.services.machine_state import _live_state

        snapshot = _live_state.snapshot() if _live_state is not None else {}
        quality = ((snapshot.get("vision") or {}).get("print_quality") or {}) if isinstance(snapshot, dict) else {}
        score = quality.get("score_percent")
        if isinstance(score, dict) or score is None:
            reason = score.get("reason") if isinstance(score, dict) else "Print quality has not been scored yet."
            return RouterResult(intent, available=False, reason=reason or UNAVAILABLE_MESSAGE)
        return RouterResult(intent, facts={
            "quality_percent": score,
            "quality_status": quality.get("status"),
            "issues": quality.get("issues", []),
        })

    if intent == Intent.GET_RECENT_PROBLEMS:
        from agent.services.machine_state import _live_state

        snapshot = _live_state.snapshot() if _live_state is not None else {}
        alerts = snapshot.get("recent_alerts", []) if isinstance(snapshot, dict) else []
        events = db.get_recent_events(limit=5)
        if not alerts and not events:
            return RouterResult(intent, facts={"problems": [], "message": "No problems have been recorded."})
        return RouterResult(intent, facts={
            "alerts": alerts,
            "events": [{"type": e["event_type"], "message": e["message"]} for e in events],
        })

    # --- knowledge ---
    if intent == Intent.KNOWLEDGE_QUERY:
        result = search_documentation(question)
        return RouterResult(
            intent,
            knowledge=result,
            available=result.get("found", False),
            reason="" if result.get("found") else result.get("message", ""),
        )

    # --- Phase 2C Historical Queries ---
    if intent == Intent.GET_ACTIVE_SHIFT:
        try:
            line_id = (context.get("line_id") if context else None) or "line_1"
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                line_label = _spoken_line(line_id)
                return RouterResult(intent, facts={"active": False, "line_id": line_id, "message": f"No active shift is currently running on {line_label}."})
            op = shift_store.get_operator(active["operator_id"])
            return RouterResult(intent, facts={
                "active": True,
                "shift_id": active["shift_id"],
                "shift_type": active["shift_type"],
                "shift_date": active["shift_date"],
                "line_id": active["line_id"],
                "operator_id": active["operator_id"],
                "operator_name": op["name"] if op else active["operator_id"],
                "target_walls": active["target_walls"],
            })
        except Exception as exc:
            log.warning("Could not resolve GET_ACTIVE_SHIFT: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query active shift.")

    if intent == Intent.GET_PREVIOUS_SHIFT:
        try:
            line_id = (context.get("line_id") if context else None) or "line_1"
            shifts = shift_store.list_shifts(status="closed", limit=10)
            line_prev = [s for s in shifts if s.get("line_id", "line_1") == line_id]
            if not line_prev:
                return RouterResult(intent, facts={"found": False, "message": f"No previous closed shift was found for {line_id}."})
            prev = line_prev[0]
            op = shift_store.get_operator(prev["operator_id"])
            op_name = op["name"] if op else prev["operator_id"]

            st = prev.get("start_time")
            et = prev.get("end_time")
            dur_sec = (et - st) if (st is not None and et is not None) else None
            dur_formatted = ""
            if dur_sec is not None and dur_sec >= 0:
                hours = int(dur_sec // 3600)
                minutes = int((dur_sec % 3600) // 60)
                dur_formatted = f"{hours} hours {minutes} minutes" if hours > 0 else f"{minutes} minutes"

            st_formatted = datetime.datetime.fromtimestamp(st).strftime("%I:%M %p").lstrip("0") if st else ""
            et_formatted = datetime.datetime.fromtimestamp(et).strftime("%I:%M %p").lstrip("0") if et else ""

            return RouterResult(intent, facts={
                "found": True,
                "shift_id": prev["shift_id"],
                "shift_type": prev["shift_type"],
                "shift_date": prev["shift_date"],
                "line_id": line_id,
                "operator_id": prev["operator_id"],
                "operator_name": op_name,
                "start_time": st,
                "end_time": et,
                "start_time_formatted": st_formatted,
                "end_time_formatted": et_formatted,
                "duration_seconds": dur_sec,
                "duration_hours": round(dur_sec / 3600.0, 2) if dur_sec is not None else None,
                "duration_formatted": dur_formatted,
                "summary_notes": prev.get("summary_notes") or "No notes recorded.",
            })
        except Exception as exc:
            log.warning("Could not resolve GET_PREVIOUS_SHIFT: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query previous shift.")


    if intent == Intent.GET_ATTENDANCE_HOURS:
        try:
            today = shift_store.local_today()
            records = shift_store.list_attendance_records(date_filter=today)
            op_id, op_name = extract_operator_tuple_from_text(question)
            if not op_id:
                spoken = extract_operator_name(question)
                if spoken and spoken.lower() not in ("i", "my", "me"):
                    op = shift_store.find_operator_by_name(spoken, active_only=False)
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]
            if not op_id and context:
                ctx_op = context.get("operator_id")
                if ctx_op:
                    op = shift_store.get_operator(ctx_op)
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]

            matching_records = records
            if op_id or op_name:
                matching_records = [
                    r for r in records
                    if (op_id and r.get("operator_id") == op_id) or
                       (op_name and (r.get("employee_name") or "").lower() == op_name.lower())
                ]
                if not matching_records:
                    # Look up latest attendance record across dates
                    with db.connect() as conn:
                        row = conn.execute(
                            "SELECT * FROM attendance_records WHERE (operator_id = ? OR employee_name LIKE ?) ORDER BY shift_start DESC LIMIT 1",
                            (op_id, f"%{op_name}%")
                        ).fetchone()
                    if row:
                        matching_records = [shift_store.enrich_attendance_record(dict(row))]

            if not matching_records and (op_name or op_id):
                return RouterResult(
                    intent,
                    facts={"records": [], "operator_name": op_name or op_id, "date": today},
                    reason=f"No attendance record found for {op_name or op_id}."
                )

            if not matching_records:
                return RouterResult(intent, available=False, reason="No attendance records found for today.")

            return RouterResult(intent, facts={"records": matching_records, "operator_name": op_name, "date": today})
        except Exception as exc:
            log.warning("Could not resolve GET_ATTENDANCE_HOURS: %s", exc)
            return RouterResult(intent, available=False, reason="Could not fetch attendance hours.")

    if intent == Intent.GET_OPERATOR_WORKING_DAYS:
        try:
            op_id, op_name = extract_operator_tuple_from_text(question)
            if not op_id:
                spoken = extract_operator_name(question)
                if spoken and spoken.lower() not in ("i", "my", "me"):
                    op = shift_store.find_operator_by_name(spoken, active_only=False)
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]
            if not op_id and context:
                ctx_op = context.get("operator_id")
                if ctx_op:
                    op = shift_store.get_operator(ctx_op)
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]

            if not op_id:
                return RouterResult(
                    intent,
                    facts={"found": False},
                    reason="Please specify which operator you want to check.",
                    available=False,
                )

            with db.connect() as conn:
                att_dates = [
                    row[0] for row in conn.execute(
                        "SELECT DISTINCT attendance_date FROM attendance_records WHERE operator_id = ? OR employee_name LIKE ?",
                        (op_id, f"%{op_name}%")
                    ).fetchall() if row[0]
                ]
                shift_dates = [
                    row[0] for row in conn.execute(
                        "SELECT DISTINCT shift_date FROM shifts WHERE operator_id = ?",
                        (op_id,)
                    ).fetchall() if row[0]
                ]

            all_dates = sorted(list(set(att_dates + shift_dates)))
            return RouterResult(intent, facts={
                "found": True,
                "operator_id": op_id,
                "operator_name": op_name,
                "working_days": len(all_dates),
                "dates": all_dates,
            })
        except Exception as exc:
            log.warning("Could not resolve GET_OPERATOR_WORKING_DAYS: %s", exc)
            return RouterResult(intent, available=False, reason="Could not calculate working days.")

    if intent == Intent.GET_PRINT_PROGRESS:
        try:
            try:
                from agent.services import machine_state
                state = machine_state.get_machine_state()
                if state and state.get("printing") and state.get("layer_measurements"):
                    return RouterResult(intent, facts={
                        "is_live": True,
                        "status": "printing",
                        "completed_mm": state.get("completed_mm", 0.0),
                        "remaining_mm": state.get("remaining_mm", 0.0),
                        "total_mm": state.get("total_mm", 0.0),
                        "percentage": state.get("percentage", 0.0),
                        "current_layer": state.get("current_layer", 0),
                    })
            except Exception:
                pass

            with db.connect() as conn:
                job_row = conn.execute(
                    "SELECT * FROM print_jobs ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                prod_row = conn.execute(
                    "SELECT * FROM production_records ORDER BY created_at DESC LIMIT 1"
                ).fetchone()

            if job_row:
                j = dict(job_row)
                ts = j.get("created_at")
                ts_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %I:%M %p") if ts else "unknown date"
                return RouterResult(intent, facts={
                    "is_live": False,
                    "found": True,
                    "job_name": j.get("print_file", "wall_print.gcode"),
                    "total_layers": j.get("total_layers", 60),
                    "layers_completed": j.get("layers_completed", 0),
                    "progress_percent": j.get("progress", 0.0),
                    "timestamp_str": ts_str,
                })
            elif prod_row:
                p = dict(prod_row)
                ts = p.get("created_at")
                ts_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %I:%M %p") if ts else "unknown date"
                return RouterResult(intent, facts={
                    "is_live": False,
                    "found": True,
                    "job_name": p.get("print_file", "wall_print.gcode"),
                    "wall_number": p.get("wall_number", 1),
                    "layers_completed": p.get("layers_completed", 0),
                    "segment_layers": p.get("segment_layers", 50),
                    "timestamp_str": ts_str,
                })
            else:
                return RouterResult(intent, available=False, reason="Live telemetry is not connected and no historical print records were found.")
        except Exception as exc:
            log.warning("Could not resolve GET_PRINT_PROGRESS: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query print progress.")

    if intent == Intent.GET_ATTENDANCE_STATUS:
        try:
            op_id, op_name = extract_operator_tuple_from_text(question)
            target_name = op_name
            if not target_name:
                cand = extract_operator_name(question)
                if cand and cand.lower() not in ("who", "anyone", "someone", "anybody", "what", "is", "kya", "shift", "attendance"):
                    target_name = cand
            today = shift_store.local_today()
            records = shift_store.list_attendance_records(date_filter=today)
            active_records = [r for r in records if r["status"] == "ACTIVE"]
            return RouterResult(
                intent,
                facts={
                    "active_records": active_records,
                    "date": today,
                    "target_operator_id": op_id,
                    "target_employee_name": target_name,
                    "all_records": records,
                }
            )
        except Exception as exc:
            log.warning("Could not resolve GET_ATTENDANCE_STATUS: %s", exc)
            return RouterResult(intent, available=False, reason="Could not fetch attendance status.")

    if intent == Intent.GET_ATTENDANCE_OVERTIME:
        try:
            op_id, op_name = extract_operator_tuple_from_text(question)
            today = shift_store.local_today()

            # Check attendance records first
            records = shift_store.list_attendance_records(date_filter=today)
            matching_records = []
            if op_id or op_name:
                matching_records = [
                    r for r in records
                    if (op_id and r.get("operator_id") == op_id) or
                       (op_name and (r.get("employee_name") or "").lower() == op_name.lower())
                ]
            elif records:
                matching_records = records

            if matching_records:
                enriched = [shift_store.enrich_attendance_record(r) for r in matching_records]
                return RouterResult(intent, facts={
                    "type": "attendance",
                    "records": enriched,
                    "operator_id": op_id,
                    "operator_name": op_name,
                    "date": today,
                })

            # Check printing jobs overtime if no attendance records
            all_jobs = db.get_print_jobs_overtime() if not op_id else db.get_print_jobs_overtime(operator_id=op_id)
            return RouterResult(intent, facts={
                "type": "print_jobs",
                "overtime_jobs": all_jobs,
                "operator_id": op_id,
                "operator_name": op_name,
                "date": today,
            })
        except Exception as exc:
            log.warning("Could not resolve GET_ATTENDANCE_OVERTIME: %s", exc)
            return RouterResult(intent, available=False, reason="Could not fetch overtime data.")

    if intent == Intent.GET_SHIFT_BY_DATE:
        date_str = extract_date_from_text(question) or time.strftime("%Y-%m-%d")
        try:
            shifts = shift_store.list_shifts(shift_date=date_str)
            operator_names = []
            for s in shifts:
                op = shift_store.get_operator(s["operator_id"])
                name = op["name"] if op else s["operator_id"]
                if name not in operator_names:
                    operator_names.append(name)
            return RouterResult(intent, facts={
                "date": date_str,
                "count": len(shifts),
                "shifts": shifts,
                "operator_names": operator_names,
            })
        except Exception as exc:
            log.warning("Could not resolve GET_SHIFT_BY_DATE: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query shifts by date.")

    if intent == Intent.GET_SHIFT_BY_OPERATOR:
        try:
            q_norm = question.lower()
            self_terms = ["my", "i", "meri", "maine", "mera", "mujhe"]
            is_self_query = any(re.search(rf"\b{re.escape(w)}\b", q_norm) for w in self_terms)

            if is_self_query:
                op_id = context.get("operator_id") if context else None
                if not op_id:
                    line_id = context.get("line_id", "line_1") if context else "line_1"
                    active = shift_store.get_active_shift_for_line(line_id)
                    if active:
                        op_id = active.get("operator_id")
                if not op_id:
                    return RouterResult(
                        intent,
                        facts={"needs_clarification": True, "clarification_type": "operator"},
                        reason="Could not determine your active operator profile. Which operator are you asking about?",
                        available=False,
                    )
            else:
                op_id = extract_operator_from_text(question)
                if not op_id and context:
                    op_id = context.get("operator_id")

            if not op_id:
                return RouterResult(
                    intent,
                    facts={"needs_clarification": True, "clarification_type": "operator"},
                    reason="Please specify which operator you want to check.",
                    available=False,
                )

            op_obj = shift_store.get_operator(op_id)
            op_name = op_obj.get("name") if op_obj else op_id

            # Determine whether this is a timing/duration query vs a wall production query
            is_timing_query = any(w in q_norm for w in [
                "when did", "timing", "start time", "end time", "shift start", "shift end",
                "kab shuru", "kab khatam", "shuru hui", "khatam hui"
            ]) or (any(w in q_norm for w in ["start", "end", "shuru", "khatam"]) and "shift" in q_norm)

            is_duration_query = any(w in q_norm for w in [
                "hour", "hours", "ghante", "ghanta", "duration", "kitne ghante"
            ]) and not any(w in q_norm for w in ["wall", "walls", "kitni wall", "kitni walls", "piece", "pieces"])

            date_str = extract_date_from_text(question)

            if is_duration_query or is_timing_query:
                shifts = shift_store.list_shifts(operator_id=op_id, shift_date=date_str, limit=50) if date_str else shift_store.list_shifts(operator_id=op_id, limit=50)
                if not shifts:
                    # Check attendance_records as fallback
                    with db.connect() as conn:
                        q_att = "SELECT * FROM attendance_records WHERE (operator_id = ? OR employee_name LIKE ?)"
                        p_att = [op_id, f"%{op_name}%"]
                        if date_str:
                            q_att += " AND attendance_date = ?"
                            p_att.append(date_str)
                        q_att += " ORDER BY shift_start DESC LIMIT 1"
                        att_row = conn.execute(q_att, p_att).fetchone()

                    if att_row:
                        ar = dict(att_row)
                        st = ar.get("shift_start")
                        et = ar.get("shift_end")
                        is_active = ar.get("status") == "ACTIVE"
                        st_formatted = datetime.datetime.fromtimestamp(st).strftime("%I:%M %p").lstrip("0") if st else ""
                        et_formatted = datetime.datetime.fromtimestamp(et).strftime("%I:%M %p").lstrip("0") if et else ("ongoing" if is_active else "")
                        dur_sec = shift_store.attendance_worked_seconds(ar)
                        hours = int(dur_sec // 3600)
                        minutes = int((dur_sec % 3600) // 60)
                        dur_formatted = f"{hours} hours {minutes} minutes" if hours > 0 else f"{minutes} minutes"
                        return RouterResult(
                            intent,
                            facts={
                                "found": True,
                                "operator_id": op_id,
                                "operator_name": op_name,
                                "is_duration_query": is_duration_query,
                                "is_timing_query": is_timing_query,
                                "is_self_query": is_self_query,
                                "shift_id": ar["id"],
                                "shift_type": "Attendance",
                                "shift_date": ar.get("attendance_date", ""),
                                "line_id": "line_1",
                                "status": ar.get("status", "COMPLETED").lower(),
                                "start_time": st,
                                "end_time": et,
                                "start_time_formatted": st_formatted,
                                "end_time_formatted": et_formatted,
                                "duration_seconds": dur_sec,
                                "duration_hours": round(dur_sec / 3600.0, 2),
                                "duration_formatted": dur_formatted if not is_active else "ongoing",
                                "duration_available": not is_active,
                            },
                            available=True,
                        )
                    return RouterResult(
                        intent,
                        facts={
                            "found": False,
                            "operator_id": op_id,
                            "operator_name": op_name,
                            "is_duration_query": is_duration_query,
                            "is_timing_query": is_timing_query,
                            "date": date_str,
                        },
                        reason=f"No shift records found for {op_name}.",
                        available=True,
                    )

                target_shift = shifts[0]
                st = target_shift.get("start_time")
                et = target_shift.get("end_time")
                is_active = target_shift.get("status") == "active"
                st_formatted = datetime.datetime.fromtimestamp(st).strftime("%I:%M %p").lstrip("0") if st else ""
                et_formatted = datetime.datetime.fromtimestamp(et).strftime("%I:%M %p").lstrip("0") if et else ""

                dur_sec = (et - st) if (st is not None and et is not None) else None
                dur_formatted = ""
                if dur_sec is not None and dur_sec >= 0:
                    hours = int(dur_sec // 3600)
                    minutes = int((dur_sec % 3600) // 60)
                    dur_formatted = f"{hours} hours {minutes} minutes" if hours > 0 else f"{minutes} minutes"

                # For ACTIVE shifts: do NOT calculate final duration with live time.
                if is_active or et is None:
                    return RouterResult(
                        intent,
                        facts={
                            "found": True,
                            "operator_id": op_id,
                            "operator_name": op_name,
                            "is_duration_query": is_duration_query,
                            "is_timing_query": is_timing_query,
                            "is_self_query": is_self_query,
                            "shift_id": target_shift["shift_id"],
                            "shift_type": target_shift["shift_type"],
                            "shift_date": target_shift["shift_date"],
                            "line_id": target_shift.get("line_id", "line_1"),
                            "status": "active",
                            "start_time": st,
                            "end_time": None,
                            "start_time_formatted": st_formatted,
                            "end_time_formatted": "ongoing",
                            "duration_seconds": None,
                            "duration_hours": None,
                            "duration_formatted": "ongoing",
                            "duration_available": False,
                        },
                        available=True,
                    )

                # For CLOSED shifts: duration = end_time - start_time
                return RouterResult(
                    intent,
                    facts={
                        "found": True,
                        "operator_id": op_id,
                        "operator_name": op_name,
                        "is_duration_query": is_duration_query,
                        "is_timing_query": is_timing_query,
                        "is_self_query": is_self_query,
                        "shift_id": target_shift["shift_id"],
                        "shift_type": target_shift["shift_type"],
                        "shift_date": target_shift["shift_date"],
                        "line_id": target_shift.get("line_id", "line_1"),
                        "status": target_shift.get("status", "closed"),
                        "start_time": st,
                        "end_time": et,
                        "start_time_formatted": st_formatted,
                        "end_time_formatted": et_formatted,
                        "duration_seconds": dur_sec,
                        "duration_hours": round(dur_sec / 3600.0, 2) if dur_sec is not None else None,
                        "duration_formatted": dur_formatted,
                        "duration_available": True,
                    },
                    available=True,
                )

            records = shift_store.get_operator_production(operator_id=op_id, shift_date=date_str)
            completed_walls = [r for r in records if r.get("status") == "completed"]
            completed_wall_nums = sorted(list({r["wall_number"] for r in completed_walls}))

            return RouterResult(
                intent,
                facts={
                    "found": True,
                    "operator_id": op_id,
                    "operator_name": op_name,
                    "completed_count": len(completed_wall_nums),
                    "completed_walls": completed_wall_nums,
                    "records_count": len(records),
                    "records": records,
                    "source": "shift_store",
                    "is_duration_query": False,
                    "is_production_query": True,
                    "wall_count": len(completed_wall_nums),
                    "walls": completed_wall_nums,
                    "date": date_str,
                },
                available=True,
            )
        except Exception as exc:
            log.warning("Could not resolve GET_SHIFT_BY_OPERATOR: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query operator records.")

    if intent == Intent.GET_WALL_HISTORY:
        wall_num = extract_wall_number(question)
        if wall_num is None:
            return RouterResult(intent, available=False, reason="Please specify which wall number you want to check.")
        try:
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM production_records WHERE wall_number = ? ORDER BY start_time ASC",
                    (wall_num,),
                ).fetchall()
            records = [dict(r) for r in rows]
            if not records:
                return RouterResult(intent, facts={"wall_number": wall_num, "found": False, "message": f"No production records found for Wall {wall_num}."})
            latest = records[-1]
            return RouterResult(intent, facts={
                "wall_number": wall_num,
                "found": True,
                "status": latest["status"],
                "layers_completed": latest["layers_completed"],
                "segment_layers": latest["segment_layers"],
                "wall_height": latest["wall_height"],
                "records_count": len(records),
            })
        except Exception as exc:
            log.warning("Could not resolve GET_WALL_HISTORY: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query wall history.")

    if intent == Intent.GET_SHIFT_ISSUES:
        try:
            # 1. Resolve operator from question text or fallback to context / active shift on context line
            op_id = extract_operator_from_text(question)
            shift_id = context.get("shift_id") if context else None
            if not op_id and context:
                op_id = context.get("operator_id")
                if not op_id and context.get("line_id"):
                    active = shift_store.get_active_shift_for_line(context["line_id"])
                    if active:
                        op_id = active.get("operator_id")
                        if not shift_id:
                            shift_id = active.get("shift_id")
            if not shift_id and context and context.get("line_id"):
                active = shift_store.get_active_shift_for_line(context["line_id"])
                if active and (not op_id or active.get("operator_id") == op_id):
                    shift_id = active.get("shift_id")

            focus = "general"
            q_low = question.lower()
            if any(w in q_low for w in ["recommend", "recommendation", "kelvin", "suggest"]):
                focus = "recommendation"
            elif any(w in q_low for w in ["verified", "rag verified"]):
                focus = "rag_verified"
            elif any(w in q_low for w in ["source", "rag source"]):
                focus = "rag_source"
            elif any(w in q_low for w in ["action", "step", "kya kiya", "kya step", "kya action", "liya"]):
                focus = "action"
            elif any(w in q_low for w in ["cause", "reason", "wajah", "kyun", "why"]):
                focus = "cause"

            # 2. Resolve explicit issue ID from question text or fallback to context
            explicit_id = extract_issue_id(question)
            used_last_issue_id = False
            if explicit_id is None and context and context.get("last_issue_id"):
                explicit_id = context.get("last_issue_id")
                used_last_issue_id = True

            # 3. Explicit Issue #ID handling with strict operator isolation
            if explicit_id is not None:
                target_issue = shift_store.get_issue(explicit_id)
                if not target_issue:
                    return RouterResult(
                        intent,
                        facts={"found": False, "issue_id": explicit_id},
                        reason=f"Issue #{explicit_id} not found.",
                        available=False,
                    )

                # Parent shift and owner lookup
                shift = shift_store.get_shift(target_issue["shift_id"]) if target_issue.get("shift_id") else None
                issue_op_id = shift.get("operator_id") if shift else None

                # Operator verification:
                # If last_issue_id is used from context, verify parent shift operator matches context operator_id
                if used_last_issue_id:
                    context_op_id = context.get("operator_id") if context else None
                    if not context_op_id or issue_op_id != context_op_id:
                        op_obj = shift_store.get_operator(context_op_id) if context_op_id else None
                        op_name = op_obj.get("name") if op_obj else context_op_id
                        return RouterResult(
                            intent,
                            facts={
                                "found": False,
                                "issue_id": explicit_id,
                                "operator_id": context_op_id,
                                "isolation_violation": True,
                            },
                            reason=f"Issue #{explicit_id} does not belong to operator {op_name}." if op_name else f"Issue #{explicit_id} does not match the active operator.",
                            available=False,
                        )

                # Explicit operator in query or context must match issue owner
                if op_id and issue_op_id != op_id:
                    op_obj = shift_store.get_operator(op_id)
                    op_name = op_obj.get("name") if op_obj else op_id
                    return RouterResult(
                        intent,
                        facts={
                            "found": False,
                            "issue_id": explicit_id,
                            "operator_id": op_id,
                            "isolation_violation": True,
                        },
                        reason=f"Issue #{explicit_id} does not belong to operator {op_name}.",
                        available=False,
                    )

                return RouterResult(
                    intent,
                    facts={
                        "found": True,
                        "issue_id": explicit_id,
                        "operator_id": issue_op_id or op_id,
                        "focus": focus,
                        "cause": target_issue.get("cause"),
                        "action_taken": target_issue.get("action_taken"),
                        "rag_recommendation": target_issue.get("rag_recommendation"),
                        "rag_verified": bool(target_issue.get("rag_verified")),
                        "rag_source": target_issue.get("rag_source"),
                        "description": target_issue.get("description"),
                        "issue_type": target_issue.get("issue_type"),
                        "result": target_issue.get("result"),
                        "severity": target_issue.get("severity"),
                    },
                    available=True,
                )

            # 4. No explicit issue ID -> require operator (do not list all operators' issues)
            if not op_id:
                return RouterResult(
                    intent,
                    facts={
                        "needs_clarification": True,
                        "clarification_type": "operator",
                    },
                    reason="Which operator's shift issues would you like to check?",
                    available=False,
                )

            date_str = extract_date_from_text(question)
            if date_str:
                shift_id = None

            if shift_id:
                issues = shift_store.list_shift_issues(shift_id=shift_id, operator_id=op_id, shift_date=date_str, limit=20)
            else:
                issues = shift_store.list_shift_issues(operator_id=op_id, shift_date=date_str, limit=20)
            if not issues:
                op_obj = shift_store.get_operator(op_id) if op_id else None
                op_name = op_obj.get("name") if op_obj else op_id
                op_str = f" for operator {op_name}" if op_name else ""
                date_str_part = f" on {date_str}" if date_str else ""
                return RouterResult(
                    intent,
                    facts={"count": 0, "operator_id": op_id, "date": date_str, "issues": []},
                    reason=f"No issues have been recorded{op_str}{date_str_part}.",
                    available=True,
                )

            # Focus == "action": return current operator's action from current or most recent issue
            if focus == "action":
                target = None
                if context and context.get("last_issue_id"):
                    target = next((iss for iss in issues if iss["issue_id"] == context["last_issue_id"]), None)
                if not target and issues:
                    target = issues[0]
                if target:
                    return RouterResult(
                        intent,
                        facts={
                            "found": True,
                            "issue_id": target.get("issue_id"),
                            "operator_id": op_id,
                            "focus": focus,
                            "cause": target.get("cause"),
                            "action_taken": target.get("action_taken"),
                            "rag_recommendation": target.get("rag_recommendation"),
                            "rag_verified": bool(target.get("rag_verified")),
                            "rag_source": target.get("rag_source"),
                            "description": target.get("description"),
                            "issue_type": target.get("issue_type"),
                            "result": target.get("result"),
                            "severity": target.get("severity"),
                        },
                        available=True,
                    )

            # Multi-issue clarification check for detail queries
            if focus in ("cause", "recommendation", "rag_verified", "rag_source") and len(issues) > 1:
                cand_ids = [iss["issue_id"] for iss in issues]
                op_obj = shift_store.get_operator(op_id) if op_id else None
                op_name = op_obj.get("name") if op_obj else op_id
                op_str = f" for {op_name}" if op_name else ""
                cand_list = ", ".join(f"Issue #{i}" for i in cand_ids)
                return RouterResult(
                    intent,
                    facts={
                        "needs_clarification": True,
                        "candidate_ids": cand_ids,
                        "operator_id": op_id,
                        "focus": focus,
                        "count": len(issues),
                    },
                    reason=f"Multiple issues found{op_str} ({cand_list}). Which issue number would you like to check?",
                    available=False,
                )

            # Single issue detail query
            if focus in ("cause", "recommendation", "rag_verified", "rag_source") and len(issues) == 1:
                single = issues[0]
                return RouterResult(
                    intent,
                    facts={
                        "found": True,
                        "issue_id": single.get("issue_id"),
                        "operator_id": op_id,
                        "focus": focus,
                        "cause": single.get("cause"),
                        "action_taken": single.get("action_taken"),
                        "rag_recommendation": single.get("rag_recommendation"),
                        "rag_verified": bool(single.get("rag_verified")),
                        "rag_source": single.get("rag_source"),
                        "description": single.get("description"),
                        "issue_type": single.get("issue_type"),
                        "result": single.get("result"),
                        "severity": single.get("severity"),
                    },
                    available=True,
                )

            # General focus: list all formatted issues for this operator
            formatted_issues = []
            for iss in issues:
                formatted_issues.append({
                    "issue_id": iss.get("issue_id"),
                    "type": iss.get("issue_type"),
                    "desc": iss.get("description"),
                    "severity": iss.get("severity"),
                    "cause": iss.get("cause"),
                    "action_taken": iss.get("action_taken"),
                    "rag_recommendation": iss.get("rag_recommendation"),
                    "rag_verified": bool(iss.get("rag_verified")),
                    "rag_source": iss.get("rag_source"),
                    "result": iss.get("result"),
                    "shift_date": iss.get("shift_date"),
                    "shift_id": iss.get("shift_id"),
                    "operator_id": iss.get("operator_id"),
                })
            return RouterResult(
                intent,
                facts={
                    "count": len(formatted_issues),
                    "operator_id": op_id,
                    "date": date_str,
                    "focus": focus,
                    "issues": formatted_issues,
                },
                available=True,
            )
        except Exception as exc:
            log.warning("Could not resolve GET_SHIFT_ISSUES: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query shift issues.")

    if intent == Intent.GET_HANDOVER_STATUS:
        try:
            h = shift_store.get_pending_handover_for_line("line_1")
            if not h:
                with db.connect() as conn:
                    row = conn.execute(
                        "SELECT * FROM handovers WHERE line_id = ? ORDER BY handover_time DESC LIMIT 1",
                        ("line_1",),
                    ).fetchone()
                h = dict(row) if row else None
            if not h:
                return RouterResult(intent, facts={"found": False, "message": "No handover record was found for Line 1."})
            return RouterResult(intent, facts={
                "found": True,
                "handover_id": h["handover_id"],
                "completed_work": h.get("completed_work") or "None",
                "pending_work": h.get("pending_work") or "None",
                "open_issues": h.get("open_issues") or "None",
                "to_shift_id": h.get("to_shift_id"),
            })
        except Exception as exc:
            log.warning("Could not resolve GET_HANDOVER_STATUS: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query handover status.")

    # --- Phase 2C Action Intents (Prepared for confirmation) ---
    if intent == Intent.SHIFT_START:
        st = extract_shift_type(question)
        tw = extract_wall_number(question) or 0
        name = extract_operator_name(question)
        return RouterResult(intent, facts={
            "action": "SHIFT_START",
            "shift_type": st,
            "target_walls": tw,
            "line_id": "line_1",
            # The name the operator said, if any. The assistant resolves it
            # against the operators table; the router never touches the
            # database for identity and never invents a name.
            "spoken_name": name,
            "requires_confirmation": not bool(name),
        })

    if intent == Intent.SHIFT_END:
        name = extract_operator_name(question)
        return RouterResult(intent, facts={
            "action": "SHIFT_END",
            "status": "closed",
            "line_id": "line_1",
            "spoken_name": name,
            "requires_confirmation": not bool(name),
        })

    if intent == Intent.LOG_WALL_PRODUCTION:
        wn = extract_wall_number(question)
        return RouterResult(intent, facts={
            "action": "LOG_WALL_PRODUCTION",
            "wall_number": wn,
            "status": "completed",
            "line_id": "line_1",
            "requires_confirmation": wn is not None,
        })

    if intent == Intent.LOG_SHIFT_ISSUE:
        wn = extract_wall_number(question)
        dt = extract_downtime(question)
        rag_rec = None
        rag_src = None
        rag_ver = 0
        search_q = re.sub(r"\bmaterial\s+slope\b", "material flow", question, flags=re.IGNORECASE)
        try:
            k = search_documentation(search_q)
            if k.get("found") and k.get("results"):
                spoken = speak_knowledge(k, search_q)
                top = k["results"][0]
                rag_rec = spoken or top.get("text", "")[:320]
                rag_src = top.get("source")
                rag_ver = _evaluate_rag_verified(top.get("text", ""))
        except Exception as exc:
            log.warning("Could not retrieve RAG guidance for LOG_SHIFT_ISSUE: %s", exc)

        issue_type = "operational_issue"
        q_low = question.lower()
        if "material flow" in q_low or "material slope" in q_low:
            issue_type = "material_flow_issue"
        elif "nozzle" in q_low:
            issue_type = "nozzle_clog"
        elif "pump" in q_low:
            issue_type = "pump_pressure"
        elif "feeder" in q_low:
            issue_type = "feeder_jam"
        elif "sensor" in q_low:
            issue_type = "sensor_fault"

        return RouterResult(intent, facts={
            "action": "LOG_SHIFT_ISSUE",
            "issue_type": issue_type,
            "description": question,
            "wall_number": wn,
            "downtime_minutes": dt,
            "severity": "warning",
            "line_id": "line_1",
            "rag_recommendation": rag_rec,
            "rag_source": rag_src,
            "rag_verified": rag_ver,
            "requires_confirmation": True,
        })

    if intent == Intent.LOG_SHIFT_HANDOVER:
        return RouterResult(intent, facts={
            "action": "LOG_SHIFT_HANDOVER",
            "line_id": "line_1",
            "requires_confirmation": True,
        })

    if intent == Intent.RESOLVE_SHIFT_ISSUE:
        t_id = extract_issue_id(question)
        action_taken, rag_recommended = extract_operator_action(question)
        cause = extract_resolution_cause(question)
        result = extract_resolution_result(question)
        domain_keywords = extract_domain_keywords(question)
        return RouterResult(intent, facts={
            "action": "RESOLVE_SHIFT_ISSUE",
            "target_issue_id": t_id,
            "action_taken": action_taken,
            "cause": cause,
            "result": result,
            "domain_keywords": domain_keywords,
            "rag_recommended": rag_recommended,
            "line_id": "line_1",
            "question": question,
            "requires_confirmation": True,
        })

    # --- Employee Wall Production Tracking Resolvers ---
    if intent == Intent.RECORD_WALLS_COMPLETED:
        try:
            op_id, op_name = extract_operator_tuple_from_text(question)
            if not op_id:
                spoken = extract_operator_name(question)
                if spoken and spoken.lower() not in ("i", "my", "me"):
                    op = shift_store.find_operator_by_name(spoken, active_only=False)
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]
            if not op_id and context:
                ctx_op = context.get("operator_id")
                if ctx_op:
                    op = shift_store.get_operator(ctx_op)
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]
            if not op_id:
                active = shift_store.get_active_shift_for_line("line_1")
                if active and active.get("operator_id"):
                    op = shift_store.get_operator(active["operator_id"])
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]

            if not op_id:
                return RouterResult(
                    intent,
                    facts={"recorded": False},
                    reason="Could not determine operator for wall production recording. Please specify employee name or operator code.",
                    available=False,
                )

            qty = _extract_wall_quantity(question)
            if qty is None or qty <= 0:
                return RouterResult(
                    intent,
                    facts={"recorded": False, "operator_id": op_id, "operator_name": op_name, "quantity": qty},
                    reason="Invalid quantity: wall count must be a positive integer greater than zero.",
                    available=False,
                )

            job_id = None
            shift_id = None
            if context:
                job_id = context.get("job_id")
                shift_id = context.get("shift_id")
            if not shift_id:
                try:
                    active_s = shift_store.get_active_shift_for_line("line_1")
                    if active_s:
                        shift_id = active_s.get("shift_id")
                except Exception:
                    shift_id = None
            if not job_id:
                try:
                    active_j = db.get_active_job()
                    if active_j:
                        job_id = active_j.get("id")
                except Exception:
                    job_id = None

            rec = shift_store.record_wall_production(
                operator_id=op_id,
                quantity=qty,
                shift_id=shift_id,
                job_id=job_id,
                notes=f"Recorded via operational assistant: {question[:120]}",
                recorded_by="operational_assistant",
            )

            rec_ts = rec.get("timestamp") or rec.get("created_at") or time.time()
            ts_str = datetime.datetime.fromtimestamp(rec_ts).strftime("%Y-%m-%d %I:%M %p")
            return RouterResult(
                intent,
                facts={
                    "recorded": True,
                    "record_id": rec["record_id"],
                    "operator_id": op_id,
                    "operator_name": op_name,
                    "quantity": qty,
                    "shift_id": rec.get("shift_id") or "unassigned",
                    "job_id": rec.get("job_id") or "unassigned",
                    "timestamp": rec["timestamp"],
                    "timestamp_str": ts_str,
                },
                available=True,
            )
        except ValueError as val_err:
            return RouterResult(intent, available=False, reason=str(val_err))
        except Exception as exc:
            log.warning("Could not resolve RECORD_WALLS_COMPLETED: %s", exc)
            return RouterResult(intent, available=False, reason="Could not record wall production.")

    if intent == Intent.GET_WALL_PRODUCTION:
        try:
            op_id, op_name = extract_operator_tuple_from_text(question)
            if not op_id:
                spoken = extract_operator_name(question)
                if spoken and spoken.lower() not in ("who", "anyone", "someone", "i", "my", "me", "what", "how", "total"):
                    op = shift_store.find_operator_by_name(spoken, active_only=False)
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]

            scope, filter_val = _extract_wall_production_scope(question)
            job_id = filter_val if scope == "job" else None
            date_from = filter_val if scope == "date" else None
            date_to = filter_val if scope == "date" else None

            totals = shift_store.get_wall_production_total(
                operator_id=op_id,
                scope=scope if scope != "date" else "date_range",
                job_id=job_id,
                date_from=date_from,
                date_to=date_to,
            )

            return RouterResult(
                intent,
                facts={
                    "found": True,
                    "operator_id": op_id,
                    "operator_name": op_name,
                    "scope": scope,
                    "job_id": job_id,
                    "date": filter_val if scope == "date" else None,
                    "total_quantity": totals["total_quantity"],
                    "record_count": totals["record_count"],
                    "by_operator": totals.get("by_operator", []),
                },
                available=True,
            )
        except Exception as exc:
            log.warning("Could not resolve GET_WALL_PRODUCTION: %s", exc)
            return RouterResult(intent, available=False, reason="Could not query wall production counts.")

    if intent == Intent.CORRECT_WALL_PRODUCTION:
        try:
            new_qty, old_qty = _extract_correction_quantities(question)
            if new_qty is None or new_qty <= 0:
                return RouterResult(
                    intent,
                    available=False,
                    reason="Invalid quantity: corrected wall count must be a positive integer greater than zero.",
                )

            op_id, op_name = extract_operator_tuple_from_text(question)
            if not op_id and context:
                ctx_op = context.get("operator_id")
                if ctx_op:
                    op = shift_store.get_operator(ctx_op)
                    if op:
                        op_id = op["operator_id"]
                        op_name = op["name"]

            records = shift_store.list_wall_production_records(
                operator_id=op_id,
                include_corrected=False,
                limit=10,
            )
            target_record = None
            if old_qty is not None and records:
                target_record = next((r for r in records if r["quantity"] == old_qty), None)
            if not target_record and records:
                target_record = records[0]

            if not target_record:
                op_suffix = f" for {op_name}" if op_name else ""
                return RouterResult(
                    intent,
                    available=False,
                    reason=f"No recent uncorrected wall production record found{op_suffix} to correct.",
                )

            correction_result = shift_store.correct_wall_production(
                original_record_id=target_record["record_id"],
                corrected_quantity=new_qty,
                reason=f"Corrected via operational assistant: {question[:120]}",
                corrected_by="operational_assistant",
            )

            orig = correction_result["original_record"]
            repl = correction_result["replacement_record"]
            return RouterResult(
                intent,
                facts={
                    "corrected": True,
                    "original_record_id": orig["record_id"],
                    "replacement_record_id": repl["record_id"],
                    "old_quantity": orig["quantity"],
                    "new_quantity": repl["quantity"],
                    "operator_id": repl["operator_id"],
                    "operator_name": repl["operator_name"],
                    "audit_trail_created": True,
                },
                available=True,
            )
        except ValueError as val_err:
            return RouterResult(intent, available=False, reason=str(val_err))
        except Exception as exc:
            log.warning("Could not resolve CORRECT_WALL_PRODUCTION: %s", exc)
            return RouterResult(intent, available=False, reason="Could not correct wall production record.")

    return RouterResult(intent, available=False, reason="")


def route(question: str, context: Optional[Dict[str, Any]] = None) -> RouterResult:
    """
    Classify + resolve in one call.

    COMBINED_QUERY resolves the live measurement AND the documentation, so the
    answer can state the real value and explain the technical background.
    """
    intent, terms = classify(question)

    if intent == Intent.COMBINED_QUERY:
        # Re-classify ignoring the knowledge signal to find the live part.
        live_intent = Intent.UNKNOWN_QUERY
        best = 0.0
        q = _normalise(question)
        for candidate, phrases, priority in _PATTERNS:
            score, _hits = _score(q, phrases)
            if score <= 0:
                continue
            weighted = priority * 1000 + score
            if weighted > best:
                live_intent, best = candidate, weighted

        live = resolve(live_intent, question, context=context) if live_intent != Intent.UNKNOWN_QUERY else None
        knowledge = search_documentation(question)
        return RouterResult(
            intent=Intent.COMBINED_QUERY,
            facts={
                "live_intent": live_intent.value,
                "live_available": bool(live and live.available),
                **(live.facts if live and live.available else {}),
                **({"live_reason": live.reason} if live and not live.available else {}),
            },
            knowledge=knowledge,
            available=bool((live and live.available) or knowledge.get("found")),
            reason="" if (live and live.available) or knowledge.get("found") else UNAVAILABLE_MESSAGE,
            matched_terms=terms,
        )

    result = resolve(intent, question, context=context)
    result.matched_terms = terms
    result.question = question

    # Full trace of one question. Every stage is logged so a wrong answer can
    # be traced to the stage that produced it rather than guessed at.
    log.info("TRANSCRIPT : %s", question)
    log.info("NORMALIZED : %s", _normalise(question))
    log.info("INTENT     : %s  (matched: %s)", intent.value, ", ".join(terms) or "-")
    if result.available:
        log.info("DATA       : %s", result.facts)
    else:
        log.info("DATA       : UNAVAILABLE -- %s", result.reason or "no reason given")
    log.info("RESPONSE   : %s", summarise(result))
    return result


# Human-readable summary used when no LLM is configured.
def summarise(result: RouterResult) -> str:
    """Deterministic sentence built ONLY from resolved facts."""
    if result.intent == Intent.UNKNOWN_QUERY and not result.facts:
        return (
            "I didn't understand that. You can ask about production, layers, material, "
            "timing, machine or camera status, problems, or the documentation."
        )

    if not result.available:
        return result.reason or UNAVAILABLE_MESSAGE

    f = result.facts
    i = result.intent

    # If the operator asked "how many walls" but production is length-based,
    # say so rather than letting millimetres masquerade as a wall count.
    asked_count = any(
        word in (result.question or "").lower()
        for word in ("wall", "walls", "kitni wall", "how many", "kitne", "count", "pieces")
    )
    count_note = ""
    if asked_count and f.get("measurement_basis") == "length_mm":
        count_note = (
            " Wall count is unavailable: this job is tracked by printed length, "
            "not by number of walls."
        )

    if i == Intent.GET_COMPLETED_PRODUCTION:
        return (f"{f['completed_mm']} millimeter printed out of {f['total_mm']} millimeter"
                + (f" ({f['percentage']} percent)." if f.get("percentage") is not None else ".")
                + count_note)
    if i == Intent.GET_REMAINING_PRODUCTION:
        return (f"{f['remaining_mm']} millimeter remaining out of {f['total_mm']} millimeter."
                + count_note)
    if i == Intent.GET_TOTAL_TARGET:
        return f"The total target is {f['total_mm']} millimeter."
    if i == Intent.GET_PRINT_RATE:
        return f"The current print rate is {f['print_rate_mm_per_min']} millimeter per minute."
    if i == Intent.GET_ELAPSED_TIME:
        return f"The print has been running for {f['elapsed']}."
    if i == Intent.GET_ETA:
        return f"Estimated time to finish is {f['eta']}."
    if i == Intent.GET_DEADLINE_STATUS:
        return f"The job is {f['deadline_status']}."
    if i == Intent.GET_CURRENT_LAYER:
        total = f" out of {f['total_layers']}" if f.get("total_layers") else ""
        return f"Current layer is {f['current_layer']}{total}."
    if i == Intent.GET_LAYER_WIDTH:
        note = f" {f['note']}" if f.get("note") else ""
        return f"The current layer width is {f['layer_width_mm']} millimeter.{note}"
    if i == Intent.GET_LAYER_HEIGHT:
        return f"The current layer height is {f['layer_height_mm']} millimeter."
    if i == Intent.GET_WALL_DIMENSIONS:
        parts = []
        if f.get("wall_height_mm") is not None:
            parts.append(f"height {f['wall_height_mm']} millimeter")
        if f.get("wall_width_mm") is not None:
            parts.append(f"width {f['wall_width_mm']} millimeter")
        return "Wall " + " and ".join(parts) + "." if parts else UNAVAILABLE_MESSAGE
    if i == Intent.GET_MATERIAL_USAGE:
        return f"{f['material_used_kg']} kilograms of material have been used."
    if i == Intent.GET_MATERIAL_REMAINING:
        return f"{f['material_remaining_kg']} kilograms of material remain."
    if i == Intent.GET_MATERIAL_RATE:
        return f"Material consumption is {f['consumption_rate_kg_per_hour']} kilograms per hour."
    if i in (Intent.GET_24_HOUR_PRODUCTION, Intent.GET_SHIFT_PRODUCTION, Intent.GET_RECENT_EVENTS):
        if f.get("produced_mm") is not None:
            return f"{f['produced_mm']} millimeter were printed in the {f['window']} window."
        if f.get("events"):
            latest = f["events"][0]
            return f"Most recent event: {latest['message']}"
        return f.get("produced_note") or UNAVAILABLE_MESSAGE
    if i == Intent.WAKE_WORD:
        return "Yes, what would you like to know?"
    if i == Intent.GET_OPERATOR:
        return f"{f['operator']} is the operator on record for this machine."
    if i == Intent.GET_MACHINE_STATUS:
        note = f" {f['note']}" if f.get("note") else ""
        return ("Yes, printing is currently running." if f.get("printing")
                else "No, printing is not currently detected.") + note
    if i == Intent.GET_CAMERA_STATUS:
        return (f"The camera is {'connected' if f.get('camera_connected') else 'not connected'}"
                + (f" at {f['fps']} frames per second." if f.get("fps") else "."))
    if i == Intent.GET_SITE_PEOPLE_COUNT:
        count = f.get("people_count")
        note = f" {f['note']}" if f.get("note") else ""
        if count == 0:
            return f"No people are currently detected on site via Camera 2.{note}"
        elif count == 1:
            return f"There is 1 person currently visible on site via Camera 2.{note}"
        else:
            return f"There are {count} people currently visible on site via Camera 2.{note}"
    if i == Intent.GET_SITE_DETECTIONS:
        if f.get("unsupported_class"):
            u_cls = f["unsupported_class"]
            return f"The loaded vision model does not support detecting '{u_cls}'. Generic objects are not inferred or mapped to industrial equipment without explicit model support."
        target_class = f.get("target_class")
        entities = f.get("entities") or []
        if target_class:
            t_entities = f.get("target_entities")
            if t_entities is None:
                t_entities = [
                    e for e in entities
                    if (e.get("class_name") if isinstance(e, dict) else getattr(e, "class_name", "")).lower() == target_class.lower()
                ]
            if not t_entities:
                return f"No {target_class}{'s' if not target_class.endswith('s') else ''} are currently detected via Camera 2."
            cnt = len(t_entities)
            plural = 's' if cnt > 1 and not target_class.endswith('s') else ''
            return f"Currently detected on Camera 2: {cnt} {target_class}{plural}."

        if not entities:
            return "No objects or persons are currently detected via Camera 2."
        counts: Dict[str, int] = {}
        for ent in entities:
            cname = ent.get("class_name") if isinstance(ent, dict) else getattr(ent, "class_name", "object")
            counts[cname] = counts.get(cname, 0) + 1
        summary_items = [f"{cnt} {name}{'s' if cnt > 1 and not name.endswith('s') else ''}" for name, cnt in counts.items()]
        return f"Currently detected on Camera 2: {', '.join(summary_items)}."
    if i in (Intent.GET_ROBOT_STATUS, Intent.GET_COMMUNICATION_STATUS):
        return (f"Machine status is {f.get('machine_status')}. "
                f"The camera is {'connected' if f.get('camera_connected') else 'not connected'}. "
                f"{f.get('machine_link_note', '')}").strip()
    if i == Intent.GET_PRINT_QUALITY:
        return f"Print quality is {f['quality_percent']} percent, rated {f['quality_status']}."
    if i == Intent.GET_RECENT_PROBLEMS:
        if not f.get("alerts") and not f.get("events"):
            return "No problems have been recorded."
        alerts = f.get("alerts") or []
        if alerts:
            return f"Most recent problem: {alerts[-1]['message']}"
        return f"Most recent event: {f['events'][0]['message']}"
    if i == Intent.KNOWLEDGE_QUERY:
        k = result.knowledge or {}
        if not k.get("found"):
            return k.get("message", "That was not found in the available documentation.")
        # The retrieved chunk is structured Markdown. Speak it, don't recite it:
        # returning `top['text']` put "## Issue:" and "**Problem:**" and the
        # filename into the operator's ear, truncated mid-word.
        spoken = speak_knowledge(k, result.question)
        if spoken:
            return spoken
        top = k["results"][0]
        return f"From {top['source']}: {top['text'][:320]}"
    if i == Intent.COMBINED_QUERY:
        pieces = []
        if f.get("live_intent") == Intent.GET_SITE_DETECTIONS.value:
            if f.get("unsupported_class"):
                pieces.append(f"The loaded vision model does not support detecting '{f['unsupported_class']}'.")
            elif f.get("entities"):
                counts: Dict[str, int] = {}
                for ent in f["entities"]:
                    cname = ent.get("class_name") if isinstance(ent, dict) else getattr(ent, "class_name", "object")
                    counts[cname] = counts.get(cname, 0) + 1
                items = [f"{cnt} {name}" for name, cnt in counts.items()]
                pieces.append(f"Currently detected on Camera 2: {', '.join(items)}.")
        elif f.get("live_available"):
            for key in ("layer_width_mm", "completed_mm", "print_rate_mm_per_min", "quality_percent"):
                if key in f:
                    pieces.append(f"{key.replace('_', ' ')}: {f[key]}")
        k = result.knowledge or {}
        if k.get("found"):
            spoken = speak_knowledge(k, result.question)
            pieces.append(
                spoken
                or f"From {k['results'][0]['source']}: {k['results'][0]['text'][:220]}"
            )
        return " ".join(pieces) if pieces else UNAVAILABLE_MESSAGE

    # --- Phase 2C Historical Query Summaries ---

    if i == Intent.GET_ATTENDANCE_HOURS:
        records = f.get("records", [])
        target_name = f.get("operator_name")
        if not records:
            if target_name:
                return f"No attendance record found for {target_name} today."
            return f"No attendance records found for {f.get('date', 'today')}."

        summary = []
        for r in records:
            name = r.get("employee_name", "Employee")
            worked = shift_store.format_worked_spoken(r.get("worked_seconds", 0))
            ot = r.get("overtime_seconds", 0)
            status_str = "ongoing" if r.get("status") == "ACTIVE" else "completed"
            if ot > 0:
                ot_spoken = shift_store.format_worked_spoken(ot)
                summary.append(f"{name} has worked {worked} ({status_str}), including {ot_spoken} of overtime")
            else:
                summary.append(f"{name} has worked {worked} ({status_str})")

        if len(summary) == 1:
            return f"{summary[0]}."
        return "Today's hours: " + ". ".join(summary) + "."

    if i == Intent.GET_ATTENDANCE_STATUS:
        active = f.get("active_records", [])
        target_name = f.get("target_employee_name")
        target_id = f.get("target_operator_id")

        if target_name or target_id:
            matched_rec = None
            for r in active:
                r_name = r.get("employee_name", "").lower()
                r_id = r.get("operator_id", "").lower()
                if (target_id and r_id == target_id.lower()) or (target_name and (target_name.lower() in r_name or r_name in target_name.lower())):
                    matched_rec = r
                    break

            display_name = (matched_rec.get("employee_name") if matched_rec else target_name).title()
            if matched_rec:
                start_clock = shift_store.format_clock_time(matched_rec.get("shift_start"))
                return f"Yes, {display_name} is currently on shift (started at {start_clock})."
            else:
                all_recs = f.get("all_records", [])
                completed_rec = None
                for r in all_recs:
                    r_name = r.get("employee_name", "").lower()
                    r_id = r.get("operator_id", "").lower()
                    if (target_id and r_id == target_id.lower()) or (target_name and (target_name.lower() in r_name or r_name in target_name.lower())):
                        completed_rec = r
                        break
                if completed_rec and completed_rec.get("status") == "COMPLETED":
                    end_clock = shift_store.format_clock_time(completed_rec.get("shift_end"))
                    return f"No, {display_name} is not currently on shift. Their shift ended at {end_clock}."
                return f"No, {display_name} is not currently on shift."

        if not active:
            return "No employees are currently clocked in. No employees are currently on shift (0)."
        names = [r["employee_name"] for r in active]
        return f"Currently clocked in: {', '.join(names)}."

    if i == Intent.GET_OPERATOR_WORKING_DAYS:
        if f.get("found"):
            count = f.get("working_days", 0)
            unit = "working day" if count == 1 else "working days"
            name = f.get("operator_name", "The employee")
            return f"{name} has completed {count} {unit}."
        return f.get("reason") or "No working records found for that operator."

    if i == Intent.GET_PRINT_PROGRESS:
        if f.get("is_live"):
            pct = f.get("percentage")
            pct_str = f" ({pct:.1f}% complete)" if pct is not None else ""
            return f"Live printing is running at layer {f.get('current_layer', 0)}{pct_str}."
        if f.get("found"):
            pct = f.get("progress_percent")
            pct_str = f" ({pct:.1f}% progress)" if pct is not None else ""
            layers_done = f.get("layers_completed", 0)
            total_layers = f.get("total_layers") or f.get("segment_layers", 0)
            return (
                f"Live telemetry is not connected. The latest stored print record from {f.get('timestamp_str', 'earlier')} "
                f"shows job {f.get('job_name', 'current')}, with {layers_done} layers completed out of {total_layers}{pct_str}."
            )
        return UNAVAILABLE_MESSAGE

    if i == Intent.GET_ATTENDANCE_OVERTIME:
        if f.get("type") == "attendance" or "records" in f:
            records = f.get("records", [])
            op_name = f.get("operator_name")
            op_id = f.get("operator_id")
            target_name = op_name or op_id

            if target_name:
                if records:
                    r = records[0]
                    ot_sec = r.get("overtime_seconds", 0)
                    ot_spoken = r.get("overtime_spoken") or shift_store.format_worked_spoken(ot_sec)
                    return f"{target_name} has {ot_spoken} of overtime."
                return f"No attendance record found for {target_name} today."

            q_low = (result.question or "").lower()
            if any(w in q_low for w in ("most", "highest", "max", "maximum", "sabse zyada")):
                overtime_records = [r for r in records if r.get("overtime_seconds", 0) > 0]
                if not overtime_records:
                    return "No employees have accumulated overtime today."
                top = max(overtime_records, key=lambda r: r.get("overtime_seconds", 0))
                top_spoken = top.get("overtime_spoken") or shift_store.format_worked_spoken(top.get("overtime_seconds", 0))
                return f"{top['employee_name']} has the most overtime today with {top_spoken}."

            # Multiple/all employees
            lines = []
            for r in records:
                name = r.get("employee_name", "Employee")
                ot_sec = r.get("overtime_seconds", 0)
                if ot_sec > 0:
                    ot_spoken = r.get("overtime_spoken") or shift_store.format_worked_spoken(ot_sec)
                    lines.append(f"{name} has {ot_spoken} of overtime")
            if lines:
                return "Today's overtime: " + ", ".join(lines) + "."
            return "No employees have accumulated overtime today."

        overtime_jobs = f.get("overtime_jobs", [])
        op_id = f.get("operator_id")
        op_name = f.get("operator_name")

        if op_id:
            if not overtime_jobs:
                return f"{op_name or op_id} has 0 overtime hours recorded for printing jobs."
            
            total_ot = sum(job.get("overtime_hours", 0) for job in overtime_jobs)
            return f"{op_name or op_id} has accumulated {total_ot:.1f} overtime hours across {len(overtime_jobs)} printing jobs."
            
        if not overtime_jobs:
            return "No printing jobs have accumulated overtime."

        total_ot = sum(job.get("overtime_hours", 0) for job in overtime_jobs)
        return f"A total of {total_ot:.1f} overtime hours have been recorded across {len(overtime_jobs)} printing jobs."


    if i == Intent.GET_ACTIVE_SHIFT:
        line_label = _spoken_line(f.get("line_id", "line_1"))
        if not f.get("active"):
            return f.get("message") or f"No active shift is currently running on {line_label}."
        return f"{f['shift_type']} shift is active on {line_label} with operator {f['operator_name']}."

    if i == Intent.GET_PREVIOUS_SHIFT:
        if not f.get("found"):
            return f.get("message") or "No previous shift found for Line 1."
        line_label = "Line 1" if f.get("line_id") == "line_1" else f.get("line_id", "Line 1")
        base = f"Previous shift on {line_label} was {f['shift_type']} shift on {f['shift_date']} with operator {f['operator_name']}."
        timing_parts = []
        if f.get("start_time_formatted"):
            timing_parts.append(f"started at {f['start_time_formatted']}")
        if f.get("end_time_formatted"):
            timing_parts.append(f"ended at {f['end_time_formatted']}")
        if f.get("duration_formatted"):
            timing_parts.append(f"total duration: {f['duration_formatted']}")
        if timing_parts:
            return f"{base} {', '.join(timing_parts).capitalize()}."
        return base

    if i == Intent.GET_SHIFT_BY_DATE:
        cnt = f.get('count', 0)
        date_val = f.get('date')
        if cnt == 0:
            return f"No shifts were recorded on {date_val}."
        names = f.get("operator_names", [])
        if names:
            names_str = ", ".join(names)
            return f"On {date_val}, the following operator(s) were on shift: {names_str}."
        return f"Found {cnt} shifts on {date_val}."

    if i == Intent.GET_SHIFT_BY_OPERATOR:
        if not f.get("found", True):
            return f.get("message") or f.get("reason") or "Operator not found."
        op_name = f.get("operator_name") or f.get("operator_id") or "Operator"
        is_self = f.get("is_self_query", False)
        subject = "You" if is_self else op_name

        if f.get("is_timing_query"):
            st_str = f.get("start_time_formatted") or "unavailable"
            et_str = f.get("end_time_formatted") or "unavailable"
            status = f.get("status")
            s_date = f.get("shift_date", "their shift")
            q_txt = (result.question or "").lower()
            asks_start = any(w in q_txt for w in ["start", "shuru"]) and not any(w in q_txt for w in ["end", "khatam"])
            asks_end = any(w in q_txt for w in ["end", "khatam"]) and not any(w in q_txt for w in ["start", "shuru"])

            if asks_start:
                if is_self:
                    return f"{op_name}, your shift started at {st_str} on {s_date}."
                return f"{op_name}'s shift started at {st_str} on {s_date}."
            if asks_end:
                if status == "active" or not et_str or et_str == "ongoing":
                    if is_self:
                        return f"{op_name}, your shift is currently active and has not ended yet."
                    return f"{op_name}'s shift is currently active and has not ended yet."
                if is_self:
                    return f"{op_name}, your shift ended at {et_str} on {s_date}."
                return f"{op_name}'s shift ended at {et_str} on {s_date}."

            if status == "active":
                if is_self:
                    return f"{op_name}, your shift started at {st_str} on {s_date} and is currently active."
                return f"{op_name} started shift at {st_str} on {s_date} and is currently active."
            timing = f" from {st_str} to {et_str}" if (st_str and et_str) else ""
            if is_self:
                return f"{op_name}, your shift on {s_date} was{timing}."
            return f"{op_name}'s shift on {s_date} was{timing}."

        if f.get("is_duration_query"):
            st_str = f.get("start_time_formatted") or "unavailable"
            status = f.get("status")
            if status == "active" or not f.get("duration_available", True):
                if is_self:
                    return f"{op_name}, you are currently on an active shift (started at {st_str}). Final duration is ongoing."
                return f"{op_name} is currently on an active shift (started at {st_str}). Final duration is ongoing."
            dur_str = f.get("duration_formatted", "0 minutes")
            et_str = f.get("end_time_formatted") or "unavailable"
            timing = f" from {st_str} to {et_str}" if (st_str and et_str) else ""
            if is_self:
                return f"{op_name}, you worked {dur_str} on {f.get('shift_date', 'your shift')}{timing}."
            return f"{op_name} worked {dur_str} on {f.get('shift_date', 'their shift')}{timing}."

        cnt = f.get("wall_count", 0)
        date_val = f.get("date")
        date_suffix = f" on {date_val}" if date_val else ""
        if cnt == 0:
            if is_self:
                return f"{op_name}, you have no completed wall records{date_suffix}."
            return f"No completed wall records found for {op_name}{date_suffix}."
        walls_str = ", ".join(f"Wall {w}" for w in f.get("walls", []))
        if is_self:
            return f"{op_name}, you printed {cnt} walls{date_suffix}: {walls_str}."
        return f"{op_name} printed {cnt} walls{date_suffix}: {walls_str}."

    if i == Intent.GET_WALL_HISTORY:
        if not f.get("found"):
            return f.get("message") or f"No records found for Wall {f.get('wall_number')}."
        return f"Wall {f['wall_number']} status is {f['status']} with {f['layers_completed']} out of {f['segment_layers']} layers completed."

    if i == Intent.GET_SHIFT_ISSUES:
        if f.get("issue_id"):
            if f.get("focus") == "action":
                act = f.get("action_taken")
                return f"Action taken for Issue #{f['issue_id']}: {act}." if act else f"No action has been recorded for Issue #{f['issue_id']}."
            desc = f.get("description") or f.get("issue_type") or "Unknown issue"
            cause = f.get("cause") or "not recorded"
            action = f.get("action_taken") or "not recorded"
            res = f.get("result") or "not recorded"
            return f"Issue #{f['issue_id']} - {desc}. Cause: {cause}; Action: {action}; Result: {res}."

        cnt = f.get("count", 0)
        if not cnt:
            op_obj = shift_store.get_operator(f.get("operator_id")) if f.get("operator_id") else None
            op_display = op_obj.get("name") if op_obj else f.get("operator_id")
            op_str = f" for {op_display}" if op_display else ""
            date_str = f" on {f['date']}" if f.get("date") else ""
            return f"No issues have been recorded{op_str}{date_str}."

        issues = f.get("issues", [])
        if f.get("operator_id") or f.get("focus") == "action":
            op_obj = shift_store.get_operator(f.get("operator_id")) if f.get("operator_id") else None
            op_display = op_obj.get("name") if op_obj else f.get("operator_id")
            parts = [f"Found {cnt} issues for {op_display}:" if op_display else f"Found {cnt} issues:"]
            for iss in issues:
                date_str = f" on {iss['shift_date']}" if iss.get("shift_date") else ""
                desc = iss.get("desc") or "Unknown issue"
                cause = iss.get("cause") or "not recorded"
                action = iss.get("action_taken") or "not recorded"
                res = iss.get("result") or "not recorded"
                parts.append(f"Issue #{iss.get('issue_id', '?')}{date_str} - {desc}. Cause: {cause}; Action: {action}; Result: {res}.")
            return " ".join(parts)

        latest = issues[0]
        return f"{cnt} issues recorded. Most recent: {latest['type']} with description: {latest['desc']}."

    if i == Intent.GET_HANDOVER_STATUS:
        if not f.get("found"):
            return f.get("message") or "No handover record found for Line 1."
        return f"Handover status: Completed work: {f['completed_work']}. Pending work: {f['pending_work']}."

    # --- Phase 2C Action Intent Summaries ---
    if i == Intent.SHIFT_START:
        return f"Start {f['shift_type']} shift on {f['line_id']} with target {f['target_walls']} walls?"

    if i == Intent.SHIFT_END:
        return f"End current shift on {f['line_id']} and mark as closed?"

    if i == Intent.LOG_WALL_PRODUCTION:
        wn = f.get('wall_number')
        if wn is None:
            return "Which wall number was completed? Please specify the wall number to record production."
        return f"Record Wall {wn} as {f.get('status', 'completed')} on {f.get('line_id', 'line_1')}?"

    if i == Intent.LOG_SHIFT_ISSUE:
        return f"Log issue on {f['line_id']} with {f['downtime_minutes']} minutes downtime?"

    if i == Intent.LOG_SHIFT_HANDOVER:
        return f"Initiate shift handover on {f['line_id']}?"

    if i == Intent.RESOLVE_SHIFT_ISSUE:
        t_id = f.get("target_issue_id")
        id_str = f"Issue #{t_id}" if t_id else "issue"
        action_str = f" with action '{f['action_taken']}'" if f.get("action_taken") else ""
        res_str = f" as {f['result']}" if f.get("result") else ""
        return f"Update {id_str}{action_str}{res_str} on {f.get('line_id', 'line_1')}?"

    # --- Employee Wall Production Tracking Summaries ---
    if i == Intent.RECORD_WALLS_COMPLETED:
        if not f.get("recorded"):
            return f.get("reason") or "Failed to record wall production."
        qty = f.get("quantity", 0)
        unit = "wall" if qty == 1 else "walls"
        op_name = f.get("operator_name") or "Operator"
        rec_id = f.get("record_id", "")
        shift_id = f.get("shift_id", "unassigned")
        job_id = f.get("job_id", "unassigned")
        ts_str = f.get("timestamp_str", "now")
        return f"Recorded {qty} {unit} completed by {op_name} (Record ID: {rec_id}, Shift: {shift_id}, Job: {job_id}, Recorded at: {ts_str})."

    if i == Intent.GET_WALL_PRODUCTION:
        if not f.get("found"):
            return f.get("reason") or "No wall production records found."
        tot = f.get("total_quantity", 0)
        cnt = f.get("record_count", 0)
        unit = "wall" if tot == 1 else "walls"
        entry_unit = "entry" if cnt == 1 else "entries"
        op_name = f.get("operator_name")
        scope = f.get("scope", "today")

        scope_phrases = {
            "today": "today",
            "this_week": "this week",
            "active_shift": "for the current shift",
            "all_time": "overall",
            "job": f"under {f.get('job_id', 'job')}",
            "date": f"on {f.get('date', '')}",
        }
        scope_str = scope_phrases.get(scope, f"for {scope}")

        if op_name:
            if tot == 0:
                return f"{op_name} has completed 0 walls {scope_str}."
            return f"{op_name} has completed {tot} {unit} {scope_str} across {cnt} {entry_unit}."
        else:
            if tot == 0:
                return f"Total walls completed {scope_str}: 0 walls."
            by_ops = f.get("by_operator", [])
            breakdown = ""
            if by_ops and len(by_ops) > 1:
                breakdown = " (" + ", ".join(f"{b['operator_name']}: {b['quantity']}" for b in by_ops) + ")"
            return f"Total walls completed {scope_str}: {tot} {unit} across {cnt} {entry_unit}{breakdown}."

    if i == Intent.CORRECT_WALL_PRODUCTION:
        if not f.get("corrected"):
            return f.get("reason") or "Failed to correct wall production record."
        orig_id = f.get("original_record_id")
        repl_id = f.get("replacement_record_id")
        new_q = f.get("new_quantity")
        op_name = f.get("operator_name") or "Operator"
        unit = "wall" if new_q == 1 else "walls"
        return f"Corrected wall production record {orig_id}: replaced with {repl_id} ({new_q} {unit} for {op_name}). The original record is marked as corrected with an audit trail."

    return UNAVAILABLE_MESSAGE
