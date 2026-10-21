"""
agent/shift_store.py
Data access layer for Phase 2B implementing the FINAL VALIDATED Phase 2A schema.
Preserves exact table names, field names, constraints, foreign keys, and indexes.
"""

import datetime
import difflib
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from agent import db
from agent.logging_setup import get_logger

log = get_logger("ShiftStore")


def _row_to_dict(row: Optional[Any]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


# ===========================================================================
# 1. Operators
# ===========================================================================

def create_operator(
    operator_id: str,
    name: str,
    employee_code: Optional[str] = None,
    role: str = "operator",
    active: int = 1,
    db_path: Optional[str] = None,
) -> str:
    """Creates a new operator record. Primary key: operator_id."""
    clean_id = operator_id.strip()
    clean_name = name.strip()
    clean_emp = employee_code.strip() if employee_code else None
    if not clean_id:
        raise ValueError("operator_id cannot be empty")
    if not clean_name:
        raise ValueError("name cannot be empty")
    if active not in (0, 1):
        raise ValueError("active must be 0 or 1")

    now = time.time()
    with db.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO operators (operator_id, name, employee_code, role, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (clean_id, clean_name, clean_emp, role.strip(), active, now),
        )
    log.info("Created operator: %s (%s)", clean_id, clean_name)
    return clean_id


register_operator = create_operator



def get_operator(operator_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an operator by operator_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM operators WHERE operator_id = ?",
            (operator_id.strip(),),
        ).fetchone()
    return _row_to_dict(row)


def find_matching_operators(
    name: str, active_only: bool = False, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Resolves a name against the operators table.
    Checks exact operator_id, employee_code, full name, and first name.
    Returns a list of all matching operator records.
    """
    clean = (name or "").strip().lower()
    if not clean:
        return []

    with db.connect(db_path) as conn:
        sql = "SELECT * FROM operators"
        if active_only:
            sql += " WHERE active = 1"
        rows = [dict(r) for r in conn.execute(sql)]

    # 1. Exact operator_id
    by_id = [r for r in rows if (r.get("operator_id") or "").strip().lower() == clean]
    if len(by_id) == 1:
        return by_id

    # 2. Exact employee_code
    by_code = [r for r in rows if (r.get("employee_code") or "").strip().lower() == clean]
    if len(by_code) == 1:
        return by_code

    # 3. Exact full name
    exact = [r for r in rows if (r.get("name") or "").strip().lower() == clean]
    if exact:
        return exact

    # 4. First name match (if clean is at least 3 letters)
    if len(clean) >= 3:
        first = [
            r for r in rows
            if (r.get("name") or "").strip().lower().split(" ")[0] == clean
        ]
        if first:
            return first

    return []


def find_operator_by_name(
    name: str, active_only: bool = True, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Resolves a SPOKEN name to an operator record, or None.
    Returns None when the name matches nothing, and also when it matches more
    than one operator: ambiguous names are never guessed.
    """
    matches = find_matching_operators(name, active_only=active_only, db_path=db_path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        log.warning("Spoken name %r matches %d operators; refusing to guess.", name, len(matches))
    return None


# --- recovering a name speech-to-text broke apart ---------------------------
#
# Confirmed on the shop floor: "Anand here, start my shift" came back from
# speech-to-text as "On and Here Start My Shift." The name was split into two
# words, one of which ("and") is rightly rejected as not-a-name, so normal
# extraction returned nothing and the operator could not start their shift.
#
# The recovery below looks ONLY at the name slot of a self-introduction -- the
# word or two beside "here", "speaking", "this is" and similar -- never the
# whole sentence, so ordinary words elsewhere cannot be mistaken for a name.

# Words that end a self-introduction: "<name> here", "<name> speaking",
# "<name> bol raha hai", "main <name> hoon".
_NAME_BEFORE = ("here", "speaking", "bol", "hoon", "hu")
# Words that begin one: "this is <name>", "main <name>".
_NAME_AFTER = ("main",)
# Never a name, even when it sits in the slot.
_NOT_OPERATOR = {"okay", "ok", "kelvin", "and", "the", "is", "am", "my", "me", "i"}

# How close a heard fragment must be to an operator's name. Fitted to the
# confirmed case -- "onand" against "anand" scores 0.80 -- while keeping plain
# words out: "and" 0.75, "hand" 0.67, "stand" 0.60 all fall short.
NAME_MATCH_THRESHOLD = 0.80


def name_slot_candidates(text: str) -> List[str]:
    """
    Fragments in the name position of a self-introduction, most likely first.

    Each slot yields the single word beside the anchor and that word joined to
    its neighbour, because speech-to-text most often breaks a name into two:
    "on and here" gives ["and", "onand"]. Empty when the sentence contains no
    self-introduction at all -- which is how the caller tells "no name was
    offered" apart from "a name was offered but not understood".
    """
    words = re.findall(r"[a-z]+", (text or "").lower())
    found: List[str] = []

    def add(fragment: str) -> None:
        if len(fragment) >= 3 and fragment not in _NOT_OPERATOR and fragment not in found:
            found.append(fragment)

    for i, word in enumerate(words):
        if word in _NAME_BEFORE and i >= 1:
            add(words[i - 1])
            if i >= 2:
                add(words[i - 2] + words[i - 1])
        if word in _NAME_AFTER and i + 1 < len(words):
            add(words[i + 1])
            if i + 2 < len(words):
                add(words[i + 1] + words[i + 2])
        if word == "this" and i + 2 < len(words) and words[i + 1] == "is":
            add(words[i + 2])
            if i + 3 < len(words):
                add(words[i + 2] + words[i + 3])
    return found


def find_operator_in_text(
    text: str, active_only: bool = True, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Recovers the ONE active operator a noisy self-introduction refers to.

    Compares each name-slot fragment against every operator's first name and
    full name (spaces removed). Returns the operator only when exactly one
    operator matches; two or more is ambiguous and returns None, as does no
    match -- an attendance record must never be written for a guessed person.
    The caller still reads the resolved full name back to the operator for
    confirmation, so a wrong recovery is heard and refused before any write.
    """
    candidates = name_slot_candidates(text)
    if not candidates:
        return None

    with db.connect(db_path) as conn:
        sql = "SELECT * FROM operators"
        if active_only:
            sql += " WHERE active = 1"
        rows = [dict(r) for r in conn.execute(sql)]

    matched: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = (row.get("name") or "").strip().lower()
        if not name:
            continue
        targets = {name.split(" ")[0], re.sub(r"[^a-z]", "", name)}
        for fragment in candidates:
            if any(
                difflib.SequenceMatcher(None, fragment, target).ratio() >= NAME_MATCH_THRESHOLD
                for target in targets
                if len(target) >= 3
            ):
                matched[row["operator_id"]] = row
                break

    if len(matched) == 1:
        operator = next(iter(matched.values()))
        log.info("Recovered operator %s from noisy transcript %r", operator["operator_id"], text)
        return operator
    if len(matched) > 1:
        log.warning(
            "Transcript %r could be any of %s; refusing to guess.", text, sorted(matched)
        )
    return None


def get_operator_by_employee_code(employee_code: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an operator by unique employee_code."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM operators WHERE employee_code = ?",
            (employee_code.strip(),),
        ).fetchone()
    return _row_to_dict(row)


def list_operators(active_only: bool = True, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists operators, optionally filtering by active."""
    query = "SELECT * FROM operators"
    params: List[Any] = []
    if active_only:
        query += " WHERE active = 1"
    query += " ORDER BY name ASC"

    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def update_operator(
    operator_id: str,
    name: Optional[str] = None,
    employee_code: Optional[str] = None,
    role: Optional[str] = None,
    active: Optional[int] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Updates mutable operator fields."""
    sets, params = [], []
    if name is not None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name cannot be empty")
        sets.append("name = ?")
        params.append(clean_name)
    if employee_code is not None:
        sets.append("employee_code = ?")
        params.append(employee_code.strip() if employee_code else None)
    if role is not None:
        sets.append("role = ?")
        params.append(role.strip())
    if active is not None:
        if active not in (0, 1):
            raise ValueError("active must be 0 or 1")
        sets.append("active = ?")
        params.append(active)

    if not sets:
        return False

    params.append(operator_id.strip())
    with db.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE operators SET {', '.join(sets)} WHERE operator_id = ?",
            params,
        )
        return cur.rowcount > 0


# ===========================================================================
# 2. Shifts
# ===========================================================================

VALID_SHIFT_TYPES = {"Morning", "Afternoon", "Night"}
VALID_SHIFT_STATUSES = {"active", "closed", "flagged", "abandoned"}


def start_shift(
    operator_id: str,
    shift_date: Optional[str] = None,
    shift_type: Optional[str] = None,
    line_id: str = "line_1",
    target_walls: int = 0,
    summary_notes: Optional[str] = None,
    start_time: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    """Starts a new shift with status 'active'. Returns shift_id."""
    clean_op_id = operator_id.strip()

    # Handle callers passing (operator_id, line_id) e.g. start_shift("OP_RAHUL", "Line 1")
    if shift_date is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(shift_date)):
        if str(shift_date) in VALID_SHIFT_TYPES:
            shift_type = str(shift_date)
            shift_date = local_today()
        else:
            line_id = str(shift_date)
            shift_date = local_today()
    if not shift_date:
        shift_date = local_today()

    if not shift_type or shift_type not in VALID_SHIFT_TYPES:
        shift_type = "Morning"

    if target_walls < 0:
        raise ValueError("target_walls cannot be negative")

    ts = start_time if start_time is not None else time.time()
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO shifts (
                operator_id, line_id, shift_date, shift_type,
                start_time, status, target_walls, summary_notes, created_at
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (clean_op_id, line_id.strip(), shift_date.strip(), shift_type, ts, target_walls, summary_notes, ts),
        )
        shift_id = int(cur.lastrowid)
    log.info("Started shift %d: line=%s, operator=%s, date=%s, type=%s", shift_id, line_id, clean_op_id, shift_date, shift_type)
    return shift_id


def get_shift(shift_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves shift details by shift_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shifts WHERE shift_id = ?",
            (shift_id,),
        ).fetchone()
    return _row_to_dict(row)


def get_active_shift_for_line(line_id: str = "line_1", db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the active shift on a specific line utilizing idx_shifts_single_active_line."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shifts WHERE line_id = ? AND status = 'active' LIMIT 1",
            (line_id.strip(),),
        ).fetchone()
    return _row_to_dict(row)


def get_active_shift(operator_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the active shift for a given operator."""
    clean_op = operator_id.strip() if operator_id else ""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shifts WHERE operator_id = ? AND status = 'active' ORDER BY start_time DESC LIMIT 1",
            (clean_op,),
        ).fetchone()
    return _row_to_dict(row)



def end_shift(
    shift_id: int,
    end_time: Optional[float] = None,
    status: str = "closed",
    summary_notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Closes a shift with terminal status: 'closed', 'flagged', or 'abandoned'."""
    if status not in ("closed", "flagged", "abandoned"):
        raise ValueError(f"Invalid terminal shift status: {status}. Must be one of ('closed', 'flagged', 'abandoned')")

    ts = end_time if end_time is not None else time.time()
    sets = ["status = ?", "end_time = ?"]
    params: List[Any] = [status, ts]
    if summary_notes is not None:
        sets.append("summary_notes = ?")
        params.append(summary_notes)
    params.append(shift_id)

    with db.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE shifts SET {', '.join(sets)} WHERE shift_id = ?",
            params,
        )
        return cur.rowcount > 0


def list_shifts(
    shift_date: Optional[str] = None,
    shift_type: Optional[str] = None,
    operator_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists shifts with optional filters on date/type (idx_shifts_date) or operator (idx_shifts_operator)."""
    query = "SELECT * FROM shifts"
    clauses, params = [], []
    if shift_date:
        clauses.append("shift_date = ?")
        params.append(shift_date.strip())
    if shift_type:
        clauses.append("shift_type = ?")
        params.append(shift_type.strip())
    if operator_id:
        clauses.append("operator_id = ?")
        params.append(operator_id.strip())
    if status:
        clauses.append("status = ?")
        params.append(status.strip())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY start_time DESC LIMIT ?"
    params.append(limit)

    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# 3. Production Records
# ===========================================================================

VALID_PRODUCTION_STATUSES = {"in_progress", "completed", "aborted", "defect"}


def record_production(
    shift_id: int,
    wall_number: int,
    job_id: Optional[int] = None,
    print_file: Optional[str] = None,
    wall_height: float = 0.0,
    segment_layers: int = 0,
    layers_completed: int = 0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    status: str = "in_progress",
    is_continuation_of: Optional[int] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Creates a production record using exact Phase 2A columns. Returns production_id."""
    if wall_height < 0.0:
        raise ValueError("wall_height cannot be negative")
    if segment_layers < 0:
        raise ValueError("segment_layers cannot be negative")
    if layers_completed < 0:
        raise ValueError("layers_completed cannot be negative")
    if status not in VALID_PRODUCTION_STATUSES:
        raise ValueError(f"Invalid production status: {status}. Must be one of {VALID_PRODUCTION_STATUSES}")

    ts = start_time if start_time is not None else time.time()
    now = time.time()
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO production_records (
                shift_id, job_id, wall_number, print_file, wall_height,
                segment_layers, layers_completed, start_time, end_time,
                status, is_continuation_of, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shift_id,
                job_id,
                wall_number,
                print_file,
                float(wall_height),
                int(segment_layers),
                int(layers_completed),
                ts,
                end_time,
                status,
                is_continuation_of,
                notes,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_production_record(production_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a production record by production_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM production_records WHERE production_id = ?",
            (production_id,),
        ).fetchone()
    return _row_to_dict(row)


def update_production_record(
    production_id: int,
    layers_completed: Optional[int] = None,
    end_time: Optional[float] = None,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Updates mutable fields on a production record."""
    sets, params = [], []
    if layers_completed is not None:
        if layers_completed < 0:
            raise ValueError("layers_completed cannot be negative")
        sets.append("layers_completed = ?")
        params.append(int(layers_completed))
    if end_time is not None:
        sets.append("end_time = ?")
        params.append(float(end_time))
    if status is not None:
        if status not in VALID_PRODUCTION_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of {VALID_PRODUCTION_STATUSES}")
        sets.append("status = ?")
        params.append(status)
    if notes is not None:
        sets.append("notes = ?")
        params.append(notes)

    if not sets:
        return False

    params.append(production_id)
    with db.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE production_records SET {', '.join(sets)} WHERE production_id = ?",
            params,
        )
        return cur.rowcount > 0


def get_shift_production(shift_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all production records logged for a shift (idx_production_shift)."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM production_records WHERE shift_id = ? ORDER BY wall_number ASC, start_time ASC",
            (shift_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_production_continuations(parent_production_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves continuation segments (idx_production_continuation)."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM production_records WHERE is_continuation_of = ? ORDER BY start_time ASC",
            (parent_production_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_operator_production(
    operator_id: str,
    shift_id: Optional[int] = None,
    shift_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all production records logged for an operator across shifts.

    Joins production_records -> shifts on shift_id, filtering by shifts.operator_id.
    Supports optional shift_id and shift_date filters.
    Returns actual stored production only, including wall, status, timestamps, job info.
    Preserves closed-shift history queryability.
    """
    clean_op = operator_id.strip() if operator_id else ""
    if not clean_op:
        return []
    query = (
        "SELECT pr.*, s.operator_id, s.shift_date, s.shift_type, s.line_id, s.status AS shift_status "
        "FROM production_records pr "
        "JOIN shifts s ON pr.shift_id = s.shift_id "
        "WHERE s.operator_id = ?"
    )
    params: List[Any] = [clean_op]
    if shift_id is not None:
        query += " AND pr.shift_id = ?"
        params.append(shift_id)
    if shift_date is not None:
        query += " AND s.shift_date = ?"
        params.append(shift_date.strip())
    query += " ORDER BY pr.start_time ASC, pr.wall_number ASC"
    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# 4. Shift Issues
# ===========================================================================

VALID_ISSUE_SEVERITIES = {"info", "warning", "critical"}


def log_issue(
    shift_id: int,
    issue_type: str,
    description: str,
    wall_number: Optional[int] = None,
    timestamp: Optional[float] = None,
    cause: Optional[str] = None,
    action_taken: Optional[str] = None,
    result: Optional[str] = None,
    severity: str = "warning",
    downtime_minutes: float = 0.0,
    rag_recommendation: Optional[str] = None,
    rag_verified: int = 0,
    rag_source: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Logs a shift issue. Returns issue_id."""
    clean_type = issue_type.strip()
    clean_desc = description.strip()
    if not clean_type:
        raise ValueError("issue_type cannot be empty")
    if not clean_desc:
        raise ValueError("description cannot be empty")
    if severity not in VALID_ISSUE_SEVERITIES:
        raise ValueError(f"Invalid severity: {severity}. Must be one of {VALID_ISSUE_SEVERITIES}")
    if downtime_minutes < 0.0:
        raise ValueError("downtime_minutes cannot be negative")

    ts = timestamp if timestamp is not None else time.time()
    now = time.time()
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO shift_issues (
                shift_id, wall_number, timestamp, issue_type, description,
                cause, action_taken, result, severity, downtime_minutes,
                rag_recommendation, rag_verified, rag_source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shift_id,
                wall_number,
                ts,
                clean_type,
                clean_desc,
                cause,
                action_taken,
                result,
                severity,
                float(downtime_minutes),
                rag_recommendation,
                int(rag_verified),
                rag_source,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_issue(issue_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an issue by issue_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shift_issues WHERE issue_id = ?",
            (issue_id,),
        ).fetchone()
    return _row_to_dict(row)


def update_issue(
    issue_id: int,
    cause: Optional[str] = None,
    action_taken: Optional[str] = None,
    result: Optional[str] = None,
    severity: Optional[str] = None,
    downtime_minutes: Optional[float] = None,
    rag_recommendation: Optional[str] = None,
    rag_verified: Optional[int] = None,
    rag_source: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Updates resolution and post-incident investigation details on an issue."""
    sets, params = [], []
    if cause is not None:
        sets.append("cause = ?")
        params.append(cause)
    if action_taken is not None:
        sets.append("action_taken = ?")
        params.append(action_taken)
    if result is not None:
        sets.append("result = ?")
        params.append(result)
    if severity is not None:
        if severity not in VALID_ISSUE_SEVERITIES:
            raise ValueError(f"Invalid severity: {severity}. Must be one of {VALID_ISSUE_SEVERITIES}")
        sets.append("severity = ?")
        params.append(severity)
    if downtime_minutes is not None:
        if downtime_minutes < 0.0:
            raise ValueError("downtime_minutes cannot be negative")
        sets.append("downtime_minutes = ?")
        params.append(float(downtime_minutes))
    if rag_recommendation is not None:
        sets.append("rag_recommendation = ?")
        params.append(rag_recommendation)
    if rag_verified is not None:
        sets.append("rag_verified = ?")
        params.append(int(rag_verified))
    if rag_source is not None:
        sets.append("rag_source = ?")
        params.append(rag_source)

    if not sets:
        return False

    params.append(issue_id)
    with db.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE shift_issues SET {', '.join(sets)} WHERE issue_id = ?",
            params,
        )
        return cur.rowcount > 0


def list_shift_issues(
    shift_id: Optional[int] = None,
    issue_type: Optional[str] = None,
    operator_id: Optional[str] = None,
    unresolved_only: bool = False,
    shift_date: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists issues utilizing idx_issues_shift or idx_issues_type_time, with optional operator, shift_date and unresolved filters."""
    clauses, params = [], []
    has_shift_join = operator_id is not None or shift_date is not None
    if has_shift_join:
        query = (
            "SELECT i.*, s.shift_date, s.shift_type, s.operator_id "
            "FROM shift_issues i JOIN shifts s ON i.shift_id = s.shift_id"
        )
        if operator_id is not None:
            clauses.append("s.operator_id = ?")
            params.append(operator_id.strip())
        if shift_date is not None:
            clauses.append("s.shift_date = ?")
            params.append(shift_date.strip())
    else:
        query = "SELECT * FROM shift_issues"

    if shift_id is not None:
        clauses.append("shift_id = ?" if not has_shift_join else "i.shift_id = ?")
        params.append(shift_id)
    if issue_type is not None:
        clauses.append("issue_type = ?" if not has_shift_join else "i.issue_type = ?")
        params.append(issue_type.strip())
    if unresolved_only:
        col = "result" if not has_shift_join else "i.result"
        clauses.append(f"({col} IS NULL OR {col} = '' OR {col} = 'not_resolved')")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    order_col = "timestamp" if not has_shift_join else "i.timestamp"
    query += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)

    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# 5. Handovers
# ===========================================================================

def create_handover(
    from_shift_id: int,
    to_shift_id: Optional[int] = None,
    line_id: str = "line_1",
    handover_time: Optional[float] = None,
    completed_work: Optional[str] = None,
    pending_work: Optional[str] = None,
    open_issues: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Initiates a shift handover. Returns handover_id."""
    if to_shift_id is not None and from_shift_id == to_shift_id:
        raise ValueError("from_shift_id and to_shift_id cannot be identical")

    ts = handover_time if handover_time is not None else time.time()
    now = time.time()
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO handovers (
                from_shift_id, to_shift_id, line_id, handover_time,
                completed_work, pending_work, open_issues, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                from_shift_id,
                to_shift_id,
                line_id.strip(),
                ts,
                completed_work,
                pending_work,
                open_issues,
                notes,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_handover(handover_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves handover details by handover_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM handovers WHERE handover_id = ?",
            (handover_id,),
        ).fetchone()
    return _row_to_dict(row)


def get_pending_handover_for_line(line_id: str = "line_1", db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the newest unassigned handover for a line utilizing idx_handovers_line_match."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM handovers
            WHERE line_id = ? AND to_shift_id IS NULL
            ORDER BY handover_time DESC LIMIT 1
            """,
            (line_id.strip(),),
        ).fetchone()
    return _row_to_dict(row)


def complete_handover(
    handover_id: int,
    to_shift_id: int,
    notes: Optional[str] = None,
    close_from_shift: bool = True,
    db_path: Optional[str] = None,
) -> bool:
    """
    Atomically binds to_shift_id to the handover record and optionally
    marks from_shift_id as 'closed'.
    """
    now = time.time()
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT from_shift_id, to_shift_id FROM handovers WHERE handover_id = ?",
            (handover_id,),
        ).fetchone()
        if not row:
            return False
        if row["from_shift_id"] == to_shift_id:
            raise ValueError("to_shift_id cannot match from_shift_id")

        from_shift_id = row["from_shift_id"]
        sets = ["to_shift_id = ?"]
        params: List[Any] = [to_shift_id]
        if notes is not None:
            sets.append("notes = ?")
            params.append(notes)
        params.append(handover_id)

        conn.execute(f"UPDATE handovers SET {', '.join(sets)} WHERE handover_id = ?", params)
        if close_from_shift:
            conn.execute(
                "UPDATE shifts SET status = 'closed', end_time = ? WHERE shift_id = ? AND status = 'active'",
                (now, from_shift_id),
            )
        return True


# ===========================================================================
# 6. Event Corrections
# ===========================================================================

def log_event_correction(
    entity_type: str,
    entity_id: int,
    field_name: str,
    new_value: str,
    corrected_by: str,
    reason: str,
    old_value: Optional[str] = None,
    source: str = "manual",
    corrected_at: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    """Logs an audit correction for an entity field. Returns correction_id."""
    clean_entity_type = entity_type.strip()
    clean_field = field_name.strip()
    clean_new_val = str(new_value).strip()
    clean_by = corrected_by.strip()
    clean_reason = reason.strip()

    if not clean_entity_type:
        raise ValueError("entity_type cannot be empty")
    if not clean_field:
        raise ValueError("field_name cannot be empty")
    if not clean_by:
        raise ValueError("corrected_by cannot be empty")
    if not clean_reason:
        raise ValueError("reason cannot be empty")

    ts = corrected_at if corrected_at is not None else time.time()
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO event_corrections (
                entity_type, entity_id, field_name, old_value,
                new_value, corrected_by, corrected_at, source, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_entity_type,
                entity_id,
                clean_field,
                old_value,
                clean_new_val,
                clean_by,
                ts,
                source.strip(),
                clean_reason,
            ),
        )
        return int(cur.lastrowid)


def get_corrections_for_entity(
    entity_type: str,
    entity_id: int,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all corrections for an entity utilizing idx_corrections_entity."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM event_corrections
            WHERE entity_type = ? AND entity_id = ?
            ORDER BY corrected_at DESC
            """,
            (entity_type.strip(), entity_id),
        ).fetchall()
    return [dict(r) for r in rows]


def list_event_corrections(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists recent event corrections."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM event_corrections ORDER BY corrected_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]

# ===========================================================================
# Attendance Records
# ===========================================================================

def local_today() -> str:
    """Server local calendar date (YYYY-MM-DD). Source of truth for 'today'."""
    return datetime.datetime.now().strftime("%Y-%m-%d")


def generate_operator_id(name: str, db_path: Optional[str] = None) -> str:
    """Project convention: OP_<NAME>_<6HEX>, unique against the operators table."""
    clean = re.sub(r"[^A-Za-z0-9]+", "_", (name or "").strip().upper()).strip("_") or "EMP"
    for _ in range(8):
        candidate = f"OP_{clean}_{uuid.uuid4().hex[:6].upper()}"
        if not get_operator(candidate, db_path=db_path):
            return candidate
    return f"OP_{clean}_{uuid.uuid4().hex[:8].upper()}"


def format_worked_hours(seconds: float) -> str:
    """UI format: 0m, 32m, 1h 15m, 4h 08m."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def format_worked_spoken(seconds: float) -> str:
    """Spoken duration: '4 hours and 18 minutes', '32 minutes'."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours <= 0:
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit}"
    hour_unit = "hour" if hours == 1 else "hours"
    minute_unit = "minute" if minutes == 1 else "minutes"
    return f"{hours} {hour_unit} and {minutes} {minute_unit}"


def format_clock_time(ts: Optional[float]) -> str:
    if ts is None:
        return "--"
    moment = datetime.datetime.fromtimestamp(float(ts))
    hour = moment.strftime("%I").lstrip("0") or "12"
    return f"{hour}:{moment.strftime('%M %p')}"


# 9-hour regular workday
STANDARD_WORKDAY_SECONDS = 9 * 3600


def attendance_worked_seconds(row: Dict[str, Any], now: Optional[float] = None) -> float:
    start = float(row.get("shift_start") or 0)
    if not start:
        return 0.0
    if (row.get("status") or "").upper() == "ACTIVE" or row.get("shift_end") is None:
        end = float(now if now is not None else time.time())
    else:
        end = float(row["shift_end"])
    return max(0.0, end - start)


def attendance_overtime_seconds(worked_seconds: float) -> float:
    """Overtime is duration worked beyond the standard 9-hour workday."""
    return max(0.0, float(worked_seconds) - STANDARD_WORKDAY_SECONDS)


def enrich_attendance_record(row: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    clock = float(now if now is not None else time.time())
    out = dict(row)
    worked = attendance_worked_seconds(out, now=clock)
    overtime = attendance_overtime_seconds(worked)
    out["worked_seconds"] = int(worked)
    out["worked_hours"] = format_worked_hours(worked)
    out["worked_display"] = format_worked_hours(worked)
    out["overtime_seconds"] = int(overtime)
    out["overtime_display"] = format_worked_hours(overtime)
    out["overtime_spoken"] = format_worked_spoken(overtime)
    out["server_now"] = clock
    return out


def get_active_attendance(
    operator_id: Optional[str] = None,
    employee_name: Optional[str] = None,
    attendance_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The one ACTIVE shift for an employee on a given local date."""
    day = (attendance_date or local_today()).strip()
    clean_op = operator_id.strip() if operator_id else None
    clean_name = employee_name.strip() if employee_name else None
    with db.connect(db_path) as conn:
        if clean_op:
            row = conn.execute(
                """
                SELECT * FROM attendance_records
                WHERE operator_id = ? AND attendance_date = ? AND status = 'ACTIVE'
                ORDER BY shift_start DESC LIMIT 1
                """,
                (clean_op, day),
            ).fetchone()
        elif clean_name:
            row = conn.execute(
                """
                SELECT * FROM attendance_records
                WHERE employee_name = ? AND attendance_date = ? AND status = 'ACTIVE'
                ORDER BY shift_start DESC LIMIT 1
                """,
                (clean_name, day),
            ).fetchone()
        else:
            return None
    return _row_to_dict(row)


def create_attendance_record(
    operator_id: Optional[str],
    employee_name: str,
    attendance_date: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates a new ACTIVE attendance record. Timestamp comes from the server clock."""
    clean_name = employee_name.strip()
    clean_op = operator_id.strip() if operator_id else None
    day = (attendance_date or local_today()).strip()

    if not clean_name:
        raise ValueError("employee_name cannot be empty")

    now = time.time()
    existing = get_active_attendance(
        operator_id=clean_op, employee_name=clean_name, attendance_date=day, db_path=db_path
    )
    if existing:
        started = format_clock_time(existing.get("shift_start"))
        raise ValueError(
            f"{clean_name} already has an active shift. He started at {started}."
        )

    with db.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO attendance_records
            (operator_id, employee_name, attendance_date, shift_start, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
            """,
            (clean_op, clean_name, day, now, now, now),
        )
        record_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM attendance_records WHERE id = ?", (record_id,)).fetchone()
        return enrich_attendance_record(dict(row), now=now)

def checkout_attendance_record(
    operator_id: Optional[str] = None,
    employee_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Checks out an ACTIVE attendance record."""
    clean_op = operator_id.strip() if operator_id else None
    clean_name = employee_name.strip() if employee_name else None
    
    if not clean_op and not clean_name:
        raise ValueError("Must provide either operator_id or employee_name")
        
    now = time.time()
    
    with db.connect(db_path) as conn:
        if clean_op:
            existing = conn.execute(
                "SELECT id FROM attendance_records WHERE operator_id = ? AND status = 'ACTIVE' ORDER BY shift_start DESC LIMIT 1",
                (clean_op,)
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT id FROM attendance_records WHERE employee_name = ? AND status = 'ACTIVE' ORDER BY shift_start DESC LIMIT 1",
                (clean_name,)
            ).fetchone()
            
        if not existing:
            raise ValueError("No ACTIVE attendance record found.")
            
        record_id = existing["id"]
        
        conn.execute(
            """
            UPDATE attendance_records 
            SET shift_end = ?, status = 'COMPLETED', updated_at = ?
            WHERE id = ?
            """,
            (now, now, record_id)
        )
        
        row = conn.execute("SELECT * FROM attendance_records WHERE id = ?", (record_id,)).fetchone()
        return enrich_attendance_record(dict(row))

def list_attendance_records(
    date_filter: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists attendance records for a given date (defaults to today)."""
    day = (date_filter or local_today()).strip()
    now = time.time()
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM attendance_records WHERE attendance_date = ? ORDER BY shift_start DESC LIMIT ?",
            (day, limit)
        ).fetchall()

    return [enrich_attendance_record(dict(r), now=now) for r in rows]


def get_attendance_record(record_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an attendance record by its primary key ID and enriches it."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM attendance_records WHERE id = ?",
            (record_id,),
        ).fetchone()
    return enrich_attendance_record(dict(row)) if row else None


def check_in_attendance(
    operator_id: str,
    timestamp: Optional[Any] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Records an operator check-in for attendance."""
    clean_op = operator_id.strip() if operator_id else ""
    op = get_operator(clean_op, db_path=db_path)
    name = op["name"] if op else clean_op

    if timestamp is not None:
        if isinstance(timestamp, (int, float)):
            start_ts = float(timestamp)
            day = datetime.datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")
        else:
            try:
                dt = datetime.datetime.fromisoformat(str(timestamp))
                start_ts = dt.timestamp()
                day = dt.strftime("%Y-%m-%d")
            except Exception:
                start_ts = time.time()
                day = local_today()
    else:
        start_ts = time.time()
        day = local_today()

    now = time.time()
    with db.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO attendance_records
            (operator_id, employee_name, attendance_date, shift_start, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
            """,
            (clean_op, name, day, start_ts, now, now),
        )
        record_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM attendance_records WHERE id = ?", (record_id,)).fetchone()
        return enrich_attendance_record(dict(row), now=now)


def check_out_attendance(
    operator_id: str,
    timestamp: Optional[Any] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Records an operator check-out for attendance."""
    clean_op = operator_id.strip() if operator_id else ""
    if timestamp is not None:
        if isinstance(timestamp, (int, float)):
            end_ts = float(timestamp)
        else:
            try:
                dt = datetime.datetime.fromisoformat(str(timestamp))
                end_ts = dt.timestamp()
            except Exception:
                end_ts = time.time()
    else:
        end_ts = time.time()

    now = time.time()
    with db.connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM attendance_records WHERE operator_id = ? AND status = 'ACTIVE' ORDER BY shift_start DESC LIMIT 1",
            (clean_op,)
        ).fetchone()
        if not existing:
            return None
        record_id = existing["id"]
        conn.execute(
            """
            UPDATE attendance_records
            SET shift_end = ?, status = 'COMPLETED', updated_at = ?
            WHERE id = ?
            """,
            (end_ts, now, record_id),
        )
        row = conn.execute("SELECT * FROM attendance_records WHERE id = ?", (record_id,)).fetchone()
        return enrich_attendance_record(dict(row))


# ===========================================================================
# 10. Persistent Employee Wall-Production Records
# ===========================================================================

def generate_wall_record_id(date_str: Optional[str] = None) -> str:
    """Generates human-readable, non-colliding wall production record ID."""
    d = (date_str or local_today()).replace("-", "")
    suffix = uuid.uuid4().hex[:8].upper()
    return f"WALL-REC-{d}-{suffix}"


def record_wall_production(
    operator_id: str,
    quantity: int,
    production_date: Optional[str] = None,
    production_time: Optional[float] = None,
    job_id: Optional[Any] = None,
    shift_id: Optional[int] = None,
    notes: Optional[str] = None,
    input_source: str = "chat",
    request_id: Optional[str] = None,
    recorded_by: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persists an employee wall production event.
    - Validates quantity > 0.
    - Validates operator exists in operators table.
    - Handles job_id and shift_id explicitly when unavailable (does not fabricate).
    - Prevents accidental duplicates via unique request_id.
    """
    clean_op = (operator_id or "").strip()
    if not clean_op:
        raise ValueError("operator_id is required.")

    # Validate operator exists
    op = get_operator(clean_op, db_path=db_path)
    if not op:
        matches = find_matching_operators(clean_op, db_path=db_path)
        if len(matches) == 1:
            clean_op = matches[0]["operator_id"]
            op = matches[0]
        else:
            raise ValueError(f"Operator '{clean_op}' does not exist in the database.")

    try:
        qty = int(quantity)
    except (ValueError, TypeError):
        raise ValueError("Wall quantity must be a valid integer.")
    if qty <= 0:
        raise ValueError(f"Wall quantity must be a positive integer greater than 0 (got {qty}).")

    # Idempotency check on request_id
    if request_id:
        clean_req = str(request_id).strip()
        with db.connect(db_path) as conn:
            existing = conn.execute(
                "SELECT * FROM wall_production_records WHERE request_id = ?",
                (clean_req,)
            ).fetchone()
            if existing:
                log.info("Idempotent hit for request_id '%s': returning existing record", clean_req)
                return _enrich_wall_production_record(dict(existing), op)

    now = time.time()
    prod_time = float(production_time) if production_time is not None else now
    day = production_date or local_today()
    actual_source = recorded_by or input_source or "chat"

    resolved_job_id = None
    if job_id is not None:
        try:
            resolved_job_id = int(job_id)
        except (ValueError, TypeError):
            job_str = str(job_id).strip()
            if notes and job_str not in notes:
                notes = f"{notes} (Job: {job_str})"
            elif not notes:
                notes = f"Job: {job_str}"
            resolved_job_id = None

    if resolved_job_id is None:
        try:
            active_job = db.get_active_job(db_path)
            if active_job:
                resolved_job_id = active_job["id"]
        except Exception:
            resolved_job_id = None

    resolved_shift_id = shift_id
    if resolved_shift_id is None:
        try:
            active_shifts = list_shifts(status="active", db_path=db_path)
            for s in active_shifts:
                if s["operator_id"] == clean_op:
                    resolved_shift_id = s["shift_id"]
                    break
        except Exception:
            resolved_shift_id = None

    rec_id = generate_wall_record_id(day)

    with db.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO wall_production_records (
                record_id, operator_id, quantity, production_date, production_time,
                job_id, shift_id, notes, input_source, created_at, request_id,
                is_corrected, correction_notes, corrected_by, corrected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL)
            """,
            (
                rec_id, clean_op, qty, day, prod_time,
                resolved_job_id, resolved_shift_id, notes, actual_source, now, request_id,
            ),
        )
        row = conn.execute(
            "SELECT * FROM wall_production_records WHERE record_id = ?",
            (rec_id,)
        ).fetchone()

    log.info("Recorded wall production: %s (%d walls by %s on %s)", rec_id, qty, op["name"], day)
    return _enrich_wall_production_record(dict(row), op)


def _enrich_wall_production_record(record: Dict[str, Any], op: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    res = dict(record)
    if not op and res.get("operator_id"):
        op = get_operator(res["operator_id"])
    res["operator_name"] = op["name"] if op else res.get("operator_id")
    res["employee_code"] = op.get("employee_code") if op else None
    res["timestamp"] = res.get("production_time") or res.get("created_at") or time.time()
    return res


def get_wall_production_total(
    operator_id: Optional[str] = None,
    date_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    job_id: Optional[int] = None,
    shift_id: Optional[int] = None,
    scope: str = "today",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sums completed walls over a validated scope (today, this_week, job, active_shift, all_time, date_range).
    Excludes corrected records (is_corrected = 1).
    """
    op = None
    clean_op = None
    if operator_id:
        clean_op = operator_id.strip()
        op = get_operator(clean_op, db_path=db_path)
        if not op:
            matches = find_matching_operators(clean_op, db_path=db_path)
            if len(matches) == 1:
                clean_op = matches[0]["operator_id"]
                op = matches[0]

    scope_lower = scope.lower().strip()
    today_str = local_today()
    s_date = date_from or start_date
    e_date = date_to or end_date

    conditions = ["is_corrected = 0"]
    params: List[Any] = []

    if clean_op:
        conditions.append("operator_id = ?")
        params.append(clean_op)

    if scope_lower == "today":
        effective_date = date_filter or today_str
        conditions.append("production_date = ?")
        params.append(effective_date)
        period_description = f"today ({effective_date})"
    elif scope_lower in ("this_week", "week"):
        curr_dt = datetime.date.fromisoformat(today_str)
        monday = curr_dt - datetime.timedelta(days=curr_dt.weekday())
        sunday = monday + datetime.timedelta(days=6)
        mon_str = monday.isoformat()
        sun_str = sunday.isoformat()
        conditions.append("production_date >= ? AND production_date <= ?")
        params.extend([mon_str, sun_str])
        period_description = f"this week ({mon_str} to {sun_str})"
    elif scope_lower == "job":
        if job_id is not None:
            conditions.append("job_id = ?")
            params.append(job_id)
            period_description = f"job #{job_id}"
        else:
            period_description = "the specified job"
    elif scope_lower == "active_shift":
        if shift_id is not None:
            conditions.append("shift_id = ?")
            params.append(shift_id)
            period_description = f"shift #{shift_id}"
        else:
            period_description = "the active shift"
    elif scope_lower in ("date_range", "date") and (s_date or e_date):
        if s_date and e_date:
            conditions.append("production_date >= ? AND production_date <= ?")
            params.extend([s_date, e_date])
            period_description = f"date range ({s_date} to {e_date})"
        elif s_date:
            conditions.append("production_date = ?")
            params.append(s_date)
            period_description = f"date {s_date}"
    elif scope_lower == "all_time":
        period_description = "all time"
    else:
        effective_date = date_filter or today_str
        conditions.append("production_date = ?")
        params.append(effective_date)
        period_description = f"today ({effective_date})"

    where_clause = " WHERE " + " AND ".join(conditions)

    with db.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM wall_production_records{where_clause} ORDER BY production_time DESC",
            params,
        ).fetchall()

    records = [_enrich_wall_production_record(dict(r), op) for r in rows]
    total_walls = sum(r["quantity"] for r in records)
    record_count = len(records)

    by_op_dict: Dict[str, Dict[str, Any]] = {}
    for r in records:
        op_id_key = r.get("operator_id") or "unassigned"
        if op_id_key not in by_op_dict:
            by_op_dict[op_id_key] = {
                "operator_id": op_id_key,
                "operator_name": r.get("operator_name") or op_id_key,
                "employee_code": r.get("employee_code"),
                "total_quantity": 0,
                "record_count": 0,
            }
        by_op_dict[op_id_key]["total_quantity"] += r["quantity"]
        by_op_dict[op_id_key]["record_count"] += 1
    by_operator = list(by_op_dict.values())

    return {
        "operator_id": clean_op,
        "operator_name": op["name"] if op else (clean_op or "All Operators"),
        "total_quantity": total_walls,
        "total_walls": total_walls,
        "record_count": record_count,
        "by_operator": by_operator,
        "scope": scope_lower,
        "period_description": period_description,
        "date": date_filter or today_str,
        "job_id": job_id,
        "shift_id": shift_id,
        "records": records,
        "has_records": record_count > 0,
    }


def correct_wall_production(
    operator_id: Optional[str] = None,
    new_quantity: Optional[int] = None,
    previous_quantity: Optional[int] = None,
    record_id: Optional[str] = None,
    original_record_id: Optional[str] = None,
    corrected_quantity: Optional[int] = None,
    reason: str = "Operator verbal correction",
    corrected_by: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Safely corrects an earlier wall production record without inflating totals.
    - Marks target record as is_corrected = 1 with audit notes.
    - Inserts corrected replacement record.
    - Records entry in event_corrections table.
    """
    target_rec_id = (original_record_id or record_id or "").strip()
    qty_val = corrected_quantity if corrected_quantity is not None else new_quantity
    if qty_val is None:
        raise ValueError("Corrected quantity is required.")

    try:
        new_qty = int(qty_val)
    except (ValueError, TypeError):
        raise ValueError("New wall quantity must be a valid integer.")
    if new_qty <= 0:
        raise ValueError(f"Corrected quantity must be positive and greater than 0 (got {new_qty}).")

    with db.connect(db_path) as conn:
        target_row = None
        if target_rec_id:
            target_row = conn.execute(
                "SELECT * FROM wall_production_records WHERE record_id = ?",
                (target_rec_id,)
            ).fetchone()
        elif operator_id:
            clean_op = operator_id.strip()
            op = get_operator(clean_op, db_path=db_path)
            if not op:
                matches = find_matching_operators(clean_op, db_path=db_path)
                if len(matches) == 1:
                    clean_op = matches[0]["operator_id"]
                    op = matches[0]
                else:
                    raise ValueError(f"Operator '{clean_op}' does not exist.")

            query = "SELECT * FROM wall_production_records WHERE operator_id = ? AND is_corrected = 0"
            params: List[Any] = [clean_op]
            if previous_quantity is not None:
                query += " AND quantity = ?"
                params.append(int(previous_quantity))
            query += " ORDER BY production_time DESC LIMIT 1"
            target_row = conn.execute(query, params).fetchone()
        else:
            raise ValueError("Either record_id or operator_id must be provided to correct a record.")

        if not target_row:
            prev_desc = f" with quantity {previous_quantity}" if previous_quantity else ""
            raise ValueError(f"No active wall production record{prev_desc} found to correct.")

        orig = dict(target_row)
        clean_op = orig["operator_id"]
        op = get_operator(clean_op, db_path=db_path)
        now = time.time()
        
        # Resolve corrected_by to a valid operator_id to satisfy foreign key constraints
        valid_c_by = clean_op
        if corrected_by:
            c_op = get_operator(corrected_by, db_path=db_path)
            if c_op:
                valid_c_by = c_op["operator_id"]
            else:
                matches = find_matching_operators(corrected_by, db_path=db_path)
                if matches:
                    valid_c_by = matches[0]["operator_id"]

        corr_notes = f"Corrected from {orig['quantity']} to {new_qty} walls. Reason: {reason}"

        conn.execute(
            """
            UPDATE wall_production_records
            SET is_corrected = 1, correction_notes = ?, corrected_by = ?, corrected_at = ?
            WHERE record_id = ?
            """,
            (corr_notes, valid_c_by, now, orig["record_id"]),
        )

        new_rec_id = generate_wall_record_id(orig["production_date"])
        conn.execute(
            """
            INSERT INTO wall_production_records (
                record_id, operator_id, quantity, production_date, production_time,
                job_id, shift_id, notes, input_source, created_at, request_id,
                is_corrected, correction_notes, corrected_by, corrected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?)
            """,
            (
                new_rec_id, clean_op, new_qty, orig["production_date"], orig["production_time"],
                orig["job_id"], orig["shift_id"], f"Correction of {orig['record_id']}: {reason}",
                "correction", now, f"Replaces {orig['record_id']} ({orig['quantity']} -> {new_qty})",
                valid_c_by, now,
            ),
        )

        try:
            conn.execute(
                """
                INSERT INTO event_corrections (
                    entity_type, entity_id, field_name, old_value, new_value,
                    corrected_by, corrected_at, source, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "wall_production_records", orig["record_id"], "quantity",
                    json.dumps({"quantity": orig["quantity"]}),
                    json.dumps({"quantity": new_qty}),
                    valid_c_by, now, "chat/voice", reason,
                ),
            )
        except Exception as e:
            log.warning("Could not insert event_corrections audit record: %s", e)

        updated_row = conn.execute(
            "SELECT * FROM wall_production_records WHERE record_id = ?",
            (new_rec_id,)
        ).fetchone()

    log.info("Corrected wall production record %s (%d -> %d) for %s", orig["record_id"], orig["quantity"], new_qty, op["name"] if op else clean_op)
    enriched = _enrich_wall_production_record(dict(updated_row), op)
    return {
        "status": "corrected",
        "success": True,
        "original_record_id": orig["record_id"],
        "original_quantity": orig["quantity"],
        "original_record": orig,
        "replacement_record": enriched,
        "corrected_record": enriched,
        "operator_id": clean_op,
        "operator_name": op["name"] if op else clean_op,
        "old_quantity": orig["quantity"],
        "new_quantity": new_qty,
        "date": orig["production_date"],
    }


def list_wall_production_records(
    operator_id: Optional[str] = None,
    date_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    job_id: Optional[Any] = None,
    shift_id: Optional[int] = None,
    include_corrected: bool = False,
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists wall production records with operator metadata."""
    conditions = []
    params: List[Any] = []
    if not include_corrected:
        conditions.append("is_corrected = 0")
    if operator_id:
        conditions.append("operator_id = ?")
        params.append(operator_id.strip())
    if date_filter:
        conditions.append("production_date = ?")
        params.append(date_filter.strip())
    if date_from:
        conditions.append("production_date >= ?")
        params.append(date_from.strip())
    if date_to:
        conditions.append("production_date <= ?")
        params.append(date_to.strip())
    if job_id is not None:
        conditions.append("job_id = ?")
        params.append(job_id)
    if shift_id is not None:
        conditions.append("shift_id = ?")
        params.append(shift_id)

    where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM wall_production_records{where_clause} ORDER BY production_time DESC LIMIT ?",
            params + [limit],
        ).fetchall()

    return [_enrich_wall_production_record(dict(r)) for r in rows]


