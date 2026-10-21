

"""
SQLite persistence for machine state, print jobs, layer measurements and events.

Design notes:
  * Derived values are never stored as trusted duplicates. `remaining_length_mm`
    is a SQLite GENERATED column computed as max(total - completed, 0), so it
    cannot drift out of sync with the values it depends on.
  * Writes are small and frequent (the vision loop updates machine_state every
    analysed frame), so WAL journaling is enabled for concurrent read access
    from the assistant while the camera loop writes.
  * Every table carries a timestamp; the assistant only trusts recent state.
"""

import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from agent import config
from agent.logging_setup import get_logger

log = get_logger("DB")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS print_jobs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name            TEXT    NOT NULL,
    material            TEXT,
    total_length_mm     REAL    NOT NULL DEFAULT 0,
    completed_length_mm REAL    NOT NULL DEFAULT 0,
    -- Derived, never stored independently: cannot drift from its inputs.
    remaining_length_mm REAL GENERATED ALWAYS AS
        (MAX(total_length_mm - completed_length_mm, 0)) VIRTUAL,
    total_layers        INTEGER,
    current_layer       INTEGER,
    status              TEXT    NOT NULL DEFAULT 'running',
    started_at          REAL    NOT NULL,
    completed_at        REAL
);

CREATE TABLE IF NOT EXISTS machine_state (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id              INTEGER REFERENCES print_jobs(id),
    printing            INTEGER,
    status              TEXT,
    current_layer       INTEGER,
    layer_width_mm      REAL,
    layer_width_px      REAL,
    completed_length_mm REAL,
    motion_score        REAL,
    confidence          REAL,
    camera_connected    INTEGER,
    timestamp           REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_machine_state_ts ON machine_state(timestamp DESC);

CREATE TABLE IF NOT EXISTS layer_measurements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       INTEGER REFERENCES print_jobs(id),
    layer_number INTEGER,
    width_px     REAL,
    width_mm     REAL,
    confidence   REAL,
    timestamp    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_layer_ts ON layer_measurements(timestamp DESC);

CREATE TABLE IF NOT EXISTS machine_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     INTEGER REFERENCES print_jobs(id),
    event_type TEXT NOT NULL,
    message    TEXT,
    timestamp  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON machine_events(timestamp DESC);

-- =======================================================================
-- FINAL VALIDATED PHASE 2A SCHEMA: Shift Management & Production Tracking
-- =======================================================================

CREATE TABLE IF NOT EXISTS operators (
    operator_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    employee_code TEXT UNIQUE,
    role          TEXT NOT NULL DEFAULT 'operator',
    active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS shifts (
    shift_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_id   TEXT NOT NULL REFERENCES operators(operator_id),
    line_id       TEXT NOT NULL DEFAULT 'line_1',
    shift_date    TEXT NOT NULL,
    shift_type    TEXT NOT NULL CHECK (shift_type IN ('Morning', 'Afternoon', 'Night')),
    start_time    REAL NOT NULL,
    end_time      REAL,
    status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closed', 'flagged', 'abandoned')),
    target_walls  INTEGER NOT NULL DEFAULT 0,
    summary_notes TEXT,
    created_at    REAL NOT NULL
);

CREATE VIEW IF NOT EXISTS print_jobs_view AS
SELECT 
    id AS job_id,
    id,
    job_name,
    material,
    started_at AS start_time,
    completed_at AS end_time,
    status,
    total_layers,
    current_layer,
    operator AS operator_id,
    -- duration calculations
    (IFNULL(completed_at, strftime('%s','now')) - started_at) / 3600.0 AS printing_duration_hours,
    9.0 AS standard_printing_hours,
    MIN((IFNULL(completed_at, strftime('%s','now')) - started_at) / 3600.0, 9.0) AS regular_printing_hours,
    MAX((IFNULL(completed_at, strftime('%s','now')) - started_at) / 3600.0 - 9.0, 0.0) AS overtime_hours,
    CASE 
        WHEN ((IFNULL(completed_at, strftime('%s','now')) - started_at) / 3600.0) > 9.0 THEN 'Overtime'
        ELSE 'Normal'
    END AS overtime_status
FROM print_jobs;
CREATE UNIQUE INDEX IF NOT EXISTS idx_shifts_single_active_line ON shifts(line_id) WHERE status='active';
CREATE INDEX IF NOT EXISTS idx_shifts_date ON shifts(shift_date, shift_type);
CREATE INDEX IF NOT EXISTS idx_shifts_operator ON shifts(operator_id, start_time DESC);

CREATE TABLE IF NOT EXISTS production_records (
    production_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    shift_id           INTEGER NOT NULL REFERENCES shifts(shift_id),
    job_id             INTEGER REFERENCES print_jobs(id),
    wall_number        INTEGER NOT NULL,
    print_file         TEXT,
    wall_height        REAL NOT NULL DEFAULT 0.0 CHECK (wall_height >= 0.0),
    segment_layers     INTEGER NOT NULL DEFAULT 0 CHECK (segment_layers >= 0),
    layers_completed   INTEGER NOT NULL DEFAULT 0 CHECK (layers_completed >= 0),
    start_time         REAL NOT NULL,
    end_time           REAL,
    status             TEXT NOT NULL DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'aborted', 'defect')),
    is_continuation_of INTEGER REFERENCES production_records(production_id),
    notes              TEXT,
    created_at         REAL NOT NULL,
    UNIQUE (shift_id, wall_number)
);
CREATE INDEX IF NOT EXISTS idx_production_shift ON production_records(shift_id, wall_number);
CREATE INDEX IF NOT EXISTS idx_production_start ON production_records(start_time, status);
CREATE INDEX IF NOT EXISTS idx_production_continuation ON production_records(is_continuation_of) WHERE is_continuation_of IS NOT NULL;

CREATE TABLE IF NOT EXISTS shift_issues (
    issue_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    shift_id           INTEGER NOT NULL REFERENCES shifts(shift_id),
    wall_number        INTEGER,
    timestamp          REAL NOT NULL,
    issue_type         TEXT NOT NULL,
    description        TEXT NOT NULL,
    cause              TEXT,
    action_taken       TEXT,
    result             TEXT,
    severity           TEXT NOT NULL DEFAULT 'warning' CHECK (severity IN ('info', 'warning', 'critical')),
    downtime_minutes   REAL NOT NULL DEFAULT 0.0 CHECK (downtime_minutes >= 0.0),
    rag_recommendation TEXT,
    rag_verified       INTEGER DEFAULT 0,
    rag_source         TEXT,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_issues_type_time ON shift_issues(issue_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_issues_shift ON shift_issues(shift_id);

CREATE TABLE IF NOT EXISTS handovers (
    handover_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    from_shift_id  INTEGER NOT NULL REFERENCES shifts(shift_id),
    to_shift_id    INTEGER REFERENCES shifts(shift_id),
    line_id        TEXT NOT NULL DEFAULT 'line_1',
    handover_time  REAL NOT NULL,
    completed_work TEXT,
    pending_work   TEXT,
    open_issues    TEXT,
    notes          TEXT,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_handovers_line_match ON handovers(line_id, handover_time DESC) WHERE to_shift_id IS NULL;

CREATE TABLE IF NOT EXISTS event_corrections (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type   TEXT NOT NULL,
    entity_id     INTEGER NOT NULL,
    field_name    TEXT NOT NULL,
    old_value     TEXT,
    new_value     TEXT NOT NULL,
    corrected_by  TEXT NOT NULL REFERENCES operators(operator_id),
    corrected_at  REAL NOT NULL,
    source        TEXT NOT NULL DEFAULT 'manual',
    reason        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_entity ON event_corrections(entity_type, entity_id);

CREATE TABLE IF NOT EXISTS attendance_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_id     TEXT REFERENCES operators(operator_id),
    employee_name   TEXT NOT NULL,
    attendance_date TEXT NOT NULL,
    shift_start     REAL NOT NULL,
    shift_end       REAL,
    status          TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'COMPLETED', 'CANCELLED')),
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attendance_operator ON attendance_records(operator_id, status);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_records(attendance_date);

CREATE TABLE IF NOT EXISTS incident_images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        TEXT UNIQUE NOT NULL,
    file_path       TEXT NOT NULL,
    original_name   TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    mime_type       TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    width           INTEGER,
    height          INTEGER,
    created_at      REAL NOT NULL,
    metadata        TEXT
);
CREATE INDEX IF NOT EXISTS idx_incident_images_hash ON incident_images(content_hash);
CREATE INDEX IF NOT EXISTS idx_incident_images_id ON incident_images(image_id);

CREATE TABLE IF NOT EXISTS incidents (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id             TEXT UNIQUE NOT NULL,
    created_at              REAL NOT NULL,
    updated_at              REAL NOT NULL,
    session_id              TEXT,
    line_id                 TEXT,
    operator_id             TEXT,
    machine_version         TEXT,
    user_question           TEXT NOT NULL,
    symptom_description     TEXT,
    visual_observations     TEXT,
    component_category      TEXT,
    defect_category         TEXT,
    error_codes             TEXT,
    possible_causes         TEXT,
    confirmed_root_cause    TEXT,
    recommended_action      TEXT,
    resolution_status       TEXT NOT NULL DEFAULT 'unverified' CHECK (resolution_status IN ('unverified', 'preliminary', 'investigating', 'resolved', 'verified', 'pending_analysis')),
    source_attribution      TEXT,
    confidence              REAL DEFAULT 0.0,
    engineering_review_status TEXT DEFAULT 'pending_review' CHECK (engineering_review_status IN ('pending_review', 'reviewed', 'approved', 'rejected')),
    verified_by             TEXT,
    verified_at             REAL,
    primary_image_id        TEXT REFERENCES incident_images(image_id),
    primary_image_hash      TEXT,
    annotated_image_id      TEXT REFERENCES incident_images(image_id),
    reference_image_id      TEXT REFERENCES incident_images(image_id),
    searchable_text         TEXT
);
CREATE INDEX IF NOT EXISTS idx_incidents_defect ON incidents(defect_category);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(resolution_status);
CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_img_id ON incidents(primary_image_id);

CREATE TABLE IF NOT EXISTS component_references (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_id            TEXT UNIQUE NOT NULL,
    component_name          TEXT NOT NULL,
    display_title           TEXT NOT NULL,
    image_id                TEXT NOT NULL REFERENCES incident_images(image_id),
    machine_version         TEXT NOT NULL DEFAULT 'Kelvin6K-V2',
    source_type             TEXT NOT NULL DEFAULT 'oem_reference',
    source_id               TEXT NOT NULL,
    verification_status     TEXT NOT NULL DEFAULT 'verified',
    caption                 TEXT NOT NULL,
    relevance_explanation   TEXT NOT NULL,
    keywords                TEXT,
    image_category          TEXT NOT NULL DEFAULT 'component_reference',
    limitations             TEXT,
    created_at              REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comp_ref_name ON component_references(component_name);
CREATE INDEX IF NOT EXISTS idx_comp_ref_status ON component_references(verification_status);
CREATE INDEX IF NOT EXISTS idx_comp_ref_category ON component_references(image_category);

CREATE TABLE IF NOT EXISTS wall_production_records (
    record_id        TEXT PRIMARY KEY,
    operator_id      TEXT NOT NULL REFERENCES operators(operator_id),
    quantity         INTEGER NOT NULL CHECK (quantity > 0),
    production_date  TEXT NOT NULL,
    production_time  REAL NOT NULL,
    job_id           INTEGER REFERENCES print_jobs(id),
    shift_id         INTEGER REFERENCES shifts(shift_id),
    notes            TEXT,
    input_source     TEXT NOT NULL DEFAULT 'chat',
    created_at       REAL NOT NULL,
    request_id       TEXT UNIQUE,
    is_corrected     INTEGER NOT NULL DEFAULT 0 CHECK (is_corrected IN (0, 1)),
    correction_notes TEXT,
    corrected_by     TEXT REFERENCES operators(operator_id),
    corrected_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_wall_prod_op_date ON wall_production_records(operator_id, production_date);
CREATE INDEX IF NOT EXISTS idx_wall_prod_date ON wall_production_records(production_date);
CREATE INDEX IF NOT EXISTS idx_wall_prod_req_id ON wall_production_records(request_id) WHERE request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wall_prod_job ON wall_production_records(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wall_prod_shift ON wall_production_records(shift_id) WHERE shift_id IS NOT NULL;
"""

_init_lock = threading.Lock()
_initialised: set = set()


@contextmanager
def connect(db_path: Optional[str] = None):
    """Connection with row access by name. Commits on success, rolls back on error."""
    path = db_path or config.DATABASE_PATH
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added after the first release. Applied with ALTER TABLE so existing
# databases upgrade in place instead of needing a rebuild. All are optional and
# operator-supplied -- when unset the dashboard reports UNAVAILABLE rather than
# inventing a value.
_MIGRATIONS = [
    ("print_jobs", "deadline_at", "REAL"),
    ("print_jobs", "operator", "TEXT"),
    ("print_jobs", "shift", "TEXT"),
    ("print_jobs", "material_total_kg", "REAL"),
    ("print_jobs", "material_used_kg", "REAL"),
    ("print_jobs", "wall_height_mm", "REAL"),
    ("print_jobs", "wall_width_mm", "REAL"),
    ("print_jobs", "layer_height_mm", "REAL"),
    ("shift_issues", "rag_recommendation", "TEXT"),
    ("shift_issues", "rag_verified", "INTEGER DEFAULT 0"),
    ("shift_issues", "rag_source", "TEXT"),
    ("component_references", "image_category", "TEXT DEFAULT 'component_reference'"),
    ("component_references", "limitations", "TEXT"),
]


def _apply_migrations(conn) -> None:
    for table, column, coltype in _MIGRATIONS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            log.info("Migrated: added %s.%s", table, column)


def init_db(db_path: Optional[str] = None) -> None:
    """Creates tables if absent and applies migrations. Safe to call repeatedly."""
    path = db_path or config.DATABASE_PATH
    with _init_lock:
        with connect(path) as conn:
            conn.executescript(_SCHEMA)
            _apply_migrations(conn)
        if path not in _initialised:
            log.info("Database ready at %s", path)
            _initialised.add(path)


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


# --- print jobs ---------------------------------------------------------

def create_print_job(
    job_name: str,
    total_length_mm: float,
    material: Optional[str] = None,
    total_layers: Optional[int] = None,
    db_path: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO print_jobs (job_name, material, total_length_mm, "
            "completed_length_mm, total_layers, current_layer, status, started_at) "
            "VALUES (?, ?, ?, 0, ?, 0, 'running', ?)",
            (job_name, material, total_length_mm, total_layers, time.time()),
        )
        return int(cur.lastrowid)


def get_active_job(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Most recently started job that is still running."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM print_jobs WHERE status = 'running' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    return _row_to_dict(row)


def get_job(job_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM print_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


def update_job_progress(
    job_id: int,
    completed_length_mm: Optional[float] = None,
    current_layer: Optional[int] = None,
    db_path: Optional[str] = None,
) -> None:
    """Updates only what is supplied. remaining_length_mm recomputes itself."""
    sets, params = [], []
    if completed_length_mm is not None:
        sets.append("completed_length_mm = ?")
        params.append(float(completed_length_mm))
    if current_layer is not None:
        sets.append("current_layer = ?")
        params.append(int(current_layer))
    if not sets:
        return
    params.append(job_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE print_jobs SET {', '.join(sets)} WHERE id = ?", params)


_JOB_DETAIL_COLUMNS = {
    "deadline_at", "operator", "shift", "material_total_kg", "material_used_kg",
    "wall_height_mm", "wall_width_mm", "layer_height_mm", "material", "total_layers",
}


def update_job_details(job_id: int, db_path: Optional[str] = None, **fields) -> None:
    """
    Sets operator-supplied job details (deadline, operator, material totals,
    wall geometry). Unknown keys are ignored rather than trusted.
    """
    sets, params = [], []
    for key, value in fields.items():
        if key in _JOB_DETAIL_COLUMNS and value is not None:
            sets.append(f"{key} = ?")
            params.append(value)
    if not sets:
        return
    params.append(job_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE print_jobs SET {', '.join(sets)} WHERE id = ?", params)


def get_state_history(
    since_seconds: float,
    limit: int = 500,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Machine-state rows newer than `since_seconds` ago, oldest first."""
    cutoff = time.time() - since_seconds
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT timestamp, printing, completed_length_mm, layer_width_mm, "
            "confidence, motion_score, status FROM machine_state "
            "WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_progress_at_or_before(timestamp: float, db_path: Optional[str] = None) -> Optional[float]:
    """
    completed_length_mm as of a past moment, used to measure production over a
    window. Returns None when no reading that old exists (so the UI can say the
    window isn't covered yet instead of implying zero production).
    """
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT completed_length_mm FROM machine_state "
            "WHERE timestamp <= ? AND completed_length_mm IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT 1",
            (timestamp,),
        ).fetchone()
    return None if row is None else row["completed_length_mm"]


def count_events_since(
    since_seconds: float,
    event_type: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    cutoff = time.time() - since_seconds
    query = "SELECT COUNT(*) AS n FROM machine_events WHERE timestamp >= ?"
    params: List[Any] = [cutoff]
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)
    with connect(db_path) as conn:
        return int(conn.execute(query, params).fetchone()["n"])


def complete_job(job_id: int, status: str = "completed", db_path: Optional[str] = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE print_jobs SET status = ?, completed_at = ? WHERE id = ?",
            (status, time.time(), job_id),
        )


# --- machine state ------------------------------------------------------

def record_machine_state(
    printing: Optional[bool] = None,
    status: Optional[str] = None,
    current_layer: Optional[int] = None,
    layer_width_mm: Optional[float] = None,
    layer_width_px: Optional[float] = None,
    completed_length_mm: Optional[float] = None,
    motion_score: Optional[float] = None,
    confidence: Optional[float] = None,
    camera_connected: Optional[bool] = None,
    job_id: Optional[int] = None,
    timestamp: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO machine_state (job_id, printing, status, current_layer, "
            "layer_width_mm, layer_width_px, completed_length_mm, motion_score, "
            "confidence, camera_connected, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                None if printing is None else int(printing),
                status,
                current_layer,
                layer_width_mm,
                layer_width_px,
                completed_length_mm,
                motion_score,
                confidence,
                None if camera_connected is None else int(camera_connected),
                timestamp if timestamp is not None else time.time(),
            ),
        )
        return int(cur.lastrowid)


def get_latest_machine_state(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM machine_state ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    state = _row_to_dict(row)
    if state is not None:
        for key in ("printing", "camera_connected"):
            if state.get(key) is not None:
                state[key] = bool(state[key])
    return state


# --- layer measurements -------------------------------------------------

def record_layer_measurement(
    layer_number: Optional[int],
    width_px: Optional[float] = None,
    width_mm: Optional[float] = None,
    confidence: Optional[float] = None,
    job_id: Optional[int] = None,
    timestamp: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO layer_measurements (job_id, layer_number, width_px, "
            "width_mm, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                job_id,
                layer_number,
                width_px,
                width_mm,
                confidence,
                timestamp if timestamp is not None else time.time(),
            ),
        )
        return int(cur.lastrowid)


def get_recent_layer_measurements(limit: int = 20, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM layer_measurements ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- events -------------------------------------------------------------

def record_event(
    event_type: str,
    message: str = "",
    job_id: Optional[int] = None,
    timestamp: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO machine_events (job_id, event_type, message, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (job_id, event_type, message, timestamp if timestamp is not None else time.time()),
        )
        return int(cur.lastrowid)


def get_recent_events(limit: int = 20, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM machine_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_print_jobs_overtime(operator_id: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Gets print jobs that have overtime from the view."""
    query = "SELECT * FROM print_jobs_view WHERE overtime_hours > 0"
    params = []
    if operator_id:
        query += " AND operator_id = ?"
        params.append(operator_id.strip())
    query += " ORDER BY start_time DESC"
    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]

def get_print_jobs_for_operator(operator_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Gets all print jobs for an operator from the view."""
    query = "SELECT * FROM print_jobs_view WHERE operator_id = ? ORDER BY start_time DESC"
    with connect(db_path) as conn:
        rows = conn.execute(query, (operator_id.strip(),)).fetchall()
    return [dict(r) for r in rows]
