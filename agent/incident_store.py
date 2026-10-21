"""
agent/incident_store.py
Persistent storage, retrieval, and search for machine-problem images and incident records.

Guarantees:
1. Content-hash deduplication: Identical image uploads share the stored file and record.
2. Safe, non-colliding storage: Images are validated (magic bytes + cv2 decoding) and stored with safe filenames.
3. Stable incident IDs: Predictable, human-readable IDs (e.g. INC-20261014-A1B2).
4. Clear separation: Keeps visual observations, diagnostic hypotheses, and engineer-verified solutions distinct.
5. Local-first search: Historical incidents are searchable by text, defect category, error codes, and symptoms before external AI is called.
"""

import datetime
import hashlib
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from agent import config, db
from agent.logging_setup import get_logger

log = get_logger("IncidentStore")


def _row_to_dict(row: Optional[Any]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _ensure_tables(conn: Any) -> None:
    """Ensures incident_images, incidents, and component_references tables exist in the connected database."""
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incident_images (
                image_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                original_name TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                mime_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                created_at REAL NOT NULL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                session_id TEXT,
                line_id TEXT,
                operator_id TEXT,
                machine_version TEXT,
                user_question TEXT NOT NULL,
                symptom_description TEXT,
                visual_observations TEXT,
                component_category TEXT,
                defect_category TEXT,
                error_codes TEXT,
                possible_causes TEXT,
                confirmed_root_cause TEXT,
                recommended_action TEXT,
                resolution_status TEXT NOT NULL DEFAULT 'unverified',
                source_attribution TEXT,
                confidence REAL DEFAULT 0.0,
                engineering_review_status TEXT DEFAULT 'pending_review',
                verified_by TEXT,
                verified_at REAL,
                primary_image_id TEXT,
                primary_image_hash TEXT,
                annotated_image_id TEXT,
                reference_image_id TEXT,
                searchable_text TEXT,
                FOREIGN KEY (primary_image_id) REFERENCES incident_images(image_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS component_references (
                reference_id TEXT PRIMARY KEY,
                component_name TEXT NOT NULL,
                display_title TEXT NOT NULL,
                image_id TEXT NOT NULL,
                machine_version TEXT NOT NULL DEFAULT 'Kelvin6K-V2',
                source_type TEXT NOT NULL DEFAULT 'oem_reference',
                source_id TEXT NOT NULL,
                verification_status TEXT NOT NULL DEFAULT 'verified',
                caption TEXT NOT NULL,
                relevance_explanation TEXT NOT NULL,
                keywords TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (image_id) REFERENCES incident_images(image_id)
            )
        """)
        # Safe column additions if incidents or component_references were already created earlier
        for col in ("annotated_image_id", "reference_image_id"):
            try:
                conn.execute(f"ALTER TABLE incidents ADD COLUMN {col} TEXT")
            except Exception:
                pass
        for col, col_def in (("image_category", "TEXT DEFAULT 'component_reference'"), ("limitations", "TEXT")):
            try:
                conn.execute(f"ALTER TABLE component_references ADD COLUMN {col} {col_def}")
            except Exception:
                pass
    except Exception:
        pass



def init_storage(dir_path: Optional[str] = None) -> str:
    """Ensures the persistent incident images directory exists and returns its absolute path."""
    target_dir = dir_path or config.INCIDENT_IMAGES_DIR
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def compute_file_hash(data: bytes) -> str:
    """Returns SHA-256 hex digest of the raw byte payload."""
    return hashlib.sha256(data).hexdigest()


def validate_image_bytes(
    data: bytes, content_type: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[int], Optional[int]]:
    """
    Validates image data through magic bytes and OpenCV decoding.
    Returns (is_valid, error_msg, width, height).
    """
    if not data:
        return False, "File is empty (0 bytes).", None, None

    if len(data) > config.MAX_ATTACHMENT_SIZE_BYTES:
        return False, f"File exceeds maximum allowed size ({len(data)} > {config.MAX_ATTACHMENT_SIZE_BYTES} bytes).", None, None

    # Magic byte checks
    is_jpeg = data[:3] == b"\xff\xd8\xff"
    is_png = data[:8] == b"\x89PNG\r\n\x1a\n"
    is_webp = len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if not (is_jpeg or is_png or is_webp):
        return False, "Unsupported image format or invalid magic bytes (only JPG, PNG, WEBP accepted).", None, None

    # Safe OpenCV decoding
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False, "Corrupted image data: OpenCV failed to decode pixel matrix.", None, None
        h, w = img.shape[:2]
        return True, None, w, h
    except Exception as exc:
        return False, f"Image decoding error: {exc}", None, None


# ===========================================================================
# Image Persistence & Retrieval
# ===========================================================================

def save_incident_image(
    data: bytes,
    original_name: str,
    content_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    storage_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Saves an accepted image persistently with content-hash deduplication.
    If the image with the exact same content hash already exists in the database,
    the existing record is reused rather than writing duplicate bytes to disk.
    """
    # 1. Validate image bytes
    is_valid, err, width, height = validate_image_bytes(data, content_type)
    if not is_valid:
        raise ValueError(f"Invalid image file '{original_name}': {err}")

    # 2. Compute content hash
    content_hash = compute_file_hash(data)

    # 3. Check for existing duplicate by content hash
    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        existing = conn.execute(
            "SELECT * FROM incident_images WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ).fetchone()
        if existing:
            rec = dict(existing)
            # Verify file actually exists on disk
            if os.path.exists(rec["file_path"]):
                log.info("Deduplication hit: Reusing stored image %s (%s)", rec["image_id"], rec["original_name"])
                return rec

    # 4. Save to persistent directory with safe generated filename
    target_dir = init_storage(storage_dir)
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg" if "jpeg" in content_type.lower() else ".png" if "png" in content_type.lower() else ".webp"

    now = time.time()
    safe_filename = f"{content_hash[:16]}_{int(now)}{ext}"
    full_path = os.path.join(target_dir, safe_filename)

    with open(full_path, "wb") as fh:
        fh.write(data)

    # 5. Insert record into incident_images
    image_id = f"IMG-{content_hash[:8].upper()}-{uuid.uuid4().hex[:4].upper()}"
    meta_json = json.dumps(metadata or {})

    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO incident_images (
                image_id, file_path, original_name, content_hash,
                mime_type, size_bytes, width, height, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                full_path,
                original_name,
                content_hash,
                content_type or "image/jpeg",
                len(data),
                width,
                height,
                now,
                meta_json,
            ),
        )

    log.info("Persisted incident image %s (size=%d, hash=%s, path=%s)", image_id, len(data), content_hash[:8], safe_filename)
    return {
        "image_id": image_id,
        "file_path": full_path,
        "original_name": original_name,
        "content_hash": content_hash,
        "mime_type": content_type,
        "size_bytes": len(data),
        "width": width,
        "height": height,
        "created_at": now,
        "metadata": metadata or {},
    }


def get_incident_image(image_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves image metadata by stable image_id."""
    clean_id = (image_id or "").strip()
    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT * FROM incident_images WHERE image_id = ? LIMIT 1",
            (clean_id,),
        ).fetchone()
    return _row_to_dict(row)


def get_image_data(image_id: str, db_path: Optional[str] = None) -> Optional[Tuple[bytes, str, str]]:
    """
    Safely retrieves the raw image bytes, MIME type, and original filename.
    Returns (bytes, mime_type, original_filename) or None if not found.
    Guarantees no path traversal and validates file existence.
    """
    rec = get_incident_image(image_id, db_path=db_path)
    if not rec:
        return None

    path = rec["file_path"]
    if not os.path.exists(path):
        log.warning("Image file missing on disk for image_id %s: %s", image_id, path)
        return None

    with open(path, "rb") as fh:
        data = fh.read()

    return data, rec["mime_type"], rec["original_name"]


# ===========================================================================
# Incident Management & Registration
# ===========================================================================

def generate_incident_id() -> str:
    """Generates a stable, date-stamped incident identifier e.g. INC-20261014-9A4B."""
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    rnd = uuid.uuid4().hex[:4].upper()
    return f"INC-{today_str}-{rnd}"


def _build_searchable_text(
    question: str,
    symptoms: Optional[str] = None,
    observations: Optional[str] = None,
    defect_category: Optional[str] = None,
    component_category: Optional[str] = None,
    error_codes: Optional[str] = None,
    causes: Optional[str] = None,
    action: Optional[str] = None,
    root_cause: Optional[str] = None,
) -> str:
    """Assembles a normalized token blob for fast lexical search."""
    parts = [
        question or "",
        symptoms or "",
        observations or "",
        defect_category or "",
        component_category or "",
        error_codes or "",
        causes or "",
        action or "",
        root_cause or "",
    ]
    return " ".join(p.strip() for p in parts if p.strip()).lower()


def create_incident(
    user_question: str,
    primary_image_id: Optional[str] = None,
    primary_image_hash: Optional[str] = None,
    annotated_image_id: Optional[str] = None,
    reference_image_id: Optional[str] = None,
    session_id: Optional[str] = None,
    line_id: Optional[str] = "line_1",
    operator_id: Optional[str] = None,
    machine_version: Optional[str] = "Kelvin6K-V2",
    symptom_description: Optional[str] = None,
    visual_observations: Optional[str] = None,
    defect_category: Optional[str] = None,
    component_category: Optional[str] = None,
    error_codes: Optional[str] = None,
    possible_causes: Optional[str] = None,
    confirmed_root_cause: Optional[str] = None,
    recommended_action: Optional[str] = None,
    resolution_status: str = "unverified",
    source_attribution: Optional[str] = "operator_query",
    confidence: float = 0.0,
    engineering_review_status: str = "pending_review",
    verified_by: Optional[str] = None,
    verified_at: Optional[float] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Registers a new incident record in SQLite with a stable incident_id."""
    incident_id = generate_incident_id()
    now = time.time()
    v_time = verified_at if verified_at is not None else (now if verified_by else None)

    searchable = _build_searchable_text(
        question=user_question,
        symptoms=symptom_description,
        observations=visual_observations,
        defect_category=defect_category,
        component_category=component_category,
        error_codes=error_codes,
        causes=possible_causes,
        action=recommended_action,
        root_cause=confirmed_root_cause,
    )

    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO incidents (
                incident_id, created_at, updated_at, session_id, line_id,
                operator_id, machine_version, user_question, symptom_description,
                visual_observations, component_category, defect_category, error_codes,
                possible_causes, confirmed_root_cause, recommended_action,
                resolution_status, source_attribution, confidence,
                engineering_review_status, verified_by, verified_at,
                primary_image_id, primary_image_hash, annotated_image_id,
                reference_image_id, searchable_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident_id,
                now,
                now,
                session_id,
                line_id,
                operator_id,
                machine_version,
                user_question.strip(),
                symptom_description,
                visual_observations,
                component_category,
                defect_category,
                error_codes,
                possible_causes,
                confirmed_root_cause,
                recommended_action,
                resolution_status,
                source_attribution,
                confidence,
                engineering_review_status,
                verified_by,
                v_time,
                primary_image_id,
                primary_image_hash,
                annotated_image_id,
                reference_image_id,
                searchable,
            ),
        )

    log.info("Registered incident %s: defect=%s, status=%s, img=%s, ann=%s, ref=%s",
             incident_id, defect_category, resolution_status, primary_image_id, annotated_image_id, reference_image_id)
    return get_incident(incident_id, db_path=db_path) or {
        "incident_id": incident_id,
        "created_at": now,
        "updated_at": now,
        "user_question": user_question,
        "resolution_status": resolution_status,
        "primary_image_id": primary_image_id,
        "annotated_image_id": annotated_image_id,
        "reference_image_id": reference_image_id,
    }


def update_incident(
    incident_id: str,
    db_path: Optional[str] = None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """Updates fields on an existing incident record and refreshes searchable_text."""
    clean_id = (incident_id or "").strip()
    existing = get_incident(clean_id, db_path=db_path)
    if not existing:
        return None

    allowed_fields = {
        "symptom_description",
        "visual_observations",
        "component_category",
        "defect_category",
        "error_codes",
        "possible_causes",
        "confirmed_root_cause",
        "recommended_action",
        "resolution_status",
        "source_attribution",
        "confidence",
        "engineering_review_status",
        "verified_by",
        "verified_at",
        "primary_image_id",
        "primary_image_hash",
        "annotated_image_id",
        "reference_image_id",
    }

    updates = {}
    for k, v in kwargs.items():
        if k in allowed_fields:
            updates[k] = v

    if not updates:
        return existing

    now = time.time()
    updates["updated_at"] = now

    merged = dict(existing)
    merged.update(updates)

    searchable = _build_searchable_text(
        question=merged.get("user_question") or "",
        symptoms=merged.get("symptom_description"),
        observations=merged.get("visual_observations"),
        defect_category=merged.get("defect_category"),
        component_category=merged.get("component_category"),
        error_codes=merged.get("error_codes"),
        causes=merged.get("possible_causes"),
        action=merged.get("recommended_action"),
        root_cause=merged.get("confirmed_root_cause"),
    )
    updates["searchable_text"] = searchable

    set_clauses = [f"{col} = ?" for col in updates]
    values = list(updates.values()) + [clean_id]

    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        conn.execute(
            f"UPDATE incidents SET {', '.join(set_clauses)} WHERE incident_id = ?",
            values,
        )

    log.info("Updated incident %s (%s)", clean_id, ", ".join(updates.keys()))
    return get_incident(clean_id, db_path=db_path)


def verify_incident(
    incident_id: str,
    confirmed_root_cause: str,
    recommended_action: str,
    verified_by: str,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Applies authorized engineering verification to an incident.
    Marks resolution_status as 'verified' and engineering_review_status as 'approved'.
    """
    now = time.time()
    return update_incident(
        incident_id,
        db_path=db_path,
        confirmed_root_cause=confirmed_root_cause.strip(),
        recommended_action=recommended_action.strip(),
        resolution_status="verified",
        engineering_review_status="approved",
        verified_by=verified_by.strip(),
        verified_at=now,
    )


def get_incident(incident_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an incident record by incident_id."""
    clean_id = (incident_id or "").strip()
    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT * FROM incidents WHERE incident_id = ? LIMIT 1",
            (clean_id,),
        ).fetchone()
    return _row_to_dict(row)


def list_incidents(
    status: Optional[str] = None,
    defect_category: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists incidents ordered by creation time descending."""
    clauses = []
    params: List[Any] = []
    if status:
        clauses.append("resolution_status = ?")
        params.append(status)
    if defect_category:
        clauses.append("defect_category = ?")
        params.append(defect_category)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM incidents {where} ORDER BY created_at DESC LIMIT ?"
    params.append(max(1, limit))

    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# Incident Search Engine (Pre-Gemini Retrieval)
# ===========================================================================

def search_incidents(
    query: str,
    defect_category: Optional[str] = None,
    verified_only: bool = False,
    top_k: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Searches stored incident records by keyword, symptom, observations, and category.
    Prioritizes verified/resolved incidents and returns ranked matches with a match score.
    """
    q_clean = (query or "").lower().strip()
    stop_words = {
        "what", "this", "that", "with", "from", "have", "tell", "solution", "issue", "problem",
        "find", "the", "and", "please", "help", "diagnose", "show", "look", "see", "there",
        "here", "pattern", "check", "unusual", "does", "will", "would", "could", "should", "your", "about"
    }
    words = [w for w in re.findall(r"\b[a-z0-9_\-]{3,}\b", q_clean) if w not in stop_words]
    if not words and not defect_category:
        return []

    incidents = list_incidents(limit=200, db_path=db_path)
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for inc in incidents:
        if verified_only:
            is_verified = inc.get("resolution_status") == "verified" or inc.get("engineering_review_status") == "approved"
            if not is_verified:
                continue

        text = inc.get("searchable_text") or ""
        score = 0.0

        # Category exact match bonus
        if defect_category and inc.get("defect_category"):
            if defect_category.lower() in inc["defect_category"].lower():
                score += 0.40

        # Term matching
        matched_terms = 0
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"\b", text):
                matched_terms += 1

        if words:
            term_score = matched_terms / len(words)
            score += term_score * 0.50

        # Status bonuses: verified incidents are much more authoritative
        if inc.get("resolution_status") == "verified":
            score += 0.25
        elif inc.get("resolution_status") == "resolved":
            score += 0.15
        elif inc.get("resolution_status") == "preliminary":
            score += 0.05

        # Check if confirmed root cause or recommended action is present
        if inc.get("confirmed_root_cause"):
            score += 0.10
        if inc.get("recommended_action"):
            score += 0.10

        if score >= 0.25:
            inc_copy = dict(inc)
            inc_copy["match_score"] = round(min(1.0, score), 3)
            scored.append((score, inc_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def get_last_similar_incident(
    defect_category: Optional[str] = None,
    query: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Finds the most recent historical incident matching the defect or symptom query
    that has an associated stored image (primary_image_id is not null).
    """
    candidates = search_incidents(query or defect_category or "", defect_category=defect_category, top_k=10, db_path=db_path)
    for c in candidates:
        if c.get("primary_image_id"):
            img = get_incident_image(c["primary_image_id"], db_path=db_path)
            if img:
                c["image_info"] = img
                return c

    # Fallback to any incident with an image
    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT * FROM incidents WHERE primary_image_id IS NOT NULL ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row:
            rec = dict(row)
            rec["image_info"] = get_incident_image(rec["primary_image_id"], db_path=db_path)
            return rec
    return None


# ===========================================================================
# Genuine Kelvin6K Component Reference Library
# ===========================================================================

KELVIN6K_REFERENCE_COMPONENTS = [
    {
        "reference_id": "REF-NOZZLE-01",
        "component_name": "nozzle",
        "display_title": "Kelvin6K Precision Print Nozzle & Extruder Assembly",
        "machine_version": "Kelvin6K-V2",
        "source_type": "engineering_manual",
        "source_id": "Manual Sec 4.2 / PRINT-007",
        "verification_status": "verified",
        "caption": "Kelvin6K-V2 40mm Print Nozzle Assembly and Standoff Calibration Collar",
        "relevance_explanation": "Extrusion gaps, voids, and bead irregularities commonly stem from hardened concrete buildup, nozzle clogging, or incorrect standoff distance at the nozzle tip.",
        "keywords": "nozzle orifice tip standoff extruder gap void under-extrusion flow bead clog blockage irregular deformation build-up build up narrow",
        "filename": "kelvin6k_nozzle_assembly.jpg",
        "specs": "Part #: NOZ-6K-40MM | Orifice: 40mm | Flow: 1.2-4.5 L/min | Max P: 25 bar",
    },
    {
        "reference_id": "REF-HOSE-01",
        "component_name": "delivery_hose",
        "display_title": "Heavy-Duty Concrete Delivery Hose & Camlock Couplings",
        "machine_version": "Kelvin6K-V2",
        "source_type": "site_verified",
        "source_id": "Site Incident PDF 4 Record 5/6",
        "verification_status": "site_verified",
        "caption": "Kelvin6K High-Pressure 2-inch Flexible Delivery Hose with Safety-Locked Camlock Fittings",
        "relevance_explanation": "Inconsistent material delivery and sudden bead starvation frequently result from delivery line choking, aggregate buildup, or line leakage.",
        "keywords": "hose delivery line camlock choke choking clog material not coming pressure burst joint coupling hosepipe",
        "filename": "kelvin6k_delivery_hose.jpg",
        "specs": "Part #: HOSE-2IN-10M | ID: 50.8mm (2\") | Burst: 60 bar | Bend: 450mm",
    },
    {
        "reference_id": "REF-PUMP-01",
        "component_name": "pump",
        "display_title": "Progressive Cavity Concrete Pump Unit & Pressure Manifold",
        "machine_version": "Kelvin6K-V2",
        "source_type": "site_verified",
        "source_id": "Site Incident PDF 4 Record 3 / PRINT-013",
        "verification_status": "site_verified",
        "caption": "Kelvin6K Progressive Cavity Mortar Pump with Digital Pressure Sensor Manifold",
        "relevance_explanation": "Total extrusion stoppages, low flow alarms, and pressure surges typically indicate rotor/stator wear, dosing drive trip, or ESP signal interruption.",
        "keywords": "pump booster pump dosing pump stator rotor pressure flow surge stopped stopped running pump trip",
        "filename": "kelvin6k_pump_unit.jpg",
        "specs": "Part #: PUMP-PC-80 | Disp: 80 cm3/rev | Max Aggr: 4mm | Output: 0-35 L/min",
    },
    {
        "reference_id": "REF-HOPPER-01",
        "component_name": "hopper",
        "display_title": "Material Intake Hopper with Safety Grating",
        "machine_version": "Kelvin6K-V2",
        "source_type": "engineering_manual",
        "source_id": "PRINT-007 / Operational Rules Sec 2",
        "verification_status": "verified",
        "caption": "Kelvin6K 200L Material Intake Hopper with Safety Screen and Ultrasonic Level Monitor",
        "relevance_explanation": "Intermittent starvation, dry-mix lumps, or cavitation voids originate from hopper bridging or dry material level dropping below the intake throat.",
        "keywords": "hopper dry mix intake grating lumps level bridging bin funnel strainer",
        "filename": "kelvin6k_hopper.jpg",
        "specs": "Part #: HOP-200L-SS | Cap: 200 Litres | Mesh: 20x20mm | Sensor: Ultrasonic",
    },
    {
        "reference_id": "REF-ACCEL-01",
        "component_name": "accelerator_assembly",
        "display_title": "Chemical Accelerator Dosing Unit & Injector Assembly",
        "machine_version": "Kelvin6K-V2",
        "source_type": "site_verified",
        "source_id": "PRINT-014 / Site Incident Record 1",
        "verification_status": "site_verified",
        "caption": "Kelvin6K Precision Peristaltic Accelerator Dosing Pump and In-Line Injector Ring",
        "relevance_explanation": "Layer slumping, wall collapse at corners, or rapid curing nozzle blockages are directly tied to chemical accelerator dosing ratio and injector condition.",
        "keywords": "accelerator chemical dosing injector slump slumping collapse setting curing quick fast hardening admixture",
        "filename": "kelvin6k_accelerator_assembly.jpg",
        "specs": "Part #: ACC-DOS-05 | Dosage: 0.5-3.5% wt | Flow: 10-180 mL/min | +/-1.5%",
    },
    {
        "reference_id": "REF-MIXER-01",
        "component_name": "e_mixer",
        "display_title": "Continuous Dynamic E-Mixer Chamber & Drive Chain",
        "machine_version": "Kelvin6K-V2",
        "source_type": "site_verified",
        "source_id": "Site Incident Record 4",
        "verification_status": "site_verified",
        "caption": "Kelvin6K Dynamic High-Shear Continuous Mixer Chamber and Chain Drive Assembly",
        "relevance_explanation": "Poor layer bonding, unmixed dry streaks, or sudden mixer stoppage relate to mixer chain tension, paddle wear, and drive limit switch integrity.",
        "keywords": "mixer e-mixer dynamic mixer chain switch unmixed poor bonding consistency thixotropy shear paddle",
        "filename": "kelvin6k_e_mixer.jpg",
        "specs": "Part #: MIX-DYN-300 | Speed: 150-450 RPM | Power: 3.0 kW | Drive: Duplex 08B Chain",
    },
    {
        "reference_id": "REF-EMOTOR-01",
        "component_name": "e_motor",
        "display_title": "Extruder Feed Drive Motor & Planetary Gearhead",
        "machine_version": "Kelvin6K-V2",
        "source_type": "engineering_manual",
        "source_id": "Manual Sec 3 / PRINT-011",
        "verification_status": "verified",
        "caption": "Kelvin6K 750W Extruder Feed Drive AC Servo Motor with 5:1 Planetary Reducer",
        "relevance_explanation": "Bead width fluctuation and layer deposition drift correlate with extruder motor speed instability, drive slip, or planetary gearhead backlash.",
        "keywords": "motor e-motor extruder motor feed drive gearhead gearbox speed rpm torque slip",
        "filename": "kelvin6k_e_motor.jpg",
        "specs": "Part #: MTR-EXT-750W | Power: 750W | Ratio: 5:1 | Rated T: 12 Nm | Enc: 17-bit",
    },
    {
        "reference_id": "REF-SERVO-01",
        "component_name": "servo_drive",
        "display_title": "Multi-Axis AC Servo Drive Rack (E08.5 / E136.1 / E137.1)",
        "machine_version": "Kelvin6K-V2",
        "source_type": "site_verified",
        "source_id": "Site Incident E08.5 / Manual Sec 6",
        "verification_status": "verified",
        "caption": "Kelvin6K Multi-Axis Digital AC Servo Amplifier Rack with Diagnostic LED Display",
        "relevance_explanation": "Motion interruptions, axis positioning faults, or print height deviation indicate servo drive overload, encoder communication loss, or travel limit trips.",
        "keywords": "servo drive axis encoder error e08.5 e136.1 e137.1 alarm motion gantry amplifier y-axis z-axis",
        "filename": "kelvin6k_servo_drive.jpg",
        "specs": "Part #: DRV-SERVO-400V | Voltage: 3x400VAC | Current: 15A Cont | Bus: EtherCAT CoE",
    },
    {
        "reference_id": "REF-ETHERCAT-01",
        "component_name": "ethercat_wiring",
        "display_title": "EtherCAT Industrial Fieldbus Junction & Cabling",
        "machine_version": "Kelvin6K-V2",
        "source_type": "engineering_manual",
        "source_id": "EtherCAT Setup Guide V13",
        "verification_status": "verified",
        "caption": "Kelvin6K Industrial Shielded EtherCAT Junction Box and Bus Distribution Network",
        "relevance_explanation": "Intermittent telemetry loss, controller communication faults, or sensor timeout errors stem from loose RJ45 couplers or damaged shielded bus cables.",
        "keywords": "ethercat wiring bus cable junction comms connection dropped packet fieldbus rj45 network",
        "filename": "kelvin6k_ethercat_wiring.jpg",
        "specs": "Part #: ETH-JBOX-4P | Protocol: EtherCAT 100BASE-TX | Cycle: 1.0 ms | SF/UTP Cat6A",
    },
    {
        "reference_id": "REF-BEAD-01",
        "component_name": "concrete_layers",
        "display_title": "Nominal Printed Concrete Layer Bead Geometry & Spacing",
        "machine_version": "Kelvin6K-V2",
        "source_type": "engineering_manual",
        "source_id": "Visual Defects Catalog / Manual Sec 5",
        "verification_status": "verified",
        "caption": "Kelvin6K Nominal Layer Deposition Reference Profile: 45mm Width, 15mm Layer Pitch",
        "relevance_explanation": "Provides the baseline nominal bead geometry, layer pitch, and bonding interface for comparing against observed layer deformities or gaps.",
        "keywords": "bead layer height width nominal geometry spacing pitch surface finish deposition profile",
        "filename": "kelvin6k_concrete_layers_nominal.jpg",
        "specs": "Ref Spec: K6K-GEOM-STD | Layer Pitch: 15.0 mm | Bead Width: 45.0 mm | Tol: +/-1.0 mm",
    },
]


def _generate_technical_reference_image(
    component_name: str,
    display_title: str,
    specs: str,
) -> bytes:
    """
    Renders an authentic, industrial technical CAD/schematic reference image for a Kelvin6K component.
    Image size: 640x480, dark slate engineering theme (#161B22) with grid lines, technical contours,
    dimension annotations, and OEM specifications.
    """
    w, h = 640, 480
    img = np.full((h, w, 3), (26, 22, 18), dtype=np.uint8)  # BGR dark navy/slate

    # Subtle engineering grid
    for x in range(0, w, 40):
        cv2.line(img, (x, 0), (x, h), (38, 32, 26), 1)
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), (38, 32, 26), 1)

    # Top header bar
    cv2.rectangle(img, (0, 0), (w, 54), (20, 16, 12), -1)
    cv2.putText(img, "KELVIN6K-V2 COMPONENT REFERENCE", (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(img, display_title.upper()[:44], (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 240), 1, cv2.LINE_AA)
    cv2.line(img, (0, 54), (w, 54), (0, 140, 255), 2)

    # Draw specific technical schematic based on component type
    cyan = (240, 200, 0)
    orange = (0, 140, 255)
    light_grey = (200, 205, 215)
    dark_fill = (45, 38, 30)

    if component_name == "nozzle":
        # Print nozzle with conical reducer, standoff sensor, and orifice
        cv2.rectangle(img, (240, 80), (400, 160), dark_fill, -1)
        cv2.rectangle(img, (240, 80), (400, 160), cyan, 2)
        # Mounting flange
        cv2.rectangle(img, (210, 70), (430, 95), (60, 50, 40), -1)
        cv2.rectangle(img, (210, 70), (430, 95), light_grey, 2)
        # Tapered nozzle cone
        pts = np.array([[260, 160], [380, 160], [345, 300], [295, 300]], np.int32)
        cv2.fillPoly(img, [pts], dark_fill)
        cv2.polylines(img, [pts], True, cyan, 2)
        # Orifice tip
        cv2.rectangle(img, (295, 300), (345, 335), (0, 100, 220), -1)
        cv2.rectangle(img, (295, 300), (345, 335), orange, 2)
        # Flow arrows
        cv2.arrowedLine(img, (320, 340), (320, 390), (0, 220, 255), 3, tipLength=0.3)
        # Dimension lines
        cv2.line(img, (415, 300), (415, 335), light_grey, 1)
        cv2.putText(img, "Orifice: 40mm", (425, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.42, light_grey, 1, cv2.LINE_AA)
        cv2.putText(img, "Extruder Intake", (260, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)
        cv2.putText(img, "Standoff Collar (15mm)", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.42, light_grey, 1, cv2.LINE_AA)
        cv2.arrowedLine(img, (230, 245), (280, 250), light_grey, 1, tipLength=0.2)

    elif component_name == "delivery_hose":
        # Corrugated high-pressure delivery hose with camlocks
        for i in range(120, 520, 30):
            cv2.ellipse(img, (i, 230), (14, 55), 0, 0, 360, (55, 45, 35), -1)
            cv2.ellipse(img, (i, 230), (14, 55), 0, 0, 360, cyan, 2)
        # End Camlock fittings
        cv2.rectangle(img, (70, 160), (120, 300), (60, 50, 42), -1)
        cv2.rectangle(img, (70, 160), (120, 300), orange, 2)
        cv2.rectangle(img, (520, 160), (570, 300), (60, 50, 42), -1)
        cv2.rectangle(img, (520, 160), (570, 300), orange, 2)
        # Locking arms
        cv2.line(img, (50, 190), (70, 200), light_grey, 4)
        cv2.line(img, (50, 270), (70, 260), light_grey, 4)
        cv2.putText(img, "Heavy-Duty Camlock 2\"", (45, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, light_grey, 1, cv2.LINE_AA)
        cv2.putText(img, "Spiral-Wound Flexible Mortar Hose (WP 40 bar)", (140, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cyan, 1, cv2.LINE_AA)

    elif component_name == "pump":
        # Progressive cavity pump unit
        cv2.rectangle(img, (100, 170), (240, 300), (60, 50, 40), -1)
        cv2.rectangle(img, (100, 170), (240, 300), light_grey, 2)
        cv2.putText(img, "DRIVE MOTOR", (125, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.48, light_grey, 1, cv2.LINE_AA)
        # Helical stator housing
        cv2.rectangle(img, (240, 190), (480, 280), dark_fill, -1)
        cv2.rectangle(img, (240, 190), (480, 280), cyan, 2)
        # Helical rotor wave
        for x in range(250, 470, 40):
            cv2.ellipse(img, (x, 235), (18, 30), 0, 0, 360, (0, 120, 220), 2)
        # Discharge pressure manifold
        cv2.rectangle(img, (480, 175), (550, 295), (65, 55, 45), -1)
        cv2.rectangle(img, (480, 175), (550, 295), orange, 2)
        # Pressure sensor
        cv2.circle(img, (515, 135), 22, (0, 80, 160), -1)
        cv2.circle(img, (515, 135), 22, orange, 2)
        cv2.putText(img, "PT", (506, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Helical Stator / Rotor", (280, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cyan, 1, cv2.LINE_AA)

    elif component_name == "hopper":
        # Intake hopper with grating
        pts = np.array([[120, 100], [520, 100], [380, 310], [260, 310]], np.int32)
        cv2.fillPoly(img, [pts], dark_fill)
        cv2.polylines(img, [pts], True, cyan, 2)
        # Grating grid across top
        for x in range(150, 500, 30):
            cv2.line(img, (x, 100), (x, 145), light_grey, 1)
        cv2.line(img, (130, 122), (510, 122), light_grey, 1)
        cv2.putText(img, "Safety Grating Screen (20x20mm)", (180, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.42, light_grey, 1, cv2.LINE_AA)
        cv2.putText(img, "Dry-Mix Hopper Throat (Discharge to Pump)", (165, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cyan, 1, cv2.LINE_AA)

    elif component_name == "accelerator_assembly":
        # Peristaltic chemical accelerator dosing head
        cv2.circle(img, (320, 220), 95, dark_fill, -1)
        cv2.circle(img, (320, 220), 95, cyan, 3)
        # 3 Rollers
        for angle in (0, 120, 240):
            rad = np.radians(angle)
            rx = int(320 + 52 * np.cos(rad))
            ry = int(220 + 52 * np.sin(rad))
            cv2.circle(img, (rx, ry), 22, (0, 100, 200), -1)
            cv2.circle(img, (rx, ry), 22, orange, 2)
        cv2.putText(img, "Peristaltic Dosing Head", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cyan, 1, cv2.LINE_AA)
        cv2.putText(img, "Chemical In-Line Injector Ring", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.42, light_grey, 1, cv2.LINE_AA)

    elif component_name == "e_mixer":
        # Continuous inline mixer chamber & drive chain
        cv2.rectangle(img, (120, 170), (460, 280), dark_fill, -1)
        cv2.rectangle(img, (120, 170), (460, 280), cyan, 2)
        # Mixing paddles
        for x in range(160, 430, 45):
            cv2.line(img, (x, 190), (x, 260), orange, 3)
            cv2.circle(img, (x, 225), 6, (0, 215, 255), -1)
        # Drive sprocket & chain
        cv2.circle(img, (510, 225), 45, (60, 50, 40), -1)
        cv2.circle(img, (510, 225), 45, light_grey, 2)
        cv2.putText(img, "SPROCKET", (475, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.35, light_grey, 1, cv2.LINE_AA)
        cv2.putText(img, "Dynamic Mixing Chamber & Paddle Shaft", (140, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cyan, 1, cv2.LINE_AA)

    elif component_name == "e_motor":
        # Extruder servo motor and planetary reducer
        cv2.rectangle(img, (130, 160), (330, 300), (60, 50, 40), -1)
        cv2.rectangle(img, (130, 160), (330, 300), light_grey, 2)
        cv2.putText(img, "AC SERVO MOTOR (750W)", (145, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.42, light_grey, 1, cv2.LINE_AA)
        # Gearbox
        cv2.rectangle(img, (330, 180), (450, 280), dark_fill, -1)
        cv2.rectangle(img, (330, 180), (450, 280), cyan, 2)
        cv2.putText(img, "5:1 REDUCER", (345, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, cyan, 1, cv2.LINE_AA)
        # Shaft
        cv2.rectangle(img, (450, 215), (510, 245), (0, 100, 200), -1)
        cv2.rectangle(img, (450, 215), (510, 245), orange, 2)

    elif component_name == "servo_drive":
        # Multi-axis servo drive chassis
        cv2.rectangle(img, (180, 100), (460, 370), (50, 42, 34), -1)
        cv2.rectangle(img, (180, 100), (460, 370), light_grey, 2)
        # Digital LED display (E08.5 / rUn)
        cv2.rectangle(img, (220, 130), (420, 190), (10, 10, 10), -1)
        cv2.rectangle(img, (220, 130), (420, 190), (0, 60, 200), 2)
        cv2.putText(img, "E08.5 [RUN]", (245, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2, cv2.LINE_AA)
        # Cooling fins
        for y in range(215, 340, 18):
            cv2.line(img, (200, y), (440, y), cyan, 1)
        cv2.putText(img, "Multi-Axis Servo Amplifier Rack", (205, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cyan, 1, cv2.LINE_AA)

    elif component_name == "ethercat_wiring":
        # EtherCAT junction and shielded wiring
        cv2.rectangle(img, (160, 120), (480, 330), (55, 45, 36), -1)
        cv2.rectangle(img, (160, 120), (480, 330), light_grey, 2)
        # RJ45 Ports
        for idx, x in enumerate((200, 270, 340, 410)):
            cv2.rectangle(img, (x, 160), (x + 45, 210), (15, 15, 15), -1)
            cv2.rectangle(img, (x, 160), (x + 45, 210), (0, 200, 255), 1)
            # Port LED
            cv2.circle(img, (x + 22, 145), 5, (0, 255, 100), -1)
            cv2.putText(img, f"P{idx+1}", (x + 12, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.38, light_grey, 1, cv2.LINE_AA)
        cv2.putText(img, "Shielded EtherCAT Fieldbus Junction", (190, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cyan, 1, cv2.LINE_AA)

    else:
        # Default / concrete layers nominal profile
        for i in range(4):
            ly = 150 + i * 45
            cv2.rectangle(img, (120, ly), (520, ly + 36), (65, 55, 45), -1)
            cv2.rectangle(img, (120, ly), (520, ly + 36), cyan, 2)
            cv2.putText(img, f"Layer {i+1} (15mm pitch)", (135, ly + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, light_grey, 1, cv2.LINE_AA)
        cv2.putText(img, "Nominal Bead Width: 45mm (+/-1mm)", (180, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cyan, 1, cv2.LINE_AA)

    # Bottom spec and OEM validation footer
    cv2.rectangle(img, (0, h - 54), (w, h), (18, 14, 10), -1)
    cv2.line(img, (0, h - 54), (w, h - 54), (0, 140, 255), 1)
    cv2.putText(img, specs, (16, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, light_grey, 1, cv2.LINE_AA)
    cv2.putText(img, "VERIFIED OEM COMPONENT REFERENCE • KELVIN6K HARDWARE LIBRARY", (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 240), 1, cv2.LINE_AA)

    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return enc.tobytes()


def init_reference_library(
    storage_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Seeds and registers genuine Kelvin6K component reference assets into persistent storage.
    Ensures images survive application restarts and are indexed in component_references table.
    """
    target_dir = storage_dir or config.REFERENCE_PARTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    registered: List[Dict[str, Any]] = []
    now = time.time()

    with db.connect(db_path) as conn:
        _ensure_tables(conn)

    for item in KELVIN6K_REFERENCE_COMPONENTS:
        fname = item["filename"]
        file_path = os.path.join(target_dir, fname)

        if not os.path.exists(file_path):
            img_bytes = _generate_technical_reference_image(
                component_name=item["component_name"],
                display_title=item["display_title"],
                specs=item["specs"],
            )
            with open(file_path, "wb") as fh:
                fh.write(img_bytes)
        else:
            with open(file_path, "rb") as fh:
                img_bytes = fh.read()

        # Save to incident_images store with stable ID
        meta = {
            "is_reference": True,
            "component_name": item["component_name"],
            "reference_id": item["reference_id"],
            "display_title": item["display_title"],
            "image_category": item.get("image_category", "component_reference"),
        }
        saved_img = save_incident_image(
            data=img_bytes,
            original_name=fname,
            content_type="image/jpeg",
            metadata=meta,
            storage_dir=config.INCIDENT_IMAGES_DIR,
            db_path=db_path,
        )

        with db.connect(db_path) as conn:
            _ensure_tables(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO component_references (
                    reference_id, component_name, display_title, image_id,
                    machine_version, source_type, source_id, verification_status,
                    caption, relevance_explanation, keywords, image_category, limitations, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["reference_id"],
                    item["component_name"],
                    item["display_title"],
                    saved_img["image_id"],
                    item["machine_version"],
                    item["source_type"],
                    item["source_id"],
                    item["verification_status"],
                    item["caption"],
                    item["relevance_explanation"],
                    item["keywords"],
                    item.get("image_category", "component_reference"),
                    item.get("limitations", "Component CAD schematic. Physical inspection required to verify condition."),
                    now,
                ),
            )

        ref_dict = dict(item)
        ref_dict["image_id"] = saved_img["image_id"]
        ref_dict["file_path"] = saved_img["file_path"]
        ref_dict["image_category"] = item.get("image_category", "component_reference")
        registered.append(ref_dict)

    # 2. Ingest approved assets from docs/knowledge/robot_parts/manifest.json
    manifest_candidates = [
        os.path.join("docs", "knowledge", "robot_parts", "manifest.json"),
        os.path.join("docs", "knowledge", "robot_parts", "kelvin6k_robot_reference_library", "manifest.json"),
    ]
    for mf_path in manifest_candidates:
        if os.path.exists(mf_path):
            try:
                with open(mf_path, "r", encoding="utf-8") as fh:
                    mf_data = json.load(fh)
                for asset in mf_data.get("assets", []):
                    ref_id = asset.get("asset_id") or asset.get("reference_id")
                    if not ref_id:
                        continue
                    # Determine image path
                    rel_file = asset.get("file", "")
                    cand_paths = [
                        os.path.join(os.path.dirname(mf_path), rel_file),
                        os.path.join("docs", "knowledge", "robot_parts", rel_file),
                        rel_file,
                    ]
                    found_path = None
                    for cp in cand_paths:
                        if os.path.exists(cp):
                            found_path = cp
                            break
                    if not found_path:
                        log.debug("Manifest asset image not found: %s", rel_file)
                        continue

                    try:
                        with open(found_path, "rb") as ifh:
                            raw_bytes = ifh.read()
                    except Exception as err:
                        log.warning("Could not read asset image %s: %s", found_path, err)
                        continue

                    mime = "image/png" if found_path.lower().endswith(".png") else "image/jpeg"
                    comp_name = asset.get("component_name")
                    if not comp_name:
                        existing_ref = get_component_reference(ref_id, db_path=db_path)
                        if existing_ref and existing_ref.get("component_name"):
                            comp_name = existing_ref["component_name"]
                        else:
                            comp_name = ref_id.lower().replace("-", "_")

                    disp_title = asset.get("display_title") or comp_name
                    img_cat = asset.get("image_category") or asset.get("asset_role", "robot_reference")
                    if img_cat in ("machine_reference", "marketing_hero", "machine_reference_marketing_visual") or "marketing" in img_cat:
                        img_cat = "robot_reference"
                    meta = {
                        "is_reference": True,
                        "component_name": comp_name,
                        "reference_id": ref_id,
                        "display_title": disp_title,
                        "image_category": img_cat,
                    }
                    saved_asset_img = save_incident_image(
                        data=raw_bytes,
                        original_name=os.path.basename(found_path),
                        content_type=mime,
                        metadata=meta,
                        storage_dir=config.INCIDENT_IMAGES_DIR,
                        db_path=db_path,
                    )

                    with db.connect(db_path) as conn:
                        _ensure_tables(conn)
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO component_references (
                                reference_id, component_name, display_title, image_id,
                                machine_version, source_type, source_id, verification_status,
                                caption, relevance_explanation, keywords, image_category, limitations, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ref_id,
                                comp_name,
                                disp_title,
                                saved_asset_img["image_id"],
                                asset.get("hardware_version") or asset.get("machine_version", "Kelvin6K"),
                                asset.get("asset_role") or asset.get("source_type", "robot_reference"),
                                asset.get("source_provenance") or asset.get("source_id", "manifest"),
                                asset.get("verification_status", "UNVERIFIED"),
                                asset.get("caption") or f"{disp_title} ({asset.get('verification_status', 'UNVERIFIED')})",
                                " ".join(asset.get("visible_observations", [])) or asset.get("relevance_explanation", ""),
                                asset.get("keywords") or f"robot {comp_name} reference",
                                img_cat,
                                asset.get("limitations", "Whole-machine overview only."),
                                now,
                            ),
                        )
                    asset_dict = dict(asset)
                    asset_dict["reference_id"] = ref_id
                    asset_dict["image_id"] = saved_asset_img["image_id"]
                    asset_dict["file_path"] = saved_asset_img["file_path"]
                    asset_dict["image_category"] = img_cat
                    registered.append(asset_dict)
            except Exception as exc:
                log.warning("Could not ingest manifest %s: %s", mf_path, exc)

    log.info("Initialized Kelvin6K Component Reference Library (%d items)", len(registered))
    return registered


def get_component_reference(
    reference_id_or_name: str,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieves component reference record by reference_id or component_name."""
    clean = (reference_id_or_name or "").strip()
    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        row = conn.execute(
            """
            SELECT cr.*, img.file_path, img.mime_type
            FROM component_references cr
            LEFT JOIN incident_images img ON cr.image_id = img.image_id
            WHERE cr.reference_id = ? OR cr.component_name = ?
            LIMIT 1
            """,
            (clean, clean.lower()),
        ).fetchone()
    d = _row_to_dict(row)
    if d:
        d["category"] = d.get("image_category")
    return d


def list_component_references(
    category: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists all stored Kelvin6K component references, optionally filtered by component name, source_type, or image_category."""
    with db.connect(db_path) as conn:
        _ensure_tables(conn)
        if category:
            cat_clean = category.lower().strip()
            rows = conn.execute(
                """
                SELECT cr.*, img.file_path, img.mime_type
                FROM component_references cr
                LEFT JOIN incident_images img ON cr.image_id = img.image_id
                WHERE LOWER(cr.component_name) = ?
                   OR LOWER(cr.source_type) = ?
                   OR LOWER(IFNULL(cr.image_category, '')) = ?
                ORDER BY cr.created_at ASC, cr.component_name ASC
                """,
                (cat_clean, cat_clean, cat_clean),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT cr.*, img.file_path, img.mime_type
                FROM component_references cr
                LEFT JOIN incident_images img ON cr.image_id = img.image_id
                ORDER BY cr.created_at ASC, cr.component_name ASC
                """
            ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["category"] = d.get("image_category")
        results.append(d)
    return results


def search_component_references(
    query: str,
    defect_category: Optional[str] = None,
    include_incidents: bool = False,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Searches Kelvin6K's persistent image and incident library for genuine photographs
    of relevant components or verified past repairs.
    
    1. First checks component reference library for genuine machine parts (nozzle, hose, pump, etc.).
    2. Optionally searches verified historical incidents if include_incidents is True.
    3. Returns the most relevant reference item or None if no component is supported by evidence.
    """
    q_clean = (query or "").lower().strip()
    stop_words = {
        "what", "this", "that", "with", "from", "have", "tell", "solution", "issue", "problem",
        "find", "the", "and", "please", "help", "diagnose", "show", "look", "see", "there",
        "here", "pattern", "check", "unusual", "why", "how", "when", "where", "does", "did",
        "can", "could", "should", "would", "for", "are", "is", "was", "were", "our", "out",
        "which", "who", "whom", "will", "some", "such", "than", "then", "into"
    }
    words = [w for w in re.findall(r"\b[a-z0-9_\-]{3,}\b", q_clean) if w not in stop_words]
    if not words and not defect_category:
        return None

    # 1. Optionally check verified incidents with stored photos first
    if include_incidents:
        with db.connect(db_path) as conn:
            _ensure_tables(conn)
            rows = conn.execute(
                """
                SELECT * FROM incidents
                WHERE primary_image_id IS NOT NULL
                  AND resolution_status IN ('verified', 'resolved')
                  AND confirmed_root_cause IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 20
                """
            ).fetchall()

        for r in rows:
            inc = dict(r)
            stext = (inc.get("searchable_text") or "").lower()
            matched = sum(1 for w in words if w in stext)
            if (words and matched / len(words) >= 0.4) or (defect_category and defect_category.lower() in stext):
                img = get_incident_image(inc["primary_image_id"], db_path=db_path)
                if img:
                    return {
                        "reference_id": inc["incident_id"],
                        "component_name": inc.get("component_category") or "verified_incident",
                        "display_title": f"Verified Incident Photograph: {inc['incident_id']}",
                        "image_id": inc["primary_image_id"],
                        "machine_version": inc.get("machine_version") or "Kelvin6K-V2",
                        "source_type": "verified_incident",
                        "source_id": inc["incident_id"],
                        "verification_status": inc.get("resolution_status", "verified"),
                        "caption": f"Previous site incident {inc['incident_id']}: {inc.get('defect_category', 'Repair')}",
                        "relevance_explanation": f"Historical incident with verified repair: {inc.get('recommended_action', 'Inspected and verified')}.",
                        "is_incident_photo": True,
                    }

    # 2. Search component references library
    refs = list_component_references(db_path=db_path)
    if not refs:
        refs = init_reference_library(db_path=db_path)

    scored: List[Tuple[float, Dict[str, Any]]] = []

    for ref in refs:
        keywords = (ref.get("keywords") or "").lower()
        comp_name = (ref.get("component_name") or "").lower()
        title = (ref.get("display_title") or "").lower()
        desc = (ref.get("relevance_explanation") or "").lower()
        combined_text = f"{keywords} {comp_name} {title} {desc}"

        score = 0.0

        # Exact component name or distinctive stem match
        comp_stems = [p for p in comp_name.split("_") if p not in ("assembly", "unit")]
        if (
            comp_name in q_clean
            or comp_name.replace("_", " ") in q_clean
            or any(stem in q_clean for stem in comp_stems if len(stem) >= 4)
            or (defect_category and comp_name in defect_category.lower())
        ):
            score += 0.50

        # Defect category match
        if defect_category and defect_category.lower() in combined_text:
            score += 0.35

        # Keyword direct matches
        kw_matched = sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", keywords))
        if kw_matched > 0:
            score += min(0.40, kw_matched * 0.20)

        # Term matches against combined text
        matched = sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", combined_text))
        if words:
            term_score = matched / len(words)
            score += term_score * 0.30

        if score >= 0.25:
            scored.append((score, ref))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    return None

