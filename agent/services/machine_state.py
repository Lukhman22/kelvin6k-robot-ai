"""
Adapter between the existing print-monitor pipeline and the voice assistant.

This is the ONLY place the assistant reads machine facts from. It sources them,
in priority order, from:

  1. the live in-memory PrinterState (when the camera loop is running in this
     process) -- via the existing agent.print_monitor query API;
  2. the SQLite machine_state table (written by the camera loop, so the
     assistant works in a separate process or after a restart).

No detection logic is duplicated here: layer detection, width measurement,
debouncing and confidence all stay in agent/print_monitor.py.

Every function returns a dict with an explicit `available` flag. Nothing is ever
fabricated: unavailable means unavailable, and low confidence is flagged rather
than smoothed over.
"""

import time
from typing import Any, Dict, Optional

from agent import config, db
from agent.logging_setup import get_logger
from agent.print_monitor import (
    get_camera_status as _live_camera_status,
    get_current_detection as _live_detection,
)

log = get_logger("MACHINE")

# Optional live state, injected by main.py when the vision loop runs in-process.
_live_state = None


def bind_live_state(state) -> None:
    """Lets the assistant read the in-process PrinterState directly."""
    global _live_state
    _live_state = state


_site_camera_service = None


def bind_site_camera_service(service) -> None:
    """Lets the assistant query the dedicated site camera (Camera 2) service."""
    global _site_camera_service
    _site_camera_service = service


def _fresh(timestamp: Optional[float]) -> bool:
    if timestamp is None:
        return False
    return (time.time() - timestamp) <= config.MACHINE_STATE_MAX_AGE_SECONDS


def _snapshot() -> Optional[Dict[str, Any]]:
    """Newest trustworthy machine reading, from live state or the database."""
    if _live_state is not None:
        detection = _live_detection(_live_state)
        if detection.get("status") != "CAMERA_ERROR" and _fresh(detection.get("timestamp")):
            detection["source"] = "live"
            return detection

    row = db.get_latest_machine_state()
    if row and _fresh(row.get("timestamp")):
        return {
            "printing": row.get("printing"),
            "status": row.get("status"),
            "layer_detected": row.get("status") in ("LAYER_DETECTED", "PRINTING"),
            "current_layer": row.get("current_layer"),
            "layer_width": row.get("layer_width_mm"),
            "layer_width_px": row.get("layer_width_px"),
            "width_unit": "mm",
            "confidence": row.get("confidence") or 0.0,
            "motion_score": row.get("motion_score"),
            "camera_connected": row.get("camera_connected"),
            "completed_length_mm": row.get("completed_length_mm"),
            "timestamp": row.get("timestamp"),
            "source": "database",
        }
    return None


def _unavailable(reason: str = "I don't have a current machine status.") -> Dict[str, Any]:
    return {"available": False, "reason": reason}


def _confidence_note(confidence: Optional[float]) -> Optional[str]:
    if confidence is None:
        return None
    if confidence < config.MIN_REPORT_CONFIDENCE:
        return (
            f"The current measurement is uncertain (confidence "
            f"{confidence * 100:.0f} percent)."
        )
    return None


# --- public API used as LLM tools ---------------------------------------

def get_current_print_status() -> Dict[str, Any]:
    """Whether printing is currently running, per the live camera/AI pipeline."""
    snap = _snapshot()
    if snap is None:
        return _unavailable()
    return {
        "available": True,
        "printing": snap.get("printing"),
        "status": snap.get("status"),
        "camera_connected": snap.get("camera_connected"),
        "source": snap.get("source"),
        "age_seconds": round(time.time() - (snap.get("timestamp") or 0), 1),
        "note": _confidence_note(snap.get("confidence")),
    }


def get_current_layer() -> Dict[str, Any]:
    """
    The current layer number.

    The camera alone cannot know the absolute layer number, so this comes from
    the active print job (updated by the machine link or the operator). If no
    job is tracking it, that is reported rather than guessed.
    """
    job = db.get_active_job()
    if job and job.get("current_layer") is not None:
        return {
            "available": True,
            "current_layer": job["current_layer"],
            "total_layers": job.get("total_layers"),
            "source": "print_job",
        }

    snap = _snapshot()
    if snap and snap.get("current_layer") is not None:
        return {"available": True, "current_layer": snap["current_layer"], "source": snap["source"]}

    return _unavailable(
        "No current layer number is being tracked. It needs an active print job "
        "or a machine link -- the camera alone cannot know the absolute layer number."
    )


def get_current_layer_width() -> Dict[str, Any]:
    """Measured width of the layer being printed, in millimetres when calibrated."""
    snap = _snapshot()
    if snap is None:
        return _unavailable()

    width_mm = snap.get("layer_width")
    width_px = snap.get("layer_width_px")
    confidence = snap.get("confidence")

    if width_mm is None and width_px is not None:
        return {
            "available": False,
            "width_px": width_px,
            "confidence": confidence,
            "reason": (
                "Layer width calibration is not configured, so only a pixel "
                "measurement exists. Set PIXELS_PER_MM to get millimetres."
            ),
        }
    if width_mm is None:
        return _unavailable("No layer width has been measured.")

    return {
        "available": True,
        "layer_width_mm": round(float(width_mm), 2),
        "layer_width_px": width_px,
        "confidence": confidence,
        "source": snap.get("source"),
        "note": _confidence_note(confidence),
    }


def get_print_progress() -> Dict[str, Any]:
    """
    How much of the wall is printed and how much remains.

    remaining_mm is always computed as max(total - completed, 0); it is never
    read from an independently stored value.
    """
    job = db.get_active_job()
    if job is None:
        return _unavailable(
            "No active print job is being tracked, so progress cannot be reported. "
            "Start one with start_print_job()."
        )

    total = float(job.get("total_length_mm") or 0.0)
    completed = float(job.get("completed_length_mm") or 0.0)
    remaining = max(total - completed, 0.0)
    percentage = round((completed / total) * 100.0, 1) if total > 0 else None

    return {
        "available": True,
        "job_name": job.get("job_name"),
        "total_mm": round(total, 1),
        "completed_mm": round(completed, 1),
        "remaining_mm": round(remaining, 1),
        "percentage": percentage,
        "current_layer": job.get("current_layer"),
        "total_layers": job.get("total_layers"),
        "status": job.get("status"),
    }


def get_machine_status() -> Dict[str, Any]:
    """Camera connection and overall machine/vision status."""
    camera = _live_camera_status(_live_state) if _live_state is not None else None
    snap = _snapshot()

    if snap is None and camera is None:
        return _unavailable()

    return {
        "available": True,
        "camera": camera or {"connected": bool(snap and snap.get("camera_connected"))},
        "status": (snap or {}).get("status"),
        "printing": (snap or {}).get("printing"),
        "confidence": (snap or {}).get("confidence"),
        "motion_score": (snap or {}).get("motion_score"),
        "source": (snap or {}).get("source"),
        "note": _confidence_note((snap or {}).get("confidence")),
    }


def get_machine_summary() -> Dict[str, Any]:
    """Everything at once -- convenient for dashboards and 'how is it going?'."""
    return {
        "print_status": get_current_print_status(),
        "layer": get_current_layer(),
        "layer_width": get_current_layer_width(),
        "progress": get_print_progress(),
        "machine": get_machine_status(),
    }


# --- job management (used by tests and the operator CLI) ----------------

def start_print_job(
    job_name: str,
    total_length_mm: float,
    material: Optional[str] = None,
    total_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """Starts tracking a print job so progress questions can be answered."""
    db.init_db()
    job_id = db.create_print_job(job_name, total_length_mm, material, total_layers)
    db.record_event("job_started", f"Started job '{job_name}' ({total_length_mm} mm)", job_id)
    log.info("Started print job %s (id=%s, %.1f mm)", job_name, job_id, total_length_mm)
    return {"available": True, "job_id": job_id, "job_name": job_name}


def update_progress(
    completed_length_mm: Optional[float] = None,
    current_layer: Optional[int] = None,
) -> Dict[str, Any]:
    """Updates the active job's progress. remaining is recomputed automatically."""
    job = db.get_active_job()
    if job is None:
        return _unavailable("No active print job to update.")
    db.update_job_progress(job["id"], completed_length_mm, current_layer)
    return get_print_progress()


def get_site_people_count(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Counts people currently visible on site using dedicated Camera 2.
    Never uses Camera 1 (which is focused on the print nozzle/bead).
    """
    if _site_camera_service is None:
        return {
            "available": False,
            "people_count": None,
            "confidence": 0.0,
            "provenance": "UNAVAILABLE",
            "reason": "Site camera (Camera 2) service is not bound.",
            "note": "",
        }

    if not getattr(config, "SITE_CAMERA_ENABLED", True):
        return {
            "available": False,
            "people_count": None,
            "confidence": 0.0,
            "provenance": "UNAVAILABLE",
            "reason": "Site camera (Camera 2) is disabled in configuration.",
            "note": "",
        }

    status = getattr(_site_camera_service, "status", "unknown")
    if status != "live":
        err = getattr(_site_camera_service, "error", None) or f"status is {status}"
        return {
            "available": False,
            "people_count": None,
            "confidence": 0.0,
            "provenance": "UNAVAILABLE",
            "reason": f"Site camera (Camera 2) is currently unavailable ({err}).",
            "note": "",
        }

    res = _site_camera_service.get_people_count(force_refresh=force_refresh)
    if not res.available:
        return {
            "available": False,
            "people_count": None,
            "confidence": 0.0,
            "provenance": "UNAVAILABLE",
            "reason": res.reason or "Could not analyze site camera frame.",
            "note": "",
        }

    note = ""
    if res.confidence and res.confidence < getattr(config, "PEOPLE_DETECTOR_CONFIDENCE_THRESHOLD", 0.65):
        note = f"The count is uncertain ({res.confidence * 100:.0f}% confidence)."

    return {
        "available": True,
        "people_count": res.people_count,
        "confidence": res.confidence,
        "details": res.details,
        "provenance": "REAL",
        "detector": res.detector,
        "timestamp": res.timestamp,
        "reason": "",
        "note": note,
    }


def get_site_detections(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Returns normalized structured entity detections from dedicated Camera 2.
    Separates vision detection from agent reasoning.
    """
    if _site_camera_service is None:
        return {
            "available": False,
            "provenance": "UNAVAILABLE",
            "reason": "Site camera (Camera 2) service is not bound.",
            "entities": [],
            "people_count": None,
            "supported_classes": [],
        }

    if not getattr(config, "SITE_CAMERA_ENABLED", True):
        return {
            "available": False,
            "provenance": "UNAVAILABLE",
            "reason": "Site camera (Camera 2) is disabled in configuration.",
            "entities": [],
            "people_count": None,
            "supported_classes": [],
        }

    status = getattr(_site_camera_service, "status", "unknown")
    if status != "live":
        err = getattr(_site_camera_service, "error", None) or f"status is {status}"
        return {
            "available": False,
            "provenance": "UNAVAILABLE",
            "reason": f"Site camera (Camera 2) is currently unavailable ({err}).",
            "entities": [],
            "people_count": None,
            "supported_classes": [],
        }

    res = _site_camera_service.get_people_count(force_refresh=force_refresh)
    if not res.available:
        return {
            "available": False,
            "provenance": "UNAVAILABLE",
            "reason": res.reason or "Could not analyze site camera frame.",
            "entities": [],
            "people_count": None,
            "supported_classes": [],
        }

    entities = res.entities_dict() if hasattr(res, "entities_dict") else getattr(res, "entities", [])
    supported_classes = []
    if hasattr(_site_camera_service, "detector") and hasattr(_site_camera_service.detector, "get_supported_classes"):
        supported_classes = _site_camera_service.detector.get_supported_classes()

    return {
        "available": True,
        "provenance": "REAL",
        "people_count": res.people_count,
        "confidence": res.confidence,
        "details": res.details,
        "entities": entities,
        "supported_classes": supported_classes,
        "detector": res.detector,
        "timestamp": res.timestamp,
        "reason": "",
    }

