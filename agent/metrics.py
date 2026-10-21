"""
Derived production metrics for the dashboard.

Everything here is COMPUTED from data that actually exists (the print job row,
the timestamped machine_state history, the system clock and configured shifts).
Nothing is invented: when the inputs for a metric are missing, the metric is
returned as UNAVAILABLE with the reason, not as a plausible-looking number.

Provenance rules used below:
  REAL         measured or computed from real measurements/operator-entered data
  MOCK         from the built-in simulator
  UNAVAILABLE  inputs missing -- shown as such
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent import config, db, shift_store
from agent.logging_setup import get_logger

log = get_logger("METRICS")

REAL = "REAL"
MOCK = "MOCK"
UNAVAILABLE = "UNAVAILABLE"

# Minimum elapsed time and progress before a rate/ETA is meaningful. Below this
# the numbers swing wildly, so they are reported as "not enough data yet".
_MIN_ELAPSED_S = 60.0
_MIN_PROGRESS_MM = 1.0


def field(value: Any, provenance: str, unit: str = "", reason: str = "") -> Dict[str, Any]:
    return {"value": value, "provenance": provenance, "unit": unit, "reason": reason}


def unavailable(reason: str, unit: str = "") -> Dict[str, Any]:
    return field(None, UNAVAILABLE, unit, reason)


def format_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None or seconds < 0:
        return None
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# --- production timing --------------------------------------------------

def production_metrics(job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Elapsed time, print rate, ETA and deadline status for the active job."""
    if job is None:
        reason = "No active print job is being tracked."
        return {
            "elapsed": unavailable(reason),
            "print_rate_mm_per_min": unavailable(reason, "mm/min"),
            "eta": unavailable(reason),
            "eta_timestamp": unavailable(reason),
            "deadline": unavailable(reason),
            "deadline_status": unavailable(reason),
        }

    now = time.time()
    started = job.get("started_at") or now
    elapsed_s = max(0.0, now - started)
    completed = float(job.get("completed_length_mm") or 0.0)
    remaining = float(job.get("remaining_length_mm") or 0.0)

    metrics: Dict[str, Any] = {
        "elapsed": field(format_duration(elapsed_s), REAL),
        "elapsed_seconds": field(round(elapsed_s, 1), REAL, "s"),
    }

    # --- print rate: real average over the job so far ---
    if elapsed_s < _MIN_ELAPSED_S or completed < _MIN_PROGRESS_MM:
        not_yet = (
            f"Not enough data yet -- a rate needs at least {int(_MIN_ELAPSED_S)}s "
            f"elapsed and {_MIN_PROGRESS_MM:g} mm of progress."
        )
        metrics["print_rate_mm_per_min"] = unavailable(not_yet, "mm/min")
        metrics["eta"] = unavailable(not_yet)
        metrics["eta_timestamp"] = unavailable(not_yet)
        rate_per_min = None
    else:
        rate_per_min = completed / (elapsed_s / 60.0)
        metrics["print_rate_mm_per_min"] = field(round(rate_per_min, 2), REAL, "mm/min")

        if rate_per_min > 0 and remaining > 0:
            eta_seconds = (remaining / rate_per_min) * 60.0
            metrics["eta"] = field(format_duration(eta_seconds), REAL)
            metrics["eta_timestamp"] = field(round(now + eta_seconds, 1), REAL, "epoch")
        elif remaining <= 0:
            metrics["eta"] = field("complete", REAL)
            metrics["eta_timestamp"] = field(None, REAL)
        else:
            metrics["eta"] = unavailable("Print rate is zero, so no finish time can be projected.")
            metrics["eta_timestamp"] = unavailable("Print rate is zero.")

    # --- deadline: operator-entered; unavailable until set ---
    deadline_at = job.get("deadline_at")
    if deadline_at is None:
        no_deadline = "No deadline set for this job. Set one with update_job_details(deadline_at=...)."
        metrics["deadline"] = unavailable(no_deadline)
        metrics["deadline_status"] = unavailable(no_deadline)
    else:
        metrics["deadline"] = field(round(float(deadline_at), 1), REAL, "epoch")
        time_left = float(deadline_at) - now
        eta_ts = metrics.get("eta_timestamp", {}).get("value")

        if remaining <= 0:
            status = "COMPLETE"
        elif time_left < 0:
            status = "OVERDUE"
        elif eta_ts is None:
            status = "UNKNOWN"
        elif eta_ts <= deadline_at:
            status = "ON TRACK"
        else:
            status = "AT RISK"

        metrics["deadline_status"] = field(
            status,
            REAL,
            "",
            "" if status != "UNKNOWN" else "Cannot compare: no ETA available yet.",
        )
        metrics["time_to_deadline"] = field(format_duration(abs(time_left)), REAL)

    return metrics


# --- shift / operator ---------------------------------------------------

def current_shift(line_id: str = "line_1") -> Dict[str, Any]:
    """
    Active shift from the database if present; otherwise derived from the
    system clock and configured shift windows.
    """
    try:
        active = shift_store.get_active_shift_for_line(line_id=line_id)
        if active:
            return field(f"{active['shift_type']} (#{active['shift_id']})", REAL)
    except Exception as exc:
        log.warning("Could not query active shift: %s", exc)

    windows = config.SHIFT_WINDOWS
    if not windows:
        return unavailable("No shift schedule configured (set SHIFT_WINDOWS in .env).")

    hour = datetime.now().hour
    for name, start, end in windows:
        in_window = start <= hour < end if start < end else (hour >= start or hour < end)
        if in_window:
            return field(name, REAL)
    return unavailable("Current time does not fall in any configured shift window.")


def current_operator(job: Optional[Dict[str, Any]], line_id: str = "line_1") -> Dict[str, Any]:
    """
    Operator from the active database shift, else from the job row,
    else OPERATOR_NAME in .env.
    """
    try:
        active = shift_store.get_active_shift_for_line(line_id=line_id)
        if active and active.get("operator_id"):
            op = shift_store.get_operator(active["operator_id"])
            op_name = op["name"] if op else active["operator_id"]
            return field(op_name, REAL)
    except Exception as exc:
        log.warning("Could not query active operator: %s", exc)

    if job and job.get("operator"):
        return field(job["operator"], REAL)
    if config.OPERATOR_NAME:
        return field(config.OPERATOR_NAME, REAL)
    return unavailable(
        "No operator recorded. Set OPERATOR_NAME in .env or store one on the job."
    )


# --- material -----------------------------------------------------------

def material_metrics(job: Optional[Dict[str, Any]], elapsed_seconds: Optional[float]) -> Dict[str, Any]:
    """
    Material used/remaining/ratio/consumption.

    There is no material sensor on this machine, so these are only available
    when the operator records totals on the job. Otherwise: UNAVAILABLE.
    """
    no_sensor = (
        "No material sensor is connected. Record material_total_kg and "
        "material_used_kg on the job to track this."
    )
    if job is None:
        return {
            "material_used_kg": unavailable("No active print job.", "kg"),
            "material_remaining_kg": unavailable("No active print job.", "kg"),
            "material_ratio_percent": unavailable("No active print job.", "%"),
            "consumption_rate_kg_per_hour": unavailable("No active print job.", "kg/h"),
            "material_type": unavailable("No active print job."),
        }

    total = job.get("material_total_kg")
    used = job.get("material_used_kg")

    result: Dict[str, Any] = {
        "material_type": field(job["material"], REAL) if job.get("material")
        else unavailable("No material type recorded on the job."),
    }

    if used is None:
        result["material_used_kg"] = unavailable(no_sensor, "kg")
        result["consumption_rate_kg_per_hour"] = unavailable(no_sensor, "kg/h")
    else:
        result["material_used_kg"] = field(round(float(used), 2), REAL, "kg")
        if elapsed_seconds and elapsed_seconds >= _MIN_ELAPSED_S and used > 0:
            rate = float(used) / (elapsed_seconds / 3600.0)
            result["consumption_rate_kg_per_hour"] = field(round(rate, 3), REAL, "kg/h")
        else:
            result["consumption_rate_kg_per_hour"] = unavailable(
                "Not enough elapsed time to compute a consumption rate.", "kg/h"
            )

    if total is None or used is None:
        result["material_remaining_kg"] = unavailable(no_sensor, "kg")
        result["material_ratio_percent"] = unavailable(no_sensor, "%")
    else:
        remaining = max(float(total) - float(used), 0.0)
        result["material_remaining_kg"] = field(round(remaining, 2), REAL, "kg")
        ratio = (float(used) / float(total) * 100.0) if float(total) > 0 else None
        result["material_ratio_percent"] = (
            field(round(ratio, 1), REAL, "%") if ratio is not None
            else unavailable("Material total is zero.", "%")
        )

    return result


# --- wall / layer geometry ---------------------------------------------

def geometry_metrics(job: Optional[Dict[str, Any]], vision_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wall and layer geometry.

    Layer width and layer height ARE measured by the vision pipeline (in mm once
    PIXELS_PER_MM is calibrated). Wall height/width are design values the camera
    cannot determine from one view, so they come from the job when recorded.
    """
    layers = (vision_snapshot or {}).get("layers", {}) if isinstance(vision_snapshot, dict) else {}

    def from_vision(key: str, unit: str, label: str) -> Dict[str, Any]:
        entry = layers.get(key)
        if isinstance(entry, dict):  # {"status": "unavailable", "reason": ...}
            return unavailable(entry.get("reason", f"{label} not measured."), unit)
        if entry is None:
            return unavailable(f"{label} not measured.", unit)
        return field(entry, REAL, unit)

    result = {
        "layer_width_mm": from_vision("layer_width_mm", "mm", "Layer width"),
        "layer_height_mm": from_vision("layer_height_mm", "mm", "Layer height"),
        "layer_width_px": from_vision("layer_width_px", "px", "Layer width"),
        "layer_height_px": from_vision("layer_height_px", "px", "Layer height"),
    }

    no_job = "No active print job." if job is None else None
    for key, unit, label in (
        ("wall_height_mm", "mm", "Wall height"),
        ("wall_width_mm", "mm", "Wall width"),
    ):
        value = (job or {}).get(key)
        if value is None:
            result[key] = unavailable(
                no_job or (
                    f"{label} is a design value the camera cannot determine from one "
                    f"view. Record it on the job to display it."
                ),
                unit,
            )
        else:
            result[key] = field(round(float(value), 1), REAL, unit)

    result["current_layer"] = (
        field((job or {}).get("current_layer"), REAL)
        if job and job.get("current_layer") is not None
        else unavailable("No current layer tracked on the job.")
    )
    result["total_layers"] = (
        field(job["total_layers"], REAL)
        if job and job.get("total_layers") is not None
        else unavailable("No total layer count recorded on the job.")
    )
    return result


# --- production history -------------------------------------------------

_WINDOWS = [("10m", 600), ("1h", 3600), ("8h", 28800), ("24h", 86400)]


def production_history() -> Dict[str, Any]:
    """
    Production over each window, measured from stored machine_state history.

    A window with no reading old enough is reported as not covered yet rather
    than as zero production -- those are different facts.
    """
    now = time.time()
    windows: Dict[str, Any] = {}

    latest = db.get_latest_machine_state()
    latest_completed = (latest or {}).get("completed_length_mm")

    for label, seconds in _WINDOWS:
        if latest_completed is None:
            windows[label] = unavailable("No completed-length history recorded yet.", "mm")
            continue
        past = db.get_progress_at_or_before(now - seconds)
        if past is None:
            windows[label] = unavailable(
                f"No data from {label} ago yet -- the system has not been running that long.",
                "mm",
            )
        else:
            windows[label] = field(round(max(latest_completed - past, 0.0), 1), REAL, "mm")

    # Graph series: completed length over the last hour.
    rows = db.get_state_history(3600, limit=240)
    series: List[Dict[str, Any]] = [
        {
            "t": round(row["timestamp"], 1),
            "completed_mm": row.get("completed_length_mm"),
            "layer_width_mm": row.get("layer_width_mm"),
            "printing": bool(row["printing"]) if row.get("printing") is not None else None,
        }
        for row in rows
    ]

    return {
        "windows": windows,
        "series": series,
        "series_provenance": REAL if series else UNAVAILABLE,
        "series_note": "" if series else "No history recorded yet -- start the backend and a job.",
    }
# --- helper for voice assistant / Alexa query ---

def get_voice_telemetry_summary(state_obj: Any) -> Dict[str, Any]:
    """
    Consolidates active job metrics for the voice assistant.
    Reads real state and honors UNAVAILABLE provenance.
    """
    job = state_obj.get_active_job() if hasattr(state_obj, "get_active_job") else None
    snap = state_obj.get_latest_vision() if hasattr(state_obj, "get_latest_vision") else {}

    geom = geometry_metrics(job, snap)
    prod = production_metrics(job)
    op = current_operator(job)

    curr_l = geom.get("current_layer", {}).get("value")
    tot_l = geom.get("total_layers", {}).get("value")

    rem_l = None
    pct = None
    if curr_l is not None and tot_l is not None and tot_l > 0:
        rem_l = max(0, tot_l - curr_l)
        pct = round((curr_l / tot_l) * 100.0, 1)

    return {
        "operator": op.get("value") or "Unknown Operator",
        "current_layer": curr_l,
        "total_layers": tot_l,
        "remaining_layers": rem_l,
        "progress_percent": pct,
        "layer_width_mm": geom.get("layer_width_mm", {}).get("value"),
        "layer_height_mm": geom.get("layer_height_mm", {}).get("value"),
        "eta": prod.get("eta", {}).get("value"),
        "status": prod.get("deadline_status", {}).get("value"),
    }