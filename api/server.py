"""
HTTP API for the Kelvin6K React dashboard.

A thin wrapper over the EXISTING Python modules -- no machine logic is
reimplemented here:

    /api/status           -> agent.services.machine_state (+ agent.sensors, agent.db)
    /api/assistant/query  -> agent.assistant / agent.query_router (tools + RAG)
    /api/stt              -> agent.listener   (faster-whisper)
    /api/tts*             -> agent.voice      (ElevenLabs / pyttsx3)
    /api/camera/*         -> api.camera_service (CameraStream + VisionPipeline)

Every value carries a provenance tag so the UI can never present a simulated or
missing number as if it were measured:

    REAL         measured from the camera/AI pipeline, or entered for the job
    MOCK         produced by the built-in simulator (SimulatedSensors)
    UNAVAILABLE  not connected / not measured -- shown as such, never guessed

READ-ONLY: there are no endpoints that command the robot, pump or accelerator.
"""

import asyncio
import datetime
import io
import json
import os
import re
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from agent import config, db, incident_store, metrics, shift_store
from agent.assistant import Assistant
from agent.logging_setup import configure_logging, get_logger
from agent.rag.retriever import index_summary
from agent.rules import RuleEngine
from agent.sensors import NotConnectedMachine, SimulatedSensors
from agent.services import machine_state as ms
from agent.state import PrinterState
from api.camera_service import CameraService

configure_logging()
log = get_logger("API")

REAL = "REAL"
MOCK = "MOCK"
UNAVAILABLE = "UNAVAILABLE"

app = FastAPI(title="Kelvin6K API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dashboard; tighten for production deployment
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-TTS-Engine"],
)

# Shared singletons -- the same objects the console app uses.
state = PrinterState()
rules = RuleEngine()
camera_service = CameraService(state, rules)
from api.site_camera_service import SiteCameraService
site_camera_service = SiteCameraService()
ms.bind_site_camera_service(site_camera_service)
assistant = Assistant(state=state)
_sensors = SimulatedSensors()
_machine_link = NotConnectedMachine()


def field(value: Any, provenance: str, unit: str = "", reason: str = "") -> Dict[str, Any]:
    """Uniform value envelope so the UI always knows where a number came from."""
    return {"value": value, "provenance": provenance, "unit": unit, "reason": reason}


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    incident_store.init_reference_library()
    ms.bind_live_state(state)
    ms.bind_site_camera_service(site_camera_service)
    camera_service.start()
    if config.SITE_CAMERA_ENABLED:
        site_camera_service.start()

    # Load speech-to-text in the background so the FIRST question does not pay
    # the model load. Threaded, so startup is not blocked either.
    import threading

    from agent import listener

    threading.Thread(target=listener.preload, name="whisper-preload", daemon=True).start()

    # Probe the Claude connection once, in the background. Without this the
    # first question pays the round trip -- and if the key is rejected it pays
    # a failing one, measured at over 3 seconds, before the latch kicks in.
    def _warm_answer_engine() -> None:
        try:
            # A real question: an unrecognised one returns before the Claude call
            # and would warm nothing.
            assistant.ask("what is the machine status")
        except Exception:
            pass  # a failed warmup must never affect startup

    threading.Thread(target=_warm_answer_engine, name="answer-warmup", daemon=True).start()
    log.info("API started (camera source=%s); speech-to-text loading in background",
             config.CAMERA_SOURCE)


@app.on_event("shutdown")
def _shutdown() -> None:
    camera_service.stop()
    site_camera_service.stop()


# --- health -------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    from agent.voice import provider_name

    rag = index_summary()
    return {
        "ok": True,
        "camera_source": config.CAMERA_SOURCE,
        "camera_status": camera_service.status,
        "llm": "gemini" if config.GEMINI_API_KEY else "offline-router",
        "llm_model": config.GEMINI_MODEL if config.GEMINI_API_KEY else None,
        "stt": f"faster-whisper ({config.WHISPER_MODEL_SIZE})",
        "tts": provider_name(),
        "rag_chunks": rag["chunks"],
        "rag_mode": "semantic" if rag["semantic"] else "lexical",
        "calibrated": config.PIXELS_PER_MM > 0,
        "read_only": True,
    }


# --- status -------------------------------------------------------------

@app.get("/api/status")
def status() -> Dict[str, Any]:
    """Everything the dashboard shows, each field tagged with its provenance."""
    progress = ms.get_print_progress()
    layer = ms.get_current_layer()
    width = ms.get_current_layer_width()
    print_status = ms.get_current_print_status()
    machine = ms.get_machine_status()

    camera = {
        "status": camera_service.status,
        "source": camera_service.source,
        "fps": round(camera_service.fps, 1),
        "resolution": list(camera_service.resolution),
        "error": camera_service.error,
        "provenance": REAL if camera_service.status == "live" else UNAVAILABLE,
    }

    camera2 = {
        "status": site_camera_service.status,
        "source": site_camera_service.source,
        "fps": round(site_camera_service.fps, 1),
        "resolution": list(site_camera_service.resolution),
        "error": site_camera_service.error,
        "provenance": REAL if site_camera_service.status == "live" else UNAVAILABLE,
        "people_count": site_camera_service.latest_people_count(),
        "fire_count": site_camera_service.latest_fire_count(),
        "detections": site_camera_service.latest_detections(),
    }

    # --- production (from the tracked print job) ---
    if progress.get("available"):
        production = {
            "job_name": field(progress["job_name"], REAL),
            "total_mm": field(progress["total_mm"], REAL, "mm"),
            "completed_mm": field(progress["completed_mm"], REAL, "mm"),
            "remaining_mm": field(progress["remaining_mm"], REAL, "mm"),
            "percentage": field(progress["percentage"], REAL, "%"),
            "current_layer": field(progress.get("current_layer"), REAL),
            "total_layers": field(progress.get("total_layers"), REAL),
        }
    else:
        reason = progress.get("reason", "No active print job.")
        production = {
            key: field(None, UNAVAILABLE, unit, reason)
            for key, unit in (
                ("job_name", ""), ("total_mm", "mm"), ("completed_mm", "mm"),
                ("remaining_mm", "mm"), ("percentage", "%"),
                ("current_layer", ""), ("total_layers", ""),
            )
        }

    # --- vision-derived ---
    vision = {
        "printing": field(
            print_status.get("printing") if print_status.get("available") else None,
            REAL if print_status.get("available") else UNAVAILABLE,
            "", print_status.get("reason", ""),
        ),
        "print_status": field(
            print_status.get("status") if print_status.get("available") else None,
            REAL if print_status.get("available") else UNAVAILABLE,
            "", print_status.get("reason", ""),
        ),
        "layer_number": field(
            layer.get("current_layer") if layer.get("available") else None,
            REAL if layer.get("available") else UNAVAILABLE,
            "", layer.get("reason", ""),
        ),
        "layer_width_mm": field(
            width.get("layer_width_mm") if width.get("available") else None,
            REAL if width.get("available") else UNAVAILABLE,
            "mm", width.get("reason", ""),
        ),
        "layer_width_px": field(
            width.get("width_px") or width.get("layer_width_px"),
            REAL if (width.get("width_px") or width.get("layer_width_px")) is not None else UNAVAILABLE,
            "px", "",
        ),
        "confidence": field(
            machine.get("confidence") if machine.get("available") else None,
            REAL if machine.get("available") else UNAVAILABLE, "", "",
        ),
        "motion_score": field(
            machine.get("motion_score") if machine.get("available") else None,
            REAL if machine.get("available") else UNAVAILABLE, "", "",
        ),
        "note": width.get("note") or print_status.get("note") or machine.get("note"),
    }

    # --- process sensors: SIMULATED until a machine link exists ---
    reading = _sensors.read()
    sensor_provenance = MOCK if reading.source == "SIMULATED" else REAL
    material = {
        "material_flow_percent": field(reading.material_flow_percent, sensor_provenance, "%"),
        "nozzle_temp_c": field(reading.nozzle_temp_c, sensor_provenance, "C"),
        "bed_temp_c": field(reading.bed_temp_c, sensor_provenance, "C"),
        "machine_state": field(reading.machine_state, sensor_provenance),
        "source_note": (
            "These process values come from the built-in SIMULATOR, not the real "
            "machine. Connect a MachineInterface to replace them."
            if sensor_provenance == MOCK else ""
        ),
    }

    # --- machine telemetry: genuinely not connected ---
    link = _machine_link.read()
    link_reason = link.unavailable_reason
    machine_telemetry = {
        "pump_running": field(link.pump_running, UNAVAILABLE, "", link_reason),
        "pump_speed_rpm": field(link.pump_speed_rpm, UNAVAILABLE, "RPM", link_reason),
        "accelerator_running": field(link.accelerator_running, UNAVAILABLE, "", link_reason),
        "accelerator_speed_rpm": field(link.accelerator_speed_rpm, UNAVAILABLE, "RPM", link_reason),
        "print_speed_mm_s": field(link.print_speed_mm_s, UNAVAILABLE, "mm/s", link_reason),
        "pressure_bar": field(link.pressure_bar, UNAVAILABLE, "bar", link_reason),
    }

    snapshot = state.snapshot()
    alerts = [
        {"level": a["level"], "message": a["message"], "seconds_ago": a["seconds_ago"]}
        for a in snapshot.get("recent_alerts", [])
    ]

    # --- derived timing / geometry / material ---
    job = db.get_active_job()
    timing = metrics.production_metrics(job)
    elapsed_seconds = timing.get("elapsed_seconds", {}).get("value")
    geometry = metrics.geometry_metrics(job, snapshot.get("vision", {}))
    material_detail = metrics.material_metrics(job, elapsed_seconds)

    # --- robot / communication ---
    machine_link_connected = link.source != "NOT_CONNECTED"
    camera_live = camera_service.status == "live"
    robot_online = camera_live or machine_link_connected

    if print_status.get("available"):
        operation_field = metrics.field(
            "PRINTING" if print_status.get("printing") else "IDLE", REAL
        )
    else:
        operation_field = metrics.unavailable(
            print_status.get("reason", "No current machine status.")
        )

    robot = {
        "robot_online": metrics.field(robot_online, REAL),
        "robot_status": metrics.field("ONLINE" if robot_online else "OFFLINE", REAL),
        "machine_state": material["machine_state"],
        "current_operation": operation_field,
        "current_shift": metrics.current_shift(),
        "current_operator": metrics.current_operator(job),
        "communication": {
            "api": metrics.field("CONNECTED", REAL),
            "camera": metrics.field(
                camera_service.status.upper(),
                REAL if camera_live else UNAVAILABLE, "", camera_service.error or "",
            ),
            "machine_link": metrics.field(
                "CONNECTED" if machine_link_connected else "NOT CONNECTED",
                REAL if machine_link_connected else UNAVAILABLE,
                "", "" if machine_link_connected else link_reason,
            ),
        },
    }

    # --- problems ---
    warnings = [a for a in alerts if a["level"] == "warning"]
    criticals = [a for a in alerts if a["level"] == "critical"]
    machine_problems = [
        a for a in alerts
        if any(w in a["message"].lower() for w in ("flow", "temp", "pump", "accelerator", "machine"))
    ]
    camera_problems = []
    if camera_service.status != "live":
        camera_problems.append({
            "level": "warning",
            "message": camera_service.error or f"Camera status: {camera_service.status}",
            "seconds_ago": 0.0,
        })

    problems = {
        "warnings": warnings,
        "critical": criticals,
        "machine": machine_problems,
        "camera": camera_problems,
        "counts": {
            "warnings": len(warnings), "critical": len(criticals),
            "machine": len(machine_problems), "camera": len(camera_problems),
        },
    }

    return {
        "timestamp": time.time(),
        "camera": camera,
        "camera2": camera2,
        "people_count": site_camera_service.latest_people_count(),
        "production": {**production, **timing},
        "vision": vision,
        "geometry": geometry,
        "material": {**material, **material_detail},
        "machine_telemetry": machine_telemetry,
        "robot": robot,
        "problems": problems,
        "alerts": alerts,
        "calibrated": config.PIXELS_PER_MM > 0,
        # Additive: existing keys above are unchanged. None until the camera
        # service has produced a measurement, or when metrology is disabled.
        "metrology": (
            camera_service.metrology.to_dict()
            if getattr(camera_service, "metrology", None) is not None
            else None
        ),
    }


@app.get("/api/metrology/announcement")
def metrology_announcement() -> Dict[str, Any]:
    """
    The most recent layer-width announcement worth speaking, if any.

    The dashboard polls this and hands the text to the EXISTING Voice
    Assistant's speak path, so there is one voice system rather than a second
    one talking over the first. `id` increments per announcement; the UI speaks
    only when it sees an id it has not spoken yet.

    The text is composed from the same MetrologyResult published above, so the
    spoken width and the width drawn on the camera can never disagree.
    """
    announcement = getattr(camera_service, "metrology_announcement", None)
    if announcement is None:
        return {"announcement": None, "enabled": config.METROLOGY_ENABLED}
    return {"announcement": announcement.to_dict(), "enabled": config.METROLOGY_ENABLED}


@app.get("/api/history")
def history() -> Dict[str, Any]:
    """Production over the last 10 minutes / 1 hour / 8 hours / 24 hours."""
    return metrics.production_history()


# --- assistant (shared by text AND voice) -------------------------------

class AttachmentResponse(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    kind: str
    observations: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class AskRequest(BaseModel):
    question: str
    speak: bool = False  # server-side TTS; the UI plays audio in the browser
    session_id: str = "default_terminal"
    line_id: str = "line_1"
    operator_id: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    tool: Optional[str] = None
    intent: Optional[str] = None
    facts: Optional[Dict[str, Any]] = None
    knowledge_sources: Optional[list] = None
    data_available: bool = True
    llm: str
    timestamp: float
    attendance_updated: bool = False
    attachments: Optional[List[AttachmentResponse]] = None
    fallback_used: bool = False
    rag_score: Optional[float] = None
    primary_image_id: Optional[str] = None
    annotated_image_id: Optional[str] = None
    reference_image_id: Optional[str] = None
    reference_component: Optional[Dict[str, Any]] = None


@app.post("/api/assistant/query", response_model=AskResponse)
@app.post("/api/ask", response_model=AskResponse)
@app.post("/api/chat", response_model=AskResponse)
async def ask(request: Request) -> AskResponse:
    """
    THE shared query service. Both the text box and the voice pipeline call
    this, so a typed and a spoken question follow exactly the same path.
    Supports both application/json and multipart/form-data with file attachments.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    processed_attachments = []

    if "multipart/form-data" in content_type:
        form = await request.form()
        question = str(form.get("question", "")).strip()
        speak = str(form.get("speak", "false")).lower() in ("true", "1")
        session_id = str(form.get("session_id", "default_terminal"))
        line_id = str(form.get("line_id", "line_1"))
        operator_id = str(form.get("operator_id", ""))

        raw_files = form.getlist("files") or form.getlist("attachments")
        if not raw_files and "file" in form:
            f_single = form.get("file")
            if f_single:
                raw_files = [f_single]

        if len(raw_files) > config.MAX_ATTACHMENT_COUNT:
            raise HTTPException(
                status_code=400,
                detail=f"Too many attachments ({len(raw_files)}). Maximum allowed is {config.MAX_ATTACHMENT_COUNT}."
            )

        for uf in raw_files:
            if hasattr(uf, "read"):
                file_bytes = await uf.read()
                filename = getattr(uf, "filename", "unnamed_file")
                ctype = getattr(uf, "content_type", None)
                try:
                    from agent import troubleshooter
                    att = troubleshooter.validate_and_process_attachment(file_bytes, filename, ctype)
                    processed_attachments.append(att)
                except ValueError as val_err:
                    raise HTTPException(status_code=400, detail=str(val_err))
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Failed to process attachment '{filename}': {exc}")
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        question = str(body.get("question", "")).strip()
        speak = bool(body.get("speak", False))
        session_id = str(body.get("session_id", "default_terminal"))
        line_id = str(body.get("line_id", "line_1"))
        operator_id = str(body.get("operator_id", ""))

    if not question and not processed_attachments:
        raise HTTPException(status_code=400, detail="question or attachment must be provided")

    if not question and processed_attachments:
        question = "Please analyze the attached image/file for robot components, defects, errors, or relevant maintenance procedures."

    from agent import query_router

    log.info("[QUERY] Question: %s (attachments: %d)", question, len(processed_attachments))
    _t_route = time.time()
    routed = query_router.route(question)
    log.info("[TIMING] Query + data     : %.2fs  (intent %s)",
             time.time() - _t_route, routed.intent.value)
    log.info("[QUERY] Intent: %s", routed.intent.value)
    if routed.available:
        log.info("[DATABASE] Query: %s", routed.intent.value)
        log.info("[DATABASE] Result: %s", routed.facts)
    else:
        log.info("[DATABASE] Result: UNAVAILABLE -- %s", routed.reason or "no reason given")
    _t_answer = time.time()
    answer = assistant.ask(
        question,
        session_id=session_id,
        line_id=line_id,
        operator_id=operator_id,
        routed=routed,
        attachments=processed_attachments or None,
    )
    log.info("[TIMING] Answer wording   : %.2fs", time.time() - _t_answer)
    log.info("[ASSISTANT] Answer: %s", answer)

    last_tb = getattr(assistant, "_last_troubleshoot_result", None)
    knowledge_sources = None
    fallback_used = False
    rag_score = None
    tb_primary_image_id = None
    tb_annotated_image_id = None
    tb_reference_image_id = None
    tb_reference_component = None

    if last_tb and last_tb.question == question:
        if last_tb.citations:
            knowledge_sources = last_tb.citations[:3]
        fallback_used = last_tb.fallback_used
        rag_score = last_tb.rag_score
        tb_primary_image_id = getattr(last_tb, "primary_image_id", None)
        tb_annotated_image_id = getattr(last_tb, "annotated_image_id", None)
        tb_reference_image_id = getattr(last_tb, "reference_image_id", None)
        raw_ref_comp = getattr(last_tb, "reference_component", None)
        tb_reference_component = raw_ref_comp if isinstance(raw_ref_comp, dict) else ({"name": str(raw_ref_comp)} if raw_ref_comp else None)
    elif routed.knowledge and routed.knowledge.get("results"):
        knowledge_sources = [r["source"] for r in routed.knowledge["results"][:3]]

    attendance_updated = False
    if (routed.intent in (
        query_router.Intent.SHIFT_START,
        query_router.Intent.SHIFT_END,
        query_router.Intent.RECORD_WALLS_COMPLETED,
        query_router.Intent.CORRECT_WALL_PRODUCTION,
    ) or any(phrase in answer.lower() for phrase in (
        "has started", "has ended", "shift has started", "shift has ended",
        "clocked in", "clocked out", "recorded", "corrected wall production",
    ))):
        attendance_updated = True
        if routed.facts is None:
            routed.facts = {}
        routed.facts["attendance_updated"] = True

    LOCAL_ATTENDANCE_INTENTS = {
        query_router.Intent.SHIFT_START,
        query_router.Intent.SHIFT_END,
        query_router.Intent.GET_ATTENDANCE_STATUS,
        query_router.Intent.GET_ATTENDANCE_HOURS,
        query_router.Intent.GET_ATTENDANCE_OVERTIME,
        query_router.Intent.GET_OPERATOR_WORKING_DAYS,
        query_router.Intent.GET_ACTIVE_SHIFT,
        query_router.Intent.GET_PREVIOUS_SHIFT,
        query_router.Intent.GET_SHIFT_BY_OPERATOR,
        query_router.Intent.RECORD_WALLS_COMPLETED,
        query_router.Intent.GET_WALL_PRODUCTION,
        query_router.Intent.CORRECT_WALL_PRODUCTION,
    }

    if routed.intent in LOCAL_ATTENDANCE_INTENTS or attendance_updated:
        llm_indicator = "local-attendance"
    elif last_tb and last_tb.question == question:
        if last_tb.source in ("rag-knowledge", "reference-library"):
            llm_indicator = "rag-knowledge"
        elif last_tb.gemini_called:
            has_img = any(a.kind == "image" for a in processed_attachments)
            llm_indicator = "gemini-vision" if has_img else "gemini"
        else:
            llm_indicator = "offline-router"
    elif getattr(assistant, "_last_gemini_called", False):
        has_img = any(a.kind == "image" for a in processed_attachments)
        llm_indicator = "gemini-vision" if has_img else "gemini"
    else:
        llm_indicator = "offline-router"

    if speak:
        try:
            from agent.voice import speak as server_speak

            server_speak(answer)
        except Exception as exc:
            log.warning("Server-side TTS failed: %s", exc)

    att_responses = [
        AttachmentResponse(
            filename=a.filename,
            content_type=a.content_type,
            size_bytes=a.size_bytes,
            kind=a.kind,
            observations=a.observations,
            width=a.width,
            height=a.height,
        )
        for a in processed_attachments
    ] if processed_attachments else None

    return AskResponse(
        question=question,
        answer=answer,
        tool=Assistant.route(question),
        intent=routed.intent.value,
        facts=routed.facts or None,
        knowledge_sources=knowledge_sources,
        data_available=routed.available,
        llm=llm_indicator,
        timestamp=time.time(),
        attendance_updated=attendance_updated,
        attachments=att_responses,
        fallback_used=fallback_used,
        rag_score=rag_score,
        primary_image_id=tb_primary_image_id,
        annotated_image_id=tb_annotated_image_id,
        reference_image_id=tb_reference_image_id,
        reference_component=tb_reference_component,
    )


# --- speech to text (reuses the existing faster-whisper pipeline) -------

def _probe_audio(path: str):
    """
    Measures an audio file by DECODING it: (duration_seconds, peak, codec).

    Why not the container header: the WebM that a browser's MediaRecorder
    produces is written as a live stream, so its header carries no Duration
    element at all. `container.duration` is therefore None for every browser
    recording, however long it actually is -- and treating that as 0 rejected
    perfectly good audio as "too short". Counting decoded samples is the only
    reliable measure, and it yields the peak amplitude for free, which
    separates "no audio" from "silent audio".
    """
    import av
    import numpy as np

    total_samples = 0
    sample_rate = 0
    peak = 0.0
    codec = "unknown"

    with av.open(path) as container:
        streams = [st for st in container.streams if st.type == "audio"]
        if not streams:
            raise ValueError("no audio stream in container")
        stream = streams[0]
        codec = stream.codec_context.name or "unknown"
        sample_rate = stream.codec_context.sample_rate or getattr(stream, "rate", 0) or 0

        for frame in container.decode(audio=0):
            total_samples += frame.samples
            if not sample_rate:
                sample_rate = frame.sample_rate or 0
            array = frame.to_ndarray()
            if array.size:
                magnitude = float(np.abs(array).max())
                # Integer formats are full-scale; normalise so `peak` is 0..1
                # regardless of whether the decoder gave int16 or float32.
                if array.dtype.kind in "iu":
                    magnitude /= float(np.iinfo(array.dtype).max)
                peak = max(peak, magnitude)

    duration = (total_samples / float(sample_rate)) if sample_rate else 0.0
    return duration, peak, codec


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribes uploaded audio with the project's existing faster-whisper model.

    Accepts whatever the browser's MediaRecorder produces (webm/opus, ogg, mp4)
    plus wav/mp3: the audio is written to a temp file and decoded by PyAV inside
    faster-whisper, so voice input works in every browser.
    """
    payload = await audio.read()
    # Full request trace: every one of these lines has, at some point, been the
    # one that identified a failure, so they all stay.
    log.info("[STT] --- request received ---")
    log.info("[STT] filename     : %s", audio.filename or "(none)")
    log.info("[STT] content type : %s", audio.content_type or "(none)")
    log.info("[STT] upload size  : %d bytes", len(payload))
    if not payload:
        log.error("[STT ERROR] Microphone capture failed -- 0 bytes received")
        raise HTTPException(status_code=400, detail="MICROPHONE CAPTURE FAILED: 0 audio bytes received")

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = handle.name
        log.info("[STT] temp file    : %s", temp_path)
        log.info(
            "[STT] file on disk : exists=%s, %d bytes",
            os.path.exists(temp_path),
            os.path.getsize(temp_path) if os.path.exists(temp_path) else -1,
        )

        from agent.listener import transcribe
        # Decode-level validation: a container header with no audio frames
        # (the classic "recording stopped instantly" case) has ~0 duration and
        # must be reported as such rather than silently returning nothing.
        try:
            probe_duration, peak, codec = _probe_audio(temp_path)
            log.info(
                "[STT] Audio format: %s, duration: %.2fs, peak amplitude: %.4f",
                codec, probe_duration, peak,
            )

            if probe_duration < 0.3:
                log.error("[STT ERROR] Only %.2fs of audio decoded -- no usable speech", probe_duration)
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Only {probe_duration:.2f} seconds of audio arrived. "
                        "Click START LISTENING, wait for LISTENING, then speak your question."
                    ),
                )
            # Real duration but a flat waveform means the microphone delivered
            # digital silence -- a different problem with a different fix, so it
            # must not be reported as "too short".
            if peak <= 0.0005:
                log.error("[STT ERROR] %.2fs of audio but peak amplitude %.5f -- silence", probe_duration, peak)
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"The recording is {probe_duration:.1f} seconds of silence. "
                        "The microphone is open but Windows is sending no sound. "
                        "Run the microphone diagnostic in the assistant panel."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            log.warning("[STT] Could not probe audio (%s) -- letting Whisper try anyway", exc)

        log.info("[STT] Transcribing with faster-whisper (%s)...", config.WHISPER_MODEL_SIZE)
        started = time.time()
        text, info = transcribe(temp_path, return_info=True)
        stt_seconds = time.time() - started
        log.info(
            "[TIMING] STT              : %.2fs  (language=%s)",
            stt_seconds, getattr(info, "language", "?"),
        )
        log.info("[STT] Transcript   : %r", text)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[STT ERROR] Transcription raised %s", type(exc).__name__)
        # Most common cause by far: a container header with no audio frames,
        # i.e. the recording was stopped before any speech was captured.
        message = str(exc)
        if "Invalid data" in message or "moov atom" in message or "EBML" in message:
            log.error("[STT ERROR] Invalid/unsupported audio format -- likely an empty recording")
            raise HTTPException(
                status_code=422,
                detail=(
                    "The recording contained no usable audio (only a file header). "
                    "Click START LISTENING, speak for 2-5 seconds, then stop."
                ),
            )
        log.error("[STT ERROR] %s", message)
        raise HTTPException(status_code=400, detail=f"could not transcribe audio: {message}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    # Whisper hallucinates stock phrases when handed silence or noise ("You",
    # "Thank you.", "Thanks for watching!"). Those must never reach the query
    # router, or a silent recording yields a confident answer to a question the
    # operator never asked.
    _NOISE_TRANSCRIPTS = {
        "you", "thank you", "thanks for watching", "thanks for watching!",
        "bye", "bye.", "silence", ".", "...", "um", "uh", "okay",
    }
    _SHORT_ALLOWLIST = {"no", "ha", "ok", "yes", "do"}
    normalised = text.lower().strip(" .!?,")

    if not text or (normalised in _NOISE_TRANSCRIPTS and normalised not in _SHORT_ALLOWLIST) or (len(normalised) < 3 and normalised not in _SHORT_ALLOWLIST):
        log.error("[STT ERROR] Empty or noise-only transcript: %r", text)
        raise HTTPException(
            status_code=422,
            detail="I couldn't hear a clear question. Please try again.",
        )

    log.info("[STT] Transcript: %s", text)

    return {
        "transcript": text,
        "engine": f"faster-whisper ({config.WHISPER_MODEL_SIZE})",
        "language": getattr(info, "language", None),
        "bytes_received": len(payload),
    }


# --- text to speech -----------------------------------------------------

class SpeakRequest(BaseModel):
    text: str


@app.post("/api/tts")
def text_to_speech(request: SpeakRequest) -> Dict[str, Any]:
    """
    Speaks through the SERVER's speaker (agent/voice.py).

    Only useful when browser and backend are the same machine. The dashboard
    prefers /api/tts/audio, which returns the audio so the BROWSER plays it on
    the operator's own speakers.
    """
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    from agent.voice import TTSError, provider_name, speak as server_speak

    log.info("[TTS] Sending answer to speak(): %s", text)
    _t_tts = time.time()
    try:
        ok = server_speak(text, raise_on_error=True)
        log.info("[TIMING] TTS (spoken)     : %.2fs", time.time() - _t_tts)
    except TTSError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"spoken": bool(ok), "text": text, "engine": provider_name()}


@app.post("/api/tts/audio")
def text_to_speech_audio(request: SpeakRequest):
    """
    Returns synthesised speech AUDIO for the browser to play.

    This is the architecture the dashboard uses:
        Python TTS -> audio bytes -> HTTP -> <audio> -> operator's speaker

    It does not depend on the server having a usable output device, and avoids
    the browser's speechSynthesis, which fails silently in several situations.
    """
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    from agent.voice import TTSError, synthesize

    try:
        path, mime = synthesize(text)
    except TTSError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.warning("TTS synthesis failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"TTS failed: {exc}")

    try:
        with open(path, "rb") as handle:
            audio_bytes = handle.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return Response(
        content=audio_bytes,
        media_type=mime,
        headers={
            "Cache-Control": "no-store",
            "X-TTS-Engine": "elevenlabs" if config.ELEVENLABS_API_KEY else "pyttsx3",
        },
    )


@app.get("/api/tts/diagnostics")
def tts_diagnostics() -> Dict[str, Any]:
    """TTS health: provider, installed voices, volume. No secrets."""
    from agent.voice import diagnostics

    return diagnostics()


# --- camera -------------------------------------------------------------

@app.get("/api/camera/stream")
def camera_stream(request: Request):
    """
    MJPEG stream for <img src="/api/camera/stream">.

    `request` is passed through so the generator stops when the browser
    disconnects -- otherwise every reload leaked a stream until the browser hit
    its per-origin connection cap and the video stopped appearing.
    """
    if camera_service.status in ("unavailable", "not started"):
        raise HTTPException(
            status_code=503,
            detail=camera_service.error or "Camera is not available.",
        )
    return StreamingResponse(
        camera_service.mjpeg_frames(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/camera/snapshot")
def camera_snapshot():
    jpeg = camera_service.latest_jpeg()
    if jpeg is None:
        raise HTTPException(
            status_code=503,
            detail=camera_service.error or "No frame available from the camera.",
        )
    return StreamingResponse(io.BytesIO(jpeg), media_type="image/jpeg")


@app.get("/api/camera/status")
def camera_status() -> Dict[str, Any]:
    return {
        "status": camera_service.status,
        "source": camera_service.source,
        "fps": round(camera_service.fps, 1),
        "resolution": list(camera_service.resolution),
        "error": camera_service.error,
        "has_frame": camera_service.latest_jpeg() is not None,
    }


# --- site camera 2 preview (reuses existing site_camera_service directly) ---

@app.get("/api/camera2/stream")
async def camera2_stream(request: Request):
    """MJPEG stream for Site Camera 2 reusing site_camera_service.latest_jpeg()."""
    if site_camera_service.status in ("unavailable", "not started", "disabled", "stopped"):
        raise HTTPException(
            status_code=503,
            detail=site_camera_service.error or f"Site camera (Camera 2) is {site_camera_service.status}.",
        )

    async def gen():
        boundary = b"--frame"
        last_sent = None
        interval = 1.0 / max(1, getattr(config, "SITE_CAMERA_FPS", 15))
        while True:
            if await request.is_disconnected():
                break
            jpeg = site_camera_service.latest_jpeg()
            if jpeg is None:
                await asyncio.sleep(0.1)
                continue
            if jpeg is not last_sent:
                last_sent = jpeg
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(interval)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/camera2/snapshot")
def camera2_snapshot():
    jpeg = site_camera_service.latest_jpeg()
    if jpeg is None:
        raise HTTPException(
            status_code=503,
            detail=site_camera_service.error or "No frame available from site camera (Camera 2).",
        )
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/camera2/detections")
def camera2_detections() -> Dict[str, Any]:
    """Returns structured instance segmentation detections from Camera 2."""
    return ms.get_site_detections()


# --- Phase 2C: Shift & Production Data API -----------------------------

class CreateOperatorRequest(BaseModel):
    operator_id: Optional[str] = None
    name: str
    employee_code: Optional[str] = None
    role: str = "operator"
    active: int = 1


class StartShiftRequest(BaseModel):
    operator_id: str
    shift_type: str  # "Morning", "Afternoon", "Night"
    shift_date: Optional[str] = None
    line_id: str = "line_1"
    target_walls: int = 0
    summary_notes: Optional[str] = None


class EndShiftRequest(BaseModel):
    status: str = "closed"  # "closed", "flagged", "abandoned"
    summary_notes: Optional[str] = None


class RecordProductionRequest(BaseModel):
    shift_id: int
    wall_number: int
    job_id: Optional[int] = None
    print_file: Optional[str] = None
    wall_height: float = 0.0
    segment_layers: int = 0
    layers_completed: int = 0
    status: str = "in_progress"  # in_progress, completed, aborted, defect
    is_continuation_of: Optional[int] = None
    notes: Optional[str] = None


class LogIssueRequest(BaseModel):
    shift_id: int
    issue_type: str
    description: str
    wall_number: Optional[int] = None
    cause: Optional[str] = None
    action_taken: Optional[str] = None
    result: Optional[str] = None
    severity: str = "warning"
    downtime_minutes: float = 0.0


class InitiateHandoverRequest(BaseModel):
    from_shift_id: int
    line_id: str = "line_1"
    completed_work: Optional[str] = None
    pending_work: Optional[str] = None
    open_issues: Optional[str] = None
    notes: Optional[str] = None


class CompleteHandoverRequest(BaseModel):
    to_shift_id: int
    notes: Optional[str] = None
    close_from_shift: bool = True


@app.get("/api/operators")
def get_operators(active_only: bool = True) -> Dict[str, Any]:
    """Lists operators, optionally filtering by active."""
    return {"operators": shift_store.list_operators(active_only=active_only)}


@app.post("/api/operators")
def create_operator(req: CreateOperatorRequest) -> Dict[str, Any]:
    """Creates a new operator."""
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail="Employee name is required.")
        
    op_id = req.operator_id
    if not op_id:
        clean_name = req.name.strip().upper().replace(" ", "_")
        import uuid
        short_id = str(uuid.uuid4())[:6].upper()
        op_id = f"OP_{clean_name}_{short_id}"

    try:
        op_id = shift_store.create_operator(
            operator_id=op_id,
            name=req.name,
            employee_code=req.employee_code,
            role=req.role,
            active=req.active,
        )
        return {
            "operator_id": op_id,
            "name": req.name,
            "employee_code": req.employee_code,
            "role": req.role,
            "active": True if req.active == 1 else False
        }
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Employee code already exists.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/api/shifts/active")
def get_active_shift(line_id: str = "line_1") -> Dict[str, Any]:
    """Gets the currently active shift on the specified line."""
    active = shift_store.get_active_shift_for_line(line_id=line_id)
    if not active:
        return {"active": False, "shift": None}
    op = shift_store.get_operator(active["operator_id"])
    return {
        "active": True,
        "shift": active,
        "operator": op,
    }


@app.post("/api/shifts/start")
def start_shift(req: StartShiftRequest) -> Dict[str, Any]:
    """Starts a new shift on a line. Blocks if line already has an active shift."""
    existing = shift_store.get_active_shift_for_line(req.line_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Line '{req.line_id}' already has an active shift (ID {existing['shift_id']}). Close it first.",
        )
    op = shift_store.get_operator(req.operator_id)
    if not op or not op.get("active"):
        raise HTTPException(status_code=400, detail=f"Operator '{req.operator_id}' not found or is inactive.")
    shift_date = req.shift_date or datetime.date.today().isoformat()
    try:
        shift_id = shift_store.start_shift(
            operator_id=req.operator_id,
            shift_date=shift_date,
            shift_type=req.shift_type,
            line_id=req.line_id,
            target_walls=req.target_walls,
            summary_notes=req.summary_notes,
        )
        return {
            "success": True,
            "shift_id": shift_id,
            "line_id": req.line_id,
            "operator_id": req.operator_id,
            "shift_type": req.shift_type,
            "shift_date": shift_date,
        }
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail=f"Unique active shift constraint violated: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/shifts/{shift_id}/end")
def end_shift(shift_id: int, req: EndShiftRequest) -> Dict[str, Any]:
    """Ends a shift and updates its status."""
    shift = shift_store.get_shift(shift_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"Shift {shift_id} not found.")
    try:
        ok = shift_store.end_shift(
            shift_id=shift_id,
            status=req.status,
            summary_notes=req.summary_notes,
        )
        return {"success": ok, "shift_id": shift_id, "status": req.status}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/shifts")
def list_shifts(
    shift_date: Optional[str] = None,
    shift_type: Optional[str] = None,
    operator_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Lists shifts matching criteria."""
    shifts = shift_store.list_shifts(
        shift_date=shift_date,
        shift_type=shift_type,
        operator_id=operator_id,
        status=status,
        limit=limit,
    )
    return {"shifts": shifts, "count": len(shifts)}


@app.post("/api/production")
def record_production(req: RecordProductionRequest) -> Dict[str, Any]:
    """Records production progress for a wall within a shift."""
    shift = shift_store.get_shift(req.shift_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"Shift {req.shift_id} not found.")
    try:
        prod_id = shift_store.record_production(
            shift_id=req.shift_id,
            wall_number=req.wall_number,
            job_id=req.job_id,
            print_file=req.print_file,
            wall_height=req.wall_height,
            segment_layers=req.segment_layers,
            layers_completed=req.layers_completed,
            status=req.status,
            is_continuation_of=req.is_continuation_of,
            notes=req.notes,
        )
        return {"success": True, "production_id": prod_id}
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail=f"Unique constraint (shift_id, wall_number) violated: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/production/shift/{shift_id}")
def get_shift_production(shift_id: int) -> Dict[str, Any]:
    """Lists all production records for a shift."""
    records = shift_store.get_shift_production(shift_id)
    return {"shift_id": shift_id, "production_records": records, "count": len(records)}


@app.post("/api/issues")
def log_issue(req: LogIssueRequest) -> Dict[str, Any]:
    """Logs an issue against a shift."""
    shift = shift_store.get_shift(req.shift_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"Shift {req.shift_id} not found.")
    try:
        issue_id = shift_store.log_issue(
            shift_id=req.shift_id,
            issue_type=req.issue_type,
            description=req.description,
            wall_number=req.wall_number,
            cause=req.cause,
            action_taken=req.action_taken,
            result=req.result,
            severity=req.severity,
            downtime_minutes=req.downtime_minutes,
        )
        return {"success": True, "issue_id": issue_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/issues/shift/{shift_id}")
def get_shift_issues(shift_id: int) -> Dict[str, Any]:
    """Lists all issues reported for a shift."""
    issues = shift_store.list_shift_issues(shift_id=shift_id)
    return {"shift_id": shift_id, "issues": issues, "count": len(issues)}


@app.post("/api/handovers/initiate")
def initiate_handover(req: InitiateHandoverRequest) -> Dict[str, Any]:
    """Initiates an outgoing shift handover."""
    from_shift = shift_store.get_shift(req.from_shift_id)
    if not from_shift:
        raise HTTPException(status_code=404, detail=f"From-shift {req.from_shift_id} not found.")
    try:
        handover_id = shift_store.create_handover(
            from_shift_id=req.from_shift_id,
            line_id=req.line_id,
            completed_work=req.completed_work,
            pending_work=req.pending_work,
            open_issues=req.open_issues,
            notes=req.notes,
        )
        return {"success": True, "handover_id": handover_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/handovers/{handover_id}/complete")
def complete_handover(handover_id: int, req: CompleteHandoverRequest) -> Dict[str, Any]:
    """Completes an initiated handover by linking to the incoming shift."""
    h = shift_store.get_handover(handover_id)
    if not h:
        raise HTTPException(status_code=404, detail=f"Handover {handover_id} not found.")
    to_shift = shift_store.get_shift(req.to_shift_id)
    if not to_shift:
        raise HTTPException(status_code=404, detail=f"To-shift {req.to_shift_id} not found.")
    try:
        ok = shift_store.complete_handover(
            handover_id=handover_id,
            to_shift_id=req.to_shift_id,
            notes=req.notes,
            close_from_shift=req.close_from_shift,
        )
        return {"success": ok, "handover_id": handover_id, "to_shift_id": req.to_shift_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

# ===========================================================================
# Attendance / Shift Check-in API
# ===========================================================================

class AttendanceCheckInRequest(BaseModel):
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None

class AttendanceCheckOutRequest(BaseModel):
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None

@app.post("/api/v1/attendance/check-in")
def attendance_check_in(req: AttendanceCheckInRequest) -> Dict[str, Any]:
    if not req.employee_id and not req.employee_name:
        raise HTTPException(status_code=400, detail="Must provide either employee_id or employee_name.")
    
    op = None
    if req.employee_id:
        op = shift_store.get_operator(req.employee_id)
        if not op:
            raise HTTPException(status_code=400, detail=f"Operator ID '{req.employee_id}' not found in the dataset.")
    elif req.employee_name:
        # Resolve by name (voice model fallback)
        operators = shift_store.list_operators()
        for existing_op in operators:
            if existing_op["name"].lower() == req.employee_name.lower():
                op = existing_op
                break
        if not op:
            raise HTTPException(status_code=400, detail=f"Employee '{req.employee_name}' not found in the dataset.")
            
    import datetime
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    try:
        record = shift_store.create_attendance_record(
            operator_id=op["operator_id"],
            employee_name=op["name"],
            attendance_date=today_str,
        )
        return record
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("Failed to create attendance record: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/attendance/check-out")
def attendance_check_out(req: AttendanceCheckOutRequest) -> Dict[str, Any]:
    if not req.employee_id and not req.employee_name:
        raise HTTPException(status_code=400, detail="Must provide either employee_id or employee_name.")
        
    try:
        record = shift_store.checkout_attendance_record(
            operator_id=req.employee_id,
            employee_name=req.employee_name,
        )
        return record
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("Failed to checkout attendance record: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/attendance")
def get_attendance(date: Optional[str] = None) -> List[Dict[str, Any]]:
    day = date or shift_store.local_today()
    return shift_store.list_attendance_records(date_filter=day)


# ===========================================================================
# Employee Wall Production Tracking API
# ===========================================================================

class RecordWallProductionRequest(BaseModel):
    operator_id: str
    quantity: int
    job_id: Optional[str] = None
    shift_id: Optional[int] = None
    notes: Optional[str] = None
    request_id: Optional[str] = None
    recorded_by: Optional[str] = "api"


class CorrectWallProductionRequest(BaseModel):
    record_id: str
    corrected_quantity: int
    reason: Optional[str] = None
    corrected_by: Optional[str] = "api"


@app.post("/api/v1/production/walls")
def record_wall_production_endpoint(req: RecordWallProductionRequest) -> Dict[str, Any]:
    """Records completed walls by an operator."""
    if req.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be a positive integer greater than zero.")
    try:
        rec = shift_store.record_wall_production(
            operator_id=req.operator_id,
            quantity=req.quantity,
            job_id=req.job_id,
            shift_id=req.shift_id,
            notes=req.notes,
            request_id=req.request_id,
            recorded_by=req.recorded_by or "api",
        )
        return {"success": True, "record": rec}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("Failed to record wall production: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


@app.get("/api/v1/production/walls")
def get_wall_production_endpoint(
    operator_id: Optional[str] = None,
    scope: str = Query("today", description="Scope: today, this_week, job, active_shift, all_time, date_range"),
    job_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    shift_id: Optional[int] = None,
    include_records: bool = Query(True, description="When true, returns the list of individual records alongside totals."),
) -> Dict[str, Any]:
    """Queries wall production totals and records."""
    try:
        totals = shift_store.get_wall_production_total(
            operator_id=operator_id,
            scope=scope,
            job_id=job_id,
            date_from=date_from,
            date_to=date_to,
            shift_id=shift_id,
        )
        records = []
        if include_records:
            records = shift_store.list_wall_production_records(
                operator_id=operator_id,
                job_id=job_id,
                date_from=date_from,
                date_to=date_to,
                shift_id=shift_id,
                include_corrected=False,
                limit=100,
            )
        return {
            "success": True,
            "scope": scope,
            "total_quantity": totals["total_quantity"],
            "record_count": totals["record_count"],
            "by_operator": totals.get("by_operator", []),
            "records": records,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("Failed to query wall production: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


@app.post("/api/v1/production/walls/correct")
def correct_wall_production_endpoint(req: CorrectWallProductionRequest) -> Dict[str, Any]:
    """Corrects a previous wall production record with an audit log."""
    if req.corrected_quantity <= 0:
        raise HTTPException(status_code=400, detail="Corrected quantity must be a positive integer greater than zero.")
    try:
        result = shift_store.correct_wall_production(
            original_record_id=req.record_id,
            corrected_quantity=req.corrected_quantity,
            reason=req.reason or "Corrected via API",
            corrected_by=req.corrected_by or "api",
        )
        return {"success": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("Failed to correct wall production record: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


# ===========================================================================
# Persistent Incidents and Incident Image Retrieval Endpoints
# ===========================================================================

class IncidentVerificationRequest(BaseModel):
    confirmed_root_cause: str
    recommended_action: str
    verified_by: str


def _detect_image_format(data: bytes) -> Optional[Tuple[str, str]]:
    """
    Returns (mime_type, file_extension) if valid image magic bytes are present, else None.
    Rejects text/html, empty data, or arbitrary non-image binaries.
    """
    if not data or len(data) < 4:
        return None
    # JPEG magic bytes: FF D8 FF
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return ("image/jpeg", ".jpg")
    # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return ("image/png", ".png")
    # WebP magic bytes: RIFF....WEBP
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ("image/webp", ".webp")
    # GIF magic bytes: GIF87a or GIF89a
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return ("image/gif", ".gif")
    return None


def _format_image_download_filename(
    image_id: str,
    record: Optional[Dict[str, Any]],
    ext: str,
) -> str:
    """Builds a meaningful, descriptive filename for download actions."""
    clean_id = (image_id or "image").replace("IMG-", "").lower()
    prefix = "defect"
    if record:
        meta = record.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        elif not isinstance(meta, dict):
            meta = {}

        orig = (record.get("original_name") or "").lower()
        if meta.get("is_annotated") or "annotated" in orig:
            prefix = "annotation"
        elif meta.get("category") or "ref_" in orig or "reference" in orig:
            prefix = "reference"
        elif "defect" in orig or "incident" in orig:
            prefix = "defect"

    return f"kelvin6k_{prefix}_{clean_id}{ext}"


@app.get("/api/incidents/images/{image_id}")
@app.head("/api/incidents/images/{image_id}")
def get_incident_image_bytes(
    image_id: str,
    download: bool = Query(False, description="When true, sets Content-Disposition to attachment to force file download."),
):
    """
    Safely retrieves raw image bytes for a persisted incident image.
    Validates image ID format to prevent path traversal, verifies DB record existence,
    inspects magic bytes to ensure genuine image binary (never HTML or corrupted binary),
    and sets appropriate Content-Type and Content-Disposition headers.
    """
    clean_id = (image_id or "").strip()
    if not clean_id or not re.match(r"^[A-Za-z0-9_-]+$", clean_id):
        raise HTTPException(status_code=400, detail="Invalid image identifier format.")

    # 1. Resolve image metadata from SQLite
    rec = incident_store.get_incident_image(clean_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Incident image '{clean_id}' not found.")

    # 2. Retrieve image data from disk
    result = incident_store.get_image_data(clean_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Image file for '{clean_id}' not found on storage disk.")

    data, recorded_mime, orig_filename = result

    # 3. Validate image magic bytes -- reject HTML or corrupted binaries
    fmt = _detect_image_format(data)
    if not fmt:
        raise HTTPException(
            status_code=500,
            detail=f"Image data for '{clean_id}' is corrupted or does not contain valid image magic bytes.",
        )

    mime_type, ext = fmt
    download_filename = _format_image_download_filename(clean_id, rec, ext)
    disposition_type = "attachment" if download else "inline"

    return Response(
        content=data,
        media_type=mime_type,
        headers={
            "Content-Disposition": f'{disposition_type}; filename="{download_filename}"',
            "Content-Type": mime_type,
            "Content-Length": str(len(data)),
            "Cache-Control": "public, max-age=86400",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/incidents/images/{image_id}/download")
def download_incident_image_bytes(image_id: str):
    """
    Dedicated download route that always serves image bytes with
    Content-Disposition: attachment and meaningful filename.
    """
    return get_incident_image_bytes(image_id=image_id, download=True)


@app.get("/api/incidents")
def list_incidents(
    status: Optional[str] = None,
    defect_category: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Lists recorded machine problem incidents ordered by recency."""
    return incident_store.list_incidents(status=status, defect_category=defect_category, limit=limit)


@app.get("/api/incidents/references")
def list_component_references(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists verified robot component reference cards and specifications."""
    return incident_store.list_component_references(category=category)


@app.get("/api/incidents/references/{reference_id}")
def get_component_reference(reference_id: str) -> Dict[str, Any]:
    """Retrieves metadata and specifications for a specific component reference."""
    ref = incident_store.get_component_reference(reference_id)
    if not ref:
        raise HTTPException(status_code=404, detail=f"Component reference '{reference_id}' not found.")
    return ref


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: str) -> Dict[str, Any]:
    """Retrieves full details of a specific incident."""
    inc = incident_store.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
    return inc


@app.post("/api/incidents/{incident_id}/verify")
def verify_incident(incident_id: str, req: IncidentVerificationRequest) -> Dict[str, Any]:
    """Applies authorized engineer verification to a recorded incident."""
    updated = incident_store.verify_incident(
        incident_id=incident_id,
        confirmed_root_cause=req.confirmed_root_cause,
        recommended_action=req.recommended_action,
        verified_by=req.verified_by,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
    return updated

