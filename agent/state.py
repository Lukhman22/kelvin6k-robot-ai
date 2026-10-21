import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from agent.rules import Alert
from agent.sensors import MachineReading, SensorReading
from agent.vision import VisionAnalysis, VisionEvent

_UNAVAILABLE = "unavailable"


@dataclass
class _AlertLogEntry:
    alert: Alert
    timestamp: float


def _value_or_unavailable(value, reason: str):
    """Renders a missing value as an explicit unavailability marker, never a guess."""
    if value is None:
        return {"status": _UNAVAILABLE, "reason": reason}
    return value


class PrinterState:
    """
    Thread-safe fusion of everything the agent knows: camera vision analysis,
    machine/process sensors, and the alert log.

    Writers: the monitoring loop (sensors + alerts) and the vision loop (camera).
    Readers: the voice assistant and the dashboard.

    This is the single source of truth -- no other module keeps its own copy.
    """

    def __init__(self, max_alert_log: int = 20):
        self._lock = threading.Lock()
        self._reading: Optional[SensorReading] = None
        self._machine: Optional[MachineReading] = None
        self._vision: Optional[VisionAnalysis] = None
        self._vision_event: Optional[VisionEvent] = None
        self._print_observation = None
        self._camera_status: str = "not started"
        self._camera_fps: float = 0.0
        self._last_question: Optional[str] = None
        self._last_answer: Optional[str] = None
        self._alert_log: List[_AlertLogEntry] = []
        self._max_alert_log = max_alert_log

    # --- writers -------------------------------------------------------

    def update(
        self,
        reading: SensorReading,
        vision_event: Optional[VisionEvent],
        new_alerts: List[Alert],
        machine: Optional[MachineReading] = None,
    ) -> None:
        with self._lock:
            self._reading = reading
            self._vision_event = vision_event
            if machine is not None:
                self._machine = machine
            now = time.time()
            for alert in new_alerts:
                self._alert_log.append(_AlertLogEntry(alert=alert, timestamp=now))
            if len(self._alert_log) > self._max_alert_log:
                self._alert_log = self._alert_log[-self._max_alert_log :]

    def update_vision(self, analysis: VisionAnalysis, camera_fps: float = 0.0) -> None:
        with self._lock:
            self._vision = analysis
            self._camera_fps = camera_fps
            self._camera_status = "live"

    def update_print_observation(self, observation) -> None:
        """Stores the debounced print status from PrintMonitor."""
        with self._lock:
            self._print_observation = observation

    def latest_print_observation(self):
        with self._lock:
            return self._print_observation

    def set_camera_status(self, status: str) -> None:
        with self._lock:
            self._camera_status = status

    def record_interaction(self, question: str, answer: str) -> None:
        with self._lock:
            self._last_question = question
            self._last_answer = answer

    def add_alerts(self, new_alerts: List[Alert]) -> None:
        with self._lock:
            now = time.time()
            for alert in new_alerts:
                self._alert_log.append(_AlertLogEntry(alert=alert, timestamp=now))
            if len(self._alert_log) > self._max_alert_log:
                self._alert_log = self._alert_log[-self._max_alert_log :]

    # --- readers -------------------------------------------------------

    def latest_vision(self) -> Optional[VisionAnalysis]:
        with self._lock:
            return self._vision

    def latest_reading(self) -> Optional[SensorReading]:
        with self._lock:
            return self._reading

    def dashboard_view(self) -> dict:
        """Compact view for the on-screen dashboard."""
        with self._lock:
            reading = self._reading
            machine = self._machine
            vision = self._vision
            observation = self._print_observation
            camera_status = self._camera_status
            camera_fps = self._camera_fps
            question = self._last_question
            answer = self._last_answer
            recent = list(self._alert_log[-5:])

        return {
            "camera_status": camera_status,
            "camera_fps": camera_fps,
            "vision": vision,
            "print_observation": observation,
            "reading": reading,
            "machine": machine,
            "last_question": question,
            "last_answer": answer,
            "alerts": [(e.alert, e.timestamp) for e in recent],
        }

    def snapshot(self) -> dict:
        """
        Full JSON-serializable state handed to the voice assistant.

        Anything not actually measured appears as
        {"status": "unavailable", "reason": "..."} so the AI can say so instead
        of inventing a number.
        """
        with self._lock:
            reading = self._reading
            machine = self._machine
            vision = self._vision
            observation = self._print_observation
            camera_status = self._camera_status
            camera_fps = self._camera_fps
            recent_alerts = list(self._alert_log[-5:])

        snapshot: Dict = {
            "data_source_notice": (
                "Values marked SIMULATED come from the built-in simulator, not the real "
                "machine. Values marked unavailable are genuinely not measurable right "
                "now -- never substitute a guess for them."
            ),
            "camera": {
                "status": camera_status,
                "fps": round(camera_fps, 1) if camera_fps else 0.0,
            },
        }

        # --- debounced print status (the authoritative answer to "is it
        # printing / is a layer being laid down / how wide is it") ---
        if observation is None:
            snapshot["print_status"] = {
                "status": _UNAVAILABLE,
                "reason": "The camera pipeline has not produced a print observation yet.",
            }
        else:
            snapshot["print_status"] = observation.to_dict()
            snapshot["print_status"]["age_seconds"] = round(time.time() - observation.timestamp, 1)
            snapshot["print_status"]["note"] = (
                "These values are debounced over several frames and confidence-checked. "
                "They are the authoritative answer for printing/layer questions."
            )

        # --- vision-derived (camera) ---
        if vision is None:
            snapshot["vision"] = {
                "status": _UNAVAILABLE,
                "reason": "The camera pipeline has not produced an analysis yet.",
            }
        else:
            layers = vision.layers
            walls = vision.walls
            quality = vision.quality
            snapshot["vision"] = {
                "source": "CAMERA",
                "age_seconds": round(time.time() - vision.timestamp, 1),
                "layers": {
                    "layers_detected": layers.detected,
                    "visible_layer_bands": _value_or_unavailable(
                        layers.visible_layer_bands,
                        layers.unavailable.get("layers", "No layer boundaries detected."),
                    ),
                    "note_on_layer_count": (
                        "This is the number of layer bands currently VISIBLE in the camera "
                        "frame -- not the total number of layers printed."
                    ),
                    "current_layer_number": {
                        "status": _UNAVAILABLE,
                        "reason": layers.unavailable.get("current_layer_number", ""),
                    },
                    "total_layers": {
                        "status": _UNAVAILABLE,
                        "reason": layers.unavailable.get("total_layers", ""),
                    },
                    "print_progress_percent": {
                        "status": _UNAVAILABLE,
                        "reason": layers.unavailable.get("print_progress_percent", ""),
                    },
                    "layer_height_px": _value_or_unavailable(layers.layer_height_px, "Not measured."),
                    "layer_height_mm": _value_or_unavailable(
                        layers.layer_height_mm,
                        layers.unavailable.get("millimetre_measurements", "Not measured."),
                    ),
                    "layer_width_px": _value_or_unavailable(
                        layers.layer_width_px,
                        layers.unavailable.get("layer_width", "Not measured."),
                    ),
                    "layer_width_mm": _value_or_unavailable(
                        layers.layer_width_mm,
                        layers.unavailable.get("millimetre_measurements", "Not measured."),
                    ),
                    "layer_consistency_percent": _value_or_unavailable(
                        layers.layer_consistency_percent, "Not enough boundaries measured."
                    ),
                    "layer_width_trend": _value_or_unavailable(
                        layers.layer_width_trend,
                        "Not enough layer-width samples collected yet.",
                    ),
                },
                "walls": {
                    "walls_detected": walls.detected,
                    "outer_wall_detected": walls.outer_wall_detected,
                    "inner_wall_detected": walls.inner_wall_detected,
                    "wall_count": {
                        "status": _UNAVAILABLE,
                        "reason": walls.unavailable.get("wall_count", ""),
                    },
                    "visible_wall_bands": _value_or_unavailable(
                        walls.visible_wall_bands, "No wall bands measured."
                    ),
                    "wall_thickness_px": _value_or_unavailable(
                        walls.wall_thickness_px, "Not measured."
                    ),
                    "wall_thickness_mm": _value_or_unavailable(
                        walls.wall_thickness_mm,
                        walls.unavailable.get("millimetre_measurements", "Not measured."),
                    ),
                    "wall_continuity_percent": _value_or_unavailable(
                        walls.wall_continuity_percent, "Not measured."
                    ),
                    "gap_count": _value_or_unavailable(walls.gap_count, "Not measured."),
                    "defects": walls.defects,
                },
                "print_quality": {
                    "score_percent": _value_or_unavailable(
                        quality.score_percent,
                        quality.unavailable.get("print_quality", "Not scorable."),
                    ),
                    "status": quality.status,
                    "issues": quality.issues,
                    "components": quality.components,
                },
                "observed_defects": [
                    {"defect": e.defect_type, "confidence": e.confidence} for e in vision.events
                ],
            }

        # --- process sensors (currently SIMULATED) ---
        if reading is None:
            snapshot["process_sensors"] = {
                "status": _UNAVAILABLE,
                "reason": "The monitoring loop has not taken a reading yet.",
            }
        else:
            snapshot["process_sensors"] = {
                "source": reading.source,
                "warning": (
                    "These values are SIMULATED, not from the real machine."
                    if reading.source == "SIMULATED"
                    else ""
                ),
                "nozzle_temp_c": reading.nozzle_temp_c,
                "bed_temp_c": reading.bed_temp_c,
                "print_height_mm": reading.print_height_mm,
                "expected_height_mm": reading.expected_height_mm,
                "material_flow_percent": reading.material_flow_percent,
                "machine_state": reading.machine_state,
            }

        # --- machine telemetry (pump, accelerator, drives, job info) ---
        if machine is None:
            machine = MachineReading()
        reason = machine.unavailable_reason
        snapshot["machine_telemetry"] = {
            "source": machine.source,
            "pump_running": _value_or_unavailable(machine.pump_running, reason),
            "pump_speed_rpm": _value_or_unavailable(machine.pump_speed_rpm, reason),
            "accelerator_running": _value_or_unavailable(machine.accelerator_running, reason),
            "accelerator_speed_rpm": _value_or_unavailable(machine.accelerator_speed_rpm, reason),
            "print_speed_mm_s": _value_or_unavailable(machine.print_speed_mm_s, reason),
            "feed_speed_mm_s": _value_or_unavailable(machine.feed_speed_mm_s, reason),
            "pressure_bar": _value_or_unavailable(machine.pressure_bar, reason),
            "motor_status": _value_or_unavailable(machine.motor_status, reason),
            "error_state": _value_or_unavailable(machine.error_state, reason),
            "current_layer_number": _value_or_unavailable(machine.current_layer_number, reason),
            "total_layers": _value_or_unavailable(machine.total_layers, reason),
        }

        snapshot["recent_alerts"] = [
            {
                "level": entry.alert.level.value,
                "message": entry.alert.message,
                "seconds_ago": round(time.time() - entry.timestamp, 1),
            }
            for entry in recent_alerts
        ]

        return snapshot
