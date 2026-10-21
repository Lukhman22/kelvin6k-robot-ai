import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from agent import config
from agent.sensors import MachineReading, SensorReading
from agent.vision import VisionAnalysis, VisionEvent


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    key: str  # stable id used for cooldown de-duplication
    level: AlertLevel
    message: str


class RuleEngine:
    """
    Turns raw sensor/vision data into operator-facing Alerts.

    Two kinds of alerts:
      - Timer-based reminders (clean pipe, maintenance) that fire once an
        interval has elapsed.
      - Condition-based warnings (temperature, height, flow, vision defects,
        machine errors) that fire when a threshold is crossed, throttled by
        a cooldown so the same warning isn't repeated every poll cycle.
    """

    def __init__(self):
        now = time.time()
        self._last_pipe_clean_reminder = now
        self._last_maintenance_reminder = now
        self._last_fired: Dict[str, float] = {}

    def _should_fire(self, key: str) -> bool:
        last = self._last_fired.get(key)
        if last is None or (time.time() - last) >= config.ALERT_COOLDOWN_SECONDS:
            self._last_fired[key] = time.time()
            return True
        return False

    def evaluate(self, reading: SensorReading, vision_event: Optional[VisionEvent]) -> List[Alert]:
        alerts: List[Alert] = []
        now = time.time()

        # --- Timer-based reminders ---
        if now - self._last_pipe_clean_reminder >= config.PIPE_CLEAN_INTERVAL_SECONDS:
            self._last_pipe_clean_reminder = now
            alerts.append(Alert(
                key="pipe_clean_reminder",
                level=AlertLevel.INFO,
                message="Reminder: please clean the printing pipe to keep material flowing smoothly.",
            ))

        if now - self._last_maintenance_reminder >= config.MAINTENANCE_INTERVAL_SECONDS:
            self._last_maintenance_reminder = now
            alerts.append(Alert(
                key="maintenance_reminder",
                level=AlertLevel.INFO,
                message="Reminder: it is time for the scheduled machine maintenance check.",
            ))

        # --- Safety / condition-based warnings ---
        if reading.nozzle_temp_c > config.NOZZLE_TEMP_MAX_C and self._should_fire("nozzle_temp_high"):
            alerts.append(Alert(
                key="nozzle_temp_high",
                level=AlertLevel.CRITICAL,
                message=(
                    f"Warning: nozzle temperature is {reading.nozzle_temp_c} degrees, "
                    f"above the safe limit of {config.NOZZLE_TEMP_MAX_C}. Reduce temperature immediately."
                ),
            ))
        elif reading.nozzle_temp_c < config.NOZZLE_TEMP_MIN_C and self._should_fire("nozzle_temp_low"):
            alerts.append(Alert(
                key="nozzle_temp_low",
                level=AlertLevel.WARNING,
                message=(
                    f"Warning: nozzle temperature is {reading.nozzle_temp_c} degrees, "
                    f"below the recommended minimum of {config.NOZZLE_TEMP_MIN_C}."
                ),
            ))

        if reading.bed_temp_c > config.BED_TEMP_MAX_C and self._should_fire("bed_temp_high"):
            alerts.append(Alert(
                key="bed_temp_high",
                level=AlertLevel.CRITICAL,
                message=(
                    f"Warning: bed temperature is {reading.bed_temp_c} degrees, "
                    f"above the safe limit of {config.BED_TEMP_MAX_C}. Check the heater immediately."
                ),
            ))

        height_deviation = abs(reading.expected_height_mm - reading.print_height_mm)
        if height_deviation > config.HEIGHT_TOLERANCE_MM and self._should_fire("height_deviation"):
            alerts.append(Alert(
                key="height_deviation",
                level=AlertLevel.WARNING,
                message=(
                    f"Warning: print height has drifted from the expected value by "
                    f"{height_deviation:.2f} millimeters. Check for layer skipping."
                ),
            ))

        if reading.material_flow_percent < config.MATERIAL_FLOW_MIN_PERCENT and self._should_fire("material_flow_low"):
            alerts.append(Alert(
                key="material_flow_low",
                level=AlertLevel.WARNING,
                message=(
                    f"Warning: material flow has dropped to {reading.material_flow_percent} percent. "
                    f"Check for a clog or filament tangle."
                ),
            ))

        if reading.machine_state == "error" and self._should_fire("machine_error"):
            alerts.append(Alert(
                key="machine_error",
                level=AlertLevel.CRITICAL,
                message="Critical: the machine is reporting an error state. Pause the print and inspect the machine now.",
            ))

        if vision_event is not None and self._should_fire(f"vision_{vision_event.defect_type}"):
            alerts.append(Alert(
                key=f"vision_{vision_event.defect_type}",
                level=AlertLevel.CRITICAL,
                message=(
                    f"Warning: possible {vision_event.defect_type} detected on camera "
                    f"with {int(vision_event.confidence * 100)} percent confidence. Please check the print."
                ),
            ))

        return alerts

    def evaluate_vision(self, analysis: VisionAnalysis) -> List[Alert]:
        """
        Camera-derived alerts. Every rule here is guarded on the measurement
        actually existing -- an unmeasurable value never produces an alert.
        Shares the same cooldown map as evaluate(), so no duplicate alerts.
        """
        alerts: List[Alert] = []

        width_mm = analysis.layers.layer_width_mm
        if width_mm is not None:
            if width_mm > config.LAYER_WIDTH_MAX_MM and self._should_fire("layer_width_high"):
                alerts.append(Alert(
                    key="layer_width_high",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Warning: measured layer width is {width_mm} millimeters, above the "
                        f"limit of {config.LAYER_WIDTH_MAX_MM}. Possible over-extrusion."
                    ),
                ))
            elif width_mm < config.LAYER_WIDTH_MIN_MM and self._should_fire("layer_width_low"):
                alerts.append(Alert(
                    key="layer_width_low",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Warning: measured layer width is {width_mm} millimeters, below the "
                        f"minimum of {config.LAYER_WIDTH_MIN_MM}. Possible under-extrusion."
                    ),
                ))

        score = analysis.quality.score_percent
        if score is not None and score < config.PRINT_QUALITY_MIN_PERCENT:
            if self._should_fire("print_quality_low"):
                alerts.append(Alert(
                    key="print_quality_low",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Warning: measured print quality has dropped to {score} percent. "
                        f"Please inspect the print."
                    ),
                ))

        for event in analysis.events:
            key = f"vision_{event.defect_type}"
            if self._should_fire(key):
                alerts.append(Alert(
                    key=key,
                    level=AlertLevel.CRITICAL,
                    message=(
                        f"Warning: {event.defect_type} detected on camera with "
                        f"{int(event.confidence * 100)} percent confidence. Please check the print."
                    ),
                ))

        return alerts

    def evaluate_machine(self, machine: MachineReading) -> List[Alert]:
        """
        Machine-telemetry alerts (pump, accelerator, drives).

        Every channel is optional: when a channel is not connected its value is
        None and no alert is produced. This deliberately stays silent rather
        than warning about equipment it cannot actually observe.
        """
        alerts: List[Alert] = []

        if machine.pump_running is False and self._should_fire("pump_stopped"):
            alerts.append(Alert(
                key="pump_stopped",
                level=AlertLevel.CRITICAL,
                message="Critical: the material pump has stopped. Check the pump immediately.",
            ))

        if machine.accelerator_running is False and self._should_fire("accelerator_stopped"):
            alerts.append(Alert(
                key="accelerator_stopped",
                level=AlertLevel.CRITICAL,
                message="Critical: the accelerator has stopped. Check the accelerator immediately.",
            ))

        if machine.error_state and self._should_fire("machine_error_state"):
            alerts.append(Alert(
                key="machine_error_state",
                level=AlertLevel.CRITICAL,
                message=f"Critical: the machine reports an error state: {machine.error_state}.",
            ))

        return alerts
