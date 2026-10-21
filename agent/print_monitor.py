"""
Print-status determination and the clean query API the voice assistant uses.

This module is the SOURCE OF TRUTH for "is printing happening / is a layer being
laid down / how wide is it". The language model never decides these -- it only
reads what this module measured.

Two things make the state trustworthy rather than jumpy:

  * confidence thresholds -- a measurement below LAYER_CONFIDENCE_THRESHOLD is
    not acted upon;
  * temporal debouncing -- the state only flips after N consecutive analysed
    frames agree (LAYER_DETECT_FRAMES / LAYER_MISSING_FRAMES /
    PRINT_ACTIVE_FRAMES / PRINT_IDLE_FRAMES), so one bad frame cannot change
    the machine state.

Printing activity is inferred from frame-to-frame change inside the ROI. That is
an honest proxy for "the machine is doing something", NOT a machine-reported
state -- when a real machine link exists its status takes priority (see
`PrintMonitor.update`).
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from agent import config
from agent.vision import VisionAnalysis


class PrintStatus(str, Enum):
    PRINTING = "PRINTING"
    NO_PRINT = "NO_PRINT"
    LAYER_DETECTED = "LAYER_DETECTED"
    LAYER_MISSING = "LAYER_MISSING"
    ABNORMAL_LAYER = "ABNORMAL_LAYER"
    CAMERA_ERROR = "CAMERA_ERROR"


@dataclass
class PrintObservation:
    """One debounced, confidence-checked view of the print."""

    timestamp: float
    printing: bool
    layer_detected: bool
    status: PrintStatus
    layer_status: PrintStatus
    confidence: float
    # Raw pixel measurement and the calibrated physical value are kept strictly
    # separate -- a pixel is never reported as a millimetre.
    layer_width_px: Optional[float] = None
    layer_width: Optional[float] = None
    width_unit: str = "mm"
    width_valid: Optional[bool] = None
    motion_score: float = 0.0
    camera_connected: bool = True
    reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The pipeline output contract."""
        return {
            "printing": self.printing,
            "layer_detected": self.layer_detected,
            "layer_width": self.layer_width,
            "layer_width_px": self.layer_width_px,
            "width_unit": self.width_unit,
            "width_valid": self.width_valid,
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "layer_status": self.layer_status.value,
            "motion_score": round(self.motion_score, 2),
            "camera_connected": self.camera_connected,
            "timestamp": self.timestamp,
            "notes": self.reasons,
        }


class PrintMonitor:
    """Turns per-frame vision analysis into a stable machine state."""

    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None
        self._detect_streak = 0
        self._missing_streak = 0
        self._active_streak = 0
        self._idle_streak = 0
        self._layer_detected = False
        self._printing = False
        self._latest: Optional[PrintObservation] = None

    # --- helpers --------------------------------------------------------

    def _motion_score(self, frame: np.ndarray, roi: Optional[tuple] = None) -> float:
        """
        Mean absolute change vs the previous analysed frame, measured INSIDE the
        ROI.

        Cropping matters: material is deposited in a small part of the view, so
        averaging over the whole frame dilutes real activity below any sensible
        threshold and printing would never register.
        """
        if cv2 is None:
            return 0.0

        if roi is not None:
            x, y, w, h = roi
            frame = frame[y : y + h, x : x + w]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0
        score = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
        self._prev_gray = gray
        return score

    @staticmethod
    def _confidence(analysis: VisionAnalysis) -> float:
        """
        Detection confidence from measurement agreement, in [0, 1]:
        did we resolve enough bands, and how consistent is the layer pitch.
        """
        layers = analysis.layers
        if not layers.detected or not layers.visible_layer_bands:
            return 0.0

        band_component = min(1.0, layers.visible_layer_bands / max(1, config.MIN_LAYER_BOUNDARIES))
        consistency = layers.layer_consistency_percent
        consistency_component = (consistency / 100.0) if consistency is not None else 0.5
        return round(0.5 * band_component + 0.5 * consistency_component, 3)

    def _width_validity(self, width_mm: Optional[float]) -> Optional[bool]:
        if width_mm is None:
            return None
        return config.LAYER_WIDTH_MIN_MM <= width_mm <= config.LAYER_WIDTH_MAX_MM

    # --- main -----------------------------------------------------------

    def update(
        self,
        frame: Optional[np.ndarray],
        analysis: Optional[VisionAnalysis],
        camera_connected: bool = True,
        machine_printing: Optional[bool] = None,
    ) -> PrintObservation:
        """
        Folds one analysed frame into the debounced state.

        `machine_printing`, when a real machine link supplies it, overrides the
        camera-based inference -- the machine knows better than a motion proxy.
        """
        now = time.time()
        reasons: Dict[str, str] = {}

        if not camera_connected or frame is None or analysis is None:
            observation = PrintObservation(
                timestamp=now,
                printing=False,
                layer_detected=False,
                status=PrintStatus.CAMERA_ERROR,
                layer_status=PrintStatus.CAMERA_ERROR,
                confidence=0.0,
                camera_connected=False,
                reasons={"camera": "No usable frame from the camera."},
            )
            self._latest = observation
            return observation

        # --- printing activity (debounced), measured inside the ROI ---
        motion = self._motion_score(frame, analysis.roi)
        if machine_printing is None:
            if motion >= config.PRINT_MOTION_THRESHOLD:
                self._active_streak += 1
                self._idle_streak = 0
            else:
                self._idle_streak += 1
                self._active_streak = 0

            if not self._printing and self._active_streak >= config.PRINT_ACTIVE_FRAMES:
                self._printing = True
            elif self._printing and self._idle_streak >= config.PRINT_IDLE_FRAMES:
                self._printing = False
            reasons["printing_source"] = (
                "Inferred from frame-to-frame movement in the camera view "
                "(no machine status link connected)."
            )
        else:
            self._printing = bool(machine_printing)
            reasons["printing_source"] = "Reported by the machine link."

        # --- layer detection (confidence + debounced) ---
        confidence = self._confidence(analysis)
        layer_seen = analysis.layers.detected and confidence >= config.LAYER_CONFIDENCE_THRESHOLD
        if analysis.layers.detected and not layer_seen:
            reasons["confidence"] = (
                f"A layer was measured but confidence {confidence:.2f} is below the "
                f"threshold of {config.LAYER_CONFIDENCE_THRESHOLD:.2f}, so it is not acted on."
            )

        if layer_seen:
            self._detect_streak += 1
            self._missing_streak = 0
        else:
            self._missing_streak += 1
            self._detect_streak = 0

        if not self._layer_detected and self._detect_streak >= config.LAYER_DETECT_FRAMES:
            self._layer_detected = True
        elif self._layer_detected and self._missing_streak >= config.LAYER_MISSING_FRAMES:
            self._layer_detected = False

        # --- width: raw pixels vs calibrated millimetres, kept separate ---
        width_px = analysis.layers.layer_width_px
        width_mm = analysis.layers.layer_width_mm
        width_valid = self._width_validity(width_mm)
        if width_mm is None and width_px is not None:
            reasons["width_units"] = (
                "Layer width calibration is not configured. Set PIXELS_PER_MM in .env "
                "to get millimetres; only a pixel measurement is available, and it is "
                "NOT reported as millimetres."
            )

        # --- overall status ---
        if self._layer_detected and width_valid is False:
            layer_status = PrintStatus.ABNORMAL_LAYER
            reasons["abnormal"] = (
                f"Measured width {width_mm} mm is outside the configured limits "
                f"{config.LAYER_WIDTH_MIN_MM}-{config.LAYER_WIDTH_MAX_MM} mm."
            )
        elif self._layer_detected:
            layer_status = PrintStatus.LAYER_DETECTED
        elif self._printing:
            layer_status = PrintStatus.LAYER_MISSING
            reasons["layer_missing"] = (
                "The machine appears active but no layer is being resolved in the camera view."
            )
        else:
            layer_status = PrintStatus.NO_PRINT

        if layer_status == PrintStatus.ABNORMAL_LAYER:
            status = PrintStatus.ABNORMAL_LAYER
        elif self._printing:
            status = PrintStatus.PRINTING
        else:
            status = PrintStatus.NO_PRINT

        observation = PrintObservation(
            timestamp=now,
            printing=self._printing,
            layer_detected=self._layer_detected,
            status=status,
            layer_status=layer_status,
            confidence=confidence,
            layer_width_px=width_px,
            layer_width=width_mm,
            width_valid=width_valid,
            motion_score=motion,
            camera_connected=True,
            reasons=reasons,
        )
        self._latest = observation
        return observation

    @property
    def latest(self) -> Optional[PrintObservation]:
        return self._latest


# ---------------------------------------------------------------------------
# Clean query API -- what the voice assistant and any control logic call.
# Each reads the shared PrinterState; none of them guess.
# ---------------------------------------------------------------------------

_STALE_AFTER_SECONDS = 10.0


def _observation(state) -> Optional[PrintObservation]:
    obs = state.latest_print_observation()
    if obs is None:
        return None
    if time.time() - obs.timestamp > _STALE_AFTER_SECONDS:
        return None
    return obs


def get_camera_status(state) -> dict:
    """{'connected': bool, 'status': str, 'fps': float}"""
    view = state.dashboard_view()
    status = view.get("camera_status", "unknown")
    return {
        "connected": status == "live",
        "status": status,
        "fps": round(float(view.get("camera_fps", 0.0)), 1),
    }


def get_current_printing_status(state) -> dict:
    """{'printing': bool|None, 'status': str, 'reason': str}"""
    obs = _observation(state)
    if obs is None:
        return {
            "printing": None,
            "status": PrintStatus.CAMERA_ERROR.value,
            "reason": "No recent camera analysis is available.",
        }
    return {
        "printing": obs.printing,
        "status": obs.status.value,
        "reason": obs.reasons.get("printing_source", ""),
    }


def get_current_layer_status(state) -> dict:
    """{'layer_detected': bool|None, 'status': str, 'confidence': float, 'reason': str}"""
    obs = _observation(state)
    if obs is None:
        return {
            "layer_detected": None,
            "status": PrintStatus.CAMERA_ERROR.value,
            "confidence": 0.0,
            "reason": "No recent camera analysis is available.",
        }
    return {
        "layer_detected": obs.layer_detected,
        "status": obs.layer_status.value,
        "confidence": round(obs.confidence, 3),
        "reason": obs.reasons.get("abnormal") or obs.reasons.get("layer_missing", ""),
    }


def get_current_detection(state) -> dict:
    """
    The complete current detection payload -- everything the pipeline measured
    this cycle, in the documented output contract. Returns a CAMERA_ERROR
    payload rather than stale or invented data when nothing recent exists.
    """
    obs = _observation(state)
    if obs is None:
        return {
            "printing": None,
            "layer_detected": None,
            "layer_width": None,
            "layer_width_px": None,
            "width_unit": "mm",
            "confidence": 0.0,
            "status": PrintStatus.CAMERA_ERROR.value,
            "timestamp": time.time(),
            "notes": {"camera": "No recent camera analysis is available."},
        }
    payload = obs.to_dict()
    payload["age_seconds"] = round(time.time() - obs.timestamp, 1)
    return payload


def get_current_layer_width(state) -> dict:
    """
    {'width': float|None, 'unit': 'mm', 'width_px': float|None,
     'valid': bool|None, 'reason': str}

    `width` is None (with a reason) whenever calibration is missing -- a pixel
    measurement is never passed off as millimetres.
    """
    obs = _observation(state)
    if obs is None:
        return {
            "width": None, "unit": "mm", "width_px": None, "valid": None,
            "reason": "No recent camera analysis is available.",
        }
    return {
        "width": obs.layer_width,
        "unit": obs.width_unit,
        "width_px": obs.layer_width_px,
        "valid": obs.width_valid,
        "reason": obs.reasons.get("width_units", ""),
    }
