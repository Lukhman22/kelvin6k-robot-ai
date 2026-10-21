"""
Vision pipeline coordinator: camera frame -> layer / wall / quality analysis.

This module previously simulated random defect detections. That simulation has
been removed: every VisionEvent produced here now comes from an actual
measurement on an actual frame. If nothing can be measured, nothing is
reported.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from agent import config
from agent.layer_detector import LayerDetector, LayerMeasurement
from agent.quality import QualityAnalyzer, QualityAssessment
from agent.wall_detector import WallAnalysis, WallDetector


@dataclass
class VisionEvent:
    """A defect actually observed in the frame (not simulated)."""

    defect_type: str
    confidence: float


@dataclass
class VisionAnalysis:
    timestamp: float
    layers: LayerMeasurement
    walls: WallAnalysis
    quality: QualityAssessment
    events: List[VisionEvent] = field(default_factory=list)
    # (x, y, w, h) of the analysed region in raw-frame coordinates, or None
    # when the whole frame was analysed.
    roi: Optional[tuple] = None


class VisionPipeline:
    """
    Runs the three analysers over each frame. Stateless per frame apart from the
    layer detector's rolling width history, which powers trend analysis.
    """

    def __init__(self):
        self.layer_detector = LayerDetector()
        self.wall_detector = WallDetector()
        self.quality_analyzer = QualityAnalyzer()

    @staticmethod
    def resolve_roi(frame: np.ndarray) -> Optional[tuple]:
        """
        Configured ROI clipped to the frame, or None for the whole frame.
        Restricting analysis to where material is actually deposited removes
        most background noise.
        """
        height, width = frame.shape[:2]
        x = max(0, min(config.ROI_X, width - 1))
        y = max(0, min(config.ROI_Y, height - 1))
        w = config.ROI_WIDTH or (width - x)
        h = config.ROI_HEIGHT or (height - y)
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        if (x, y, w, h) == (0, 0, width, height):
            return None
        return (x, y, w, h)

    def analyze(self, frame: np.ndarray) -> VisionAnalysis:
        roi = self.resolve_roi(frame)
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y : y + h, x : x + w]

        layers = self.layer_detector.analyze(frame)
        walls = self.wall_detector.analyze(frame)
        quality = self.quality_analyzer.analyze(layers, walls)

        events: List[VisionEvent] = []

        # Wall defects are direct observations of missing material.
        if walls.wall_continuity_percent is not None and walls.wall_continuity_percent < 95.0:
            # Confidence scales with how far continuity has fallen.
            confidence = round(min(0.95, (95.0 - walls.wall_continuity_percent) / 50.0 + 0.5), 2)
            events.append(VisionEvent(defect_type="missing material in wall", confidence=confidence))

        # Very uneven layer pitch is the visual signature of a layer shift.
        if layers.layer_consistency_percent is not None and layers.layer_consistency_percent < 70.0:
            confidence = round(min(0.95, (70.0 - layers.layer_consistency_percent) / 50.0 + 0.5), 2)
            events.append(VisionEvent(defect_type="layer shift or uneven layers", confidence=confidence))

        return VisionAnalysis(
            timestamp=time.time(),
            layers=layers,
            walls=walls,
            quality=quality,
            events=events,
            roi=roi,
        )

    @staticmethod
    def primary_event(analysis: VisionAnalysis) -> Optional[VisionEvent]:
        """The highest-confidence observed defect, or None if the frame is clean."""
        if not analysis.events:
            return None
        return max(analysis.events, key=lambda e: e.confidence)
