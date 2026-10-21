"""
Print-quality assessment derived strictly from measurements that were actually
obtained this frame.

The score is a documented weighted average of the sub-metrics that are
available. If none are available, the score is None ("cannot be determined")
-- it is never randomised or defaulted to a flattering number.

Sub-metrics and what they detect:
  layer_consistency   -- uneven layer pitch => layer inconsistency / shifting
  wall_continuity     -- missing material / gaps / under-extrusion
  width_stability     -- drifting bead width => over- or under-extrusion
"""

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional

from agent.layer_detector import LayerMeasurement
from agent.wall_detector import WallAnalysis

# Relative weights of each sub-metric in the overall score.
_WEIGHTS = {
    "layer_consistency": 0.4,
    "wall_continuity": 0.4,
    "width_stability": 0.2,
}


@dataclass
class QualityAssessment:
    score_percent: Optional[float] = None
    status: str = "UNKNOWN"  # GOOD | FAIR | POOR | UNKNOWN
    issues: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)
    unavailable: Dict[str, str] = field(default_factory=dict)


class QualityAnalyzer:
    def analyze(self, layers: LayerMeasurement, walls: WallAnalysis) -> QualityAssessment:
        components: Dict[str, float] = {}
        issues: List[str] = []
        unavailable: Dict[str, str] = {}

        if layers.layer_consistency_percent is not None:
            components["layer_consistency"] = layers.layer_consistency_percent
            if layers.layer_consistency_percent < 85.0:
                issues.append(
                    f"layer pitch is uneven ({layers.layer_consistency_percent} percent consistent) "
                    "-- possible layer inconsistency or layer shifting"
                )
        else:
            unavailable["layer_consistency"] = "No reliable layer boundaries measured."

        if walls.wall_continuity_percent is not None:
            components["wall_continuity"] = walls.wall_continuity_percent
            if walls.wall_continuity_percent < 95.0:
                issues.append(
                    f"wall continuity is {walls.wall_continuity_percent} percent "
                    "-- gaps or missing material detected"
                )
        else:
            unavailable["wall_continuity"] = "No wall structure measured."

        # Width stability: a strong trend in bead width suggests extrusion drift.
        if layers.layer_width_trend is not None:
            stability = 100.0 if layers.layer_width_trend == "stable" else 75.0
            components["width_stability"] = stability
            if layers.layer_width_trend == "increasing":
                issues.append("bead width is increasing -- possible over-extrusion")
            elif layers.layer_width_trend == "decreasing":
                issues.append("bead width is decreasing -- possible under-extrusion")
        else:
            unavailable["width_stability"] = (
                "Not enough layer-width samples yet to judge extrusion stability."
            )

        if not components:
            unavailable["print_quality"] = (
                "Print quality cannot be scored: none of the underlying visual "
                "measurements are available from the current camera view."
            )
            return QualityAssessment(unavailable=unavailable)

        total_weight = sum(_WEIGHTS[name] for name in components)
        score = sum(components[name] * _WEIGHTS[name] for name in components) / total_weight
        score = round(score, 1)

        if score >= 90:
            status = "GOOD"
        elif score >= 75:
            status = "FAIR"
        else:
            status = "POOR"

        if len(components) < len(_WEIGHTS):
            missing = sorted(set(_WEIGHTS) - set(components))
            unavailable["partial_score"] = (
                "Score is based only on the available metrics "
                f"({', '.join(sorted(components))}); missing: {', '.join(missing)}."
            )

        return QualityAssessment(
            score_percent=score,
            status=status,
            issues=issues,
            components={k: round(v, 1) for k, v in components.items()},
            unavailable=unavailable,
        )
