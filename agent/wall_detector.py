"""
Wall analysis from a single camera view.

What this CAN measure:
  - vertical wall bands visible in the frame (via vertical-edge projection)
  - the thickness of the dominant (outer) wall band, in pixels -> mm if calibrated
  - wall continuity: what fraction of the wall's height has unbroken material
  - gaps / missing sections along the wall

What this CANNOT do from one fixed camera, and never claims:
  - reliably separate outer from inner walls (a single viewpoint usually sees
    only the outer surface; inner walls are occluded)
  - give a trustworthy total wall count for the part
When the view does not support it, the analysis says so explicitly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from agent import config
from agent.segmentation import segment_part

# A single camera can only claim an inner wall when clearly separated vertical
# bands are visible. Below this, we report outer-wall-only rather than guessing.
_MIN_BANDS_FOR_INNER_WALL = 3

_WALL_COUNT_UNRELIABLE = "Wall count cannot be reliably determined from the current camera view."


@dataclass
class WallAnalysis:
    detected: bool
    outer_wall_detected: bool = False
    inner_wall_detected: bool = False
    visible_wall_bands: Optional[int] = None
    wall_thickness_px: Optional[float] = None
    wall_thickness_mm: Optional[float] = None
    wall_continuity_percent: Optional[float] = None
    gap_count: Optional[int] = None
    defects: List[str] = field(default_factory=list)
    unavailable: Dict[str, str] = field(default_factory=dict)


class WallDetector:
    """
    Classical wall analysis (vertical-edge projection + continuity scan).

    Designed so a trained segmentation model can replace `analyze()` later: keep
    the same WallAnalysis return type and the rest of the system is unchanged.
    """

    def _to_mm(self, pixels: Optional[float]) -> Optional[float]:
        if pixels is None or config.PIXELS_PER_MM <= 0:
            return None
        return round(pixels / config.PIXELS_PER_MM, 3)

    def analyze(self, frame: np.ndarray) -> WallAnalysis:
        if cv2 is None:
            return WallAnalysis(
                detected=False,
                unavailable={"walls": "OpenCV (cv2) is not installed."},
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        unavailable: Dict[str, str] = {}
        if config.PIXELS_PER_MM <= 0:
            unavailable["millimetre_measurements"] = (
                "PIXELS_PER_MM is not calibrated in .env, so wall thickness is reported "
                "in pixels only."
            )

        seg = segment_part(gray)
        if not seg.ok:
            unavailable["walls"] = seg.reason
            return WallAnalysis(detected=False, unavailable=unavailable)

        region = seg.region

        # Vertical bands: columns that contain a lot of material.
        column_fill = (region > 0).mean(axis=0)
        occupied = column_fill > 0.5
        bands = 0
        in_band = False
        for value in occupied:
            if value and not in_band:
                bands += 1
                in_band = True
            elif not value:
                in_band = False

        # Wall thickness: median horizontal run of material per row.
        row_widths = [int(np.count_nonzero(row)) for row in region]
        row_widths = [rw for rw in row_widths if rw > 0]
        thickness_px = float(np.median(row_widths)) if row_widths else None

        # Continuity measures HORIZONTAL completeness of the deposited material,
        # looking only at rows that carry material. Empty rows between layers are
        # the normal inter-layer spacing, not defects, so they are excluded --
        # counting them would flag every healthy layered print as broken.
        # (Missing whole layers shows up in the layer detector's pitch
        # consistency instead.)
        material_rows = [row for row in region if np.count_nonzero(row) > 0]

        if not material_rows:
            unavailable["walls"] = "No material rows found inside the detected structure."
            return WallAnalysis(detected=False, unavailable=unavailable)

        span = max(1, region.shape[1])
        fills = [np.count_nonzero(row) / span for row in material_rows]
        continuity = round(float(np.mean(fills)) * 100.0, 1)

        # A gap is a hole INSIDE the wall span: background pixels between the
        # first and last material pixel of a row (a genuinely missing section).
        gap_count = 0
        for row in material_rows:
            idx = np.flatnonzero(row)
            interior = row[idx[0] : idx[-1] + 1]
            holes, in_hole = 0, False
            for pixel in interior:
                if pixel == 0 and not in_hole:
                    holes += 1
                    in_hole = True
                elif pixel != 0:
                    in_hole = False
            gap_count += holes

        defects: List[str] = []
        if continuity < 95.0:
            defects.append(f"wall continuity {continuity} percent -- material missing in places")
        if gap_count > 0:
            defects.append(f"{gap_count} gap(s) detected inside the wall")

        inner_wall = bands >= _MIN_BANDS_FOR_INNER_WALL
        if not inner_wall:
            unavailable["inner_wall"] = (
                "Only the outer wall surface is visible from this camera angle. "
                "Inner walls are occluded and cannot be measured."
            )
        unavailable["wall_count"] = _WALL_COUNT_UNRELIABLE

        return WallAnalysis(
            detected=True,
            outer_wall_detected=True,
            inner_wall_detected=inner_wall,
            visible_wall_bands=bands,
            wall_thickness_px=round(thickness_px, 2) if thickness_px is not None else None,
            wall_thickness_mm=self._to_mm(thickness_px),
            wall_continuity_percent=continuity,
            gap_count=gap_count,
            defects=defects,
            unavailable=unavailable,
        )
