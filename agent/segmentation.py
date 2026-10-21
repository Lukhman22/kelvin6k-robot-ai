"""
Shared foreground segmentation for the vision analysers.

Separating the printed part from the background is the first step for both the
layer detector and the wall detector, so it lives here once.

Critically, this module REFUSES to segment frames that don't actually contain a
distinguishable part (blank wall, lens cap on, washed-out exposure). Without
those guards a featureless frame segments as one huge "perfect" object and the
downstream analysers report a flawless print of nothing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

# A frame flatter than this has no visible structure to measure.
MIN_CONTRAST_STD = 5.0
# Foreground must be a plausible share of the frame -- not a speck, not everything.
MIN_FOREGROUND_RATIO = 0.005
MAX_FOREGROUND_RATIO = 0.90
MIN_CONTOUR_AREA = 200


@dataclass
class Segmentation:
    ok: bool
    reason: str = ""
    mask: Optional[np.ndarray] = None  # full-frame binary mask
    bbox: Optional[Tuple[int, int, int, int]] = None  # union bbox (x, y, w, h)
    region: Optional[np.ndarray] = None  # mask cropped to bbox


def segment_part(gray: np.ndarray) -> Segmentation:
    """Segments the printed structure, or explains why it can't be done."""
    if cv2 is None:
        return Segmentation(
            ok=False,
            reason="OpenCV (cv2) is not installed.",
        )

    if float(gray.std()) < MIN_CONTRAST_STD:
        return Segmentation(
            ok=False,
            reason="The frame has almost no contrast -- no printed structure is visible.",
        )

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ratio = float(np.count_nonzero(mask)) / mask.size

    # Otsu marks the bright class as foreground; invert if the part is the
    # darker of the two.
    if ratio > MAX_FOREGROUND_RATIO:
        mask = cv2.bitwise_not(mask)
        ratio = float(np.count_nonzero(mask)) / mask.size

    if ratio < MIN_FOREGROUND_RATIO:
        return Segmentation(ok=False, reason="No printed material visible in the frame.")
    if ratio > MAX_FOREGROUND_RATIO:
        return Segmentation(
            ok=False,
            reason="Cannot separate the printed part from the background in this frame.",
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not significant:
        return Segmentation(ok=False, reason="No structure large enough to analyse.")

    # Union bounding box over every significant piece -- individual layers are
    # often separate contours, so using only the largest would see one layer.
    x, y, w, h = cv2.boundingRect(np.vstack(significant))
    if h < 5 or w < 5:
        return Segmentation(ok=False, reason="The detected structure is too small to measure.")

    return Segmentation(
        ok=True,
        mask=mask,
        bbox=(x, y, w, h),
        region=mask[y : y + h, x : x + w],
    )
