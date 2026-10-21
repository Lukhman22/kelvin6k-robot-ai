"""
Real-time detection of printed layer bands from a single camera view.

What this CAN measure from one camera:
  - horizontal layer boundaries currently visible in the frame
  - the vertical pitch (layer height) between those boundaries, in pixels
  - the horizontal width of the deposited band, in pixels
  - consistency/variation of those measurements over time
  - millimetre conversions, but ONLY when PIXELS_PER_MM has been calibrated

What this CANNOT know from one camera, and therefore never reports:
  - the absolute layer number being printed (needs the machine's layer counter
    or the sliced G-code)
  - the total layer count / print progress (needs the sliced G-code or the
    machine's job info)
Those are reported as unavailable, with the reason, rather than guessed.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from agent import config
from agent.segmentation import Segmentation, segment_part


@dataclass
class LayerMeasurement:
    detected: bool
    visible_layer_bands: Optional[int] = None
    layer_height_px: Optional[float] = None
    layer_height_mm: Optional[float] = None
    layer_width_px: Optional[float] = None
    layer_width_mm: Optional[float] = None
    layer_consistency_percent: Optional[float] = None
    layer_width_trend: Optional[str] = None  # "increasing" | "decreasing" | "stable"
    boundary_rows: List[int] = field(default_factory=list)
    # Why a given field is unavailable, keyed by field name.
    unavailable: Dict[str, str] = field(default_factory=dict)


def _find_peaks(profile: np.ndarray, min_distance: int, threshold: float) -> List[int]:
    """
    Minimal 1-D peak finder (avoids a scipy dependency): local maxima above
    `threshold`, greedily selected strongest-first with a minimum separation.
    """
    candidates = [
        i
        for i in range(1, len(profile) - 1)
        if profile[i] >= threshold and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]
    ]
    candidates.sort(key=lambda i: profile[i], reverse=True)

    chosen: List[int] = []
    for idx in candidates:
        if all(abs(idx - kept) >= min_distance for kept in chosen):
            chosen.append(idx)
    return sorted(chosen)


class LayerDetector:
    """
    Detects layer boundaries as strong horizontal edges stacked vertically.

    The algorithm is deliberately classical (Sobel + projection profile + peak
    finding) rather than a trained model: it is transparent, needs no training
    data, and degrades to "not detected" instead of inventing numbers. A trained
    segmentation model can replace `analyze()` later while keeping the same
    return type.
    """

    def __init__(self):
        self._width_history = deque(maxlen=config.LAYER_WIDTH_HISTORY)

    def _to_mm(self, pixels: Optional[float]) -> Optional[float]:
        if pixels is None or config.PIXELS_PER_MM <= 0:
            return None
        return round(pixels / config.PIXELS_PER_MM, 3)

    def _measure_band_width_px(self, seg: Segmentation) -> Optional[float]:
        """
        Width of the deposited band: the median horizontal run-length of printed
        material across the rows that actually contain material.
        """
        if not seg.ok:
            return None
        row_widths = [int(np.count_nonzero(row)) for row in seg.region]
        row_widths = [rw for rw in row_widths if rw > 0]
        if not row_widths:
            return None
        return float(np.median(row_widths))

    def _trend(self) -> Optional[str]:
        """Direction of recent layer-width samples via least-squares slope."""
        if len(self._width_history) < 5:
            return None
        values = np.array(self._width_history, dtype=float)
        x = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x, values, 1)[0])
        mean = float(values.mean())
        if mean <= 0:
            return None
        # Treat <0.5% change per sample as flat.
        relative = slope / mean
        if relative > 0.005:
            return "increasing"
        if relative < -0.005:
            return "decreasing"
        return "stable"

    def _detect_bands(self, seg: Segmentation):
        """
        Finds layer BANDS (deposited beads), not edges: contiguous runs of rows
        that contain material inside the part's bounding box.

        Returns (band_centre_rows, band_edge_rows) or (None, None) if the frame
        has no coherent structure.
        """
        if not seg.ok:
            return None, None

        x, y, w, h = seg.bbox
        region = seg.region

        # A row belongs to a band if a meaningful share of the part's width is
        # material there. Measured against the widest row so partial views and
        # tapered parts still register.
        row_counts = (region > 0).sum(axis=1)
        peak_width = int(row_counts.max()) if row_counts.size else 0
        if peak_width <= 0:
            return None, None
        occupied = row_counts > (0.5 * peak_width)

        centres, edges = [], []
        start = None
        for i, value in enumerate(occupied):
            if value and start is None:
                start = i
            elif not value and start is not None:
                centres.append(y + (start + i - 1) / 2.0)
                edges.extend([y + start, y + i - 1])
                start = None
        if start is not None:
            centres.append(y + (start + len(occupied) - 1) / 2.0)
            edges.extend([y + start, y + len(occupied) - 1])

        return centres, edges

    def analyze(self, frame: np.ndarray) -> LayerMeasurement:
        if cv2 is None:
            return LayerMeasurement(
                detected=False,
                unavailable={"layers": "OpenCV (cv2) is not installed."},
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        seg = segment_part(gray)
        if not seg.ok:
            return LayerMeasurement(detected=False, unavailable={"layers": seg.reason})

        band_centres, band_edges = self._detect_bands(seg)

        if band_centres is None or len(band_centres) < config.MIN_LAYER_BOUNDARIES:
            # Fall back to horizontal-edge peaks: a solid part with faint layer
            # lines segments as one block, but still shows periodic edges.
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            profile = np.abs(sobel_y).mean(axis=1)
            if profile.size >= 10 and float(profile.max()) > 0:
                threshold = float(profile.mean() + profile.std())
                peaks_fallback = _find_peaks(profile, min_distance=4, threshold=threshold)
                if len(peaks_fallback) > len(band_centres or []):
                    band_centres = [float(p) for p in peaks_fallback]
                    band_edges = peaks_fallback

        if not band_centres:
            return LayerMeasurement(
                detected=False,
                unavailable={"layers": "No layer structure could be resolved in this frame."},
            )

        peaks = [int(round(c)) for c in band_centres]

        unavailable = {
            "current_layer_number": (
                "The absolute layer number cannot be derived from a camera view. "
                "It requires the machine's layer counter or the sliced G-code."
            ),
            "total_layers": (
                "Total layer count is not visible to the camera. "
                "It requires the sliced G-code or the machine's job information."
            ),
            "print_progress_percent": (
                "Print progress requires a known total layer count or model height, "
                "which is not available from the camera alone."
            ),
        }

        if len(peaks) < config.MIN_LAYER_BOUNDARIES:
            unavailable["layers"] = (
                f"Only {len(peaks)} layer bands detected, below the minimum of "
                f"{config.MIN_LAYER_BOUNDARIES} needed for a reliable measurement."
            )
            return LayerMeasurement(detected=False, unavailable=unavailable)

        # Band centre-to-centre spacing = layer pitch (layer height).
        spacings = np.diff(np.array(band_centres, dtype=float))
        layer_height_px = float(np.median(spacings))

        # Consistency: how uniform the layer pitch is (100% = perfectly even).
        consistency = None
        if layer_height_px > 0 and len(spacings) >= 2:
            cv_ratio = float(np.std(spacings) / layer_height_px)
            consistency = round(max(0.0, min(1.0, 1.0 - cv_ratio)) * 100.0, 1)

        width_px = self._measure_band_width_px(seg)
        if width_px is not None:
            self._width_history.append(width_px)

        if config.PIXELS_PER_MM <= 0:
            unavailable["millimetre_measurements"] = (
                "PIXELS_PER_MM is not calibrated in .env, so measurements are reported "
                "in pixels only. Calibrate it to get millimetre values."
            )
        if width_px is None:
            unavailable["layer_width"] = "No coherent printed band found in this frame."

        return LayerMeasurement(
            detected=True,
            visible_layer_bands=len(peaks),
            layer_height_px=round(layer_height_px, 2),
            layer_height_mm=self._to_mm(layer_height_px),
            layer_width_px=round(width_px, 2) if width_px is not None else None,
            layer_width_mm=self._to_mm(width_px),
            layer_consistency_percent=consistency,
            layer_width_trend=self._trend(),
            boundary_rows=[int(e) for e in (band_edges or [])],
            unavailable=unavailable,
        )
