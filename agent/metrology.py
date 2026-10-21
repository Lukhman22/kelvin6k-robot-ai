"""
Perpendicular layer-width metrology for the printed concrete bead.

This is deliberately separate from `layer_detector.py`. That module measures the
MEDIAN HORIZONTAL RUN of material per row, which is the right cheap proxy for
"is a layer being laid down" but is NOT a width: on a bead that curves or runs
diagonally, a horizontal run overstates the true thickness by 1/cos(angle).

Here the width is measured the way a metrologist would:

    mask -> largest contour -> centreline -> N stations along it
         -> local tangent at each station -> ray-cast along the NORMAL
         -> distance between the two real edges

A station where either edge cannot be found is INVALID and contributes nothing.
No width is ever invented, and a millimetre value is produced only when a
calibration exists -- without one every station reports pixels and the overall
status is MEASUREMENT_INVALID.

Nothing in here touches the camera, the database, or the existing vision
pipeline. It takes a frame and returns a result object.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from agent import config
from agent.logging_setup import get_logger

log = get_logger("METROLOGY")

# Status values. Kept as plain strings so they cross the API and the overlay
# without a serialisation step.
PASS = "PASS"
UNDER_WIDTH = "UNDER-WIDTH"
OVER_WIDTH = "OVER-WIDTH"
INVALID = "MEASUREMENT INVALID"

# How the mask was produced. Reported verbatim on the overlay so the operator is
# never told a detector is running that is not.
MASK_CV = "CV SEGMENTATION"
MASK_NONE = "NO DETECTION"

# Centreline sampling. The first and last few percent of the skeleton are
# skipped: the tangent there is dominated by the bead's end cap, so a
# perpendicular taken at the very tip measures the cap, not the bead.
_END_MARGIN = 0.08
# Window (in centreline points) used for the local tangent. Too small and pixel
# jitter rotates the normal; too large and a real curve gets averaged away.
_TANGENT_WINDOW = 7
# A ray is abandoned after this many pixels -- longer than any plausible bead at
# this working distance, so a runaway ray cannot spin.
_MAX_RAY_PX = 400
# A station is rejected when one half-width is this many times the other: the
# centreline has drifted off centre and the two edges are not a matched pair.
_MAX_ASYMMETRY = 3.0


@dataclass
class Station:
    """One measurement station along the centreline."""

    index: int                      # 1-based, so W1..W10 read naturally
    x: float                        # centreline point, image coordinates
    y: float
    normal: Tuple[float, float]     # unit vector perpendicular to the tangent
    left: Optional[Tuple[float, float]] = None
    right: Optional[Tuple[float, float]] = None
    width_px: Optional[float] = None
    width_mm: Optional[float] = None
    valid: bool = False
    reason: str = ""

    @property
    def label(self) -> str:
        return f"W{self.index}"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width_px": round(self.width_px, 2) if self.width_px is not None else None,
            "width_mm": round(self.width_mm, 2) if self.width_mm is not None else None,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass
class MetrologyResult:
    """One complete measurement pass. The single source of truth for both the
    overlay and the voice announcement -- they must never disagree."""

    timestamp: float
    mask_method: str = MASK_NONE
    contour_ok: bool = False
    centerline_ok: bool = False
    stations: List[Station] = field(default_factory=list)
    contour: Optional[np.ndarray] = None      # drawing only
    centerline: Optional[np.ndarray] = None   # drawing only

    average_width_mm: Optional[float] = None
    deviation_mm: Optional[float] = None
    status: str = INVALID
    reason: str = ""
    pixels_per_mm: float = 0.0
    calibrated: bool = False

    @property
    def station_count(self) -> int:
        return len(self.stations)

    @property
    def valid_count(self) -> int:
        return sum(1 for s in self.stations if s.valid)

    def to_dict(self) -> dict:
        """API-safe payload. Contour and centreline are omitted: they are
        thousands of points and only the overlay needs them."""
        return {
            "mask_method": self.mask_method,
            "contour_ok": self.contour_ok,
            "centerline_ok": self.centerline_ok,
            "stations_total": self.station_count,
            "stations_valid": self.valid_count,
            "stations": [s.to_dict() for s in self.stations],
            "average_width_mm": self.average_width_mm,
            "deviation_mm": self.deviation_mm,
            "target_width_mm": config.METROLOGY_TARGET_WIDTH_MM,
            "tolerance_mm": config.METROLOGY_TOLERANCE_MM,
            "range_min_mm": config.metrology_range_min(),
            "range_max_mm": config.metrology_range_max(),
            "status": self.status,
            "reason": self.reason,
            "pixels_per_mm": self.pixels_per_mm,
            "calibrated": self.calibrated,
            "timestamp": self.timestamp,
        }


# --- segmentation ----------------------------------------------------------


def segment_bead(frame: np.ndarray) -> Tuple[Optional[np.ndarray], str, str]:
    """
    Isolates the concrete bead.

    Classical CV, not a learned model: this project has no concrete-trained
    network, and a COCO-pretrained YOLO has no class that matches a concrete
    bead. The method is reported so the overlay can state what actually ran.

    Returns (mask, method, reason).
    """
    if frame is None or frame.size == 0:
        return None, MASK_NONE, "No frame."

    if cv2 is None:
        return None, MASK_NONE, "OpenCV (cv2) is not installed."

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

    # A frame with no structure segments as one huge "perfect" object, so refuse
    # it rather than measure noise. This mirrors agent/segmentation.py's guard.
    if float(gray.std()) < config.METROLOGY_MIN_CONTRAST:
        return None, MASK_NONE, (
            f"Frame has almost no contrast (std {gray.std():.1f}); nothing to measure."
        )

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ratio = float(np.count_nonzero(mask)) / mask.size
    if ratio > 0.5:                       # bead is the darker class -> invert
        mask = cv2.bitwise_not(mask)
        ratio = float(np.count_nonzero(mask)) / mask.size

    if ratio < config.METROLOGY_MIN_AREA_RATIO:
        return None, MASK_NONE, "No bead-sized region found in the frame."
    if ratio > config.METROLOGY_MAX_AREA_RATIO:
        return None, MASK_NONE, "Cannot separate the bead from the background."

    # Close pinholes and shave speckle, so the boundary a ray hits is the bead's
    # real edge rather than a hole in the middle of it.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, MASK_CV, ""


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """The bead's outline: the single largest external contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < config.METROLOGY_MIN_CONTOUR_AREA:
        return None
    return biggest


# --- centreline ------------------------------------------------------------


def _principal_axis(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The bead's long axis, by PCA over the filled pixels.

    Returns (centroid, along, across) as unit vectors. This only has to be
    roughly right: it fixes the direction the bead is SLICED in, and the
    centreline itself comes from the material in each slice.
    """
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    centroid = points.mean(axis=0)
    centred = points - centroid
    # Covariance eigenvectors: the largest eigenvalue is the long axis.
    cov = np.cov(centred.T)
    values, vectors = np.linalg.eigh(cov)
    along = vectors[:, int(np.argmax(values))]
    along = along / (np.linalg.norm(along) or 1.0)
    across = np.array([-along[1], along[0]], dtype=np.float32)
    return centroid, along.astype(np.float32), across


def _centre_of_slice(
    mask: np.ndarray, origin: np.ndarray, across: np.ndarray, reach: float
) -> Optional[np.ndarray]:
    """
    Midpoint of the material along one slice across the bead.

    Walks the slice from one side to the other and averages the positions where
    the mask is set. Taking the mean of the material -- rather than the midpoint
    of the first and last hit -- keeps the point centred when one edge is
    ragged.
    """
    height, width = mask.shape[:2]
    hits = []
    step = 1.0
    distance = -reach
    while distance <= reach:
        px = origin[0] + across[0] * distance
        py = origin[1] + across[1] * distance
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < width and 0 <= iy < height and mask[iy, ix] > 0:
            hits.append((px, py))
        distance += step
    if len(hits) < 2:
        return None
    return np.array(hits, dtype=np.float32).mean(axis=0)


def centerline_points(mask: np.ndarray, samples: int = 60) -> Optional[np.ndarray]:
    """
    An ordered centreline that follows the bead's real shape.

    Morphological skeletonisation was tried first and rejected: cv2.ximgproc
    (opencv-contrib) is not installed here, and the dependency-free
    erode/open/subtract skeleton comes out as a DISCONNECTED point cloud, which
    has no traversal order and broke immediately on a diagonal bead.

    Slicing across the principal axis avoids the problem entirely -- the points
    are generated in order, one per slice, so no path-walking is needed. The
    curve still follows the material, because every point is the centre of the
    mask in that slice rather than a point on a straight line.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None

    centroid, along, across = _principal_axis(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    projections = (points - centroid) @ along
    lo, hi = float(projections.min()), float(projections.max())
    if hi - lo < config.METROLOGY_MIN_AXIS_LENGTH_PX:
        return None

    # Slices must reach past the widest part of the bead in both directions.
    reach = float(np.abs((points - centroid) @ across).max()) + 4.0

    path = []
    for i in range(samples):
        t = lo + (hi - lo) * i / max(1, samples - 1)
        origin = centroid + along * t
        centre = _centre_of_slice(mask, origin, across, reach)
        if centre is not None:
            path.append(centre)

    if len(path) < 4:
        return None
    return np.array(path, dtype=np.float32)


def _smooth(path: np.ndarray, window: int) -> np.ndarray:
    """Moving average along the path, so pixel staircasing does not rotate the
    local tangent from station to station."""
    if len(path) < window or window < 3:
        return path
    kernel = np.ones(window, dtype=np.float32) / float(window)
    xs = np.convolve(path[:, 0], kernel, mode="valid")
    ys = np.convolve(path[:, 1], kernel, mode="valid")
    return np.stack([xs, ys], axis=1)


def centerline(mask: np.ndarray) -> Optional[np.ndarray]:
    """An ordered, smoothed centreline through the bead. None when too short."""
    points = centerline_points(mask, config.METROLOGY_CENTERLINE_SAMPLES)
    if points is None:
        return None
    smoothed = _smooth(points, config.METROLOGY_SMOOTH_WINDOW)
    return smoothed if len(smoothed) >= 2 else None


# --- perpendicular width ---------------------------------------------------


def _tangent_at(path: np.ndarray, index: int, window: int = _TANGENT_WINDOW) -> Tuple[float, float]:
    """Unit tangent from a finite difference across a window of the path."""
    half = max(1, window // 2)
    lo = max(0, index - half)
    hi = min(len(path) - 1, index + half)
    dx = float(path[hi][0] - path[lo][0])
    dy = float(path[hi][1] - path[lo][1])
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 1.0, 0.0
    return dx / length, dy / length


def _cast_ray(
    mask: np.ndarray, x: float, y: float, dx: float, dy: float
) -> Optional[Tuple[float, float]]:
    """
    Walks outward from (x, y) until the mask stops being set, and returns the
    last point still inside the bead -- its edge along this direction.

    None when the ray leaves the frame or runs past the cap without exiting,
    which is what makes a station INVALID instead of guessing a width.
    """
    height, width = mask.shape[:2]
    if not (0 <= int(round(x)) < width and 0 <= int(round(y)) < height):
        return None
    if mask[int(round(y)), int(round(x))] == 0:
        return None                                    # not inside to begin with

    last_inside = (x, y)
    step = 0.5                                         # sub-pixel walk
    distance = step
    while distance <= _MAX_RAY_PX:
        px = x + dx * distance
        py = y + dy * distance
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < width and 0 <= iy < height):
            return None                                # ran off frame: unreliable
        if mask[iy, ix] == 0:
            # The edge lies between the last set pixel and this clear one;
            # taking the midpoint removes a systematic half-step under-read.
            return ((last_inside[0] + px) / 2.0, (last_inside[1] + py) / 2.0)
        last_inside = (px, py)
        distance += step
    return None                                        # never exited: unreliable


def measure_stations(
    mask: np.ndarray, path: np.ndarray, count: int, pixels_per_mm: float
) -> List[Station]:
    """
    Places `count` stations along the centreline and measures the bead width
    perpendicular to the local tangent at each one.
    """
    stations: List[Station] = []
    n = len(path)
    if n < 2 or count < 1:
        return stations

    lo = int(n * _END_MARGIN)
    hi = int(n * (1.0 - _END_MARGIN))
    if hi - lo < count:
        lo, hi = 0, n - 1

    for i in range(count):
        position = lo if count == 1 else lo + round(i * (hi - lo) / (count - 1))
        position = int(min(max(position, 0), n - 1))
        x, y = float(path[position][0]), float(path[position][1])

        tx, ty = _tangent_at(path, position)
        nx, ny = -ty, tx                               # rotate the tangent 90 degrees

        station = Station(index=i + 1, x=x, y=y, normal=(nx, ny))

        left = _cast_ray(mask, x, y, -nx, -ny)
        right = _cast_ray(mask, x, y, nx, ny)
        if left is None or right is None:
            station.reason = "edge not found on both sides"
            stations.append(station)
            continue

        half_left = math.dist((x, y), left)
        half_right = math.dist((x, y), right)
        if min(half_left, half_right) <= 0.5:
            station.reason = "centreline sits on an edge"
            stations.append(station)
            continue
        if max(half_left, half_right) / max(1e-6, min(half_left, half_right)) > _MAX_ASYMMETRY:
            station.reason = "edges not a matched pair"
            stations.append(station)
            continue

        station.left, station.right = left, right
        station.width_px = math.dist(left, right)
        if pixels_per_mm > 0:
            station.width_mm = station.width_px / pixels_per_mm
            station.valid = True
        else:
            # Geometry succeeded but there is no scale, so no millimetre value
            # exists. Reporting the pixel count as millimetres would be a lie.
            station.reason = "no pixel-to-millimetre calibration"
        stations.append(station)

    return stations


# --- validation ------------------------------------------------------------


def classify(average_mm: Optional[float], valid_count: int) -> Tuple[str, str]:
    """Status and the reason for it, from the valid stations alone."""
    if valid_count < config.METROLOGY_MIN_VALID_STATIONS:
        return INVALID, (
            f"Only {valid_count} of {config.METROLOGY_STATIONS} stations measured; "
            f"at least {config.METROLOGY_MIN_VALID_STATIONS} are needed."
        )
    if average_mm is None:
        return INVALID, "No millimetre value is available."
    if average_mm < config.metrology_range_min():
        return UNDER_WIDTH, ""
    if average_mm > config.metrology_range_max():
        return OVER_WIDTH, ""
    return PASS, ""


def measure_layer(frame: np.ndarray, pixels_per_mm: Optional[float] = None) -> MetrologyResult:
    """
    Full pass over one frame. Never raises: any failure produces a result whose
    status is MEASUREMENT INVALID with the reason filled in, because the camera
    loop must keep running whatever happens here.
    """
    scale = config.metrology_pixels_per_mm() if pixels_per_mm is None else pixels_per_mm
    result = MetrologyResult(
        timestamp=time.time(),
        pixels_per_mm=scale,
        calibrated=scale > 0,
    )

    try:
        mask, method, reason = segment_bead(frame)
        result.mask_method = method
        if mask is None:
            result.reason = reason
            return result

        contour = largest_contour(mask)
        if contour is None:
            result.reason = "No contour large enough to measure."
            return result
        result.contour = contour
        result.contour_ok = True

        # Measure against the contour's own filled mask, so a second object
        # elsewhere in the frame cannot stop a ray.
        bead = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(bead, [contour], -1, 255, thickness=cv2.FILLED)

        path = centerline(bead)
        if path is None:
            result.reason = "Could not trace a centreline through the bead."
            return result
        result.centerline = path
        result.centerline_ok = True

        result.stations = measure_stations(
            bead, path, config.METROLOGY_STATIONS, scale
        )

        widths = [s.width_mm for s in result.stations if s.valid and s.width_mm is not None]
        if widths:
            result.average_width_mm = round(float(np.mean(widths)), 2)
            result.deviation_mm = round(
                result.average_width_mm - config.METROLOGY_TARGET_WIDTH_MM, 2
            )

        result.status, status_reason = classify(result.average_width_mm, result.valid_count)
        if status_reason:
            result.reason = status_reason
        elif not result.calibrated:
            result.reason = (
                "PIXELS_PER_MM is not calibrated, so only pixel geometry is available. "
                "Run calibration_test.py at the working height."
            )
        return result

    except Exception as exc:                 # never take the camera loop down
        log.warning("Metrology failed on this frame: %s", exc)
        result.status = INVALID
        result.reason = f"Measurement error: {exc}"
        return result
