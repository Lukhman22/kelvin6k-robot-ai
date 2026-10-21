"""
api/site_camera_service.py - Isolated Camera 2 worker for site people counting.

Owns dedicated Camera 2 completely independent from Camera 1.
Camera 1 continues to run at 30 FPS with metrology, whereas Camera 2 runs
at a lighter cadence and maintains a thread-safe JPEG buffer and a 2-second
cached PeopleCountResult.
"""

import threading
import time
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

from agent import config
from agent.camera import CameraError, CameraStream
from agent.logging_setup import get_logger
from agent.people_detector import PeopleCountResult, PeopleDetector, draw_people_overlay

log = get_logger("SITE-CAMERA")

_default_service: Optional["SiteCameraService"] = None


def get_site_camera_service() -> Optional["SiteCameraService"]:
    """Returns the active SiteCameraService instance, if created."""
    global _default_service
    return _default_service


class SiteCameraService:
    """Dedicated background service for Camera 2 (Site People Counting)."""

    def __init__(self, detector: Optional[PeopleDetector] = None):
        global _default_service
        self.detector = detector or PeopleDetector()
        _default_service = self
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._jpeg: Optional[bytes] = None

        self._cache_lock = threading.Lock()
        self._cached_result: Optional[PeopleCountResult] = None
        self._cached_time: float = 0.0
        self.cache_ttl: float = getattr(config, "SITE_CAMERA_CACHE_SECONDS", 2.0)

        self.status = "not started"  # not started | connecting | live | unavailable | stopped | disabled
        self.error: Optional[str] = None
        self.fps = 0.0
        self.resolution = (0, 0)
        self.source = getattr(config, "SITE_CAMERA_SOURCE", "usb")
        self.index = getattr(config, "SITE_CAMERA_INDEX", 1)

    def start(self) -> None:
        if not getattr(config, "CAMERA_ENABLED", True) or not getattr(config, "SITE_CAMERA_ENABLED", True):
            self.status = "disabled"
            self.error = "Site camera is disabled in configuration"
            log.info("Site camera is disabled in configuration")
            return

        if cv2 is None:
            self.status = "unavailable"
            self.error = "OpenCV (cv2) is not installed. Site camera service cannot run."
            log.warning("Site camera cannot start: OpenCV (cv2) is not installed")
            return
            
        if self.detector is None:
            self.detector = PeopleDetector()

        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="site-camera-service")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=4.0)
            self._thread = None
        if self.status != "disabled":
            self.status = "stopped"

    def latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._jpeg

    def latest_people_count(self) -> Optional[int]:
        """Returns the latest YOLO people count from Camera 2 if live, or None."""
        if self.status != "live":
            return None
        with self._cache_lock:
            if self._cached_result is not None:
                return self._cached_result.people_count
        return 0

    def latest_detections(self) -> List[Dict[str, Any]]:
        """Returns the latest YOLO instance segmentation entity detections from Camera 2 if live, or empty list."""
        if self.status != "live":
            return []
        with self._cache_lock:
            if self._cached_result is not None:
                if hasattr(self._cached_result, "entities_dict"):
                    return self._cached_result.entities_dict()
                return getattr(self._cached_result, "entities", [])
        return []

    def latest_fire_count(self) -> int:
        """Returns the number of fire/flame entities currently detected by Camera 2 if live, or 0."""
        if self.status != "live":
            return 0
        with self._cache_lock:
            if self._cached_result is not None:
                entities = getattr(self._cached_result, "entities", [])
                return sum(1 for e in entities if getattr(e, "class_name", "").lower() in ("fire", "flame"))
        return 0

    def get_people_count(self, force_refresh: bool = False) -> PeopleCountResult:
        """
        Returns the latest people count from Camera 2 with a 2-second result cache.
        Enforces that Camera 2 MUST be live to return a valid count.
        """
        if not getattr(config, "SITE_CAMERA_ENABLED", True) or self.status == "disabled":
            return PeopleCountResult(
                people_count=None,
                available=False,
                reason="Site camera (Camera 2) is disabled in configuration.",
                detector="none",
            )

        if self.status != "live":
            return PeopleCountResult(
                people_count=None,
                available=False,
                reason=f"Site camera (Camera 2) is {self.status}: {self.error or 'no live video'}.",
                detector="none",
            )

        now = time.time()
        with self._cache_lock:
            if (
                not force_refresh
                and self._cached_result is not None
                and (now - self._cached_time) < self.cache_ttl
            ):
                return self._cached_result

        jpeg = self.latest_jpeg()
        if not jpeg:
            return PeopleCountResult(
                people_count=None,
                available=False,
                reason="Site camera (Camera 2) has no captured frames yet.",
                detector="none",
            )

        result = self.detector.detect_people(jpeg)
        with self._cache_lock:
            self._cached_result = result
            self._cached_time = now
        return result

    def _run(self) -> None:
        self.status = "connecting"
        camera = CameraStream(
            source=self.source,
            index=self.index,
            width=getattr(config, "SITE_CAMERA_WIDTH", 1280),
            height=getattr(config, "SITE_CAMERA_HEIGHT", 720),
            fps=getattr(config, "SITE_CAMERA_FPS", 15),
            name_hint=getattr(config, "SITE_CAMERA_NAME_HINT", ""),
            rtsp_url=getattr(config, "SITE_CAMERA_RTSP_URL", "") or None,
        )

        try:
            camera.open()
        except CameraError as exc:
            self.status = "unavailable"
            self.error = str(exc).splitlines()[0]
            log.warning("Site camera (Camera 2) unavailable: %s", self.error)
            return

        self.status = "live"
        self.error = None
        self.resolution = camera.actual_resolution
        log.info("Site camera (Camera 2) live (%s, %dx%d)", camera.source, *self.resolution)

        try:
            frame_no = 0
            frame_skip = 2  # run YOLO inference every 2 frames for smooth video streaming
            latest_result = PeopleCountResult(
                people_count=0, confidence=0.85, details="Initialising detector", boxes=[]
            )

            while self._running:
                frame = camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                self.fps = camera.measured_fps
                frame_no += 1

                if frame_no % frame_skip == 0:
                    try:
                        latest_result = self.detector.detect_frame(frame)
                    except Exception as det_exc:
                        log.warning("Site camera detection error: %s", det_exc)

                annotated = draw_people_overlay(frame, latest_result, self.fps)

                if cv2 is not None:
                    ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        with self._frame_lock:
                            self._jpeg = buffer.tobytes()

                with self._cache_lock:
                    self._cached_result = latest_result
                    self._cached_time = time.time()

                time.sleep(0.03)
        except Exception as exc:
            self.status = "unavailable"
            self.error = str(exc)
            log.warning("Site camera (Camera 2) stopped with error: %s", exc)
        finally:
            camera.release()
            if self.status not in ("unavailable", "disabled"):
                self.status = "stopped"
            log.info("Site camera (Camera 2) stopped.")
