"""
Background camera worker for the web API.

Owns the camera in one place and reuses the EXISTING pipeline -- CameraStream,
VisionPipeline, PrintMonitor, PrinterState, db -- so detection logic is not
duplicated for the web UI. It publishes:

  * the latest annotated JPEG frame (for MJPEG streaming), and
  * the same PrinterState / SQLite machine_state the console app uses.

If the camera cannot be opened, nothing is faked: the frame is None and the
status says UNAVAILABLE with the reason.
"""

import asyncio
import threading
import time
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

from agent import config, db
from agent.camera import CameraError, CameraStream
from agent.dashboard import draw_overlay
from agent.logging_setup import get_logger
from agent.print_monitor import PrintMonitor
from agent.rules import RuleEngine
from agent.state import PrinterState
from agent.vision import VisionPipeline

log = get_logger("CAMERA-API")


class CameraService:
    """Single background thread that captures, analyses, and publishes frames."""

    def __init__(self, state: PrinterState, rules: Optional[RuleEngine] = None):
        self.state = state
        self.rules = rules or RuleEngine()
        self.pipeline = VisionPipeline()
        self.monitor = PrintMonitor()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._last_persist = 0.0

        # Latest metrology pass and the announcement it produced, if any.
        # Both are read by api/server.py; neither is required for the camera
        # to run, so a metrology failure leaves them as they were.
        from agent.metrology_voice import AnnouncementGate

        self.metrology = None
        self.metrology_announcement = None
        self._last_measure = 0.0
        self.voice_gate = AnnouncementGate()

        self.status = "not started"   # not started | connecting | live | unavailable | stopped
        self.error: Optional[str] = None
        self.fps = 0.0
        self.resolution = (0, 0)
        self.source = config.CAMERA_SOURCE

    # --- lifecycle ---

    def start(self) -> None:
        if not getattr(config, "CAMERA_ENABLED", True):
            self.status = "disabled"
            self.error = "Camera is disabled in configuration"
            log.info("Main camera is disabled in configuration")
            return

        if cv2 is None:
            self.status = "unavailable"
            self.error = "OpenCV (cv2) is not installed. Camera service cannot run."
            log.warning("Main camera cannot start: OpenCV (cv2) is not installed")
            return
            
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera-service")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    # --- frame access ---

    def latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._jpeg

    def _measure(self, frame) -> None:
        """
        Perpendicular layer-width metrology, and the voice announcement it may
        trigger.

        Runs inside the existing FRAME_SKIP gate, so it costs nothing on the
        frames that are only being streamed. Wrapped so a metrology failure can
        never stop the camera loop.
        """
        if not config.METROLOGY_ENABLED:
            return
        # Own clock: measuring on every analysed frame cost 32 ms of a 33 ms
        # budget at 30 fps, which would have eaten the stream's headroom.
        now = time.time()
        if now - self._last_measure < config.METROLOGY_MIN_INTERVAL_SEC:
            return
        self._last_measure = now
        try:
            from agent.metrology import measure_layer

            self.metrology = measure_layer(frame)
            self.metrology_announcement = self.voice_gate.consider(self.metrology)
        except Exception as exc:
            log.warning("Metrology skipped for this frame: %s", exc)

    def _publish(self, frame, analysis) -> None:
        if cv2 is None:
            return
        annotated = draw_overlay(
            frame, analysis, self.fps, self.monitor.latest, metrology=self.metrology
        )
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._frame_lock:
                self._jpeg = buffer.tobytes()

    def _persist(self, observation) -> None:
        now = time.time()
        if now - self._last_persist < config.STATE_PERSIST_INTERVAL_SECONDS:
            return
        self._last_persist = now
        try:
            job = db.get_active_job()
            job_id = job["id"] if job else None
            db.record_machine_state(
                printing=observation.printing,
                status=observation.status.value,
                current_layer=(job or {}).get("current_layer"),
                layer_width_mm=observation.layer_width,
                layer_width_px=observation.layer_width_px,
                # Stamp the job's real progress onto each state row so the
                # dashboard can measure production over time windows.
                completed_length_mm=(job or {}).get("completed_length_mm"),
                motion_score=observation.motion_score,
                confidence=observation.confidence,
                camera_connected=observation.camera_connected,
                job_id=job_id,
            )
        except Exception as exc:
            log.warning("Could not persist machine state: %s", exc)

    # --- worker ---

    def _run(self) -> None:
        self.status = "connecting"
        camera = CameraStream()

        try:
            camera.open()
        except CameraError as exc:
            # No camera: report it honestly and keep the API alive.
            self.status = "unavailable"
            self.error = str(exc).splitlines()[0]
            self.state.set_camera_status(f"unavailable: {self.error}")
            self.state.update_print_observation(
                self.monitor.update(None, None, camera_connected=False)
            )
            log.warning("Camera unavailable: %s", self.error)
            return

        self.status = "live"
        self.error = None
        self.resolution = camera.actual_resolution
        self.source = camera.source
        self.state.set_camera_status("live")
        log.info("Camera service live (%s, %dx%d)", camera.source, *self.resolution)

        frame_no = 0
        analyze_every = max(1, config.FRAME_SKIP)
        analysis = None

        try:
            while self._running:
                frame = camera.read()
                if frame is None:
                    if camera.consecutive_failures >= config.CAMERA_READ_FAILURE_LIMIT:
                        self.status = "reconnecting"
                        self.state.set_camera_status("reconnecting")
                        self.metrology = None
                        self.voice_gate.reset()
                        self.state.update_print_observation(
                            self.monitor.update(None, None, camera_connected=False)
                        )
                    time.sleep(0.02)
                    continue

                if self.status != "live":
                    self.status = "live"
                    self.state.set_camera_status("live")

                self.fps = camera.measured_fps
                frame_no += 1

                if frame_no % analyze_every == 0:
                    analysis = self.pipeline.analyze(frame)
                    self.state.update_vision(analysis, self.fps)
                    observation = self.monitor.update(frame, analysis, camera_connected=True)
                    self.state.update_print_observation(observation)
                    self._persist(observation)

                    alerts = self.rules.evaluate_vision(analysis)
                    if alerts:
                        self.state.add_alerts(alerts)

                    self._measure(frame)

                self._publish(frame, analysis)
        except Exception as exc:
            self.status = "unavailable"
            self.error = str(exc)
            log.warning("Camera service stopped with an error: %s", exc)
        finally:
            camera.release()
            if self.status not in ("unavailable",):
                self.status = "stopped"
            self.state.set_camera_status("stopped")
            log.info("Camera service stopped.")

    async def mjpeg_frames(self, request=None, max_seconds: float = 0.0):
        """
        Async generator yielding multipart MJPEG chunks for <img src>.

        Async (not a sync generator) for two reasons that were breaking the
        browser view:

          * a sync generator occupies an anyio threadpool worker for the whole
            life of the stream; leaked streams exhausted the pool and the API
            stopped answering;
          * without a disconnect check the `while True` loop ran forever after
            the browser went away, so reloading the page leaked a stream each
            time until the browser hit its 6-connections-per-origin cap and the
            video simply stopped appearing.

        Now the loop exits as soon as the client disconnects.
        """
        boundary = b"--frame"
        started = time.time()
        interval = 1.0 / max(1, config.CAMERA_FPS)
        last_sent: Optional[bytes] = None

        try:
            while True:
                if request is not None and await request.is_disconnected():
                    break
                if max_seconds and (time.time() - started) > max_seconds:
                    break

                jpeg = self.latest_jpeg()
                if jpeg is None:
                    await asyncio.sleep(0.1)
                    continue

                # Skip re-sending an identical frame; keeps bandwidth sane when
                # the camera is slower than the stream interval.
                if jpeg is not last_sent:
                    last_sent = jpeg
                    yield (
                        boundary
                        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass  # client went away mid-write
        finally:
            log.debug("MJPEG stream closed after %.1fs", time.time() - started)
