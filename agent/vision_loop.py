"""
Camera + vision + dashboard loop.

Runs on the MAIN thread because OpenCV's GUI (imshow/waitKey) must own it.
If the camera can't be opened, this returns cleanly so the rest of the
application (monitoring, alerts, voice assistant) keeps running.
"""

import sys
import threading
import time
from typing import Callable, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

from agent import config, db
from agent.camera import CameraError, CameraStream
from agent.dashboard import DashboardRenderer, draw_overlay
from agent.logging_setup import get_logger
from agent.print_monitor import PrintMonitor
from agent.rules import Alert, RuleEngine
from agent.state import PrinterState
from agent.vision import VisionPipeline

log = get_logger("VISION")

_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 = ESC
_ASK_KEYS = {ord(" ")}  # SPACE: ask a question without leaving the video window
_WINDOW = "kelvin6k -- live print monitoring"


class VisionLoop:
    def __init__(
        self,
        state: PrinterState,
        rules: RuleEngine,
        on_alerts: Optional[Callable[[List[Alert]], None]] = None,
        on_voice_trigger: Optional[Callable[[], None]] = None,
    ):
        self.state = state
        self.rules = rules
        self.on_alerts = on_alerts
        # Called when SPACE is pressed in the live window.
        self.on_voice_trigger = on_voice_trigger
        self.pipeline = VisionPipeline()
        self.monitor = PrintMonitor()
        self.dashboard = DashboardRenderer()
        self._running = False
        self._last_persist = 0.0

    def _create_window(self, composed) -> None:
        """
        Resizable window scaled to fit the laptop screen. The composed image
        (video + dashboard panel) is wider than many laptop displays, so it is
        shown scaled rather than cropped.
        """
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        height, width = composed.shape[:2]
        scale = min(config.WINDOW_MAX_WIDTH / width, config.WINDOW_MAX_HEIGHT / height, 1.0)
        cv2.resizeWindow(_WINDOW, int(width * scale), int(height * scale))

    def _handle_key(self, key: int) -> bool:
        """Returns False when the loop should stop."""
        if key in _QUIT_KEYS:
            return False
        if key in _ASK_KEYS and self.on_voice_trigger is not None:
            # Run off the GUI thread so the video keeps updating while
            # recording/answering.
            threading.Thread(target=self.on_voice_trigger, daemon=True).start()
        return True

    def _persist(self, observation) -> None:
        """
        Writes the observation to SQLite so the assistant can answer even in a
        different process or after a restart. Rate-limited, and never allowed to
        break the camera loop.
        """
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
                motion_score=observation.motion_score,
                confidence=observation.confidence,
                camera_connected=observation.camera_connected,
                job_id=job_id,
            )
            if observation.layer_detected and observation.layer_width_px is not None:
                db.record_layer_measurement(
                    layer_number=(job or {}).get("current_layer"),
                    width_px=observation.layer_width_px,
                    width_mm=observation.layer_width,
                    confidence=observation.confidence,
                    job_id=job_id,
                )
        except Exception as exc:
            log.warning("Could not persist machine state: %s", exc)

    def stop(self) -> None:
        self._running = False

    def run(self) -> bool:
        """
        Runs until Q/ESC or stop(). Returns True if the camera ran, False if it
        was unavailable (caller can then fall back to a headless mode).
        """
        if cv2 is None:
            print("[CAMERA] OpenCV (cv2) is not installed.", file=sys.stderr)
            print("[CAMERA] Continuing without the camera -- monitoring and voice still work.")
            self.state.set_camera_status("unavailable: cv2 not installed")
            return False

        if not getattr(config, "CAMERA_ENABLED", True):
            print("[CAMERA] Camera is disabled in configuration.")
            self.state.set_camera_status("disabled: configuration")
            return False
            
        camera = CameraStream()
        try:
            camera.open()
        except CameraError as exc:
            print(f"[CAMERA] {exc}", file=sys.stderr)
            print("[CAMERA] Continuing without the camera -- monitoring and voice still work.")
            self.state.set_camera_status(f"unavailable: {exc}")
            return False

        width, height = camera.actual_resolution
        source_label = "RTSP stream" if camera.source == "rtsp" else f"USB index {camera.index}"
        print(f"[CAMERA] Live via {source_label} ({width}x{height}). Press Q or ESC to quit.")
        self.state.set_camera_status("live")

        analysis = None
        frame_no = 0
        window_created = False
        analyze_every = max(1, config.FRAME_SKIP)
        self._running = True
        if config.SHOW_CAMERA_WINDOW:
            print("[CAMERA] In the video window: SPACE = ask a question, Q/ESC = quit.")

        try:
            while self._running:
                frame = camera.read()
                if frame is None:
                    # Dropped frame. camera.read() handles RTSP reconnection
                    # internally; we just report status and keep going.
                    if camera.consecutive_failures > 0:
                        self.state.set_camera_status(
                            "live" if camera.is_open else "reconnecting"
                        )
                    if camera.consecutive_failures >= config.CAMERA_READ_FAILURE_LIMIT:
                        self.state.update_print_observation(
                            self.monitor.update(None, None, camera_connected=False)
                        )
                    if not self._handle_key(cv2.waitKey(1) & 0xFF):
                        break
                    continue

                frame_no += 1
                if frame_no % analyze_every == 0:
                    analysis = self.pipeline.analyze(frame)
                    self.state.update_vision(analysis, camera.measured_fps)

                    # Debounced print status -- the source of truth for the
                    # voice assistant and any future feed-control logic.
                    observation = self.monitor.update(frame, analysis, camera_connected=True)
                    self.state.update_print_observation(observation)
                    self._persist(observation)

                    alerts = self.rules.evaluate_vision(analysis)
                    if alerts:
                        self.state.add_alerts(alerts)
                        if self.on_alerts is not None:
                            self.on_alerts(alerts)

                if config.SHOW_CAMERA_WINDOW:
                    overlaid = draw_overlay(
                        frame, analysis, camera.measured_fps, self.monitor.latest
                    )
                    composed = self.dashboard.render(overlaid, self.state.dashboard_view())

                    if not window_created:
                        self._create_window(composed)
                        window_created = True
                    cv2.imshow(_WINDOW, composed)

                    if not self._handle_key(cv2.waitKey(1) & 0xFF):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            camera.release()
            if config.SHOW_CAMERA_WINDOW:
                cv2.destroyAllWindows()
            self.state.set_camera_status("stopped")
            print("[CAMERA] Stopped.")

        return True
