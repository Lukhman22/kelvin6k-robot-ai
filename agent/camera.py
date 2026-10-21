"""
Camera capture for the connected external USB camera.

Windows notes (why this file is more than a thin VideoCapture wrapper):
  * OpenCV's default backend on Windows is Media Foundation (MSMF). Many
    external USB/UVC cameras open under MSMF but then fail on read with
    `cap_msmf.cpp ... OnReadSample() is called with error status: -2147024890`.
    DirectShow (CAP_DSHOW) drives the same cameras reliably, so it is tried
    FIRST on Windows.
  * A camera can also open at its default mode but fail at a requested
    resolution, so several modes are attempted, and MJPG is tried because most
    USB cameras only reach higher resolutions/frame rates in MJPG.
  * "Opened" is never trusted on its own -- frames must actually arrive before
    an attempt is accepted.
"""

import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import quote

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from agent import config
from agent.logging_setup import get_logger

log = get_logger("CAMERA")

_WINDOWS = sys.platform.startswith("win")

# Frames to pull before trusting a camera; the first frame(s) are often junk or
# fail while the driver warms up.
_WARMUP_FRAMES = 5
_REQUIRED_GOOD_FRAMES = 2

# Modes tried in order. The configured mode goes first; these are the fallbacks
# that virtually every UVC camera supports. None = leave the driver's default.
_FALLBACK_MODES: List[Optional[Tuple[int, int]]] = [
    (640, 480),
    (1280, 720),
    (800, 600),
    (320, 240),
    None,
]

# Guards against two parts of the application opening the same camera at once,
# which on Windows makes the second one fail with "in use by another app".
_open_lock = threading.Lock()
_open_indexes: set = set()


class CameraError(Exception):
    """Raised when the camera cannot be opened or read."""


# EZVIZ is a Hikvision brand and models differ in stream path. Rather than
# requiring you to know the exact model, these are probed in order until one
# actually delivers frames; the working one is then printed so you can pin it
# in .env as CAMERA_RTSP_PATH.
EZVIZ_RTSP_PATHS = [
    "/H.264/ch1/main/av_stream",   # classic EZVIZ / Hikvision main stream
    "/H.264/ch1/sub/av_stream",    # sub stream (lower latency)
    "/Streaming/Channels/101",     # Hikvision-style main
    "/Streaming/Channels/102",     # Hikvision-style sub
    "/h264/ch1/main/av_stream",    # lowercase variant on some firmware
    "/ch1/main",                   # short form
    "/live/ch0",                   # some OEM firmware
]


def build_rtsp_url(path: Optional[str] = None) -> str:
    """
    Builds the RTSP URL from configuration.

    Uses CAMERA_RTSP_URL verbatim when set; otherwise assembles
    rtsp://user:password@ip:port/path from the individual settings.
    Credentials come from the environment only -- never hard-coded.
    """
    if config.CAMERA_RTSP_URL:
        return config.CAMERA_RTSP_URL

    if not config.CAMERA_IP:
        raise CameraError(
            "CAMERA_SOURCE=rtsp but neither CAMERA_RTSP_URL nor CAMERA_IP is set in .env.\n"
            "For an EZVIZ camera set:\n"
            "  CAMERA_IP=192.168.1.x\n"
            "  CAMERA_USERNAME=admin\n"
            "  CAMERA_PASSWORD=<6-character verification code from the camera label>\n"
            "or set the full CAMERA_RTSP_URL directly."
        )

    user = quote(config.CAMERA_USERNAME, safe="")
    password = quote(config.CAMERA_PASSWORD, safe="")
    chosen = path or config.CAMERA_RTSP_PATH
    chosen = chosen if chosen.startswith("/") else "/" + chosen
    credentials = f"{user}:{password}@" if config.CAMERA_USERNAME else ""
    return f"rtsp://{credentials}{config.CAMERA_IP}:{config.CAMERA_RTSP_PORT}{chosen}"


def mask_rtsp_url(url: str) -> str:
    """Same URL with the password replaced by ***, safe to print or log."""
    return re.sub(r"(rtsp://[^:/@]+:)[^@]*(@)", r"\1***\2", url)


class _quiet_opencv:
    """
    Silences OpenCV's internal videoio logging while probing.

    Probing deliberately tries combinations that will fail, and OpenCV prints a
    wall of MSMF/FFMPEG/obsensor warnings for each one. Those would bury the
    diagnostics we actually print.
    """

    def __enter__(self):
        self._previous = None
        try:
            logging_module = cv2.utils.logging
            self._previous = logging_module.getLogLevel()
            logging_module.setLogLevel(logging_module.LOG_LEVEL_SILENT)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._previous is not None:
                cv2.utils.logging.setLogLevel(self._previous)
        except Exception:
            pass
        return False


@dataclass
class OpenAttempt:
    """One (index, backend, mode) combination that was tried."""

    index: int
    backend: str
    requested: str
    fourcc: str
    ok: bool
    detail: str

    def __str__(self) -> str:
        mark = "OK  " if self.ok else "FAIL"
        return f"  [{mark}] index={self.index} backend={self.backend} mode={self.requested} fourcc={self.fourcc} -- {self.detail}"


def _backend_candidates() -> List[Tuple[str, int]]:
    """Backends to try, best-first for this platform."""
    if cv2 is None:
        return []
    preference = config.CAMERA_BACKEND

    dshow = ("DSHOW", getattr(cv2, "CAP_DSHOW", 700))
    msmf = ("MSMF", getattr(cv2, "CAP_MSMF", 1400))
    any_backend = ("ANY", getattr(cv2, "CAP_ANY", 0))

    if preference == "dshow":
        return [dshow]
    if preference == "msmf":
        return [msmf]
    if preference == "any":
        return [any_backend]

    # auto
    if _WINDOWS:
        # DirectShow first: it is the one that works with external USB cameras
        # when MSMF throws OnReadSample errors.
        return [dshow, msmf, any_backend]

    v4l2 = getattr(cv2, "CAP_V4L2", None)
    if v4l2 is not None:
        return [("V4L2", v4l2), any_backend]
    return [any_backend]


def list_camera_names() -> List[str]:
    """
    DirectShow device names in index order, so index N here is index N for
    cv2.VideoCapture(N, cv2.CAP_DSHOW).

    Requires the optional `pygrabber` package (pip install pygrabber). Returns
    [] when it isn't available or we're not on Windows -- name-based selection
    is then skipped and index scanning is used instead.
    """
    if not _WINDOWS:
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        return []


def list_device_formats(index: int) -> List[Tuple[int, int, str]]:
    """
    Modes the camera actually advertises, as [(width, height, media_type), ...].

    Read straight from the DirectShow device via the optional `pygrabber`
    package, so we use what the camera really supports instead of assuming
    1280x720/MJPG. Returns [] when unavailable.
    """
    if not _WINDOWS:
        return []
    graph = None
    try:
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        graph.add_video_input_device(index)
        formats = graph.get_input_device().get_formats()
        modes: List[Tuple[int, int, str]] = []
        for fmt in formats:
            width, height = fmt.get("width"), fmt.get("height")
            media = str(fmt.get("media_type_str", ""))
            if width and height and (width, height, media) not in modes:
                modes.append((int(width), int(height), media))
        return modes
    except Exception:
        return []
    finally:
        # Always let go of the device -- holding it would block our own capture.
        try:
            if graph is not None:
                graph.remove_filters()
        except Exception:
            pass


def _validate_frames(capture: "cv2.VideoCapture", timeout: Optional[float] = None) -> Tuple[bool, str]:
    """
    Actually pulls frames -- isOpened() alone lies on Windows and on RTSP.

    Time-based rather than a fixed read count: network streams (and some USB
    cameras) need a second or two before the first frame arrives, and a handful
    of instant reads would declare a perfectly good camera dead.
    """
    deadline = time.time() + (timeout if timeout is not None else config.CAMERA_WARMUP_SECONDS)
    good = 0
    attempts = 0
    last_error = "no frames returned"

    while time.time() < deadline:
        ok, frame = capture.read()
        attempts += 1
        if ok and frame is not None and frame.size > 0:
            good += 1
            if good >= _REQUIRED_GOOD_FRAMES:
                return True, f"{good} valid frames, {frame.shape[1]}x{frame.shape[0]}"
        else:
            last_error = (
                f"read() returned no frame after {attempts} attempts "
                f"(driver/backend rejected the stream)"
            )
        time.sleep(0.05)

    return False, last_error


def _try_open(
    index: int,
    backend_name: str,
    backend_id: int,
    mode: Optional[Tuple[int, int]],
    fps: int,
    use_mjpg: bool,
) -> Tuple[Optional["cv2.VideoCapture"], OpenAttempt]:
    """Opens one (index, backend, mode, fourcc) combination and verifies frames."""
    requested = f"{mode[0]}x{mode[1]}" if mode else "driver default"
    fourcc_name = "MJPG" if use_mjpg else "default"

    capture = cv2.VideoCapture(index, backend_id)
    if not capture.isOpened():
        capture.release()
        return None, OpenAttempt(index, backend_name, requested, fourcc_name, False, "could not open device")

    if use_mjpg:
        # Most USB cameras need MJPG for anything above small/low-fps modes.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if mode is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, mode[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, mode[1])
    if fps:
        capture.set(cv2.CAP_PROP_FPS, fps)

    ok, detail = _validate_frames(capture)
    if not ok:
        capture.release()
        return None, OpenAttempt(index, backend_name, requested, fourcc_name, False, detail)

    return capture, OpenAttempt(index, backend_name, requested, fourcc_name, True, detail)


class CameraStream:
    """
    Wrapper around OpenCV VideoCapture for the connected USB/web camera.

    Public API is unchanged: open(), read(), measured_fps, actual_resolution,
    release(), and use as a context manager. read() still returns None on a
    dropped frame so callers keep running instead of crashing.
    """

    def __init__(
        self,
        index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        name_hint: Optional[str] = None,
        allow_scan: Optional[bool] = None,
        source: Optional[str] = None,
        rtsp_url: Optional[str] = None,
    ):
        # "usb" -> local UVC camera by index; "rtsp" -> EZVIZ / IP camera.
        self.source = (config.CAMERA_SOURCE if source is None else source).strip().lower()
        self._explicit_rtsp_url = rtsp_url
        self.rtsp_url: str = ""
        self._consecutive_failures = 0
        self._reconnect_attempts = 0

        self.index = config.CAMERA_INDEX if index is None else index
        self.width = config.CAMERA_WIDTH if width is None else width
        self.height = config.CAMERA_HEIGHT if height is None else height
        self.fps = config.CAMERA_FPS if fps is None else fps
        self.name_hint = config.CAMERA_NAME_HINT if name_hint is None else name_hint
        self.allow_scan = config.CAMERA_AUTO_SCAN if allow_scan is None else allow_scan

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_times = deque(maxlen=30)
        self._claimed_index: Optional[int] = None

        # Populated by open() -- useful diagnostics for callers/tests.
        self.backend_name: str = ""
        self.device_name: str = ""
        self.attempts: List[OpenAttempt] = []

    # --- index selection ------------------------------------------------

    def _candidate_indexes(self) -> List[int]:
        """
        Configured index first, then a name match, then a scan bounded by the
        number of cameras Windows ACTUALLY reports.

        Scanning past the real device count is what produced hundreds of
        "Camera index out of range" errors, so when the device list is known we
        never probe beyond it. CAMERA_SCAN_MAX_INDEX is only the ceiling for the
        blind case where no device list is available.
        """
        candidates = [self.index]

        names = list_camera_names()
        if self.name_hint and names:
            hint = self.name_hint.lower()
            for i, name in enumerate(names):
                if hint in name.lower() and i not in candidates:
                    # Preferred: the device whose name matches the hint.
                    candidates.insert(0, i)

        if self.allow_scan:
            if names:
                # Device list known -> only real devices are worth probing.
                upper = len(names) - 1
            else:
                upper = config.CAMERA_SCAN_MAX_INDEX
            for i in range(upper + 1):
                if i not in candidates:
                    candidates.append(i)

        # Never probe an index we know doesn't exist.
        if names:
            candidates = [i for i in candidates if i < len(names)] or [self.index]

        return candidates

    def _mode_candidates(self, index: int) -> List[Optional[Tuple[int, int]]]:
        """Configured mode, then the camera's own advertised modes, then fallbacks."""
        modes: List[Optional[Tuple[int, int]]] = [(self.width, self.height)]

        for width, height, _media in list_device_formats(index):
            if (width, height) not in modes:
                modes.append((width, height))

        for mode in _FALLBACK_MODES:
            if mode not in modes:
                modes.append(mode)
        return modes

    def _fourcc_candidates(self, index: int) -> List[bool]:
        """
        Whether to request MJPG. Many USB cameras need it for higher modes, but
        some (like simple UVC cameras that only offer YUY2) fail when it is
        forced -- so only try MJPG when the device advertises it, or when we
        can't read the device's format list at all.
        """
        formats = list_device_formats(index)
        if not formats:
            return [True, False]
        has_mjpg = any("mjp" in media.lower() for _w, _h, media in formats)
        return [True, False] if has_mjpg else [False]

    # --- lifecycle ------------------------------------------------------

    def open(self) -> None:
        """
        Opens the camera, trying backends (DirectShow first on Windows),
        indexes, resolutions, and MJPG until frames actually arrive.

        Raises CameraError listing everything that was tried if none work.
        """
        if cv2 is None:
            raise CameraError("OpenCV (cv2) is not installed. Camera capture is unavailable.")

        if self._capture is not None:
            return  # already open -- don't grab the device twice

        if self.source == "rtsp":
            self._open_rtsp()
            return

        self.attempts = []
        names = list_camera_names()

        with _quiet_opencv():
            self._open_first_working(names)

        if self._capture is None:
            raise CameraError(self._failure_report())

    # --- RTSP / IP camera (EZVIZ) ---------------------------------------

    def _rtsp_url_candidates(self) -> List[str]:
        """
        URLs to try. An explicit URL or a non-default CAMERA_RTSP_PATH is used
        alone; otherwise the known EZVIZ/Hikvision paths are probed in order.
        """
        if self._explicit_rtsp_url:
            return [self._explicit_rtsp_url]
        if config.CAMERA_RTSP_URL:
            return [config.CAMERA_RTSP_URL]
        if config.CAMERA_RTSP_PATH and config.CAMERA_RTSP_PATH != EZVIZ_RTSP_PATHS[0]:
            return [build_rtsp_url(config.CAMERA_RTSP_PATH)]
        return [build_rtsp_url(path) for path in EZVIZ_RTSP_PATHS]

    def _open_rtsp(self) -> None:
        """Opens the RTSP stream with low-latency settings, probing stream paths."""
        # FFMPEG transport options must be set before VideoCapture is created.
        # TCP avoids the packet loss/corruption that plagues RTSP over UDP.
        # stimeout is in microseconds -- this is the connection timeout.
        timeout_us = int(config.CAMERA_RTSP_TIMEOUT_SECONDS * 1_000_000)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{config.CAMERA_RTSP_TRANSPORT}|stimeout;{timeout_us}|max_delay;500000"
        )

        candidates = self._rtsp_url_candidates()
        failures: List[str] = []

        for url in candidates:
            safe_url = mask_rtsp_url(url)
            log.info("Trying RTSP stream %s", safe_url)

            with _quiet_opencv():
                capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if not capture.isOpened():
                    capture.release()
                    failures.append(f"  [FAIL] {safe_url} -- could not connect")
                    continue

                # Keep only the newest frame: RTSP buffering is the main source
                # of latency, and a stale frame is worse than none here.
                try:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                ok, detail = _validate_frames(capture)

            if not ok:
                capture.release()
                failures.append(f"  [FAIL] {safe_url} -- connected but {detail}")
                continue

            self._capture = capture
            self.rtsp_url = url
            self.backend_name = "FFMPEG/RTSP"
            self.device_name = safe_url
            self._consecutive_failures = 0
            self._reconnect_attempts = 0

            actual_w, actual_h = self.actual_resolution
            log.info("RTSP stream connected")
            log.info("  url               : %s", safe_url)
            log.info("  transport         : %s", config.CAMERA_RTSP_TRANSPORT)
            log.info("  actual resolution : %dx%d", actual_w, actual_h)
            log.info("  driver FPS        : %s", self.reported_fps)
            log.info("  frames received   : yes (%s)", detail)
            if len(candidates) > 1:
                path = "/" + url.split("/", 3)[-1] if url.count("/") >= 3 else url
                log.info("  Pin this in .env: CAMERA_RTSP_PATH=%s", path)
            return

        raise CameraError(
            "Could not open the RTSP camera.\n\nTried:\n"
            + "\n".join(failures)
            + "\n\nCheck that:\n"
            "  - the camera IP is correct and reachable (try: ping <CAMERA_IP>)\n"
            "  - the PC and camera are on the same network\n"
            "  - CAMERA_USERNAME / CAMERA_PASSWORD are correct (EZVIZ: the\n"
            "    6-character VERIFICATION CODE on the camera label, not your\n"
            "    EZVIZ account password)\n"
            "  - RTSP is enabled on the camera and port 554 is reachable\n"
            "  - if you know the exact stream path, set CAMERA_RTSP_PATH or\n"
            "    the full CAMERA_RTSP_URL in .env"
        )

    def reconnect(self) -> bool:
        """
        Tears down and re-opens the stream. Returns True on success.
        Used for network cameras, which drop out far more often than USB ones.
        """
        limit = config.CAMERA_MAX_RECONNECT_ATTEMPTS
        if limit and self._reconnect_attempts >= limit:
            return False

        self._reconnect_attempts += 1
        attempt_label = f"{self._reconnect_attempts}" + (f"/{limit}" if limit else "")
        print(f"[CAMERA] Connection lost -- reconnect attempt {attempt_label} "
              f"in {config.CAMERA_RECONNECT_DELAY_SECONDS:.0f}s...", file=sys.stderr)

        self.release()
        time.sleep(config.CAMERA_RECONNECT_DELAY_SECONDS)

        try:
            self.open()
            print("[CAMERA] Reconnected.")
            return True
        except CameraError as exc:
            print(f"[CAMERA] Reconnect failed: {str(exc).splitlines()[0]}", file=sys.stderr)
            return False

    def _open_first_working(self, names: List[str]) -> None:
        for index in self._candidate_indexes():
            with _open_lock:
                if index in _open_indexes:
                    self.attempts.append(
                        OpenAttempt(index, "-", "-", "-", False, "already opened by this application")
                    )
                    continue

            modes = self._mode_candidates(index)
            fourccs = self._fourcc_candidates(index)

            for backend_name, backend_id in _backend_candidates():
                for mode in modes:
                    for use_mjpg in fourccs:
                        capture, attempt = _try_open(
                            index, backend_name, backend_id, mode, self.fps, use_mjpg
                        )
                        self.attempts.append(attempt)
                        if capture is None:
                            continue

                        with _open_lock:
                            if index in _open_indexes:
                                capture.release()
                                continue
                            _open_indexes.add(index)

                        self._capture = capture
                        self._claimed_index = index
                        self.index = index
                        self.backend_name = backend_name
                        self.device_name = names[index] if index < len(names) else ""
                        self._print_diagnostics(mode, use_mjpg)
                        return

    def _print_diagnostics(self, mode, use_mjpg: bool) -> None:
        actual_w, actual_h = self.actual_resolution
        requested = f"{mode[0]}x{mode[1]}" if mode else "driver default"
        name = f" ({self.device_name})" if self.device_name else ""
        print("[CAMERA] Opened successfully")
        print(f"[CAMERA]   index              : {self.index}{name}")
        print(f"[CAMERA]   backend            : {self.backend_name}")
        print(f"[CAMERA]   requested resolution: {requested}")
        print(f"[CAMERA]   actual resolution  : {actual_w}x{actual_h}")
        print(f"[CAMERA]   requested FPS      : {self.fps}")
        print(f"[CAMERA]   driver FPS         : {self.reported_fps}")
        print(f"[CAMERA]   fourcc             : {'MJPG' if use_mjpg else 'driver default'}")
        print(f"[CAMERA]   frames received    : yes")

    def _failure_report(self) -> str:
        names = list_camera_names()
        lines = [
            f"Could not open any working camera (tried {len(self.attempts)} combinations).",
            "",
            "Attempts:",
        ]
        lines += [str(a) for a in self.attempts]
        lines += [
            "",
            "Detected camera devices: " + (", ".join(f"{i}={n}" for i, n in enumerate(names)) if names else
                                           "unknown (install pygrabber for names: pip install pygrabber)"),
            "",
            "Things to check:",
            "  - Is the external USB camera plugged in and enumerated by Windows?",
            "  - Close anything else using it (Camera app, Teams, Zoom, OBS, browser tabs).",
            "  - Settings > Privacy & security > Camera: allow desktop apps to access the camera.",
            "  - Run `python find_cameras.py` to scan every index/backend and preview them.",
            "  - Set CAMERA_INDEX (and optionally CAMERA_NAME_HINT / CAMERA_BACKEND) in .env.",
        ]
        return "\n".join(lines)

    # --- capture --------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """
        Returns the next frame, or None if the frame was dropped.

        For network cameras this also counts consecutive failures and triggers a
        reconnect once CAMERA_READ_FAILURE_LIMIT is hit -- a temporary Wi-Fi
        glitch must not take the application down.
        """
        if self._capture is None:
            return None

        ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            self._consecutive_failures += 1
            if (
                self.source == "rtsp"
                and self._consecutive_failures >= config.CAMERA_READ_FAILURE_LIMIT
            ):
                self._consecutive_failures = 0
                self.reconnect()  # failure is reported and tolerated, never raised
            return None

        self._consecutive_failures = 0
        self._frame_times.append(time.time())
        return frame

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def measured_fps(self) -> float:
        """Actual measured FPS over the recent frame window (0.0 until warmed up)."""
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed

    @property
    def reported_fps(self) -> float:
        if self._capture is None:
            return 0.0
        return round(float(self._capture.get(cv2.CAP_PROP_FPS)), 1)

    @property
    def actual_resolution(self) -> tuple:
        if self._capture is None:
            return (0, 0)
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    def is_connected(self) -> bool:
        """True while a stream is open (USB or RTSP)."""
        return self._capture is not None

    def close(self) -> None:
        """Alias for release() -- clean shutdown."""
        self.release()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._claimed_index is not None:
            with _open_lock:
                _open_indexes.discard(self._claimed_index)
            self._claimed_index = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def __del__(self):
        # Last-resort cleanup so a forgotten stream doesn't hold the device.
        try:
            self.release()
        except Exception:
            pass


def probe_cameras(max_index: Optional[int] = None) -> List[dict]:
    """
    Scans indexes across backends and reports which ones deliver real frames.
    Used by find_cameras.py. Each result:
        {index, name, backend, resolution, fps, working}
    """
    if max_index is None:
        max_index = config.CAMERA_SCAN_MAX_INDEX

    names = list_camera_names()
    # Bound the scan by the real device count when Windows reports one, so we
    # don't emit "Camera index out of range" errors for indexes that can't exist.
    upper = (len(names) - 1) if names else max_index
    results: List[dict] = []

    with _quiet_opencv():
        for index in range(upper + 1):
            name = names[index] if index < len(names) else ""
            entry = {"index": index, "name": name, "backend": None,
                     "resolution": None, "fps": None, "working": False}

            # Use the modes the device advertises; fall back to a safe probe.
            advertised = list_device_formats(index)
            probe_mode = (advertised[0][0], advertised[0][1]) if advertised else (640, 480)
            use_mjpg = any("mjp" in m[2].lower() for m in advertised) if advertised else True

            for backend_name, backend_id in _backend_candidates():
                capture, _ = _try_open(index, backend_name, backend_id, probe_mode, 30, use_mjpg)
                if capture is None:
                    capture, _ = _try_open(index, backend_name, backend_id, None, 30, False)
                if capture is not None:
                    entry.update(
                        backend=backend_name,
                        resolution=(
                            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        ),
                        fps=round(float(capture.get(cv2.CAP_PROP_FPS)), 1),
                        working=True,
                    )
                    capture.release()
                    break

            results.append(entry)

    return results


def supported_modes(index: int, backend_name: Optional[str] = None) -> List[Tuple[int, int]]:
    """
    Detects which of the common modes the camera actually delivers, by setting
    each and reading back what the driver reports. Used instead of assuming
    1280x720 works.
    """
    backends = _backend_candidates()
    if backend_name:
        backends = [b for b in backends if b[0] == backend_name] or backends

    modes_to_test = [(320, 240), (640, 480), (800, 600), (1280, 720), (1920, 1080)]
    supported: List[Tuple[int, int]] = []

    for _, b_id in backends:
        capture = cv2.VideoCapture(index, b_id)
        if not capture.isOpened():
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for width, height in modes_to_test:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = capture.read()
            if ok and frame is not None and frame.size > 0:
                actual = (frame.shape[1], frame.shape[0])
                if actual not in supported:
                    supported.append(actual)
        capture.release()
        if supported:
            break

    return supported
