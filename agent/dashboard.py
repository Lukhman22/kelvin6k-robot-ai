"""
Live dashboard rendered with OpenCV: the camera frame with detection overlay,
plus a side panel showing vision, machine, alert, and voice state.

Anything not actually measured renders as "--" or "N/A (not connected)" so the
screen never implies a value the system doesn't have.
"""

from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from agent import config
from agent.rules import AlertLevel

_PANEL_WIDTH = 460
# Enough vertical room for every section (camera, walls, quality, machine,
# alerts, voice) regardless of the camera's resolution.
_MIN_PANEL_HEIGHT = 1000
_BG = (28, 28, 30)
_FG = (235, 235, 235)
_DIM = (150, 150, 155)
_ACCENT = (120, 200, 255)
_OK = (120, 220, 140)
_WARN = (90, 190, 255)
_CRIT = (90, 90, 245)
_FONT = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0) if cv2 is not None else 0


def _fmt(value, suffix: str = "", unavailable: str = "--") -> str:
    if value is None:
        return unavailable
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return f"{value}{suffix}"


def draw_overlay(
    frame: np.ndarray, analysis, fps: float, observation=None, metrology=None
) -> np.ndarray:
    """
    Draws the ROI, detected layer boundaries, and headline readouts.

    `metrology` is optional and defaults to None, so every existing call
    site keeps working unchanged. When a MetrologyResult is supplied the
    perpendicular width measurement is drawn on top.
    """
    if cv2 is None or frame is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]

    # ROI offset: analysis coordinates are relative to the cropped region.
    off_x, off_y, roi_w = 0, 0, w
    if analysis is not None and analysis.roi is not None:
        off_x, off_y, roi_w, roi_h = analysis.roi
        cv2.rectangle(out, (off_x, off_y), (off_x + roi_w, off_y + roi_h), (255, 190, 90), 2)
        cv2.putText(out, "ROI", (off_x + 5, off_y + 20), _FONT, 0.5, (255, 190, 90), 1)

    if analysis is not None and analysis.layers.detected:
        for row in analysis.layers.boundary_rows:
            cv2.line(out, (off_x, off_y + int(row)), (off_x + roi_w, off_y + int(row)), (0, 220, 255), 1)

    # Print status band (debug view only -- never required in production).
    if observation is not None:
        lines = [
            f"PRINTING: {'YES' if observation.printing else 'NO'}",
            f"LAYER: {'DETECTED' if observation.layer_detected else 'NOT DETECTED'}",
            (f"WIDTH: {observation.layer_width} mm" if observation.layer_width is not None
             else f"WIDTH: {observation.layer_width_px} px (uncalibrated)"
             if observation.layer_width_px is not None else "WIDTH: --"),
            f"CONFIDENCE: {observation.confidence * 100:.0f}%",
            f"CAMERA: {'CONNECTED' if observation.camera_connected else 'ERROR'}",
            f"STATUS: {observation.status.value}",
        ]
        colors = [
            _OK if observation.printing else _DIM,
            _OK if observation.layer_detected else _DIM,
            _FG if observation.layer_width is not None else _DIM,
            _FG,
            _OK if observation.camera_connected else _CRIT,
            _CRIT if observation.status.value == "ABNORMAL_LAYER" else _ACCENT,
        ]
        y = 60
        cv2.rectangle(out, (0, 40), (300, 40 + 24 * len(lines)), (0, 0, 0), -1)
        for text, color in zip(lines, colors):
            cv2.putText(out, text, (10, y), _FONT, 0.5, color, 1)
            y += 24

    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 23), _FONT, 0.6, _ACCENT, 2)

    if analysis is None:
        status_text = "LAYER STATUS: NOT DETECTED"
        cv2.putText(out, status_text, (140, 23), _FONT, 0.6, _DIM, 2)
        if metrology is not None:
            out = draw_metrology(out, metrology)
        return out

    layers = analysis.layers
    status = "DETECTED" if layers.detected else "NOT DETECTED"
    color = _OK if layers.detected else _DIM

    width_mm = layers.layer_width_mm
    width_text = f"{width_mm} mm" if width_mm is not None else (
        f"{layers.layer_width_px} px (uncalibrated)" if layers.layer_width_px is not None else "--"
    )

    cv2.putText(out, f"LAYER STATUS: {status}", (140, 23), _FONT, 0.6, color, 2)
    cv2.putText(out, f"BANDS VISIBLE: {_fmt(layers.visible_layer_bands)}", (420, 23), _FONT, 0.6, _FG, 2)
    cv2.putText(out, f"LAYER WIDTH: {width_text}", (700, 23), _FONT, 0.6, _FG, 2)
    if metrology is not None:
        out = draw_metrology(out, metrology)
    return out


class DashboardRenderer:
    """Renders the side dashboard panel and composes it with the video frame."""

    def _blank_panel(self, height: int) -> np.ndarray:
        return np.full((height, _PANEL_WIDTH, 3), _BG, dtype=np.uint8)

    def _section(self, panel: np.ndarray, y: int, title: str) -> int:
        cv2.putText(panel, title, (14, y), _FONT, 0.55, _ACCENT, 2)
        cv2.line(panel, (14, y + 8), (_PANEL_WIDTH - 14, y + 8), (70, 70, 75), 1)
        return y + 30

    def _row(self, panel: np.ndarray, y: int, label: str, value: str, color=_FG) -> int:
        cv2.putText(panel, label, (18, y), _FONT, 0.45, _DIM, 1)
        cv2.putText(panel, value, (250, y), _FONT, 0.45, color, 1)
        return y + 22

    def render(self, frame: Optional[np.ndarray], view: dict) -> np.ndarray:
        if cv2 is None:
            return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
        vision = view.get("vision")
        reading = view.get("reading")
        machine = view.get("machine")

        # The panel needs a fixed minimum height for all sections to fit; if the
        # camera frame is shorter, the frame is padded rather than the panel
        # clipped, so low-resolution cameras still show the full dashboard.
        frame_height = frame.shape[0] if frame is not None else _MIN_PANEL_HEIGHT
        height = max(frame_height, _MIN_PANEL_HEIGHT)
        panel = self._blank_panel(height)
        y = 30

        # --- LIVE CAMERA ---
        y = self._section(panel, y, "LIVE CAMERA")
        y = self._row(panel, y, "Camera", str(view.get("camera_status", "--")))
        y = self._row(panel, y, "FPS", f"{view.get('camera_fps', 0.0):.1f}")

        observation = view.get("print_observation")
        if observation is not None:
            status_color = _CRIT if observation.status.value == "ABNORMAL_LAYER" else (
                _OK if observation.printing else _DIM)
            y = self._row(panel, y, "Printing", "YES" if observation.printing else "NO", status_color)
            y = self._row(panel, y, "Print status", observation.status.value, status_color)
            y = self._row(panel, y, "Layer", observation.layer_status.value,
                          _OK if observation.layer_detected else _DIM)
            y = self._row(panel, y, "Confidence", f"{observation.confidence * 100:.0f}%")
            # Shown so PRINT_MOTION_THRESHOLD can be tuned by observation:
            # note this idle vs printing, then set the threshold between them.
            y = self._row(panel, y, "Motion (ROI)", f"{observation.motion_score:.2f}")
            if observation.layer_width is not None:
                wcolor = _OK if observation.width_valid else _CRIT
                y = self._row(panel, y, "Layer width", f"{observation.layer_width} mm", wcolor)
            elif observation.layer_width_px is not None:
                y = self._row(panel, y, "Layer width",
                              f"{observation.layer_width_px} px (uncalib.)", _DIM)

        if vision is None:
            y = self._row(panel, y, "Vision", "no analysis yet", _DIM)
        else:
            layers, walls, quality = vision.layers, vision.walls, vision.quality
            y = self._row(panel, y, "Layer status", "DETECTED" if layers.detected else "NOT DETECTED",
                          _OK if layers.detected else _DIM)
            y = self._row(panel, y, "Visible layer bands", _fmt(layers.visible_layer_bands))
            y = self._row(panel, y, "Current layer no.", "N/A (needs machine data)", _DIM)
            # Layer width is already shown in the print-status rows above.
            y = self._row(panel, y, "Width trend", _fmt(layers.layer_width_trend))
            y = self._row(panel, y, "Layer consistency", _fmt(layers.layer_consistency_percent, " %"))
            y = self._row(panel, y, "Print progress", "N/A (needs G-code)", _DIM)

            y += 8
            y = self._section(panel, y, "WALL ANALYSIS")
            y = self._row(panel, y, "Outer wall", _fmt(walls.outer_wall_detected))
            y = self._row(panel, y, "Inner wall",
                          "YES" if walls.inner_wall_detected else "not visible",
                          _FG if walls.inner_wall_detected else _DIM)
            y = self._row(panel, y, "Wall count", "not reliable from view", _DIM)
            if walls.wall_thickness_mm is not None:
                y = self._row(panel, y, "Wall thickness", f"{walls.wall_thickness_mm} mm")
            else:
                y = self._row(panel, y, "Wall thickness",
                              f"{_fmt(walls.wall_thickness_px, ' px')} (uncalib.)", _DIM)
            y = self._row(panel, y, "Wall continuity", _fmt(walls.wall_continuity_percent, " %"))
            y = self._row(panel, y, "Gaps", _fmt(walls.gap_count))

            y += 8
            y = self._section(panel, y, "PRINT QUALITY")
            if quality.score_percent is None:
                y = self._row(panel, y, "Quality", "cannot be scored yet", _DIM)
            else:
                qcolor = _OK if quality.status == "GOOD" else (_WARN if quality.status == "FAIR" else _CRIT)
                y = self._row(panel, y, "Quality", f"{quality.score_percent} %  {quality.status}", qcolor)

        # --- MACHINE STATUS ---
        y += 8
        y = self._section(panel, y, "MACHINE STATUS")
        if reading is None:
            y = self._row(panel, y, "Process sensors", "no reading yet", _DIM)
        else:
            y = self._row(panel, y, "Source", reading.source, _WARN)
            y = self._row(panel, y, "Material flow", f"{reading.material_flow_percent} %")
            y = self._row(panel, y, "Machine state", reading.machine_state)
            y = self._row(panel, y, "Nozzle temp", f"{reading.nozzle_temp_c} C")

        if machine is None or machine.source == "NOT_CONNECTED":
            y = self._row(panel, y, "Pump", "N/A (not connected)", _DIM)
            y = self._row(panel, y, "Pump speed", "N/A (not connected)", _DIM)
            y = self._row(panel, y, "Accelerator", "N/A (not connected)", _DIM)
            y = self._row(panel, y, "Accel. speed", "N/A (not connected)", _DIM)
            y = self._row(panel, y, "Print speed", "N/A (not connected)", _DIM)
        else:
            y = self._row(panel, y, "Pump", _fmt(machine.pump_running, unavailable="N/A"))
            y = self._row(panel, y, "Pump speed", _fmt(machine.pump_speed_rpm, " RPM", "N/A"))
            y = self._row(panel, y, "Accelerator", _fmt(machine.accelerator_running, unavailable="N/A"))
            y = self._row(panel, y, "Accel. speed", _fmt(machine.accelerator_speed_rpm, " RPM", "N/A"))
            y = self._row(panel, y, "Print speed", _fmt(machine.print_speed_mm_s, " mm/s", "N/A"))

        # --- ALERTS ---
        y += 8
        y = self._section(panel, y, "ALERTS")
        alerts = view.get("alerts") or []
        if not alerts:
            y = self._row(panel, y, "None", "", _DIM)
        else:
            for alert, _ts in alerts[-4:]:
                color = {
                    AlertLevel.INFO: _FG,
                    AlertLevel.WARNING: _WARN,
                    AlertLevel.CRITICAL: _CRIT,
                }[alert.level]
                text = alert.message[:52] + ("..." if len(alert.message) > 52 else "")
                cv2.putText(panel, text, (18, y), _FONT, 0.38, color, 1)
                y += 18

        # --- VOICE ---
        y += 10
        y = self._section(panel, y, "VOICE")
        question = view.get("last_question") or "--"
        answer = view.get("last_answer") or "--"
        cv2.putText(panel, "Q: " + question[:46], (18, y), _FONT, 0.4, _FG, 1)
        y += 20
        for line_no in range(2):
            chunk = answer[line_no * 46 : (line_no + 1) * 46]
            if not chunk:
                break
            prefix = "A: " if line_no == 0 else "   "
            cv2.putText(panel, prefix + chunk, (18, y), _FONT, 0.4, _OK, 1)
            y += 18

        cv2.putText(panel, "Q or ESC to quit", (18, height - 14), _FONT, 0.4, _DIM, 1)

        if frame is None:
            return panel

        if frame.shape[0] < height:
            pad = np.full((height - frame.shape[0], frame.shape[1], 3), _BG, dtype=np.uint8)
            frame = np.vstack([frame, pad])
        return np.hstack([frame, panel])


# ---------------------------------------------------------------------------
# Perpendicular layer-width metrology overlay.
#
# Drawn server-side and streamed as MJPEG, like the rest of this overlay, so
# the React component stays a plain <img> and needs no change.
# ---------------------------------------------------------------------------

_METRO_CONTOUR = (255, 0, 0)        # BGR blue: the bead outline
_METRO_CENTER = (255, 255, 255)     # white dotted centreline
_METRO_STATION = (0, 230, 255)      # amber perpendicular measurement lines
_METRO_INVALID = (90, 90, 245)      # red for a station that could not measure


def _dotted_polyline(img, points, colour, thickness=2, on=8, off=8):
    """
    Draws a dashed polyline.

    OpenCV has no dashed-line primitive, so the path is walked and alternate
    spans are skipped, carrying the remainder across segment boundaries so the
    dash rhythm stays even around curves.
    """
    if points is None or len(points) < 2:
        return
    carry = 0.0
    drawing = True
    for i in range(len(points) - 1):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[i + 1][0]), float(points[i + 1][1])
        seg = float(np.hypot(x2 - x1, y2 - y1))
        if seg < 1e-6:
            continue
        pos = 0.0
        while pos < seg:
            span = (on if drawing else off) - carry
            end = min(pos + span, seg)
            if drawing:
                ax = x1 + (x2 - x1) * (pos / seg)
                ay = y1 + (y2 - y1) * (pos / seg)
                bx = x1 + (x2 - x1) * (end / seg)
                by = y1 + (y2 - y1) * (end / seg)
                cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), colour, thickness)
            if end - pos >= span:
                drawing = not drawing
                carry = 0.0
            else:
                carry += end - pos
            pos = end


def _block(img, x, y, title, lines, colour=_FG, width=250):
    """
    A titled readout block over a dark translucent backing, so the text stays
    legible against bright concrete.
    """
    height = 22 + 18 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 8, y - 16), (x + width, y + height - 12), (20, 20, 22), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, title, (x, y), _FONT, 0.45, _ACCENT, 1)
    yy = y + 20
    for text, tint in lines:
        cv2.putText(img, text, (x, yy), _FONT, 0.42, tint or colour, 1)
        yy += 18


def _status_colour(status: str):
    if status == "PASS":
        return _OK
    if status in ("UNDER-WIDTH", "OVER-WIDTH"):
        return _WARN
    return _CRIT


def draw_metrology(frame: np.ndarray, result) -> np.ndarray:
    """
    Draws the blue bead contour, the white dotted centreline, the W1..Wn
    perpendicular measurement lines and their widths, and the readout blocks.

    Never raises: a drawing failure must not take down the camera stream.
    """
    if cv2 is None or frame is None:
        return frame
    try:
        out = frame
        height, width = out.shape[:2]

        if getattr(result, "contour", None) is not None:
            cv2.drawContours(out, [result.contour], -1, _METRO_CONTOUR, 2)

        if getattr(result, "centerline", None) is not None:
            _dotted_polyline(out, result.centerline, _METRO_CENTER, 2)

        for station in getattr(result, "stations", []) or []:
            if station.left is not None and station.right is not None:
                colour = _METRO_STATION if station.valid else _METRO_INVALID
                p1 = (int(station.left[0]), int(station.left[1]))
                p2 = (int(station.right[0]), int(station.right[1]))
                cv2.line(out, p1, p2, colour, 2)
                cv2.circle(out, p1, 3, colour, -1)
                cv2.circle(out, p2, 3, colour, -1)
                label = station.label
                if station.width_mm is not None:
                    label = f"{station.label} {station.width_mm:.1f}"
                cv2.putText(out, label, (p2[0] + 6, p2[1]), _FONT, 0.4, colour, 1)
            else:
                cv2.circle(out, (int(station.x), int(station.y)), 4, _METRO_INVALID, 1)
                cv2.putText(out, f"{station.label} --",
                            (int(station.x) + 6, int(station.y)), _FONT, 0.4,
                            _METRO_INVALID, 1)

        status = getattr(result, "status", "MEASUREMENT INVALID")
        colour = _status_colour(status)
        valid = getattr(result, "valid_count", 0)
        total = getattr(result, "station_count", 0)

        local = "--"
        for station in getattr(result, "stations", []) or []:
            if station.valid and station.width_mm is not None:
                local = f"{station.width_mm:.1f} mm"
                break

        _block(out, 14, 50, "METROLOGY DEBUG", [
            (f"MASK: {getattr(result, 'mask_method', 'NO DETECTION')}", None),
            (f"CONTOUR: {'OK' if getattr(result, 'contour_ok', False) else 'NONE'}", None),
            (f"CENTERLINE: {'OK' if getattr(result, 'centerline_ok', False) else 'NONE'}", None),
            (f"STATIONS: {total}/{config.METROLOGY_STATIONS}", None),
            (f"VALID WIDTHS: {valid}/{total if total else config.METROLOGY_STATIONS}", None),
            (f"LOCAL WIDTH: {local}", None),
            (f"STATUS: {status}", colour),
            ("OVERLAY: LIVE", None),
        ])

        calibrated = bool(getattr(result, "calibrated", False))
        right_x = max(300, width - 262)
        _block(out, right_x, 50, "AI VISION ANALYSIS", [
            ("CALIBRATED SCALE" if calibrated else "NOT CALIBRATED",
             _OK if calibrated else _CRIT),
            (f"REF: {config.METROLOGY_CAMERA_HEIGHT_MIN_CM:.0f}-"
             f"{config.METROLOGY_CAMERA_HEIGHT_MAX_CM:.0f}cm FIXED HT", None),
            (f"TARGET: {config.METROLOGY_TARGET_WIDTH_MM:.1f} mm", None),
            (f"ENG: {config.metrology_range_min():.1f}-"
             f"{config.metrology_range_max():.1f} mm", None),
            (f"SCALE: {getattr(result, 'pixels_per_mm', 0.0):.3f} px/mm", None),
        ])

        # Given its own backing block: drawn as bare text over the bead it was
        # unreadable against bright concrete.
        _block(out, 14, max(210, height // 2 - 20), "MEASUREMENT ENGINE", [
            (f"{config.METROLOGY_STATIONS} PERPENDICULAR STATIONS", None),
        ], width=250)

        log_lines = []
        for station in (getattr(result, "stations", []) or [])[: config.METROLOGY_STATIONS]:
            if station.width_mm is not None:
                log_lines.append((f"{station.label} = {station.width_mm:.1f} mm", None))
            else:
                log_lines.append((f"{station.label} = --", _METRO_INVALID))
        if log_lines:
            _block(out, right_x, max(190, height // 2 - 90),
                   "SEGMENTED WIDTH LOG", log_lines, width=200)

        avg = getattr(result, "average_width_mm", None)
        dev = getattr(result, "deviation_mm", None)
        avg_text = f"{avg:.1f} mm" if avg is not None else "--"
        dev_text = f"{dev:+.1f} mm" if dev is not None else "--"
        _block(out, 14, height - 116, "RESULT", [
            (f"AVERAGE WIDTH: {avg_text}", colour),
            (f"DEVIATION: {dev_text}", None),
            (f"TARGET: {config.METROLOGY_TARGET_WIDTH_MM:.1f} mm   "
             f"TOL: +/-{config.METROLOGY_TOLERANCE_MM:.1f} mm", None),
            (f"SCALE MODE: MANUAL ({config.METROLOGY_CAMERA_HEIGHT_MIN_CM:.0f}-"
             f"{config.METROLOGY_CAMERA_HEIGHT_MAX_CM:.0f}cm)", None),
        ], width=330)

        reason = getattr(result, "reason", "")
        if reason:
            cv2.putText(out, reason[:88], (14, height - 10), _FONT, 0.38, _DIM, 1)
        return out
    except Exception:   # drawing must never break the camera stream
        return frame
