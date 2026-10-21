"""
Tests for the perpendicular layer-width metrology.

The geometry is verified against SYNTHETIC beads of a known width, because that
is the only way to check a measurement when there is no calibrated camera and no
physical reference: if the engine cannot measure a bead it drew itself, it
cannot measure concrete.

The headline test is `test_width_is_perpendicular_not_horizontal` -- a bead at
45 degrees whose true width is 45 mm reads ~45 mm here and ~63 mm under the
horizontal-run method the existing layer detector uses.
"""

import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config  # noqa: E402
from agent import metrology as M  # noqa: E402
from agent.metrology_voice import AnnouncementGate, compose  # noqa: E402

# 4 px per mm: a 180 px bead is exactly the 45.0 mm target.
PPM = 4.0
# Synthetic edges are perfect, so the engine should land within a few tenths.
# The tolerance the machine is held to is +/-2.0 mm, so this is well inside it.
TOLERANCE_MM = 0.75


def straight_bead(width_px: int, length: int = 540) -> np.ndarray:
    img = np.full((400, 700, 3), 40, np.uint8)
    cv2.rectangle(
        img, (80, 200 - width_px // 2), (80 + length, 200 + width_px // 2),
        (200, 200, 200), -1,
    )
    return img


def angled_bead(width_px: int, angle_deg: float) -> np.ndarray:
    img = np.full((500, 700, 3), 40, np.uint8)
    rad = math.radians(angle_deg)
    half = 250
    cx, cy = 350, 250
    p1 = (int(cx - half * math.cos(rad)), int(cy - half * math.sin(rad)))
    p2 = (int(cx + half * math.cos(rad)), int(cy + half * math.sin(rad)))
    cv2.line(img, p1, p2, (200, 200, 200), width_px)
    return img


def curved_bead(width_px: int) -> np.ndarray:
    img = np.full((500, 700, 3), 40, np.uint8)
    points = [
        (int(60 + i * 5.8), int(250 + 120 * math.sin(i / 100 * math.pi)))
        for i in range(101)
    ]
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (200, 200, 200), width_px)
    return img


def result_for(image: np.ndarray, ppm: float = PPM) -> M.MetrologyResult:
    return M.measure_layer(image, pixels_per_mm=ppm)


class TestGeometry(unittest.TestCase):
    """The measurement itself."""

    def test_straight_bead_measures_its_known_width(self):
        result = result_for(straight_bead(180))
        self.assertIsNotNone(result.average_width_mm)
        self.assertAlmostEqual(result.average_width_mm, 45.0, delta=TOLERANCE_MM)
        self.assertEqual(result.valid_count, config.METROLOGY_STATIONS)

    def test_width_is_perpendicular_not_horizontal(self):
        """
        The core requirement. A bead of constant width must measure the same at
        every angle; a horizontal run would inflate it by 1/cos(angle).
        """
        for angle in (0, 15, 30, 45, 60):
            with self.subTest(angle=angle):
                result = result_for(angled_bead(180, angle))
                self.assertIsNotNone(
                    result.average_width_mm, f"no measurement at {angle} degrees"
                )
                self.assertAlmostEqual(
                    result.average_width_mm, 45.0, delta=TOLERANCE_MM,
                    msg=f"{angle} degrees measured {result.average_width_mm} mm",
                )

    def test_curved_bead_follows_the_shape(self):
        result = result_for(curved_bead(180))
        self.assertTrue(result.centerline_ok)
        self.assertAlmostEqual(result.average_width_mm, 45.0, delta=TOLERANCE_MM)
        # A straight centreline would not deviate; a followed one does.
        ys = [float(p[1]) for p in result.centerline]
        self.assertGreater(max(ys) - min(ys), 50, "centreline did not follow the curve")

    def test_stations_are_placed_along_the_bead(self):
        result = result_for(straight_bead(180))
        self.assertEqual(result.station_count, config.METROLOGY_STATIONS)
        labels = [s.label for s in result.stations]
        self.assertEqual(labels[0], "W1")
        self.assertEqual(labels[-1], f"W{config.METROLOGY_STATIONS}")
        xs = [s.x for s in result.stations]
        self.assertEqual(xs, sorted(xs), "stations must run along the bead in order")

    def test_normals_are_perpendicular_to_the_measurement_line(self):
        """Each measurement line must be parallel to that station's normal."""
        result = result_for(angled_bead(180, 30))
        for station in result.stations:
            if not station.valid:
                continue
            with self.subTest(station=station.label):
                dx = station.right[0] - station.left[0]
                dy = station.right[1] - station.left[1]
                length = math.hypot(dx, dy)
                dot = abs((dx / length) * station.normal[0] + (dy / length) * station.normal[1])
                self.assertAlmostEqual(dot, 1.0, delta=0.02)


class TestValidation(unittest.TestCase):
    """Status, validity, and the refusal to invent a number."""

    def test_under_width_is_reported(self):
        result = result_for(straight_bead(145))       # 36.25 mm
        print("REASON:", getattr(result, "reason", "none")); self.assertEqual(result.status, M.UNDER_WIDTH)
        self.assertLess(result.average_width_mm, config.metrology_range_min())

    def test_over_width_is_reported(self):
        result = result_for(straight_bead(215))       # 53.75 mm
        self.assertEqual(result.status, M.OVER_WIDTH)
        self.assertGreater(result.average_width_mm, config.metrology_range_max())

    def test_in_tolerance_passes(self):
        for width_px in (174, 180, 186):              # 43.5, 45.0, 46.5 mm
            with self.subTest(width_px=width_px):
                self.assertEqual(result_for(straight_bead(width_px)).status, M.PASS)

    def test_deviation_is_measured_against_the_target(self):
        result = result_for(straight_bead(180))
        self.assertAlmostEqual(
            result.deviation_mm,
            result.average_width_mm - config.METROLOGY_TARGET_WIDTH_MM,
            places=2,
        )

    def test_without_calibration_no_millimetre_is_invented(self):
        """
        The project ships with PIXELS_PER_MM unset. In that state the geometry
        still works, but a pixel count must never be presented as millimetres.
        """
        result = result_for(straight_bead(180), ppm=0.0)
        self.assertFalse(result.calibrated)
        self.assertIsNone(result.average_width_mm)
        self.assertEqual(result.status, M.INVALID)
        self.assertEqual(result.valid_count, 0)
        for station in result.stations:
            self.assertIsNone(station.width_mm)
            self.assertIsNotNone(station.width_px, "pixel geometry should still exist")

    def test_blank_frame_is_refused(self):
        blank = np.full((400, 700, 3), 40, np.uint8)
        result = result_for(blank)
        self.assertEqual(result.status, M.INVALID)
        self.assertEqual(result.mask_method, M.MASK_NONE)
        self.assertIsNone(result.average_width_mm)
        self.assertTrue(result.reason)

    def test_too_few_valid_stations_is_invalid_not_a_guess(self):
        stations = [
            M.Station(index=i + 1, x=0.0, y=0.0, normal=(0.0, 1.0), valid=i < 2)
            for i in range(config.METROLOGY_STATIONS)
        ]
        valid = sum(1 for s in stations if s.valid)
        status, reason = M.classify(45.0, valid)
        self.assertEqual(status, M.INVALID)
        self.assertIn("stations", reason)

    def test_mask_method_is_reported_honestly(self):
        """
        There is no concrete-trained network in this project, so the overlay
        must not claim a YOLO detection.
        """
        result = result_for(straight_bead(180))
        self.assertEqual(result.mask_method, M.MASK_CV)
        self.assertNotIn("YOLO", result.mask_method.upper())

    def test_measure_layer_never_raises(self):
        for bad in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((5, 5, 3), np.uint8)):
            with self.subTest(case=type(bad).__name__):
                result = M.measure_layer(bad, pixels_per_mm=PPM)
                self.assertEqual(result.status, M.INVALID)


class TestAnnouncements(unittest.TestCase):
    """Wording, and not speaking on every frame."""

    def _result(self, status, average, timestamp=0.0):
        result = M.MetrologyResult(timestamp=timestamp)
        result.status = status
        result.average_width_mm = average
        return result

    def test_wording_matches_the_status(self):
        self.assertIn("correct", compose(self._result(M.PASS, 45.7)).lower())
        under = compose(self._result(M.UNDER_WIDTH, 36.1))
        self.assertIn("below the required range", under)
        self.assertIn("36.1 millimeters", under)
        self.assertIn("43 to 47 millimeters", under)
        over = compose(self._result(M.OVER_WIDTH, 52.0))
        self.assertIn("above the required range", over)
        self.assertIn("52.0 millimeters", over)
        self.assertIn(
            "unable to reliably measure", compose(self._result(M.INVALID, None)).lower()
        )

    def test_no_width_is_spoken_when_none_was_measured(self):
        """An unmeasurable frame must not produce a number of any kind."""
        text = compose(self._result(M.INVALID, None))
        self.assertNotRegex(text, r"\d", f"invalid announcement contained a figure: {text!r}")
        self.assertNotIn("millimeters", text)

    def test_does_not_speak_on_every_frame(self):
        gate = AnnouncementGate()
        now = 1000.0
        spoken = []
        for _ in range(300):                       # 30 s at 10 fps, unchanging
            now += 0.1
            said = gate.consider(self._result(M.UNDER_WIDTH, 36.1, now), now=now)
            if said:
                spoken.append(said)
        self.assertGreaterEqual(len(spoken), 1)
        self.assertLessEqual(
            len(spoken), 4,
            f"300 frames produced {len(spoken)} announcements; the cooldown is not holding",
        )

    def test_every_status_transition_is_announced(self):
        """
        Including the recovery. An earlier version let the cooldown swallow
        transitions, which lost the UNDER-WIDTH -> PASS announcement entirely.
        """
        gate = AnnouncementGate()
        now = 1000.0
        spoken = []

        def feed(status, average, frames):
            nonlocal now
            for _ in range(frames):
                now += 0.1
                said = gate.consider(self._result(status, average, now), now=now)
                if said:
                    spoken.append(said.status)

        feed(M.PASS, 45.7, 50)
        feed(M.UNDER_WIDTH, 36.1, 80)
        feed(M.OVER_WIDTH, 52.0, 60)
        feed(M.PASS, 45.2, 50)

        self.assertEqual(spoken, [M.PASS, M.UNDER_WIDTH, M.OVER_WIDTH, M.PASS])

    def test_flapping_status_is_not_announced(self):
        """A measurement bouncing across the tolerance edge must stay silent
        until it settles."""
        gate = AnnouncementGate()
        now = 0.0
        spoken = 0
        for i in range(40):
            status = M.PASS if i % 2 == 0 else M.UNDER_WIDTH
            for _ in range(5):                     # 0.5 s per status
                now += 0.1
                if gate.consider(self._result(status, 45.0, now), now=now):
                    spoken += 1
        self.assertEqual(spoken, 0)

    def test_announcement_matches_the_measurement_exactly(self):
        """Voice and overlay must never disagree."""
        gate = AnnouncementGate()
        now = 2000.0
        said = None
        for _ in range(60):
            now += 0.1
            result = self._result(M.UNDER_WIDTH, 36.1, now)
            said = gate.consider(result, now=now) or said
        self.assertIsNotNone(said)
        self.assertEqual(said.average_width_mm, 36.1)
        self.assertIn("36.1", said.text)
        self.assertEqual(said.status, M.UNDER_WIDTH)

    def test_ids_increase_so_the_ui_can_tell_repeats_apart(self):
        gate = AnnouncementGate()
        now = 0.0
        ids = []
        for status in (M.PASS, M.UNDER_WIDTH, M.PASS):
            for _ in range(60):
                now += 0.1
                said = gate.consider(self._result(status, 45.0, now), now=now)
                if said:
                    ids.append(said.id)
        self.assertEqual(ids, sorted(set(ids)))
        self.assertGreaterEqual(len(ids), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
