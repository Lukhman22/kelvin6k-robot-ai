"""
Regression test suite for Phase F: Maintenance Knowledge & Dataset RAG Alignment.

Validates:
1. Exact authoritative maintenance intervals for concrete printing:
   - Accelerator hole: every 5 layers, big Allen key, no wire, under 5 seconds stop.
   - Printing pipe: every 4 hours of operation, operating manual.
   - Pump seals: weekly inspection and replacement.
   - Hopper: clean after each wall (1 wall).
   - Front of pump: clean after every 2 walls.
   - Accelerator coupling: inspect daily.
   - Pump during pauses: run every 5 minutes.
   - Alignment and restart after cleaning: 1 hour maximum window.
   - Printed walls: cure 3 times per day.
2. English and Hinglish query support.
3. Explicit uncertainty reporting when information is missing from documentation.
4. Conflicting parameter reporting (must not pick arbitrary values).
5. Zero regression on attendance, shift tracking, and live machine routing.
"""

import unittest

from agent import query_router as qr
from agent.rag.retriever import search_documentation


class TestMaintenanceRAGAlignment(unittest.TestCase):
    """Authoritative maintenance rules and RAG retrieval alignment."""

    def test_accelerator_hole_cleaning_intervals(self):
        """Accelerator hole must be cleaned every 5 layers with big Allen key, never wire."""
        english_res = qr.route("How often do I clean the accelerator hole?")
        self.assertEqual(english_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary = qr.summarise(english_res).lower()
        self.assertIn("5 layers", summary)
        self.assertIn("allen key", summary)
        self.assertIn("wire", summary)  # safety note: do NOT use wire

        hinglish_res1 = qr.route("Accelerator hole kitni layers ke baad clean karna hai?")
        self.assertEqual(hinglish_res1.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary1 = qr.summarise(hinglish_res1).lower()
        self.assertIn("5 layers", summary1)
        self.assertIn("allen key", summary1)

        hinglish_res2 = qr.route("Kitni layers ke baad accelerator hole clean karna hai?")
        self.assertEqual(hinglish_res2.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary2 = qr.summarise(hinglish_res2).lower()
        self.assertIn("5 layers", summary2)

    def test_printing_pipe_cleaning_intervals(self):
        """Printing pipe must be cleaned every 4 hours of operation."""
        english_res = qr.route("How often do I clean the printing pipe?")
        self.assertEqual(english_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary = qr.summarise(english_res).lower()
        self.assertTrue("4 hours" in summary or "four hours" in summary)

        hinglish_res = qr.route("Pipe kitne ghante mein clean karna hai?")
        self.assertEqual(hinglish_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_h = qr.summarise(hinglish_res).lower()
        self.assertTrue("4 hours" in summary_h or "four hours" in summary_h)

        hours_op_res = qr.route("How many hours of operation before cleaning the pipe?")
        self.assertEqual(hours_op_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_op = qr.summarise(hours_op_res).lower()
        self.assertTrue("4 hours" in summary_op or "four hours" in summary_op)

    def test_pump_seal_inspection_intervals(self):
        """Pump seals must be inspected weekly."""
        english_res = qr.route("How often are pump seals checked?")
        self.assertEqual(english_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary = qr.summarise(english_res).lower()
        self.assertIn("weekly", summary)

        hinglish_res = qr.route("Pump seal kab check karna hai?")
        self.assertEqual(hinglish_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_h = qr.summarise(hinglish_res).lower()
        self.assertIn("weekly", summary_h)

    def test_hopper_and_pump_front_cleaning(self):
        """Hopper after each wall (1 wall); front of pump after every 2 walls."""
        hopper_res = qr.route("When do I clean the hopper?")
        self.assertEqual(hopper_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_hopper = qr.summarise(hopper_res).lower()
        self.assertTrue("each wall" in summary_hopper or "1 wall" in summary_hopper)

        pump_front_res = qr.route("When do I clean the front of the pump?")
        self.assertEqual(pump_front_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_pf = qr.summarise(pump_front_res).lower()
        self.assertTrue("2 walls" in summary_pf or "two walls" in summary_pf)

        hinglish_two_walls = qr.route("Har do wall ke baad kya clean karna hai?")
        self.assertEqual(hinglish_two_walls.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_hw = qr.summarise(hinglish_two_walls).lower()
        self.assertTrue("2 walls" in summary_hw or "two walls" in summary_hw)

    def test_accelerator_coupling_inspection(self):
        """Accelerator coupling must be inspected daily."""
        english_res = qr.route("Has the accelerator coupling been checked today?")
        self.assertEqual(english_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary = qr.summarise(english_res).lower()
        self.assertIn("daily", summary)

        hinglish_res = qr.route("Coupling roz check karni hai?")
        self.assertEqual(hinglish_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_h = qr.summarise(hinglish_res).lower()
        self.assertIn("daily", summary_h)

    def test_pump_pause_and_restart_intervals(self):
        """Run pump every 5 minutes during pauses; alignment and restart within 1 hour max."""
        pause_res = qr.route("How often should the pump run during a pause?")
        self.assertEqual(pause_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_pause = qr.summarise(pause_res).lower()
        self.assertTrue("5 minutes" in summary_pause or "five minutes" in summary_pause)

        restart_res = qr.route("What is the maximum restart time?")
        self.assertEqual(restart_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_restart = qr.summarise(restart_res).lower()
        self.assertTrue("1 hour" in summary_restart or "one hour" in summary_restart)

    def test_curing_schedule(self):
        """Printed walls must be cured 3 times per day."""
        english_res = qr.route("How often should walls be cured?")
        self.assertEqual(english_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary = qr.summarise(english_res).lower()
        self.assertTrue("3 times" in summary or "three times" in summary)

        hinglish_res = qr.route("Curing din mein kitni baar karni hai?")
        self.assertEqual(hinglish_res.intent, qr.Intent.KNOWLEDGE_QUERY)
        summary_h = qr.summarise(hinglish_res).lower()
        self.assertTrue("3 times" in summary_h or "three times" in summary_h)

    def test_conflicting_parameters_reported_as_conflict(self):
        """Assistant must report conflicts explicitly and not assume a standard."""
        thick_res = qr.route("What is the standard layer thickness?")
        summary_thick = qr.summarise(thick_res).lower()
        self.assertIn("conflicting", summary_thick)

        speed_res = qr.route("What feedrate should I use?")
        summary_speed = qr.summarise(speed_res).lower()
        self.assertIn("conflicting", summary_speed)

    def test_unanswerable_maintenance_query_uncertainty(self):
        """Undocumented queries must return explicit not-found responses."""
        ballscrew_res = qr.route("What is the lubrication interval for the z axis ballscrew?")
        summary_b = qr.summarise(ballscrew_res).lower()
        self.assertTrue(
            "not found in the available documentation" in summary_b
            or not ballscrew_res.available
        )

        oil_res = qr.route("When should we change the hydraulic oil?")
        self.assertFalse(oil_res.available)

        grease_res = qr.route("How often should we replace the gearbox grease?")
        summary_g = qr.summarise(grease_res).lower()
        self.assertTrue(
            "not found in the available documentation" in summary_g
            or not grease_res.available
        )

    def test_attendance_and_shift_intents_unaffected(self):
        """Attendance and shift tracking queries must retain their exact intents."""
        regression_cases = {
            "hours worked today": qr.Intent.GET_ATTENDANCE_HOURS,
            "who is currently working": qr.Intent.GET_ATTENDANCE_STATUS,
            "overtime today": qr.Intent.GET_ATTENDANCE_OVERTIME,
            "start morning shift": qr.Intent.SHIFT_START,
            "end shift": qr.Intent.SHIFT_END,
            "who worked on line 1": qr.Intent.GET_ACTIVE_SHIFT,
        }
        for query, expected_intent in regression_cases.items():
            res = qr.route(query)
            self.assertEqual(res.intent, expected_intent, f"Regression on {query}")

    def test_live_measurements_unaffected(self):
        """Live machine telemetry queries must retain their exact intents."""
        regression_cases = {
            "current layer width kitni hai?": qr.Intent.GET_LAYER_WIDTH,
            "what is the machine status": qr.Intent.GET_MACHINE_STATUS,
            "kitni wall print ho gayi hai?": qr.Intent.GET_COMPLETED_PRODUCTION,
            "current layer kya hai?": qr.Intent.GET_CURRENT_LAYER,
            "kitna material use hua hai": qr.Intent.GET_MATERIAL_USAGE,
        }
        for query, expected_intent in regression_cases.items():
            res = qr.route(query)
            self.assertEqual(res.intent, expected_intent, f"Regression on {query}")


if __name__ == "__main__":
    unittest.main()
