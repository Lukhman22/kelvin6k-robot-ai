"""
tests/test_historical_operator_date_queries.py

Regression test suite for combined operator_id + shift_date historical voice queries.
Verifies:
1. Aman + date production (filtering by operator_id + shift_date)
2. Aman + date issues (filtering by operator_id + shift_date)
3. Dynamic matching for ANY registered operator name (zero hardcoded names)
4. Existing no-date queries remain completely unchanged in behavior
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config, db, shift_store
from agent import query_router as qr


class HistoricalOperatorDateQueryTests(unittest.TestCase):
    """Test suite for operator + date queries with isolated temporary database."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db = config.DATABASE_PATH
        config.DATABASE_PATH = os.path.join(self._tmp.name, "test_hist.db")
        db.init_db()

        # Register operator Aman Verma
        shift_store.create_operator(
            operator_id="OP_AMAN",
            name="Aman Verma",
            employee_code="EMP_AMAN_01",
            role="operator",
            active=1,
        )

        # Aman Shift 1: 2026-09-24
        s1 = shift_store.start_shift("OP_AMAN", shift_date="2026-09-24", shift_type="Morning", line_id="line_1")
        shift_store.record_production(s1, wall_number=1, wall_height=3000.0, segment_layers=30, layers_completed=30, status="completed")
        shift_store.record_production(s1, wall_number=2, wall_height=3000.0, segment_layers=30, layers_completed=30, status="completed")
        shift_store.log_issue(
            s1,
            issue_type="nozzle_clog",
            description="Nozzle clogged due to dry mix",
            cause="Dry mix",
            action_taken="Cleaned nozzle with wire",
            result="resolved",
        )
        shift_store.end_shift(s1, status="closed")

        # Aman Shift 2: 2026-09-25
        s2 = shift_store.start_shift("OP_AMAN", shift_date="2026-09-25", shift_type="Morning", line_id="line_1")
        shift_store.record_production(s2, wall_number=3, wall_height=3000.0, segment_layers=30, layers_completed=30, status="completed")
        shift_store.log_issue(
            s2,
            issue_type="hose_kink",
            description="Material hose kinked",
            cause="Sharp bend",
            action_taken="Straightened hose",
            result="resolved",
        )
        shift_store.end_shift(s2, status="closed")

        # Aman Shift 3: 2026-09-21 (production only, no issues)
        s3 = shift_store.start_shift("OP_AMAN", shift_date="2026-09-21", shift_type="Night", line_id="line_1")
        shift_store.record_production(s3, wall_number=4, wall_height=3000.0, segment_layers=30, layers_completed=30, status="completed")
        shift_store.end_shift(s3, status="closed")

        # Register dynamic operator Pooja Sharma
        shift_store.create_operator(
            operator_id="OP_POOJA",
            name="Pooja Sharma",
            employee_code="EMP_POOJA_02",
            role="operator",
            active=1,
        )

        # Pooja Shift: 2026-09-24 (same date as Aman Shift 1)
        sp = shift_store.start_shift("OP_POOJA", shift_date="2026-09-24", shift_type="Night", line_id="line_1")
        shift_store.record_production(sp, wall_number=5, wall_height=3000.0, segment_layers=30, layers_completed=30, status="completed")
        shift_store.log_issue(
            sp,
            issue_type="feeder_stall",
            description="Feeder stalled from aggregate",
            cause="Aggregate jam",
            action_taken="Cleared hopper",
            result="resolved",
        )
        shift_store.end_shift(sp, status="closed")

        # Register another dynamic operator Deepak Rao
        shift_store.create_operator(
            operator_id="OP_DEEPAK",
            name="Deepak Rao",
            employee_code="EMP_DEEPAK_03",
            role="operator",
            active=1,
        )
        sd = shift_store.start_shift("OP_DEEPAK", shift_date="2026-09-27", shift_type="Morning", line_id="line_1")
        shift_store.record_production(sd, wall_number=6, wall_height=3000.0, segment_layers=30, layers_completed=30, status="completed")
        shift_store.log_issue(
            sd,
            issue_type="pump_pressure",
            description="Pump pressure low",
            cause="Air lock",
            action_taken="Bled pump line",
            result="resolved",
        )
        shift_store.end_shift(sd, status="closed")

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        self._tmp.cleanup()

    # -------------------------------------------------------------------------
    # 1. Aman + Date Production Tests
    # -------------------------------------------------------------------------

    def test_aman_date_production_september_24(self):
        query = "What did Aman print on September 24, 2026?"
        intent, hits = qr.classify(query)
        self.assertEqual(intent, qr.Intent.GET_SHIFT_BY_OPERATOR)

        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_BY_OPERATOR)
        self.assertTrue(result.available)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertEqual(result.facts["date"], "2026-09-24")
        self.assertEqual(result.facts["wall_count"], 2)
        self.assertEqual(result.facts["walls"], [1, 2])
        # Pooja's Wall 5 must not be included
        self.assertNotIn(5, result.facts["walls"])
        # Aman's other dates must not be included
        self.assertNotIn(3, result.facts["walls"])
        self.assertNotIn(4, result.facts["walls"])

        summary = qr.summarise(result)
        self.assertIn("Aman Verma printed 2 walls on 2026-09-24: Wall 1, Wall 2.", summary)

    def test_aman_date_production_september_25(self):
        query = "What did Aman print on September 25, 2026?"
        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_BY_OPERATOR)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertEqual(result.facts["date"], "2026-09-25")
        self.assertEqual(result.facts["wall_count"], 1)
        self.assertEqual(result.facts["walls"], [3])

        summary = qr.summarise(result)
        self.assertIn("Aman Verma printed 1 walls on 2026-09-25: Wall 3.", summary)

    def test_aman_date_production_september_21(self):
        query = "How many walls did Aman print on September 21, 2026?"
        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_BY_OPERATOR)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertEqual(result.facts["date"], "2026-09-21")
        self.assertEqual(result.facts["wall_count"], 1)
        self.assertEqual(result.facts["walls"], [4])

    def test_aman_date_production_no_records_on_date(self):
        query = "What did Aman print on September 28, 2026?"
        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_BY_OPERATOR)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertEqual(result.facts["date"], "2026-09-28")
        self.assertEqual(result.facts["wall_count"], 0)
        self.assertEqual(result.facts["walls"], [])

        summary = qr.summarise(result)
        self.assertIn("No completed wall records found for Aman Verma on 2026-09-28.", summary)

    # -------------------------------------------------------------------------
    # 2. Aman + Date Issues Tests
    # -------------------------------------------------------------------------

    def test_aman_date_issues_september_24(self):
        query = "What issues did Aman have on September 24, 2026?"
        intent, hits = qr.classify(query)
        self.assertEqual(intent, qr.Intent.GET_SHIFT_ISSUES)

        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_ISSUES)
        self.assertTrue(result.available)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertEqual(result.facts["date"], "2026-09-24")
        self.assertEqual(result.facts["count"], 1)
        issues = result.facts["issues"]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "nozzle_clog")
        self.assertEqual(issues[0]["action_taken"], "Cleaned nozzle with wire")

        summary = qr.summarise(result)
        self.assertIn("Found 1 issues for Aman Verma:", summary)
        self.assertIn("Nozzle clogged due to dry mix", summary)
        self.assertIn("Cleaned nozzle with wire", summary)
        # Must not contain Pooja's feeder issue or Aman's hose issue
        self.assertNotIn("feeder_stall", summary)
        self.assertNotIn("Material hose kinked", summary)

    def test_aman_date_issues_september_25(self):
        query = "What problem did Aman have on September 25, 2026?"
        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_ISSUES)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertEqual(result.facts["date"], "2026-09-25")
        self.assertEqual(result.facts["count"], 1)
        self.assertEqual(result.facts["issues"][0]["type"], "hose_kink")

    def test_aman_date_issues_empty_on_date(self):
        query = "What issues did Aman have on September 21, 2026?"
        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_ISSUES)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertEqual(result.facts["date"], "2026-09-21")
        self.assertEqual(result.facts["count"], 0)
        self.assertEqual(result.facts["issues"], [])

        summary = qr.summarise(result)
        self.assertIn("No issues have been recorded for Aman Verma on 2026-09-21.", summary)

    # -------------------------------------------------------------------------
    # 3. Dynamic Registered Operator Name Tests (No Hardcoded Names)
    # -------------------------------------------------------------------------

    def test_dynamic_operator_pooja_production_with_date(self):
        query = "What did Pooja print on September 24, 2026?"
        intent, _ = qr.classify(query)
        self.assertEqual(intent, qr.Intent.GET_SHIFT_BY_OPERATOR)

        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_BY_OPERATOR)
        self.assertEqual(result.facts["operator_id"], "OP_POOJA")
        self.assertEqual(result.facts["date"], "2026-09-24")
        self.assertEqual(result.facts["wall_count"], 1)
        self.assertEqual(result.facts["walls"], [5])

        summary = qr.summarise(result)
        self.assertIn("Pooja Sharma printed 1 walls on 2026-09-24: Wall 5.", summary)

    def test_dynamic_operator_pooja_issues_with_date(self):
        query = "What problems did Pooja have on September 24, 2026?"
        intent, _ = qr.classify(query)
        self.assertEqual(intent, qr.Intent.GET_SHIFT_ISSUES)

        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_ISSUES)
        self.assertEqual(result.facts["operator_id"], "OP_POOJA")
        self.assertEqual(result.facts["date"], "2026-09-24")
        self.assertEqual(result.facts["count"], 1)
        self.assertEqual(result.facts["issues"][0]["type"], "feeder_stall")

        summary = qr.summarise(result)
        self.assertIn("Found 1 issues for Pooja Sharma:", summary)
        self.assertIn("Feeder stalled from aggregate", summary)

    def test_dynamic_operator_deepak_production_and_issues(self):
        # Production
        q_prod = "What did Deepak print on September 27, 2026?"
        r_prod = qr.route(q_prod)
        self.assertEqual(r_prod.intent, qr.Intent.GET_SHIFT_BY_OPERATOR)
        self.assertEqual(r_prod.facts["operator_id"], "OP_DEEPAK")
        self.assertEqual(r_prod.facts["date"], "2026-09-27")
        self.assertEqual(r_prod.facts["walls"], [6])

        # Issues
        q_iss = "What issues did Deepak have on September 27, 2026?"
        r_iss = qr.route(q_iss)
        self.assertEqual(r_iss.intent, qr.Intent.GET_SHIFT_ISSUES)
        self.assertEqual(r_iss.facts["operator_id"], "OP_DEEPAK")
        self.assertEqual(r_iss.facts["date"], "2026-09-27")
        self.assertEqual(r_iss.facts["issues"][0]["type"], "pump_pressure")

    # -------------------------------------------------------------------------
    # 4. Existing No-Date Queries Remain Unchanged
    # -------------------------------------------------------------------------

    def test_existing_no_date_aman_production(self):
        query = "What did Aman print?"
        intent, _ = qr.classify(query)
        self.assertEqual(intent, qr.Intent.GET_SHIFT_BY_OPERATOR)

        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_BY_OPERATOR)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertIsNone(result.facts["date"])
        # Should return all walls completed by Aman across all dates (1, 2, 3, 4)
        self.assertEqual(result.facts["wall_count"], 4)
        self.assertEqual(result.facts["walls"], [1, 2, 3, 4])

        summary = qr.summarise(result)
        self.assertIn("Aman Verma printed 4 walls: Wall 1, Wall 2, Wall 3, Wall 4.", summary)
        self.assertNotIn(" on ", summary)

    def test_existing_no_date_aman_issues(self):
        query = "What problems did Aman have?"
        intent, _ = qr.classify(query)
        self.assertEqual(intent, qr.Intent.GET_SHIFT_ISSUES)

        result = qr.route(query)
        self.assertEqual(result.intent, qr.Intent.GET_SHIFT_ISSUES)
        self.assertEqual(result.facts["operator_id"], "OP_AMAN")
        self.assertIsNone(result.facts["date"])
        self.assertEqual(result.facts["count"], 2)

    def test_existing_no_date_other_operators_routing(self):
        # Register Rahul and Anand to verify existing patterns and dynamic lookup
        shift_store.create_operator("OP_RAHUL", "Rahul Sharma", "EMP_R", role="operator", active=1)
        shift_store.create_operator("OP_ANAND", "Anand Sharma", "EMP_A", role="operator", active=1)

        # 1. Rahul queries
        intent, _ = qr.classify("What did Rahul print?")
        self.assertEqual(intent, qr.Intent.GET_SHIFT_BY_OPERATOR)

        intent, _ = qr.classify("How many walls did Rahul print?")
        self.assertEqual(intent, qr.Intent.GET_SHIFT_BY_OPERATOR)

        intent, _ = qr.classify("What problem did Rahul have?")
        self.assertEqual(intent, qr.Intent.GET_SHIFT_ISSUES)

        # 2. Anand queries
        intent, _ = qr.classify("What did Anand print?")
        self.assertEqual(intent, qr.Intent.GET_SHIFT_BY_OPERATOR)

        intent, _ = qr.classify("What problem did Anand have?")
        self.assertEqual(intent, qr.Intent.GET_SHIFT_ISSUES)


if __name__ == "__main__":
    unittest.main()
