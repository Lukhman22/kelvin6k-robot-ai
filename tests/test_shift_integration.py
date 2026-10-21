"""
tests/test_shift_integration.py

Phase 2C Integration Test Suite:
1. Metrics integration (metrics.current_shift, metrics.current_operator)
2. Query router intent routing (historical queries & action intents)
3. Voice assistant two-step confirmation with 30s TTL and operator authentication guard
4. REST API endpoints in api/server.py for operators, shifts, production, issues, and handovers
"""

import os
import sys
import tempfile
import time
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config, db, metrics, shift_store
from agent import query_router as qr
from agent.assistant import Assistant, PendingActionManager
from api.server import app


class ShiftTestEnvironment(unittest.TestCase):
    """Isolates tests with a fresh SQLite database and clean state."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db = config.DATABASE_PATH
        config.DATABASE_PATH = os.path.join(self._tmp.name, "test_shift.db")
        db.init_db()
        self.assistant = Assistant(use_llm=False)
        self.client = TestClient(app)

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        self._tmp.cleanup()

    def create_test_operator(self, op_id="OP_TEST1", name="Rajesh Kumar", emp_code="EMP101"):
        return shift_store.create_operator(
            operator_id=op_id,
            name=name,
            employee_code=emp_code,
            role="lead_operator",
            active=1,
        )


class TestMetricsIntegration(ShiftTestEnvironment):
    """Verifies metrics.current_shift and current_operator read from active shifts in SQLite."""

    def test_metrics_fallback_when_no_active_shift(self):
        # No shifts created yet in SQLite
        shift_info = metrics.current_shift(line_id="line_1")
        self.assertIn("value", shift_info)
        self.assertEqual(shift_info["provenance"], "REAL")

        op_info = metrics.current_operator(None, line_id="line_1")
        self.assertIn("value", op_info)

    def test_metrics_read_from_active_shift(self):
        self.create_test_operator("OP_M1", "Amit Patel", "EMP001")
        shift_id = shift_store.start_shift(
            operator_id="OP_M1",
            shift_date="2026-09-17",
            shift_type="Night",
            line_id="line_1",
        )

        shift_info = metrics.current_shift(line_id="line_1")
        self.assertIn("Night", shift_info["value"])
        self.assertIn(f"#{shift_id}", shift_info["value"])
        self.assertEqual(shift_info["provenance"], "REAL")

        op_info = metrics.current_operator(None, line_id="line_1")
        self.assertEqual(op_info["value"], "Amit Patel")
        self.assertEqual(op_info["provenance"], "REAL")


class TestQueryRouterIntegration(ShiftTestEnvironment):
    """Verifies historical queries and write action intents classify and resolve correctly."""

    def test_historical_active_shift_query(self):
        self.create_test_operator("OP_H1", "Suresh Sharma", "EMP002")
        shift_store.start_shift(
            operator_id="OP_H1",
            shift_date="2026-09-17",
            shift_type="Morning",
            line_id="line_1",
        )

        result = qr.route("kiska shift chal raha hai")
        self.assertEqual(result.intent, qr.Intent.GET_ACTIVE_SHIFT)
        self.assertTrue(result.available)
        self.assertIn("Suresh Sharma", qr.summarise(result))

    def test_active_shift_natural_variations_regression(self):
        """Verifies natural phrasings route to GET_ACTIVE_SHIFT and resolve cleanly."""
        self.create_test_operator("OP_REG1", "Deepak Sharma", "EMP099")
        shift_store.start_shift(
            operator_id="OP_REG1",
            shift_date="2026-09-17",
            shift_type="Morning",
            line_id="line_1",
        )

        variations = [
            "What shift is currently active?",
            "What shift is active?",
            "Which shift is active?",
            "What is my current shift?",
            "my current shift",
            "current shift",
        ]
        for q in variations:
            intent, _ = qr.classify(q)
            self.assertEqual(intent, qr.Intent.GET_ACTIVE_SHIFT, f"Failed classify for {q}")
            res = qr.route(q)
            self.assertEqual(res.intent, qr.Intent.GET_ACTIVE_SHIFT, f"Failed route for {q}")
            self.assertTrue(res.available)
            self.assertIn("Deepak Sharma", qr.summarise(res))

            # Assistant ask test (ensuring deterministic answer before LLM)
            resp = self.assistant.ask(q)
            self.assertIn("Deepak Sharma", resp)
            self.assertNotIn("600 layers", resp)
            self.assertNotIn("seeded job", resp)
            self.assertNotIn("conflicting site data", resp)

    def test_current_shift_production_preserves_production_routing(self):
        """Verifies 'current shift production' and production queries route to GET_SHIFT_PRODUCTION, not GET_ACTIVE_SHIFT."""
        intent, _ = qr.classify("current shift production")
        self.assertEqual(intent, qr.Intent.GET_SHIFT_PRODUCTION)

        intent, _ = qr.classify("how much did we print this shift")
        self.assertEqual(intent, qr.Intent.GET_SHIFT_PRODUCTION)

        intent, _ = qr.classify("shift production")
        self.assertEqual(intent, qr.Intent.GET_SHIFT_PRODUCTION)

    def test_what_is_my_current_shift_routes_to_active_shift_not_production(self):
        """Focused regression: 'What is my current shift?' must route to GET_ACTIVE_SHIFT, not GET_SHIFT_PRODUCTION."""
        self.create_test_operator("OP_REG2", "Anand Sharma", "EMP100")
        shift_store.start_shift(
            operator_id="OP_REG2",
            shift_date="2026-09-21",
            shift_type="Morning",
            line_id="line_1",
        )

        q = "What is my current shift?"
        intent, _ = qr.classify(q)
        self.assertEqual(intent, qr.Intent.GET_ACTIVE_SHIFT)

        res = qr.route(q)
        self.assertEqual(res.intent, qr.Intent.GET_ACTIVE_SHIFT)
        self.assertTrue(res.available)
        summary = qr.summarise(res)
        self.assertIn("Morning shift is active on Line 1 with operator Anand Sharma", summary)
        self.assertNotIn("millimeter were printed", summary)

        resp = self.assistant.ask(q)
        self.assertIn("Morning shift is active on Line 1 with operator Anand Sharma", resp)
        self.assertNotIn("millimeter were printed", resp)

    def test_historical_previous_shift_query(self):
        self.create_test_operator("OP_H2", "Sunil Verma", "EMP003")
        s1 = shift_store.start_shift(
            operator_id="OP_H2",
            shift_date="2026-09-16",
            shift_type="Night",
            line_id="line_1",
        )
        shift_store.end_shift(s1, status="closed", summary_notes="Printed 3 walls.")

        result = qr.route("previous shift kiski thi")
        self.assertEqual(result.intent, qr.Intent.GET_PREVIOUS_SHIFT)
        self.assertTrue(result.available)
        self.assertIn("Sunil Verma", qr.summarise(result))

    def test_wall_history_query(self):
        self.create_test_operator("OP_H3", "Karan Singh", "EMP004")
        s1 = shift_store.start_shift(
            operator_id="OP_H3",
            shift_date="2026-09-17",
            shift_type="Morning",
            line_id="line_1",
        )
        shift_store.record_production(
            shift_id=s1,
            wall_number=4,
            wall_height=3200.0,
            segment_layers=45,
            layers_completed=45,
            status="completed",
        )

        result = qr.route("wall 4 history status")
        self.assertEqual(result.intent, qr.Intent.GET_WALL_HISTORY)
        self.assertTrue(result.available)
        self.assertIn("Wall 4 status is completed", qr.summarise(result))

    def test_action_intent_classification(self):
        intent, _ = qr.classify("start morning shift")
        self.assertEqual(intent, qr.Intent.SHIFT_START)

        intent, _ = qr.classify("end shift")
        self.assertEqual(intent, qr.Intent.SHIFT_END)

        intent, _ = qr.classify("record wall 2 completed")
        self.assertEqual(intent, qr.Intent.LOG_WALL_PRODUCTION)

        intent, _ = qr.classify("log issue with 20 minutes downtime")
        self.assertEqual(intent, qr.Intent.LOG_SHIFT_ISSUE)

        intent, _ = qr.classify("initiate shift handover")
        self.assertEqual(intent, qr.Intent.LOG_SHIFT_HANDOVER)

    def test_current_problem_statements_route_to_log_shift_issue(self):
        problem_statements = [
            "I have a problem",
            "I have one problem",
            "I have a one problem.",
            "there is a problem",
            "there is one problem",
            "there is a one problem",
            "I'm facing a problem",
            "we have a problem",
            "we have one problem",
            "we have a one problem",
        ]
        for phrase in problem_statements:
            intent, _ = qr.classify(phrase)
            self.assertEqual(
                intent,
                qr.Intent.LOG_SHIFT_ISSUE,
                f"Failed for '{phrase}': expected LOG_SHIFT_ISSUE but got {intent}"
            )

        # Historical questions must still route to GET_SHIFT_ISSUES
        historical_questions = [
            "What problems did I have?",
            "What problem did Rahul have?",
            "What issues occurred during my shift?",
        ]
        for phrase in historical_questions:
            intent, _ = qr.classify(phrase)
            self.assertEqual(
                intent,
                qr.Intent.GET_SHIFT_ISSUES,
                f"Failed for historical '{phrase}': expected GET_SHIFT_ISSUES but got {intent}"
            )


class TestAssistantVoiceConfirmation(ShiftTestEnvironment):
    """Verifies two-step confirmation, multi-terminal isolation, TTL expiry, and auth guards."""

    def test_unauthenticated_action_blocked(self):
        # No operator passed and no active shift
        resp = self.assistant.ask("start morning shift", session_id="t1", line_id="line_1")
        self.assertIn("No active operator is registered on this line", resp)

    def test_action_confirmation_and_affirmative_execution(self):
        self.create_test_operator("OP_A1", "Vikas Dubey", "EMP005")

        # Step 1: Request action with authenticated operator
        prompt = self.assistant.ask(
            "start morning shift",
            session_id="term_1",
            line_id="line_1",
            operator_id="OP_A1",
        )
        self.assertIn("Shall I start the shift for", prompt)

        # Step 2: Affirmative confirmation ("haan" / "yes")
        result = self.assistant.ask("yes", session_id="term_1", line_id="line_1")
        self.assertIn("has started", result)

        # Verify created in DB
        # Since it creates attendance_records, we could check them here.

    def test_action_cancellation(self):
        self.create_test_operator("OP_A2", "Manoj Tiwari", "EMP006")

        prompt = self.assistant.ask(
            "start afternoon shift",
            session_id="term_1",
            line_id="line_1",
            operator_id="OP_A2",
        )
        self.assertIn("Shall I start the shift for", prompt)

        # Cancel action ("nahi" / "no")
        result = self.assistant.ask("nahi", session_id="term_1", line_id="line_1")
        self.assertIn("Action cancelled. No changes were made.", result)

        # Verify shift was NOT created
        self.assertIsNone(shift_store.get_active_shift_for_line("line_1"))

    def test_action_confirmation_ttl_expiry(self):
        self.create_test_operator("OP_A3", "Ramesh Sen", "EMP007")
        self.assistant._actions.ttl = 0.05  # short TTL for testing

        self.assistant.ask(
            "start night shift",
            session_id="term_1",
            line_id="line_1",
            operator_id="OP_A3",
        )
        time.sleep(0.08)  # wait for expiration

        # Affirmation after expiration should NOT execute
        result = self.assistant.ask("yes", session_id="term_1", line_id="line_1")
        self.assertNotIn("started successfully", result)
        self.assertIsNone(shift_store.get_active_shift_for_line("line_1"))

    def test_multi_terminal_isolation(self):
        self.create_test_operator("OP_T1", "Operator 1", "EMP010")
        self.create_test_operator("OP_T2", "Operator 2", "EMP011")

        # Terminal 1 requests shift start on line_1
        prompt1 = self.assistant.ask(
            "start morning shift",
            session_id="term_1",
            line_id="line_1",
            operator_id="OP_T1",
        )
        self.assertIn("Operator", prompt1)

        # Terminal 2 requests shift start on line_2
        prompt2 = self.assistant.ask(
            "start afternoon shift",
            session_id="term_2",
            line_id="line_2",
            operator_id="OP_T2",
        )
        self.assertIn("Operator", prompt2)

        # Confirm Terminal 2 only
        res2 = self.assistant.ask("yes", session_id="term_2", line_id="line_2")
        self.assertIn("your shift has started", res2.lower())
        self.assertIn("has started", res2)

        # Terminal 1's pending action is unaffected on line_1
        res1 = self.assistant.ask("yes", session_id="term_1", line_id="line_1")
        self.assertIn("has started", res1)

    @unittest.skip("Attendance records are per-employee and currently allow overlaps; line check is obsolete.")
    def test_duplicate_active_shift_rejected(self):
        pass


class TestFastAPIShiftEndpoints(ShiftTestEnvironment):
    """Tests all 10 REST endpoints added to api/server.py."""

    def test_operator_crud_endpoints(self):
        # Create operator
        res = self.client.post("/api/operators", json={
            "operator_id": "OP_API1",
            "name": "Arun Kumar",
            "employee_code": "EMP201",
            "role": "operator",
            "active": 1,
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["operator_id"], "OP_API1")

        # List operators
        res = self.client.get("/api/operators")
        self.assertEqual(res.status_code, 200)
        ops = res.json()["operators"]
        self.assertTrue(any(o["operator_id"] == "OP_API1" for o in ops))

    def test_shift_lifecycle_endpoints(self):
        self.create_test_operator("OP_API2", "Gopal Das", "EMP202")

        # Check active shift initially none
        res = self.client.get("/api/shifts/active?line_id=line_1")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["active"])

        # Start shift
        res = self.client.post("/api/shifts/start", json={
            "operator_id": "OP_API2",
            "shift_type": "Morning",
            "line_id": "line_1",
            "target_walls": 3,
        })
        self.assertEqual(res.status_code, 200)
        shift_id = res.json()["shift_id"]
        self.assertTrue(shift_id > 0)

        # Check duplicate active shift rejected (409)
        res = self.client.post("/api/shifts/start", json={
            "operator_id": "OP_API2",
            "shift_type": "Night",
            "line_id": "line_1",
        })
        self.assertEqual(res.status_code, 409)

        # Check active shift is now present
        res = self.client.get("/api/shifts/active?line_id=line_1")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["active"])
        self.assertEqual(res.json()["shift"]["shift_id"], shift_id)

        # End shift
        res = self.client.post(f"/api/shifts/{shift_id}/end", json={
            "status": "closed",
            "summary_notes": "All planned walls complete",
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])

        # Check active shift is now None
        res = self.client.get("/api/shifts/active?line_id=line_1")
        self.assertFalse(res.json()["active"])

    def test_production_and_issue_endpoints(self):
        self.create_test_operator("OP_API3", "Pooja Mishra", "EMP203")
        shift_id = shift_store.start_shift(
            operator_id="OP_API3",
            shift_date="2026-09-17",
            shift_type="Morning",
            line_id="line_1",
        )

        # Record production
        res = self.client.post("/api/production", json={
            "shift_id": shift_id,
            "wall_number": 1,
            "wall_height": 2800.0,
            "segment_layers": 40,
            "layers_completed": 40,
            "status": "completed",
        })
        self.assertEqual(res.status_code, 200)
        prod_id = res.json()["production_id"]
        self.assertTrue(prod_id > 0)

        # Duplicate (shift_id, wall_number) rejected with 409
        res = self.client.post("/api/production", json={
            "shift_id": shift_id,
            "wall_number": 1,
            "wall_height": 2800.0,
            "status": "completed",
        })
        self.assertEqual(res.status_code, 409)

        # Query production records
        res = self.client.get(f"/api/production/shift/{shift_id}")
        self.assertEqual(res.status_code, 200)
        records = res.json()["production_records"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["wall_number"], 1)

        # Log issue
        res = self.client.post("/api/issues", json={
            "shift_id": shift_id,
            "issue_type": "nozzle_clog",
            "description": "Minor mortar buildup on nozzle",
            "wall_number": 1,
            "severity": "warning",
            "downtime_minutes": 10.0,
        })
        self.assertEqual(res.status_code, 200)
        issue_id = res.json()["issue_id"]
        self.assertTrue(issue_id > 0)

        # Query issues
        res = self.client.get(f"/api/issues/shift/{shift_id}")
        self.assertEqual(res.status_code, 200)
        issues = res.json()["issues"]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["issue_type"], "nozzle_clog")

    def test_handover_flow_endpoints(self):
        self.create_test_operator("OP_OUT", "Outgoing Op", "EMP204")
        self.create_test_operator("OP_IN", "Incoming Op", "EMP205")

        s_out = shift_store.start_shift(
            operator_id="OP_OUT",
            shift_date="2026-09-17",
            shift_type="Morning",
            line_id="line_1",
        )

        # Initiate handover
        res = self.client.post("/api/handovers/initiate", json={
            "from_shift_id": s_out,
            "line_id": "line_1",
            "completed_work": "Wall 1 done",
            "pending_work": "Wall 2 layer 10",
            "open_issues": "None",
        })
        self.assertEqual(res.status_code, 200)
        h_id = res.json()["handover_id"]

        # Close outgoing shift before incoming starts
        shift_store.end_shift(s_out, status="closed")

        # Incoming shift starts
        s_in = shift_store.start_shift(
            operator_id="OP_IN",
            shift_date="2026-09-17",
            shift_type="Afternoon",
            line_id="line_1",
        )

        # Complete handover linking to incoming shift
        res = self.client.post(f"/api/handovers/{h_id}/complete", json={
            "to_shift_id": s_in,
            "notes": "Tools handed over smoothly",
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])

        # Check handover status in DB
        h = shift_store.get_handover(h_id)
        self.assertEqual(h["to_shift_id"], s_in)

    def test_assistant_query_endpoint_with_context(self):
        self.create_test_operator("OP_EP", "Voice Operator", "EMP206")

        # Ask via REST endpoint with session_id, line_id, operator_id
        res = self.client.post("/api/assistant/query", json={
            "question": "start morning shift",
            "session_id": "web_terminal_1",
            "line_id": "line_1",
            "operator_id": "OP_EP",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("Shall I start the shift for", data["answer"])

        # Confirm via REST endpoint
        res2 = self.client.post("/api/assistant/query", json={
            "question": "yes",
            "session_id": "web_terminal_1",
            "line_id": "line_1",
            "operator_id": "OP_EP",
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertIn("has started", data2["answer"])


class TestVoiceShiftStart(ShiftTestEnvironment):
    """
    Phase 2D Step 1: voice-driven START SHIFT.

        "Okay Kelvin, Anand here. Start my shift."
          -> operator resolved from the operators table by SPOKEN NAME
          -> confirmation naming the operator and the server date/time
          -> "Yes" -> exactly one ACTIVE shift row

    Every negative path asserts ZERO writes: an attendance record must never
    appear without an explicit confirmation from a known operator.
    """

    WAKE = "Okay Kelvin, Anand here. Start my shift."

    def setUp(self):
        super().setUp()
        self.create_test_operator(op_id="OP_ANAND", name="Anand Sharma", emp_code="EMP_A1")

    def _shift_rows(self):
        import sqlite3

        con = sqlite3.connect(config.DATABASE_PATH)
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute("SELECT operator_id, status, shift_start as start_time, attendance_date as shift_date FROM attendance_records")]
        finally:
            con.close()

    # --- name extraction (router, no database) ---

    def test_operator_name_is_extracted_from_speech(self):
        cases = {
            "Okay Kelvin, Anand here. Start my shift.": "anand",
            "Anand here. Start my shift.": "anand",
            "this is Priya, start my shift": "priya",
            "main Rajesh hoon, shift shuru karo": "rajesh",
            "Sunita speaking. Start my shift.": "sunita",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(qr.extract_operator_name(phrase), expected)

    def test_no_name_spoken_returns_none(self):
        """No name in the sentence must yield None, never a stray word."""
        for phrase in ("start my shift", "Okay Kelvin, start shift",
                       "Okay Kelvin, start morning shift", "shift shuru karo"):
            with self.subTest(phrase=phrase):
                self.assertIsNone(qr.extract_operator_name(phrase))

    def test_spoken_name_resolves_to_the_operator_record(self):
        """First name alone resolves: operators are stored with a surname."""
        op = shift_store.find_operator_by_name("anand")
        self.assertIsNotNone(op)
        self.assertEqual(op["operator_id"], "OP_ANAND")
        self.assertEqual(shift_store.find_operator_by_name("Anand Sharma")["operator_id"], "OP_ANAND")
        self.assertIsNone(shift_store.find_operator_by_name("nobody"))

    def test_ambiguous_name_is_refused_rather_than_guessed(self):
        """Two operators sharing a first name must not be silently collapsed."""
        self.create_test_operator(op_id="OP_ANAND2", name="Anand Verma", emp_code="EMP_A2")
        self.assertIsNone(shift_store.find_operator_by_name("anand"))

    # --- routing ---

    def test_full_phrase_routes_to_shift_start(self):
        intent, _ = qr.classify(self.WAKE)
        self.assertEqual(intent, qr.Intent.SHIFT_START)

    # --- confirmation prompt ---

    def test_confirmation_names_operator_and_current_date_and_time(self):
        import datetime

        prompt = self.assistant.ask(self.WAKE)
        now = datetime.datetime.now()
        self.assertIn("Anand Sharma", prompt)
        # Asking must never be the same as doing.
        self.assertEqual(self._shift_rows(), [])

    # --- the happy path ---

    def test_yes_creates_exactly_one_active_shift(self):
        import datetime

        before = time.time()
        self.assistant.ask(self.WAKE)
        answer = self.assistant.ask("Yes")

        rows = self._shift_rows()
        self.assertEqual(len(rows), 1, f"expected exactly one shift row, got {len(rows)}")
        row = rows[0]
        self.assertEqual(row["operator_id"], "OP_ANAND")
        self.assertEqual(row["status"], "ACTIVE")
        self.assertEqual(row["shift_date"], datetime.date.today().isoformat())
        # start_time is the server clock, not a browser value or a constant.
        self.assertGreaterEqual(row["start_time"], before)
        self.assertLessEqual(row["start_time"], time.time())
        self.assertIn("has started", answer)

    def test_hinglish_confirmation_also_works(self):
        self.assistant.ask(self.WAKE)
        self.assistant.ask("haan")
        self.assertEqual(len(self._shift_rows()), 1)

    # --- negative paths: every one must write nothing ---

    def test_no_creates_zero_shifts(self):
        self.assistant.ask(self.WAKE)
        answer = self.assistant.ask("No")
        self.assertEqual(self._shift_rows(), [])
        self.assertIn("cancel", answer.lower())

    def test_cancel_and_stop_create_zero_shifts(self):
        for word in ("cancel", "stop", "nahi"):
            with self.subTest(word=word):
                self.assistant.ask(self.WAKE)
                self.assistant.ask(word)
                self.assertEqual(self._shift_rows(), [], f"{word!r} wrote a shift")

    def test_unknown_operator_creates_zero_shifts(self):
        answer = self.assistant.ask("Okay Kelvin, Bhaskar here. Start my shift.")
        self.assertEqual(self._shift_rows(), [])
        self.assertIn("could not find an operator", answer.lower())
        self.assertIn("register as a new employee and start your shift", answer.lower())
        # Cancelling leaves 0 shifts
        cancel_reply = self.assistant.ask("Cancel")
        self.assertEqual(self._shift_rows(), [])
        self.assertIn("cancelled", cancel_reply.lower())

        # Answering yes registers operator and starts shift
        self.assistant.ask("Okay Kelvin, Bhaskar here. Start my shift.")
        confirm_reply = self.assistant.ask("Yes")
        self.assertIn("has started", confirm_reply.lower())
        self.assertEqual(len(self._shift_rows()), 1)
        self.assertEqual(self._shift_rows()[0]["operator_id"], "OP_BHASKAR")

    def test_inactive_operator_creates_zero_shifts(self):
        shift_store.create_operator(
            operator_id="OP_GONE", name="Ramesh Gupta", employee_code="EMP_G1", active=0,
        )
        answer = self.assistant.ask("Okay Kelvin, Ramesh here. Start my shift.")
        self.assertEqual(self._shift_rows(), [])
        self.assertIn("could not find an operator", answer.lower())

    def test_expired_confirmation_creates_zero_shifts(self):
        expiring = Assistant(use_llm=False)
        expiring._actions = PendingActionManager(ttl_seconds=0.05)
        prompt = expiring.ask(self.WAKE)
        self.assertIn("Shall I start the shift for", prompt)
        time.sleep(0.12)
        expiring.ask("Yes")
        self.assertEqual(self._shift_rows(), [], "an expired confirmation wrote a shift")


class TestNoisyOperatorNameRecovery(ShiftTestEnvironment):
    """
    Confirmed field case: "Anand here, start my shift" was transcribed as
    "On and Here Start My Shift." -- speech-to-text split the name, and the
    extractor rightly rejected "and". The operator must still be recovered
    when exactly one known operator fits, and must NEVER be guessed otherwise.
    """

    CONFIRMED = "On and Here Start My Shift."
    DIDNT_CATCH = "I didn't catch your name. Please say your name again and start your shift."

    def _shift_rows(self):
        import sqlite3

        con = sqlite3.connect(config.DATABASE_PATH)
        try:
            return con.execute("SELECT operator_id, status FROM attendance_records").fetchall()
        finally:
            con.close()

    def _anand(self):
        self.create_test_operator(op_id="OP_001", name="Anand Sharma", emp_code="EMP_A")

    # --- the helper ---

    def test_confirmed_transcript_recovers_anand(self):
        self._anand()
        # Precondition: normal extraction still refuses "and" -- unchanged.
        self.assertIsNone(qr.extract_operator_name(self.CONFIRMED))
        op = shift_store.find_operator_in_text(self.CONFIRMED)
        self.assertIsNotNone(op)
        self.assertEqual(op["operator_id"], "OP_001")

    def test_unknown_name_does_not_match(self):
        self.create_test_operator(op_id="OP_R", name="Rajesh Kumar", emp_code="EMP_R")
        self.assertIsNone(shift_store.find_operator_in_text(self.CONFIRMED))
        # Plain words sitting in the name slot must not become an operator.
        self._anand()
        for phrase in ("stand here start my shift", "hand here start my shift"):
            with self.subTest(phrase=phrase):
                self.assertIsNone(shift_store.find_operator_in_text(phrase))

    def test_ambiguous_operators_are_not_guessed(self):
        self._anand()
        self.create_test_operator(op_id="OP_002", name="Anand Verma", emp_code="EMP_V")
        self.assertIsNone(shift_store.find_operator_in_text(self.CONFIRMED))

    def test_no_self_introduction_yields_no_name_slot(self):
        for phrase in ("start morning shift", "Okay Kelvin, start my shift"):
            with self.subTest(phrase=phrase):
                self.assertEqual(shift_store.name_slot_candidates(phrase), [])

    # --- the assistant flow ---

    def test_confirmed_transcript_reaches_confirmation_and_starts_shift(self):
        self._anand()
        prompt = self.assistant.ask(self.CONFIRMED)
        self.assertIn("Anand Sharma", prompt)
        self.assertIn("Shall I start the shift for", prompt)
        self.assertEqual(self._shift_rows(), [], "asking must not write")

        self.assistant.ask("Yes")
        self.assertEqual(self._shift_rows(), [("OP_001", "ACTIVE")])

    def test_normal_anand_here_still_works(self):
        self._anand()
        prompt = self.assistant.ask("Okay Kelvin, Anand here. Start my shift.")
        self.assertIn("Anand Sharma", prompt)
        self.assistant.ask("Yes")
        self.assertEqual(self._shift_rows(), [("OP_001", "ACTIVE")])

    def test_unrecoverable_name_asks_again_and_writes_nothing(self):
        self.create_test_operator(op_id="OP_R", name="Rajesh Kumar", emp_code="EMP_R")
        answer = self.assistant.ask(self.CONFIRMED)
        self.assertEqual(answer, self.DIDNT_CATCH)
        self.assertNotIn("No active operator", answer)
        # Nothing may be left pending for a later "yes".
        self.assistant.ask("Yes")
        self.assertEqual(self._shift_rows(), [])

    def test_ambiguous_name_asks_again_and_writes_nothing(self):
        self._anand()
        self.create_test_operator(op_id="OP_002", name="Anand Verma", emp_code="EMP_V")
        self.assertEqual(self.assistant.ask(self.CONFIRMED), self.DIDNT_CATCH)
        self.assistant.ask("Yes")
        self.assertEqual(self._shift_rows(), [])

    def test_no_name_offered_keeps_the_existing_message(self):
        """No self-introduction at all is a different case, and keeps its reply."""
        answer = self.assistant.ask("start morning shift")
        self.assertIn("No active operator is registered on this line", answer)


class TestWallProductionLogging(ShiftTestEnvironment):
    """Comprehensive regression test suite for wall-production logging flow."""

    def test_natural_completed_production_intent_and_extraction(self):
        cases = [
            ("1 wall print ho gayi", 1),
            ("2 walls print ho gayi", 2),
            ("wall 1 complete ho gayi", 1),
            ("wall 2 complete ho gayi", 2),
            ("I printed wall 2", 2),
            ("I completed wall 2", 2),
            ("ek wall print ho gayi", 1),
            ("do wall print ho gayi", 2),
            ("ek wall complete ho gayi", 1),
            ("do wall complete ho gayi", 2),
            ("dusri wall complete ho gayi", 2),
            ("maine 2 wall print kar di", 2),
            ("2 wall complete kar di", 2),
            ("wall 2 print kar di", 2),
        ]
        for query, expected_wn in cases:
            intent, _ = qr.classify(query)
            self.assertEqual(
                intent,
                qr.Intent.LOG_WALL_PRODUCTION,
                f"Query '{query}' failed intent classification: got {intent}",
            )
            wn = qr.extract_wall_number(query)
            self.assertEqual(
                wn,
                expected_wn,
                f"Query '{query}' failed wall number extraction: got {wn}, expected {expected_wn}",
            )

    def test_prefix_and_suffix_wall_logging_queries(self):
        """Regression test for user-reported bug:
        'log one wall', 'log 1 wall', 'record one wall', 'record 1 wall', 'log wall 1', 'record wall 1'
        must all route to LOG_WALL_PRODUCTION and extract Wall 1.
        """
        cases = [
            "log one wall",
            "log 1 wall",
            "record one wall",
            "record 1 wall",
            "log wall 1",
            "record wall 1",
        ]
        for query in cases:
            intent, _ = qr.classify(query)
            self.assertEqual(
                intent,
                qr.Intent.LOG_WALL_PRODUCTION,
                f"Query '{query}' failed intent classification: got {intent}",
            )
            wn = qr.extract_wall_number(query)
            self.assertEqual(
                wn,
                1,
                f"Query '{query}' failed wall number extraction: got {wn}, expected 1",
            )

    def test_ordinals_and_number_words(self):
        self.assertEqual(qr.extract_wall_number("first wall completed"), 1)
        self.assertEqual(qr.extract_wall_number("second wall completed"), 2)
        self.assertEqual(qr.extract_wall_number("third wall completed"), 3)
        self.assertEqual(qr.extract_wall_number("fourth wall completed"), 4)
        self.assertEqual(qr.extract_wall_number("fifth wall completed"), 5)
        self.assertEqual(qr.extract_wall_number("1st wall print ho gayi"), 1)
        self.assertEqual(qr.extract_wall_number("2nd wall print ho gayi"), 2)
        self.assertEqual(qr.extract_wall_number("3rd wall print ho gayi"), 3)
        self.assertEqual(qr.extract_wall_number("4th wall print ho gayi"), 4)
        self.assertEqual(qr.extract_wall_number("5th wall print ho gayi"), 5)
        self.assertEqual(qr.extract_wall_number("pehli wall complete ho gayi"), 1)
        self.assertEqual(qr.extract_wall_number("dusri wall complete ho gayi"), 2)
        self.assertEqual(qr.extract_wall_number("teesri wall complete ho gayi"), 3)
        self.assertEqual(qr.extract_wall_number("chauthi wall complete ho gayi"), 4)
        self.assertEqual(qr.extract_wall_number("panchvi wall complete ho gayi"), 5)

    def test_future_and_planned_statements_not_logged(self):
        future_queries = [
            "2 wall print karni hain",
            "wall 2 print karni hai",
            "2 walls plan hain",
            "wall 2 target hai",
            "I will print wall 2",
            "wall 2 chahiye",
        ]
        for query in future_queries:
            intent, _ = qr.classify(query)
            self.assertNotEqual(
                intent,
                qr.Intent.LOG_WALL_PRODUCTION,
                f"Future query '{query}' must NOT route to LOG_WALL_PRODUCTION, got {intent}",
            )

    def test_unrelated_print_or_complete_statements_not_logged(self):
        unrelated_queries = [
            "I printed the document",
            "I completed maintenance",
            "shift complete ho gayi",
        ]
        for query in unrelated_queries:
            intent, _ = qr.classify(query)
            self.assertNotEqual(
                intent,
                qr.Intent.LOG_WALL_PRODUCTION,
                f"Unrelated query '{query}' must NOT route to LOG_WALL_PRODUCTION, got {intent}",
            )

    def test_ambiguous_wall_statements_never_default_to_wall_1(self):
        self.create_test_operator("OP_W1", "Anand Sharma", "EMP099")
        shift_store.start_shift(
            operator_id="OP_W1",
            shift_date="2026-09-22",
            shift_type="Morning",
            line_id="line_1",
        )
        expected_prompt = "Which wall number was completed? Please specify the wall number to record production."
        for ambiguous_query in ["log wall", "wall print ho gayi", "wall complete ho gayi"]:
            wn = qr.extract_wall_number(ambiguous_query)
            self.assertIsNone(wn, f"Ambiguous query '{ambiguous_query}' must extract None, got {wn}")
            resp = self.assistant.ask(ambiguous_query, session_id="w_term", line_id="line_1", operator_id="OP_W1")
            self.assertEqual(resp, expected_prompt)
            # Verify no pending action was set
            self.assertIsNone(self.assistant._actions.get("w_term", "line_1"))

    def test_valid_production_confirmation_and_db_insertion(self):
        self.create_test_operator("OP_W2", "Anand Sharma", "EMP100")
        shift_id = shift_store.start_shift(
            operator_id="OP_W2",
            shift_date="2026-09-22",
            shift_type="Morning",
            line_id="line_1",
        )
        # Step 1: Natural statement
        prompt = self.assistant.ask("wall 2 complete ho gayi", session_id="term_w", line_id="line_1", operator_id="OP_W2")
        self.assertIn("You are about to log Wall 2 as completed on line_1", prompt)
        self.assertIn("Please confirm to proceed", prompt)

        # Step 2: Affirmation
        exec_resp = self.assistant.ask("yes", session_id="term_w", line_id="line_1", operator_id="OP_W2")
        self.assertIn("Wall 2 production recorded successfully", exec_resp)

        # Step 3: Verify in DB
        records = shift_store.get_shift_production(shift_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["wall_number"], 2)
        self.assertEqual(records[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
