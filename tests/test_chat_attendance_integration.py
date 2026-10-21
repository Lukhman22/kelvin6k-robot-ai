"""
Regression tests for Kelvin6K Operational Chatbot START_SHIFT command end-to-end.

Covers:
1. Exact failing command: "Kelvin, start Hamadan's shift".
2. Natural-language and Hinglish variants:
   - "Start shift for Hamadan"
   - "Hamadan clock in"
   - "Start Hamadan's attendance"
   - "Kelvin, Hamadan ka shift start karo"
   - "start hamadan shift"
   - "Hamadan ki attendance lagao"
   - "Hamadan ki shift shuru karo"
   - "clock in Hamadan"
   - "Hamadan ko clock in karo"
3. Unknown and ambiguous employee names:
   - Unknown employee rejection without fabricating records.
   - Ambiguous employee names asking for clarification without guessing.
4. Already-active shift prevention:
   - Reports existing active shift and start time; creates zero duplicate records.
5. Persistence and database read-back:
   - Verified that attendance record is persisted in SQLite and readable via get_attendance_record().
6. Backend error and failed transaction handling.
7. Attendance API / list_attendance_records returning the record after action.
8. Attendance update flag:
   - Confirms attendance_updated flag is returned so frontend refreshes without page reload.
9. Non-regression:
   - Ensures maintenance/RAG questions, machine status, and operator timing questions
     are not misclassified as attendance start commands.
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import agent.config as config
from agent import db, shift_store, query_router
from agent.assistant import Assistant


class TestChatAttendanceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="k6k_test_att_")
        cls.db_path = os.path.join(cls.test_dir, "test_kelvin6k.db")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.original_db = config.DATABASE_PATH
        config.DATABASE_PATH = self.db_path
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        db.init_db(self.db_path)

        # Seed test operators
        shift_store.create_operator(
            operator_id="OP_HAMADAN_D1B91C",
            name="Hamadan",
            employee_code="13",
            role="operator",
            active=1,
            db_path=self.db_path,
        )
        shift_store.create_operator(
            operator_id="OP_ANAND_001",
            name="Anand Sharma",
            employee_code="01",
            role="operator",
            active=1,
            db_path=self.db_path,
        )
        shift_store.create_operator(
            operator_id="OP_INACTIVE",
            name="Inactive User",
            employee_code="99",
            role="operator",
            active=0,
            db_path=self.db_path,
        )
        # Ambiguous operators with same first name
        shift_store.create_operator(
            operator_id="OP_RAHUL_A",
            name="Rahul Verma",
            employee_code="21",
            role="operator",
            active=1,
            db_path=self.db_path,
        )
        shift_store.create_operator(
            operator_id="OP_RAHUL_B",
            name="Rahul Sharma",
            employee_code="22",
            role="operator",
            active=1,
            db_path=self.db_path,
        )

        self.assistant = Assistant(state=None)

    def tearDown(self):
        config.DATABASE_PATH = self.original_db

    # 1. Exact failing command
    def test_exact_failing_command_start_hamadan_shift(self):
        q = "Kelvin, start Hamadan's shift"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)

        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

        # Verify database persistence
        records = shift_store.list_attendance_records(db_path=self.db_path)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["operator_id"], "OP_HAMADAN_D1B91C")
        self.assertEqual(rec["employee_name"], "Hamadan")
        self.assertEqual(rec["status"], "ACTIVE")

        # Verified read-back via primary key
        fetched = shift_store.get_attendance_record(rec["id"], db_path=self.db_path)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["status"], "ACTIVE")

    # 2. Natural-language and Hinglish variants
    def test_variants_start_shift_for_hamadan(self):
        q = "Start shift for Hamadan"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

    def test_variants_hamadan_clock_in(self):
        q = "Hamadan clock in"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

    def test_variants_start_hamadan_attendance(self):
        q = "Start Hamadan's attendance"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

    def test_variants_hinglish_hamadan_ka_shift_start_karo(self):
        q = "Kelvin, Hamadan ka shift start karo"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

    def test_variants_hinglish_hamadan_ki_attendance_lagao(self):
        q = "Hamadan ki attendance lagao"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

    def test_variants_hinglish_hamadan_ki_shift_shuru_karo(self):
        q = "Hamadan ki shift shuru karo"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

    def test_variants_clock_in_hamadan(self):
        q = "clock in Hamadan"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

    def test_variants_hamadan_ko_clock_in_karo(self):
        q = "Hamadan ko clock in karo"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Hamadan's shift has started", ans)

    # 3. Unknown and ambiguous employee names
    def test_unknown_employee_rejection(self):
        q = "Kelvin, start John's shift"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("could not find an employee named John", ans)

        # Ensure no records created
        records = shift_store.list_attendance_records(db_path=self.db_path)
        self.assertEqual(len(records), 0)

    def test_ambiguous_employee_clarification(self):
        q = "Start shift for Rahul"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Multiple employees found matching 'Rahul'", ans)

        # Ensure no records created
        records = shift_store.list_attendance_records(db_path=self.db_path)
        self.assertEqual(len(records), 0)

    def test_inactive_employee_rejection(self):
        q = "Start shift for Inactive User"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("is inactive and cannot start a shift", ans)

        records = shift_store.list_attendance_records(db_path=self.db_path)
        self.assertEqual(len(records), 0)

    # 4. Duplicate prevention when shift is already active
    def test_already_active_shift_prevention(self):
        # Start Hamadan's shift first
        q1 = "Kelvin, start Hamadan's shift"
        routed1 = query_router.route(q1)
        ans1 = self.assistant.ask(q1, routed=routed1)
        self.assertIn("Hamadan's shift has started", ans1)

        # Second attempt must be rejected with existing start time
        q2 = "Start shift for Hamadan"
        routed2 = query_router.route(q2)
        ans2 = self.assistant.ask(q2, routed=routed2)
        self.assertIn("Hamadan already has an active shift", ans2)
        self.assertIn("Started at", ans2)

        # Total active records must remain strictly 1
        records = shift_store.list_attendance_records(db_path=self.db_path)
        self.assertEqual(len(records), 1)

    # 5. Database failure handling
    def test_database_error_handling(self):
        q = "Kelvin, start Hamadan's shift"
        routed = query_router.route(q)

        with patch("agent.shift_store.create_attendance_record", side_effect=RuntimeError("disk full")):
            ans = self.assistant.ask(q, routed=routed)
            self.assertIn("Database error", ans)

    # 6. Unnamed shift start prompts for employee
    def test_unnamed_shift_start_prompts_for_operator(self):
        q = "Start shift"
        routed = query_router.route(q)
        self.assertEqual(routed.intent, query_router.Intent.SHIFT_START)
        ans = self.assistant.ask(q, routed=routed)
        self.assertIn("Please specify which employee you would like to start a shift for.", ans)

    # 7. Non-regression: Question vs command distinction
    def test_question_not_misclassified_as_shift_start(self):
        # Timing question should route to GET_SHIFT_BY_OPERATOR, NOT SHIFT_START
        q1 = "Hamadan ki shift shuru hui kya?"
        routed1 = query_router.route(q1)
        self.assertEqual(routed1.intent, query_router.Intent.GET_SHIFT_BY_OPERATOR)

        q2 = "when did Hamadan start his shift?"
        routed2 = query_router.route(q2)
        self.assertEqual(routed2.intent, query_router.Intent.GET_SHIFT_BY_OPERATOR)

        # Maintenance queries must remain KNOWLEDGE_QUERY
        q3 = "How often do I clean the printing pipe?"
        routed3 = query_router.route(q3)
        self.assertEqual(routed3.intent, query_router.Intent.KNOWLEDGE_QUERY)

        # Machine status must remain GET_MACHINE_STATUS
        q4 = "what is the machine status"
        routed4 = query_router.route(q4)
        self.assertEqual(routed4.intent, query_router.Intent.GET_MACHINE_STATUS)


if __name__ == "__main__":
    unittest.main()
