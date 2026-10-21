"""
tests/test_shift_store_phase2.py

Focused validation test suite for Phase 2 shift_store changes:
1. RAG fields save/read/update on shift_issues (rag_recommendation, rag_verified, rag_source).
2. Existing issue records without RAG arguments preserve backward compatibility and defaults.
3. Operator production filtering: Anand vs Suresh strict operator isolation.
4. Operator production shift_id filtering.
5. Closed-shift history accessibility.
6. Cross-operator protection (cannot access foreign shift records) & empty operator handling.
7. Chronological ordering and complete production metadata preservation (wall, status, timestamps, job info).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config, db, shift_store


class TestShiftStorePhase2(unittest.TestCase):
    """Isolates tests with a fresh SQLite database and clean state."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db = config.DATABASE_PATH
        config.DATABASE_PATH = os.path.join(self._tmp.name, "test_phase2.db")
        db.init_db()

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        self._tmp.cleanup()

    # -----------------------------------------------------------------------
    # 1. RAG Fields Save / Read / Update
    # -----------------------------------------------------------------------

    def test_rag_fields_save_read_and_update(self):
        shift_store.create_operator(
            operator_id="OP_ANAND",
            name="Anand Sharma",
            employee_code="EMP101",
        )
        sid = shift_store.start_shift(
            operator_id="OP_ANAND",
            shift_date="2026-09-22",
            shift_type="Morning",
            line_id="line_1",
        )

        # 1a. Log issue with RAG fields populated
        issue_id = shift_store.log_issue(
            shift_id=sid,
            issue_type="nozzle_clog",
            description="Mortar clogging at nozzle tip",
            wall_number=2,
            rag_recommendation="Flush nozzle with 10% vinegar solution and check pressure valve",
            rag_verified=1,
            rag_source="SOP-042-NOZZLE-MAINT",
        )

        # 1b. Verify get_issue returns all 3 fields
        rec = shift_store.get_issue(issue_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["rag_recommendation"], "Flush nozzle with 10% vinegar solution and check pressure valve")
        self.assertEqual(rec["rag_verified"], 1)
        self.assertEqual(rec["rag_source"], "SOP-042-NOZZLE-MAINT")

        # 1c. Verify list_shift_issues returns all 3 fields for shift_id and operator_id queries
        issues_by_shift = shift_store.list_shift_issues(shift_id=sid)
        self.assertEqual(len(issues_by_shift), 1)
        self.assertEqual(issues_by_shift[0]["rag_recommendation"], "Flush nozzle with 10% vinegar solution and check pressure valve")
        self.assertEqual(issues_by_shift[0]["rag_verified"], 1)
        self.assertEqual(issues_by_shift[0]["rag_source"], "SOP-042-NOZZLE-MAINT")

        issues_by_op = shift_store.list_shift_issues(operator_id="OP_ANAND")
        self.assertEqual(len(issues_by_op), 1)
        self.assertEqual(issues_by_op[0]["rag_recommendation"], "Flush nozzle with 10% vinegar solution and check pressure valve")
        self.assertEqual(issues_by_op[0]["rag_verified"], 1)
        self.assertEqual(issues_by_op[0]["rag_source"], "SOP-042-NOZZLE-MAINT")

        # 1d. Verify update_issue can update RAG fields
        ok = shift_store.update_issue(
            issue_id=issue_id,
            action_taken="Flushed nozzle with vinegar",
            result="resolved",
            rag_recommendation="Updated SOP: Standard flush procedure",
            rag_verified=0,
            rag_source="SOP-042-REV2",
        )
        self.assertTrue(ok)
        updated = shift_store.get_issue(issue_id)
        self.assertEqual(updated["action_taken"], "Flushed nozzle with vinegar")
        self.assertEqual(updated["result"], "resolved")
        self.assertEqual(updated["rag_recommendation"], "Updated SOP: Standard flush procedure")
        self.assertEqual(updated["rag_verified"], 0)
        self.assertEqual(updated["rag_source"], "SOP-042-REV2")

    # -----------------------------------------------------------------------
    # 2. Existing Callers Backward Compatibility
    # -----------------------------------------------------------------------

    def test_existing_callers_without_rag_values_continue_working(self):
        shift_store.create_operator(
            operator_id="OP_ANAND",
            name="Anand Sharma",
            employee_code="EMP101",
        )
        sid = shift_store.start_shift(
            operator_id="OP_ANAND",
            shift_date="2026-09-22",
            shift_type="Morning",
            line_id="line_1",
        )

        # Call log_issue without any RAG parameters (existing caller pattern)
        issue_id = shift_store.log_issue(sid, "feeder_jam", "Feeder stopped feeding aggregate")
        rec = shift_store.get_issue(issue_id)
        self.assertIsNotNone(rec)
        self.assertIsNone(rec["rag_recommendation"])
        self.assertEqual(rec["rag_verified"], 0)
        self.assertIsNone(rec["rag_source"])

    # -----------------------------------------------------------------------
    # 3. Anand vs Suresh Production Filtering (Strict Operator Isolation)
    # -----------------------------------------------------------------------

    def test_operator_production_filtering_anand_vs_suresh(self):
        shift_store.create_operator("OP_ANAND", "Anand Sharma", "EMP101")
        shift_store.create_operator("OP_SURESH", "Suresh Patel", "EMP102")

        anand_shift = shift_store.start_shift("OP_ANAND", "2026-09-22", "Morning", line_id="line_1")
        suresh_shift = shift_store.start_shift("OP_SURESH", "2026-09-22", "Morning", line_id="line_2")

        # Anand logs Wall 1 and Wall 2
        shift_store.record_production(anand_shift, wall_number=1, start_time=100.0, print_file="anand_w1.gcode", status="completed")
        shift_store.record_production(anand_shift, wall_number=2, start_time=200.0, print_file="anand_w2.gcode", status="in_progress")

        # Suresh logs Wall 1
        shift_store.record_production(suresh_shift, wall_number=1, start_time=150.0, print_file="suresh_w1.gcode", status="completed")

        # Query Anand's production
        anand_records = shift_store.get_operator_production("OP_ANAND")
        self.assertEqual(len(anand_records), 2)
        self.assertEqual([r["wall_number"] for r in anand_records], [1, 2])
        self.assertTrue(all(r["operator_id"] == "OP_ANAND" for r in anand_records))
        self.assertTrue(all(r["line_id"] == "line_1" for r in anand_records))
        self.assertEqual(anand_records[0]["print_file"], "anand_w1.gcode")
        self.assertEqual(anand_records[0]["status"], "completed")
        self.assertEqual(anand_records[1]["status"], "in_progress")

        # Query Suresh's production
        suresh_records = shift_store.get_operator_production("OP_SURESH")
        self.assertEqual(len(suresh_records), 1)
        self.assertEqual(suresh_records[0]["wall_number"], 1)
        self.assertEqual(suresh_records[0]["operator_id"], "OP_SURESH")
        self.assertEqual(suresh_records[0]["line_id"], "line_2")
        self.assertEqual(suresh_records[0]["print_file"], "suresh_w1.gcode")

        # No cross-leakage: Anand has zero of Suresh's files/shifts, and vice versa
        anand_files = {r["print_file"] for r in anand_records}
        suresh_files = {r["print_file"] for r in suresh_records}
        self.assertNotIn("suresh_w1.gcode", anand_files)
        self.assertNotIn("anand_w1.gcode", suresh_files)

    # -----------------------------------------------------------------------
    # 4. Optional shift_id Filter
    # -----------------------------------------------------------------------

    def test_operator_production_shift_id_filter(self):
        shift_store.create_operator("OP_ANAND", "Anand Sharma", "EMP101")
        s1 = shift_store.start_shift("OP_ANAND", "2026-09-21", "Night", line_id="line_1")
        shift_store.record_production(s1, wall_number=1, start_time=100.0, status="completed")
        shift_store.end_shift(s1, status="closed")

        s2 = shift_store.start_shift("OP_ANAND", "2026-09-22", "Morning", line_id="line_1")
        shift_store.record_production(s2, wall_number=2, start_time=200.0, status="completed")

        # Filter by shift 1
        s1_records = shift_store.get_operator_production("OP_ANAND", shift_id=s1)
        self.assertEqual(len(s1_records), 1)
        self.assertEqual(s1_records[0]["wall_number"], 1)
        self.assertEqual(s1_records[0]["shift_id"], s1)

        # Filter by shift 2
        s2_records = shift_store.get_operator_production("OP_ANAND", shift_id=s2)
        self.assertEqual(len(s2_records), 1)
        self.assertEqual(s2_records[0]["wall_number"], 2)
        self.assertEqual(s2_records[0]["shift_id"], s2)

        # Without shift filter returns all
        all_records = shift_store.get_operator_production("OP_ANAND")
        self.assertEqual(len(all_records), 2)

    # -----------------------------------------------------------------------
    # 5. Closed-Shift History Accessible
    # -----------------------------------------------------------------------

    def test_closed_shift_history_accessible(self):
        shift_store.create_operator("OP_ANAND", "Anand Sharma", "EMP101")
        s1 = shift_store.start_shift("OP_ANAND", "2026-09-20", "Morning", line_id="line_1")
        shift_store.record_production(s1, wall_number=1, start_time=50.0, status="completed")
        shift_store.end_shift(s1, status="closed")

        records = shift_store.get_operator_production("OP_ANAND")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["wall_number"], 1)
        self.assertEqual(records[0]["shift_status"], "closed")

    # -----------------------------------------------------------------------
    # 6. Cross-Operator Isolation & Empty Input Handling
    # -----------------------------------------------------------------------

    def test_cross_operator_shift_id_isolation_and_empty_input(self):
        shift_store.create_operator("OP_ANAND", "Anand Sharma", "EMP101")
        shift_store.create_operator("OP_SURESH", "Suresh Patel", "EMP102")
        suresh_shift = shift_store.start_shift("OP_SURESH", "2026-09-22", "Morning", line_id="line_1")
        shift_store.record_production(suresh_shift, wall_number=1, start_time=100.0, status="completed")

        # Anand cannot access Suresh's production even if Suresh's shift_id is supplied
        isolated = shift_store.get_operator_production("OP_ANAND", shift_id=suresh_shift)
        self.assertEqual(isolated, [])

        # Empty string or whitespace operator_id returns empty list
        self.assertEqual(shift_store.get_operator_production(""), [])
        self.assertEqual(shift_store.get_operator_production("   "), [])
        self.assertEqual(shift_store.get_operator_production("NON_EXISTENT_OP"), [])

    # -----------------------------------------------------------------------
    # 7. Production Metadata and Chronological Ordering
    # -----------------------------------------------------------------------

    def test_production_chronological_ordering_and_metadata(self):
        shift_store.create_operator("OP_ANAND", "Anand Sharma", "EMP101")
        s = shift_store.start_shift("OP_ANAND", "2026-09-22", "Morning", line_id="line_1")
        shift_store.record_production(s, wall_number=2, start_time=200.0, end_time=250.0, wall_height=2.5, layers_completed=20, status="completed")
        shift_store.record_production(s, wall_number=1, start_time=100.0, end_time=150.0, wall_height=3.0, layers_completed=24, status="completed")

        records = shift_store.get_operator_production("OP_ANAND")
        self.assertEqual(len(records), 2)
        # Ordered chronologically by start_time ASC
        self.assertEqual(records[0]["wall_number"], 1)
        self.assertEqual(records[0]["wall_height"], 3.0)
        self.assertEqual(records[0]["layers_completed"], 24)
        self.assertEqual(records[1]["wall_number"], 2)


if __name__ == "__main__":
    unittest.main()
