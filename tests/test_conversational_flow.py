import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config, shift_store, db
from agent.assistant import Assistant


class TestConversationalFlow(unittest.TestCase):
    """
    End-to-end integration tests verifying the 9 conversational flow requirements:
    1. Operator onboarding & shift start (new & existing)
    2. Addressing current operator by name
    3. Shift start with date & time confirmation
    4. Human-driven interaction: wall count query & production logging
    5. Issue logging with strict RAG verification logic
    6. Issue action & combined action-resolution without RAG re-trigger
    7. Shift end summary (name, date, start/end time, walls, issues) & closing
    8. Operator isolation (next operator gets fresh context & separate DB records)
    9. Historical queries via SQLite
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_conv.db")
        db.init_db(self.db_path)
        self._patcher = patch.object(config, "DATABASE_PATH", self.db_path)
        self._patcher.start()
        if hasattr(config, "DB_PATH"):
            self._db_path_patcher = patch.object(config, "DB_PATH", self.db_path)
            self._db_path_patcher.start()
        else:
            self._db_path_patcher = None
        self.assistant = Assistant(use_llm=False)

    def tearDown(self):
        if self._db_path_patcher:
            self._db_path_patcher.stop()
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_flow_complete_lifecycle(self):
        # -------------------------------------------------------------
        # Stage 1 & 2 & 3: New Operator "Rahul" says: "Hey Kelvin, Rahul here. Start my shift."
        # -------------------------------------------------------------
        resp = self.assistant.ask("Hey Kelvin, Rahul here. Start my shift.")
        # Kelvin must address Rahul by name, state date & time, mention registration confirmation
        self.assertIn("rahul", resp.lower())
        self.assertIn("could not find an operator named rahul", resp.lower())
        self.assertIn("register as a new employee and start your shift", resp.lower())

        # Confirm registration and start
        confirm_resp = self.assistant.ask("Yes")
        self.assertIn("rahul", confirm_resp.lower())
        self.assertIn("has started", confirm_resp.lower())

        # Verify DB operator and shift
        op = shift_store.find_operator_by_name("Rahul")
        self.assertIsNotNone(op)
        self.assertEqual(op["name"], "Rahul")
        active_shift = shift_store.get_active_shift_for_line("line_1")
        self.assertIsNotNone(active_shift)
        self.assertEqual(active_shift["operator_id"], op["operator_id"])
        self.assertEqual(active_shift["status"], "active")

        # Verify session context
        sess = self.assistant._session_ctx.get("default_terminal", "line_1")
        self.assertEqual(sess.current_operator_id, op["operator_id"])
        self.assertEqual(sess.current_operator_name, "Rahul")
        self.assertEqual(sess.current_shift_id, active_shift["shift_id"])

        # -------------------------------------------------------------
        # Stage 4: Production queries and completion logging
        # -------------------------------------------------------------
        # "How many walls have I printed?"
        how_many = self.assistant.ask("How many walls have I printed?")
        self.assertIn("rahul", how_many.lower())
        self.assertIn("no completed wall records", how_many.lower())

        # "I completed two walls."
        log_wall_prompt = self.assistant.ask("I completed two walls.")
        self.assertIn("rahul", log_wall_prompt.lower())
        self.assertIn("you are about to log wall 2 as completed on line_1", log_wall_prompt.lower())
        
        # Confirm wall logging
        log_confirm = self.assistant.ask("Yes")
        self.assertIn("rahul", log_confirm.lower())
        self.assertIn("wall 2 production recorded successfully", log_confirm.lower())

        # Verify production record in DB
        prod_records = shift_store.get_shift_production(active_shift["shift_id"])
        self.assertEqual(len(prod_records), 1)
        self.assertEqual(prod_records[0]["wall_number"], 2)
        self.assertEqual(prod_records[0]["status"], "completed")

        # Now "How many walls have I printed?" should show 1 wall (Wall 2)
        how_many_now = self.assistant.ask("How many walls have I printed?")
        self.assertIn("rahul", how_many_now.lower())
        self.assertIn("wall 2", how_many_now.lower())

        # -------------------------------------------------------------
        # Stage 5: Problem logging with RAG
        # -------------------------------------------------------------
        # "I have a material flow problem."
        issue_prompt = self.assistant.ask("I have a material flow problem.")
        self.assertIn("rahul", issue_prompt.lower())
        self.assertIn("you are about to log an issue on line_1", issue_prompt.lower())

        # Confirm issue logging
        issue_confirm = self.assistant.ask("Yes")
        self.assertIn("rahul", issue_confirm.lower())
        self.assertIn("issue #1 logged successfully", issue_confirm.lower())

        # Verify DB issue record and RAG fields
        issues = shift_store.list_shift_issues(shift_id=active_shift["shift_id"])
        self.assertEqual(len(issues), 1)
        iss = issues[0]
        self.assertEqual(iss["issue_type"], "material_flow_issue")
        self.assertTrue(iss["rag_recommendation"])
        # Strict RAG verification check:
        # If the recommendation was site-verified, rag_verified is 1 and guidance is presented
        # If documented/unverified, rag_verified is 0 and "No verified solution" is stated
        if iss["rag_verified"] == 1:
            self.assertIn("recommendation:", issue_confirm.lower())
        else:
            self.assertEqual(iss["rag_verified"], 0)
            self.assertIn("no verified solution", issue_confirm.lower())

        # -------------------------------------------------------------
        # Stage 6: Operator solution & Combined resolution
        # -------------------------------------------------------------
        # Test combined statement: "I cleaned the nozzle, the material flow is normal."
        res_prompt = self.assistant.ask("I cleaned the nozzle, the material flow is normal.")
        self.assertIn("rahul", res_prompt.lower())
        self.assertIn("action 'i cleaned the nozzle'", res_prompt.lower())
        self.assertIn("status 'resolved'", res_prompt.lower())

        # Confirm resolution
        res_confirm = self.assistant.ask("Yes")
        self.assertIn("rahul", res_confirm.lower())
        self.assertIn("issue #1 updated successfully as resolved", res_confirm.lower())

        # Verify DB issue updated
        iss_updated = shift_store.get_issue(1)
        self.assertEqual(iss_updated["action_taken"], "I cleaned the nozzle")
        self.assertEqual(iss_updated["result"], "resolved")

        # -------------------------------------------------------------
        # Stage 7: Shift end summary & closing
        # -------------------------------------------------------------
        end_prompt = self.assistant.ask("End my shift.")
        self.assertIn("rahul", end_prompt.lower())
        self.assertIn("shall i end the shift for rahul?", end_prompt.lower())

        # Confirm shift end
        end_confirm = self.assistant.ask("Yes")
        self.assertIn("rahul", end_confirm.lower())
        self.assertIn("your shift has ended", end_confirm.lower())
        self.assertIn("duration:", end_confirm.lower())

        # Verify DB shift is closed
        closed_shift = shift_store.get_shift(active_shift["shift_id"])
        self.assertEqual(closed_shift["status"], "closed")
        self.assertIsNotNone(closed_shift["end_time"])

        # -------------------------------------------------------------
        # Stage 8: Next operator gets clean isolation
        # -------------------------------------------------------------
        # Suresh starts a new shift
        suresh_resp = self.assistant.ask("Hey Kelvin, Suresh here. Start my shift.")
        self.assertIn("suresh", suresh_resp.lower())
        self.assertIn("register as a new employee and start your shift", suresh_resp.lower())
        suresh_confirm = self.assistant.ask("Yes")
        self.assertIn("suresh", suresh_confirm.lower())
        self.assertIn("has started", suresh_confirm.lower())

        suresh_op = shift_store.find_operator_by_name("Suresh")
        self.assertIsNotNone(suresh_op)
        suresh_shift = shift_store.get_active_shift_for_line("line_1")
        self.assertEqual(suresh_shift["operator_id"], suresh_op["operator_id"])
        self.assertNotEqual(suresh_shift["shift_id"], closed_shift["shift_id"])

        # Suresh asks "How many walls have I printed?" -> should be 0, not Rahul's 1
        suresh_walls = self.assistant.ask("How many walls have I printed?")
        self.assertIn("suresh", suresh_walls.lower())
        self.assertIn("no completed wall records", suresh_walls.lower())

        # End Suresh's shift
        self.assistant.ask("End my shift.")
        self.assistant.ask("Yes")

        # -------------------------------------------------------------
        # Stage 9: Historical voice queries via SQLite
        # -------------------------------------------------------------
        today_date = datetime.date.today().strftime("%d %B")

        # "Who worked on <today_date>?"
        who_worked_date = self.assistant.ask(f"Who worked on {today_date}?")
        self.assertIn("rahul", who_worked_date.lower())
        self.assertIn("suresh", who_worked_date.lower())

        # "Who worked on Line 1?"
        # With no active shift currently, must strictly report no active shift running on Line 1
        who_line = self.assistant.ask("Who worked on Line 1?")
        self.assertIn("no active shift is currently running on line 1", who_line.lower())

        # "How many walls did Rahul print?"
        rahul_walls = self.assistant.ask("How many walls did Rahul print?")
        self.assertIn("rahul printed 1 walls: wall 2", rahul_walls.lower())

        # "What problem did Rahul have?"
        rahul_prob = self.assistant.ask("What problem did Rahul have?")
        self.assertIn("issue #1", rahul_prob.lower())
        self.assertIn("material flow", rahul_prob.lower())
        self.assertIn("action: i cleaned the nozzle", rahul_prob.lower())
        self.assertIn("result: resolved", rahul_prob.lower())

        # "When did Rahul's shift start?"
        rahul_start = self.assistant.ask("When did Rahul's shift start?")
        self.assertIn("rahul's shift started at", rahul_start.lower())

        # "When did Rahul's shift end?"
        rahul_end = self.assistant.ask("When did Rahul's shift end?")
        self.assertIn("rahul's shift ended at", rahul_end.lower())

        # "How many hours did Rahul work?"
        rahul_hours = self.assistant.ask("How many hours did Rahul work?")
        self.assertIn("rahul worked", rahul_hours.lower())

    def test_confirmation_mode_remains_active_on_unclear_or_wake_word(self):
        # 1. New operator Aman starts shift
        resp = self.assistant.ask("Hey Kelvin, Aman here. Start my shift.")
        self.assertIn("aman", resp.lower())
        self.assertIn("could not find an operator named aman", resp.lower())
        self.assertIn("register as a new employee and start your shift", resp.lower())

        expected_reask = "Aman, please answer Yes or No. New employee detected. I could not find an operator named Aman. Would you like to register as a new employee and start your shift?"

        # 2. Test phrases that must NOT be treated as confirmation:
        # "Okay Kelvin" (wake word / unclear)
        self.assertEqual(self.assistant.ask("Okay Kelvin"), expected_reask)

        # "Sure" (must NOT confirm, must keep pending and re-ask)
        self.assertEqual(self.assistant.ask("Sure"), expected_reask)

        # "Kar do" (must NOT confirm, must keep pending and re-ask)
        self.assertEqual(self.assistant.ask("Kar do"), expected_reask)

        # "Start karo" (must NOT confirm, must keep pending and re-ask)
        self.assertEqual(self.assistant.ask("Start karo"), expected_reask)

        # Unrelated question / command "What time is it?"
        self.assertEqual(self.assistant.ask("What time is it?"), expected_reask)

        # "I don't know" (uncertain)
        self.assertEqual(self.assistant.ask("I don't know"), expected_reask)

        # 3. Explicit YES executes
        confirm_resp = self.assistant.ask("Yes")
        self.assertIn("aman", confirm_resp.lower())
        self.assertIn("has started", confirm_resp.lower())

        # Verify DB operator and shift created
        op = shift_store.find_operator_by_name("Aman")
        self.assertIsNotNone(op)
        active_shift = shift_store.get_active_shift_for_line("line_1")
        self.assertIsNotNone(active_shift)
        self.assertEqual(active_shift["operator_id"], op["operator_id"])

    def test_confirmation_cancellation_and_rejection(self):
        expected_rohit_reask = "Rohit, please answer Yes or No. New employee detected. I could not find an operator named Rohit. Would you like to register as a new employee and start your shift?"

        # 1. Rejection with "No"
        self.assistant.ask("Hey Kelvin, Rohit here. Start my shift.")
        self.assertEqual(self.assistant.ask("Sure"), expected_rohit_reask)
        no_resp = self.assistant.ask("No")
        self.assertEqual(no_resp, "Action cancelled. No changes were made.")
        self.assertIsNone(shift_store.find_operator_by_name("Rohit"))
        self.assertIsNone(shift_store.get_active_shift_for_line("line_1"))

        # 2. Rejection with "Nahi"
        self.assistant.ask("Hey Kelvin, Rohit here. Start my shift.")
        nahi_resp = self.assistant.ask("Nahi")
        self.assertEqual(nahi_resp, "Action cancelled. No changes were made.")
        self.assertIsNone(shift_store.find_operator_by_name("Rohit"))

        # 3. Rejection with "Not"
        self.assistant.ask("Hey Kelvin, Rohit here. Start my shift.")
        not_resp = self.assistant.ask("Not now")
        self.assertEqual(not_resp, "Action cancelled. No changes were made.")
        self.assertIsNone(shift_store.find_operator_by_name("Rohit"))

        # 4. Cancellation with "Cancel"
        self.assistant.ask("Hey Kelvin, Rohit here. Start my shift.")
        cancel_resp = self.assistant.ask("Cancel")
        self.assertEqual(cancel_resp, "Action cancelled. No changes were made.")
        self.assertIsNone(shift_store.find_operator_by_name("Rohit"))

        # 5. Cancellation with "Stop"
        self.assistant.ask("Hey Kelvin, Rohit here. Start my shift.")
        stop_resp = self.assistant.ask("Stop")
        self.assertEqual(stop_resp, "Action cancelled. No changes were made.")
        self.assertIsNone(shift_store.find_operator_by_name("Rohit"))

        # 6. Explicit "Haan" executes
        self.assistant.ask("Hey Kelvin, Rohit here. Start my shift.")
        haan_resp = self.assistant.ask("Haan")
        self.assertIn("rohit", haan_resp.lower())
        self.assertIn("has started", haan_resp.lower())
        self.assertIsNotNone(shift_store.find_operator_by_name("Rohit"))
        self.assertIsNotNone(shift_store.get_active_shift_for_line("line_1"))

        # 7. Explicit "Confirm" executes for shift end
        end_prompt = self.assistant.ask("End my shift.")
        self.assertIn("shall i end the shift for rohit?", end_prompt.lower())
        self.assertEqual(self.assistant.ask("Kar do"), "Rohit, please answer Yes or No. Shall I end the shift for Rohit?")
        confirm_end = self.assistant.ask("Confirm")
        self.assertIn("has ended", confirm_end.lower())
        self.assertIsNone(shift_store.get_active_shift_for_line("line_1"))

    def test_current_problem_statement_routing_and_issue_logging(self):
        # 1. Start shift for Aman
        self.assistant.ask("Hey Kelvin, Aman here. Start my shift.")
        self.assistant.ask("Yes")

        # 2. Say "I have a one problem."
        # Must route to LOG_SHIFT_ISSUE, NOT GET_SHIFT_ISSUES or GET_RECENT_PROBLEMS
        resp = self.assistant.ask("I have a one problem.")
        self.assertNotIn("most recent problem", resp.lower())
        self.assertIn("you are about to log an issue", resp.lower())

        # 3. Confirm logging
        confirm_resp = self.assistant.ask("Yes")
        self.assertIn("logged successfully", confirm_resp.lower())

        # 4. Verify DB has recorded the issue
        active = shift_store.get_active_shift_for_line("line_1")
        issues = shift_store.list_shift_issues(shift_id=active["shift_id"])
        self.assertEqual(len(issues), 1)

    def test_issue_workflow_no_db_write_before_confirmation_and_same_issue_saving(self):
        # 1. Start shift for Aman
        self.assistant.ask("Hey Kelvin, Aman here. Start my shift.")
        self.assistant.ask("Yes")
        active = shift_store.get_active_shift_for_line("line_1")
        self.assertIsNotNone(active)

        # Verify DB has 0 issues initially
        self.assertEqual(len(shift_store.list_shift_issues(shift_id=active["shift_id"])), 0)

        # 2. Operator says "I have a problem" -> Kelvin asks what problem
        resp1 = self.assistant.ask("I have a problem.")
        self.assertIn("aman, what problem are you facing?", resp1.lower())
        # Verify NO DB write
        self.assertEqual(len(shift_store.list_shift_issues(shift_id=active["shift_id"])), 0)

        # 3. Problem is stated -> search existing RAG
        # In no-site-verified case: only say no site-verified solution was found, and ask what action was taken
        resp2 = self.assistant.ask("Pump pressure drop")
        self.assertIn("aman", resp2.lower())
        self.assertIn("no site-verified solution was found", resp2.lower())
        self.assertIn("what action you took", resp2.lower())
        # Verify NO DB write
        self.assertEqual(len(shift_store.list_shift_issues(shift_id=active["shift_id"])), 0)

        # 4. Operator states action with generic 'working' phrasing -> must NOT default result to resolved
        resp3 = self.assistant.ask("I checked the pressure valve and am working on it.")
        self.assertIn("you are about to log an issue for 'pump pressure drop' with action 'i checked the pressure valve and am working on it'", resp3.lower())
        self.assertNotIn("status 'resolved'", resp3.lower())
        self.assertIn("please confirm to proceed", resp3.lower())
        # Verify NO DB write before confirmation
        self.assertEqual(len(shift_store.list_shift_issues(shift_id=active["shift_id"])), 0)

        # 5. Test NO/CANCEL does not save
        cancel_resp = self.assistant.ask("No")
        self.assertIn("action cancelled. no changes were made.", cancel_resp.lower())
        # Verify NO DB write on cancellation
        self.assertEqual(len(shift_store.list_shift_issues(shift_id=active["shift_id"])), 0)

        # 6. Re-run flow and test clear resolution phrase ("it is working now"):
        self.assistant.ask("I have a problem.")
        self.assistant.ask("Pump pressure drop")
        resp_res = self.assistant.ask("I checked the pressure valve and it is working now.")
        self.assertIn("status 'resolved'", resp_res.lower())
        self.assertEqual(len(shift_store.list_shift_issues(shift_id=active["shift_id"])), 0)

        confirm_resp = self.assistant.ask("Yes")
        self.assertIn("logged successfully", confirm_resp.lower())
        self.assertIn("action 'i checked the pressure valve'", confirm_resp.lower())
        self.assertIn("status 'resolved'", confirm_resp.lower())

        # 7. Verify SAME-issue saving (exactly 1 issue in DB, no second issue)
        issues = shift_store.list_shift_issues(shift_id=active["shift_id"])
        self.assertEqual(len(issues), 1)
        iss = issues[0]
        self.assertEqual(iss["shift_id"], active["shift_id"])
        self.assertEqual(iss["description"], "Pump pressure drop")
        self.assertEqual(iss["action_taken"], "I checked the pressure valve")
        self.assertEqual(iss["result"], "resolved")
        # In no-SITE-VERIFIED case: unverified RAG guidance is NOT stored
        self.assertIsNone(iss["rag_recommendation"])
        self.assertEqual(iss["rag_verified"], 0)
        self.assertIsNone(iss["rag_source"])
        self.assertIsNotNone(iss["timestamp"])

        # 8. Verify saving when operator does NOT state result (result remains None, no default to "resolved")
        self.assistant.ask("I have a problem.")
        self.assistant.ask("Cooling fan vibration")
        self.assistant.ask("I tightened the mount screws.")
        self.assistant.ask("Yes")
        issues2 = shift_store.list_shift_issues(shift_id=active["shift_id"])
        self.assertEqual(len(issues2), 2)
        iss2 = [i for i in issues2 if i["description"] == "Cooling fan vibration"][0]
        self.assertEqual(iss2["action_taken"], "I tightened the mount screws")
        self.assertIsNone(iss2["result"])
        self.assertIsNone(iss2["rag_recommendation"])
        self.assertEqual(iss2["rag_verified"], 0)
        self.assertIsNone(iss2["rag_source"])

    @patch("agent.assistant.search_documentation")
    def test_issue_workflow_site_verified_rag_preserves_metadata(self, mock_search):
        # Mock RAG returning a SITE-VERIFIED solution
        mock_search.return_value = {
            "found": True,
            "results": [
                {
                    "text": "Verification Status: SITE-VERIFIED\nIssue: Feeder jam\nProcedure: Stop feeder motor and clear aggregate blockage.",
                    "source": "feeder_sop.md, Section 2",
                }
            ],
            "message": "",
        }

        # 1. Start shift for Aman
        self.assistant.ask("Hey Kelvin, Aman here. Start my shift.")
        self.assistant.ask("Yes")
        active = shift_store.get_active_shift_for_line("line_1")

        # 2. Operator reports problem
        self.assistant.ask("I have a problem.")
        resp = self.assistant.ask("Feeder jam")
        self.assertIn("i found a site-verified solution:", resp.lower())
        self.assertIn("what action you took", resp.lower())

        # Verify no DB write yet
        self.assertEqual(len(shift_store.list_shift_issues(shift_id=active["shift_id"])), 0)

        # 3. Operator states action and confirmation
        prompt = self.assistant.ask("I cleared aggregate blockage and it is working now.")
        self.assertIn("status 'resolved'", prompt.lower())
        self.assistant.ask("Yes")

        # 4. Verify DB row preserves rag_recommendation, rag_verified=1, and rag_source
        issues = shift_store.list_shift_issues(shift_id=active["shift_id"])
        self.assertEqual(len(issues), 1)
        iss = issues[0]
        self.assertEqual(iss["description"], "Feeder jam")
        self.assertEqual(iss["action_taken"], "I cleared aggregate blockage")
        self.assertEqual(iss["result"], "resolved")
        self.assertEqual(iss["rag_verified"], 1)
        self.assertIsNotNone(iss["rag_recommendation"])
        self.assertIn("clear aggregate blockage", iss["rag_recommendation"].lower())
        self.assertEqual(iss["rag_source"], "feeder_sop.md, Section 2")


if __name__ == "__main__":
    unittest.main()

