"""
tests/test_issue_resolution.py

Dedicated regression tests for shift issue resolution lifecycle:
1. Unresolved issue filtering (NULL, '', and 'not_resolved' included; 'resolved' excluded).
2. Operator issue history query and routing.
3. RAG recommendation vs. operator history query separation.
4. Explicit issue ID selection.
5. Session last_issue_id reuse.
6. Action taken without result requires confirmation and does not prematurely write to DB.
7. Applied RAG recommendation preserves advisory source.
8. Vague Kelvin reference never becomes operator action.
9. Conditional/future statements do not update issue.
10. Invalid/closed last_issue_id fallback.
11. Multiple unresolved issues require clarification (no guessing).
12. Clarification selection and refusing 'both'.
13. Cancellation leaves database unchanged.
14. Safe cause extraction.
15. Preserves operator's meaningful wording.
16. Multi-issue historical report formatting with 'not recorded' for NULL fields.
17. SessionManager non-reentrant lock deadlock avoidance test.
18. Missing result never defaults to resolved test.
19. Consistent cancellation response across states test.
20. Explicit issue ID on already-resolved issue blocked test.
21. Semantic correction: Issue ID alone must NOT start resolution flow.
22. Context-specific cancellation: 'no'/'nahi' preserves disambiguation but marks not_resolved in awaiting_outcome.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config, db, shift_store
from agent import query_router as qr
from agent.assistant import Assistant


class IssueResolutionTestSuite(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db = config.DATABASE_PATH
        config.DATABASE_PATH = os.path.join(self._tmp.name, "test_resolution.db")
        db.init_db()
        self.assistant = Assistant(use_llm=False)

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        self._tmp.cleanup()

    def create_operator(self, op_id="OP001", name="Anand Sharma", code="EMP101"):
        return shift_store.create_operator(
            operator_id=op_id,
            name=name,
            employee_code=code,
            role="lead_operator",
            active=1,
        )

    def start_active_shift(self, op_id="OP001", line_id="line_1"):
        self.create_operator(op_id=op_id)
        sid = shift_store.start_shift(
            operator_id=op_id,
            shift_date="2026-09-22",
            shift_type="Morning",
            line_id=line_id,
            target_walls=5,
        )
        return {"shift_id": sid}

    def test_1_unresolved_issue_filtering(self):
        shift = self.start_active_shift()
        sid = shift["shift_id"]
        id1 = shift_store.log_issue(sid, "nozzle_clog", "Nozzle blocked with mortar")
        id2 = shift_store.log_issue(sid, "pump_pressure", "Pump pressure dropped")
        id3 = shift_store.log_issue(sid, "sensor_fault", "Sensor calibration issue")
        shift_store.update_issue(id2, action_taken="Adjusted pressure valve", result="not_resolved")
        shift_store.update_issue(id3, action_taken="Replaced sensor", result="resolved")
        unresolved = shift_store.list_shift_issues(shift_id=sid, unresolved_only=True)
        unresolved_ids = [i["issue_id"] for i in unresolved]
        self.assertIn(id1, unresolved_ids)
        self.assertIn(id2, unresolved_ids)
        self.assertNotIn(id3, unresolved_ids)

    def test_2_operator_issue_history_query_and_routing(self):
        shift = self.start_active_shift(op_id="OP001")
        sid = shift["shift_id"]
        id1 = shift_store.log_issue(sid, "nozzle_clog", "Nozzle blocked with mortar")
        shift_store.update_issue(id1, cause="Mortar buildup", action_taken="Water flush", result="resolved")
        anand_issues = shift_store.list_shift_issues(operator_id="OP001")
        self.assertEqual(len(anand_issues), 1)
        self.assertEqual(anand_issues[0]["action_taken"], "Water flush")
        self.assertEqual(anand_issues[0]["cause"], "Mortar buildup")
        routed = qr.route("What problems did Anand Sharma have?")
        self.assertEqual(routed.intent, qr.Intent.GET_SHIFT_ISSUES)
        self.assertEqual(routed.facts["operator_id"], "OP001")
        self.assertEqual(len(routed.facts["issues"]), 1)
        self.assertEqual(routed.facts["issues"][0]["action_taken"], "Water flush")

    def test_3_rag_recommendation_vs_operator_history(self):
        intent_kelvin, _ = qr.classify("What solution did Kelvin recommend?")
        self.assertEqual(intent_kelvin, qr.Intent.KNOWLEDGE_QUERY)
        intent_anand, _ = qr.classify("Anand ne kya solution kiya tha?")
        self.assertEqual(intent_anand, qr.Intent.GET_SHIFT_ISSUES)

    def test_4_explicit_issue_id_selection(self):
        shift = self.start_active_shift()
        sid = shift["shift_id"]
        id10 = shift_store.log_issue(sid, "nozzle_clog", "Nozzle blockage")
        id12 = shift_store.log_issue(sid, "feeder_jam", "Feeder jam")
        resp = self.assistant.ask(f"Issue {id10} solve ho gaya", session_id="s4", line_id="line_1")
        self.assertIn(f"Issue #{id10}", resp)
        self.assertIn("confirm", resp.lower())
        confirm_resp = self.assistant.ask("yes", session_id="s4", line_id="line_1")
        self.assertIn(f"Issue #{id10} updated successfully as resolved", confirm_resp)
        iss10 = shift_store.list_shift_issues(shift_id=sid)
        row10 = next(i for i in iss10 if i["issue_id"] == id10)
        row12 = next(i for i in iss10 if i["issue_id"] == id12)
        self.assertEqual(row10["result"], "resolved")
        self.assertIsNone(row12["result"])

    def test_5_session_last_issue_id_reuse(self):
        self.start_active_shift()
        self.assistant.ask("log issue with 15 minutes downtime", session_id="s5", line_id="line_1")
        log_confirm = self.assistant.ask("yes", session_id="s5", line_id="line_1")
        self.assertIn("logged successfully", log_confirm)
        sess = self.assistant._session_ctx.get("s5", "line_1")
        self.assertIsNotNone(sess.last_issue_id)
        logged_id = sess.last_issue_id
        resolve_prompt = self.assistant.ask("Nozzle clean kiya aur problem solve ho gayi", session_id="s5", line_id="line_1")
        self.assertIn(f"Issue #{logged_id}", resolve_prompt)
        self.assertIn("confirm", resolve_prompt.lower())
        exec_resp = self.assistant.ask("yes", session_id="s5", line_id="line_1")
        self.assertIn(f"Issue #{logged_id} updated successfully", exec_resp)

    def test_6_action_taken_without_result_requires_confirmation(self):
        shift = self.start_active_shift()
        iss_id = shift_store.log_issue(shift["shift_id"], "nozzle_clog", "Nozzle clog ho gaya")
        resp = self.assistant.ask("Nozzle clean kiya", session_id="s6", line_id="line_1")
        self.assertIn("Recorded action 'Nozzle clean kiya'", resp)
        self.assertIn("Did this resolve the issue", resp)
        db_issues = shift_store.list_shift_issues(shift_id=shift["shift_id"])
        row = next(i for i in db_issues if i["issue_id"] == iss_id)
        self.assertIsNone(row["action_taken"])
        self.assertIsNone(row["result"])
        confirm_prompt = self.assistant.ask("haan", session_id="s6", line_id="line_1")
        self.assertIn(f"Issue #{iss_id}", confirm_prompt)
        self.assertIn("action 'Nozzle clean kiya'", confirm_prompt)
        final_resp = self.assistant.ask("yes", session_id="s6", line_id="line_1")
        self.assertIn(f"Issue #{iss_id} updated successfully as resolved", final_resp)
        db_issues_after = shift_store.list_shift_issues(shift_id=shift["shift_id"])
        row_after = next(i for i in db_issues_after if i["issue_id"] == iss_id)
        self.assertEqual(row_after["action_taken"], "Nozzle clean kiya")
        self.assertEqual(row_after["result"], "resolved")

    def test_7_applied_rag_recommendation(self):
        shift = self.start_active_shift()
        iss_id = shift_store.log_issue(shift["shift_id"], "nozzle_clog", "Nozzle buildup")
        self.assistant._session_ctx.update(
            "s7", "line_1",
            last_issue_id=iss_id,
            last_rag_recommendation="Flush nozzle with warm water and restart pump.",
        )
        resp = self.assistant.ask(
            "Kelvin ne jo recommend kiya tha woh apply kar diya aur issue solve ho gaya",
            session_id="s7", line_id="line_1",
        )
        self.assertIn(f"Issue #{iss_id}", resp)
        self.assertIn("Applied Kelvin recommendation", resp)
        final_resp = self.assistant.ask("yes", session_id="s7", line_id="line_1")
        self.assertIn(f"Issue #{iss_id} updated successfully as resolved", final_resp)
        row = shift_store.list_shift_issues(shift_id=shift["shift_id"])[0]
        self.assertIn("Flush nozzle with warm water", row["action_taken"])
        self.assertEqual(row["result"], "resolved")

    def test_8_vague_kelvin_reference_never_action(self):
        shift = self.start_active_shift()
        iss_id = shift_store.log_issue(shift["shift_id"], "nozzle_clog", "Nozzle buildup")
        self.assistant._session_ctx.update(
            "s8", "line_1",
            last_issue_id=iss_id,
            last_rag_recommendation="Flush nozzle with warm water",
        )
        act, is_rag = qr.extract_operator_action("Kelvin ne bola tha")
        self.assertIsNone(act)
        self.assertFalse(is_rag)
        act2, is_rag2 = qr.extract_operator_action("Kelvin ne recommend kiya")
        self.assertIsNone(act2)
        self.assertFalse(is_rag2)
        self.assistant.ask("Kelvin ne bola tha", session_id="s8", line_id="line_1")
        row = shift_store.list_shift_issues(shift_id=shift["shift_id"])[0]
        self.assertIsNone(row["action_taken"])
        self.assertIsNone(row["result"])

    def test_9_conditional_future_statements_no_update(self):
        shift = self.start_active_shift()
        shift_store.log_issue(shift["shift_id"], "nozzle_clog", "Nozzle clog")
        res = qr.extract_resolution_result("Agar nozzle clean karenge toh solve ho jayega")
        self.assertIsNone(res)
        res2 = qr.extract_resolution_result("Check karte hain agar theek hoga")
        self.assertIsNone(res2)
        row = shift_store.list_shift_issues(shift_id=shift["shift_id"])[0]
        self.assertIsNone(row["result"])

    def test_10_invalid_or_closed_last_issue_id_fallback(self):
        prev_shift = self.start_active_shift(op_id="OP001", line_id="line_1")
        old_issue_id = shift_store.log_issue(prev_shift["shift_id"], "sensor_fault", "Old sensor issue")
        shift_store.end_shift(prev_shift["shift_id"], status="closed")
        curr_shift_id = shift_store.start_shift("OP001", "2026-09-22", "Afternoon", "line_1", 5)
        new_issue_id = shift_store.log_issue(curr_shift_id, "pump_drop", "New pump drop")
        self.assistant._session_ctx.update("s10", "line_1", last_issue_id=old_issue_id)
        resp = self.assistant.ask("Issue solve ho gaya", session_id="s10", line_id="line_1")
        self.assertIn(f"Issue #{new_issue_id}", resp)
        self.assertNotIn(f"Issue #{old_issue_id}", resp)

    def test_11_multiple_unresolved_issues_clarification(self):
        shift = self.start_active_shift()
        sid = shift["shift_id"]
        id1 = shift_store.log_issue(sid, "nozzle_clog", "Nozzle clogging")
        id2 = shift_store.log_issue(sid, "hopper_empty", "Hopper material empty")
        resp = self.assistant.ask("Issue solve ho gaya", session_id="s11", line_id="line_1")
        self.assertIn("Multiple unresolved issues", resp)
        self.assertIn(f"Issue #{id1}", resp)
        self.assertIn(f"Issue #{id2}", resp)
        self.assertIn("Which issue would you like to update?", resp)
        pending = self.assistant._actions.get("s11", "line_1")
        self.assertIsNone(pending)

    def test_12_clarification_selection_and_refuse_both(self):
        shift = self.start_active_shift()
        sid = shift["shift_id"]
        id1 = shift_store.log_issue(sid, "nozzle_clog", "Nozzle clogging")
        id2 = shift_store.log_issue(sid, "hopper_empty", "Hopper material empty")
        self.assistant.ask("Issue solve ho gaya", session_id="s12", line_id="line_1")
        refuse_resp = self.assistant.ask("both", session_id="s12", line_id="line_1")
        self.assertIn("Please resolve one issue at a time", refuse_resp)
        select_resp = self.assistant.ask(f"Issue {id1}", session_id="s12", line_id="line_1")
        self.assertIn(f"Issue #{id1}", select_resp)
        self.assertIn("confirm", select_resp.lower())
        final_resp = self.assistant.ask("yes", session_id="s12", line_id="line_1")
        self.assertIn(f"Issue #{id1} updated successfully as resolved", final_resp)
        issues = shift_store.list_shift_issues(shift_id=sid)
        row1 = next(i for i in issues if i["issue_id"] == id1)
        row2 = next(i for i in issues if i["issue_id"] == id2)
        self.assertEqual(row1["result"], "resolved")
        self.assertIsNone(row2["result"])

    def test_13_cancellation_leaves_database_unchanged(self):
        shift = self.start_active_shift()
        iss_id = shift_store.log_issue(shift["shift_id"], "nozzle_clog", "Nozzle clog")
        prompt = self.assistant.ask(f"Issue {iss_id} solve ho gaya", session_id="s13", line_id="line_1")
        self.assertIn("confirm", prompt.lower())
        cancel_resp = self.assistant.ask("cancel", session_id="s13", line_id="line_1")
        self.assertEqual(cancel_resp, "Action cancelled. No changes were made.")
        row = shift_store.list_shift_issues(shift_id=shift["shift_id"])[0]
        self.assertIsNone(row["result"])
        self.assertIsNone(row["action_taken"])

    def test_14_cause_extraction_safe(self):
        c1 = qr.extract_resolution_cause("cause material buildup tha")
        self.assertEqual(c1, "material buildup")
        c2 = qr.extract_resolution_cause("reason was material blockage")
        self.assertEqual(c2, "material blockage")
        c3 = qr.extract_resolution_cause("wajah material blockage thi")
        self.assertEqual(c3, "material blockage")
        c4 = qr.extract_resolution_cause("iska reason kya tha")
        self.assertIsNone(c4)

    def test_15_preserve_operator_meaningful_action(self):
        act, _ = qr.extract_operator_action("Nozzle ko water se flush kiya aur pump restart kiya")
        self.assertEqual(act, "Nozzle ko water se flush kiya aur pump restart kiya")
        self.assertNotEqual(act, "Nozzle clean kiya")

    def test_16_get_shift_issues_multi_issue_report_formatting(self):
        shift = self.start_active_shift(op_id="OP001")
        sid = shift["shift_id"]
        id1 = shift_store.log_issue(sid, "nozzle_clog", "Nozzle blocked with mortar")
        shift_store.update_issue(id1, cause="Mortar dry", action_taken="Water flush", result="resolved")
        shift_store.log_issue(sid, "feeder_jam", "Feeder stuck")
        routed = qr.route("What problems did Anand Sharma have?")
        summary = qr.summarise(routed)
        self.assertIn("Found 2 issues for Anand Sharma:", summary)
        self.assertIn("Cause: Mortar dry; Action: Water flush; Result: resolved", summary)
        self.assertIn("Cause: not recorded; Action: not recorded; Result: not recorded", summary)

    def test_17_session_manager_no_deadlock(self):
        """Correction 1: SessionManager.update() must not deadlock with non-reentrant lock."""
        res = self.assistant._session_ctx.update("term_dl", "line_1", last_issue_id=99)
        self.assertEqual(res.last_issue_id, 99)
        res2 = self.assistant._session_ctx.get("term_dl", "line_1")
        self.assertEqual(res2.last_issue_id, 99)

    def test_18_missing_result_never_defaults_to_resolved(self):
        """Correction 2: Missing result must NEVER default to resolved."""
        shift = self.start_active_shift()
        iss_id = shift_store.log_issue(shift["shift_id"], "feeder_jam", "Feeder jam")
        # Operator states an action without stating an outcome
        resp = self.assistant.ask(f"Issue {iss_id} nozzle clean kiya", session_id="s18", line_id="line_1")
        self.assertIn("Did this resolve the issue", resp)
        # Verify DB action_taken and result remain NULL while pending outcome
        row = shift_store.list_shift_issues(shift_id=shift["shift_id"])[0]
        self.assertIsNone(row["result"])
        self.assertIsNone(row["action_taken"])

    def test_19_cancellation_consistent_response(self):
        """Correction 3: Cancellation across states returns 'Action cancelled. No changes were made.'"""
        shift = self.start_active_shift()
        sid = shift["shift_id"]
        id1 = shift_store.log_issue(sid, "nozzle_clog", "Nozzle clog")
        id2 = shift_store.log_issue(sid, "feeder_jam", "Feeder jam")

        # 1. Cancel in disambiguation
        self.assistant.ask("Issue solve ho gaya", session_id="s19_a", line_id="line_1")
        c1 = self.assistant.ask("cancel", session_id="s19_a", line_id="line_1")
        self.assertEqual(c1, "Action cancelled. No changes were made.")

        # 2. Cancel in awaiting outcome
        self.assistant.ask("Nozzle clean kiya", session_id="s19_b", line_id="line_1")
        c2 = self.assistant.ask("cancel", session_id="s19_b", line_id="line_1")
        self.assertEqual(c2, "Action cancelled. No changes were made.")

        # 3. Cancel in pending action confirmation
        self.assistant.ask(f"Issue {id1} solve ho gaya", session_id="s19_c", line_id="line_1")
        c3 = self.assistant.ask("cancel", session_id="s19_c", line_id="line_1")
        self.assertEqual(c3, "Action cancelled. No changes were made.")

    def test_20_explicit_issue_id_already_resolved_blocked(self):
        """Correction 4: Explicit issue ID for an already-resolved issue is blocked."""
        shift = self.start_active_shift()
        iss_id = shift_store.log_issue(shift["shift_id"], "sensor_fault", "Sensor failure")
        shift_store.update_issue(iss_id, action_taken="Replaced wire", result="resolved")

        resp = self.assistant.ask(f"Issue {iss_id} solve ho gaya", session_id="s20", line_id="line_1")
        self.assertEqual(resp, f"Issue #{iss_id} is already resolved.")
        pending = self.assistant._actions.get("s20", "line_1")
        self.assertIsNone(pending)

    def test_21_issue_id_alone_does_not_start_resolution(self):
        """Semantic Correction: Issue ID alone must NOT start resolution flow."""
        shift = self.start_active_shift()
        iss_id = shift_store.log_issue(shift["shift_id"], "feeder_jam", "Feeder jam")
        # Operator provides issue ID alone without action or resolution language
        resp = self.assistant.ask(f"Issue {iss_id}", session_id="s21", line_id="line_1")
        # Must ask what user wants to do, not enter resolution flow
        self.assertIn(f"What would you like to do with Issue #{iss_id}?", resp)
        # Verify no awaiting_outcome or pending action
        sess = self.assistant._session_ctx.get("s21", "line_1")
        self.assertIsNone(sess.awaiting_outcome)
        pending = self.assistant._actions.get("s21", "line_1")
        self.assertIsNone(pending)
        # Verify database remains unchanged (action_taken and result remain NULL)
        row = shift_store.list_shift_issues(shift_id=shift["shift_id"])[0]
        self.assertIsNone(row["action_taken"])
        self.assertIsNone(row["result"])

    def test_22_disambiguation_vs_outcome_no_nahi_distinction(self):
        """Context-specific cancellation: 'no'/'nahi' preserves disambiguation but marks not_resolved in awaiting_outcome."""
        shift = self.start_active_shift()
        sid = shift["shift_id"]
        id1 = shift_store.log_issue(sid, "nozzle_clog", "Nozzle clog 1")
        id2 = shift_store.log_issue(sid, "feeder_jam", "Feeder jam 2")

        # 1. In awaiting_disambiguation: 'no'/'nahi' does NOT cancel; keeps prompting
        self.assistant.ask("Issue solve ho gaya", session_id="s22_dis", line_id="line_1")
        sess_dis = self.assistant._session_ctx.get("s22_dis", "line_1")
        self.assertIsNotNone(sess_dis.awaiting_disambiguation)
        dis_prompt = self.assistant.ask("nahi", session_id="s22_dis", line_id="line_1")
        self.assertIn("Please specify which issue to update", dis_prompt)
        sess_dis_after = self.assistant._session_ctx.get("s22_dis", "line_1")
        self.assertIsNotNone(sess_dis_after.awaiting_disambiguation)

        # 2. In awaiting_disambiguation: explicit 'cancel' aborts cleanly
        cancel_dis = self.assistant.ask("cancel", session_id="s22_dis", line_id="line_1")
        self.assertEqual(cancel_dis, "Action cancelled. No changes were made.")
        self.assertIsNone(self.assistant._session_ctx.get("s22_dis", "line_1").awaiting_disambiguation)

        # 3. In awaiting_outcome: 'no'/'nahi' marks result='not_resolved'
        self.assistant.ask(f"Issue {id1} nozzle clean kiya", session_id="s22_out", line_id="line_1")
        outcome_prompt = self.assistant.ask("nahi", session_id="s22_out", line_id="line_1")
        self.assertIn(f"Issue #{id1}", outcome_prompt)
        self.assertIn("status 'not_resolved'", outcome_prompt)
        confirm_resp = self.assistant.ask("yes", session_id="s22_out", line_id="line_1")
        self.assertIn(f"Issue #{id1} updated successfully as not_resolved", confirm_resp)
        row1 = next(i for i in shift_store.list_shift_issues(shift_id=sid) if i["issue_id"] == id1)
        self.assertEqual(row1["result"], "not_resolved")
        self.assertIn("nozzle clean kiya", row1["action_taken"].lower())

        # 4. In awaiting_outcome: explicit 'cancel' aborts without DB change
        self.assistant.ask(f"Issue {id2} feeder check kiya", session_id="s22_out2", line_id="line_1")
        cancel_out = self.assistant.ask("cancel", session_id="s22_out2", line_id="line_1")
        self.assertEqual(cancel_out, "Action cancelled. No changes were made.")
        row2 = next(i for i in shift_store.list_shift_issues(shift_id=sid) if i["issue_id"] == id2)
        self.assertIsNone(row2["result"])
        self.assertIsNone(row2["action_taken"])

    def test_23_what_action_did_i_take_returns_current_operator_action(self):
        """Bug 1: 'What action did I take?' returns the current operator's action from current/most recent issue."""
        shift = self.start_active_shift(op_id="OP001")
        sid = shift["shift_id"]
        iss_id = shift_store.log_issue(sid, "nozzle_clog", "Nozzle clogged with dry mortar")
        shift_store.update_issue(iss_id, action_taken="Flushed nozzle with warm water")

        resp = self.assistant.ask("What action did I take?", session_id="s23", line_id="line_1")
        self.assertNotIn("Which operator's shift issues would you like to check?", resp)
        self.assertIn(f"Issue #{iss_id}", resp)
        self.assertIn("Flushed nozzle with warm water", resp)

    def test_24_what_problem_occurred_during_my_shift_returns_shift_issue(self):
        """Bug 2: 'What problem occurred during my shift?' returns shift_issues, not camera warning."""
        shift = self.start_active_shift(op_id="OP001")
        sid = shift["shift_id"]
        iss_id = shift_store.log_issue(sid, "pump_pressure", "Pump pressure dropped below 2 bar")

        resp = self.assistant.ask("What problem occurred during my shift?", session_id="s24", line_id="line_1")
        self.assertNotIn("Camera stream", resp)
        self.assertIn(f"Issue #{iss_id}", resp)
        self.assertIn("Pump pressure dropped below 2 bar", resp)

    def test_25_operator_shift_history_isolation(self):
        """Operator isolation: Operator B cannot see Operator A's shift issues or actions."""
        shift_a = self.start_active_shift(op_id="OP001", line_id="line_1")
        sid_a = shift_a["shift_id"]
        iss_a = shift_store.log_issue(sid_a, "nozzle_clog", "Anand nozzle blocked")
        shift_store.update_issue(iss_a, action_taken="Anand cleared nozzle")
        shift_store.end_shift(sid_a, status="closed")

        self.create_operator(op_id="OP002", name="Suresh Patel", code="EMP102")
        sid_b = shift_store.start_shift("OP002", "2026-09-22", "Afternoon", "line_1", 5)

        # Suresh queries without having any issues logged on his shift
        resp_prob = self.assistant.ask("What problem occurred during my shift?", session_id="s25", line_id="line_1")
        self.assertNotIn("Anand nozzle blocked", resp_prob)
        self.assertIn("No issues have been recorded", resp_prob)

        resp_act = self.assistant.ask("What action did I take?", session_id="s25", line_id="line_1")
        self.assertNotIn("Anand cleared nozzle", resp_act)
        self.assertIn("No issues have been recorded", resp_act)

    def test_26_stt_material_slope_recognized_as_material_flow_issue(self):
        """Bug 1: STT transcribing 'material flow' as 'material slope' must still create material flow issue."""
        shift = self.start_active_shift(op_id="OP001")
        sid = shift["shift_id"]

        q = "I have a material slope problem."
        intent, terms = qr.classify(q)
        self.assertEqual(intent, qr.Intent.LOG_SHIFT_ISSUE)

        routed = qr.resolve(intent, q)
        self.assertEqual(routed.facts["issue_type"], "material_flow_issue")
        self.assertIsNotNone(routed.facts["rag_recommendation"])
        self.assertIn("Inspect the material supply", routed.facts["rag_recommendation"])
        self.assertIn("printing_issues.md", routed.facts["rag_source"])
        self.assertEqual(routed.facts["rag_verified"], 0)

        # Confirm logging
        prompt = self.assistant.ask(q, session_id="s26", line_id="line_1")
        self.assertIn("Please confirm to proceed", prompt)
        confirm_resp = self.assistant.ask("yes", session_id="s26", line_id="line_1")
        self.assertIn("logged successfully", confirm_resp)

        issues = shift_store.list_shift_issues(shift_id=sid)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["issue_type"], "material_flow_issue")
        self.assertIn("Inspect the material supply", issues[0]["rag_recommendation"])
        self.assertIn("printing_issues.md", issues[0]["rag_source"])
        self.assertEqual(issues[0]["rag_verified"], 0)

    def test_27_combined_action_and_resolution_statement(self):
        """Bug 2: Natural combined statement 'I cleaned the nozzle, the material flow is normal.' updates action and resolves."""
        shift = self.start_active_shift(op_id="OP001")
        sid = shift["shift_id"]
        iss_id = shift_store.log_issue(sid, "material_flow_issue", "I have a material flow problem.")

        # Combined statement without calling RAG again
        resp = self.assistant.ask("I cleaned the nozzle, the material flow is normal.", session_id="s27", line_id="line_1")
        self.assertIn(f"Issue #{iss_id}", resp)
        self.assertIn("action 'I cleaned the nozzle'", resp)
        self.assertIn("status 'resolved'", resp)
        self.assertIn("Please confirm to proceed", resp)

        # Before confirmation, DB unchanged
        pre_row = shift_store.get_issue(iss_id)
        self.assertIsNone(pre_row["action_taken"])
        self.assertIsNone(pre_row["result"])

        # Confirm resolution
        confirm_resp = self.assistant.ask("yes", session_id="s27", line_id="line_1")
        self.assertIn(f"Issue #{iss_id} updated successfully as resolved", confirm_resp)

        # After confirmation, DB has both action_taken and result="resolved"
        post_row = shift_store.get_issue(iss_id)
        self.assertEqual(post_row["action_taken"], "I cleaned the nozzle")
        self.assertEqual(post_row["result"], "resolved")


if __name__ == "__main__":
    unittest.main()

