"""
Regression test suite for Kelvin6K Operator Knowledge RAG Ingestion & Retrieval.

Validates:
1. Retrieval accuracy across all 14 operator knowledge issues:
   - K6K-CLEAN-001: Pump and hose cleaning intervals (every 3-4 hours, after each wall, 1-hour idle).
   - K6K-HOSE-001: Delivery hose routing and positioning.
   - K6K-FLOW-001: Material flow drop; safety review for hammering pump & pulsing feedrate.
   - K6K-SPEED-001: Feedrate guidance (straight 100/150/170, curved 100-120).
   - K6K-LAYER-001: Manual layer thickness (40-45 mm vs 40-50 mm conflicts).
   - K6K-CURE-001: Green strength / wall slumping thresholds (3-4 layers vs 5th vs 10-12 layers).
   - K6K-PAUSE-001: Pause-time material stagnation & safety gate (5 min hopper vs 15-20 min set).
   - K6K-VISION-001: Camera false stops / lighting observations.
   - K6K-ENV-001: Historical environmental and machine parameter notebook.
   - K6-FAULT-E136.1: Absolute encoder cable damage.
   - K6-FAULT-E137.1: Servo/robot position mismatch; encoder reset safety gate.
   - K6-Z-001: Z-axis downward motion blocked; H0D11 parameter safety gate.
   - K6-QUALITY-001: Early layer accelerator check / 50% hardness definition.
   - K6-SAFETY-001: Prevent wall contact and collision during printing.
2. English and Hinglish query retrieval parity.
3. Safety gates: no unapproved workarounds recommended as standard procedures.
4. Preserving conflicting values and flagging them rather than picking one arbitrarily.
5. Distinguishing possible causes from confirmed root causes.
6. Explicit uncertainty reporting for missing/unsupported documentation.
7. Deduplication: exact 14 chunks, 1 per issue, preserving existing index chunks.
"""

import unittest

from agent.rag.retriever import get_store, index_summary, search_documentation
from agent.spoken import parse_record, speak_record


class TestOperatorRAGPipeline(unittest.TestCase):
    """Validation of operator knowledge RAG ingestion, retrieval, and safety grounding."""

    @classmethod
    def setUpClass(cls):
        cls.store = get_store(refresh=True)

    def test_deduplication_and_chunk_integrity(self):
        """Verifies exact 14 chunks for operator knowledge, one per issue, no duplicates."""
        summary = index_summary()
        self.assertIn("kelvin6k_operator_knowledge.jsonl", summary["documents"])
        self.assertEqual(summary["documents"]["kelvin6k_operator_knowledge.jsonl"], 14)

        op_chunks = [c for c in self.store.chunks if c.filename == "kelvin6k_operator_knowledge.jsonl"]
        self.assertEqual(len(op_chunks), 14)

        expected_issues = {
            "K6K-CLEAN-001", "K6K-HOSE-001", "K6K-FLOW-001", "K6K-SPEED-001",
            "K6K-LAYER-001", "K6K-CURE-001", "K6K-PAUSE-001", "K6K-VISION-001",
            "K6K-ENV-001", "K6-FAULT-E136.1", "K6-FAULT-E137.1", "K6-Z-001",
            "K6-QUALITY-001", "K6-SAFETY-001"
        }
        found_issues = set()
        for chunk in op_chunks:
            for iid in expected_issues:
                if f"Issue: {iid}" in chunk.text:
                    found_issues.add(iid)
        self.assertEqual(found_issues, expected_issues, "All 14 issue IDs must be present and distinct.")

    def test_cleaning_intervals_retrieval_and_conflicts(self):
        """K6K-CLEAN-001: cleaning interval queries in English and Hinglish."""
        queries = [
            "How frequently should the pump and hose be cleaned?",
            "Pump aur hose ki cleaning kitni der mein karni chahiye?",
            "What is the cleaning interval for pump and delivery hose?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-CLEAN-001", top["text"])
            self.assertIn("3 to 4 hours", top["text"])
            self.assertIn("one-hour", top["text"].lower())
            # Conflicting values must be flagged
            self.assertIn("CONFLICT", top["text"])
            spoken = speak_record(top["text"], q)
            self.assertIn("conflicting", spoken.lower())

    def test_delivery_hose_positioning_retrieval(self):
        """K6K-HOSE-001: delivery hose positioning queries in English and Hinglish."""
        queries = [
            "How should the delivery hose between pump and hopper be positioned?",
            "Delivery hose ko kaise lagana chahiye?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-HOSE-001", top["text"])
            self.assertIn("bend", top["text"].lower())

    def test_material_flow_drop_and_safety_gate(self):
        """K6K-FLOW-001: flow interruption, unapproved workaround, safety review gate."""
        queries = [
            "What should an operator do when concrete material flow suddenly drops or stops?",
            "Material flow achanak kam ho gaya, kya check karna chahiye?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-FLOW-001", top["text"])
            # Workaround must NOT be approved
            self.assertIn("UNAPPROVED WORKAROUND", top["text"])
            self.assertIn("SAFETY REVIEW REQUIRED", top["text"])
            spoken = speak_record(top["text"], q)
            self.assertIn("unapproved workaround", spoken.lower())
            self.assertIn("not be recommended", spoken.lower())

        # Why query distinguishes possible cause from confirmed cause
        why_res = search_documentation("Why did concrete material flow stop?")
        top_why = why_res["results"][0]
        spoken_why = speak_record(top_why["text"], "Why did concrete material flow stop?")
        self.assertIn("possible cause", spoken_why.lower())
        self.assertIn("not confirmed", spoken_why.lower())

    def test_feedrate_straight_vs_curved_guidance(self):
        """K6K-SPEED-001: straight wall (100/150/170) vs curved path (100-120)."""
        queries = [
            "What feedrates were reported for straight walls and complex curved paths?",
            "Curved wall ke liye operator ne kaunsa feedrate range report kiya?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-SPEED-001", top["text"])
            self.assertIn("150", top["text"])
            self.assertIn("100–120", top["text"])
            self.assertIn("jerking", top["text"].lower())
            self.assertIn("ENGINEERING APPROVAL", top["text"])

    def test_layer_thickness_measurement_and_conflicts(self):
        """K6K-LAYER-001: 40-45 mm vs 40-50 mm conflicts and tape measure check."""
        queries = [
            "How is layer bead thickness measured and corrected?",
            "Layer 45 mm se zyada ho raha hai source mein kya observation hai?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-LAYER-001", top["text"])
            self.assertIn("40–45 mm", top["text"])
            self.assertIn("tape measure", top["text"].lower())
            self.assertIn("CONFLICT", top["text"])

    def test_green_strength_slumping_thresholds(self):
        """K6K-CURE-001: green strength 3-4 layers vs fifth layer vs 10-12 layers."""
        queries = [
            "How quickly must lower concrete layers harden to support subsequent layers?",
            "Lower layers ko kitni der mein hard hona chahiye?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-CURE-001", top["text"])
            self.assertIn("green strength", top["text"].lower())
            self.assertIn("CONFLICT", top["text"])
            spoken = speak_record(top["text"], q)
            self.assertIn("conflicting", spoken.lower())

    def test_pause_material_stagnation_safety_gate(self):
        """K6K-PAUSE-001: 5 min hopper vs 15-20 min set, no autonomous extrusion."""
        queries = [
            "What happens if printing is paused while accelerator mixed material remains in the hopper?",
            "Pause ke dauran hopper mein material hard hone ke baare mein kya report hai?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-PAUSE-001", top["text"])
            self.assertIn("5 minutes", top["text"])
            self.assertIn("15–20 minutes", top["text"])
            self.assertIn("CONFLICT", top["text"])
            self.assertIn("SAFETY REVIEW REQUIRED", top["text"])

    def test_camera_false_stops_lighting_observation(self):
        """K6K-VISION-001: shadows and lighting observations."""
        queries = [
            "Have shadows or shop lighting caused the camera system to stop the machine incorrectly?",
            "Kya camera lighting ki wajah se false stop report hua tha?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-VISION-001", top["text"])
            self.assertIn("OBSERVATION ONLY", top["text"])
            spoken = speak_record(top["text"], q)
            self.assertIn("observation", spoken.lower())

    def test_environmental_notebook_data_source(self):
        """K6K-ENV-001: 10-15 days temperature, humidity, pump speed notebook."""
        queries = [
            "Is there historical environmental and process data that could support analysis?",
            "Operator notebook mein kya environmental data note kiya gaya hai?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6K-ENV-001", top["text"])
            self.assertIn("notebook", top["text"].lower())
            self.assertIn("temperature", top["text"].lower())

    def test_fault_code_e136_1_encoder_cable(self):
        """K6-FAULT-E136.1: absolute encoder cable damage and repair."""
        queries = [
            "What was reported as the cause of error E136.1?",
            "Error code E136.1 alarm aane par kya check karna hai?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6-FAULT-E136.1", top["text"])
            self.assertIn("encoder cable", top["text"].lower())
            self.assertIn("POSSIBLE - NOT CONFIRMED", top["text"])

    def test_fault_code_e137_1_safety_gate(self):
        """K6-FAULT-E137.1: servo position mismatch, encoder reset safety gate."""
        queries = [
            "What was reported as the cause of error E137.1?",
            "E137.1 ka reported explanation kya hai, aur kya verify karna baaki hai?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6-FAULT-E137.1", top["text"])
            self.assertIn("position mismatch", top["text"].lower())
            self.assertIn("REQUIRES AUTHORIZATION", top["text"])
            spoken = speak_record(top["text"], q)
            self.assertIn("authorization", spoken.lower())

    def test_z_axis_h0d11_safety_gate(self):
        """K6-Z-001: Z-axis limit, H0D11 parameter modification safety gate."""
        queries = [
            "What was reported when the Z-axis could not move downward?",
            "H0D11 ke baare mein notes kya kehte hain? Kya yeh approved procedure hai?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6-Z-001", top["text"])
            self.assertIn("H0D11", top["text"])
            self.assertIn("REQUIRES AUTHORIZATION", top["text"])
            spoken = speak_record(top["text"], q)
            self.assertIn("authorization", spoken.lower())

    def test_early_layer_accelerator_check(self):
        """K6-QUALITY-001: early-layer accelerator check and 50% hardness definition."""
        queries = [
            "What should be checked during the first few layers if concrete is not hardening sufficiently?",
            "Pehle layers mein concrete hard nahi ho raha toh kya check karein?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6-QUALITY-001", top["text"])
            self.assertIn("50% hardness", top["text"])
            self.assertIn("NEEDS VALIDATION", top["text"])

    def test_printing_safety_wall_collision(self):
        """K6-SAFETY-001: prevent touching wall and robot collision."""
        queries = [
            "What precautions were reported to reduce wall damage or robot collision?",
            "Printing ke time wall ko touch karna ya collision se bachne ke liye kya safety rules hain?",
        ]
        for q in queries:
            res = search_documentation(q)
            self.assertTrue(res["found"], f"Expected results for query: {q}")
            top = res["results"][0]
            self.assertIn("K6-SAFETY-001", top["text"])
            self.assertIn("touch the wall", top["text"].lower())

    def test_unsupported_query_returns_explicit_uncertainty(self):
        """Questions outside documented facts produce explicit uncertainty responses."""
        unsupported_queries = [
            "What is the gear ratio of the planetary gearbox on the concrete pump?",
            "What is the Wi-Fi password for the robot control cabin?",
            "Kaunsa hydraulic oil brand recommend kiya gaya hai pump gearbox ke liye?",
        ]
        for q in unsupported_queries:
            res = search_documentation(q)
            self.assertFalse(res["found"], f"Expected found=False for unsupported query: {q}")
            self.assertEqual(len(res["results"]), 0)
            self.assertIn("not found in the available documentation", res["message"])

    def test_cause_distinction_and_citation_formatting(self):
        """Verifies citation includes file name, page, and issue heading."""
        res = search_documentation("Why did error E136.1 occur?")
        self.assertTrue(res["found"])
        top = res["results"][0]
        self.assertIn("kelvin6k_operator_knowledge.jsonl", top["source"])
        self.assertIn("page 10", top["source"])
        self.assertIn("K6-FAULT-E136.1", top["source"])

        spoken = speak_record(top["text"], "Why did error E136.1 occur?")
        self.assertIn("possible cause", spoken.lower())
        self.assertIn("not confirmed", spoken.lower())

    def test_query_router_operator_integration(self):
        """Verifies query router dispatches operator queries to KNOWLEDGE_QUERY with grounded answers."""
        from agent import query_router as qr

        queries = [
            ("Pump aur hose ki cleaning kitni der mein karni chahiye?", "3 to 4 hours"),
            ("Material flow achanak kam ho gaya, kya check karna chahiye?", "unapproved"),
            ("Curved wall ke liye operator ne kaunsa feedrate range report kiya?", "100–120"),
            ("Layer 45 mm se zyada ho raha hai; source mein kya observation hai?", "40–45 mm"),
            ("Pause ke dauran hopper mein material hard hone ke baare mein kya report hai?", "5 minutes"),
            ("What was reported as the cause of error E136.1?", "encoder cable"),
            ("E137.1 ka reported explanation kya hai, aur kya verify karna baaki hai?", "mismatch"),
            ("H0D11 ke baare mein notes kya kehte hain? Kya yeh approved procedure hai?", "H0D11"),
            ("Kya camera lighting ki wajah se false stop report hua tha?", "observation"),
        ]
        for q, expected_content in queries:
            res = qr.route(q)
            self.assertEqual(res.intent, qr.Intent.KNOWLEDGE_QUERY, f"Query {q} did not route to KNOWLEDGE_QUERY")
            self.assertTrue(res.available, f"Query {q} was marked unavailable")
            summary = qr.summarise(res)
            self.assertIn(expected_content.lower(), summary.lower(), f"Expected {expected_content} in summary for {q}")

    def test_zero_regression_attendance_and_live_intents(self):
        """Ensures existing attendance, shift, and live queries route to their respective functions without regression."""
        from agent import query_router as qr

        # Attendance query
        att_res = qr.route("Hamadan ki shift shuru hui kya?")
        self.assertIn(att_res.intent, {qr.Intent.GET_ATTENDANCE_STATUS, qr.Intent.GET_ACTIVE_SHIFT, qr.Intent.GET_SHIFT_BY_OPERATOR})

        # Clock-in / Shift-start write intent
        start_res = qr.route("Start Hamadan shift")
        self.assertEqual(start_res.intent, qr.Intent.SHIFT_START)

        # Clock-out / Shift-end write intent
        end_res = qr.route("End Hamadan shift")
        self.assertEqual(end_res.intent, qr.Intent.SHIFT_END)

        # Live print status query
        status_res = qr.route("Printing chal rahi hai?")
        self.assertEqual(status_res.intent, qr.Intent.GET_MACHINE_STATUS)


if __name__ == "__main__":
    unittest.main()
