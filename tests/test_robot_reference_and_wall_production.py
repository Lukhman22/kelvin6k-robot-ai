"""
Automated tests for Kelvin6K Robot Reference Library & Employee Wall Production Tracking.

Verifies:
1. Reference Library catalog (manifest.json) integrity and asset deduplication in SQLite.
2. Troubleshooter / Query Router reference library query handling (e.g. "Show me our robot reference photographs").
3. Image role separation (reference images vs defect annotations).
4. Employee Wall Production data access layer (validation > 0, idempotency, correction audit trail).
5. Natural language query routing for recording, counting, and correcting completed walls.
6. Local provenance classification (never mislabeling local deterministic actions as Gemini).
7. FastAPI REST API endpoints for wall production.
"""

import json
import os
import time
import unittest
from unittest.mock import patch

from agent.db import connect
from agent import incident_store
from agent import shift_store
from agent.query_router import Intent, classify, route, summarise, extract_operator_tuple_from_text, _extract_wall_quantity
from agent.troubleshooter import troubleshoot, TroubleshootResult
from fastapi.testclient import TestClient
from api.server import app


class TestRobotReferenceAndWallProduction(unittest.TestCase):
    """Test suite for Robot Reference Knowledge and Employee Wall Production."""

    @classmethod
    def setUpClass(cls):
        # Ensure database tables and reference library are initialized
        incident_store.init_reference_library()
        with connect() as conn:
            cur = conn.cursor()
            now = time.time()
            cur.execute(
                "INSERT OR IGNORE INTO operators (operator_id, name, employee_code, role, active, created_at) "
                "VALUES ('OP_TEST_999', 'Test Operator', 'OP-999', 'operator', 1, ?)",
                (now,),
            )
            cur.execute(
                "INSERT OR IGNORE INTO operators (operator_id, name, employee_code, role, active, created_at) "
                "VALUES ('OP_HAMADAN_D1B91C', 'Hamadan', '13', 'operator', 1, ?)",
                (now,),
            )

    # =========================================================================
    # PART A: Robot Component Reference Library
    # =========================================================================

    def test_manifest_catalog_integrity(self):
        """Manifest catalog must exist and contain verified reference assets across 3 categories."""
        manifest_path = os.path.join("docs", "knowledge", "robot_parts", "manifest.json")
        self.assertTrue(os.path.isfile(manifest_path), f"Missing {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("library_id", data)
        self.assertIn("roles_and_categories", data)
        self.assertIn("assets", data)

        assets = data["assets"]
        self.assertGreaterEqual(len(assets), 10)

        # Check categories
        categories = {a["image_category"] for a in assets}
        self.assertIn("robot_reference", categories)
        self.assertIn("component_reference", categories)
        self.assertIn("technical_spec_source", categories)

        # Verify key assets exist
        asset_ids = {a["asset_id"] for a in assets}
        self.assertIn("K6K-REF-001", asset_ids)
        self.assertIn("ATOM-SPEC-DETAILS-01", asset_ids)

        # Verify each asset has mandatory metadata
        for asset in assets:
            self.assertTrue(asset["asset_id"])
            self.assertTrue(asset["component_name"])
            self.assertTrue(asset["image_category"])
            self.assertIn("limitations", asset)

    def test_reference_library_ingestion_and_deduplication(self):
        """Reference library initialization must populate DB and deduplicate on repeat runs."""
        # Initial run in setUpClass populated DB; re-running must not raise errors or duplicate
        incident_store.init_reference_library()

        with connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM component_references WHERE reference_id = 'K6K-REF-001'")
            count = cur.fetchone()[0]

        self.assertEqual(count, 1, "Reference K6K-REF-001 must be uniquely deduplicated")

        ref = incident_store.get_component_reference("K6K-REF-001")
        self.assertIsNotNone(ref)
        self.assertEqual(ref["category"], "robot_reference")
        self.assertIn("outdoor_gantry_robot", ref["component_name"])
        self.assertIn("overview only", ref["limitations"].lower())

    def test_troubleshooter_reference_library_query(self):
        """Troubleshooter engine must handle reference photograph queries deterministically without Gemini."""
        queries = [
            "Show me our robot reference photographs",
            "Which reference photograph shows the overall rover configuration?",
            "Show me the feed tube routing reference",
            "What does the printhead mounted look like?",
        ]

        for q in queries:
            ans = troubleshoot(q)
            self.assertFalse(ans.gemini_called, f"Gemini should NOT be called for '{q}'")
            self.assertIsNotNone(ans)
            self.assertTrue(ans.answer)
            self.assertIn("/api/incidents/images/", ans.answer)
            self.assertIn("Limitation:", ans.answer)

    # =========================================================================
    # PART B: Employee Wall Production Tracking Data Layer
    # =========================================================================

    def test_record_wall_production_validation(self):
        """Wall production must reject zero, negative, or invalid quantities."""
        # Reject zero
        with self.assertRaises(ValueError) as ctx:
            shift_store.record_wall_production(operator_id="Hamadan", quantity=0)
        self.assertIn("positive integer", str(ctx.exception).lower())

        # Reject negative
        with self.assertRaises(ValueError) as ctx:
            shift_store.record_wall_production(operator_id="Hamadan", quantity=-3)
        self.assertIn("positive integer", str(ctx.exception).lower())

    def test_record_wall_production_success_and_idempotency(self):
        """Valid wall production record must persist with operator lookup and handle idempotency."""
        req_id = "test-req-wall-101"
        rec1 = shift_store.record_wall_production(
            operator_id="Hamadan",
            quantity=4,
            job_id="TEST-JOB-42",
            notes="Completed northern section",
            request_id=req_id,
        )
        self.assertTrue(rec1["record_id"].startswith("WALL-"))
        self.assertEqual(rec1["operator_name"], "Hamadan")
        self.assertEqual(rec1["quantity"], 4)
        self.assertEqual(rec1["is_corrected"], 0)

        # Idempotent replay: sending same request_id must return existing record
        rec2 = shift_store.record_wall_production(
            operator_id="Hamadan",
            quantity=4,
            job_id="TEST-JOB-42",
            request_id=req_id,
        )
        self.assertEqual(rec1["record_id"], rec2["record_id"])

    def test_get_wall_production_totals(self):
        """Totals query must compute aggregate wall count and operator breakdown correctly."""
        # Record known quantities for Test Operator (OP-999)
        shift_store.record_wall_production(operator_id="Test Operator", quantity=2, job_id="JOB-T1")
        shift_store.record_wall_production(operator_id="Test Operator", quantity=3, job_id="JOB-T1")

        totals = shift_store.get_wall_production_total(operator_id="Test Operator", scope="today")
        self.assertGreaterEqual(totals["total_quantity"], 5)
        self.assertGreaterEqual(totals["record_count"], 2)

        op_names = [b["operator_name"] for b in totals["by_operator"]]
        self.assertIn("Test Operator", op_names)

    def test_correct_wall_production_audit_trail(self):
        """Correcting a record must mark original is_corrected=1, log audit event, and update totals without inflation."""
        # 1. Create original record with 4 walls
        orig = shift_store.record_wall_production(
            operator_id="Test Operator",
            quantity=4,
            job_id="JOB-AUDIT-01",
        )
        orig_id = orig["record_id"]

        # 2. Correct the record from 4 to 3 walls
        correction_result = shift_store.correct_wall_production(
            original_record_id=orig_id,
            corrected_quantity=3,
            reason="Counting mistake during shift",
            corrected_by="supervisor",
        )

        self.assertEqual(correction_result["status"], "corrected")
        self.assertEqual(correction_result["original_quantity"], 4)
        new_rec = correction_result["replacement_record"]
        self.assertEqual(new_rec["quantity"], 3)

        # 3. Verify original record in DB is flagged as is_corrected = 1
        with connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT is_corrected, quantity FROM wall_production_records WHERE record_id = ?", (orig_id,))
            row = cur.fetchone()
            self.assertEqual(row[0], 1, "Original record must be marked corrected")
            self.assertEqual(row[1], 4, "Original quantity must remain 4 for audit trail")

            # 4. Verify event_corrections entry exists
            cur.execute(
                "SELECT entity_id, old_value, new_value, reason FROM event_corrections WHERE entity_id = ?",
                (orig_id,),
            )
            audit_row = cur.fetchone()

        self.assertIsNotNone(audit_row, "Correction event must be logged in event_corrections")
        self.assertEqual(audit_row[0], orig_id)
        self.assertIn('"quantity": 4', audit_row[1])
        self.assertIn('"quantity": 3', audit_row[2])
        self.assertEqual(audit_row[3], "Counting mistake during shift")

    # =========================================================================
    # PART C: Natural Language Query Routing & Intent Classification
    # =========================================================================

    def test_natural_language_wall_production_classification(self):
        """Router must accurately classify natural language recording, count, and correction commands."""
        # Recording commands
        rec_phrases = [
            "Hamadan completed 4 walls; save it",
            "Log 2 completed walls for operator Hamadan",
            "Record 3 walls printed by operator code 13",
            "Save 5 walls completed by Anand Sharma",
            "Operator Hamadan built 2 walls",
        ]
        for p in rec_phrases:
            intent, _ = classify(p)
            self.assertEqual(
                intent,
                Intent.RECORD_WALLS_COMPLETED,
                f"Phrase '{p}' should classify as RECORD_WALLS_COMPLETED, got {intent}",
            )

        # Count queries
        count_phrases = [
            "How many walls did Hamadan print today?",
            "Total walls completed this week",
            "How many walls printed in total?",
            "Show wall production for today",
            "How many walls did operator 13 complete?",
        ]
        for p in count_phrases:
            intent, _ = classify(p)
            self.assertEqual(
                intent,
                Intent.GET_WALL_PRODUCTION,
                f"Phrase '{p}' should classify as GET_WALL_PRODUCTION, got {intent}",
            )

        # Correction commands
        correction_phrases = [
            "Correct the previous record: it should be 3 walls, not 4",
            "Correction: Hamadan printed 3 walls, not 4",
            "Change last wall entry: it was 2 walls not 5",
        ]
        for p in correction_phrases:
            intent, _ = classify(p)
            self.assertEqual(
                intent,
                Intent.CORRECT_WALL_PRODUCTION,
                f"Phrase '{p}' should classify as CORRECT_WALL_PRODUCTION, got {intent}",
            )

    def test_natural_language_wall_production_resolution(self):
        """Routing and resolving natural language commands must update DB and produce clear answers."""
        # Execute recording via route()
        routed = route("Hamadan completed 2 walls; save it")
        self.assertEqual(routed.intent, Intent.RECORD_WALLS_COMPLETED)
        self.assertTrue(routed.available)
        summary = summarise(routed)
        self.assertIn("Hamadan", summary)
        self.assertIn("2", summary)

        # Execute query via route()
        routed_query = route("How many walls did Hamadan print today?")
        self.assertEqual(routed_query.intent, Intent.GET_WALL_PRODUCTION)
        self.assertTrue(routed_query.available)
        summary_q = summarise(routed_query)
        self.assertIn("Hamadan", summary_q)
        self.assertIn("wall", summary_q.lower())

    def test_employee_code_extraction(self):
        """Operator extractor must support codes like 'operator code 13' or 'OP-101'."""
        op_tuple = extract_operator_tuple_from_text("Record 3 walls printed by operator code 13")
        self.assertIsNotNone(op_tuple)
        self.assertEqual(op_tuple[1], "Hamadan")

    def test_quantity_extraction(self):
        """Wall quantity extractor must correctly parse numeric and word numbers."""
        self.assertEqual(_extract_wall_quantity("Hamadan completed 4 walls; save it"), 4)
        self.assertEqual(_extract_wall_quantity("Log two completed walls"), 2)
        self.assertEqual(_extract_wall_quantity("Record 12 walls printed"), 12)

    # =========================================================================
    # PART D: FastAPI REST API Endpoints
    # =========================================================================

    def test_rest_api_wall_production(self):
        """FastAPI REST API must handle validation, recording, queries, and audit corrections."""
        with TestClient(app) as client:
            # 1. Validation: reject zero or negative quantity
            bad_res = client.post("/api/v1/production/walls", json={"operator_id": "Hamadan", "quantity": 0})
            self.assertEqual(bad_res.status_code, 400)

            bad_res2 = client.post("/api/v1/production/walls", json={"operator_id": "Hamadan", "quantity": -2})
            self.assertEqual(bad_res2.status_code, 400)

            # 2. Record valid walls
            rec_res = client.post(
                "/api/v1/production/walls",
                json={
                    "operator_id": "Hamadan",
                    "quantity": 3,
                    "job_id": "API-TEST-JOB-1",
                    "notes": "Testing REST API",
                },
            )
            self.assertEqual(rec_res.status_code, 200)
            data = rec_res.json()
            self.assertTrue(data["success"])
            rec_id = data["record"]["record_id"]
            self.assertEqual(data["record"]["quantity"], 3)
            self.assertEqual(data["record"]["operator_name"], "Hamadan")

            # 3. Query wall production
            get_res = client.get("/api/v1/production/walls?scope=today")
            self.assertEqual(get_res.status_code, 200)
            totals = get_res.json()
            self.assertTrue(totals["success"])
            self.assertGreaterEqual(totals["total_quantity"], 3)
            self.assertIn("records", totals)

            # 4. Correct the record
            corr_res = client.post(
                "/api/v1/production/walls/correct",
                json={
                    "record_id": rec_id,
                    "corrected_quantity": 2,
                    "reason": "REST API audit test correction",
                },
            )
            self.assertEqual(corr_res.status_code, 200)
            corr_data = corr_res.json()
            self.assertTrue(corr_data["success"])
            self.assertEqual(corr_data["status"], "corrected")
            self.assertEqual(corr_data["replacement_record"]["quantity"], 2)

    def test_rest_api_reference_library(self):
        """FastAPI REST API must list verified reference assets and individual reference details."""
        with TestClient(app) as client:
            list_res = client.get("/api/incidents/references")
            self.assertEqual(list_res.status_code, 200)
            refs = list_res.json()
            self.assertGreaterEqual(len(refs), 10)

            ref_res = client.get("/api/incidents/references/K6K-REF-001")
            self.assertEqual(ref_res.status_code, 200)
            ref_data = ref_res.json()
            self.assertEqual(ref_data["reference_id"], "K6K-REF-001")
            self.assertEqual(ref_data["category"], "robot_reference")
            self.assertIn("limitations", ref_data)


if __name__ == "__main__":
    unittest.main()
