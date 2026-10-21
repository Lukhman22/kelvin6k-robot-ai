"""
Comprehensive test suite for Kelvin6K Visual Troubleshooting & Robot-Part Reference Images.

Validates Manager Requirements:
1. When operator uploads an image of a defect, Kelvin generates an annotated image highlighting
   the visible defect with an industrial bounding box, label pill tag, and AI disclaimer banner.
2. Defect localization integrity: When a defect cannot be reliably isolated (e.g. smooth bead or
   low-contrast image), NO arbitrary bounding box is fabricated; the original image is displayed
   with an honest limitation notice.
3. Relevant real robot-part images: Persistent component reference library provides genuine Kelvin6K
   photographs/schematics (nozzle, hose, pump, hopper, accelerator, mixer, motor, servo drive,
   wiring, layers) with component name, source ID, machine version (Kelvin6K-V2), verification status,
   and relevance explanation.
4. Honest fallback: If no reference image exists for a problem, states so honestly without inventing
   hardware photos or placeholders.
5. RAG-first, Gemini-second: Historical incidents and documentation are retrieved first; stored solutions
   and photographs are returned without calling Gemini (gemini_called == False).
6. 6-Section structured response format maintained across all flows:
   - ### 1. What I can see
   - ### 2. Relevant historical cases (### Historical record found)
   - ### 3. Most plausible explanations
   - ### 4. Recommended next checks
   - ### 5. Solution
   - ### 6. Confidence and limitations
7. Persistence: Stored images, annotations, and reference components survive backend restarts.
8. FastAPI endpoints: GET /api/incidents/references and GET /api/incidents/references/{reference_id}.
"""

import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from agent import config, db, incident_store
from agent.troubleshooter import (
    Attachment,
    TroubleshootResult,
    detect_and_create_annotation,
    troubleshoot,
)


def _make_test_image(defect_type: str = "gap") -> bytes:
    """Creates synthetic concrete printing test frames."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 205  # light gray bed
    if defect_type == "gap":
        # Two separated horizontal beads with an empty gap
        cv2.rectangle(img, (60, 200), (260, 280), (85, 85, 85), -1)
        cv2.rectangle(img, (380, 200), (580, 280), (85, 85, 85), -1)
    elif defect_type == "corner":
        # L-shaped corner with bulging bead contour
        pts = np.array(
            [[100, 100], [420, 100], [440, 150], [420, 360], [330, 360], [330, 190], [100, 190]],
            np.int32,
        )
        cv2.fillPoly(img, [pts], (85, 85, 85))
    elif defect_type == "tearing":
        # Jagged/torn bead contour
        pts = []
        for x in range(50, 590, 12):
            y_top = 210 + (16 if (x // 12) % 2 == 0 else -16)
            pts.append([x, y_top])
        for x in range(578, 40, -12):
            y_bot = 290 + (16 if (x // 12) % 2 == 0 else -16)
            pts.append([x, y_bot])
        cv2.fillPoly(img, [np.array(pts, np.int32)], (85, 85, 85))
    elif defect_type == "uniform":
        # Flat solid color: no features, low contrast
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    else:
        # Uniform smooth continuous bead
        cv2.rectangle(img, (60, 210), (580, 290), (85, 85, 85), -1)

    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()


class TestVisualTroubleshootingAndReferenceImages(unittest.TestCase):
    """Full test suite for visual troubleshooting and reference image retrieval."""

    @classmethod
    def setUpClass(cls):
        # Initialize component reference library
        incident_store.init_reference_library()

    def test_01_reference_library_seeding_and_integrity(self):
        """Verifies that all 10 genuine Kelvin6K machine components are seeded with full specifications."""
        refs = incident_store.list_component_references()
        self.assertGreaterEqual(len(refs), 10, "Expected at least 10 seeded component references.")

        required_components = {
            "nozzle",
            "delivery_hose",
            "pump",
            "hopper",
            "accelerator_assembly",
            "e_mixer",
            "e_motor",
            "servo_drive",
            "ethercat_wiring",
            "concrete_layers",
        }
        found_names = {r["component_name"] for r in refs}
        self.assertTrue(
            required_components.issubset(found_names),
            f"Missing required components. Found: {found_names}",
        )

        # Inspect Print Nozzle Assembly metadata
        nozzle = incident_store.get_component_reference("REF-NOZZLE-01")
        self.assertIsNotNone(nozzle)
        self.assertEqual(nozzle["machine_version"], "Kelvin6K-V2")
        self.assertEqual(nozzle["verification_status"], "verified")
        self.assertEqual(nozzle["source_type"], "engineering_manual")
        self.assertIn("nozzle", nozzle["keywords"])
        self.assertIn("orifice", nozzle["keywords"])
        self.assertIn("gap", nozzle["keywords"])
        self.assertTrue(bool(nozzle["relevance_explanation"]))
        self.assertTrue(bool(nozzle["image_id"]))

        # Verify image bytes can be loaded from disk
        img_res = incident_store.get_image_data(nozzle["image_id"])
        self.assertIsNotNone(img_res)
        img_bytes, mime, fn = img_res
        self.assertGreater(len(img_bytes), 1000)
        self.assertIn(mime, ("image/jpeg", "image/png"))

    def test_02_detect_and_create_annotation_layer_gap(self):
        """Tests that a layer gap defect is accurately localized with bounding box and disclaimer banner."""
        img_bytes = _make_test_image("gap")
        att = Attachment(
            filename="defect_gap.jpg",
            content_type="image/jpeg",
            size_bytes=len(img_bytes),
            data=img_bytes,
            kind="image",
            width=640,
            height=480,
        )

        ann = detect_and_create_annotation(attachment=att, primary_image_id="test_parent_01")
        self.assertIsNotNone(ann, "Expected defect annotation to be generated for gap image.")
        self.assertIn("annotated_image_id", ann)
        self.assertEqual(ann["defect_type"], "Layer Gap / Bead Discontinuity")
        self.assertEqual(ann["primary_image_id"], "test_parent_01")

        # Verify bounding box [x, y, w, h]
        bbox = ann["bbox"]
        self.assertEqual(len(bbox), 4)
        x, y, w, h = bbox
        self.assertGreater(w, 10, "Bounding box width should be non-trivial.")
        self.assertGreater(h, 10, "Bounding box height should be non-trivial.")

        # Load annotated image from store and verify size
        saved = incident_store.get_image_data(ann["annotated_image_id"])
        self.assertIsNotNone(saved)
        saved_bytes, mime, fn = saved
        self.assertGreater(len(saved_bytes), 1000)

    def test_03_detect_and_create_annotation_corner_bulge(self):
        """Tests that corner deformation is localized with appropriate defect type."""
        img_bytes = _make_test_image("corner")
        att = Attachment(
            filename="corner_bulge.jpg",
            content_type="image/jpeg",
            size_bytes=len(img_bytes),
            data=img_bytes,
            kind="image",
            width=640,
            height=480,
        )

        ann = detect_and_create_annotation(attachment=att, primary_image_id="test_parent_02")
        self.assertIsNotNone(ann)
        self.assertEqual(ann["defect_type"], "Layer Deformation / Corner Bulge")
        self.assertGreaterEqual(ann["confidence"], 0.70)

    def test_04_no_annotation_fabricated_for_uniform_or_normal_image(self):
        """CRITICAL: Verifies NO arbitrary bounding box is fabricated when defect cannot be localized."""
        # 1. Flat low-contrast image
        flat_bytes = _make_test_image("uniform")
        att_flat = Attachment(
            filename="flat_gray.jpg",
            content_type="image/jpeg",
            size_bytes=len(flat_bytes),
            data=flat_bytes,
            kind="image",
        )
        ann_flat = detect_and_create_annotation(attachment=att_flat)
        self.assertIsNone(ann_flat, "Must return None for flat uniform image (never fabricate markers).")

        # 2. End-to-end troubleshoot with smooth bead
        normal_bytes = _make_test_image("normal")
        att_normal = Attachment(
            filename="normal_bead.jpg",
            content_type="image/jpeg",
            size_bytes=len(normal_bytes),
            data=normal_bytes,
            kind="image",
        )
        mock_obs = "Visual inspection of 640x480px concrete print capture. Single continuous bead profile visible within frame."
        with patch("agent.troubleshooter.extract_visual_observations", return_value=mock_obs):
            res = troubleshoot("find the issue in this and tell me the solution", attachments=[att_normal])
        # Annotated image ID should be None
        self.assertIsNone(res.annotated_image_id)
        # Should display original image with honest localization limitation note
        self.assertIn("Defect localization could not be reliably isolated with high confidence", res.answer)
        self.assertIn("displaying original uploaded image without artificial markers", res.answer)

    def test_05_search_component_references_matching(self):
        """Tests that queries retrieve the correct genuine Kelvin6K machine component reference."""
        # Query 1: Gap / void -> Nozzle Assembly
        ref_nozzle = incident_store.search_component_references("Why is there a layer gap and void in my print?")
        self.assertIsNotNone(ref_nozzle)
        self.assertEqual(ref_nozzle["component_name"], "nozzle")
        self.assertIn("REF-NOZZLE-01", ref_nozzle["reference_id"])

        # Query 2: Hose / pressure / delivery line -> Delivery Hose
        ref_hose = incident_store.search_component_references("Material delivery hose blockage and line resistance")
        self.assertIsNotNone(ref_hose)
        self.assertEqual(ref_hose["component_name"], "delivery_hose")
        self.assertIn("REF-HOSE-01", ref_hose["reference_id"])

        # Query 3: Accelerator pipe clogging -> Accelerator Assembly
        ref_accel = incident_store.search_component_references("Premature setting and chemical accelerator dosing assembly injector blockage")
        self.assertIsNotNone(ref_accel)
        self.assertEqual(ref_accel["component_name"], "accelerator_assembly")
        self.assertIn("REF-ACCEL-01", ref_accel["reference_id"])

        # Query 4: Motor overheating -> Extruder Motor
        ref_motor = incident_store.search_component_references("Extruder motor temperature overheating alarm")
        self.assertIsNotNone(ref_motor)
        self.assertEqual(ref_motor["component_name"], "e_motor")
        self.assertIn("REF-EMOTOR-01", ref_motor["reference_id"])

    def test_06_honest_fallback_when_no_reference_image_exists(self):
        """CRITICAL: Tests that when no reference image exists, Kelvin honestly states so without fabricating photos."""
        # Non-machine query
        ref_none = incident_store.search_component_references("analyze this unusual pattern")
        self.assertIsNone(ref_none, "Should return None when query does not match any machine component.")

        # Test formatting output when reference_info is None
        from agent.troubleshooter import _format_rag_answer
        hits = [{"source": "Test Manual", "text": "Problem: Sample.\nCause: Sample.\nSolution: Sample."}]
        ans, _ = _format_rag_answer(
            question="sample query",
            hits=hits,
            attachments=[],
            reference_info=None,
        )
        self.assertIn("No verified reference image of this component is currently available", ans)

    def test_07_rag_first_workflow_with_visual_response(self):
        """Tests end-to-end RAG-first troubleshooting returns annotated image, reference image, and 6 sections without Gemini."""
        img_bytes = _make_test_image("gap")
        att = Attachment(
            filename="layer_gap_defect.jpg",
            content_type="image/jpeg",
            size_bytes=len(img_bytes),
            data=img_bytes,
            kind="image",
            width=640,
            height=480,
        )

        mock_obs = (
            "Visual inspection of 640x480px concrete print capture. "
            "Visible bead discontinuity with local gaps or voids between concrete layers. "
            "Note: Image quality and perspective limit analysis to visible external geometry; "
            "internal pump line pressure cannot be determined from images alone."
        )

        with patch("agent.troubleshooter.extract_visual_observations", return_value=mock_obs):
            res = troubleshoot("find the issue in this and tell me the solution", attachments=[att])

        # RAG-first decision flow
        self.assertTrue(res.rag_found)
        self.assertEqual(res.source, "rag-knowledge")
        self.assertFalse(res.gemini_called, "Must NOT call Gemini when RAG hit is sufficient.")

        # Visual response fields
        self.assertIsNotNone(res.primary_image_id)
        self.assertIsNotNone(res.annotated_image_id)
        self.assertIsNotNone(res.reference_image_id)
        self.assertIsNotNone(res.reference_component)

        # 6 Structured sections validation
        self.assertIn("### 1. What I can see", res.answer)
        self.assertIn("AI-Assisted Visual Observation: Layer Gap", res.answer)
        self.assertIn("AI-assisted visual observation highlighting visible defect", res.answer)
        self.assertIn("View Original Upload", res.answer)

        self.assertIn("### 2. Relevant historical cases", res.answer)
        self.assertIn("### Historical record found", res.answer)
        self.assertIn("Relevant component reference image", res.answer)
        self.assertIn("Kelvin6K-V2", res.answer)
        self.assertIn("Verification Status:", res.answer)

        self.assertIn("### 3. Most plausible explanations", res.answer)
        self.assertIn("### 4. Recommended next checks", res.answer)
        self.assertIn("### 5. Solution", res.answer)
        self.assertIn("### 6. Confidence and limitations", res.answer)

    def test_08_historical_incident_prioritized_with_both_images(self):
        """Tests that when a verified incident is matched, its stored photograph and reference image are returned without Gemini."""
        # Create a verified historical incident with an image
        inc_img_bytes = _make_test_image("gap")
        saved_img = incident_store.save_incident_image(
            data=inc_img_bytes,
            original_name="past_gap_repair.jpg",
            content_type="image/jpeg",
        )
        inc = incident_store.create_incident(
            user_question="Historical layer gap incident",
            symptom_description="Severe bead gap and void on outer perimeter wall",
            defect_category="layer_gap",
            primary_image_id=saved_img["image_id"],
            resolution_status="verified",
            engineering_review_status="approved",
            confirmed_root_cause="Concrete aggregate oversized particle caught in 25mm nozzle opening.",
            recommended_action="Safely stop pump, depressurize delivery line, remove nozzle tip, wash in clean water, and verify orifice.",
            verified_by="Lead Field Engineer Marcus Vance",
        )

        try:
            test_att = Attachment(
                filename="current_gap.jpg",
                content_type="image/jpeg",
                size_bytes=len(inc_img_bytes),
                data=inc_img_bytes,
                kind="image",
            )

            mock_obs = (
                "Visual inspection of 640x480px concrete print capture. "
                "Visible bead discontinuity with local gaps or voids between concrete layers."
            )

            with patch("agent.troubleshooter.extract_visual_observations", return_value=mock_obs):
                res = troubleshoot(
                    "Have we faced this concrete-layer defect before?",
                    attachments=[test_att],
                )

            self.assertTrue(res.rag_found)
            self.assertFalse(res.gemini_called)
            self.assertIn("Historical Incident Recall", res.answer)
            self.assertIn(inc["incident_id"], res.answer)
            self.assertIn("Lead Field Engineer Marcus Vance", res.answer)
        finally:
            with db.connect() as conn:
                conn.execute("DELETE FROM incidents WHERE incident_id = ?", (inc["incident_id"],))

    def test_09_database_persistence_across_reconnection(self):
        """Tests that images, annotations, and references persist in SQLite and are retrievable after reconnection."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_f:
            tmp_db = tmp_f.name

        try:
            # Seed reference library in temporary DB
            incident_store.init_reference_library(db_path=tmp_db)
            refs = incident_store.list_component_references(db_path=tmp_db)
            self.assertEqual(len(refs), 10)

            # Store an image
            test_data = _make_test_image("gap")
            saved = incident_store.save_incident_image(
                data=test_data,
                original_name="test_persist.jpg",
                content_type="image/jpeg",
                metadata={"test": "persistence"},
                db_path=tmp_db,
            )
            img_id = saved["image_id"]

            # Query afresh from SQLite
            with db.connect(tmp_db) as conn:
                row = conn.execute("SELECT * FROM incident_images WHERE image_id = ?", (img_id,)).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["original_name"], "test_persist.jpg")

            # Load image data afresh
            img_ret = incident_store.get_image_data(img_id, db_path=tmp_db)
            self.assertIsNotNone(img_ret)
            data, mime, fn = img_ret
            self.assertEqual(data, test_data)
        finally:
            if os.path.exists(tmp_db):
                try:
                    os.remove(tmp_db)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
