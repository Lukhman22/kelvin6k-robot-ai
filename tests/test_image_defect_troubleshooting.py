"""
Comprehensive test suite for Image-Based Concrete Printing Defect Troubleshooting (RAG-First Multimodal Upgrade).

Validates:
1. End-to-end defect diagnosis with uploaded image and generic query ("find the issue in this and tell me the solution").
2. Output follows the required 6-section structure:
   - 1. What I can see
   - 2. Relevant historical cases
   - 3. Most plausible explanations
   - 4. Recommended next checks
   - 5. Solution
   - 6. Confidence and limitations
3. Direct visual observations are separated from causes; never infers internal pressure, accelerator flow, or motor temperature.
4. Local computer vision analysis extracts objective visual observations when external Vision API is unavailable.
5. Symptom-based query expansion maps visual observations to domain defect terms (layer gap, void, irregular deposition, slumping, tearing).
6. Grounded historical RAG hits cite authoritative documentation and distinguish unconfirmed vs confirmed causes.
7. Missing or non-matching defects fall back to external AI guidance with clear disclaimers.
8. Industrial safety enforcement:
   - No clearing pressurized lines (requires zero-pressure verification).
   - Prohibits unauthorized travel limit (H0D11) and encoder resets (E136.1, E137.1).
   - Warns against striking equipment or rapid feedrate pulsing.
9. Verification of all indexed knowledge sources: operator audio notes (14), manager notes (12), handwritten notes (16).
10. FastAPI /api/assistant/query integration with multipart/form-data binary upload.
"""

from unittest.mock import MagicMock, patch
import io
import pytest
import unittest
import numpy as np
import cv2

from fastapi.testclient import TestClient
from api.server import app
from agent import config
from agent.rag.retriever import get_store, search_documentation
from agent.troubleshooter import (
    Attachment,
    TroubleshootResult,
    troubleshoot,
    extract_local_cv_observations,
    extract_visual_observations,
    synthesize_search_queries,
    _format_rag_answer,
    _evaluate_safety_warning,
)


def _create_synthetic_image(defect_type: str = "gap") -> bytes:
    """Helper creating synthetic concrete bead images for testing."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 210  # light grey bed
    if defect_type == "gap":
        # Two separated bead segments with a void between them
        cv2.rectangle(img, (60, 200), (250, 280), (90, 90, 90), -1)
        cv2.rectangle(img, (380, 200), (580, 280), (90, 90, 90), -1)
    elif defect_type == "corner":
        # L-shaped corner with bulging/irregular bead
        pts = np.array([[100, 100], [420, 100], [440, 150], [420, 360], [330, 360], [330, 190], [100, 190]], np.int32)
        cv2.fillPoly(img, [pts], (90, 90, 90))
    elif defect_type == "tearing":
        # Jagged bead contour
        pts = []
        for x in range(50, 590, 12):
            y_top = 210 + (16 if (x // 12) % 2 == 0 else -16)
            pts.append([x, y_top])
        for x in range(578, 40, -12):
            y_bot = 290 + (16 if (x // 12) % 2 == 0 else -16)
            pts.append([x, y_bot])
        cv2.fillPoly(img, [np.array(pts, np.int32)], (90, 90, 90))
    else:
        # Uniform smooth bead
        cv2.rectangle(img, (60, 210), (580, 290), (90, 90, 90), -1)

    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()


class TestImageDefectTroubleshooting(unittest.TestCase):
    """Validation of image-based defect troubleshooting pipeline."""

    @classmethod
    def setUpClass(cls):
        cls._orig_key = config.GEMINI_API_KEY
        config.GEMINI_API_KEY = ""
        cls.store = get_store(refresh=True)

    @classmethod
    def tearDownClass(cls):
        config.GEMINI_API_KEY = cls._orig_key

    def test_knowledge_base_completeness(self):
        """Verifies operator audio notes (14), manager notes (12), and handwritten notes (16) are indexed."""
        chunks = self.store.chunks
        op_chunks = [c for c in chunks if c.filename == "kelvin6k_operator_knowledge.jsonl"]
        mgr_chunks = [c for c in chunks if c.filename == "kelvin6k_manager_whatsapp_notes.jsonl"]
        hw_chunks = [c for c in chunks if c.filename == "kelvin6k_handwritten_fault_notes.jsonl"]

        self.assertEqual(len(op_chunks), 14, "Operator audio notes must maintain exactly 14 chunks.")
        self.assertEqual(len(mgr_chunks), 12, "Manager WhatsApp notes must contain 12 chunks.")
        self.assertEqual(len(hw_chunks), 16, "Handwritten fault notes must contain 16 chunks.")

    def test_local_cv_extracts_gap_and_limits(self):
        """Tests that local CV extraction identifies bead discontinuity and states physical limitations."""
        img_bytes = _create_synthetic_image("gap")
        att = Attachment(
            filename="test_gap.jpg",
            content_type="image/jpeg",
            size_bytes=len(img_bytes),
            data=img_bytes,
            kind="image",
            width=640,
            height=480,
        )
        obs = extract_local_cv_observations(att)
        self.assertIn("640x480px", obs)
        self.assertIn("Visible bead discontinuity", obs)
        self.assertIn("internal pump line pressure", obs)
        self.assertIn("cannot be determined from images alone", obs)

    def test_synthesize_search_queries_expands_domain_terms(self):
        """Tests that symptom phrases are generated for generic prompts with image observations."""
        queries = synthesize_search_queries(
            question="find the issue in this and tell me the solution",
            observations="Visible bead discontinuity with local gaps or voids between concrete layers.",
        )
        self.assertTrue(any("layer gap" in q for q in queries))
        self.assertIn("layer gap void", queries)

    def test_end_to_end_gap_defect_diagnosis_6_sections(self):
        """Tests that a gap defect image with generic question answers from RAG with 6 structured sections."""
        img_bytes = _create_synthetic_image("gap")
        att = Attachment(
            filename="bead_gap_defect.jpg",
            content_type="image/jpeg",
            size_bytes=len(img_bytes),
            data=img_bytes,
            kind="image",
            width=640,
            height=480,
        )

        res = troubleshoot("find the issue in this and tell me the solution", attachments=[att])

        self.assertTrue(res.rag_found, "Should match Layer Gap or Void in RAG.")
        self.assertEqual(res.source, "rag-knowledge")
        self.assertGreaterEqual(res.rag_score, 0.35)
        self.assertIn("visual_defects.md, Issue: Layer Gap or Void", res.citations[0])

        # Verify all 6 structured sections
        self.assertIn("### 1. What I can see", res.answer)
        self.assertIn("### 2. Relevant historical cases", res.answer)
        self.assertIn("### Historical record found", res.answer)
        self.assertIn("### 3. Most plausible explanations", res.answer)
        self.assertIn("### 4. Recommended next checks", res.answer)
        self.assertIn("### 5. Solution", res.answer)
        self.assertIn("### 6. Confidence and limitations", res.answer)

        # Verify unconfirmed cause preserved
        self.assertIn("Possible cause / unconfirmed", res.answer)
        # Verify safety & limits
        self.assertIn("depressur", res.answer.lower())
        self.assertIn("cannot be determined from images alone", res.answer)

    def test_end_to_end_corner_irregularity_diagnosis(self):
        """Tests that corner deformation image maps to Layer Deformation or Irregular Deposition."""
        img_bytes = _create_synthetic_image("corner")
        att = Attachment(
            filename="corner_bulge.jpg",
            content_type="image/jpeg",
            size_bytes=len(img_bytes),
            data=img_bytes,
            kind="image",
            width=640,
            height=480,
        )

        res = troubleshoot("what is wrong with this?", attachments=[att])

        self.assertTrue(res.rag_found)
        self.assertEqual(res.source, "rag-knowledge")
        self.assertIn("visual_defects.md, Issue: Layer Deformation or Irregular Deposition", res.citations[0])
        self.assertIn("### 1. What I can see", res.answer)
        self.assertIn("### 5. Solution", res.answer)

    def test_mocked_gemini_vision_observation_pipeline(self):
        """Tests that when Gemini Vision generates rich observations, they feed RAG correctly."""
        img_bytes = _create_synthetic_image("gap")
        att = Attachment(
            filename="photo.jpg",
            content_type="image/jpeg",
            size_bytes=len(img_bytes),
            data=img_bytes,
            kind="image",
        )

        mock_vision_text = "Severe through-layer vertical crack running across 4 printed concrete layers."

        with patch.object(config, "GEMINI_API_KEY", "mock-gemini-key"):
            with patch("google.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_resp = MagicMock()
                mock_resp.text = mock_vision_text
                mock_client.models.generate_content.return_value = mock_resp
                mock_client_cls.return_value = mock_client

                obs = extract_visual_observations(att)
                self.assertIn(mock_vision_text, obs)

                res = troubleshoot("diagnose this", attachments=[att])
                self.assertTrue(res.rag_found)
                self.assertTrue(any("Vertical or Through-Layer Crack" in c for c in res.citations))

    def test_safety_warnings_triggered(self):
        """Verifies industrial safety rules trigger immediate warnings."""
        # Pressurized nozzle clearing
        w1 = _evaluate_safety_warning("How to clear clogged nozzle while running?", "pump on pressurized")
        self.assertIsNotNone(w1)
        self.assertIn("DANGER", w1)
        self.assertIn("relieve hydraulic/material line pressure", w1)

        # Hammering pump / pulsing feedrate
        w2 = _evaluate_safety_warning("Can I hit the pump with a hammer to clear it?", "hammer strike")
        self.assertIsNotNone(w2)
        self.assertIn("DANGER", w2)
        self.assertIn("Never hit pressurized equipment", w2)

        # Travel limit override H0D11
        w3 = _evaluate_safety_warning("How do I change parameter H0D11 to jog downward?", "H0D11 travel limit")
        self.assertIsNotNone(w3)
        self.assertIn("Engineering authorization is required", w3)

        # Encoder resets
        w4 = _evaluate_safety_warning("Reset servo encoder for error E137.1", "E137.1 encoder reset")
        self.assertIsNotNone(w4)
        self.assertIn("authorized technicians", w4)

    def test_fallback_when_defect_not_in_records(self):
        """Tests fallback behavior when image shows a novel defect not in RAG."""
        img_bytes = _create_synthetic_image("normal")
        att = Attachment(
            filename="exotic_defect.jpg",
            content_type="image/jpeg",
            size_bytes=len(img_bytes),
            data=img_bytes,
            kind="image",
        )

        # Mock search to return no hits
        with patch("agent.troubleshooter.search_documentation") as mock_search, \
             patch("agent.troubleshooter._call_gemini_fallback") as mock_fallback:
            mock_search.return_value = {"found": False, "results": []}
            mock_fallback.return_value = (
                "### 1. What I can see\n- Surface texture visible.\n"
                "### 2. Relevant historical cases\n**No sufficient matching historical record found.**\n"
                "**General AI guidance (Gemini):**\n"
                "### 3. Most plausible explanations\n- **Possible causes:** Toolpath resonance.\n"
                "### 4. Recommended next checks\n- **Safe diagnostic checks:** Inspect gantry belt tension.\n"
                "### 5. Solution\n- **Recommended next step:** Run calibration pattern.\n"
                "### 6. Confidence and limitations\n- **Information needed:** Controller log export.",
                None,
            )

            res = troubleshoot("analyze this unusual pattern", attachments=[att])
            self.assertEqual(res.source, "gemini-fallback")
            self.assertTrue(res.fallback_used)
            self.assertIn("No sufficient matching historical record found", res.answer)
            self.assertIn("General AI guidance (Gemini)", res.answer)
            self.assertIn("Safe diagnostic checks", res.answer)

    def test_fastapi_multipart_image_upload_endpoint(self):
        """Tests FastAPI /api/assistant/query endpoint with multipart/form-data binary upload."""
        with TestClient(app) as client:
            img_bytes = _create_synthetic_image("gap")

            response = client.post(
                "/api/assistant/query",
                data={"question": "find the issue in this and tell me the solution", "speak": "false"},
                files={"files": ("bead_gap_defect.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("answer", data)
            self.assertIn("What I can see", data["answer"])
        self.assertIn("Relevant historical cases", data["answer"])
        self.assertIn("Solution", data["answer"])
        self.assertIsNotNone(data.get("attachments"))
        self.assertEqual(len(data["attachments"]), 1)
        self.assertEqual(data["attachments"][0]["filename"], "bead_gap_defect.jpg")
        self.assertEqual(data["attachments"][0]["kind"], "image")
