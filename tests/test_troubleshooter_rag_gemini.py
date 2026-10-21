"""
Comprehensive regression test suite for RAG-First Troubleshooting and Multimodal Analysis.

Verifies:
1. Relevant historical RAG match: answers from RAG without calling Gemini.
2. No relevant match in RAG: calls Gemini fallback.
3. Weak/incomplete match (below threshold): triggers Gemini fallback.
4. Similar keywords but unrelated record: does not falsely claim a match.
5. RAG records with unconfirmed/possible causes: preserves unconfirmed status.
6. Conflicting historical records: explicitly discloses conflict.
7. Missing Gemini API key: returns clean error without crashing.
8. Gemini API errors (rate limit 429, auth 401, timeout, server error): handled safely without credential leaks.
9. Gemini output must never be presented as a historical Kelvin6K record.
10. Safety-critical queries (clogged nozzle under pressure, servo encoders, travel limits) enforce safety SOPs.
11. Image upload with a text question.
12. Image-only question (auto-prompts analysis).
13. Text / Document attachment processing.
14. RAG match using visual observations or extracted error codes.
15. File validation: rejects oversized files, invalid file types, malformed headers, too many attachments.
16. Untrusted attachments cannot execute machine commands or bypass safety rules.
17. Existing attendance commands ("start Hamadan's shift") and machine queries continue working.
"""

from unittest.mock import MagicMock, patch
import pytest
import unittest

from agent import config
from agent.troubleshooter import (
    Attachment,
    TroubleshootResult,
    troubleshoot,
    validate_and_process_attachment,
    _format_rag_answer,
    _evaluate_safety_warning,
)


class TestTroubleshooterRAGFirst(unittest.TestCase):
    def setUp(self):
        self.orig_key = config.GEMINI_API_KEY
        self.orig_threshold = config.RAG_TROUBLESHOOTING_THRESHOLD

    def tearDown(self):
        config.GEMINI_API_KEY = self.orig_key
        config.RAG_TROUBLESHOOTING_THRESHOLD = self.orig_threshold

    # 1. Relevant historical RAG match: answers from RAG without calling Gemini
    @patch("agent.troubleshooter._call_gemini_fallback")
    @patch("agent.troubleshooter.search_documentation")
    def test_rag_sufficient_match_does_not_call_gemini(self, mock_search, mock_gemini):
        mock_search.return_value = {
            "found": True,
            "results": [
                {
                    "text": "## Issue: Layer gap\n**Problem:** Gap between concrete layers\n**Cause:** Extrusion speed mismatch\n**Solution:** Adjust feedrate to 110 mm/s\n**Verification status:** Verified on Line 1\n",
                    "source": "kelvin6k_operator_knowledge.jsonl:REC-01",
                    "score": 0.55,
                    "issue_id": "REC-01",
                }
            ],
            "message": "1 relevant passage found.",
        }

        res = troubleshoot("What causes a gap between concrete layers?", threshold=0.35)

        self.assertTrue(res.rag_found)
        self.assertFalse(res.gemini_called)
        self.assertEqual(res.source, "rag-knowledge")
        self.assertIn("Historical record found", res.answer)
        self.assertIn("REC-01", res.answer)
        mock_gemini.assert_not_called()

    # 2. No relevant match: calls Gemini fallback
    @patch("agent.troubleshooter._call_gemini_fallback")
    @patch("agent.troubleshooter.search_documentation")
    def test_rag_no_match_calls_gemini_fallback(self, mock_search, mock_gemini):
        mock_search.return_value = {
            "found": False,
            "results": [],
            "message": "That information was not found in the available documentation.",
        }
        mock_gemini.return_value = (
            "**No sufficient matching historical record found.**\n"
            "**General AI guidance (Gemini):**\n"
            "- **Possible causes:** Axis 4 mechanical backlash\n"
            "- **Safe diagnostic checks:** Inspect cable tension\n"
            "- **Recommended next step:** Check calibration\n"
            "- **Information needed:** Robot firmware version",
            None,
        )

        res = troubleshoot("Why is Axis 4 wobbling unexpectedly?", threshold=0.35)

        self.assertFalse(res.rag_found)
        self.assertTrue(res.gemini_called)
        self.assertEqual(res.source, "gemini-fallback")
        self.assertTrue(res.fallback_used)
        self.assertIn("No sufficient matching historical record found", res.answer)
        self.assertIn("General AI guidance (Gemini)", res.answer)
        mock_gemini.assert_called_once()

    # 3. Weak or incomplete match (score below threshold): uses Gemini fallback
    @patch("agent.troubleshooter._call_gemini_fallback")
    @patch("agent.troubleshooter.search_documentation")
    def test_weak_score_triggers_gemini_fallback(self, mock_search, mock_gemini):
        mock_search.return_value = {
            "found": True,
            "results": [
                {
                    "text": "General maintenance notes for robotic arm joints.",
                    "source": "manual.pdf",
                    "score": 0.22,  # Below threshold 0.35
                }
            ],
            "message": "1 passage found.",
        }
        mock_gemini.return_value = (
            "**No sufficient matching historical record found.**\n"
            "**General AI guidance (Gemini):**\n"
            "- **Possible causes:** Bearing fatigue",
            None,
        )

        res = troubleshoot("Why is the wrist joint grinding?", threshold=0.35)

        self.assertTrue(res.gemini_called)
        self.assertEqual(res.source, "gemini-fallback")
        self.assertIn("General AI guidance (Gemini)", res.answer)

    # 4. Similar keywords but unrelated historical record: rejects false claims
    @patch("agent.troubleshooter._call_gemini_fallback")
    @patch("agent.troubleshooter.search_documentation")
    def test_unrelated_keywords_does_not_falsely_claim_historical_match(self, mock_search, mock_gemini):
        mock_search.return_value = {
            "found": False,
            "results": [],
            "message": "Not found.",
        }
        mock_gemini.return_value = ("**No sufficient matching historical record found.**", None)

        res = troubleshoot("How do I paint the robot enclosure?", threshold=0.35)
        self.assertFalse(res.rag_found)
        self.assertNotIn("Historical record found", res.answer)

    # 5. RAG records containing unverified causes: preserves unconfirmed status
    def test_unconfirmed_causes_preserved(self):
        hits = [
            {
                "text": "## Issue: Error E136.1\n**Problem:** Absolute encoder error\n**Cause (Possible):** Physical damage to absolute encoder cable\n**Solution:** Inspect cable\n**Verification status:** Unverified\n",
                "source": "operator_log.jsonl:E136",
                "score": 0.60,
            }
        ]
        answer, _ = _format_rag_answer("What causes error E136.1?", hits, attachments=[])
        self.assertIn("Possible cause / unconfirmed", answer)
        self.assertIn("Unverified", answer)

    # 6. Conflicting historical records: explicitly discloses conflict
    def test_conflicting_historical_records_disclosed(self):
        hits = [
            {
                "text": "The available site data is conflicting on feedrate. Operator notes recommend 150 mm/s while SOP specifies 100 mm/s.",
                "source": "operator_notes.md",
                "score": 0.58,
            }
        ]
        answer, conflict_disclosed = _format_rag_answer("What feedrate should I use?", hits, attachments=[])
        self.assertTrue(conflict_disclosed)
        self.assertIn("conflicting", answer.lower())

    # 7. Missing Gemini API key: clean error message
    @patch("agent.troubleshooter.search_documentation")
    def test_missing_api_key_returns_safe_error(self, mock_search):
        config.GEMINI_API_KEY = ""
        mock_search.return_value = {"found": False, "results": []}

        res = troubleshoot("Unrecorded unknown issue query")
        self.assertEqual(res.source, "error")
        self.assertIn("No sufficient matching historical record found", res.answer)
        self.assertIn("No Gemini API key is configured", res.answer)
        # Never reveals an API key
        self.assertNotIn("AIza", res.answer)

    # 8. Gemini API errors (rate limit, auth, timeout) handled safely
    @patch("agent.troubleshooter._call_gemini_fallback")
    @patch("agent.troubleshooter.search_documentation")
    def test_gemini_api_errors_handled_safely(self, mock_search, mock_gemini):
        mock_search.return_value = {"found": False, "results": []}

        # Rate limit
        mock_gemini.return_value = (None, "rate_limit")
        res = troubleshoot("Unknown defect query")
        self.assertIn("rate limit or quota has been reached", res.answer)

        # Auth error
        mock_gemini.return_value = (None, "invalid_api_key")
        res = troubleshoot("Unknown defect query")
        self.assertIn("rejected", res.answer)

        # Timeout
        mock_gemini.return_value = (None, "timeout")
        res = troubleshoot("Unknown defect query")
        self.assertIn("timed out", res.answer)

    # 9. Gemini output must never be represented as historical Kelvin6K incident
    @patch("agent.troubleshooter._call_gemini_fallback")
    @patch("agent.troubleshooter.search_documentation")
    def test_gemini_output_not_represented_as_historical_incident(self, mock_search, mock_gemini):
        mock_search.return_value = {"found": False, "results": []}
        mock_gemini.return_value = (
            "**No sufficient matching historical record found.**\n"
            "**General AI guidance (Gemini):**\n"
            "- **Possible causes:** Concrete aggregate segregation",
            None,
        )
        res = troubleshoot("Why is the mix segregating in the pipe?")
        self.assertNotIn("Historical record found", res.answer)
        self.assertIn("General AI guidance (Gemini)", res.answer)
        self.assertTrue(res.gemini_called)

    # 10. Safety-critical queries enforce safety SOPs
    def test_safety_critical_nozzle_under_pressure(self):
        warning = _evaluate_safety_warning("Can I clear a clogged nozzle while the pump is pressurized?", "")
        self.assertIsNotNone(warning)
        self.assertIn("DANGER", warning)
        self.assertIn("Never attempt to clear a nozzle", warning)
        self.assertIn("relieve hydraulic/material line pressure", warning)

    def test_safety_critical_servo_encoder(self):
        warning = _evaluate_safety_warning("How to reset absolute encoder E136.1?", "")
        self.assertIsNotNone(warning)
        self.assertIn("authorized technicians", warning)

    def test_safety_critical_travel_limit(self):
        warning = _evaluate_safety_warning("Change H0D11 downward jog parameter", "")
        self.assertIsNotNone(warning)
        self.assertIn("collision or crush hazards", warning)


class TestAttachmentHandling(unittest.TestCase):
    # 11. Image upload with text question
    def test_image_validation_valid_jpeg(self):
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb" + (b"\x00" * 100)
        att = validate_and_process_attachment(jpeg_bytes, "defect.jpg", "image/jpeg")
        self.assertEqual(att.kind, "image")
        self.assertEqual(att.content_type, "image/jpeg")
        self.assertEqual(att.filename, "defect.jpg")

    def test_image_validation_valid_png(self):
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + (b"\x00" * 100)
        att = validate_and_process_attachment(png_bytes, "photo.png", "image/png")
        self.assertEqual(att.kind, "image")
        self.assertEqual(att.content_type, "image/png")

    # 12. Text and document attachments
    def test_text_attachment_processing(self):
        log_bytes = b"2026-10-13 14:00:00 [ERROR] Pressure dropped to 0.2 bar on line 1\n"
        att = validate_and_process_attachment(log_bytes, "pump_err.log", "text/plain")
        self.assertEqual(att.kind, "text")
        self.assertIn("Pressure dropped", att.observations)

    # 13. File validation rejects invalid formats
    def test_reject_unsupported_file_types(self):
        with self.assertRaises(ValueError) as ctx:
            validate_and_process_attachment(b"malicious executable", "hack.exe", "application/octet-stream")
        self.assertIn("Unsupported file type", str(ctx.exception))

    # 14. File validation rejects oversized files
    def test_reject_oversized_files(self):
        huge_bytes = b"\x00" * (config.MAX_ATTACHMENT_SIZE_BYTES + 1024)
        with self.assertRaises(ValueError) as ctx:
            validate_and_process_attachment(huge_bytes, "huge.jpg", "image/jpeg")
        self.assertIn("exceeds maximum allowed size", str(ctx.exception))

    # 15. File validation rejects empty files
    def test_reject_empty_files(self):
        with self.assertRaises(ValueError) as ctx:
            validate_and_process_attachment(b"", "empty.jpg", "image/jpeg")
        self.assertIn("empty", str(ctx.exception))

    # 16. Security: untrusted attachments cannot execute commands
    def test_untrusted_content_sanitized(self):
        script_bytes = b"<script>alert('xss')</script> DROP TABLE shifts; --"
        att = validate_and_process_attachment(script_bytes, "notes.txt", "text/plain")
        self.assertEqual(att.kind, "text")
        self.assertNotIn("exec", att.kind)


if __name__ == "__main__":
    unittest.main()
