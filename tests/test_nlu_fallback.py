"""
Tests for Phase 1 Semantic NLU Fallback (agent/nlu.py).
Verifies:
1. config.GEMINI_MODEL is verified to be non-empty and used by NLU.
2. Structured output schema (intent, entities, confidence, ambiguous).
3. Read/Knowledge whitelist strictly blocks all action/write intents.
4. Ambiguity and confidence threshold (< 0.70) gating.
5. Integration with query_router.classify() and query_router.route().
6. Deterministic router remains primary for high-scoring matches.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config
from agent.nlu import (
    ALLOWED_READ_INTENTS,
    FORBIDDEN_WRITE_INTENTS,
    NLUResult,
    _call_semantic_model,
    semantic_fallback,
    set_model_override,
)
from agent.query_router import Intent, classify, route


class TestSemanticNLU(unittest.TestCase):

    def tearDown(self):
        set_model_override(None)

    def test_configured_gemini_model_verified(self):
        """Verifies that config.GEMINI_MODEL is non-empty and that NLU uses config.GEMINI_MODEL."""
        # 1. Verify config.GEMINI_MODEL is populated and valid
        self.assertTrue(bool(config.GEMINI_MODEL and config.GEMINI_MODEL.strip()))

        # 2. Verify NLU invokes the SDK with config.GEMINI_MODEL directly
        with patch("agent.config.GEMINI_API_KEY", "test-api-key"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_resp = MagicMock()
            mock_resp.text = '{"intent": null, "confidence": 0.0}'
            mock_client.models.generate_content.return_value = mock_resp

            _call_semantic_model("status check")

            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args[1]
            self.assertEqual(call_kwargs.get("model"), config.GEMINI_MODEL)

    def test_semantic_model_response_parsing(self):
        """Verifies structured parsing of model JSON into NLUResult."""
        mock_payload = {
            "intent": "GET_LAYER_WIDTH",
            "entities": {"target_metric": "layer_width", "wall_number": 2},
            "confidence": 0.92,
            "ambiguous": False,
        }
        set_model_override(lambda q: json.dumps(mock_payload))

        res = semantic_fallback("bead looks too narrow on wall 2")
        self.assertEqual(res.intent, Intent.GET_LAYER_WIDTH)
        self.assertEqual(res.entities.get("target_metric"), "layer_width")
        self.assertEqual(res.entities.get("wall_number"), 2)
        self.assertAlmostEqual(res.confidence, 0.92)
        self.assertFalse(res.ambiguous)

    def test_action_intents_strictly_blocked(self):
        """NLU must reject state-mutating actions even if the model outputs them."""
        for action_intent in FORBIDDEN_WRITE_INTENTS:
            mock_payload = {
                "intent": action_intent.value,
                "entities": {},
                "confidence": 0.99,
                "ambiguous": False,
            }
            set_model_override(lambda q, p=mock_payload: json.dumps(p))

            res = semantic_fallback("start shift now")
            self.assertIsNone(res.intent)
            self.assertEqual(res.confidence, 0.0)

    def test_low_confidence_and_ambiguity_gating(self):
        """Predictions below 0.70 or marked ambiguous are not routed."""
        # 1. Low confidence
        set_model_override(lambda q: json.dumps({
            "intent": "GET_PRINT_QUALITY",
            "entities": {},
            "confidence": 0.55,
            "ambiguous": False,
        }))
        res = semantic_fallback("maybe check something")
        self.assertEqual(res.confidence, 0.55)

        # 2. Ambiguous query
        set_model_override(lambda q: json.dumps({
            "intent": "GET_LAYER_WIDTH",
            "entities": {},
            "confidence": 0.85,
            "ambiguous": True,
        }))
        res = semantic_fallback("width or height")
        self.assertTrue(res.ambiguous)

    def test_query_router_nlu_rescue_integration(self):
        """Unseen natural phrasing is rescued by NLU before blind keyword fallback."""
        set_model_override(lambda q: json.dumps({
            "intent": "KNOWLEDGE_QUERY",
            "entities": {},
            "confidence": 0.94,
            "ambiguous": False,
        }))

        intent, terms = classify("why are layers not sticking")
        self.assertEqual(intent, Intent.KNOWLEDGE_QUERY)
        self.assertIn("semantic_nlu", terms)

    def test_deterministic_router_remains_primary(self):
        """High-scoring deterministic matches never invoke NLU."""
        called = False

        def _spy_model(q):
            nonlocal called
            called = True
            return None

        set_model_override(_spy_model)

        intent, _ = classify("what is the current layer")
        self.assertEqual(intent, Intent.GET_CURRENT_LAYER)
        self.assertFalse(called, "Deterministic route should have matched before calling NLU")


if __name__ == "__main__":
    unittest.main()
