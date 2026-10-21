"""
Regression tests for STT fallback and safe audio handling:
- Auto language detection default kept (WHISPER_LANGUAGE='auto').
- te -> hi fallback when hi_prob >= en_prob in all_language_probs.
- te -> en fallback when en_prob > hi_prob in all_language_probs.
- Direct pass-through without re-transcription when 'en' or 'hi' is detected.
- Safe empty / zero-sample audio handling.
- Safe VAD silence handling (e.g. ValueError: max() iterable argument is empty).
- Verification that no initial_prompt is introduced and model remains 'base'.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent import config
from agent.listener import decode_options, transcribe


class TestSTTFallbackAndVADHandling(unittest.TestCase):
    def setUp(self):
        self._orig_lang = config.WHISPER_LANGUAGE
        self._orig_model_size = config.WHISPER_MODEL_SIZE
        config.WHISPER_LANGUAGE = "auto"

    def tearDown(self):
        config.WHISPER_LANGUAGE = self._orig_lang
        config.WHISPER_MODEL_SIZE = self._orig_model_size

    def test_whisper_config_defaults(self):
        """Verify WHISPER_LANGUAGE=auto, base model kept, and no initial_prompt in decode_options."""
        self.assertEqual(config.WHISPER_LANGUAGE, "auto")
        self.assertEqual(config.WHISPER_MODEL_SIZE, "base")

        opts = decode_options()
        self.assertIsNone(opts.get("language"))
        self.assertNotIn("initial_prompt", opts)
        self.assertTrue(opts.get("vad_filter"))

    def test_te_to_hi_regression_fallback(self):
        """
        When Whisper detects 'te' (Telugu) and all_language_probs has hi >= en,
        re-transcribe with language='hi'.
        """
        audio = np.ones(16000, dtype=np.float32)

        first_info = SimpleNamespace(
            language="te",
            duration_after_vad=1.0,
            all_language_probs=[("te", 0.52), ("hi", 0.33), ("en", 0.10)],
        )
        first_segment = SimpleNamespace(text="తెలుగు టెక్స్ట్")

        second_info = SimpleNamespace(
            language="hi",
            duration_after_vad=1.0,
            all_language_probs=[("hi", 0.85), ("en", 0.10)],
        )
        second_segment = SimpleNamespace(text="प्रिंट चल रहा है क्या?")

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            ([first_segment], first_info),
            ([second_segment], second_info),
        ]

        with patch("agent.listener._get_model", return_value=mock_model):
            result = transcribe(audio)

        self.assertEqual(result, "प्रिंट चल रहा है क्या?")
        self.assertEqual(mock_model.transcribe.call_count, 2)

        # First call: auto detection (language=None)
        first_call_opts = mock_model.transcribe.call_args_list[0][1]
        self.assertIsNone(first_call_opts.get("language"))
        self.assertNotIn("initial_prompt", first_call_opts)

        # Second call: forced fallback language='hi'
        second_call_opts = mock_model.transcribe.call_args_list[1][1]
        self.assertEqual(second_call_opts.get("language"), "hi")
        self.assertNotIn("initial_prompt", second_call_opts)

    def test_ur_to_hi_regression_fallback(self):
        """
        When Whisper detects 'ur' (Urdu) for Hindustani/Hindi speech,
        always re-transcribe with language='hi' (Devanagari script),
        even if English loanwords cause en_prob > hi_prob.
        """
        audio = np.ones(16000, dtype=np.float32)

        first_info = SimpleNamespace(
            language="ur",
            duration_after_vad=1.0,
            all_language_probs=[("ur", 0.88), ("en", 0.08), ("hi", 0.03)],
        )
        first_segment = SimpleNamespace(text="کیا پرنٹ चल रहा है")

        second_info = SimpleNamespace(
            language="hi",
            duration_after_vad=1.0,
            all_language_probs=[("hi", 0.90), ("ur", 0.08)],
        )
        second_segment = SimpleNamespace(text="क्या प्रिंट चल रहा है?")

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            ([first_segment], first_info),
            ([second_segment], second_info),
        ]

        with patch("agent.listener._get_model", return_value=mock_model):
            result = transcribe(audio)

        self.assertEqual(result, "क्या प्रिंट चल रहा है?")
        self.assertEqual(mock_model.transcribe.call_count, 2)

        # First call: auto detection (language=None)
        first_call_opts = mock_model.transcribe.call_args_list[0][1]
        self.assertIsNone(first_call_opts.get("language"))

        # Second call: forced fallback language='hi'
        second_call_opts = mock_model.transcribe.call_args_list[1][1]
        self.assertEqual(second_call_opts.get("language"), "hi")

    def test_te_to_en_regression_fallback(self):
        """
        When Whisper detects 'te' (Telugu) and all_language_probs has en > hi,
        re-transcribe with language='en'.
        """
        audio = np.ones(16000, dtype=np.float32)

        first_info = SimpleNamespace(
            language="te",
            duration_after_vad=1.0,
            all_language_probs=[("te", 0.48), ("en", 0.35), ("hi", 0.12)],
        )
        first_segment = SimpleNamespace(text="తెలుగు")

        second_info = SimpleNamespace(
            language="en",
            duration_after_vad=1.0,
            all_language_probs=[("en", 0.90)],
        )
        second_segment = SimpleNamespace(text="What is the current layer?")

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            ([first_segment], first_info),
            ([second_segment], second_info),
        ]

        with patch("agent.listener._get_model", return_value=mock_model):
            result = transcribe(audio)

        self.assertEqual(result, "What is the current layer?")
        self.assertEqual(mock_model.transcribe.call_count, 2)

        second_call_opts = mock_model.transcribe.call_args_list[1][1]
        self.assertEqual(second_call_opts.get("language"), "en")

    def test_te_to_hi_dict_probs_format(self):
        """When all_language_probs is formatted as a dict, fallback still works."""
        audio = np.ones(16000, dtype=np.float32)

        first_info = SimpleNamespace(
            language="te",
            duration_after_vad=1.0,
            all_language_probs={"te": 0.5, "hi": 0.35, "en": 0.15},
        )
        second_info = SimpleNamespace(language="hi", duration_after_vad=1.0)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            ([SimpleNamespace(text="తెలుగు")], first_info),
            ([SimpleNamespace(text="नमस्ते")], second_info),
        ]

        with patch("agent.listener._get_model", return_value=mock_model):
            result = transcribe(audio)

        self.assertEqual(result, "नमस्ते")
        self.assertEqual(mock_model.transcribe.call_count, 2)
        second_call_opts = mock_model.transcribe.call_args_list[1][1]
        self.assertEqual(second_call_opts.get("language"), "hi")

    def test_no_fallback_on_direct_en_or_hi_detection(self):
        """Direct detection of 'en' or 'hi' must NOT re-transcribe."""
        audio = np.ones(16000, dtype=np.float32)

        info_en = SimpleNamespace(
            language="en",
            duration_after_vad=1.0,
            all_language_probs=[("en", 0.88), ("hi", 0.05)],
        )
        segment_en = SimpleNamespace(text="Hello Kelvin")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([segment_en], info_en)

        with patch("agent.listener._get_model", return_value=mock_model):
            result = transcribe(audio)

        self.assertEqual(result, "Hello Kelvin")
        self.assertEqual(mock_model.transcribe.call_count, 1)

    def test_explicit_whisper_language_skips_auto_fallback(self):
        """When WHISPER_LANGUAGE is explicitly set (e.g. 'en'), no auto fallback occurs."""
        config.WHISPER_LANGUAGE = "en"
        audio = np.ones(16000, dtype=np.float32)

        info = SimpleNamespace(language="en", duration_after_vad=1.0, all_language_probs=None)
        segment = SimpleNamespace(text="English only query")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([segment], info)

        with patch("agent.listener._get_model", return_value=mock_model):
            result = transcribe(audio)

        self.assertEqual(result, "English only query")
        self.assertEqual(mock_model.transcribe.call_count, 1)

    def test_safe_handling_empty_and_zero_audio(self):
        """Empty audio inputs return empty string safely without calling model."""
        with patch("agent.listener._get_model") as mock_get_model:
            self.assertEqual(transcribe(None), "")
            self.assertEqual(transcribe(np.zeros(0, dtype=np.float32)), "")
            self.assertEqual(transcribe(np.array([], dtype=np.float32)), "")
            mock_get_model.assert_not_called()

    def test_safe_handling_vad_empty_audio_raising_value_error(self):
        """
        When faster-whisper raises ValueError ('max() iterable argument is empty')
        because VAD filtered out all speech frames, transcribe returns '' safely.
        """
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = ValueError("max() iterable argument is empty")

        with patch("agent.listener._get_model", return_value=mock_model):
            result = transcribe(audio)

        self.assertEqual(result, "")

    def test_safe_handling_vad_zero_duration_after_vad(self):
        """When duration_after_vad is 0, transcribe returns '' safely."""
        audio = np.ones(16000, dtype=np.float32)
        info = SimpleNamespace(language="en", duration_after_vad=0.0, all_language_probs=[])

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], info)

        with patch("agent.listener._get_model", return_value=mock_model):
            result = transcribe(audio)

        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
    

