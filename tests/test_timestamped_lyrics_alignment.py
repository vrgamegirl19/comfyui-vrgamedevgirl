import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lyrics", ROOT / "general" / "lyrics.py")
lyrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lyrics)


class TimestampedLyricsAlignmentTests(unittest.TestCase):
    def test_reference_word_alignment_uses_explicit_or_detected_language(self):
        for language, detected, alignment_succeeds in (
            ("german", False, True),
            ("auto", True, True),
            ("auto", True, False),
        ):
            with self.subTest(language=language, alignment_succeeds=alignment_succeeds):
                model = Mock()
                model.transcribe.return_value = SimpleNamespace(language="de", segments=[])
                model.align.return_value = SimpleNamespace(segments=[]) if alignment_succeeds else None
                stable_whisper = SimpleNamespace(load_model=Mock(return_value=model))
                with patch.dict(sys.modules, {"stable_whisper": stable_whisper}):
                    (payload,) = lyrics.VRGDG_TimestampedLyricsExtractor().extract_timestamped_lyrics(
                        {"waveform": torch.zeros(1, 1, 32), "sample_rate": 16},
                        language=language,
                        reference_lyrics="Wenn Seelen schwer",
                        segment_mode="reference_scene_words",
                    )
                self.assertEqual(model.transcribe.call_count, int(detected))
                self.assertEqual(model.align.call_args.args[1], "Wenn Seelen schwer")
                self.assertEqual(model.align.call_args.kwargs["language"], "de")
                self.assertEqual(
                    json.loads(payload)["mode"],
                    "align_reference_words" if alignment_succeeds else "transcribe_reference_words",
                )

    def test_unrecognized_words_keep_acoustic_vocal_range(self):
        extractor = lyrics.VRGDG_TimestampedLyricsExtractor()
        units = [
            {"type": "vocal", "text": "Wenn Seelen schwer"},
            {"type": "vocal", "text": "und Nächte lang"},
        ]
        acoustic = [
            {"words": [{"start": 25.0, "end": 26.0, "text": "Thank"}]},
            {"words": [{"start": 299.0, "end": 300.0, "text": "you"}]},
        ]

        aligned = extractor._reference_scene_word_segments(units, acoustic, 313.0)

        self.assertEqual(aligned[0]["start"], 25.0)
        self.assertEqual(aligned[1]["end"], 300.0)
        self.assertEqual(aligned[0]["text"], "Wenn Seelen schwer")
        self.assertEqual(aligned[1]["text"], "und Nächte lang")


if __name__ == "__main__":
    unittest.main()
