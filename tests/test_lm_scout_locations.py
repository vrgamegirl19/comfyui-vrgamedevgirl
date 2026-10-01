"""Unit tests for the LM Extract location scout (Reference Builder)."""

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

img_mod = importlib.import_module(f"{ROOT.name}.llm.image_prompt_generation")

_REPLY = (
    '{"locations": ['
    '{"name": "Corner Bodega Exterior", "description": "Red awnings glow under amber streetlights."},'
    '{"name": "corner bodega exterior", "description": "Duplicate that must be dropped."},'
    '{"name": "Karaoke Booth", "description": "Mirrored walls and a neon menu board."}]}'
)


class LmScoutLocationsTests(unittest.TestCase):
    def _run(self, payload, reply=_REPLY):
        with patch.object(img_mod, "_run_builder_text_llm", return_value=(reply, {"runner": "lm_studio"})) as llm:
            return img_mod._generate_lm_scout_locations({"runner": "lm_studio", **payload}), llm

    def test_parses_json_and_dedupes(self):
        result, _ = self._run({"model_file": "m", "lyrics_text": "Scene 1: night streets"})
        self.assertEqual([item["name"] for item in result["locations"]], ["Corner Bodega Exterior", "Karaoke Booth"])

    def test_style_theme_and_lyrics_reach_prompt(self):
        _, llm = self._run({"model_file": "m", "lyrics_text": "night streets", "style_theme": "No warehouses"})
        prompt = llm.call_args.args[1]
        self.assertIn("No warehouses", prompt)
        self.assertIn("night streets", prompt)
        self.assertIn("senior music video location scout", prompt)

    def test_requires_source_text(self):
        with self.assertRaises(ValueError):
            self._run({"model_file": "m"})

    def test_bad_reply_raises_with_preview(self):
        with self.assertRaises(ValueError) as ctx:
            self._run({"model_file": "m", "lyrics_text": "x"}, reply="sorry, no")
        self.assertIn("sorry, no", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
