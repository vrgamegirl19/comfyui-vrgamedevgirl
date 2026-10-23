"""The agent API's video style and FX wording matches the Video Builder's."""

import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

vs = importlib.import_module(f"{ROOT.name}.agent_api.video_style")
CASES = json.loads((Path(__file__).parent / "video_style_cases.json").read_text(encoding="utf-8"))


class VideoStyleFxTests(unittest.TestCase):
    def test_style_matches_the_video_builder_cases(self):
        for case in CASES["style"]:
            with self.subTest(case["name"]):
                self.assertEqual(vs.video_style_verbiage(case["card"], case["defaults"]), case["expected"])

    def test_fx_matches_the_video_builder_cases(self):
        for case in CASES["fx"]:
            with self.subTest(case["name"]):
                self.assertEqual(vs.fx_accent(case["defaults"], case["shot_index"]), case["expected"])

    def test_the_label_names_the_style_in_the_opening_line(self):
        self.assertEqual(vs.video_style_label({}, {"video_style": "film_noir"}), "Film noir")
        self.assertEqual(vs.video_style_label({}, {"video_style": "Cinematic realism"}), "Cinematic realism")
        self.assertEqual(vs.video_style_label({}, {"video_style": "custom", "video_style_custom": "Teal haze"}), "Teal haze")

    def test_the_style_goes_after_the_first_paragraph_once(self):
        out = vs.insert_video_style("detailed_description:\nA noir style.\n\n[Shot 1] She walks.", "Film noir: High contrast.")
        self.assertEqual(out, "detailed_description:\nA noir style.\n\nFilm noir: High contrast.\n\n[Shot 1] She walks.")
        self.assertEqual(vs.insert_video_style(out, "Film noir: High contrast."), out)

    def test_fx_goes_at_the_end_of_each_shot_once(self):
        core = "[Shot 1] She walks.\n\n[Shot 2] At 0:03.0, she turns."
        defaults = {"fx_preset": "lighting"}
        out = vs.apply_fx(core, defaults)
        self.assertIn(f"She walks. {vs.fx_accent(defaults, 0)}\n\n[Shot 2]", out)
        self.assertTrue(out.endswith(f"she turns. {vs.fx_accent(defaults, 1)}"))
        self.assertEqual(vs.apply_fx(out, defaults), out)
        self.assertEqual(len(out) - len(core), vs.fx_reserve(defaults, 2))
        self.assertEqual(vs.apply_fx(core, {}), core)

    def test_the_exported_data_matches_the_video_builder(self):
        result = subprocess.run(["node", "scripts/export_video_style_data.mjs", "--check"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
