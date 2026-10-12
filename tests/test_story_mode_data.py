"""Story Mode data (storyboard/story_mode.py) and its JS twin agree on tests/story_mode_cases.json."""

import importlib
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

story_mode = importlib.import_module(f"{ROOT.name}.storyboard.story_mode")
scene_helpers = importlib.import_module(f"{ROOT.name}.storyboard.scene_helpers")
CASES = json.loads((ROOT / "tests" / "story_mode_cases.json").read_text(encoding="utf-8"))


class StoryModeDataTests(unittest.TestCase):
    def test_parse_cases(self):
        for item in CASES["parse"]:
            with self.subTest(item["name"]):
                book, names = story_mode.parse_story_book_text(item["text"])
                self.assertEqual(book, item["book"])
                self.assertEqual(names, item["setting_names"])
                text = story_mode.story_book_to_text(book, story_mode.sync_story_settings(book, names))
                self.assertEqual(text, item["text_out"])
                self.assertEqual(story_mode.parse_story_book_text(text)[0], book)

    def test_split_cases(self):
        for item in CASES["split"]:
            with self.subTest(item["name"]):
                chapters, names = story_mode.chapters_from_split(item["paragraphs"], item["split"])
                self.assertEqual(chapters, item["chapters"])
                self.assertEqual(names, item["setting_names"])
                self.assertTrue(story_mode.chapters_keep_text(item["paragraphs"], chapters))

    def test_normalize_cases(self):
        for item in CASES["normalize"]:
            with self.subTest(item["name"]):
                self.assertEqual(story_mode.normalize_story_mode_fields(item["input"]), item["output"])

    def test_continuation_cases(self):
        for item in CASES["continuations"]:
            with self.subTest(item["name"]):
                plan = story_mode.resolve_story_continuations(item["plan"], item["scenes"], item["acts"], item["mode"])
                got = {entry["scene_id"]: [entry["continues_previous"], entry["continuation_source"]] for entry in plan}
                self.assertEqual(got, item["expect"])

    def test_beat_context_says_the_scene_continues(self):
        layer = {"story_scene_plan": [
            {"scene_id": "s1", "story_slice": "She reads."},
            {"scene_id": "s2", "story_slice": "She cries.", "continues_previous": True,
             "continuation_direction": "She drops the letter."},
        ]}
        _arc, brief = story_mode.story_beat_context(layer, "s2", 2)
        self.assertIn("next moment of Scene 1 in one unbroken camera take", brief)
        self.assertIn("Then: She drops the letter.", brief)
        self.assertNotIn("Continuation:", story_mode.story_beat_context(layer, "s1", 1)[1])

    def test_story_layer_normalizer_keeps_story_mode_data(self):
        layer = {"overall_story_idea": "idea", "story_flow_version": 2,
                 "story_book": {"title": "T", "chapters": [{"title": "One", "paragraphs": ["p"]}]}}
        normalized = scene_helpers._normalize_story_layer(layer)
        self.assertEqual(normalized["story_flow_version"], 2)
        self.assertEqual(normalized["story_book"]["chapters"][0]["paragraphs"], ["p"])
        self.assertNotIn("story_acts", scene_helpers._normalize_story_layer({"overall_story_idea": "idea"}))

    def test_changed_text_is_detected(self):
        chapters = [{"paragraphs": ["a"]}, {"paragraphs": ["c"]}]
        self.assertFalse(story_mode.chapters_keep_text(["a", "b", "c"], chapters))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the JS twin")
    def test_js_twin(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("story_mode_data.mjs"))],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
