"""Story Mode: Update Story From This Scene adapts the act and chapter to the user's change."""

import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

story_revise = importlib.import_module(f"{ROOT.name}.storyboard.story_revise")


def payload():
    scenes = [{"id": f"s{n}", "start": (n - 1) * 4, "end": n * 4, "lyrics": f"line {n}"} for n in range(1, 5)]
    layer = {
        "story_flow_version": 2,
        "story_book": {"title": "T", "premise": "P", "chapters": [
            {"id": "ch_1", "title": "Home", "setting_id": "setting_home", "paragraphs": ["Mara waits.", "Mara leaves."]}]},
        "story_settings": [{"id": "setting_home", "name": "Home", "look": ""}],
        "story_acts": [{"id": "act_1", "chapter_id": "ch_1", "title": "Waiting", "summary": "Mara waits alone.",
                        "cast_ids": ["mara"], "scene_ids": ["s1", "s2", "s3", "s4"]}],
        "story_scene_plan": [
            {"scene_id": "s1", "act_id": "act_1", "story_slice": "Mara sits.", "spot": "Kitchen", "cast_ids": ["mara"]},
            {"scene_id": "s2", "act_id": "act_1", "story_slice": "Mara stares.", "spot": "Kitchen", "mood": "lonely",
             "cast_ids": ["mara", "joe"], "cast_source": "manual", "user_note": "Joe arrives with flowers."},
            {"scene_id": "s3", "act_id": "act_1", "story_slice": "Mara cries.", "spot": "Kitchen", "cast_ids": ["mara"]},
            {"scene_id": "s4", "act_id": "act_1", "story_slice": "Mara leaves.", "spot": "Porch", "cast_ids": ["mara"]},
        ],
        "story_main_cast_id": "mara",
    }
    refs = {"subjects": [{"id": "mara", "name": "Mara", "description": "red coat"},
                         {"id": "joe", "name": "Joe", "description": "conductor"}]}
    return {"story_layer": layer, "scenes": scenes, "reference_builder": refs, "scene_id": "s2"}


class ReviseFromSceneTests(unittest.TestCase):
    def run_step(self, reply):
        calls = []

        def fake(payload_, instruction, **kwargs):
            calls.append(instruction)
            return json.dumps(reply), {"runner": "fake", "used_model": "m"}

        with patch.object(story_revise, "_run", fake):
            return story_revise.revise_story_from_scene(payload()), calls[0]

    def test_the_users_change_is_kept_and_the_act_and_chapter_adapt(self):
        result, prompt = self.run_step({
            "scene": {"story_slice": "Joe arrives with flowers; Mara freezes.", "spot": "Front Door", "mood": "stunned",
                      "performer": "Mara", "interaction": "Joe holds out the flowers; Mara does not take them."},
            "act_summary": "Mara waits until Joe arrives with flowers.",
            "chapter_paragraphs": ["Mara waits.", "Joe arrives with flowers.", "Mara leaves."],
            "other_scenes": [{"scene": 3, "reason": "Joe is in the kitchen now."}, {"scene": 2, "reason": "self"},
                             {"scene": 9, "reason": "no such scene"}],
        })
        self.assertIn("Characters in the scene: Mara, Joe", prompt)
        self.assertIn("Direction: Joe arrives with flowers.", prompt)
        layer = result["story_layer"]
        entry = next(item for item in layer["story_scene_plan"] if item["scene_id"] == "s2")
        self.assertEqual(entry["cast_ids"], ["mara", "joe"], "the user's characters stay")
        self.assertEqual((entry["cast_source"], entry["user_note"]), ("manual", "Joe arrives with flowers."))
        self.assertEqual(entry["story_slice"], "Joe arrives with flowers; Mara freezes.")
        self.assertEqual(entry["performer_id"], "mara")
        self.assertEqual(entry["interaction"], "Joe holds out the flowers; Mara does not take them.")
        self.assertEqual(layer["story_acts"][0]["summary"], "Mara waits until Joe arrives with flowers.")
        self.assertEqual(layer["story_book"]["chapters"][0]["paragraphs"][1], "Joe arrives with flowers.")
        self.assertIn("Joe arrives with flowers.", layer["story_book_text"])
        self.assertEqual(result["other_scenes"], [{"scene_id": "s3", "reason": "Joe is in the kitchen now."}])
        untouched = next(item for item in layer["story_scene_plan"] if item["scene_id"] == "s3")
        self.assertEqual(untouched["story_slice"], "Mara cries.", "other scenes are only named, not rewritten")

    def test_an_empty_reply_changes_nothing(self):
        with self.assertRaises(ValueError):
            self.run_step({"scene": {}, "act_summary": "", "chapter_paragraphs": [], "other_scenes": []})


if __name__ == "__main__":
    unittest.main()
