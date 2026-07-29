import ast
import re
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "VRGDG_StoryboardBuilderNodes.py"
STORYBOARD_UI_PATH = Path(__file__).resolve().parents[1] / "web" / "VRGDG_StoryboardBuilderUI.js"
BUILDER_UI_PATH = Path(__file__).resolve().parents[1] / "web" / "VRGDG_MusicVideoBuilderUI.js"
HELPERS = {
    "_clean_scene_text",
    "_storyboard_starting_shot_value",
    "_storyboard_starting_shot_subject",
    "_storyboard_starting_shot_sentence",
    "_ensure_storyboard_starting_shot",
}


def load_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


HELPER_NAMESPACE = load_helpers()
ensure_starting_shot = HELPER_NAMESPACE["_ensure_storyboard_starting_shot"]


class StoryboardStartingShotTests(unittest.TestCase):
    def test_storyboard_payload_and_builder_bridge_include_required_opening(self):
        storyboard_source = STORYBOARD_UI_PATH.read_text(encoding="utf-8")
        builder_source = BUILDER_UI_PATH.read_text(encoding="utf-8")
        self.assertIn("starting_shot: requiresStartingShot", storyboard_source)
        self.assertIn('"REQUIRED Storyboard opening shot"', builder_source)
        self.assertIn("ensureStoryboardRequiredStartingShot(", builder_source)

    def test_eyes_shot_is_added_as_an_explicit_opening(self):
        scene = {
            "starting_shot": {
                "required": True,
                "selected_starting_shot": "eyes shot",
            },
            "visible_subjects": ["Noira"],
        }
        prompt = ensure_starting_shot(
            "Noira dances while the camera performs a full-circle orbit.",
            scene,
        )
        self.assertTrue(
            prompt.startswith(
                "The video begins with an extreme close-up of Noira's eyes."
            )
        )
        self.assertIn("full-circle orbit", prompt)

    def test_existing_explicit_eyes_opening_is_not_duplicated(self):
        scene = {
            "starting_shot": {
                "required": True,
                "selected_starting_shot": "eyes shot",
            },
            "visible_subjects": ["Noira"],
        }
        original = (
            "The video opens with an extreme close-up of Noira's eyes. "
            "The camera then performs a full-circle orbit."
        )
        self.assertEqual(ensure_starting_shot(original, scene), original)

    def test_non_required_shot_does_not_change_i2v_prompt(self):
        scene = {
            "starting_shot": None,
            "shot_type": "eyes shot",
            "visible_subjects": ["Noira"],
        }
        original = "Noira dances while the camera performs a full-circle orbit."
        self.assertEqual(ensure_starting_shot(original, scene), original)


if __name__ == "__main__":
    unittest.main()
