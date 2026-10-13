"""Story Mode has its own editable LLM instructions, separate from every other mode."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

instructions = importlib.import_module(f"{ROOT.name}.llm.builder_instructions")
story_book = importlib.import_module(f"{ROOT.name}.storyboard.story_book")
story_plan = importlib.import_module(f"{ROOT.name}.storyboard.story_plan")
prompts = importlib.import_module(f"{ROOT.name}.llm.prompts.story_mode")
storyboard_prompts = importlib.import_module(f"{ROOT.name}.llm.prompts.storyboard")


def payload(folder=""):
    scenes = [{"id": f"s{n}", "start": (n - 1) * 4, "end": n * 4, "lyrics": "line"} for n in range(1, 9)]
    layer = {"story_flow_version": 2, "story_book": {"title": "T", "premise": "P", "chapters": [
        {"id": "ch_1", "title": "Home", "setting_id": "setting_home", "paragraphs": ["Mara waits.", "Mara leaves."]}]},
        "story_settings": [{"id": "setting_home", "name": "Home", "look": ""}]}
    return {"story_layer": layer, "scenes": scenes, "project_folder": folder,
            "reference_builder": {"subjects": [{"id": "mara", "name": "Mara"}]}}


class StoryModeInstructionTests(unittest.TestCase):
    def test_every_story_mode_key_is_registered_with_its_default(self):
        for key, text in instructions.STORY_MODE_INSTRUCTION_DEFAULTS.items():
            self.assertEqual(instructions._BUILDER_INSTRUCTION_DEFAULTS[key], text)
            self.assertIn(key, instructions._BUILDER_INSTRUCTION_LABELS)
        self.assertEqual(instructions._BUILDER_INSTRUCTION_DEFAULTS["minimax_h3_story_mode_reference_to_video"],
                         instructions._BUILDER_INSTRUCTION_DEFAULTS["minimax_h3_reference_to_video"],
                         "the Story Mode copy starts as the standard MiniMax text")

    def test_a_project_edit_is_used_by_story_mode_only(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "project_context" / "custom_builder_instructions"
            target.mkdir(parents=True)
            (target / "story_mode_acts.txt").write_text("MY ACT RULES for about {{target_acts}} acts.\n", encoding="utf-8")
            prompts_seen = []

            def fake(payload_, instruction, *, label, **kwargs):
                prompts_seen.append(instruction)
                return json.dumps({"acts": []}), {"runner": "fake", "used_model": "m"}

            with patch.object(story_plan, "_run", fake):
                story_plan.build_story_acts(payload(folder))
            self.assertTrue(prompts_seen[0].startswith("MY ACT RULES for about 2 acts.\nReturn JSON only:"))
            self.assertEqual(story_book.story_template(payload(folder), "story_mode_book"), prompts.STORY_BOOK_TEMPLATE,
                             "an instruction the project did not edit stays the default")
        self.assertEqual(story_book.story_template(payload(), "story_mode_acts"), prompts.STORY_ACTS_TEMPLATE)

    def test_unknown_tokens_in_an_edited_template_are_left_alone(self):
        self.assertEqual(prompts.render_story_template("{{target_acts}} acts, {{nonsense}}", {"target_acts": 3}),
                         "3 acts, {{nonsense}}\n")

    def test_the_classic_beat_prompt_never_uses_the_story_mode_copy(self):
        kw = dict(flf_mode=False, beat_word_limit=80, lyric_story_strength_guidance_text="G", user_story_arc="",
                  song_story_brief="", previous_beat="", previous_lyrics="", previous_end_state="",
                  previous_carry_forward="", current_lyrics="", next_lyrics="", scene_defaults_json="{}",
                  performance_assignment_json="{}", extra_subjects_json="[]", scene_json="{}")
        story = storyboard_prompts._storyboard_scene_beat_instruction(**kw, story_mode=True, rules_template="MINE\n\n")
        classic = storyboard_prompts._storyboard_scene_beat_instruction(**kw, story_mode=False, rules_template="MINE\n\n")
        self.assertTrue(story.startswith("MINE"))
        self.assertTrue(classic.startswith("You are a music video scene-story planner."))


if __name__ == "__main__":
    unittest.main()
