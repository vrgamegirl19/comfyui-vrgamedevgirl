"""Story Mode continuations: the scene plan says which scenes carry straight on, and they become masked scenes."""

import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

story_plan = importlib.import_module(f"{ROOT.name}.storyboard.story_plan")
builder_runner = importlib.import_module(f"{ROOT.name}.llm.builder_runner")
orchestrator = importlib.import_module(f"{ROOT.name}.agent_api.orchestrator.story_mode_orchestrator")

SCENES = [{"id": f"s{n}", "start": (n - 1) * 5, "end": n * 5, "lyrics": f"line {n}"} for n in range(1, 4)]
REPLY = json.dumps({"scenes": [
    {"scene": 1, "story_slice": "She reads.", "spot": "Kitchen", "mood": "calm", "performance": "singing",
     "continues": False, "continuation": "She opens the letter."},
    {"scene": 2, "story_slice": "She cries.", "spot": "Kitchen", "mood": "sad", "performance": "singing",
     "continues": True, "continuation": "She drops the letter."},
    {"scene": 3, "story_slice": "She leaves.", "spot": "Porch", "mood": "set", "performance": "singing",
     "continues": True, "continuation": "She walks down the steps."},
]})


def layer(mode, plan=None):
    return {
        "story_flow_version": 2,
        "story_continuation_mode": mode,
        "story_book": {"chapters": [{"id": "ch_1", "title": "Home", "setting_id": "", "paragraphs": ["She reads."]}]},
        "story_acts": [{"id": "act_1", "chapter_id": "ch_1", "title": "A", "summary": "S", "cast_ids": [],
                        "scene_ids": ["s1", "s2", "s3"]}],
        "story_scene_plan": plan or [],
    }


def plan_with(mode, plan=None):
    def fake(payload, instruction, **kwargs):
        return REPLY, {}
    with patch.object(builder_runner, "_run_builder_text_llm", fake):
        result = story_plan.build_story_scene_plan({"story_layer": layer(mode, plan), "scenes": SCENES,
                                                    "spot_min_scenes": 1})
    return {entry["scene_id"]: entry for entry in result["story_layer"]["story_scene_plan"]}


class StoryContinuationTests(unittest.TestCase):
    def test_the_plan_asks_the_llm_and_keeps_its_answer_and_direction(self):
        plan = plan_with("llm")
        self.assertEqual([plan[s]["continues_previous"] for s in ("s1", "s2", "s3")], [False, True, True])
        self.assertEqual(plan["s2"]["continuation_source"], "llm")
        self.assertEqual(plan["s2"]["continuation_direction"], "She drops the letter.")
        self.assertTrue(plan["s1"]["llm_continues"] is False and plan["s3"]["llm_continues"] is True)

    def test_same_spot_mode_uses_the_spots(self):
        plan = plan_with("location")
        self.assertEqual([plan[s]["continues_previous"] for s in ("s1", "s2", "s3")], [False, True, False])
        self.assertEqual(plan["s2"]["continuation_source"], "auto")

    def test_a_manual_choice_and_its_direction_survive_a_replan(self):
        old = [{"scene_id": "s2", "act_id": "act_1", "continues_previous": False, "continuation_source": "manual",
                "continuation_direction": "Mine."}]
        plan = plan_with("llm", old)
        self.assertEqual([plan["s2"]["continues_previous"], plan["s2"]["continuation_source"]], [False, "manual"])
        self.assertEqual(plan["s2"]["continuation_direction"], "Mine.")

    def test_the_api_makes_continued_scenes_masked_and_turns_its_own_back_off(self):
        session = {
            "video_engine": "minimax_h3",
            "minimax_h3_settings": {"video_mode": "reference_to_video", "render_pass": "single", "continuity_mode": "off"},
            "segments": [{"id": "s1", "start": 0, "end": 5}, {"id": "s2", "start": 5, "end": 10},
                         {"id": "s3", "start": 10, "end": 15}],
        }
        plan = [{"scene_id": "s1"}, {"scene_id": "s2", "continues_previous": True, "continuation_direction": "Go."},
                {"scene_id": "s3", "continues_previous": True}]
        changed = orchestrator.apply_continuations_to_segments(session, plan)
        self.assertEqual(changed, ["s2", "s3"])
        s2 = session["segments"][1]
        self.assertTrue(s2["use_scene_minimax_h3_settings"])
        self.assertEqual(s2["minimax_h3_settings"]["continuity_mode"], "latent_continuation_masked")
        self.assertEqual(s2["minimax_h3_settings"]["location_transition_preset"], "masked")
        self.assertTrue(s2["minimax_h3_settings"]["continuity_prompt_from_last_frame"])
        self.assertEqual(s2["minimax_h3_continuation_direction"], "Go.")
        self.assertEqual(s2["minimax_h3_masked_quick_prev"]["locked"], False)
        self.assertEqual(orchestrator.apply_continuations_to_segments(session, plan), [], "a second run changes nothing")

        plan[1]["continues_previous"] = False
        self.assertEqual(orchestrator.apply_continuations_to_segments(session, plan), ["s2"])
        self.assertFalse(s2["use_scene_minimax_h3_settings"])
        self.assertNotIn("minimax_h3_settings", s2)
        self.assertNotIn("story_masked_continuation", s2)

    def test_the_api_leaves_modes_that_cannot_continue_and_ltx_projects_alone(self):
        plan = [{"scene_id": "s1"}, {"scene_id": "s2", "continues_previous": True}]
        segments = [{"id": "s1", "start": 0, "end": 5}, {"id": "s2", "start": 5, "end": 10}]
        i2v = {"video_engine": "minimax_h3", "minimax_h3_settings": {"video_mode": "image_to_video"},
               "segments": [dict(s) for s in segments]}
        self.assertEqual(orchestrator.apply_continuations_to_segments(i2v, plan), [])
        ltx = {"video_engine": "ltx", "segments": [dict(s) for s in segments]}
        self.assertEqual(orchestrator.apply_continuations_to_segments(ltx, plan), [])


if __name__ == "__main__":
    unittest.main()
