"""MiniMax prompts for agents: the LLM writes shots, the builder format is applied, prompts are saved on the scenes."""

import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
mm = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.minimax_prompt_orchestrator")
from test_agent_api_references_llm import Base  # noqa: E402


class MiniMaxPromptTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        refs = session["flux_reference_builder"]
        refs["locations"] = [{"id": "loc1", "name": "Rooftop", "description": "a neon rooftop", "image": {"path": ""}}]
        refs["subject_scene_map"] = {s["id"]: ["darrel"] for s in self.segments}
        refs["scene_map"] = {s["id"]: "loc1" for s in self.segments}
        session["builder_storyboard_defaults"] = {"video_style": "cinematic_realism"}
        session["segments"][0]["story_beat"] = "Darrel paces the rooftop."
        self.write_session(session)

    def run_with(self, reply, params=None):
        calls = []

        def fake(payload):
            calls.append(payload)
            return {"prompt": reply(payload) if callable(reply) else reply}

        with patch.object(mm.vid_gen, "_generate_builder_t2v_prompt", fake):
            return mm.create_minimax_prompts("Song", params or {"limit": 2}), calls

    def test_prompts_are_written_in_the_saved_format_with_the_loaded_model(self):
        result, calls = self.run_with('{"shots":[{"description":"The camera opens wide. <Subject 1> (Darrel) paces, turning at the rail."}]}')
        self.assertEqual((result["created"], result["failed"]), (2, 0))
        request = calls[0]
        self.assertEqual(request["lmstudio_model"], "gemma-4-e4b-it")
        self.assertEqual(request["builder_instruction_key"], "minimax_h3_reference_to_video")
        self.assertTrue(request["t2i_prompt"].startswith("MiniMax H3 shot-description task."))
        self.assertIn("Darrel paces the rooftop.".replace("Darrel", "<Subject 1>"), request["t2i_prompt"])
        saved = self.read_session()["segments"][0]
        self.assertEqual(saved["minimax_h3_prompt"], "detailed_description:\nThe target video is in a cinematic_realism music-video style.\n\n"
                                                     "[Shot 1] The camera opens wide. <Subject 1> (Darrel) paces, turning at the rail. "
                                                     '<Subject 1> (Darrel) sings the lyric line, "line 1".')
        self.assertEqual(saved["video_prompt_type"], "rtv")
        self.assertEqual(saved["minimax_h3_prompt_origin"], "gemma")
        self.assertEqual(saved["minimax_h3_mode"], "reference_to_video")
        self.assertNotIn("minimax_h3_prompt", self.read_session()["segments"][2])
        self.assertEqual({m for m, _ in self.lm.requests}, {"GET"})

    def test_the_lyric_is_in_the_prompt_in_double_quotes_even_with_a_negative_word(self):
        session = self.read_session()
        session["segments"][0]["lyric_text"] = "Don't you ever feel like you are on your own,\nI am right here waiting"
        self.write_session(session)
        reply = ('{"shots":[{"description":"The camera pushes in. <Subject 1> (Darrel) sings the lyric line, '
                 "Don't you ever feel like you are on your own, I am right here waiting.\"}]}")
        self.run_with(reply, {"limit": 1})
        prompt = self.read_session()["segments"][0]["minimax_h3_prompt"]
        self.assertIn('"Don\'t you ever feel like you are on your own, I am right here waiting"', prompt)

    def test_the_storyboard_builder_gets_the_video_prompts_and_beats(self):
        reply = '{"shots":[{"description":"A long enough shot description for the test scene."}]}'
        result, _calls = self.run_with(reply, {"limit": 2})
        self.assertTrue(result["storyboard"]["saved"], result["storyboard"])
        with open(os.path.join(self.folder, "storyboard", "storyboard.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        first = saved["scenes"][0]
        self.assertTrue(first["video_prompt"].startswith("detailed_description:"))
        self.assertEqual((first["status"], first["video_prompt_origin"], first["video_prompt_type"]), ("video_prompt_ready", "gemma", "rtv"))
        self.assertEqual(saved["scenes"][2]["video_prompt"], "")
        with open(os.path.join(self.folder, "prompts", "i2v_prompts.txt"), encoding="utf-8") as handle:
            self.assertIn("I2V1=detailed_description:", handle.read())

    def test_only_missing_prompts_are_written_unless_replacing(self):
        reply = '{"shots":[{"description":"A long enough shot description for the test scene."}]}'
        self.run_with(reply, {"limit": 2})
        _result, calls = self.run_with(reply, {"limit": 1})
        self.assertEqual(calls[0]["scene_id"], "seg_2")
        _result, calls = self.run_with(reply, {"limit": 1, "replace_existing": True})
        self.assertEqual(calls[0]["scene_id"], "seg_0")

    def test_scenes_without_any_reference_image_fail_and_are_reported(self):
        session = self.read_session()
        session["flux_reference_builder"]["subjects"][0]["image"] = {"path": ""}
        self.write_session(session)
        result, calls = self.run_with('{"shots":[{"description":"A long enough shot description for the test scene."}]}')
        self.assertEqual((result["created"], result["failed"]), (0, 2))
        self.assertEqual(calls, [])
        self.assertIn("no mapped", result["failures"][0]["error"])

    def test_a_too_long_prompt_is_retried_with_a_smaller_budget(self):
        long_reply = json.dumps({"shots": [{"description": "x" * 7100 + "."}]})
        short_reply = '{"shots":[{"description":"A long enough shot description for the test scene."}]}'
        replies = iter([long_reply, short_reply])
        result, calls = self.run_with(lambda _payload: next(replies), {"limit": 1})
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["prompts"][0]["attempts"], 2)

    def test_unusable_llm_output_is_reported_not_saved(self):
        result, _calls = self.run_with("not json", {"limit": 1})
        self.assertEqual(result["failed"], 1)
        self.assertNotIn("minimax_h3_prompt", self.read_session()["segments"][0])


if __name__ == "__main__":
    unittest.main()
