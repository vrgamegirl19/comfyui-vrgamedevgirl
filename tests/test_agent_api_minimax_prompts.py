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


def setUpModule():
    # These tests use the project's runner settings, so the ones saved for every project on this machine are left out.
    patcher = patch.object(importlib.import_module(f"{pkg_name}.builder.project"), "_load_global_llm_runner", lambda: {})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class MiniMaxPromptTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        refs = session["flux_reference_builder"]
        refs["locations"] = [{"id": "loc1", "name": "Rooftop", "description": "a neon rooftop", "image": {"path": ""}}]
        refs["subject_scene_map"] = {s["id"]: ["darrel"] for s in self.segments}
        refs["scene_map"] = {s["id"]: "loc1" for s in self.segments}
        session["builder_storyboard_defaults"] = {"video_style": "cinematic_realism"}
        session["use_structured_outputs"] = True
        session["segments"][0]["story_beat"] = "Darrel paces the rooftop."
        self.write_session(session)

    def run_with(self, reply, params=None):
        calls = []

        def fake(payload):
            calls.append(payload)
            return {"prompt": reply(payload) if callable(reply) else reply}

        with patch.object(mm.vid_gen, "_generate_builder_t2v_prompt", fake):
            return mm.create_minimax_prompts("Song", params or {"limit": 2}), calls

    def test_theatrical_presets_reach_the_existing_api_prompt_job(self):
        session = self.read_session()
        session["builder_storyboard_defaults"].update(
            camera_flow="pretty_little_malfunction", performance_style="theatrical",
        )
        session["default_facial_performance"] = "theatrical"
        self.write_session(session)
        result, calls = self.run_with(
            '{"shots":[{"description":"<Subject 1> paces with a cheeky grin."}]}', {"limit": 1},
        )
        self.assertEqual(result["failed"], 0)
        task = calls[0]["t2i_prompt"]
        self.assertIn("Theatrical camera flow:", task)
        self.assertIn("purposeful whip-pan", task)
        self.assertIn("Theatrical performance direction:", task)
        self.assertIn("Theatrical facial performance direction:", task)
        self.assertEqual({method for method, _ in self.lm.requests}, {"GET"})
        assembly = importlib.import_module(f"{pkg_name}.minimax.prompt_assembly")
        brief = assembly.build_minimax_prompt_context(session["segments"][0], session)
        self.assertIn("Theatrical camera flow:", brief["instruction_text"])
        video_inputs = importlib.import_module(f"{pkg_name}.agent_api.video_prompt_inputs")
        request = video_inputs.scene_video_prompt_inputs(session, session["segments"][0]["id"], "t2v")
        self.assertIn("Theatrical performance direction:", request["user_notes"])

    def test_prompts_are_written_in_the_saved_format_with_the_loaded_model(self):
        result, calls = self.run_with('{"shots":[{"description":"The camera opens wide. <Subject 1> (Darrel) paces, turning at the rail."}]}')
        self.assertEqual((result["created"], result["failed"]), (2, 0))
        request = calls[0]
        self.assertEqual(request["lmstudio_model"], "gemma-4-e4b-it")
        self.assertEqual(request["builder_instruction_key"], "minimax_h3_reference_to_video")
        self.assertTrue(request["t2i_prompt"].startswith("MiniMax H3 shot-description task."))
        self.assertIn("Darrel paces the rooftop.".replace("Darrel", "<Subject 1>"), request["t2i_prompt"])
        saved = self.read_session()["segments"][0]
        prompt = saved["minimax_h3_prompt"]
        self.assertTrue(prompt.startswith("subject_definitions:\n<Subject 1> is "), prompt[:120])
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Audio 1> is the complete synchronized song and vocal track", prompt)
        self.assertIn("\n\nretention_analysis:\n", prompt)
        self.assertIn("detailed_description:\nThe target video is in a Cinematic realism music-video style: "
                      "Naturalistic practical lighting, restrained color grading, balanced contrast, subtle film grain, "
                      "realistic skin texture, neutral tones, believable materials, and polished cinematic clarity throughout.\n\n"
                      "[Shot 1] The camera opens wide. <Subject 1> (Darrel) paces, turning at the rail. "
                      '<Subject 1> (Darrel) sings <d>[English] Line 1.</d> in sync with <Audio 1>.', prompt)
        self.assertTrue(prompt.rstrip().endswith("complete audience-facing song/music track."))
        self.assertLessEqual(len(prompt), 7000)
        self.assertEqual(saved["video_prompt_type"], "rtv")
        self.assertEqual(saved["minimax_h3_prompt_origin"], "gemma")
        self.assertEqual(saved["minimax_h3_mode"], "reference_to_video")
        self.assertNotIn("minimax_h3_prompt", self.read_session()["segments"][2])
        self.assertEqual({m for m, _ in self.lm.requests}, {"GET"})

    def test_a_chosen_temporal_world_effect_is_added_last_and_fits_the_limit(self):
        session = self.read_session()
        session["builder_storyboard_defaults"]["temporal_world_effect"] = "realtime_subjects_timelapse_world"
        self.write_session(session)
        _result, calls = self.run_with('{"shots":[{"description":"The camera opens wide. <Subject 1> (Darrel) paces, turning at the rail."}]}', {"limit": 1})
        prompt = self.read_session()["segments"][0]["minimax_h3_prompt"]
        self.assertIn("Real-time characters / time-lapse world", prompt)
        self.assertTrue(prompt.rstrip().endswith("never time-stretch, reverse, or accelerate protected voices."))
        self.assertEqual(prompt.count("Temporal / World Effect"), 1)
        self.assertLessEqual(len(prompt), 7000)
        self.assertIn("MANDATORY exact temporal / world effect verbiage", calls[0]["t2i_prompt"])

    def test_a_chosen_video_style_and_fx_are_added_like_the_video_builder(self):
        session = self.read_session()
        session["builder_storyboard_defaults"].update(video_style="film_noir", fx_preset="lighting", temporal_world_effect="frozen_world")
        self.write_session(session)
        _result, calls = self.run_with('{"shots":[{"description":"The camera opens wide. <Subject 1> (Darrel) paces, turning at the rail."}]}', {"limit": 1})
        prompt = self.read_session()["segments"][0]["minimax_h3_prompt"]
        self.assertEqual(prompt.count("in a Film noir music-video style: High-contrast black-and-white imagery"), 1)
        self.assertEqual(prompt.count("High-contrast black-and-white imagery"), 1)
        self.assertLess(prompt.index("High-contrast black-and-white"), prompt.index("[Shot 1] The camera"))
        self.assertIn("FX accent: A brief practical-light flicker cascade", prompt)
        self.assertLess(prompt.index("FX accent:"), prompt.index("Temporal / World Effect"))
        self.assertTrue(prompt.rstrip().endswith("never time-stretch, reverse, or accelerate protected voices."))
        self.assertLessEqual(len(prompt), 7000)
        self.assertIn("exact video style look to the opening line", calls[0]["t2i_prompt"])

    def test_built_in_audio_prompts_have_no_audio_reference(self):
        session = self.read_session()
        session["minimax_h3_settings"] = {**(session.get("minimax_h3_settings") or {}), "audio_mode": "built_in_audio"}
        self.write_session(session)
        self.run_with('{"shots":[{"description":"A long enough shot description for the test scene."}]}', {"limit": 1})
        prompt = self.read_session()["segments"][0]["minimax_h3_prompt"]
        self.assertTrue(prompt.startswith("subject_definitions:"))
        self.assertNotIn("<Audio 1>", prompt)
        self.assertIn("MiniMax generates the native audio", prompt)

    def test_request_carries_the_actual_fractional_scene_duration(self):
        session = self.read_session()
        session["segments"][0].update(start=34.25, end=37.17)
        self.write_session(session)
        result, calls = self.run_with(
            '{"shots":[{"description":"The camera tracks as <Subject 1> walks across the rooftop."}]}',
            {"scene_ids": [session["segments"][0]["id"]]},
        )
        self.assertEqual(result["created"], 1)
        task = calls[0]["t2i_prompt"]
        self.assertIn("Duration: 2.92s", task)
        self.assertIn("Shot 1: 0–2.92s (2.92 seconds available).", task)
        self.assertIn("Reserve time for the required singing or speaking", task)
        self.assertIn("Write natural, fluent shot prose that H3 can follow literally", task)
        self.assertIn("her boot briefly crossing the foreground", task)
        self.assertIn("Keep ownership and spatial checks in your reasoning", task)
        self.assertNotIn("60 to 110", task)

    def test_default_compact_prompt_uses_pictures_in_shot_prose_and_grounding_rules(self):
        session = self.read_session()
        session.pop("use_structured_outputs", None)
        location_image = os.path.join(self.temp, "rooftop.png")
        Path(location_image).write_bytes(b"png")
        session["flux_reference_builder"]["locations"][0]["image"] = {"path": location_image}
        self.write_session(session)
        result, calls = self.run_with('{"shots":[{"description":"<Subject 1> (Darrel) walks toward the rooftop rail."}]}', {"limit": 1})
        self.assertEqual(result["failed"], 0, result)
        prompt = self.read_session()["segments"][0]["minimax_h3_prompt"]
        self.assertTrue(prompt.startswith("detailed_description:"))
        self.assertIn("[Shot 1] <Subject 1> walks", prompt)
        self.assertNotIn("<Subject 1> (", prompt)
        self.assertIn("environment from <Picture 2>", prompt)
        self.assertNotIn("<Subject 1> is", prompt)
        self.assertNotIn("subject_definitions:", prompt)
        self.assertIn("omit unsupported carryover props", calls[0]["t2i_prompt"])
        self.assertIn("introduce its appearance and physical placement", calls[0]["t2i_prompt"])

    def test_reference_composition_copy_is_retried_before_saving(self):
        count = 0

        def reply(_payload):
            nonlocal count
            count += 1
            description = ("Opening at eye level in the composition of <Picture 1>, a tight frame holds <Subject 1>."
                           if count == 1 else "An eye-level tight shot shows <Subject 1> (Darrel) turning toward the camera.")
            return json.dumps({"shots": [{"description": description}]})

        result, calls = self.run_with(reply, {"limit": 1})
        self.assertEqual(result["failed"], 0, result)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("composition of <Picture 1>", self.read_session()["segments"][0]["minimax_h3_prompt"])

    def test_lyric_free_option_preserves_instrumental_opening_and_singer_timing(self):
        session = self.read_session()
        session["omit_lyrics_from_video_prompts"] = True
        session["video_type"] = "singing"
        scene = session["segments"][0]
        scene.update(lyric_text="Secret song words", lyric_performance_mode="cue_map",
                     facial_performance="sad_wounded", lyric_cue_map=[
                         {"type": "instrumental", "start": 0, "end": 1},
                         {"type": "vocal", "start": 1, "end": 2, "text": "Secret song words",
                          "singer_id": "darrel", "singer_name": "Darrel"},
                     ])
        self.write_session(session)
        reply = json.dumps({"shots": [
            {"description": "The camera tracks left. His mouth moves."},
            {"description": 'He sings [sad, singing] with watery eyes. A dolly pushes closer.'},
        ]})
        result, calls = self.run_with(reply, {"limit": 1})
        self.assertEqual(result["failed"], 0, result)
        self.assertTrue(calls[0]["omit_lyrics_from_video_prompts"])
        prompt = self.read_session()["segments"][0]["minimax_h3_prompt"]
        self.assertNotRegex(prompt, r"Secret song words|mouth|lip|jaw|<d>")
        self.assertNotIn("sings", prompt.split("[Shot 2]")[0])
        self.assertIn("<Subject 1> sings in sync", prompt)
        self.assertIn("[sad, singing]", prompt)
        self.assertIn("during 1s–2s", prompt)
        self.assertIn("watery eyes", prompt)

    def test_the_lyric_is_in_the_prompt_in_a_d_tag_even_with_a_negative_word(self):
        session = self.read_session()
        session["segments"][0]["lyric_text"] = "Don't you ever feel like you are on your own,\nI am right here waiting"
        self.write_session(session)
        reply = ('{"shots":[{"description":"The camera pushes in. <Subject 1> (Darrel) sings the lyric line, '
                 "Don't you ever feel like you are on your own, I am right here waiting.\"}]}")
        self.run_with(reply, {"limit": 1})
        prompt = self.read_session()["segments"][0]["minimax_h3_prompt"]
        self.assertIn("<d>[English] Don't you ever feel like you are on your own, I am right here waiting.</d>", prompt)

    def test_scene_emotion_and_custom_facial_inputs_reach_llm_and_tagged_lyric_is_preserved(self):
        session = self.read_session()
        scene = session["segments"][0]
        scene.update(facial_performance="custom", facial_performance_custom="Angry",
                     emotion_expression_tags="Start angry, then end sad")
        self.write_session(session)
        reply = json.dumps({"shots": [{"description":
            '<Subject 1> sings with a challenging stare. <d>[English, angry, singing] line 1.</d> '
            'By the end his gaze falls, [sad, singing].'}]})
        result, calls = self.run_with(reply, {"limit": 1})
        self.assertEqual(result["failed"], 0, result)
        self.assertEqual(calls[0]["emotion_expression_tags"], "Start angry, then end sad")
        self.assertEqual(calls[0]["facial_performance_custom"], "Angry")
        self.assertEqual(calls[0]["lyric_text"], "line 1")
        prompt = self.read_session()["segments"][0]["minimax_h3_prompt"]
        self.assertRegex(prompt, r"<d>\[English, angry, singing\] line 1\.\s*</d>")
        self.assertNotIn('"line 1"', prompt)
        self.assertIn("By the end his gaze falls, [sad, singing]", prompt)

    def test_the_storyboard_builder_gets_the_video_prompts_and_beats(self):
        reply = '{"shots":[{"description":"A long enough shot description for the test scene."}]}'
        result, _calls = self.run_with(reply, {"limit": 2})
        self.assertTrue(result["storyboard"]["saved"], result["storyboard"])
        with open(os.path.join(self.folder, "storyboard", "storyboard.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        first = saved["scenes"][0]
        self.assertTrue(first["video_prompt"].startswith("subject_definitions:"))
        self.assertIn("detailed_description:", first["video_prompt"])
        self.assertEqual((first["status"], first["video_prompt_origin"], first["video_prompt_type"]), ("video_prompt_ready", "gemma", "rtv"))
        self.assertEqual(saved["scenes"][2]["video_prompt"], "")
        with open(os.path.join(self.folder, "prompts", "i2v_prompts.txt"), encoding="utf-8") as handle:
            self.assertIn("I2V1=subject_definitions:", handle.read())

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
