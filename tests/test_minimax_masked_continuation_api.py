"""Latent Continuation Masked through the Agent API: settings, scene field, prompt context and the prompt writer."""

import importlib
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
settings = importlib.import_module(f"{pkg_name}.minimax.settings_payload")
assembly = importlib.import_module(f"{pkg_name}.minimax.prompt_assembly")
shot_prompt = importlib.import_module(f"{pkg_name}.minimax.shot_prompt")
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
mm = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.minimax_prompt_orchestrator")
from test_agent_api_references_llm import Base  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_masked_continuity_is_accepted_in_every_spelling_and_saved_in_one(self):
        for spelling in ("latent_continuation_masked", "latent_masked", "Latent-Masked", "latent_masked_av"):
            with self.subTest(spelling=spelling):
                self.assertEqual(settings.validate_minimax_h3_patch({"continuity_mode": spelling}), {})
                normalized = settings.normalize_minimax_h3_settings({"continuity_mode": spelling})
                self.assertEqual(normalized["continuity_mode"], "latent_continuation_masked")
        # the older modes still work, and a made-up mode is an error instead of a silent "off"
        self.assertEqual(settings.validate_minimax_h3_patch({"continuity_mode": "latent"}), {})
        self.assertIn("continuity_mode", settings.validate_minimax_h3_patch({"continuity_mode": "sideways"}))

    def test_context_frames_take_the_masked_sizes_and_nothing_else(self):
        for frames in (16, 22, 39, 56, 90, 141, 192):
            self.assertEqual(settings.validate_minimax_h3_patch({"latent_context_frames": frames}), {})
        self.assertIn("latent_context_frames", settings.validate_minimax_h3_patch({"latent_context_frames": 33}))

    def test_the_masked_location_transition_preset_is_accepted(self):
        self.assertEqual(settings.validate_minimax_h3_patch({"location_transition_preset": "masked"}), {})
        self.assertIn("location_transition_preset", settings.validate_minimax_h3_patch({"location_transition_preset": "wipe"}))

    def test_the_schema_tells_agents_about_the_masked_options(self):
        schema = settings.minimax_h3_settings_schema()["settings"]
        self.assertIn("latent_continuation_masked", schema["continuity_mode"]["enum"])
        self.assertIn("masked", schema["location_transition_preset"]["enum"])
        self.assertEqual(schema["latent_context_frames"]["enum"], [16, 22, 39, 56, 90, 141, 192])
        for key in ("continuity_mode", "latent_context_frames", "location_transition_preset"):
            self.assertIn("masked", schema[key]["description"])

    def test_the_render_payload_carries_masked_continuity_for_one_pass_and_two_pass(self):
        for render_pass, workflow in (
            ("single", settings.WORKFLOW_SINGLE),
            ("two_pass", settings.WORKFLOW_TWO_PASS),
            ("three_pass", settings.WORKFLOW_ADVANCED),
        ):
            with self.subTest(render_pass=render_pass):
                normalized = settings.normalize_minimax_h3_settings({
                    "video_mode": "reference_to_video", "render_pass": render_pass,
                    "continuity_mode": "latent_masked", "latent_context_frames": 90,
                })
                self.assertEqual(settings.minimax_workflow_key(normalized), workflow)
                payload = settings.build_minimax_render_payload(normalized)
                self.assertEqual(payload["continuity_mode"], "latent_continuation_masked")
                self.assertEqual(payload["latent_context_frames"], 90)
                self.assertEqual(payload["minimax_h3_latent_context_frames"], 90)


class SceneFieldTests(unittest.TestCase):
    def test_the_continuation_direction_can_be_patched_on_a_scene(self):
        self.assertEqual(mutations._unsupported_scene_fields({"minimax_h3_continuation_direction": "he sits"}), [])
        self.assertIn("minimax_h3_continuation_direction", mutations._SCENE_PATCH_TEXT_FIELDS)

    def test_the_direction_start_can_be_patched_and_reset(self):
        self.assertEqual(mutations._unsupported_scene_fields({"minimax_h3_continuation_start_seconds": 1.2}), [])
        self.assertIn("minimax_h3_continuation_start_seconds", mutations._SCENE_PATCH_OPTIONAL_NUMBER_FIELDS)


class HoldTimeTests(unittest.TestCase):
    def test_the_direction_starts_half_a_second_in_unless_the_author_says_otherwise(self):
        for duration in (2.0, 4.0, 4.94, 10.0, 30.0):
            with self.subTest(duration=duration):
                self.assertEqual(assembly.continuation_hold_seconds(duration), 0.5)
                self.assertEqual(assembly.continuation_hold_seconds(duration, None), 0.5)

    def test_the_start_can_move_from_half_a_second_up_to_half_the_scene(self):
        # the same limits the Video Builder's slider uses (miniMaxH3ContinuationStartLimits)
        self.assertEqual(assembly.continuation_start_limits(4.94), (0.5, 2.4))
        self.assertEqual(assembly.continuation_start_limits(4.72), (0.5, 2.3))
        self.assertEqual(assembly.continuation_start_limits(10.0), (0.5, 5.0))
        self.assertEqual(assembly.continuation_start_limits(0.8), (0.5, 0.5))  # too short to move
        self.assertEqual(assembly.continuation_start_limits(0.0), (0.5, 0.5))

    def test_a_chosen_start_is_kept_between_the_limits(self):
        cases = {1.2: 1.2, 2.4: 2.4, 9: 2.4, 0.1: 0.5, 0: 0.5, -3: 0.5, "1.7": 1.7, "bad": 0.5, "": 0.5, float("nan"): 0.5}
        for requested, expected in cases.items():
            with self.subTest(requested=requested):
                self.assertEqual(assembly.continuation_hold_seconds(4.94, requested), expected)
        self.assertEqual(assembly.continuation_hold_seconds(0.8, 0.7), 0.5)  # the limits meet, so only 0.5 s is possible


class PromptContextTests(unittest.TestCase):
    @staticmethod
    def _session(mode):
        return {"minimax_h3_settings": {"continuity_mode": mode, "video_mode": "reference_to_video"}, "segments": [], "flux_reference_builder": {}}

    def test_a_masked_scene_gets_the_continuation_brief_with_its_direction(self):
        segment = {
            "id": "s2", "start": 10.0, "end": 14.72, "lyric_text": "I keep it moving",
            "minimax_h3_continuation_direction": "he lifts his left arm, then points at the camera",
        }
        context = assembly.build_minimax_prompt_context(segment, self._session("latent_masked"))
        continuation = context["continuation"]
        self.assertEqual(continuation["hold_seconds"], 0.5)
        self.assertEqual(continuation["start_seconds"], 0.5)
        self.assertEqual((continuation["scene_seconds"], continuation["seconds_left"]), (4.72, 4.22))
        self.assertEqual(continuation["start_limits"], {"min": 0.5, "max": 2.3})
        self.assertEqual(continuation["start_field"], "minimax_h3_continuation_start_seconds")
        self.assertEqual(continuation["direction"], "he lifts his left arm, then points at the camera")
        self.assertEqual(continuation["direction_field"], "minimax_h3_continuation_direction")
        rules = " ".join(continuation["rules"])
        self.assertIn("For the first 0.5 seconds", rules)
        self.assertIn("At about 0.5 seconds", rules)
        # the agent is told how long the scene is and that the movement must finish inside it
        self.assertIn("The scene is 4.72 seconds long, so that movement has 4.22 seconds", rules)
        self.assertIn("completely finished before the scene ends", rules)
        self.assertIn("keep singing", rules)  # the scene has lyrics
        self.assertIn("continuation.rules", context["instruction_text"])

    def test_the_authors_start_second_is_used_for_the_timing(self):
        segment = {"id": "s2", "start": 10.0, "end": 14.94, "minimax_h3_continuation_start_seconds": 1.8,
                   "minimax_h3_continuation_direction": "he turns and sits"}
        continuation = assembly.build_minimax_prompt_context(segment, self._session("latent_masked"))["continuation"]
        self.assertEqual((continuation["hold_seconds"], continuation["seconds_left"]), (1.8, 3.14))
        self.assertIn("For the first 1.8 seconds", " ".join(continuation["rules"]))
        # a value beyond half the scene is held at half the scene
        segment["minimax_h3_continuation_start_seconds"] = 4.0
        self.assertEqual(assembly.build_minimax_prompt_context(segment, self._session("latent_masked"))["continuation"]["hold_seconds"], 2.4)

    def test_the_vocal_rule_is_left_out_for_a_scene_without_lyrics_and_other_modes_get_no_brief(self):
        segment = {"id": "s2", "start": 0.0, "end": 5.0, "lyric_no_lip_sync": True}
        rules = " ".join(assembly.build_minimax_prompt_context(segment, self._session("latent_continuation_masked"))["continuation"]["rules"])
        self.assertNotIn("keep singing", rules)
        for mode in ("off", "spatial_reference"):
            self.assertNotIn("continuation", assembly.build_minimax_prompt_context(segment, self._session(mode)))


class ShotTaskTests(unittest.TestCase):
    def test_the_last_shot_of_a_saved_prompt_is_what_the_next_scene_starts_from(self):
        saved = (
            "detailed_description:\nThe target video is in a cinematic style.\n\n"
            "[Shot 1] The camera tracks backward. He walks.\n\n[Shot 2] At 00:02.000, the camera pushes in on his face.\n\n"
            "overall_soundscape:\nThe song."
        )
        self.assertEqual(shot_prompt.last_shot_text(saved), "the camera pushes in on his face.")
        self.assertEqual(shot_prompt.last_shot_text("no shots here"), "")

    def test_the_task_text_carries_the_timing_the_direction_and_the_vocal_rule(self):
        continuation = {"hold_seconds": 1.5, "scene_seconds": 4.72, "seconds_left": 3.22, "direction": "he sits down on the couch"}
        text = shot_prompt.continuation_task_text(continuation, "He walks across the roof.", has_vocals=True)
        self.assertIn("SCENE TIMING: This scene is 4.72 seconds long. The direction starts at 1.5 seconds", text)
        self.assertIn("leaves 3.22 seconds for it", text)
        self.assertIn("perform its actions in quicker succession rather than leaving any out", text)
        self.assertNotIn("soften", text)
        self.assertIn("Continuing seamlessly from the previous shot", text)
        self.assertIn("PREVIOUS SCENE'S LAST SHOT", text)
        self.assertIn("He walks across the roof.", text)
        self.assertIn('AUTHOR\'S DIRECTION FOR THIS SCENE — MANDATORY, THE FINISHED DESCRIPTION MUST CONTAIN IT: "he sits down on the couch"', text)
        self.assertIn('"For the first 1.5 seconds, ..."', text)
        self.assertIn('"At about 1.5 seconds, ..."', text)
        self.assertIn("first describe the natural movement that gets the subject there", text)
        self.assertIn("keep singing", text)
        # without a direction the scene simply advances one small step, and without vocals nothing about singing is added
        plain = shot_prompt.continuation_task_text({"hold_seconds": 1.0, "direction": ""}, "", has_vocals=False)
        self.assertIn("advance them one small natural step at a time", plain)
        self.assertNotIn("AUTHOR'S DIRECTION FOR THIS SCENE", plain)
        self.assertNotIn("singing", plain)


class PictureTaskTests(unittest.TestCase):
    CONTINUATION = {"hold_seconds": 1.5, "direction": "he sits down on the couch"}

    def test_with_the_final_frame_the_task_is_the_builders_masked_contract(self):
        text = shot_prompt.continuation_task_text(self.CONTINUATION, with_picture=True, has_vocals=True)
        self.assertIn("FRAME-TO-FRAME CONTINUITY — HIGHEST PRIORITY", text)
        self.assertIn("Attached Picture 1 is the previous rendered scene's actual final frame", text)
        self.assertIn("this scene is the very next moment of the same uninterrupted take, not a new shot", text)
        self.assertIn("then carry on exactly as the AUTHOR'S DIRECTION at the end of this scene concept says", text)
        self.assertIn("OPENING SUBJECT VISIBILITY — IMAGE-AWARE", text)
        self.assertIn("The performer is mid-performance", text)
        self.assertIn("first describe the natural movement that gets the subject there", text)
        self.assertNotIn("PREVIOUS SCENE'S LAST SHOT", text)
        # the last shot is the stand-in only when there is no picture
        stand_in = shot_prompt.continuation_task_text(self.CONTINUATION, "He walks.", with_picture=False)
        self.assertIn("PREVIOUS SCENE'S LAST SHOT", stand_in)
        self.assertNotIn("Attached Picture 1 is the previous", stand_in)

    def test_the_location_contract_follows_the_builders_presets(self):
        roof = {"id": "l1", "name": "Rooftop", "description": "A neon rooftop. Wind and signs."}
        lobby = {"id": "l2", "name": "Lobby", "description": "Polished marble."}
        contract = shot_prompt.location_continuity_contract
        self.assertEqual(contract(None, roof), "")
        same = contract(roof, roof, "masked")
        self.assertIn("LOCATION PHASE — ESTABLISHED CURRENT LOCATION: Rooftop (A neon rooftop.)", same)
        masked = contract(lobby, roof, "masked", hold_seconds=2.0, has_vocals=True)
        self.assertIn("LOCATION PHASE — MASKED CONTINUATION TRANSITION", masked)
        self.assertIn("for the first 2 seconds simply continue the opening frame's action in Rooftop (A neon rooftop.)", masked)
        self.assertIn("At about 2 seconds, begin ONE smooth, motivated movement", masked)
        self.assertIn("keeps singing the scene's lyrics on camera through the whole movement", masked)
        self.assertIn("end looking deeper into Lobby (Polished marble.)", masked)
        self.assertNotIn("keeps singing", contract(lobby, roof, "masked", has_vocals=False))
        self.assertIn("PHYSICAL THRESHOLD TURN", contract(lobby, roof, "normal"))
        self.assertIn("Apply this scene's authored transition direction: slow dolly", contract(lobby, roof, "custom", custom="slow dolly"))
        self.assertIn("CUSTOM TRANSITION FALLBACK", contract(lobby, roof, "custom"))
        for preset in ("surreal", "cinematic", "inner_world", "match", "motion", "creative_auto"):
            self.assertIn("LOCATION PHASE", contract(lobby, roof, preset))


class PromptWriterTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        refs = session["flux_reference_builder"]
        refs["locations"] = [{"id": "loc1", "name": "Rooftop", "description": "a neon rooftop", "image": {"path": ""}}]
        refs["subject_scene_map"] = {s["id"]: ["darrel"] for s in self.segments}
        refs["scene_map"] = {s["id"]: "loc1" for s in self.segments}
        session["builder_storyboard_defaults"] = {"video_style": "cinematic_realism"}
        session["minimax_h3_settings"] = {"video_mode": "reference_to_video", "continuity_mode": "latent_continuation_masked"}
        session["segments"][0]["minimax_h3_prompt"] = (
            "detailed_description:\nThe target video is in a style.\n\n[Shot 1] The camera tracks backward. <Subject 1> walks the alley."
        )
        session["segments"][1]["minimax_h3_continuation_direction"] = "he turns to the camera and sits down"
        self.write_session(session)

    def test_a_continued_scene_is_written_as_the_next_moment_of_the_previous_scene(self):
        calls = []

        def fake(payload):
            calls.append(payload)
            return {"prompt": '{"shots":[{"description":"Continuing seamlessly from the previous shot. <Subject 1> (Darrel) walks on."}]}'}

        with patch.object(mm.vid_gen, "_generate_builder_t2v_prompt", fake):
            mm.create_minimax_prompts("Song", {"scene_ids": [self.segments[1]["id"]], "replace_existing": True})
        task = calls[0]["t2i_prompt"]
        self.assertIn("CONTINUATION — HIGHEST PRIORITY", task)
        self.assertIn("The camera tracks backward. <Subject 1> walks the alley.", task)  # the previous scene's last shot
        self.assertIn("he turns to the camera and sits down", task)
        self.assertIn("THE FINISHED DESCRIPTION MUST CONTAIN IT", task)

    def _write(self, frame, replies):
        calls = []

        def fake(payload):
            calls.append(payload)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return {"prompt": reply}

        shot = '{"shots":[{"description":"Continuing seamlessly from the previous shot. <Subject 1> (Darrel) walks on."}]}'
        replies = [shot if r == "ok" else r for r in replies]
        with patch.object(mm.vid_gen, "_generate_builder_t2v_prompt", fake):
            result = mm.write_continued_scene_prompt("Song", self.segments[1]["id"], frame, self.segments[0]["id"])
        return result, calls

    def test_the_final_frame_is_shown_to_the_llm_like_the_builder_does(self):
        result, calls = self._write("C:/frames/last.png", ["ok"])
        request = calls[0]
        self.assertEqual(request["builder_instruction_key"], "minimax_h3_frame_continuity")
        self.assertTrue(request["frame_continuity_prompt"])
        self.assertEqual(request["image_references"], [{"path": "C:/frames/last.png", "frame_continuity_source": True}])
        self.assertIn("Attached Picture 1 is the previous rendered scene's actual final frame", request["t2i_prompt"])
        self.assertIn("LOCATION PHASE — ESTABLISHED CURRENT LOCATION", request["t2i_prompt"])
        self.assertTrue(result["used_previous_frame"])
        saved = self.read_session()["segments"][1]
        self.assertEqual(saved["minimax_h3_prompt_origin"], "previous_final_frame")
        self.assertEqual(saved["minimax_h3_continuity_prompt_frame_path"], "C:/frames/last.png")
        self.assertEqual(saved["minimax_h3_continuity_prompt_source_scene_id"], self.segments[0]["id"])
        self.assertTrue(saved["minimax_h3_continuity_prompt_created_at"])
        self.assertIn("Continuing seamlessly from the previous shot.", saved["minimax_h3_prompt"])

    def test_a_model_that_cannot_read_images_falls_back_to_the_previous_shot_and_says_so(self):
        result, calls = self._write("C:/frames/last.png", [RuntimeError("this model does not support image input"), "ok"])
        self.assertEqual(len(calls), 2)
        self.assertIn("image_references", calls[0])
        self.assertNotIn("image_references", calls[1])
        self.assertEqual(calls[1]["builder_instruction_key"], "minimax_h3_reference_to_video")
        self.assertIn("PREVIOUS SCENE'S LAST SHOT", calls[1]["t2i_prompt"])
        self.assertFalse(result["used_previous_frame"])
        self.assertIn("could not read the previous scene's final frame", result["warnings"][0])
        self.assertEqual(self.read_session()["segments"][1]["minimax_h3_prompt_origin"], "gemma")

    def test_the_first_scene_and_other_modes_are_written_as_before(self):
        session = self.read_session()
        session["minimax_h3_settings"]["continuity_mode"] = "off"
        self.write_session(session)
        calls = []

        def fake(payload):
            calls.append(payload)
            return {"prompt": '{"shots":[{"description":"The camera opens wide. <Subject 1> (Darrel) paces."}]}'}

        with patch.object(mm.vid_gen, "_generate_builder_t2v_prompt", fake):
            mm.create_minimax_prompts("Song", {"scene_ids": [self.segments[1]["id"]], "replace_existing": True})
        self.assertNotIn("CONTINUATION — HIGHEST PRIORITY", calls[0]["t2i_prompt"])


if __name__ == "__main__":
    unittest.main()
