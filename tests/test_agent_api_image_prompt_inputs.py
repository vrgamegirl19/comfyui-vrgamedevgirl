"""Image prompt jobs must send the scene's notes and references, not just a scene id (the writers read them from the request)."""

import asyncio
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
llm_jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs.llm_jobs")
inputs_mod = importlib.import_module(f"{pkg_name}.agent_api.image_prompt_inputs")
video_inputs_mod = importlib.import_module(f"{pkg_name}.agent_api.video_prompt_inputs")
vid_mod = importlib.import_module(f"{pkg_name}.llm.video_prompt_generation")
img_mod = importlib.import_module(f"{pkg_name}.llm.image_prompt_generation")

JobStatus = jobs_mod.JobStatus


def _session():
    return {
        "project_name": "Robot Test",
        "revision": 1,
        "builder_story_layer": {"enabled": True, "user_story_arc": "A robot learns to sing.", "song_story_brief": "Loneliness to joy.", "lyric_story_strength": 6},
        "flux_reference_builder": {
            "subjects": [{"id": "sub1", "name": "Robo", "description": "a rusty chrome robot with one blue eye"}],
            "locations": [{"id": "loc1", "name": "Junkyard", "description": "heaps of scrap under a purple sky"}],
            "subject_scene_map": {"scene_001": ["sub1"]},
            "scene_map": {"scene_001": "loc1"},
        },
        "segments": [
            {"id": "scene_001", "start": 0.0, "end": 4.0, "label": "Scene 1", "notes": "Robot stands alone at dusk, low angle.",
             "t2i_prompt": "A rusty robot at dusk", "i2v_notes": "Slow push in.",
             "lyric_text": "Neon lights glowing in the rain", "story_beat": "Opening", "shot_type": "wide"},
            {"id": "scene_002", "start": 4.0, "end": 8.0, "label": "Scene 2", "notes": "", "lyric_text": ""},
            {"id": "scene_003", "start": 8.0, "end": 12.0, "label": "Scene 3", "lyric_text": "[instrumental]", "no_character_present": True,
             "notes": "Empty street."},
        ],
    }


class SceneImageInputsTests(unittest.TestCase):
    def test_the_notes_the_agent_wrote_reach_the_writer(self):
        inputs = inputs_mod.scene_image_prompt_inputs(_session(), "scene_001", "zimage")
        notes = inputs["user_notes"]
        self.assertIn("Scene notes:\nRobot stands alone at dusk, low angle.", notes)
        self.assertIn("Mapped subject / character:\nRobo", notes)
        self.assertIn("Mapped location:\nJunkyard", notes)
        self.assertIn("Reference subject description:\nRobo: a rusty chrome robot with one blue eye", notes)
        self.assertIn("Reference location description:\nheaps of scrap under a purple sky", notes)
        self.assertIn("Lyric line as still-image mood context:\nNeon lights glowing in the rain", notes)
        self.assertIn("Scene story beat:\nOpening", notes)
        self.assertIn("Still shot direction:\nwide", notes)
        self.assertIn("User story arc:\nA robot learns to sing.", notes)
        self.assertIn("Lyric story strength:\n6/10", notes)
        self.assertTrue(notes.rstrip().endswith(inputs_mod.IMAGE_PREP_RULE))

    def test_the_instruction_preset_key_is_sent_so_a_saved_instruction_applies(self):
        for mode, key in (("zimage", "zimage_t2i"), ("ernie_image", "ernie_t2i"), ("krea2_2pass", "krea2_t2i"),
                          ("flux_klein", "flux_klein_t2i"), ("flux", "flux_klein_t2i"), ("nano_banana", "nano_b_t2i"),
                          ("nb", "nano_b_t2i"), ("flow_gpt", "flow_gpt_t2i"), ("", "zimage_t2i"), ("unknown", "zimage_t2i")):
            inputs = inputs_mod.scene_image_prompt_inputs(_session(), "scene_001", mode)
            self.assertEqual(inputs["builder_instruction_key"], key, mode)
        self.assertEqual(inputs_mod.scene_image_prompt_inputs(_session(), "scene_001", "nb")["prompt_mode"], "nano_banana")

    def test_reference_context_is_filled_for_flux_and_nano_banana(self):
        flux = inputs_mod.scene_image_prompt_inputs(_session(), "scene_001", "flux_klein")["reference_context"]
        self.assertEqual(flux["subject_description"], "Robo: a rusty chrome robot with one blue eye")
        self.assertEqual((flux["location_name"], flux["location_description"]), ("Junkyard", "heaps of scrap under a purple sky"))
        nano = inputs_mod.scene_image_prompt_inputs(_session(), "scene_001", "nano_banana")["reference_context"]
        self.assertTrue(nano["has_subject_reference"] and nano["has_location_reference"])

    def test_a_scene_with_no_notes_still_gets_a_direction_like_in_the_builder(self):
        session = _session()
        session.pop("builder_story_layer")
        notes = inputs_mod.scene_image_prompt_inputs(session, "scene_002", "zimage")["user_notes"]
        self.assertIn("Scene:\nScene 2", notes)
        self.assertIn("Direction:\nCreate a cinematic image prompt that fits this scene.", notes)

    def test_instrumental_lyrics_and_no_character_scenes_are_handled(self):
        inputs = inputs_mod.scene_image_prompt_inputs(_session(), "scene_003", "zimage")
        self.assertNotIn("Lyric line as still-image mood context", inputs["user_notes"])
        self.assertTrue(inputs["no_character_present"])
        self.assertEqual(inputs["reference_context"]["subject_description"], "")

    def test_scenes_can_be_named_by_number_and_unknown_scenes_are_rejected(self):
        self.assertIn("Robot stands alone", inputs_mod.scene_image_prompt_inputs(_session(), "1", "zimage")["user_notes"])
        with self.assertRaises(errors.ValidationError):
            inputs_mod.scene_image_prompt_inputs(_session(), "nope", "zimage")

    def test_values_sent_in_the_request_win(self):
        inputs = inputs_mod.scene_image_prompt_inputs(_session(), "scene_001", "zimage")
        merged = inputs_mod.merge_request_overrides(inputs, {"user_notes": "My own notes", "mode": "zimage", "lyric_text": ""})
        self.assertEqual(merged["user_notes"], "My own notes")
        self.assertEqual(merged["lyric_text"], "Neon lights glowing in the rain", "an empty value does not erase the scene's")
        self.assertNotIn("mode", merged)


class SceneVideoInputsTests(unittest.TestCase):
    def test_a_scene_with_an_image_prompt_sends_it_with_notes_and_references(self):
        inputs = video_inputs_mod.scene_video_prompt_inputs(_session(), "scene_001", "i2v")
        self.assertEqual(inputs["t2i_prompt"], "A rusty robot at dusk")
        self.assertIn("Slow push in.", inputs["user_notes"])
        self.assertIn("singing in sync", inputs["user_notes"])
        self.assertEqual(inputs["subject_context"], "Robo: a rusty chrome robot with one blue eye")
        self.assertEqual(inputs["location_context"], "Junkyard\nheaps of scrap under a purple sky")
        self.assertEqual(inputs["builder_instruction_key"], "i2v")
        self.assertEqual(video_inputs_mod.scene_video_prompt_inputs(_session(), "scene_001", "t2v")["builder_instruction_key"], "t2v")

    def test_a_scene_without_an_image_prompt_uses_its_scene_text(self):
        inputs = video_inputs_mod.scene_video_prompt_inputs(_session(), "scene_003", "i2v")
        self.assertIn("Empty street.", inputs["t2i_prompt"])

    def test_no_character_and_instrumental_scenes_get_the_matching_notes(self):
        inputs = video_inputs_mod.scene_video_prompt_inputs(_session(), "scene_003", "i2v")
        self.assertIn("no main character is present", inputs["user_notes"])
        self.assertEqual(inputs["subject_context"], "")
        session = _session()
        session["segments"][2]["no_character_present"] = False
        notes = video_inputs_mod.scene_video_prompt_inputs(session, "scene_003", "i2v")["user_notes"]
        self.assertIn("instrumental / no sung lyrics", notes)

    def test_visual_only_scenes_never_ask_for_singing(self):
        session = _session()
        session["segments"][0]["lyric_no_lip_sync"] = True
        inputs = video_inputs_mod.scene_video_prompt_inputs(session, "scene_001", "i2v")
        self.assertEqual(inputs["performance_mode"], "no_lip_sync")
        self.assertIn("visual-only", inputs["user_notes"])
        self.assertNotIn("singing in sync", inputs["user_notes"])

    def test_a_scene_with_nothing_to_write_from_is_rejected_with_advice(self):
        session = _session()
        session.pop("builder_story_layer")
        session["segments"][1].update({"notes": "", "lyric_text": "", "label": "Scene 2"})
        with self.assertRaises(errors.ValidationError) as caught:
            video_inputs_mod.scene_video_prompt_inputs(session, "scene_002", "i2v")
        self.assertIn("image prompt first", str(caught.exception))


class ImagePromptJobTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_image_inputs_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)
        self.project_dir = os.path.join(self.test_dir, "TestLLMProject")
        os.makedirs(self.project_dir, exist_ok=True)
        session = _session()
        session.pop("builder_story_layer")  # with a story layer every scene already has notes
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(session, handle)
        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir
        self.addCleanup(self._restore_env)
        self.manager = jobs_mod.JobManager()
        llm_jobs_mod.register_llm_job_handlers(self.manager)

    def _restore_env(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)

    def _run(self, job_type, params):
        async def go():
            job = self.manager.submit_job(job_type, project_id="TestLLMProject", params=params, is_gpu=False)
            for _ in range(100):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)
            return job
        return asyncio.run(go())

    def test_the_scene_image_prompt_job_sends_the_scene_notes_to_the_writer(self):
        with patch.object(img_mod, "_generate_builder_t2i_prompt", return_value={"prompt": "A lone robot at dusk"}) as writer:
            job = self._run("llm.scene_image_prompt", {"scene_id": "scene_001", "mode": "zimage"})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        payload = writer.call_args[0][0]
        self.assertIn("Robot stands alone at dusk", payload["user_notes"])
        self.assertEqual(payload["builder_instruction_key"], "zimage_t2i")
        self.assertEqual(payload["scene_id"], "scene_001")
        self.assertEqual(payload["project_folder"], self.project_dir)

    def test_a_request_with_only_a_scene_id_no_longer_trips_the_notes_check(self):
        """The writer refuses an empty user_notes; the job must always provide some."""
        seen = {}

        def writer(payload):
            seen.update(payload)
            if not (payload.get("user_notes") or "").strip():
                raise ValueError("Enter scene notes or provide a reference image.")
            return {"prompt": "ok"}

        with patch.object(img_mod, "_generate_builder_t2i_prompt", side_effect=writer):
            job = self._run("llm.scene_image_prompt", {"scene_id": "scene_002"})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        self.assertIn("Create a cinematic image prompt", seen["user_notes"])

    def test_flux_and_nano_banana_jobs_get_the_same_inputs(self):
        for mode, name in (("flux_klein", "_generate_flux_klein_prompt"), ("nano_banana", "_generate_nb_image_prompt")):
            with patch.object(img_mod, name, return_value={"prompt": "p"}) as writer:
                job = self._run("llm.scene_image_prompt", {"scene_id": "scene_001", "mode": mode})
            self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
            payload = writer.call_args[0][0]
            self.assertIn("Robot stands alone", payload["user_notes"], mode)
            self.assertEqual(payload["reference_context"]["location_name"], "Junkyard", mode)

    def test_the_batch_job_sends_each_scenes_own_notes(self):
        seen = []

        def writer(payload):
            seen.append((payload["scene_id"], payload["user_notes"]))
            return {"prompt": f"prompt for {payload['scene_id']}"}

        with patch.object(img_mod, "_generate_builder_t2i_prompt", side_effect=writer):
            job = self._run("llm.batch_prompts", {"kind": "image", "scope": "all", "mode": "zimage"})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        self.assertEqual([item["status"] for item in job.result["results"]], ["success"] * 3)
        by_scene = dict(seen)
        self.assertIn("Robot stands alone", by_scene["scene_001"])
        self.assertIn("Empty street.", by_scene["scene_003"])
        self.assertNotIn("Robot stands alone", by_scene["scene_003"])

    def test_notes_sent_in_the_request_are_used_instead(self):
        with patch.object(img_mod, "_generate_builder_t2i_prompt", return_value={"prompt": "p"}) as writer:
            job = self._run("llm.scene_image_prompt", {"scene_id": "scene_001", "user_notes": "Agent supplied notes"})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        self.assertEqual(writer.call_args[0][0]["user_notes"], "Agent supplied notes")


    def test_the_video_prompt_jobs_send_the_scene_inputs(self):
        for job_type, params, name in (
            ("llm.scene_video_prompt", {"scene_id": "scene_001"}, "_generate_builder_i2v_prompt"),
            ("llm.scene_video_prompt", {"scene_id": "scene_001", "mode": "t2v"}, "_generate_builder_t2v_prompt"),
            ("llm.scene_chained_video_prompt", {"scene_id": "scene_001"}, "_generate_builder_chained_i2v_prompt"),
        ):
            with patch.object(vid_mod, name, return_value={"prompt": "moving"}) as writer:
                job = self._run(job_type, params)
            self.assertEqual(job.status, JobStatus.SUCCEEDED, f"{job_type} {job.error}")
            payload = writer.call_args[0][0]
            self.assertEqual(payload["t2i_prompt"], "A rusty robot at dusk", name)
            self.assertIn("Slow push in.", payload["user_notes"], name)
            self.assertEqual(payload["location_context"].splitlines()[0], "Junkyard", name)

    def test_the_video_batch_sends_each_scenes_own_inputs(self):
        seen = {}

        def writer(payload):
            seen[payload["scene_id"]] = payload["t2i_prompt"]
            return {"prompt": "moving"}

        with patch.object(vid_mod, "_generate_builder_i2v_prompt", side_effect=writer):
            job = self._run("llm.batch_prompts", {"kind": "video", "scope": "all"})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        self.assertEqual(seen["scene_001"], "A rusty robot at dusk")
        self.assertIn("Empty street.", seen["scene_003"])


if __name__ == "__main__":
    unittest.main()
