"""Raw takes: the API remembers each scene's untrimmed render, lists them, and trims from one without a file path."""

import asyncio
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator")
video_orch = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.video_orchestrator")
minimax_inputs = importlib.import_module(f"{pkg_name}.runner.minimax_inputs")
video_files = importlib.import_module(f"{pkg_name}.runner.video_files")

JobStatus = jobs_mod.JobStatus
MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41"


class SceneTakesTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_takes_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)
        self.project_dir = os.path.join(self.test_dir, "TakesProject")
        self.scratch = os.path.join(self.test_dir, "output", "VRGDG_MiniMaxH3", "TakesProject_abc12345", "scene_0001")
        os.makedirs(self.project_dir)
        self.session = {
            "project_name": "TakesProject", "revision": 1,
            "segments": [
                {"id": "scene_001", "start": 0.0, "end": 4.0, "minimax_h3_prompt": "p", "i2v_prompt": "p", "video_path": "", "video_history": []},
                {"id": "scene_002", "start": 4.0, "end": 8.0, "minimax_h3_prompt": "p", "i2v_prompt": "p", "video_path": "", "video_history": []},
            ],
        }
        self.write_session()
        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir
        self.addCleanup(self._restore_env)
        patcher = patch.object(minimax_inputs, "_minimax_h3_output_location", lambda folder, number, create=True: (
            os.path.join(os.path.dirname(self.scratch), f"scene_{number:04d}"), "x"))
        patcher.start()
        self.addCleanup(patcher.stop)
        probe = patch.object(minimax_inputs, "_probe_media_duration_seconds", lambda path: 5.0)
        probe.start()
        self.addCleanup(probe.stop)
        self.manager = jobs_mod.JobManager()
        orch_mod.register_video_orchestrator_handlers(self.manager)

    def _restore_env(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)

    def write_session(self):
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)

    def read_session(self):
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def make_take(self, name, age_seconds):
        os.makedirs(self.scratch, exist_ok=True)
        path = os.path.join(self.scratch, name)
        with open(path, "wb") as handle:
            handle.write(MP4)
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))
        return path

    def run_trim(self, params):
        async def go():
            job = self.manager.submit_job("video.trim", project_id="TakesProject", params=params, is_gpu=False)
            for _ in range(100):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)
            return job
        return asyncio.run(go())

    # ---- listing -------------------------------------------------------------------------------------------------
    def test_takes_are_listed_newest_first_without_the_pass_one_backup(self):
        old = self.make_take("MiniMaxH3_scene_0001_00001-audio.mp4", 600)
        new = self.make_take("MiniMaxH3_scene_0001_00002-audio.mp4", 10)
        self.make_take("MiniMaxH3_scene_0001_stage1_00001-audio.mp4", 5)
        self.make_take("notes.txt", 5)
        listing = orch_mod.list_scene_takes("TakesProject", "scene_001")
        self.assertEqual([t["path"] for t in listing["takes"]], [new, old])
        self.assertEqual([t["index"] for t in listing["takes"]], [0, 1])
        self.assertEqual(listing["usable"], 2)
        self.assertEqual(listing["takes"][0]["duration_seconds"], 5.0)
        self.assertEqual(listing["takes"][0]["frame_count"], 120)

    def test_scenes_can_be_named_by_number_and_other_scenes_have_their_own_takes(self):
        self.make_take("MiniMaxH3_scene_0001_00001-audio.mp4", 10)
        self.assertEqual(orch_mod.list_scene_takes("TakesProject", "1")["usable"], 1)
        self.assertEqual(orch_mod.list_scene_takes("TakesProject", "scene_002")["usable"], 0)
        with self.assertRaises(errors.SceneNotFoundError):
            orch_mod.list_scene_takes("TakesProject", "nope")

    def test_a_recorded_take_whose_file_is_gone_is_listed_as_missing_with_a_reason(self):
        gone = os.path.join(self.scratch, "MiniMaxH3_scene_0001_00001-audio.mp4")
        self.session["segments"][0].update(raw_video_path=gone, raw_video_history=[gone])
        self.write_session()
        listing = orch_mod.list_scene_takes("TakesProject", "scene_001")
        self.assertEqual([(t["exists"], t["recorded"], t["is_latest_render"]) for t in listing["takes"]], [(False, True, True)])
        self.assertEqual(listing["usable"], 0)
        self.assertIn("deletes its scratch renders", listing["note"])
        with self.assertRaises(errors.ValidationError) as caught:
            video_orch.resolve_scene_take("TakesProject", "scene_001", "latest")
        self.assertIn("no raw take on disk", str(caught.exception))

    def test_takes_rendered_under_another_path_spelling_are_listed_but_not_picked_by_latest(self):
        other = os.path.join(os.path.dirname(os.path.dirname(self.scratch)), "TakesProject_deadbeef", "scene_0001")
        os.makedirs(other)
        legacy = os.path.join(other, "MiniMaxH3_scene_0001_00001-audio.mp4")
        with open(legacy, "wb") as handle:
            handle.write(MP4)
        unrelated = os.path.join(os.path.dirname(os.path.dirname(self.scratch)), "OtherProject_deadbeef", "scene_0001")
        os.makedirs(unrelated)
        with open(os.path.join(unrelated, "MiniMaxH3_scene_0001_00001-audio.mp4"), "wb") as handle:
            handle.write(MP4)
        listing = orch_mod.list_scene_takes("TakesProject", "scene_001")
        self.assertEqual([(t["path"], t["other_folder"]) for t in listing["takes"]], [(legacy, True)])
        with self.assertRaises(errors.ValidationError) as caught:
            video_orch.resolve_scene_take("TakesProject", "scene_001", "latest")
        self.assertIn("another scratch folder", str(caught.exception))
        self.assertEqual(video_orch.resolve_scene_take("TakesProject", "scene_001", 0), legacy)
        own = self.make_take("MiniMaxH3_scene_0001_00002-audio.mp4", 1)
        self.assertEqual(video_orch.resolve_scene_take("TakesProject", "scene_001", "latest"), own)

    # ---- resolving a take ----------------------------------------------------------------------------------------
    def test_a_take_is_named_by_latest_or_index_and_never_by_a_path(self):
        old = self.make_take("MiniMaxH3_scene_0001_00001-audio.mp4", 600)
        new = self.make_take("MiniMaxH3_scene_0001_00002-audio.mp4", 10)
        resolve = video_orch.resolve_scene_take
        self.assertEqual(resolve("TakesProject", "scene_001", "latest"), new)
        self.assertEqual(resolve("TakesProject", "scene_001", 1), old)
        self.assertEqual(resolve("TakesProject", "scene_001", "1"), old)
        for bad in (5, "C:/Windows/notepad.exe", "../../x.mp4"):
            with self.assertRaises(errors.ValidationError):
                resolve("TakesProject", "scene_001", bad)

    def test_a_missing_take_is_refused_by_index_too(self):
        keep = self.make_take("MiniMaxH3_scene_0001_00002-audio.mp4", 10)
        gone = os.path.join(self.scratch, "MiniMaxH3_scene_0001_00001-audio.mp4")
        self.session["segments"][0]["raw_video_history"] = [gone]
        self.write_session()
        listing = orch_mod.list_scene_takes("TakesProject", "scene_001")
        missing = next(t for t in listing["takes"] if not t["exists"])
        with self.assertRaises(errors.ValidationError) as caught:
            video_orch.resolve_scene_take("TakesProject", "scene_001", missing["index"])
        self.assertIn("file is gone", str(caught.exception))
        self.assertEqual(video_orch.resolve_scene_take("TakesProject", "scene_001", "latest"), keep)

    # ---- trimming from a take ----------------------------------------------------------------------------------
    def test_trim_from_the_latest_take_to_its_last_frame(self):
        self.make_take("MiniMaxH3_scene_0001_00001-audio.mp4", 600)
        new = self.make_take("MiniMaxH3_scene_0001_00002-audio.mp4", 10)
        target = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-retrim-audio.mp4")
        with patch.object(video_files, "_trim_scene_video", return_value={"video_path": target, "thumbnail_path": ""}) as trim:
            job = self.run_trim({"scene_id": "scene_001", "take": "latest", "start": 1.0, "to_end": True})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        sent = trim.call_args[0][0]
        self.assertEqual(sent["source_path"], new)
        self.assertEqual(sent["start"], 1.0)
        self.assertEqual(sent["frames"], 95)  # (5.0 - 1.0 - 0.01) s at 24 fps
        self.assertAlmostEqual(sent["duration"], 95 / 24)
        self.assertEqual(sent["label"], "retrim")
        self.assertTrue(sent["mark_as_audio_video"])
        self.assertEqual((job.result["source_path"], job.result["frames"]), (new, 95))
        saved = self.read_session()["segments"][0]
        self.assertEqual(saved["video_path"], target)
        self.assertIn(target, saved["video_history"])

    def test_trim_from_a_take_with_an_explicit_duration(self):
        self.make_take("MiniMaxH3_scene_0001_00001-audio.mp4", 10)
        target = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-x.mp4")
        with patch.object(video_files, "_trim_scene_video", return_value={"video_path": target, "thumbnail_path": ""}) as trim:
            job = self.run_trim({"scene_id": "scene_001", "take": 0, "start": 0.5, "duration": 3.0, "frames": 72})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        sent = trim.call_args[0][0]
        self.assertEqual((sent["start"], sent["duration"], sent["frames"]), (0.5, 3.0, 72))

    def test_trim_with_no_take_on_disk_fails_with_the_reason(self):
        job = self.run_trim({"scene_id": "scene_001", "take": "latest", "start": 0, "to_end": True})
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertIn("no raw take on disk", json.dumps(job.error))

    def test_a_source_path_still_wins_over_take_and_keeps_the_old_behaviour(self):
        source = os.path.join(self.test_dir, "other.mp4")
        with open(source, "wb") as handle:
            handle.write(MP4)
        target = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-trim.mp4")
        with patch.object(video_files, "_trim_scene_video", return_value={"video_path": target, "thumbnail_path": ""}) as trim:
            job = self.run_trim({"scene_id": "scene_001", "source_path": source, "take": "latest", "start": 0.5, "duration": 3.0})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        sent = trim.call_args[0][0]
        self.assertEqual((sent["source_path"], sent["label"], sent["mark_as_audio_video"]), (source, "trim", False))

    # ---- the render remembers its raw take ---------------------------------------------------------------------
    def test_a_render_records_the_raw_take_before_it_is_trimmed(self):
        raw = os.path.join(self.test_dir, "raw_render.mp4")
        with open(raw, "wb") as handle:
            handle.write(MP4)
        collected = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-audio.mp4")
        os.makedirs(os.path.dirname(collected), exist_ok=True)
        shutil.copy2(raw, collected)
        import wave
        audio = os.path.join(self.test_dir, "a.wav")
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(bytes(4) * 44100)
        self.session["audio_file"] = audio
        self.session["audio_path"] = audio
        self.write_session()
        orch_mod.set_comfy_client(orch_mod.FakeComfyClient())
        with patch.object(video_orch, "resolve_comfy_video_path", return_value=raw), \
                patch.object(minimax_inputs, "_prepare_scene_audio_clip", return_value={"audio_path": audio}), \
                patch.object(video_files, "_collect_scene_video", return_value={"video_path": collected, "thumbnail_path": "", "backup_path": ""}):
            async def go():
                job = self.manager.submit_job("video.render", project_id="TakesProject", params={"scene_id": "scene_001", "mode": "i2v"}, is_gpu=True)
                for _ in range(100):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)
                return job
            job = asyncio.run(go())
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        saved = self.read_session()["segments"][0]
        self.assertEqual(saved["raw_video_path"], raw)
        self.assertEqual(saved["raw_video_history"], [raw])
        listing = orch_mod.list_scene_takes("TakesProject", "scene_001")
        self.assertEqual([(t["path"], t["recorded"], t["is_latest_render"]) for t in listing["takes"]], [(raw, True, True)])


if __name__ == "__main__":
    unittest.main()
