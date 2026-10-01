"""Unit tests for Agent API Phase A4: Video Generation and Lifecycle (Section 6.9, Section 24.2)."""

import asyncio
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator")
video_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.video_orchestrator")
media_mod = importlib.import_module(f"{pkg_name}.builder.media")
video_files_mod = importlib.import_module(f"{pkg_name}.runner.video_files")
minimax_mod = importlib.import_module(f"{pkg_name}.runner.minimax_inputs")
ltx_workflows = importlib.import_module(f"{pkg_name}.runner.ltx_workflows")
minimax_workflows = importlib.import_module(f"{pkg_name}.runner.minimax_workflows")

ComfyExecutionError = errors.ComfyExecutionError
JobCancelledError = errors.JobCancelledError
JobNotFoundError = errors.JobNotFoundError
SceneNotFoundError = errors.SceneNotFoundError
ValidationError = errors.ValidationError

Job = jobs_mod.Job
JobManager = jobs_mod.JobManager
JobStatus = jobs_mod.JobStatus

FakeComfyClient = orch_mod.FakeComfyClient
set_comfy_client = orch_mod.set_comfy_client
build_video_graph_for_mode = orch_mod.build_video_graph_for_mode
render_scene_video_async = orch_mod.render_scene_video_async
run_scene_video_render_job = orch_mod.run_scene_video_render_job
run_video_trim_job = orch_mod.run_video_trim_job
run_video_match_color_job = orch_mod.run_video_match_color_job
recover_scene_video = orch_mod.recover_scene_video
select_scene_video = orch_mod.select_scene_video
delete_scene_video = orch_mod.delete_scene_video
scan_project_scene_videos = orch_mod.scan_project_scene_videos
list_project_final_videos = orch_mod.list_project_final_videos
register_video_orchestrator_handlers = orch_mod.register_video_orchestrator_handlers


class TestAgentApiVideoJobs(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_video_jobs_test_")
        self.project_dir = os.path.join(self.test_dir, "TestVideoProject")
        os.makedirs(self.project_dir, exist_ok=True)

        # Create dummy video and audio files on disk
        self.dummy_video_path = os.path.join(self.test_dir, "dummy_video.mp4")
        with open(self.dummy_video_path, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

        self.dummy_audio_path = os.path.join(self.test_dir, "dummy_audio.wav")
        import wave
        with wave.open(self.dummy_audio_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"\x00\x00\x00\x00" * 44100 * 5)

        # Initial builder session with 2 scenes
        self.session = {
            "project_name": "TestVideoProject",
            "audio_file": self.dummy_audio_path,
            "revision": 1,
            "segments": [
                {
                    "id": "scene_001",
                    "start": 0.0,
                    "end": 4.0,
                    "i2v_prompt": "Cinematic camera orbiting a neon high-rise",
                    "video_path": "",
                    "rendered_video_path": "",
                    "thumbnail_path": "",
                    "video_history": [],
                    "video_history_index": -1,
                },
                {
                    "id": "scene_002",
                    "start": 4.0,
                    "end": 8.0,
                    "i2v_prompt": "Fast drone chase through neon alleys",
                    "video_path": "",
                    "rendered_video_path": "",
                    "thumbnail_path": "",
                    "video_history": [],
                    "video_history_index": -1,
                },
            ],
            "settings": {"video_mode": "i2v"},
        }
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            json.dump(self.session, f)

        # Set environment root
        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir

        # Setup FakeComfyClient
        self.fake_comfy = FakeComfyClient()
        set_comfy_client(self.fake_comfy)

        self.manager = JobManager()
        register_video_orchestrator_handlers(self.manager)

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(video_orch_mod, "resolve_comfy_video_path")
    def test_01_render_scene_video_job_success(self, mock_resolve):
        mock_resolve.return_value = self.dummy_video_path

        # Mock video collection and audio prep
        with patch.object(minimax_mod, "_prepare_scene_audio_clip") as mock_prep, \
             patch.object(video_files_mod, "_collect_scene_video") as mock_collect:
            mock_prep.return_value = {"audio_path": self.dummy_audio_path}
            target_vid = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-audio.mp4")
            os.makedirs(os.path.dirname(target_vid), exist_ok=True)
            shutil.copy2(self.dummy_video_path, target_vid)
            mock_collect.return_value = {
                "video_path": target_vid,
                "thumbnail_path": os.path.join(self.project_dir, "scene_video_thumbnails", "video_0001-audio.jpg"),
                "backup_path": "",
            }

            async def _test():
                job = self.manager.submit_job(
                    "video.render",
                    project_id="TestVideoProject",
                    params={"scene_id": "scene_001", "mode": "i2v"},
                    is_gpu=True,
                )

                for _ in range(50):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)

                self.assertEqual(job.status, JobStatus.SUCCEEDED, f"Job failed with error: {job.error}")
                self.assertEqual(job.progress.stage, "completed")
                self.assertEqual(job.progress.percent, 100.0)

                # Verify session state on disk
                with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                    saved_session = json.load(f)

                seg = saved_session["segments"][0]
                self.assertEqual(seg["video_path"], target_vid)
                self.assertEqual(seg["rendered_video_path"], target_vid)
                self.assertEqual(seg["preview_mode"], "video")
                self.assertEqual(len(seg["video_history"]), 1)
                self.assertEqual(seg["video_history_index"], 0)
                self.assertGreater(saved_session["revision"], 1)

            asyncio.run(_test())

    def test_02_video_trim_job(self):
        with patch.object(video_files_mod, "_trim_scene_video") as mock_trim:
            trimmed_target = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-trim.mp4")
            mock_trim.return_value = {
                "video_path": trimmed_target,
                "thumbnail_path": os.path.join(self.project_dir, "scene_video_thumbnails", "video_0001-trim.jpg"),
            }

            # Set an active video on scene 1
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                session = json.load(f)
            session["segments"][0]["video_path"] = self.dummy_video_path
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
                json.dump(session, f)

            async def _test():
                job = self.manager.submit_job(
                    "video.trim",
                    project_id="TestVideoProject",
                    params={"scene_id": "scene_001", "start": 0.5, "duration": 3.0},
                    is_gpu=False,
                )

                for _ in range(50):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)

                self.assertEqual(job.status, JobStatus.SUCCEEDED)
                self.assertEqual(job.result["video_path"], trimmed_target)

                # Verify session state
                with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                    saved_session = json.load(f)
                self.assertEqual(saved_session["segments"][0]["video_path"], trimmed_target)
                self.assertIn(trimmed_target, saved_session["segments"][0]["video_history"])

            asyncio.run(_test())

    def test_03_video_match_color_job(self):
        with patch.object(video_files_mod, "_apply_scene_start_color_match") as mock_color:
            matched_target = os.path.join(self.project_dir, "rendered_scene_videos", "video_0002-color.mp4")
            mock_color.return_value = {
                "video_path": matched_target,
                "thumbnail_path": "",
                "applied": True,
            }

            # Set video paths on scene 1 and 2
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                session = json.load(f)
            session["segments"][0]["video_path"] = self.dummy_video_path
            session["segments"][1]["video_path"] = self.dummy_video_path
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
                json.dump(session, f)

            async def _test():
                job = self.manager.submit_job(
                    "video.match_start_color",
                    project_id="TestVideoProject",
                    params={"scene_id": "scene_002"},
                    is_gpu=False,
                )

                for _ in range(50):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)

                self.assertEqual(job.status, JobStatus.SUCCEEDED)
                self.assertEqual(job.result["video_path"], matched_target)

                with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                    saved_session = json.load(f)
                self.assertEqual(saved_session["segments"][1]["video_path"], matched_target)

            asyncio.run(_test())

    def test_04_recover_scene_video(self):
        with patch.object(media_mod, "_restore_scene_video") as mock_restore:
            recovered_path = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-audio.mp4")
            mock_restore.return_value = {
                "video_path": recovered_path,
                "thumbnail_path": "",
            }

            res = recover_scene_video("TestVideoProject", "scene_001", source_path=self.dummy_video_path)
            self.assertEqual(res["video_path"], recovered_path)

            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                saved_session = json.load(f)
            self.assertEqual(saved_session["segments"][0]["video_path"], recovered_path)
            self.assertEqual(saved_session["segments"][0]["preview_mode"], "video")

    def test_05_select_scene_video(self):
        with patch.object(media_mod, "_restore_scene_video") as mock_restore:
            selected_path = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-audio.mp4")
            mock_restore.return_value = {
                "video_path": selected_path,
                "thumbnail_path": "",
            }

            res = select_scene_video("TestVideoProject", "scene_001", source_path=self.dummy_video_path)
            self.assertEqual(res["video_path"], selected_path)

            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                saved_session = json.load(f)
            self.assertEqual(saved_session["segments"][0]["video_path"], selected_path)
            self.assertEqual(saved_session["segments"][0]["preview_mode"], "video")

    def test_06_delete_scene_video(self):
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            session = json.load(f)
        session["segments"][0]["video_path"] = self.dummy_video_path
        session["segments"][0]["rendered_video_path"] = self.dummy_video_path
        session["segments"][0]["preview_mode"] = "video"
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            json.dump(session, f)

        res = delete_scene_video("TestVideoProject", "scene_001")
        self.assertTrue(res["cleared"])

        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            saved_session = json.load(f)
        self.assertEqual(saved_session["segments"][0]["video_path"], "")
        self.assertEqual(saved_session["segments"][0]["rendered_video_path"], "")
        self.assertEqual(saved_session["segments"][0]["preview_mode"], "image")

    def test_07_scan_project_scene_videos(self):
        with patch.object(media_mod, "_scan_builder_scene_videos") as mock_scan:
            mock_scan.return_value = {
                "videos": {"1": self.dummy_video_path},
                "video_backups": {"1": []},
            }
            res = scan_project_scene_videos("TestVideoProject")
            self.assertIn("videos", res)
            self.assertIn("1", res["videos"])

    def test_08_build_video_graph_dispatch(self):
        modes = [
            ("t2v", ltx_workflows, "_build_t2v_api_prompt"),
            ("rtv", ltx_workflows, "_build_rtv_api_prompt"),
            ("ingredients", ltx_workflows, "_build_ingredients_api_prompt"),
            ("flf", ltx_workflows, "_build_flf_api_prompt"),
            ("id_lora", ltx_workflows, "_build_id_lora_api_prompt"),
            ("minimax_h3", minimax_workflows, "_build_minimax_h3_api_prompt"),
            ("minimax_h3_2pass", minimax_workflows, "_build_minimax_h3_2pass_api_prompt"),
            ("minimax_h3_advanced_2pass", minimax_workflows, "_build_minimax_h3_advanced_2pass_api_prompt"),
            ("minimax_h3_3pass", minimax_workflows, "_build_minimax_h3_3pass_api_prompt"),
            ("i2v", ltx_workflows, "_build_i2v_api_prompt"),
        ]
        for mode_name, mod, func_name in modes:
            with patch.object(mod, func_name) as mock_func:
                mock_func.return_value = {"prompt": {"1": {"class_type": "MockNode"}}}
                res = build_video_graph_for_mode(mode_name, {"project_folder": self.project_dir})
                self.assertIn("prompt", res)
                mock_func.assert_called_once()

    def test_09_batch_video_render_and_stitch(self):
        # Set dummy video on scene 1, leave scene 2 empty
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            session = json.load(f)
        session["segments"][0]["video_path"] = self.dummy_video_path
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            json.dump(session, f)

        with patch.object(video_orch_mod, "resolve_comfy_video_path") as mock_res, \
             patch.object(minimax_mod, "_prepare_scene_audio_clip") as mock_prep, \
             patch.object(video_files_mod, "_collect_scene_video") as mock_collect, \
             patch.object(video_files_mod, "_stitch_scene_videos") as mock_stitch:

            mock_res.return_value = self.dummy_video_path
            mock_prep.return_value = {"audio_path": self.dummy_audio_path}
            mock_collect.return_value = {
                "video_path": self.dummy_video_path,
                "thumbnail_path": "",
            }
            mock_stitch.return_value = {
                "final_video_path": os.path.join(self.project_dir, "FINAL_VIDEO.mp4"),
                "scene_count": 2,
            }

            async def _test():
                # Test batch with missing scope: only scene 2 rendered
                job = self.manager.submit_job(
                    "videos.render_batch",
                    project_id="TestVideoProject",
                    params={"scope": "missing", "skip_final_stitch": False},
                    is_gpu=True,
                )

                for _ in range(50):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)

                self.assertEqual(job.status, JobStatus.SUCCEEDED)
                self.assertEqual(job.result["total_targets"], 1)
                self.assertEqual(job.result["processed"], 1)
                self.assertEqual(job.result["results"][0]["scene_id"], "scene_002")

            asyncio.run(_test())

    def test_10_video_stitch_job(self):
        with patch.object(video_files_mod, "_stitch_scene_videos") as mock_stitch:
            mock_stitch.return_value = {
                "final_video_path": os.path.join(self.project_dir, "FINAL_VIDEO.mp4"),
                "scene_count": 2,
            }

            # Set videos on both scenes
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                session = json.load(f)
            session["segments"][0]["video_path"] = self.dummy_video_path
            session["segments"][1]["video_path"] = self.dummy_video_path
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
                json.dump(session, f)

            async def _test():
                job = self.manager.submit_job(
                    "video.stitch",
                    project_id="TestVideoProject",
                    params={"output_prefix": "FINAL_VIDEO"},
                    is_gpu=False,
                )

                for _ in range(50):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)

                self.assertEqual(job.status, JobStatus.SUCCEEDED)
                self.assertIn("final_video_path", job.result)

            asyncio.run(_test())

    def test_11_image_slideshow_job(self):
        with patch.object(video_files_mod, "_render_image_slideshow") as mock_ss:
            mock_ss.return_value = {
                "video_path": os.path.join(self.project_dir, "slideshow.mp4"),
            }

            async def _test():
                job = self.manager.submit_job(
                    "video.slideshow",
                    project_id="TestVideoProject",
                    params={"fps": 24},
                    is_gpu=False,
                )

                for _ in range(50):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)

                self.assertEqual(job.status, JobStatus.SUCCEEDED)
                self.assertIn("video_path", job.result)

            asyncio.run(_test())

    def test_12_list_project_final_videos(self):
        # Create a final video file on disk
        final_file = os.path.join(self.project_dir, "FINAL_VIDEO.mp4")
        with open(final_file, "wb") as f:
            f.write(b"final video test content")

        finals = list_project_final_videos("TestVideoProject")
        self.assertGreaterEqual(len(finals), 1)
        self.assertEqual(finals[0]["filename"], "FINAL_VIDEO.mp4")
        self.assertEqual(finals[0]["path"], final_file)


if __name__ == "__main__":
    unittest.main()
