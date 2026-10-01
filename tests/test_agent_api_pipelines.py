"""Unit tests for Agent API Phase A4 Step 8: Full Pipelines & Dry-Run Planning (Section 6.13)."""

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
pipeline_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.pipeline_orchestrator")
image_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.image_orchestrator")
video_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.video_orchestrator")
video_files = importlib.import_module(f"{pkg_name}.runner.video_files")

JobCancelledError = errors.JobCancelledError
SceneNotFoundError = errors.SceneNotFoundError
ValidationError = errors.ValidationError

Job = jobs_mod.Job
JobManager = jobs_mod.JobManager
JobStatus = jobs_mod.JobStatus
FakeComfyClient = orch_mod.FakeComfyClient
set_comfy_client = orch_mod.set_comfy_client


class TestAgentApiPipelines(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_pipelines_test_")
        self.project_dir = os.path.join(self.test_dir, "TestPipelineProject")
        os.makedirs(self.project_dir, exist_ok=True)

        # Create dummy audio file
        self.dummy_audio_path = os.path.join(self.test_dir, "dummy_audio.wav")
        import wave
        with wave.open(self.dummy_audio_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"\x00\x00\x00\x00" * 44100 * 10)

        # Scene 2 approved image already exists
        img_dir2 = os.path.join(self.project_dir, "scene_image_previews", "scene_0002")
        os.makedirs(img_dir2, exist_ok=True)
        self.scene2_image = os.path.join(img_dir2, "image_0002.png")
        with open(self.scene2_image, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4")

        # Initial builder session with 2 scenes:
        # Scene 1: Needs image and video
        # Scene 2: Has approved image, needs video
        self.session = {
            "project_name": "TestPipelineProject",
            "audio_file": self.dummy_audio_path,
            "video_engine": "ltx",
            "video_mode": "i2v",
            "image_model_mode": "zimage",
            "revision": 1,
            "segments": [
                {
                    "id": "scene_001",
                    "start": 0.0,
                    "end": 4.0,
                    "lyric_text": "Neon rain cascading down the streets",
                    "t2i_prompt": "",
                    "i2v_prompt": "",
                },
                {
                    "id": "scene_002",
                    "start": 4.0,
                    "end": 8.0,
                    "lyric_text": "Reflections shimmering on wet asphalt",
                    "t2i_prompt": "Wet city street reflection",
                    "i2v_prompt": "Slow camera pan across reflections",
                    "approved_image_path": self.scene2_image,
                    "image_history": [self.scene2_image],
                    "image_history_index": 0,
                },
            ],
        }

        self.session_file = os.path.join(self.project_dir, "vrgdg_builder_session.json")
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.session, f, indent=2)

        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir

        self.orig_client = orch_mod.get_comfy_client()
        self.fake_client = FakeComfyClient()
        set_comfy_client(self.fake_client)

    def tearDown(self):
        set_comfy_client(self.orig_client)
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ==========================================================================
    # 1. Dry-Run Planning (get_pipeline_plan)
    # ==========================================================================

    def test_pipeline_plan_resume_missing(self):
        """Test dry run with resume_missing mode."""
        plan = pipeline_orch_mod.get_pipeline_plan("TestPipelineProject", {"build_mode": "resume_missing"})
        self.assertTrue(plan["can_run"])
        self.assertEqual(len(plan["missing_prerequisites"]), 0)

        summary = plan["summary"]
        self.assertEqual(summary["total_target_scenes"], 2)
        # Scene 1 needs image, Scene 2 already has one
        self.assertEqual(summary["images_to_generate"], 1)
        # Both scenes need videos
        self.assertEqual(summary["videos_to_render"], 2)
        # Scene 1 needs image prompt + video prompt
        self.assertTrue(summary["prompts_to_generate"] >= 1)
        self.assertTrue(summary["estimated_gpu_seconds"] > 0)
        self.assertTrue(summary["will_stitch"])

        # Check scene 1 plan
        s1 = plan["target_scenes"][0]
        self.assertEqual(s1["scene_id"], "scene_001")
        self.assertFalse(s1["has_image"])
        self.assertFalse(s1["has_video"])
        self.assertTrue(s1["actions"]["generate_image"])
        self.assertTrue(s1["actions"]["render_video"])

        # Check scene 2 plan
        s2 = plan["target_scenes"][1]
        self.assertEqual(s2["scene_id"], "scene_002")
        self.assertTrue(s2["has_image"])
        self.assertFalse(s2["has_video"])
        self.assertFalse(s2["actions"]["generate_image"])
        self.assertTrue(s2["actions"]["render_video"])

    def test_pipeline_plan_fresh_rebuild(self):
        """Test dry run with fresh_rebuild mode."""
        plan = pipeline_orch_mod.get_pipeline_plan("TestPipelineProject", {"build_mode": "fresh_rebuild"})
        summary = plan["summary"]
        # Both scenes should generate fresh images and videos
        self.assertEqual(summary["images_to_generate"], 2)
        self.assertEqual(summary["videos_to_render"], 2)

    def test_pipeline_plan_redo_videos(self):
        """Test dry run with redo_videos mode."""
        plan = pipeline_orch_mod.get_pipeline_plan("TestPipelineProject", {"build_mode": "redo_videos"})
        summary = plan["summary"]
        # Images should be skipped
        self.assertEqual(summary["images_to_generate"], 0)
        # Videos should be rendered
        self.assertEqual(summary["videos_to_render"], 2)

    def test_pipeline_plan_missing_audio_prerequisite(self):
        """Test that missing audio file is caught in prerequisites."""
        # Point to missing audio
        self.session["audio_file"] = os.path.join(self.test_dir, "nonexistent.wav")
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.session, f, indent=2)

        plan = pipeline_orch_mod.get_pipeline_plan("TestPipelineProject", {})
        self.assertFalse(plan["can_run"])
        self.assertTrue(any("audio file" in p.lower() for p in plan["missing_prerequisites"]))

    def test_pipeline_plan_scope_filtering(self):
        """Test scope filtering in dry run."""
        plan = pipeline_orch_mod.get_pipeline_plan(
            "TestPipelineProject",
            {"scope": "all", "scene_ids": ["scene_001"]},
        )
        self.assertEqual(plan["summary"]["total_target_scenes"], 1)
        self.assertEqual(plan["target_scenes"][0]["scene_id"], "scene_001")

    # ==========================================================================
    # 2. End-to-End Build Full Video Pipeline (build_full_video)
    # ==========================================================================

    def test_build_full_video_pipeline_zimage_i2v(self):
        """Test end-to-end pipeline execution for zimage + i2v path."""
        manager = JobManager()
        job = manager.submit_job(
            "pipeline.build_full_video",
            project_id="TestPipelineProject",
            params={
                "build_mode": "resume_missing",
                "stitch": True,
            },
        )

        def _mock_generate_image(project_id, scene_id, **kwargs):
            img_p = os.path.join(self.project_dir, "scene_image_previews", scene_id, "mock_img.png")
            os.makedirs(os.path.dirname(img_p), exist_ok=True)
            with open(img_p, "wb") as f:
                f.write(b"mock_image")
            return {"image_path": img_p}

        def _mock_render_video(project_id, scene_id, **kwargs):
            v_dir = os.path.join(self.project_dir, "rendered_scene_videos")
            os.makedirs(v_dir, exist_ok=True)
            v_p = os.path.join(v_dir, f"{scene_id}.mp4")
            with open(v_p, "wb") as f:
                f.write(b"mock_video")
            # Update session directly as real render does
            with pipeline_orch_mod._BUILDER_SAVE_LOCK:
                _, s = pipeline_orch_mod._get_active_session_and_folder(project_id)
                for seg in s["segments"]:
                    if seg.get("id") == scene_id:
                        seg["video_path"] = v_p
                        seg["rendered_video_path"] = v_p
                pipeline_orch_mod._persist_session(self.project_dir, s)
            return {"video_path": v_p}

        final_mock_video = os.path.join(self.project_dir, "FINAL_VIDEO.mp4")
        with open(final_mock_video, "wb") as f:
            f.write(b"final_stitched_video")

        with patch.object(pipeline_orch_mod, "generate_scene_image_async", side_effect=_mock_generate_image), \
             patch.object(pipeline_orch_mod, "render_scene_video_async", side_effect=_mock_render_video), \
             patch.object(video_files, "_stitch_scene_videos", return_value={"final_video_path": final_mock_video}):

            res = asyncio.run(pipeline_orch_mod.run_build_full_video_job(job, manager))

        self.assertEqual(res["pipeline"], "full_video")
        self.assertEqual(res["scenes_processed"], 2)
        # Scene 1 generated an image, scene 2 already had one
        self.assertEqual(res["images_generated"], 1)
        # Both scenes rendered video
        self.assertEqual(res["videos_rendered"], 2)
        self.assertEqual(res["final_video_path"], final_mock_video)

        # Verify session was updated with prompts and videos
        with open(self.session_file, "r", encoding="utf-8") as f:
            sess = json.load(f)
        s1 = sess["segments"][0]
        s2 = sess["segments"][1]
        self.assertTrue(s1.get("t2i_prompt"))
        self.assertTrue(s1.get("i2v_prompt"))
        self.assertTrue(s1.get("video_path"))
        self.assertTrue(s2.get("video_path"))

    def test_build_full_video_pipeline_minimax_h3(self):
        """Test end-to-end pipeline execution for MiniMax H3 path."""
        self.session["video_engine"] = "minimax_h3"
        self.session["video_mode"] = "minimax_h3"
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.session, f, indent=2)

        manager = JobManager()
        job = manager.submit_job(
            "pipeline.build_full_video",
            project_id="TestPipelineProject",
            params={
                "build_mode": "fresh_rebuild",
                "stitch": False,
            },
        )

        def _mock_generate_image(project_id, scene_id, **kwargs):
            img_p = os.path.join(self.project_dir, "scene_image_previews", scene_id, "mock_img.png")
            os.makedirs(os.path.dirname(img_p), exist_ok=True)
            with open(img_p, "wb") as f:
                f.write(b"mock_image")
            return {"image_path": img_p}

        def _mock_render_video(project_id, scene_id, **kwargs):
            v_p = os.path.join(self.project_dir, "rendered_scene_videos", f"{scene_id}_minimax.mp4")
            os.makedirs(os.path.dirname(v_p), exist_ok=True)
            with open(v_p, "wb") as f:
                f.write(b"mock_minimax_video")
            return {"video_path": v_p}

        with patch.object(pipeline_orch_mod, "generate_scene_image_async", side_effect=_mock_generate_image), \
             patch.object(pipeline_orch_mod, "render_scene_video_async", side_effect=_mock_render_video):

            res = asyncio.run(pipeline_orch_mod.run_build_full_video_job(job, manager))

        self.assertEqual(res["scenes_processed"], 2)
        self.assertEqual(res["images_generated"], 2)
        self.assertEqual(res["videos_rendered"], 2)

        # Verify minimax_h3_prompt populated
        with open(self.session_file, "r", encoding="utf-8") as f:
            sess = json.load(f)
        self.assertTrue(sess["segments"][0].get("minimax_h3_prompt"))
        self.assertTrue(sess["segments"][1].get("minimax_h3_prompt"))

    def test_build_full_video_pipeline_cancellation(self):
        """Test that job cancellation is handled immediately."""
        manager = JobManager()
        job = manager.submit_job(
            "pipeline.build_full_video",
            project_id="TestPipelineProject",
            params={"build_mode": "fresh_rebuild"},
        )
        job.cancel_requested = True

        with self.assertRaises(JobCancelledError):
            asyncio.run(pipeline_orch_mod.run_build_full_video_job(job, manager))

    # ==========================================================================
    # 3. First/Last Frame Pipeline (build_flf)
    # ==========================================================================

    def test_build_flf_pipeline(self):
        """Test FLF pipeline job."""
        manager = JobManager()
        job = manager.submit_job(
            "pipeline.build_flf",
            project_id="TestPipelineProject",
            params={"stitch": True},
        )

        def _mock_generate_image(project_id, scene_id, **kwargs):
            img_p = os.path.join(self.project_dir, "scene_image_previews", scene_id, "flf_start.png")
            os.makedirs(os.path.dirname(img_p), exist_ok=True)
            with open(img_p, "wb") as f:
                f.write(b"flf_image")
            return {"image_path": img_p}

        def _mock_render_video(project_id, scene_id, **kwargs):
            v_dir = os.path.join(self.project_dir, "rendered_scene_videos")
            os.makedirs(v_dir, exist_ok=True)
            v_p = os.path.join(v_dir, f"{scene_id}_flf.mp4")
            with open(v_p, "wb") as f:
                f.write(b"flf_video")
            with pipeline_orch_mod._BUILDER_SAVE_LOCK:
                _, s = pipeline_orch_mod._get_active_session_and_folder(project_id)
                for seg in s["segments"]:
                    if seg.get("id") == scene_id:
                        seg["video_path"] = v_p
                pipeline_orch_mod._persist_session(self.project_dir, s)
            return {"video_path": v_p}

        final_mock_video = os.path.join(self.project_dir, "FINAL_FLF_VIDEO.mp4")
        with open(final_mock_video, "wb") as f:
            f.write(b"final_flf_stitched_video")

        with patch.object(pipeline_orch_mod, "generate_scene_image_async", side_effect=_mock_generate_image), \
             patch.object(pipeline_orch_mod, "render_scene_video_async", side_effect=_mock_render_video), \
             patch.object(video_files, "_stitch_scene_videos", return_value={"final_video_path": final_mock_video}):

            res = asyncio.run(pipeline_orch_mod.run_build_flf_job(job, manager))

        self.assertEqual(res["pipeline"], "flf")
        self.assertEqual(res["scenes_processed"], 2)
        self.assertEqual(res["videos_rendered"], 2)
        self.assertEqual(res["final_video_path"], final_mock_video)

    # ==========================================================================
    # 4. Handler Registration
    # ==========================================================================

    def test_register_pipeline_orchestrator_handlers(self):
        """Test registering pipeline handlers with JobManager."""
        manager = JobManager()
        pipeline_orch_mod.register_pipeline_orchestrator_handlers(manager)
        self.assertIn("pipeline.build_full_video", manager._handlers)
        self.assertIn("pipeline.build_flf", manager._handlers)


if __name__ == "__main__":
    unittest.main()
