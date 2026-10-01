"""Unit tests for Agent API Phase A4: Image Generation and Lifecycle (Section 6.8, Section 24.1)."""

import asyncio
import base64
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator")
media_mod = importlib.import_module(f"{pkg_name}.builder.media")
paths_mod = importlib.import_module(f"{pkg_name}.runner.paths")

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
build_image_graph_for_mode = orch_mod.build_image_graph_for_mode
generate_scene_image_async = orch_mod.generate_scene_image_async
run_scene_image_generation_job = orch_mod.run_scene_image_generation_job
run_batch_image_generation_job = orch_mod.run_batch_image_generation_job
approve_scene_image = orch_mod.approve_scene_image
revert_scene_image = orch_mod.revert_scene_image
delete_scene_image = orch_mod.delete_scene_image
save_scene_image_custom = orch_mod.save_scene_image_custom
extract_frame_from_video_to_image = orch_mod.extract_frame_from_video_to_image
register_image_orchestrator_handlers = orch_mod.register_image_orchestrator_handlers


class TestAgentApiImageJobs(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_image_jobs_test_")
        self.project_dir = os.path.join(self.test_dir, "TestImageProject")
        os.makedirs(self.project_dir, exist_ok=True)

        # Create sample dummy image file on disk
        self.dummy_image_path = os.path.join(self.test_dir, "dummy_preview.png")
        img = Image.new("RGB", (64, 64), color="magenta")
        img.save(self.dummy_image_path, format="PNG")

        # Initial builder session with 3 scenes
        self.session = {
            "project_name": "TestImageProject",
            "revision": 1,
            "segments": [
                {
                    "id": "scene_001",
                    "start": 0.0,
                    "end": 4.0,
                    "t2i_prompt": "A futuristic city in the clouds at sunset",
                    "image_history": [],
                    "image_history_index": -1,
                    "approved_image_path": "",
                    "custom_image_path": "",
                },
                {
                    "id": "scene_002",
                    "start": 4.0,
                    "end": 8.0,
                    "t2i_prompt": "Flying vehicles moving between skyscrapers",
                    "image_history": [],
                    "image_history_index": -1,
                    "approved_image_path": "",
                    "custom_image_path": "",
                },
                {
                    "id": "scene_003",
                    "start": 8.0,
                    "end": 12.0,
                    "t2i_prompt": "",
                    "image_history": [],
                    "image_history_index": -1,
                    "approved_image_path": "",
                    "custom_image_path": "",
                },
            ],
            "settings": {},
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
        register_image_orchestrator_handlers(self.manager)

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_build_image_graph_for_all_modes(self):
        payload = {
            "prompt": "Test prompt",
            "width": 1024,
            "height": 576,
            "project_folder": self.project_dir,
            "api_key": "test_api_key",
            "source_image_path": self.dummy_image_path,
        }
        modes = ["zimage", "krea2", "krea2_2pass", "ernie_image", "flux_klein", "nano_banana", "z_upscale_enhance"]
        for m in modes:
            graph_res = build_image_graph_for_mode(m, payload)
            self.assertIsInstance(graph_res, dict, f"Failed for mode {m}")
            self.assertIn("prompt", graph_res, f"No 'prompt' graph returned for {m}")
            self.assertGreater(len(graph_res["prompt"]), 0, f"Empty prompt graph for {m}")

    @patch.object(media_mod, "_resolve_comfy_image_path")
    def test_02_generate_scene_image_job_success(self, mock_resolve_image):
        mock_resolve_image.return_value = self.dummy_image_path

        async def _test():
            job = self.manager.submit_job(
                "image.generate",
                project_id="TestImageProject",
                params={"scene_id": "scene_001", "mode": "zimage"},
                is_gpu=True,
            )

            # Wait for job completion
            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.SUCCEEDED)
            self.assertEqual(job.progress.stage, "completed")
            self.assertEqual(job.progress.percent, 100.0)

            # Verify session state on disk
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                saved_session = json.load(f)

            seg = saved_session["segments"][0]
            self.assertEqual(len(seg["image_history"]), 1)
            self.assertEqual(seg["image_history_index"], 0)
            saved_preview = seg["image_history"][0]
            self.assertTrue(os.path.isfile(saved_preview))
            self.assertIn("scene_image_previews", saved_preview)
            self.assertEqual(seg["approved_image_path"], "")
            self.assertEqual(seg["custom_image_path"], "")
            self.assertEqual(seg["preview_mode"], "image")
            self.assertGreater(saved_session["revision"], 1)

        asyncio.run(_test())

    def test_03_generate_scene_image_missing_prompt_fails(self):
        async def _test():
            # Scene 3 has empty prompt
            job = self.manager.submit_job(
                "image.generate",
                project_id="TestImageProject",
                params={"scene_id": "scene_003"},
                is_gpu=True,
            )

            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.FAILED)
            self.assertIn("no prompt specified", str(job.error).lower())

        asyncio.run(_test())

    @patch.object(media_mod, "_resolve_comfy_image_path")
    def test_04_batch_image_generation(self, mock_resolve_image):
        mock_resolve_image.return_value = self.dummy_image_path

        # Mark scene 1 as already having an image
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            session = json.load(f)
        session["segments"][0]["image_history"] = [self.dummy_image_path]
        session["segments"][0]["image_history_index"] = 0
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            json.dump(session, f)

        async def _test():
            # Run batch with resume_missing: should only target scene 2 (scene 3 has no prompt and will fail or skip)
            job = self.manager.submit_job(
                "images.generate_batch",
                project_id="TestImageProject",
                params={
                    "run_mode": "resume_missing",
                    "scene_ids": ["scene_001", "scene_002"],
                    "mode": "zimage",
                },
                is_gpu=True,
            )

            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.SUCCEEDED)
            res = job.result
            self.assertEqual(res["total_targets"], 1)
            self.assertEqual(res["processed"], 1)
            self.assertEqual(res["results"][0]["scene_id"], "scene_002")
            self.assertEqual(res["results"][0]["status"], "success")

            # Check that scene 2 now has an archived image
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                saved_session = json.load(f)
            self.assertEqual(len(saved_session["segments"][1]["image_history"]), 1)

        asyncio.run(_test())

    def test_05_approve_scene_image(self):
        # Setup scene 1 with 2 history items
        fake_hist_1 = os.path.join(self.project_dir, "fake_img1.png")
        fake_hist_2 = os.path.join(self.project_dir, "fake_img2.png")
        shutil.copy2(self.dummy_image_path, fake_hist_1)
        shutil.copy2(self.dummy_image_path, fake_hist_2)

        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            session = json.load(f)
        session["segments"][0]["image_history"] = [fake_hist_1, fake_hist_2]
        session["segments"][0]["image_history_index"] = 0
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            json.dump(session, f)

        # Approve without explicit image_path -> should use image_history[0]
        res = approve_scene_image("TestImageProject", "scene_001")
        self.assertEqual(res["approved_image_path"], os.path.abspath(fake_hist_1))

        # Approve with explicit image_path -> should set to fake_hist_2
        res2 = approve_scene_image("TestImageProject", "scene_001", image_path=fake_hist_2)
        self.assertEqual(res2["approved_image_path"], os.path.abspath(fake_hist_2))

        # Verify on disk
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            saved_session = json.load(f)
        self.assertEqual(saved_session["segments"][0]["approved_image_path"], os.path.abspath(fake_hist_2))

    def test_06_revert_scene_image(self):
        # Setup scene 1 with 3 history items
        h1 = os.path.join(self.project_dir, "fake_img1.png")
        h2 = os.path.join(self.project_dir, "fake_img2.png")
        h3 = os.path.join(self.project_dir, "fake_img3.png")
        for h in [h1, h2, h3]:
            shutil.copy2(self.dummy_image_path, h)

        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            session = json.load(f)
        session["segments"][0]["image_history"] = [h1, h2, h3]
        session["segments"][0]["image_history_index"] = 2  # at latest take
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            json.dump(session, f)

        # Revert delta = -1 -> index becomes 1
        res1 = revert_scene_image("TestImageProject", "scene_001", delta=-1)
        self.assertEqual(res1["history_index"], 1)
        self.assertEqual(res1["current_image"], h2)

        # Revert to specific index = 0
        res2 = revert_scene_image("TestImageProject", "scene_001", index=0)
        self.assertEqual(res2["history_index"], 0)
        self.assertEqual(res2["current_image"], h1)

    def test_07_delete_scene_image(self):
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            session = json.load(f)
        session["segments"][0]["approved_image_path"] = self.dummy_image_path
        session["segments"][0]["custom_image_path"] = self.dummy_image_path
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            json.dump(session, f)

        res = delete_scene_image("TestImageProject", "scene_001")
        self.assertTrue(res["cleared"])

        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            saved_session = json.load(f)
        self.assertEqual(saved_session["segments"][0]["approved_image_path"], "")
        self.assertEqual(saved_session["segments"][0]["custom_image_path"], "")
        self.assertTrue(saved_session["segments"][0].get("image_assignment_cleared"))

    def test_08_save_scene_image_custom(self):
        # Convert dummy image to data URL
        with open(self.dummy_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        res = save_scene_image_custom("TestImageProject", "scene_001", image_data=data_url)
        self.assertIn("custom_image_path", res)
        self.assertTrue(os.path.isfile(res["custom_image_path"]))

        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            saved_session = json.load(f)
        self.assertEqual(saved_session["segments"][0]["custom_image_path"], res["custom_image_path"])

    @patch.object(media_mod, "_extract_video_final_frame_as_scene_image")
    def test_09_extract_frame_from_video_to_image(self, mock_extract):
        mock_extract.return_value = {
            "saved_path": self.dummy_image_path,
            "scene_number": 2,
            "source_path": os.path.join(self.project_dir, "fake_video.mp4"),
        }

        # Create dummy video file on disk
        fake_vid = os.path.join(self.project_dir, "fake_video.mp4")
        with open(fake_vid, "wb") as f:
            f.write(b"fake video data")

        res = extract_frame_from_video_to_image("TestImageProject", "scene_002", source_video_path=fake_vid)
        self.assertEqual(res["saved_path"], self.dummy_image_path)

        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
            saved_session = json.load(f)
        seg = saved_session["segments"][1]
        self.assertEqual(seg["custom_image_path"], self.dummy_image_path)
        self.assertIn(self.dummy_image_path, seg["image_history"])


if __name__ == "__main__":
    unittest.main()
