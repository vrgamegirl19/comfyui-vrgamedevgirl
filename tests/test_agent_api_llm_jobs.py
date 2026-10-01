"""Unit tests for Agent API Phase A4: LLM Prompt Jobs and Instruction Presets (Section 6.7)."""

import asyncio
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
llm_jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs.llm_jobs")
instructions_mod = importlib.import_module(f"{pkg_name}.llm.builder_instructions")
cache_mod = importlib.import_module(f"{pkg_name}.llm.cache")

img_mod = importlib.import_module(f"{pkg_name}.llm.image_prompt_generation")
vid_mod = importlib.import_module(f"{pkg_name}.llm.video_prompt_generation")

Job = jobs_mod.Job
JobManager = jobs_mod.JobManager
JobStatus = jobs_mod.JobStatus
is_llm_runner_gpu = llm_jobs_mod.is_llm_runner_gpu
register_llm_job_handlers = llm_jobs_mod.register_llm_job_handlers


class TestAgentApiLlmJobs(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_llm_jobs_test_")
        self.project_dir = os.path.join(self.test_dir, "TestLLMProject")
        os.makedirs(self.project_dir, exist_ok=True)

        # Setup initial session.json with 2 scenes
        self.session = {
            "project_name": "TestLLMProject",
            "revision": 1,
            "segments": [
                {
                    "id": "scene_001",
                    "start": 0.0,
                    "end": 4.0,
                    "lyric_text": "Neon lights glowing in the rain",
                    "t2i_prompt": "",
                    "i2v_prompt": "",
                },
                {
                    "id": "scene_002",
                    "start": 4.0,
                    "end": 8.0,
                    "lyric_text": "Walking down the midnight avenue",
                    "t2i_prompt": "An existing image prompt",
                    "i2v_prompt": "",
                },
            ],
            "settings": {"text_runner": "lm_studio"},
        }
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            json.dump(self.session, f)

        # Set environment root
        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir

        self.manager = JobManager()
        register_llm_job_handlers(self.manager)

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_runner_gpu_classification(self):
        # External runners do not use local GPU
        self.assertFalse(is_llm_runner_gpu({"text_runner": "lm_studio"}))
        self.assertFalse(is_llm_runner_gpu({"text_runner": "own_server"}))
        self.assertFalse(is_llm_runner_gpu({"text_runner": "llm_api"}))

        # Local models take GPU
        self.assertTrue(is_llm_runner_gpu({"text_runner": "builtin"}))
        self.assertTrue(is_llm_runner_gpu({"text_runner": "gemma4"}))
        self.assertTrue(is_llm_runner_gpu({}))

    @patch.object(img_mod, "_generate_builder_concept_prompts")
    def test_02_concept_prompts_job(self, mock_gen):
        mock_gen.return_value = {"concepts": ["Concept A: Cyberpunk neon", "Concept B: Dark alley"]}

        async def _test():
            job = self.manager.submit_job(
                "llm.concepts",
                project_id="TestLLMProject",
                params={"story_idea": "Cyberpunk music video"},
                is_gpu=False,
            )

            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.SUCCEEDED)
            self.assertEqual(job.result["concepts"][0], "Concept A: Cyberpunk neon")

        asyncio.run(_test())

    @patch.object(img_mod, "_generate_builder_t2i_prompt")
    def test_03_scene_image_prompt_job(self, mock_gen):
        mock_gen.return_value = {"prompt": "Cinematic shot of neon reflections in rain puddle, 8k, photorealistic"}

        async def _test():
            job = self.manager.submit_job(
                "llm.scene_image_prompt",
                project_id="TestLLMProject",
                params={"scene_id": "scene_001", "mode": "zimage"},
                is_gpu=False,
            )

            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.SUCCEEDED, f"Job failed with error: {job.error}")
            self.assertIn("neon reflections", job.result["prompt"])

            # Verify prompt saved to vrgdg_builder_session.json
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                saved = json.load(f)
            seg1 = saved["segments"][0]
            self.assertEqual(seg1["t2i_prompt"], "Cinematic shot of neon reflections in rain puddle, 8k, photorealistic")
            self.assertEqual(seg1["t2i_prompt_origin"], "llm")
            self.assertGreater(saved["revision"], 1)

        asyncio.run(_test())

    @patch.object(vid_mod, "_generate_builder_i2v_prompt")
    def test_04_scene_video_prompt_job(self, mock_gen):
        mock_gen.return_value = {"prompt": "Slow camera dolly-in as rain drops ripple on asphalt"}

        async def _test():
            job = self.manager.submit_job(
                "llm.scene_video_prompt",
                project_id="TestLLMProject",
                params={"scene_id": "scene_001", "mode": "i2v"},
                is_gpu=False,
            )

            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.SUCCEEDED)
            self.assertIn("Slow camera dolly-in", job.result["prompt"])

            # Verify saved
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                saved = json.load(f)
            seg1 = saved["segments"][0]
            self.assertEqual(seg1["i2v_prompt"], "Slow camera dolly-in as rain drops ripple on asphalt")
            self.assertEqual(seg1["i2v_prompt_origin"], "llm")

        asyncio.run(_test())

    @patch.object(img_mod, "_generate_builder_t2i_prompt")
    def test_05_batch_prompts_job_with_resume_missing(self, mock_gen):
        mock_gen.return_value = {"prompt": "Generated batch prompt"}

        async def _test():
            # Scene 1 has no t2i_prompt, Scene 2 already has an existing prompt.
            # With resume_missing, only Scene 1 should be generated.
            job = self.manager.submit_job(
                "llm.batch_prompts",
                project_id="TestLLMProject",
                params={
                    "kind": "image",
                    "scope": "all",
                    "run_mode": "resume_missing",
                    "mode": "zimage",
                },
                is_gpu=False,
            )

            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.SUCCEEDED)
            self.assertEqual(job.result["processed"], 1)
            self.assertEqual(job.result["total_targets"], 1)

            # Check that scene 1 was updated, scene 2 remained unchanged
            with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["segments"][0]["t2i_prompt"], "Generated batch prompt")
            self.assertEqual(saved["segments"][1]["t2i_prompt"], "An existing image prompt")

        asyncio.run(_test())

    @patch.object(vid_mod, "_enhance_builder_video_prompt")
    def test_06_enhance_prompt_job(self, mock_enh):
        mock_enh.return_value = {"prompt": "Enhanced: ultra detailed cinematic pan"}

        async def _test():
            job = self.manager.submit_job(
                "llm.enhance_prompt",
                project_id="TestLLMProject",
                params={"scene_id": "scene_001", "draft_prompt": "simple pan"},
                is_gpu=False,
            )

            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.SUCCEEDED)
            self.assertEqual(job.result["prompt"], "Enhanced: ultra detailed cinematic pan")

        asyncio.run(_test())

    def test_07_instruction_presets_management(self):
        # 1. Get default instruction
        state = instructions_mod._get_builder_instruction({"key": "zimage_t2i", "project_folder": self.project_dir})
        self.assertEqual(state["key"], "zimage_t2i")
        self.assertIn("default_text", state)

        # 2. Save custom instruction override
        saved = instructions_mod._save_builder_instruction({
            "key": "zimage_t2i",
            "project_folder": self.project_dir,
            "scope": "all_scenes",
            "text": "Custom photography instruction for this project.",
        })
        self.assertTrue(saved["has_all_scenes_custom"])
        self.assertEqual(saved["text"], "Custom photography instruction for this project.")

        # 3. Reset override
        reset = instructions_mod._reset_builder_instruction({
            "key": "zimage_t2i",
            "project_folder": self.project_dir,
            "scope": "all_scenes",
        })
        self.assertFalse(reset["has_all_scenes_custom"])
        self.assertEqual(reset["source"], "default")

    def test_08_cache_clearing(self):
        result = cache_mod._clear_vrgdg_llm_caches(clear_cuda_cache=False, clear_hf_pipeline_cache=True)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
