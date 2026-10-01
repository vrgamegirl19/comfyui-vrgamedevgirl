"""Unit tests for Agent API Phase A4 Step 6: MiniMax H3 Latents & Continuity (Section 6.10, Invariant 4)."""

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
latent_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.latent_orchestrator")
video_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.video_orchestrator")
minimax_latent_mod = importlib.import_module(f"{pkg_name}.minimax.latent_manager")
minimax_inputs_mod = importlib.import_module(f"{pkg_name}.runner.minimax_inputs")

PredecessorMissingError = errors.PredecessorMissingError
LatentStaleError = errors.LatentStaleError
SceneNotFoundError = errors.SceneNotFoundError
ValidationError = errors.ValidationError

Job = jobs_mod.Job
JobManager = jobs_mod.JobManager
JobStatus = jobs_mod.JobStatus
FakeComfyClient = orch_mod.FakeComfyClient
set_comfy_client = orch_mod.set_comfy_client
SceneLatentManager = minimax_latent_mod.SceneLatentManager


class TestAgentApiLatents(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_latents_test_")
        self.project_dir = os.path.join(self.test_dir, "TestLatentProject")
        os.makedirs(self.project_dir, exist_ok=True)

        # Create dummy video and audio files
        self.dummy_video_path = os.path.join(self.test_dir, "dummy_video.mp4")
        with open(self.dummy_video_path, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

        self.dummy_audio_path = os.path.join(self.test_dir, "dummy_audio.wav")
        import wave
        with wave.open(self.dummy_audio_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"\x00\x00\x00\x00" * 44100 * 30)

        # Initial builder session with 3 scenes
        self.session = {
            "project_name": "TestLatentProject",
            "audio_file": self.dummy_audio_path,
            "video_engine": "minimax_h3",
            "video_mode": "minimax_h3",
            "revision": 1,
            "segments": [
                {
                    "id": "scene_001",
                    "start": 0.0,
                    "end": 4.0,
                    "lyric_text": "Scene 1 lyrics",
                    "minimax_h3_prompt": "Scene 1 minimax prompt",
                },
                {
                    "id": "scene_002",
                    "start": 4.0,
                    "end": 8.0,
                    "lyric_text": "Scene 2 lyrics",
                    "minimax_h3_prompt": "Scene 2 minimax prompt",
                },
                {
                    "id": "scene_003",
                    "start": 8.0,
                    "end": 12.0,
                    "lyric_text": "Scene 3 lyrics",
                    "minimax_h3_prompt": "Scene 3 minimax prompt",
                },
            ],
            "minimax_h3_settings": {
                "continuity_mode": "latent_continuation",
            },
        }

        self.session_file = os.path.join(self.project_dir, "vrgdg_builder_session.json")
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.session, f, indent=2)

        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir

        self.fake_client = FakeComfyClient()
        set_comfy_client(self.fake_client)

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_latent_file(self, scene_number: int, token_count: int = 12, frame_count: int = 73) -> str:
        folder = os.path.join(self.project_dir, "latents")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"scene_{scene_number:03d}.latent")
        with open(path, "wb") as f:
            f.write(b"SIMULATED_RAW_LATENT_TENSORS")
        sidecar = path + ".json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({
                "scene_number": str(scene_number),
                "token_count": str(token_count),
                "frame_count": str(frame_count),
                "fps": "24.0",
                "timestamp": "1234567.89",
                "tail_padding_frames": "0",
            }, f)
        return path

    def test_get_project_latents_status(self):
        """Test listing latent status for all scenes in project."""
        # Initial: no latents
        status = latent_orch_mod.get_project_latents_status("TestLatentProject")
        self.assertEqual(status["count"], 3)
        self.assertEqual(status["dirty_count"], 0)
        self.assertFalse(status["scenes"][0]["exists"])

        # Create scene 1 latent and mark scene 2 dirty
        self._create_latent_file(1)
        SceneLatentManager.mark_dirty(self.project_dir, 2, reason="Test dirty flag")

        status2 = latent_orch_mod.get_project_latents_status("TestLatentProject")
        self.assertTrue(status2["scenes"][0]["exists"])
        self.assertEqual(status2["scenes"][0]["token_count"], 12)
        self.assertEqual(status2["scenes"][0]["frame_count"], 73)
        self.assertTrue(status2["scenes"][1]["dirty"])
        self.assertIn(2, status2["dirty_scenes"])

    def test_get_dirty_latents(self):
        """Test querying list of dirty scene numbers."""
        dirty = latent_orch_mod.get_dirty_latents("TestLatentProject")
        self.assertEqual(dirty["dirty_scenes"], [])

        SceneLatentManager.mark_dirty(self.project_dir, 2)
        SceneLatentManager.mark_dirty(self.project_dir, 3)

        dirty2 = latent_orch_mod.get_dirty_latents("TestLatentProject")
        self.assertEqual(dirty2["dirty_scenes"], [2, 3])
        self.assertEqual(dirty2["count"], 2)

    def test_get_scene_latent_status_scene1(self):
        """Scene 1 is opening scene; predecessor is never needed."""
        status = latent_orch_mod.get_scene_latent_status("TestLatentProject", "scene_001")
        self.assertFalse(status["predecessor_needed"])
        self.assertTrue(status["predecessor_exists"])
        self.assertEqual(status["predecessor_scene"], 0)
        self.assertFalse(status["exists"])

    def test_get_scene_latent_status_scene2_predecessor_check(self):
        """Scene 2 requires Scene 1 latent as predecessor."""
        # When Scene 1 has no latent
        status = latent_orch_mod.get_scene_latent_status("TestLatentProject", "scene_002")
        self.assertTrue(status["predecessor_needed"])
        self.assertEqual(status["predecessor_scene"], 1)
        self.assertFalse(status["predecessor_exists"])
        self.assertFalse(status["predecessor_dirty"])

        # When Scene 1 latent exists
        self._create_latent_file(1)
        status2 = latent_orch_mod.get_scene_latent_status("TestLatentProject", "scene_002")
        self.assertTrue(status2["predecessor_exists"])
        self.assertFalse(status2["predecessor_dirty"])

        # When Scene 1 is marked dirty
        SceneLatentManager.mark_dirty(self.project_dir, 1)
        status3 = latent_orch_mod.get_scene_latent_status("TestLatentProject", "scene_002")
        self.assertTrue(status3["predecessor_dirty"])

    def test_delete_scene_latent(self):
        """Test deleting scene latent and marking successor dirty."""
        self._create_latent_file(1)
        self._create_latent_file(2)

        res = latent_orch_mod.delete_scene_latent("TestLatentProject", "scene_001")
        self.assertTrue(res["deleted"])
        self.assertFalse(SceneLatentManager.latent_exists(self.project_dir, 1))
        # Successor (Scene 2) must be marked dirty per Invariant 4
        self.assertTrue(SceneLatentManager.is_dirty(self.project_dir, 2))

    def test_delete_all_latents(self):
        """Test deleting all latents across the project."""
        self._create_latent_file(1)
        self._create_latent_file(2)
        self._create_latent_file(3)

        res = latent_orch_mod.delete_scene_latent("TestLatentProject", "scene_001", all_latents=True)
        self.assertTrue(res["deleted"])
        self.assertEqual(res["removed_files"], 6)  # 3 .latent + 3 .latent.json
        self.assertFalse(SceneLatentManager.latent_exists(self.project_dir, 1))
        self.assertFalse(SceneLatentManager.latent_exists(self.project_dir, 2))
        self.assertFalse(SceneLatentManager.latent_exists(self.project_dir, 3))

    def test_validate_latent_continuity_invariant4(self):
        """Validate Invariant 4 error codes: PREDECESSOR_MISSING and LATENT_STALE."""
        # Scene 1 never fails
        latent_orch_mod.validate_latent_continuity(self.project_dir, 1)

        # Scene 2 fails with PredecessorMissingError if Scene 1 latent absent
        with self.assertRaises(PredecessorMissingError) as cm:
            latent_orch_mod.validate_latent_continuity(self.project_dir, 2)
        self.assertEqual(cm.exception.code, "PREDECESSOR_MISSING")
        self.assertEqual(cm.exception.status, 409)
        self.assertEqual(cm.exception.details["predecessor_scene"], 1)

        # Create Scene 1 latent and mark it dirty -> fails with LatentStaleError
        self._create_latent_file(1)
        SceneLatentManager.mark_dirty(self.project_dir, 1)
        with self.assertRaises(LatentStaleError) as cm2:
            latent_orch_mod.validate_latent_continuity(self.project_dir, 2)
        self.assertEqual(cm2.exception.code, "LATENT_STALE")
        self.assertEqual(cm2.exception.status, 409)
        self.assertEqual(cm2.exception.details["stale_scene"], 1)

        # Clean Scene 1 latent -> passes validation
        SceneLatentManager.clear_dirty(self.project_dir, 1)
        latent_orch_mod.validate_latent_continuity(self.project_dir, 2)

    def test_render_scene_video_async_latent_continuity_checks(self):
        """render_scene_video_async enforces latent continuity for MiniMax H3."""
        # Scene 2 without Scene 1 latent must fail immediately with PredecessorMissingError
        with self.assertRaises(PredecessorMissingError):
            asyncio.run(
                video_orch_mod.render_scene_video_async(
                    "TestLatentProject",
                    "scene_002",
                    params={"mode": "minimax_h3", "continuity_mode": "latent_continuation"},
                )
            )

        # Create Scene 1 latent and mark dirty -> fails with LatentStaleError
        self._create_latent_file(1)
        SceneLatentManager.mark_dirty(self.project_dir, 1)
        with self.assertRaises(LatentStaleError):
            asyncio.run(
                video_orch_mod.render_scene_video_async(
                    "TestLatentProject",
                    "scene_002",
                    params={"mode": "minimax_h3", "continuity_mode": "latent_continuation"},
                )
            )

        # Clean Scene 1 latent -> render proceeds
        SceneLatentManager.clear_dirty(self.project_dir, 1)
        self._create_latent_file(3)  # Pre-existing Scene 3 latent

        def _mock_resolve_video(*args, **kwargs):
            p = tempfile.mktemp(suffix=".mp4", dir=self.test_dir)
            with open(p, "wb") as f:
                f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")
            return p

        with patch.object(video_orch_mod, "resolve_comfy_video_path", side_effect=_mock_resolve_video):
            res = asyncio.run(
                video_orch_mod.render_scene_video_async(
                    "TestLatentProject",
                    "scene_002",
                    params={"mode": "minimax_h3", "continuity_mode": "latent_continuation"},
                )
            )
            self.assertIn("video_path", res)
            # Re-rendering Scene 2 must mark Scene 3 dirty per Invariant 4
            self.assertTrue(SceneLatentManager.is_dirty(self.project_dir, 3))

    def test_run_rebuild_dirty_latents_job(self):
        """Test rebuilding dirty chain in strict order."""
        # Setup: Latent 1 exists, scenes 2 and 3 are dirty
        self._create_latent_file(1)
        self._create_latent_file(2)
        self._create_latent_file(3)
        SceneLatentManager.mark_dirty(self.project_dir, 2)
        SceneLatentManager.mark_dirty(self.project_dir, 3)

        manager = JobManager()
        job = manager.submit_job("latents.rebuild", project_id="TestLatentProject")

        def _mock_resolve_video(*args, **kwargs):
            p = tempfile.mktemp(suffix=".mp4", dir=self.test_dir)
            with open(p, "wb") as f:
                f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")
            return p

        with patch.object(video_orch_mod, "resolve_comfy_video_path", side_effect=_mock_resolve_video):
            res = asyncio.run(
                latent_orch_mod.run_rebuild_dirty_latents_job(
                    job.id,
                    {"project_id": "TestLatentProject"},
                    event_bus=None,
                )
            )

        self.assertEqual(res["count"], 2)
        self.assertEqual(res["rebuilt_scenes"][0]["scene_number"], 2)
        self.assertEqual(res["rebuilt_scenes"][1]["scene_number"], 3)
        # All dirty flags must now be cleared
        self.assertEqual(SceneLatentManager.list_dirty(self.project_dir), [])

    def test_minimax_stage_recover(self):
        """Test discovering and backing up intermediate stage outputs."""
        scratch_dir = tempfile.mkdtemp(prefix="minimax_scratch_")
        try:
            # Create simulated stage1 and stage2 output videos in scratch
            s1_path = os.path.join(scratch_dir, "scene_0001_stage1-audio.mp4")
            with open(s1_path, "wb") as f:
                f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

            s2_path = os.path.join(scratch_dir, "scene_0001_stage2-audio.mp4")
            with open(s2_path, "wb") as f:
                f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

            recovered = latent_orch_mod.minimax_stage_recover(
                "TestLatentProject",
                "scene_001",
                payload={"output_folder": scratch_dir},
            )

            self.assertEqual(recovered["scene_number"], 1)
            self.assertEqual(recovered["stage_outputs"]["stage1_path"], os.path.abspath(s1_path))
            self.assertEqual(recovered["stage_outputs"]["stage2_path"], os.path.abspath(s2_path))
            self.assertIn("stage1", recovered["backups"])
            self.assertIn("stage2", recovered["backups"])

            backup_file = recovered["backups"]["stage1"]["backup_path"]
            self.assertTrue(os.path.isfile(backup_file))
            self.assertIn("rendered_scene_videos_backup", backup_file)
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_cleanup_minimax_output(self):
        """Test cleaning up MiniMax H3 scratch output directory."""
        import folder_paths
        root = os.path.realpath(os.path.join(folder_paths.get_output_directory(), "VRGDG_MiniMaxH3"))
        scratch_out, _ = minimax_inputs_mod._minimax_h3_output_location(self.project_dir, 1, create=True)
        self.assertTrue(os.path.isdir(scratch_out))

        res = latent_orch_mod.cleanup_minimax_output("TestLatentProject", payload={"scene_number": 1})
        self.assertTrue(res["removed"])
        self.assertFalse(os.path.isdir(scratch_out))

    def test_get_minimax_project_index(self):
        """Test generating and reading MINIMAX_PROJECT_FILES.md index."""
        index_res = latent_orch_mod.get_minimax_project_index("TestLatentProject")
        self.assertTrue(os.path.isfile(index_res["path"]))
        self.assertIn("MiniMax H3 project files", index_res["content"])
        self.assertIn("Primary MiniMax data", index_res["content"])


if __name__ == "__main__":
    unittest.main()
