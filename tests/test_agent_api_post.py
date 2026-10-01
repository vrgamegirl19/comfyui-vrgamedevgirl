"""Unit tests for Agent API Phase A4 Step 7: Post-Processing & Face Fix (Section 6.11, 23.1, 23.2)."""

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
post_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.post_orchestrator")
lut_video_tools = importlib.import_module(f"{pkg_name}.post_process.lut_video_tools")
face_fix_mod = importlib.import_module(f"{pkg_name}.post_process.face_fix")

JobCancelledError = errors.JobCancelledError
SceneNotFoundError = errors.SceneNotFoundError
ValidationError = errors.ValidationError

Job = jobs_mod.Job
JobManager = jobs_mod.JobManager
JobStatus = jobs_mod.JobStatus
FakeComfyClient = orch_mod.FakeComfyClient
set_comfy_client = orch_mod.set_comfy_client


class TestAgentApiPost(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_post_test_")
        self.project_dir = os.path.join(self.test_dir, "TestPostProject")
        os.makedirs(self.project_dir, exist_ok=True)

        # Create dummy video and image directories and files inside project
        v_dir = os.path.join(self.project_dir, "rendered_scene_videos")
        os.makedirs(v_dir, exist_ok=True)
        self.scene1_video = os.path.join(v_dir, "video_0001.mp4")
        with open(self.scene1_video, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

        self.scene2_video = os.path.join(v_dir, "video_0002.mp4")
        with open(self.scene2_video, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

        i_dir1 = os.path.join(self.project_dir, "scene_image_previews", "scene_0001")
        i_dir2 = os.path.join(self.project_dir, "scene_image_previews", "scene_0002")
        os.makedirs(i_dir1, exist_ok=True)
        os.makedirs(i_dir2, exist_ok=True)

        self.scene1_image = os.path.join(i_dir1, "image_0001.png")
        with open(self.scene1_image, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4")

        self.scene2_image = os.path.join(i_dir2, "image_0002.png")
        with open(self.scene2_image, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4")

        # Initial builder session
        self.session = {
            "project_name": "TestPostProject",
            "revision": 1,
            "segments": [
                {
                    "id": "scene_001",
                    "video_path": self.scene1_video,
                    "approved_image_path": self.scene1_image,
                    "video_history": [self.scene1_video],
                    "video_history_index": 0,
                    "image_history": [self.scene1_image],
                    "image_history_index": 0,
                },
                {
                    "id": "scene_002",
                    "video_path": self.scene2_video,
                    "approved_image_path": self.scene2_image,
                    "video_history": [self.scene2_video],
                    "video_history_index": 0,
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
    # 1. LUT & Preset Catalog Services
    # ==========================================================================

    def test_list_luts_service(self):
        """Test listing available LUT files."""
        with patch.object(lut_video_tools, "list_luts", return_value={"luts": ["cinematic.cube"]}):
            res = post_orch_mod.list_luts_service()
            self.assertIn("luts", res)
            self.assertEqual(res["luts"], ["cinematic.cube"])

    def test_upload_lut_service(self):
        """Test uploading a .cube LUT file with security check."""
        lut_dir = os.path.join(self.test_dir, "luts")
        with patch.object(post_orch_mod, "LUTS_DIR", lut_dir):
            # Success with .cube
            res = post_orch_mod.upload_lut_service("my_lut.cube", b"LUT_DATA_CONTENT")
            self.assertEqual(res["name"], "my_lut.cube")
            self.assertTrue(os.path.isfile(res["path"]))

            # Failure with non-.cube
            with self.assertRaises(ValidationError):
                post_orch_mod.upload_lut_service("bad_script.py", b"print('hack')")

    def test_delete_preview_service(self):
        """Test deleting a preview frame."""
        with patch.object(lut_video_tools, "delete_lut_preview", return_value=True):
            res = post_orch_mod.delete_preview_service("prev_001")
            self.assertTrue(res["deleted"])
            self.assertEqual(res["id"], "prev_001")

    def test_adjust_presets(self):
        """Test reading and writing adjust presets."""
        with patch.object(lut_video_tools, "list_adjust_presets", return_value={"presets": {"moody": {"contrast": 1.2}}}):
            presets = post_orch_mod.get_adjust_presets()
            self.assertIn("presets", presets)

        with patch.object(lut_video_tools, "save_adjust_preset", return_value={"name": "vibrant", "contrast": 1.5}):
            res = post_orch_mod.put_adjust_preset("vibrant", {"contrast": 1.5})
            self.assertIn("preset", res)

        with self.assertRaises(ValidationError):
            post_orch_mod.put_adjust_preset("", {"contrast": 1.5})

    # ==========================================================================
    # 2. Project Containment (F14 Security Rule)
    # ==========================================================================

    def test_project_containment_validation(self):
        """Test that media files outside project root are rejected (F14)."""
        outside_video = os.path.join(self.test_dir, "outside_video.mp4")
        with open(outside_video, "wb") as f:
            f.write(b"dummy")

        seg = {"video_path": outside_video}
        with self.assertRaises(ValidationError) as cm:
            post_orch_mod._resolve_scene_video_file(self.project_dir, seg, "TestPostProject", "scene_001")
        self.assertIn("inside project folder", str(cm.exception))

        # Missing file on disk
        missing_video = os.path.join(self.project_dir, "rendered_scene_videos", "nonexistent.mp4")
        seg_missing = {"video_path": missing_video}
        with self.assertRaises(FileNotFoundError):
            post_orch_mod._resolve_scene_video_file(self.project_dir, seg_missing, "TestPostProject", "scene_001")

        # Empty video path
        with self.assertRaises(ValidationError):
            post_orch_mod._resolve_scene_video_file(self.project_dir, {}, "TestPostProject", "scene_001")

    # ==========================================================================
    # 3. Preview Services (Mode S)
    # ==========================================================================

    def test_preview_scene_lut_video_and_image(self):
        """Test previewing LUT on video and image."""
        with patch.object(lut_video_tools, "preview_lut_on_media", return_value={"preview_id": "p1", "preview_path": "p1.jpg"}):
            # Video mode
            res_v = post_orch_mod.preview_scene_lut("TestPostProject", "scene_001", {"lut_name": "test.cube", "media_type": "video"})
            self.assertEqual(res_v["preview_id"], "p1")

            # Image mode
            res_i = post_orch_mod.preview_scene_lut("TestPostProject", "scene_001", {"lut_name": "test.cube", "media_type": "image"})
            self.assertEqual(res_i["preview_id"], "p1")

        # Scene not found
        with self.assertRaises(SceneNotFoundError):
            post_orch_mod.preview_scene_lut("TestPostProject", "scene_999", {})

    def test_preview_scene_grain_and_adjust(self):
        """Test previewing film grain and adjustments."""
        with patch.object(lut_video_tools, "preview_film_grain_on_media", return_value={"preview_id": "g1"}):
            res_g = post_orch_mod.preview_scene_grain("TestPostProject", "scene_001", {"grain_intensity": 0.05})
            self.assertEqual(res_g["preview_id"], "g1")

        with patch.object(lut_video_tools, "preview_adjust_on_media", return_value={"preview_id": "a1"}):
            res_a = post_orch_mod.preview_scene_adjust("TestPostProject", "scene_001", {"settings": {"brightness": 0.1}})
            self.assertEqual(res_a["preview_id"], "a1")

    # ==========================================================================
    # 4. Post-Processing Job Handlers (Mode J)
    # ==========================================================================

    def test_run_post_lut_job_video(self):
        """Test applying LUT to scene video and verifying history append."""
        manager = JobManager()
        job = manager.submit_job(
            "post.lut",
            project_id="TestPostProject",
            params={"scene_id": "scene_001", "lut_name": "teal_orange.cube", "media_type": "video"},
        )

        def _mock_apply_lut_to_video(**kwargs):
            out_p = kwargs["output_path"]
            with open(out_p, "wb") as f:
                f.write(b"lut_processed_video")
            return {"thumbnail_path": out_p + ".thumb.jpg"}

        with patch.object(lut_video_tools, "apply_lut_to_video", side_effect=_mock_apply_lut_to_video):
            res = asyncio.run(post_orch_mod.run_post_lut_job(job, manager))

        self.assertEqual(res["scene_id"], "scene_001")
        self.assertEqual(res["history_index"], 1)
        self.assertEqual(res["history_count"], 2)
        self.assertTrue(os.path.isfile(res["output_path"]))

        # Verify session persistence
        with open(self.session_file, "r", encoding="utf-8") as f:
            sess = json.load(f)
        seg0 = sess["segments"][0]
        self.assertEqual(seg0["video_path"], res["output_path"])
        self.assertEqual(len(seg0["video_history"]), 2)
        self.assertEqual(seg0["video_history_index"], 1)
        self.assertEqual(sess["revision"], 2)

    def test_run_post_lut_job_image(self):
        """Test applying LUT to scene image and verifying image history append."""
        manager = JobManager()
        job = manager.submit_job(
            "post.lut",
            project_id="TestPostProject",
            params={"scene_id": "scene_001", "lut_name": "vintage.cube", "media_type": "image"},
        )

        def _mock_apply_lut_to_image(**kwargs):
            out_p = kwargs["output_path"]
            with open(out_p, "wb") as f:
                f.write(b"lut_processed_image")
            return {"applied": True}

        with patch.object(lut_video_tools, "apply_lut_to_image", side_effect=_mock_apply_lut_to_image):
            res = asyncio.run(post_orch_mod.run_post_lut_job(job, manager))

        self.assertEqual(res["scene_id"], "scene_001")
        self.assertTrue(os.path.isfile(res["output_path"]))

        with open(self.session_file, "r", encoding="utf-8") as f:
            sess = json.load(f)
        seg0 = sess["segments"][0]
        self.assertEqual(seg0["custom_image_path"], res["output_path"])
        self.assertEqual(len(seg0["image_history"]), 2)

    def test_run_post_grain_job(self):
        """Test film grain job on video."""
        manager = JobManager()
        job = manager.submit_job(
            "post.film_grain",
            project_id="TestPostProject",
            params={"scene_id": "scene_002", "grain_intensity": 0.05, "media_type": "video"},
        )

        def _mock_apply_grain_to_video(**kwargs):
            out_p = kwargs["output_path"]
            with open(out_p, "wb") as f:
                f.write(b"grain_video")
            return {"thumbnail_path": out_p + ".thumb.jpg"}

        with patch.object(lut_video_tools, "apply_film_grain_to_video", side_effect=_mock_apply_grain_to_video):
            res = asyncio.run(post_orch_mod.run_post_grain_job(job, manager))

        self.assertEqual(res["scene_id"], "scene_002")
        self.assertTrue(os.path.isfile(res["output_path"]))

        with open(self.session_file, "r", encoding="utf-8") as f:
            sess = json.load(f)
        seg1 = sess["segments"][1]
        self.assertEqual(seg1["video_path"], res["output_path"])
        self.assertEqual(len(seg1["video_history"]), 2)

    def test_run_post_adjust_job(self):
        """Test color/tone adjust job on video."""
        manager = JobManager()
        job = manager.submit_job(
            "post.adjust",
            project_id="TestPostProject",
            params={"scene_id": "scene_001", "settings": {"contrast": 1.2, "saturation": 1.1}},
        )

        def _mock_apply_adjust(**kwargs):
            out_p = kwargs["output_path"]
            with open(out_p, "wb") as f:
                f.write(b"adjusted_video")
            return {"thumbnail_path": out_p + ".thumb.jpg"}

        with patch.object(lut_video_tools, "apply_adjust_to_video", side_effect=_mock_apply_adjust):
            res = asyncio.run(post_orch_mod.run_post_adjust_job(job, manager))

        self.assertEqual(res["scene_id"], "scene_001")
        self.assertTrue(os.path.isfile(res["output_path"]))

    def test_run_post_apply_all_job(self):
        """Test applying complete post stack to multiple scenes."""
        manager = JobManager()
        job = manager.submit_job(
            "post.apply_all",
            project_id="TestPostProject",
            params={
                "stack": {
                    "lut": {"lut_name": "film.cube", "strength": 8.0},
                    "adjust": {"settings": {"contrast": 1.1}},
                    "film_grain": {"enabled": True, "grain_intensity": 0.03},
                },
            },
        )

        def _mock_video_pass(**kwargs):
            out_p = kwargs["output_path"]
            with open(out_p, "wb") as f:
                f.write(b"pass_output")
            return {}

        with patch.object(lut_video_tools, "apply_lut_to_video", side_effect=_mock_video_pass), \
             patch.object(lut_video_tools, "apply_adjust_to_video", side_effect=_mock_video_pass), \
             patch.object(lut_video_tools, "apply_film_grain_to_video", side_effect=_mock_video_pass):
            res = asyncio.run(post_orch_mod.run_post_apply_all_job(job, manager))

        self.assertEqual(res["count"], 2)
        self.assertEqual(len(res["processed_scenes"]), 2)

        with open(self.session_file, "r", encoding="utf-8") as f:
            sess = json.load(f)
        self.assertEqual(len(sess["segments"][0]["video_history"]), 2)
        self.assertEqual(len(sess["segments"][1]["video_history"]), 2)

    # ==========================================================================
    # 5. Face Fix Services & Job Handlers (Mode S, Mode J)
    # ==========================================================================

    def test_estimate_scene_face_fix_anchors(self):
        """Test estimating face fix anchors (Mode S)."""
        with patch.object(face_fix_mod, "estimate_face_fix_anchors", return_value={"total_anchors": 3, "estimated_runs": 1}):
            res = post_orch_mod.estimate_scene_face_fix_anchors("TestPostProject", "scene_001", {"threshold": 0.8})
            self.assertEqual(res["total_anchors"], 3)

    def test_face_fix_prepare_and_enhance_jobs(self):
        """Test prepare and enhance anchor face fix jobs."""
        manager = JobManager()

        # 1. Prepare job
        prep_job = manager.submit_job("face_fix.prepare", project_id="TestPostProject", params={"scene_id": "scene_001"})
        dummy_manifest = os.path.join(self.project_dir, "face_fix_manifest.json")
        with open(dummy_manifest, "w", encoding="utf-8") as f:
            json.dump({"runs": [], "anchors": []}, f)

        with patch.object(face_fix_mod, "prepare_face_fix", return_value={"manifest_path": dummy_manifest, "anchors": [{"run_index": 0, "order": 0}], "runs": [{"run_index": 0}]}):
            prep_res = asyncio.run(post_orch_mod.run_face_fix_prepare_job(prep_job, manager))
            self.assertEqual(prep_res["manifest_path"], dummy_manifest)

        # 2. Enhance anchor job
        enh_job = manager.submit_job(
            "face_fix.enhance_anchor",
            project_id="TestPostProject",
            params={"manifest_path": dummy_manifest, "run_index": 0, "order": 0},
        )
        with patch.object(face_fix_mod, "accept_enhanced_anchor", return_value={"accepted": True}):
            enh_res = asyncio.run(post_orch_mod.run_face_fix_enhance_anchor_job(enh_job, manager))
            self.assertTrue(enh_res["accepted"])

    def test_face_fix_ltx_and_finalize_jobs(self):
        """Test LTX run and finalize jobs."""
        manager = JobManager()
        dummy_manifest = os.path.join(self.project_dir, "face_fix_manifest.json")
        with open(dummy_manifest, "w", encoding="utf-8") as f:
            json.dump({"runs": [], "anchors": []}, f)

        # 1. LTX run job
        ltx_job = manager.submit_job(
            "face_fix.ltx_run",
            project_id="TestPostProject",
            params={"manifest_path": dummy_manifest, "run_index": 0},
        )
        with patch.object(face_fix_mod, "build_ltx_face_fix_prompt", return_value={"prompt": {"1": {}}, "frame_count": 5}), \
             patch.object(face_fix_mod, "accept_ltx_frame_batch", return_value={"batch_accepted": True}):
            ltx_res = asyncio.run(post_orch_mod.run_face_fix_ltx_run_job(ltx_job, manager))
            self.assertTrue(ltx_res["batch_accepted"])

        # 2. Finalize job
        fin_video = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001_facefix.mp4")
        with open(fin_video, "wb") as f:
            f.write(b"face_fixed_video")

        fin_job = manager.submit_job(
            "face_fix.finalize",
            project_id="TestPostProject",
            params={"manifest_path": dummy_manifest, "scene_id": "scene_001"},
        )
        with patch.object(face_fix_mod, "finalize_face_fix", return_value={"output_video_path": fin_video, "finalized": True}):
            fin_res = asyncio.run(post_orch_mod.run_face_fix_finalize_job(fin_job, manager))
            self.assertTrue(fin_res["finalized"])
            self.assertEqual(fin_res["video_path"], fin_video)

        # Verify video history append
        with open(self.session_file, "r", encoding="utf-8") as f:
            sess = json.load(f)
        self.assertEqual(sess["segments"][0]["video_path"], fin_video)
        self.assertEqual(len(sess["segments"][0]["video_history"]), 2)

    def test_face_fix_auto_job(self):
        """Test end-to-end auto face fix job execution."""
        manager = JobManager()
        auto_job = manager.submit_job(
            "face_fix.auto",
            project_id="TestPostProject",
            params={"scene_id": "scene_001"},
        )

        dummy_manifest = os.path.join(self.project_dir, "face_fix_auto_manifest.json")
        with open(dummy_manifest, "w", encoding="utf-8") as f:
            json.dump({}, f)

        fin_video = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001_auto_facefix.mp4")
        with open(fin_video, "wb") as f:
            f.write(b"auto_face_fixed_video")

        with patch.object(face_fix_mod, "prepare_face_fix", return_value={
            "manifest_path": dummy_manifest,
            "anchors": [{"run_index": 0, "order": 0}],
            "runs": [{"run_index": 0}],
        }), \
             patch.object(face_fix_mod, "accept_enhanced_anchor", return_value={"accepted": True}), \
             patch.object(face_fix_mod, "build_ltx_face_fix_prompt", return_value={"prompt": {"1": {}}, "frame_count": 5}), \
             patch.object(face_fix_mod, "accept_ltx_frame_batch", return_value={"batch_accepted": True}), \
             patch.object(face_fix_mod, "finalize_face_fix", return_value={"output_video_path": fin_video, "finalized": True}):

            res = asyncio.run(post_orch_mod.run_face_fix_auto_job(auto_job, manager))

        self.assertTrue(res["repaired"])
        self.assertEqual(res["video_path"], fin_video)

        # Test early return when runs is empty
        auto_job_empty = manager.submit_job(
            "face_fix.auto",
            project_id="TestPostProject",
            params={"scene_id": "scene_001"},
        )
        with patch.object(face_fix_mod, "prepare_face_fix", return_value={"manifest_path": dummy_manifest, "anchors": [], "runs": []}):
            res_empty = asyncio.run(post_orch_mod.run_face_fix_auto_job(auto_job_empty, manager))
            self.assertFalse(res_empty["repaired"])
            self.assertIn("No qualifying faces", res_empty["reason"])

    def test_register_post_orchestrator_handlers(self):
        """Test registering all 9 handlers with JobManager."""
        manager = JobManager()
        post_orch_mod.register_post_orchestrator_handlers(manager)

        expected = [
            "post.lut",
            "post.film_grain",
            "post.adjust",
            "post.apply_all",
            "face_fix.prepare",
            "face_fix.enhance_anchor",
            "face_fix.ltx_run",
            "face_fix.finalize",
            "face_fix.auto",
        ]
        for h in expected:
            self.assertIn(h, manager._handlers)


if __name__ == "__main__":
    unittest.main()
