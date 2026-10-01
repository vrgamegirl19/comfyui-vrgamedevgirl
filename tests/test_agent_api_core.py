"""Unit tests for Phase A2: Agent API Core (envelope, errors, auth, paths, schemas, modes, projects, routes)."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
envelope = importlib.import_module(f"{pkg_name}.agent_api.envelope")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
auth = importlib.import_module(f"{pkg_name}.agent_api.auth")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
schemas = importlib.import_module(f"{pkg_name}.agent_api.schemas")
modes = importlib.import_module(f"{pkg_name}.agent_api.modes")
projects = importlib.import_module(f"{pkg_name}.agent_api.projects")
router = importlib.import_module(f"{pkg_name}.agent_api.router")


class AgentApiCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    # 1. Envelope & Errors
    def test_envelope_success(self):
        resp = envelope.api_success(data={"greeting": "hello"}, revision=5)
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.body)
        self.assertTrue(body["ok"])
        self.assertEqual(body["data"], {"greeting": "hello"})
        self.assertEqual(body["revision"], 5)

    def test_envelope_error(self):
        resp = envelope.api_error(code="CUSTOM_ERR", message="Something failed", details={"key": "val"}, status=400)
        self.assertEqual(resp.status, 400)
        body = json.loads(resp.body)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "CUSTOM_ERR")
        self.assertEqual(body["error"]["message"], "Something failed")
        self.assertEqual(body["error"]["details"], {"key": "val"})

    def test_envelope_exception_mapping(self):
        exc_not_found = errors.ProjectNotFoundError("missing_proj")
        resp = envelope.api_exception(exc_not_found)
        self.assertEqual(resp.status, 404)
        body = json.loads(resp.body)
        self.assertEqual(body["error"]["code"], errors.PROJECT_NOT_FOUND)

        exc_conflict = errors.RevisionConflictError(current_revision=10, incoming_revision=8)
        resp_conflict = envelope.api_exception(exc_conflict)
        self.assertEqual(resp_conflict.status, 409)
        body_conflict = json.loads(resp_conflict.body)
        self.assertEqual(body_conflict["error"]["code"], errors.REVISION_CONFLICT)

    # 2. Auth
    def test_auth_loopback_allowed_by_default(self):
        req = MagicMock()
        req.remote = "127.0.0.1"
        req.headers = {}
        config = {"enabled": True, "require_token_on_loopback": False, "token": "test-secret-token"}
        # Should not raise
        auth.verify_auth(req, config)

    def test_auth_remote_requires_token(self):
        req = MagicMock()
        req.remote = "192.168.1.50"
        req.headers = {}
        config = {"enabled": True, "require_token_on_loopback": False, "token": "test-secret-token"}
        with self.assertRaises(errors.AuthError):
            auth.verify_auth(req, config)

        # Valid Bearer token
        req.headers = {"Authorization": "Bearer test-secret-token"}
        auth.verify_auth(req, config)

        # Wrong Bearer token
        req.headers = {"Authorization": "Bearer wrong-token"}
        with self.assertRaises(errors.AuthError):
            auth.verify_auth(req, config)

    # 3. Path Security & Project ID Resolution (C8)
    def test_path_resolution_and_traversal_rejection(self):
        allowed_root = os.path.join(self.temp_dir, "output")
        os.makedirs(allowed_root, exist_ok=True)
        proj_folder = os.path.join(allowed_root, "MyProject")
        os.makedirs(proj_folder, exist_ok=True)

        with patch.object(paths, "get_allowed_project_roots", return_value=[allowed_root]):
            # Valid project resolves
            resolved = paths.resolve_project_folder("MyProject")
            self.assertEqual(os.path.normcase(resolved), os.path.normcase(proj_folder))

            # Non-existent project raises ProjectNotFoundError
            with self.assertRaises(errors.ProjectNotFoundError):
                paths.resolve_project_folder("NonExistentProject")

            # Path traversal / escape attempts raise ValidationError or PathOutsideRootError
            with self.assertRaises(errors.ValidationError):
                paths.resolve_project_folder("../../outside")

            with self.assertRaises(errors.ValidationError):
                paths.resolve_project_folder("sub/dir")

            with self.assertRaises(errors.ValidationError):
                paths.resolve_project_folder("")

    # 4. Settings Schemas (C5)
    def test_settings_schemas_extraction_and_validation(self):
        sample_session = {
            "video_engine": "minimax_h3",
            "video_model_mode": "image_to_video",
            "image_model_mode": "zimage",
            "minimax_h3_settings": {"aspect_ratio": "16:9", "steps": 25},
            "gemma_context_limit": 8000,
        }
        effective = schemas.extract_effective_settings(sample_session)
        self.assertEqual(effective["settings_version"], 1)
        self.assertEqual(effective["project"]["video_engine"], "minimax_h3")
        self.assertEqual(effective["minimax_h3"]["aspect_ratio"], "16:9")
        self.assertEqual(effective["minimax_h3"]["steps"], 25)
        self.assertEqual(effective["llm"]["gemma_context_limit"], 8000)

        # Validation test
        schemas.validate_settings_patch({"project": {"video_engine": "ltx_video"}})
        with self.assertRaises(errors.SettingsInvalidError):
            schemas.validate_settings_patch("not-a-dict")
        with self.assertRaises(errors.SettingsInvalidError):
            schemas.validate_settings_patch({"minimax_h3": "not-a-dict-value"})

    # 5. Modes Catalog (C9)
    def test_modes_catalog(self):
        catalog = modes.get_modes_catalog()
        self.assertIn("video_engines", catalog)
        self.assertIn("image_modes", catalog)
        self.assertIn("enums", catalog)
        self.assertIn("minimax_h3", catalog["video_engines"])
        self.assertIn("ltx_video", catalog["video_engines"])
        self.assertIn("zimage", catalog["image_modes"])
        self.assertIn("flux_klein", catalog["image_modes"])

    # 6. Projects & Scenes Read Queries
    def test_project_queries_and_scene_serialization(self):
        allowed_root = os.path.join(self.temp_dir, "output")
        proj_folder = os.path.join(allowed_root, "Test_Track")
        os.makedirs(proj_folder, exist_ok=True)
        images_dir = os.path.join(proj_folder, "images")
        scene_videos_dir = os.path.join(proj_folder, "scene_videos")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(scene_videos_dir, exist_ok=True)

        # Write dummy assets
        with open(os.path.join(images_dir, "image_0001.png"), "wb") as f:
            f.write(b"PNG_DATA")
        with open(os.path.join(scene_videos_dir, "video_0001.mp4"), "wb") as f:
            f.write(b"MP4_DATA")

        session_data = {
            "project_name": "Test Track",
            "project_folder": proj_folder,
            "revision": 3,
            "updated": 1700000000.0,
            "segments": [
                {
                    "id": "seg_0001",
                    "start": 0.0,
                    "end": 4.5,
                    "notes": "Intro beat",
                    "t2i_prompt": "Neon cyberpunk city",
                    "i2v_prompt": "Camera pans slowly",
                },
                {
                    "id": "seg_0002",
                    "start": 4.5,
                    "end": 8.0,
                    "notes": "Verse begins",
                    "t2i_prompt": "Singer at microphone",
                },
            ],
        }
        with open(os.path.join(proj_folder, "vrgdg_builder_session.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(session_data))

        with patch.object(paths, "get_allowed_project_roots", return_value=[allowed_root]), \
             patch.object(projects, "get_allowed_project_roots", return_value=[allowed_root]):
            # 1. list_projects
            proj_list = projects.list_projects()
            self.assertEqual(len(proj_list), 1)
            self.assertEqual(proj_list[0]["id"], "Test_Track")
            self.assertEqual(proj_list[0]["scene_count"], 2)
            self.assertEqual(proj_list[0]["revision"], 3)

            # 2. get_project_detail
            detail = projects.get_project_detail("Test_Track")
            self.assertEqual(detail["id"], "Test_Track")
            self.assertEqual(detail["revision"], 3)
            self.assertIn("settings", detail)
            self.assertIn("scenes", detail)
            self.assertEqual(len(detail["scenes"]), 2)

            # 3. get_project_scenes & filters
            all_scenes = projects.get_project_scenes("Test_Track")
            self.assertEqual(len(all_scenes), 2)
            self.assertEqual(all_scenes[0]["id"], "seg_0001")
            self.assertEqual(all_scenes[0]["number"], 1)
            self.assertEqual(all_scenes[0]["status"], "has_video")
            self.assertIsNotNone(all_scenes[0]["approved_image"])
            self.assertIsNotNone(all_scenes[0]["rendered_video"])
            self.assertEqual(all_scenes[1]["status"], "has_prompt")
            self.assertIsNone(all_scenes[1]["approved_image"])

            # Filter by has_video
            video_scenes = projects.get_project_scenes("Test_Track", has_video=True)
            self.assertEqual(len(video_scenes), 1)
            self.assertEqual(video_scenes[0]["id"], "seg_0001")

            # 4. get_scene_detail
            scene_1 = projects.get_scene_detail("Test_Track", "seg_0001")
            self.assertEqual(scene_1["id"], "seg_0001")
            scene_by_num = projects.get_scene_detail("Test_Track", "2")
            self.assertEqual(scene_by_num["id"], "seg_0002")

            # 5. get_project_summary
            summary = projects.get_project_summary("Test_Track")
            self.assertEqual(summary["total_scenes"], 2)
            self.assertEqual(summary["scenes_with_image"], 1)
            self.assertEqual(summary["scenes_with_video"], 1)
            self.assertEqual(summary["scenes_with_prompt"], 2)
            self.assertTrue(summary["disk_usage_bytes"] > 0)

            # 6. get_project_assets
            assets = projects.get_project_assets("Test_Track")
            self.assertTrue(len(assets) >= 2)
            kinds = {a["kind"] for a in assets}
            self.assertIn("image", kinds)
            self.assertIn("video", kinds)

    # 7. Router registration
    def test_router_registration(self):
        mock_server = MagicMock()
        mock_server.routes = MagicMock()
        mock_server.routes.get = MagicMock(return_value=lambda f: f)
        mock_server.routes.post = MagicMock(return_value=lambda f: f)

        router._VRGDG_AGENT_API_ROUTES_REGISTERED = False
        router.register_agent_api_routes(mock_server)
        self.assertTrue(router._VRGDG_AGENT_API_ROUTES_REGISTERED)
        self.assertTrue(mock_server.routes.get.call_count >= 8)


if __name__ == "__main__":
    unittest.main()
