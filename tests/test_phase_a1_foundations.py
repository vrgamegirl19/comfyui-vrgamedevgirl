import ast
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from builder_source import read_builder_backend_source


ROOT = Path(__file__).resolve().parents[1]
BACKEND_SOURCE = read_builder_backend_source()


def load_foundations_helpers():
    tree = ast.parse(BACKEND_SOURCE)
    names = {
        "_MAX_SESSION_BACKUPS",
        "_MAX_REMOVED_SCENE_ASSETS",
        "_prune_session_backups",
        "_prune_removed_scene_assets",
        "_redact_session_secrets",
        "_find_ffmpeg_path",
    }
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(target, "id", "") in names for target in node.targets))
    ]
    namespace = {
        "os": os,
        "shutil": shutil,
        "json": json,
        "time": time,
        "subprocess": subprocess,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "builder_foundations", "exec"), namespace)
    return namespace


class PhaseA1FoundationsTests(unittest.TestCase):
    def setUp(self):
        self.helpers = load_foundations_helpers()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def test_prune_session_backups_caps_at_max(self):
        prune_fn = self.helpers["_prune_session_backups"]
        max_keep = self.helpers["_MAX_SESSION_BACKUPS"]
        self.assertEqual(max_keep, 25)

        backup_dir = os.path.join(self.temp_dir, "session_backups")
        os.makedirs(backup_dir, exist_ok=True)

        base_time = 1700000000.0
        # Create 32 backup files
        for i in range(1, 33):
            path = os.path.join(backup_dir, f"vrgdg_builder_session_20260930_{i:04d}.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"backup": i}))
            mtime = base_time + (i * 10)
            os.utime(path, (mtime, mtime))

        prune_fn(self.temp_dir)

        remaining = sorted(os.listdir(backup_dir))
        self.assertEqual(len(remaining), 25)
        # Oldest 7 (1 to 7) should have been pruned; 8 to 32 should remain
        self.assertNotIn("vrgdg_builder_session_20260930_0001.json", remaining)
        self.assertNotIn("vrgdg_builder_session_20260930_0007.json", remaining)
        self.assertIn("vrgdg_builder_session_20260930_0008.json", remaining)
        self.assertIn("vrgdg_builder_session_20260930_0032.json", remaining)

    def test_prune_removed_scene_assets_caps_at_max(self):
        prune_fn = self.helpers["_prune_removed_scene_assets"]
        max_keep = self.helpers["_MAX_REMOVED_SCENE_ASSETS"]
        self.assertEqual(max_keep, 20)

        removed_dir = os.path.join(self.temp_dir, "removed_scene_assets")
        os.makedirs(removed_dir, exist_ok=True)

        base_time = 1700000000.0
        # Create 28 removed asset directories
        for i in range(1, 29):
            folder_path = os.path.join(removed_dir, f"scene_001_removed_20260930_{i:04d}")
            os.makedirs(folder_path, exist_ok=True)
            with open(os.path.join(folder_path, "asset.txt"), "w") as f:
                f.write("test")
            mtime = base_time + (i * 10)
            os.utime(folder_path, (mtime, mtime))

        prune_fn(self.temp_dir)

        remaining = sorted(os.listdir(removed_dir))
        self.assertEqual(len(remaining), 20)
        # Oldest 8 (1 to 8) should have been pruned; 9 to 28 should remain
        self.assertNotIn("scene_001_removed_20260930_0001", remaining)
        self.assertNotIn("scene_001_removed_20260930_0008", remaining)
        self.assertIn("scene_001_removed_20260930_0009", remaining)
        self.assertIn("scene_001_removed_20260930_0028", remaining)

    def test_redact_session_secrets_cleans_credentials(self):
        redact_fn = self.helpers["_redact_session_secrets"]

        dirty_session = {
            "project_name": "My Project",
            "api_key": "top-secret-123",
            "lm_studio_api_key": "lm-key-456",
            "llm_api_key_project": "proj-key-789",
            "own_server_key": "server-secret",
            "custom_service_api_key": "custom-123",
            "llm_settings": {
                "api_key": "nested-secret",
                "custom_provider_api_key": "provider-secret",
                "provider": "openrouter",
                "temperature": 0.7,
            },
            "lm_studio_settings": {
                "lm_studio_api_key": "studio-secret",
                "endpoint": "http://127.0.0.1:1234",
            },
            "segments": [{"prompt": "A beautiful sunset"}],
        }

        cleaned = redact_fn(dirty_session)

        # Original dict unchanged (deep copy semantics)
        self.assertEqual(dirty_session["api_key"], "top-secret-123")

        # Top level secrets redacted
        self.assertEqual(cleaned["api_key"], "")
        self.assertEqual(cleaned["lm_studio_api_key"], "")
        self.assertEqual(cleaned["llm_api_key_project"], "")
        self.assertEqual(cleaned["own_server_key"], "")
        self.assertEqual(cleaned["custom_service_api_key"], "")

        # Nested settings redacted
        self.assertEqual(cleaned["llm_settings"]["api_key"], "")
        self.assertEqual(cleaned["llm_settings"]["custom_provider_api_key"], "")
        self.assertEqual(cleaned["lm_studio_settings"]["lm_studio_api_key"], "")

        # Non-secret fields intact
        self.assertEqual(cleaned["project_name"], "My Project")
        self.assertEqual(cleaned["llm_settings"]["provider"], "openrouter")
        self.assertEqual(cleaned["llm_settings"]["temperature"], 0.7)
        self.assertEqual(cleaned["lm_studio_settings"]["endpoint"], "http://127.0.0.1:1234")
        self.assertEqual(len(cleaned["segments"]), 1)

    def test_find_ffmpeg_path_resolves_or_informative_error(self):
        ffmpeg_fn = self.helpers["_find_ffmpeg_path"]
        try:
            path = ffmpeg_fn()
            self.assertIsInstance(path, str)
            self.assertTrue(len(path) > 0)
        except RuntimeError as exc:
            self.assertIn("ffmpeg was not found", str(exc))

    def test_stale_revision_rejection_contract(self):
        # Verify the contract in builder backend source
        self.assertIn("existing_revision > incoming_revision", BACKEND_SOURCE)
        self.assertIn('"stale": True', BACKEND_SOURCE)
        self.assertIn('"revision": new_revision', BACKEND_SOURCE)


if __name__ == "__main__":
    unittest.main()
