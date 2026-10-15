"""API keys saved for every project: stored apart from the project session and used by the Agent API."""

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

builder_project = importlib.import_module(f"{ROOT.name}.builder.project")
llm_runtime = importlib.import_module(f"{ROOT.name}.agent_api.llm_runtime")


class GlobalLlmKeyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.keys_file = os.path.join(self.temp, "llm_keys.json")
        patcher = patch.object(builder_project, "_global_llm_keys_path", lambda: self.keys_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_keys_are_saved_loaded_and_cleared_one_at_a_time(self):
        self.assertEqual(builder_project._load_global_llm_keys(), {})
        builder_project._save_global_llm_keys({"llm_api_key": " sk-1 ", "elevenlabs_api_key": "el-1"})
        builder_project._save_global_llm_keys({"own_server_api_key": "own-1", "own_server_url": "http://host:8000/v1"})
        self.assertEqual(builder_project._load_global_llm_keys(), {
            "llm_api_key": "sk-1", "elevenlabs_api_key": "el-1",
            "own_server_api_key": "own-1", "own_server_url": "http://host:8000/v1",
        })
        builder_project._save_global_llm_keys({"llm_api_key": ""})
        builder_project._save_global_llm_keys({"own_server_api_key": ""})
        self.assertEqual(builder_project._load_global_llm_keys(), {"elevenlabs_api_key": "el-1"})

    def test_a_key_not_named_in_a_save_is_left_alone(self):
        builder_project._save_global_llm_keys({"llm_api_key": "sk-1"})
        builder_project._save_global_llm_keys({})
        builder_project._save_global_llm_keys(None)
        self.assertEqual(builder_project._load_global_llm_keys(), {"llm_api_key": "sk-1"})

    def test_a_session_save_keeps_the_keys_out_of_the_project_file(self):
        project = os.path.join(self.temp, "proj")
        builder_project._save_builder_session({
            "project_folder": project,
            "session": {"segments": [], "global_llm_keys": {"llm_api_key": "sk-secret"}},
        })
        with open(os.path.join(project, "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertNotIn("sk-secret", text)
        self.assertEqual(builder_project._load_global_llm_keys()["llm_api_key"], "sk-secret")
        self.assertEqual(builder_project._load_builder_session(project)["global_llm_keys"]["llm_api_key"], "sk-secret")

    def test_the_agent_api_uses_the_global_keys_unless_the_project_has_its_own(self):
        builder_project._save_global_llm_keys({
            "llm_api_key": "sk-global", "own_server_api_key": "own-global", "own_server_url": "http://host:8000/v1/",
            "elevenlabs_api_key": "el-global",
        })
        plain = llm_runtime.llm_payload_from_session({"own_server_url": "http://host:8000/v1"})
        self.assertEqual((plain["llm_api_key"], plain["own_server_api_key"]), ("sk-global", "own-global"))
        other = llm_runtime.llm_payload_from_session({"own_server_url": "http://elsewhere/v1"})
        self.assertEqual(other["own_server_api_key"], "", "the global server key only goes to its own server")
        own = llm_runtime.llm_payload_from_session({"llm_api_key_project": "sk-project", "own_server_api_key_project": "own-project"})
        self.assertEqual((own["llm_api_key"], own["own_server_api_key"]), ("sk-project", "own-project"))
        self.assertEqual(llm_runtime.project_elevenlabs_key({}), "el-global")
        self.assertEqual(llm_runtime.project_elevenlabs_key({"elevenlabs_api_key_project": "el-project"}), "el-project")


if __name__ == "__main__":
    unittest.main()
