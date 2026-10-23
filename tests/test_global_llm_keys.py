"""API keys and LLM runner settings saved for every project: stored apart from the project session and used by the Agent API."""

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
        self.runner_file = os.path.join(self.temp, "llm_runner.json")
        for name, target in (("_global_llm_keys_path", self.keys_file), ("_global_llm_runner_path", self.runner_file)):
            patcher = patch.object(builder_project, name, lambda target=target: target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_keys_are_saved_loaded_and_cleared_one_at_a_time(self):
        self.assertEqual(builder_project._load_global_llm_keys(), {})
        builder_project._save_global_llm_keys({"llm_api_key": " sk-1 ", "elevenlabs_api_key": "el-1"})
        builder_project._save_global_llm_keys({"own_server_api_key": "own-1"})
        self.assertEqual(builder_project._load_global_llm_keys(), {
            "llm_api_key": "sk-1", "elevenlabs_api_key": "el-1", "own_server_api_key": "own-1",
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

    def test_the_agent_api_uses_the_global_keys_and_falls_back_to_an_older_project_key(self):
        builder_project._save_global_llm_keys({
            "llm_api_key": "sk-global", "own_server_api_key": "own-global", "elevenlabs_api_key": "el-global",
        })
        plain = llm_runtime.llm_payload_from_session({"own_server_url": "http://elsewhere/v1"})
        self.assertEqual((plain["llm_api_key"], plain["own_server_api_key"]), ("sk-global", "own-global"))
        both = llm_runtime.llm_payload_from_session({"llm_api_key_project": "sk-project", "own_server_api_key_project": "own-project"})
        self.assertEqual((both["llm_api_key"], both["own_server_api_key"]), ("sk-global", "own-global"))
        builder_project._save_global_llm_keys({"llm_api_key": "", "own_server_api_key": ""})
        old = llm_runtime.llm_payload_from_session({"llm_api_key_project": "sk-project", "own_server_api_key_project": "own-project"})
        self.assertEqual((old["llm_api_key"], old["own_server_api_key"]), ("sk-project", "own-project"))
        self.assertEqual(llm_runtime.project_elevenlabs_key({}), "el-global")
        self.assertEqual(llm_runtime.project_elevenlabs_key({"elevenlabs_api_key_project": "el-project"}), "el-project")

    def test_runner_settings_are_merged_and_unknown_fields_ignored(self):
        self.assertEqual(builder_project._load_global_llm_runner(), {})
        builder_project._save_global_llm_runner({"text_gemma_runner": "own_server", "own_server_url": "http://a/v1", "not_a_field": 1})
        builder_project._save_global_llm_runner({"text_gemma_runner": "builtin", "gemma_context_limit": 32768})
        builder_project._save_global_llm_runner(None)
        self.assertEqual(builder_project._load_global_llm_runner(), {
            "text_gemma_runner": "builtin", "own_server_url": "http://a/v1", "gemma_context_limit": 32768,
        })

    def test_the_save_route_and_a_session_save_store_the_runner_outside_the_project(self):
        result = builder_project._save_global_llm_settings({
            "runner": {"text_gemma_runner": "own_server", "own_server_model": "gemma-4"}, "keys": {"own_server_api_key": "own-1"},
        })
        self.assertEqual(result["global_llm_runner"]["own_server_model"], "gemma-4")
        self.assertEqual(result["global_llm_keys"]["own_server_api_key"], "own-1")
        project = os.path.join(self.temp, "proj")
        builder_project._save_builder_session({
            "project_folder": project,
            "session": {"segments": [], "global_llm_runner": {"text_gemma_runner": "builtin"}},
        })
        with open(os.path.join(project, "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            self.assertNotIn("global_llm_runner", json.load(handle))
        loaded = builder_project._load_builder_session(project)["global_llm_runner"]
        self.assertEqual((loaded["text_gemma_runner"], loaded["own_server_model"]), ("builtin", "gemma-4"))

    def test_the_agent_api_uses_the_global_runner_over_the_project(self):
        builder_project._save_global_llm_runner({"text_gemma_runner": "own_server", "own_server_url": "http://global/v1"})
        payload = llm_runtime.llm_payload_from_session({"text_gemma_runner": "lm_studio", "own_server_url": "http://project/v1"})
        self.assertEqual((payload["text_runner"], payload["own_server_url"]), ("own_server", "http://global/v1"))
        self.assertEqual(llm_runtime.llm_payload_from_session({}, {"text_runner": "builtin"})["text_runner"], "builtin")


if __name__ == "__main__":
    unittest.main()
