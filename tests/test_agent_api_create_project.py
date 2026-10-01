"""Projects created through the API must be usable and must not disturb the saved model defaults."""

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

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
builder_project = importlib.import_module(f"{pkg_name}.builder.project")


class CreateProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        os.makedirs(self.root)
        self.defaults_file = os.path.join(self.temp, "model_defaults.json")
        # Only these keys are model defaults (builder.project._MODEL_DEFAULT_KEYS).
        self.saved_defaults = {
            "text_gemma_runner": "lm_studio",
            "lm_studio_base_url": "http://127.0.0.1:1234/v1",
            "lm_studio_model": "my-local-model",
            "image_model_mode": "zimage",
        }
        with open(self.defaults_file, "w", encoding="utf-8") as handle:
            json.dump({"saved_at": "x", "defaults": self.saved_defaults}, handle)
        for target, value in (
            (builder_project, ("_model_defaults_path", lambda: self.defaults_file)),
            (builder_project, ("_project_target_from_payload", lambda payload, key: os.path.join(self.root, payload["project_name"]))),
            (paths, ("get_allowed_project_roots", lambda: [self.root])),
        ):
            patcher = patch.object(target, value[0], value[1])
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_new_project_has_a_session_that_the_api_can_use(self):
        created = mutations.create_project("Fresh Song")
        session_path = os.path.join(self.root, "Fresh Song", "vrgdg_builder_session.json")
        self.assertTrue(os.path.isfile(session_path), "the session file must exist")
        self.assertGreaterEqual(created["revision"], 1)
        # The rest of the API can work on it straight away.
        result = mutations.patch_project_settings("Fresh Song", {"minimax_h3": {"video_mode": "reference_to_video"}})
        self.assertEqual(result["settings"]["minimax_h3"]["video_mode"], "reference_to_video")

    def test_a_new_project_starts_from_the_saved_model_defaults(self):
        mutations.create_project("Fresh Song")
        with open(os.path.join(self.root, "Fresh Song", "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            session = json.load(handle)
        for key, value in self.saved_defaults.items():
            self.assertEqual(session[key], value)
        self.assertEqual(session["segments"], [])
        self.assertEqual(session["video_engine"], "minimax_h3")

    def test_creating_a_project_does_not_wipe_the_saved_model_defaults(self):
        mutations.create_project("Fresh Song")
        with open(self.defaults_file, "r", encoding="utf-8") as handle:
            after = json.load(handle)["defaults"]
        for key, value in self.saved_defaults.items():
            self.assertEqual(after.get(key), value, f"saved default '{key}' was changed")


if __name__ == "__main__":
    unittest.main()
