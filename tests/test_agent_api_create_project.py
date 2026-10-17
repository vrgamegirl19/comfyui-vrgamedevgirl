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
            (builder_project, ("_project_target_from_payload",
                               lambda payload, key: os.path.join(self.root, payload.get("project_name") or payload[key]))),
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

    def test_a_duplicate_keeps_the_scenes_and_its_own_copy_of_the_audio(self):
        mutations.create_project("Fresh Song")
        source = os.path.join(self.root, "Fresh Song")
        audio = os.path.join(source, "project_audio", "project_audio.wav")
        os.makedirs(os.path.dirname(audio))
        with open(audio, "wb") as handle:
            handle.write(b"RIFF")
        session_path = os.path.join(source, "vrgdg_builder_session.json")
        with open(session_path, "r", encoding="utf-8") as handle:
            session = json.load(handle)
        session.update({"audio_path": audio, "segments": [{"id": "seg_1", "start": 0.0, "end": 2.0, "lyric_text": "la"}]})
        with open(session_path, "w", encoding="utf-8") as handle:
            json.dump(session, handle)

        copied = mutations.duplicate_project("Fresh Song", "Fresh Song Copy")
        with open(os.path.join(copied["project_folder"], "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            branch = json.load(handle)
        self.assertEqual([s["id"] for s in branch["segments"]], ["seg_1"])
        self.assertTrue(os.path.isfile(branch["audio_path"]))
        self.assertEqual(os.path.commonpath([branch["audio_path"], copied["project_folder"]]), copied["project_folder"])

    def test_every_llm_runner_setting_is_a_global_default(self):
        session = {
            "llm_api_provider": "openai", "llm_api_model": "gpt-x", "own_server_url": "http://127.0.0.1:8000/v1",
            "own_server_model": "m", "own_server_output_token_limit": 4096, "own_server_timeout": 900,
            "llm_api_key_project": "secret", "own_server_api_key_project": "secret",
        }
        defaults = builder_project._extract_model_defaults(session)
        self.assertEqual(defaults, {k: v for k, v in session.items() if not k.endswith("_key_project")})


if __name__ == "__main__":
    unittest.main()
