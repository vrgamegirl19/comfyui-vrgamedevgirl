"""GET /projects/{pid}?include=...: groups, top-level session keys, and unknown keys."""

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
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
projects = importlib.import_module(f"{pkg_name}.agent_api.projects")

BASE_FIELDS = {"id", "name", "project_folder", "revision", "updated", "video_engine", "image_mode"}


class ProjectIncludeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        folder = os.path.join(self.root, "Cut and Gun")
        os.makedirs(folder)
        self.audio = os.path.join(folder, "song.wav")
        Path(self.audio).write_bytes(b"RIFF")
        session = {
            "project_name": "Cut and Gun", "project_folder": folder, "revision": 7, "video_engine": "minimax_h3",
            "audio_path": self.audio, "audio_duration": 181.5, "detected_tempo_bpm": 96.0,
            "lm_studio_api_key": "sk-secret", "llm_api_key_project": "sk-project-secret",
            "flux_reference_builder": {"subjects": [{"id": "character_a", "name": "A"}], "locations": [],
                                       "use_subject_reference": True},
            "segments": [{"id": "seg_0001", "start": 0.0, "end": 4.0}],
        }
        with open(os.path.join(folder, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(session, handle)
        patcher = patch.object(paths, "get_allowed_project_roots", return_value=[self.root])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_top_level_session_keys_are_returned_by_name(self):
        detail = projects.get_project_detail("Cut and Gun", ["audio_path", "audio_duration", "detected_tempo_bpm"])
        self.assertEqual(detail["audio_path"], self.audio)
        self.assertEqual(detail["audio_duration"], 181.5)
        self.assertEqual(detail["detected_tempo_bpm"], 96.0)
        self.assertEqual(set(detail) - BASE_FIELDS, {"audio_path", "audio_duration", "detected_tempo_bpm"})

    def test_a_session_object_key_is_returned_whole(self):
        detail = projects.get_project_detail("Cut and Gun", ["flux_reference_builder"])
        self.assertEqual(detail["flux_reference_builder"]["subjects"][0]["id"], "character_a")
        self.assertTrue(detail["flux_reference_builder"]["use_subject_reference"])

    def test_groups_still_work_and_mix_with_session_keys(self):
        detail = projects.get_project_detail("Cut and Gun", ["audio", "detected_tempo_bpm"])
        self.assertEqual(detail["audio"]["duration"], 181.5)
        self.assertEqual(detail["detected_tempo_bpm"], 96.0)
        self.assertNotIn("scenes", detail)
        self.assertNotIn("settings", detail)

    def test_no_include_still_returns_every_group(self):
        detail = projects.get_project_detail("Cut and Gun")
        for group in ("settings", "scenes", "audio", "story", "references"):
            self.assertIn(group, detail)

    def test_unknown_keys_are_an_error_that_names_them(self):
        with self.assertRaises(errors.ValidationError) as ctx:
            projects.get_project_detail("Cut and Gun", ["audio_path", "no_such_key", "bogus"])
        self.assertIn("no_such_key", ctx.exception.message)
        self.assertIn("bogus", ctx.exception.message)
        self.assertNotIn("audio_path", ctx.exception.details["unknown"])
        self.assertEqual(ctx.exception.status, 400)

    def test_api_keys_are_never_returned(self):
        detail = projects.get_project_detail("Cut and Gun", ["lm_studio_api_key", "llm_api_key_project"])
        self.assertEqual(detail["lm_studio_api_key"], "")
        self.assertEqual(detail["llm_api_key_project"], "")


if __name__ == "__main__":
    unittest.main()
