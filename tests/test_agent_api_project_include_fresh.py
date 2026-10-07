"""GET /projects/{pid}?include=<Builder session key> on a fresh project that has not saved that key yet."""

import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
projects = importlib.import_module(f"{pkg_name}.agent_api.projects")
from builder_source import function_source, read_builder_module  # noqa: E402


def builder_session_keys():
    """Top-level keys of the session the Builder saves (currentSessionData in session.mjs)."""
    source = function_source(read_builder_module("session.mjs"), "currentSessionData")
    body = source[source.index("return {"):]
    return set(re.findall(r"^\s+([A-Za-z_]\w*):\s", body, re.M))


class FreshProjectIncludeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        folder = os.path.join(self.root, "API Retest")
        os.makedirs(folder)
        # What POST /projects writes before any reference, audio or story has been saved.
        session = {"project_name": "API Retest", "project_folder": folder, "revision": 1,
                   "video_engine": "minimax_h3", "segments": []}
        with open(os.path.join(folder, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(session, handle)
        patcher = patch.object(paths, "get_allowed_project_roots", return_value=[self.root])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_flux_reference_builder_is_an_empty_object_before_the_first_reference(self):
        detail = projects.get_project_detail("API Retest", ["flux_reference_builder"])
        self.assertEqual(detail["flux_reference_builder"], {})
        self.assertEqual(projects.get_project_detail("API Retest", ["references"])["references"], {})

    def test_other_unsaved_builder_keys_are_accepted(self):
        detail = projects.get_project_detail("API Retest", ["audio_path", "audio_duration", "detected_tempo_bpm",
                                                            "builder_story_layer", "minimax_h3_settings"])
        self.assertIn(detail["audio_path"], ("", None))
        self.assertIsNone(detail["detected_tempo_bpm"])
        self.assertEqual(detail["builder_story_layer"], {})
        self.assertIn("minimax_h3_settings", detail)

    def test_unsaved_api_keys_stay_blank(self):
        detail = projects.get_project_detail("API Retest", ["lm_studio_api_key"])
        self.assertIn(detail["lm_studio_api_key"], ("", None))

    def test_truly_unknown_names_are_still_a_400(self):
        with self.assertRaises(errors.ValidationError) as ctx:
            projects.get_project_detail("API Retest", ["flux_reference_builder", "bogus"])
        self.assertEqual(ctx.exception.details["unknown"], ["bogus"])

    def test_known_keys_are_the_keys_the_builder_saves(self):
        session_keys = importlib.import_module(f"{pkg_name}.agent_api.session_keys")
        self.assertEqual(set(session_keys.BUILDER_SESSION_DATA_KEYS), builder_session_keys())
        self.assertTrue(set(session_keys.BUILDER_SESSION_DATA_KEYS) <= session_keys.KNOWN_SESSION_KEYS)
        for key in ("audio_path", "project_name", "project_folder", "revision", "updated"):
            self.assertIn(key, session_keys.KNOWN_SESSION_KEYS)


if __name__ == "__main__":
    unittest.main()
