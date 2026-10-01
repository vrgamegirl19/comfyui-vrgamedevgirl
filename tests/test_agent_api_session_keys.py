"""The API must read and write the same session keys the Video Builder saves."""

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


class VideoModeTests(unittest.TestCase):
    def test_a_minimax_project_renders_with_minimax_even_if_the_old_ltx_mode_is_set(self):
        # Real projects keep video_model_mode (the LTX mode, e.g. "rtv") after switching to MiniMax.
        self.assertEqual(paths.session_video_mode({"video_engine": "minimax_h3", "video_model_mode": "rtv"}), "minimax_h3")

    def test_other_engines_use_their_saved_ltx_mode(self):
        self.assertEqual(paths.session_video_mode({"video_engine": "ltx", "video_model_mode": "flf"}), "flf")
        self.assertEqual(paths.session_video_mode({"video_engine": "ltx"}), "i2v")
        self.assertEqual(paths.session_video_mode({}), "i2v")


class StoryKeyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        os.makedirs(self.root)
        for target, name, value in (
            (builder_project, "_model_defaults_path", lambda: os.path.join(self.temp, "model_defaults.json")),
            (builder_project, "_project_target_from_payload", lambda payload, key: os.path.join(self.root, payload["project_name"])),
            (paths, "get_allowed_project_roots", lambda: [self.root]),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        mutations.create_project("Song")
        self.session_file = os.path.join(self.root, "Song", "vrgdg_builder_session.json")

    def session(self):
        with open(self.session_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_the_story_is_saved_under_the_key_the_ui_reads(self):
        mutations.put_project_story("Song", {"story_idea": "A man who never stays"})
        session = self.session()
        self.assertEqual(session["builder_story_layer"], {"story_idea": "A man who never stays"})
        self.assertNotIn("builderStoryLayer", session)
        self.assertEqual(mutations.get_project_story("Song"), {"story_idea": "A man who never stays"})

    def test_a_story_saved_under_the_old_key_is_still_read(self):
        session = self.session()
        session["builderStoryLayer"] = {"story_idea": "old"}
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump(session, handle)
        self.assertEqual(mutations.get_project_story("Song"), {"story_idea": "old"})


if __name__ == "__main__":
    unittest.main()
