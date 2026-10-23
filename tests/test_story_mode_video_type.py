"""Story Mode (video_type "story") renders like Singing everywhere the server reads the video type."""

import importlib
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name):
    """Import a submodule of this package without running the package's ComfyUI __init__."""
    package = "vrgdg_story_mode_pkg"
    if package not in sys.modules:
        stub = types.ModuleType(package)
        stub.__path__ = [str(ROOT)]
        sys.modules[package] = stub
    return importlib.import_module(f"{package}.{module_name}")


class StoryModeVideoTypeTests(unittest.TestCase):
    def test_python_performance_twins_read_story_as_singing(self):
        session = {"video_type": "story"}
        scene = {}
        inputs = _load("agent_api.video_prompt_inputs")
        self.assertEqual(inputs._performance_mode(session, scene), "singing")
        sync = _load("storyboard.session_sync")
        self.assertEqual(sync._performance_mode(session, scene), "singing")
        helpers = _load("storyboard.scene_helpers")
        self.assertEqual(helpers._normalize_performance_mode("story"), "singing")

    def test_story_scenes_keep_their_own_performance(self):
        session = {"video_type": "story"}
        inputs = _load("agent_api.video_prompt_inputs")
        sync = _load("storyboard.session_sync")
        speaking = {"performance_mode": "speaking"}
        self.assertEqual(inputs._performance_mode(session, speaking), "speaking")
        self.assertEqual(sync._performance_mode(session, speaking), "speaking")
        # Other video types still ignore a scene's performance_mode, as before.
        self.assertEqual(sync._performance_mode({"video_type": "singing"}, speaking), "singing")

    def test_story_scene_without_lip_sync_stays_visual_only(self):
        inputs = _load("agent_api.video_prompt_inputs")
        self.assertEqual(inputs._performance_mode({"video_type": "story"}, {"lyric_no_lip_sync": True}), "no_lip_sync")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the video type tests")
    def test_frontend_video_type_helpers(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("story_mode_video_type.mjs"))],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
