"""PUT /story/settings rejects numbers outside the Builder's 0-10 sliders instead of clamping them."""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
story = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.storyboard_orchestrator")
from test_agent_api_references_llm import Base  # noqa: E402

DEFAULT_NUMBERS = ("camera_motion_speed", "character_motion_speed", "temporal_background_intensity")


class StorySettingsRangeTests(Base):
    def assert_rejected(self, params, key):
        before = self.read_session()
        with self.assertRaises(errors.ValidationError) as ctx:
            story.set_story_settings("Song", params)
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, errors.VALIDATION_ERROR)
        self.assertIn(f"{key} must be a number from 0 to 10", ctx.exception.message)
        self.assertEqual(self.read_session(), before, "nothing is saved")

    def test_out_of_range_defaults_are_rejected(self):
        for key in DEFAULT_NUMBERS:
            for bad in (11, -1, 10.5, "abc", float("nan"), float("inf")):
                with self.subTest(key=key, value=bad):
                    self.assert_rejected({"defaults": {key: bad}}, key)

    def test_out_of_range_lyric_story_strength_is_rejected(self):
        for bad in (11, -1, "abc", float("nan")):
            with self.subTest(value=bad):
                self.assert_rejected({"story": {"lyric_story_strength": bad}}, "lyric_story_strength")

    def test_the_whole_range_is_accepted(self):
        for value in (0, 4, 7.5, 10, "8"):
            with self.subTest(value=value):
                story.set_story_settings("Song", {
                    "defaults": {key: value for key in DEFAULT_NUMBERS},
                    "story": {"lyric_story_strength": value},
                })
                session = self.read_session()
                for key in DEFAULT_NUMBERS:
                    self.assertEqual(session["builder_storyboard_defaults"][key], float(value), key)
                self.assertEqual(session["builder_story_layer"]["lyric_story_strength"], float(value))


if __name__ == "__main__":
    unittest.main()
