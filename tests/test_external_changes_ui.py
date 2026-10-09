"""Open Video Builder and Storyboard windows merge Agent API / MCP edits without losing unsaved edits."""

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExternalChangesUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI behavior tests")
    def test_merge_rules_and_storyboard_save_protection(self) -> None:
        result = subprocess.run(
            ["node", "--test", str(ROOT / "tests" / "external_changes.mjs")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_builder_and_storyboard_are_wired_to_the_project_events(self) -> None:
        web = ROOT / "web"
        builder = (web / "music_video_builder" / "builder.mjs").read_text(encoding="utf-8")
        self.assertIn("createExternalChangeSync({", builder)
        self.assertIn("externalChangeSync?.dispose()", builder)
        storyboard = (web / "storyboard_builder" / "storyboard.mjs").read_text(encoding="utf-8")
        self.assertIn("window.addEventListener(STORYBOARD_EXTERNAL_CHANGE_EVENT, onExternalChange)", storyboard)
        self.assertIn("window.removeEventListener(STORYBOARD_EXTERNAL_CHANGE_EVENT, onExternalChange)", storyboard)
        events = (ROOT / "agent_api" / "project_events.py").read_text(encoding="utf-8")
        sync = (web / "music_video_builder" / "external_changes.mjs").read_text(encoding="utf-8")
        self.assertIn('EVENT_NAME = "vrgdg.project_changed"', events)
        self.assertIn('PROJECT_CHANGED_EVENT = "vrgdg.project_changed"', sync)
        # Every Storyboard file save goes through the revision-checked helper.
        for path in (web / "storyboard_builder").glob("*.mjs"):
            self.assertNotIn('postJson("/vrgdg/storyboard/save"', path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
