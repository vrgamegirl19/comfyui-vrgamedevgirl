"""Story Builder (Story Mode) UI: tab progress rules and a smoke test of the window on a fake DOM."""

import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the Story Builder tests")
class StoryBuilderTests(unittest.TestCase):
    def run_node(self, name):
        result = subprocess.run(["node", "--test", str(Path(__file__).with_name(name))],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_progress_rules(self):
        self.run_node("story_builder_data.mjs")

    def test_window_renders_every_tab(self):
        self.run_node("story_builder_ui.mjs")


if __name__ == "__main__":
    unittest.main()
