import shutil
import subprocess
import unittest
from pathlib import Path


class TimelinePromptSaveTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI behavior tests")
    def test_prompt_persistence_and_selection(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("timeline_prompt_save.cjs"))],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
