"""Check Scene Options relocation and the saved global timing control."""

import shutil
import subprocess
import unittest
from pathlib import Path


class SceneOptionsLayoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI tests")
    def test_scene_options_and_global_timing_control(self) -> None:
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("scene_options_layout.cjs"))],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
