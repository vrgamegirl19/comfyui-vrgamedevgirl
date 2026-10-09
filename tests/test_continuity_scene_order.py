"""Visual continuity must follow scene timing independently of custom audio placement."""

import shutil
import subprocess
import unittest
from pathlib import Path


class ContinuitySceneOrderTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI tests")
    def test_continuity_order_and_predecessor_request(self) -> None:
        """Exercise the real predecessor selector and MiniMax continuity request in all video types."""
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("continuity_scene_order.cjs"))],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
