import shutil
import subprocess
import unittest
from pathlib import Path


class RenderETATests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for ETA tests")
    def test_estimates(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("render_eta.mjs"))],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
