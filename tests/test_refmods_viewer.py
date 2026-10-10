import shutil
import subprocess
import unittest
from pathlib import Path


class RefModsViewerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the viewer tests")
    def test_viewer_helpers(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("refmods_viewer.mjs"))],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
