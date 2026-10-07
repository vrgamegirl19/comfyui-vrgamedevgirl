import shutil
import subprocess
import unittest
from pathlib import Path


class RefModTrimTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the trim tests")
    def test_trim_helpers(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("refmod_trim.mjs"))],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
