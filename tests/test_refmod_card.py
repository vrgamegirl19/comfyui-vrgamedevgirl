import shutil
import subprocess
import unittest
from pathlib import Path


class RefModCardJsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the RefMod card tests")
    def test_refmod_picker_reads_the_library_when_opened(self):
        result = subprocess.run(["node", "--test", str(Path(__file__).with_name("refmod_card.mjs"))],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
