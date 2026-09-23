import shutil
import subprocess
import unittest
from pathlib import Path


class ReferenceBindingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI behavior tests")
    def test_reference_bindings(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("minimax_reference_bindings.cjs"))],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
