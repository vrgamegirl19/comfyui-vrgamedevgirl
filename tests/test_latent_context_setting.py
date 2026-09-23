import shutil
import subprocess
import unittest
from pathlib import Path


class LatentContextSettingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI behavior tests")
    def test_context_setting_persistence(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("latent_context_setting.cjs"))],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
