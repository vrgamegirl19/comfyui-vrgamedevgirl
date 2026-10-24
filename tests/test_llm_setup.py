"""Builder LLM setup and resume behavior, including the nested Runner dialog."""

import shutil
import subprocess
import unittest
from pathlib import Path


class LlmSetupTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI behavior tests")
    def test_llm_setup_and_resume(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("llm_setup.cjs"))],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
