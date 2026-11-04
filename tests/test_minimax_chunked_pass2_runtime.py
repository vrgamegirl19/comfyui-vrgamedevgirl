"""Runs the chunked second pass node's CPU checks under ComfyUI's own Python.

The node imports ComfyUI modules (``comfy``, ``comfy_extras``) that the unit-test process does not have, and importing
them would change what other tests see. So ``chunked_pass2_runtime_checks.py`` runs in a subprocess, from the ComfyUI
folder, with ``..\\..\\python_embeded\\python.exe``. The test is skipped when that interpreter is not there.
"""

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
PYTHON = COMFY_ROOT.parent / "python_embeded" / "python.exe"
CHECKS = ROOT / "tests" / "chunked_pass2_runtime_checks.py"
SKIPPED = 77  # the checks' own "not running under ComfyUI's Python"


class ChunkedPass2RuntimeTests(unittest.TestCase):
    def test_runtime_checks_pass_under_comfyui_python(self) -> None:
        if not PYTHON.is_file():
            self.skipTest(f"ComfyUI's Python was not found at {PYTHON}")
        done = subprocess.run(
            [str(PYTHON), str(CHECKS)], cwd=str(COMFY_ROOT), capture_output=True, text=True, timeout=900,
        )
        if done.returncode == SKIPPED:
            self.skipTest(done.stdout.strip() or "ComfyUI modules could not be imported")
        self.assertEqual(done.returncode, 0, f"{done.stdout[-2500:]}\n{done.stderr[-4000:]}")


if __name__ == "__main__":
    unittest.main()
