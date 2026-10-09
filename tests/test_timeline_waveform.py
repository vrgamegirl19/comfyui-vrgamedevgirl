"""Waveform samples use the same timestamp scale as scene cards and playback."""

import shutil
import subprocess
import unittest
from pathlib import Path


class TimelineWaveformTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI tests")
    def test_waveform_time_scale(self) -> None:
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("timeline_waveform.cjs"))],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
