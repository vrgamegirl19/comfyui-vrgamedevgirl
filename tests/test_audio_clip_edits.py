"""Edited audio keeps source offsets and renders real silence between pieces."""

import importlib
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
audio_clips = importlib.import_module(f"{ROOT.name}.builder.audio_clips")


class AudioClipEditsTests(unittest.TestCase):
    def test_reimport_keeps_previous_clip_sources(self) -> None:
        """Importing another file for the same scene cannot overwrite its old pieces."""
        audio = importlib.import_module(f"{ROOT.name}.builder.audio")
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "speech.wav"
            payload = {"project_folder": folder, "scene_number": 1, "source_path": str(source),
                       "preserve_source": True}
            with patch.object(audio, "_read_audio_peaks", return_value={"duration": 1, "peaks": []}):
                source.write_bytes(b"first original")
                first = audio._save_scene_audio(payload)
                source.write_bytes(b"second original")
                second = audio._save_scene_audio(payload)
            self.assertNotEqual(first["saved_path"], second["saved_path"])
            self.assertEqual(Path(first["saved_path"]).read_bytes(), b"first original")
            self.assertEqual(Path(second["saved_path"]).read_bytes(), b"second original")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_frontend_audio_edits(self) -> None:
        result = subprocess.run(["node", "--test", str(ROOT / "tests/audio_clip_edits.cjs")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_mix_has_exact_silent_gaps_and_preserves_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "speech.wav"
            rate = 44100
            values = [int((8000 if i < rate else 16000) * math.sin(2 * math.pi * 440 * i / rate))
                      for i in range(rate * 2)]
            with wave.open(str(source), "wb") as output:
                output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                output.writeframes(struct.pack(f"<{len(values)}h", *values))
            original = source.read_bytes()
            clips = [{"path": str(source), "start": 0.5, "source_start": 0, "duration": 0.5},
                     {"path": str(source), "start": 2, "source_start": 1, "duration": 0.5}]
            result = audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": clips, "duration": 4})
            with wave.open(result["audio_path"], "rb") as rendered:
                self.assertEqual(rendered.getnframes(), rate * 4)
                self.assertEqual(rendered.getnchannels(), 2)
                samples = struct.unpack(f"<{rate * 4 * 2}h", rendered.readframes(rate * 4))
            def rms(start: float, end: float) -> float:
                """Measure the selected region of the rendered stereo track."""
                region = samples[int(start * rate) * 2:int(end * rate) * 2]
                return math.sqrt(sum(value * value for value in region) / len(region))
            self.assertEqual(rms(0, 0.4), 0)
            self.assertEqual(rms(1.1, 1.9), 0)
            self.assertEqual(rms(2.6, 4), 0)
            self.assertGreater(rms(0.6, 0.9), 3000)
            self.assertGreater(rms(2.1, 2.4), rms(0.6, 0.9) * 1.8)
            self.assertEqual(source.read_bytes(), original)
            cached = audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": clips, "duration": 4})
            self.assertEqual(cached["audio_path"], result["audio_path"])
            silence = audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": [], "duration": 4})
            with wave.open(silence["audio_path"], "rb") as rendered:
                self.assertFalse(any(rendered.readframes(rate * 4)))

    def test_invalid_timing_is_rejected_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": [], "duration": value})


if __name__ == "__main__":
    unittest.main()
