import importlib.util
import pathlib
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "VRGDG_EnsureVideoAudio.py"

SPEC = importlib.util.spec_from_file_location("vrgdg_ensure_video_audio", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BrokenLazyAudio:
    def __getitem__(self, key):
        raise RuntimeError("ffmpeg found no audio stream")


class EnsureVideoAudioTests(unittest.TestCase):
    def test_valid_audio_passes_through(self):
        waveform = torch.ones((1, 2, 16000), dtype=torch.float32)
        source = {"waveform": waveform, "sample_rate": 32000}

        audio, has_source_audio, status = MODULE.ensure_video_audio(
            source, {"source_duration": 10.0}
        )

        self.assertIs(audio["waveform"], waveform)
        self.assertEqual(audio["sample_rate"], 32000)
        self.assertTrue(has_source_audio)
        self.assertIn("Using source audio", status)

    def test_missing_audio_becomes_video_length_silence(self):
        audio, has_source_audio, status = MODULE.ensure_video_audio(
            BrokenLazyAudio(),
            {"source_duration": 1.25},
            fallback_sample_rate=32000,
            fallback_channels=2,
        )

        self.assertEqual(audio["waveform"].shape, (1, 2, 40000))
        self.assertEqual(audio["sample_rate"], 32000)
        self.assertEqual(torch.count_nonzero(audio["waveform"]).item(), 0)
        self.assertFalse(has_source_audio)
        self.assertIn("generated 1.250s", status)

    def test_duration_can_fall_back_to_frame_count_and_fps(self):
        audio, has_source_audio, _ = MODULE.ensure_video_audio(
            BrokenLazyAudio(),
            {"source_frame_count": 60, "source_fps": 24},
            fallback_sample_rate=8000,
            fallback_channels=1,
        )

        self.assertEqual(audio["waveform"].shape, (1, 1, 20000))
        self.assertFalse(has_source_audio)


if __name__ == "__main__":
    unittest.main()
