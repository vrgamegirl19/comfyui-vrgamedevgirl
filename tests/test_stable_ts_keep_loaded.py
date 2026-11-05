"""Back-to-back scene timing can keep one stable-ts model loaded instead of reloading it per scene."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

lyrics = importlib.import_module(f"{ROOT.name}.general.lyrics")


class FakeModel:
    def __getattr__(self, name):
        raise RuntimeError("fake stable-ts model")


class KeepLoadedTests(unittest.TestCase):
    def setUp(self):
        lyrics.unload_stable_ts_models()
        self.loads = []
        fake = types.SimpleNamespace(load_model=lambda name, device: self.loads.append(name) or FakeModel())
        patcher = patch.dict(sys.modules, {"stable_whisper": fake})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lyrics.unload_stable_ts_models)

    def run_once(self, **kwargs):
        audio = {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}
        try:
            lyrics.VRGDG_TimestampedLyricsExtractor().extract_timestamped_lyrics(audio, reference_lyrics="hold the light", **kwargs)
        except Exception:
            pass

    def test_kept_model_is_reused_and_unload_after_drops_it(self):
        self.run_once(unload_after=False)
        self.run_once(unload_after=False)
        self.assertEqual(len(self.loads), 1)
        self.run_once(unload_after=True)
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(lyrics.unload_stable_ts_models(), 0)

    def test_default_still_unloads_every_run(self):
        self.run_once()
        self.run_once()
        self.assertEqual(len(self.loads), 2)

    def test_clear_memory_unloads_a_kept_model(self):
        self.run_once(unload_after=False)
        self.assertEqual(lyrics.unload_stable_ts_models(), 1)
        self.run_once(unload_after=False)
        self.assertEqual(len(self.loads), 2)


if __name__ == "__main__":
    unittest.main()
