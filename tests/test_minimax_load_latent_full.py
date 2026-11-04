"""Load Whole Latent returns the scene's whole saved latent, picture and sound, unsliced.

Load Latent cuts the tail of a latent as context for the next scene. A re-decode needs every token and the audio that
goes with it, however long the scene is.
"""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
latent_continuation = importlib.import_module(f"{pkg_name}.minimax.latent_continuation")
latent_manager = importlib.import_module(f"{pkg_name}.minimax.latent_manager")

TOKENS = 62  # a long scene: more than the 192 frames Load Latent can hand back as context


class LoadWholeLatentTests(unittest.TestCase):
    def setUp(self):
        self.project = tempfile.mkdtemp(prefix="vrgdg_whole_latent_")
        self.addCleanup(shutil.rmtree, self.project, ignore_errors=True)
        generator = torch.Generator().manual_seed(7)
        self.video = torch.randn(1, 24, TOKENS, 6, 10, generator=generator)
        self.audio = torch.randn(1, 8, 4, 100, generator=generator)
        latent_manager.SceneLatentManager.save_latent(self.project, 3, {"video": self.video, "audio": self.audio})

    def load(self, scene=3):
        return latent_continuation.VRGDG_MiniMaxH3LoadLatentFull().load(self.project, scene)

    def test_the_whole_latent_comes_back_unchanged(self):
        latent, frames = self.load()
        self.assertTrue(torch.equal(latent["video"], self.video))
        self.assertTrue(torch.equal(latent["audio"], self.audio))
        self.assertEqual(frames, latent_manager._tokens_to_frames(TOKENS))

    def test_the_sampler_style_samples_hold_picture_and_sound(self):
        latent, _frames = self.load()
        video, audio = latent["samples"].tensors
        self.assertTrue(torch.equal(video, self.video))
        self.assertTrue(torch.equal(audio, self.audio))

    def test_load_latent_would_have_cut_a_long_scene(self):
        sliced, tokens = latent_continuation.VRGDG_MiniMaxH3LoadLatent().load(self.project, 3, context_frames=192)
        self.assertLess(tokens, TOKENS)
        self.assertLess(sliced["video"].shape[2], TOKENS)

    def test_a_missing_latent_says_which_scene(self):
        with self.assertRaisesRegex(FileNotFoundError, "scene 004"):
            self.load(scene=4)

    def test_it_is_a_registered_node_with_a_single_latent_output(self):
        self.assertIs(latent_continuation.NODE_CLASS_MAPPINGS["VRGDG_MiniMaxH3LoadLatentFull"],
                      latent_continuation.VRGDG_MiniMaxH3LoadLatentFull)
        self.assertEqual(latent_continuation.VRGDG_MiniMaxH3LoadLatentFull.RETURN_TYPES, ("LATENT", "INT"))

    def test_it_is_reloaded_when_the_latent_changes(self):
        node = latent_continuation.VRGDG_MiniMaxH3LoadLatentFull
        before = node.IS_CHANGED(self.project, 3)
        latent_manager.SceneLatentManager.save_latent(self.project, 3, {"video": self.video * 2, "audio": self.audio})
        self.assertNotEqual(before, node.IS_CHANGED(self.project, 3))
        self.assertEqual(node.IS_CHANGED(self.project, 9), "missing")


if __name__ == "__main__":
    unittest.main()
