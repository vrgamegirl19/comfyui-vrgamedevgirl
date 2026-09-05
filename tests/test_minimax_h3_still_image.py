import ast
import importlib
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "VRGDG_MiniMaxH3ImageReference.py"
INIT_SOURCE = ROOT / "__init__.py"


def load_prompt_helper():
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_still_prompt"
    )
    namespace = {"ValueError": ValueError, "str": str, "int": int, "max": max, "range": range}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace["_build_still_prompt"]


class MiniMaxH3StillImageTests(unittest.TestCase):
    def test_text_to_image_prompt_uses_static_base_format(self):
        result = load_prompt_helper()("A red fox in fresh snow", 0, "balanced")
        self.assertIn("integrated_multimodal_description:", result)
        self.assertIn("A red fox in fresh snow", result)
        self.assertIn("camera is completely locked", result)
        self.assertIn("overall_soundscape: Silence.", result)

    def test_reference_edit_prompt_labels_primary_and_supporting_images(self):
        result = load_prompt_helper()("Replace the coat with a blue jacket", 3, "preserve")
        self.assertIn("<Picture 1> is the primary source image to edit", result)
        self.assertIn("<Picture 2> is an additional visual reference", result)
        self.assertIn("<Picture 3> is an additional visual reference", result)
        self.assertIn("fully_preserved", result)
        self.assertIn("Replace the coat with a blue jacket", result)

    def test_creative_mode_uses_weak_reference(self):
        result = load_prompt_helper()("Turn this into a watercolor", 1, "creative")
        self.assertIn("weak_reference", result)
        self.assertIn("allowing composition", result)

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            load_prompt_helper()("   ", 0, "balanced")

    def test_nodes_are_registered(self):
        source = NODE_SOURCE.read_text(encoding="utf-8")
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"VRGDG_MiniMaxH3StillImage"', source)
        self.assertIn('"VRGDG_MiniMaxH3SelectStillFrame"', source)
        self.assertIn('".VRGDG_MiniMaxH3ImageReference"', init_source)

    @unittest.skipUnless(torch is not None, "Torch is provided by ComfyUI's Python environment.")
    def test_still_latent_samples_video_and_locks_audio(self):
        comfyui_root = ROOT.parents[1]
        for path in (str(ROOT), str(comfyui_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        module = importlib.import_module("VRGDG_MiniMaxH3ImageReference")
        latent, _ = module._empty_av_latent(64, 64, 5)
        locked = module._lock_silent_audio(latent)
        video_mask, audio_mask = locked["noise_mask"].unbind()
        self.assertTrue(torch.all(video_mask == 1))
        self.assertTrue(torch.all(audio_mask == 0))
        self.assertEqual(tuple(video_mask.shape), tuple(latent["samples"].unbind()[0].shape))
        self.assertEqual(tuple(audio_mask.shape), tuple(latent["samples"].unbind()[1].shape))


if __name__ == "__main__":
    unittest.main()
