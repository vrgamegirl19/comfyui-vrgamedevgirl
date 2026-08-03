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
NODE_SOURCE = ROOT / "VRGDG_MiniMaxH3AudioDrive.py"
INIT_SOURCE = ROOT / "__init__.py"


def load_helpers():
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    names = {"_fit_audio_latent"}
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch, "ValueError": ValueError}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace


class MiniMaxH3AudioDriveTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "Torch is provided by ComfyUI's Python environment.")
    def test_audio_latent_is_trimmed_to_generation_duration(self):
        fit_audio = load_helpers()["_fit_audio_latent"]
        encoded = torch.arange(1 * 32 * 2 * 12, dtype=torch.float32).reshape(1, 32, 2, 12)
        template = torch.zeros((1, 32, 2, 8), dtype=torch.float32)
        result = fit_audio(encoded, template)
        self.assertEqual(tuple(result.shape), tuple(template.shape))
        self.assertTrue(torch.equal(result, encoded[..., :8]))

    @unittest.skipUnless(torch is not None, "Torch is provided by ComfyUI's Python environment.")
    def test_audio_latent_is_zero_padded_to_generation_duration(self):
        fit_audio = load_helpers()["_fit_audio_latent"]
        encoded = torch.ones((1, 32, 2, 5), dtype=torch.float32)
        template = torch.zeros((1, 32, 2, 9), dtype=torch.float32)
        result = fit_audio(encoded, template)
        self.assertEqual(tuple(result.shape), tuple(template.shape))
        self.assertTrue(torch.equal(result[..., :5], encoded))
        self.assertEqual(torch.count_nonzero(result[..., 5:]).item(), 0)

    def test_node_is_registered_and_documents_original_audio_passthrough(self):
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        node_source = NODE_SOURCE.read_text(encoding="utf-8")
        self.assertIn('".VRGDG_MiniMaxH3AudioDrive"', init_source)
        self.assertIn('RETURN_NAMES = ("audio_driven_av_latent", "original_audio")', node_source)
        self.assertIn("torch.ones_like(video_latent)", node_source)
        self.assertIn("torch.zeros_like(encoded_audio)", node_source)
        self.assertIn("return output, source_audio", node_source)

    @unittest.skipUnless(torch is not None, "Torch is provided by ComfyUI's Python environment.")
    def test_node_locks_audio_and_returns_original_waveform_object(self):
        comfyui_root = ROOT.parents[1]
        for path in (str(comfyui_root), str(ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        module = importlib.import_module("VRGDG_MiniMaxH3AudioDrive")

        class FakeAudioVAE:
            audio_sample_rate = 32000

            def encode(self, waveform):
                self.received_shape = tuple(waveform.shape)
                return torch.full((1, 32, 2, 7), 0.25, dtype=torch.float32)

        video = torch.zeros((1, 24, 2, 4, 4), dtype=torch.float32)
        empty_audio = torch.zeros((1, 32, 2, 10), dtype=torch.float32)
        av_latent = {
            "samples": module.comfy.nested_tensor.NestedTensor((video, empty_audio)),
            "keep": "metadata",
        }
        source_audio = {
            "waveform": torch.ones((1, 2, 3200), dtype=torch.float32),
            "sample_rate": 32000,
        }
        vae = FakeAudioVAE()

        locked, passthrough = module.VRGDG_MiniMaxH3AudioDrive().apply_audio_drive(
            av_latent, source_audio, vae
        )

        locked_video, locked_audio = locked["samples"].unbind()
        video_mask, audio_mask = locked["noise_mask"].unbind()
        self.assertIs(passthrough, source_audio)
        self.assertEqual(locked["keep"], "metadata")
        self.assertIs(locked_video, video)
        self.assertEqual(tuple(locked_audio.shape), tuple(empty_audio.shape))
        self.assertEqual(torch.count_nonzero(locked_audio[..., 7:]).item(), 0)
        self.assertTrue(torch.all(video_mask == 1))
        self.assertTrue(torch.all(audio_mask == 0))
        self.assertEqual(vae.received_shape, (1, 3200, 2))


if __name__ == "__main__":
    unittest.main()
