import ast
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "post_process/lut_video_tools.py"


def load_iv_adjustments():
    spec = importlib.util.spec_from_file_location("vrgdg_iv_adjustments_test", ROOT / "post_process/luts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IV = load_iv_adjustments()


def load_functions(*names):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"VRGDG_LUTS": IV.VRGDG_LUTS}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


FUNCTIONS = load_functions("_frames_to_tensor", "_tensor_to_frames", "_write_ffmpeg_cube")


class LutVideoEffectTests(unittest.TestCase):
    def test_frame_round_trip_matches_the_previous_cpu_conversion(self):
        rng = np.random.default_rng(3)
        frames = [rng.integers(0, 256, (6, 5, 3), dtype=np.uint8) for _ in range(4)]
        tensor = FUNCTIONS["_frames_to_tensor"](frames, "cpu")
        # Previous path: BGR->RGB on the CPU, then float32 / 255.
        expected = np.stack([frame[..., ::-1] for frame in frames]).astype(np.float32) / 255.0
        self.assertTrue(torch.equal(tensor, torch.from_numpy(expected)))
        # Previous write path: clip(x * 255).astype(uint8), then RGB->BGR.
        graded = tensor * 1.07 - 0.03
        expected_frames = np.clip(graded.numpy() * 255.0, 0, 255).astype(np.uint8)[..., ::-1]
        self.assertTrue(np.array_equal(FUNCTIONS["_tensor_to_frames"](graded), expected_frames))

    def test_ffmpeg_cube_keeps_the_parsed_lut_and_domain(self):
        with tempfile.TemporaryDirectory() as folder:
            size = 3
            values = [f"{r / 2:.3f} {g / 2 * 0.9:.3f} {b / 2 * 0.8 + 0.1:.3f}" for b in range(size) for g in range(size) for r in range(size)]
            Path(folder, "fox_ears.cube").write_text(
                "TITLE \"fennec\"\n# comment\nLUT_3D_SIZE 3\nDOMAIN_MIN 0 0.1 0\nDOMAIN_MAX 1 0.9 1\n" + "\n".join(values) + "\n",
                encoding="utf-8",
            )
            original_dir = IV.LUTS_DIR
            IV.LUTS_DIR = folder
            try:
                FUNCTIONS["_write_ffmpeg_cube"]("fox_ears.cube", os.path.join(folder, "lut.cube"))
                original = IV.VRGDG_LUTS._parse_cube_file(os.path.join(folder, "fox_ears.cube"))
                rewritten = IV.VRGDG_LUTS._parse_cube_file(os.path.join(folder, "lut.cube"))
            finally:
                IV.LUTS_DIR = original_dir
        self.assertEqual(rewritten["size"], 3)
        self.assertTrue(torch.allclose(rewritten["lut"], original["lut"], atol=1e-6))
        self.assertTrue(torch.allclose(rewritten["domain_min"], original["domain_min"]))
        self.assertTrue(torch.allclose(rewritten["domain_max"], original["domain_max"]))


if __name__ == "__main__":
    unittest.main()
