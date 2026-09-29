import ast
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "VRGDG_StandaloneVideoEnhancerNodes.py"
UI_SOURCE = ROOT / "web" / "VRGDG_StandaloneVideoEnhancer.js"
INIT_SOURCE = ROOT / "__init__.py"


def load_effect_helpers():
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    names = {
        "_auto_batch_size",
        "_apply_unsharp",
        "_apply_seeded_grain",
        "_apply_effects_batch",
    }
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace


class StandaloneVideoEnhancerTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "Torch is provided by ComfyUI's Python environment.")
    def test_grain_is_stable_across_different_batch_boundaries(self):
        helpers = load_effect_helpers()
        apply_effects = helpers["_apply_effects_batch"]
        frames = torch.full((4, 12, 16, 3), 0.5, dtype=torch.float32)
        settings = {
            "sharpen_enabled": False,
            "grain_enabled": True,
            "grain_intensity": 0.04,
            "saturation_mix": 0.5,
            "seed": 42,
            "use_gpu": False,
        }
        complete = apply_effects(frames, settings, 100)
        split = torch.cat(
            (
                apply_effects(frames[:2], settings, 100),
                apply_effects(frames[2:], settings, 102),
            ),
            dim=0,
        )
        self.assertTrue(torch.equal(complete, split))

    def test_automatic_batch_size_decreases_for_larger_frames(self):
        tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_auto_batch_size"
        )
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
        auto_batch = namespace["_auto_batch_size"]
        self.assertEqual(auto_batch(1280, 720), 16)
        self.assertEqual(auto_batch(1920, 1080), 8)
        self.assertEqual(auto_batch(2560, 1440), 4)
        self.assertEqual(auto_batch(3072, 1728), 2)
        self.assertEqual(auto_batch(3840, 2160), 1)

    def test_fake_upscale_preserves_aspect_orientation_and_never_downscales(self):
        tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_output_dimensions"
        )
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
        output_dimensions = namespace["_output_dimensions"]
        self.assertEqual(output_dimensions(1920, 1080, "2k"), (2560, 1440))
        self.assertEqual(output_dimensions(1080, 1920, "4k"), (2160, 3840))
        self.assertEqual(output_dimensions(1280, 960, "3k"), (3072, 2304))
        self.assertEqual(output_dimensions(3840, 2160, "2k"), (3840, 2160))
        self.assertEqual(output_dimensions(1920, 1080, "original"), (1920, 1080))

    def test_backend_exposes_preview_and_checkpointed_render_routes(self):
        source = NODE_SOURCE.read_text(encoding="utf-8")
        for route in (
            "/vrgdg/video_enhancer/upload",
            "/vrgdg/video_enhancer/load",
            "/vrgdg/video_enhancer/preview",
            "/vrgdg/video_enhancer/render/start",
            "/vrgdg/video_enhancer/render/status",
            "/vrgdg/video_enhancer/render/cancel",
            "/vrgdg/video_enhancer/media",
        ):
            self.assertIn(route, source)
        self.assertIn('"completed_segments": sorted(completed)', source)
        self.assertIn("frames_per_segment", source)
        self.assertIn("_process_with_retry", source)
        self.assertIn('"1:a?"', source)
        self.assertIn('job.pop("process", None)', source)
        self.assertIn("shutil.rmtree(segments_folder, ignore_errors=True)", source)
        self.assertIn("cv2.INTER_LANCZOS4", source)
        self.assertIn('"upscale_resolution": upscale_resolution', source)

    def test_node_and_clean_ui_are_registered(self):
        source = NODE_SOURCE.read_text(encoding="utf-8")
        ui = UI_SOURCE.read_text(encoding="utf-8")
        init = INIT_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"VRGDGStandaloneVideoEnhancer": VRGDGStandaloneVideoEnhancer', source)
        self.assertIn('".VRGDG_StandaloneVideoEnhancerNodes"', init)
        self.assertIn('const NODE_NAME = "VRGDGStandaloneVideoEnhancer"', ui)
        self.assertIn("Preview Current Frame", ui)
        self.assertIn("Resume From Checkpoint", ui)
        self.assertIn("loadFinalComparison", ui)
        self.assertIn("Fake upscale / Output size", ui)
        self.assertIn("4K UHD / 2160p", ui)
        self.assertIn('this.addDOMWidget("video_enhancer_ui"', ui)


if __name__ == "__main__":
    unittest.main()
