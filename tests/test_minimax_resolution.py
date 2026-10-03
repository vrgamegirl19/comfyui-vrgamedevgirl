"""Tests for the shared MiniMax H3 output-resolution math."""

import importlib.util
import os
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"vrgdg_{name}", os.path.join(_ROOT, "minimax", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolution = _load("resolution")
tile_plan = _load("tile_plan")


class ResolutionTests(unittest.TestCase):
    def test_preset_megapixels_match_the_builder_ui(self):
        wide = "16:9 (Widescreen)"
        self.assertEqual(resolution.preset_megapixels("1k", wide), 0.5625)
        self.assertEqual(resolution.preset_megapixels("2k", wide), 1.9922)
        self.assertEqual(resolution.preset_megapixels("4k", wide), 7.9688)
        self.assertIsNone(resolution.preset_megapixels("custom", wide))

    def test_frame_sizes_are_multiples_of_32(self):
        self.assertEqual(resolution.frame_size(1.9922, "16:9 (Widescreen)"), (1920, 1088))
        self.assertEqual(resolution.frame_size(0.5625, "16:9 (Widescreen)"), (1024, 576))
        self.assertEqual(resolution.frame_size(1.9922, "9:16 (Portrait Widescreen)"), (1088, 1920))
        for megapixels in (0.4, 0.9, 1.5, 2.1, 3.3):
            for aspect in ("16:9 (Widescreen)", "1:1 (Square)", "21:9 (Ultrawide)", "3:4 (Portrait Standard)"):
                width, height = resolution.frame_size(megapixels, aspect)
                self.assertEqual(width % 32, 0)
                self.assertEqual(height % 32, 0)

    def test_portrait_preset_uses_the_long_edge(self):
        self.assertEqual(resolution.output_frame_size({"resolution_preset": "2k", "aspect_ratio": "9:16 (Portrait Widescreen)"}), (1088, 1920))

    def test_custom_uses_megapixels_and_preset_wins_over_them(self):
        self.assertEqual(resolution.resolved_megapixels({"resolution_preset": "custom", "megapixels": 0.9, "aspect_ratio": "16:9 (Widescreen)"}), 0.9)
        self.assertEqual(resolution.resolved_megapixels({"resolution_preset": "2k", "megapixels": 0.9, "aspect_ratio": "16:9 (Widescreen)"}), 1.9922)


class VramPresetTests(unittest.TestCase):
    def test_retired_and_empty_presets_normalize(self):
        self.assertEqual(tile_plan.normalize_vram_preset("32gb"), "24gb")
        self.assertEqual(tile_plan.normalize_vram_preset("custom"), "24gb")
        self.assertEqual(tile_plan.normalize_vram_preset(""), tile_plan.DEFAULT_VRAM_PRESET)
        self.assertEqual(tile_plan.normalize_vram_preset(" 12GB "), "12gb")
        with self.assertRaises(ValueError):
            tile_plan.normalize_vram_preset("48gb")


if __name__ == "__main__":
    unittest.main()
