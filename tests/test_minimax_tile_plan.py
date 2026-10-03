"""Tests for the resolution-driven tile plan used by MiniMax H3 2 Pass Advanced."""

import importlib.util
import math
import os
import unittest

_MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "minimax", "tile_plan.py")
_SPEC = importlib.util.spec_from_file_location("vrgdg_minimax_tile_plan", _MODULE_PATH)
tile_plan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tile_plan)


def frame_size(megapixels: float, ratio_w: int = 16, ratio_h: int = 9):
    """Same rounding as miniMaxH3FrameSize in minimax_h3.mjs."""
    scale = math.sqrt(megapixels * 1048576 / (ratio_w * ratio_h))
    return round(ratio_w * scale / 32) * 32, round(ratio_h * scale / 32) * 32


class SolveEqualTilesTests(unittest.TestCase):
    def test_matches_node_solver_for_1080p(self):
        self.assertEqual(tile_plan.solve_equal_tiles(1920, 2, 128), (1024, 128))
        self.assertEqual(tile_plan.solve_equal_tiles(1920, 3, 128), (736, 144))
        self.assertEqual(tile_plan.solve_equal_tiles(1088, 2, 128), (608, 128))
        self.assertEqual(tile_plan.solve_equal_tiles(1088, 3, 128), (448, 128))

    def test_single_tile_covers_frame_without_overlap(self):
        self.assertEqual(tile_plan.solve_equal_tiles(1920, 1, 128), (1920, 0))

    def test_tiles_cover_the_frame_exactly(self):
        for total in (1280, 1920, 2240, 3840):
            for count in range(2, 6):
                tile, overlap = tile_plan.solve_equal_tiles(total, count, 128)
                self.assertEqual(count * tile - (count - 1) * overlap, total)
                self.assertEqual(tile % 32, 0)
                self.assertEqual(overlap % 16, 0)


class PlanSpatialTilesTests(unittest.TestCase):
    def grid(self, plan):
        return f"{plan['grid_rows']}x{plan['grid_cols']}"

    def test_grids_match_the_builder_ui_planner(self):
        four_k = frame_size(7.9688)
        for preset, expected in (("24gb", "3x4"), ("16gb", "4x5"), ("12gb", "5x5"), ("8gb", "6x7")):
            self.assertEqual(self.grid(tile_plan.plan_spatial_tiles(*four_k, preset)), expected, preset)
        two_k = frame_size(1.9922)
        self.assertEqual(self.grid(tile_plan.plan_spatial_tiles(*two_k, "24gb")), "2x2")
        self.assertEqual(self.grid(tile_plan.plan_spatial_tiles(*two_k, "16gb")), "2x2")

    def test_chunk_and_overlap_come_from_the_preset(self):
        plan = tile_plan.plan_spatial_tiles(3840, 2176, "24gb")
        self.assertEqual(plan["chunk_length"], 153)
        self.assertEqual(plan["spatial_w_overlap"], 160)
        for preset in tile_plan.VRAM_PRESETS:
            self.assertEqual(tile_plan.plan_spatial_tiles(1920, 1088, preset)["chunk_length"] % 17, 0)

    def test_solved_tiles_respect_node_limits(self):
        for width, height in ((1280, 704), (1920, 1088), (2240, 1248), (3840, 2176), (704, 1280)):
            for preset in tile_plan.VRAM_PRESETS:
                plan = tile_plan.plan_spatial_tiles(width, height, preset)
                self.assertGreaterEqual(plan["tile_width"], plan["min_tile_size"])
                self.assertGreaterEqual(plan["tile_height"], plan["min_tile_size"])
                self.assertEqual(plan["tile_width"] % 32, 0)
                self.assertEqual(plan["tile_height"] % 32, 0)
                self.assertLessEqual(plan["fade_width"], max(plan["spatial_w_overlap"], 0))
                self.assertEqual(
                    plan["grid_cols"] * plan["tile_width"] - (plan["grid_cols"] - 1) * plan["solved_overlap_w"], width)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            tile_plan.plan_spatial_tiles(1920, 1080, "32gb")
        for preset in ("32gb", "48gb", "custom"):
            with self.assertRaises(ValueError):
                tile_plan.plan_spatial_tiles(1920, 1088, preset)

    def test_32gb_is_not_a_preset(self):
        self.assertNotIn("32gb", tile_plan.VRAM_PRESETS)
        self.assertIn(tile_plan.DEFAULT_VRAM_PRESET, tile_plan.VRAM_PRESETS)

    def test_hidden_settings_match_the_tested_workflow(self):
        hidden = tile_plan.HIDDEN_ADVANCED_SETTINGS
        self.assertEqual(hidden["tile_size_mode"], "rows_cols")
        self.assertEqual((hidden["overlap_mode"], hidden["overlap_blend"]), ("later", "linear"))
        self.assertFalse(hidden["brightness_match"])


if __name__ == "__main__":
    unittest.main()
