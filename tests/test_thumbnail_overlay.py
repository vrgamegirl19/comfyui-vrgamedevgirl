"""The Thumbnail dialog's Overlay engine: the YouTube look made by layers on the chosen frame, with no image model."""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
overlay = importlib.import_module(f"{pkg_name}.builder.thumbnail_overlay")
thumbnail = importlib.import_module(f"{pkg_name}.builder.thumbnail")

WIDTH, HEIGHT = 640, 360


def busy_left_frame() -> Image.Image:
    """A busy left half (noise: lots of edges) and a calm right half (a smooth gradient)."""
    rng = np.random.default_rng(3)
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[:, : WIDTH // 2] = rng.integers(0, 256, (HEIGHT, WIDTH // 2, 3), dtype=np.uint8)
    ramp = np.linspace(30, 90, WIDTH - WIDTH // 2, dtype=np.uint8)[None, :, None]
    array[:, WIDTH // 2:] = np.broadcast_to(ramp, (HEIGHT, WIDTH - WIDTH // 2, 3))
    return Image.fromarray(array)


def busy_right_frame() -> Image.Image:
    return busy_left_frame().transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def color_count(image: Image.Image, box: dict, color, tolerance=40) -> int:
    """How many pixels inside the title box are close to ``color``."""
    crop = pixels(image)[box["y"]: box["y"] + box["height"], box["x"]: box["x"] + box["width"]].astype(int)
    return int((np.abs(crop - np.array(color)).max(axis=2) <= tolerance).sum())


class StrengthTests(unittest.TestCase):
    def test_a_strength_is_a_whole_number_from_0_to_100(self):
        self.assertEqual(overlay.overlay_strength(60), 60)
        self.assertEqual(overlay.overlay_strength("35.4"), 35)
        self.assertEqual(overlay.overlay_strength(-5), 0)
        self.assertEqual(overlay.overlay_strength(900), 100)
        for bad in (None, "", "abc", float("nan"), float("inf"), [1]):
            self.assertEqual(overlay.overlay_strength(bad), overlay.OVERLAY_DEFAULT_STRENGTH, repr(bad))

    def test_strength_0_without_a_title_is_the_frame_byte_for_byte(self):
        frame = busy_left_frame()
        result, info = overlay.render_overlay(frame, "", "yellow", "auto", 0)
        self.assertTrue(np.array_equal(pixels(result), pixels(frame)))
        self.assertIsNone(info["title_box"])
        self.assertEqual(info["lines"], 0)

    def test_a_stronger_overlay_moves_further_from_the_frame(self):
        frame = busy_right_frame()
        source = pixels(frame).astype(float)
        moved = [float(np.abs(pixels(overlay.render_overlay(frame, "", "yellow", "auto", strength)[0]).astype(float) - source).mean())
                 for strength in (0, 20, 60, 100)]
        self.assertEqual(moved[0], 0.0)
        self.assertEqual(moved, sorted(moved))
        self.assertGreater(moved[1], 0.0)

    def test_the_grade_adds_color_and_contrast_to_a_flat_picture(self):
        flat = Image.new("RGB", (WIDTH, HEIGHT), (120, 90, 80))
        graded = overlay.grade_frame(flat, 60)
        before, after = pixels(flat)[HEIGHT // 2, WIDTH // 2].astype(int), pixels(graded)[HEIGHT // 2, WIDTH // 2].astype(int)
        self.assertGreater(int(after.max() - after.min()), int(before.max() - before.min()), "more color")
        self.assertEqual(overlay.grade_frame(flat, 0).mode, "RGB")
        # The vignette darkens the corners and leaves the middle alone.
        self.assertLess(int(pixels(graded)[2, 2].sum()), int(pixels(graded)[HEIGHT // 2, WIDTH // 2].sum()))


class TitleTests(unittest.TestCase):
    def test_the_title_goes_in_the_calm_part_of_the_frame(self):
        _, info = overlay.render_overlay(busy_left_frame(), "HEAT", "yellow", "auto", 60)
        box = info["title_box"]
        self.assertGreaterEqual(box["x"], WIDTH // 2 - 10, "not over the busy left half")
        self.assertLessEqual(box["x"] + box["width"], WIDTH)
        _, info = overlay.render_overlay(busy_right_frame(), "HEAT", "yellow", "auto", 60)
        self.assertLessEqual(info["title_box"]["x"] + info["title_box"]["width"], WIDTH // 2 + 10, "the calm half is now on the left")

    def test_the_position_limits_where_it_looks(self):
        calm_everywhere = Image.new("RGB", (WIDTH, HEIGHT), (40, 40, 60))
        boxes = {where: overlay.render_overlay(calm_everywhere, "HEAT", "yellow", where, 60)[1]["title_box"] for where in overlay.OVERLAY_POSITIONS}
        self.assertLessEqual(boxes["top"]["y"] + boxes["top"]["height"], int(HEIGHT * 0.42) + 1)
        self.assertGreaterEqual(boxes["bottom"]["y"], int(HEIGHT * 0.58) - 1)
        self.assertLessEqual(boxes["left"]["x"] + boxes["left"]["width"], int(WIDTH * 0.45) + 1)
        self.assertGreaterEqual(boxes["right"]["x"], int(WIDTH * 0.55) - 1)
        self.assertLess(boxes["auto"]["y"], HEIGHT // 3, "auto prefers the upper part when everything is equally calm")

    def test_every_box_stays_inside_the_frame_and_off_the_edge(self):
        for where in overlay.OVERLAY_POSITIONS:
            box = overlay.render_overlay(busy_left_frame(), "HEAT", "yellow", where, 60)[1]["title_box"]
            self.assertGreater(box["x"], 0, where)
            self.assertGreater(box["y"], 0, where)
            self.assertLessEqual(box["x"] + box["width"], WIDTH, where)
            self.assertLessEqual(box["y"] + box["height"], HEIGHT, where)

    def test_the_title_is_drawn_in_its_color_and_only_near_the_box_when_the_grade_is_off(self):
        frame = busy_right_frame()
        for name, look in overlay.OVERLAY_COLORS.items():
            with self.subTest(color=name):
                result, info = overlay.render_overlay(frame, "HEAT", name, "auto", 0)
                self.assertEqual(info["color"], name)
                self.assertGreater(color_count(result, info["title_box"], look["fill"], tolerance=60), 800, "letters of that color")
        result, info = overlay.render_overlay(frame, "HEAT", "yellow", "auto", 0)
        box = info["title_box"]
        pad = int(box["height"] * 0.9)
        outside = np.ones((HEIGHT, WIDTH), dtype=bool)
        outside[max(0, box["y"] - pad): box["y"] + box["height"] + pad, max(0, box["x"] - pad): box["x"] + box["width"] + pad] = False
        self.assertTrue(np.array_equal(pixels(result)[outside], pixels(frame)[outside]), "nothing else in the frame changed")

    def test_a_long_title_wraps_to_fit_and_a_short_one_stays_on_one_line(self):
        _, short = overlay.render_overlay(busy_left_frame(), "HEAT", "yellow", "auto", 60)
        self.assertEqual(short["lines"], 1)
        _, long = overlay.render_overlay(busy_left_frame(), "Never look back again tonight my friend", "yellow", "auto", 60)
        self.assertGreaterEqual(long["lines"], 2)
        self.assertLessEqual(long["lines"], 3)
        box = long["title_box"]
        self.assertLessEqual(box["x"] + box["width"], WIDTH)
        self.assertLessEqual(box["y"] + box["height"], HEIGHT)
        _, one_word = overlay.render_overlay(busy_left_frame(), "SUPERCALIFRAGILISTICEXPIALIDOCIOUS", "yellow", "auto", 60)
        self.assertEqual(one_word["lines"], 1, "a single word cannot be split")
        self.assertLessEqual(one_word["title_box"]["width"], WIDTH)

    def test_the_title_is_tidied_and_cut_to_a_length_that_can_be_read(self):
        _, info = overlay.render_overlay(busy_left_frame(), "   " + "word " * 60, "yellow", "auto", 60)
        self.assertGreaterEqual(info["lines"], 2)
        _, blank = overlay.render_overlay(busy_left_frame(), "   \n\t ", "yellow", "auto", 60)
        self.assertIsNone(blank["title_box"])

    def test_an_unknown_color_or_position_falls_back_to_the_defaults(self):
        _, info = overlay.render_overlay(busy_left_frame(), "HEAT", "chartreuse", "diagonal", 60)
        self.assertEqual((info["color"], info["position"]), ("yellow", "auto"))

    def test_without_a_font_on_the_system_the_default_font_still_draws_the_title(self):
        with patch.object(overlay, "_FONT_CANDIDATES", ()):
            result, info = overlay.render_overlay(busy_right_frame(), "HEAT", "white", "auto", 0)
        self.assertEqual(info["font"], "default")
        self.assertGreater(color_count(result, info["title_box"], (255, 255, 255), tolerance=40), 100)

    def test_the_calmest_place_is_looked_for_on_the_frame_as_it_was(self):
        frame = busy_left_frame()
        weak = overlay.render_overlay(frame, "HEAT", "yellow", "auto", 0)[1]["title_box"]
        strong = overlay.render_overlay(frame, "HEAT", "yellow", "auto", 100)[1]["title_box"]
        self.assertEqual(weak, strong)


class MakeOverlayTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.project = os.path.join(self.folder.name, "project")
        os.makedirs(self.project)
        self.source = os.path.join(self.project, "source.png")
        busy_left_frame().save(self.source)

    def test_each_overlay_is_kept_as_its_own_numbered_png_in_the_thumbnail_folder(self):
        first = overlay._make_thumbnail_overlay({"project_folder": self.project, "source_path": self.source, "title": "HEAT"})
        second = overlay._make_thumbnail_overlay({"project_folder": self.project, "source_path": self.source, "title": "HEAT", "color": "cyan"})
        self.assertEqual(os.path.basename(first["image_path"]), "thumbnail_0001.png")
        self.assertEqual(os.path.basename(second["image_path"]), "thumbnail_0002.png")
        self.assertEqual(os.path.dirname(first["image_path"]), os.path.join(self.project, "thumbnail"))
        self.assertEqual((first["width"], first["height"]), (WIDTH, HEIGHT))
        self.assertEqual((first["color"], second["color"]), ("yellow", "cyan"))
        with Image.open(first["image_path"]) as saved:
            self.assertEqual(saved.size, (WIDTH, HEIGHT))
            self.assertEqual(saved.mode, "RGB")

    def test_it_shares_the_numbering_with_the_images_the_other_workflows_make(self):
        thumbnail._save_thumbnail_image({"project_folder": self.project, "source_path": self.source})
        made = overlay._make_thumbnail_overlay({"project_folder": self.project, "source_path": self.source})
        self.assertEqual(os.path.basename(made["image_path"]), "thumbnail_0002.png")

    def test_the_options_are_passed_on_and_a_bad_strength_is_the_default(self):
        made = overlay._make_thumbnail_overlay({
            "project_folder": self.project, "source_path": self.source, "title": "HEAT", "position": "right", "strength": "nope"})
        self.assertEqual((made["position"], made["strength"]), ("right", overlay.OVERLAY_DEFAULT_STRENGTH))
        self.assertGreaterEqual(made["title_box"]["x"], int(WIDTH * 0.55) - 1)

    def test_no_title_means_only_the_grade(self):
        made = overlay._make_thumbnail_overlay({"project_folder": self.project, "source_path": self.source})
        self.assertIsNone(made["title_box"])

    def test_a_missing_project_or_frame_is_refused_and_leaves_nothing_behind(self):
        with self.assertRaisesRegex(ValueError, "Project folder is empty"):
            overlay._make_thumbnail_overlay({"project_folder": "", "source_path": self.source})
        with self.assertRaisesRegex(ValueError, "Choose the frame first"):
            overlay._make_thumbnail_overlay({"project_folder": self.project, "source_path": "  "})
        with self.assertRaisesRegex(FileNotFoundError, "The frame was not found"):
            overlay._make_thumbnail_overlay({"project_folder": self.project, "source_path": os.path.join(self.project, "nope.png")})
        self.assertFalse(os.path.isdir(os.path.join(self.project, "thumbnail")) and os.listdir(os.path.join(self.project, "thumbnail")))

    def test_the_build_says_it_can_make_an_overlay(self):
        capabilities = thumbnail._thumbnail_capabilities()
        self.assertTrue(capabilities["overlay"])
        self.assertTrue(capabilities["frame"])
        self.assertEqual(capabilities["version"], 1)


if __name__ == "__main__":
    unittest.main()
