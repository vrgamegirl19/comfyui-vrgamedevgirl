import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load_studio():
    package = importlib.util.module_from_spec(importlib.util.spec_from_loader("vrgdg_refmod_test", loader=None, is_package=True))
    package.__path__ = [str(ROOT)]
    sys.modules["vrgdg_refmod_test"] = package
    minimax = importlib.util.module_from_spec(importlib.util.spec_from_loader("vrgdg_refmod_test.minimax", loader=None, is_package=True))
    minimax.__path__ = [str(ROOT / "minimax")]
    sys.modules["vrgdg_refmod_test.minimax"] = minimax
    return (
        importlib.import_module("vrgdg_refmod_test.minimax.refmod_studio"),
        importlib.import_module("vrgdg_refmod_test.minimax.refmod_picker"),
    )


studio, picker = _load_studio()


class RefModStudioRequestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.images = []
        for name in ("front.png", "left.png", "right.png"):
            path = os.path.join(self.temp.name, name)
            with open(path, "wb") as handle:
                handle.write(b"x")
            self.images.append(path)
        self.image = self.images[0]

    def request(self, **overrides):
        payload = {"type": "identity", "name": "hero", "paths": list(self.images)}
        payload.update(overrides)
        return payload

    def test_valid_request_maps_mode_and_defaults(self):
        result = studio.validate_request(self.request())
        self.assertEqual(result["mode"], "encode")
        self.assertEqual(result["steps"], 500)
        self.assertFalse(result["overwrite"])
        result = studio.validate_request(self.request(mode="Compressed Reference", steps=150))
        self.assertEqual((result["mode"], result["steps"]), ("training", 150))

    def test_inactive_types_are_rejected(self):
        for concept_type in ("voice", "singing", "music_style", "sound_fx", "ambience", "bogus"):
            with self.assertRaises(ValueError):
                studio.validate_request(self.request(type=concept_type))

    def test_new_types_are_creatable(self):
        for concept_type in ("clothing_men", "clothing_women", "object", "prop", "vehicle", "creature", "background", "style"):
            self.assertEqual(studio.validate_request(self.request(type=concept_type, paths=[self.image]))["type"], concept_type)
        with self.assertRaises(ValueError):
            studio.validate_request(self.request(type="clothing"))

    def test_reference_builder_save_collects_the_card_image_then_extras(self):
        saved = []
        def fake_save(data, name):
            saved.append((data, name))
            return f"/tmp/{len(saved)}.png"
        paths = studio.collect_image_paths({
            "image": {"data": "AAAA", "name": "card.png"},
            "extra_images": [{"path": self.images[1]}, {"data": "BBBB"}, "junk", {}],
        }, save=fake_save)
        self.assertEqual(paths, ["/tmp/1.png", self.images[1], "/tmp/2.png"])
        self.assertEqual(saved, [("AAAA", "card.png"), ("BBBB", "reference.png")])
        self.assertEqual(studio.collect_image_paths({"image": {"path": self.image}}), [self.image])
        self.assertEqual(studio.collect_image_paths({}), [])

    def test_identity_works_with_any_number_of_images(self):
        for count in (1, 2, 3, 5):
            paths = (self.images * 2)[:count]
            self.assertEqual(studio.validate_request(self.request(paths=paths))["paths"], paths)
        with self.assertRaises(ValueError):
            studio.validate_request(self.request(paths=[]))

    def test_name_is_cleaned_and_required(self):
        self.assertEqual(studio.validate_request(self.request(name=" my/hero:1 "))["name"], "my_hero_1")
        for name in ("", "   ", "..."):
            with self.assertRaises(ValueError):
                studio.validate_request(self.request(name=name))

    def test_images_are_required_and_must_exist(self):
        with self.assertRaises(ValueError):
            studio.validate_request(self.request(paths=[]))
        with self.assertRaises(ValueError):
            studio.validate_request(self.request(paths=[os.path.join(self.temp.name, "missing.png")]))
        with self.assertRaises(ValueError):
            studio.validate_request(self.request(paths=[os.path.join(self.temp.name, "a.txt")]))

    def test_image_order_is_kept(self):
        extra = os.path.join(self.temp.name, "extra.png")
        with open(extra, "wb") as handle:
            handle.write(b"x")
        paths = [self.images[2], self.images[0], extra, self.images[1]]
        self.assertEqual(studio.validate_request(self.request(paths=paths))["paths"], paths)

    def test_steps_are_bounded(self):
        for steps in (-1, 2001, "many"):
            with self.assertRaises(ValueError):
                studio.validate_request(self.request(steps=steps))

    def test_fixed_settings_are_not_user_controlled(self):
        self.assertEqual(studio.FIXED_SETTINGS["ref_resolution"], 1024)
        self.assertEqual(studio.FIXED_SETTINGS["max_tokens"], 5120)
        self.assertNotIn("max_tokens", studio.validate_request(self.request(max_tokens=1)))


class RefModCanvasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def png(self, name, size, colour=(255, 255, 255), box=None, subject=(30, 30, 30)):
        from PIL import Image

        image = Image.new("RGB", size, colour)
        if box:
            image.paste(subject, box)
        path = os.path.join(self.temp.name, name)
        image.save(path)
        return path

    def test_canvas_matches_the_browser_twin(self):
        sizes = [(500, 560), (480, 560), (300, 1150)]
        self.assertEqual(studio.canvas_for(sizes, 1.0)[:2], (448, 1024))
        self.assertEqual(studio.canvas_for([(500, 560), (300, 1000)], 0.5)[:2], (320, 512))
        self.assertEqual(studio.canvas_for([(200, 300)], 1.0)[:2], (320, 320))
        self.assertEqual(studio.canvas_for([(611, 661), (581, 661), (330, 1179)], 1.0)[:2], (544, 1024))

    def test_quality_presets_match_the_browser(self):
        self.assertEqual(studio.QUALITY_SCALES, {"maximum": 1.0, "high": 0.8, "balanced": 0.6, "compact": 0.45, "draft": 0.3})
        self.assertEqual(studio.DEFAULT_QUALITY, "balanced")

    def test_lower_quality_means_fewer_tokens(self):
        sizes = [(600, 600), (400, 1100)]
        tokens = []
        for key in ("maximum", "high", "balanced", "compact", "draft"):
            width, height, _ = studio.canvas_for(sizes, studio.QUALITY_SCALES[key])
            tokens.append(studio.tokens_for_canvas(6, width, height))
        self.assertEqual(tokens, sorted(tokens, reverse=True))

    def test_rounding_is_half_up_like_the_browser(self):
        self.assertEqual(studio._snap32(528), 544)
        self.assertEqual(studio._snap32(5), 32)

    def test_tokens_follow_frames_and_canvas(self):
        self.assertEqual(studio.tokens_for_canvas(6, 512, 512), 1536)

    def test_images_share_one_canvas_without_cutting(self):
        wide = self.png("wide.png", (640, 320), box=(100, 50, 540, 270))
        tall = self.png("tall.png", (320, 960), box=(60, 80, 260, 900))
        tensors, canvas = studio.prepare_images([wide, tall], [None, None], "maximum")
        self.assertEqual(canvas, (640, 960))
        for tensor in tensors:
            self.assertEqual(tuple(tensor.shape), (1, 960, 640, 3))

    def test_crop_removes_the_margin_and_shrinks_the_canvas(self):
        wide = self.png("wide.png", (640, 320), box=(100, 50, 540, 270))
        tensors, canvas = studio.prepare_images([wide], [[90, 40, 550, 280]], "maximum")
        self.assertEqual(canvas, (448, 320))
        self.assertEqual(tuple(tensors[0].shape), (1, 320, 448, 3))

    def test_padding_uses_the_image_border_colour(self):
        grey = self.png("grey.png", (200, 200), colour=(120, 120, 120), box=(60, 60, 140, 140))
        tall = self.png("tall.png", (100, 400), colour=(255, 255, 255), box=(20, 50, 80, 350))
        tensors, canvas = studio.prepare_images([grey, tall], [None, None], "maximum")
        corner = tensors[0][0, 0, 0]
        self.assertAlmostEqual(float(corner[0]), 120 / 255, places=2)

    def test_quality_scales_everything_down(self):
        big = self.png("big.png", (1200, 800), box=(100, 100, 1100, 700))
        _, full = studio.prepare_images([big], [None], "maximum")
        _, balanced = studio.prepare_images([big], [None], "balanced")
        self.assertEqual(full, (1024, 672))
        self.assertLess(balanced[0], full[0])

    def test_bad_crops_are_rejected(self):
        image = self.png("a.png", (300, 300), box=(50, 50, 250, 250))
        with self.assertRaises(ValueError):
            studio.prepare_images([image], [[10, 10, 20, 20]], "maximum")
        with self.assertRaises(ValueError):
            studio.prepare_images([image], [["a", "b", "c", "d"]], "maximum")

    def test_request_validation_covers_crops_and_quality(self):
        image = os.path.join(self.temp.name, "front.png")
        from PIL import Image

        Image.new("RGB", (100, 100)).save(image)
        base = {"type": "clothing_women", "name": "n", "paths": [image]}
        request = studio.validate_request(base)
        self.assertEqual(request["crops"], [None])
        self.assertEqual(request["quality"], "balanced")
        with self.assertRaises(ValueError):
            studio.validate_request({**base, "crops": [None, None]})
        with self.assertRaises(ValueError):
            studio.validate_request({**base, "quality": "ultra"})
        self.assertEqual(studio.validate_request({**base, "quality": "compact"})["quality"], "compact")


class RefModProgressContextTests(unittest.TestCase):
    def setUp(self):
        from server import PromptServer

        self.server_class = PromptServer
        self.had_instance = "instance" in vars(PromptServer)
        self.previous = vars(PromptServer).get("instance")
        self.addCleanup(self.restore)

    def restore(self):
        if self.had_instance:
            self.server_class.instance = self.previous
        elif "instance" in vars(self.server_class):
            del self.server_class.instance

    def test_sets_ids_when_no_prompt_has_run(self):
        from types import SimpleNamespace

        self.server_class.instance = SimpleNamespace(last_node_id=None)
        studio._ensure_progress_context()
        self.assertEqual(self.server_class.instance.last_prompt_id, "vrgdg_refmod_studio")
        self.assertEqual(self.server_class.instance.last_node_id, "vrgdg_refmod_studio")

    def test_keeps_ids_from_an_earlier_prompt(self):
        from types import SimpleNamespace

        self.server_class.instance = SimpleNamespace(last_prompt_id="abc", last_node_id="7")
        studio._ensure_progress_context()
        self.assertEqual((self.server_class.instance.last_prompt_id, self.server_class.instance.last_node_id), ("abc", "7"))


class RefModDescribePromptTests(unittest.TestCase):
    def test_every_creatable_type_has_a_prompt(self):
        for concept_type in studio.ACTIVE_TYPES:
            self.assertIn(concept_type, picker.DESCRIBE_INSTRUCTIONS)

    def test_every_creatable_type_has_a_describe_prompt(self):
        for concept_type in studio.ACTIVE_TYPES:
            self.assertIn(concept_type, picker.DESCRIBE_INSTRUCTIONS)

    def test_object_like_prompts_leave_out_people_and_backgrounds(self):
        for key in ("object", "prop", "vehicle", "creature"):
            self.assertIn("background", picker.DESCRIBE_INSTRUCTIONS[key].lower())

    def test_clothing_prompt_excludes_the_person(self):
        self.assertIn("do not describe the person", picker.DESCRIBE_INSTRUCTIONS["clothing"])

    def test_identity_prompt_is_the_agreed_text(self):
        self.assertTrue(picker.DESCRIBE_INSTRUCTIONS["identity"].startswith(
            "These images all show the same character or subject."))


class RefModDescribeSelectionTests(unittest.TestCase):
    def test_identity_keeps_its_first_three_images(self):
        images = list(range(20))
        chosen = picker._pick_images_to_describe(images, "identity", 6)
        self.assertEqual(chosen[:3], [0, 1, 2])
        self.assertEqual(len(chosen), 6)

    def test_other_types_spread_across_the_list(self):
        chosen = picker._pick_images_to_describe(list(range(20)), "style", 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(chosen[0], 0)

    def test_short_lists_are_used_whole(self):
        self.assertEqual(picker._pick_images_to_describe([0, 1], "identity", 6), [0, 1])
        self.assertEqual(picker._pick_images_to_describe([0, 1, 2], "identity", 6), [0, 1, 2])


class RefModUploadTests(unittest.TestCase):
    def test_upload_rejects_non_images(self):
        with self.assertRaises(ValueError):
            picker._save_upload("notes.txt", b"hello")
        with self.assertRaises(ValueError):
            picker._save_upload("broken.png", b"not an image")


if __name__ == "__main__":
    unittest.main()
