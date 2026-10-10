"""Tests for the Agent API twin of the Video Builder's "Choose MiniMax References" picker."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
scene_inputs = importlib.import_module(f"{pkg_name}.minimax.scene_inputs")


def errors_value_error():
    return ValueError


def _image(name):
    return {"path": f"C:/refs/{name}.png", "name": f"{name}.png"}


class ReferenceChoicesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.allowed_root = os.path.join(self.temp_dir, "output")
        self.folder = os.path.join(self.allowed_root, "Refs")
        os.makedirs(self.folder, exist_ok=True)
        self.start_image = os.path.join(self.folder, "image_0001.png")
        Path(self.start_image).write_bytes(b"PNG")
        self.session = {
            "project_name": "Refs",
            "project_folder": self.folder,
            "revision": 1,
            "video_engine": "minimax_h3",
            "minimax_h3_settings": {"video_mode": "reference_to_video"},
            "segments": [
                {"id": "seg_1", "label": "Scene 1", "start": 0.0, "end": 4.0, "approved_image_path": self.start_image},
                {"id": "seg_2", "label": "Scene 2", "start": 4.0, "end": 8.0},
            ],
            "flux_reference_builder": {
                "use_subject_reference": True,
                "subject_count": 2,
                "subjects": [
                    {"id": "ava", "name": "Ava", "description": "lead", "image": _image("ava")},
                    {"id": "ben", "name": "Ben", "description": "drummer", "image": _image("ben")},
                    {"id": "no_image", "name": "Ghost", "description": "x", "image": {}},
                ],
                "locations": [
                    {"id": "roof", "name": "Rooftop", "description": "city roof", "image": _image("roof")},
                    {"id": "alley", "name": "Alley", "description": "wet alley", "image": _image("alley")},
                    {"id": "club", "name": "Club", "description": "neon club", "image": _image("club")},
                ],
                "extras_enabled": True,
                "extra_subjects": [
                    {"id": "crowd", "title": "Crowd", "description": "dancers", "send_to_minimax": True, "image": _image("crowd")},
                    {"id": "unused", "title": "Unused", "description": "nobody", "send_to_minimax": True, "image": _image("unused")},
                ],
                "extra_scene_map": {"seg_1": [{"extra_id": "crowd", "interaction": "background"}]},
                "ingredients_sheets": [{"id": "props", "name": "Props", "image": _image("props")}],
                "subject_scene_map": {"seg_1": ["ava"], "seg_2": ["ben"]},
                # one location per scene: this is all the scene mapping can hold
                "scene_map": {"seg_1": "roof", "seg_2": "alley"},
                "ingredients_scene_map": {},
            },
        }
        Path(self.folder, "vrgdg_builder_session.json").write_text(json.dumps(self.session, indent=2), encoding="utf-8")
        patcher = patch.object(paths, "get_allowed_project_roots", return_value=[self.allowed_root])
        patcher.start()
        self.addCleanup(patcher.stop)

    def _saved(self):
        return json.loads(Path(self.folder, "vrgdg_builder_session.json").read_text(encoding="utf-8"))

    def _keys(self, items):
        return [item["key"] for item in items]

    # ---- reading
    def test_every_reference_with_an_image_is_offered_not_just_the_mapped_location(self):
        choices = mutations.get_scene_minimax_references("Refs", "seg_1")
        keys = self._keys(choices["available"])
        for key in ("subject:ava", "subject:ben", "location:roof", "location:alley", "location:club", "ingredients:props", "extra:crowd"):
            self.assertIn(key, keys)
        self.assertNotIn("subject:no_image", keys)  # no image, so it cannot be sent
        self.assertNotIn("extra:unused", keys)  # an extra only appears for the scenes it is mapped to
        by_key = {item["key"]: item for item in choices["available"]}
        self.assertTrue(by_key["location:roof"]["in_scene_mapping"])
        self.assertFalse(by_key["location:alley"]["in_scene_mapping"])
        self.assertEqual(by_key["location:club"]["image_path"], "C:/refs/club.png")

    def test_the_selected_order_and_image_numbers_follow_the_scene_mapping_until_chosen_by_hand(self):
        choices = mutations.get_scene_minimax_references("Refs", "seg_1")
        self.assertFalse(choices["custom"])
        self.assertEqual(self._keys(choices["selected"]), ["subject:ava", "extra:crowd", "location:roof"])
        # Image 1 is the scene's start frame only when that option is on, so numbering starts at 1 here
        self.assertEqual([item["image_number"] for item in choices["selected"]], [1, 2, 3])
        self.assertEqual(choices["automatic_keys"], ["subject:ava", "extra:crowd", "location:roof"])
        self.assertEqual(choices["limits"], {"max_images": 9, "start_frame_image_1": False, "max_choices": 9})
        self.assertEqual((choices["scene_id"], choices["scene_number"], choices["video_mode"]), ("seg_1", 1, "reference_to_video"))

    def test_scenes_can_be_addressed_by_number(self):
        self.assertEqual(mutations.get_scene_minimax_references("Refs", "2")["scene_id"], "seg_2")
        with self.assertRaises(errors.SceneNotFoundError):
            mutations.get_scene_minimax_references("Refs", "seg_99")

    def test_reference_mode_ignores_legacy_start_flag_in_picker_numbering(self):
        session = self._saved()
        session["segments"][0]["minimax_h3_use_scene_image_as_start_frame"] = True
        Path(self.folder, "vrgdg_builder_session.json").write_text(json.dumps(session), encoding="utf-8")
        choices = mutations.get_scene_minimax_references("Refs", "seg_1")
        self.assertEqual(choices["limits"], {"max_images": 9, "start_frame_image_1": False, "max_choices": 9})
        self.assertEqual([item["image_number"] for item in choices["selected"]], [1, 2, 3])

    # ---- writing
    def test_several_locations_can_be_chosen_in_a_custom_order(self):
        keys = ["location:alley", "subject:ava", "location:roof", "location:club"]
        result = mutations.set_scene_minimax_references("Refs", "seg_1", keys=keys)
        self.assertTrue(result["custom"])
        # the mapped extra for this scene is always sent, after the chosen ones
        self.assertEqual(self._keys(result["selected"]), [*keys, "extra:crowd"])
        self.assertEqual(result["revision"], 2)
        self.assertEqual(self._saved()["segments"][0]["minimax_h3_reference_keys"], keys)
        # the render reads the same list: these are the image paths MiniMax gets, in this order
        paths_sent = scene_inputs.render_reference_image_paths(self._saved(), self._saved()["segments"][0], "reference_to_video", 0)
        self.assertEqual(paths_sent[:4], ["C:/refs/alley.png", "C:/refs/ava.png", "C:/refs/roof.png", "C:/refs/club.png"])

    def test_automatic_hands_the_scene_back_to_its_mappings(self):
        mutations.set_scene_minimax_references("Refs", "seg_1", keys=["location:club"])
        result = mutations.set_scene_minimax_references("Refs", "seg_1", automatic=True)
        self.assertFalse(result["custom"])
        self.assertIsNone(self._saved()["segments"][0]["minimax_h3_reference_keys"])
        self.assertEqual(self._keys(result["selected"]), ["subject:ava", "extra:crowd", "location:roof"])

    def test_an_empty_list_chooses_nothing_by_hand(self):
        result = mutations.set_scene_minimax_references("Refs", "seg_1", keys=[])
        self.assertTrue(result["custom"])
        self.assertEqual(self._keys(result["selected"]), ["extra:crowd"])  # only the always-sent mapped extra

    def test_bad_requests_are_refused_with_the_allowed_keys(self):
        with self.assertRaisesRegex(errors.ValidationError, "Not available for this scene: location:nowhere.*Available keys: .*location:roof"):
            mutations.set_scene_minimax_references("Refs", "seg_1", keys=["location:nowhere"])
        with self.assertRaisesRegex(errors.ValidationError, "extra:unused"):
            mutations.set_scene_minimax_references("Refs", "seg_1", keys=["extra:unused"])
        with self.assertRaisesRegex(errors.ValidationError, "Repeated: location:roof"):
            mutations.set_scene_minimax_references("Refs", "seg_1", keys=["location:roof", "location:roof"])
        with self.assertRaisesRegex(errors.ValidationError, "must be a list"):
            mutations.set_scene_minimax_references("Refs", "seg_1", keys="location:roof")
        with self.assertRaisesRegex(errors.ValidationError, "not both"):
            mutations.set_scene_minimax_references("Refs", "seg_1", keys=["location:roof"], automatic=True)
        with self.assertRaisesRegex(errors.ValidationError, "or `automatic: true`"):
            mutations.set_scene_minimax_references("Refs", "seg_1")
        self.assertNotIn("minimax_h3_reference_keys", self._saved()["segments"][0])  # nothing was saved

    def test_too_many_choices_are_refused(self):
        available = [{"key": f"location:l{n}"} for n in range(10)]
        keys = [item["key"] for item in available]
        with self.assertRaisesRegex(errors_value_error(), "at most 8 chosen references.*start frame.*9 were given"):
            scene_inputs.validate_reference_keys({"available": available, "limits": {"max_choices": 8, "start_frame_image_1": True}}, keys[:9])
        self.assertEqual(
            scene_inputs.validate_reference_keys({"available": available, "limits": {"max_choices": 9, "start_frame_image_1": False}}, keys[:9]),
            keys[:9],
        )

    def test_a_stale_revision_is_a_conflict(self):
        with self.assertRaises(errors.RevisionConflictError):
            mutations.set_scene_minimax_references("Refs", "seg_1", keys=["location:roof"], if_match_revision=99)


if __name__ == "__main__":
    unittest.main()
