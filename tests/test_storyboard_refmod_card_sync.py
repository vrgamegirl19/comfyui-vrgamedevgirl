"""The headless Storyboard cards keep a RefMod card's RefMod, and a live Reference Builder card wins over the saved copy.

Twins of ``refmodCardFields`` (scene_output.mjs) and ``liveReference`` (persistence.mjs).
"""
import copy
import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_refmod_card_sync_test"
for name, folder in [(PACKAGE, ROOT), (PACKAGE + ".storyboard", ROOT / "storyboard"), (PACKAGE + ".builder", ROOT / "builder"),
                     (PACKAGE + ".minimax", ROOT / "minimax"), (PACKAGE + ".core", ROOT / "core")]:
    module = types.ModuleType(name)
    module.__path__ = [str(folder)]
    sys.modules.setdefault(name, module)
sync = importlib.import_module(PACKAGE + ".storyboard.session_sync")

REFMOD = {"name": "background/City Street at Dawn", "folder": "background", "type": "background", "kind": "image", "tokens": 100,
          "frames": 1, "strength": 0.8}


def session(location_picture="locations/City_06.png", refmod=True, subject_refmod=True):
    location = {"id": "city", "name": "City Street", "description": "a street at dawn", "image": {"path": location_picture}}
    if refmod:
        location.update({"source": "refmod", "refmod": dict(REFMOD)})
    anna = {"id": "anna", "name": "Anna", "description": "a singer", "image": {"path": ""}}
    if subject_refmod:
        anna.update({"source": "refmod", "refmod": {"name": "identity/anna", "kind": "video", "strength": 1}})
    return {
        "video_engine": "minimax_h3",
        "flux_reference_builder": {"subjects": [anna], "locations": [location], "scene_map": {"1": "city"},
                                   "subject_scene_map": {"1": ["anna"]}},
        "segments": [{"id": "1", "start": 0.0, "end": 4.0, "label": "Scene 1"}],
    }


class SlimReferenceTests(unittest.TestCase):
    def test_a_refmod_card_keeps_its_refmod(self):
        card = sync.scene_cards(session())[0]
        location = card["location_ref"]
        self.assertEqual(location["source"], "refmod")
        self.assertEqual(location["refmod"]["name"], "background/City Street at Dawn")
        self.assertEqual(location["refmod"]["strength"], 0.8)
        subject = card["subject_refs"][0]
        self.assertEqual((subject["source"], subject["refmod"]["name"]), ("refmod", "identity/anna"))

    def test_a_picture_card_has_no_refmod_fields(self):
        card = sync.scene_cards(session(refmod=False, subject_refmod=False))[0]
        for reference in (card["location_ref"], card["subject_refs"][0]):
            self.assertFalse({"source", "refmod"} & set(reference), reference)


class LiveCardWinsTests(unittest.TestCase):
    def test_the_live_picture_and_refmod_replace_the_saved_copy_of_a_location(self):
        fresh = sync.scene_cards(session("locations/City_06.png"))[0]
        saved = copy.deepcopy(fresh)
        saved["location_ref"] = {"id": "city", "name": "City Street", "description": "old text", "image": {"path": "locations/City_02.png"}}
        merged = sync.merge_scene_cards({"scenes": [saved]}, [fresh])[0]["location_ref"]
        self.assertEqual(merged["image"]["path"], "locations/City_06.png")
        self.assertEqual(merged["source"], "refmod")
        self.assertEqual(merged["description"], "a street at dawn")

    def test_the_saved_copy_fills_in_what_the_live_card_does_not_have(self):
        fresh = sync.scene_cards(session())[0]
        saved = copy.deepcopy(fresh)
        saved["location_ref"]["note_for_storyboard"] = "kept"
        merged = sync.merge_scene_cards({"scenes": [saved]}, [fresh])[0]["location_ref"]
        self.assertEqual(merged["note_for_storyboard"], "kept")

    def test_a_live_card_without_a_picture_keeps_the_saved_one(self):
        fresh = sync.scene_cards(session())[0]
        saved = copy.deepcopy(fresh)
        saved["subject_refs"][0]["image"] = {"path": "subjects/anna.png", "name": "anna.png", "data": ""}
        merged = sync.merge_scene_cards({"scenes": [saved]}, [fresh])[0]["subject_refs"][0]
        self.assertEqual(merged["image"]["path"], "subjects/anna.png")
        self.assertEqual(merged["source"], "refmod")

    def test_a_different_location_takes_the_live_card(self):
        fresh = sync.scene_cards(session())[0]
        saved = copy.deepcopy(fresh)
        saved["location_ref"] = {"id": "beach", "name": "Beach", "description": "sand", "image": {"path": "locations/beach.png"}}
        merged = sync.merge_scene_cards({"scenes": [saved]}, [fresh])[0]
        self.assertEqual(merged["location_ref"], fresh["location_ref"])
        self.assertEqual(merged["setting"], fresh["setting"])

    def test_a_scene_without_a_saved_location_takes_the_live_card(self):
        fresh = sync.scene_cards(session())[0]
        saved = copy.deepcopy(fresh)
        saved["location_ref"] = None
        merged = sync.merge_scene_cards({"scenes": [saved]}, [fresh])[0]
        self.assertEqual(merged["location_ref"], fresh["location_ref"])


if __name__ == "__main__":
    unittest.main()
