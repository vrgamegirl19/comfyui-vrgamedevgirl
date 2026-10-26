"""Bulk scene edits fail loudly, a location save keeps what it is not sent, and a card can be bound to a saved RefMod."""

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
refmod_library = importlib.import_module(f"{pkg_name}.minimax.refmod_library")
refmod_scene = importlib.import_module(f"{pkg_name}.minimax.refmod_scene")

REFMODS = {
    "vehicle/C8_Corvette": {"name": "vehicle/C8_Corvette", "folder": "vehicle", "type": "vehicle", "kind": "video", "frames": 3, "tokens": 1488},
    "identity/Darrel_noclothes": {"name": "identity/Darrel_noclothes", "folder": "identity", "type": "identity", "kind": "video", "frames": 3, "tokens": 2976},
}


class CardAndBulkEditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.allowed_root = os.path.join(self.temp_dir, "output")
        self.folder = os.path.join(self.allowed_root, "Test_Track")
        os.makedirs(self.folder)
        self.session_file = os.path.join(self.folder, "vrgdg_builder_session.json")
        session = {
            "project_name": "Test Track", "project_folder": self.folder, "revision": 1, "video_engine": "minimax_h3",
            "segments": [
                {"id": "seg_0001", "label": "Scene 1", "start": 0.0, "end": 4.0},
                {"id": "seg_0002", "label": "Scene 2", "start": 4.0, "end": 8.0},
                {"id": "seg_0003", "label": "Scene 3", "start": 8.0, "end": 12.0},
            ],
            "flux_reference_builder": {"subjects": [], "locations": []},
        }
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump(session, handle)
        for target in (patch.object(paths, "get_allowed_project_roots", return_value=[self.allowed_root]),
                       patch.object(refmod_library, "find_refmod", side_effect=REFMODS.get),
                       patch.object(refmod_library, "list_refmods", return_value=list(REFMODS.values()))):
            target.start()
            self.addCleanup(target.stop)

    def saved(self):
        with open(self.session_file, encoding="utf-8") as handle:
            return json.load(handle)

    def segment(self, scene_id):
        return next(s for s in self.saved()["segments"] if s["id"] == scene_id)

    # -- bulk edits ----------------------------------------------------------------------------------------------------

    def test_a_bulk_patch_and_move_take_a_scene_number(self):
        result = mutations.bulk_scene_operations("Test_Track", [
            {"op": "patch", "scene_id": "2", "fields": {"timeline_note": "Rain starts"}},
            {"op": "resize", "scene_id": "3", "duration": 2.0},
            {"op": "patch", "scene_id": "seg_0001", "fields": {"notes": "Open on the door"}},
        ])
        self.assertEqual(result["applied_operations"], 3)
        self.assertEqual(self.segment("seg_0002")["timeline_note"], "Rain starts")
        self.assertEqual(self.segment("seg_0001")["notes"], "Open on the door")
        self.assertEqual(self.segment("seg_0003")["end"], 10.0)

    def test_a_bad_operation_fails_the_batch_and_saves_nothing(self):
        before = self.saved()
        bad = [
            ([], errors.ValidationError),
            ("patch", errors.ValidationError),
            ([{"op": "paint", "scene_id": "1"}], errors.ValidationError),
            ([{"scene_id": "1", "fields": {"notes": "x"}}], errors.ValidationError),
            ([{"op": "patch", "fields": {"notes": "x"}}], errors.ValidationError),
            ([{"op": "patch", "scene_id": "1"}], errors.ValidationError),
            ([{"op": "move", "scene_id": "1"}], errors.ValidationError),
            ([{"op": "resize", "scene_id": "1"}], errors.ValidationError),
            (["patch"], errors.ValidationError),
            ([{"op": "patch", "scene_id": "1", "fields": {"notes": "ok"}}, {"op": "patch", "scene_id": "9", "fields": {"notes": "x"}}], errors.SceneNotFoundError),
        ]
        for operations, expected in bad:
            with self.subTest(operations=operations), self.assertRaises(expected):
                mutations.bulk_scene_operations("Test_Track", operations)
        self.assertEqual(self.saved(), before)

    # -- locations -----------------------------------------------------------------------------------------------------

    def test_a_location_save_keeps_the_fields_it_is_not_sent(self):
        mutations.upsert_reference_location("Test_Track", "loc_1", {"name": "Foyer", "description": "Marble floor", "trigger_phrase": "in the foyer"})
        result = mutations.upsert_reference_location("Test_Track", "loc_1", {"image": {"path": "C:/plates/foyer.png"}})
        location = result["location"]
        self.assertEqual((location["name"], location["description"], location["trigger_phrase"]), ("Foyer", "Marble floor", "in the foyer"))
        self.assertEqual(location["image"], {"path": "C:/plates/foyer.png"})
        self.assertEqual(mutations.upsert_reference_location("Test_Track", "loc_2", {})["location"]["name"], "Location")

    # -- RefMod cards --------------------------------------------------------------------------------------------------

    def test_a_character_is_bound_to_a_saved_refmod(self):
        result = mutations.upsert_reference_subject("Test_Track", "darrel_bare", {
            "name": "Darrel Bare", "reference_type": "character",
            "source": "refmod", "refmod": {"name": "identity/Darrel_noclothes", "strength": 0.8},
        })
        card = result["subject"]
        self.assertEqual(card["source"], "refmod")
        self.assertEqual(card["refmod"], {"name": "identity/Darrel_noclothes", "folder": "identity", "type": "identity", "kind": "video",
                                          "tokens": 2976, "frames": 3, "strength": 0.8})
        self.assertTrue(refmod_scene.is_refmod_card(card))
        self.assertEqual(self.saved()["flux_reference_builder"]["subjects"][0]["refmod"]["name"], "identity/Darrel_noclothes")

    def test_a_refmod_may_be_sent_as_its_name_and_defaults_to_full_strength(self):
        card = mutations.upsert_reference_subject("Test_Track", "vette", {"name": "Corvette", "reference_type": "vehicle", "refmod": "vehicle/C8_Corvette"})["subject"]
        self.assertEqual((card["source"], card["refmod"]["name"], card["refmod"]["strength"]), ("refmod", "vehicle/C8_Corvette", 1.0))
        card = mutations.upsert_reference_subject("Test_Track", "vette2", {"name": "Corvette", "source": "refmod", "refmod_name": "vehicle/C8_Corvette"})["subject"]
        self.assertEqual(card["refmod"]["kind"], "video")

    def test_a_save_that_does_not_mention_the_refmod_keeps_it_and_image_takes_it_off(self):
        mutations.upsert_reference_subject("Test_Track", "vette", {"name": "Corvette", "source": "refmod", "refmod": {"name": "vehicle/C8_Corvette", "strength": 0.5}})
        kept = mutations.upsert_reference_subject("Test_Track", "vette", {"description": "Red convertible"})["subject"]
        self.assertEqual((kept["source"], kept["refmod"]["strength"], kept["description"]), ("refmod", 0.5, "Red convertible"))
        stronger = mutations.upsert_reference_subject("Test_Track", "vette", {"source": "refmod", "refmod": {"name": "vehicle/C8_Corvette"}})["subject"]
        self.assertEqual(stronger["refmod"]["strength"], 0.5, "the saved strength stays when the same RefMod is picked again")
        off = mutations.upsert_reference_subject("Test_Track", "vette", {"source": "image"})["subject"]
        self.assertEqual(off["source"], "image")
        self.assertNotIn("refmod", off)
        self.assertFalse(refmod_scene.is_refmod_card(off))

    def test_a_location_is_bound_to_a_saved_refmod(self):
        card = mutations.upsert_reference_location("Test_Track", "garage", {"name": "Garage", "source": "refmod", "refmod": {"name": "vehicle/C8_Corvette"}})["location"]
        self.assertTrue(refmod_scene.is_refmod_card(card))

    def test_a_refmod_that_is_not_saved_is_refused_with_the_names_that_are(self):
        with self.assertRaises(errors.ValidationError) as caught:
            mutations.upsert_reference_subject("Test_Track", "vette", {"source": "refmod", "refmod": {"name": "vehicle/Camaro"}})
        message = str(caught.exception)
        self.assertIn("vehicle/Camaro", message)
        self.assertIn("vehicle/C8_Corvette", message)
        for payload in ({"source": "refmod"}, {"source": "paint"}, {"refmod": {"name": "vehicle/C8_Corvette", "strength": 3}},
                        {"refmod": {"name": "vehicle/C8_Corvette", "strength": "strong"}}):
            with self.subTest(payload=payload), self.assertRaises(errors.ValidationError):
                mutations.upsert_reference_subject("Test_Track", "vette", payload)
        self.assertEqual(self.saved()["flux_reference_builder"]["subjects"], [])


if __name__ == "__main__":
    unittest.main()
