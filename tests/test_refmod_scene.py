import importlib.util
import json
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests" / "refmod_scene_cases.json").read_text(encoding="utf-8"))


def _load():
    package = types.ModuleType("vrgdg_scene_test")
    package.__path__ = [str(ROOT / "minimax")]
    sys.modules["vrgdg_scene_test"] = package
    spec = importlib.util.spec_from_file_location("vrgdg_scene_test.refmod_scene", ROOT / "minimax" / "refmod_scene.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vrgdg_scene_test.refmod_scene"] = module
    spec.loader.exec_module(module)
    return module


scene = _load()


class RefModSceneTests(unittest.TestCase):
    def test_shared_cases(self):
        for case in CASES:
            with self.subTest(case["name"]):
                subjects = scene.scene_subject_cards(case.get("scene") or {}, case["subjects"])
                items = scene.compose_items(subjects, case["extras"], case["location"], case["all_subjects"], case["override"])
                labelled = scene.assign_labels(items, include_audio=case["include_audio"])
                actual = [{"card_id": i["card_id"], "category": i["category"], "label": i["label"], "mod_name": i["mod_name"]} for i in labelled]
                self.assertEqual(actual, case["expected"])
                self.assertEqual(scene.total_tokens(items), case["expected_tokens"])

    def test_token_report_matches_the_browser_rules(self):
        def person(name, tokens, strength=1, category="character"):
            return {"name": name, "tokens": tokens, "strength": strength, "category": category}
        report = scene.token_report([person("The man", 520), person("Darrel", 2394), person("Brandon", 1440)])
        self.assertEqual(report["imbalance"], {"weak": "The man", "strong": "Darrel", "ratio": 4.6})
        self.assertEqual(report["total"], 4354)
        self.assertFalse(report["over_limit"])
        self.assertIsNone(scene.token_report([person("A", 520), person("B", 1000, 0.6)])["imbalance"])
        self.assertIsNone(scene.token_report([person("A", 1000), person("Meadow", 5000, 1, "background")])["imbalance"])
        self.assertTrue(scene.token_report([person("A", 3500), person("B", 3500)])["over_limit"])

    def test_scene_selection_uses_the_standard_scene_maps(self):
        brad = {"id": "s1", "name": "Brad", "reference_type": "character", "source": "refmod",
                "refmod": {"name": "identity/brad", "kind": "video", "tokens": 100, "strength": 1}}
        darrel = dict(brad, id="s2", name="Darrel", refmod={"name": "identity/darrel", "kind": "video", "tokens": 100, "strength": 1})
        meadow = {"id": "l1", "name": "Meadow", "reference_type": "environment", "source": "refmod",
                  "refmod": {"name": "background/meadow", "kind": "image", "tokens": 50, "strength": 1}}
        session = {"flux_reference_builder": {
            "use_subject_reference": True, "subject_count": 2, "subjects": [brad, darrel], "locations": [meadow],
            "subject_scene_map": {"scene-a": "s1,s2", "scene-b": "s2"}, "scene_map": {"scene-a": "l1"},
        }}
        first = scene.refmod_items_for_scene(session, {"id": "scene-a"}, 0)
        self.assertEqual([i["mod_name"] for i in first], ["identity/brad", "identity/darrel", "background/meadow"])
        second = scene.refmod_items_for_scene(session, {"id": "scene-b"}, 1)
        self.assertEqual([i["mod_name"] for i in second], ["identity/darrel"])
        labels = [i["label"] for i in scene.assign_labels(first, include_audio=True)]
        self.assertEqual(labels, ["<Video 1>", "<Video 2>", "<Picture 1>", "<Audio 1>"])

    def test_a_no_character_scene_uses_no_character_refmod(self):
        # Cut and Gun, Scene 4: no_character_present with a RefMod location; Ayame is the project's first character.
        ayame = {"id": "character_a", "name": "Ayame", "reference_type": "character", "source": "refmod",
                 "image": {"path": "C:/refs/ayame.png"},
                 "refmod": {"name": "identity/Ayame_Realistic", "kind": "video", "tokens": 3072, "strength": 1}}
        floors = {"id": "loc_safehouse", "name": "Safehouse floors", "reference_type": "environment", "source": "refmod",
                  "refmod": {"name": "background/safehouse_floors", "kind": "video", "tokens": 1536, "strength": 1}}
        session = {"flux_reference_builder": {
            "use_subject_reference": True, "subject_count": 1, "subjects": [ayame], "locations": [floors],
            "subject_scene_map": {"seg_a": ["character_a"]}, "scene_map": {"seg_5d0f59ee3de8": "loc_safehouse"},
        }}
        segment = {"id": "seg_5d0f59ee3de8", "no_character_present": True}
        items = scene.assign_labels(scene.refmod_items_for_scene(session, segment, 3), include_audio=True)
        self.assertEqual([(i["key"], i["label"]) for i in items],
                         [("location:loc_safehouse", "<Video 1>"), ("audio:scene", "<Audio 1>")])
        # The same project's character scenes still get Ayame first.
        sung = scene.assign_labels(scene.refmod_items_for_scene(session, {"id": "seg_a"}, 0))
        self.assertEqual([i["mod_name"] for i in sung], ["identity/Ayame_Realistic"])

    def test_payload_lists_visual_mods_only(self):
        card = {"id": "s1", "name": "Brad", "reference_type": "character", "source": "refmod",
                "refmod": {"name": "identity/brad", "kind": "video", "tokens": 100, "strength": 0.5}}
        items = scene.assign_labels(scene.compose_items([card], [], None, [card]), include_audio=True)
        payload = scene.reference_payload(items)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["strength"], 0.5)


class RefModLabelsJsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the label tests")
    def test_browser_twin_matches(self):
        result = subprocess.run(["node", "--test", str(Path(__file__).with_name("refmod_labels.mjs"))],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
