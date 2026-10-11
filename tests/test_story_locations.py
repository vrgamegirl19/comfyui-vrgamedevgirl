"""Story Mode locations: setting looks and spot cards from the story (never the lyrics), merged and mapped."""

import copy
import importlib
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

locations = importlib.import_module(f"{ROOT.name}.storyboard.story_locations")
story = importlib.import_module(f"{ROOT.name}.agent_api.orchestrator.storyboard_orchestrator")
story_mode = importlib.import_module(f"{ROOT.name}.agent_api.orchestrator.story_mode_orchestrator")
CASES = json.loads((ROOT / "tests" / "story_locations_cases.json").read_text(encoding="utf-8"))
from test_agent_api_references_llm import Base  # noqa: E402


def layer_with_plan(scene_ids):
    half = len(scene_ids) // 2
    spots = ["Kitchen", "Kitchen", "Bedroom", "Bedroom", "Kitchen"]
    plan = [{"scene_id": sid, "act_id": "act_1" if i < half else "act_2", "story_slice": f"Mara does thing {i + 1}.",
             "spot": spots[i % len(spots)] if i < half else "Sleeper Car", "mood": "", "performance_mode": "singing",
             "note_sources": []} for i, sid in enumerate(scene_ids)]
    return {
        "story_flow_version": 2,
        "story_book": {"title": "T", "chapters": [
            {"id": "ch_1", "title": "Home", "setting_id": "setting_home", "paragraphs": ["Mara packs."]},
            {"id": "ch_2", "title": "Away", "setting_id": "setting_the_night_train", "paragraphs": ["Mara rides."]},
        ]},
        "story_settings": [{"id": "setting_home", "name": "Home", "look": ""},
                           {"id": "setting_the_night_train", "name": "The Night Train", "look": "Brass and green velvet."}],
        "story_acts": [{"id": "act_1", "chapter_id": "ch_1", "scene_ids": scene_ids[:half]},
                       {"id": "act_2", "chapter_id": "ch_2", "scene_ids": scene_ids[half:]}],
        "story_scene_plan": plan,
    }


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def __call__(self, payload, instruction, *, label, **kwargs):
        self.prompts.append(instruction)
        if label == "Story Mode Setting Looks":
            return json.dumps({"settings": [{"name": "Home", "look": "A 1970s farmhouse, pine and warm lamplight."}]}), {}
        spots = [line[2:].split(":")[0] for line in instruction.splitlines() if line.startswith("- ") and ":" in line]
        return json.dumps({"spots": [{"name": name, "description": f"{name} described."} for name in spots]}), {}


class StoryLocationTests(unittest.TestCase):
    def test_card_names(self):
        for item in CASES["card_names"]:
            self.assertEqual(locations.card_name(item["setting"], item["spot"]), item["name"])

    def test_apply_cases(self):
        for item in CASES["apply"]:
            with self.subTest(item["name"]):
                count = [0]

                def new_id(taken):
                    count[0] += 1
                    taken.add(f"loc_new_{count[0]}")
                    return f"loc_new_{count[0]}"

                before = copy.deepcopy(item["refs"])
                refs, counts = locations.apply_story_locations(item["refs"], item["locations"], new_id,
                                                               item["replace_descriptions"])
                self.assertEqual(refs, item["refs_after"])
                self.assertEqual(counts, {"added": item["added"], "updated": item["updated"]})
                self.assertEqual(item["refs"], before, "the input is not changed")

    def test_spots_are_grouped_by_setting(self):
        ids = [f"s{i}" for i in range(1, 11)]
        groups = locations.spot_groups(layer_with_plan(ids), ids)
        self.assertEqual([g["setting_name"] for g in groups], ["Home", "The Night Train"])
        self.assertEqual([s["card_name"] for s in groups[0]["spots"]], ["Home - Kitchen", "Home - Bedroom"])
        self.assertEqual(groups[0]["spots"][0]["scene_ids"], ["s1", "s2", "s5"])

    def test_locations_come_from_the_story_never_the_lyrics(self):
        ids = [f"s{i}" for i in range(1, 11)]
        payload = {
            "story_layer": layer_with_plan(ids),
            "scenes": [{"id": sid, "lyrics": "meet me in the abandoned parking garage"} for sid in ids],
            "lyrics": "meet me in the abandoned parking garage",
            "style_theme": "No warehouses.",
        }
        llm = FakeLLM()
        with patch.object(locations, "_run", llm):
            result = locations.build_story_locations(payload)
        self.assertEqual(len(llm.prompts), 3, "one looks call (Home only) and one spots call per setting")
        for prompt in llm.prompts:
            self.assertNotIn("parking garage", prompt)
        self.assertIn("No warehouses.", llm.prompts[1])
        self.assertIn("A 1970s farmhouse", llm.prompts[1], "spots are built on the setting look")
        names = [item["name"] for item in result["locations"]]
        self.assertEqual(names, ["Home - Kitchen", "Home - Bedroom", "The Night Train - Sleeper Car"])
        self.assertEqual(result["story_layer"]["story_settings"][0]["look"], "A 1970s farmhouse, pine and warm lamplight.")
        self.assertEqual(result["story_layer"]["story_settings"][1]["look"], "Brass and green velvet.", "a user look is kept")

    def test_needs_a_plan(self):
        with self.assertRaises(ValueError):
            locations.build_story_locations({"story_layer": {}, "scenes": [{"id": "s1"}]})

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the JS twin")
    def test_js_twin(self):
        result = subprocess.run(["node", "--test", str(Path(__file__).with_name("story_locations_data.mjs"))],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class StoryLocationApiTests(Base):
    def test_cards_are_saved_and_every_scene_is_mapped(self):
        session = self.read_session()
        ids = [s["id"] for s in session["segments"]]
        session["builder_story_layer"] = layer_with_plan(ids)
        self.write_session(session)
        with patch.object(locations, "_run", FakeLLM()):
            result = story_mode.create_story_locations("Song", {"style_theme": "No warehouses."})
        self.assertEqual(result["added"], 3)
        refs = self.read_session()["flux_reference_builder"]
        cards = {card["name"]: card for card in refs["locations"]}
        self.assertEqual(cards["Home - Kitchen"]["story_setting_id"], "setting_home")
        self.assertTrue(cards["Home - Kitchen"]["id"].startswith("loc_"))
        self.assertEqual({sid for sid in ids}, set(refs["scene_map"]))
        self.assertEqual(refs["scene_map"][ids[-1]], cards["The Night Train - Sleeper Car"]["id"])
        self.assertTrue(refs["use_location_references"])
        cards_for_scene_1 = story.scene_cards(self.read_session())[0]
        self.assertEqual(cards_for_scene_1["location_ref"]["name"], "Home - Kitchen")


if __name__ == "__main__":
    unittest.main()
