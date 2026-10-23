"""Story Mode acts and scene plan: pure rules, the JS twin, and the LLM builders on 20 and 46 scene timelines."""

import importlib
import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

acts_rules = importlib.import_module(f"{ROOT.name}.storyboard.story_acts")
story_plan = importlib.import_module(f"{ROOT.name}.storyboard.story_plan")
prompts = importlib.import_module(f"{ROOT.name}.llm.prompts.story_mode")
CASES = json.loads((ROOT / "tests" / "story_acts_cases.json").read_text(encoding="utf-8"))


class ActRuleTests(unittest.TestCase):
    def test_validate_cases(self):
        for item in CASES["validate"]:
            with self.subTest(item["name"]):
                acts = CASES["acts"] if item["acts"] == "base" else item["acts"]
                self.assertEqual(acts_rules.validate_story_acts(acts, CASES["chapters"], CASES["scene_ids"]), item["problems"])

    def test_edit_cases(self):
        for item in CASES["move"]:
            with self.subTest(item["name"]):
                acts = acts_rules.move_act_border(CASES["acts"], item["act_index"], item["scene_id"], CASES["scene_ids"])
                self.assertEqual([a["scene_ids"] for a in acts], item["scene_ids_after"])
        for item in CASES["merge"]:
            with self.subTest(item["name"]):
                acts = acts_rules.merge_acts(CASES["acts"], item["act_index"])
                self.assertEqual([a["id"] for a in acts], item["ids_after"])
                self.assertEqual([a["scene_ids"] for a in acts], item["scene_ids_after"])
                self.assertEqual([a["cast_ids"] for a in acts], item["cast_after"])
                self.assertEqual([a["summary"] for a in acts], item["summary_after"])
        for item in CASES["split"]:
            with self.subTest(item["name"]):
                acts = acts_rules.split_act(CASES["acts"], item["act_index"], item["scene_id"])
                self.assertEqual([a["id"] for a in acts], item["ids_after"])
                self.assertEqual([a["scene_ids"] for a in acts], item["scene_ids_after"])
                self.assertEqual([a["title"] for a in acts], item["titles_after"])

    def test_fill_cast_case(self):
        story_mode = importlib.import_module(f"{ROOT.name}.storyboard.story_mode")
        item = CASES["fill_cast"]
        mapped, changed = story_mode.fill_scene_cast(item["subject_scene_map"], item["acts"], item["subject_ids"])
        self.assertEqual(mapped, item["map_after"])
        self.assertEqual(changed, item["changed"])

    def test_apply_cast_case(self):
        story_mode = importlib.import_module(f"{ROOT.name}.storyboard.story_mode")
        item = CASES["apply_cast"]
        mapped, changed = story_mode.apply_scene_cast(item["subject_scene_map"], item["plan"], item["acts"], item["subject_ids"])
        self.assertEqual(mapped, item["map_after"])
        self.assertEqual(changed, item["changed"])

    def test_chapter_allocation(self):
        self.assertEqual(acts_rules.allocate_chapter_scenes([3, 1, 2], 20), [9, 4, 7])
        self.assertEqual(acts_rules.allocate_chapter_scenes([1] * 5, 3), [1, 1, 1, 0, 0])
        self.assertEqual(acts_rules.chapter_ranges([9, 0, 3]), [(1, 9), None, (10, 12)])

    def test_act_range_repair_and_notes(self):
        self.assertEqual(acts_rules.repair_act_ranges([(5, 9), (1, 4), (30, 40), ("x",)], 1, 9), [(1, 4), (5, 9)])
        self.assertEqual(acts_rules.repair_act_ranges([], 3, 7), [(3, 7)])
        self.assertEqual(acts_rules.keep_notes_inside_acts([(1, 4), (5, 9)], [(3, 6)]), [(1, 2), (3, 9)])
        self.assertEqual(acts_rules.keep_notes_inside_acts([(1, 4), (5, 9)], [(4, 8)]), [(1, 3), (4, 9)])
        self.assertEqual(acts_rules.fallback_act_ranges(4, 10, 19), [(10, 12), (13, 15), (16, 17), (18, 19)])

    def test_spot_runs(self):
        spots = ["Kitchen", "Kitchen", "Hall", "Kitchen", "Porch", "Porch", "Porch", ""]
        self.assertEqual(acts_rules.enforce_spot_runs(spots, 3), ["Kitchen"] * 4 + ["Porch"] * 4)
        self.assertEqual(acts_rules.enforce_spot_runs(["A", "B", "B", "B"], 3, locked={0}), ["A", "B", "B", "B"])
        self.assertEqual(acts_rules.enforce_spot_runs([], 3), [])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the JS twin")
    def test_js_twin(self):
        result = subprocess.run(["node", "--test", str(Path(__file__).with_name("story_acts_data.mjs"))],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def make_payload(scene_count, *, director_note_scene=None, note_scenes=None):
    scenes = []
    for index in range(scene_count):
        number = index + 1
        scenes.append({
            "id": f"seg_{number}",
            "scene_number": number,
            "timeline_start": index * 5.0,
            "timeline_end": number * 5.0,
            "lyric_section": "Chorus" if number % 8 < 4 else "Verse",
            # The lyrics name places that are not in the story. None of them may become a spot.
            "lyrics": "" if number % 7 == 0 else f"down in the parking garage line {number}",
            "timeline_note": "She runs out to the Porch." if number == director_note_scene else "",
        })
    markers = []
    if note_scenes:
        first, last = note_scenes
        markers.append({"start": (first - 1) * 5.0 + 0.5, "end": last * 5.0 - 0.5, "note": "She finds the letter."})
    book = {"title": "T", "chapters": [
        {"id": f"ch_{n}", "title": f"Chapter {n}", "setting_id": "setting_home" if n % 2 else "setting_train",
         "paragraphs": [f"Paragraph {n}.{p}" for p in range(count)]}
        for n, count in enumerate((3, 2, 3, 2), start=1)
    ]}
    layer = {"story_flow_version": 2, "story_book": book, "lyric_story_strength": 5,
             "story_settings": [{"id": "setting_home", "name": "Home", "look": ""},
                                {"id": "setting_train", "name": "The Night Train", "look": ""}]}
    return {
        "story_layer": layer,
        "scenes": scenes,
        "timeline_markers": markers,
        "reference_builder": {"subjects": [{"id": "mara", "name": "Mara", "description": "a young woman"}]},
    }


class ScriptedLLM:
    """Acts: one act every 4 scenes from each chapter's first scene. Plan: alternating spots, the Porch for a
    scene with a Director Note, and the last scene of each act left out (unless it has a note)."""

    def __init__(self):
        self.calls = {"Story Mode Acts": 0, "Story Mode Scene Plan": 0}
        self.prompts = []

    def __call__(self, payload, instruction, *, label, **kwargs):
        self.calls[label] += 1
        self.prompts.append(instruction)
        numbers = [int(n) for n in re.findall(r"^Scene (\d+) \(", instruction, re.M)]
        if label == "Story Mode Acts":
            acts = [{"first_scene": n, "title": f"Act at {n}", "summary": f"Things happen from {n}.", "cast": ["Mara"]}
                    for n in numbers[::4]]
            return json.dumps({"acts": acts}), {"runner": "fake", "used_model": "m"}
        scenes = []
        last_line = next(line for line in instruction.splitlines() if line.startswith(f"Scene {numbers[-1]} ("))
        for n in numbers if "director note" in last_line else numbers[:-1]:
            note_line = next(line for line in instruction.splitlines() if line.startswith(f"Scene {n} ("))
            spot = "Porch" if "director note" in note_line else ("Kitchen" if n % 2 else "Hall")
            scenes.append({"scene": n, "story_slice": f"Slice {n}.", "spot": spot, "mood": "tense",
                           "performance": "story" if n % 5 == 0 else "singing"})
        return json.dumps({"scenes": scenes}), {"runner": "fake", "used_model": "m"}


class StoryPlanBuilderTests(unittest.TestCase):
    def run_steps(self, payload):
        llm = ScriptedLLM()
        with patch.object(story_plan, "_run", llm):
            acts = story_plan.build_story_acts(payload)["story_layer"]
            payload["story_layer"].update(acts)
            plan = story_plan.build_story_scene_plan(payload)["story_layer"]
        return llm, acts["story_acts"], plan["story_scene_plan"]

    def check_timeline(self, scene_count):
        payload = make_payload(scene_count, director_note_scene=4, note_scenes=(6, 9))
        llm, acts, plan = self.run_steps(payload)
        scene_ids = [s["id"] for s in payload["scenes"]]
        chapters = payload["story_layer"]["story_book"]["chapters"]
        self.assertEqual(acts_rules.validate_story_acts(acts, chapters, scene_ids), [])
        self.assertEqual(llm.calls["Story Mode Acts"], 4, "one acts call per chapter")
        self.assertEqual(llm.calls["Story Mode Scene Plan"], len(acts), "one plan call per act")
        self.assertEqual([entry["scene_id"] for entry in plan], scene_ids, "every scene once, in timeline order")
        number = {sid: i + 1 for i, sid in enumerate(scene_ids)}
        for index, act in enumerate(acts):
            first_in_chapter = index == 0 or acts[index - 1]["chapter_id"] != act["chapter_id"]
            starts_inside_note = number[act["scene_ids"][0]] in (7, 8, 9)
            # Acts never cut a timed note; only a chapter border (set by the story) can fall inside one.
            self.assertFalse(starts_inside_note and not first_in_chapter, f"{act['id']} cuts the timed note (scenes 6-9)")
        for entry, scene in zip(plan, payload["scenes"]):
            self.assertTrue(entry["story_slice"])
            self.assertNotIn("garage", entry["spot"].lower(), "a lyric place became a spot")
            if not scene["lyrics"]:
                self.assertEqual(entry["performance_mode"], "instrumental")
        self.assertEqual(plan[3]["spot"], "Porch", "the Director Note's spot is kept")
        self.assertIn("director", plan[3]["note_sources"])
        self.assertIn("timeline", plan[6]["note_sources"])
        for prompt in llm.prompts:
            self.assertIn(prompts.LYRICS_ARE_MOOD_ONLY, prompt)
        return acts, plan

    def test_twenty_scenes(self):
        acts, plan = self.check_timeline(20)
        self.assertGreaterEqual(len(acts), 4)

    def test_forty_six_scenes_runs_in_small_batches(self):
        acts, plan = self.check_timeline(46)
        self.assertGreater(len(acts), 8)
        runs = acts_rules._spot_runs([entry["spot"] for entry in plan])
        short = [(a, b) for a, b in runs if b - a + 1 < 3 and not any(plan[i]["note_sources"] for i in range(a, b + 1))]
        # A short run can only remain at a chapter edge, where the next chapter's spots start fresh.
        chapter_of = {sid: act["chapter_id"] for act in acts for sid in act["scene_ids"]}
        for a, b in short:
            edges = {chapter_of[plan[a]["scene_id"]], chapter_of[plan[b]["scene_id"]]}
            neighbours = {chapter_of[plan[i]["scene_id"]] for i in (a - 1, b + 1) if 0 <= i < len(plan)}
            self.assertTrue(neighbours - edges or len(plan) < 3, f"short spot run {a + 1}-{b + 1} inside a chapter")

    def test_one_chapter_can_be_rebuilt_alone(self):
        payload = make_payload(20)
        _llm, acts, plan = self.run_steps(payload)
        payload["story_layer"].update({"story_acts": acts, "story_scene_plan": plan})
        llm = ScriptedLLM()
        with patch.object(story_plan, "_run", llm):
            result = story_plan.build_story_acts({**payload, "chapter_id": "ch_2"})["story_layer"]
        self.assertEqual(llm.calls["Story Mode Acts"], 1)
        others = [a for a in acts if a["chapter_id"] != "ch_2"]
        kept = [a for a in result["story_acts"] if a["chapter_id"] != "ch_2"]
        self.assertEqual([a["scene_ids"] for a in kept], [a["scene_ids"] for a in others])
        rebuilt = {sid for a in result["story_acts"] if a["chapter_id"] == "ch_2" for sid in a["scene_ids"]}
        self.assertFalse(any(entry["scene_id"] in rebuilt for entry in result["story_scene_plan"]))

    def test_a_failed_llm_still_gives_every_scene_an_act_and_plan(self):
        payload = make_payload(12)

        def broken(*args, **kwargs):
            raise RuntimeError("LM Studio is not answering")

        with patch.object(story_plan, "_run", broken):
            acts = story_plan.build_story_acts(payload)["story_layer"]["story_acts"]
            payload["story_layer"]["story_acts"] = acts
            plan = story_plan.build_story_scene_plan(payload)["story_layer"]["story_scene_plan"]
        self.assertEqual(sum(len(a["scene_ids"]) for a in acts), 12)
        self.assertEqual(len(plan), 12)
        self.assertTrue(all(entry["spot"] for entry in plan), "spots fall back to the setting name")

    def test_a_scene_with_no_spot_takes_a_loaded_place_not_the_act_title(self):
        payload = make_payload(12)
        payload["story_layer"]["story_settings"] = []
        for chapter in payload["story_layer"]["story_book"]["chapters"]:
            chapter["setting_id"] = ""
            chapter["title"] = f"Act {chapter['id'][3:]}"
        payload["reference_builder"]["locations"] = [{"id": "l1", "name": "Hallway", "description": "A narrow hallway."}]

        def broken(*args, **kwargs):
            raise RuntimeError("LM Studio is not answering")

        with patch.object(story_plan, "_run", broken):
            acts = story_plan.build_story_acts(payload)["story_layer"]["story_acts"]
            payload["story_layer"]["story_acts"] = acts
            plan = story_plan.build_story_scene_plan(payload)["story_layer"]["story_scene_plan"]
        self.assertEqual({entry["spot"] for entry in plan}, {"Hallway"})

    def test_scene_cast_performer_and_interaction_with_user_changes_kept(self):
        payload = make_payload(8)
        payload["reference_builder"]["subjects"].append({"id": "joe", "name": "Joe", "description": "a conductor"})
        payload["story_layer"]["story_main_cast_id"] = "mara"
        payload["story_layer"]["story_relationships"] = [{"a": "mara", "b": "joe", "type": "love_interest", "note": ""}]
        _llm, acts, _plan = self.run_steps(payload)
        payload["story_layer"]["story_acts"] = acts
        scene_ids = [s["id"] for s in payload["scenes"]]
        # The user put Joe alone in scene 2 and wrote what he does.
        payload["story_layer"]["story_scene_plan"] = [{"scene_id": scene_ids[1], "act_id": acts[0]["id"],
                                                       "cast_ids": ["joe"], "cast_source": "manual",
                                                       "user_note": "Joe arrives with flowers."}]

        def scripted(payload_, instruction, *, label, **kwargs):
            numbers = [int(n) for n in re.findall(r"^Scene (\d+) \(", instruction, re.M)]
            scripted.prompts.append(instruction)
            scenes = [{"scene": n, "story_slice": f"Slice {n}.", "spot": "Kitchen", "mood": "warm", "performance": "singing",
                       "continues": False, "continuation": "", "cast": ["Mara", "Joe", "Stranger"], "performer": "Mara",
                       "interaction": "Mara reaches for Joe's hand."} for n in numbers]
            return json.dumps({"scenes": scenes}), {"runner": "fake", "used_model": "m"}

        scripted.prompts = []
        with patch.object(story_plan, "_run", scripted):
            plan = story_plan.build_story_scene_plan(payload)["story_layer"]["story_scene_plan"]
        by_id = {entry["scene_id"]: entry for entry in plan}
        self.assertEqual(by_id[scene_ids[0]]["cast_ids"], ["mara", "joe"], "names outside the cast are dropped")
        self.assertEqual(by_id[scene_ids[0]]["performer_id"], "mara")
        self.assertEqual(by_id[scene_ids[0]]["interaction"], "Mara reaches for Joe's hand.")
        self.assertEqual(by_id[scene_ids[0]]["cast_source"], "story")
        self.assertEqual((by_id[scene_ids[1]]["cast_ids"], by_id[scene_ids[1]]["cast_source"]), (["joe"], "manual"))
        self.assertEqual(by_id[scene_ids[1]]["performer_id"], "", "the performer must be in the scene")
        self.assertEqual(by_id[scene_ids[1]]["interaction"], "", "one character has no interaction")
        self.assertEqual(by_id[scene_ids[1]]["user_note"], "Joe arrives with flowers.")
        first_prompt = scripted.prompts[0]
        self.assertIn("user direction, characters in this scene: Joe | user direction: Joe arrives with flowers.", first_prompt)
        self.assertIn("Mara is the main character", first_prompt)
        self.assertIn("Relationships:\n- Mara and Joe: love interests", first_prompt)

    def test_plan_needs_valid_acts(self):
        payload = make_payload(6)
        payload["story_layer"]["story_acts"] = [{"id": "act_1", "chapter_id": "ch_1", "scene_ids": ["seg_1"]}]
        with self.assertRaises(ValueError):
            story_plan.build_story_scene_plan(payload)


if __name__ == "__main__":
    unittest.main()
