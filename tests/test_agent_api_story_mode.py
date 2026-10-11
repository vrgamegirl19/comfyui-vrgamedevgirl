"""Story Mode story steps for agents: book, chapter rewrite, pasted story split and cast scan."""

import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
story = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.storyboard_orchestrator")
story_mode = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.story_mode_orchestrator")
book_funcs = importlib.import_module(f"{pkg_name}.storyboard.story_book")
from test_agent_api_references_llm import Base  # noqa: E402

BOOK = """Title: The Last Train North
Premise: Darrel follows his father's trail.

Chapter 1: The Station
Setting: The Night Train

Darrel waits on the empty platform.

The conductor waves him aboard.

Chapter 2: Coming Home
Setting: Home

Darrel opens the kitchen door.
"""


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.instructions = []

    def __call__(self, payload, instruction, **kwargs):
        # The cast check after a book or chapter finds no one outside the cast.
        if kwargs.get("label") == "Story Mode Cast Check":
            return '{"characters": []}', {}
        self.instructions.append(instruction)
        return self.replies.pop(0), {"runner": "lm_studio", "used_model": payload.get("lmstudio_model", "")}


class StoryModeStepTests(Base):
    def layer(self):
        return self.read_session()["builder_story_layer"]

    def test_book_is_drafted_from_the_premise_and_saved(self):
        session = self.read_session()
        session["builder_story_layer"] = {"user_story_arc": "Verse 1: old arc", "lyric_story_strength": 7}
        self.write_session(session)
        llm = FakeLLM(BOOK)
        with patch.object(book_funcs, "_run", llm):
            result = story_mode.create_story_book("Song", {"story_idea": "A son searches for his father"})
        prompt = llm.instructions[0]
        self.assertIn("A son searches for his father", prompt)
        self.assertIn("Darrel", prompt, "the cast is sent")
        self.assertIn("10 scenes", prompt)
        self.assertIn("emotion only", prompt)
        layer = self.layer()
        self.assertEqual(layer["story_flow_version"], 2)
        self.assertEqual(layer["lyric_story_strength"], 5, "Story Mode starts at the emotion-only default")
        self.assertEqual(layer["user_story_arc"], "Verse 1: old arc", "older story fields are kept")
        self.assertEqual([c["setting_id"] for c in layer["story_book"]["chapters"]],
                         ["setting_the_night_train", "setting_home"])
        self.assertEqual([s["name"] for s in layer["story_settings"]], ["The Night Train", "Home"])
        self.assertEqual(layer["story_acts"], [])
        self.assertTrue(layer["story_book_text"].startswith("Title: The Last Train North"))
        self.assertEqual(result["story_book"]["title"], "The Last Train North")

    def test_a_reply_without_a_story_is_an_error(self):
        with patch.object(book_funcs, "_run", FakeLLM("")):
            with self.assertRaises(ValueError):
                story_mode.create_story_book("Song", {"story_idea": "idea"})

    def test_pasted_story_keeps_every_word(self):
        text = "Darrel waits on the platform.\n\nThe train comes.\n\nHe walks home.\n\nThe kitchen is cold."
        split = {"chapters": [{"start_paragraph": 1, "title": "The Station", "setting": "The Night Train"},
                              {"start_paragraph": 3, "title": "Home Again", "setting": "Home"}]}
        with patch.object(book_funcs, "_run", FakeLLM(json.dumps(split))):
            story_mode.split_story("Song", {"text": text})
        chapters = self.layer()["story_book"]["chapters"]
        self.assertEqual([c["paragraphs"] for c in chapters],
                         [["Darrel waits on the platform.", "The train comes."], ["He walks home.", "The kitchen is cold."]])
        self.assertEqual(chapters[1]["setting_id"], "setting_home")

    def test_pasted_story_with_headings_keeps_them(self):
        split = {"chapters": [{"start_paragraph": 1, "title": "Other", "setting": "Station"},
                              {"start_paragraph": 3, "title": "Other", "setting": "House"}]}
        with patch.object(book_funcs, "_run", FakeLLM(json.dumps(split))):
            story_mode.split_story("Song", {"text": BOOK})
        chapters = self.layer()["story_book"]["chapters"]
        self.assertEqual([c["title"] for c in chapters], ["The Station", "Coming Home"])
        self.assertEqual([c["setting_id"] for c in chapters], ["setting_the_night_train", "setting_home"],
                         "settings the user wrote win over the LLM's")

    def test_chapter_rewrite_replaces_one_chapter_and_clears_its_acts(self):
        session = self.read_session()
        session["builder_story_layer"] = {
            "story_flow_version": 2,
            "story_book": {"title": "T", "chapters": [
                {"id": "ch_1", "title": "One", "setting_id": "setting_home", "paragraphs": ["old"]},
                {"id": "ch_2", "title": "Two", "setting_id": "setting_home", "paragraphs": ["keep"]},
            ]},
            "story_settings": [{"id": "setting_home", "name": "Home", "look": "warm wood"}],
            "story_acts": [{"id": "act_1", "chapter_id": "ch_1"}, {"id": "act_2", "chapter_id": "ch_2"}],
            "story_scene_plan": [{"scene_id": "s1", "act_id": "act_1"}, {"scene_id": "s2", "act_id": "act_2"}],
        }
        self.write_session(session)
        llm = FakeLLM("Chapter 1: One Again\n\nA new first chapter.")
        with patch.object(book_funcs, "_run", llm):
            story_mode.rewrite_story_chapter("Song", {"chapter_id": "ch_1", "direction": "make it rain"})
        self.assertIn("make it rain", llm.instructions[0])
        layer = self.layer()
        chapters = layer["story_book"]["chapters"]
        self.assertEqual(chapters[0]["paragraphs"], ["A new first chapter."])
        self.assertEqual(chapters[0]["setting_id"], "setting_home", "a rewrite without a Setting line keeps the setting")
        self.assertEqual(chapters[1]["paragraphs"], ["keep"])
        self.assertEqual([a["id"] for a in layer["story_acts"]], ["act_2"])
        self.assertEqual([p["scene_id"] for p in layer["story_scene_plan"]], ["s2"])
        self.assertEqual(layer["story_settings"][0]["look"], "warm wood")

    def test_chapter_rewrite_needs_a_known_chapter(self):
        with self.assertRaises(errors.ValidationError):
            story_mode.rewrite_story_chapter("Song", {})
        with self.assertRaises(errors.ValidationError):
            story_mode.rewrite_story_chapter("Song", {"chapter_id": "ch_9"})

    def test_cast_scan_lists_only_people_without_a_card(self):
        session = self.read_session()
        session["builder_story_layer"] = {"story_flow_version": 2, "story_book_text": BOOK,
                                          **story_mode.story_mode_text_fields({}, {"story_book_text": BOOK})}
        self.write_session(session)
        found = {"characters": [
            {"name": "Darrel", "role": "main", "description": "already cast"},
            {"name": "The Conductor", "role": "supporting", "description": "Older man in a navy uniform."},
            {"name": "the conductor", "role": "supporting", "description": "duplicate"},
        ]}
        with patch.object(book_funcs, "_run", FakeLLM(json.dumps(found))):
            result = story_mode.scan_story_cast("Song", {})
        self.assertEqual(result["characters"], [
            {"name": "The Conductor", "role": "supporting", "description": "Older man in a navy uniform."},
        ])

    def test_book_text_can_be_edited_through_story_settings(self):
        story.set_story_settings("Song", {"story": {"story_book_text": BOOK}})
        layer = self.layer()
        self.assertEqual(layer["story_flow_version"], 2)
        self.assertEqual(len(layer["story_book"]["chapters"]), 2)
        with self.assertRaises(errors.ValidationError):
            story.set_story_settings("Song", {"story": {"story_acts": []}})

    def write_book(self):
        story.set_story_settings("Song", {"story": {"story_book_text": BOOK}})

    def test_acts_and_plan_are_saved_and_the_plan_sets_each_scene_performance(self):
        self.write_book()
        plan_funcs = importlib.import_module(f"{pkg_name}.storyboard.story_plan")

        def fake(payload, instruction, *, label, **kwargs):
            import re
            numbers = [int(n) for n in re.findall(r"^Scene (\d+) \(", instruction, re.M)]
            if label == "Story Mode Acts":
                return json.dumps({"acts": [{"first_scene": numbers[0], "title": "A", "summary": "S", "cast": ["Darrel"]}]}), {}
            performance = {1: "story", 2: "speaking"}
            return json.dumps({"scenes": [
                {"scene": n, "story_slice": f"slice {n}", "spot": "Platform", "mood": "calm",
                 "performance": performance.get(n, "singing")} for n in numbers
            ]}), {}

        with patch.object(plan_funcs, "_run", fake):
            acts = story_mode.create_story_acts("Song", {})
            plan = story_mode.create_story_scene_plan("Song", {})
        self.assertEqual(sum(len(a["scene_ids"]) for a in acts["story_acts"]), 10)
        self.assertEqual(acts["story_acts"][0]["cast_ids"], ["darrel"])
        self.assertEqual(len(plan["story_scene_plan"]), 10)
        segments = self.read_session()["segments"]
        self.assertEqual((segments[0]["performance_mode"], segments[0]["lyric_no_lip_sync"]), ("no_lip_sync", True))
        self.assertEqual((segments[1]["performance_mode"], segments[1]["lyric_no_lip_sync"]), ("speaking", False))
        self.assertEqual((segments[2]["performance_mode"], segments[2]["lyric_no_lip_sync"]), ("singing", False))
        layer = self.layer()
        self.assertEqual(len(layer["story_scene_plan"]), 10)

    def test_act_edits_are_checked(self):
        self.write_book()
        ids = [s["id"] for s in self.read_session()["segments"]]
        good = [{"id": "act_1", "chapter_id": "ch_1", "title": "A", "scene_ids": ids[:4]},
                {"id": "act_2", "chapter_id": "ch_2", "title": "B", "scene_ids": ids[4:]}]
        story.set_story_settings("Song", {"story": {"story_acts": good, "story_spot_min_scenes": 4}})
        self.assertEqual(self.layer()["story_spot_min_scenes"], 4)
        bad = [{"id": "act_1", "chapter_id": "ch_1", "title": "A", "scene_ids": ids[:4]}]
        with self.assertRaises(errors.ValidationError):
            story.set_story_settings("Song", {"story": {"story_acts": bad}})

    def test_plan_needs_acts(self):
        self.write_book()
        with self.assertRaises(errors.ValidationError):
            story_mode.create_story_scene_plan("Song", {})

    def test_whole_story_mode_flow_through_the_api(self):
        import re
        plan_funcs = importlib.import_module(f"{pkg_name}.storyboard.story_plan")
        location_funcs = importlib.import_module(f"{pkg_name}.storyboard.story_locations")
        runner = importlib.import_module(f"{pkg_name}.llm.builder_runner")
        session = self.read_session()
        session["video_type"] = "story"
        for index, segment in enumerate(session["segments"]):
            segment["lyric_text"] = f"under the parking garage lights {index + 1}"
        self.write_session(session)

        def acts_and_plan(payload, instruction, *, label, **kwargs):
            numbers = [int(n) for n in re.findall(r"^Scene (\d+) \(", instruction, re.M)]
            if label == "Story Mode Acts":
                return json.dumps({"acts": [{"first_scene": numbers[0], "title": "Arrive", "summary": "Darrel arrives.",
                                             "cast": ["Darrel"]}]}), {}
            return json.dumps({"scenes": [{"scene": n, "story_slice": f"Darrel does step {n}.", "spot": "Platform" if n < 6 else "Kitchen",
                                           "mood": "hopeful", "performance": "singing"} for n in numbers]}), {}

        def spots(payload, instruction, *, label, **kwargs):
            if label == "Story Mode Setting Looks":
                return json.dumps({"settings": [{"name": "The Night Train", "look": "Brass and velvet."},
                                                {"name": "Home", "look": "Pine and lamplight."}]}), {}
            names = [line[2:].split(":")[0] for line in instruction.splitlines() if line.startswith("- ") and ":" in line]
            return json.dumps({"spots": [{"name": n, "description": f"{n} card."} for n in names]}), {}

        beat_prompts = []

        def beat(payload, instruction, **kwargs):
            beat_prompts.append(instruction)
            return "Darrel steps forward under warm light.", {"runner": "fake", "used_model": "m"}

        with patch.object(book_funcs, "_run", FakeLLM(BOOK)), patch.object(plan_funcs, "_run", acts_and_plan),                 patch.object(location_funcs, "_run", spots), patch.object(runner, "_run_builder_text_llm", beat):
            story_mode.create_story_book("Song", {"story_idea": "A son goes home"})
            story_mode.create_story_acts("Song", {})
            story_mode.create_story_scene_plan("Song", {})
            located = story_mode.create_story_locations("Song", {})
            beats = story.create_scene_beats("Song", {})
        self.assertEqual(beats["created"], 10)
        cards = {c["name"] for c in self.read_session()["flux_reference_builder"]["locations"]}
        self.assertTrue({"The Night Train - Platform", "Home - Kitchen"} <= cards, cards)
        self.assertFalse(any("garage" in name.lower() for name in cards))
        self.assertEqual(located["added"], 2)
        self.assertEqual(len(beat_prompts), 10, "no cast repair: the act cast was mapped onto every scene")
        third = beat_prompts[2]
        self.assertIn("Scene 3 (Platform) - Darrel does step 3.", third)
        self.assertIn("mood and emotion only", third)
        self.assertIn("The Night Train - Platform", third, "the beat uses the scene's spot card")
        self.assertIn("Home - Kitchen", beat_prompts[7])
        subject_map = self.read_session()["flux_reference_builder"]["subject_scene_map"]
        self.assertTrue(all(subject_map.get(seg["id"]) for seg in self.read_session()["segments"]))

    def test_beats_in_story_mode_need_the_plan(self):
        self.write_book()
        with self.assertRaises(errors.ValidationError):
            story.create_scene_beats("Song", {})

    def test_old_story_arc_converts_into_a_book_and_is_kept(self):
        session = self.read_session()
        arc = ("Verse 1:\nDarrel waits on the platform.\nScene 2 (Platform) — He checks the watch.\n\n"
               "Chorus 1: The train arrives.\n\n[Bridge]\nPart of him wants to stay home.")
        session["builder_story_layer"] = {"user_story_arc": arc, "overall_story_idea": "idea"}
        self.write_session(session)
        split = {"chapters": [{"start_paragraph": 1, "title": "Station", "setting": "The Night Train"},
                              {"start_paragraph": 3, "title": "Home", "setting": "Home"}]}
        llm = FakeLLM(json.dumps(split))
        with patch.object(book_funcs, "_run", llm):
            story_mode.convert_story_arc("Song", {})
        self.assertNotIn("Verse 1", llm.instructions[0])
        layer = self.layer()
        self.assertEqual(layer["user_story_arc"], arc, "the old arc is kept")
        self.assertEqual([c["paragraphs"] for c in layer["story_book"]["chapters"]], [
            ["Darrel waits on the platform. He checks the watch.", "The train arrives."],
            ["Part of him wants to stay home."],
        ])
        self.assertEqual(layer["story_flow_version"], 2)

    def test_convert_needs_an_old_arc(self):
        with self.assertRaises(errors.ValidationError):
            story_mode.convert_story_arc("Song", {})

    def test_router_offers_the_story_mode_steps(self):
        source = (ROOT / "agent_api" / "router.py").read_text(encoding="utf-8")
        self.assertIn("**STORY_MODE_JOB_TYPES", source)
        self.assertEqual(set(story_mode.STORY_MODE_JOB_TYPES), set(story_mode.STORY_MODE_STEPS))


if __name__ == "__main__":
    unittest.main()
