"""Story Mode cast: the story is written about the picked characters only, unless new people are allowed."""

import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

story_book = importlib.import_module(f"{ROOT.name}.storyboard.story_book")
builder_runner = importlib.import_module(f"{ROOT.name}.llm.builder_runner")

BOOK = "Title: Alone\nPremise: P\n\nChapter 1: Night\nSetting: Apartment\n\nBrandon waits by the window."
REFS = {"subjects": [
    {"id": "subj_brandon", "name": "Brandon", "description": "Tall, lean.", "reference_type": "character"},
    {"id": "subj_hostess", "name": "Hostess", "description": "Red dress.", "reference_type": "character"},
    {"id": "subj_waiter", "name": "Waiter", "description": "White shirt.", "reference_type": "character"},
]}


def payload(**layer):
    return {
        "story_layer": {"story_flow_version": 2, **layer},
        "scenes": [{"id": f"s{i}", "end": (i + 1) * 4} for i in range(6)],
        "reference_builder": REFS,
        "story_idea": "A player who never settles down.",
    }


class FakeLLM:
    """Writes ``reply``; the cast check finds ``extras``; the removal pass returns ``cleaned``."""

    def __init__(self, reply=BOOK, extras=(), cleaned=""):
        self.prompts = []
        self.labels = []
        self.reply = reply
        self.extras = list(extras)
        self.cleaned = cleaned

    def __call__(self, payload, instruction, *, label="", **kwargs):
        self.prompts.append(instruction)
        self.labels.append(label)
        if label in ("Story Mode Cast Check", "Story Mode Cast Scan"):
            people = [{"name": name, "role": "supporting", "description": "x"} for name in self.extras]
            return json.dumps({"characters": people}), {}
        if label == "Story Mode Remove People":
            return self.cleaned, {}
        return self.reply, {}


BOOK_WITH_OTHERS = ("Title: Alone\nPremise: P\n\nChapter 1: Lobby\nSetting: Condo Lobby\n\nA bellhop pushes a cart past "
                    "Brandon.\n\nChapter 2: Penthouse\nSetting: Penthouse\n\nElena waits for Brandon on the sofa.")
BOOK_CLEANED = ("Title: Alone\nPremise: P\n\nChapter 1: Lobby\nSetting: Condo Lobby\n\nBrandon waits by an empty cart."
                "\n\nChapter 2: Penthouse\nSetting: Penthouse\n\nBrandon reads a message from a woman he never calls.")


class StoryCastTests(unittest.TestCase):
    def draft(self, **layer):
        fake = FakeLLM()
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            story_book.draft_story_book(payload(**layer))
        return fake.prompts[0]

    def test_picked_cast_is_the_only_cast(self):
        prompt = self.draft(story_cast_ids=["subj_brandon"])
        self.assertIn("The story has exactly 1 character", prompt)
        self.assertIn("- Brandon (character, main character): Tall, lean.", prompt)
        self.assertNotIn("Hostess", prompt)
        self.assertNotIn("Waiter", prompt)
        self.assertNotIn("You may add a few new minor people", prompt)

    def test_no_pick_means_every_card_and_still_no_new_people(self):
        prompt = self.draft()
        self.assertIn("The story has exactly 3 characters", prompt)
        for name in ("Brandon", "Hostess", "Waiter"):
            self.assertIn(name, prompt)

    def test_an_old_allow_new_people_setting_is_ignored(self):
        prompt = self.draft(story_cast_ids=["subj_brandon", "subj_hostess"], story_allow_new_cast=True)
        self.assertNotIn("You may add a few new minor people", prompt)
        self.assertIn("The story has exactly 2 characters", prompt)
        self.assertNotIn("Waiter", prompt)

    def test_main_character_roles_and_relationships_reach_the_prompt(self):
        prompt = self.draft(story_main_cast_id="subj_hostess",
                            story_cast_roles={"subj_brandon": "love_interest", "subj_waiter": "antagonist"},
                            story_relationships=[{"a": "subj_hostess", "b": "subj_brandon", "type": "love_interest",
                                                  "note": "estranged"}])
        self.assertIn("- Hostess (character, main character)", prompt)
        self.assertIn("- Brandon (character, love interest)", prompt)
        self.assertIn("- Waiter (character, antagonist)", prompt)
        self.assertIn("Relationships:\n- Hostess and Brandon: love interests (estranged)", prompt)
        self.assertIn("The story has exactly 3 characters", prompt, "relationship lines are not counted as people")
        self.assertIn("Hostess performs the song", prompt, "the main character is the performer")

    def test_ids_of_removed_cards_fall_back_to_every_card(self):
        prompt = self.draft(story_cast_ids=["subj_gone"])
        self.assertIn("The story has exactly 3 characters", prompt)

    def test_chapter_rewrite_keeps_the_cast_rule(self):
        book = {"title": "Alone", "chapters": [
            {"id": "ch_1", "title": "Night", "setting_id": "setting_apartment", "paragraphs": ["Brandon waits."]}]}
        fake = FakeLLM("Chapter 1: Night\nSetting: Apartment\n\nBrandon paces.")
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            story_book.rewrite_story_chapter({
                **payload(story_cast_ids=["subj_brandon"], story_book=book,
                          story_settings=[{"id": "setting_apartment", "name": "Apartment"}]),
                "chapter_id": "ch_1",
            })
        self.assertIn("The story has exactly 1 character", fake.prompts[0])
        self.assertNotIn("Hostess", fake.prompts[0])


    def test_draft_writes_out_people_outside_the_cast(self):
        fake = FakeLLM(BOOK_WITH_OTHERS, extras=["Bellhop", "Elena", "Brandon"], cleaned=BOOK_CLEANED)
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            result = story_book.draft_story_book(payload(story_cast_ids=["subj_brandon"]))
        self.assertEqual(fake.labels, ["Story Mode Book", "Story Mode Cast Check", "Story Mode Remove People"])
        self.assertEqual(result["removed_people"], ["Bellhop", "Elena"])
        self.assertIn("these people are gone: Bellhop, Elena", fake.prompts[2])
        text = result["story_layer"]["story_book_text"]
        self.assertNotIn("bellhop", text.lower())
        self.assertNotIn("Elena", text)
        self.assertEqual([c["id"] for c in result["story_book"]["chapters"]], ["ch_1", "ch_2"])

    def test_a_clean_draft_is_kept_after_one_check(self):
        fake = FakeLLM(BOOK)
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            result = story_book.draft_story_book(payload(story_cast_ids=["subj_brandon"]))
        self.assertEqual(fake.labels, ["Story Mode Book", "Story Mode Cast Check"])
        self.assertEqual(result["removed_people"], [])

    def test_an_old_allow_new_people_setting_still_checks_the_cast(self):
        fake = FakeLLM(BOOK_WITH_OTHERS, extras=["Bellhop"])
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            story_book.draft_story_book(payload(story_allow_new_cast=True))
        self.assertEqual(fake.labels[:2], ["Story Mode Book", "Story Mode Cast Check"])

    def test_a_broken_removal_keeps_the_draft(self):
        fake = FakeLLM(BOOK_WITH_OTHERS, extras=["Bellhop"], cleaned="Chapter 1: Lobby\nSetting: Condo Lobby\n\nOnly one.")
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            result = story_book.draft_story_book(payload(story_cast_ids=["subj_brandon"]))
        self.assertEqual(result["removed_people"], [])
        self.assertIn("bellhop", result["story_layer"]["story_book_text"].lower())

    def test_cast_scan_counts_unpicked_cards_as_outsiders(self):
        book, _names = story_book.parse_story_book_text(BOOK_WITH_OTHERS)
        fake = FakeLLM(extras=["Hostess"])
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            result = story_book.scan_story_cast(payload(story_cast_ids=["subj_brandon"], story_book=book))
        self.assertIn("- Brandon", fake.prompts[0])
        self.assertNotIn("- Hostess", fake.prompts[0])
        self.assertEqual([c["name"] for c in result["characters"]], ["Hostess"])

    def test_remove_people_action(self):
        book, _names = story_book.parse_story_book_text(BOOK_WITH_OTHERS)
        fake = FakeLLM(cleaned=BOOK_CLEANED)
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            result = story_book.remove_story_people({
                **payload(story_cast_ids=["subj_brandon"], story_book=book, story_acts=[{"id": "act_1"}]),
                "names": ["Elena", " "],
            })
        self.assertEqual(result["removed_people"], ["Elena"])
        self.assertNotIn("Elena", result["story_layer"]["story_book_text"])
        self.assertEqual(result["story_layer"]["story_acts"], [])
        with self.assertRaises(ValueError):
            story_book.remove_story_people({**payload(story_book=book), "names": []})


class SceneItemTests(unittest.TestCase):
    """Clothing and prop cards are things the scene's people wear or use, never cast members."""

    def test_items_are_not_cast_and_say_who_wears_them(self):
        story_mode = importlib.import_module(f"{ROOT.name}.storyboard.story_mode")
        mara = {"id": "mara", "name": "Mara", "reference_type": "character"}
        joe = {"id": "joe", "name": "Joe"}
        dress = {"id": "dress", "name": "Red Dress", "description": "Silk, knee length.", "reference_type": "outfit", "wears": "joe"}
        lamp = {"id": "lamp", "name": "Oil Lamp", "reference_type": "prop"}
        self.assertTrue(story_mode.is_story_person_card(mara))
        self.assertTrue(story_mode.is_story_person_card(joe))
        self.assertFalse(story_mode.is_story_person_card(dress))
        text = story_mode.scene_items_text([dress, lamp], [mara, joe])
        self.assertIn("Joe wears Red Dress (Silk, knee length.)", text)
        self.assertIn("Oil Lamp is part of the set", text)
        keys = {"id": "keys", "name": "Car Keys", "reference_type": "object", "wears": "joe"}
        self.assertIn("Joe holds and uses Car Keys the way it is normally used", story_mode.scene_items_text([keys], [mara, joe]))
        self.assertTrue(story_mode.is_story_person_card({"id": "wolf", "name": "Grey Wolf", "reference_type": "creature"}))
        self.assertEqual(story_mode.scene_items_text([], [mara]), "")
        style = {"id": "look", "name": "2dpoly", "reference_type": "style"}
        self.assertEqual(story_mode.scene_items_text([style], [mara]), "")

    def test_story_cast_leaves_out_clothing(self):
        refs = {"subjects": REFS["subjects"] + [{"id": "dress", "name": "Red Dress", "reference_type": "outfit"}]}
        cast = story_book.story_cast({**payload(), "reference_builder": refs})
        self.assertNotIn("Red Dress", [subject.get("name") for subject in cast])

    def test_act_keeps_the_users_location_and_story(self):
        story_mode = importlib.import_module(f"{ROOT.name}.storyboard.story_mode")
        act = story_mode.normalize_story_acts([{"id": "a1", "location_note": "A rooftop at dusk.", "direction": "She burns the letter."}, {"id": "a2"}])
        self.assertEqual(act[0]["location_note"], "A rooftop at dusk.")
        self.assertNotIn("location_note", act[1])
        text = story_mode.act_user_text(act[0])
        self.assertIn("A rooftop at dusk.", text)
        self.assertIn("She burns the letter.", text)
        self.assertEqual(story_mode.act_user_text(act[1]), "")
        layer = {"story_acts": act, "story_scene_plan": [{"scene_id": "s1", "act_id": "a1", "story_slice": "x"}]}
        _arc, brief = story_mode.story_beat_context(layer, "s1", 1)
        self.assertIn("She burns the letter.", brief)


class RealismRuleTests(unittest.TestCase):
    """Every Story Mode step that writes people or places is told to be believable and natural, even with a saved template."""

    def test_realism_rules_reach_every_step(self):
        prompts = importlib.import_module(f"{ROOT.name}.llm.prompts.story_mode")
        saved = "Short saved rules."
        texts = {
            "book": prompts.story_book_instruction(premise="p", cast_text="- A", tone_notes="", scene_count=10, song_seconds=60, lyric_guidance="g", lyrics="", previous_book="", seed="1", template=saved),
            "chapter": prompts.story_chapter_rewrite_instruction(book_text="b", chapter_number=1, chapter_text="c", direction="", cast_text="- A", lyric_guidance="g", template=saved),
            "plan": prompts.story_scene_plan_instruction(act_title="a", act_summary="s", setting_name="n", chapter_text="c", scenes_text="t", cast_text="- A", spot_min_scenes=3, previous_spot="", lyric_guidance="g", known_spots="", template=saved),
        }
        for name, text in texts.items():
            with self.subTest(name):
                self.assertIn(prompts.BELIEVABLE_PLACES_RULE, text)
                self.assertIn(prompts.NATURAL_ACTIONS_RULE, text)
        looks = prompts.story_setting_looks_instruction(settings_text="s", book_summary="b", style_theme="", cast_text="", template=saved)
        self.assertIn(prompts.BELIEVABLE_PLACES_RULE, looks)
        self.assertNotIn(prompts.NATURAL_ACTIONS_RULE, looks)

    def test_beat_rules_keep_the_cast_believable(self):
        story_layer = importlib.import_module(f"{ROOT.name}.storyboard.story_layer")
        with patch.object(story_book, "story_template", return_value="Saved beat rules.\n"):
            self.assertIn("Keep it believable", story_layer._story_mode_beat_rules({}))


class UserLocationsAndPersonalityTests(unittest.TestCase):
    """The user's own places and character personalities reach the book, chapter and setting-look prompts."""

    def test_book_prompt_carries_places_and_personality(self):
        fake = FakeLLM()
        layer = {"story_user_locations": "Grandma's kitchen: yellow wallpaper",
                 "story_character_notes": {"subj_brandon": "guarded, funny when nervous"}}
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            story_book.draft_story_book(payload(**layer))
        prompt = fake.prompts[0]
        self.assertIn("Places the user wants", prompt)
        self.assertIn("Grandma's kitchen: yellow wallpaper", prompt)
        self.assertIn("Personality and emotions: guarded, funny when nervous", prompt)

    def test_loaded_location_cards_stand_in_when_the_locations_box_is_empty(self):
        cards = [{"name": "Old Mill", "description": "Stone walls,\nmoss."}, {"name": "Home - Kitchen", "story_setting_id": "setting_home"}]
        data = {**payload(), "reference_builder": {"locations": cards}}
        self.assertEqual(story_book.user_places(data), [("Old Mill", "Stone walls, moss.")])
        fake = FakeLLM()
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            story_book.draft_story_book(data)
        self.assertIn("Old Mill: Stone walls, moss.", fake.prompts[0])
        typed = {**payload(story_user_locations="Barn"), "reference_builder": {"locations": cards}}
        self.assertEqual(story_book.user_places(typed), [("Barn", "")], "the typed list wins over loaded cards")

    def test_no_places_block_when_the_user_lists_none(self):
        prompts = importlib.import_module(f"{ROOT.name}.llm.prompts.story_mode")
        text = prompts.story_book_instruction(premise="p", cast_text="- A", tone_notes="", scene_count=10, song_seconds=60,
                                              lyric_guidance="g", lyrics="", previous_book="", seed="1")
        self.assertNotIn("Places the user wants", text)
        looks = prompts.story_setting_looks_instruction(settings_text="s", book_summary="b", style_theme="", cast_text="",
                                                        places_text="The porch: peeling paint")
        self.assertIn("The porch: peeling paint", looks)


class UserPlacesMatchingTests(unittest.TestCase):
    PLACES = [("Grandma's kitchen", "Yellow wallpaper."), ("The night train", "Empty carriage."), ("Barn", "")]

    def test_matches_by_whole_words_without_case_or_the(self):
        self.assertEqual(story_book.match_user_place(self.PLACES, "Kitchen")[0], "Grandma's kitchen")
        self.assertEqual(story_book.match_user_place(self.PLACES, "night TRAIN")[0], "The night train")
        self.assertIsNone(story_book.match_user_place(self.PLACES, "Bar"))
        self.assertIsNone(story_book.match_user_place(self.PLACES, "Penthouse"))

    def test_places_are_read_one_per_line_with_details_after_the_colon(self):
        places = story_book.user_places({"story_layer": {"story_user_locations": "- Porch: peeling paint\nGarage"}})
        self.assertEqual(places, [("Porch", "peeling paint"), ("Garage", "")])

    def test_cards_are_built_from_the_list_before_any_scene_is_planned(self):
        story_locations = importlib.import_module(f"{ROOT.name}.storyboard.story_locations")
        data = {"story_layer": {"story_flow_version": 2,
                                "story_user_locations": "Grandma's kitchen: yellow wallpaper\nThe night train",
                                "story_settings": [{"id": "setting_kitchen", "name": "Kitchen", "look": ""}]},
                "scenes": [{"id": "s1"}]}
        result = story_locations.build_story_locations(data)
        self.assertEqual([(item["name"], item["description"]) for item in result["locations"]],
                         [("Grandma's kitchen", "yellow wallpaper"), ("The night train", "")])
        self.assertEqual(result["locations"][0]["story_setting_id"], "setting_kitchen")
        self.assertEqual(result["locations"][0]["scene_ids"], [])
        self.assertEqual(result["story_layer"]["story_settings"][0]["look"], "yellow wallpaper")
        with self.assertRaises(ValueError):
            story_locations.build_story_locations({"story_layer": {"story_flow_version": 2}, "scenes": [{"id": "s1"}]})

    def test_places_come_from_the_story_before_acts_and_acts_and_scenes_use_them(self):
        story_locations = importlib.import_module(f"{ROOT.name}.storyboard.story_locations")
        story_plan = importlib.import_module(f"{ROOT.name}.storyboard.story_plan")
        prompts = importlib.import_module(f"{ROOT.name}.llm.prompts.story_mode")
        layer = {"story_flow_version": 2, "story_user_locations": "Grandma's kitchen: yellow wallpaper",
                 "story_book": {"title": "T", "premise": "", "chapters": [
                     {"id": "ch_1", "title": "Home", "setting_id": "setting_home", "paragraphs": ["She cooks, then sits on the porch."]}]},
                 "story_settings": [{"id": "setting_home", "name": "Home", "look": "Pine and brick."}]}
        seen = []

        def fake(payload_, instruction, *, label, **kwargs):
            seen.append(label)
            if label == "Story Mode Place List":
                return json.dumps({"settings": [{"name": "Home", "spots": ["Kitchen", "Back Porch"]}]}), {"runner": "fake"}
            return json.dumps({"spots": [{"name": "Back Porch", "description": "Peeling paint."}]}), {"runner": "fake"}

        with patch.object(story_locations, "_run", fake):
            result = story_locations.build_story_locations({"story_layer": layer, "scenes": [{"id": "s1"}]})
        by_name = {item["name"]: item for item in result["locations"]}
        self.assertEqual(sorted(by_name), ["Grandma's kitchen", "Home - Back Porch"])
        self.assertEqual(by_name["Grandma's kitchen"]["description"], "yellow wallpaper")
        self.assertEqual(by_name["Grandma's kitchen"]["scene_ids"], [])
        self.assertEqual(seen, ["Story Mode Place List", "Story Mode Spots"], "the user's kitchen needs no LLM text")
        # The made cards reach the acts and the scene plan, and a scene's spot snaps to a made card.
        refs = {"locations": [
            {"id": "a", "name": "Grandma's kitchen", "description": "yellow wallpaper"},
            {"id": "b", "name": "Home - Back Porch", "description": "Peeling paint.", "story_setting_id": "setting_home"}]}
        data = {"story_layer": layer, "reference_builder": refs}
        self.assertEqual(story_book.made_places_text(data, "setting_home"),
                         "- Grandma's kitchen: yellow wallpaper\n- Back Porch: Peeling paint.")
        acts = prompts.story_acts_instruction(chapter_number=1, chapter_title="t", setting_name="Home", chapter_text="c",
                                              scenes_text="t", cast_text="- A", notes_text="", target_acts=2,
                                              made_places_text=story_book.made_places_text(data, "setting_home"))
        self.assertIn("Places already made", acts)
        self.assertIn("- Back Porch: Peeling paint.", acts)
        self.assertEqual(story_plan._snap_to_made_place(story_book.made_place_cards(data, "setting_home"), "the porch"), "Back Porch")
        # A spot no card fits is left blank (the scene takes the spot before it), so the plan never adds a place.
        self.assertEqual(story_plan._snap_to_made_place(story_book.made_place_cards(data, "setting_home"), "Garage"), "")
        self.assertEqual(story_plan._snap_to_made_place([], "Garage"), "Garage", "with no cards yet the spot is kept")

    def test_scene_plan_and_acts_prompts_carry_the_places(self):
        prompts = importlib.import_module(f"{ROOT.name}.llm.prompts.story_mode")
        plan = prompts.story_scene_plan_instruction(
            act_title="a", act_summary="s", setting_name="n", chapter_text="c", scenes_text="t", cast_text="- A",
            spot_min_scenes=3, previous_spot="", lyric_guidance="g", known_spots="", places_text="Porch: paint")
        acts = prompts.story_acts_instruction(chapter_number=1, chapter_title="t", setting_name="n", chapter_text="c",
                                              scenes_text="t", cast_text="- A", notes_text="", target_acts=2,
                                              places_text="Porch: paint")
        for text in (plan, acts):
            self.assertIn("Places the user wants", text)
            self.assertIn("Porch: paint", text)


class ActSummaryTests(unittest.TestCase):
    def test_the_act_description_is_rewritten_from_its_planned_scenes(self):
        story_plan = importlib.import_module(f"{ROOT.name}.storyboard.story_plan")
        seen = []

        def fake(payload_, instruction, *, label, **kwargs):
            seen.append(instruction)
            return json.dumps({"summary": "Brandon leaves the penthouse by the stairs."}), {"runner": "fake", "used_model": "m"}

        data = {
            "story_layer": {"story_flow_version": 2, "story_acts": [{"id": "a1", "chapter_id": "c1", "title": "Leaving", "summary": "OLD TEXT", "scene_ids": ["s0", "s1"], "direction": "He takes the stairs."}],
                             "story_scene_plan": [{"scene_id": "s0", "act_id": "a1", "story_slice": "Brandon picks up his coat.", "spot": "Hall"}, {"scene_id": "s1", "act_id": "a1", "story_slice": "He walks down the stairs.", "spot": "Stairwell"}]},
            "scenes": [{"id": "s0", "end": 4}, {"id": "s1", "end": 8}], "reference_builder": REFS, "act_id": "a1",
        }
        with patch.object(story_plan, "_run", fake):
            result = story_plan.rewrite_act_summary(data)
        self.assertEqual(result["story_layer"]["story_acts"][0]["summary"], "Brandon leaves the penthouse by the stairs.")
        self.assertEqual(result["story_layer"]["story_acts"][0]["title"], "Leaving")
        self.assertIn("He walks down the stairs.", seen[0])
        self.assertIn("He takes the stairs.", seen[0])
        self.assertIn("OLD TEXT", seen[0])

if __name__ == "__main__":
    unittest.main()
