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
        self.assertIn("Mara holds or uses Oil Lamp", text)
        self.assertEqual(story_mode.scene_items_text([], [mara]), "")

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


if __name__ == "__main__":
    unittest.main()
