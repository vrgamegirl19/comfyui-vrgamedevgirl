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
        self.assertIn("- Brandon (character): Tall, lean.", prompt)
        self.assertNotIn("Hostess", prompt)
        self.assertNotIn("Waiter", prompt)
        self.assertNotIn("You may add a few new minor people", prompt)

    def test_no_pick_means_every_card_and_still_no_new_people(self):
        prompt = self.draft()
        self.assertIn("The story has exactly 3 characters", prompt)
        for name in ("Brandon", "Hostess", "Waiter"):
            self.assertIn(name, prompt)

    def test_allowing_new_people(self):
        prompt = self.draft(story_cast_ids=["subj_brandon", "subj_hostess"], story_allow_new_cast=True)
        self.assertIn("You may add a few new minor people", prompt)
        self.assertNotIn("Waiter", prompt)

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

    def test_allowing_new_people_skips_the_check(self):
        fake = FakeLLM(BOOK_WITH_OTHERS, extras=["Bellhop"])
        with patch.object(builder_runner, "_run_builder_text_llm", fake):
            result = story_book.draft_story_book(payload(story_allow_new_cast=True))
        self.assertEqual(fake.labels, ["Story Mode Book"])
        self.assertIn("bellhop", result["story_layer"]["story_book_text"].lower())

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


if __name__ == "__main__":
    unittest.main()
