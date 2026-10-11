"""Story Mode cast: the story is written about the picked characters only, unless new people are allowed."""

import importlib
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
    def __init__(self, reply=BOOK):
        self.prompts = []
        self.reply = reply

    def __call__(self, payload, instruction, **kwargs):
        self.prompts.append(instruction)
        return self.reply, {}


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


if __name__ == "__main__":
    unittest.main()
