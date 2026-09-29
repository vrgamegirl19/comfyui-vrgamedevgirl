import importlib.util
import sys
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "storyboard" / "cast_guard.py"
_SPEC = importlib.util.spec_from_file_location("vrgdg_cast_guard", _PATH)
cast_guard = importlib.util.module_from_spec(_SPEC)
sys.modules["vrgdg_cast_guard"] = cast_guard
_SPEC.loader.exec_module(cast_guard)

MAN = {"name": "The man", "description": "The man has wavy brown hair. He wears a khaki shirt and his necklace glints."}
WOMAN = {"name": "the woman", "description": "The woman has blonde hair. She wears a floral dress and her sandals click."}
ANNA = {"name": "Anna Lee", "description": "Anna is a singer with red hair."}
BEN = {"name": "Ben", "description": "Ben plays guitar."}


class CastGuardTests(unittest.TestCase):
    def test_no_guard_when_everyone_is_in_the_cast(self):
        self.assertIsNone(cast_guard.build_cast_guard([MAN, WOMAN], [MAN, WOMAN]))

    def test_excluded_woman_is_flagged_by_noun_and_pronoun(self):
        guard = cast_guard.build_cast_guard([MAN, WOMAN], [MAN])
        text = "The man stands on the porch. The woman approaches from behind. He turns and takes her hand."
        leaks = cast_guard.cast_leaks(text, guard)
        self.assertIn("her", leaks)
        self.assertIn("the woman", leaks)
        self.assertEqual(cast_guard.cast_leaks("The man looks at his necklace.", guard), [])
        self.assertEqual(cast_guard.strip_cast_leaks(text, guard), "The man stands on the porch.")

    def test_man_is_not_flagged_inside_woman(self):
        guard = cast_guard.build_cast_guard([MAN, WOMAN], [WOMAN])
        self.assertEqual(cast_guard.cast_leaks("The woman smiles at the man.", guard), ["the man"])
        self.assertEqual(cast_guard.cast_leaks("The woman smiles.", guard), [])

    def test_pronouns_are_not_used_when_cast_shares_the_gender(self):
        guard = cast_guard.build_cast_guard([ANNA, BEN, MAN], [ANNA, BEN])
        self.assertEqual(cast_guard.cast_leaks("She and he walk together.", guard), [])

    def test_plural_wording_is_flagged_for_a_single_cast_member(self):
        guard = cast_guard.build_cast_guard([MAN, WOMAN], [MAN])
        self.assertIn("their", cast_guard.cast_leaks("The camera holds on their outfits.", guard))
        both = cast_guard.build_cast_guard([MAN, WOMAN, BEN], [MAN, WOMAN])
        self.assertEqual(cast_guard.cast_leaks("They walk together.", both), [])

    def test_named_subject_first_name_is_flagged(self):
        guard = cast_guard.build_cast_guard([ANNA, BEN], [BEN])
        self.assertEqual(cast_guard.cast_leaks("Ben waits while Anna sings.", guard), ["anna"])

    def test_arc_entries_are_filtered_per_scene(self):
        guards = {1: cast_guard.build_cast_guard([MAN, WOMAN], [MAN]), 2: cast_guard.build_cast_guard([MAN, WOMAN], [MAN, WOMAN])}
        arc = (
            "Intro:\n"
            "Scene 1 (Porch) \u2014 The man waits. The woman arrives. He smiles.\n"
            "Scene 2 (Porch) \u2014 The man and the woman walk."
        )
        result = cast_guard.strip_story_arc_entry_leaks(arc, guards)
        self.assertIn("Scene 1 (Porch) \u2014 The man waits.", result)
        self.assertNotIn("arrives", result)
        self.assertIn("The man and the woman walk.", result)

    def test_fully_removed_entry_gets_a_neutral_body(self):
        guards = {1: cast_guard.build_cast_guard([MAN, WOMAN], [MAN])}
        result = cast_guard.strip_story_arc_entry_leaks("Scene 1 (Porch) \u2014 The woman waits.", guards)
        self.assertEqual(result, "Scene 1 (Porch) \u2014 The man remain the focus of the scene.")

    def test_cast_wall_text_names_cast_and_excluded(self):
        text = cast_guard.cast_wall_text(cast_guard.build_cast_guard([MAN, WOMAN], [MAN]))
        self.assertIn("The man", text)
        self.assertIn("the woman", text)


if __name__ == "__main__":
    unittest.main()
