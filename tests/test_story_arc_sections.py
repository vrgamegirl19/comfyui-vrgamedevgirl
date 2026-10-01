import ast
import re
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "storyboard/story_layer.py"
HELPERS = {
    "_parse_story_arc_lyric_sections",
    "_cap_story_arc_words",
    "_story_arc_section_word_limit",
    "_story_arc_entry_word_limit",
    "_story_arc_scene_map",
    "_normalize_story_arc_output",
    "_feeling_word_hits",
    "_split_story_arc_sections",
    "_story_arc_scene_entries_enabled",
}
CONSTANTS = {"_FEELING_WORDS"}


def load_story_arc_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    helper_nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in HELPERS)
        or (isinstance(node, ast.Assign) and any(getattr(target, "id", "") in CONSTANTS for target in node.targets))
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


HELPER_NAMESPACE = load_story_arc_helpers()
parse_sections = HELPER_NAMESPACE["_parse_story_arc_lyric_sections"]
section_word_limit = HELPER_NAMESPACE["_story_arc_section_word_limit"]
normalize_output = HELPER_NAMESPACE["_normalize_story_arc_output"]
scene_map = HELPER_NAMESPACE["_story_arc_scene_map"]
entry_word_limit = HELPER_NAMESPACE["_story_arc_entry_word_limit"]
feeling_hits = HELPER_NAMESPACE["_feeling_word_hits"]
split_sections = HELPER_NAMESPACE["_split_story_arc_sections"]
entries_enabled = HELPER_NAMESPACE["_story_arc_scene_entries_enabled"]


class StoryArcSectionTests(unittest.TestCase):
    def test_scene_level_headers_collapse_into_real_song_sections(self):
        blocks = [
            ("Instrumental", 3),
            ("Verse 1", 4),
            ("Pre-Chorus", 2),
            ("Chorus", 4),
            ("Verse 2", 4),
            ("Pre-Chorus", 2),
            ("Chorus", 4),
            ("Break", 2),
            ("Instrumental", 12),
        ]
        lines = []
        scene_number = 0
        for label, repetitions in blocks:
            for _ in range(repetitions):
                scene_number += 1
                lines.extend((f"[{label}]", f"scene lyric {scene_number}"))

        parsed = parse_sections("\n".join(lines))

        self.assertEqual(
            [label for label, _body in parsed],
            [
                "Instrumental",
                "Verse 1",
                "Pre-Chorus",
                "Chorus",
                "Verse 2",
                "Pre-Chorus 2",
                "Chorus 2",
                "Break",
                "Instrumental 2",
            ],
        )
        self.assertIn("scene lyric 1", parsed[0][1])
        self.assertIn("scene lyric 3", parsed[0][1])
        self.assertIn("scene lyric 37", parsed[-1][1])

    def test_nonconsecutive_repeated_sections_remain_separate(self):
        parsed = parse_sections("[Chorus]\nfirst\n[Verse]\nsecond\n[Chorus]\nthird")
        self.assertEqual([label for label, _body in parsed], ["Chorus", "Verse", "Chorus 2"])

    def test_long_structures_receive_a_smaller_per_section_limit(self):
        self.assertEqual(section_word_limit(9), 100)
        self.assertEqual(section_word_limit(37), 40)
        self.assertEqual(section_word_limit(100), 30)

    def test_structure_error_identifies_first_mismatch(self):
        with self.assertRaisesRegex(
            ValueError,
            "Expected 3 headings but Qwen Local returned 2.*expected 'Verse', received 'Chorus'",
        ):
            normalize_output(
                "Intro:\nOpening action.\nChorus:\nFinal action.",
                ["Intro", "Verse", "Chorus"],
                100,
                "Qwen Local",
            )

    def test_instruction_echo_preamble_is_ignored_before_valid_sections(self):
        normalized = normalize_output(
            "What user says preserve exactly section order and output headings:\n"
            "Do not change the requested structure.\n"
            "Verse 1:\nOpening action.\n"
            "Chorus:\nFinal action.",
            ["Verse 1", "Chorus"],
            100,
            "Qwen Local",
        )
        self.assertEqual(
            normalized,
            "Verse 1:\nOpening action.\n\nChorus:\nFinal action.",
        )

    def test_invented_story_heading_is_not_treated_as_instruction_echo(self):
        with self.assertRaisesRegex(ValueError, "Qwen Local changed the lyric structure"):
            normalize_output(
                "Prologue:\nInvented opening.\nVerse 1:\nOpening action.",
                ["Verse 1"],
                100,
                "Qwen Local",
            )

    def test_story_arc_generation_has_automatic_format_retry(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        llm_source = (Path(__file__).resolve().parents[1] / "llm" / "prompts" / "storyboard.py").read_text(encoding="utf-8")
        self.assertIn("CRITICAL FORMAT VALIDATION FAILURE", llm_source)
        self.assertIn("after an automatic format retry", source)

    def test_inline_headings_are_accepted(self):
        normalized = normalize_output(
            "Verse 1: The performer enters the archive.\n"
            "Chorus: The performer reaches the central chamber.",
            ["Verse 1", "Chorus"],
            100,
            "Gemma Local",
        )
        self.assertEqual(
            normalized,
            "Verse 1:\nThe performer enters the archive.\n\n"
            "Chorus:\nThe performer reaches the central chamber.",
        )

    def test_adjacent_repeated_sections_are_merged(self):
        normalized = normalize_output(
            "Instrumental: Opening atmosphere.\n"
            "instrumental: More atmosphere.\n"
            "Verse 1: The performer enters.\n"
            "Chorus: The performer sings.\n"
            "instrumental: Closing atmosphere.",
            ["Instrumental", "Verse 1", "Chorus", "Instrumental 2"],
            100,
            "Gemma Local",
        )
        self.assertIn("Opening atmosphere. More atmosphere.", normalized)
        self.assertIn("Closing atmosphere.", normalized)

    def test_missing_headings_include_response_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "No heading lines were detected.*Response preview"):
            normalize_output(
                "The performer enters the archive without section labels.",
                ["Verse 1"],
                100,
                "Gemma Local",
            )

    def test_scene_map_groups_scenes_under_matching_sections_in_order(self):
        rows = [
            {"scene_number": 1, "lyric_section": "Verse", "location": "Footpath"},
            {"scene_number": 2, "lyric_section": "Verse", "location": "Pass"},
            {"scene_number": 3, "lyric_section": "Chorus", "location": "Overlook"},
            {"scene_number": 4, "lyric_section": "Verse", "location": "Cliffside"},
        ]
        mapped = scene_map(rows, ["Verse", "Chorus", "Verse 2"])
        self.assertEqual([label for label, _rows in mapped], ["Verse", "Chorus", "Verse 2"])
        self.assertEqual([[row["scene_number"] for row in group] for _label, group in mapped], [[1, 2], [3], [4]])

    def test_scene_map_attaches_unmatched_runs_to_the_previous_section(self):
        rows = [
            {"scene_number": 1, "lyric_section": "Instrumental", "location": "Chamber"},
            {"scene_number": 2, "lyric_section": "Verse", "location": "Footpath"},
            {"scene_number": 3, "lyric_section": "Instrumental", "location": "Pass"},
            {"scene_number": 4, "lyric_section": "Chorus", "location": "Overlook"},
        ]
        mapped = scene_map(rows, ["Verse", "Chorus"])
        self.assertEqual([[row["scene_number"] for row in group] for _label, group in mapped], [[1, 2, 3], [4]])

    def test_scene_map_is_empty_when_sections_cannot_be_aligned(self):
        rows = [{"scene_number": 1, "lyric_section": "Verse", "location": "Footpath"}]
        self.assertEqual(scene_map(rows, ["Verse", "Bridge"]), [])
        self.assertEqual(scene_map([], ["Verse"]), [])

    def test_entry_word_limit_shrinks_with_scene_count_but_keeps_a_floor(self):
        self.assertEqual(entry_word_limit(10, 55, 1500), 55)
        self.assertEqual(entry_word_limit(60, 55, 1500), 25)
        self.assertEqual(entry_word_limit(500, 55, 1500), 22)

    def test_scene_entries_with_stray_colons_do_not_break_section_headings(self):
        normalized = normalize_output(
            "Verse 1:\n"
            "Scene 1 (Footpath) \u2014 The man tests a loose rock and steadies it.\n"
            "Scene 2 (Pass): The pair pause on the grass and look at the saddle.\n"
            "Chorus:\n"
            "Scene 3 (Overlook) \u2014 He runs across the grass with arms wide.",
            ["Verse 1", "Chorus"],
            200,
            "Gemma Local",
        )
        self.assertIn("Scene 2 (Pass) \u2014 The pair pause", normalized)
        self.assertTrue(normalized.startswith("Verse 1:"))
        self.assertIn("Chorus:\nScene 3 (Overlook)", normalized)

    def test_feeling_words_need_two_hits(self):
        self.assertEqual(feeling_hits("She feels grief and longing."), ["feels", "grief", "longing"])
        self.assertEqual(feeling_hits("She feels the rail and turns."), [])

    def test_sections_split_by_required_labels(self):
        text = "Verse 1:\nHe walks.\nHe stops.\n\nChorus:\nHe spins."
        self.assertEqual(split_sections(text, ["Verse 1", "Chorus"]), {"Verse 1": "He walks.\nHe stops.", "Chorus": "He spins."})

    def test_scene_notes_run_for_detailed_and_rich_unless_set(self):
        self.assertFalse(entries_enabled({}, {}, "standard"))
        self.assertTrue(entries_enabled({}, {}, "detailed"))
        self.assertTrue(entries_enabled({}, {}, "rich"))
        self.assertTrue(entries_enabled({"story_arc_scene_entries": True}, {}, "compact"))
        self.assertFalse(entries_enabled({"story_arc_scene_entries": "off"}, {}, "rich"))

    def test_scene_entries_stay_on_separate_lines(self):
        normalized = normalize_output(
            "Verse 1:\n"
            "Scene 1 (Footpath) \u2014 The man tests a loose rock and steadies it.\n"
            "Scene 2 (Pass) \u2014 The pair pause on the grass.",
            ["Verse 1"],
            200,
            "Gemma Local",
        )
        self.assertEqual(
            normalized,
            "Verse 1:\nScene 1 (Footpath) \u2014 The man tests a loose rock and steadies it.\n"
            "Scene 2 (Pass) \u2014 The pair pause on the grass.",
        )


if __name__ == "__main__":
    unittest.main()
