import ast
import re
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "VRGDG_StoryboardBuilderNodes.py"
HELPERS = {
    "_parse_story_arc_lyric_sections",
    "_cap_story_arc_words",
    "_story_arc_section_word_limit",
    "_normalize_story_arc_output",
}


def load_story_arc_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in HELPERS
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


HELPER_NAMESPACE = load_story_arc_helpers()
parse_sections = HELPER_NAMESPACE["_parse_story_arc_lyric_sections"]
section_word_limit = HELPER_NAMESPACE["_story_arc_section_word_limit"]
normalize_output = HELPER_NAMESPACE["_normalize_story_arc_output"]


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
        self.assertIn("CORRECTION: Your previous answer did not follow", source)
        self.assertIn("after an automatic format retry", source)


if __name__ == "__main__":
    unittest.main()
