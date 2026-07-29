import ast
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT_SOURCE = ROOT / "HumoAutomationExtra2.py"
BUILDER_SOURCE = ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js"


def load_timestamp_extractor_harness():
    tree = ast.parse(ALIGNMENT_SOURCE.read_text(encoding="utf-8"), filename=str(ALIGNMENT_SOURCE))
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VRGDG_TimestampedLyricsExtractor"
    )
    method_names = {
        "_word_items_from_segments",
        "_align_reference_unit",
        "_is_reference_marker_line",
        "_is_instrumental_marker_line",
        "_reference_units",
        "_segments_from_reference_units",
    }
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    harness_class = ast.ClassDef(
        name="TimestampExtractorHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    namespace = {
        "math": math,
        "re": re,
        "_assign_unaligned_reference_units": lambda units, aligned: {
            index: [index] for index in aligned
        },
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[harness_class], type_ignores=[])), str(ALIGNMENT_SOURCE), "exec"),
        namespace,
    )
    harness = namespace["TimestampExtractorHarness"]()
    harness._clean_lyric = lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
    harness._normalize_for_match = lambda value: re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value or "").lower(),
    ).strip()
    return harness


class ExactReferenceLineTimingTests(unittest.TestCase):
    def setUp(self):
        self.extractor = load_timestamp_extractor_harness()

    def build(self, units, stable_segments, total_duration=30.0):
        return self.extractor._segments_from_reference_units(
            units,
            stable_segments,
            total_duration,
            "[instrumental]",
            2.0,
            include_instrumental_gaps=True,
            min_scene_seconds=10.0,
            max_scene_seconds=10.0,
            vocal_tail_padding_seconds=0.6,
            exact_reference_lines=True,
            preserve_reference_units=True,
        )

    def assert_no_gaps_or_overlaps(self, segments):
        for index, segment in enumerate(segments):
            self.assertGreater(segment["end"], segment["start"])
            if index:
                self.assertAlmostEqual(segment["start"], segments[index - 1]["end"], places=3)

    def test_exact_mode_uses_word_timing_but_ignores_scene_minimum_and_maximum(self):
        segments = self.build(
            [
                {"type": "vocal", "text": "Hello world"},
                {"type": "vocal", "text": "Second line"},
            ],
            [
                {
                    "words": [
                        {"start": 5.0, "end": 5.35, "text": "Hello"},
                        {"start": 5.5, "end": 6.0, "text": "world"},
                        {"start": 6.3, "end": 6.65, "text": "Second"},
                        {"start": 6.8, "end": 7.1, "text": "line"},
                    ]
                }
            ],
        )

        vocals = [segment for segment in segments if segment["type"] == "vocal"]
        instrumentals = [segment for segment in segments if segment["type"] == "instrumental"]
        self.assertEqual([segment["text"] for segment in vocals], ["Hello world", "Second line"])
        self.assertAlmostEqual(vocals[0]["start"], 5.0)
        self.assertAlmostEqual(vocals[0]["end"], 6.3)
        self.assertAlmostEqual(vocals[1]["start"], 6.3)
        self.assertAlmostEqual(vocals[1]["end"], 7.7)
        self.assertTrue(all(segment["duration"] < 10.0 for segment in vocals))
        self.assertTrue(any(segment["duration"] > 10.0 for segment in instrumentals))
        self.assert_no_gaps_or_overlaps(segments)

    def test_unaligned_exact_line_gets_text_sized_timing_not_rest_of_song(self):
        segments = self.build(
            [
                {"type": "vocal", "text": "Detected words"},
                {"type": "vocal", "text": "Missing final lyric"},
            ],
            [
                {
                    "words": [
                        {"start": 5.0, "end": 5.3, "text": "Detected"},
                        {"start": 5.45, "end": 5.8, "text": "words"},
                    ]
                }
            ],
            total_duration=120.0,
        )

        vocals = [segment for segment in segments if segment["type"] == "vocal"]
        self.assertEqual([segment["text"] for segment in vocals], ["Detected words", "Missing final lyric"])
        self.assertLess(vocals[-1]["duration"], 5.0)
        self.assertLess(vocals[-1]["end"], 10.0)
        self.assertEqual(segments[-1]["type"], "instrumental")
        self.assertAlmostEqual(segments[-1]["end"], 120.0)
        self.assert_no_gaps_or_overlaps(segments)

    def test_explicit_instrumental_intro_ends_at_first_detected_lyric(self):
        segments = self.build(
            [
                {"type": "instrumental", "text": "[instrumental]"},
                {"type": "vocal", "text": "First real lyric"},
            ],
            [
                {
                    "words": [
                        {"start": 32.0, "end": 32.25, "text": "First"},
                        {"start": 32.4, "end": 32.65, "text": "real"},
                        {"start": 32.8, "end": 33.1, "text": "lyric"},
                    ]
                }
            ],
            total_duration=40.0,
        )

        self.assertEqual(segments[0]["type"], "instrumental")
        self.assertAlmostEqual(segments[0]["start"], 0.0)
        self.assertAlmostEqual(segments[0]["end"], 32.0)
        self.assertEqual(segments[1]["text"], "First real lyric")
        self.assertAlmostEqual(segments[1]["start"], 32.0)
        self.assert_no_gaps_or_overlaps(segments)

    def test_only_explicit_instrumental_markers_create_no_vocal_units(self):
        units = self.extractor._reference_units(
            "\n".join([
                "[intro]",
                "Intro lyric",
                "[break]",
                "Break lyric",
                "[instrumental]",
                "[instrumental break]",
                "[outro]",
                "Outro lyric",
                "[interlude]",
                "Interlude lyric",
            ]),
            "reference_lines",
            "[instrumental]",
        )

        self.assertEqual(
            [(unit["type"], unit["text"]) for unit in units],
            [
                ("vocal", "Intro lyric"),
                ("vocal", "Break lyric"),
                ("instrumental", "[instrumental]"),
                ("instrumental", "[instrumental break]"),
                ("vocal", "Outro lyric"),
                ("vocal", "Interlude lyric"),
            ],
        )
        stanza_units = self.extractor._reference_units(
            "Before the break\n[break]\nAfter the break",
            "reference_stanzas",
            "[instrumental]",
        )
        self.assertEqual(
            [(unit["type"], unit["text"]) for unit in stanza_units],
            [
                ("vocal", "Before the break"),
                ("vocal", "After the break"),
            ],
        )

    def test_ui_documents_only_the_two_explicit_instrumental_markers(self):
        builder_source = BUILDER_SOURCE.read_text(encoding="utf-8")
        wizard_source = (ROOT / "web" / "VRGDG_MusicVideoWizardUI.js").read_text(encoding="utf-8")

        for source in (builder_source, wizard_source):
            self.assertIn("[instrumental break]", source)
            self.assertIn("[intro]", source)
            self.assertIn("[outro]", source)
            self.assertIn("[break]", source)

    def test_ui_only_hides_minimum_and_maximum_for_exact_mode(self):
        source = BUILDER_SOURCE.read_text(encoding="utf-8")
        self.assertIn('minSceneField.style.display = isBeatMode || isExactMode ? "none" : "";', source)
        self.assertIn('maxSceneField.style.display = isBeatMode || isExactMode ? "none" : "";', source)
        self.assertIn('vocalTailField.style.display = isBeatMode ? "none" : "";', source)
        self.assertIn("The only disabled rules are minimum and maximum scene duration", source)

    def test_reference_line_units_override_minimum_and_maximum(self):
        short_segments = self.extractor._segments_from_reference_units(
            [
                {"type": "vocal", "text": "Playing in the rain"},
            ],
            [
                {
                    "words": [
                        {"start": 16.92, "end": 17.2, "text": "Playing"},
                        {"start": 17.35, "end": 17.5, "text": "in"},
                        {"start": 17.62, "end": 17.78, "text": "the"},
                        {"start": 17.9, "end": 18.2, "text": "rain"},
                    ]
                }
            ],
            30.0,
            "[instrumental]",
            2.0,
            include_instrumental_gaps=False,
            min_scene_seconds=3.0,
            max_scene_seconds=10.0,
            vocal_tail_padding_seconds=0.0,
            exact_reference_lines=False,
            preserve_reference_units=True,
        )
        short_vocal = next(segment for segment in short_segments if segment["type"] == "vocal")
        self.assertEqual(short_vocal["text"], "Playing in the rain")
        self.assertAlmostEqual(short_vocal["start"], 16.92)
        self.assertAlmostEqual(short_vocal["end"], 18.2)
        self.assertLess(short_vocal["duration"], 3.0)

        long_segments = self.extractor._segments_from_reference_units(
            [{"type": "vocal", "text": "One continuous long lyric line"}],
            [
                {
                    "words": [
                        {"start": 1.0, "end": 1.2, "text": "One"},
                        {"start": 3.0, "end": 3.2, "text": "continuous"},
                        {"start": 6.0, "end": 6.2, "text": "long"},
                        {"start": 9.0, "end": 9.2, "text": "lyric"},
                        {"start": 12.0, "end": 12.2, "text": "line"},
                    ]
                }
            ],
            20.0,
            "[instrumental]",
            2.0,
            include_instrumental_gaps=False,
            min_scene_seconds=1.0,
            max_scene_seconds=4.0,
            vocal_tail_padding_seconds=0.0,
            exact_reference_lines=False,
            preserve_reference_units=True,
        )
        long_vocals = [segment for segment in long_segments if segment["type"] == "vocal"]
        self.assertEqual(len(long_vocals), 1)
        self.assertGreater(long_vocals[0]["duration"], 4.0)


if __name__ == "__main__":
    unittest.main()
