"""Scenes from timed lyric lines and the min/max scene-length rule."""

import importlib
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

ls = importlib.import_module(f"{ROOT.name}.builder.lyric_scenes")


def scene(start, end, text="la la", scene_id=None, **extra):
    return {"id": scene_id or f"s{start}", "start": float(start), "end": float(end), "lyric_text": text, **extra}


def durations(segments):
    return [round(s["end"] - s["start"], 3) for s in sorted(segments, key=lambda s: s["start"])]


def words_of(segments):
    """Sung words in time order (the "[instrumental]" marker a one-word scene leaves on its other half is not a word)."""
    return [
        word for s in sorted(segments, key=lambda s: s["start"]) if not ls.is_instrumental_lyric_text(s["lyric_text"])
        for word in s["lyric_text"].split()
    ]


class LyricTextTests(unittest.TestCase):
    def test_instrumental_markers(self):
        for text in ("[instrumental]", "Instrumental", "instrumental section.", "[Intro] [instrumental]", "[b-roll]"):
            self.assertTrue(ls.is_instrumental_lyric_text(text), text)
        for text in ("", "Don't get it twisted", "Instrumental of my heart"):
            self.assertFalse(ls.is_instrumental_lyric_text(text), text)

    def test_headers_are_stripped_but_non_vocal_markers_stay(self):
        self.assertEqual(ls.clean_timestamped_lyric_text("[Verse 1] Slide in the room , I'm the ghost"), "Slide in the room, I'm the ghost")
        self.assertEqual(ls.clean_timestamped_lyric_text("[instrumental]"), "[instrumental]")

    def test_merge_ignores_instrumental_text_and_repeats(self):
        self.assertEqual(ls.merge_lyric_text("one two", "three"), "one two\nthree")
        self.assertEqual(ls.merge_lyric_text("[instrumental]", "three"), "three")
        self.assertEqual(ls.merge_lyric_text("[instrumental]", "[instrumental]"), "[instrumental]")
        self.assertEqual(ls.merge_lyric_text("same", "same"), "same")

    def test_split_divides_lines_then_words_and_keeps_instrumental(self):
        self.assertEqual(ls.split_lyric_text("a\nb\nc\nd", 0.5), ("a\nb", "c\nd"))
        self.assertEqual(ls.split_lyric_text("one two three four", 0.5), ("one two", "three four"))
        self.assertEqual(ls.split_lyric_text("[instrumental]", 0.5), ("[instrumental]", "[instrumental]"))
        left, right = ls.split_lyric_text("solo", 0.5)
        self.assertEqual(left, "solo")
        self.assertTrue(ls.is_instrumental_lyric_text(right))


class SectionLabelTests(unittest.TestCase):
    LYRICS = "[Intro]\nDon't get it twisted.\n\n[Verse 1]\nSlide in the room, I'm the ghost\nShe looking for a king\n\n[bridge]\nI get the bag, then I'm OUT\n\n[Final Chorus]\nBanging like a drum\n"

    def test_sections_follow_the_headers(self):
        segments = [scene(0, 3, "Don't get it twisted."), scene(3, 7, "Slide in the room, I'm the ghost\nShe looking for a king"),
                    scene(7, 9, "I get the bag, then I'm OUT"), scene(9, 12, "[instrumental]")]
        applied = ls.apply_lyric_sections(segments, self.LYRICS)
        self.assertEqual([s["lyric_section"] for s in segments], ["Intro", "Verse 1", "bridge", "instrumental"])
        self.assertEqual(applied, 3)

    def test_existing_sections_are_kept(self):
        segments = [scene(0, 3, "Don't get it twisted.", lyric_section="Custom")]
        ls.apply_lyric_sections(segments, self.LYRICS)
        self.assertEqual(segments[0]["lyric_section"], "Custom")


class FromTimedLinesTests(unittest.TestCase):
    def payload(self):
        return {
            "duration": 30.0,
            "segment_mode": "reference_lines",
            "segments": [
                {"start": 4.0, "end": 8.0, "text": "[Verse 1] Slide in the room", "type": "vocal"},
                {"start": 8.0, "end": 12.0, "text": "She looking for a king", "type": "vocal"},
                {"start": 20.0, "end": 24.0, "text": "I get the bag", "type": "vocal"},
            ],
        }

    def test_gaps_and_the_tail_become_instrumental_scenes(self):
        scenes = ls.segments_from_timestamped_payload(self.payload(), min_gap_seconds=2.0, max_scene_seconds=10.0)
        self.assertEqual([(s["start"], s["end"]) for s in scenes], [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 24.0), (24.0, 30.0)])
        self.assertEqual([s["lyric_text"] for s in scenes][1], "Slide in the room")
        self.assertTrue(all(s["lyric_no_lip_sync"] for s in (scenes[0], scenes[3], scenes[5])))
        self.assertEqual([s["label"] for s in scenes], [f"SCENE {i}" for i in range(1, 7)])
        self.assertEqual(len({s["id"] for s in scenes}), 6)

    def test_gaps_can_be_turned_off(self):
        scenes = ls.segments_from_timestamped_payload(self.payload(), include_instrumental_gaps=False)
        self.assertEqual([s["lyric_text"] for s in scenes], ["Slide in the room", "She looking for a king", "I get the bag"])
        self.assertEqual(scenes[0]["start"], 0.0)  # the first scene starts at the song start

    def test_a_long_whisper_chunk_is_split_near_a_word_boundary(self):
        words = [{"text": f"w{i}", "start": float(i), "end": i + 0.9} for i in range(30)]
        payload = {"duration": 30.0, "segments": [{"start": 0.0, "end": 30.0, "text": "x", "type": "vocal", "words": words}]}
        scenes = ls.segments_from_timestamped_payload(payload, segment_mode="whisper_chunks", max_scene_seconds=8.0)
        self.assertGreaterEqual(len(scenes), 3)
        self.assertTrue(all(s["end"] - s["start"] <= 10.0 + 0.1 for s in scenes))
        self.assertEqual(" ".join(w for s in scenes for w in s["lyric_text"].split()), " ".join(f"w{i}" for i in range(30)))
        self.assertTrue(any("split near a word boundary" in s["timeline_note"] for s in scenes))

    def test_reference_modes_never_split_a_line(self):
        payload = {"duration": 20.0, "segments": [{"start": 0.0, "end": 20.0, "text": "one very long line", "type": "vocal"}]}
        scenes = ls.segments_from_timestamped_payload(payload, segment_mode="exact_reference_lines", max_scene_seconds=8.0)
        self.assertEqual(len(scenes), 1)

    def test_normalize_merges_short_scenes_into_a_vocal_neighbor(self):
        scenes = [scene(0, 4, "one"), scene(4, 5.2, "two"), scene(5.2, 11, "three")]
        result = ls.normalize_scene_durations(scenes, min_scene_seconds=3.5, max_scene_seconds=10.0)
        self.assertEqual([(s["start"], s["end"]) for s in result], [(0.0, 5.2), (5.2, 11.0)])
        self.assertEqual(result[0]["lyric_text"], "one\ntwo")

    def test_normalize_leaves_reference_units_alone(self):
        scenes = [scene(0, 1, "a"), scene(1, 2, "b")]
        result = ls.normalize_scene_durations(scenes, segment_mode="reference_lines")
        self.assertEqual(len(result), 2)


class LengthRuleTests(unittest.TestCase):
    def test_a_long_scene_is_split_into_equal_parts_with_its_lyrics(self):
        segments = [scene(0, 12, "one two three four five six", scene_id="a")]
        ls.enforce_scene_lengths(segments, 3.5, 10.0)
        self.assertEqual(durations(segments), [6.0, 6.0])
        self.assertEqual(words_of(segments), "one two three four five six".split())
        self.assertEqual(len({s["id"] for s in segments}), 2)

    def test_very_long_scenes_split_into_as_many_equal_parts_as_needed(self):
        segments = [scene(0, 25, "a b c d e f g h i j", scene_id="a")]
        ls.enforce_scene_lengths(segments, 3.5, 10.0)
        self.assertEqual(len(segments), 3)
        self.assertTrue(all(abs(d - 25 / 3) < 0.01 for d in durations(segments)))

    def test_short_scenes_are_merged_up_to_the_minimum(self):
        segments = [scene(0, 2.1, "a", scene_id="a"), scene(2.1, 4.2, "b", scene_id="b"), scene(4.2, 8.0, "c", scene_id="c")]
        ls.enforce_scene_lengths(segments, 3.5, 10.0)
        self.assertTrue(all(3.5 <= d <= 10.0 for d in durations(segments)), durations(segments))
        self.assertEqual(words_of(segments), ["a", "b", "c"])

    def test_a_short_scene_between_two_long_ones_rebalances(self):
        segments = [scene(0, 9, "a", scene_id="a"), scene(9, 11.1, "b", scene_id="b"), scene(11.1, 20.1, "c", scene_id="c")]
        ls.enforce_scene_lengths(segments, 3.5, 10.0)
        self.assertTrue(all(3.5 <= d <= 10.0 for d in durations(segments)), durations(segments))

    def test_locked_scenes_are_never_touched(self):
        segments = [scene(0, 14, "a b", scene_id="locked"), scene(14, 15, "c", scene_id="x"), scene(15, 20, "d", scene_id="y")]
        ls.enforce_scene_lengths(segments, 3.5, 10.0, locked_ids={"locked"})
        locked = next(s for s in segments if s["id"] == "locked")
        self.assertEqual((locked["start"], locked["end"]), (0.0, 14.0))

    def test_instrumental_scenes_do_not_swallow_lyrics(self):
        segments = [scene(0, 2, "[instrumental]", scene_id="i"), scene(2, 9, "a b c", scene_id="v")]
        ls.enforce_scene_lengths(segments, 3.5, 10.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["lyric_text"], "a b c")

    def test_a_timeline_that_already_fits_is_left_alone(self):
        segments = [scene(0, 5, "a"), scene(5, 12, "b")]
        before = [dict(s) for s in segments]
        ls.enforce_scene_lengths(segments, 3.5, 10.0)
        self.assertEqual([(s["start"], s["end"]) for s in segments], [(s["start"], s["end"]) for s in before])
        self.assertIsNone(ls.next_length_fix(segments, 3.5, 10.0))

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            ls.next_length_fix([scene(0, 5)], 10.0, 3.5)
        with self.assertRaises(ValueError):
            ls.next_length_fix([scene(0, 5)], 0, 3.5)

    def test_random_timelines_end_up_inside_the_limits_without_losing_lyrics(self):
        rng = random.Random(7)
        for trial in range(200):
            cursor, segments, words = 0.0, [], []
            for index in range(rng.randint(2, 25)):
                length = rng.choice([0.4, 1.1, 2.1, 2.9, 3.6, 5.0, 7.5, 9.9, 12.0, 17.0, 26.0])
                text = " ".join(f"t{trial}_{index}_{n}" for n in range(rng.randint(1, 8)))
                segments.append(scene(cursor, cursor + length, text, scene_id=f"s{index}"))
                words.extend(text.split())
                cursor += length
            total = cursor
            ls.enforce_scene_lengths(segments, 3.5, 10.0)
            ordered = sorted(segments, key=lambda s: s["start"])
            self.assertAlmostEqual(ordered[0]["start"], 0.0, places=3)
            self.assertAlmostEqual(ordered[-1]["end"], total, places=2)
            for left, right in zip(ordered, ordered[1:]):
                self.assertAlmostEqual(left["end"], right["start"], places=3, msg=f"gap in trial {trial}")
            self.assertEqual(words_of(segments), words, f"lyrics changed in trial {trial}")
            if total >= 3.5:
                for d in durations(segments):
                    self.assertLessEqual(d, 10.0 + 0.02, f"trial {trial}: {durations(segments)}")
                    self.assertGreaterEqual(d, 3.5 - 0.02, f"trial {trial}: {durations(segments)}")


if __name__ == "__main__":
    unittest.main()
