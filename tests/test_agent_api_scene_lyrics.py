"""Scene lyrics through the Agent API: PATCH /scenes/{id} must save them and the timeline bulk replace must keep them."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
timeline = importlib.import_module(f"{pkg_name}.builder.timeline")
lyric_scenes = importlib.import_module(f"{pkg_name}.builder.lyric_scenes")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")

OLD_LYRICS = ("Hello", "darkness my old friend", "I've come to talk with you again")


class SceneLyricsApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.allowed_root = os.path.join(self.temp_dir, "output")
        self.folder = os.path.join(self.allowed_root, "Robot_Test")
        os.makedirs(self.folder, exist_ok=True)
        segments = []
        for index, lyric in enumerate(OLD_LYRICS):
            segments.append({
                "id": f"seg_000{index + 1}", "label": f"Scene {index + 1}", "start": index * 4.0, "end": index * 4.0 + 4.0,
                "lyric_text": lyric, "lyric_singers": ["Lead"],
            })
        self.session_file = os.path.join(self.folder, "vrgdg_builder_session.json")
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump({"project_name": "Robot Test", "project_folder": self.folder, "revision": 1, "segments": segments}, handle)
        patcher = patch.object(paths, "get_allowed_project_roots", return_value=[self.allowed_root])
        patcher.start()
        self.addCleanup(patcher.stop)

    def saved(self):
        with open(self.session_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ---- PATCH /scenes/{id} -----------------------------------------------------------------------------------
    def test_patching_lyric_text_is_saved(self):
        res = mutations.patch_scene("Robot_Test", "seg_0001", {"lyric_text": "Brand new words"}, if_match_revision=1)
        self.assertEqual(res["scene"]["lyric_text"], "Brand new words")
        saved = self.saved()["segments"][0]
        self.assertEqual(saved["lyric_text"], "Brand new words")
        self.assertIs(saved["lyric_no_lip_sync"], False)
        self.assertEqual(saved["lyric_singers"], ["Lead"], "other lyric fields are left alone")

    def test_patching_an_instrumental_lyric_turns_lip_sync_off_like_the_builder_does(self):
        mutations.patch_scene("Robot_Test", "seg_0002", {"lyric_text": "[instrumental]"})
        self.assertIs(self.saved()["segments"][1]["lyric_no_lip_sync"], True)
        mutations.patch_scene("Robot_Test", "seg_0002", {"lyric_text": "Words again"})
        self.assertIs(self.saved()["segments"][1]["lyric_no_lip_sync"], False)
        # An explicit flag in the same request wins.
        mutations.patch_scene("Robot_Test", "seg_0002", {"lyric_text": "Words", "lyric_no_lip_sync": True})
        self.assertIs(self.saved()["segments"][1]["lyric_no_lip_sync"], True)

    def test_the_other_scene_text_fields_the_docs_promise_are_saved_too(self):
        mutations.patch_scene("Robot_Test", "seg_0001", {"story_beat": "Opening", "minimax_h3_pass2_prompt": "sharp", "lyric_singers": "Ana"})
        saved = self.saved()["segments"][0]
        self.assertEqual((saved["story_beat"], saved["minimax_h3_pass2_prompt"], saved["lyric_singers"]), ("Opening", "sharp", ["Ana"]))

    def test_an_unsupported_field_is_rejected_and_nothing_is_saved(self):
        before = self.saved()
        with self.assertRaises(errors.ValidationError) as caught:
            mutations.patch_scene("Robot_Test", "seg_0001", {"lyric_txt": "typo", "lyric_text": "good"})
        message = str(caught.exception)
        self.assertIn("lyric_txt", message)
        self.assertIn("lyric_text", message, "the error lists what can be changed")
        after = self.saved()
        self.assertEqual(after["revision"], before["revision"], "no revision bump for a rejected patch")
        self.assertEqual(after["segments"][0]["lyric_text"], "Hello", "a rejected patch changes nothing, even its valid fields")

    def test_an_echoed_id_is_not_an_error(self):
        mutations.patch_scene("Robot_Test", "seg_0001", {"id": "seg_0001", "lyric_text": "ok"})
        self.assertEqual(self.saved()["segments"][0]["lyric_text"], "ok")

    def test_bulk_patch_also_keeps_the_instrumental_flag_in_step(self):
        mutations.bulk_scene_operations("Robot_Test", [{"op": "patch", "scene_id": "seg_0001", "fields": {"lyric_text": "[instrumental]"}}])
        self.assertIs(self.saved()["segments"][0]["lyric_no_lip_sync"], True)

    # ---- POST /timeline/bulk ----------------------------------------------------------------------------------
    def test_replacing_the_timeline_keeps_every_word_of_the_old_lyrics(self):
        res = mutations.timeline_bulk("Robot_Test", "0 - 2\n2 - 6\n6 - 12", mode="ranges", action="replace")
        self.assertEqual((res["scene_count"], res["lyrics"]), (3, "carried_over"))
        scenes = self.saved()["segments"]
        carried = " ".join(scene.get("lyric_text", "") for scene in scenes)
        carried = " ".join(carried.replace("[instrumental]", " ").split())
        self.assertEqual(carried, " ".join(" ".join(OLD_LYRICS).split()))
        self.assertGreaterEqual(res["scenes_with_lyrics"], 3)

    def test_replacing_with_the_same_boundaries_gives_each_scene_its_own_lyric_back(self):
        mutations.timeline_bulk("Robot_Test", "0 - 4\n4 - 8\n8 - 12", mode="ranges", action="replace")
        scenes = self.saved()["segments"]
        self.assertEqual([scene["lyric_text"] for scene in scenes], list(OLD_LYRICS))
        self.assertEqual([scene["lyric_singers"] for scene in scenes], [["Lead"]] * 3)
        self.assertTrue(all(scene["lyric_no_lip_sync"] is False for scene in scenes))

    def test_merging_two_old_scenes_into_one_joins_their_lyrics(self):
        mutations.timeline_bulk("Robot_Test", "0 - 8\n8 - 12", mode="ranges", action="replace")
        scenes = self.saved()["segments"]
        self.assertEqual(scenes[0]["lyric_text"], "Hello\ndarkness my old friend")
        self.assertEqual(scenes[1]["lyric_text"], OLD_LYRICS[2])

    def test_lyrics_written_on_the_lines_are_used_instead(self):
        text = "0:00 --> 0:02.5 Take me to church\n0:02.5 --> 0:06 [instrumental]\n0:06 --> 0:09"
        res = mutations.timeline_bulk("Robot_Test", text, mode="ranges", action="replace")
        self.assertEqual((res["lyrics"], res["scenes_with_lyrics"]), ("from_text", 2))
        scenes = self.saved()["segments"]
        self.assertEqual(scenes[0]["lyric_text"], "Take me to church")
        self.assertIs(scenes[0]["lyric_no_lip_sync"], False)
        self.assertIs(scenes[1]["lyric_no_lip_sync"], True)
        self.assertNotIn("lyric_text", scenes[2], "a line with no words stays empty; old lyrics are not mixed in")

    def test_replacing_a_timeline_that_has_no_lyrics_reports_none(self):
        mutations.patch_scene("Robot_Test", "seg_0001", {"lyric_text": ""})
        mutations.patch_scene("Robot_Test", "seg_0002", {"lyric_text": ""})
        mutations.patch_scene("Robot_Test", "seg_0003", {"lyric_text": ""})
        res = mutations.timeline_bulk("Robot_Test", "5\n5", mode="durations", action="replace")
        self.assertEqual((res["lyrics"], res["scenes_with_lyrics"]), ("none", 0))

    def test_append_works_and_carries_lyrics_from_the_lines(self):
        res = mutations.timeline_bulk("Robot_Test", "3 La la\n2", mode="durations", action="append")
        self.assertEqual((res["scene_count"], res["lyrics"]), (5, "from_text"))
        scenes = self.saved()["segments"]
        self.assertEqual((scenes[3]["start"], scenes[3]["end"], scenes[3]["lyric_text"]), (12.0, 15.0, "La la"))
        self.assertEqual(scenes[4]["start"], 15.0)
        self.assertEqual(scenes[2]["lyric_text"], OLD_LYRICS[2], "the existing scenes are untouched")

    def test_an_unknown_action_is_rejected_instead_of_saving_nothing(self):
        with self.assertRaises(errors.ValidationError):
            mutations.timeline_bulk("Robot_Test", "5", mode="durations", action="merge")
        self.assertEqual(self.saved()["revision"], 1)


class BulkParserTests(unittest.TestCase):
    def test_ranges_accept_words_after_the_times_whatever_they_contain(self):
        scenes = timeline.parse_bulk_scenes("0:10 --> 0:14.5 Take me to church\n14.5 - 20 Hold on - wait", mode="ranges")
        self.assertEqual(scenes[0], {"start": 10.0, "end": 14.5, "lyric_text": "Take me to church"})
        self.assertEqual(scenes[1], {"start": 14.5, "end": 20.0, "lyric_text": "Hold on - wait"})

    def test_plain_ranges_durations_and_markers_still_parse(self):
        self.assertEqual(timeline.parse_bulk_timings("0 --> 3\n3 to 7.5\n7.5 – 9", mode="ranges"), [(0.0, 3.0), (3.0, 7.5), (7.5, 9.0)])
        self.assertEqual(timeline.parse_bulk_timings("1. 2\n- 3.5", mode="durations"), [(0.0, 2.0), (2.0, 5.5)])
        self.assertEqual(timeline.parse_bulk_timings("0\n4\n9", mode="markers"), [(0.0, 4.0), (4.0, 9.0)])
        self.assertEqual(timeline.parse_bulk_timings("00:00:01,000 --> 00:00:03,500", mode="ranges"), [(1.0, 3.5)])

    def test_durations_and_markers_can_carry_words_too(self):
        self.assertEqual(timeline.parse_bulk_scenes("2.5 La la\n3", mode="durations")[0]["lyric_text"], "La la")
        markers = timeline.parse_bulk_scenes("0 First line\n4 Second line\n9", mode="markers")
        self.assertEqual([m.get("lyric_text") for m in markers], ["First line", "Second line"])

    def test_bad_lines_still_say_which_line_is_wrong(self):
        with self.assertRaises(ValueError) as caught:
            timeline.parse_bulk_scenes("0 - 3\nnot a range", mode="ranges")
        self.assertIn("Line 2", str(caught.exception))
        with self.assertRaises(ValueError):
            timeline.parse_bulk_scenes("5 - 4", mode="ranges")


class CarryLyricsOverTests(unittest.TestCase):
    def test_a_lyric_in_a_gap_goes_to_the_nearest_new_scene(self):
        old = [{"start": 4.0, "end": 5.0, "lyric_text": "Lost words"}]
        carried = lyric_scenes.carry_lyrics_over(old, [(0.0, 2.0), (6.0, 8.0)])
        self.assertEqual(carried[0], {})
        self.assertEqual(carried[1]["lyric_text"], "Lost words")

    def test_a_long_old_scene_is_divided_between_the_new_scenes_it_covers(self):
        old = [{"start": 0.0, "end": 8.0, "lyric_text": "one two three four five six"}]
        carried = lyric_scenes.carry_lyrics_over(old, [(0.0, 4.0), (4.0, 8.0)])
        self.assertEqual((carried[0]["lyric_text"], carried[1]["lyric_text"]), ("one two three", "four five six"))

    def test_nothing_is_invented_when_there_are_no_lyrics(self):
        self.assertEqual(lyric_scenes.carry_lyrics_over([{"start": 0, "end": 4}], [(0.0, 4.0)]), [{}])
        self.assertEqual(lyric_scenes.carry_lyrics_over([], [(0.0, 4.0)]), [{}])


if __name__ == "__main__":
    unittest.main()
