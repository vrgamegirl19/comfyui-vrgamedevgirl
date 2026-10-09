"""PATCH /scenes/{id} edits every scene-card field on the timeline and the Storyboard card, like a UI edit."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
projects = importlib.import_module(f"{pkg_name}.agent_api.projects")
story = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.storyboard_orchestrator")
storyboard_store = importlib.import_module(f"{pkg_name}.storyboard.persistence")
card_fields = importlib.import_module(f"{pkg_name}.storyboard.scene_card_fields")

PROJECT = "Card_Test"

# One value for every editable scene-card field.
FULL_PATCH = {
    "label": "Opening",
    "lyric_text": "We run through the rain",
    "lyric_section": "Verse 1",
    "story_beat": "She leaves the house.",
    "timeline_note": "Director: hold on her face first.",
    "i2v_notes": "Slow push in, rain streaks.",
    "notes": "Planning: practical lights only.",
    "prompt_summary": "A woman leaves at night.",
    "shot_type": "Medium close-up",
    "camera_motion": "Dolly in",
    "character_motion": "Walks briskly",
    "performance_mode": "speaking",
    "performance_style": "Restrained",
    "facial_performance": "determined",
    "facial_performance_custom": "jaw set, eyes wet",
    "include_microphone": True,
    "audio_direction": "Rain on tin roof.",
    "continuity": "Red coat stays buttoned.",
    "flf_start_state": "Door closed",
    "flf_transformation": "Door opens",
    "flf_end_state": "Door open",
    "flf_carry_forward": "Wet coat",
    "video_style": "custom",
    "video_style_custom": "Teal night grade",
    "temporal_world_effect_override": "custom",
    "temporal_world_effect_custom": "Rain slows down",
    "trigger_phrase": "ohwx woman",
    "trigger_position": "end",
    "no_character_present": False,
    "lyric_no_lip_sync": False,
    "lyric_instrumental": False,
    "lyric_singers": ["Ana"],
    "lyric_cue_map": [{"speaker_id": "ana", "text": "We run"}],
    "lyric_shot_word_timing_enabled": True,
    "lyric_performance_mode": "cue_map",
    "speaker_assignments": [{"speaker_id": "ana", "speaker_name": "Ana", "text": "We run through the rain"}],
    "video_prompt_type": "rtv",
    "image_prompt": "A woman in a red coat at a rainy door",
    "video_prompt": "subject_definitions: manual prompt",
    "video_prompt_origin": "manual",
    "minimax_h3_pass2_prompt": "sharper rain",
    "subject_ids": ["ana"],
    "location_id": "porch",
}

# Segment keys and card keys each field must land in.
SEGMENT_EXPECT = {
    "timeline_note": "timeline_note", "i2v_notes": "i2v_notes", "notes": "notes", "lyric_text": "lyric_text",
    "video_style": "minimax_h3_video_style", "video_style_custom": "minimax_h3_video_style_custom",
    "speaker_assignments": "minimax_speaker_assignments", "image_prompt": "t2i_prompt", "video_prompt": "minimax_h3_prompt",
}
CARD_EXPECT = {"timeline_note": "timeline_note", "i2v_notes": "motion_summary", "notes": "notes", "lyric_text": "lyrics"}


class FakeServer:
    def __init__(self):
        self.events = []

    def send_sync(self, event, data, sid=None):
        self.events.append((event, data))


class SceneCardBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        self.folder = os.path.join(self.root, PROJECT)
        os.makedirs(self.folder)
        self.session_file = os.path.join(self.folder, "vrgdg_builder_session.json")
        self.write_session({
            "project_name": PROJECT, "project_folder": self.folder, "revision": 3, "video_engine": "minimax_h3",
            "builder_storyboard_defaults": {"camera_motion_speed": 0, "minimax_h3_cut_frequency": 0, "fx_custom_json": "{}"},
            "flux_reference_builder": {
                "subjects": [{"id": "ana", "name": "Ana", "description": "a woman in a red coat", "image": {"path": ""}},
                             {"id": "ben", "name": "Ben", "description": "a tall man", "image": {"path": ""}}],
                "locations": [{"id": "porch", "name": "Porch", "description": "a rainy porch", "image": {"path": ""}}],
                "subject_scene_map": {"seg_1": ["ben"]}, "scene_map": {},
            },
            "segments": [
                {"id": "seg_1", "label": "Scene 1", "start": 0.0, "end": 4.0, "lyric_text": "line one",
                 "minimax_h3_prompt": "old prompt", "shot_type": "Wide", "include_microphone": True, "timeline_note": "old note"},
                {"id": "seg_2", "label": "Scene 2", "start": 4.0, "end": 8.0, "lyric_text": "line two",
                 "minimax_h3_prompt": "MANUAL PROMPT TWO", "minimax_h3_prompt_origin": "manual"},
                {"id": "seg_3", "label": "Scene 3", "start": 8.0, "end": 12.0, "lyric_text": "line three"},
            ],
        })
        patcher = patch.object(paths, "get_allowed_project_roots", return_value=[self.root])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.server = FakeServer()
        server_module = types.ModuleType("server")
        server_module.PromptServer = types.SimpleNamespace(instance=self.server)
        modules = patch.dict(sys.modules, {"server": server_module})
        modules.start()
        self.addCleanup(modules.stop)

    def write_session(self, data):
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def session(self):
        with open(self.session_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def segment(self, scene_id):
        return next(s for s in self.session()["segments"] if s["id"] == scene_id)

    def storyboard_path(self):
        return os.path.join(self.folder, "storyboard", "storyboard.json")

    def storyboard(self):
        with open(self.storyboard_path(), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def card(self, scene_id):
        return next((s for s in self.storyboard()["scenes"] if s["id"] == scene_id), None)

    def save_storyboard(self):
        """A saved Storyboard with a manual prompt, a Storyboard-only card, a deleted card and its own settings."""
        storyboard_store._save_storyboard({"project_folder": self.folder, "storyboard": {
            "project_video_engine": "minimax_h3", "mode": "image_to_video_prep",
            "source_scene_ids": ["seg_1", "seg_2", "seg_3"],
            "image_aesthetic": "editorial", "fx_preset": "custom", "fx_custom_json": '{"grain": 0.2}',
            "camera_motion_speed": 0, "character_motion_speed": 0, "minimax_h3_cut_frequency": 0,
            "temporal_allow_background_extras": False,
            "reference_builder": {"subjects": [{"id": "ana", "name": "Ana"}], "locations": [], "locations_cleared": False},
            "scenes": [
                {"id": "seg_1", "scene_number": 1, "label": "Scene 1", "prompt_summary": "Saved summary",
                 "trigger_phrase": "keep trigger", "include_microphone": True, "video_prompt": "old prompt"},
                {"id": "seg_2", "scene_number": 2, "label": "Scene 2", "video_prompt": "MANUAL PROMPT TWO",
                 "image_prompt": "MANUAL IMAGE TWO", "notes": "storyboard-only planning two"},
                {"id": "extra_card", "scene_number": 3, "label": "Storyboard only", "notes": "keep me"},
            ],
        }})


class SceneCardEditTests(SceneCardBase):
    def test_every_field_is_saved_on_the_timeline_and_the_storyboard_card(self):
        self.save_storyboard()
        result = mutations.patch_scene(PROJECT, "seg_1", FULL_PATCH, if_match_revision=3)
        segment = self.segment("seg_1")
        card = self.card("seg_1")
        for name, value in FULL_PATCH.items():
            field = card_fields.FIELDS_BY_NAME[name]
            if field.kind in ("references", "reference"):
                continue
            expected = card_fields.normalize_value(field, value)
            if field.segment_keys and name not in ("video_prompt",):
                key = SEGMENT_EXPECT.get(name, field.segment_keys[0])
                self.assertEqual(segment[key], expected, f"segment {key}")
            card_value = card[CARD_EXPECT.get(name, field.card_key)]
            if name == "speaker_assignments":
                card_value = [{k: cue[k] for k in ("speaker_id", "speaker_name", "text")} for cue in card_value]
                expected = [{k: cue[k] for k in ("speaker_id", "speaker_name", "text")} for cue in expected]
            self.assertEqual(card_value, expected, f"card {name}")
        self.assertEqual(segment["video_notes"], FULL_PATCH["i2v_notes"])
        self.assertEqual(segment["minimax_h3_prompt"], FULL_PATCH["video_prompt"])
        self.assertEqual(segment["minimax_h3_prompt_origin"], "manual")
        for key in ("flux_prompt", "nb_prompt", "flow_gpt_prompt"):
            self.assertEqual(segment[key], FULL_PATCH["image_prompt"], "a zimage project edits every image prompt like the Builder")
        refs = self.session()["flux_reference_builder"]
        self.assertEqual(refs["subject_scene_map"]["seg_1"], ["ana"])
        self.assertEqual(refs["scene_map"]["seg_1"], "porch")
        self.assertEqual([ref["id"] for ref in card["subject_refs"]], ["ana"])
        self.assertEqual(card["location_ref"]["id"], "porch")
        self.assertEqual(card["status"], "video_prompt_ready")
        self.assertIn("timeline_note", result["storyboard_card"])
        self.assertGreater(result["revision"], 3)

    def test_clears_false_and_zero_stay_cleared_after_reopening(self):
        self.save_storyboard()
        mutations.patch_scene(PROJECT, "seg_1", {
            "timeline_note": "", "shot_type": "", "include_microphone": False, "lyric_singers": [],
            "prompt_summary": "", "trigger_phrase": "", "subject_ids": [], "image_prompt": "",
            "minimax_h3_continuation_start_seconds": 0,
        })
        segment, card = self.segment("seg_1"), self.card("seg_1")
        self.assertEqual((segment["timeline_note"], segment["shot_type"], segment["include_microphone"]), ("", "", False))
        self.assertEqual(segment["lyric_singers"], [])
        self.assertEqual(segment["minimax_h3_continuation_start_seconds"], 0)
        self.assertEqual((card["timeline_note"], card["shot_type"], card["include_microphone"]), ("", "", False))
        self.assertEqual((card["prompt_summary"], card["trigger_phrase"], card["image_prompt"]), ("", "", ""))
        self.assertEqual(self.session()["flux_reference_builder"]["subject_scene_map"]["seg_1"], [],
                         "an explicit empty mapping, not a fallback to an older one")
        # Reopening the Storyboard merges the timeline in; the cleared values stay cleared there too.
        reopened = {c["id"]: c for c in story.scene_cards(self.session(), self.folder)}
        self.assertEqual((reopened["seg_1"]["shot_type"], reopened["seg_1"]["timeline_note"]), ("", ""))
        self.assertIs(reopened["seg_1"]["include_microphone"], False)
        self.assertEqual(reopened["seg_1"]["prompt_summary"], "")

    def test_unrelated_storyboard_data_and_other_cards_are_preserved(self):
        self.save_storyboard()
        before = self.storyboard()
        mutations.patch_scene(PROJECT, "seg_1", {"timeline_note": "new director note"})
        after = self.storyboard()
        self.assertEqual(self.card("seg_2"), next(s for s in before["scenes"] if s["id"] == "seg_2"))
        self.assertEqual(self.card("extra_card")["notes"], "keep me")
        for key in ("image_aesthetic", "fx_preset", "fx_custom_json", "camera_motion_speed", "character_motion_speed",
                    "minimax_h3_cut_frequency", "temporal_allow_background_extras", "source_scene_ids"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(self.card("seg_1")["prompt_summary"], "Saved summary")
        self.assertEqual(self.card("seg_1")["trigger_phrase"], "keep trigger")
        self.assertEqual(after["revision"], before["revision"] + 1)

    def test_full_storyboard_sync_keeps_manual_prompts_defaults_and_card_edits(self):
        self.save_storyboard()
        session = self.session()
        session["segments"] = [s for s in session["segments"] if s["id"] != "seg_3"]  # removed on the timeline
        self.write_session(session)
        result = story.sync_storyboard_files(PROJECT)
        self.assertTrue(result["saved"], result)
        saved = self.storyboard()
        ids = [s["id"] for s in saved["scenes"]]
        self.assertEqual(ids, ["seg_1", "seg_2", "extra_card"], "Storyboard-only card kept, deleted timeline scene gone")
        self.assertEqual(self.card("seg_2")["video_prompt"], "MANUAL PROMPT TWO")
        self.assertEqual(self.card("seg_1")["prompt_summary"], "Saved summary")
        self.assertEqual(self.card("seg_1")["trigger_phrase"], "keep trigger")
        self.assertEqual(saved["fx_custom_json"], "{}", "the Builder's saved scene defaults win, like Save Storyboard")
        self.assertEqual((saved["camera_motion_speed"], saved["minimax_h3_cut_frequency"]), (0, 0), "zero stays zero")
        self.assertIs(saved["temporal_allow_background_extras"], False)
        self.assertEqual(saved["image_aesthetic"], "editorial")

    def test_a_card_deleted_in_the_storyboard_stays_deleted(self):
        storyboard_store._save_storyboard({"project_folder": self.folder, "storyboard": {
            "source_scene_ids": ["seg_1", "seg_2", "seg_3"],
            "scenes": [{"id": "seg_1", "label": "Scene 1"}, {"id": "seg_2", "label": "Scene 2"}],
        }})
        mutations.patch_scene(PROJECT, "seg_3", {"timeline_note": "timeline only"})
        self.assertIsNone(self.card("seg_3"))
        self.assertEqual(self.segment("seg_3")["timeline_note"], "timeline only")
        before = self.session()
        with self.assertRaises(errors.ValidationError):
            mutations.patch_scene(PROJECT, "seg_3", {"timeline_note": "x", "prompt_summary": "card only"})
        self.assertEqual(self.session(), before, "nothing is saved when the card cannot take the edit")

    def test_without_a_saved_storyboard_only_card_only_fields_create_one(self):
        mutations.patch_scene(PROJECT, "seg_2", {"timeline_note": "note"})
        self.assertFalse(os.path.exists(self.storyboard_path()), "the Storyboard builds its cards from the timeline")
        mutations.patch_scene(PROJECT, "seg_2", {"trigger_phrase": "ohwx"})
        self.assertEqual(self.card("seg_2")["trigger_phrase"], "ohwx")
        self.assertEqual(self.card("seg_2")["timeline_note"], "note")
        self.assertEqual(self.card("seg_2")["video_prompt"], "MANUAL PROMPT TWO")
        self.assertEqual([s["id"] for s in self.storyboard()["scenes"]], ["seg_1", "seg_2", "seg_3"])

    def test_bad_values_unknown_ids_and_alias_conflicts_are_rejected_without_saving(self):
        before = self.session()
        for patch_data in (
            {"include_microphone": "yes"},
            {"performance_mode": "rapping"},
            {"subject_ids": ["nobody"]},
            {"timeline_note": "a", "director_note": "b"},
            {"lyric_cue_map": "not a list"},
            {"no_such_field": 1},
        ):
            with self.subTest(patch=patch_data), self.assertRaises(errors.ValidationError):
                mutations.patch_scene(PROJECT, "seg_1", patch_data)
        self.assertEqual(self.session(), before)

    def test_aliases_name_the_ui_fields(self):
        mutations.patch_scene(PROJECT, "seg_1", {"director_notes": "D", "video_notes": "V", "planning_notes": "P", "lyrics": "L"})
        segment = self.segment("seg_1")
        self.assertEqual((segment["timeline_note"], segment["i2v_notes"], segment["notes"], segment["lyric_text"]), ("D", "V", "P", "L"))

    def test_stale_if_match_is_refused(self):
        with self.assertRaises(errors.RevisionConflictError):
            mutations.patch_scene(PROJECT, "seg_1", {"timeline_note": "late"}, if_match_revision=2)
        self.assertEqual(self.segment("seg_1")["timeline_note"], "old note")

    def test_open_windows_are_told_exactly_what_changed(self):
        self.save_storyboard()
        mutations.patch_scene(PROJECT, "seg_1", {"timeline_note": "new", "subject_ids": ["ana"]})
        name, data = self.server.events[-1]
        self.assertEqual(name, "vrgdg.project_changed")
        self.assertEqual(data["source"], "agent_api")
        self.assertEqual(data["project_folder"], self.folder)
        self.assertEqual(data["revision"], self.session()["revision"])
        self.assertEqual(data["builder_save_revision"], self.session()["builder_save_revision"])
        change = data["change"]["scenes"]["seg_1"]
        self.assertEqual(data["change"]["kind"], "scene_fields")
        self.assertIn("timeline_note", change["segment"])
        self.assertEqual(change["references"], ["subjects"])
        self.assertIn("timeline_note", change["card"])
        mutations.patch_scene(PROJECT, "seg_1", {"start": 0.5})
        self.assertEqual(self.server.events[-1][1]["change"]["kind"], "project", "timing edits reload the project")

    def test_a_failed_session_save_restores_the_storyboard(self):
        self.save_storyboard()
        before = self.storyboard()
        with patch.object(mutations, "_persist_session", side_effect=errors.RevisionConflictError(9, 3)):
            with self.assertRaises(errors.RevisionConflictError):
                mutations.patch_scene(PROJECT, "seg_1", {"prompt_summary": "should roll back"})
        self.assertEqual(self.storyboard(), before)

    def test_scene_detail_reads_back_the_complete_card(self):
        self.save_storyboard()
        mutations.patch_scene(PROJECT, "seg_1", {"timeline_note": "Director", "i2v_notes": "Video", "notes": "Planning"})
        card = projects.get_scene_detail(PROJECT, "seg_1")["scene_card"]
        self.assertEqual((card["timeline_note"], card["motion_summary"], card["notes"]), ("Director", "Video", "Planning"))
        self.assertEqual(card["trigger_phrase"], "keep trigger", "Storyboard-only fields are part of the card")


class StoryboardRevisionTests(SceneCardBase):
    def test_a_window_holding_an_older_revision_cannot_overwrite_api_card_edits(self):
        self.save_storyboard()
        loaded = self.storyboard()["revision"]
        mutations.patch_scene(PROJECT, "seg_1", {"prompt_summary": "API edit"})
        with self.assertRaises(storyboard_store.StoryboardConflictError):
            storyboard_store._save_storyboard({"project_folder": self.folder, "expected_revision": loaded,
                                               "storyboard": {"scenes": []}})
        self.assertEqual(self.card("seg_1")["prompt_summary"], "API edit")
        current = self.storyboard()["revision"]
        storyboard_store._save_storyboard({"project_folder": self.folder, "expected_revision": current,
                                           "storyboard": self.storyboard()})
        self.assertEqual(self.storyboard()["revision"], current + 1)


if __name__ == "__main__":
    unittest.main()
