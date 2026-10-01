"""Unit tests for Phase A3: Agent API Mutations (Scene CRUD, Timeline ops, Journaling, References, Lyrics, Audio, Prompts)."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
timeline = importlib.import_module(f"{pkg_name}.builder.timeline")
prompt_assembly = importlib.import_module(f"{pkg_name}.minimax.prompt_assembly")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")


class AgentApiMutationsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

        self.allowed_root = os.path.join(self.temp_dir, "output")
        os.makedirs(self.allowed_root, exist_ok=True)
        self.proj_folder = os.path.join(self.allowed_root, "Test_Track")
        os.makedirs(self.proj_folder, exist_ok=True)

        # Asset folders
        self.images_dir = os.path.join(self.proj_folder, "zimage_approved")
        self.scene_videos_dir = os.path.join(self.proj_folder, "rendered_scene_videos")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.scene_videos_dir, exist_ok=True)

        # Create 2 initial dummy image files
        with open(os.path.join(self.images_dir, "image_0001.png"), "wb") as f:
            f.write(b"PNG_1")
        with open(os.path.join(self.images_dir, "image_0002.png"), "wb") as f:
            f.write(b"PNG_2")

        # Initial session
        self.session_data = {
            "project_name": "Test Track",
            "project_folder": self.proj_folder,
            "revision": 1,
            "video_engine": "minimax_h3",
            "video_mode": "text_to_video",
            "segments": [
                {
                    "id": "seg_0001",
                    "label": "Scene 1",
                    "start": 0.0,
                    "end": 4.0,
                    "notes": "Intro beat",
                    "t2i_prompt": "Cyberpunk city night",
                    "i2v_prompt": "Camera pans slowly",
                    "approved_image_path": os.path.join(self.images_dir, "image_0001.png"),
                },
                {
                    "id": "seg_0002",
                    "label": "Scene 2",
                    "start": 4.0,
                    "end": 8.0,
                    "notes": "Verse begins",
                    "t2i_prompt": "Performer at microphone",
                    "approved_image_path": os.path.join(self.images_dir, "image_0002.png"),
                },
            ],
            "flux_reference_builder": {
                "subjects": [],
                "locations": [],
            },
            "subject_scene_map": {},
            "scene_map": {},
            "beats": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "tempo_bpm": 120.0,
        }
        self.session_file = os.path.join(self.proj_folder, "vrgdg_builder_session.json")
        with open(self.session_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.session_data, indent=2))

        self.path_patch = patch.object(paths, "get_allowed_project_roots", return_value=[self.allowed_root])
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    # 1. Scene Creation & File Slot Reservation
    def test_create_scene_append(self):
        res = mutations.create_scene("Test_Track", position="append", duration=3.5, label="Scene 3")
        self.assertIn("scene", res)
        self.assertEqual(res["scene"]["start"], 8.0)
        self.assertEqual(res["scene"]["end"], 11.5)
        self.assertEqual(res["revision"], 2)

        # Check session on disk
        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(len(saved["segments"]), 3)
        self.assertEqual(saved["segments"][2]["label"], "Scene 3")

    def test_create_scene_insert_before_shifts_assets(self):
        # Insert before scene 2 (slot 2) -> scene 2's image should shift to image_0003.png
        res = mutations.create_scene("Test_Track", position="before", ref_scene_id="seg_0002", duration=2.0)
        self.assertEqual(res["scene"]["start"], 4.0)
        self.assertEqual(res["scene"]["end"], 6.0)

        # Image 0002 should have been shifted to 0003
        self.assertTrue(os.path.isfile(os.path.join(self.images_dir, "image_0003.png")))
        self.assertFalse(os.path.isfile(os.path.join(self.images_dir, "image_0002.png")))

        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(len(saved["segments"]), 3)
        # Seg 2 should now start at 6.0 and its image path should point to image_0003.png
        seg2 = next(s for s in saved["segments"] if s["id"] == "seg_0002")
        self.assertEqual(seg2["start"], 6.0)
        self.assertIn("image_0003.png", seg2["approved_image_path"])

    # 2. Scene Deletion & Removal Renumbering
    def test_delete_scene_shifts_later_assets_down(self):
        # Delete scene 1 -> image_0001 archived, image_0002 shifted to image_0001
        res = mutations.delete_scene("Test_Track", "seg_0001", ripple=True)
        self.assertEqual(res["deleted_scene_id"], "seg_0001")

        self.assertTrue(os.path.isfile(os.path.join(self.images_dir, "image_0001.png")))
        self.assertFalse(os.path.isfile(os.path.join(self.images_dir, "image_0002.png")))

        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(len(saved["segments"]), 1)
        self.assertEqual(saved["segments"][0]["id"], "seg_0002")
        # Due to ripple, scene 2 should now start at 0.0
        self.assertEqual(saved["segments"][0]["start"], 0.0)
        self.assertEqual(saved["segments"][0]["end"], 4.0)

    # 3. Scene Split & Merge
    def test_split_scene(self):
        res = mutations.split_scene("Test_Track", "seg_0001", at_time=2.5, clear_right_media=True)
        left = res["left_scene"]
        right = res["right_scene"]
        self.assertEqual(left["start"], 0.0)
        self.assertEqual(left["end"], 2.5)
        self.assertEqual(right["start"], 2.5)
        self.assertEqual(right["end"], 4.0)
        self.assertNotIn("approved_image_path", right)

        # Total scenes is now 3
        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(len(saved["segments"]), 3)

    def test_merge_scenes(self):
        res = mutations.merge_scenes("Test_Track", "seg_0001", with_direction="next")
        merged = res["merged_scene"]
        self.assertEqual(merged["start"], 0.0)
        self.assertEqual(merged["end"], 8.0)
        self.assertIn("Intro beat", merged["notes"])
        self.assertIn("Verse begins", merged["notes"])

        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(len(saved["segments"]), 1)

    # 4. Scene Move & Resize
    def test_move_scene_timing(self):
        res = mutations.move_scene("Test_Track", "seg_0002", start_time=5.0, ripple=False)
        self.assertEqual(res["scene"]["start"], 5.0)
        self.assertEqual(res["scene"]["end"], 9.0)

    def test_resize_scene_timing(self):
        res = mutations.resize_scene("Test_Track", "seg_0001", duration=3.0, ripple=True)
        self.assertEqual(res["scene"]["end"], 3.0)
        # Later scene should ripple
        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["segments"][1]["start"], 3.0)

    # 5. Scene Patch & Latent Dirty Marking
    def test_patch_scene_marks_latent_dirty(self):
        res = mutations.patch_scene("Test_Track", "seg_0001", {"t2i_prompt": "New enhanced prompt"})
        self.assertEqual(res["scene"]["t2i_prompt"], "New enhanced prompt")

    # 6. Timeline Batch: Snap, Close Gaps, Bulk
    def test_timeline_close_gaps(self):
        # Lock scene 1 video so rolling edits don't stretch it, then move scene 2 to create a 2s gap
        mutations.patch_scene("Test_Track", "seg_0001", {"video_path": "video.mp4"})
        mutations.patch_scene("Test_Track", "seg_0002", {"start": 6.0, "end": 10.0})
        res = mutations.timeline_close_gaps("Test_Track")
        self.assertTrue(res["removed_duration"] > 0)
        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["segments"][1]["start"], 4.0)

    def test_timeline_snap(self):
        mutations.patch_scene("Test_Track", "seg_0001", {"end": 4.12})
        res = mutations.timeline_snap("Test_Track", scope="edge", scene_id="seg_0001", edge="end")
        self.assertTrue(res["snapped"])
        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Should snap to nearest beat (4.0)
        self.assertEqual(saved["segments"][0]["end"], 4.0)

    def test_timeline_bulk_durations(self):
        bulk_text = "3.5\n4.5\n2.0\n"
        res = mutations.timeline_bulk("Test_Track", bulk_text, mode="durations", action="replace")
        self.assertEqual(res["scene_count"], 3)
        with open(self.session_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(len(saved["segments"]), 3)
        self.assertEqual(saved["segments"][0]["end"], 3.5)
        self.assertEqual(saved["segments"][1]["end"], 8.0)

    # 7. Transactional Journal Rollback (Section 15.6)
    def test_timeline_journal_rollback(self):
        journal = timeline.TimelineJournal(self.proj_folder, "test_op")
        journal.start()
        # Rename image_0001 -> image_0099
        src = os.path.join(self.images_dir, "image_0001.png")
        dst = os.path.join(self.images_dir, "image_0099.png")
        os.rename(src, dst)
        journal.record_renames([(src, dst)])

        # Trigger rollback
        journal.rollback()
        self.assertTrue(os.path.isfile(src))
        self.assertFalse(os.path.isfile(dst))
        self.assertFalse(os.path.isfile(journal.journal_path))

    # 8. References CRUD (Section 6.5)
    def test_references_crud_and_scene_mapping(self):
        # 1. Upsert subject
        res_subj = mutations.upsert_reference_subject("Test_Track", "subj_001", {
            "name": "Kira",
            "description": "Cyberpunk singer with blue hair",
            "reference_type": "character",
        })
        self.assertEqual(res_subj["subject"]["name"], "Kira")

        # 2. Upsert location
        res_loc = mutations.upsert_reference_location("Test_Track", "loc_001", {
            "name": "Neon Alley",
            "description": "Rainy neon alleyway",
        })
        self.assertEqual(res_loc["location"]["name"], "Neon Alley")

        # 3. Update scene mapping
        res_map = mutations.update_scene_reference_mapping("Test_Track", {
            "subjects": {"seg_0001": ["subj_001"]},
            "locations": {"seg_0001": "loc_001"},
        })
        self.assertEqual(res_map["scene_mapping"]["subjects"]["seg_0001"], ["subj_001"])

        # 4. Get references
        all_refs = mutations.get_project_references("Test_Track")
        self.assertEqual(len(all_refs["subjects"]), 1)
        self.assertEqual(len(all_refs["locations"]), 1)

        # 5. Delete subject and verify cleanup from subject_scene_map
        del_res = mutations.delete_reference("Test_Track", "subjects", "subj_001")
        self.assertTrue(del_res["deleted"])

        all_refs_after = mutations.get_project_references("Test_Track")
        self.assertEqual(len(all_refs_after["subjects"]), 0)
        self.assertEqual(all_refs_after["scene_mapping"]["subjects"]["seg_0001"], [])

    # 9. Lyrics, Audio & Beats
    def test_lyrics_and_beats(self):
        # Set lyrics & srt
        lyrics_res = mutations.set_project_lyrics("Test_Track", lyrics_text="In the neon city rain\nI feel no pain", srt_text="1\n00:00:00,000 --> 00:00:04,000\nIn the neon city rain\n")
        self.assertTrue(lyrics_res["lyrics_saved"])
        self.assertTrue(lyrics_res["srt_saved"])

        read_lyrics = mutations.get_project_lyrics("Test_Track")
        self.assertIn("neon city", read_lyrics["lyrics_text"])
        self.assertTrue(read_lyrics["has_srt"])

        # Calibrate beats (+0.25s)
        cal_res = mutations.calibrate_beats("Test_Track", offset_seconds=0.25)
        self.assertTrue(cal_res["calibrated"])
        beats_res = mutations.get_audio_beats("Test_Track")
        self.assertEqual(beats_res["beats"][0], 1.25)

    # 10. Prompt Context, Assemble, and Validate (Section 18)
    def test_prompt_context_assemble_validate(self):
        ctx = mutations.get_prompt_context("Test_Track", "seg_0001")
        self.assertIn("shot_plan", ctx)
        self.assertIn("budget", ctx)
        self.assertIn("instruction_text", ctx)

        # Assemble prompt with 1 shot
        shots = ["A wide cinematic establishing shot pans across the glowing rainy skyline as sirens echo in the distance."]
        asm = mutations.assemble_minimax_prompt_endpoint("Test_Track", "seg_0001", shots=shots, save=True)
        self.assertTrue(asm["valid"])
        self.assertIn("integrated_multimodal_description:", asm["prompt"])
        self.assertIn("[Shot 1]", asm["prompt"])
        self.assertTrue(asm["saved"])

        # Validate prompt
        val = mutations.validate_minimax_prompt_endpoint("Test_Track", "seg_0001", asm["prompt"])
        self.assertTrue(val["valid"])

        # Set specific prompt field
        set_res = mutations.set_scene_prompt_field_endpoint("Test_Track", "seg_0001", "t2i_prompt", "Solo artist in rain", origin="agent")
        self.assertEqual(set_res["prompt"], "Solo artist in rain")
        self.assertEqual(set_res["origin"], "agent")

    # 11. Project Preflight Settings (Section 16)
    def test_settings_preflight(self):
        pre = mutations.preflight_project_settings("Test_Track")
        self.assertTrue(pre["valid"])
        self.assertIn("effective_settings", pre)

    # 12. Project Validation & Mismatched Asset Numbering (C17 / R10)
    def test_validate_project_mismatched_assets(self):
        # Initial state should be valid (scene 1 has image_0001, scene 2 has image_0002)
        val = mutations.validate_project("Test_Track")
        mismatch_issues = [i for i in val["issues"] if i.get("type") == "mismatched_asset_numbering"]
        self.assertEqual(len(mismatch_issues), 0)

        # Create mismatched files on disk
        with open(os.path.join(self.images_dir, "image_0005.png"), "wb") as f:
            f.write(b"PNG_5")
        with open(os.path.join(self.scene_videos_dir, "video_0001.mp4"), "wb") as f:
            f.write(b"VID_1")

        # Deliberately misalign scene 2's asset to image_0005.png and video_0001.mp4
        with open(self.session_file, "r", encoding="utf-8") as f:
            sess = json.load(f)
        sess["segments"][1]["approved_image_path"] = os.path.join(self.images_dir, "image_0005.png")
        sess["segments"][1]["video_path"] = os.path.join(self.scene_videos_dir, "video_0001.mp4")
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(sess, f)

        val_after = mutations.validate_project("Test_Track")
        self.assertFalse(val_after["valid"])
        mismatches = [i for i in val_after["issues"] if i.get("type") == "mismatched_asset_numbering"]
        self.assertEqual(len(mismatches), 2)
        image_mismatch = next(i for i in mismatches if i["asset_type"] == "image")
        self.assertEqual(image_mismatch["expected_number"], 2)
        self.assertEqual(image_mismatch["actual_number"], 5)
        video_mismatch = next(i for i in mismatches if i["asset_type"] == "video")
        self.assertEqual(video_mismatch["expected_number"], 2)
        self.assertEqual(video_mismatch["actual_number"], 1)


if __name__ == "__main__":
    unittest.main()

