"""Tests for the named MiniMax H3 video profiles shared by every project."""

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
profiles = importlib.import_module(f"{pkg_name}.builder.video_profiles")
payload_mod = importlib.import_module(f"{pkg_name}.minimax.settings_payload")


class VideoProfileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.join(self._tmp.name, "VRGDG_Video_Profiles", "minimax_h3")
        patcher = patch.object(profiles, "profile_root", return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _settings(self, **overrides):
        base = {
            "video_mode": "reference_to_video", "render_pass": "three_pass", "resolution_preset": "1k",
            "advanced_two_pass_vram_preset": "12gb", "steps": 25, "pass1_use_te_speed": True,
            "pass2_use_block_sparse_attention": True,
            # Not part of a profile:
            "audio_mode": "input_audio", "continuity_mode": "latent_continuation",
            "continuity_prompt_from_last_frame": True, "latent_context_frames": 39,
            "location_transition_preset": "surreal", "location_transition_custom": "x",
            "ref_pass_profiles": {"single": {"steps": 3}}, "two_pass_defaults_version": 1,
            "advanced_two_pass_defaults_version": 4,
        }
        base.update(overrides)
        return base

    def test_save_list_load_round_trip_keeps_the_video_selection_and_its_settings(self):
        saved = profiles.save_video_profile("My 2 Pass Advanced", self._settings())
        self.assertEqual(saved["name"], "My 2 Pass Advanced")
        listing = profiles.list_video_profiles()
        self.assertEqual([item["name"] for item in listing], ["My 2 Pass Advanced"])
        self.assertEqual((listing[0]["video_mode"], listing[0]["render_pass"]), ("reference_to_video", "three_pass"))
        loaded = profiles.load_video_profile("my 2 pass advanced")  # names are not case sensitive
        settings = loaded["settings"]
        self.assertEqual(settings["render_pass"], "three_pass")
        self.assertEqual(settings["video_mode"], "reference_to_video")
        self.assertEqual(settings["resolution_preset"], "1k")
        self.assertEqual(settings["advanced_two_pass_vram_preset"], "12gb")
        self.assertEqual(settings["steps"], 25)
        self.assertIs(settings["pass1_use_te_speed"], True)
        self.assertIs(settings["pass2_use_block_sparse_attention"], True)

    def test_audio_continuity_and_internal_keys_are_never_saved(self):
        profiles.save_video_profile("P", self._settings())
        with open(os.path.join(self.root, "p.json"), "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)["settings"]
        for key in profiles.EXCLUDED_PROFILE_KEYS:
            self.assertNotIn(key, on_disk, key)
        # Everything else the Video Builder keeps for the video selection is there.
        for key in ("diffusion_model_name", "clip_name", "aspect_ratio", "megapixels", "sampler_name",
                    "two_pass_lora_name", "use_loras", "advanced_two_pass_pass1_sampler"):
            self.assertIn(key, on_disk, key)

    def test_loading_filters_again_so_an_edited_file_cannot_carry_excluded_keys(self):
        profiles.save_video_profile("P", self._settings())
        path = os.path.join(self.root, "p.json")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["settings"]["audio_mode"] = "built_in_audio"
        data["settings"]["continuity_mode"] = "exact_start_frame"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        loaded = profiles.load_video_profile("P")["settings"]
        self.assertNotIn("audio_mode", loaded)
        self.assertNotIn("continuity_mode", loaded)

    def test_saving_over_an_existing_name_needs_overwrite(self):
        profiles.save_video_profile("Cinema", self._settings(steps=10))
        with self.assertRaises(profiles.ProfileExistsError):
            profiles.save_video_profile("cinema", self._settings(steps=11))
        self.assertEqual(profiles.load_video_profile("Cinema")["settings"]["steps"], 10)
        profiles.save_video_profile("cinema", self._settings(steps=11), overwrite=True)
        self.assertEqual(profiles.load_video_profile("Cinema")["settings"]["steps"], 11)
        self.assertEqual(len(profiles.list_video_profiles()), 1)

    def test_names_that_map_to_the_same_file_are_refused(self):
        profiles.save_video_profile("A/B", self._settings())
        with self.assertRaises(ValueError) as caught:
            profiles.save_video_profile("A_B", self._settings(), overwrite=True)
        self.assertIn("too similar", str(caught.exception))

    def test_bad_names_and_empty_settings_are_rejected(self):
        for name in ("", "   ", "x" * 61):
            with self.assertRaises(ValueError):
                profiles.save_video_profile(name, self._settings())
        with self.assertRaises(ValueError):
            profiles.save_video_profile("Empty", {})
        self.assertEqual(profiles.list_video_profiles(), [])

    def test_delete_removes_the_profile_and_missing_ones_report_not_found(self):
        profiles.save_video_profile("Keep", self._settings())
        profiles.save_video_profile("Drop", self._settings())
        self.assertEqual(profiles.delete_video_profile("drop"), {"name": "Drop"})
        self.assertEqual([item["name"] for item in profiles.list_video_profiles()], ["Keep"])
        with self.assertRaises(FileNotFoundError):
            profiles.delete_video_profile("Drop")
        with self.assertRaises(FileNotFoundError):
            profiles.load_video_profile("Drop")

    def test_unreadable_files_are_skipped_in_the_list(self):
        profiles.save_video_profile("Good", self._settings())
        with open(os.path.join(self.root, "broken.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with open(os.path.join(self.root, "other.json"), "w", encoding="utf-8") as handle:
            json.dump({"name": "", "settings": {}}, handle)
        self.assertEqual([item["name"] for item in profiles.list_video_profiles()], ["Good"])

    def test_list_is_sorted_by_name_ignoring_case(self):
        for name in ("zeta", "Alpha", "beta"):
            profiles.save_video_profile(name, self._settings())
        self.assertEqual([item["name"] for item in profiles.list_video_profiles()], ["Alpha", "beta", "zeta"])

    def test_listing_with_no_profile_folder_is_empty(self):
        self.assertEqual(profiles.list_video_profiles(), [])


class PipelineIsNotPartOfAProfileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(profiles, "profile_root", return_value=os.path.join(self._tmp.name, "profiles"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_neither_saving_nor_loading_carries_the_pipeline(self):
        saved = profiles.save_video_profile("From a RefMod project", {
            "pipeline": "refmod", "video_mode": "reference_to_video", "render_pass": "two_pass",
        })
        self.assertNotIn("pipeline", saved["settings"])
        self.assertNotIn("pipeline", profiles.load_video_profile("From a RefMod project")["settings"])

    def test_a_profile_file_saved_before_this_rule_cannot_bring_the_pipeline_back(self):
        # This is what the Builder used to receive: the load filled in the default pipeline, "standard", which
        # replaced a RefMod project's pipeline and hid the RefMod pickers.
        profiles.save_video_profile("Old", {"video_mode": "reference_to_video", "render_pass": "two_pass"})
        path = os.path.join(profiles.profile_root(), "old.json")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["settings"]["pipeline"] = "standard"
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        self.assertNotIn("pipeline", profiles.load_video_profile("Old")["settings"])


class ExclusionListTests(unittest.TestCase):
    def test_excluded_keys_are_real_settings(self):
        known = set(payload_mod.minimax_h3_defaults()) | set(payload_mod._OPTIONAL_SETTINGS)
        legacy_only = {"ref_pass_mode"}  # old saves only; not in the current defaults
        for key in profiles.EXCLUDED_PROFILE_KEYS - legacy_only:
            self.assertIn(key, known, f"{key} is not a MiniMax H3 setting (typo?)")

    def test_the_sections_the_ui_calls_audio_and_between_scene_continuity_are_excluded(self):
        for key in ("audio_mode", "continuity_mode", "latent_context_frames", "continuity_prompt_from_last_frame",
                    "location_transition_preset", "location_transition_custom"):
            self.assertIn(key, profiles.EXCLUDED_PROFILE_KEYS)

    def test_the_project_pipeline_is_excluded_so_a_profile_cannot_switch_refmod_back_to_standard(self):
        self.assertIn("pipeline", profiles.EXCLUDED_PROFILE_KEYS)

    def test_video_selection_settings_are_not_excluded(self):
        for key in ("video_mode", "render_pass", "diffusion_model_name", "resolution_preset", "megapixels",
                    "advanced_two_pass_vram_preset", "pass1_use_te_speed", "use_loras", "loras", "seed"):
            self.assertNotIn(key, profiles.EXCLUDED_PROFILE_KEYS)


if __name__ == "__main__":
    unittest.main()
