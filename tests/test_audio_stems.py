import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_audio_stems():
    """Import builder/audio_stems.py without starting ComfyUI: the package gets a stand-in root and folder_paths is stubbed."""
    stubbed = "folder_paths" not in sys.modules
    if stubbed:
        sys.modules["folder_paths"] = types.SimpleNamespace(
            get_output_directory=lambda: tempfile.gettempdir(), get_input_directory=lambda: tempfile.gettempdir(),
            get_temp_directory=lambda: tempfile.gettempdir(),
        )
    package = types.ModuleType("vrgdg_stems_pkg")
    package.__path__ = [str(ROOT)]
    sys.modules["vrgdg_stems_pkg"] = package
    try:
        from vrgdg_stems_pkg.builder import audio_stems
    finally:
        # Other tests must not see the stand-in.
        if stubbed:
            del sys.modules["folder_paths"]

    return audio_stems


stems_module = load_audio_stems()
RATE = stems_module.SAMPLE_RATE


def tone(seconds, level=0.5):
    samples = int(round(seconds * RATE))
    return np.full((2, samples), level, dtype=np.float32)


class RegionTests(unittest.TestCase):
    def test_regions_are_clamped_sorted_and_short_ones_dropped(self):
        regions = stems_module.normalize_regions(
            [{"start": 3.0, "end": 9.0}, {"start": -1.0, "end": 1.0, "fade_ms": 5}, {"start": 2.0, "end": 2.001}, "x", {"start": "a", "end": 1}],
            4.0,
        )
        self.assertEqual([(r["start"], r["end"]) for r in regions], [(0.0, 1.0), (3.0, 4.0)])
        self.assertEqual(regions[0]["fade_ms"], 5.0)
        self.assertEqual(regions[1]["fade_ms"], stems_module.DEFAULT_FADE_MS)

    def test_millisecond_times_are_kept(self):
        regions = stems_module.normalize_regions([{"start": 2.343, "end": 4.56}], 4.56)
        self.assertEqual((regions[0]["start"], regions[0]["end"]), (2.343, 4.56))


class EnvelopeTests(unittest.TestCase):
    def test_no_regions_mutes_everything(self):
        self.assertEqual(float(stems_module.keep_envelope([], 1000).max()), 0.0)

    def test_region_is_kept_and_outside_is_muted(self):
        envelope = stems_module.keep_envelope([{"start": 1.0, "end": 2.0, "fade_ms": 0.0}], int(3 * RATE))
        self.assertEqual(float(envelope[: int(0.99 * RATE)].max()), 0.0)
        self.assertEqual(float(envelope[int(1.01 * RATE): int(1.99 * RATE)].min()), 1.0)
        self.assertEqual(float(envelope[int(2.01 * RATE):].max()), 0.0)

    def test_fade_ramps_inside_the_region_edges(self):
        envelope = stems_module.keep_envelope([{"start": 1.0, "end": 2.0, "fade_ms": 100.0}], int(3 * RATE))
        first = int(1.0 * RATE)
        self.assertEqual(float(envelope[first]), 0.0)
        self.assertTrue(0.0 < float(envelope[first + int(0.05 * RATE)]) < 1.0)
        self.assertEqual(float(envelope[first + int(0.2 * RATE)]), 1.0)
        self.assertEqual(float(envelope[int(2.0 * RATE) - 1]), 0.0)

    def test_overlapping_regions_do_not_add_up(self):
        regions = [{"start": 0.0, "end": 1.0, "fade_ms": 0.0}, {"start": 0.5, "end": 1.5, "fade_ms": 0.0}]
        self.assertEqual(float(stems_module.keep_envelope(regions, int(2 * RATE)).max()), 1.0)


class MixTests(unittest.TestCase):
    def stems(self):
        return {"vocals": tone(2.0, 0.4), "drums": tone(2.0, 0.1), "bass": tone(2.0, 0.1), "other": tone(2.0, 0.1)}

    def settings(self, **changes):
        base = stems_module.normalize_stem_settings({}, stems_module.FOUR_STEMS, 2.0)
        for name, values in changes.items():
            base[name].update(values)
        return base

    def level(self, mix, seconds):
        return float(mix[0, int(seconds * RATE)])

    def test_stems_with_no_settings_play_in_full(self):
        mix = stems_module.mix_stems(self.stems(), self.settings())
        self.assertAlmostEqual(self.level(mix, 0.5), 0.7, places=5)

    def test_a_masked_stem_is_only_audible_inside_its_regions(self):
        regions = [{"start": 1.0, "end": 2.0, "fade_ms": 0.0}]
        mix = stems_module.mix_stems(self.stems(), self.settings(vocals={"mask": True, "regions": regions}))
        self.assertAlmostEqual(self.level(mix, 0.5), 0.3, places=5)
        self.assertAlmostEqual(self.level(mix, 1.5), 0.7, places=5)

    def test_every_stem_has_its_own_regions(self):
        settings = self.settings(
            vocals={"mask": True, "regions": [{"start": 1.0, "end": 2.0, "fade_ms": 0.0}]},
            drums={"mask": True, "regions": [{"start": 0.0, "end": 1.0, "fade_ms": 0.0}]},
        )
        mix = stems_module.mix_stems(self.stems(), settings)
        self.assertAlmostEqual(self.level(mix, 0.5), 0.3, places=5)  # drums, bass, other
        self.assertAlmostEqual(self.level(mix, 1.5), 0.4 + 0.2, places=5)  # vocals, bass, other

    def test_a_masked_stem_with_no_regions_is_silent(self):
        mix = stems_module.mix_stems(self.stems(), self.settings(vocals={"mask": True, "regions": []}))
        self.assertAlmostEqual(self.level(mix, 1.0), 0.3, places=5)

    def test_levels_are_in_decibels_and_mute_removes_a_stem(self):
        mix = stems_module.mix_stems(self.stems(), self.settings(vocals={"db": -6.0206}, drums={"mute": True}, bass={"db": -100.0}))
        self.assertAlmostEqual(self.level(mix, 1.0), 0.2 + 0.1, places=3)

    def test_loud_mix_is_scaled_under_full_scale(self):
        stems = {name: tone(1.0, 0.9) for name in stems_module.FOUR_STEMS}
        mix = stems_module.mix_stems(stems, stems_module.normalize_stem_settings({}, stems_module.FOUR_STEMS, 1.0))
        self.assertLessEqual(float(np.abs(mix).max()), 0.98 + 1e-6)

    def test_settings_are_cleaned_up_per_stem(self):
        settings = stems_module.normalize_stem_settings(
            {"vocals": {"mask": True, "regions": [{"start": 3.0, "end": 9.0}], "db": 99, "mute": 1}, "drums": "junk"},
            stems_module.SIX_STEMS, 4.0,
        )
        self.assertEqual(list(settings), list(stems_module.SIX_STEMS))
        self.assertEqual(settings["vocals"]["regions"][0]["end"], 4.0)
        self.assertEqual((settings["vocals"]["db"], settings["vocals"]["mute"], settings["vocals"]["mask"]), (24.0, True, True))
        self.assertEqual(settings["drums"], {"mask": False, "regions": [], "db": 0.0, "mute": False})
        self.assertEqual(settings["guitar"]["mask"], False)

    def test_scenes_split_before_six_stems_existed_have_the_four_basic_ones(self):
        self.assertEqual(stems_module._stem_names({}), list(stems_module.FOUR_STEMS))
        self.assertEqual(stems_module._stem_names({"separation": {"stem_names": ["vocals", "guitar", "x"]}}), ["vocals", "guitar"])
        self.assertEqual(stems_module.STEMS_BY_MODEL["htdemucs_6s"], stems_module.SIX_STEMS)


class FileTests(unittest.TestCase):
    def test_wav_round_trip_is_exact_length_and_peaks_have_the_requested_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "a.wav")
            audio = tone(1.5, 0.25)
            stems_module._write_wav(path, audio)
            loaded = stems_module._read_wav(path)
            self.assertEqual(loaded.shape, audio.shape)
            self.assertAlmostEqual(float(loaded[0, 10]), 0.25, places=3)
            self.assertEqual(len(stems_module.peaks(loaded, 100)), 100)

    def test_fit_cuts_and_pads_to_the_exact_sample_count(self):
        self.assertEqual(stems_module._fit(tone(2.0), 1000).shape[1], 1000)
        padded = stems_module._fit(tone(0.01), 2000)
        self.assertEqual(padded.shape[1], 2000)
        self.assertEqual(float(padded[0, -1]), 0.0)

    def test_an_empty_project_folder_never_means_the_current_folder(self):
        for empty in ("", "  ", None):
            with self.assertRaises(ValueError):
                stems_module.scene_folder(empty, "seg_a")

    def test_scene_folder_stays_inside_the_project(self):
        with tempfile.TemporaryDirectory() as project:
            inside = stems_module.scene_folder(project, "seg_abc123", create=True)
            self.assertTrue(os.path.isdir(inside))
            self.assertEqual(os.path.commonpath([os.path.abspath(project), inside]), os.path.abspath(project))
            for bad in ("", "..", "../x", "a/b", "a\\b", "x" * 200):
                with self.assertRaises(ValueError):
                    stems_module.scene_folder(project, bad)
            with self.assertRaises(ValueError):
                stems_module.file_path(project, "seg_abc123", "../../secret")


class RenderOverrideTests(unittest.TestCase):
    def build_scene(self, project, seconds):
        folder = stems_module.scene_folder(project, "seg_1", create=True)
        stems_module._write_wav(os.path.join(folder, "masked_mix.wav"), tone(seconds))
        (Path(folder) / stems_module.META_NAME).write_text(json.dumps({"mix": {"duration_seconds": seconds}}), encoding="utf-8")

    def test_a_scene_without_a_mask_renders_normally(self):
        with tempfile.TemporaryDirectory() as project:
            self.assertIsNone(stems_module.audio_override_for_scene({"id": "seg_1"}, project, 4.0))
            self.assertIsNone(stems_module.audio_override_for_scene({"id": "seg_1", "audio_mask": {"enabled": False}}, project, 4.0))

    def test_an_enabled_mask_uses_the_built_mix(self):
        with tempfile.TemporaryDirectory() as project:
            self.build_scene(project, 4.0)
            override = stems_module.audio_override_for_scene({"id": "seg_1", "audio_mask": {"enabled": True}}, project, 4.0)
            self.assertTrue(override["path"].endswith("masked_mix.wav"))
            self.assertEqual((override["start_seconds"], override["duration_seconds"]), (0.0, 4.0))

    def test_an_enabled_mask_without_a_mix_or_with_a_different_length_is_refused(self):
        with tempfile.TemporaryDirectory() as project:
            scene = {"id": "seg_1", "audio_mask": {"enabled": True}}
            with self.assertRaises(ValueError):
                stems_module.audio_override_for_scene(scene, project, 4.0)
            self.build_scene(project, 4.0)
            with self.assertRaises(ValueError):
                stems_module.audio_override_for_scene(scene, project, 5.0)


class SceneListTests(unittest.TestCase):
    def test_only_scenes_with_saved_stems_are_listed(self):
        with tempfile.TemporaryDirectory() as project:
            self.assertEqual(stems_module.list_scene_stems({"project_folder": project})["scene_ids"], [])
            for scene_id, has_meta in (("seg_b", True), ("seg_a", True), ("seg_empty", False)):
                folder = stems_module.scene_folder(project, scene_id, create=True)
                if has_meta:
                    (Path(folder) / stems_module.META_NAME).write_text("{}", encoding="utf-8")
            self.assertEqual(stems_module.list_scene_stems({"project_folder": project})["scene_ids"], ["seg_a", "seg_b"])

    def test_a_missing_project_folder_is_refused(self):
        with self.assertRaises(ValueError):
            stems_module.list_scene_stems({"project_folder": ""})

    def test_deleting_a_scene_removes_its_folder(self):
        with tempfile.TemporaryDirectory() as project:
            folder = stems_module.scene_folder(project, "seg_a", create=True)
            (Path(folder) / stems_module.META_NAME).write_text("{}", encoding="utf-8")
            stems_module._write_wav(os.path.join(folder, "vocals.wav"), tone(0.1))
            self.assertEqual(stems_module.delete_scene_stems({"project_folder": project, "scene_id": "seg_a"})["removed"], 2)
            self.assertFalse(os.path.isdir(folder))
            self.assertEqual(stems_module.list_scene_stems({"project_folder": project})["scene_ids"], [])


class SceneStatesTests(unittest.TestCase):
    def test_many_scenes_come_back_in_one_call_and_a_bad_one_does_not_break_the_rest(self):
        with tempfile.TemporaryDirectory() as project:
            folder = stems_module.scene_folder(project, "seg_a", create=True)
            for name in stems_module.FOUR_STEMS:
                stems_module._write_wav(os.path.join(folder, f"{name}.wav"), tone(0.5, 0.2))
            (Path(folder) / stems_module.META_NAME).write_text(json.dumps({"separation": {"stem_names": list(stems_module.FOUR_STEMS)}}), encoding="utf-8")
            result = stems_module.scene_states({"project_folder": project, "scene_ids": ["seg_a", "seg_none", "../bad"]})
            self.assertTrue(result["states"]["seg_a"]["exists"])
            self.assertFalse(result["states"]["seg_none"]["exists"])
            self.assertFalse(result["states"]["../bad"]["exists"])

    def test_scene_ids_must_be_a_list_and_the_batch_is_capped(self):
        with tempfile.TemporaryDirectory() as project:
            with self.assertRaises(ValueError):
                stems_module.scene_states({"project_folder": project, "scene_ids": "seg_a"})
            ids = [f"seg_{index}" for index in range(stems_module.MAX_BATCH_SCENES + 20)]
            self.assertEqual(len(stems_module.scene_states({"project_folder": project, "scene_ids": ids})["states"]), stems_module.MAX_BATCH_SCENES)


class PeaksCacheTests(unittest.TestCase):
    def make_scene(self, project):
        folder = stems_module.scene_folder(project, "seg_a", create=True)
        for name in stems_module.FOUR_STEMS:
            stems_module._write_wav(os.path.join(folder, f"{name}.wav"), tone(0.5, 0.2))
        stems_module._write_wav(os.path.join(folder, "original.wav"), tone(0.5, 0.2))
        (Path(folder) / stems_module.META_NAME).write_text(json.dumps({"separation": {"stem_names": list(stems_module.FOUR_STEMS)}}), encoding="utf-8")
        return folder

    def test_peaks_are_saved_and_reused_without_reading_the_wavs_again(self):
        with tempfile.TemporaryDirectory() as project:
            folder = self.make_scene(project)
            first = stems_module.scene_state({"project_folder": project, "scene_id": "seg_a"})
            self.assertTrue(os.path.isfile(os.path.join(folder, stems_module.PEAKS_NAME)))
            calls = []
            original_read = stems_module._read_wav
            stems_module._read_wav = lambda path: calls.append(path) or original_read(path)
            try:
                second = stems_module.scene_state({"project_folder": project, "scene_id": "seg_a"})
            finally:
                stems_module._read_wav = original_read
            self.assertEqual(calls, [])
            self.assertEqual(first["peaks"], second["peaks"])
            self.assertEqual(first["duration"], 0.5)

    def test_a_changed_wav_gets_new_peaks(self):
        with tempfile.TemporaryDirectory() as project:
            folder = self.make_scene(project)
            before = stems_module.scene_state({"project_folder": project, "scene_id": "seg_a"})["peaks"]["vocals"]
            stems_module._write_wav(os.path.join(folder, "vocals.wav"), tone(0.5, 0.9))
            after = stems_module.scene_state({"project_folder": project, "scene_id": "seg_a"})["peaks"]["vocals"]
            self.assertNotEqual(before, after)
            self.assertAlmostEqual(max(after), 0.9, places=2)

    def test_the_peaks_file_is_removed_with_the_scene(self):
        with tempfile.TemporaryDirectory() as project:
            folder = self.make_scene(project)
            stems_module.scene_state({"project_folder": project, "scene_id": "seg_a"})
            stems_module.delete_scene_stems({"project_folder": project, "scene_id": "seg_a"})
            self.assertFalse(os.path.isdir(folder))


class MaskedMixTimingTests(unittest.TestCase):
    def timing(self):
        from vrgdg_stems_pkg.minimax.latent_manager import calculate_minimax_h3_timing

        return calculate_minimax_h3_timing

    def test_a_scene_length_mix_fits_a_scene_whose_boundary_has_float_noise(self):
        # Scene 17 of a real project: 79.96000000000001 to 84.52, rendered from a 4.56 s masked mix that starts at 0.
        plan = self.timing()(79.96000000000001, 84.52, 0, 0, source_start_seconds=0, source_duration_seconds=84.52 - 79.96000000000001)
        self.assertEqual(plan.final_trim_duration_seconds, 4.56)

    def test_audio_that_really_is_too_short_is_still_refused(self):
        with self.assertRaises(ValueError):
            self.timing()(79.96, 84.52, 0, 0, source_start_seconds=0, source_duration_seconds=4.4)


if __name__ == "__main__":
    unittest.main()
