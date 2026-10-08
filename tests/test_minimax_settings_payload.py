"""MiniMax H3 settings model and render payload parity between the Video Builder UI and the Agent API."""

import importlib
import importlib.util
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
payload_mod = importlib.import_module(f"{pkg_name}.minimax.settings_payload")
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
schemas = importlib.import_module(f"{pkg_name}.agent_api.schemas")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")

VIDEO_RENDER = ROOT / "web" / "music_video_builder" / "video_render.mjs"

# Keys the browser sets per scene or per call, not from saved settings.
SCENE_SPECIFIC_KEYS = {
    "project_folder", "scene_number", "latent_exact_frame_path", "audio_path", "prompt", "pass2_prompt",
    "timeline_start_seconds", "timeline_end_seconds", "source_start_seconds", "image_paths",
    "video_references", "last_frame_path", "source_duration_seconds",
}


def _js_payload_keys():
    """Top-level keys of the render payload literal in video_render.mjs, plus the per-pass spread keys."""
    source = VIDEO_RENDER.read_text(encoding="utf-8")
    start = source.index("      const payload = {\n", source.index("build_minimax_h3_prompt") - 12000)
    block = source[start:source.index("\n      };\n", start)]
    keys = set(re.findall(r"^        ([a-z0-9_]+):", block, flags=re.MULTILINE))
    for key in ("te_speed", "feedforward", "block_sparse_attention"):
        for number in (1, 2):
            keys.add(f"pass{number}_use_{key}")
    return keys - SCENE_SPECIFIC_KEYS


def _settings(**overrides):
    return payload_mod.normalize_minimax_h3_settings(overrides)


class DefaultsTests(unittest.TestCase):
    def test_flf_transition_settings_normalize_and_survive_json_reload(self):
        import json

        settings = _settings(i2v_transition_style=" SURREAL_MORPH ",
                             i2v_transition_direction="  petals become stars  ")
        reloaded = _settings(**json.loads(json.dumps(settings)))
        self.assertEqual(reloaded["i2v_transition_style"], "surreal_morph")
        self.assertEqual(reloaded["i2v_transition_direction"], "petals become stars")
        self.assertEqual(_settings(i2v_transition_style="invalid")["i2v_transition_style"], "natural")
        self.assertEqual(payload_mod.SETTINGS_ENUMS["i2v_transition_style"], (
            "natural", "surreal_morph", "dreamlike_dissolve",
            "environment_transformation", "camera_reveal", "custom",
        ))

    def test_defaults_file_matches_the_ui(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        result = subprocess.run(
            [node, str(ROOT / "scripts" / "export_minimax_defaults.mjs"), "--check"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_tiling_is_derived_from_the_output_resolution_not_stored(self):
        defaults = payload_mod.minimax_h3_defaults()
        for key in ("resolution_preset", "megapixels", "advanced_two_pass_vram_preset", "advanced_two_pass_pass1_resolution_preset"):
            self.assertIn(key, defaults)
        for key in ('two_pass_final_width', 'two_pass_final_height', 'advanced_two_pass_pass2_megapixels', 'advanced_two_pass_pass2_resolution_preset', 'advanced_two_pass_tile_width', 'advanced_two_pass_grid_rows', 'advanced_two_pass_chunk_length', 'advanced_two_pass_fade_width', 'advanced_two_pass_overlap_mode', 'advanced_two_pass_brightness_match', 'advanced_two_pass_dynamic_fade', 'advanced_two_pass_upscaler_device'):
            self.assertNotIn(key, defaults)
        self.assertEqual(defaults["resolution_preset"], "1k")
        self.assertEqual(defaults["advanced_two_pass_vram_preset"], "16gb")


class NormalizeAndValidateTests(unittest.TestCase):
    def test_partial_settings_fill_with_defaults_and_ignore_bad_values(self):
        settings = payload_mod.normalize_minimax_h3_settings({
            "steps": 25, "render_pass": "three_pass", "advanced_two_pass_vram_preset": "bogus", "mystery": 1,
            "advanced_two_pass_grid_rows": 99,
        })
        self.assertEqual(settings["steps"], 25)
        self.assertEqual(settings["render_pass"], "three_pass")
        self.assertEqual(settings["advanced_two_pass_vram_preset"], "16gb")  # invalid saved value -> default
        self.assertNotIn("mystery", settings)
        self.assertNotIn("advanced_two_pass_grid_rows", settings)  # retired: ignored

    def test_retired_vram_presets_map_to_24gb(self):
        for saved, expected in (("8gb", "8gb"), ("24gb", "24gb"), ("32gb", "24gb"), ("custom", "24gb")):
            settings = payload_mod.normalize_minimax_h3_settings({"advanced_two_pass_vram_preset": saved})
            self.assertEqual(settings["advanced_two_pass_vram_preset"], expected, saved)

    def test_patch_validation_reports_each_problem(self):
        problems = payload_mod.validate_minimax_h3_patch({
            "advanced_two_pass_vram_preset": "32gb",
            "two_pass_final_width": 1920,
            "advanced_two_pass_grid_rows": 12,
            "resolution_preset": "8k",
            "render_pass": "five_pass",
            "use_loras": "yes",
            "loras": [{"strength": 1}],
            "no_such_setting": 1,
            "advanced_two_pass_pass1_resolution_preset": "1k",
        })
        self.assertEqual(
            set(problems),
            {"advanced_two_pass_vram_preset", "two_pass_final_width", "advanced_two_pass_grid_rows",
             "resolution_preset", "render_pass", "use_loras", "loras", "no_such_setting"},
        )
        self.assertIn("retired", problems["two_pass_final_width"])
        self.assertIn("resolution_preset", problems["two_pass_final_width"])

    def test_valid_patch_passes(self):
        self.assertEqual(payload_mod.validate_minimax_h3_patch({
            "render_pass": "three_pass",
            "resolution_preset": "4k",
            "advanced_two_pass_vram_preset": "24gb",
            "loras": [{"name": "style.safetensors", "strength": 0.8, "apply_to": "pass2"}],
        }), {})

    def test_settings_schema_describes_limits(self):
        full = payload_mod.minimax_h3_settings_schema()
        schema = full["settings"]
        self.assertEqual(schema["resolution_preset"]["enum"], ["custom", "1k", "2k", "1440p", "4k"])
        self.assertEqual(schema["advanced_two_pass_vram_preset"]["enum"], ["8gb", "12gb", "16gb", "24gb"])
        self.assertEqual(schema["megapixels"]["maximum"], 16)
        self.assertIn("two_pass_final_width", full["retired_settings"])
        self.assertNotIn("advanced_two_pass_grid_rows", schema)


class ResolutionMigrationTests(unittest.TestCase):
    """Older saves took the resolution of the pass type they rendered with."""

    def test_new_projects_use_the_default_preset(self):
        settings = payload_mod.normalize_minimax_h3_settings({})
        self.assertEqual((settings["resolution_preset"], settings["megapixels"]), ("1k", 0.5625))

    def test_legacy_saves_migrate_per_render_pass(self):
        single = payload_mod.normalize_minimax_h3_settings({"render_pass": "single", "video_mode": "reference_to_video", "megapixels": 0.9})
        self.assertEqual((single["resolution_preset"], single["megapixels"]), ("custom", 0.9))
        two = payload_mod.normalize_minimax_h3_settings({
            "render_pass": "two_pass", "video_mode": "reference_to_video", "two_pass_final_width": 1920, "two_pass_final_height": 1080,
        })
        self.assertEqual((two["resolution_preset"], two["megapixels"]), ("2k", 1.9922))
        advanced = payload_mod.normalize_minimax_h3_settings({
            "render_pass": "three_pass", "video_mode": "reference_to_video", "advanced_two_pass_pass2_resolution_preset": "4k",
        })
        self.assertEqual((advanced["resolution_preset"], advanced["megapixels"]), ("4k", 7.9688))
        custom = payload_mod.normalize_minimax_h3_settings({
            "render_pass": "three_pass", "video_mode": "reference_to_video",
            "advanced_two_pass_pass2_resolution_preset": "custom", "advanced_two_pass_pass2_megapixels": 3.3,
        })
        self.assertEqual((custom["resolution_preset"], custom["megapixels"]), ("custom", 3.3))

    def test_a_saved_preset_wins_and_follows_the_aspect_ratio(self):
        settings = payload_mod.normalize_minimax_h3_settings({"resolution_preset": "2k", "aspect_ratio": "9:16 (Portrait Widescreen)", "megapixels": 0.3})
        self.assertEqual((settings["resolution_preset"], settings["megapixels"]), ("2k", 1.9922))
        self.assertEqual(payload_mod.normalize_minimax_h3_settings({"resolution_preset": "custom", "megapixels": 1.3})["megapixels"], 1.3)


class SavedSessionCompatibilityTests(unittest.TestCase):
    """Behavior found by checking a real saved project (numbers saved as text, per-pass toggles)."""

    def test_numbers_and_booleans_saved_as_text_are_read_back(self):
        settings = payload_mod.normalize_minimax_h3_settings({
            "two_pass_lora_strength": "1",
            "three_pass_lightx_lora_strength": "0.5",
            "use_loras": "true",
            "steps": "30",
        })
        self.assertEqual(settings["two_pass_lora_strength"], 1.0)
        self.assertEqual(settings["three_pass_lightx_lora_strength"], 0.5)
        self.assertIs(settings["use_loras"], True)
        self.assertEqual(settings["steps"], 30)

    def test_api_patches_stay_strict_about_text_numbers(self):
        self.assertIn("steps", payload_mod.validate_minimax_h3_patch({"steps": "30"}))
        self.assertEqual(payload_mod.validate_minimax_h3_patch({"steps": 30}), {})

    def test_per_pass_toggles_and_profiles_round_trip_and_reach_the_payload(self):
        settings = payload_mod.normalize_minimax_h3_settings({
            "two_pass_use_te_speed": True,
            "pass1_use_te_speed": False,
            "pass2_use_feedforward": True,
            "ref_pass_profiles": {"three_pass": {"render_pass": "three_pass"}},
        })
        self.assertEqual(settings["ref_pass_profiles"], {"three_pass": {"render_pass": "three_pass"}})
        payload = payload_mod.build_minimax_render_payload(settings)
        self.assertIs(payload["pass1_use_te_speed"], False)  # explicit per-pass value wins
        self.assertIs(payload["pass2_use_te_speed"], True)   # unset falls back to two_pass_use_te_speed
        self.assertIs(payload["pass2_use_feedforward"], True)
        self.assertEqual(payload_mod.validate_minimax_h3_patch({"pass1_use_te_speed": True}), {})
        self.assertIn("pass1_use_te_speed", payload_mod.minimax_h3_settings_schema()["settings"])


class SceneSettingsTests(unittest.TestCase):
    def test_scene_overrides_apply_only_when_scene_opts_in(self):
        session = {"minimax_h3_settings": {"steps": 30}}
        segment = {"use_scene_minimax_h3_settings": False, "minimax_h3_settings": {"steps": 10}}
        self.assertEqual(payload_mod.minimax_h3_settings_for_scene(session, segment)["steps"], 30)
        segment["use_scene_minimax_h3_settings"] = True
        self.assertEqual(payload_mod.minimax_h3_settings_for_scene(session, segment)["steps"], 10)


class RenderPayloadTests(unittest.TestCase):
    def test_pass_selection_and_workflow_key(self):
        self.assertEqual(payload_mod.minimax_workflow_key(_settings()), "minimax_h3")
        ref = {"video_mode": "reference_to_video"}
        self.assertEqual(payload_mod.minimax_workflow_key(_settings(render_pass="two_pass", **ref)), "minimax_h3_2pass")
        self.assertEqual(
            payload_mod.minimax_workflow_key(_settings(render_pass="three_pass", **ref)),
            "minimax_h3_advanced_2pass",
        )
        # Multi-pass only applies to reference modes, as in the UI.
        self.assertEqual(
            payload_mod.minimax_workflow_key(_settings(render_pass="three_pass", video_mode="text_to_video")),
            "minimax_h3",
        )

    def test_advanced_payload_uses_the_shared_resolution_and_vram_preset(self):
        settings = _settings(
            render_pass="three_pass", video_mode="reference_to_video",
            resolution_preset="4k", advanced_two_pass_vram_preset="12gb", advanced_two_pass_pass2_steps=3,
        )
        payload = payload_mod.build_minimax_render_payload(settings)
        self.assertEqual(payload["advanced_pass2_megapixels"], 7.9688)
        self.assertEqual(payload["advanced_vram_preset"], "12gb")
        self.assertEqual(payload["pass2_steps"], 3)
        self.assertEqual(payload["video_mode"], "reference_to_video")
        # Tiles, chunks, fades and the upscaler device are planned by the graph, not sent as settings.
        for key in ("advanced_tile_size_mode", "advanced_grid_rows", "advanced_chunk_length", "advanced_fade_width",
                    "advanced_overlap_mode", "advanced_brightness_match", "advanced_upscaler_device"):
            self.assertNotIn(key, payload)

    def test_retired_vram_preset_still_renders_as_24gb(self):
        payload = payload_mod.build_minimax_render_payload(
            _settings(render_pass="three_pass", video_mode="reference_to_video", advanced_two_pass_vram_preset="32gb")
        )
        self.assertEqual(payload["advanced_vram_preset"], "24gb")

    def test_two_pass_uses_two_pass_values_and_the_shared_final_size(self):
        payload = payload_mod.build_minimax_render_payload(
            _settings(render_pass="two_pass", video_mode="reference_to_video", two_pass_pass2_steps=6, resolution_preset="2k")
        )
        self.assertEqual(payload["pass2_steps"], 6)
        self.assertEqual((payload["final_width"], payload["final_height"]), (1920, 1088))
        self.assertIn("latent_upscaler_name", payload)
        portrait = payload_mod.build_minimax_render_payload(
            _settings(render_pass="two_pass", video_mode="reference_to_video", resolution_preset="2k", aspect_ratio="9:16 (Portrait Widescreen)")
        )
        self.assertEqual((portrait["final_width"], portrait["final_height"]), (1088, 1920))

    def test_one_resolution_drives_every_pass_type(self):
        for render_pass in ("single", "two_pass", "three_pass"):
            payload = payload_mod.build_minimax_render_payload(
                _settings(render_pass=render_pass, video_mode="reference_to_video", resolution_preset="1k")
            )
            self.assertEqual(payload["megapixels"], 0.5625, render_pass)
        advanced = payload_mod.build_minimax_render_payload(
            _settings(render_pass="three_pass", video_mode="reference_to_video", resolution_preset="1k")
        )
        self.assertEqual(advanced["advanced_pass2_megapixels"], 0.5625)

    def test_single_pass_has_no_multipass_only_keys(self):
        payload = payload_mod.build_minimax_render_payload(_settings())
        self.assertNotIn("final_width", payload)
        self.assertNotIn("latent_upscaler_name", payload)
        self.assertIn("use_te_speed", payload)

    def test_multipass_rejects_built_in_audio(self):
        with self.assertRaises(ValueError):
            payload_mod.build_minimax_render_payload(
                _settings(render_pass="three_pass", video_mode="reference_to_video", audio_mode="built_in_audio")
            )

    def test_overrides_win(self):
        payload = payload_mod.build_minimax_render_payload(_settings(), {"seed": 123})
        self.assertEqual(payload["seed"], 123)

    def test_payload_covers_every_key_the_browser_sends(self):
        """Fails when video_render.mjs gains a payload key the Python builder does not emit."""
        js_keys = _js_payload_keys()
        self.assertGreater(len(js_keys), 80)
        emitted = set()
        for render_pass in ("single", "two_pass", "three_pass"):
            emitted |= set(payload_mod.build_minimax_render_payload(
                _settings(render_pass=render_pass, video_mode="reference_to_video")
            ))
        self.assertEqual(sorted(js_keys - emitted), [], "Add these browser payload keys to build_minimax_render_payload.")


class PatchSettingsMappingTests(unittest.TestCase):
    def _patch(self, session, patch_body):
        saved = {}

        def fake_persist(folder, sess):
            saved["session"] = sess
            return {"session": sess, "revision": 8}

        with patch.object(mutations, "_get_active_session_and_folder", return_value=("C:/p", session)), \
                patch.object(mutations, "_persist_session", side_effect=fake_persist):
            result = mutations.patch_project_settings("proj", patch_body)
        return result, saved["session"]

    def test_minimax_group_is_saved_where_the_ui_reads_it(self):
        session = {"minimax_h3_settings": {"steps": 30}}
        result, saved = self._patch(session, {"minimax_h3": {"render_pass": "three_pass", "advanced_two_pass_vram_preset": "12gb"}})
        self.assertEqual(saved["minimax_h3_settings"]["steps"], 30)
        self.assertEqual(saved["minimax_h3_settings"]["render_pass"], "three_pass")
        self.assertNotIn("minimax_h3", saved)
        self.assertEqual(result["settings"]["minimax_h3"]["advanced_two_pass_vram_preset"], "12gb")
        self.assertEqual(result["settings"]["minimax_h3"]["steps"], 30)
        self.assertEqual(result["revision"], 8)

    def test_flat_groups_update_top_level_session_keys(self):
        _, saved = self._patch({}, {"project": {"video_engine": "ltx"}, "post_process": {"lut_enabled": True}})
        self.assertEqual(saved["video_engine"], "ltx")
        self.assertTrue(saved["lut_enabled"])

    def test_invalid_minimax_setting_is_rejected(self):
        with self.assertRaises(errors.SettingsInvalidError):
            schemas.validate_settings_patch({"minimax_h3": {"advanced_two_pass_grid_rows": 12, "bogus": 1}})


if __name__ == "__main__":
    unittest.main()
