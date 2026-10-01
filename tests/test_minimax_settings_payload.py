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
    def test_defaults_file_matches_the_ui(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        result = subprocess.run(
            [node, str(ROOT / "scripts" / "export_minimax_defaults.mjs"), "--check"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_new_seam_settings_are_available(self):
        defaults = payload_mod.minimax_h3_defaults()
        for key in (
            "advanced_two_pass_brightness_match", "advanced_two_pass_dynamic_fade",
            "advanced_two_pass_dynamic_fade_min", "advanced_two_pass_masked_area_noise",
            "advanced_two_pass_pass1_resolution_preset", "advanced_two_pass_pass2_resolution_preset",
            "advanced_two_pass_vram_preset", "advanced_two_pass_grid_rows",
        ):
            self.assertIn(key, defaults)


class NormalizeAndValidateTests(unittest.TestCase):
    def test_partial_settings_fill_with_defaults_and_ignore_bad_values(self):
        settings = payload_mod.normalize_minimax_h3_settings({
            "steps": 25, "render_pass": "three_pass", "advanced_two_pass_grid_rows": 99, "mystery": 1,
        })
        self.assertEqual(settings["steps"], 25)
        self.assertEqual(settings["render_pass"], "three_pass")
        self.assertEqual(settings["advanced_two_pass_grid_rows"], 2)  # invalid saved value -> default
        self.assertNotIn("mystery", settings)

    def test_patch_validation_reports_each_problem(self):
        problems = payload_mod.validate_minimax_h3_patch({
            "advanced_two_pass_grid_rows": 12,
            "advanced_two_pass_fade_width": 50,
            "advanced_two_pass_chunk_length": 100,
            "render_pass": "five_pass",
            "use_loras": "yes",
            "loras": [{"strength": 1}],
            "no_such_setting": 1,
            "advanced_two_pass_dynamic_fade": "widening",
        })
        self.assertEqual(
            set(problems),
            {"advanced_two_pass_grid_rows", "advanced_two_pass_fade_width", "advanced_two_pass_chunk_length",
             "render_pass", "use_loras", "loras", "no_such_setting"},
        )

    def test_valid_patch_passes(self):
        self.assertEqual(payload_mod.validate_minimax_h3_patch({
            "render_pass": "three_pass",
            "advanced_two_pass_dynamic_fade": "narrowing",
            "advanced_two_pass_spatial_w_overlap": 192,
            "advanced_two_pass_chunk_length": 170,
            "advanced_two_pass_masked_area_noise": 0.0,
            "loras": [{"name": "style.safetensors", "strength": 0.8, "apply_to": "pass2"}],
        }), {})

    def test_settings_schema_describes_limits(self):
        schema = payload_mod.minimax_h3_settings_schema()["settings"]
        self.assertEqual(schema["advanced_two_pass_dynamic_fade"]["enum"], ["off", "narrowing", "widening"])
        self.assertEqual(schema["advanced_two_pass_grid_cols"]["maximum"], 9)
        self.assertEqual(schema["advanced_two_pass_fade_width"]["multipleOf"], 32)


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

    def test_advanced_payload_carries_seam_and_tile_settings(self):
        settings = _settings(
            render_pass="three_pass", video_mode="reference_to_video",
            advanced_two_pass_grid_rows=2, advanced_two_pass_grid_cols=3,
            advanced_two_pass_spatial_w_overlap=192, advanced_two_pass_dynamic_fade="narrowing",
            advanced_two_pass_pass2_megapixels=7.97, advanced_two_pass_pass2_steps=3,
        )
        payload = payload_mod.build_minimax_render_payload(settings)
        self.assertEqual(payload["advanced_grid_rows"], 2)
        self.assertEqual(payload["advanced_grid_cols"], 3)
        self.assertEqual(payload["advanced_spatial_w_overlap"], 192)
        self.assertEqual(payload["advanced_dynamic_fade"], "narrowing")
        self.assertTrue(payload["advanced_brightness_match"])
        self.assertEqual(payload["advanced_pass2_megapixels"], 7.97)
        self.assertEqual(payload["pass2_steps"], 3)
        self.assertEqual(payload["video_mode"], "reference_to_video")

    def test_two_pass_uses_two_pass_values_and_final_size(self):
        payload = payload_mod.build_minimax_render_payload(
            _settings(render_pass="two_pass", video_mode="reference_to_video", two_pass_pass2_steps=6)
        )
        self.assertEqual(payload["pass2_steps"], 6)
        self.assertIn("final_width", payload)
        self.assertIn("latent_upscaler_name", payload)

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
        result, saved = self._patch(session, {"minimax_h3": {"render_pass": "three_pass", "advanced_two_pass_dynamic_fade": "narrowing"}})
        self.assertEqual(saved["minimax_h3_settings"]["steps"], 30)
        self.assertEqual(saved["minimax_h3_settings"]["render_pass"], "three_pass")
        self.assertNotIn("minimax_h3", saved)
        self.assertEqual(result["settings"]["minimax_h3"]["advanced_two_pass_dynamic_fade"], "narrowing")
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
