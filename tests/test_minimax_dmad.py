"""DMAD sampling and the sigma shift: settings, the Python twin of the UI preset, and the compiled graphs."""

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

payload_mod = importlib.import_module(f"{ROOT.name}.minimax.settings_payload")
workflows = importlib.import_module(f"{ROOT.name}.runner.minimax_workflows")
patches = importlib.import_module(f"{ROOT.name}.runner.minimax_patches")
models = importlib.import_module(f"{ROOT.name}.runner.models")

DMAD_LORA = payload_mod.DMAD_LORA_NAME
REFINER = payload_mod.DMAD_REFINER_LORA_NAME
MINIMAX_H3_MJS = ROOT / "web" / "music_video_builder" / "minimax_h3.mjs"


def _settings(**overrides):
    return payload_mod.normalize_minimax_h3_settings(overrides)


class SigmaSettingsTests(unittest.TestCase):
    def test_sigma_shift_is_off_by_default_and_sent_in_the_render_payload(self) -> None:
        payload = payload_mod.build_minimax_render_payload(_settings())
        self.assertFalse(payload["sigma_pass1_enabled"])
        self.assertFalse(payload["sigma_pass2_enabled"])
        self.assertEqual((payload["sigma_pass1_video"], payload["sigma_pass1_audio"]), (12.0, 3.0))

    def test_sigma_values_are_decimals_and_range_checked(self) -> None:
        settings = _settings(sigma_pass1_enabled=True, sigma_pass1_video=2.5, sigma_pass1_audio="1.5")
        self.assertEqual((settings["sigma_pass1_video"], settings["sigma_pass1_audio"]), (2.5, 1.5))
        problems = payload_mod.validate_minimax_h3_patch({"sigma_pass1_video": 500, "sigma_pass2_audio": 0})
        self.assertEqual(set(problems), {"sigma_pass1_video", "sigma_pass2_audio"})
        self.assertEqual(payload_mod.validate_minimax_h3_patch({"sigma_pass1_video": 2.5, "use_dmad": True}), {})


class DmadSettingsTests(unittest.TestCase):
    def test_dmad_fills_in_the_loras_sampler_settings_and_sigma_shift(self) -> None:
        on = _settings(**payload_mod.set_minimax_h3_dmad({"video_mode": "reference_to_video", "render_pass": "two_pass"}, True))
        self.assertTrue(on["use_dmad"])
        self.assertEqual(on["loras"], [{"name": DMAD_LORA, "strength": 1, "apply_to": "pass1"}])
        self.assertEqual((on["use_loras"], on["lora_count"]), (True, 1))
        self.assertEqual((on["two_pass_lora_name"], on["two_pass_lora_strength"]), (REFINER, 0.7))
        self.assertEqual((on["two_pass_pass1_sampler"], on["two_pass_pass1_scheduler"], on["two_pass_pass1_steps"]), ("lcm", "simple", 4))
        self.assertEqual((on["two_pass_pass2_steps"], on["two_pass_pass2_denoise"]), (3, 0.2))
        self.assertEqual((on["sampler_name"], on["steps"]), ("lcm", 4))
        self.assertEqual((on["sigma_pass1_enabled"], on["sigma_pass1_video"], on["sigma_pass1_audio"]), (True, 12.0, 2.0))
        self.assertFalse(on["sigma_pass2_enabled"])

    def test_standard_puts_back_what_dmad_replaced(self) -> None:
        before = {
            "video_mode": "reference_to_video", "render_pass": "two_pass", "steps": 17, "sampler_name": "euler",
            "two_pass_pass1_steps": 25, "two_pass_lora_name": "my_turbo.safetensors", "two_pass_lora_strength": 0.9,
            "use_loras": True, "lora_count": 1, "loras": [{"name": "style.safetensors", "strength": 0.4, "apply_to": "both"}],
            "sigma_pass2_enabled": True, "sigma_pass2_video": 7, "sigma_pass2_audio": 1.5,
        }
        on = payload_mod.set_minimax_h3_dmad(before, True)
        self.assertEqual([item["name"] for item in on["loras"]], [DMAD_LORA, "style.safetensors"])
        off = payload_mod.set_minimax_h3_dmad(on, False)
        self.assertFalse(off["use_dmad"])
        self.assertEqual(off["dmad_backup"], {})
        self.assertEqual(off["loras"], before["loras"])
        for key in ("steps", "sampler_name", "two_pass_pass1_steps", "two_pass_lora_name", "two_pass_lora_strength",
                    "sigma_pass2_enabled", "sigma_pass2_video", "sigma_pass2_audio", "lora_count", "use_loras"):
            self.assertEqual(off[key], before[key], key)

    def test_standard_with_no_extra_loras_before_leaves_the_list_off(self) -> None:
        off = payload_mod.set_minimax_h3_dmad(payload_mod.set_minimax_h3_dmad({}, True), False)
        self.assertEqual((off["use_loras"], off["lora_count"], off["loras"]), (False, 0, []))

    def test_dmad_has_no_two_pass_advanced(self) -> None:
        on = payload_mod.set_minimax_h3_dmad({"video_mode": "reference_to_video", "render_pass": "three_pass"}, True)
        self.assertEqual(on["render_pass"], "two_pass")
        self.assertEqual(_settings(video_mode="reference_to_video", render_pass="three_pass", use_dmad=True)["render_pass"], "two_pass")
        self.assertEqual(_settings(video_mode="reference_to_video", render_pass="three_pass")["render_pass"], "three_pass")

    def test_dmad_fills_in_saved_pass_profiles_too(self) -> None:
        current = {"video_mode": "reference_to_video", "render_pass": "single",
                   "ref_pass_profiles": {"two_pass": {"render_pass": "two_pass", "steps": 9}}}
        on = payload_mod.set_minimax_h3_dmad(current, True)
        self.assertTrue(on["ref_pass_profiles"]["two_pass"]["use_dmad"])
        self.assertEqual(on["ref_pass_profiles"]["two_pass"]["loras"][0]["name"], DMAD_LORA)
        off = payload_mod.set_minimax_h3_dmad(on, False)
        self.assertFalse(off["ref_pass_profiles"]["two_pass"]["use_dmad"])
        self.assertEqual(off["ref_pass_profiles"]["two_pass"]["steps"], 9)

    def test_an_api_patch_that_turns_dmad_on_fills_in_the_preset_first(self) -> None:
        current = {"video_mode": "reference_to_video", "render_pass": "two_pass"}
        merged = payload_mod.apply_minimax_h3_settings_patch(current, {"use_dmad": True, "two_pass_pass2_steps": 2})
        self.assertTrue(merged["use_dmad"])
        self.assertEqual(merged["two_pass_pass1_sampler"], "lcm")
        self.assertEqual(merged["two_pass_pass2_steps"], 2)
        self.assertEqual(payload_mod.apply_minimax_h3_settings_patch(merged, {"use_dmad": True}), merged)
        back = payload_mod.apply_minimax_h3_settings_patch(merged, {"use_dmad": False})
        self.assertFalse(back["use_dmad"])
        self.assertEqual(back["two_pass_pass1_sampler"], "res_multistep")

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_python_twin_matches_the_ui_preset(self) -> None:
        script = (
            f"import {{ cloneMiniMaxH3Settings, setMiniMaxH3Dmad }} from {json.dumps(MINIMAX_H3_MJS.as_uri())};"
            "const cases = JSON.parse(process.argv[1]);"
            "console.log(JSON.stringify(cases.map(({ settings, enabled }) => setMiniMaxH3Dmad(cloneMiniMaxH3Settings(settings), enabled))));"
        )
        cases = [
            {"settings": {"video_mode": "reference_to_video", "render_pass": "two_pass"}, "enabled": True},
            {"settings": {"video_mode": "reference_to_video", "render_pass": "single", "steps": 17, "sampler_name": "euler"}, "enabled": True},
            {"settings": {"pipeline": "refmod", "render_pass": "three_pass"}, "enabled": True},
            {"settings": {"video_mode": "reference_to_video", "use_loras": True, "lora_count": 2,
                          "loras": [{"name": "a.safetensors", "strength": 0.5, "apply_to": "pass2"},
                                    {"name": "b.safetensors", "strength": 1, "apply_to": "both"}]}, "enabled": True},
        ]
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script, json.dumps(cases)],
            capture_output=True, text=True, timeout=60, check=True,
        )
        ui_results = json.loads(result.stdout)
        keys = (*payload_mod.DMAD_PRESET, "use_loras", "lora_count", "loras", "use_dmad", "render_pass")
        for case, ui in zip(cases, ui_results):
            with self.subTest(case=case["settings"]):
                python = _settings(**payload_mod.set_minimax_h3_dmad(case["settings"], case["enabled"]))
                for key in keys:
                    self.assertEqual(python[key], ui[key], key)
                self.assertEqual(python["dmad_backup"], ui["dmad_backup"])


class SigmaGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vrgdg_dmad_")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.image = self.project / "first.png"
        Image.new("RGB", (64, 64), "red").save(self.image)
        self.audio = self.project / "source.wav"
        with wave.open(str(self.audio), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(b"\0\0\0\0" * 44100 * 2)
        self.scene = {
            "prompt": "The singer turns toward the camera.",
            "project_folder": str(self.project),
            "audio_path": str(self.audio),
            "source_duration_seconds": 2,
            "scene_duration_seconds": 1,
            "scene_number": 1,
            "timeline_start_seconds": 0,
            "timeline_end_seconds": 1,
            "image_paths": [str(self.image)],
        }
        choices = [
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors",
            "minimax_h3_latent_upscaler_3d_bf16.safetensors",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            DMAD_LORA,
            REFINER,
        ]
        for target, name in ((models, "_folder_choices"), (models, "_lora_choices")):
            started = patch.object(target, name, return_value=choices)
            started.start()
            self.addCleanup(started.stop)
        nodes = patch.object(patches, "_get_comfy_node_mappings", return_value={"MiniMaxH3SigmaShift": object})
        nodes.start()
        self.addCleanup(nodes.stop)

    def _payload(self, **setting_overrides):
        settings = _settings(video_mode="reference_to_video", **setting_overrides)
        return {**payload_mod.build_minimax_render_payload(settings), **self.scene}

    def _sigma_nodes(self, graph):
        return {key: node for key, node in graph.items() if node.get("class_type") == "MiniMaxH3SigmaShift"}

    def test_no_sigma_node_while_sigma_shift_is_off(self) -> None:
        for builder, passes in ((workflows._build_minimax_h3_api_prompt, "single"), (workflows._build_minimax_h3_2pass_api_prompt, "two_pass")):
            with self.subTest(passes=passes):
                built = builder(self._payload(render_pass=passes))
                self.assertEqual(self._sigma_nodes(built["prompt"]), {})

    def test_single_pass_sigma_shift_sits_in_front_of_the_scheduler_and_guider(self) -> None:
        built = workflows._build_minimax_h3_api_prompt(self._payload(
            render_pass="single", sigma_pass1_enabled=True, sigma_pass1_video=8.5, sigma_pass1_audio=1.25,
        ))
        graph = built["prompt"]
        sigma = self._sigma_nodes(graph)
        self.assertEqual(list(sigma), ["9401"])
        self.assertEqual(sigma["9401"]["inputs"]["shift_video"], 8.5)
        self.assertEqual(sigma["9401"]["inputs"]["shift_audio"], 1.25)
        self.assertEqual(graph["124"]["inputs"]["model"], ["9401", 0])
        self.assertEqual(graph["126"]["inputs"]["model"], ["9401", 0])
        self.assertEqual(built["sigma_shift"]["shift_video"], 8.5)

    def test_two_pass_sigma_shift_is_per_pass(self) -> None:
        built = workflows._build_minimax_h3_2pass_api_prompt(self._payload(
            render_pass="two_pass", sigma_pass1_enabled=False, sigma_pass2_enabled=True, sigma_pass2_video=5, sigma_pass2_audio=2,
        ))
        graph = built["prompt"]
        sigma = self._sigma_nodes(graph)
        self.assertEqual(list(sigma), ["9402"])
        self.assertEqual(sigma["9402"]["inputs"]["model"], ["207", 0])
        for node_id in ("192", "193"):
            self.assertEqual(graph[node_id]["inputs"]["model"], ["9402", 0])
        for node_id in ("124", "126"):
            self.assertEqual(graph[node_id]["inputs"]["model"], ["141", 0])
        self.assertIsNone(built["two_pass"]["sigma_shift"]["pass1"])
        self.assertEqual(built["two_pass"]["sigma_shift"]["pass2"]["shift_video"], 5)

    def test_sigma_shift_needs_the_comfyui_node(self) -> None:
        with patch.object(patches, "_get_comfy_node_mappings", return_value={}):
            with self.assertRaisesRegex(ValueError, "MiniMaxH3SigmaShift"):
                workflows._build_minimax_h3_2pass_api_prompt(self._payload(render_pass="two_pass", sigma_pass1_enabled=True))

    def test_dmad_two_pass_graph_matches_the_tested_ab_workflow(self) -> None:
        dmad = payload_mod.set_minimax_h3_dmad({"video_mode": "reference_to_video", "render_pass": "two_pass"}, True)
        built = workflows._build_minimax_h3_2pass_api_prompt({
            **payload_mod.build_minimax_render_payload(_settings(**dmad)), **self.scene,
        })
        graph = built["prompt"]
        # Pass 1: loader -> DMAD LoRA -> sigma shift -> scheduler and guider, lcm, simple, 4 steps.
        dmad_lora = next(node for node in graph.values()
                         if node.get("class_type") == "LoraLoaderModelOnly" and node["inputs"]["lora_name"] == DMAD_LORA)
        self.assertEqual(dmad_lora["inputs"]["strength_model"], 1)
        self.assertEqual(dmad_lora["inputs"]["model"], ["141", 0])
        sigma = graph["9401"]
        self.assertEqual((sigma["inputs"]["shift_video"], sigma["inputs"]["shift_audio"]), (12.0, 2.0))
        self.assertEqual(graph[sigma["inputs"]["model"][0]], dmad_lora)
        self.assertEqual(graph["124"]["inputs"]["model"], ["9401", 0])
        self.assertEqual(graph["123"]["inputs"]["sampler_name"], "lcm")
        self.assertEqual((graph["124"]["inputs"]["scheduler"], graph["124"]["inputs"]["steps"]), ("simple", 4))
        # Pass 2: loader -> refiner at 0.7, no sigma shift, 3 steps at 0.2 denoise.
        self.assertEqual(graph["207"]["inputs"]["lora_name"], REFINER)
        self.assertEqual(graph["207"]["inputs"]["strength_model"], 0.7)
        self.assertEqual(graph["192"]["inputs"]["model"], ["207", 0])
        self.assertEqual(graph["190"]["inputs"]["value"], 3)
        self.assertEqual(graph["191"]["inputs"]["value"], 0.2)
        self.assertEqual(graph["210"]["inputs"]["sampler_name"], "res_multistep")

    def test_missing_selected_dmad_lora_is_rejected_instead_of_silently_removed(self) -> None:
        dmad = payload_mod.set_minimax_h3_dmad({"video_mode": "reference_to_video", "render_pass": "two_pass"}, True)
        # Reproduce the installed filename differing from the selected preset.
        installed = models._folder_choices("loras")
        installed = [name for name in installed if name != DMAD_LORA]
        installed.append("dmad_minimax_h3_4step_full_critic.safetensors")
        with patch.object(models, "_folder_choices", return_value=installed), patch.object(models, "_lora_choices", return_value=installed):
            for builder in (workflows._build_minimax_h3_2pass_api_prompt, workflows._build_minimax_h3_api_prompt):
                with self.subTest(builder=builder.__name__):
                    with self.assertRaisesRegex(ValueError, DMAD_LORA):
                        builder({**payload_mod.build_minimax_render_payload(_settings(**dmad)), **self.scene})

    def test_dmad_single_pass_graph_has_the_lora_lcm_and_sigma_shift(self) -> None:
        dmad = payload_mod.set_minimax_h3_dmad({"video_mode": "reference_to_video", "render_pass": "single"}, True)
        built = workflows._build_minimax_h3_api_prompt({
            **payload_mod.build_minimax_render_payload(_settings(**dmad)), **self.scene,
        })
        graph = built["prompt"]
        loras = [node for node in graph.values() if node.get("class_type") == "LoraLoaderModelOnly"]
        self.assertEqual([node["inputs"]["lora_name"] for node in loras], [DMAD_LORA])
        self.assertEqual(graph["9401"]["inputs"]["shift_audio"], 2.0)
        self.assertEqual(graph["124"]["inputs"]["model"], ["9401", 0])
        self.assertEqual(graph["123"]["inputs"]["sampler_name"], "lcm")
        self.assertEqual(graph["124"]["inputs"]["steps"], 4)

    def test_dmad_cannot_render_without_an_enabled_first_pass_adapter(self) -> None:
        dmad = payload_mod.set_minimax_h3_dmad({"video_mode": "reference_to_video", "render_pass": "two_pass"}, True)
        payload = {**payload_mod.build_minimax_render_payload(_settings(**dmad)), **self.scene}
        self.assertTrue(payload["use_dmad"])
        invalid = [
            {"use_loras": False}, {"lora_count": 0}, {"loras": []},
            {"loras": [{"name": DMAD_LORA, "apply_to": "pass2", "strength": 1}]},
            {"loras": [{"name": DMAD_LORA, "apply_to": "pass1", "strength": 0}]},
        ]
        for builder in (workflows._build_minimax_h3_api_prompt, workflows._build_minimax_h3_2pass_api_prompt):
            for patch in invalid:
                with self.subTest(builder=builder.__name__, patch=patch):
                    with self.assertRaisesRegex(ValueError, "DMAD rendering requires"):
                        builder({**payload, **patch})


if __name__ == "__main__":
    unittest.main()
