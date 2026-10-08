"""Exercise real I2V graph compilation, audio preparation and settings migration."""

import importlib
import json
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

settings = importlib.import_module(f"{ROOT.name}.minimax.settings_payload")
workflows = importlib.import_module(f"{ROOT.name}.runner.minimax_workflows")
models = importlib.import_module(f"{ROOT.name}.runner.models")


class MiniMaxI2VTwoPassTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vrgdg_i2v_two_pass_")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.first = self.project / "first.png"
        self.last = self.project / "last.png"
        Image.new("RGB", (64, 64), "red").save(self.first)
        Image.new("RGB", (64, 64), "blue").save(self.last)
        self.audio = self.project / "source.wav"
        with wave.open(str(self.audio), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(b"\0\0\0\0" * 44100 * 2)
        self.payload = {
            "video_mode": "image_to_video",
            "prompt": "The subject turns toward the camera.",
            "project_folder": str(self.project),
            "audio_path": str(self.audio),
            "source_duration_seconds": 2,
            "scene_duration_seconds": 1,
            "image_paths": [str(self.first)],
            "pass1_steps": 23,
            "pass2_steps": 7,
            "pass1_seed": 123,
            "pass2_seed": 456,
            "pass1_sampler_name": "euler",
            "pass2_sampler_name": "res_multistep",
            "pass2_denoise": 0.3,
            "final_width": 1344,
            "final_height": 768,
            "latent_upscale_scale": 2,
        }
        # Model weights are not loaded while compiling. Supply a deterministic
        # model registry, retaining the compiler's actual model-choice validation.
        self.registry = patch.object(models, "_folder_choices", return_value=[
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors",
            "minimax_h3_latent_upscaler_3d_bf16.safetensors",
            "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            "character.safetensors",
        ])
        registry = self.registry.start()
        self.addCleanup(self.registry.stop)
        lora_registry = patch.object(models, "_lora_choices", return_value=registry.return_value)
        lora_registry.start()
        self.addCleanup(lora_registry.stop)

    def test_compiler_uses_i2v_template_and_preserves_requested_pass_settings(self) -> None:
        result = workflows._build_minimax_h3_2pass_api_prompt(self.payload)
        graph = result["prompt"]
        self.assertIn("minimax_i2v_", Path(result["workflow_path"]).name)
        self.assertEqual(graph["136"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(graph["214"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(graph["136"]["inputs"]["width"], ["186", 1])
        self.assertEqual(graph["214"]["inputs"]["width"], ["212", 1])
        self.assertEqual(graph["193"]["inputs"]["conditioning"], ["214", 0])
        self.assertNotIn("ref_image_size", graph["136"]["inputs"])
        self.assertIn("fl2va", graph["141"]["inputs"]["model_name"])
        self.assertIn("fl2v", graph["207"]["inputs"]["lora_name"])
        for nid, key, expected in (
            ("124", "steps", 23), ("190", "value", 7),
            ("129", "noise_seed", 123), ("211", "noise_seed", 456),
            ("123", "sampler_name", "euler"), ("191", "value", 0.3),
            ("115", "value", 1344), ("184", "value", 768),
        ):
            self.assertEqual(graph[nid]["inputs"][key], expected)
        self.assertTrue(Path(result["prepared_audio"]["audio_path"]).is_file())
        self.assertEqual(graph["142"]["inputs"]["audio"], ["172", 1])
        self.assertEqual(graph["188"]["inputs"]["model_name"],
                         "minimax_h3_latent_upscaler_3d_bf16.safetensors")
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], graph)

    def test_optional_last_frame_is_encoded_independently_for_each_pass(self) -> None:
        result = workflows._build_minimax_h3_2pass_api_prompt({
            **self.payload, "last_frame_path": str(self.last),
        })
        graph = result["prompt"]
        self.assertEqual(json.loads(graph["180"]["inputs"]["image_paths"]),
                         [str(self.first), str(self.last)])
        for nid in ("136", "214"):
            self.assertEqual(graph[nid]["inputs"]["last_frame"], ["180", 1])
            aligned = graph[f"i2v_keyframe_timing_{nid}"]["inputs"]
            self.assertEqual(aligned["conditioning"], [nid, 0])
            self.assertEqual(aligned["first_frame_index"], 0)
            self.assertEqual(aligned["last_frame_index"], result["post_render_trim"]["frames"] - 1)
        self.assertEqual(graph["193"]["inputs"]["conditioning"], ["i2v_keyframe_timing_214", 0])

    def test_single_pass_flf_has_both_images_and_keeps_endpoint_before_padding(self) -> None:
        result = workflows._build_minimax_h3_api_prompt({
            **self.payload, "last_frame_path": str(self.last), "diffusion_model_name":
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "megapixels": 0.6,
        })
        graph = result["prompt"]
        self.assertEqual(graph["136"]["inputs"]["last_frame"], ["180", 1])
        self.assertEqual(graph["i2v_keyframe_timing_136"]["inputs"]["last_frame_index"],
                         result["post_render_trim"]["frames"] - 1)

    def test_loop_can_use_the_same_image_for_both_frames(self) -> None:
        result = workflows._build_minimax_h3_2pass_api_prompt({
            **self.payload, "last_frame_path": str(self.first),
        })
        self.assertEqual(json.loads(result["prompt"]["180"]["inputs"]["image_paths"]),
                         [str(self.first), str(self.first)])
        self.assertIn("last_frame", result["prompt"]["214"]["inputs"])

    def test_extra_character_lora_can_target_both_passes(self) -> None:
        graph = workflows._build_minimax_h3_2pass_api_prompt({
            **self.payload, "use_loras": True, "lora_count": 1,
            "loras": [{"name": "character.safetensors", "strength": 0.7, "apply_to": "both"}],
        })["prompt"]
        pass1 = graph[graph["126"]["inputs"]["model"][0]]
        pass2 = graph[graph["193"]["inputs"]["model"][0]]
        self.assertEqual(pass1["inputs"]["lora_name"], "character.safetensors")
        self.assertEqual(pass2["inputs"]["lora_name"], "character.safetensors")
        self.assertEqual(pass1["inputs"]["model"], ["141", 0])
        self.assertEqual(pass2["inputs"]["model"], ["207", 0])

    def test_reference_two_pass_still_uses_its_original_template(self) -> None:
        result = workflows._build_minimax_h3_2pass_api_prompt({
            **self.payload, "video_mode": "reference_to_video",
        })
        self.assertNotIn("minimax_i2v_", Path(result["workflow_path"]).name)
        self.assertEqual(result["prompt"]["136"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(result["prompt"]["193"]["inputs"]["conditioning"], ["136", 0])

    def test_invalid_i2v_requests_fail_before_queueing(self) -> None:
        for extra, message in (
            ({"image_paths": []}, "requires a scene image"),
            ({"two_pass_lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"},
             "FL2V/I2V Turbo LoRA"),
            ({"audio_mode": "built_in_audio"}, "Input Audio only"),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                workflows._build_minimax_h3_2pass_api_prompt({**self.payload, **extra})

    def test_old_saves_and_scene_overrides_choose_the_correct_renderer(self) -> None:
        legacy = settings.normalize_minimax_h3_settings({
            "video_mode": "image_to_video", "render_pass": "two_pass",
        })
        self.assertEqual(settings.minimax_workflow_key(legacy), "minimax_h3")
        session = {"minimax_h3_settings": {**legacy, "render_pass": "single"}}
        scene = {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": {
            "video_mode": "image_to_video", "render_pass": "two_pass",
            "i2v_pass_settings_version": 1, "two_pass_pass1_steps": 29,
            "continuity_mode": "latent_continuation_masked", "warmup_frames": 5,
        }}
        locked = settings.minimax_h3_settings_for_scene(session, scene)
        self.assertEqual(settings.minimax_workflow_key(locked), "minimax_h3_2pass")
        payload = settings.build_minimax_render_payload(locked)
        self.assertEqual(payload["pass1_steps"], 29)
        self.assertEqual(payload["continuity_mode"], "off")
        self.assertEqual(payload["pre_frames"], 5)
        self.assertIn("fl2va", payload["diffusion_model_name"])
        self.assertEqual(settings.minimax_workflow_key(settings.minimax_h3_settings_for_scene(session, {})),
                         "minimax_h3")


if __name__ == "__main__":
    unittest.main()
