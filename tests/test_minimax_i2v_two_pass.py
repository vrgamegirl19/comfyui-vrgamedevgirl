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
drafts = importlib.import_module(f"{ROOT.name}.runner.minimax_drafts")


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
            "scene_number": 1, "timeline_start_seconds": 0, "timeline_end_seconds": 1,
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
            "H3\\dmad_minimax_h3_4step_full_critic_comfyui.safetensors",
            "H3/minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors",
        ])
        registry = self.registry.start()
        self.addCleanup(self.registry.stop)
        lora_registry = patch.object(models, "_lora_choices", return_value=registry.return_value)
        lora_registry.start()
        self.addCleanup(lora_registry.stop)
        compiler_loras = patch.object(workflows, "_lora_choices", return_value=registry.return_value)
        compiler_loras.start()
        self.addCleanup(compiler_loras.stop)

    def test_enhancer_overrides_only_pass_two_and_survives_draft_finalize(self):
        for audio_mode in ("input_audio", "built_in_audio"):
            for mode in ("reference_to_video", "image_to_video", "image_reference_to_video", "text_to_video"):
                with self.subTest(audio_mode=audio_mode, mode=mode):
                    payload = {**self.payload, "video_mode": mode, "audio_mode": audio_mode,
                               "scene_number": 1, "timeline_start_seconds": 0, "timeline_end_seconds": 1,
                               "use_loras": True, "lora_count": 1,
                               "loras": [{"name": "character.safetensors", "strength": 0.6, "apply_to": "pass1"}]}
                    normal = workflows._build_minimax_h3_2pass_api_prompt(payload)
                    payload.update(two_pass_use_enhancer=True, two_pass_lora_name="[none]")
                    result = workflows._build_minimax_h3_2pass_api_prompt(payload)
                    graph = result["prompt"]
                    for node_id in ("124", "125", "126", "129"):
                        self.assertEqual(graph[node_id], normal["prompt"][node_id])
                    self.assertEqual(graph["207"]["inputs"]["model"], ["141", 0])
                    self.assertEqual(graph["207"]["inputs"]["lora_name"],
                                     "H3\\dmad_minimax_h3_4step_full_critic_comfyui.safetensors")
                    self.assertEqual(graph["207"]["inputs"]["strength_model"], 1.0)
                    refiner = graph["pass2_enhancer_refiner"]["inputs"]
                    self.assertEqual(refiner["model"], ["207", 0])
                    self.assertEqual(refiner["strength_model"], 1.0)
                    self.assertEqual(graph["193"]["inputs"]["model"], ["pass2_enhancer_refiner", 0])
                    self.assertEqual(graph["192"]["class_type"], "ManualSigmas")
                    self.assertEqual(graph["192"]["inputs"], {"sigmas": "0.7, 0.494118, 0"})
                    self.assertEqual(graph["194"]["inputs"]["sigmas"], ["192", 0])
                    self.assertEqual(graph["210"]["inputs"]["sampler_name"], "er_sde")
                    self.assertEqual(result["two_pass"]["pass2_steps"], 2)
                    self.assertEqual(graph["142"]["inputs"]["audio"], normal["prompt"]["142"]["inputs"]["audio"])
                    draft = drafts._build_minimax_draft(result, payload)
                    self.assertNotIn("194", draft["prompt"])
                    self.assertEqual(draft["draft_steps"], {"pass1": 23, "pass2": 2})
                    Path(draft["prompt"]["draft_save"]["inputs"]["checkpoint_path"]).touch()
                    final = drafts._build_minimax_finalize({**payload, "draft_id": draft["draft_id"]})
                    for node_id in ("192", "193", "194", "207", "pass2_enhancer_refiner"):
                        self.assertEqual(final["prompt"][node_id], graph[node_id])
                    # Switching off restores the caller's normal Turbo and sampler settings.
                    restored = workflows._build_minimax_h3_2pass_api_prompt({
                        **payload, "two_pass_use_enhancer": False,
                        "two_pass_lora_name": normal["prompt"]["207"]["inputs"]["lora_name"],
                    })
                    for node_id in ("192", "193", "194", "207", "210"):
                        self.assertEqual(restored["prompt"][node_id], normal["prompt"][node_id])

    def test_enhancer_reports_both_missing_files_before_queueing(self):
        with patch.object(workflows, "_lora_choices", return_value=[]):
            with self.assertRaises(ValueError) as error:
                workflows._build_minimax_h3_2pass_api_prompt({**self.payload, "two_pass_use_enhancer": True})
        self.assertIn("DMAD", str(error.exception))
        self.assertIn("Open-H3 Refiner", str(error.exception))
        self.assertIn("ComfyUI/models/loras", str(error.exception))

    def test_draft_and_finalize_use_real_compiled_graph_for_input_and_native_audio(self) -> None:
        for audio_mode in ("input_audio", "built_in_audio"):
            for mode in ("image_to_video", "reference_to_video", "image_reference_to_video"):
                with self.subTest(audio_mode=audio_mode, mode=mode):
                    payload = {**self.payload, "audio_mode": audio_mode, "video_mode": mode,
                               "scene_number": 1, "timeline_start_seconds": 0, "timeline_end_seconds": 1}
                    built = workflows._build_minimax_h3_2pass_api_prompt(payload)
                    draft = drafts._build_minimax_draft(built, payload)
                    self.assertEqual(draft["prompt"]["125"]["inputs"], built["prompt"]["125"]["inputs"])
                    self.assertNotIn("194", draft["prompt"])
                    Path(draft["prompt"]["draft_save"]["inputs"]["checkpoint_path"]).touch()
                    final = drafts._build_minimax_finalize({**payload, "draft_id": draft["draft_id"]})
                    self.assertEqual(final["prompt"]["125"]["class_type"], "VRGDG_MiniMaxH3LoadDraft")
                    self.assertEqual(final["prompt"]["194"], built["prompt"]["194"])
                    self.assertEqual(final["post_render_trim"], built["post_render_trim"])

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

    def test_image_reference_rebuilds_conditioning_for_the_upscaled_sampler(self) -> None:
        for audio_mode in ("input_audio", "built_in_audio"):
            for ref_size in ("match", "max"):
                for last_frame in (False, True):
                    with self.subTest(audio_mode=audio_mode, ref_size=ref_size, last_frame=last_frame):
                        payload = {
                            **self.payload, "video_mode": "image_reference_to_video", "audio_mode": audio_mode,
                            "image_paths": [str(self.first), str(self.last)], "ref_image_size": ref_size,
                            "latent_upscale_scale": 1.5,
                            "two_pass_lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                        }
                        if last_frame:
                            end = self.project / "end.png"
                            Image.new("RGB", (64, 64), "green").save(end)
                            payload["last_frame_path"] = str(end)
                        graph = workflows._build_minimax_h3_2pass_api_prompt(payload)["prompt"]
                        self.assertEqual(graph["136"]["class_type"], "VRGDG_MiniMaxH3ImageReferenceToVideo")
                        self.assertEqual(graph["214"]["class_type"], graph["136"]["class_type"])
                        self.assertEqual(graph["193"]["inputs"]["conditioning"], ["214", 0])
                        self.assertEqual(graph["214"]["inputs"]["width"], ["212", 1])
                        self.assertEqual(graph["214"]["inputs"]["height"], ["213", 1])
                        for dimension, source in (("212", "186"), ("213", "187")):
                            self.assertEqual(graph[dimension]["inputs"]["expression"], "round(a / 16 * b) * 16")
                            self.assertEqual(graph[dimension]["inputs"]["values.a"], [source, 1])
                            self.assertEqual(graph[dimension]["inputs"]["values.b"], ["185", 0])
                        for node_id in ("136", "214"):
                            inputs = graph[node_id]["inputs"]
                            self.assertEqual(inputs["first_frame"], ["180", 0])
                            self.assertEqual(inputs["ref_images.ref_image_0"], ["180", 2 if last_frame else 1])
                            self.assertEqual(inputs["ref_image_size"], ref_size)
                            if last_frame:
                                self.assertEqual(inputs["last_frame"], ["180", 1])
                            else:
                                self.assertNotIn("last_frame", inputs)
                        for node in graph.values():
                            for value in node["inputs"].values():
                                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                                    self.assertIn(value[0], graph)

    def test_reference_two_pass_still_uses_its_original_template(self) -> None:
        result = workflows._build_minimax_h3_2pass_api_prompt({
            **self.payload, "video_mode": "reference_to_video",
        })
        self.assertNotIn("minimax_i2v_", Path(result["workflow_path"]).name)
        self.assertEqual(result["prompt"]["136"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(result["prompt"]["193"]["inputs"]["conditioning"], ["136", 0])

    def test_text_two_pass_compiles_without_reference_media_and_can_finalize(self) -> None:
        for audio_mode in ("input_audio", "built_in_audio"):
            with self.subTest(audio_mode=audio_mode):
                payload = {**self.payload, "video_mode": "text_to_video", "image_paths": [],
                           "audio_mode": audio_mode,
                           "diffusion_model_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                           "two_pass_lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"}
                built = workflows._build_minimax_h3_2pass_api_prompt(payload)
                graph = built["prompt"]
                self.assertEqual(built["reference_inputs"], {"image_count": 0, "video_count": 0})
                self.assertEqual(json.loads(graph["180"]["inputs"]["image_paths"]), [])
                self.assertEqual(graph["136"]["class_type"], "MiniMaxH3ReferenceToVideo")
                self.assertEqual(graph["193"]["inputs"]["conditioning"], ["136", 0])
                self.assertEqual(graph["124"]["inputs"]["steps"], 23)
                self.assertEqual(graph["190"]["inputs"]["value"], 7)
                draft = drafts._build_minimax_draft(built, payload)
                Path(draft["prompt"]["draft_save"]["inputs"]["checkpoint_path"]).touch()
                final = drafts._build_minimax_finalize({**payload, "draft_id": draft["draft_id"]})
                self.assertEqual(final["prompt"]["194"], graph["194"])

    def test_text_two_pass_migration_and_renderer_selection(self) -> None:
        legacy = settings.normalize_minimax_h3_settings({"video_mode": "text_to_video", "render_pass": "two_pass"})
        self.assertEqual(settings.minimax_workflow_key(legacy), "minimax_h3")
        selected = settings.normalize_minimax_h3_settings({**legacy, "render_pass": "two_pass", "two_pass_pass1_steps": 29})
        self.assertEqual(settings.minimax_workflow_key(selected), "minimax_h3_2pass")
        self.assertEqual(settings.build_minimax_render_payload(selected)["pass1_steps"], 29)
        reloaded = settings.normalize_minimax_h3_settings(json.loads(json.dumps(selected)))
        self.assertEqual(settings.minimax_workflow_key(reloaded), "minimax_h3_2pass")
        locked = {"use_scene_minimax_h3_settings": True,
                  "minimax_h3_settings": {"video_mode": "text_to_video", "render_pass": "two_pass"}}
        self.assertEqual(settings.minimax_workflow_key(settings.minimax_h3_settings_for_scene(
            {"minimax_h3_settings": selected}, locked)), "minimax_h3")

    def test_invalid_i2v_requests_fail_before_queueing(self) -> None:
        for extra, message in (
            ({"image_paths": []}, "requires a scene image"),
            ({"two_pass_lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"},
             "FL2V/I2V Turbo LoRA"),
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
