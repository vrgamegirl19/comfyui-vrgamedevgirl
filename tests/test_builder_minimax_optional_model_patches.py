import ast
import copy
import json
import os
import random
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "VRGDG_WorkflowRunnerNodes.py"
TEMPLATE_PATH = ROOT / "Workflows" / "UsedForUIDoNotTouch" / "minimax_audio_driven_builder_latent_upscale_2pass_api.json"
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


def load_two_pass_builder(sparse_method="Sol-Attn (adaptive tau)"):
    module = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_minimax_h3_2pass_api_prompt"
    )
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    timing = types.SimpleNamespace(
        workflow_duration_input_seconds=5.0,
        final_trim_start_seconds=0.0,
        final_trim_duration_seconds=5.0,
        final_frame_count=120,
        to_dict=lambda: {},
    )

    def first_value(payload, *keys, default=None):
        return next((payload[key] for key in keys if payload.get(key) is not None), default)

    def int_value(payload, key, default, low, high):
        return max(low, min(high, int(payload.get(key, default))))

    def float_value(payload, key, default, low, high):
        return max(low, min(high, float(payload.get(key, default))))

    def bool_value(payload, key, default=False):
        return bool(payload.get(key, default))

    def set_input(prompt, node_id, name, value):
        prompt[str(node_id)]["inputs"][name] = value

    namespace = {
        "copy": copy,
        "random": random,
        "json": json,
        "os": os,
        "_load_api_template": lambda _path: (str(TEMPLATE_PATH), copy.deepcopy(template)),
        "_minimax_h3_2pass_api_template_path": lambda: str(TEMPLATE_PATH),
        "_first_payload_value": first_value,
        "_minimax_h3_effective_warmup_frames": lambda payload: first_value(payload, "warmup_frames", "pre_frames", default=0),
        "_minimax_h3_latent_continuation_mode": lambda payload: payload.get("continuity_mode", "off"),
        "_minimax_h3_is_latent_mode": lambda mode: mode in ("latent_continuation", "latent_continuation_exact_frame"),
        "_patch_minimax_h3_latent_continuation": lambda _prompt, _payload: {"enabled": False},
        "_patch_minimax_h3_save_latent": lambda _prompt, _payload, _timing=None: {"enabled": False},
        "_int_payload": int_value,
        "_float_payload": float_value,
        "_bool_payload": bool_value,
        "_probe_media_duration_seconds": lambda _path: 5.0,
        "calculate_minimax_h3_timing": lambda *_args, **_kwargs: timing,
        "_trim_minimax_h3_audio_context": lambda path, *_args: {"audio_path": path},
        "_minimax_h3_image_paths": lambda _payload: [],
        "_minimax_h3_video_references": lambda _payload: [],
        "_require_model_choice": lambda *_args: None,
        "_set_api_input": set_input,
        "_clean_lora_name": lambda value: str(value),
        "_NONE_LORA": "[none]",
        "_model_choice_exists": lambda *_args: True,
        "_get_comfy_node_mappings": lambda: {
            "MiniMaxChunkFeedForward": object(),
            "BlockSparseAttention": types.SimpleNamespace(INPUT_TYPES=lambda: {
                "required": {
                    "selection": ("COMFY_DYNAMICCOMBO_V3", {"options": [
                        {"key": sparse_method, "inputs": {"required": {"tau": ("FLOAT", {"default": 1.3})}}},
                    ]}),
                },
            }),
            "H3FastVAEDecode": object(),
        },
        "_minimax_h3_output_location": lambda _folder, _scene: (_folder, "scene_0001"),
    }
    exec(compile(ast.Module(body=[node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in {"_patch_minimax_h3_optional_model_paths", "_patch_minimax_h3_te_speed", "_patch_minimax_h3_fast_decode"}] + [function], type_ignores=[]), str(RUNNER_PATH), "exec"), namespace)
    return namespace["_build_minimax_h3_2pass_api_prompt"]


def load_single_pass_builder():
    namespace = load_two_pass_builder().__globals__.copy()
    module = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_build_minimax_h3_api_prompt", "_api_node_id_by_class", "_patch_minimax_h3_optional_model_paths"}]
    namespace.update({
        "_MINIMAX_H3_ASPECT_RATIOS": {"16:9 (Widescreen)"},
        "_minimax_h3_api_template_path": lambda: str(TEMPLATE_PATH.parent / "minimax_audio_driven_builder_api.json"),
        "_minimax_h3_built_in_audio_api_template_path": lambda: str(TEMPLATE_PATH.parent / "minimax_built_in_audio_builder_api.json"),
        "_load_api_template": lambda path: (path, json.loads(Path(path).read_text(encoding="utf-8"))),
        "_patch_minimax_h3_advanced_settings": lambda *_args: {},
        "_patch_minimax_h3_loras": lambda *_args: {},
        "_patch_minimax_h3_turbo": lambda *_args: {"enabled": False},
        "_patch_minimax_h3_memory_efficient_sage_attention": lambda *_args: {"enabled": False},
    })
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(RUNNER_PATH), "exec"), namespace)
    return namespace["_build_minimax_h3_api_prompt"]


class BuilderMiniMaxOptionalModelPatchTests(unittest.TestCase):
    def payload(self, folder, audio, **updates):
        payload = {
            "prompt": "A subject moves through the scene.",
            "audio_path": audio,
            "project_folder": folder,
            "timeline_start_seconds": 0,
            "timeline_end_seconds": 5,
        }
        payload.update(updates)
        return payload

    def test_ui_sends_per_pass_options_for_both_two_pass_modes(self):
        self.assertTrue('miniMaxAccelerationControls.flatMap' in BUILDER_SOURCE)
        self.assertTrue('pass${pass}_use_${key}' in BUILDER_SOURCE)
        self.assertTrue('two_pass_use_fast_vae_decode: (twoPass || threePass)' in BUILDER_SOURCE)

    def test_single_pass_checkbox_combinations_and_audio_modes(self):
        build = load_single_pass_builder()
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "scene.wav")
            Path(audio).touch()
            for audio_mode in ("input_audio", "built_in_audio"):
                baseline = build(self.payload(folder, audio, audio_mode=audio_mode))["prompt"]
                original_model = baseline["124"]["inputs"]["model"]
                for feed, sparse in ((False, False), (True, False), (False, True), (True, True)):
                    with self.subTest(audio_mode=audio_mode, feed=feed, sparse=sparse):
                        result = build(self.payload(
                            folder, audio, audio_mode=audio_mode, video_mode="reference_to_video",
                            use_feedforward=feed, use_block_sparse_attention=sparse,
                        ))
                        prompt = result["prompt"]
                        expected = (["MiniMaxChunkFeedForward"] if feed else []) + (["BlockSparseAttention"] if sparse else [])
                        patches = result["optional_patch_nodes"]
                        self.assertEqual([item["class_type"] for item in patches], expected)
                        current = original_model
                        for item in patches:
                            self.assertEqual(prompt[item["node"]]["inputs"]["model"], current)
                            current = [item["node"], 0]
                        self.assertEqual(prompt["124"]["inputs"]["model"], current)
                        self.assertEqual(prompt["126"]["inputs"]["model"], current)
                        if not expected:
                            self.assertEqual(prompt, baseline)

    def test_single_pass_missing_nodes_fail_before_queueing(self):
        build = load_single_pass_builder()
        build.__globals__["_get_comfy_node_mappings"] = lambda: {}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "nodes are missing: MiniMaxChunkFeedForward"):
                build(self.payload(folder, "", audio_mode="built_in_audio", use_feedforward=True))

    def test_single_pass_ui_persists_and_sends_options(self):
        for key in ("use_feedforward", "use_block_sparse_attention", "use_te_speed"):
            self.assertTrue(f"{key}: !twoPass && !threePass ? miniMaxSettings.{key} : undefined" in BUILDER_SOURCE)
        self.assertTrue('...currentSettings,' in BUILDER_SOURCE)
        self.assertTrue('miniMaxAccelerationSettings,' in BUILDER_SOURCE)

    def test_unchecked_prompt_keeps_hidden_graph_unchanged(self):
        build = load_two_pass_builder()
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "scene.wav")
            Path(audio).touch()
            result = build(self.payload(folder, audio))
        classes = [node.get("class_type") for node in result["prompt"].values()]
        self.assertNotIn("MiniMaxChunkFeedForward", classes)
        self.assertNotIn("BlockSparseAttention", classes)
        self.assertFalse(result["two_pass"]["feedforward_enabled"])
        self.assertFalse(result["two_pass"]["block_sparse_attention_enabled"])
        self.assertFalse(result["two_pass"]["fast_vae_decode_enabled"])
        self.assertEqual(result["prompt"]["122"]["class_type"], "VAEDecode")

    def test_checked_fast_decode_replaces_final_decoder_with_batch_size_eight(self):
        build = load_two_pass_builder()
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "scene.wav")
            Path(audio).touch()
            result = build(self.payload(folder, audio, two_pass_use_fast_vae_decode=True))
        decoder = result["prompt"]["122"]
        self.assertEqual(decoder["class_type"], "H3FastVAEDecode")
        self.assertEqual(decoder["inputs"]["samples"], ["194", 0])
        self.assertEqual(decoder["inputs"]["vae"], ["119", 0])
        self.assertEqual(decoder["inputs"]["tile_batch_size"], 8)
        self.assertTrue(result["two_pass"]["fast_vae_decode_enabled"])
        self.assertEqual(result["two_pass"]["fast_vae_decode_tile_batch_size"], 8)

    def test_checked_prompt_adds_feedforward_then_sparse_to_both_passes(self):
        build = load_two_pass_builder()
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "scene.wav")
            Path(audio).touch()
            result = build(self.payload(
                folder,
                audio,
                two_pass_use_feedforward=True,
                two_pass_use_block_sparse_attention=True,
            ))
        prompt = result["prompt"]
        patches = result["two_pass"]["optional_patch_nodes"]
        self.assertEqual(
            [(item["target"], item["class_type"]) for item in patches],
            [
                ("pass1", "MiniMaxChunkFeedForward"),
                ("pass1", "BlockSparseAttention"),
                ("pass2", "MiniMaxChunkFeedForward"),
                ("pass2", "BlockSparseAttention"),
            ],
        )
        for sampler_id, target in (("124", "pass1"), ("192", "pass2")):
            sparse_id = prompt[sampler_id]["inputs"]["model"][0]
            sparse = prompt[sparse_id]
            self.assertEqual(sparse["class_type"], "BlockSparseAttention")
            self.assertEqual(sparse["inputs"]["selection"], "Sol-Attn (adaptive tau)")
            self.assertEqual(sparse["inputs"]["selection.tau"], 1.3)
            feed_id = sparse["inputs"]["model"][0]
            feed = prompt[feed_id]
            self.assertEqual(feed["class_type"], "MiniMaxChunkFeedForward")
            self.assertEqual(feed["inputs"]["chunks"], 8)
            self.assertEqual(feed["inputs"]["seq_threshold"], 4096)
            self.assertIn(target, sparse["_meta"]["title"])

    def test_updated_sparse_method_is_sent_to_both_passes(self):
        build = load_two_pass_builder(sparse_method="sol-attn")
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "scene.wav")
            Path(audio).touch()
            result = build(self.payload(folder, audio, two_pass_use_block_sparse_attention=True))
        prompt = result["prompt"]
        for sampler_id in ("124", "192"):
            sparse = prompt[prompt[sampler_id]["inputs"]["model"][0]]
            self.assertEqual(sparse["class_type"], "BlockSparseAttention")
            self.assertEqual(sparse["inputs"]["selection"], "sol-attn")
            self.assertEqual(sparse["inputs"]["selection.tau"], 1.3)

    def test_unknown_method_fails_before_queueing_instead_of_switching_algorithms(self):
        build = load_two_pass_builder(sparse_method="vsa")
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "scene.wav")
            Path(audio).touch()
            with self.assertRaisesRegex(ValueError, "supported Sol-Attn method"):
                build(self.payload(folder, audio, two_pass_use_block_sparse_attention=True))

    def test_default_steps_acceleration_and_random_seeds(self):
        build = load_two_pass_builder()
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "scene.wav")
            Path(audio).touch()
            payload = self.payload(folder, audio)
            first = build(payload)["prompt"]
            second = build(payload)["prompt"]
            self.assertEqual(first["190"]["inputs"]["value"], 2)
            for node_id in ("129", "211"):
                self.assertGreaterEqual(first[node_id]["inputs"]["noise_seed"], 0)
                self.assertNotEqual(first[node_id]["inputs"]["noise_seed"], second[node_id]["inputs"]["noise_seed"])
            classes = {node["class_type"] for node in first.values()}
            self.assertFalse(classes & {"TESpeedMiniMaxH3", "MiniMaxChunkFeedForward", "BlockSparseAttention", "H3FastVAEDecode"})
            fixed = build({**payload, "pass1_seed": 123, "pass2_seed": 456})["prompt"]
            self.assertEqual(fixed["129"]["inputs"]["noise_seed"], 123)
            self.assertEqual(fixed["211"]["inputs"]["noise_seed"], 456)

    def test_acceleration_is_independent_for_each_pass_and_advanced(self):
        from test_builder_minimax_advanced_two_pass import BuilderMiniMaxAdvancedTwoPassTests
        base = load_two_pass_builder()
        case = BuilderMiniMaxAdvancedTwoPassTests()
        advanced_ns = case._advanced_prompt_namespace(case._NewSpatialSplit)
        base_mappings = base.__globals__["_get_comfy_node_mappings"]()
        advanced_mappings = advanced_ns["_get_comfy_node_mappings"]()
        base.__globals__["_get_comfy_node_mappings"] = lambda: {**base_mappings, **advanced_mappings}
        advanced_ns["_build_minimax_h3_2pass_api_prompt"] = base
        advanced = advanced_ns["_build_minimax_h3_advanced_2pass_api_prompt"]

        def model_classes(prompt, node_id):
            classes = []
            while node_id:
                node = prompt[node_id]
                classes.append(node["class_type"])
                node_id = (node["inputs"].get("model") or [None])[0]
            return classes

        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "scene.wav")
            Path(audio).touch()
            for build, final_sampler in ((base, "192"), (advanced, "9306")):
                for key, class_type in (("te_speed", "TESpeedMiniMaxH3"), ("feedforward", "MiniMaxChunkFeedForward"), ("block_sparse_attention", "BlockSparseAttention")):
                    for first, second in ((False, False), (True, False), (False, True), (True, True)):
                        with self.subTest(advanced=build is advanced, option=key, first=first, second=second):
                            payload = self.payload(folder, audio, two_pass_use_te_speed=False,
                                two_pass_use_fast_vae_decode=True,
                                **{f"pass1_use_{key}": first, f"pass2_use_{key}": second})
                            prompt = build(payload)["prompt"]
                            self.assertEqual(class_type in model_classes(prompt, "124"), first)
                            self.assertEqual(class_type in model_classes(prompt, final_sampler), second)
                            self.assertEqual(prompt["122"]["class_type"], "H3FastVAEDecode")
                            if build is advanced:
                                self.assertEqual(prompt["9307"]["class_type"], "VAEDecode")
                            for node in prompt.values():
                                for value in node["inputs"].values():
                                    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                                        self.assertIn(str(value[0]), prompt)

    def test_single_pass_replaces_easy_cache_and_preserves_sampler_settings(self):
        build = load_single_pass_builder()
        module = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        wanted = {"_patch_minimax_h3_advanced_settings", "_optional_api_node_id_by_class", "_replace_api_input_refs"}
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        build.__globals__["_MINIMAX_H3_SAGE_ATTENTION_MODES"] = {"auto"}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(RUNNER_PATH), "exec"), build.__globals__)
        with tempfile.TemporaryDirectory() as folder:
            for audio_mode in ("input_audio", "built_in_audio"):
                audio = os.path.join(folder, "scene.wav")
                Path(audio).touch()
                result = build(self.payload(folder, audio, audio_mode=audio_mode,
                    video_mode="reference_to_video", use_te_speed=True, use_fast_vae_decode=True,
                    easy_cache_bypass=False, steps=13, denoise=0.85, sampler_name="euler", scheduler="simple"))
                prompt = result["prompt"]
                self.assertNotIn("EasyCache", [node["class_type"] for node in prompt.values()])
                self.assertEqual(prompt["124"]["inputs"]["steps"], 13)
                self.assertEqual(prompt["124"]["inputs"]["denoise"], 0.85)
                self.assertEqual(prompt["123"]["inputs"]["sampler_name"], "euler")
                self.assertEqual(prompt["9210"]["class_type"], "TESpeedMiniMaxH3")
                self.assertEqual(prompt["126"]["inputs"]["model"], ["9210", 0])
                self.assertEqual(prompt["122"]["class_type"], "H3FastVAEDecode")


if __name__ == "__main__":
    unittest.main()
