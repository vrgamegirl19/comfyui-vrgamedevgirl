import ast
import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "VRGDG_WorkflowRunnerNodes.py"
TEMPLATE_PATH = ROOT / "Workflows" / "UsedForUIDoNotTouch" / "minimax_audio_driven_builder_latent_upscale_2pass_api.json"
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


def load_two_pass_builder():
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
            "BlockSparseAttention": object(),
            "H3FastVAEDecode": object(),
        },
        "_minimax_h3_output_location": lambda _folder, _scene: (_folder, "scene_0001"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER_PATH), "exec"), namespace)
    return namespace["_build_minimax_h3_2pass_api_prompt"]


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

    def test_ui_defaults_off_and_sends_checkbox_values(self):
        self.assertIn("two_pass_use_feedforward: false", BUILDER_SOURCE)
        self.assertIn("two_pass_use_block_sparse_attention: false", BUILDER_SOURCE)
        self.assertIn("two_pass_use_fast_vae_decode: false", BUILDER_SOURCE)
        self.assertIn('makeCheckbox("Use FeedForward (lower VRAM for longer scenes)"', BUILDER_SOURCE)
        self.assertIn('makeCheckbox("Use Block Sparse Attention (faster)"', BUILDER_SOURCE)
        self.assertIn('makeCheckbox("Use Fast Batched VAE Decode (batch size 8)"', BUILDER_SOURCE)
        self.assertIn("two_pass_use_feedforward: twoPass ? miniMaxSettings.two_pass_use_feedforward : undefined", BUILDER_SOURCE)
        self.assertIn("two_pass_use_block_sparse_attention: twoPass ? miniMaxSettings.two_pass_use_block_sparse_attention : undefined", BUILDER_SOURCE)
        self.assertIn("two_pass_use_fast_vae_decode: twoPass ? miniMaxSettings.two_pass_use_fast_vae_decode : undefined", BUILDER_SOURCE)

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


if __name__ == "__main__":
    unittest.main()
