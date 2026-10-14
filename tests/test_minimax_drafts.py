import ast
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("minimax_drafts", ROOT / "runner/minimax_drafts.py")
drafts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drafts)


class MiniMaxDraftTests(unittest.TestCase):
    def result(self, image=False, advanced=False):
        name = "minimax_i2v_audio_driven_builder_latent_upscale_2pass_api.json" if image else "minimax_audio_driven_builder_latent_upscale_2pass_api.json"
        prompt = json.loads((ROOT / "Workflows/UsedForUIDoNotTouch" / name).read_text(encoding="utf-8"))
        prompt["9850"] = {"class_type": "VRGDG_MiniMaxH3SaveLatent", "inputs": {"latent": prompt["122"]["inputs"]["samples"]}}
        prompt["122"]["inputs"]["samples"] = ["9850", 0]
        if advanced:
            prompt["9306"] = {"class_type": "MMH3UltimateUpscale", "inputs": {"latent": ["125", 0], "model": prompt["192"]["inputs"]["model"]}}
            prompt["9850"]["inputs"]["latent"] = ["9306", 0]
            prompt["9308"] = copy.deepcopy(prompt["142"])
        return {"prompt": prompt, "advanced_two_pass" if advanced else "two_pass": {}, "timing": {"h3_frame_count": 240}, "post_render_trim": {"duration": 10}}

    def test_real_standard_image_and_advanced_graphs_split_without_resampling(self):
        for image, advanced in [(False, False), (True, False), (False, True), (True, True)]:
            with self.subTest(image=image, advanced=advanced), tempfile.TemporaryDirectory() as folder:
                payload = {"project_folder": folder, "scene_number": 1, "timeline_start_seconds": 0, "timeline_end_seconds": 10}
                original = self.result(image, advanced)
                draft = drafts._build_minimax_draft(original, payload)
                graph = draft["prompt"]
                self.assertEqual(graph["125"]["class_type"], "SamplerCustomAdvanced")
                self.assertNotIn("194", graph)
                self.assertNotIn("9306", graph)
                self.assertNotIn("9308", graph)
                self.assertEqual(graph["9850"]["inputs"]["latent"], ["125", 0])
                self.assertEqual(graph["draft_save"]["inputs"]["latent"], ["125", 0])
                self.assertEqual(graph["142"]["inputs"]["images"], ["draft_decode", 0])
                self.assertEqual(original["prompt"]["122"]["inputs"]["samples"], ["9850", 0])
                Path(graph["draft_save"]["inputs"]["checkpoint_path"]).touch()
                final = drafts._build_minimax_finalize({**payload, "draft_id": draft["draft_id"]})
                self.assertEqual(final["prompt"]["125"]["class_type"], "VRGDG_MiniMaxH3LoadDraft")
                self.assertNotIn("draft_save", final["prompt"])
                self.assertNotIn("9308", final["prompt"])
                self.assertIn("9306" if advanced else "194", final["prompt"])
                self.assertEqual(final["prompt"]["142"]["inputs"]["images"], original["prompt"]["142"]["inputs"]["images"])

    def test_missing_checkpoint_and_changed_scene_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            payload = {"project_folder": folder, "scene_number": 1, "timeline_start_seconds": 0, "timeline_end_seconds": 10}
            draft = drafts._build_minimax_draft(self.result(), payload)
            request = {**payload, "draft_id": draft["draft_id"]}
            with self.assertRaises(FileNotFoundError):
                drafts._build_minimax_finalize(request)
            Path(draft["prompt"]["draft_save"]["inputs"]["checkpoint_path"]).touch()
            for change in [{"scene_number": 2}, {"timeline_end_seconds": 12}]:
                with self.assertRaises(ValueError):
                    drafts._build_minimax_finalize({**request, **change})
            with self.assertRaises(ValueError):
                drafts._build_minimax_finalize({**request, "draft_id": "../outside"})

    def test_checkpoint_round_trip_preserves_audio_video_and_masks(self):
        import torch
        class Nested:
            def __init__(self, tensors):
                self.tensors = tensors
        source = ROOT / "minimax/latent_continuation.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"_checkpoint_value", "VRGDG_MiniMaxH3SaveDraft", "VRGDG_MiniMaxH3LoadDraft"}
        namespace = {"torch": torch, "os": os, "NestedTensor": Nested, "HAS_NESTED_TENSOR": True}
        exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names], type_ignores=[]), str(source), "exec"), namespace)
        video = torch.randn(1, 2, 3, 4, 4, dtype=torch.bfloat16)
        audio = torch.randn(1, 2, 3, 4)
        latent = {"samples": Nested((video, audio)), "noise_mask": Nested((torch.ones_like(video), torch.zeros_like(audio))), "batch_index": [0]}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = os.path.join(folder, "pass1.pt")
            saved = namespace["VRGDG_MiniMaxH3SaveDraft"]().save(latent, checkpoint)[0]
            self.assertIs(saved, latent)
            loaded = namespace["VRGDG_MiniMaxH3LoadDraft"]().load(checkpoint)[0]
            self.assertTrue(torch.equal(loaded["samples"].tensors[0], video))
            self.assertTrue(torch.equal(loaded["samples"].tensors[1], audio))
            self.assertEqual(loaded["samples"].tensors[0].dtype, video.dtype)
            self.assertTrue(torch.equal(loaded["noise_mask"].tensors[1], torch.zeros_like(audio)))
            self.assertEqual(loaded["batch_index"], [0])


if __name__ == "__main__":
    unittest.main()
