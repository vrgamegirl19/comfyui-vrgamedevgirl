import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from array import array
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from builder_source import read_builder_source, read_runner_source


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = read_builder_source()
RUNNER_SOURCE = read_runner_source()

_SPEC = importlib.util.spec_from_file_location(
    "vrgdg_minimax_h3_latent_manager", ROOT / "minimax/latent_manager.py"
)
MANAGER = importlib.util.module_from_spec(_SPEC)
# Dataclasses resolve their string annotations through sys.modules, as a normal import would register the module.
sys.modules[_SPEC.name] = MANAGER
_SPEC.loader.exec_module(MANAGER)


class UnsafeLatentPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (self.marker,)


class FakeNestedTensor:
    """Stands in for comfy.nested_tensor.NestedTensor, which needs a full ComfyUI install."""

    def __init__(self, tensors):
        self.tensors = tuple(tensors)


def _fake_common_upscale(samples, width, height, upscale_method, crop):
    return torch.nn.functional.interpolate(samples, size=(height, width), mode="bicubic", align_corners=False)


class FakeVideoVae:
    """Records how it is used: decode -> pictures of 16 x the latent size, encode -> one latent token per H3 group."""

    def __init__(self):
        self.calls = []

    def decode(self, latent):
        tokens = int(latent.shape[2])
        frames = 17 * ((tokens - 2) // 5) + 5 if tokens > 2 else 5
        self.calls.append(("decode", tuple(latent.shape)))
        return torch.full((frames, int(latent.shape[3]) * 16, int(latent.shape[4]) * 16, 3), 0.5)

    def encode(self, pictures):
        frames, height, width = int(pictures.shape[0]), int(pictures.shape[1]), int(pictures.shape[2])
        tokens = 2 + 5 * ((frames - 5) // 17)
        self.calls.append(("encode", tuple(pictures.shape)))
        return torch.full((1, 24, tokens, height // 16, width // 16), 9.0)


def _loader_classes():
    module = ast.parse((ROOT / "minimax/latent_continuation.py").read_text(encoding="utf-8"))
    names = {"VRGDG_MiniMaxH3LoadLatent", "VRGDG_MiniMaxH3LoadExactFrame", "VRGDG_MiniMaxH3ApplyMaskedContinuation"}
    body = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = {
        "os": os, "hashlib": hashlib, "torch": torch, "Any": Any,
        "SceneLatentManager": MANAGER.SceneLatentManager,
        "plan_masked_context": MANAGER.plan_masked_context,
        "comfy": types.SimpleNamespace(utils=types.SimpleNamespace(common_upscale=_fake_common_upscale)),
        "HAS_NESTED_TENSOR": True,
        "NestedTensor": FakeNestedTensor,
        "_require_masked_av_support": lambda: None,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "latent_loaders", "exec"), namespace)
    return namespace


def _runner_namespace():
    """The latent-continuation helpers of the workflow runner, executed in isolation."""
    module = ast.parse(RUNNER_SOURCE)
    wanted = {
        "_int_payload",
        "_first_payload_value",
        "_api_node_id_by_class",
        "_minimax_h3_latent_continuation_mode",
        "_minimax_h3_is_latent_mode",
        "_minimax_h3_latent_context_frames_setting",
        "_minimax_h3_tail_padding_frames",
        "_minimax_h3_effective_warmup_frames",
        "_minimax_h3_cooldown_frames",
        "_patch_minimax_h3_latent_continuation",
        "_patch_minimax_h3_latent_continuation_masked",
        "_minimax_h3_masked_latent_plan",
    }
    constants = {"_MMH3_LATENT_MASKED_MODE", "_MMH3_RETIRED_LATENT_MODES", "_MMH3_CONDITIONING_CLASSES"}
    body = [
        node for node in module.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in constants)
    ]
    namespace = {
        "json": json,
        "os": os,
        "SceneLatentManager": MANAGER.SceneLatentManager,
        "plan_latent_context": MANAGER.plan_latent_context,
        "plan_masked_context": MANAGER.plan_masked_context,
        "normalize_masked_context_frames": MANAGER.normalize_masked_context_frames,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "latent_continuation_runner", "exec"), namespace)
    return namespace


class LatentContextPlanTests(unittest.TestCase):
    def test_plain_plan_takes_the_tail_of_the_latent(self):
        plan = MANAGER.plan_latent_context(72, None, 22, exact_frame=False)
        self.assertEqual((plan["start_token"], plan["end_token"]), (65, 72))
        self.assertEqual(plan["warmup_frames"], 22)
        self.assertIsNone(plan["image_frame_offset"])

    def test_exact_plan_snaps_to_the_token_grid_and_lands_on_the_last_visible_frame(self):
        total = 72
        for padding in range(0, 17):
            for context_frames in (22, 39, 56):
                plan = MANAGER.plan_latent_context(total, padding, context_frames, exact_frame=True)
                visible = MANAGER._tokens_to_frames(total) - padding
                first_context_frame = sum(
                    MANAGER._FRAME_PER_TOKEN[k % 5] for k in range(plan["start_token"])
                )
                with self.subTest(padding=padding, context_frames=context_frames):
                    self.assertTrue(plan["exact"])
                    self.assertEqual(plan["start_token"] % 5, 0)
                    self.assertLessEqual(plan["end_token"], total)
                    # the exact image sits on the predecessor's real last frame
                    self.assertEqual(plan["image_frame_offset"], visible - 1 - first_context_frame)
                    # and the context never overlaps that frame
                    self.assertLess(plan["context_frames"], plan["warmup_frames"])

    def test_exact_plan_falls_back_when_the_latent_is_too_short(self):
        plan = MANAGER.plan_latent_context(3, 0, 22, exact_frame=True)
        self.assertFalse(plan["exact"])


class MaskedContextPlanTests(unittest.TestCase):
    def test_window_is_phase_aligned_and_exact_for_every_size_and_padding(self):
        total = 72
        for padding in range(0, 40):
            for frames in MANAGER.MASKED_CONTEXT_FRAMES:
                try:
                    plan = MANAGER.plan_masked_context(total, padding, frames)
                except ValueError:
                    self.assertLess(MANAGER._tokens_to_frames(total) - padding, frames - 17)
                    continue
                visible = MANAGER._tokens_to_frames(total) - padding
                with self.subTest(padding=padding, frames=frames):
                    self.assertEqual(plan["start_token"] % 5, 0)
                    self.assertEqual(plan["end_token"] % 5, 2)
                    self.assertEqual(plan["context_frames"], frames)
                    # the head reaches the predecessor's last visible frame, so no frame is left to regenerate
                    self.assertGreaterEqual(plan["end_frame"], visible)
                    self.assertEqual(plan["lost_tail_frames"], 0)
                    self.assertLess(plan["head_tail_frames"], 17)
                    # what the render trims: the head minus the real frames of it that lie past the visible end
                    self.assertEqual(plan["warmup_frames"], frames - plan["head_tail_frames"])
                    self.assertEqual(plan["end_frame"] - visible, plan["head_tail_frames"])

    def test_39_frames_is_12_tokens_and_65_audio_ticks(self):
        plan = MANAGER.plan_masked_context(72, 0, 39)
        self.assertEqual(plan["tokens"], 12)
        self.assertEqual(plan["end_audio_tick"] - plan["start_audio_tick"], 65)

    def test_window_ends_on_the_boundary_at_or_after_the_visible_end(self):
        plan = MANAGER.plan_masked_context(72, 5, 39)
        self.assertEqual((plan["start_token"], plan["end_token"]), (60, 72))
        self.assertEqual((plan["head_tail_frames"], plan["lost_tail_frames"]), (5, 0))
        self.assertEqual(plan["warmup_frames"], 34)

    def test_warmup_leaves_the_first_visible_frame_on_the_predecessor_next_frame(self):
        # measured scene 9 -> 10: 107 frames, 1 padding -> 106 visible. The old window ended at frame 90 and left 16
        # frames to regenerate (a pop at the join). The head now ends on frame 107 (one real cool-down frame).
        plan = MANAGER.plan_masked_context(32, 1, 39)
        self.assertEqual((plan["start_frame"], plan["end_frame"]), (68, 107))
        self.assertEqual((plan["lost_tail_frames"], plan["head_tail_frames"], plan["warmup_frames"]), (0, 1, 38))
        # render frame 38 is source frame 106, the frame right after the predecessor's last visible frame (105)
        self.assertEqual(plan["start_frame"] + plan["warmup_frames"], 106)
        # scene 10 -> 11: 175 frames, 1 padding
        plan = MANAGER.plan_masked_context(52, 1, 39)
        self.assertEqual(plan["start_frame"] + plan["warmup_frames"], 175 - 1)

    def test_unknown_sizes_fall_back_to_39_and_short_latents_are_rejected(self):
        self.assertEqual(MANAGER.normalize_masked_context_frames(22), 39)
        self.assertEqual(MANAGER.normalize_masked_context_frames("141"), 141)
        with self.assertRaisesRegex(ValueError, "too short"):
            MANAGER.plan_masked_context(7, 0, 39)


class MaskedContinuationNodeTests(unittest.TestCase):
    def setUp(self):
        self.classes = _loader_classes()
        self.node = self.classes["VRGDG_MiniMaxH3ApplyMaskedContinuation"]()

    @staticmethod
    def _target(video_tokens=40, audio_ticks=200, mask=None):
        video = torch.zeros(1, 24, video_tokens, 4, 6)
        audio = torch.ones(1, 32, 2, audio_ticks)
        latent = {"samples": FakeNestedTensor((video, audio))}
        if mask is not None:
            latent["noise_mask"] = mask
        return latent

    @staticmethod
    def _context(tokens=12, ticks=65, spatial=(4, 6)):
        video = torch.full((1, 24, tokens, *spatial), 7.0)
        audio = torch.full((1, 32, 2, ticks), 3.0)
        return {"samples": FakeNestedTensor((video, audio)), "video": video, "audio": audio}

    def test_head_is_copied_and_masked_and_the_rest_is_generated(self):
        (out,) = self.node.apply(self._target(), self._context())
        video, audio = out["samples"].tensors
        video_mask, audio_mask = out["noise_mask"].tensors
        self.assertTrue(torch.all(video[:, :, :12] == 7.0))
        self.assertTrue(torch.all(video[:, :, 12:] == 0.0))
        self.assertTrue(torch.all(video_mask[:, :, :12] == 0.0))
        self.assertTrue(torch.all(video_mask[:, :, 12:] == 1.0))
        # audio is generated untouched unless include_audio is set
        self.assertTrue(torch.all(audio == 1.0))
        self.assertTrue(torch.all(audio_mask == 1.0))

    def test_audio_drive_mask_is_kept_and_audio_is_not_replaced(self):
        base = self._target()
        video, audio = base["samples"].tensors
        drive_mask = FakeNestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
        (out,) = self.node.apply(self._target(mask=drive_mask), self._context())
        video_mask, audio_mask = out["noise_mask"].tensors
        self.assertTrue(torch.all(audio_mask == 0.0))
        self.assertTrue(torch.all(video_mask[:, :, :12] == 0.0))
        self.assertTrue(torch.all(video_mask[:, :, 12:] == 1.0))

    def test_include_audio_copies_and_protects_the_predecessor_audio(self):
        (out,) = self.node.apply(self._target(), self._context(), include_audio=True)
        audio = out["samples"].tensors[1]
        audio_mask = out["noise_mask"].tensors[1]
        self.assertTrue(torch.all(audio[..., :65] == 3.0))
        self.assertTrue(torch.all(audio[..., 65:] == 1.0))
        self.assertTrue(torch.all(audio_mask[..., :65] == 0.0))
        self.assertTrue(torch.all(audio_mask[..., 65:] == 1.0))

    def test_mismatched_resolution_is_resized_and_oversized_context_is_rejected(self):
        (out,) = self.node.apply(self._target(), self._context(spatial=(8, 12)))
        self.assertEqual(tuple(out["samples"].tensors[0].shape), (1, 24, 40, 4, 6))
        with self.assertRaisesRegex(ValueError, "fills the whole"):
            self.node.apply(self._target(video_tokens=12), self._context())


class ShotPromptDecimalTests(unittest.TestCase):
    def test_decimal_seconds_survive_the_negative_sentence_filter(self):
        spec = importlib.util.spec_from_file_location("vrgdg_shot_prompt_decimal", ROOT / "minimax/shot_prompt.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        text = "For the first 1.5 seconds, he keeps walking. At about 1.5 seconds, he points. He does not cut."
        self.assertEqual(
            module.strip_negative_sentences(text),
            "For the first 1.5 seconds, he keeps walking. At about 1.5 seconds, he points.",
        )


class MaskedHeadResizeTests(unittest.TestCase):
    def setUp(self):
        self.node = _loader_classes()["VRGDG_MiniMaxH3ApplyMaskedContinuation"]()

    @staticmethod
    def _latents(target_hw, context_hw, tokens=12, target_tokens=40):
        target = {"samples": FakeNestedTensor((torch.zeros(1, 24, target_tokens, *target_hw), torch.ones(1, 32, 2, 200)))}
        video = torch.full((1, 24, tokens, *context_hw), 7.0)
        context = {"samples": FakeNestedTensor((video, torch.zeros(1, 32, 2, 65))), "video": video, "audio": None}
        return target, context

    def test_a_size_mismatch_goes_through_pictures_when_a_vae_is_connected(self):
        target, context = self._latents((56, 100), (52, 96))
        vae = FakeVideoVae()
        (out,) = self.node.apply(target, context, vae=vae)
        self.assertEqual([call[0] for call in vae.calls], ["decode", "encode"])
        self.assertEqual(vae.calls[0][1], (1, 24, 12, 52, 96))
        # the pictures were resized to the target before they were encoded
        self.assertEqual(vae.calls[1][1], (39, 56 * 16, 100 * 16, 3))
        video = out["samples"].tensors[0]
        self.assertEqual(tuple(video.shape), (1, 24, 40, 56, 100))
        self.assertTrue(torch.all(video[:, :, :12] == 9.0))
        self.assertTrue(torch.all(out["noise_mask"].tensors[0][:, :, :12] == 0.0))

    def test_a_matching_size_never_touches_the_vae(self):
        target, context = self._latents((56, 100), (56, 100))
        vae = FakeVideoVae()
        (out,) = self.node.apply(target, context, vae=vae)
        self.assertEqual(vae.calls, [])
        self.assertTrue(torch.all(out["samples"].tensors[0][:, :, :12] == 7.0))

    def test_without_a_vae_the_latent_is_interpolated_as_before(self):
        target, context = self._latents((56, 100), (52, 96))
        (out,) = self.node.apply(target, context)
        self.assertEqual(tuple(out["samples"].tensors[0].shape), (1, 24, 40, 56, 100))


class SceneLatentStorageTests(unittest.TestCase):
    def _save(self, folder, scene, tokens=12, **kwargs):
        latent = {"video": torch.zeros(1, 24, tokens, 4, 4), "audio": torch.zeros(1, 32, 2, 8)}
        return MANAGER.SceneLatentManager.save_latent(folder, scene, latent, **kwargs)

    def test_tail_padding_round_trips_through_the_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            self._save(folder, 3, metadata={"tail_padding_frames": 7})
            info = MANAGER.SceneLatentManager.get_latent_info(folder, 3)
            self.assertEqual(info["tail_padding_frames"], 7)
            self.assertEqual(info["token_count"], 12)

    def test_torch_fallback_round_trips_tensors_and_metadata_safely(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(MANAGER, "HAS_SAFETENSORS", False):
            self._save(folder, 1, metadata={"tail_padding_frames": 7})
            loaded = MANAGER.SceneLatentManager.load_latent(folder, 1)
            self.assertEqual(loaded["video"].shape, (1, 24, 12, 4, 4))
            self.assertEqual(loaded["audio"].shape, (1, 32, 2, 8))
            self.assertEqual(loaded["metadata"]["tail_padding_frames"], "7")

    def test_pickle_fallback_does_not_execute_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = os.path.join(folder, "pickle_executed")
            path = MANAGER.SceneLatentManager.get_path(folder, 1)
            torch.save(UnsafeLatentPayload(marker), path)
            self.assertIsNone(MANAGER.SceneLatentManager.load_latent(folder, 1))
            self.assertFalse(os.path.exists(marker))

    def test_latents_saved_without_padding_report_it_as_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            self._save(folder, 3)
            self.assertIsNone(MANAGER.SceneLatentManager.get_latent_info(folder, 3)["tail_padding_frames"])

    def test_saving_a_scene_marks_an_existing_successor_dirty(self):
        with tempfile.TemporaryDirectory() as folder:
            self._save(folder, 2)
            self._save(folder, 1)
            self.assertTrue(MANAGER.SceneLatentManager.is_dirty(folder, 2))
            self._save(folder, 2)
            self.assertFalse(MANAGER.SceneLatentManager.is_dirty(folder, 2))

    def test_deleting_a_scene_latent_keeps_the_numbering_of_the_others(self):
        with tempfile.TemporaryDirectory() as folder:
            for scene in (1, 2, 3):
                self._save(folder, scene)
            MANAGER.SceneLatentManager.delete_latent(folder, 2)
            self.assertFalse(MANAGER.SceneLatentManager.latent_exists(folder, 2))
            self.assertTrue(MANAGER.SceneLatentManager.latent_exists(folder, 1))
            self.assertTrue(MANAGER.SceneLatentManager.latent_exists(folder, 3))

    def test_making_room_for_a_scene_shifts_it_and_later_latents_up(self):
        with tempfile.TemporaryDirectory() as folder:
            for scene in (1, 2, 3):
                self._save(folder, scene, tokens=10 + scene)
            MANAGER.SceneLatentManager.make_room_for_scene(folder, 2)
            self.assertFalse(MANAGER.SceneLatentManager.latent_exists(folder, 2))
            self.assertEqual(MANAGER.SceneLatentManager.get_latent_info(folder, 1)["token_count"], 11)
            self.assertEqual(MANAGER.SceneLatentManager.get_latent_info(folder, 3)["token_count"], 12)
            self.assertEqual(MANAGER.SceneLatentManager.get_latent_info(folder, 4)["token_count"], 13)

    def test_delete_all_removes_only_scene_latent_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for scene in (1, 2):
                self._save(folder, scene)
            keep = Path(folder) / "latents" / "notes.txt"
            keep.write_text("keep", encoding="utf-8")
            self.assertGreater(MANAGER.SceneLatentManager.delete_all_latents(folder), 0)
            self.assertEqual([p.name for p in (Path(folder) / "latents").iterdir()], ["notes.txt"])


class LatentLoaderCacheTests(unittest.TestCase):
    def test_overwritten_files_change_both_loader_fingerprints(self):
        classes = _loader_classes()
        with tempfile.TemporaryDirectory() as folder:
            cases = [
                (Path(MANAGER.SceneLatentManager.get_path(folder, 1)),
                 lambda: classes["VRGDG_MiniMaxH3LoadLatent"].IS_CHANGED(folder, 1, 22)),
                (Path(folder) / "last.png",
                 lambda: classes["VRGDG_MiniMaxH3LoadExactFrame"].IS_CHANGED(str(Path(folder) / "last.png"))),
            ]
            for path, fingerprint in cases:
                with self.subTest(loader=path.name):
                    path.write_bytes(b"first render")
                    stat = path.stat()
                    first = fingerprint()
                    self.assertEqual(first, fingerprint())
                    path.write_bytes(b"other render")
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    self.assertNotEqual(first, fingerprint())


class LatentWarmupTimingTests(unittest.TestCase):
    def test_missing_audio_handles_keep_the_full_warmup(self):
        for source_start in (0, 0.25, 2):
            with self.subTest(source_start=source_start):
                plan = MANAGER.calculate_minimax_h3_timing(
                    10, 15, 22, source_start_seconds=source_start,
                    source_duration_seconds=source_start + 5, pad_warmup=True,
                )
                self.assertAlmostEqual(plan.final_trim_start_seconds, 22 / 24)
                self.assertAlmostEqual(plan.audio_leading_padding_seconds, max(0, 22 / 24 - source_start))
                self.assertAlmostEqual(plan.audio_trim_start_seconds, max(0, source_start - 22 / 24))
                self.assertAlmostEqual(plan.audio_trim_duration_seconds, 5 + 22 / 24)
                self.assertEqual(plan.final_trim_duration_seconds, 5)

    def test_non_latent_timing_still_clamps_the_audio_handle(self):
        plan = MANAGER.calculate_minimax_h3_timing(10, 15, 22, source_start_seconds=0)
        self.assertEqual(plan.final_trim_start_seconds, 0)
        self.assertEqual(plan.audio_leading_padding_seconds, 0)
        self.assertEqual(plan.audio_trim_duration_seconds, 5)

    def test_padded_warmup_still_obeys_h3_frame_limit(self):
        with self.assertRaisesRegex(ValueError, "exceeding"):
            MANAGER.calculate_minimax_h3_timing(10, 25, 22, source_start_seconds=0, pad_warmup=True)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for audio alignment verification")
    def test_trimmed_audio_contains_silence_then_source_at_the_planned_offset(self):
        module = ast.parse(RUNNER_SOURCE)
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_trim_minimax_h3_audio_context")
        namespace = {"os": os, "subprocess": subprocess, "wave": wave, "_find_ffmpeg_path": lambda: shutil.which("ffmpeg")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "audio_context", "exec"), namespace)
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "source.wav")
            with wave.open(source, "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(array("h", [1200, -1200] * (44100 * 6)).tobytes())
            for source_start in (0, 0.25, 2):
                with self.subTest(source_start=source_start):
                    plan = MANAGER.calculate_minimax_h3_timing(
                        10, 11, 22, source_start_seconds=source_start,
                        source_duration_seconds=6, pad_warmup=True,
                    )
                    result = namespace["_trim_minimax_h3_audio_context"](source, folder, 2, plan)
                    with wave.open(result["audio_path"], "rb") as handle:
                        samples = array("h", handle.readframes(handle.getnframes()))
                    padding_samples = round(plan.audio_leading_padding_seconds * 44100) * 2
                    self.assertTrue(all(sample == 0 for sample in samples[:max(0, padding_samples - 2)]))
                    self.assertEqual(list(samples[padding_samples + 2:padding_samples + 4]), [1200, -1200])
                    scene_start = round(plan.final_trim_start_seconds * 44100) * 2
                    self.assertEqual(list(samples[scene_start:scene_start + 2]), [1200, -1200])
                    self.assertAlmostEqual(result["duration"], plan.audio_trim_duration_seconds, delta=1 / 44100)


class RunnerLatentContinuationTests(unittest.TestCase):
    def setUp(self):
        self.ns = _runner_namespace()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        MANAGER.SceneLatentManager.save_latent(
            self.folder.name,
            21,
            {"video": torch.zeros(1, 24, 72, 4, 4), "audio": torch.zeros(1, 32, 2, 8)},
            metadata={"tail_padding_frames": 5},
        )
        self.image = Path(self.folder.name) / "last.png"
        self.image.write_bytes(b"png")
        self.payload = {
            "project_folder": self.folder.name,
            "scene_number": 22,
            "latent_context_frames": 22,
            "pre_frames": 0,
        }

    def test_the_retired_modes_continue_masked(self):
        mode = self.ns["_minimax_h3_latent_continuation_mode"]
        for name in ("Latent Continuation", "latent", "continuation", "latent-exact", "latent_continuation_exact_frame"):
            with self.subTest(name=name):
                self.assertEqual(mode({"continuity_mode": name}), "latent_continuation_masked")
        self.assertEqual(mode({"continuity_mode": "off"}), "off")

    def test_warmup_covers_the_head_only_when_continuation_is_active(self):
        warmup = self.ns["_minimax_h3_effective_warmup_frames"]
        self.assertEqual(warmup({**self.payload, "continuity_mode": "off"}), 0)
        self.assertEqual(warmup({**self.payload, "continuity_mode": "latent_continuation"}), 34)
        self.assertEqual(warmup({**self.payload, "continuity_mode": "latent_continuation", "scene_number": 1}), 0)
        # the Render settings warm-up and cool-down do not apply to masked continuation
        self.assertEqual(warmup({**self.payload, "continuity_mode": "latent_continuation", "pre_frames": 40}), 34)
        self.assertEqual(warmup({**self.payload, "continuity_mode": "off", "pre_frames": 40}), 40)
        cooldown = self.ns["_minimax_h3_cooldown_frames"]
        self.assertEqual(cooldown({"continuity_mode": "latent_continuation_masked", "cooldown_frames": 12}), 0)
        self.assertEqual(cooldown({"continuity_mode": "off", "cooldown_frames": 12}), 12)
        self.assertEqual(cooldown({"continuity_mode": "off", "tail_loss_frames": 7}), 7)

    def test_masked_mode_names_and_warmup(self):
        mode = self.ns["_minimax_h3_latent_continuation_mode"]
        self.assertEqual(mode({"continuity_mode": "Latent Continuation Masked"}), "latent_continuation_masked")
        self.assertEqual(mode({"continuity_mode": "latent-masked"}), "latent_continuation_masked")
        self.assertTrue(self.ns["_minimax_h3_is_latent_mode"]("latent_continuation_masked"))
        warmup = self.ns["_minimax_h3_effective_warmup_frames"]
        masked = {**self.payload, "continuity_mode": "latent_continuation_masked"}
        # the 39 head frames minus the 5 real cool-down frames of them past the visible end (the 22 frame default is not a masked size)
        self.assertEqual(warmup(masked), 34)
        self.assertEqual(warmup({**masked, "latent_context_frames": 90}), 85)
        self.assertEqual(warmup({**masked, "scene_number": 1}), 0)

    @staticmethod
    def _single_pass_prompt(audio_drive=True):
        prompt = {
            "136": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {}},
            "125": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["126", 0], "latent_image": ["172", 0]}},
            "126": {"class_type": "BasicGuider", "inputs": {"conditioning": ["136", 0]}},
        }
        if audio_drive:
            prompt["172"] = {"class_type": "VRGDG_MiniMaxH3AudioDrive", "inputs": {"av_latent": ["136", 1]}}
        else:
            prompt["125"]["inputs"]["latent_image"] = ["136", 1]
        return prompt

    def test_masked_mode_puts_the_head_in_the_sampler_latent_and_leaves_conditioning_alone(self):
        prompt = self._single_pass_prompt()
        result = self.ns["_patch_minimax_h3_latent_continuation"](
            prompt, {**self.payload, "continuity_mode": "latent_continuation_masked"}
        )
        self.assertTrue(result["enabled"])
        self.assertEqual(prompt["125"]["inputs"]["latent_image"], [result["apply_node_id"], 0])
        apply = prompt[result["apply_node_id"]]
        self.assertEqual(apply["class_type"], "VRGDG_MiniMaxH3ApplyMaskedContinuation")
        self.assertEqual(apply["inputs"]["latent"], ["172", 0])
        self.assertFalse(apply["inputs"]["include_audio"])
        load = prompt[result["load_node_id"]]["inputs"]
        self.assertTrue(load["masked_av"])
        self.assertEqual((load["scene_number"], load["context_frames"]), (21, 39))
        self.assertEqual(result["warmup_frames"], 34)
        self.assertEqual(prompt["126"]["inputs"]["conditioning"], ["136", 0])
        self.assertNotIn("VRGDG_MiniMaxH3ApplyLatentGuide", {node["class_type"] for node in prompt.values()})

    def test_masked_mode_carries_the_predecessor_audio_only_without_audio_drive(self):
        prompt = self._single_pass_prompt(audio_drive=False)
        result = self.ns["_patch_minimax_h3_latent_continuation"](
            prompt, {**self.payload, "continuity_mode": "latent_continuation_masked"}
        )
        self.assertEqual(prompt[result["apply_node_id"]]["inputs"]["latent"], ["136", 1])
        self.assertTrue(prompt[result["apply_node_id"]]["inputs"]["include_audio"])

    @staticmethod
    def _two_pass_prompt():
        prompt = RunnerLatentContinuationTests._single_pass_prompt()
        prompt.update({
            "181": {"class_type": "MiniMaxH3AVLatentSeparateT8", "inputs": {"av_latent": ["125", 0]}},
            "189": {
                "class_type": "VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent",
                "inputs": {"original_av_latent": ["125", 0], "upscaled_video_latent": ["182", 0]},
            },
            "194": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["193", 0], "latent_image": ["189", 0]}},
        })
        return prompt

    def test_masked_mode_applies_the_head_again_to_the_second_pass(self):
        prompt = self._two_pass_prompt()
        result = self.ns["_patch_minimax_h3_latent_continuation"](
            prompt, {**self.payload, "continuity_mode": "latent_continuation_masked"}
        )
        # pass 1 is protected exactly as in single pass
        self.assertEqual(prompt["125"]["inputs"]["latent_image"], [result["apply_node_id"], 0])
        self.assertEqual(prompt[result["apply_node_id"]]["inputs"]["latent"], ["172", 0])
        # pass 2 reads the upscaled latent, whose video mask the upscale resets, so the head goes in again
        second = prompt[result["second_pass_apply_node_id"]]
        self.assertEqual(second["class_type"], "VRGDG_MiniMaxH3ApplyMaskedContinuation")
        self.assertEqual(second["inputs"]["latent"], ["189", 0])
        self.assertEqual(second["inputs"]["context_latent"], [result["load_node_id"], 0])
        self.assertFalse(second["inputs"]["include_audio"])
        self.assertEqual(prompt["194"]["inputs"]["latent_image"], [result["second_pass_apply_node_id"], 0])
        # one predecessor window feeds both passes
        loads = [n for n in prompt.values() if n["class_type"] == "VRGDG_MiniMaxH3LoadLatent"]
        self.assertEqual(len(loads), 1)

    def test_masked_mode_connects_the_video_vae_to_both_apply_nodes(self):
        prompt = self._two_pass_prompt()
        prompt["136"]["inputs"]["vae"] = ["119", 0]
        result = self.ns["_patch_minimax_h3_latent_continuation"](
            prompt, {**self.payload, "continuity_mode": "latent_continuation_masked"}
        )
        self.assertEqual(prompt[result["apply_node_id"]]["inputs"]["vae"], ["119", 0])
        self.assertEqual(prompt[result["second_pass_apply_node_id"]]["inputs"]["vae"], ["119", 0])
        # no video VAE in the graph: the input is simply left off
        bare = self._single_pass_prompt()
        result = self.ns["_patch_minimax_h3_latent_continuation"](
            bare, {**self.payload, "continuity_mode": "latent_continuation_masked"}
        )
        self.assertNotIn("vae", bare[result["apply_node_id"]]["inputs"])

    def test_masked_mode_works_on_the_image_to_video_node_and_drops_its_first_frame_keyframe(self):
        payload = {**self.payload, "continuity_mode": "latent_continuation_masked"}
        prompt = self._two_pass_prompt()
        prompt["136"] = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"vae": ["119", 0], "first_frame": ["180", 0], "last_frame": ["180", 1]},
        }
        result = self.ns["_patch_minimax_h3_latent_continuation"](prompt, payload)
        self.assertTrue(result["enabled"])
        self.assertTrue(result["dropped_first_frame"])
        # the head replaces the opening keyframe, the closing keyframe stays
        self.assertNotIn("first_frame", prompt["136"]["inputs"])
        self.assertEqual(prompt["136"]["inputs"]["last_frame"], ["180", 1])
        # the video VAE is found through the image node, so a resized head can go through pictures
        self.assertEqual(prompt[result["apply_node_id"]]["inputs"]["vae"], ["119", 0])
        self.assertEqual(prompt[result["second_pass_apply_node_id"]]["inputs"]["vae"], ["119", 0])
        # Reference to Video has no first frame input to drop
        prompt = self._single_pass_prompt()
        self.assertFalse(self.ns["_patch_minimax_h3_latent_continuation"](prompt, payload)["dropped_first_frame"])

    def test_masked_mode_is_refused_for_2_pass_advanced(self):
        # 2 Pass Advanced samples pass 2 inside MMH3 Ultimate Upscale, which builds its own masks
        prompt = self._single_pass_prompt()
        prompt["9306"] = {"class_type": "MMH3UltimateUpscale", "inputs": {"latent": ["125", 0]}}
        with self.assertRaisesRegex(ValueError, "not available in 2 Pass Advanced"):
            self.ns["_patch_minimax_h3_latent_continuation"](
                prompt, {**self.payload, "continuity_mode": "latent_continuation_masked"}
            )
        self.assertEqual(prompt["125"]["inputs"]["latent_image"], ["172", 0])

    def test_masked_mode_rejects_unmappable_graphs_and_missing_predecessors(self):
        payload = {**self.payload, "continuity_mode": "latent_continuation_masked"}
        # two samplers that are not a first pass plus an upscaled second pass
        prompt = self._single_pass_prompt()
        prompt["194"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["189", 0]}}
        with self.assertRaisesRegex(ValueError, "supports Single pass and 2 Pass renders"):
            self.ns["_patch_minimax_h3_latent_continuation"](prompt, payload)
        # three samplers
        prompt = self._two_pass_prompt()
        prompt["300"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["194", 0]}}
        with self.assertRaisesRegex(ValueError, "supports Single pass and 2 Pass renders"):
            self.ns["_patch_minimax_h3_latent_continuation"](prompt, payload)
        with self.assertRaises(FileNotFoundError):
            self.ns["_patch_minimax_h3_latent_continuation"](
                self._single_pass_prompt(), {**payload, "scene_number": 30}
            )

    def test_all_builder_timing_calls_preserve_latent_warmup(self):
        calls = [
            node for node in ast.walk(ast.parse(RUNNER_SOURCE))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "calculate_minimax_h3_timing"
        ]
        self.assertEqual(len(calls), 3)  # advanced two-pass reuses the two-pass builder
        for call in calls:
            for mode in ("off", "latent_continuation", "latent_continuation_masked"):
                for scene in (1, 22):
                    with self.subTest(line=call.lineno, mode=mode, scene=scene):
                        payload = {**self.payload, "continuity_mode": mode, "scene_number": scene}
                        warmup = self.ns["_minimax_h3_effective_warmup_frames"](payload)
                        namespace = {**self.ns, "calculate_minimax_h3_timing": MANAGER.calculate_minimax_h3_timing,
                                     "payload": payload, "scene_number": scene, "timeline_start": 10,
                                     "timeline_end": 15, "source_start": 0, "source_duration": 5,
                                     "warmup_frames": warmup, "cooldown_frames": 0, "audio_mode": "input_audio"}
                        timing = eval(compile(ast.Expression(call), "builder_timing", "eval"), namespace)
                        if mode != "off" and scene > 1:
                            prompt = self._single_pass_prompt()
                            guide = self.ns["_patch_minimax_h3_latent_continuation"](prompt, payload)
                            self.assertAlmostEqual(timing.final_trim_start_seconds, guide["warmup_frames"] / 24)
                            self.assertEqual(timing.audio_leading_padding_seconds, timing.final_trim_start_seconds)
                        else:
                            self.assertEqual(timing.audio_leading_padding_seconds, 0)

    def test_missing_predecessor_latent_is_a_clear_error(self):
        payload = {**self.payload, "scene_number": 30, "continuity_mode": "latent_continuation_masked"}
        with self.assertRaises(FileNotFoundError):
            self.ns["_patch_minimax_h3_latent_continuation"](self._single_pass_prompt(), payload)

    def test_tail_padding_is_the_render_length_minus_the_visible_scene(self):
        padding = self.ns["_minimax_h3_tail_padding_frames"]
        self.assertEqual(
            padding({"actual_warmup_seconds": 0.917, "final_frame_count": 174, "h3_frame_count": 209}), 13
        )
        self.assertIsNone(padding({}))

    def test_tail_padding_uses_the_stitched_frame_count(self):
        # 125.49s -> 135.43s is 238.56 frames; the stitcher keeps round(3250.32) - round(3011.76) = 238
        plan = MANAGER.calculate_minimax_h3_timing(125.49, 135.43, 0, 0)
        self.assertEqual(plan.final_frame_count, 238)
        self.assertEqual(self.ns["_minimax_h3_tail_padding_frames"](plan), plan.h3_frame_count - 238)


class BuilderLatentContinuationWiringTests(unittest.TestCase):
    def test_only_the_masked_mode_is_offered_and_the_retired_ones_normalise_to_it(self):
        self.assertIn('value: "latent_continuation_masked"', BUILDER_SOURCE)
        self.assertNotIn('value: "latent_continuation", label', BUILDER_SOURCE)
        self.assertNotIn('value: "latent_continuation_exact_frame"', BUILDER_SOURCE)
        self.assertIn('"latent", "latent_continuation", "continuation"].includes(clean)) return "latent_continuation_masked"', BUILDER_SOURCE)
        self.assertIn("function isMiniMaxH3LatentContinuationMode(mode)", BUILDER_SOURCE)

    def test_masked_mode_has_its_own_prompt_contract_and_is_not_limited_to_single_pass(self):
        # the masked head already holds the previous scene's motion, so the prompt must continue it, not transition
        self.assertIn("this scene is the very next moment of the same uninterrupted take", BUILDER_SOURCE)
        self.assertIn("LOCATION PHASE — MASKED CONTINUATION TRANSITION", BUILDER_SOURCE)
        self.assertIn('continuity_mode === "latent_continuation_masked"', BUILDER_SOURCE)
        # 2 Pass is allowed, the Builder no longer stops it before the graph is built. 2 Pass Advanced is not available
        self.assertNotIn("Latent Continuation Masked works with Single pass only", BUILDER_SOURCE)
        self.assertIn("2 Pass (the exact head is applied again in pass 2). Not available in 2 Pass Advanced.", BUILDER_SOURCE)

    def test_render_payload_carries_the_latent_settings(self):
        for key in (
            "continuity_mode: continuityInput?.continuityMode",
            "latent_context_frames: latentContextFrames",
        ):
            self.assertIn(key, BUILDER_SOURCE)

    def test_the_predecessor_frame_is_not_injected_as_a_reference_image(self):
        start = BUILDER_SOURCE.index("if (isMiniMaxH3LatentContinuationMode(continuityMode)) {")
        end = BUILDER_SOURCE.index("previousSegment,\n      };", start)
        block = BUILDER_SOURCE[start:end]
        self.assertIn('framePath: ""', block)
        self.assertIn("promptFramePath", block)
        self.assertNotIn("exactFramePath", block)

    def test_the_continuation_direction_box_is_in_the_prompting_tab_and_swaps_with_the_prompt_box(self):
        source = BUILDER_SOURCE.replace("\r\n", "\n")
        # it left Between-scene continuity and sits above the prompt box in the LLM Prompting tab
        self.assertNotIn("miniMaxLocationTransitionControls, miniMaxContinuationDirectionField", source)
        self.assertIn(
            'miniMaxPromptActions,\n        miniMaxContinuationDirectionField,\n        miniMaxContinuationStartField,\n        makeField("MiniMax H3 prompt", miniMaxPrompt),',
            source,
        )
        # the prompt box is off exactly when the render writes the prompt from the previous scene's final frame
        self.assertIn("const promptFromLastFrame = Boolean(segment) && miniMaxH3FrameContinuityPromptEnabled(segment);", source)
        self.assertIn("miniMaxPrompt.disabled = promptFromLastFrame;", source)
        self.assertIn("miniMaxContinuationDirection.disabled = !promptFromLastFrame;", source)
        # the pop-out mirrors the box and both states
        self.assertIn("const directionMirror = mirrorTextarea(direction, direction.placeholder);", source)
        self.assertIn("promptMirror.disabled = prompt.disabled;", source)
        self.assertIn("directionMirror.disabled = direction.disabled;", source)
        self.assertIn("direction: miniMaxContinuationDirection,", source)

    def test_dirty_badge_only_shows_on_scenes_that_use_latent_continuation(self):
        start = BUILDER_SOURCE.index("async function loadDirtyLatentBadges()")
        end = BUILDER_SOURCE.index("function openSceneOptions", start)
        loader = BUILDER_SOURCE[start:end]
        self.assertIn("isMiniMaxH3LatentContinuationMode(miniMaxH3ContinuityModeForSegment(seg))", loader)
        self.assertIn("dirtySet.has(slot) && usesLatentContinuation", loader)
        # changing a scene's continuity mode refreshes the badges straight away
        handler_start = BUILDER_SOURCE.index('miniMaxContinuityMode.addEventListener("change"')
        handler_end = BUILDER_SOURCE.index("miniMaxAddSpeakerCueButton.onclick", handler_start)
        self.assertIn("loadDirtyLatentBadges()", BUILDER_SOURCE[handler_start:handler_end])

    def test_deleting_a_video_removes_its_stale_latent(self):
        start = BUILDER_SOURCE.index("async function deleteSelectedMedia(")
        end = BUILDER_SOURCE.index("function sendPromptToEnhance", start)
        self.assertIn("deleteStaleSceneLatents(media.segment)", BUILDER_SOURCE[start:end])
        # a video delete must never renumber the other scenes' latents
        self.assertIn("payload.reindex = false", BUILDER_SOURCE)

    def test_the_timeline_card_has_a_quick_button_for_masked_continuation(self):
        source = BUILDER_SOURCE.replace("\r\n", "\n")
        # shown on every base scene except the first, which has no predecessor
        self.assertIn("if (!isOverlay && state.segments.indexOf(segment) > 0) {", source)
        self.assertIn("Promise.resolve(toggleSceneMaskedContinuation(segment)).catch", source)
        # it locks the scene and sets masked continuity with the Masked transition, only where masked is allowed
        self.assertIn("async function toggleSceneMaskedContinuation(segment) {", source)
        self.assertIn('isMiniMaxH3ContinuityAllowedForMode("latent_continuation_masked", base.video_mode, base.render_pass)', source)
        self.assertIn("segment.use_scene_minimax_h3_settings = true;", source)
        # clicking it again turns it off and restores what the scene had before
        self.assertIn("segment.minimax_h3_masked_quick_prev = {", source)
        self.assertIn("delete segment.minimax_h3_masked_quick_prev;", source)
        self.assertIn('continuity_mode: "latent_continuation_masked",\n      location_transition_preset: "masked",', source)


if __name__ == "__main__":
    unittest.main()
