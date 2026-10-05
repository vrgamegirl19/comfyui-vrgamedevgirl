"""MiniMax H3 Latent Continuation Custom Nodes for ComfyUI.

Provides dedicated nodes to save, load, inject, and trim MiniMax H3 latents
for lossless, native temporal chaining across multi-scene video projects.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import torch
import node_helpers
import comfy.model_management

from .latent_manager import SceneLatentManager, plan_latent_context, plan_masked_context

try:
    from comfy.nested_tensor import NestedTensor
    HAS_NESTED_TENSOR = True
except ImportError:
    HAS_NESTED_TENSOR = False


def _require_masked_av_support() -> None:
    """Fail clearly when ComfyUI predates the H3 per-token AV noise masks (PR 15375, ComfyUI 0.34.0)."""
    try:
        import comfy.ldm.minimax.model as h3_model
    except Exception as exc:
        raise RuntimeError(f"Latent Continuation Masked could not import ComfyUI's MiniMax H3 model: {exc}") from exc
    if not hasattr(h3_model, "mask_row_values"):
        raise RuntimeError(
            "Latent Continuation Masked needs a ComfyUI build with MiniMax H3 per-token AV noise masks "
            "(PR 15375, ComfyUI 0.34.0 or newer). Update ComfyUI and restart it."
        )


class VRGDG_MiniMaxH3SaveLatent:
    """Save a rendered scene's raw sampled AV latent tensor to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "project_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Project folder path (e.g. ComfyUI/output/MyProject)",
                }),
                "scene_number": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Scene number (1-based index)",
                }),
            },
            "optional": {
                "frame_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Optional explicit frame count (0 = auto-compute from latent tokens)",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                }),
                "tail_padding_frames": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999999,
                    "tooltip": (
                        "Frames at the end of this render that fall after the scene's visible last frame "
                        "(alignment padding / cool-down). -1 = unknown. Lets the next scene's exact-frame "
                        "continuation find the real last frame inside the latent."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "saved_path")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "VRGDG/MiniMax H3 Latent Continuation"
    DESCRIPTION = (
        "Serializes the raw MiniMax H3 sampled latent to <project_folder>/latents/scene_NNN.latent. "
        "Acts as a passthrough so it can be inserted seamlessly before VAE Decode."
    )

    def save(
        self,
        latent: dict[str, Any],
        project_folder: str = "",
        scene_number: int = 1,
        frame_count: int = 0,
        fps: float = 24.0,
        tail_padding_frames: int = -1,
    ) -> tuple[dict[str, Any], str]:
        saved_path = ""
        clean_folder = str(project_folder or "").strip().strip('"')
        if clean_folder:
            try:
                saved_path = SceneLatentManager.save_latent(
                    project_folder=clean_folder,
                    scene_number=scene_number,
                    samples_dict_or_tensor=latent,
                    frame_count=frame_count if frame_count > 0 else None,
                    fps=fps,
                    model_type="MiniMax-H3",
                    metadata={"tail_padding_frames": int(tail_padding_frames)} if tail_padding_frames >= 0 else None,
                )
            except Exception as exc:
                print(f"[VRGDG Latent Save] Warning: Failed to serialize scene {scene_number:03d}: {exc}")
        else:
            print(f"[VRGDG Latent Save] Skipped: No project_folder provided for scene {scene_number:03d}")

        return (latent, saved_path)


class VRGDG_MiniMaxH3LoadLatent:
    """Load a predecessor scene's latent and slice to the requested context frames window."""

    @classmethod
    def IS_CHANGED(cls, project_folder, scene_number, context_frames, exact_frame_mode=False, masked_av=False, run_after=None):
        folder = str(project_folder or "").strip().strip('"')
        path = SceneLatentManager.get_path(folder, scene_number)
        if not os.path.isfile(path):
            # A graph can save the predecessor in the same run (see run_after), so the file may not exist yet.
            return "missing"
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Project folder path",
                }),
                "scene_number": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Scene number to load (e.g. predecessor scene N-1)",
                }),
                "context_frames": ("INT", {
                    "default": 22,
                    "min": 1,
                    "max": 192,
                    "step": 1,
                    "tooltip": (
                        "Number of temporal context frames to slice from the tail (16, 22, 39, 56). "
                        "Masked mode uses 39, 90, 141 or 192."
                    ),
                }),
            },
            "optional": {
                "exact_frame_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Cut the context so it ends just before the predecessor's real last visible frame "
                        "(skipping any padding after it) on the model's 5-token grid, leaving that last frame "
                        "to be supplied as an exact image."
                    ),
                }),
                "masked_av": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Latent Continuation Masked: slice a phase-aligned run (starts on a 5-token boundary, ends "
                        "before the predecessor's padding) with exactly matching audio ticks, for "
                        "VRGDG H3 Apply Masked Continuation."
                    ),
                }),
                "run_after": ("*", {
                    "tooltip": (
                        "Optional. Connect the LATENT output of the Save Latent node that writes the predecessor in "
                        "the same graph, so this node loads the file only after it has been saved."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("context_latent", "sliced_tokens")
    FUNCTION = "load"
    CATEGORY = "VRGDG/MiniMax H3 Latent Continuation"
    DESCRIPTION = (
        "Loads scene_NNN.latent from disk, extracts the trailing context window based on the "
        "requested context frames, and returns the sliced latent tensor."
    )

    def load(
        self,
        project_folder: str = "",
        scene_number: int = 1,
        context_frames: int = 22,
        exact_frame_mode: bool = False,
        masked_av: bool = False,
        run_after: Any = None,
    ) -> tuple[dict[str, Any], int]:
        clean_folder = str(project_folder or "").strip().strip('"')
        if not clean_folder:
            raise ValueError("project_folder is required to load a scene latent")

        data = SceneLatentManager.load_latent(clean_folder, scene_number)
        if not data:
            raise FileNotFoundError(
                f"No latent found for scene {scene_number:03d} in {clean_folder}/latents/. "
                "Render the predecessor scene first."
            )

        video = data["video"]
        audio = data.get("audio")

        total_tokens = int(video.shape[2])
        raw_padding = (data.get("metadata") or {}).get("tail_padding_frames")
        try:
            tail_padding = int(float(raw_padding)) if raw_padding not in (None, "") else None
        except (TypeError, ValueError):
            tail_padding = None
        if masked_av:
            plan = plan_masked_context(total_tokens, tail_padding, context_frames)
        else:
            plan = plan_latent_context(total_tokens, tail_padding, context_frames, exact_frame=bool(exact_frame_mode))
        tokens_to_slice = int(plan["tokens"])

        # Slice the planned token window along the time axis (dim 2)
        sliced_video = video[:, :, plan["start_token"]:plan["end_token"], :, :].clone()
        if exact_frame_mode:
            print(
                f"[VRGDG Latent Load] Exact-frame window: tokens {plan['start_token']}-{plan['end_token']} of {total_tokens} "
                f"({plan['context_frames']} context frames, warm-up {plan['warmup_frames']}f, "
                f"tail padding {'unknown - re-render the predecessor for an exact seam' if tail_padding is None else str(tail_padding) + 'f'})"
            )

        sliced_audio = None
        if isinstance(audio, torch.Tensor) and audio.ndim >= 2 and masked_av:
            # The audio ticks that played over exactly the sliced video frames (40 Hz grid)
            total_audio_steps = int(audio.shape[-1])
            a1 = max(1, min(total_audio_steps, int(plan["end_audio_tick"])))
            a0 = max(0, min(a1 - 1, int(plan["start_audio_tick"])))
            sliced_audio = audio[..., a0:a1].clone()
        elif isinstance(audio, torch.Tensor) and audio.ndim >= 2:
            # Audio latent temporal rate is approx 40 Hz (rescaled from 24 fps)
            audio_steps = max(1, round(context_frames * 40 / 24))
            total_audio_steps = int(audio.shape[-1])
            audio_to_slice = min(audio_steps, total_audio_steps)
            sliced_audio = audio[..., -audio_to_slice:].clone()

        if sliced_audio is not None and HAS_NESTED_TENSOR:
            packed_samples = NestedTensor((sliced_video, sliced_audio))
        elif HAS_NESTED_TENSOR:
            empty_audio = torch.zeros((int(sliced_video.shape[0]), 32, 2, 1), dtype=sliced_video.dtype)
            packed_samples = NestedTensor((sliced_video, empty_audio))
        else:
            packed_samples = sliced_video

        out_latent = {
            "samples": packed_samples,
            "video": sliced_video,
            "audio": sliced_audio,
        }
        if masked_av:
            print(
                f"[VRGDG Latent Load] Masked window: tokens {plan['start_token']}-{plan['end_token']} of {total_tokens} "
                f"({plan['context_frames']} frames, {plan['lost_tail_frames']}f of the predecessor's visible tail "
                f"after the window, tail padding {'unknown' if tail_padding is None else str(tail_padding) + 'f'})"
            )
        print(
            f"[VRGDG Latent Load] Loaded scene {scene_number:03d}: sliced {tokens_to_slice} tokens "
            f"({plan['context_frames']} frames context window)"
        )
        return (out_latent, tokens_to_slice)


class VRGDG_MiniMaxH3ApplyLatentGuide:
    """Inject a sliced predecessor latent as a native MiniMax H3 AddGuide temporal keyframe."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "context_latent": ("LATENT",),
            },
            "optional": {
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Frame index where the continuation guide starts (typically 0)",
                }),
                "include_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Also anchor the predecessor's audio tail. Leave off when the scene audio is already "
                        "driven by a source track (Audio Drive / audio reference); the predecessor audio would "
                        "conflict with it at the same timeline position."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "apply"
    CATEGORY = "VRGDG/MiniMax H3 Latent Continuation"
    DESCRIPTION = (
        "Bypasses pixel VAE encoding by attaching the raw predecessor latent directly into the "
        "conditioning keyframes (minimax_keyframes) at frame_idx=0. Video only by default."
    )

    def apply(
        self,
        positive: list[Any],
        latent: dict[str, Any],
        context_latent: dict[str, Any],
        frame_idx: int = 0,
        include_audio: bool = False,
    ) -> tuple[list[Any], dict[str, Any]]:
        # Extract video and audio tensors from context_latent
        video, audio = SceneLatentManager._extract_streams(context_latent)

        # Inspect target latent spatial dimensions and adapt context video if needed
        try:
            target_video, _ = SceneLatentManager._extract_streams(latent)
            if isinstance(target_video, torch.Tensor) and target_video.ndim == 5:
                target_h = int(target_video.shape[-2])
                target_w = int(target_video.shape[-1])
                old_h = int(video.shape[-2])
                old_w = int(video.shape[-1])
                if (old_h, old_w) != (target_h, target_w):
                    video = SceneLatentManager.resize_latent_video(video, target_h, target_w)
                    print(
                        f"[VRGDG Latent Guide] Spatially adapted context latent from {old_h}x{old_w} "
                        f"to target {target_h}x{target_w} ({target_w * 16}x{target_h * 16}px) for keyframe layout."
                    )
        except Exception as exc:
            print(f"[VRGDG Latent Guide] Note: could not inspect target latent shape ({exc}); using context latent as-is.")

        target_device = (
            comfy.model_management.intermediate_device()
            if hasattr(comfy, "model_management") and hasattr(comfy.model_management, "intermediate_device")
            else video.device
        )

        keyframe: dict[str, Any] = {
            "resolved_frame_index": int(frame_idx),
            "latent": video.to(target_device),
        }
        audio_attached = bool(include_audio) and isinstance(audio, torch.Tensor)
        if audio_attached:
            keyframe["audio_latent"] = audio.to(target_device)

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        patched_positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})

        ref_audio_frames = []
        for ref in positive[0][1].get("minimax_refs", []) or []:
            ref_latent = ref.get("audio_latent")
            if ref_latent is not None:
                ref_audio_frames.append(f"{int(ref.get('ref_audio_t', 0))}/{int(ref_latent.shape[-1])}")
        print(
            f"[VRGDG Latent Guide] Attached native latent guide keyframe at frame_idx {frame_idx} "
            f"({video.shape[2]} tokens, spatial {video.shape[-2]}x{video.shape[-1]}, "
            f"audio {'attached, ' + str(int(audio.shape[-1])) + ' frames' if audio_attached else 'not attached'}; "
            f"existing keyframes {len(keyframes) - 1}, ref audio (meta/latent) {ref_audio_frames or 'none'})"
        )
        return (patched_positive, latent)


class VRGDG_MiniMaxH3ApplyMaskedContinuation:
    """Write the predecessor's latent into the head of the target latent and protect it with a denoise mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Target AV latent about to be sampled (after Audio Drive when the audio is locked).",
                }),
                "context_latent": ("LATENT", {
                    "tooltip": "Predecessor window from VRGDG H3 Load Latent with masked_av enabled.",
                }),
            },
            "optional": {
                "include_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Also copy and protect the predecessor's audio ticks. Use only with built-in audio. "
                        "With Audio Drive the song audio is already locked and must not be replaced."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply"
    CATEGORY = "VRGDG/MiniMax H3 Latent Continuation"
    DESCRIPTION = (
        "Latent Continuation Masked: copies the predecessor's raw video latent into the first tokens of the "
        "target latent and sets a zero denoise mask there, so the model keeps those frames and generates the "
        "rest. Needs ComfyUI with MiniMax H3 per-token AV noise masks (PR 15375, ComfyUI 0.34.0+)."
    )

    @staticmethod
    def _mask_like(existing, template: torch.Tensor) -> torch.Tensor:
        if (
            isinstance(existing, torch.Tensor)
            and tuple(existing.shape[2:]) == tuple(template.shape[2:])
            and int(existing.shape[0]) == 1
        ):
            return existing.clone().to(torch.float32)
        return torch.ones((1, 1, *template.shape[2:]), dtype=torch.float32, device=template.device)

    def apply(
        self,
        latent: dict[str, Any],
        context_latent: dict[str, Any],
        include_audio: bool = False,
    ) -> tuple[dict[str, Any]]:
        _require_masked_av_support()
        if not HAS_NESTED_TENSOR:
            raise RuntimeError("Latent Continuation Masked needs ComfyUI's NestedTensor support.")

        target_video, target_audio = SceneLatentManager._extract_streams(latent)
        if not isinstance(target_audio, torch.Tensor):
            raise ValueError("Latent Continuation Masked needs a joint video+audio target latent.")
        ctx_video, ctx_audio = SceneLatentManager._extract_streams(context_latent)
        if int(target_video.shape[0]) != 1 or int(ctx_video.shape[0]) != 1:
            raise ValueError("Latent Continuation Masked supports batch size 1 only.")
        if int(ctx_video.shape[1]) != int(target_video.shape[1]):
            raise ValueError(
                f"Latent Continuation Masked: predecessor has {int(ctx_video.shape[1])} latent channels, "
                f"target has {int(target_video.shape[1])}."
            )

        target_h, target_w = int(target_video.shape[-2]), int(target_video.shape[-1])
        if (int(ctx_video.shape[-2]), int(ctx_video.shape[-1])) != (target_h, target_w):
            print(
                f"[VRGDG Latent Masked] Spatially adapted predecessor latent from {int(ctx_video.shape[-2])}x"
                f"{int(ctx_video.shape[-1])} to {target_h}x{target_w}."
            )
            ctx_video = SceneLatentManager.resize_latent_video(ctx_video, target_h, target_w)

        head = int(ctx_video.shape[2])
        if head >= int(target_video.shape[2]):
            raise ValueError(
                f"Latent Continuation Masked: the {head}-token context fills the whole {int(target_video.shape[2])}-token "
                "target. Render a longer scene or use a smaller context."
            )

        out_video = target_video.clone()
        out_audio = target_audio.clone()
        out_video[:, :, :head] = ctx_video.to(device=out_video.device, dtype=out_video.dtype)

        existing_video_mask, existing_audio_mask = None, None
        if latent.get("noise_mask") is not None:
            try:
                existing_video_mask, existing_audio_mask = SceneLatentManager._extract_streams(latent["noise_mask"])
            except ValueError:
                pass
        video_mask = self._mask_like(existing_video_mask, out_video)
        audio_mask = self._mask_like(existing_audio_mask, out_audio)
        video_mask[:, :, :head] = 0.0

        audio_ticks = 0
        if include_audio and isinstance(ctx_audio, torch.Tensor):
            audio_ticks = min(int(ctx_audio.shape[-1]), int(out_audio.shape[-1]) - 1)
            out_audio[..., :audio_ticks] = ctx_audio[..., :audio_ticks].to(device=out_audio.device, dtype=out_audio.dtype)
            audio_mask[..., :audio_ticks] = 0.0

        out = dict(latent)
        out["samples"] = NestedTensor((out_video, out_audio))
        out["noise_mask"] = NestedTensor((video_mask, audio_mask))
        audio_note = f"copied, {audio_ticks} ticks" if audio_ticks else "left to the audio drive / generated"
        print(
            f"[VRGDG Latent Masked] Protected {head} predecessor tokens at the head of {int(out_video.shape[2])} "
            f"(audio {audio_note})"
        )
        return (out,)


class VRGDG_MiniMaxH3LoadExactFrame:
    """Load one image file (the predecessor scene's last frame) as an IMAGE tensor."""

    @classmethod
    def IS_CHANGED(cls, image_path):
        path = os.path.abspath(str(image_path or "").strip().strip('"'))
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the image file, e.g. the previous scene's extracted last frame",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "VRGDG/MiniMax H3 Latent Continuation"
    DESCRIPTION = "Loads a single image file as a 1-frame IMAGE batch for the exact last-frame anchor."

    def load(self, image_path: str = "") -> tuple[torch.Tensor]:
        import numpy as np
        from PIL import Image, ImageOps

        path = os.path.abspath(str(image_path or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Exact last-frame image was not found: {path}")
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return (torch.from_numpy(array).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3SaveLatent": VRGDG_MiniMaxH3SaveLatent,
    "VRGDG_MiniMaxH3LoadLatent": VRGDG_MiniMaxH3LoadLatent,
    "VRGDG_MiniMaxH3ApplyLatentGuide": VRGDG_MiniMaxH3ApplyLatentGuide,
    "VRGDG_MiniMaxH3ApplyMaskedContinuation": VRGDG_MiniMaxH3ApplyMaskedContinuation,
    "VRGDG_MiniMaxH3LoadExactFrame": VRGDG_MiniMaxH3LoadExactFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3SaveLatent": "VRGDG H3 Save Latent",
    "VRGDG_MiniMaxH3LoadLatent": "VRGDG H3 Load Latent",
    "VRGDG_MiniMaxH3ApplyLatentGuide": "VRGDG H3 Apply Latent Continuation Guide",
    "VRGDG_MiniMaxH3ApplyMaskedContinuation": "VRGDG H3 Apply Masked Continuation",
    "VRGDG_MiniMaxH3LoadExactFrame": "VRGDG H3 Load Exact Last Frame",
}
