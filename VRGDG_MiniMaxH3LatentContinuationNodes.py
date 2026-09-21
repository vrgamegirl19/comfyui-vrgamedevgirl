"""MiniMax H3 Latent Continuation Custom Nodes for ComfyUI.

Provides dedicated nodes to save, load, inject, and trim MiniMax H3 latents
for lossless, native temporal chaining across multi-scene video projects.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

import torch
import node_helpers
import comfy.model_management

try:
    from .VRGDG_MiniMaxH3LatentManager import (
        SceneLatentManager,
        _frames_to_tokens,
        _tokens_to_frames,
        plan_latent_context,
        scene_latent_manager,
    )
except ImportError:
    from VRGDG_MiniMaxH3LatentManager import (
        SceneLatentManager,
        _frames_to_tokens,
        _tokens_to_frames,
        plan_latent_context,
        scene_latent_manager,
    )

try:
    from comfy.nested_tensor import NestedTensor
    HAS_NESTED_TENSOR = True
except ImportError:
    HAS_NESTED_TENSOR = False


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
    def IS_CHANGED(cls, project_folder, scene_number, context_frames, exact_frame_mode=False):
        folder = str(project_folder or "").strip().strip('"')
        path = SceneLatentManager.get_path(folder, scene_number)
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
                    "max": 141,
                    "step": 1,
                    "tooltip": "Number of temporal context frames to slice from the tail (16, 22, 39, 56)",
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
        if isinstance(audio, torch.Tensor) and audio.ndim >= 2:
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


class VRGDG_MiniMaxH3TrimContinuation:
    """Slice leading temporal context frames from the decoded video output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "context_frames": ("INT", {
                    "default": 22,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Number of leading context frames to trim from the decoded output",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trimmed_images",)
    FUNCTION = "trim"
    CATEGORY = "VRGDG/MiniMax H3 Latent Continuation"
    DESCRIPTION = (
        "Slices off the leading context frames so only newly generated frames remain for "
        "seamless downstream video assembly and timeline playback."
    )

    def trim(self, images: torch.Tensor, context_frames: int = 22) -> tuple[torch.Tensor]:
        if context_frames <= 0 or images is None:
            return (images,)

        total_frames = int(images.shape[0])
        if total_frames <= context_frames:
            print(
                f"[VRGDG Latent Trim] Warning: Image batch has only {total_frames} frames; "
                f"cannot trim {context_frames} frames. Returning full batch."
            )
            return (images,)

        trimmed = images[context_frames:].clone()
        print(f"[VRGDG Latent Trim] Trimmed {context_frames} context frames: {total_frames} -> {trimmed.shape[0]} frames")
        return (trimmed,)


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
    "VRGDG_MiniMaxH3TrimContinuation": VRGDG_MiniMaxH3TrimContinuation,
    "VRGDG_MiniMaxH3LoadExactFrame": VRGDG_MiniMaxH3LoadExactFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3SaveLatent": "VRGDG H3 Save Latent",
    "VRGDG_MiniMaxH3LoadLatent": "VRGDG H3 Load Latent",
    "VRGDG_MiniMaxH3ApplyLatentGuide": "VRGDG H3 Apply Latent Continuation Guide",
    "VRGDG_MiniMaxH3TrimContinuation": "VRGDG H3 Trim Continuation Output",
    "VRGDG_MiniMaxH3LoadExactFrame": "VRGDG H3 Load Exact Last Frame",
}
