"""MiniMax H3 Scene Latent Storage & Management for VRGDG Video Builder.

Provides serialized latent persistence between render passes to enable native
latent-space temporal chaining and eliminate pixel-based VAE re-encoding drift.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Any, Mapping

import torch

try:
    import safetensors.torch
    import safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from comfy.nested_tensor import NestedTensor
    HAS_NESTED_TENSOR = True
except ImportError:
    HAS_NESTED_TENSOR = False


# MiniMax H3 token to frame conversion table
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _tokens_to_frames(token_count: int) -> int:
    """Calculate video frame count from MiniMax H3 temporal latent token count."""
    if token_count <= 0:
        return 0
    if token_count <= 2:
        return 5
    return sum(_FRAME_PER_TOKEN[k % 5] for k in range(token_count))


def _frames_to_tokens(frame_count: int) -> int:
    """Calculate MiniMax H3 temporal latent token count from video frame count."""
    if frame_count <= 5:
        return 2
    if frame_count in (16, 17):
        return 5
    tokens = 2
    total_frames = 5
    while True:
        next_frames = total_frames + _FRAME_PER_TOKEN[tokens % 5]
        if next_frames > frame_count:
            break
        total_frames = next_frames
        tokens += 1
    return max(2, tokens)


def plan_latent_context(
    total_tokens: int,
    tail_padding_frames: int | None,
    context_frames: int,
    exact_frame: bool = False,
) -> dict[str, Any]:
    """Decide which tail of a predecessor latent becomes the next scene's temporal context.

    Plain mode: the last ``context_frames`` worth of tokens, warm-up == context length.

    Exact-frame mode: the saved latent can run past the predecessor's visible last frame
    (alignment padding / cool-down), and tokens can only be cut on token boundaries. So:
      * the visible end is snapped to a token boundary (``end``, ``rem`` frames left over),
      * the last visible token is replaced by an exact image of the real last frame,
      * the context start is snapped to a multiple of 5 tokens so its first token is a
        1-frame token like the start of every H3 clip (the grid the model expects),
      * ``warmup_frames`` spans context + the replaced token + the leftover frames, so the
        image lands on warm-up frame ``warmup_frames - 1`` = the predecessor's last visible frame.
    """
    total = max(0, int(total_tokens))
    wanted = min(_frames_to_tokens(int(context_frames)), total)
    plain = {
        "exact": False,
        "start_token": total - wanted,
        "end_token": total,
        "tokens": wanted,
        "context_frames": _tokens_to_frames(wanted),
        "warmup_frames": _tokens_to_frames(wanted),
        "image_frame_offset": None,
        "visible_end_token": total,
        "tail_padding_frames": int(tail_padding_frames or 0),
    }
    if not exact_frame or total < 4:
        return plain

    pad = max(0, int(tail_padding_frames or 0))
    visible = max(0, _tokens_to_frames(total) - pad)
    end = total
    while end > 2 and _tokens_to_frames(end) > visible:
        end -= 1
    rem = max(0, visible - _tokens_to_frames(end))

    ctx_end = end - 1
    dropped_span = _FRAME_PER_TOKEN[ctx_end % 5]
    # smallest 5-token-aligned start that keeps the context within the requested size
    start = max(0, -(-(ctx_end - wanted) // 5) * 5)
    tokens = ctx_end - start
    if tokens < 2:
        return plain
    ctx_frames = sum(_FRAME_PER_TOKEN[k % 5] for k in range(tokens))
    warmup = ctx_frames + dropped_span + rem
    return {
        "exact": True,
        "start_token": start,
        "end_token": ctx_end,
        "tokens": tokens,
        "context_frames": ctx_frames,
        "warmup_frames": warmup,
        "image_frame_offset": warmup - 1,
        "visible_end_token": end,
        "tail_padding_frames": pad,
    }


class SceneLatentManager:
    """Universal latent manager for MiniMax H3 scene renders."""

    SUBDIR = "latents"
    FILENAME_PATTERN = re.compile(r"^scene_(\d{3,})\.latent$", re.IGNORECASE)

    @classmethod
    def _latents_folder(cls, project_folder: str) -> str:
        folder = os.path.join(os.path.abspath(project_folder), cls.SUBDIR)
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def ensure_latents_dir(cls, project_folder: str) -> str:
        return cls._latents_folder(project_folder)

    @classmethod
    def get_path(cls, project_folder: str, scene_number: int) -> str:
        return os.path.join(cls._latents_folder(project_folder), f"scene_{int(scene_number):03d}.latent")

    @classmethod
    def get_dirty_path(cls, project_folder: str, scene_number: int) -> str:
        return os.path.join(cls._latents_folder(project_folder), f"scene_{int(scene_number):03d}.dirty")

    @classmethod
    def latent_exists(cls, project_folder: str, scene_number: int) -> bool:
        if not project_folder or not os.path.isdir(project_folder):
            return False
        return os.path.isfile(cls.get_path(project_folder, scene_number))

    @classmethod
    def is_dirty(cls, project_folder: str, scene_number: int) -> bool:
        if not project_folder or not os.path.isdir(project_folder):
            return False
        return os.path.isfile(cls.get_dirty_path(project_folder, scene_number))

    @classmethod
    def mark_dirty(cls, project_folder: str, scene_number: int, reason: str = "") -> None:
        """Mark a scene's latent as dirty (e.g. successor should be re-rendered)."""
        if not project_folder or not os.path.isdir(project_folder):
            return
        dirty_path = cls.get_dirty_path(project_folder, scene_number)
        try:
            with open(dirty_path, "w", encoding="utf-8") as f:
                json.dump({
                    "scene_number": int(scene_number),
                    "timestamp": time.time(),
                    "reason": reason or "Predecessor scene re-rendered or modified",
                }, f, indent=2)
        except Exception as exc:
            print(f"[VRGDG Latent] Failed to mark scene {scene_number:03d} dirty: {exc}")

    @classmethod
    def clear_dirty(cls, project_folder: str, scene_number: int) -> None:
        """Clear dirty flag on a scene."""
        if not project_folder or not os.path.isdir(project_folder):
            return
        dirty_path = cls.get_dirty_path(project_folder, scene_number)
        if os.path.isfile(dirty_path):
            try:
                os.remove(dirty_path)
            except OSError:
                pass

    @classmethod
    def list_dirty(cls, project_folder: str) -> list[int]:
        """List all scene numbers that have active dirty flags."""
        if not project_folder or not os.path.isdir(project_folder):
            return []
        folder = os.path.join(os.path.abspath(project_folder), cls.SUBDIR)
        if not os.path.isdir(folder):
            return []
        dirty_scenes = []
        try:
            for fname in os.listdir(folder):
                if fname.startswith("scene_") and fname.endswith(".dirty"):
                    try:
                        num = int(fname.split("_")[1].split(".")[0])
                        dirty_scenes.append(num)
                    except (IndexError, ValueError):
                        continue
        except OSError:
            pass
        dirty_scenes.sort()
        return dirty_scenes

    @classmethod
    def _extract_streams(cls, source: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract video (and optional audio) tensors from diverse ComfyUI latent inputs."""
        if isinstance(source, Mapping):
            if "samples" in source:
                return cls._extract_streams(source["samples"])
            if "video" in source:
                video = source["video"]
                audio = source.get("audio")
                return video, audio

        # Handle torch.Tensor directly first so unbind is not invoked on standard tensors
        if isinstance(source, torch.Tensor):
            return source, None

        # Handle ComfyUI NestedTensor (holds .tensors tuple of (video, audio))
        if getattr(source, "is_nested", False) or hasattr(source, "tensors"):
            tensors = getattr(source, "tensors", None)
            if isinstance(tensors, (tuple, list)):
                if len(tensors) >= 2:
                    return tensors[0], tensors[1]
                elif len(tensors) == 1:
                    return tensors[0], None

        if isinstance(source, (tuple, list)):
            if len(source) >= 2:
                return source[0], source[1]
            elif len(source) == 1:
                return source[0], None

        # Fallback for custom objects with unbind that are not torch.Tensor
        if hasattr(source, "unbind") and callable(source.unbind):
            try:
                streams = tuple(source.unbind())
                if len(streams) >= 2:
                    return streams[0], streams[1]
                elif len(streams) == 1:
                    return streams[0], None
            except Exception:
                pass

        raise ValueError(f"Unable to extract video latent from input of type {type(source)}")

    @classmethod
    def resize_latent_video(
        cls,
        video: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Spatially resize a MiniMax H3 5D video latent [B, C, T, H, W] to (target_h, target_w).

        Uses bicubic interpolation in float32 for maximum precision over flattened (B*T, C) slices,
        maintaining original dtype and device. If aspect ratios differ significantly (>5%),
        it scales to cover target dimensions and center crops to avoid anamorphic distortion.
        """
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            return video

        b, c, t, h, w = video.shape
        target_h = int(target_h)
        target_w = int(target_w)
        if (h, w) == (target_h, target_w):
            return video

        # Reshape to [b * t, c, h, w] for 2D spatial interpolation
        video_2d = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).to(torch.float32)

        src_ar = w / max(1, h)
        tgt_ar = target_w / max(1, target_h)

        # If aspect ratio is essentially identical (within 5%), direct bicubic interpolation
        if abs(src_ar - tgt_ar) / max(src_ar, tgt_ar) < 0.05:
            out_2d = torch.nn.functional.interpolate(
                video_2d,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            )
        else:
            # Scale to cover target, then center crop
            scale = max(target_h / h, target_w / w)
            scaled_h = max(target_h, round(h * scale))
            scaled_w = max(target_w, round(w * scale))
            scaled_2d = torch.nn.functional.interpolate(
                video_2d,
                size=(scaled_h, scaled_w),
                mode="bicubic",
                align_corners=False,
            )
            top = max(0, (scaled_h - target_h) // 2)
            left = max(0, (scaled_w - target_w) // 2)
            out_2d = scaled_2d[:, :, top:top + target_h, left:left + target_w]
            if (out_2d.shape[-2], out_2d.shape[-1]) != (target_h, target_w):
                out_2d = torch.nn.functional.interpolate(
                    out_2d,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )

        return out_2d.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).to(dtype=video.dtype)

    @classmethod
    def save_latent(
        cls,
        project_folder: str,
        scene_number: int,
        samples_dict_or_tensor: Any,
        frame_count: int | None = None,
        fps: float = 24.0,
        model_type: str = "MiniMax-H3",
        metadata: dict | None = None,
    ) -> str:
        """Serialize a scene's raw temporal latent to disk.

        Args:
            project_folder: Path to the project root directory
            scene_number: Scene index (1-based)
            samples_dict_or_tensor: ComfyUI sampler output dict or latent tensor
            frame_count: Optional explicit frame count
            fps: Frame rate (defaults to 24.0)
            model_type: Model identifier (defaults to "MiniMax-H3")
            metadata: Optional additional metadata dictionary

        Returns:
            Absolute file path of the saved .latent file
        """
        video, audio = cls._extract_streams(samples_dict_or_tensor)
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError(
                f"Video latent must be a 5D tensor [B, C, T, H, W], got: {getattr(video, 'shape', None)}"
            )

        video_cpu = video.detach().to("cpu").contiguous()
        audio_cpu = audio.detach().to("cpu").contiguous() if isinstance(audio, torch.Tensor) else None

        token_count = int(video_cpu.shape[2])
        computed_frames = frame_count if frame_count is not None and frame_count > 0 else _tokens_to_frames(token_count)

        target_path = cls.get_path(project_folder, scene_number)
        meta_dict = {
            "scene_number": str(scene_number),
            "frame_count": str(computed_frames),
            "token_count": str(token_count),
            "fps": str(float(fps)),
            "model_type": str(model_type),
            "timestamp": str(time.time()),
            "video_shape": str(list(video_cpu.shape)),
            "has_audio": "true" if audio_cpu is not None else "false",
        }
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                meta_dict[str(k)] = str(v)

        tensors = {"video": video_cpu}
        if audio_cpu is not None:
            tensors["audio"] = audio_cpu

        saved = False
        if HAS_SAFETENSORS:
            try:
                safetensors.torch.save_file(tensors, target_path, metadata=meta_dict)
                saved = True
            except Exception as exc:
                print(f"[VRGDG Latent] safetensors save failed, falling back to torch.save: {exc}")

        if not saved:
            payload = {
                "video": video_cpu,
                "audio": audio_cpu,
                "metadata": meta_dict,
            }
            torch.save(payload, target_path)

        # Also write sidecar JSON for quick UI reads without loading tensor memory
        sidecar_path = target_path + ".json"
        try:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)
        except Exception:
            pass

        # Clear any dirty flag for this scene since it was just freshly rendered
        cls.clear_dirty(project_folder, scene_number)

        # Mark subsequent scenes as dirty to warn that predecessor changed
        successor_scene = int(scene_number) + 1
        if cls.latent_exists(project_folder, successor_scene):
            cls.mark_dirty(project_folder, successor_scene, reason=f"Scene {scene_number:03d} re-rendered")

        print(
            f"[VRGDG Latent] Saved scene {scene_number:03d} latent ({token_count} tokens, "
            f"{computed_frames} frames, audio: {'yes' if audio_cpu is not None else 'no'}) -> {target_path}"
        )
        return target_path

    @classmethod
    def load_latent(cls, project_folder: str, scene_number: int) -> dict[str, Any] | None:
        """Load a scene's latent tensor and metadata from disk.

        Returns:
            Dict with keys: 'samples', 'video', 'audio', 'frame_count', 'token_count', 'fps', 'metadata'
            or None if the file does not exist.
        """
        path = cls.get_path(project_folder, scene_number)
        if not os.path.isfile(path):
            return None

        video = None
        audio = None
        metadata: dict[str, Any] = {}

        if HAS_SAFETENSORS:
            try:
                tensors = safetensors.torch.load_file(path)
                video = tensors.get("video")
                audio = tensors.get("audio")
                try:
                    with safetensors.safe_open(path, framework="pt") as sf:
                        metadata = sf.metadata() or {}
                except Exception:
                    pass
            except Exception:
                pass

        if video is None:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    video = data.get("video")
                    audio = data.get("audio")
                    metadata = data.get("metadata") or {}
                elif isinstance(data, torch.Tensor):
                    video = data
            except Exception as exc:
                print(f"[VRGDG Latent] Failed to load latent from {path}: {exc}")
                return None

        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            print(f"[VRGDG Latent] Invalid video tensor in {path}")
            return None

        # Assemble the ComfyUI samples structure
        if audio is not None and HAS_NESTED_TENSOR:
            samples = NestedTensor((video, audio))
        elif HAS_NESTED_TENSOR:
            # Fallback zero audio tensor to satisfy H3 joint AV requirements if expected
            empty_audio = torch.zeros((int(video.shape[0]), 32, 2, 1), dtype=video.dtype)
            samples = NestedTensor((video, empty_audio))
        else:
            samples = video

        token_count = int(video.shape[2])
        frame_count = int(metadata.get("frame_count", 0)) or _tokens_to_frames(token_count)
        fps = float(metadata.get("fps", 24.0))

        return {
            "samples": samples,
            "video": video,
            "audio": audio,
            "scene_number": int(scene_number),
            "token_count": token_count,
            "frame_count": frame_count,
            "fps": fps,
            "metadata": metadata,
            "path": path,
        }

    @classmethod
    def get_latent_info(cls, project_folder: str, scene_number: int) -> dict[str, Any]:
        """Query metadata and file stats for a scene latent without loading tensors."""
        path = cls.get_path(project_folder, scene_number)
        exists = os.path.isfile(path)
        if not exists:
            return {
                "ok": True,
                "exists": False,
                "path": path,
                "scene_number": int(scene_number),
                "size_bytes": 0,
                "frame_count": 0,
                "token_count": 0,
                "dirty": cls.is_dirty(project_folder, scene_number),
            }

        size_bytes = os.path.getsize(path)
        sidecar_path = path + ".json"
        meta: dict[str, Any] = {}
        if os.path.isfile(sidecar_path):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        return {
            "ok": True,
            "exists": True,
            "path": path,
            "scene_number": int(scene_number),
            "size_bytes": size_bytes,
            "frame_count": int(meta.get("frame_count", 0)),
            "token_count": int(meta.get("token_count", 0)),
            "fps": float(meta.get("fps", 24.0)),
            "timestamp": float(meta.get("timestamp", 0.0) or os.path.getmtime(path)),
            # frames past the scene's visible end inside this latent; None for latents saved before this was recorded
            "tail_padding_frames": (
                int(float(meta["tail_padding_frames"])) if meta.get("tail_padding_frames") not in (None, "") else None
            ),
            "dirty": cls.is_dirty(project_folder, scene_number),
        }

    @classmethod
    def delete_latent(cls, project_folder: str, scene_number: int, reindex: bool = False) -> bool:
        """Safely delete a scene's latent file and sidecars."""
        path = cls.get_path(project_folder, scene_number)
        deleted = False
        for p in (path, path + ".json", cls.get_dirty_path(project_folder, scene_number)):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                    deleted = True
                except OSError as exc:
                    print(f"[VRGDG Latent] Failed to delete {p}: {exc}")
        if reindex:
            cls.reindex_latents(project_folder, scene_number)
        return deleted

    @classmethod
    def delete_all_latents(cls, project_folder: str) -> int:
        """Delete every scene latent (and its sidecars / dirty flags) in a project. Returns files removed."""
        if not project_folder or not os.path.isdir(project_folder):
            return 0
        folder = os.path.join(os.path.abspath(project_folder), cls.SUBDIR)
        if not os.path.isdir(folder):
            return 0
        removed = 0
        for fname in os.listdir(folder):
            if not fname.startswith("scene_") or not fname.endswith((".latent", ".latent.json", ".dirty")):
                continue
            try:
                os.remove(os.path.join(folder, fname))
                removed += 1
            except OSError as exc:
                print(f"[VRGDG Latent] Failed to delete {fname}: {exc}")
        return removed

    @classmethod
    def reindex_latents(cls, project_folder: str, deleted_scene_number: int) -> None:
        """Shift scene latent indices down by 1 for all scenes after the deleted scene."""
        if not project_folder or not os.path.isdir(project_folder):
            return
        folder = os.path.join(os.path.abspath(project_folder), cls.SUBDIR)
        if not os.path.isdir(folder):
            return

        del_num = int(deleted_scene_number)
        # Find all scene latent numbers > del_num in ascending order
        existing_indices = []
        for fname in os.listdir(folder):
            m = cls.FILENAME_PATTERN.match(fname)
            if m:
                num = int(m.group(1))
                if num > del_num and num not in existing_indices:
                    existing_indices.append(num)

        existing_indices.sort()
        for num in existing_indices:
            new_num = num - 1
            old_base = os.path.join(folder, f"scene_{num:03d}")
            new_base = os.path.join(folder, f"scene_{new_num:03d}")

            for ext in (".latent", ".latent.json", ".dirty"):
                old_file = old_base + ext
                new_file = new_base + ext
                if os.path.isfile(old_file):
                    try:
                        if os.path.isfile(new_file):
                            os.remove(new_file)
                        os.rename(old_file, new_file)
                    except OSError as exc:
                        print(f"[VRGDG Latent] Reindex rename failed {old_file} -> {new_file}: {exc}")

    @classmethod
    def copy_latents_folder(cls, source_project_folder: str, target_project_folder: str) -> None:
        """Copy all latent files when branching / saving project as."""
        if not source_project_folder or not os.path.isdir(source_project_folder):
            return
        src_dir = os.path.join(os.path.abspath(source_project_folder), cls.SUBDIR)
        if not os.path.isdir(src_dir):
            return
        dst_dir = os.path.join(os.path.abspath(target_project_folder), cls.SUBDIR)
        os.makedirs(dst_dir, exist_ok=True)

        for item in os.listdir(src_dir):
            s_path = os.path.join(src_dir, item)
            d_path = os.path.join(dst_dir, item)
            if os.path.isfile(s_path):
                try:
                    shutil.copy2(s_path, d_path)
                except Exception as exc:
                    print(f"[VRGDG Latent] Copy latent failed {s_path} -> {d_path}: {exc}")


scene_latent_manager = SceneLatentManager()

__all__ = [
    "SceneLatentManager",
    "scene_latent_manager",
    "_tokens_to_frames",
    "_frames_to_tokens",
]
