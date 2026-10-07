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
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP
from typing import Any, Mapping, Optional

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


# Context sizes of the masked continuation. Each is 39 + 51k frames, i.e. a whole number of H3 phase groups that also
# lands on an exact 24 fps / 40 Hz video+audio boundary (39 frames = 12 video tokens = 65 audio ticks).
MASKED_CONTEXT_FRAMES = (39, 90, 141, 192)
_MASKED_AUDIO_TICKS_PER_FRAME = 40 / 24


def normalize_masked_context_frames(value: Any) -> int:
    """Snap a requested context size to a valid masked-continuation size (39 when unknown)."""
    try:
        frames = int(float(value))
    except (TypeError, ValueError):
        return MASKED_CONTEXT_FRAMES[0]
    return frames if frames in MASKED_CONTEXT_FRAMES else MASKED_CONTEXT_FRAMES[0]


def plan_masked_context(
    total_tokens: int,
    tail_padding_frames: int | None,
    context_frames: int,
) -> dict[str, Any]:
    """Pick the predecessor frames that are copied, unchanged, into the head of the next scene's latent.

    The copied run must start on H3 phase 0 (a multiple of 5 tokens) so its tokens sit on the same temporal
    phase at the head of the new latent, and it must end on a phase-2 boundary (5k + 2 tokens) so its video and
    audio lengths are exact. Raises ``ValueError`` when the predecessor is too short for the requested context.

    Where the window ends decides how clean the join is. The window ends on the first such boundary at or after the
    predecessor's last visible frame, so the head carries the predecessor's real frames right up to its end (and up
    to 16 real frames of the cool-down after it, ``head_tail_frames``). The new scene's first visible frame is then
    already a real, protected frame, not a regenerated one, and nothing between the two scenes is re-imagined.
    Ending the window before the visible end instead leaves up to 16 frames (``lost_tail_frames``) that the model
    has to regenerate differently from the predecessor, which shows as a pop at the join.

    ``warmup_frames`` is what the new render trims off the front: the window minus the frames of it that lie past
    the predecessor's visible end. Only when the latent has no boundary after the visible end does the window fall
    back to ending before it, and the warm-up then also covers the lost frames.
    """
    frames = normalize_masked_context_frames(context_frames)
    wanted = 2 + 5 * ((frames - 5) // 17)
    total = max(0, int(total_tokens))
    pad = max(0, int(tail_padding_frames or 0))
    visible = max(0, _tokens_to_frames(total) - pad)

    end = next(
        (
            token for token in range(wanted, total + 1)
            if token % 5 == 2 and _tokens_to_frames(token) >= visible
        ),
        None,
    )
    if end is None:
        end = total
        while end >= wanted and (end % 5 != 2 or _tokens_to_frames(end) > visible):
            end -= 1
    if end < wanted:
        raise ValueError(
            f"The predecessor latent ({total} tokens, {visible} visible frames) is too short for a {frames}-frame "
            "masked context. Pick a smaller context size or re-render the predecessor longer."
        )

    start = end - wanted
    start_frame = _tokens_to_frames(start)
    end_frame = _tokens_to_frames(end)
    head_frames = end_frame - start_frame
    head_tail = max(0, end_frame - visible)
    lost_tail = max(0, visible - end_frame)
    return {
        "start_token": start,
        "end_token": end,
        "tokens": wanted,
        "context_frames": head_frames,
        "warmup_frames": head_frames - head_tail + lost_tail,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_audio_tick": round(start_frame * _MASKED_AUDIO_TICKS_PER_FRAME),
        "end_audio_tick": round(end_frame * _MASKED_AUDIO_TICKS_PER_FRAME),
        "lost_tail_frames": lost_tail,
        "head_tail_frames": head_tail,
        "tail_padding_frames": pad,
    }


# Exact scene timing for MiniMax H3 renders. It needs no ComfyUI imports, so the Builder runner can plan a render
# before patching a hidden workflow, and the same plan drives the post-render trim.
H3_FPS = 24
H3_FRAME_STEP = 17
H3_FRAME_OFFSET = 5
H3_MIN_FRAME_COUNT = 5
H3_MAX_FRAME_COUNT = 362


def _decimal(value, name):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite number.")
    return result


def _non_negative_int(value, name):
    number = _decimal(value, name)
    if number < 0 or number != number.to_integral_value():
        raise ValueError(f"{name} must be a non-negative whole number.")
    return int(number)


def _seconds(value):
    """Return a stable JSON-friendly seconds value."""
    return float(value.quantize(Decimal("0.000000001")))


def align_h3_frame_count(frame_count):
    """Round a frame count up to MiniMax H3's 17n+5 frame grid."""
    frames = max(H3_MIN_FRAME_COUNT, _non_negative_int(frame_count, "frame_count"))
    return frames + ((H3_FRAME_OFFSET - frames) % H3_FRAME_STEP)


def frames_covering_duration(duration_seconds, fps=H3_FPS):
    """Return the number of whole frames needed to cover a duration."""
    duration = _decimal(duration_seconds, "duration_seconds")
    frame_rate = _non_negative_int(fps, "fps")
    if duration < 0:
        raise ValueError("duration_seconds must not be negative.")
    if frame_rate <= 0:
        raise ValueError("fps must be greater than zero.")
    return int((duration * frame_rate).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class MiniMaxH3TimingPlan:
    timeline_start_seconds: float
    timeline_end_seconds: float
    scene_duration_seconds: float
    source_start_seconds: float
    source_duration_seconds: Optional[float]
    requested_warmup_frames: int
    requested_cooldown_frames: int
    actual_warmup_seconds: float
    actual_cooldown_seconds: float
    audio_trim_start_seconds: float
    audio_trim_duration_seconds: float
    context_duration_seconds: float
    context_frame_count: int
    workflow_duration_input_seconds: float
    h3_frame_count: int
    h3_render_duration_seconds: float
    alignment_padding_seconds: float
    final_trim_start_seconds: float
    final_trim_duration_seconds: float
    discard_after_scene_seconds: float
    final_frame_count: int
    audio_leading_padding_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


def calculate_minimax_h3_timing(
    timeline_start_seconds,
    timeline_end_seconds,
    warmup_frames=0,
    cooldown_frames=0,
    *,
    source_start_seconds=None,
    source_duration_seconds=None,
    fps=H3_FPS,
    max_frame_count=H3_MAX_FRAME_COUNT,
    pad_warmup=False,
):
    """Create the complete render/trim timing plan for one Builder scene.

    Timeline start/end are authoritative.  Warm-up and cool-down are context
    frames and never alter ``final_trim_duration_seconds``.  If the source audio
    starts or ends too close to a boundary, only the unavailable handle is
    clamped; the selected scene itself must still exist in the source audio.
    With pad_warmup, missing leading audio is silence so a continuation guide
    can keep its full warm-up and be trimmed off with the audio.

    ``workflow_duration_input_seconds`` is intentionally based on the ceiling
    frame count.  Passing it through the current hidden workflow's seconds-to-
    frames expression can therefore never produce a render shorter than the
    requested context.
    """
    frame_rate = _non_negative_int(fps, "fps")
    if frame_rate != H3_FPS:
        raise ValueError(f"MiniMax H3 timing requires {H3_FPS} FPS.")

    timeline_start = _decimal(timeline_start_seconds, "timeline_start_seconds")
    timeline_end = _decimal(timeline_end_seconds, "timeline_end_seconds")
    if timeline_start < 0:
        raise ValueError("timeline_start_seconds must not be negative.")
    if timeline_end <= timeline_start:
        raise ValueError("timeline_end_seconds must be greater than timeline_start_seconds.")
    scene_duration = timeline_end - timeline_start

    warm_frames = _non_negative_int(warmup_frames, "warmup_frames")
    cool_frames = _non_negative_int(cooldown_frames, "cooldown_frames")
    requested_warmup = Decimal(warm_frames) / frame_rate
    requested_cooldown = Decimal(cool_frames) / frame_rate

    source_start = timeline_start if source_start_seconds is None else _decimal(
        source_start_seconds, "source_start_seconds"
    )
    if source_start < 0:
        raise ValueError("source_start_seconds must not be negative.")

    source_duration = None
    if source_duration_seconds is not None:
        source_duration = _decimal(source_duration_seconds, "source_duration_seconds")
        if source_duration < 0:
            raise ValueError("source_duration_seconds must not be negative.")
        if source_start + scene_duration > source_duration:
            raise ValueError(
                "The selected scene extends beyond the available source audio."
            )

    available_warmup = min(requested_warmup, source_start)
    actual_warmup = requested_warmup if pad_warmup else available_warmup
    audio_leading_padding = actual_warmup - available_warmup
    actual_cooldown = requested_cooldown
    if source_duration is not None:
        audio_after_scene = source_duration - (source_start + scene_duration)
        actual_cooldown = min(requested_cooldown, max(Decimal(0), audio_after_scene))

    audio_trim_start = source_start - available_warmup
    context_duration = actual_warmup + scene_duration + actual_cooldown
    context_frames = frames_covering_duration(context_duration, frame_rate)
    h3_frames = align_h3_frame_count(context_frames)
    maximum = _non_negative_int(max_frame_count, "max_frame_count")
    if h3_frames > maximum:
        raise ValueError(
            "The scene plus available warm-up/cool-down requires "
            f"{h3_frames} H3 frames, exceeding the configured maximum of {maximum}."
        )

    render_duration = Decimal(h3_frames) / frame_rate
    workflow_duration = Decimal(context_frames) / frame_rate
    alignment_padding = render_duration - context_duration
    final_trim_start = actual_warmup
    discard_after_scene = render_duration - (actual_warmup + scene_duration)
    # frames the stitcher keeps for this scene: timeline boundaries rounded to whole frames
    final_frame_count = max(1, int(
        (timeline_end * frame_rate).to_integral_value(rounding=ROUND_HALF_UP)
        - (timeline_start * frame_rate).to_integral_value(rounding=ROUND_HALF_UP)
    ))

    return MiniMaxH3TimingPlan(
        timeline_start_seconds=_seconds(timeline_start),
        timeline_end_seconds=_seconds(timeline_end),
        scene_duration_seconds=_seconds(scene_duration),
        source_start_seconds=_seconds(source_start),
        source_duration_seconds=None if source_duration is None else _seconds(source_duration),
        requested_warmup_frames=warm_frames,
        requested_cooldown_frames=cool_frames,
        actual_warmup_seconds=_seconds(actual_warmup),
        actual_cooldown_seconds=_seconds(actual_cooldown),
        audio_trim_start_seconds=_seconds(audio_trim_start),
        audio_trim_duration_seconds=_seconds(context_duration),
        context_duration_seconds=_seconds(context_duration),
        context_frame_count=context_frames,
        workflow_duration_input_seconds=_seconds(workflow_duration),
        h3_frame_count=h3_frames,
        h3_render_duration_seconds=_seconds(render_duration),
        alignment_padding_seconds=_seconds(alignment_padding),
        final_trim_start_seconds=_seconds(final_trim_start),
        final_trim_duration_seconds=_seconds(scene_duration),
        discard_after_scene_seconds=_seconds(discard_after_scene),
        final_frame_count=final_frame_count,
        audio_leading_padding_seconds=_seconds(audio_leading_padding),
    )


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
        except Exception as exc:
            print(f"[VRGDG Latent] Failed to write latent info {sidecar_path}: {exc}")

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
                data = torch.load(path, map_location="cpu", weights_only=True)
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
    def make_room_for_scene(cls, project_folder: str, scene_number: int) -> None:
        """Shift scene latent indices up by 1 for the given scene and every later scene."""
        if not project_folder or not os.path.isdir(project_folder):
            return
        folder = os.path.join(os.path.abspath(project_folder), cls.SUBDIR)
        if not os.path.isdir(folder):
            return

        first = int(scene_number)
        indices = sorted({int(m.group(1)) for m in map(cls.FILENAME_PATTERN.match, os.listdir(folder)) if m and int(m.group(1)) >= first}, reverse=True)
        for num in indices:
            old_base = os.path.join(folder, f"scene_{num:03d}")
            new_base = os.path.join(folder, f"scene_{num + 1:03d}")
            for ext in (".latent", ".latent.json", ".dirty"):
                if os.path.isfile(old_base + ext):
                    os.rename(old_base + ext, new_base + ext)

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
    "calculate_minimax_h3_timing",
]
