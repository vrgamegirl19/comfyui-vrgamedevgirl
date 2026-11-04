"""Chunked MiniMax H3 second pass: the learned upscale and the refinement sampler, one temporal window at a time.

The Builder's normal 2 Pass upscales the whole first-pass clip and refines it in one sampler call, so the second pass
needs memory for every frame of the scene. This node does the same work in overlapping windows (see ``chunk_plan``):
only the current window sits on the GPU, the refined latent is assembled on the CPU, and the first-pass audio is kept
exactly. The graph compiler (``runner/minimax_chunked_pass2.py``) swaps it in for scenes longer than a threshold.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import comfy.model_management
import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced

from .chunk_plan import (
    CHUNK_SECONDS, FPS, OVERLAP_SECONDS, TAIL_TRIM_SECONDS, budget_window_tokens, frame_at,
    tokens_per_latent_frame, window_plan,
)
from .latent_continuation import VRGDG_MiniMaxH3ApplyMaskedContinuation
from .latent_manager import SceneLatentManager
from .latent_upscaler import VRGDG_MiniMaxH3LearnedLatentUpscale

LOG_PREFIX = "[VRGDG Chunked Pass 2]"
UPSCALE_MODES = ("per_window", "whole_clip")
OVERLAP_MODES = ("locked", "crossfade")


def av_parts(latent: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate the native H3 AV boundary without changing its tensor layout."""
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("Connect a joint MiniMax H3 video+audio latent, before AV separation.")
    video, audio = samples.unbind()
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
        raise ValueError("Expected H3 video [B,24,T,H,W] and audio [B,32,2,T40].")
    return video, audio


def cuda_peak_gb() -> Optional[float]:
    """Peak allocated VRAM in GB since the last reset, or None without CUDA."""
    try:
        return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    except Exception:
        return None


def cuda_reset_peak() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def format_gb(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f} GB"


def chunk_conditioning(
    conditioning: List[Any], frame_start: int, frame_end: int, height: int, width: int,
) -> List[Any]:
    """Keep global references; clip timed video/audio guides and rebase their timestamps."""
    result = []
    for embedding, metadata in conditioning:
        meta = metadata.copy()
        if "minimax_keyframes" in meta:
            guides = []
            for guide in meta["minimax_keyframes"]:
                origin = guide["resolved_frame_index"]
                # The two modalities can have different lengths; crop and anchor each separately.
                video = guide.get("latent")
                if video is not None:
                    count = video.shape[2]
                    indices = [
                        i for i in range(count)
                        if origin + frame_at(i) < frame_end and origin + frame_at(i + 1) > frame_start
                    ]
                    if indices:
                        first, last = indices[0] // 5 * 5, indices[-1] + 1
                        item = {k: v for k, v in guide.items() if k not in ("latent", "audio_latent")}
                        item["resolved_frame_index"] = origin + frame_at(first) - frame_start
                        item["latent"] = comfy.utils.common_upscale(
                            video[:, :, first:last].clone(), width, height, "bilinear", "disabled")
                        guides.append(item)
                audio = guide.get("audio_latent")
                if audio is not None:
                    first = max(0, round((frame_start - origin) * 40 / FPS))
                    last = min(audio.shape[-1], round((frame_end - origin) * 40 / FPS))
                    if last > first:
                        item = {k: v for k, v in guide.items() if k not in ("latent", "audio_latent")}
                        item["resolved_frame_index"] = origin + first * FPS / 40 - frame_start
                        item["audio_latent"] = audio[..., first:last].clone()
                        guides.append(item)
            meta["minimax_keyframes"] = guides
        result.append([embedding, meta])
    return result


def join_chunk(
    assembled: torch.Tensor, chunk: torch.Tensor, start: int, covered: int, mode: str, protect: int = 0,
) -> int:
    """Merge a refined window into the CPU latent in place and return the new covered length.

    Locked history is never overwritten. ``crossfade`` blends the overlap, except the first ``protect`` tokens of the
    clip, which hold a predecessor's frames that must stay exactly as they are.
    """
    overlap = max(0, covered - start)
    if overlap and mode == "crossfade":
        skip = min(overlap, max(0, protect - start))
        blend = overlap - skip
        if blend:
            weight = torch.linspace(0, 1, blend + 2, dtype=chunk.dtype)[1:-1].view(1, 1, -1, 1, 1)
            old = assembled[:, :, start + skip:covered]
            old.lerp_(chunk[:, :, skip:overlap], weight)
    end = start + chunk.shape[2]
    assembled[:, :, covered:end] = chunk[:, :, overlap:]
    return end


def run_chunks(
    latent: Dict[str, Any], model: Any, conditioning: List[Any], upscale_model: Dict[str, Any],
    upscaler: Any, sampler: Any, sigmas: torch.Tensor, scale: float,
    chunk_seconds: float, overlap_seconds: float, overlap_mode: str, seed: int, *,
    tail_trim_seconds: float = 0.0, max_window_tokens: int = 0, upscale_mode: str = "per_window",
    head_for_size: Optional[Callable[[int, int], torch.Tensor]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Upscale, then sample each window, retaining only current work and the CPU assembly.

    ``upscale_mode`` ``per_window`` upscales only the current window, so the learned upscaler never holds more than
    one window. ``whole_clip`` runs it once over the entire first-pass clip first and cuts windows from the result.
    ``max_window_tokens`` sizes windows by video tokens instead of seconds (0 is off).
    ``head_for_size(height, width)`` returns the protected predecessor frames for a masked continuation, already at
    the upscaled size. They become the first tokens of the refined clip and are locked while every window samples.
    The report gives the PyTorch-allocated peak VRAM of the upscale and of the sampling of every window.
    """
    if upscale_mode not in UPSCALE_MODES:
        raise ValueError(f"upscale_mode must be per_window or whole_clip, got {upscale_mode!r}.")
    if overlap_mode not in OVERLAP_MODES:
        raise ValueError(f"overlap_mode must be locked or crossfade, got {overlap_mode!r}.")
    video, audio = av_parts(latent)
    video = video.to(device="cpu")
    audio = audio.to(device="cpu")
    up_height, up_width = round(video.shape[-2] * scale), round(video.shape[-1] * scale)
    per_frame = tokens_per_latent_frame(up_height, up_width)
    window_tokens, budget_note = budget_window_tokens(int(max_window_tokens), per_frame)
    plan = window_plan(video.shape[2], chunk_seconds, overlap_seconds, tail_trim_seconds, window_tokens)
    whole = None
    report: List[str] = []
    if budget_note:
        print(f"{LOG_PREFIX} {budget_note}")
        report.append(budget_note)
    if upscale_mode == "whole_clip":
        cuda_reset_peak()
        whole = upscaler.upscale({"samples": video.clone()}, upscale_model, scale)[0]["samples"].to(device="cpu")
        if whole.shape[2] != video.shape[2]:
            raise ValueError("The learned upscaler changed the number of video tokens.")
        line = f"Whole-clip upscale: peak VRAM {format_gb(cuda_peak_gb())} (PyTorch allocated)."
        print(f"{LOG_PREFIX} {line}")
        report.append(line)
    assembled: Optional[torch.Tensor] = None
    covered = 0
    head_tokens = 0
    first = plan[0]
    summary = (
        f"Plan: {len(plan)} windows of {first[1] - first[0]} tokens x {per_frame} tokens/frame "
        f"= {(first[1] - first[0]) * per_frame} video tokens each; "
        f"upscale {upscale_mode}; tail trim {first[1] - first[2] if len(plan) > 1 else 0} tokens."
    )
    print(f"{LOG_PREFIX} {summary}")
    report.insert(0, summary)
    progress = comfy.utils.ProgressBar(len(plan))
    for index, (start, end, keep_end) in enumerate(plan):
        comfy.model_management.throw_exception_if_processing_interrupted()
        frame_start, frame_end = frame_at(start), frame_at(end)
        audio_start = round(frame_start * 40 / FPS)
        audio_end = round(frame_end * 40 / FPS)
        chunk_audio = audio[..., audio_start:audio_end].clone()
        if chunk_audio.shape[-1] != audio_end - audio_start:
            raise ValueError("First-pass audio latent does not cover the video's complete duration.")
        message = (
            f"Chunk {index + 1}/{len(plan)}: {frame_start / FPS:.3f}-{frame_end / FPS:.3f}s, "
            f"video tokens {start}:{end} (keeps to {keep_end}), audio ticks {audio_start}:{audio_end}"
        )
        print(f"{LOG_PREFIX} {message}")
        upscale_peak = None
        if whole is not None:
            upscaled = whole[:, :, start:end].clone()
        else:
            cuda_reset_peak()
            upscaled = upscaler.upscale(
                {"samples": video[:, :, start:end].clone()}, upscale_model, scale)[0]["samples"]
            upscaled = upscaled.to(device="cpu")
            upscale_peak = cuda_peak_gb()
        if assembled is None:
            assembled = torch.empty(
                (*upscaled.shape[:2], video.shape[2], *upscaled.shape[3:]), dtype=upscaled.dtype, device="cpu")
            if head_for_size is not None:
                head = head_for_size(int(upscaled.shape[-2]), int(upscaled.shape[-1]))
                head_tokens = int(head.shape[2])
                if head_tokens >= keep_end or head_tokens >= video.shape[2]:
                    raise ValueError(
                        f"The {head_tokens}-token continuation head does not fit the first window (it keeps "
                        f"{keep_end} tokens). Use a longer chunk length."
                    )
                assembled[:, :, :head_tokens] = head.to(device="cpu", dtype=assembled.dtype)
                covered = head_tokens
                line = f"Locked {head_tokens} predecessor tokens at the head of the clip."
                print(f"{LOG_PREFIX} {line}")
                report.append(line)
        overlap = max(0, covered - start)
        locked = overlap if overlap_mode == "locked" else max(0, min(overlap, head_tokens - start))
        video_mask = torch.ones_like(upscaled)
        if locked:
            upscaled[:, :, :locked] = assembled[:, :, start:start + locked]
            video_mask[:, :, :locked] = 0
        piece = {
            "samples": NestedTensor((upscaled, chunk_audio)),
            "noise_mask": NestedTensor((video_mask, torch.zeros_like(chunk_audio))),
        }
        cond = chunk_conditioning(conditioning, frame_start, frame_end, *upscaled.shape[-2:])
        guider = Guider_Basic(model)
        guider.set_conds(cond)
        noise = Noise_RandomNoise((seed + start) % (1 << 64))
        cuda_reset_peak()
        sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, piece)[0]
        sample_peak = cuda_peak_gb()
        refined = sampled["samples"].unbind()[0].to(device="cpu")
        covered = join_chunk(
            assembled, refined[:, :, :keep_end - start], start, covered, overlap_mode, protect=head_tokens)
        # The counters are reset before each stage, so the two numbers are separate peaks, not one combined peak.
        stages = f"sampling {format_gb(sample_peak)}" if upscale_peak is None else (
            f"upscale {format_gb(upscale_peak)}, sampling {format_gb(sample_peak)}")
        report.append(f"{message}; peak VRAM (PyTorch allocated) {stages}")
        print(f"{LOG_PREFIX} Chunk {index + 1}/{len(plan)} peak VRAM (PyTorch allocated) {stages}")
        # Do not retain the previous sampler/guider's GPU work while upscaling the next window.
        del sampled, refined, piece, guider, cond, upscaled, video_mask, chunk_audio
        progress.update(1)
    if covered != video.shape[2]:
        raise RuntimeError(f"Assembled {covered} of {video.shape[2]} video tokens.")
    result = {"samples": NestedTensor((assembled, audio.clone()))}
    report.append(
        f"Assembled {video.shape[2]} tokens / {frame_at(video.shape[2])} frames; "
        f"{overlap_mode} overlap; first-pass audio preserved exactly."
    )
    return result, "\n".join(report)


def predecessor_head_resolver(head_latent: Dict[str, Any], head_vae: Any) -> Callable[[int, int], torch.Tensor]:
    """A function that returns the predecessor's frames at a given latent size.

    It runs the Builder's own Apply Masked Continuation node on a small blank latent, so a head saved at another
    size is resized through pictures exactly as the whole-clip second pass would resize it.
    """
    def resolve(height: int, width: int) -> torch.Tensor:
        context_video, _ = SceneLatentManager._extract_streams(head_latent)
        head = int(context_video.shape[2])
        blank = {"samples": NestedTensor((
            torch.zeros(1, int(context_video.shape[1]), head + 5, height, width, dtype=context_video.dtype),
            torch.zeros(1, 32, 2, 8, dtype=context_video.dtype),
        ))}
        applied = VRGDG_MiniMaxH3ApplyMaskedContinuation().apply(blank, head_latent, include_audio=False, vae=head_vae)[0]
        return applied["samples"].unbind()[0][:, :, :head].to(device="cpu")

    return resolve


class VRGDG_MiniMaxH3ChunkedPass2:
    """Learned upscale plus the second-pass sampler, one temporal window at a time."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {"required": {
            "latent": ("LATENT", {"tooltip": "The first-pass joint video+audio latent."}),
            "model": ("MODEL",),
            "conditioning": ("CONDITIONING",),
            "upscale_model": ("VRGDG_MINIMAX_H3_LATENT_UPSCALE_MODEL",),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
            "chunk_seconds": ("FLOAT", {"default": CHUNK_SECONDS, "min": 0.1, "max": 120.0, "step": 0.1}),
            "overlap_seconds": ("FLOAT", {"default": OVERLAP_SECONDS, "min": 0.0, "max": 30.0, "step": 0.1}),
            "overlap_mode": (list(OVERLAP_MODES), {
                "tooltip": "locked pins the previous refined overlap while the next window samples. crossfade "
                           "refines both versions of the overlap and blends them.",
            }),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
        }, "optional": {
            "tail_trim_seconds": ("FLOAT", {
                "default": TAIL_TRIM_SECONDS, "min": 0.0, "max": 10.0, "step": 0.1,
                "tooltip": "Throw away this much of each window's tail (snapped to 0.708 s groups) and let the next "
                           "window redo it with refined history on the left and content on the right. Needs an "
                           "overlap of at least two groups (1.6 s). 0 keeps every window's tail."}),
            "max_window_tokens": ("INT", {
                "default": 0, "min": 0, "max": 2000000, "step": 1000,
                "tooltip": "Size windows by video tokens instead of seconds. 0 turns the budget off. A budget below "
                           "the smallest window (7 latent frames) uses that window and says so."}),
            "upscale": (list(UPSCALE_MODES), {
                "tooltip": "per_window upscales only the current window, so upscaler memory is one window. "
                           "whole_clip upscales the entire clip first: window edges get real neighbours, but "
                           "upscaler memory grows with the clip."}),
            "head_latent": ("LATENT", {
                "tooltip": "Masked continuation: the predecessor window from VRGDG H3 Load Latent (masked_av). It "
                           "becomes the locked first tokens of the refined clip."}),
            "head_vae": ("VAE", {
                "tooltip": "Video VAE, used only to resize a head that was saved at another size."}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("refined_av_latent", "chunk_report")
    FUNCTION = "sample"
    CATEGORY = "VRGDG/MiniMax H3"
    DESCRIPTION = (
        "MiniMax H3 second pass in temporal windows: the normal learned upscale and SamplerCustomAdvanced for one "
        "window at a time, assembled on the CPU, with the first-pass audio preserved exactly. The Video Builder uses "
        "it for scenes longer than its Pass 2 chunking threshold."
    )

    def sample(
        self, latent: Dict[str, Any], model: Any, conditioning: List[Any], upscale_model: Dict[str, Any],
        sampler: Any, sigmas: torch.Tensor, scale: float, chunk_seconds: float, overlap_seconds: float,
        overlap_mode: str, seed: int, tail_trim_seconds: float = TAIL_TRIM_SECONDS, max_window_tokens: int = 0,
        upscale: str = "per_window", head_latent: Optional[Dict[str, Any]] = None, head_vae: Any = None,
    ) -> Tuple[Dict[str, Any], str]:
        head = predecessor_head_resolver(head_latent, head_vae) if head_latent is not None else None
        return run_chunks(
            latent, model, conditioning, upscale_model, VRGDG_MiniMaxH3LearnedLatentUpscale(), sampler, sigmas,
            scale, chunk_seconds, overlap_seconds, overlap_mode, seed,
            tail_trim_seconds=tail_trim_seconds, max_window_tokens=max_window_tokens,
            upscale_mode=upscale, head_for_size=head,
        )


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3ChunkedPass2": VRGDG_MiniMaxH3ChunkedPass2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3ChunkedPass2": "VRGDG H3 Chunked Pass 2",
}
