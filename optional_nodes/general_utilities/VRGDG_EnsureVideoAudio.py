"""Audio fallback for video workflows that must also accept silent sources."""

from __future__ import annotations

import math

import torch


def _video_duration_seconds(video_info) -> float:
    if not isinstance(video_info, dict):
        raise ValueError("Ensure Video Audio requires video_info from the video loader.")

    for key in ("source_duration", "loaded_duration"):
        try:
            duration = float(video_info.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if math.isfinite(duration) and duration > 0.0:
            return duration

    try:
        frame_count = float(video_info.get("source_frame_count", 0.0) or 0.0)
        fps = float(video_info.get("source_fps", 0.0) or 0.0)
    except (TypeError, ValueError):
        frame_count = 0.0
        fps = 0.0
    if math.isfinite(frame_count) and math.isfinite(fps) and frame_count > 0.0 and fps > 0.0:
        return frame_count / fps
    raise ValueError("The video loader did not report a usable video duration.")


def ensure_video_audio(audio, video_info, fallback_sample_rate=32000, fallback_channels=2):
    """Return materialized source audio or video-length silence."""
    failure = "the video has no readable audio stream"
    try:
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise ValueError("audio waveform is not a [batch, channels, samples] tensor")
        if waveform.shape[-1] < 1 or sample_rate < 1:
            raise ValueError("audio stream is empty")
        return {"waveform": waveform, "sample_rate": sample_rate}, True, (
            f"Using source audio: {waveform.shape[-1] / sample_rate:.3f}s at {sample_rate} Hz."
        )
    except Exception as exc:  # VHS defers extraction until the AUDIO mapping is accessed.
        failure = str(exc).strip() or failure

    duration = _video_duration_seconds(video_info)
    sample_rate = max(8000, min(192000, int(fallback_sample_rate)))
    channels = max(1, min(8, int(fallback_channels)))
    sample_count = max(1, int(round(duration * sample_rate)))
    waveform = torch.zeros((1, channels, sample_count), dtype=torch.float32, device="cpu")
    status = (
        f"No readable source audio; generated {duration:.3f}s of {channels}-channel silence "
        f"at {sample_rate} Hz. VHS reported: {failure[:300]}"
    )
    print(f"[VRGDG Ensure Video Audio] {status}")
    return {"waveform": waveform, "sample_rate": sample_rate}, False, status


class VRGDGEnsureVideoAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_info": ("VHS_VIDEOINFO",),
                "fallback_sample_rate": ([32000, 44100, 48000], {"default": 32000}),
                "fallback_channels": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "BOOLEAN", "STRING")
    RETURN_NAMES = ("audio", "has_source_audio", "status")
    FUNCTION = "ensure"
    CATEGORY = "VRGDG/Video/Audio"
    DESCRIPTION = (
        "Passes readable source audio through unchanged. If a video has no readable audio stream, "
        "creates silence matching the source-video duration so audio-dependent workflows can continue."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def ensure(self, audio, video_info, fallback_sample_rate=32000, fallback_channels=2):
        return ensure_video_audio(audio, video_info, fallback_sample_rate, fallback_channels)


NODE_CLASS_MAPPINGS = {
    "VRGDGEnsureVideoAudio": VRGDGEnsureVideoAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDGEnsureVideoAudio": "VRGDG Ensure Video Audio",
}
