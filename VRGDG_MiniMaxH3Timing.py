"""Exact scene timing calculations for MiniMax H3 Video Builder renders.

This module deliberately has no ComfyUI dependencies.  The Builder runner can
use it before patching a hidden workflow, and the same plan can later drive the
post-render trim.
"""

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Optional


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
):
    """Create the complete render/trim timing plan for one Builder scene.

    Timeline start/end are authoritative.  Warm-up and cool-down are context
    frames and never alter ``final_trim_duration_seconds``.  If the source audio
    starts or ends too close to a boundary, only the unavailable handle is
    clamped; the selected scene itself must still exist in the source audio.

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

    actual_warmup = min(requested_warmup, source_start)
    actual_cooldown = requested_cooldown
    if source_duration is not None:
        audio_after_scene = source_duration - (source_start + scene_duration)
        actual_cooldown = min(requested_cooldown, max(Decimal(0), audio_after_scene))

    audio_trim_start = source_start - actual_warmup
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
    )
