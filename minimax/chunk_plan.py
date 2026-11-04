"""Window planning for the chunked MiniMax H3 second pass (pure Python, no torch).

A long 2 Pass render is refined in overlapping temporal windows instead of one pass over the whole clip. H3 video
tokens run on a 1,4,4,4,4 temporal cycle (5 tokens = 17 pixel frames), so a window starts on a multiple of 5 tokens
and holds ``5k + 2`` tokens. The torch side lives in ``chunked_pass2.py``; the Builder's graph compiler uses the
helpers here to decide whether a scene is long enough to chunk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

FPS = 24
MIN_WINDOW_TOKENS = 7  # one 5-token group plus 2: the smallest window on the H3 grid
MAX_CLIP_FRAMES = 362  # the longest clip H3 renders (about 15.08 s)

PASS2_CHUNKING_MODES = ("auto", "off")
DEFAULT_PASS2_CHUNKING = "auto"
DEFAULT_MIN_SECONDS = 8.0
MIN_SECONDS_RANGE = (1.0, 15.0)

# What the Builder asks the chunk node for. These are the values the Heat A/B test used.
CHUNK_SECONDS = 3.0
OVERLAP_SECONDS = 1.6
OVERLAP_MODE = "locked"
TAIL_TRIM_SECONDS = 0.7
UPSCALE_MODE = "per_window"


def frame_at(token: int) -> int:
    """Pixel-frame start of an H3 token on the 1,4,4,4,4 temporal cycle."""
    groups, tail = divmod(token, 5)
    return groups * 17 + (0 if tail == 0 else 1 + (tail - 1) * 4)


def tokens_for_frames(frames: int) -> int:
    """Video tokens of a ``17k + 5`` frame clip (``5k + 2`` tokens)."""
    if frames < 5 or (frames - 5) % 17:
        raise ValueError(f"An H3 clip has 17k+5 frames, got {frames}.")
    return (frames - 5) // 17 * 5 + 2


def tokens_per_latent_frame(height: int, width: int) -> int:
    """Video tokens one latent frame costs. H3 patches the latent 2x2 and pads an odd side up to the patch size, so a
    68x120 latent is 34x60 = 2040 tokens and a 69x121 latent is 35x61 = 2135."""
    return max(1, ((height + 1) // 2) * ((width + 1) // 2))


def budget_window_tokens(max_window_tokens: int, per_frame: int) -> Tuple[int, str]:
    """``(latent frames per window, note)`` for a token budget.

    A budget of 0 is "off" and returns 0 (the caller then uses seconds). A positive budget never falls back to
    seconds: one too small for even the smallest window returns that window and a note that says so.
    """
    if max_window_tokens <= 0:
        return 0, ""
    frames = int(max_window_tokens) // max(1, per_frame)
    if frames >= MIN_WINDOW_TOKENS:
        return frames, ""
    return MIN_WINDOW_TOKENS, (
        f"max_window_tokens {int(max_window_tokens)} is below the smallest supported window "
        f"({MIN_WINDOW_TOKENS} latent frames = {MIN_WINDOW_TOKENS * per_frame} video tokens at {per_frame} tokens/frame); "
        "using that size."
    )


def window_plan(
    total_tokens: int,
    seconds: float,
    overlap_seconds: float,
    trim_seconds: float = 0.0,
    window_tokens: int = 0,
) -> List[Tuple[int, int, int]]:
    """Phase-aligned windows as ``(start, end, keep_end)``.

    Starts are multiples of 5 and lengths are ``5k + 2``. ``keep_end`` is where a window stops being final: the tail
    ``end - keep_end`` tokens were sampled with no future context, so they are thrown away and the next window redoes
    them with refined history on the left and real content on the right. The last window keeps everything.
    ``trim_seconds`` is snapped to whole 5-token groups (0.708 s each) and always leaves at least 7 tokens of locked
    history, so an overlap too short for it trims nothing. ``window_tokens`` (latent frames per window, from a token
    budget) overrides ``seconds`` when above zero.
    """
    if total_tokens < 2 or total_tokens % 5 != 2:
        raise ValueError("The pass-one H3 latent must contain 5k+2 video tokens (17k+5 frames).")
    if window_tokens > 0:
        groups = max(1, (int(window_tokens) - 2) // 5)
    else:
        groups = max(1, round((seconds * FPS - 5) / 17))
    overlap_groups = min(groups - 1, max(0, round((overlap_seconds * FPS - 5) / 17)))
    trim_groups = min(max(0, overlap_groups - 1), max(0, round(trim_seconds * FPS / 17)))
    window = groups * 5 + 2
    stride = (groups - overlap_groups) * 5
    result: List[Tuple[int, int, int]] = []
    start = 0
    while True:
        end = min(total_tokens, start + window)
        last = end == total_tokens
        result.append((start, end, end if last else end - trim_groups * 5))
        if last:
            return result
        start += stride


def normalize_pass2_chunking(value: Any) -> str:
    """``auto`` or ``off``. Anything else, including a missing value, is ``off``."""
    text = str(value if value is not None else "").strip().lower()
    return text if text in PASS2_CHUNKING_MODES else "off"


def normalize_min_seconds(value: Any) -> float:
    """The length above which a render is chunked, clamped to 1 to 15 s."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = DEFAULT_MIN_SECONDS
    if seconds != seconds:  # NaN
        seconds = DEFAULT_MIN_SECONDS
    return min(max(seconds, MIN_SECONDS_RANGE[0]), MIN_SECONDS_RANGE[1])


def chunking_decision(mode: Any, min_seconds: Any, h3_frame_count: Any) -> Dict[str, Any]:
    """Whether a render of ``h3_frame_count`` frames gets the chunked second pass.

    The length is what H3 renders (the scene plus warm-up and alignment padding), because that is what costs
    memory. A render longer than ``min_seconds`` is chunked, one at or under it keeps the single whole-clip pass.
    """
    mode = normalize_pass2_chunking(mode)
    threshold = normalize_min_seconds(min_seconds)
    try:
        frames = int(h3_frame_count)
    except (TypeError, ValueError):
        frames = 0
    seconds = frames / FPS
    decision: Dict[str, Any] = {
        "mode": mode, "min_seconds": threshold, "rendered_seconds": round(seconds, 3), "enabled": False, "reason": "",
    }
    if mode == "off":
        decision["reason"] = "Pass 2 chunking is off."
    elif frames <= 0:
        decision["reason"] = "The render length is unknown."
    elif seconds <= threshold:
        decision["reason"] = f"{seconds:.2f} s is not longer than {threshold:g} s, so pass 2 runs in one piece."
    else:
        decision["enabled"] = True
        try:
            plan = window_plan(tokens_for_frames(frames), CHUNK_SECONDS, OVERLAP_SECONDS, TAIL_TRIM_SECONDS)
            decision["windows"] = len(plan)
        except ValueError:
            decision["windows"] = None
        decision["reason"] = f"{seconds:.2f} s is longer than {threshold:g} s, so pass 2 runs in windows."
    return decision
