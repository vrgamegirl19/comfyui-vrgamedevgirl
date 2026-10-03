"""Shared MiniMax H3 output-resolution math for single pass, 2 Pass and 2 Pass Advanced.

One resolution setting (aspect ratio + preset or custom megapixels) drives every
render pass type. Pure functions only. ``frame_size`` and ``preset_megapixels`` are the
Python twins of ``miniMaxH3FrameSize`` and ``advancedPresetMegapixels`` in
``web/music_video_builder``; keep them in step.
"""

import math
import re
from typing import Any, Dict, Optional, Tuple

RESOLUTION_PRESETS: Tuple[str, ...] = ("custom", "1k", "2k", "1440p", "4k")
PRESET_LONG_EDGE_PX: Dict[str, int] = {"1k": 1024, "2k": 1920, "1440p": 2560, "4k": 3840}
FRAME_MULTIPLE_PX = 32
DEFAULT_ASPECT_RATIO = "16:9 (Widescreen)"


def _js_round(value: float) -> int:
    """Round half up like JavaScript's ``Math.round`` (Python's ``round`` rounds half to even)."""
    return int(math.floor(value + 0.5))


def aspect_ratio_parts(aspect_ratio: Any) -> Tuple[int, int]:
    """Return the ``(width, height)`` ratio from a label such as ``16:9 (Widescreen)``."""
    match = re.search(r"(\d+)\s*:\s*(\d+)", str(aspect_ratio or ""))
    ratio_w = int(match.group(1)) if match else 16
    ratio_h = int(match.group(2)) if match else 9
    return (ratio_w or 16), (ratio_h or 9)


def frame_size(megapixels: float, aspect_ratio: Any = DEFAULT_ASPECT_RATIO) -> Tuple[int, int]:
    """Frame size in pixels (multiples of 32) for a megapixel target and aspect ratio."""
    ratio_w, ratio_h = aspect_ratio_parts(aspect_ratio)
    target = max(0.1, float(megapixels) if megapixels else 2.0)
    scale = math.sqrt(target * 1048576 / (ratio_w * ratio_h))
    return (
        _js_round(ratio_w * scale / FRAME_MULTIPLE_PX) * FRAME_MULTIPLE_PX,
        _js_round(ratio_h * scale / FRAME_MULTIPLE_PX) * FRAME_MULTIPLE_PX,
    )


def preset_megapixels(preset: Any, aspect_ratio: Any = DEFAULT_ASPECT_RATIO) -> Optional[float]:
    """Megapixels of a resolution preset at an aspect ratio, or ``None`` for ``custom``."""
    long_edge = PRESET_LONG_EDGE_PX.get(str(preset or "").strip().lower())
    if not long_edge:
        return None
    ratio_w, ratio_h = aspect_ratio_parts(aspect_ratio)
    scale = long_edge / max(ratio_w, ratio_h)
    width = _js_round(ratio_w * scale / FRAME_MULTIPLE_PX) * FRAME_MULTIPLE_PX
    height = _js_round(ratio_h * scale / FRAME_MULTIPLE_PX) * FRAME_MULTIPLE_PX
    return round((width * height) / 1048576, 4)


def resolved_megapixels(settings: Dict[str, Any]) -> float:
    """Output megapixels for a settings dict: the preset's size, else the custom value."""
    preset = preset_megapixels(settings.get("resolution_preset"), settings.get("aspect_ratio"))
    if preset is not None:
        return preset
    return float(settings.get("megapixels") or 0.9)


def output_frame_size(settings: Dict[str, Any]) -> Tuple[int, int]:
    """Output width and height in pixels for a settings dict."""
    return frame_size(resolved_megapixels(settings), settings.get("aspect_ratio"))


DEFAULT_RESOLUTION_PRESET = "1k"


def _clamp_megapixels(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.1, min(16.0, number or fallback))


def migrate_resolution(raw: Dict[str, Any], render_pass: str) -> Tuple[str, float]:
    """Output ``(resolution_preset, megapixels)`` for a saved settings dict.

    Python twin of ``resolveMiniMaxH3Resolution`` in ``minimax_h3.mjs``. New saves carry
    ``resolution_preset``. Older saves took the resolution of the pass type they rendered
    with: single pass ``megapixels``, 2 Pass final width x height, or 2 Pass Advanced Pass 2
    megapixels/preset.
    """
    aspect = raw.get("aspect_ratio") or DEFAULT_ASPECT_RATIO
    saved = str(raw.get("resolution_preset") or "").strip().lower()
    if saved in RESOLUTION_PRESETS:
        preset_mp = preset_megapixels(saved, aspect)
        return saved, preset_mp if preset_mp is not None else _clamp_megapixels(raw.get("megapixels"), 0.9)
    final_w = raw.get("two_pass_final_width")
    final_h = raw.get("two_pass_final_height")
    has_final_size = _is_positive(final_w) and _is_positive(final_h)
    has_legacy = (
        raw.get("megapixels") is not None
        or has_final_size
        or raw.get("advanced_two_pass_pass2_megapixels") is not None
        or raw.get("advanced_two_pass_pass2_resolution_preset") is not None
    )
    if not has_legacy:
        default_mp = preset_megapixels(DEFAULT_RESOLUTION_PRESET, aspect)
        return DEFAULT_RESOLUTION_PRESET, default_mp if default_mp is not None else 0.5625
    preset = "custom"
    if render_pass == "three_pass":
        legacy_preset = str(raw.get("advanced_two_pass_pass2_resolution_preset") or "").strip().lower()
        preset = legacy_preset if legacy_preset in RESOLUTION_PRESETS else "custom"
        preset_mp = preset_megapixels(preset, aspect)
        megapixels = preset_mp if preset_mp is not None else _clamp_megapixels(raw.get("advanced_two_pass_pass2_megapixels"), 2.0)
    elif render_pass == "two_pass" and has_final_size:
        megapixels = _clamp_megapixels((float(final_w) * float(final_h)) / 1048576, 0.9)
    else:
        megapixels = _clamp_megapixels(raw.get("megapixels"), 0.9)
    if preset == "custom":
        size = frame_size(megapixels, aspect)
        for key in RESOLUTION_PRESETS:
            candidate = preset_megapixels(key, aspect)
            if candidate is not None and frame_size(candidate, aspect) == size:
                return key, candidate
    return preset, round(megapixels, 4)


def _is_positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False
