"""Resolution-driven tile planning for MiniMax H3 2 Pass Advanced.

Pure functions only (no ComfyUI or torch imports) so the same plan can be used by
the ``VRGDG_MiniMaxH3SpatialTilePlan`` node, the Builder backend and the tests.

``plan_spatial_tiles`` is the Python twin of ``miniMaxH3TilePlan`` in
``web/music_video_builder/minimax_h3.mjs`` and ``solve_equal_tiles`` mirrors the
equal-tile solver inside Comfyui-MMH3-UltimateUpscale's ``MMH3 Spatial Split
Params`` (rows_cols mode). Keep all three in step.
"""

import math
from typing import Any, Dict, Optional, Tuple

# VRAM presets: tile-area target (megapixels), temporal chunk length (frames) and
# the desired spatial overlap (pixels). 2 Pass Advanced is for cards up to 24 GB;
# bigger cards should use the plain 2 Pass or single pass workflows.
VRAM_PRESETS: Dict[str, Dict[str, Any]] = {
    "8gb": {"tile_megapixels": 0.2, "chunk": 51, "overlap": 128},
    "12gb": {"tile_megapixels": 0.3, "chunk": 85, "overlap": 128},
    "16gb": {"tile_megapixels": 0.43, "chunk": 119, "overlap": 128},
    "24gb": {"tile_megapixels": 0.65, "chunk": 153, "overlap": 160},
}
DEFAULT_VRAM_PRESET = "16gb"
# Saved projects from before the 24 GB cap may still hold these values.
LEGACY_VRAM_PRESETS: Dict[str, str] = {"32gb": "24gb", "custom": "24gb"}

# Output order of the VRGDG_MiniMaxH3SpatialTilePlan node. The graph builder links by
# these names, so the node and the builder cannot drift apart.
PLAN_OUTPUT_NAMES: Tuple[str, ...] = (
    "grid_rows", "grid_cols", "spatial_w_overlap", "spatial_h_overlap", "fade_width", "fade_height",
    "min_tile_size", "chunk_length", "temporal_overlap", "tile_width", "tile_height", "summary",
)

# Settings of MMH3 Spatial/Temporal Split Params and the learned upscaler that are no
# longer user options. Values come from the tested audio workflow
# (example_workflow_vrgdg_settings_audio_autotile.json); only the resolution-dependent
# ones are produced by plan_spatial_tiles / VRGDG_MiniMaxH3SpatialTilePlan.
HIDDEN_ADVANCED_SETTINGS: Dict[str, Any] = {
    "tile_size_mode": "rows_cols",
    "anchor_strength": 0.999,
    "overlap_mode": "later",
    "overlap_blend": "linear",
    "brightness_match": False,
    "dynamic_fade": "off",
    "dynamic_fade_min": 32,
    "masked_area_noise": 0.0,
    "upscaler_device": "cuda",
    "upscaler_precision": "bf16",
}

LATENT_TOKEN_PX = 16  # one MiniMax H3 latent token in pixels
PATCH_GRID_PX = 2 * LATENT_TOKEN_PX  # the model's 2x2 latent patch grid
MAX_GRID = 9
DEFAULT_FADE_PX = 64
DEFAULT_MIN_TILE_PX = 256
TEMPORAL_OVERLAP_FRAMES = 17


def normalize_vram_preset(value: Any) -> str:
    """Return a supported preset key, mapping retired values (32gb, custom) to 24gb.

    Raises:
        ValueError: If the value is not a known or retired preset.
    """
    key = str(value or "").strip().lower()
    if not key:
        return DEFAULT_VRAM_PRESET
    key = LEGACY_VRAM_PRESETS.get(key, key)
    if key not in VRAM_PRESETS:
        raise ValueError(
            f"Unknown VRAM preset '{value}'. 2 Pass Advanced supports: {', '.join(VRAM_PRESETS)}."
        )
    return key


def solve_equal_tiles(total_px: int, count: int, base_overlap_px: int,
                      granularity: int = LATENT_TOKEN_PX) -> Tuple[int, int]:
    """Solve ``(tile_px, overlap_px)`` so ``count`` equal tiles cover ``total_px``.

    Mirrors ``_solve_equal_tiles`` in Comfyui-MMH3-UltimateUpscale:
    ``count * tile - (count - 1) * overlap == total_px``, overlap a multiple of
    ``granularity`` and every tile aligned to two latent tokens.

    Args:
        total_px: Frame size along the axis in pixels.
        count: Number of tiles along the axis.
        base_overlap_px: Desired overlap in pixels.
        granularity: Overlap granularity in pixels (one latent token).

    Returns:
        The solved ``(tile_px, overlap_px)``.
    """
    g = int(granularity)
    pg = 2 * g
    total = int(total_px)
    if count <= 1:
        return -(-total // pg) * pg, 0
    start = -(-((total + (count - 1) * int(base_overlap_px)) // count) // pg) * pg
    upper = total - g * (count - 1)
    if start <= upper:
        for size in range(start, upper + 1, pg):
            num = count * size - total
            if num % (count - 1) == 0:
                overlap = num // (count - 1)
                if overlap % g == 0 and 0 <= overlap <= size - g:
                    return size, overlap
    size = start if start <= upper else total
    overlap = int(round((count * size - total) / (count - 1) / g) * g)
    overlap = max(0, min(overlap, size - g))
    return size, overlap


def _best_grid(width: int, height: int, target_area: float) -> Tuple[int, int]:
    """Pick the rows x cols grid whose tiles best match the target area and frame shape."""
    best: Optional[Tuple[float, int, int]] = None
    for rows in range(1, MAX_GRID + 1):
        for cols in range(1, MAX_GRID + 1):
            tile_w = width / cols
            tile_h = height / rows
            area_error = math.log((tile_w * tile_h) / target_area)
            shape_error = math.log(tile_w / tile_h / (width / height))
            # Going over the target area costs double: that is what runs out of memory.
            cost = (2 * area_error if area_error > 0 else -area_error) + 0.5 * abs(shape_error)
            if best is None or cost < best[0] - 1e-9:
                best = (cost, rows, cols)
    assert best is not None
    return best[1], best[2]


def plan_spatial_tiles(width: int, height: int, vram_preset: str = DEFAULT_VRAM_PRESET) -> Dict[str, Any]:
    """Compute every resolution-dependent tiling setting for a Pass 2 frame size.

    Args:
        width: Pass 2 frame width in pixels (multiple of 32).
        height: Pass 2 frame height in pixels (multiple of 32).
        vram_preset: One of ``VRAM_PRESETS``.

    Returns:
        A dict with the grid, the requested overlap, fade, minimum tile size,
        temporal chunk settings and the tile size / overlap the node will solve.

    Raises:
        ValueError: On an unknown preset or a size that is not a multiple of 32.
    """
    key = str(vram_preset or "").strip().lower()
    if key not in VRAM_PRESETS:
        raise ValueError(
            f"Unknown VRAM preset '{vram_preset}'. 2 Pass Advanced supports: {', '.join(VRAM_PRESETS)}."
        )
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError(f"Frame size must be positive multiples of 32 pixels; got {width}x{height}.")
    preset = VRAM_PRESETS[key]
    rows, cols = _best_grid(width, height, preset["tile_megapixels"] * 1048576)
    overlap = int(preset["overlap"])
    # The node errors if a solved tile is below min_tile_size, so drop to fewer tiles
    # on an axis until the solved tile fits.
    while True:
        tile_w, solved_ow = solve_equal_tiles(width, cols, overlap)
        tile_h, solved_oh = solve_equal_tiles(height, rows, overlap)
        if tile_w >= DEFAULT_MIN_TILE_PX and tile_h >= DEFAULT_MIN_TILE_PX:
            break
        if tile_w < DEFAULT_MIN_TILE_PX and cols > 1:
            cols -= 1
        elif tile_h < DEFAULT_MIN_TILE_PX and rows > 1:
            rows -= 1
        else:
            break
    return {
        "vram_preset": key,
        "width": width,
        "height": height,
        "grid_rows": rows,
        "grid_cols": cols,
        "spatial_w_overlap": overlap,
        "spatial_h_overlap": overlap,
        "fade_width": min(DEFAULT_FADE_PX, overlap),
        "fade_height": min(DEFAULT_FADE_PX, overlap),
        "min_tile_size": DEFAULT_MIN_TILE_PX,
        "chunk_length": int(preset["chunk"]),
        "temporal_overlap": TEMPORAL_OVERLAP_FRAMES,
        "tile_width": tile_w,
        "tile_height": tile_h,
        "solved_overlap_w": solved_ow,
        "solved_overlap_h": solved_oh,
    }


def describe_plan(plan: Dict[str, Any]) -> str:
    """One-line human summary of a plan for logs and the node's text output."""
    return (
        f"{plan['grid_rows']}x{plan['grid_cols']} tiles of {plan['tile_height']}x{plan['tile_width']}px "
        f"over {plan['height']}x{plan['width']}px ({plan['vram_preset']}; overlap h={plan['solved_overlap_h']} "
        f"w={plan['solved_overlap_w']}, chunk {plan['chunk_length']} frames)"
    )
