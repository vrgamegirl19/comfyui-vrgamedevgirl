"""MiniMax H3 video settings model and render payload builder, shared by the Video Builder and the Agent API.

The browser keeps every MiniMax H3 option in ``session["minimax_h3_settings"]`` and
turns it into a render payload in ``web/music_video_builder/video_render.mjs``. This
module is the Python twin of that logic so agents (MCP) and the server-side
orchestrator render with the same options as the UI:

* ``h3_settings_defaults.json`` lists every setting name, type and default. It is
  generated from the UI with ``node scripts/export_minimax_defaults.mjs``.
* ``normalize_minimax_h3_settings`` / ``validate_minimax_h3_patch`` coerce and check values.
* ``build_minimax_render_payload`` maps settings to the payload keys the
  ``runner/minimax_workflows.py`` graph builders read.

Pure Python: no ComfyUI, torch or aiohttp imports, so it can be copied as is.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .resolution import RESOLUTION_PRESETS, migrate_resolution, output_frame_size
from .scene_inputs import canonical_continuity_mode
from .tile_plan import VRAM_PRESETS, normalize_vram_preset


_DEFAULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h3_settings_defaults.json")

SETTINGS_ENUMS: Dict[str, Tuple[str, ...]] = {
    "pipeline": ("standard", "refmod"),
    "video_mode": ("text_to_video", "image_to_video", "image_reference_to_video", "reference_to_video", "video_to_video"),
    "render_pass": ("single", "two_pass", "three_pass"),
    "audio_mode": ("input_audio", "built_in_audio"),
    "ref_image_size": ("max", "match"),
    "resolution_preset": RESOLUTION_PRESETS,
    "advanced_two_pass_vram_preset": tuple(VRAM_PRESETS),
    "advanced_two_pass_pass1_resolution_preset": RESOLUTION_PRESETS,
    "continuity_mode": (
        "off", "spatial_reference", "exact_start_frame", "latent_continuation",
        "latent_continuation_exact_frame", "latent_continuation_masked",
    ),
    "location_transition_preset": (
        "normal", "surreal", "cinematic", "inner_world", "match", "motion", "creative_auto", "masked", "custom",
    ),
}

# Integer settings that only take one of a fixed set of values. 90, 141 and 192 belong to latent_continuation_masked
# (39 + 51k frames, exact for video and audio); the other latent modes use 16, 22, 39 and 56.
_INT_CHOICES: Dict[str, Tuple[int, ...]] = {
    "latent_context_frames": (16, 22, 39, 56, 90, 141, 192),
}

# One line per setting that agents get wrong without it, shown by minimax_h3_settings_schema.
SETTING_NOTES: Dict[str, str] = {
    "continuity_mode": (
        "Reference to Video and Video to Video only. latent_continuation_masked copies the previous scene's saved latent "
        "into the head of this scene and protects it, so the scene continues the same take. It works in render_pass "
        "single, two_pass and three_pass (pass 1 only). Aliases such as latent_masked are accepted."
    ),
    "latent_context_frames": (
        "Frames of the previous scene's latent used as context. latent_continuation_masked takes 39 (recommended), 90, "
        "141 or 192 and falls back to 39 for any other value. The other latent modes take 16, 22, 39 or 56."
    ),
    "continuity_prompt_from_last_frame": (
        "With continuity_mode latent_continuation_masked, a render of scene 2 or later first writes the scene's prompt from the "
        "previous scene's rendered final frame (the loaded LLM is shown the frame), saves it on the scene, then renders it. The "
        "previous scene must be rendered first. A loaded model that cannot read images falls back to the previous scene's last shot as text."
    ),
    "location_transition_preset": (
        "How the scene prompt moves between two different mapped locations. masked is written for "
        "latent_continuation_masked: continue for about a third of the scene, then make one smooth move. "
        "Used by the final-frame prompt writer (the Video Builder's and the API's continuity_prompt_from_last_frame) and by "
        "minimax-prompts for a continued scene."
    ),
}

# (min, max) inclusive. Matches the clamps in cloneMiniMaxH3Settings.
_EXPLICIT_RANGES: Dict[str, Tuple[float, float]] = {}

# Settings that no longer exist. One output resolution (resolution_preset / megapixels) now drives every
# pass type, and 2 Pass Advanced derives its tiles, chunks, fades and upscaler device from that resolution
# and the VRAM preset (minimax/tile_plan.py). Saved values are ignored; patches get a pointer to the new key.
RETIRED_SETTINGS: Dict[str, str] = {
    "two_pass_final_width": "use resolution_preset / megapixels (one output resolution for every pass type)",
    "two_pass_final_height": "use resolution_preset / megapixels (one output resolution for every pass type)",
    "advanced_two_pass_pass2_megapixels": "use resolution_preset / megapixels (the Pass 2 size is the output resolution)",
    "advanced_two_pass_pass2_resolution_preset": "use resolution_preset (the Pass 2 size is the output resolution)",
    **{
        f"advanced_two_pass_{name}": "derived from the output resolution and advanced_two_pass_vram_preset"
        for name in (
            "tile_size_mode", "tile_width", "tile_height", "grid_rows", "grid_cols", "chunk_length",
            "temporal_overlap", "anchor_strength", "spatial_w_overlap", "spatial_h_overlap", "fade_width",
            "fade_height", "min_tile_size", "overlap_mode", "overlap_blend", "brightness_match",
            "dynamic_fade", "dynamic_fade_min", "masked_area_noise", "upscaler_device", "upscaler_precision",
        )
    },
}

# Saved by the UI but absent from the defaults: per-pass acceleration toggles (they fall back to
# the matching two_pass_use_* value when unset) and the per-pass profile cache used when switching
# render passes. Kept so saved projects round-trip and agents can set them.
_OPTIONAL_SETTINGS: Dict[str, Any] = {
    **{f"pass{number}_use_{name}": False for number in (1, 2) for name in ("te_speed", "feedforward", "block_sparse_attention")},
    "ref_pass_profiles": {},
}

_LORA_APPLY_TO = ("both", "pass1", "pass2")
_REFERENCE_MODES = ("reference_to_video", "image_reference_to_video")

WORKFLOW_SINGLE = "minimax_h3"
WORKFLOW_TWO_PASS = "minimax_h3_2pass"
WORKFLOW_ADVANCED = "minimax_h3_advanced_2pass"


def load_minimax_h3_defaults() -> Dict[str, Any]:
    """Return a fresh copy of the UI defaults (every setting the Video Builder panel saves)."""
    with open(_DEFAULTS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


_DEFAULTS_CACHE: Optional[Dict[str, Any]] = None


def minimax_h3_defaults() -> Dict[str, Any]:
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is None:
        _DEFAULTS_CACHE = load_minimax_h3_defaults()
    return copy.deepcopy(_DEFAULTS_CACHE)


def _numeric_range(key: str) -> Optional[Tuple[float, float]]:
    if key in _EXPLICIT_RANGES:
        return _EXPLICIT_RANGES[key]
    if key == "megapixels" or key.endswith("_megapixels"):
        return (0.1, 16)
    if key == "steps" or key.endswith("_steps") or key == "steps_before_turbo":
        return (1, 1000)
    if key == "denoise" or key.endswith("_denoise"):
        return (0, 1)
    return None


# JSON writes 2.0 as 2, so a whole-number default can still belong to a decimal setting.
_DECIMAL_KEY_TOKENS = ("megapixels", "denoise", "strength", "scale", "percent", "threshold", "depth", "noise")


def _is_decimal_setting(key: str, default: Any) -> bool:
    if isinstance(default, bool):
        return False
    return isinstance(default, float) or (isinstance(default, int) and any(token in key for token in _DECIMAL_KEY_TOKENS))


def _lenient(value: Any, default: Any) -> Any:
    """Turn text from older saves into the number or boolean it stands for ("0.5" -> 0.5)."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if isinstance(default, bool):
        return {"true": True, "false": False}.get(text.lower(), value)
    if isinstance(default, (int, float)):
        try:
            number = float(text)
        except ValueError:
            return value
        return int(number) if number.is_integer() and not isinstance(default, float) else number
    return value


def _coerce(key: str, value: Any, default: Any) -> Any:
    """Coerce ``value`` to the type of ``default`` or raise ValueError with a short reason."""
    if _is_decimal_setting(key, default):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return float(value)
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if value in (0, 1) and not isinstance(value, float):
            return bool(value)
        raise ValueError("must be true or false")
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a whole number")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("must be a whole number")
        return int(value)
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError("must be text")
        return value
    if isinstance(default, list):
        if not isinstance(value, list):
            raise ValueError("must be a list")
        return value
    return value


def _check_value(key: str, value: Any, default: Any, lenient: bool = False) -> Any:
    """Coerce and range/enum-check one setting. Raises ValueError with a reason.

    ``lenient`` accepts numbers and booleans saved as text, for reading saved sessions.
    API patches stay strict so an agent gets a clear error instead of a silent conversion.
    """
    coerced = _coerce(key, _lenient(value, default) if lenient else value, default)
    allowed = SETTINGS_ENUMS.get(key)
    if allowed is not None:
        text = str(coerced).strip().lower()
        if key == "continuity_mode":
            text = canonical_continuity_mode(text) or text
        if text not in allowed:
            raise ValueError(f"must be one of: {', '.join(allowed)}")
        return text
    choices = _INT_CHOICES.get(key)
    if choices is not None and coerced not in choices:
        raise ValueError(f"must be one of: {', '.join(str(item) for item in choices)}")
    bounds = _numeric_range(key)
    if bounds is not None and isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
        if not bounds[0] <= coerced <= bounds[1]:
            raise ValueError(f"must be between {bounds[0]:g} and {bounds[1]:g}")
    if key == "loras":
        _check_loras(coerced)
    return coerced


def _check_loras(loras: List[Any]) -> None:
    if len(loras) > 4:
        raise ValueError("supports at most 4 LoRAs")
    for index, item in enumerate(loras):
        if not isinstance(item, dict) or not str(item.get("name") or item.get("lora_name") or "").strip():
            raise ValueError(f"item {index + 1} needs a 'name'")
        apply_to = str(item.get("apply_to") or "pass1").strip().lower()
        if apply_to not in _LORA_APPLY_TO:
            raise ValueError(f"item {index + 1} apply_to must be one of: {', '.join(_LORA_APPLY_TO)}")


def validate_minimax_h3_patch(patch: Dict[str, Any]) -> Dict[str, str]:
    """Return ``{setting: problem}`` for every unknown key or invalid value in ``patch``."""
    defaults = {**minimax_h3_defaults(), **copy.deepcopy(_OPTIONAL_SETTINGS)}
    problems: Dict[str, str] = {}
    for key, value in patch.items():
        if key in RETIRED_SETTINGS:
            problems[key] = f"retired MiniMax H3 setting: {RETIRED_SETTINGS[key]}"
            continue
        if key not in defaults:
            problems[key] = "unknown MiniMax H3 setting"
            continue
        try:
            _check_value(key, value, defaults[key])
        except ValueError as exc:
            problems[key] = str(exc)
    return problems


def normalize_minimax_h3_settings(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Defaults overlaid with every valid saved value. Invalid or unknown saved keys fall back to defaults."""
    settings = minimax_h3_defaults()
    if not isinstance(raw, dict):
        return settings
    for key, default in list(settings.items()):
        if key not in raw:
            continue
        try:
            value = raw[key]
            if key == "advanced_two_pass_vram_preset":
                value = normalize_vram_preset(value)  # retired 32gb / custom saves map to 24gb
            settings[key] = _check_value(key, value, default, lenient=True)
        except ValueError:
            continue
    if settings.get("pipeline") == "refmod":
        # The RefMod pipeline has one mode and no 2 Pass Advanced.
        settings["video_mode"] = "reference_to_video"
        if settings.get("render_pass") == "three_pass":
            settings["render_pass"] = "two_pass"
        if settings.get("continuity_mode") in ("spatial_reference", "exact_start_frame"):
            settings["continuity_mode"] = "off"
    # One output resolution for every pass type; older saves took it from the pass type they rendered with.
    settings["resolution_preset"], settings["megapixels"] = migrate_resolution(raw, settings["render_pass"])
    for key, default in _OPTIONAL_SETTINGS.items():
        if key in raw:
            try:
                settings[key] = _check_value(key, raw[key], default, lenient=True)
            except ValueError:
                continue
    return settings


def minimax_h3_settings_for_scene(session: Dict[str, Any], segment: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Project settings, overlaid with the scene's own settings when the scene opts in (mirrors the UI)."""
    project_raw = session.get("minimax_h3_settings") if isinstance(session, dict) else None
    merged = dict(project_raw) if isinstance(project_raw, dict) else {}
    if isinstance(segment, dict) and segment.get("use_scene_minimax_h3_settings"):
        scene_raw = segment.get("minimax_h3_settings")
        if isinstance(scene_raw, dict):
            merged.update(scene_raw)
        if not scene_raw or "video_mode" not in scene_raw:
            mode = segment.get("minimax_h3_mode")
            if mode:
                merged["video_mode"] = mode
        # The pipeline belongs to the whole project, so a scene with its own settings follows it.
        merged["pipeline"] = (project_raw or {}).get("pipeline", "standard") if isinstance(project_raw, dict) else "standard"
    return normalize_minimax_h3_settings(merged)


def minimax_render_pass(settings: Dict[str, Any]) -> str:
    """Return ``single``, ``two_pass`` or ``three_pass``. Multi-pass only applies to reference modes."""
    render_pass = str(settings.get("render_pass") or "single")
    if render_pass in ("two_pass", "three_pass") and settings.get("video_mode") in _REFERENCE_MODES:
        return render_pass
    return "single"


def minimax_workflow_key(settings: Dict[str, Any]) -> str:
    """Graph builder key (see ``build_video_graph_for_mode``) for the settings' render pass."""
    return {
        "single": WORKFLOW_SINGLE,
        "two_pass": WORKFLOW_TWO_PASS,
        "three_pass": WORKFLOW_ADVANCED,
    }[minimax_render_pass(settings)]


def build_minimax_render_payload(settings: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map normalized settings to the payload keys the MiniMax graph builders read.

    Covers every settings-derived key ``video_render.mjs`` sends. Scene-specific
    keys (project folder, scene number, prompt, audio, references, timing) are the
    caller's job. ``overrides`` replaces any key afterwards, like the UI's per-call options.
    """
    s = settings
    render_pass = minimax_render_pass(s)
    two_pass = render_pass == "two_pass"
    advanced = render_pass == "three_pass"
    multi = two_pass or advanced
    if advanced and s.get("audio_mode") == "built_in_audio":
        raise ValueError(
            "MiniMax H3 2 Pass Advanced currently supports Input Audio only. "
            "Switch audio_mode to input_audio, or use Single or 2 Pass, before rendering."
        )

    def pick(single: Any, two: Any, adv: Any) -> Any:
        return two if two_pass else adv if advanced else single

    payload: Dict[str, Any] = {
        "audio_mode": s["audio_mode"],
        "pipeline": s["pipeline"],
        "video_mode": s["video_mode"],
        "continuity_mode": s["continuity_mode"] or "off",
        "latent_context_frames": s["latent_context_frames"],
        "minimax_h3_latent_context_frames": s["latent_context_frames"],
        "pre_frames": max(0, int(s["warmup_frames"])),
        "tail_loss_frames": max(0, int(s["cooldown_frames"])),
        "seed": s["seed"],
        "aspect_ratio": s["aspect_ratio"],
        "megapixels": s["megapixels"],
        "diffusion_model_name": s["diffusion_model_name"],
        "clip_name": s["clip_name"],
        "video_vae_name": s["video_vae_name"],
        "audio_vae_name": s["audio_vae_name"],
        "sampler_name": s["sampler_name"],
        "scheduler": s["scheduler"],
        "steps": s["steps"],
        "denoise": s["denoise"],
        "ref_image_size": s["ref_image_size"],
        "two_pass_lora_name": s["two_pass_lora_name"],
        "two_pass_lora_strength": s["two_pass_lora_strength"],
        "use_fast_vae_decode": s["use_fast_vae_decode"],
        "te_speed_processing_control": s["two_pass_te_speed_processing_control"],
        "te_speed_start_percent": s["two_pass_te_speed_start_percent"],
        "te_speed_end_percent": s["two_pass_te_speed_end_percent"],
        "te_speed_mcs": s["two_pass_te_speed_mcs"],
        "te_speed_cache_depth": s["two_pass_te_speed_cache_depth"],
        "te_speed_device": s["two_pass_te_speed_device"],
        "three_pass_lightx_lora_name": s["three_pass_lightx_lora_name"],
        "three_pass_lightx_lora_strength": s["three_pass_lightx_lora_strength"],
        "pass1_steps": pick(s["steps"], s["two_pass_pass1_steps"], s["advanced_two_pass_pass1_steps"]),
        "pass1_denoise": pick(s["denoise"], s["two_pass_pass1_denoise"], s["advanced_two_pass_pass1_denoise"]),
        "pass1_sampler_name": pick(s["sampler_name"], s["two_pass_pass1_sampler"], s["advanced_two_pass_pass1_sampler"]),
        "pass1_scheduler": pick(s["scheduler"], s["two_pass_pass1_scheduler"], s["advanced_two_pass_pass1_scheduler"]),
        "pass1_seed": pick(s["seed"], s["two_pass_pass1_seed"], s["advanced_two_pass_pass1_seed"]),
        "pass2_steps": pick(4, s["two_pass_pass2_steps"], s["advanced_two_pass_pass2_steps"]),
        "pass2_denoise": pick(0.2, s["two_pass_pass2_denoise"], s["advanced_two_pass_pass2_denoise"]),
        "pass2_sampler_name": pick(s["sampler_name"], s["two_pass_pass2_sampler"], s["advanced_two_pass_pass2_sampler"]),
        "pass2_scheduler": pick(s["scheduler"], s["two_pass_pass2_scheduler"], s["advanced_two_pass_pass2_scheduler"]),
        "pass2_seed": pick(s["seed"], s["two_pass_pass2_seed"], s["advanced_two_pass_pass2_seed"]),
        "sage_attention": s["sage_attention"],
        "use_memory_efficient_sage_attention": s["use_memory_efficient_sage_attention"],
        "enable_fp16_accumulation": s["enable_fp16_accumulation"],
        "use_loras": s["use_loras"],
        "lora_count": s["lora_count"],
        "loras": s["loras"],
        "use_turbo_lora": s["use_turbo_lora"],
        "turbo_lora_name": s["turbo_lora_name"],
        "turbo_lora_strength": s["turbo_lora_strength"],
        "easy_cache_bypass": s["easy_cache_bypass"],
        "easy_cache_reuse_threshold": s["easy_cache_reuse_threshold"],
        "easy_cache_start_percent": s["easy_cache_start_percent"],
        "easy_cache_end_percent": s["easy_cache_end_percent"],
        "easy_cache_verbose": s["easy_cache_verbose"],
    }

    # Per-pass acceleration toggles: pass{n}_use_{te_speed,feedforward,block_sparse_attention}.
    for key in ("te_speed", "feedforward", "block_sparse_attention"):
        for number in (1, 2):
            payload[f"pass{number}_use_{key}"] = s.get(f"pass{number}_use_{key}", s.get(f"two_pass_use_{key}"))

    if two_pass:
        payload["final_width"], payload["final_height"] = output_frame_size(s)
        payload["latent_upscale_scale"] = s["two_pass_latent_upscale_scale"]
    if multi:
        payload["latent_upscaler_name"] = s["two_pass_latent_upscaler_name"]
        payload["two_pass_use_feedforward"] = s["two_pass_use_feedforward"]
        payload["two_pass_use_block_sparse_attention"] = s["two_pass_use_block_sparse_attention"]
        payload["two_pass_use_fast_vae_decode"] = s["two_pass_use_fast_vae_decode"]
        payload["final_resize_method"] = s["two_pass_final_resize_method"]
        payload["output_crf"] = s["two_pass_output_crf"]
    else:
        payload["use_te_speed"] = s["use_te_speed"]
        payload["use_feedforward"] = s["use_feedforward"]
        payload["use_block_sparse_attention"] = s["use_block_sparse_attention"]

    # Legacy 3-pass profile values the UI still forwards for every pass.
    for number in (1, 2, 3):
        for field in ("megapixels", "steps", "denoise", "sampler", "scheduler", "seed", "te_speed"):
            payload[f"three_pass_pass{number}_{field}"] = s[f"three_pass_pass{number}_{field}"]

    # 2 Pass Advanced: the Pass 2 size is the shared output resolution. Tiles, chunks, fades and the
    # upscaler device are planned from it and the VRAM preset when the graph is built.
    payload["advanced_pass1_megapixels"] = s["advanced_two_pass_pass1_megapixels"]
    payload["advanced_pass2_megapixels"] = s["megapixels"]
    payload["advanced_vram_preset"] = normalize_vram_preset(s["advanced_two_pass_vram_preset"])

    if overrides:
        payload.update(overrides)
    return payload


SEED_FIELDS = (
    "seed",
    "two_pass_pass1_seed",
    "two_pass_pass2_seed",
    "advanced_two_pass_pass1_seed",
    "advanced_two_pass_pass2_seed",
)


def random_seed_value(rng: Optional[Any] = None) -> int:
    """Same range as the UI's ``randomSeedValue`` (1 to 2147483647)."""
    import random

    return (rng or random).randint(1, 2147483647)


def randomize_minimax_seeds(settings: Dict[str, Any], rng: Optional[Any] = None) -> Dict[str, int]:
    """Give every MiniMax seed field a fresh random value, as ``setMiniMaxH3SeedRandom`` does.

    Returns the new values so the caller can save them with the project.
    """
    fresh = {field: random_seed_value(rng) for field in SEED_FIELDS}
    settings.update(fresh)
    return fresh


def minimax_h3_settings_schema() -> Dict[str, Any]:
    """Describe every MiniMax H3 setting for agents: type, default, allowed values and limits."""
    defaults = minimax_h3_defaults()
    settings: Dict[str, Any] = {}
    for key, default in defaults.items():
        if isinstance(default, bool):
            kind = "boolean"
        elif _is_decimal_setting(key, default):
            kind = "number"
        elif isinstance(default, int):
            kind = "integer"
        elif isinstance(default, float):
            kind = "number"
        elif isinstance(default, list):
            kind = "array"
        elif isinstance(default, dict):
            kind = "object"
        else:
            kind = "string"
        entry: Dict[str, Any] = {"type": kind, "default": default}
        if key in SETTINGS_ENUMS:
            entry["enum"] = list(SETTINGS_ENUMS[key])
        if key in _INT_CHOICES:
            entry["enum"] = list(_INT_CHOICES[key])
        if key in SETTING_NOTES:
            entry["description"] = SETTING_NOTES[key]
        bounds = _numeric_range(key)
        if bounds is not None and kind in ("integer", "number"):
            entry["minimum"], entry["maximum"] = bounds
        settings[key] = entry
    for key, default in _OPTIONAL_SETTINGS.items():
        settings[key] = {
            "type": "boolean" if isinstance(default, bool) else "object",
            "default": default,
            "optional": True,
        }
    return {
        "settings": settings,
        "render_passes": list(SETTINGS_ENUMS["render_pass"]),
        "retired_settings": dict(RETIRED_SETTINGS),
    }
