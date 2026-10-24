"""Per-scene Scene Defaults, sharing the project's existing setting names."""

import copy
import math
from typing import Any, Dict

SCENE_DEFAULT_KEYS = (
    "camera_flow",
    "custom_camera_flow_sequence",
    "camera_guidance",
    "character_guidance",
    "camera_motion_speed",
    "character_motion_speed",
    "minimax_h3_cut_frequency",
    "minimax_h3_cut_style",
    "performance_style",
    "facial_performance",
    "facial_performance_custom",
    "global_consistency_phrase",
    "image_aesthetic",
    "image_shot_flow",
    "video_style",
    "video_style_custom",
    "temporal_world_effect",
    "temporal_world_effect_custom",
    "temporal_allow_background_extras",
    "temporal_background_intensity",
    "temporal_environment_time_passage",
    "temporal_protected_characters",
    "temporal_protected_custom",
    "fx_preset",
    "fx_custom_json",
)
NUMBER_KEYS = {
    "camera_motion_speed", "character_motion_speed", "minimax_h3_cut_frequency",
    "temporal_background_intensity",
}
BOOL_KEYS = {"temporal_allow_background_extras", "temporal_environment_time_passage"}


def normalize_scene_defaults(value: Any, *, strict: bool = False) -> Dict[str, Any]:
    """Keep only scene-level settings; false, zero and empty text are explicit overrides."""
    if not isinstance(value, dict):
        if strict:
            raise ValueError("scene_defaults must be an object.")
        return {}
    unknown = set(value) - set(SCENE_DEFAULT_KEYS)
    if strict and unknown:
        raise ValueError("Unknown scene_defaults fields: " + ", ".join(sorted(unknown)))
    result: Dict[str, Any] = {}
    for key in SCENE_DEFAULT_KEYS:
        if key not in value:
            continue
        item = value[key]
        valid = isinstance(item, str) and len(item) <= 20000
        if key in NUMBER_KEYS:
            valid = isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and 0 <= item <= 10
        elif key in BOOL_KEYS:
            valid = isinstance(item, bool)
        elif key == "custom_camera_flow_sequence":
            valid = isinstance(item, list) and len(item) <= 1000 and all(isinstance(row, (str, dict)) for row in item)
        if not valid:
            if strict:
                raise ValueError("Invalid scene_defaults value for " + key)
            continue
        result[key] = copy.deepcopy(item)
    return result
