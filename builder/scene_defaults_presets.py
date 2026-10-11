"""Named Scene Defaults beside saved video profiles in ComfyUI's output folder."""

import json
import os
import re
from threading import Lock

from ..core.atomic_write import atomic_write_json
from .video_profiles import clean_profile_name

_SAVE_LOCK = Lock()
TEXT_KEYS = frozenset({
    "cameraFlow", "performanceStyle", "facialPerformance", "facialPerformanceCustom",
    "globalConsistencyPhrase", "storyArcDetail", "imageAesthetic", "imageShotFlow",
    "imageWorldStyle", "imageCustomStyle",
})
SPEED_KEYS = frozenset({"cameraMotionSpeed", "characterMotionSpeed", "cutFrequency"})
BASELINE = {"videoStyle": "", "temporalWorldEffect": "", "fxPreset": ""}
WINDOWS_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL"}
WINDOWS_DEVICE_NAMES.update(f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10))


class SceneDefaultsPresetExistsError(ValueError):
    """Replacing a named preset needs an explicit overwrite."""


def preset_root():
    import folder_paths

    return os.path.join(folder_paths.get_output_directory(), "VRGDG_Video_Profiles", "scene_defaults")


def _preset_path(name):
    stem = re.sub(r"[^A-Za-z0-9_. -]+", "_", name).strip(" ._").lower() or "preset"
    # Device names are invalid filenames on Windows, even when followed by .json.
    if stem.split(".")[0].upper() in WINDOWS_DEVICE_NAMES:
        stem = f"preset_{stem}"
    return os.path.join(preset_root(), f"{stem}.json")


def filter_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Scene Defaults settings must be an object.")
    result = dict(BASELINE)
    for key in TEXT_KEYS & settings.keys():
        value = settings[key]
        if not isinstance(value, str) or len(value) > 20000:
            raise ValueError(f"Invalid Scene Defaults value: {key}.")
        result[key] = value
    for key in SPEED_KEYS & settings.keys():
        try:
            value = float(settings[key])
            if isinstance(settings[key], bool) or not value.is_integer() or not 0 <= value <= 10:
                raise ValueError()
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{key} must be a whole number from 0 to 10.") from None
        result[key] = int(value)
    if "customCameraFlowSequence" in settings:
        sequence = settings["customCameraFlowSequence"]
        if not isinstance(sequence, list) or len(sequence) > 1000:
            raise ValueError("Invalid custom camera flow sequence.")
        clean = []
        for item in sequence:
            if not isinstance(item, dict):
                raise ValueError("Invalid custom camera flow entry.")
            row = {key: item[key] for key in ("shot", "camera") if key in item}
            if any(not isinstance(value, str) or len(value) > 20000 for value in row.values()):
                raise ValueError("Invalid custom camera flow entry.")
            if not row.get("shot", "").strip():
                raise ValueError("Custom camera flow entries need a shot description.")
            clean.append(row)
        result["customCameraFlowSequence"] = clean
    if result.get("cameraFlow") == "custom" and not result.get("customCameraFlowSequence"):
        raise ValueError("Custom camera flow needs at least one saved shot.")
    return result


def load_scene_defaults_preset(name):
    name = clean_profile_name(name)
    with open(_preset_path(name), encoding="utf-8-sig") as handle:
        preset = json.load(handle)
    return {"name": clean_profile_name(preset["name"]), "settings": filter_settings(preset["settings"])}


def list_scene_defaults_presets():
    root = preset_root()
    if not os.path.isdir(root):
        return []
    presets = []
    for filename in os.listdir(root):
        if not filename.lower().endswith(".json"):
            continue
        try:
            with open(os.path.join(root, filename), encoding="utf-8-sig") as handle:
                preset = json.load(handle)
            name = clean_profile_name(preset["name"])
            filter_settings(preset["settings"])
            if os.path.normcase(_preset_path(name)) == os.path.normcase(os.path.join(root, filename)):
                presets.append({"name": name})
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return sorted(presets, key=lambda item: item["name"].casefold())


def save_scene_defaults_preset(name, settings, overwrite=False):
    name = clean_profile_name(name)
    preset = {"version": 1, "name": name, "settings": filter_settings(settings)}
    path = _preset_path(name)
    with _SAVE_LOCK:
        if os.path.exists(path):
            existing = load_scene_defaults_preset(name)
            if existing["name"].casefold() != name.casefold():
                raise ValueError(f"Name is too similar to '{existing['name']}'.")
            if not overwrite:
                raise SceneDefaultsPresetExistsError(f"A Scene Defaults preset named '{name}' already exists.")
        atomic_write_json(path, preset)
    return preset
