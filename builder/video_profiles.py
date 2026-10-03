"""Named MiniMax H3 video profiles shared by every project.

A profile is a snapshot of the video settings a user chose in the Video Builder: the video type
(text / image / reference to video), the render pass (single, 2 Pass, 2 Pass Advanced) and everything
that belongs to it (models, resolution, sampler, LoRAs, acceleration, ...). The lock-this-scene state,
the audio mode and the between-scene continuity settings are not part of a profile.

Each profile is one JSON file under ``<ComfyUI output>/VRGDG_Video_Profiles/minimax_h3/``. This module
only reads and writes those files; the HTTP routes live in ``builder/routes.py``.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from ..core.atomic_write import atomic_write_json
from ..minimax.settings_payload import normalize_minimax_h3_settings

PROFILE_ENGINE = "minimax_h3"
PROFILE_VERSION = 1
MAX_PROFILE_NAME_LENGTH = 60

# Never saved in or applied from a profile.
EXCLUDED_PROFILE_KEYS = frozenset({
    # Audio
    "audio_mode",
    # Between-scene continuity
    "continuity_mode", "continuity_prompt_from_last_frame", "latent_context_frames",
    "location_transition_preset", "location_transition_custom",
    # Internal bookkeeping: the per-pass cache and the settings-version markers. Applying a stale
    # version marker would make the Builder reset newer defaults on load.
    "ref_pass_profiles", "ref_pass_mode", "two_pass_defaults_version", "advanced_two_pass_defaults_version",
})


class ProfileExistsError(ValueError):
    """Raised when saving over an existing profile without ``overwrite``."""

    def __init__(self, name: str):
        super().__init__(f"A video profile named '{name}' already exists.")
        self.name = name


def profile_root() -> str:
    import folder_paths  # imported late so the pure helpers work without ComfyUI

    return os.path.join(folder_paths.get_output_directory(), "VRGDG_Video_Profiles", PROFILE_ENGINE)


def clean_profile_name(value: Any) -> str:
    """Return the trimmed display name, or raise ValueError when it is empty or too long."""
    name = re.sub(r"\s+", " ", str(value or "")).strip()
    if not name:
        raise ValueError("Profile name is empty.")
    if len(name) > MAX_PROFILE_NAME_LENGTH:
        raise ValueError(f"Profile name is longer than {MAX_PROFILE_NAME_LENGTH} characters.")
    return name


def _file_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_. -]+", "_", name).strip(" ._")
    return stem.lower() or "profile"


def _profile_path(name: str) -> str:
    return os.path.join(profile_root(), f"{_file_stem(name)}.json")


def filter_profile_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validated MiniMax settings without the keys a profile must not carry."""
    normalized = normalize_minimax_h3_settings(settings if isinstance(settings, dict) else {})
    return {key: value for key, value in normalized.items() if key not in EXCLUDED_PROFILE_KEYS}


def _read_profile_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("settings"), dict) or not str(data.get("name") or "").strip():
        return None
    return data


def list_video_profiles() -> List[Dict[str, Any]]:
    """Summaries of every saved profile, sorted by name. Unreadable files are skipped."""
    root = profile_root()
    if not os.path.isdir(root):
        return []
    profiles = []
    for filename in os.listdir(root):
        if not filename.lower().endswith(".json"):
            continue
        data = _read_profile_file(os.path.join(root, filename))
        if data is None:
            continue
        settings = data["settings"]
        profiles.append({
            "name": str(data["name"]),
            "saved_at": str(data.get("saved_at") or ""),
            "video_mode": str(settings.get("video_mode") or ""),
            "render_pass": str(settings.get("render_pass") or ""),
        })
    return sorted(profiles, key=lambda item: item["name"].lower())


def load_video_profile(name: Any) -> Dict[str, Any]:
    """The named profile with its settings filtered again, so an edited file cannot carry excluded keys."""
    display_name = clean_profile_name(name)
    data = _read_profile_file(_profile_path(display_name))
    if data is None or str(data["name"]).strip().lower() != display_name.lower():
        raise FileNotFoundError(f"Video profile '{display_name}' was not found.")
    return {
        "name": str(data["name"]),
        "saved_at": str(data.get("saved_at") or ""),
        "settings": filter_profile_settings(data["settings"]),
    }


def save_video_profile(name: Any, settings: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """Save the settings under a name. Raises ProfileExistsError unless ``overwrite`` is set."""
    display_name = clean_profile_name(name)
    if not isinstance(settings, dict) or not settings:
        raise ValueError("There are no video settings to save.")
    path = _profile_path(display_name)
    existing = _read_profile_file(path)
    if existing is not None:
        existing_name = str(existing["name"]).strip()
        if existing_name.lower() != display_name.lower():
            raise ValueError(f"The name is too similar to the existing profile '{existing_name}'.")
        if not overwrite:
            raise ProfileExistsError(existing_name)
    saved_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    filtered = filter_profile_settings(settings)
    atomic_write_json(path, {
        "version": PROFILE_VERSION,
        "engine": PROFILE_ENGINE,
        "name": display_name,
        "saved_at": saved_at,
        "settings": filtered,
    })
    print(f"[VRGDG Video Profiles] Saved '{display_name}' ({filtered.get('video_mode')}, {filtered.get('render_pass')})")
    return {"name": display_name, "saved_at": saved_at, "settings": filtered}


def delete_video_profile(name: Any) -> Dict[str, Any]:
    """Delete the named profile file."""
    display_name = clean_profile_name(name)
    path = _profile_path(display_name)
    data = _read_profile_file(path)
    if data is None or str(data["name"]).strip().lower() != display_name.lower():
        raise FileNotFoundError(f"Video profile '{display_name}' was not found.")
    os.remove(path)
    print(f"[VRGDG Video Profiles] Deleted '{data['name']}'")
    return {"name": str(data["name"])}
