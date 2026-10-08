"""Named Video Builder layout profiles ("UI profiles") shared by every project.

A UI profile remembers how the Builder looks: whether the left and right side panels are hidden, their widths,
the height of the timeline, and whether the floating LLM Prompting window is open, where it sits and how big it is. It is independent of the video profiles in
``video_profiles.py``, which hold video settings.

Each profile is one JSON file under ``<ComfyUI output>/VRGDG_UI_Profiles/music_video_builder/``, and a small
``_last_selected.json`` there remembers the profile chosen last so the Builder loads it on start. This module only
reads and writes those files; the HTTP routes live in ``builder/routes.py``.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from ..core.atomic_write import atomic_write_json

PROFILE_VERSION = 1
MAX_PROFILE_NAME_LENGTH = 60
LAST_SELECTED_FILE = "_last_selected.json"
DEFAULT_LAYOUT_FILE = "_default_layout.json"

# (minimum, maximum) pixels. The timeline maximum is generous because the browser window limits it further.
PANEL_LIMITS = {
    "left_panel_width": (180, 520),
    "right_panel_width": (280, 720),
    "timeline_panel_height": (190, 4000),
    "llm_popout_width": (300, 1600),
    "llm_popout_height": (200, 2000),
}
# Screen position of the floating window. None means "not placed yet", so it opens beside the right panel.
POSITION_LIMITS = {"llm_popout_x": (-2000, 10000), "llm_popout_y": (-200, 10000)}
DEFAULT_LAYOUT = {
    "left_collapsed": False,
    "right_collapsed": False,
    "llm_popout_open": False,
    "left_panel_width": 260,
    "right_panel_width": 360,
    "timeline_panel_height": 300,
    "llm_popout_width": 460,
    "llm_popout_height": 460,
    "llm_popout_x": None,
    "llm_popout_y": None,
}
_FLAG_KEYS = ("left_collapsed", "right_collapsed", "llm_popout_open")


class UiProfileExistsError(ValueError):
    """Raised when saving over an existing profile without ``overwrite``."""

    def __init__(self, name: str):
        super().__init__(f"A UI profile named '{name}' already exists.")
        self.name = name


def profile_root() -> str:
    import folder_paths  # imported late so the pure helpers work without ComfyUI

    return os.path.join(folder_paths.get_output_directory(), "VRGDG_UI_Profiles", "music_video_builder")


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


def normalize_layout(layout: Any) -> Dict[str, Any]:
    """The layout with every key present and every size clamped, whatever the input held."""
    source = layout if isinstance(layout, dict) else {}
    result: Dict[str, Any] = {key: bool(source.get(key, DEFAULT_LAYOUT[key])) for key in _FLAG_KEYS}
    for key, (low, high) in PANEL_LIMITS.items():
        try:
            value = int(round(float(source.get(key, DEFAULT_LAYOUT[key]))))
        except (TypeError, ValueError):
            value = DEFAULT_LAYOUT[key]
        result[key] = max(low, min(high, value))
    for key, (low, high) in POSITION_LIMITS.items():
        raw = source.get(key)
        try:
            result[key] = None if raw is None or isinstance(raw, bool) else max(low, min(high, int(round(float(raw)))))
        except (TypeError, ValueError):
            result[key] = None
    return result


def _read_profile_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("layout"), dict) or not str(data.get("name") or "").strip():
        return None
    return data


def _find_profile(name: Any) -> Dict[str, Any]:
    display_name = clean_profile_name(name)
    data = _read_profile_file(_profile_path(display_name))
    if data is None or str(data["name"]).strip().lower() != display_name.lower():
        raise FileNotFoundError(f"UI profile '{display_name}' was not found.")
    return data


def list_ui_profiles() -> List[Dict[str, Any]]:
    """Summaries of every saved profile, sorted by name. Unreadable files are skipped."""
    root = profile_root()
    if not os.path.isdir(root):
        return []
    profiles = []
    for filename in os.listdir(root):
        if not filename.lower().endswith(".json"):
            continue
        data = _read_profile_file(os.path.join(root, filename))
        if data is not None:
            profiles.append({"name": str(data["name"]), "saved_at": str(data.get("saved_at") or "")})
    return sorted(profiles, key=lambda item: item["name"].lower())


def load_ui_profile(name: Any) -> Dict[str, Any]:
    """The named profile with its layout normalized again, so an edited file cannot carry bad sizes."""
    data = _find_profile(name)
    return {
        "name": str(data["name"]),
        "saved_at": str(data.get("saved_at") or ""),
        "layout": normalize_layout(data["layout"]),
    }


def save_ui_profile(name: Any, layout: Any, overwrite: bool = False) -> Dict[str, Any]:
    """Save the layout under a name. Raises UiProfileExistsError unless ``overwrite`` is set."""
    display_name = clean_profile_name(name)
    path = _profile_path(display_name)
    existing = _read_profile_file(path)
    if existing is not None:
        existing_name = str(existing["name"]).strip()
        if existing_name.lower() != display_name.lower():
            raise ValueError(f"The name is too similar to the existing profile '{existing_name}'.")
        if not overwrite:
            raise UiProfileExistsError(existing_name)
    saved_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    normalized = normalize_layout(layout)
    atomic_write_json(path, {
        "version": PROFILE_VERSION,
        "name": display_name,
        "saved_at": saved_at,
        "layout": normalized,
    })
    set_last_ui_profile(display_name)
    print(f"[VRGDG UI Profiles] Saved '{display_name}'")
    return {"name": display_name, "saved_at": saved_at, "layout": normalized}


def update_ui_profile_layout(name: Any, layout: Any) -> Dict[str, Any]:
    """Replace the layout of an existing profile, for example after the user resized the timeline."""
    data = _find_profile(name)
    return save_ui_profile(data["name"], layout, overwrite=True)


def get_last_ui_profile_name() -> str:
    """Name of the profile chosen last (selected or saved) that still exists, or an empty string."""
    try:
        with open(os.path.join(profile_root(), LAST_SELECTED_FILE), "r", encoding="utf-8-sig") as handle:
            name = str(json.load(handle).get("name") or "").strip()
    except (OSError, ValueError, AttributeError):
        return ""
    if not name:
        return ""
    data = _read_profile_file(_profile_path(name))
    return str(data["name"]) if data is not None and str(data["name"]).strip().lower() == name.lower() else ""


def set_last_ui_profile(name: Any) -> str:
    """Remember the profile chosen last so the Builder loads it on start. An empty name means no profile."""
    clean = str(name or "").strip()
    if clean:
        clean = load_ui_profile(clean)["name"]
    atomic_write_json(os.path.join(profile_root(), LAST_SELECTED_FILE), {"name": clean})
    return clean


def load_default_ui_layout() -> Optional[Dict[str, Any]]:
    """The layout kept for "Default layout" (no profile selected), or None when it was never saved."""
    try:
        with open(os.path.join(profile_root(), DEFAULT_LAYOUT_FILE), "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return normalize_layout(data.get("layout")) if isinstance(data, dict) and isinstance(data.get("layout"), dict) else None


def save_default_ui_layout(layout: Any) -> Dict[str, Any]:
    """Keep the layout used when no profile is selected, so the Builder comes back the way it was left."""
    normalized = normalize_layout(layout)
    os.makedirs(profile_root(), exist_ok=True)
    atomic_write_json(os.path.join(profile_root(), DEFAULT_LAYOUT_FILE), {"version": PROFILE_VERSION, "layout": normalized})
    return normalized


def delete_ui_profile(name: Any) -> Dict[str, Any]:
    """Delete the named profile file."""
    data = _find_profile(name)
    os.remove(_profile_path(str(data["name"])))
    print(f"[VRGDG UI Profiles] Deleted '{data['name']}'")
    return {"name": str(data["name"])}
