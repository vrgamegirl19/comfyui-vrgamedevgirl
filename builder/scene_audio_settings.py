"""Speaking-scene audio settings and non-destructive ripple timing.

The browser preview route and Agent API both use this service. Source audio
files are never padded or overwritten: silence is represented by timeline gaps.
"""

import copy
import math
from typing import Any

from .timeline import shift_segment_timing
from .elevenlabs_speech import dialogue_speaker, normalize_dialogue, validate_dialogue


DEFAULTS = {"silence_before": 0.0, "silence_after": 0.0, "fit_duration": True}
SCENE_DEFAULTS = {"use_project_defaults": True, **DEFAULTS}


def validate_settings(value: dict[str, Any], scene: bool = False) -> dict[str, Any]:
    """Validate a partial settings patch without silently coercing bad values."""
    allowed = SCENE_DEFAULTS if scene else DEFAULTS
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Audio settings must be an object with keys: {', '.join(allowed)}.")
    result = {}
    for key, item in value.items():
        if isinstance(allowed[key], bool):
            if not isinstance(item, bool):
                raise ValueError(f"{key} must be a boolean.")
            result[key] = item
        else:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{key} must be a number in seconds.")
            if not math.isfinite(item) or not 0 <= item <= 60:
                raise ValueError(f"{key} must be between 0 and 60 seconds.")
            result[key] = round(float(item), 4)
    return result


def require_speaking(session: dict[str, Any]) -> None:
    """Reject audio-settings operations outside Speaking mode."""
    if session.get("video_type") != "speaking":
        raise ValueError("Scene audio settings are available only in Speaking video mode.")


def effective_settings(session: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    """Resolve saved overrides against the project's defaults."""
    defaults = {**DEFAULTS, **validate_settings(session.get("speaking_audio_defaults") or {})}
    saved = {**SCENE_DEFAULTS, **validate_settings(scene.get("scene_audio_settings") or {}, True)}
    return defaults if saved["use_project_defaults"] else {key: saved[key] for key in DEFAULTS}


def scene_clips(session: dict[str, Any], scene_id: str) -> list[dict[str, Any]]:
    """Return dialogue clips owned by this scene; independent music stays separate."""
    return [clip for clip in session.get("audio_clips") or []
            if clip.get("scene_id") == scene_id and (not clip.get("role") or clip["role"] == "dialogue")]


def materialize_scene_clips(session: dict[str, Any]) -> None:
    """Migrate legacy per-scene sources, matching audioClipsForState in the UI."""
    if isinstance(session.get("audio_clips"), list):
        return
    clips = []
    for scene in session.get("segments") or []:
        if not scene.get("custom_audio_path"):
            continue
        duration = float(scene.get("custom_audio_duration") or scene["end"] - scene["start"])
        clips.append({
            "id": f"audio_{scene['id']}", "scene_id": scene["id"],
            "path": scene["custom_audio_path"],
            "name": scene.get("custom_audio_name") or scene.get("label") or "Audio",
            "start": float(scene.get("custom_audio_timeline_start", scene["start"])),
            "source_start": float(scene.get("custom_audio_source_start") or 0),
            "duration": duration,
            "full_duration": float(scene.get("custom_audio_full_duration") or duration),
            "peaks": scene.get("custom_audio_peaks") or [], "lane": 0,
            "role": "dialogue", "volume": 1, "muted": False, "include_in_generation": True,
        })
    if not clips and session.get("audio_path") and float(session.get("audio_duration") or 0) > 0:
        clips.append({
            "id": "project_audio", "scene_id": "", "path": session["audio_path"],
            "name": "Project dialogue", "start": 0, "source_start": 0,
            "duration": session["audio_duration"], "full_duration": session["audio_duration"],
            "peaks": session.get("audio_peaks") or [], "lane": 0, "role": "dialogue",
            "volume": 1, "muted": False, "include_in_generation": True,
        })
    session["audio_clips"] = clips


def audio_settings_view(session: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    """Describe settings and source timing for the dedicated dialog and MCP."""
    require_speaking(session)
    materialize_scene_clips(session)
    clips = scene_clips(session, scene["id"])
    span = (max(c["start"] + c["duration"] for c in clips) - min(c["start"] for c in clips)) if clips else 0
    settings = effective_settings(session, scene)
    return {
        "dialogue": normalize_dialogue(scene.get("scene_dialogue")),
        "scene_id": scene["id"], "settings": {**SCENE_DEFAULTS, **(scene.get("scene_audio_settings") or {})},
        "effective_settings": settings,
        "project_defaults": {**DEFAULTS, **(session.get("speaking_audio_defaults") or {})},
        "clips": clips, "audio_duration": round(span, 4),
        "total_duration": round(span + settings["silence_before"] + settings["silence_after"], 4),
        "scene_duration": round(scene["end"] - scene["start"], 4),
        "needs_render": bool(scene.get("scene_audio_render_dirty")),
    }


def apply_scene_timing(session: dict[str, Any], scene: dict[str, Any]) -> None:
    """Place owned dialogue after opening silence and ripple subsequent scene audio."""
    clips = scene_clips(session, scene["id"])
    if not clips:
        return
    settings = effective_settings(session, scene)
    old_end = float(scene["end"])
    first = min(float(c["start"]) for c in clips)
    offset = float(scene["start"]) + settings["silence_before"] - first
    for clip in clips:
        clip["start"] = round(float(clip["start"]) + offset, 4)
    scene["custom_audio_timeline_start"] = round(float(scene["start"]) + settings["silence_before"], 4)
    if not settings["fit_duration"]:
        return
    new_end = max(c["start"] + c["duration"] for c in clips) + settings["silence_after"]
    delta = round(new_end - old_end, 4)
    scene["end"] = round(new_end, 4)
    if abs(delta) < 0.0001:
        return
    shifted = set()
    for later in session["segments"]:
        if later["id"] != scene["id"] and float(later["start"]) >= old_end - 0.0001:
            shift_segment_timing(later, delta)
            shifted.add(later["id"])
    for clip in session["audio_clips"]:
        if clip.get("scene_id") in shifted:
            clip["start"] = round(float(clip["start"]) + delta, 4)


def update_audio_settings(
    session: dict[str, Any], settings: dict[str, Any], scene_id: str | None = None,
    attachment: dict[str, Any] | None = None,
    dialogue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an edited copy, preserving source files, scene IDs and asset numbering.

    ``attachment`` is trusted metadata from the existing audio import service;
    an empty object removes scene-owned dialogue. Project edits affect only
    scenes inheriting defaults. No scene insertion, deletion or renaming occurs.
    """
    require_speaking(session)
    if dialogue is not None and scene_id is None:
        raise ValueError("Dialogue drafts require an existing scene.")
    patch = validate_settings(settings, scene_id is not None)
    if session.get("timing_frozen") or any(s.get("video_status") == "running" for s in session.get("segments", [])):
        raise ValueError("Unfreeze timing and wait for scene rendering to finish before editing audio.")
    result = copy.deepcopy(session)
    materialize_scene_clips(result)
    scenes = sorted(result.get("segments") or [], key=lambda s: float(s["start"]))
    target = next((s for s in scenes if s["id"] == scene_id), None)
    if scene_id is not None and target is None:
        raise ValueError("Select an existing base scene.")
    if target is not None:
        if dialogue is not None:
            draft = validate_dialogue(dialogue)
            if draft["speaker_id"]:
                dialogue_speaker(result.get("flux_reference_builder"), draft["speaker_id"])
            target["scene_dialogue"] = draft
        target["scene_audio_settings"] = {**SCENE_DEFAULTS, **(target.get("scene_audio_settings") or {}), **patch}
        if attachment is not None:
            result["audio_clips"] = [c for c in result["audio_clips"] if c not in scene_clips(result, scene_id)]
            target.update({
                "custom_audio_path": attachment.get("saved_path", ""),
                "custom_audio_name": attachment.get("audio_name", ""),
                "custom_audio_duration": attachment.get("duration", 0),
                "custom_audio_full_duration": attachment.get("duration", 0),
                "custom_audio_source_start": 0, "custom_audio_timeline_start": target["start"],
                "custom_audio_peaks": attachment.get("peaks", []), "custom_audio_beats": [],
            })
            if attachment:
                result["audio_clips"].append({
                    "id": f"audio_{scene_id}", "scene_id": scene_id, "path": attachment["saved_path"],
                    "name": attachment.get("audio_name") or "Scene dialogue", "start": target["start"],
                    "source_start": 0, "duration": attachment["duration"], "full_duration": attachment["duration"],
                    "peaks": attachment.get("peaks", []), "lane": 0, "role": "dialogue", "volume": 1,
                    "muted": False, "include_in_generation": True,
                })
        targets = [target]
    else:
        result["speaking_audio_defaults"] = {**DEFAULTS, **(result.get("speaking_audio_defaults") or {}), **patch}
        targets = [s for s in scenes if (s.get("scene_audio_settings") or {}).get("use_project_defaults", True)]
    for scene in targets:
        old = next(s for s in session["segments"] if s["id"] == scene["id"])
        before = copy.deepcopy(scene_clips(result, scene["id"]))
        apply_scene_timing(result, scene)
        changed = (attachment is not None or scene["end"] - scene["start"] != old["end"] - old["start"]
                   or before != scene_clips(result, scene["id"])
                   or effective_settings(session, old) != effective_settings(result, scene))
        if changed and (scene.get("video_path") or scene.get("video_output")):
            scene["scene_audio_render_dirty"] = True
    return result
