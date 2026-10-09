"""Project queries, scene serialization, and asset summaries for the Agent API (6.2, 6.4)."""

import copy
import json
import os
import urllib.parse
from typing import Any, Dict, List, Optional

from ..builder.paths import _session_path
from ..builder.project import _load_builder_session, _redact_session_secrets
from ..minimax.latent_manager import SceneLatentManager
from ..minimax.scene_inputs import normalize_continuity_mode

from .errors import ProjectNotFoundError, SceneNotFoundError, ValidationError
from .paths import get_allowed_project_roots, get_project_id, resolve_project_folder
from .schemas import extract_effective_settings
from .session_keys import KNOWN_SESSION_KEYS, UNSAVED_SESSION_DEFAULTS


def _build_asset_dict(
    project_folder: str,
    relative_path: str,
    kind: str,
    asset_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Build a standard asset descriptor object with cache-busting URL."""
    abs_path = os.path.join(project_folder, relative_path)
    if not os.path.isfile(abs_path):
        return None
    try:
        stat = os.stat(abs_path)
        mtime = int(stat.st_mtime)
        size = stat.st_size
    except OSError:
        return None

    norm_rel = relative_path.replace("\\", "/")
    encoded_path = urllib.parse.quote(abs_path)
    url = f"/vrgdg/video_editor/video?path={encoded_path}&video_cache_bust={mtime}"

    return {
        "id": asset_id or os.path.splitext(os.path.basename(relative_path))[0],
        "kind": kind,
        "filename": os.path.basename(relative_path),
        "path_rel": norm_rel,
        "url": url,
        "bytes": size,
        "mtime": mtime,
        "exists": True,
    }


def _asset_from_path(project_folder: str, path: Any, kind: str, asset_id: str) -> Optional[Dict[str, Any]]:
    """An asset descriptor for a file the session points at (an absolute path), or None if it is not there."""
    text = str(path or "").strip().strip('"')
    if not text or not os.path.isfile(text):
        return None
    absolute = os.path.abspath(text)
    root = os.path.abspath(project_folder)
    inside = os.path.commonpath([root, absolute]) == root if os.path.splitdrive(root)[0].lower() == os.path.splitdrive(absolute)[0].lower() else False
    relative = os.path.relpath(absolute, root) if inside else absolute
    asset = _build_asset_dict(root if inside else os.path.dirname(absolute), relative if inside else os.path.basename(absolute), kind, asset_id)
    if asset and not inside:
        asset["path_rel"] = absolute.replace("\\", "/")
    return asset


def _first_asset(project_folder: str, paths: List[Any], kind: str, asset_id: str) -> Optional[Dict[str, Any]]:
    for path in paths:
        asset = _asset_from_path(project_folder, path, kind, asset_id)
        if asset:
            return asset
    return None


def _scene_image_paths(segment: Dict[str, Any]) -> List[Any]:
    history = segment.get("image_history") if isinstance(segment.get("image_history"), list) else []
    try:
        selected = history[int(segment.get("image_history_index"))]
    except (TypeError, ValueError, IndexError):
        selected = ""
    return [segment.get("approved_image_path"), selected, segment.get("custom_image_path"), segment.get("ref_image_path")]


def list_projects(root: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all available Video Builder projects across allowed roots."""
    roots = [os.path.abspath(root)] if root and os.path.isdir(root) else get_allowed_project_roots()
    projects: List[Dict[str, Any]] = []
    seen_ids = set()

    for r in roots:
        if not os.path.isdir(r):
            continue
        try:
            entries = os.listdir(r)
        except OSError:
            continue

        for name in entries:
            folder = os.path.join(r, name)
            if not os.path.isdir(folder):
                continue
            session_file = _session_path(folder)
            if not os.path.isfile(session_file):
                continue

            pid = name
            if pid in seen_ids:
                continue

            try:
                stat = os.stat(session_file)
                mtime = stat.st_mtime
                with open(session_file, "r", encoding="utf-8-sig") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    continue

                segments = data.get("segments", [])
                scene_count = len(segments) if isinstance(segments, list) else 0
                revision = int(data.get("revision") or data.get("builder_save_revision") or 0)
                audio_path = str(data.get("audio_path") or "").strip()

                projects.append({
                    "id": pid,
                    "name": str(data.get("project_name") or pid),
                    "updated": mtime,
                    "revision": revision,
                    "scene_count": scene_count,
                    "has_audio": bool(audio_path and os.path.isfile(audio_path)),
                    "video_engine": str(data.get("video_engine") or "minimax_h3"),
                    "image_mode": str(data.get("image_model_mode") or "zimage"),
                })
                seen_ids.add(pid)
            except Exception:
                continue

    projects.sort(key=lambda item: item.get("updated", 0), reverse=True)
    return projects


PROJECT_INCLUDE_GROUPS = ("settings", "scenes", "audio", "story", "references")


def get_project_detail(project_id: str, include: Optional[List[str]] = None) -> Dict[str, Any]:
    """Retrieve full project details with revision and settings.

    ``include`` names groups (``PROJECT_INCLUDE_GROUPS``) and/or top-level session keys such as
    ``audio_path`` or ``flux_reference_builder``, which are returned under their own name with API
    keys blanked. A Builder session key (``session_keys.KNOWN_SESSION_KEYS``) the project has not saved
    yet is returned empty. A name that is none of these raises ``ValidationError`` listing it.
    """
    folder = resolve_project_folder(project_id)
    load_result = _load_builder_session(folder)
    session = load_result.get("session")
    if not isinstance(session, dict):
        raise ProjectNotFoundError(project_id)

    requested = [str(item).strip() for item in include if str(item).strip()] if include else []
    includes = set(item.lower() for item in requested) if requested else None

    pid = get_project_id(folder)
    revision = int(session.get("revision") or session.get("builder_save_revision") or 0)

    result: Dict[str, Any] = {
        "id": pid,
        "name": str(session.get("project_name") or pid),
        "project_folder": folder,
        "revision": revision,
        "updated": session.get("updated", 0),
        "video_engine": str(session.get("video_engine") or "minimax_h3"),
        "image_mode": str(session.get("image_model_mode") or "zimage"),
    }

    session_keys = [key for key in requested if key.lower() not in PROJECT_INCLUDE_GROUPS and key not in result]
    unknown = [key for key in session_keys if key not in session and key not in KNOWN_SESSION_KEYS]
    if unknown:
        raise ValidationError(
            f"Unknown include key(s): {', '.join(unknown)}. Use a group ({', '.join(PROJECT_INCLUDE_GROUPS)}) "
            "or a top-level key of the project session.",
            details={"unknown": unknown, "groups": list(PROJECT_INCLUDE_GROUPS)},
        )
    if session_keys:
        # A Builder key this project has not saved yet reads as its empty value (null for most).
        values = {
            key: session[key] if key in session else copy.deepcopy(UNSAVED_SESSION_DEFAULTS.get(key))
            for key in session_keys
        }
        result.update(_redact_session_secrets(values))

    if includes is None or "settings" in includes:
        result["settings"] = extract_effective_settings(session)

    if includes is None or "scenes" in includes:
        result["scenes"] = get_project_scenes(project_id)

    if includes is None or "audio" in includes:
        audio_path = str(session.get("audio_path") or "").strip()
        result["audio"] = {
            "attached": bool(audio_path and os.path.isfile(audio_path)),
            "path": audio_path if (audio_path and os.path.isfile(audio_path)) else "",
            "duration": float(session.get("audio_duration", 0.0) or 0.0),
        }

    if includes is None or "story" in includes:
        result["story"] = session.get("builder_story_layer") or {}

    if includes is None or "references" in includes:
        result["references"] = session.get("flux_reference_builder") or {}

    return result


def _scene_locked_settings(segment: Dict[str, Any]) -> Dict[str, Any]:
    """The scene's Audio Mask and locked MiniMax H3 settings, under the session names the Video Builder saves.

    Read-only copies for the scene GET: ``audio_mask`` (``audio_mask.mjs``), the scene settings lock
    (``use_scene_minimax_h3_settings`` + the scene's own ``minimax_h3_settings``) and the continuity the last render
    recorded (``minimax_h3_continuity_mode_used`` normalized like ``timeline_state.mjs`` reads it, and
    ``minimax_h3_continuity_source_scene_id``). A field the scene never saved is ``None`` (objects), ``False``,
    ``"off"`` or ``""``.
    """
    mask = segment.get("audio_mask")
    scene_settings = segment.get("minimax_h3_settings")
    return {
        "audio_mask": copy.deepcopy(mask) if isinstance(mask, dict) else None,
        "use_scene_minimax_h3_settings": bool(segment.get("use_scene_minimax_h3_settings")),
        "minimax_h3_settings": copy.deepcopy(scene_settings) if isinstance(scene_settings, dict) else None,
        "minimax_h3_continuity_mode_used": normalize_continuity_mode(segment.get("minimax_h3_continuity_mode_used")),
        "minimax_h3_continuity_source_scene_id": str(segment.get("minimax_h3_continuity_source_scene_id") or ""),
    }


def get_project_scenes(
    project_id: str,
    has_image: Optional[bool] = None,
    has_video: Optional[bool] = None,
    has_prompt: Optional[bool] = None,
    status: Optional[str] = None,
    locked_settings: bool = False,
) -> List[Dict[str, Any]]:
    """List project scenes with resolved assets, prompts, and timing.

    ``locked_settings`` adds the scene's Audio Mask and locked MiniMax H3 settings (``_scene_locked_settings``). The
    scene GET asks for them; the scene list leaves them out to stay small.
    """
    folder = resolve_project_folder(project_id)
    load_result = _load_builder_session(folder)
    session = load_result.get("session")
    if not isinstance(session, dict):
        raise ProjectNotFoundError(project_id)

    segments = session.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    scenes: List[Dict[str, Any]] = []

    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue

        scene_id = str(segment.get("id") or f"seg_{index:04d}")
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        duration = max(0.0, end - start)

        t2i = str(segment.get("t2i_prompt") or "").strip()
        i2v = str(segment.get("i2v_prompt") or "").strip()
        minimax_prompt = str(segment.get("minimax_h3_prompt") or "").strip()
        notes = str(segment.get("notes") or "").strip()
        lyric_text = str(segment.get("lyric_text") or "").strip()

        # The files the Video Builder saved on the scene. A fixed legacy name is the fallback.
        image_asset = _first_asset(folder, _scene_image_paths(segment), "image", f"img_{index:04d}") or _build_asset_dict(
            folder, os.path.join("images", f"image_{index:04d}.png"), "image", f"img_{index:04d}")
        video_asset = _first_asset(folder, [segment.get("video_path"), segment.get("rendered_video_path")], "video", f"vid_{index:04d}") or _build_asset_dict(
            folder, os.path.join("scene_videos", f"video_{index:04d}.mp4"), "video", f"vid_{index:04d}")
        thumbnail_asset = _first_asset(folder, [segment.get("video_thumbnail_path"), segment.get("thumbnail_path")], "image", f"thumb_{index:04d}")
        audio_asset = _build_asset_dict(folder, os.path.join("minimax_h3_scene_audio", f"scene_audio_{index:04d}.wav"), "audio", f"aud_{index:04d}") or _build_asset_dict(
            folder, os.path.join("scene_audio", f"audio_{index:04d}.wav"), "audio", f"aud_{index:04d}")

        # Determine status
        scene_status = "empty"
        if video_asset:
            scene_status = "has_video"
        elif image_asset:
            scene_status = "has_image"
        elif t2i or i2v or minimax_prompt:
            scene_status = "has_prompt"

        # Apply filters
        if has_image is not None and bool(image_asset) != has_image:
            continue
        if has_video is not None and bool(video_asset) != has_video:
            continue
        if has_prompt is not None and bool(t2i or i2v or minimax_prompt) != has_prompt:
            continue
        if status is not None and scene_status != status:
            continue

        scenes.append({
            "id": scene_id,
            "number": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "status": scene_status,
            "lyrics": lyric_text or notes,
            "story_beat": str(segment.get("story_beat") or "").strip(),
            "t2i_prompt": t2i,
            "i2v_prompt": i2v,
            "minimax_h3_prompt": minimax_prompt,
            "minimax_h3_continuation_direction": str(segment.get("minimax_h3_continuation_direction") or "").strip(),
            "minimax_h3_i2v_frame_mode": str(segment.get("minimax_h3_i2v_frame_mode") or (
                "flf" if segment.get("first_last_frame_end_image_path") else "normal"
            )),
            "first_last_frame_end_image_path": str(segment.get("first_last_frame_end_image_path") or ""),
            "minimax_h3_continuation_start_seconds": segment.get("minimax_h3_continuation_start_seconds"),
            "enhanced_prompt": str(segment.get("enhance_prompt") or "").strip(),
            "no_character_present": bool(segment.get("no_character_present")),
            "lyric_no_lip_sync": bool(segment.get("lyric_no_lip_sync")),
            "lyric_singers": [str(s) for s in segment.get("lyric_singers") or []],
            "approved_image": image_asset,
            "rendered_video": video_asset,
            "video_thumbnail": thumbnail_asset,
            "scene_audio": audio_asset,
        })
        if locked_settings:
            scenes[-1].update(_scene_locked_settings(segment))

    return scenes


def get_scene_detail(project_id: str, scene_id: str) -> Dict[str, Any]:
    """Get single scene details by scene ID or 1-based number, with its Audio Mask and locked MiniMax H3 settings."""
    scenes = get_project_scenes(project_id, locked_settings=True)
    target = str(scene_id).strip()

    for s in scenes:
        if s["id"] == target or str(s["number"]) == target:
            return s

    raise SceneNotFoundError(scene_id, project_id)


def get_project_summary(project_id: str) -> Dict[str, Any]:
    """Compute lightweight summary statistics and disk usage for a project."""
    folder = resolve_project_folder(project_id)
    load_result = _load_builder_session(folder)
    session = load_result.get("session", {})

    scenes = get_project_scenes(project_id)

    scenes_with_image = sum(1 for s in scenes if s.get("approved_image"))
    scenes_with_video = sum(1 for s in scenes if s.get("rendered_video"))
    scenes_with_prompt = sum(1 for s in scenes if s.get("t2i_prompt") or s.get("i2v_prompt") or s.get("minimax_h3_prompt"))

    # Compute disk usage
    total_bytes = 0
    try:
        for root_dir, _, filenames in os.walk(folder):
            for fname in filenames:
                try:
                    total_bytes += os.path.getsize(os.path.join(root_dir, fname))
                except OSError:
                    pass
    except OSError:
        pass

    # Latents check
    dirty_latents = []
    try:
        dirty_latents = SceneLatentManager.list_dirty(folder)
    except Exception:
        pass

    audio_path = str(session.get("audio_path") or "").strip()

    return {
        "project_id": get_project_id(folder),
        "total_scenes": len(scenes),
        "scenes_with_prompt": scenes_with_prompt,
        "scenes_with_image": scenes_with_image,
        "scenes_with_video": scenes_with_video,
        "dirty_latents_count": len(dirty_latents),
        "audio_attached": bool(audio_path and os.path.isfile(audio_path)),
        "audio_duration": float(session.get("audio_duration", 0.0) or 0.0),
        "disk_usage_bytes": total_bytes,
        "revision": int(session.get("revision") or session.get("builder_save_revision") or 0),
    }


def get_project_assets(project_id: str) -> List[Dict[str, Any]]:
    """Scan and list all media assets belonging to a project."""
    folder = resolve_project_folder(project_id)
    assets: List[Dict[str, Any]] = []

    subfolders = [
        ("images", "image"),
        ("scene_videos", "video"),
        ("final_videos", "video"),
        ("scene_audio", "audio"),
        ("project_audio", "audio"),
        ("latents", "latent"),
    ]

    for sub, kind in subfolders:
        dir_path = os.path.join(folder, sub)
        if not os.path.isdir(dir_path):
            continue
        try:
            for fname in sorted(os.listdir(dir_path)):
                rel_path = os.path.join(sub, fname)
                asset = _build_asset_dict(folder, rel_path, kind)
                if asset:
                    assets.append(asset)
        except OSError:
            continue

    return assets
