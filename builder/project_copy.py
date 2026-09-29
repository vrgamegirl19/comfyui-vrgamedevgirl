"""Copying, branching and rebasing Video Builder projects and their assets."""

import json
import os
import re
import shutil
from ..core.atomic_write import atomic_write_json

from .paths import _context_folder, _copy_file_if_exists, _copy_file_into_folder, _images_folder, _is_inside_folder, _is_internal_approved_image_path, _looks_like_filesystem_path, _unique_preview_path
from .audio import _copy_or_convert_project_audio, _scene_audio_folder, _scene_audio_path
from .media import _scene_image_path


_BRANCH_KEEP_KEYS = ("lyrics", "notes", "prompts", "mappings", "images", "videos", "overlays")


_BRANCH_VIDEO_DIRS = {"rendered_scene_videos", "rendered_scene_videos_backup", "scene_video_thumbnails"}


_BRANCH_IMAGE_DIRS = {"zimage_approved", "scene_image_previews"}


_BRANCH_SEGMENT_NOTE_KEYS = ("timeline_note", "notes", "i2v_notes", "flux_notes", "nb_notes", "enhance_notes", "story_beat")


_BRANCH_SEGMENT_PROMPT_KEYS = (
    "t2i_prompt", "flux_prompt", "nb_prompt", "enhance_prompt", "i2v_prompt", "t2v_prompt",
    "flow_gpt_prompt", "ernie_t2i_prompt", "krea2_t2i_prompt", "minimax_h3_prompt", "minimax_h3_pass2_prompt",
)


_BRANCH_SEGMENT_IMAGE_KEYS = (
    "custom_image_path", "custom_image_data", "custom_image_name", "approved_image_path", "image",
    "image_history", "image_history_index", "ref_image_path", "flux_subject_image_path",
    "flux_location_image_path", "image_output", "image_status", "minimax_h3_continuity_frame_path",
    "adjust_preview_image_path",
)


_BRANCH_SEGMENT_VIDEO_KEYS = (
    "video_path", "video_folder", "video_thumbnail_path", "video_history", "video_thumbnail_history",
    "video_backup_paths", "video_backup_thumbnail_paths", "video_history_index", "video_output",
    "video_original_path", "video_original_thumbnail_path", "video_source_path",
    "minimax_h3_continuity_source_video_path", "minimax_h3_video_references",
    "minimax_h3_stage1_path", "minimax_h3_stage1_source_path", "minimax_h3_stage1_backup_path",
    "minimax_h3_stage2_path", "minimax_h3_stage2_source_path", "minimax_h3_stage2_backup_path",
)


_MINIMAX_H3_STAGE_PATH_KEYS = (
    "minimax_h3_stage1_path", "minimax_h3_stage1_source_path", "minimax_h3_stage1_backup_path",
    "minimax_h3_stage2_path", "minimax_h3_stage2_source_path", "minimax_h3_stage2_backup_path",
)


_BRANCH_SEGMENT_MAPPING_KEYS = (
    "subject_ids", "location_id", "reference_subject_ids", "reference_location_id",
    "flux_image_ingredients", "nb_image_ingredients", "minimax_h3_reference_keys",
)


_BRANCH_SEGMENT_AUDIO_KEYS = ("custom_audio_path", "custom_audio_name")


def _normalize_branch_keep(payload):
    raw = payload.get("keep") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return {key: True for key in _BRANCH_KEEP_KEYS}
    return {key: bool(raw.get(key, False)) for key in _BRANCH_KEEP_KEYS}


def _rewrite_project_path_string(text, old_folder, new_folder):
    value = str(text or "")
    stripped = value.strip().strip('"')
    if not stripped or not old_folder or not new_folder or not _looks_like_filesystem_path(stripped):
        return value
    try:
        old_abs = os.path.abspath(old_folder)
        new_abs = os.path.abspath(new_folder)
        raw_abs = os.path.abspath(stripped)
    except (OSError, ValueError, TypeError):
        return value
    if not _is_inside_folder(raw_abs, old_abs):
        return value
    relative = os.path.relpath(raw_abs, old_abs)
    rewritten = os.path.abspath(os.path.join(new_abs, relative))
    old_exists = os.path.exists(raw_abs)
    new_exists = os.path.exists(rewritten)
    if old_exists and not new_exists:
        return ""
    return rewritten


def _rewrite_project_paths(value, old_folder, new_folder):
    if isinstance(value, dict):
        return {key: _rewrite_project_paths(item, old_folder, new_folder) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_project_paths(item, old_folder, new_folder) for item in value]
    if isinstance(value, str):
        return _rewrite_project_path_string(value, old_folder, new_folder)
    return value


def _clear_remaining_source_paths(value, old_folder):
    if not old_folder:
        return value
    if isinstance(value, dict):
        return {key: _clear_remaining_source_paths(item, old_folder) for key, item in value.items()}
    if isinstance(value, list):
        return [_clear_remaining_source_paths(item, old_folder) for item in value]
    if isinstance(value, str) and _looks_like_filesystem_path(value):
        try:
            if _is_inside_folder(os.path.abspath(value.strip().strip('"')), old_folder):
                return ""
        except (OSError, ValueError, TypeError):
            return value
    return value


def _rewrite_project_json_file(path, old_folder, new_folder):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    data = _clear_remaining_source_paths(_rewrite_project_paths(data, old_folder, new_folder), old_folder)
    atomic_write_json(path, data)


def _rewrite_project_sidecar_json(project_folder, old_folder):
    files = [
        os.path.join(project_folder, "storyboard", "storyboard.json"),
        os.path.join(project_folder, "subject_location", "reference_descriptions.json"),
    ]
    prompts = os.path.join(project_folder, "prompts")
    if os.path.isdir(prompts):
        for name in os.listdir(prompts):
            if name.lower().endswith(".json"):
                files.append(os.path.join(prompts, name))
    for path in files:
        try:
            _rewrite_project_json_file(path, old_folder, project_folder)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue


def _is_project_root_export_video(name):
    stem, ext = os.path.splitext(str(name or ""))
    if ext.lower() != ".mp4":
        return False
    upper = stem.upper()
    return upper.startswith("FINAL_VIDEO") or upper.startswith("PREVIEW_SCENES_")


def _remove_project_root_export_videos(project_folder):
    if not os.path.isdir(project_folder):
        return
    for name in os.listdir(project_folder):
        if not _is_project_root_export_video(name):
            continue
        path = os.path.join(project_folder, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _branch_copy_ignore(keep, source=""):
    skip_names = {"__pycache__", "session_backups"}
    source_abs = os.path.normcase(os.path.abspath(source)) if source else ""
    if not keep.get("videos"):
        skip_names |= _BRANCH_VIDEO_DIRS
        skip_names.add("render_logs")
    if not keep.get("images"):
        skip_names |= _BRANCH_IMAGE_DIRS
    if not keep.get("mappings"):
        skip_names.add("subject_location")
    if not keep.get("prompts"):
        skip_names.add("prompts")
    if not keep.get("notes") and not keep.get("prompts"):
        skip_names.add("storyboard")

    def ignore(directory, names):
        ignored = [name for name in names if name in skip_names or name.endswith(".tmp")]
        basename = os.path.basename(directory)
        if source_abs and os.path.normcase(os.path.abspath(directory)) == source_abs:
            ignored.extend(name for name in names if _is_project_root_export_video(name))
        if not keep.get("mappings") and basename == "project_context" and "flux_references" in names:
            ignored.append("flux_references")
        if not keep.get("images") and basename == "storyboard":
            ignored.extend(
                name for name in names
                if os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
            )
        return ignored

    return ignore


def _copy_project_tree(source, target, keep):
    if not source or not os.path.isdir(source):
        os.makedirs(target, exist_ok=True)
        return
    os.makedirs(target, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=_branch_copy_ignore(keep, source))
    _remove_project_root_export_videos(target)
    shutil.rmtree(os.path.join(target, "session_backups"), ignore_errors=True)


def _empty_keep_value(key, current):
    if key.endswith("_index"):
        return -1
    if key in {"image", "image_output", "video_output"}:
        return None
    if key == "image_status" or key == "video_status":
        return "none"
    if isinstance(current, list):
        return []
    if isinstance(current, dict):
        return {}
    if isinstance(current, bool):
        return False
    return ""


def _strip_segment_keep_fields(segment, keep):
    if not isinstance(segment, dict):
        return segment
    if not keep.get("lyrics"):
        segment["lyric_text"] = ""
    if not keep.get("notes"):
        for key in _BRANCH_SEGMENT_NOTE_KEYS:
            if key in segment:
                segment[key] = ""
    if not keep.get("prompts"):
        for key in _BRANCH_SEGMENT_PROMPT_KEYS:
            if key in segment:
                segment[key] = ""
    if not keep.get("mappings"):
        for key in _BRANCH_SEGMENT_MAPPING_KEYS:
            if key in segment:
                segment[key] = _empty_keep_value(key, segment.get(key))
    if not keep.get("images"):
        for key in _BRANCH_SEGMENT_IMAGE_KEYS:
            if key in segment:
                segment[key] = _empty_keep_value(key, segment.get(key))
        segment["preview_mode"] = "image"
    if not keep.get("videos"):
        for key in _BRANCH_SEGMENT_VIDEO_KEYS:
            if key in segment:
                segment[key] = _empty_keep_value(key, segment.get(key))
        segment["video_status"] = "none"
        if isinstance(segment.get("minimax_h3_video_references"), list):
            segment["minimax_h3_video_references"] = []
    if not keep.get("images") and not keep.get("videos"):
        for key in _BRANCH_SEGMENT_AUDIO_KEYS:
            if key in segment:
                segment[key] = ""
    return segment


def _apply_branch_keep_to_session(session, keep):
    if not isinstance(session, dict):
        return {}
    if all(keep.get(key) for key in _BRANCH_KEEP_KEYS):
        return session
    segments = session.get("segments") if isinstance(session.get("segments"), list) else []
    session["segments"] = [_strip_segment_keep_fields(item if isinstance(item, dict) else {}, keep) for item in segments]
    overlays = session.get("overlay_segments") if isinstance(session.get("overlay_segments"), list) else []
    if keep.get("overlays"):
        session["overlay_segments"] = [_strip_segment_keep_fields(item if isinstance(item, dict) else {}, keep) for item in overlays]
    else:
        session["overlay_segments"] = []
        if isinstance(session.get("overlay_track"), dict):
            session["overlay_track"] = {**session["overlay_track"], "enabled": False}
    if not keep.get("notes"):
        session["timeline_markers"] = []
    if not keep.get("prompts"):
        for key in ("prompt_json_path", "i2v_motion_json_path", "theme_style_path", "story_idea_path", "subject_scene_path"):
            session[key] = ""
    if not keep.get("mappings"):
        session["flux_reference_builder"] = {}
        session["id_lora_reference_builder"] = {}
    if not keep.get("images"):
        session["flux_global_image_ingredients"] = []
        session["use_flux_global_image_ingredients"] = False
        session["builder_agent_reference_images"] = []
        session["builder_story_reference_images"] = []
        session["builder_story_source_path"] = ""
    if not keep.get("lyrics"):
        mapper = session.get("lyric_mapper") if isinstance(session.get("lyric_mapper"), dict) else {}
        if mapper:
            mapper["source_text"] = ""
            session["lyric_mapper"] = mapper
    return session


def _branch_scene_number(segment, index):
    if isinstance(segment, dict):
        for key in ("scene_slot_number", "overlay_slot_number", "scene_number"):
            try:
                value = int(segment.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value
    return max(1, int(index or 1))


def _copy_external_media_into_branch(source_path, project_folder, scene_number, key, copied):
    text = str(source_path or "").strip().strip('"')
    if not text:
        return ""
    try:
        abs_path = os.path.abspath(text)
    except (OSError, ValueError, TypeError):
        return ""
    if _is_inside_folder(abs_path, project_folder):
        return abs_path if os.path.isfile(abs_path) else ""
    if abs_path in copied:
        return copied[abs_path]
    if not os.path.isfile(abs_path):
        return ""
    backup_dir = os.path.join(project_folder, "rendered_scene_videos_backup", f"scene_{int(scene_number):04d}")
    os.makedirs(backup_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    ext = os.path.splitext(abs_path)[1] or ".mp4"
    safe_key = re.sub(r"[^A-Za-z0-9_-]+", "_", str(key or "media")).strip("_") or "media"
    target = os.path.join(backup_dir, f"{safe_key}_{stem}{ext}")
    index = 2
    while os.path.exists(target):
        target = os.path.join(backup_dir, f"{safe_key}_{stem}_{index:02d}{ext}")
        index += 1
    shutil.copy2(abs_path, target)
    copied[abs_path] = target
    return target


def _relocate_external_minimax_stage_media(session, project_folder):
    if not isinstance(session, dict) or not project_folder:
        return session
    copied = {}

    def relocate_segment(segment, index):
        if not isinstance(segment, dict):
            return
        scene_number = _branch_scene_number(segment, index)
        for key in _MINIMAX_H3_STAGE_PATH_KEYS:
            if key in segment:
                segment[key] = _copy_external_media_into_branch(
                    segment.get(key, ""), project_folder, scene_number, key, copied,
                )
        backups = segment.get("video_backup_paths")
        if isinstance(backups, list):
            segment["video_backup_paths"] = [
                _copy_external_media_into_branch(item, project_folder, scene_number, "video_backup", copied)
                for item in backups
            ]
            segment["video_backup_paths"] = [item for item in segment["video_backup_paths"] if item]
        thumbs = segment.get("video_backup_thumbnail_paths")
        if isinstance(thumbs, list):
            segment["video_backup_thumbnail_paths"] = [
                _copy_external_media_into_branch(item, project_folder, scene_number, "video_backup_thumb", copied)
                for item in thumbs
            ]
            segment["video_backup_thumbnail_paths"] = [item for item in segment["video_backup_thumbnail_paths"] if item]
        if segment.get("video_source_path"):
            segment["video_source_path"] = _copy_external_media_into_branch(
                segment.get("video_source_path", ""), project_folder, scene_number, "video_source", copied,
            )

    segments = session.get("segments") if isinstance(session.get("segments"), list) else []
    for index, segment in enumerate(segments, start=1):
        relocate_segment(segment, index)
    overlays = session.get("overlay_segments") if isinstance(session.get("overlay_segments"), list) else []
    for index, segment in enumerate(overlays, start=1):
        relocate_segment(segment, index)
    return session


def _strip_storyboard_images(project_folder):
    path = os.path.join(project_folder, "storyboard", "storyboard.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    scenes = data.get("scenes") if isinstance(data.get("scenes"), list) else []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene["image_path"] = ""
        scene["image_data"] = ""
        scene["image_name"] = ""
    data["scenes"] = scenes
    atomic_write_json(path, data)


def _prune_unkept_project_folders(project_folder, keep):
    removals = []
    if not keep.get("videos"):
        removals.extend(_BRANCH_VIDEO_DIRS)
        removals.append("render_logs")
    if not keep.get("images"):
        removals.extend(_BRANCH_IMAGE_DIRS)
        _strip_storyboard_images(project_folder)
    if not keep.get("mappings"):
        removals.append("subject_location")
        flux_refs = os.path.join(project_folder, "project_context", "flux_references")
        if os.path.isdir(flux_refs):
            shutil.rmtree(flux_refs, ignore_errors=True)
    if not keep.get("prompts"):
        removals.append("prompts")
    if not keep.get("notes") and not keep.get("prompts"):
        removals.append("storyboard")
    for name in removals:
        path = os.path.join(project_folder, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def _project_rebased_path(project_folder, old_project_folder, raw_path):
    text = str(raw_path or "").strip().strip('"')
    if not text or not old_project_folder:
        return ""
    try:
        old_abs = os.path.abspath(old_project_folder)
        raw_abs = os.path.abspath(text)
        if _is_inside_folder(raw_abs, old_abs):
            return os.path.abspath(os.path.join(project_folder, os.path.relpath(raw_abs, old_abs)))
    except Exception:
        return ""
    return ""


def _snapshot_project_assets(project_folder, session, audio_path, old_project_folder=""):
    project_folder = os.path.abspath(project_folder)
    if audio_path and os.path.isfile(audio_path):
        copied_audio = _copy_or_convert_project_audio(
            audio_path,
            os.path.join(project_folder, "project_audio"),
            "project_audio" + os.path.splitext(audio_path)[1],
        )
        if copied_audio:
            audio_path = copied_audio
    elif old_project_folder:
        rebased_audio = _project_rebased_path(project_folder, old_project_folder, audio_path)
        if rebased_audio:
            audio_path = rebased_audio

    context_map = {
        "prompt_json_path": "ConceptPrompts.txt",
        "theme_style_path": "themestyle.txt",
        "story_idea_path": "storyconcept.txt",
        "subject_scene_path": "subjectsandscenes.txt",
    }
    for key, filename in context_map.items():
        raw_path = str(session.get(key, "") or "").strip()
        if raw_path and os.path.isfile(raw_path):
            copied_path = _copy_file_into_folder(raw_path, _context_folder(project_folder), filename)
            if copied_path:
                session[key] = copied_path
        else:
            rebased_path = _project_rebased_path(project_folder, old_project_folder, raw_path)
            if rebased_path:
                session[key] = rebased_path

    return audio_path, session


def _copy_reference_asset(project_folder, scene_number, key, source_path):
    source = str(source_path or "").strip().strip('"')
    if not source or not os.path.isfile(source):
        return ""
    ext = os.path.splitext(source)[1].lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        ext = ".bin"
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key or "asset")).strip("_") or "asset"
    folder = os.path.join(_context_folder(project_folder), f"scene_{max(1, int(scene_number or 1)):04d}")
    return _copy_file_if_exists(source, os.path.join(folder, f"{safe_key}{ext}"))


def _copy_session_assets_to_project(project_folder, session):
    project_folder = os.path.abspath(project_folder)
    if isinstance(session.get("flux_global_image_ingredients"), list):
        global_folder = os.path.join(_context_folder(project_folder), "flux_global")
        for ingredient_index, ingredient in enumerate(session["flux_global_image_ingredients"], start=1):
            if not isinstance(ingredient, dict):
                continue
            source = str(ingredient.get("path", "") or "").strip().strip('"')
            if not source or not os.path.isfile(source):
                continue
            ext = os.path.splitext(source)[1].lower() or ".png"
            copied = _copy_file_if_exists(source, os.path.join(global_folder, f"global_ingredient_{ingredient_index}{ext}"))
            if copied:
                ingredient["path"] = copied
    segments = session.get("segments", [])
    if not isinstance(segments, list):
        session["segments"] = []
        return session

    for scene_number, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue

        approved = str(segment.get("approved_image_path", "") or "").strip()
        if approved and os.path.isfile(approved):
            ext = os.path.splitext(approved)[1] or ".png"
            segment["approved_image_path"] = _copy_file_if_exists(
                approved,
                _scene_image_path(project_folder, scene_number, ext),
            )

        history = segment.get("image_history", [])
        new_history = []
        if isinstance(history, list):
            for item in history:
                item_path = str(item or "").strip()
                if not item_path or not os.path.isfile(item_path):
                    continue
                if item_path == approved or _is_internal_approved_image_path(item_path):
                    continue
                ext = os.path.splitext(item_path)[1] or ".png"
                copied = _copy_file_if_exists(item_path, _unique_preview_path(project_folder, scene_number, ext))
                if copied and copied not in new_history:
                    new_history.append(copied)
        segment["image_history"] = new_history
        if new_history:
            try:
                current_index = int(segment.get("image_history_index", len(new_history) - 1) or 0)
            except (TypeError, ValueError):
                current_index = len(new_history) - 1
            segment["image_history_index"] = max(0, min(len(new_history) - 1, current_index))
        else:
            segment["image_history_index"] = -1

        video_path = str(segment.get("video_path", "") or "").strip()
        if video_path and os.path.isfile(video_path):
            target_video = os.path.join(project_folder, "rendered_scene_videos", f"video_{scene_number:04d}-audio.mp4")
            segment["video_path"] = _copy_file_if_exists(video_path, target_video)
            segment["video_folder"] = os.path.dirname(segment["video_path"])
            segment["video_status"] = "done"

        custom_audio = str(segment.get("custom_audio_path", "") or "").strip()
        if custom_audio and os.path.isfile(custom_audio):
            ext = os.path.splitext(custom_audio)[1] or ".wav"
            segment["custom_audio_path"] = _copy_file_if_exists(
                custom_audio,
                _scene_audio_path(project_folder, scene_number, ext),
            )

        for key in (
            "custom_image_path",
            "ref_image_path",
            "flux_subject_image_path",
            "flux_location_image_path",
        ):
            copied = _copy_reference_asset(project_folder, scene_number, key, segment.get(key, ""))
            if copied:
                segment[key] = copied
        if isinstance(segment.get("flux_image_ingredients"), list):
            for ingredient_index, ingredient in enumerate(segment["flux_image_ingredients"], start=1):
                if not isinstance(ingredient, dict):
                    continue
                copied = _copy_reference_asset(
                    project_folder,
                    scene_number,
                    f"flux_ingredient_{ingredient_index}",
                    ingredient.get("path", ""),
                )
                if copied:
                    ingredient["path"] = copied

    overlay_segments = session.get("overlay_segments", [])
    if isinstance(overlay_segments, list):
        overlay_segments = _assign_overlay_scene_numbers(overlay_segments)
        for overlay_index, segment in enumerate(overlay_segments, start=1):
            if not isinstance(segment, dict):
                continue
            scene_number = _overlay_scene_number(segment, overlay_index)
            segment["track"] = "overlay"
            approved = str(segment.get("approved_image_path", "") or "").strip()
            if approved and os.path.isfile(approved):
                ext = os.path.splitext(approved)[1] or ".png"
                segment["approved_image_path"] = _copy_file_if_exists(
                    approved,
                    _scene_image_path(project_folder, scene_number, ext),
                )
            video_path = str(segment.get("video_path", "") or "").strip()
            if video_path and os.path.isfile(video_path):
                target_video = os.path.join(project_folder, "rendered_scene_videos", f"video_{scene_number:04d}-audio.mp4")
                segment["video_path"] = _copy_file_if_exists(video_path, target_video)
                segment["video_folder"] = os.path.dirname(segment["video_path"])
                segment["video_status"] = "done"
            for key in (
                "custom_image_path",
                "ref_image_path",
                "flux_subject_image_path",
                "flux_location_image_path",
            ):
                copied = _copy_reference_asset(project_folder, scene_number, key, segment.get(key, ""))
                if copied:
                    segment[key] = copied

    return session


def _rebase_project_owned_paths(project_folder, old_project_folder, session):
    if not old_project_folder:
        return session
    session = _rewrite_project_paths(session, old_project_folder, project_folder)
    session = _clear_remaining_source_paths(session, old_project_folder)
    _rewrite_project_sidecar_json(project_folder, old_project_folder)
    return session


def _project_path_candidates(project_folder, old_project_folder, raw_path, scene_number=None):
    text = str(raw_path or "").strip().strip('"')
    if not text:
        return []
    abs_text = os.path.abspath(text)
    candidates = [text, abs_text]
    if old_project_folder:
        try:
            old_abs = os.path.abspath(old_project_folder)
            if _is_inside_folder(abs_text, old_abs):
                candidates.append(os.path.join(project_folder, os.path.relpath(abs_text, old_abs)))
        except Exception:
            pass
    base = os.path.basename(text)
    if base:
        candidates.extend([
            os.path.join(project_folder, base),
            os.path.join(_images_folder(project_folder), base),
            os.path.join(_context_folder(project_folder), base),
            os.path.join(project_folder, "project_audio", base),
            os.path.join(_scene_audio_folder(project_folder), base),
            os.path.join(project_folder, "rendered_scene_videos", base),
        ])
    if scene_number:
        scene = int(scene_number)
        candidates.extend([
            _scene_image_path(project_folder, scene, ".png"),
            _scene_image_path(project_folder, scene, ".jpg"),
            _scene_image_path(project_folder, scene, ".jpeg"),
            _scene_image_path(project_folder, scene, ".webp"),
            _scene_audio_path(project_folder, scene, ".wav"),
            _scene_audio_path(project_folder, scene, ".mp3"),
            _scene_audio_path(project_folder, scene, ".m4a"),
            os.path.join(project_folder, "rendered_scene_videos", f"video_{scene:04d}-audio.mp4"),
        ])
    return candidates


def _overlay_scene_number(segment, fallback_index):
    if isinstance(segment, dict):
        for key in ("overlay_slot_number", "scene_slot_number", "slot_number"):
            try:
                value = int(segment.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if value >= 10001:
                return value
    return 10000 + int(fallback_index or 1)


def _assign_overlay_scene_numbers(overlay_segments):
    if not isinstance(overlay_segments, list):
        return overlay_segments
    used = set()
    existing = []
    for segment in overlay_segments:
        if isinstance(segment, dict):
            value = _overlay_scene_number(segment, 0)
            if value >= 10001:
                existing.append(value)
    next_slot = max([10000] + existing) + 1
    for index, segment in enumerate(overlay_segments, start=1):
        if not isinstance(segment, dict):
            continue
        slot = _overlay_scene_number(segment, index)
        if slot in used:
            slot = max(next_slot, 10000 + index)
            while slot in used:
                slot += 1
            next_slot = slot + 1
        segment["overlay_slot_number"] = slot
        used.add(slot)
    return overlay_segments


def _resolve_project_asset_path(project_folder, old_project_folder, raw_path, scene_number=None):
    for candidate in _project_path_candidates(project_folder, old_project_folder, raw_path, scene_number):
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return str(raw_path or "")
