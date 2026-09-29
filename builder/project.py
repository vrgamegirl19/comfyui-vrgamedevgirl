"""Video Builder projects: create, save, load, export and import, sessions, model defaults, and per-scene file renumbering."""

import json
import os
import re
import shutil
import time
import tempfile
import zipfile
import threading
import folder_paths
from ..core.atomic_write import atomic_write_json, atomic_write_text
from ..minimax.latent_manager import SceneLatentManager

from .paths import _context_folder, _copy_file_if_exists, _default_project_folder, _images_folder, _is_internal_approved_image_path, _model_defaults_path, _newest_file, _prompts_folder, _render_logs_folder, _resolve_existing_file, _safe_project_name, _scene_notes_path, _scene_preview_folder, _session_path, _unique_folder_path, _vrgdg_textfile_path, _wizard_draft_path, _wizard_folder, _wizard_lyrics_path
from .audio import _convert_audio_to_wav, _segments_to_srt, _srt_path
from .media import _scene_image_path
from .project_copy import _apply_branch_keep_to_session, _assign_overlay_scene_numbers, _copy_project_tree, _copy_session_assets_to_project, _normalize_branch_keep, _overlay_scene_number, _project_rebased_path, _prune_unkept_project_folders, _rebase_project_owned_paths, _relocate_external_minimax_stage_media, _resolve_project_asset_path, _snapshot_project_assets


_BUILDER_SAVE_LOCK = threading.RLock()


def _default_context_paths():
    return {
        "concept_prompts_path": _vrgdg_textfile_path("ConceptPrompts", "ConceptPrompts.txt"),
        "i2v_motion_notes_path": _vrgdg_textfile_path("I2VMotionNotes", "I2VMotionNotes.txt"),
        "theme_style_path": _vrgdg_textfile_path("themestyle", "themestyle.txt"),
        "story_idea_path": _vrgdg_textfile_path("storyconcept", "storyconcept.txt"),
        "subject_scene_path": _vrgdg_textfile_path("subjectandscenes", "subjectsandscenes.txt"),
    }


def _project_prompt_creator_paths(project_folder):
    folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not folder:
        raise ValueError("Create or load a project before importing Prompt Creator data.")

    context = _context_folder(folder)
    audio_folder = os.path.join(folder, "audio")
    paths = {
        "project_folder": folder,
        "audio_path": _newest_file(audio_folder, (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".mp4")),
        "srt_path": _srt_path(folder),
        "lyric_segments_path": os.path.join(folder, "prompts", "lyric_segments.json"),
        "concept_prompts_path": os.path.join(context, "ConceptPrompts.txt"),
        "i2v_motion_notes_path": os.path.join(context, "I2VMotionNotes.txt"),
        "theme_style_path": os.path.join(context, "themestyle.txt"),
        "story_idea_path": os.path.join(context, "storyconcept.txt"),
        "subject_scene_path": os.path.join(context, "subjectsandscenes.txt"),
    }
    exists = {key: bool(value and os.path.isfile(value)) for key, value in paths.items() if key.endswith("_path")}
    paths["exists"] = exists
    paths["ready"] = bool(exists.get("srt_path") and exists.get("concept_prompts_path"))
    return paths


def _json_file_has_text_values(path):
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except Exception:
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                return bool(handle.read().strip())
        except Exception:
            return False
    if isinstance(data, dict):
        return any(str(value or "").strip() for value in data.values())
    if isinstance(data, list):
        return any(str(item or "").strip() for item in data)
    return False


def _is_prompt_creator_output_folder(context_folder):
    marker_path = os.path.join(context_folder, "prompt_creator_output.json")
    if os.path.isfile(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if str(data.get("type", "") or "") == "vrgdg_prompt_creator_output":
                return True
        except Exception:
            return True

    project_folder = os.path.dirname(context_folder)
    legacy_markers = (
        os.path.join(project_folder, "prompt_creator_draft.json"),
        os.path.join(project_folder, "prompts", "lyric_segments.json"),
        os.path.join(context_folder, "full_lyrics.txt"),
    )
    return any(os.path.isfile(path) for path in legacy_markers)


def _last_prompt_creator_pointer_source(exclude_project_folder=""):
    pointer_path = os.path.join(folder_paths.get_output_directory(), "VRGDG_LastPromptCreatorProject.json")
    if not os.path.isfile(pointer_path):
        return "", ""
    try:
        with open(pointer_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return "", ""
    if str(data.get("type", "") or "") != "vrgdg_last_prompt_creator_project":
        return "", ""
    project_folder = os.path.abspath(str(data.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        return "", ""
    exclude = os.path.normcase(os.path.abspath(str(exclude_project_folder or ""))) if exclude_project_folder else ""
    if exclude and os.path.normcase(project_folder) == exclude:
        return "", ""
    raw_context = str(data.get("context_folder", "") or "").strip().strip('"')
    context_folder = os.path.abspath(raw_context) if raw_context else _context_folder(project_folder)
    concept_path = os.path.join(context_folder, "ConceptPrompts.txt")
    srt_path = _srt_path(project_folder)
    if not os.path.isfile(concept_path) or not os.path.isfile(srt_path):
        return "", ""
    if not _json_file_has_text_values(concept_path):
        return "", ""
    return project_folder, context_folder


def _latest_prompt_creator_source(exclude_project_folder=""):
    pointer_project, pointer_context = _last_prompt_creator_pointer_source(exclude_project_folder)
    if pointer_project and pointer_context:
        return pointer_project, pointer_context

    output_dir = folder_paths.get_output_directory()
    exclude = os.path.normcase(os.path.abspath(str(exclude_project_folder or ""))) if exclude_project_folder else ""
    candidates = []
    for root, dirs, _files in os.walk(output_dir):
        if os.path.basename(root) != "project_context":
            continue
        project_folder = os.path.dirname(root)
        if exclude and os.path.normcase(os.path.abspath(project_folder)) == exclude:
            continue
        concept_path = os.path.join(root, "ConceptPrompts.txt")
        srt_path = _srt_path(project_folder)
        if not os.path.isfile(concept_path) or not os.path.isfile(srt_path):
            continue
        if not _is_prompt_creator_output_folder(root):
            continue
        if not _json_file_has_text_values(concept_path):
            continue
        motion_path = os.path.join(root, "I2VMotionNotes.txt")
        has_motion = _json_file_has_text_values(motion_path)
        related = [
            concept_path,
            srt_path,
            motion_path,
            os.path.join(root, "themestyle.txt"),
            os.path.join(root, "storyconcept.txt"),
            os.path.join(root, "subjectsandscenes.txt"),
        ]
        newest = max((os.path.getmtime(path) for path in related if os.path.isfile(path)), default=0)
        candidates.append((1 if has_motion else 0, newest, project_folder, root))
    if not candidates:
        raise ValueError("No previous Prompt Creator output was found. Run Prompt Creator first, then import it into this project.")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2], candidates[0][3]


def _copy_prompt_creator_outputs_from_source(project_folder, source_project_folder=""):
    target = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not target:
        raise ValueError("Create or load a project before importing Prompt Creator data.")
    os.makedirs(target, exist_ok=True)
    os.makedirs(_context_folder(target), exist_ok=True)
    os.makedirs(os.path.join(target, "audio"), exist_ok=True)
    if source_project_folder:
        source_project = os.path.abspath(str(source_project_folder or "").strip().strip('"'))
        source_context = _context_folder(source_project)
        if os.path.normcase(source_project) == os.path.normcase(target):
            return _project_prompt_creator_paths(target)
        if not os.path.isfile(os.path.join(source_context, "ConceptPrompts.txt")) or not os.path.isfile(_srt_path(source_project)):
            raise ValueError("The selected Prompt Creator project does not have saved ConceptPrompts.txt and builder_segments.srt outputs.")
    else:
        source_project, source_context = _latest_prompt_creator_source(target)
    copied = {}
    for filename in ("ConceptPrompts.txt", "I2VMotionNotes.txt", "themestyle.txt", "storyconcept.txt", "subjectsandscenes.txt", "subject.txt", "full_lyrics.txt"):
        source_path = os.path.join(source_context, filename)
        if os.path.isfile(source_path):
            copied[filename] = _copy_file_if_exists(source_path, os.path.join(_context_folder(target), filename))
    source_lyrics = os.path.join(source_project, "prompts", "lyric_segments.json")
    if os.path.isfile(source_lyrics):
        copied["lyric_segments.json"] = _copy_file_if_exists(source_lyrics, os.path.join(_prompts_folder(target), "lyric_segments.json"))
    source_srt = _srt_path(source_project)
    if os.path.isfile(source_srt):
        copied["builder_segments.srt"] = _copy_file_if_exists(source_srt, _srt_path(target))
    source_audio = _newest_file(os.path.join(source_project, "audio"), (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".mp4"))
    if source_audio:
        if os.path.splitext(source_audio)[1].lower() == ".m4a":
            copied["audio"] = _convert_audio_to_wav(source_audio, os.path.join(target, "audio", "project_audio.wav"))
        else:
            copied["audio"] = _copy_file_if_exists(source_audio, os.path.join(target, "audio", os.path.basename(source_audio)))
    result = _project_prompt_creator_paths(target)
    result["source_project_folder"] = source_project
    result["copied"] = copied
    return result


def _copy_latest_prompt_creator_outputs(project_folder):
    return _copy_prompt_creator_outputs_from_source(project_folder, "")


def _project_target_from_payload(payload, preferred_key="project_folder"):
    raw = str(payload.get(preferred_key, "") or "").strip().strip('"')
    if not raw:
        raw = str(payload.get("project_name", "") or "").strip().strip('"')
    if not raw:
        raw = f"VRGDG_Project_{time.strftime('%Y%m%d_%H%M%S')}"
    if os.path.isabs(raw) or os.path.dirname(raw):
        return os.path.abspath(raw)
    project_root = str(payload.get("project_root", "") or "").strip().strip('"')
    if project_root:
        if not os.path.isabs(project_root):
            raise ValueError("Custom project root must be a full absolute folder path.")
        return os.path.join(os.path.abspath(project_root), _safe_project_name(raw))
    return os.path.join(folder_paths.get_output_directory(), _safe_project_name(raw))


def _new_builder_project(payload):
    target = _unique_folder_path(_project_target_from_payload(payload, "project_folder"))
    os.makedirs(target, exist_ok=True)
    os.makedirs(_images_folder(target), exist_ok=True)
    os.makedirs(_prompts_folder(target), exist_ok=True)
    os.makedirs(_context_folder(target), exist_ok=True)
    os.makedirs(os.path.join(target, "latents"), exist_ok=True)
    for filename in ("ConceptPrompts.txt", "I2VMotionNotes.txt", "themestyle.txt", "storyconcept.txt", "subjectsandscenes.txt", "full_lyrics.txt"):
        path = os.path.join(_context_folder(target), filename)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("")
    return {
        "project_folder": target,
        "session_path": _session_path(target),
        "srt_path": _srt_path(target),
        "images_folder": _images_folder(target),
        "prompts_folder": _prompts_folder(target),
        "context_folder": _context_folder(target),
        "concept_prompts_path": os.path.join(_context_folder(target), "ConceptPrompts.txt"),
        "i2v_motion_notes_path": os.path.join(_context_folder(target), "I2VMotionNotes.txt"),
        "theme_style_path": os.path.join(_context_folder(target), "themestyle.txt"),
        "story_idea_path": os.path.join(_context_folder(target), "storyconcept.txt"),
        "subject_scene_path": os.path.join(_context_folder(target), "subjectsandscenes.txt"),
    }


def _save_builder_project_as(payload):
    source = os.path.abspath(str(payload.get("source_project_folder", "") or "").strip().strip('"'))
    if not source:
        source = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))

    target = _project_target_from_payload(payload, "target_project_folder")
    target = _unique_folder_path(target)
    if source and os.path.isdir(source):
        try:
            common = os.path.commonpath([source, target])
        except ValueError:
            common = ""
    else:
        common = ""
    if source and common == source:
        raise ValueError("Save Project As target cannot be inside the current project folder.")

    keep = _normalize_branch_keep(payload)
    os.makedirs(target, exist_ok=True)
    os.makedirs(_images_folder(target), exist_ok=True)
    os.makedirs(_prompts_folder(target), exist_ok=True)
    os.makedirs(_context_folder(target), exist_ok=True)
    if source and os.path.isdir(source):
        _copy_project_tree(source, target, keep)
        try:
            SceneLatentManager.copy_latents_folder(source, target)
        except Exception as exc:
            print(f"[VRGDG Latent] Failed to copy latents during branch: {exc}")

    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    session = _apply_branch_keep_to_session(session, keep)
    _prune_unkept_project_folders(target, keep)
    segments = session.get("segments", [])
    if not isinstance(segments, list):
        segments = []
        session["segments"] = segments
    overlay_segments = session.get("overlay_segments", [])
    if not isinstance(overlay_segments, list):
        overlay_segments = []
        session["overlay_segments"] = overlay_segments
    overlay_segments = _assign_overlay_scene_numbers(overlay_segments)
    session["overlay_segments"] = overlay_segments
    audio_raw = str(payload.get("audio_path", "") or "").strip().strip('"')
    audio_path = _resolve_existing_file(audio_raw, "Audio file") if audio_raw else ""
    audio_path, session = _snapshot_project_assets(target, session, audio_path, source)
    session = _copy_session_assets_to_project(target, session)
    session = _rebase_project_owned_paths(target, source, session)
    if keep.get("videos"):
        session = _relocate_external_minimax_stage_media(session, target)
    segments = session.get("segments") if isinstance(session.get("segments"), list) else []
    session = {
        **session,
        "audio_path": audio_path,
        "project_folder": target,
        "updated": time.time(),
        "segments": segments,
    }

    with open(_session_path(target), "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(_srt_path(target), "w", encoding="utf-8") as handle:
        handle.write(_segments_to_srt(segments))
    scene_notes_path = _write_scene_notes_json(target, segments)
    if keep.get("mappings"):
        _save_reference_descriptions(target, session)
    if keep.get("notes") or keep.get("prompts"):
        _save_project_context_files(target, session)
    return {
        "project_folder": target,
        "session_path": _session_path(target),
        "srt_path": _srt_path(target),
        "scene_notes_path": scene_notes_path,
        "images_folder": _images_folder(target),
        "prompts_folder": _prompts_folder(target),
        "context_folder": _context_folder(target),
        "session": session,
    }


def _fallback_project_context_text(filename, session):
    """Build portable context from canonical session state when no legacy file exists."""
    session = session if isinstance(session, dict) else {}
    story_layer = session.get("builder_story_layer") if isinstance(session.get("builder_story_layer"), dict) else {}
    storyboard = session.get("builder_storyboard_defaults") if isinstance(session.get("builder_storyboard_defaults"), dict) else {}
    segments = session.get("segments") if isinstance(session.get("segments"), list) else []
    refs = session.get("flux_reference_builder") if isinstance(session.get("flux_reference_builder"), dict) else {}

    if filename == "storyconcept.txt":
        lines = []
        for label, key in (
            ("Overall story idea", "overall_story_idea"),
            ("Story arc", "user_story_arc"),
            ("Song/story brief", "song_story_brief"),
        ):
            value = str(story_layer.get(key) or "").strip()
            if value:
                lines.append(f"{label}: {value}")
        if not lines:
            for index, segment in enumerate(segments, start=1):
                if not isinstance(segment, dict):
                    continue
                value = str(segment.get("scene_summary") or segment.get("timeline_note") or segment.get("t2i_prompt") or "").strip()
                if value:
                    lines.append(f"Scene {index}: {value}")
        return "\n\n".join(lines) or "Use the saved scene prompts, lyrics, and timeline order as the canonical project story."

    if filename == "subjectsandscenes.txt":
        lines = []
        for kind, items_key in (("Subject", "subjects"), ("Location", "locations")):
            items = refs.get(items_key) if isinstance(refs.get(items_key), list) else []
            for index, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or f"{kind} {index}").strip()
                description = str(item.get("description") or "").strip()
                lines.append(f"{kind}: {name}" + (f"\n{description}" if description else ""))
        for map_name in ("subject_scene_map", "performer_scene_map", "scene_map", "scene_trigger_map"):
            mapping = refs.get(map_name)
            if isinstance(mapping, dict) and mapping:
                lines.append(f"{map_name}:\n" + json.dumps(mapping, indent=2, ensure_ascii=False))
        return "\n\n".join(lines) or "No subjects or locations are currently mapped; use the saved scene records as canonical scene context."

    if filename == "themestyle.txt":
        lines = []
        for label, source, key in (
            ("Image world style", story_layer, "image_world_style"),
            ("Custom style direction", story_layer, "image_custom_style_direction"),
            ("Image aesthetic", storyboard, "image_aesthetic"),
            ("Video style", storyboard, "video_style"),
            ("Custom video style", storyboard, "video_style_custom"),
            ("Performance style", storyboard, "performance_style"),
            ("Global consistency", storyboard, "global_consistency_phrase"),
        ):
            value = str(source.get(key) or "").strip()
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines) or "Preserve the visual style, wardrobe, lighting, materials, and continuity established by the saved scene prompts and reference images."
    return ""


def _save_project_context_files(project_folder, session):
    context_files = session.get("project_context_files") if isinstance(session, dict) else None
    if not isinstance(context_files, dict):
        return []
    context = _context_folder(project_folder)
    saved = []
    for filename in ("storyconcept.txt", "subjectsandscenes.txt", "themestyle.txt"):
        value = str(context_files.get(filename, "") or "").strip()
        path = os.path.join(context, filename)
        # A caller that did not manage to read a context file must never erase
        # an existing non-empty project brief during autosave.
        if not value and os.path.isfile(path):
            with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
                value = handle.read().strip()
        if not value:
            value = _fallback_project_context_text(filename, session)
        atomic_write_text(path, value.rstrip() + "\n")
        saved.append(path)
    return saved


def _validate_saved_project(project_folder, session, context_paths):
    required = [_session_path(project_folder), _srt_path(project_folder)]
    required.extend(context_paths or [])
    refs = session.get("flux_reference_builder") if isinstance(session, dict) else None
    if isinstance(refs, dict):
        required.append(os.path.join(project_folder, "subject_location", "reference_descriptions.json"))
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise IOError("Project save validation failed; missing files: " + ", ".join(missing))
    empty = [os.path.basename(path) for path in (context_paths or []) if os.path.getsize(path) == 0]
    if empty:
        raise IOError("Project save validation failed; empty context files: " + ", ".join(empty))
    if isinstance(refs, dict):
        with open(required[-1], "r", encoding="utf-8-sig") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("subjects"), list) or not isinstance(manifest.get("locations"), list):
            raise IOError("Project save validation failed; reference-builder manifest is invalid.")
        expected = json.loads(json.dumps(refs, ensure_ascii=False))
        expected.setdefault("subjects", [])
        expected.setdefault("locations", [])
        if manifest != expected:
            raise IOError("Project save validation failed; reference-builder manifest does not match the saved session state.")


def _render_log_duration_text(milliseconds):
    try:
        total_seconds = max(0, int(round(float(milliseconds or 0) / 1000.0)))
    except (TypeError, ValueError):
        total_seconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _render_log_text(log):
    log = log if isinstance(log, dict) else {}
    summary = log.get("summary") if isinstance(log.get("summary"), dict) else {}
    scenes = log.get("scenes") if isinstance(log.get("scenes"), list) else []
    lines = [
        "VRGDG Video Builder Render Log",
        "=" * 32,
        f"Session: {log.get('id', '')}",
        f"Status: {str(log.get('status') or 'unknown').upper()}",
        f"Project: {log.get('project_folder', '')}",
        f"Mode: {log.get('mode_label') or log.get('scene_scope') or 'Render All'}",
        f"Started: {log.get('started_at', '')}",
        f"Finished: {log.get('ended_at', '')}",
        "",
        "Summary",
        "-" * 32,
        f"Total wall time: {_render_log_duration_text(summary.get('total_ms', log.get('total_ms', 0)))}",
        f"Active scene rendering: {_render_log_duration_text(summary.get('render_ms', 0))}",
        f"Between-render time: {_render_log_duration_text(summary.get('between_render_ms', 0))}",
        f"Setup time: {_render_log_duration_text(summary.get('setup_ms', 0))}",
        f"Final stitching: {_render_log_duration_text(summary.get('stitch_ms', 0))}",
        f"Other overhead: {_render_log_duration_text(summary.get('overhead_ms', 0))}",
        f"Scenes completed: {int(summary.get('completed_scenes', 0) or 0)}/{int(summary.get('target_scenes', len(scenes)) or 0)}",
        f"Existing scenes skipped: {int(summary.get('skipped_existing_scenes', 0) or 0)}",
        f"Average render per completed scene: {_render_log_duration_text(summary.get('average_render_ms', 0))}",
    ]
    if log.get("final_video_path"):
        lines.append(f"Final video: {log.get('final_video_path')}")
    if log.get("error"):
        lines.extend(["", f"Error: {log.get('error')}"])
    lines.extend(["", "Scene Details", "-" * 32])
    if not scenes:
        lines.append("No scene render timing has been recorded yet.")
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        label = scene.get("label") or f"Scene {scene.get('scene_number', '?')}"
        lines.extend([
            f"{label} [{str(scene.get('status') or 'pending').upper()}]",
            f"  Total scene step: {_render_log_duration_text(scene.get('total_ms', 0))}",
            f"  Preparation: {_render_log_duration_text(scene.get('preparation_ms', 0))}",
            f"  Video render: {_render_log_duration_text(scene.get('render_ms', 0))}",
            f"  Post-processing/cleanup: {_render_log_duration_text(scene.get('post_ms', 0))}",
            f"  Time since previous render: {_render_log_duration_text(scene.get('gap_before_render_ms', 0))}",
        ])
        if scene.get("video_path"):
            lines.append(f"  Video: {scene.get('video_path')}")
        if scene.get("error"):
            lines.append(f"  Error: {scene.get('error')}")
    return "\n".join(lines).rstrip() + "\n"


def _save_builder_render_log(payload):
    project_folder_raw = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_folder_raw:
        raise ValueError("Project folder is required before saving a render log.")
    project_folder = os.path.abspath(project_folder_raw)
    os.makedirs(project_folder, exist_ok=True)
    log = payload.get("log") if isinstance(payload.get("log"), dict) else {}
    if not log:
        raise ValueError("Render log data is empty.")
    log_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(log.get("id") or "").strip()).strip("._")
    if not log_id:
        log_id = f"render_{time.strftime('%Y%m%d_%H%M%S')}"
    log = {**log, "id": log_id, "project_folder": project_folder}
    logs_folder = _render_logs_folder(project_folder)
    os.makedirs(logs_folder, exist_ok=True)
    json_path = os.path.join(logs_folder, f"{log_id}.json")
    text_path = os.path.join(logs_folder, f"{log_id}.txt")
    log["report_json_path"] = json_path
    log["report_text_path"] = text_path
    json_temp = json_path + ".tmp"
    text_temp = text_path + ".tmp"
    with open(json_temp, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(text_temp, "w", encoding="utf-8") as handle:
        handle.write(_render_log_text(log))
    os.replace(json_temp, json_path)
    os.replace(text_temp, text_path)

    session_path = _session_path(project_folder)
    if os.path.isfile(session_path):
        try:
            with open(session_path, "r", encoding="utf-8-sig") as handle:
                session = json.load(handle)
            if not isinstance(session, dict):
                session = {}
        except Exception:
            session = {}
        logs = session.get("render_logs") if isinstance(session.get("render_logs"), list) else []
        logs = [item for item in logs if isinstance(item, dict) and item.get("id") != log_id]
        logs.append(log)
        session["render_logs"] = logs[-20:]
        session["active_render_log_id"] = log_id if log.get("status") == "running" else ""
        session["updated"] = time.time()
        session_temp = session_path + ".render-log.tmp"
        with open(session_temp, "w", encoding="utf-8") as handle:
            json.dump(session, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(session_temp, session_path)
    return {
        "log": log,
        "report_json_path": json_path,
        "report_text_path": text_path,
    }


def _save_canonical_full_lyrics(project_folder, lyrics):
    """Persist the user's complete source lyrics without timeline reconstruction."""
    text = str(lyrics or "").strip()
    if not text:
        return ""
    context_folder = _context_folder(project_folder)
    os.makedirs(context_folder, exist_ok=True)
    path = os.path.join(context_folder, "full_lyrics.txt")
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    os.replace(temporary_path, path)
    return path


# Per-scene files are numbered by base-scene position; insert-track slots start at 10001 and never match.
_SCENE_ASSET_FOLDERS = (
    "rendered_scene_videos",
    "rendered_scene_videos_backup",
    "scene_video_thumbnails",
    "zimage_approved",
    "scene_image_previews",
    "scene_audio",
    "scene_audio_trimmed",
    "scene_srt",
    "project_context",
    "minimax_h3_scene_audio",
    "image_to_video_clips",
    "text_to_video_clips",
    "reference_to_video_clips",
    "ingredients_to_video_clips",
    "id_lora_i2v_clips",
    "first_last_frame_clips",
)


_SCENE_ASSET_NAME = re.compile(r"^((?:video|image|audio|scene_audio|scene)_)(\d{4})(?!\d)")


def _scene_asset_number(name):
    match = _SCENE_ASSET_NAME.match(name)
    return int(match.group(2)) if match else 0


def _renumbered_scene_asset_name(name, number):
    return _SCENE_ASSET_NAME.sub(lambda match: f"{match.group(1)}{number:04d}", name, count=1)


def _shift_scene_assets(project_folder, first_number, delta):
    """Renumber every per-scene file numbered first_number or later by delta (+1 or -1).

    Returns (old_path, new_path) pairs, folders included, so callers can update the paths they store.
    """
    renamed = []
    for folder_name in _SCENE_ASSET_FOLDERS:
        folder = os.path.join(project_folder, folder_name)
        if not os.path.isdir(folder):
            continue
        # Shifting up renames from the highest number down so no file lands on one that has not moved yet.
        entries = sorted(((_scene_asset_number(name), name) for name in os.listdir(folder)), reverse=delta > 0)
        for number, name in entries:
            if number < first_number:
                continue
            source = os.path.join(folder, name)
            target = os.path.join(folder, _renumbered_scene_asset_name(name, number + delta))
            os.rename(source, target)
            renamed.append((source, target))
            if os.path.isdir(target):
                for inner in os.listdir(target):
                    if _scene_asset_number(inner) == number:
                        inner_target = os.path.join(target, _renumbered_scene_asset_name(inner, number + delta))
                        os.rename(os.path.join(target, inner), inner_target)
                        renamed.append((os.path.join(source, inner), inner_target))
    return renamed


def _renumber_scene_assets_after_removal(project_folder, removed_scene_number):
    """Keep per-scene file numbers equal to scene positions after a base scene is removed.

    The removed scene's own files move to removed_scene_assets/, then every later scene's files shift down one number.
    """
    removed = int(removed_scene_number)
    archive = os.path.join(project_folder, "removed_scene_assets", f"scene_{removed:04d}_{time.strftime('%Y%m%d_%H%M%S')}")
    for folder_name in _SCENE_ASSET_FOLDERS:
        folder = os.path.join(project_folder, folder_name)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if _scene_asset_number(name) == removed:
                os.makedirs(os.path.join(archive, folder_name), exist_ok=True)
                shutil.move(os.path.join(folder, name), os.path.join(archive, folder_name, name))
    renamed = _shift_scene_assets(project_folder, removed + 1, -1)
    SceneLatentManager.delete_latent(project_folder, removed)
    SceneLatentManager.reindex_latents(project_folder, removed)
    return renamed


def _renumber_scene_assets_after_insert(project_folder, inserted_scene_number):
    """Keep per-scene file numbers equal to scene positions after a base scene is inserted or split off.

    The scene at inserted_scene_number and every later scene shift up one number, so the new scene starts with no files.
    """
    inserted = int(inserted_scene_number)
    renamed = _shift_scene_assets(project_folder, inserted, 1)
    SceneLatentManager.make_room_for_scene(project_folder, inserted)
    return renamed


def _scene_numbers_from_folder(folder, pattern):
    numbers = set()
    if not os.path.isdir(folder):
        return numbers
    regex = re.compile(pattern, re.IGNORECASE)
    for name in os.listdir(folder):
        match = regex.match(name)
        if match and os.path.isfile(os.path.join(folder, name)):
            numbers.add(int(match.group(1)))
    return numbers


def _project_scene_numbers(project_folder):
    numbers = set()
    numbers.update(_scene_numbers_from_folder(_images_folder(project_folder), r"^image_(\d+)\.(?:png|jpe?g|webp)$"))
    numbers.update(_scene_numbers_from_folder(os.path.join(project_folder, "rendered_scene_videos"), r"^video_(\d+)-audio\.mp4$"))
    preview_root = os.path.join(project_folder, "scene_image_previews")
    if os.path.isdir(preview_root):
        for name in os.listdir(preview_root):
            match = re.match(r"^scene_(\d+)$", name, re.IGNORECASE)
            if match and os.path.isdir(os.path.join(preview_root, name)):
                numbers.add(int(match.group(1)))
    return numbers


def _scene_preview_paths(project_folder, scene_number):
    folder = _scene_preview_folder(project_folder, scene_number)
    if not os.path.isdir(folder):
        return []
    paths = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            paths.append(os.path.abspath(path))
    paths.sort(key=lambda item: os.path.getmtime(item))
    return paths


def _backup_session_file(project_folder):
    path = _session_path(project_folder)
    if not os.path.isfile(path):
        return ""
    backup_folder = os.path.join(project_folder, "session_backups")
    os.makedirs(backup_folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = os.path.join(backup_folder, f"vrgdg_builder_session_{stamp}.json")
    index = 2
    while os.path.exists(target):
        target = os.path.join(backup_folder, f"vrgdg_builder_session_{stamp}_{index:02d}.json")
        index += 1
    shutil.copy2(path, target)
    return target


def _rehydrate_builder_session(project_folder, session):
    old_project_folder = str(session.get("project_folder", "") or "")
    project_folder = os.path.abspath(project_folder)

    # Portable imports can contain paths in deeply nested reference-builder,
    # FLF, history, LUT, grain, and adjustment structures. Rebase every path
    # owned by the exported project before the field-specific recovery below.
    def rebase_nested_project_paths(value):
        if isinstance(value, dict):
            return {key: rebase_nested_project_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase_nested_project_paths(item) for item in value]
        if not isinstance(value, str) or not old_project_folder or not os.path.isabs(value):
            return value
        rebased = _project_rebased_path(project_folder, old_project_folder, value)
        return rebased if rebased and os.path.exists(rebased) else value

    session = rebase_nested_project_paths(session)
    session["project_folder"] = project_folder
    manifest_path = os.path.join(project_folder, "subject_location", "reference_descriptions.json")
    if not isinstance(session.get("flux_reference_builder"), dict) and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8-sig") as handle:
                manifest = json.load(handle)
            if isinstance(manifest, dict) and isinstance(manifest.get("subjects"), list) and isinstance(manifest.get("locations"), list):
                session["flux_reference_builder"] = manifest
        except Exception as exc:
            print(f"[VRGDG Music Builder] Reference-builder manifest restore skipped: {exc}")
    context = _context_folder(project_folder)
    session.setdefault("theme_style_path", os.path.join(context, "themestyle.txt"))
    session.setdefault("story_idea_path", os.path.join(context, "storyconcept.txt"))
    session.setdefault("subject_scene_path", os.path.join(context, "subjectsandscenes.txt"))
    session["audio_path"] = _resolve_project_asset_path(project_folder, old_project_folder, session.get("audio_path", ""))
    for key in ("prompt_json_path", "theme_style_path", "story_idea_path", "subject_scene_path"):
        session[key] = _resolve_project_asset_path(project_folder, old_project_folder, session.get(key, ""))
    if isinstance(session.get("flux_global_image_ingredients"), list):
        for ingredient in session["flux_global_image_ingredients"]:
            if not isinstance(ingredient, dict):
                continue
            ingredient["path"] = _resolve_project_asset_path(
                project_folder,
                old_project_folder,
                ingredient.get("path", ""),
            )

    segments = session.get("segments", [])
    if not isinstance(segments, list):
        session["segments"] = []
        segments = session["segments"]
    overlay_segments = session.get("overlay_segments", [])
    if not isinstance(overlay_segments, list):
        session["overlay_segments"] = []
        overlay_segments = session["overlay_segments"]
    overlay_segments = _assign_overlay_scene_numbers(overlay_segments)

    # Only rebuild timeline scenes from loose media files when the session has no
    # saved scene list. Otherwise deleted scenes can come back from old files.
    existing_count = len(segments)
    if existing_count == 0:
        asset_numbers = _project_scene_numbers(project_folder)
        base_asset_numbers = [number for number in asset_numbers if number < 10000]
        target_count = max(base_asset_numbers) if base_asset_numbers else 0
        for index in range(1, target_count + 1):
            start = float((index - 1) * 4)
            segments.append({
                "id": f"recovered_scene_{index}",
                "label": f"Scene {index}",
                "start": start,
                "end": start + 4,
                "source": "recovered",
            })

    cleaned_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        is_recovered = str(segment.get("source", "") or "").lower() == "recovered" or str(segment.get("id", "") or "").startswith("recovered_scene_")
        if is_recovered:
            start = float(segment.get("start", 0) or 0)
            end = float(segment.get("end", start) or start)
            overlaps_real = False
            for other in segments:
                if other is segment or not isinstance(other, dict):
                    continue
                other_recovered = str(other.get("source", "") or "").lower() == "recovered" or str(other.get("id", "") or "").startswith("recovered_scene_")
                if other_recovered:
                    continue
                other_start = float(other.get("start", 0) or 0)
                other_end = float(other.get("end", other_start) or other_start)
                if min(end, other_end) - max(start, other_start) > 0.05:
                    overlaps_real = True
                    break
            if overlaps_real:
                continue
        cleaned_segments.append(segment)
    session["segments"] = cleaned_segments
    segments = cleaned_segments

    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        if not str(segment.get("label", "") or "").strip() or str(segment.get("label", "")).lower() == "new scene":
            segment["label"] = f"Scene {index}"
        for key in (
            "approved_image_path",
            "custom_image_path",
            "ref_image_path",
            "flux_subject_image_path",
            "flux_location_image_path",
            "video_path",
            "custom_audio_path",
        ):
            segment[key] = _resolve_project_asset_path(project_folder, old_project_folder, segment.get(key, ""), index)
        if isinstance(segment.get("image_history"), list):
            segment["image_history"] = [
                _resolve_project_asset_path(project_folder, old_project_folder, item, index)
                for item in segment["image_history"]
            ]
            segment["image_history"] = [item for item in segment["image_history"] if item]
        else:
            segment["image_history"] = []
        if isinstance(segment.get("flux_image_ingredients"), list):
            for ingredient in segment["flux_image_ingredients"]:
                if not isinstance(ingredient, dict):
                    continue
                ingredient["path"] = _resolve_project_asset_path(
                    project_folder,
                    old_project_folder,
                    ingredient.get("path", ""),
                    index,
                )
        approved = _resolve_project_asset_path(project_folder, old_project_folder, segment.get("approved_image_path", ""), index)
        image_assignment_cleared = bool(segment.get("image_assignment_cleared", False))
        if not approved and not image_assignment_cleared:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = _scene_image_path(project_folder, index, ext)
                if os.path.isfile(candidate):
                    approved = os.path.abspath(candidate)
                    break
        if approved and os.path.isfile(approved):
            segment["approved_image_path"] = approved
            segment["image_history"] = [
                item for item in segment["image_history"]
                if item != approved and not _is_internal_approved_image_path(item)
            ]
        if not image_assignment_cleared:
            for preview_path in _scene_preview_paths(project_folder, index):
                if preview_path not in segment["image_history"]:
                    segment["image_history"].append(preview_path)
        if segment["image_history"] and not isinstance(segment.get("image_history_index"), int):
            segment["image_history_index"] = len(segment["image_history"]) - 1
        video_path = os.path.join(project_folder, "rendered_scene_videos", f"video_{index:04d}-audio.mp4")
        if os.path.isfile(video_path):
            segment["video_path"] = os.path.abspath(video_path)
            segment["video_folder"] = os.path.dirname(os.path.abspath(video_path))
            segment["video_status"] = "done"
    for index, segment in enumerate(overlay_segments, start=1):
        if not isinstance(segment, dict):
            continue
        scene_number = _overlay_scene_number(segment, index)
        if not str(segment.get("label", "") or "").strip() or str(segment.get("label", "")).lower() == "new scene":
            segment["label"] = f"Insert {index}"
        segment["track"] = "overlay"
        for key in (
            "approved_image_path",
            "custom_image_path",
            "ref_image_path",
            "flux_subject_image_path",
            "flux_location_image_path",
            "video_path",
            "custom_audio_path",
        ):
            segment[key] = _resolve_project_asset_path(project_folder, old_project_folder, segment.get(key, ""), scene_number)
        if isinstance(segment.get("image_history"), list):
            segment["image_history"] = [
                _resolve_project_asset_path(project_folder, old_project_folder, item, scene_number)
                for item in segment["image_history"]
            ]
            segment["image_history"] = [item for item in segment["image_history"] if item]
        else:
            segment["image_history"] = []
        for preview_path in _scene_preview_paths(project_folder, scene_number):
            if preview_path not in segment["image_history"]:
                segment["image_history"].append(preview_path)
        video_path = os.path.join(project_folder, "rendered_scene_videos", f"video_{scene_number:04d}-audio.mp4")
        if os.path.isfile(video_path):
            segment["video_path"] = os.path.abspath(video_path)
            segment["video_folder"] = os.path.dirname(os.path.abspath(video_path))
            segment["video_status"] = "done"
    return session


def _prompt_key_number(key):
    match = re.search(r"(\d+)", str(key or ""))
    return int(match.group(1)) if match else 999999


def _load_prompt_json(path):
    json_path = _resolve_existing_file(path, "Prompt JSON")
    with open(json_path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    prompts = []
    if isinstance(data, dict):
        for key in sorted(data.keys(), key=_prompt_key_number):
            prompts.append(str(data.get(key, "") or "").strip())
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                prompts.append(item.strip())
            elif isinstance(item, dict):
                for key in sorted(item.keys(), key=_prompt_key_number):
                    prompts.append(str(item.get(key, "") or "").strip())
    else:
        raise ValueError("Prompt JSON must be an object or list.")
    if not prompts:
        raise ValueError("Prompt JSON did not contain any prompt text.")
    return {"prompt_json_path": json_path, "prompts": prompts}


def _resolve_editable_text_file(path):
    raw_path = str(path or "").strip().strip('"')
    if not raw_path:
        raise ValueError("Text file path is empty.")
    file_path = os.path.normpath(os.path.abspath(raw_path))
    if os.path.splitext(file_path)[1].lower() not in {".txt", ".json"}:
        raise ValueError("Only .txt or .json files can be edited here.")
    return file_path


def _load_editable_text_file(payload):
    file_path = _resolve_editable_text_file(payload.get("path", ""))
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Text file was not found: {file_path}")
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as handle:
        return {"path": file_path, "content": handle.read()}


def _save_editable_text_file(payload):
    file_path = _resolve_editable_text_file(payload.get("path", ""))
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    content = str(payload.get("content", "") or "")
    with open(file_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return {"path": file_path}


_MODEL_DEFAULT_KEYS = (
    "text_gemma_runner",
    "qwen_model_file",
    "qwen_mmproj_file",
    "gemma_model_file",
    "llm_max_tokens",
    "gemma_context_limit",
    "gemma_output_token_limit",
    "gemma_gpu_layers",
    "lm_studio_base_url",
    "lm_studio_model",
    "lm_studio_api_key",
    "lm_studio_context_limit",
    "lm_studio_output_token_limit",
    "image_model_mode",
    "zimage_settings",
    "reference_krea2_settings",
    "flux_klein_settings",
    "ernie_image_settings",
    "krea2_2pass_settings",
    "z_enhance_settings",
    "video_model_mode",
    "i2v_video_settings",
)


def _scrub_model_defaults_project_sources(defaults):
    if not isinstance(defaults, dict):
        return {}
    cleaned = json.loads(json.dumps(defaults))
    for key in ("zimage_settings", "ernie_image_settings", "krea2_2pass_settings"):
        settings = cleaned.get(key)
        if not isinstance(settings, dict):
            continue
        settings["use_image_to_image"] = False
        settings["image_to_image_path"] = ""
        settings["image_to_image_data"] = ""
        settings["image_to_image_name"] = ""
    return cleaned


def _extract_model_defaults(session):
    if not isinstance(session, dict):
        return {}
    defaults = {}
    for key in _MODEL_DEFAULT_KEYS:
        value = session.get(key)
        if value is not None:
            defaults[key] = value
    return _scrub_model_defaults_project_sources(defaults)


def _save_model_defaults(session):
    defaults = _extract_model_defaults(session)
    if not defaults:
        return ""
    target = _model_defaults_path()
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "defaults": defaults,
    }
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target


def _load_model_defaults():
    target = _model_defaults_path()
    if not os.path.isfile(target):
        return {"path": target, "defaults": {}, "saved_at": ""}
    with open(target, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        payload = {}
    defaults = payload.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
    defaults = _scrub_model_defaults_project_sources(defaults)
    return {
        "path": target,
        "defaults": defaults,
        "saved_at": str(payload.get("saved_at", "") or ""),
    }


def _write_scene_notes_json(project_folder, segments):
    notes = {}
    for index, segment in enumerate(segments if isinstance(segments, list) else [], start=1):
        if isinstance(segment, dict):
            notes[f"SceneNote{index}"] = str(segment.get("timeline_note", "") or "")
    path = _scene_notes_path(project_folder)
    atomic_write_json(path, notes)
    return path


def _load_scene_notes_json(project_folder):
    folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not folder:
        return {}
    path = _scene_notes_path(folder)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    notes = {}
    for raw_key, raw_value in data.items():
        match = re.search(r"(\d+)", str(raw_key or ""))
        if match:
            notes[int(match.group(1))] = str(raw_value or "").strip()
    return notes


def _save_reference_descriptions(project_folder, session):
    """Expose Reference Builder descriptions as portable project assets."""
    refs = session.get("flux_reference_builder") if isinstance(session, dict) else None
    if not isinstance(refs, dict):
        return ""
    root = os.path.join(os.path.abspath(project_folder), "subject_location")
    subject_dir = os.path.join(root, "subject")
    location_dir = os.path.join(root, "location")
    os.makedirs(subject_dir, exist_ok=True)
    os.makedirs(location_dir, exist_ok=True)
    # Keep the complete normalized catalog, including image paths and every
    # scene mapping.  The text files below remain convenient human-editable
    # exports, while this manifest is the lossless restore source.
    manifest = json.loads(json.dumps(refs, ensure_ascii=False))
    manifest.setdefault("subjects", [])
    manifest.setdefault("locations", [])
    for manifest_key, folder in (("subjects", subject_dir), ("locations", location_dir)):
        items = refs.get(manifest_key) if isinstance(refs.get(manifest_key), list) else []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or f"{manifest_key[:-1].title()} {index}").strip()
            description = str(item.get("description") or "").strip()
            atomic_write_text(os.path.join(folder, f"{_safe_project_name(name)}.txt"), description + ("\n" if description else ""))
    manifest_path = os.path.join(root, "reference_descriptions.json")
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _write_minimax_project_index(project_folder, session):
    """Write a human-readable map of MiniMax project files and session fields."""
    if str(session.get("video_engine", "") or "").strip().lower() != "minimax_h3":
        return ""
    path = os.path.join(os.path.abspath(project_folder), "MINIMAX_PROJECT_FILES.md")
    content = """# MiniMax H3 project files

Generated by the Video Builder. The session JSON is the source of truth for MiniMax prompts and settings.

## Primary MiniMax data

- `vrgdg_builder_session.json` — project `video_engine`, `minimax_h3_settings`, two/three-pass flags, and per-scene `minimax_h3_prompt`, mode, audio mode, references, continuity, stage outputs, and render history.
- `subject_location/reference_descriptions.json` — portable character and location names, IDs, and descriptions.
- `subject_location/subject/<name>.txt` — one character/subject description per file.
- `subject_location/location/<name>.txt` — one location description per file.

## References and audio

- `project_context/flux_references/subjects/` — character reference images.
- `project_context/flux_references/locations/` — location reference images.
- `project_context/flux_references/ingredients_sheets/` — ingredient/reference sheets when used.
- `project_audio/` — project/global audio copied into the project.
- `scene_audio/` — scene audio assets, including MiniMax scene audio when applicable.
- `scene_audio_trimmed/` — trimmed scene-audio derivatives when created.

## Rendered MiniMax outputs

- `rendered_scene_videos/video_####-audio.mp4` — final per-scene clips.
- `rendered_scene_videos_backup/scene_####/` — replaced/backed-up scene clips.
- `scene_video_thumbnails/` — rendered clip thumbnails.
- `render_logs/` — MiniMax render progress and diagnostics.

## Not MiniMax-specific

- `prompts/t2i_prompts.txt` — image-generation prompts.
- `prompts/i2v_prompts.txt` — LTX image-to-video prompts; MiniMax prompts are in the session JSON.
- `builder_segments.srt`, `SceneNotes.json`, and other shared context files.
"""
    atomic_write_text(path, content)
    return path


def _save_builder_session_unlocked(payload):
    audio_raw = str(payload.get("audio_path", "") or "").strip().strip('"')
    audio_path = _resolve_existing_file(audio_raw, "Audio file") if audio_raw else ""
    project_folder = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_folder:
        if audio_path:
            project_folder = _default_project_folder(audio_path, payload.get("project_name", ""))
        else:
            project_name = payload.get("project_name", "") or f"VRGDG_Project_{time.strftime('%Y%m%d_%H%M%S')}"
            project_folder = os.path.join(folder_paths.get_output_directory(), _safe_project_name(project_name))
    project_folder = os.path.abspath(project_folder)
    os.makedirs(project_folder, exist_ok=True)
    os.makedirs(_images_folder(project_folder), exist_ok=True)
    os.makedirs(_prompts_folder(project_folder), exist_ok=True)
    os.makedirs(_context_folder(project_folder), exist_ok=True)

    session = dict(payload.get("session")) if isinstance(payload.get("session"), dict) else {}
    segments = session.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    session_path = _session_path(project_folder)
    if isinstance(payload.get("project_context_files"), dict):
        session["project_context_files"] = dict(payload["project_context_files"])
    incoming_revision = int(session.get("builder_save_revision") or 0)
    if incoming_revision > 0 and os.path.isfile(session_path):
        try:
            with open(session_path, "r", encoding="utf-8-sig") as handle:
                existing_session = json.load(handle)
            existing_revision = int(existing_session.get("builder_save_revision") or 0) if isinstance(existing_session, dict) else 0
            if existing_revision > incoming_revision:
                print(
                    "[VRGDG Music Builder] Ignored stale session snapshot: "
                    f"incoming revision {incoming_revision} < saved revision {existing_revision}."
                )
                session = existing_session
                segments = session.get("segments", []) if isinstance(session.get("segments"), list) else []
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[VRGDG Music Builder] Save revision check skipped: {exc}")
    # Autosave requests can race a reference-builder panel update. If the
    # incoming snapshot has a null catalog, retain the last known catalog
    # instead of turning a valid project into a prompt-only project.
    if session.get("flux_reference_builder") is None and os.path.isfile(session_path):
        try:
            with open(session_path, "r", encoding="utf-8-sig") as handle:
                previous = json.load(handle)
            previous_refs = previous.get("flux_reference_builder") if isinstance(previous, dict) else None
            if isinstance(previous_refs, dict):
                session["flux_reference_builder"] = previous_refs
        except Exception as exc:
            print(f"[VRGDG Music Builder] Previous reference-builder state could not be recovered: {exc}")
    overlay_segments = session.get("overlay_segments", [])
    if not isinstance(overlay_segments, list):
        overlay_segments = []
        session["overlay_segments"] = overlay_segments
    overlay_segments = _assign_overlay_scene_numbers(overlay_segments)
    audio_path, session = _snapshot_project_assets(project_folder, session, audio_path)
    session = {
        **session,
        "audio_path": audio_path,
        "project_folder": project_folder,
        "updated": time.time(),
        "segments": segments,
    }

    # The complete text pasted into Line Mapper/Auto Build is the project's
    # canonical lyric source. Never rebuild this file from timestamped scene
    # notes because those notes legitimately contain detected instrumental gaps
    # and may split or repeat song sections.
    lyric_mapper = session.get("lyric_mapper") if isinstance(session.get("lyric_mapper"), dict) else {}
    full_lyrics_path = _save_canonical_full_lyrics(project_folder, lyric_mapper.get("source_text"))

    srt_text = _segments_to_srt(segments)
    _backup_session_file(project_folder)
    atomic_write_text(_srt_path(project_folder), srt_text)
    context_paths = _save_project_context_files(project_folder, session)
    reference_descriptions_path = _save_reference_descriptions(project_folder, session)
    minimax_project_index_path = _write_minimax_project_index(project_folder, session)
    model_defaults_path = _save_model_defaults(session)
    scene_notes_path = _write_scene_notes_json(project_folder, segments)

    t2i_lines = []
    i2v_lines = []
    for segment in sorted(list(segments) + list(overlay_segments), key=lambda item: float(item.get("start", 0) or 0)):
        if str(segment.get("t2i_prompt", "")).strip():
            t2i_lines.append(str(segment.get("t2i_prompt", "")).strip())
        if str(segment.get("i2v_prompt", "")).strip():
            i2v_lines.append(str(segment.get("i2v_prompt", "")).strip())
    atomic_write_text(os.path.join(_prompts_folder(project_folder), "t2i_prompts.txt"), "\n\n".join(t2i_lines).strip() + ("\n" if t2i_lines else ""))
    atomic_write_text(os.path.join(_prompts_folder(project_folder), "i2v_prompts.txt"), "\n\n".join(i2v_lines).strip() + ("\n" if i2v_lines else ""))
    # The session is the transaction commit marker. All derived project files
    # are atomically replaced first, then the canonical session is replaced
    # last so a failed save cannot advertise an uncommitted snapshot.
    atomic_write_json(_session_path(project_folder), session)
    _validate_saved_project(project_folder, session, context_paths)

    return {
        "project_folder": project_folder,
        "session_path": _session_path(project_folder),
        "srt_path": _srt_path(project_folder),
        "images_folder": _images_folder(project_folder),
        "prompts_folder": _prompts_folder(project_folder),
        "context_folder": _context_folder(project_folder),
        "context_paths": context_paths,
        "full_lyrics_path": full_lyrics_path,
        "model_defaults_path": model_defaults_path,
        "scene_notes_path": scene_notes_path,
        "reference_descriptions_path": reference_descriptions_path,
        "minimax_project_index_path": minimax_project_index_path,
        "session": session,
    }


def _save_builder_session(payload):
    # Autosave, Quick Save, and Reference Builder Save share this transaction.
    # Serialize them so their individually atomic file replacements cannot
    # interleave into a mixed-generation project snapshot.
    with _BUILDER_SAVE_LOCK:
        return _save_builder_session_unlocked(payload)


def _prepare_builder_project_export(project_folder):
    project_folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    session_path = _session_path(project_folder)
    if not os.path.isdir(project_folder) or not os.path.isfile(session_path):
        raise FileNotFoundError("The Builder project or its session file was not found.")
    with open(session_path, "r", encoding="utf-8") as handle:
        session = json.load(handle)
    if not isinstance(session, dict):
        raise ValueError("The Builder project session is invalid.")

    # Pull any still-external scene assets into the project before packaging it.
    old_project_folder = str(session.get("project_folder", "") or project_folder)
    session = _copy_session_assets_to_project(project_folder, session)
    portable_folder = os.path.join(project_folder, "portable_assets")
    portable_extensions = {
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
        ".mp4", ".mov", ".mkv", ".webm", ".avi",
        ".wav", ".mp3", ".m4a", ".flac", ".ogg",
        ".srt", ".txt", ".json", ".csv",
    }
    copied_external_paths = {}

    def localize_external_assets(value, key_path="asset"):
        if isinstance(value, dict):
            return {key: localize_external_assets(item, f"{key_path}_{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [localize_external_assets(item, f"{key_path}_{index + 1}") for index, item in enumerate(value)]
        if not isinstance(value, str):
            return value
        source = value.strip().strip('"')
        if not os.path.isabs(source) or not os.path.isfile(source):
            return value
        try:
            if os.path.commonpath([project_folder, os.path.abspath(source)]) == project_folder:
                return os.path.abspath(source)
        except ValueError:
            pass
        extension = os.path.splitext(source)[1].lower()
        if extension not in portable_extensions:
            return value
        source_key = os.path.normcase(os.path.abspath(source))
        if source_key in copied_external_paths:
            return copied_external_paths[source_key]
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key_path).strip("._")[-80:] or "asset"
        safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(source)).strip("._") or f"file{extension}"
        destination = os.path.join(portable_folder, f"{len(copied_external_paths) + 1:04d}_{safe_key}_{safe_base}")
        copied = _copy_file_if_exists(source, destination)
        if copied:
            copied_external_paths[source_key] = copied
            return copied
        return value

    session = localize_external_assets(session, "session")
    session = _rebase_project_owned_paths(project_folder, old_project_folder, session)
    session["project_folder"] = project_folder
    session["updated"] = time.time()
    with open(session_path, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    project_name = _safe_project_name(os.path.basename(project_folder))
    temp_handle = tempfile.NamedTemporaryFile(prefix="vrgdg_builder_export_", suffix=".zip", delete=False)
    zip_path = temp_handle.name
    temp_handle.close()
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.writestr("vrgdg_project_package.json", json.dumps({
                "format": "vrgdg_builder_project",
                "version": 1,
                "project_name": project_name,
                "created": time.time(),
            }, indent=2))
            for root, folders, files in os.walk(project_folder):
                folders[:] = [name for name in folders if name not in {"__pycache__"}]
                for filename in files:
                    source = os.path.join(root, filename)
                    relative = os.path.relpath(source, project_folder).replace(os.sep, "/")
                    already_compressed = os.path.splitext(filename)[1].lower() in {
                        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".mp3", ".m4a", ".flac", ".ogg",
                        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".zip",
                    }
                    archive.write(source, relative, compress_type=zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED)
        return zip_path, f"{project_name}.vrgdg.zip"
    except Exception:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        raise


def _safe_builder_zip_members(archive):
    members = archive.infolist()
    if not members:
        raise ValueError("The selected ZIP file is empty.")
    total_size = 0
    for member in members:
        normalized = member.filename.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part not in {"", "."}]
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or any(part == ".." for part in parts):
            raise ValueError(f"Unsafe path in project ZIP: {member.filename}")
        unix_mode = (member.external_attr >> 16) & 0o170000
        if unix_mode == 0o120000:
            raise ValueError(f"Symbolic links are not allowed in project ZIPs: {member.filename}")
        total_size += max(0, int(member.file_size or 0))
        if member.file_size > 1024 * 1024 * 1024 and member.compress_size and member.file_size > member.compress_size * 1000:
            raise ValueError(f"Suspicious compression ratio in project ZIP: {member.filename}")
    if total_size > 500 * 1024 * 1024 * 1024:
        raise ValueError("The uncompressed project is larger than the 500 GB safety limit.")
    names = {member.filename.replace("\\", "/").strip("/") for member in members}
    if "vrgdg_builder_session.json" not in names:
        raise ValueError("This ZIP is not a portable Video Builder project (vrgdg_builder_session.json is missing).")
    return members


def _import_builder_project_zip(zip_path, requested_name=""):
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = _safe_builder_zip_members(archive)
        manifest = {}
        try:
            manifest = json.loads(archive.read("vrgdg_project_package.json").decode("utf-8"))
        except (KeyError, ValueError, UnicodeDecodeError):
            manifest = {}
        default_name = manifest.get("project_name") or os.path.basename(zip_path).replace(".vrgdg.zip", "").replace(".zip", "")
        project_name = _safe_project_name(requested_name or default_name)
        target = _unique_folder_path(os.path.join(folder_paths.get_output_directory(), project_name))
        os.makedirs(target, exist_ok=False)
        try:
            target_real = os.path.realpath(target)
            for member in members:
                name = member.filename.replace("\\", "/").strip("/")
                if not name or name == "vrgdg_project_package.json":
                    continue
                destination = os.path.realpath(os.path.join(target, *name.split("/")))
                if os.path.commonpath([target_real, destination]) != target_real:
                    raise ValueError(f"Unsafe path in project ZIP: {member.filename}")
                if member.is_dir():
                    os.makedirs(destination, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with archive.open(member, "r") as source, open(destination, "wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            session_path = _session_path(target)
            imported_session = {}
            if os.path.isfile(session_path):
                with open(session_path, "r", encoding="utf-8-sig") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    imported_session = loaded
            old_folder = str(imported_session.get("project_folder", "") or "").strip().strip('"')
            imported_session = _rebase_project_owned_paths(target, old_folder, imported_session)
            imported_session["project_folder"] = target
            imported_session["updated"] = time.time()
            with open(session_path, "w", encoding="utf-8") as handle:
                json.dump(imported_session, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            result = _load_builder_session(target)
            result["imported_project_name"] = project_name
            return result
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise


def _save_wizard_draft(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    os.makedirs(project_folder, exist_ok=True)
    folder = _wizard_folder(project_folder)
    os.makedirs(folder, exist_ok=True)
    draft = payload.get("draft") if isinstance(payload.get("draft"), dict) else {}
    lyrics = str(payload.get("lyrics", "") or draft.get("lyrics", "") or "").replace("\r\n", "\n").replace("\r", "\n")
    draft = {
        **draft,
        "lyrics": lyrics,
        "updated": time.time(),
    }
    with open(_wizard_draft_path(project_folder), "w", encoding="utf-8") as handle:
        json.dump(draft, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(_wizard_lyrics_path(project_folder), "w", encoding="utf-8") as handle:
        handle.write(lyrics)
        if lyrics and not lyrics.endswith("\n"):
            handle.write("\n")
    raw_outputs = payload.get("raw_outputs") if isinstance(payload.get("raw_outputs"), dict) else {}
    for name, value in raw_outputs.items():
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name or "").strip()).strip("._") or "raw_output"
        if not safe_name.endswith(".txt") and not safe_name.endswith(".json"):
            safe_name += ".txt"
        with open(os.path.join(folder, safe_name), "w", encoding="utf-8") as handle:
            if isinstance(value, (dict, list)):
                json.dump(value, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            else:
                handle.write(str(value or ""))
                if value and not str(value).endswith("\n"):
                    handle.write("\n")
    return {
        "wizard_folder": folder,
        "wizard_draft_path": _wizard_draft_path(project_folder),
        "wizard_lyrics_path": _wizard_lyrics_path(project_folder),
        "draft": draft,
    }


def _load_wizard_draft(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    path = _wizard_draft_path(project_folder)
    draft = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            draft = loaded
    lyrics_path = _wizard_lyrics_path(project_folder)
    if os.path.isfile(lyrics_path) and not str(draft.get("lyrics", "")).strip():
        with open(lyrics_path, "r", encoding="utf-8") as handle:
            draft["lyrics"] = handle.read()
    return {
        "wizard_folder": _wizard_folder(project_folder),
        "wizard_draft_path": path,
        "wizard_lyrics_path": lyrics_path,
        "draft": draft,
        "exists": bool(draft),
    }


def _load_builder_session(project_folder):
    folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not folder:
        raise ValueError("Project folder is empty.")
    path = _session_path(folder)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Builder session was not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as handle:
        session = json.load(handle)
    if not isinstance(session, dict):
        raise ValueError("Builder session is not a JSON object.")
    session = _rehydrate_builder_session(folder, session)
    scene_note_fallbacks = _load_scene_notes_json(folder)
    segments = session.get("segments", [])
    if scene_note_fallbacks and isinstance(segments, list):
        for index, segment in enumerate(segments, start=1):
            if not isinstance(segment, dict):
                continue
            if not str(segment.get("timeline_note", "") or "").strip() and scene_note_fallbacks.get(index):
                segment["timeline_note"] = scene_note_fallbacks[index]
    return {
        "project_folder": folder,
        "session_path": path,
        "srt_path": _srt_path(folder),
        "scene_notes_path": _scene_notes_path(folder),
        "session": session,
    }


def _list_builder_projects(project_root=""):
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    projects = []
    roots = [output_dir]
    custom_root = str(project_root or "").strip().strip('"')
    if custom_root and os.path.isabs(custom_root):
        custom_root = os.path.abspath(custom_root)
        if os.path.normcase(custom_root) != os.path.normcase(output_dir):
            roots.append(custom_root)
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            folder = os.path.abspath(os.path.join(root, name))
            folder_key = os.path.normcase(folder)
            if folder_key in seen or not os.path.isdir(folder):
                continue
            session_path = _session_path(folder)
            if not os.path.isfile(session_path):
                continue
            seen.add(folder_key)
            try:
                mtime = os.path.getmtime(session_path)
            except OSError:
                mtime = 0
            scene_count = 0
            try:
                with open(session_path, "r", encoding="utf-8-sig") as handle:
                    session = json.load(handle)
                segments = session.get("segments", []) if isinstance(session, dict) else []
                scene_count = len(segments) if isinstance(segments, list) else 0
            except Exception:
                scene_count = 0
            try:
                can_delete = os.path.commonpath([output_dir, folder]) == output_dir
            except ValueError:
                can_delete = False
            projects.append({
                "name": name,
                "project_folder": folder,
                "session_path": os.path.abspath(session_path),
                "updated": mtime,
                "scene_count": scene_count,
                "can_delete": can_delete,
            })
    projects.sort(key=lambda item: item.get("updated", 0), reverse=True)
    return {"projects": projects, "output_dir": output_dir, "project_roots": roots}


def _delete_builder_project(payload):
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    try:
        common = os.path.commonpath([output_dir, project_folder])
    except ValueError:
        common = ""
    if common != output_dir:
        raise ValueError("Project is outside the ComfyUI output folder, so it was not deleted.")
    if not os.path.isdir(project_folder):
        return {"deleted": False, "project_folder": project_folder, "reason": "Project folder was already missing."}
    if not os.path.isfile(_session_path(project_folder)):
        raise ValueError("This folder does not look like a Music Video Builder project.")
    shutil.rmtree(project_folder)
    return {"deleted": True, "project_folder": project_folder}
