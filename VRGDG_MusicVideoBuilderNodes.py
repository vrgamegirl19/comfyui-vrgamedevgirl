import json
import math
import os
import re
import asyncio
import subprocess
import shutil
import time
import wave
import base64

import folder_paths
from aiohttp import web
from PIL import Image
from server import PromptServer

from .VRGDG_VideoEditorNodes import (
    _clean_gemma_prompt_text,
    _clean_visual_gemma_text,
    _image_from_data_url,
    _I2V_INSTRUCTIONS,
    _TEXT_ONLY_T2I_INSTRUCTIONS,
    _VISUAL_T2I_INSTRUCTIONS,
)
from .VRGDG_WorkflowRunnerNodes import _resolve_comfy_image_path


_VRGDG_MUSIC_BUILDER_ROUTES_REGISTERED = False


def _vrgdg_textfile_path(folder_name, file_name):
    return os.path.join(
        folder_paths.get_output_directory(),
        "VRGDG_TEMP",
        "TextFiles",
        folder_name,
        file_name,
    )


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


def _newest_file(folder, extensions):
    if not os.path.isdir(folder):
        return ""
    candidates = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(tuple(extensions)):
            candidates.append(path)
    if not candidates:
        return ""
    return max(candidates, key=lambda path: os.path.getmtime(path))


def _default_audio_srt_paths():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    srt_folder = os.path.join(repo_dir, "srt_files")
    audio_folder = os.path.join(folder_paths.get_output_directory(), "VRGDG_AudioFiles")
    return {
        "audio_path": _newest_file(audio_folder, (".wav", ".mp3", ".flac", ".m4a", ".ogg")),
        "srt_path": _newest_file(srt_folder, (".srt",)),
        "audio_folder": audio_folder,
        "srt_folder": srt_folder,
    }


def _open_native_picker(kind):
    try:
        return _open_tk_picker(kind)
    except Exception as tk_exc:
        try:
            return _open_powershell_picker(kind)
        except Exception as ps_exc:
            raise RuntimeError(f"Native file dialog is not available. tkinter error: {tk_exc}; PowerShell error: {ps_exc}")


def _open_tk_picker(kind):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(f"Native file dialog is not available: {exc}")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "audio":
            path = filedialog.askopenfilename(
                title="Choose audio file",
                filetypes=[
                    ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                    ("All files", "*.*"),
                ],
            )
        elif kind == "srt":
            path = filedialog.askopenfilename(
                title="Choose SRT file",
                filetypes=[("SRT files", "*.srt"), ("All files", "*.*")],
            )
        elif kind == "project_folder":
            path = filedialog.askdirectory(title="Choose project folder")
        else:
            raise ValueError(f"Unknown picker type: {kind}")
        return str(path or "")
    finally:
        root.destroy()


def _open_powershell_picker(kind):
    if kind == "audio":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose audio file'
$dialog.Filter = 'Audio files (*.wav;*.mp3;*.flac;*.m4a;*.ogg)|*.wav;*.mp3;*.flac;*.m4a;*.ogg|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write($dialog.FileName) }
"""
    elif kind == "srt":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose SRT file'
$dialog.Filter = 'SRT files (*.srt)|*.srt|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write($dialog.FileName) }
"""
    elif kind == "image":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose image file'
$dialog.Filter = 'Image files (*.png;*.jpg;*.jpeg;*.webp)|*.png;*.jpg;*.jpeg;*.webp|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write($dialog.FileName) }
"""
    elif kind == "project_folder":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose project folder'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write($dialog.SelectedPath) }
"""
    else:
        raise ValueError(f"Unknown picker type: {kind}")

    result = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-Command", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "PowerShell picker failed.").strip())
    return result.stdout.strip()


def _resolve_existing_file(raw_path, label="file"):
    text = str(raw_path or "").strip().strip('"')
    if not text:
        raise ValueError(f"{label} path is empty.")
    path = os.path.abspath(text)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def _safe_project_name(value):
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_. -]+", "_", text).strip(" ._")
    return text or "VRGDG_MusicVideoBuilder"


def _default_project_folder(audio_path, project_name):
    parent = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    name = _safe_project_name(project_name or f"{stem}_builder")
    return os.path.join(parent, name)


def _unique_folder_path(path):
    folder = os.path.abspath(str(path or "").strip().strip('"'))
    if not folder:
        raise ValueError("Project folder is empty.")
    if not os.path.exists(folder):
        return folder
    for index in range(2, 10000):
        candidate = f"{folder}_{index:03d}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"Could not create a unique project folder for: {folder}")


def _project_target_from_payload(payload, preferred_key="project_folder"):
    raw = str(payload.get(preferred_key, "") or "").strip().strip('"')
    if not raw:
        raw = str(payload.get("project_name", "") or "").strip().strip('"')
    if not raw:
        raw = f"VRGDG_Project_{time.strftime('%Y%m%d_%H%M%S')}"
    if os.path.isabs(raw) or os.path.dirname(raw):
        return os.path.abspath(raw)
    return os.path.join(folder_paths.get_output_directory(), _safe_project_name(raw))


def _new_builder_project(payload):
    target = _unique_folder_path(_project_target_from_payload(payload, "project_folder"))
    os.makedirs(target, exist_ok=True)
    os.makedirs(_images_folder(target), exist_ok=True)
    os.makedirs(_prompts_folder(target), exist_ok=True)
    os.makedirs(_context_folder(target), exist_ok=True)
    for filename in ("ConceptPrompts.txt", "I2VMotionNotes.txt", "themestyle.txt", "storyconcept.txt", "subjectsandscenes.txt"):
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

    os.makedirs(target, exist_ok=True)
    os.makedirs(_images_folder(target), exist_ok=True)
    os.makedirs(_prompts_folder(target), exist_ok=True)
    os.makedirs(_context_folder(target), exist_ok=True)

    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    segments = session.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    overlay_segments = session.get("overlay_segments", [])
    if not isinstance(overlay_segments, list):
        overlay_segments = []
        session["overlay_segments"] = overlay_segments
    audio_raw = str(payload.get("audio_path", "") or "").strip().strip('"')
    audio_path = _resolve_existing_file(audio_raw, "Audio file") if audio_raw else ""
    audio_path, session = _snapshot_project_assets(target, session, audio_path, source)
    session = _copy_session_assets_to_project(target, session)
    session = _rebase_project_owned_paths(target, source, session)
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
    return {
        "project_folder": target,
        "session_path": _session_path(target),
        "srt_path": _srt_path(target),
        "images_folder": _images_folder(target),
        "prompts_folder": _prompts_folder(target),
        "context_folder": _context_folder(target),
        "session": session,
    }


def _session_path(project_folder):
    return os.path.join(project_folder, "vrgdg_builder_session.json")


def _srt_path(project_folder):
    return os.path.join(project_folder, "builder_segments.srt")


def _images_folder(project_folder):
    return os.path.join(project_folder, "zimage_approved")


def _prompts_folder(project_folder):
    return os.path.join(project_folder, "prompts")


def _context_folder(project_folder):
    return os.path.join(project_folder, "project_context")


def _scene_image_path(project_folder, scene_number, extension=".png"):
    scene = max(1, int(scene_number or 1))
    ext = str(extension or ".png").lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        ext = ".png"
    return os.path.join(_images_folder(project_folder), f"image_{scene:04d}{ext}")


def _is_internal_approved_image_path(path):
    parts = os.path.normpath(str(path or "")).split(os.sep)
    return "zimage_approved" in {part.lower() for part in parts}


def _scene_preview_folder(project_folder, scene_number):
    scene = max(1, int(scene_number or 1))
    return os.path.join(project_folder, "scene_image_previews", f"scene_{scene:04d}")


def _scene_audio_folder(project_folder):
    return os.path.join(project_folder, "scene_audio")


def _scene_audio_path(project_folder, scene_number, extension=".wav"):
    scene = max(1, int(scene_number or 1))
    ext = str(extension or ".wav").lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        ext = ".wav"
    return os.path.join(_scene_audio_folder(project_folder), f"audio_{scene:04d}{ext}")


def _unique_preview_path(project_folder, scene_number, extension=".png"):
    folder = _scene_preview_folder(project_folder, scene_number)
    os.makedirs(folder, exist_ok=True)
    ext = str(extension or ".png").lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        ext = ".png"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(folder, f"preview_{stamp}{ext}")
    if not os.path.exists(base):
        return base
    index = 2
    while True:
        candidate = os.path.join(folder, f"preview_{stamp}_{index:02d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _is_inside_folder(path, folder):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(folder)]) == os.path.abspath(folder)
    except Exception:
        return False


def _copy_file_into_folder(source_path, target_folder, target_name=None):
    if not source_path:
        return ""
    source = os.path.abspath(str(source_path or "").strip().strip('"'))
    if not os.path.isfile(source):
        return ""
    os.makedirs(target_folder, exist_ok=True)
    name = target_name or os.path.basename(source)
    safe_stem = _safe_project_name(os.path.splitext(name)[0])
    ext = os.path.splitext(name)[1] or os.path.splitext(source)[1]
    target = os.path.join(target_folder, f"{safe_stem}{ext}")
    if os.path.abspath(source) != os.path.abspath(target):
        shutil.copy2(source, target)
    return target


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
        copied_audio = _copy_file_into_folder(
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


def _copy_file_if_exists(source_path, target_path):
    source = str(source_path or "").strip().strip('"')
    if not source or not os.path.isfile(source):
        return ""
    source = os.path.abspath(source)
    target = os.path.abspath(target_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.normcase(source) == os.path.normcase(target):
        return target
    shutil.copy2(source, target)
    return target


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
        for overlay_index, segment in enumerate(overlay_segments, start=1):
            if not isinstance(segment, dict):
                continue
            scene_number = 10000 + overlay_index
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
    for key in ("audio_path", "prompt_json_path", "theme_style_path", "story_idea_path", "subject_scene_path"):
        rebased = _project_rebased_path(project_folder, old_project_folder, session.get(key, ""))
        if rebased:
            session[key] = rebased
    if isinstance(session.get("flux_global_image_ingredients"), list):
        for ingredient in session["flux_global_image_ingredients"]:
            if not isinstance(ingredient, dict):
                continue
            rebased = _project_rebased_path(project_folder, old_project_folder, ingredient.get("path", ""))
            if rebased:
                ingredient["path"] = rebased

    segments = session.get("segments", [])
    overlay_segments = session.get("overlay_segments", [])
    if not isinstance(segments, list):
        return session
    if not isinstance(overlay_segments, list):
        overlay_segments = []
    for segment in list(segments) + list(overlay_segments):
        if not isinstance(segment, dict):
            continue
        for key in (
            "approved_image_path",
            "custom_image_path",
            "ref_image_path",
            "flux_subject_image_path",
            "flux_location_image_path",
            "video_path",
            "custom_audio_path",
        ):
            rebased = _project_rebased_path(project_folder, old_project_folder, segment.get(key, ""))
            if rebased:
                segment[key] = rebased
        if isinstance(segment.get("image_history"), list):
            segment["image_history"] = [
                _project_rebased_path(project_folder, old_project_folder, item) or item
                for item in segment["image_history"]
            ]
        if isinstance(segment.get("flux_image_ingredients"), list):
            for ingredient in segment["flux_image_ingredients"]:
                if not isinstance(ingredient, dict):
                    continue
                rebased = _project_rebased_path(project_folder, old_project_folder, ingredient.get("path", ""))
                if rebased:
                    ingredient["path"] = rebased
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


def _resolve_project_asset_path(project_folder, old_project_folder, raw_path, scene_number=None):
    for candidate in _project_path_candidates(project_folder, old_project_folder, raw_path, scene_number):
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return str(raw_path or "")


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
    session["project_folder"] = project_folder
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

    existing_count = len(segments)
    asset_numbers = _project_scene_numbers(project_folder)
    base_asset_numbers = [number for number in asset_numbers if number < 10000]
    target_count = max(existing_count, max(base_asset_numbers) if base_asset_numbers else 0)
    for index in range(existing_count + 1, target_count + 1):
        start = float((index - 1) * 4)
        segments.append({
            "id": f"recovered_scene_{index}",
            "label": f"Scene {index}",
            "start": start,
            "end": start + 4,
            "source": "recovered",
        })

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
        if not approved:
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
        scene_number = 10000 + index
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


def _format_srt_time(seconds):
    total_ms = max(0, int(round(float(seconds or 0) * 1000)))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_srt_time(text):
    match = re.match(r"^\s*(\d+):(\d+):(\d+)[,.](\d+)\s*$", str(text or ""))
    if not match:
        raise ValueError(f"Invalid SRT time: {text}")
    hours, minutes, seconds, millis = [int(part) for part in match.groups()]
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _parse_srt_segments(srt_text):
    blocks = re.split(r"\n\s*\n", str(srt_text or "").strip(), flags=re.MULTILINE)
    segments = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_index = next((idx for idx, line in enumerate(lines) if "-->" in line), -1)
        if timing_index < 0:
            continue
        left, right = [part.strip() for part in lines[timing_index].split("-->", 1)]
        start = _parse_srt_time(left)
        end = max(start + 0.1, _parse_srt_time(right))
        label = " ".join(lines[timing_index + 1:]).strip() or f"Scene {len(segments) + 1}"
        segments.append(
            {
                "id": f"srt_{len(segments) + 1}_{int(start * 1000)}",
                "start": round(start, 3),
                "end": round(end, 3),
                "label": label[:80] or f"Scene {len(segments) + 1}",
                "notes": label,
                "t2i_prompt": "",
                "i2v_prompt": "",
                "ref_image_path": "",
                "use_vision_reference": False,
                "image": None,
                "source": "srt",
            }
        )
    return segments


def _load_srt_segments(path):
    srt_path = _resolve_existing_file(path, "SRT file")
    with open(srt_path, "r", encoding="utf-8-sig") as handle:
        segments = _parse_srt_segments(handle.read())
    if not segments:
        raise ValueError("No SRT timing blocks were found.")
    return {"srt_path": srt_path, "segments": segments}


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


def _read_text_file(path, label):
    if not str(path or "").strip():
        return ""
    text_path = _resolve_existing_file(path, label)
    with open(text_path, "r", encoding="utf-8-sig") as handle:
        return handle.read().strip()


def _resolve_editable_text_file(path):
    raw_path = str(path or "").strip().strip('"')
    if not raw_path:
        raise ValueError("Text file path is empty.")
    file_path = os.path.normpath(os.path.abspath(raw_path))
    if os.path.splitext(file_path)[1].lower() != ".txt":
        raise ValueError("Only .txt files can be edited here.")
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


def _segments_to_srt(segments):
    lines = []
    ordered = sorted(segments, key=lambda item: float(item.get("start", 0) or 0))
    for index, segment in enumerate(ordered, start=1):
        start = float(segment.get("start", 0) or 0)
        end = max(start + 0.1, float(segment.get("end", start + 4) or start + 4))
        text = str(segment.get("label") or segment.get("t2i_prompt") or f"Scene {index}").strip()
        lines.extend([str(index), f"{_format_srt_time(start)} --> {_format_srt_time(end)}", text, ""])
    return "\n".join(lines).strip() + "\n"


def _read_audio_peaks_with_torchaudio(audio_path, target_peaks=1600):
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        return {"duration": 0, "sample_rate": sample_rate, "channels": 0, "peaks": []}
    channels = int(waveform.shape[0])
    mono = waveform.mean(dim=0).abs()
    total_samples = int(mono.numel())
    duration = total_samples / float(sample_rate or 1)
    peak_count = max(1, min(int(target_peaks or 1600), total_samples))
    samples_per_peak = max(1, math.ceil(total_samples / peak_count))
    padded = peak_count * samples_per_peak - total_samples
    if padded > 0:
        mono = torch.nn.functional.pad(mono, (0, padded))
    chunks = mono.reshape(peak_count, samples_per_peak)
    peaks_tensor = torch.sqrt(torch.mean(chunks * chunks, dim=1)).clamp(0, 1)
    return {
        "duration": duration,
        "sample_rate": int(sample_rate or 0),
        "channels": channels,
        "peaks": [float(value) for value in peaks_tensor.cpu().tolist()],
    }


def _read_audio_peaks_with_wave(audio_path, target_peaks=1600):
    with wave.open(audio_path, "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        duration = frames / float(sample_rate or 1)
        if sample_width not in (1, 2, 4):
            raise ValueError("Only 8-bit, 16-bit, and 32-bit WAV files are supported by the fallback reader.")
        raw = handle.readframes(frames)

    if not raw:
        return {"duration": duration, "sample_rate": sample_rate, "channels": channels, "peaks": []}

    import audioop

    mono = raw
    if channels > 1:
        mono = audioop.tomono(raw, sample_width, 0.5, 0.5)
    total_samples = len(mono) // sample_width
    peak_count = max(1, min(int(target_peaks or 1600), total_samples))
    samples_per_peak = max(1, math.ceil(total_samples / peak_count))
    peaks = []
    for offset in range(0, len(mono), samples_per_peak * sample_width):
        chunk = mono[offset: offset + samples_per_peak * sample_width]
        if not chunk:
            continue
        rms = audioop.rms(chunk, sample_width)
        maximum = float((1 << ((sample_width * 8) - 1)) - 1)
        peaks.append(min(1.0, rms / maximum if maximum else 0.0))

    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "peaks": peaks,
    }


def _read_audio_peaks(audio_path, target_peaks=1600):
    try:
        return _read_audio_peaks_with_torchaudio(audio_path, target_peaks)
    except Exception as torch_exc:
        try:
            return _read_audio_peaks_with_wave(audio_path, target_peaks)
        except Exception as wave_exc:
            raise ValueError(
                "Could not read audio for waveform. Try a standard WAV, MP3, FLAC, or M4A file. "
                f"torchaudio error: {torch_exc}; wav fallback error: {wave_exc}"
            )


def _estimate_beats_from_peaks(peaks, duration):
    values = [float(value or 0) for value in peaks or []]
    total_duration = float(duration or 0)
    if len(values) < 8 or total_duration <= 0:
        return []
    step = total_duration / len(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(max(0.0, variance))
    threshold = mean + (std * 0.65)
    min_gap = max(0.22, min(0.55, total_duration / 500))
    beats = []
    last_time = -999.0
    for index in range(1, len(values) - 1):
        value = values[index]
        if value < threshold:
            continue
        if value < values[index - 1] or value < values[index + 1]:
            continue
        beat_time = index * step
        if beat_time - last_time < min_gap:
            if beats and value > values[int(beats[-1] / step)]:
                beats[-1] = beat_time
                last_time = beat_time
            continue
        beats.append(round(beat_time, 3))
        last_time = beat_time
    return beats


def _looks_like_gemma_repeat_failure(text):
    sample = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not sample:
        return False

    compact = re.sub(r"[^a-z0-9_<>\-|]+", "", sample)
    for marker in (
        "completion-completion-completion",
        "thought-thought-thought",
        "de-facto-de-facto-de-facto",
        "de-fleshed",
        "cast-cast-cast",
        "prompt-cast-cast",
        "thoughtthoughtthought",
        "ownnessownnessownness",
        "nessnessnessness",
        "end_anow",
        "thought_turn",
        "turn_turn",
        "<|channel>",
        "<channel|>",
    ):
        if marker in compact or marker in sample:
            return True

    if re.search(r"([a-z]{2,16})\1{5,}", compact):
        return True

    if re.search(r"\b([a-zA-Z_]{3,})(?:[-\s]+\1){5,}\b", sample):
        return True

    unicode_tokens = re.findall(r"[\w']+", sample, flags=re.UNICODE)
    unicode_tokens = [token.strip("_'") for token in unicode_tokens if token.strip("_'")]
    if len(unicode_tokens) >= 16:
        token_counts = {}
        for token in unicode_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        if token_counts and max(token_counts.values()) >= 10 and max(token_counts.values()) / float(len(unicode_tokens)) >= 0.20:
            return True
        for size in (2, 3, 4):
            if len(unicode_tokens) < size * 4:
                continue
            phrase_counts = {}
            for index in range(len(unicode_tokens) - size + 1):
                phrase = " ".join(unicode_tokens[index:index + size])
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            if phrase_counts and max(phrase_counts.values()) >= 8:
                return True

    words = re.findall(r"[a-zA-Z_][a-zA-Z_']{2,}", sample)
    if len(words) < 18:
        return False

    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    common_words = {"the", "and", "with", "that", "this", "from", "into", "while", "during"}
    repeated_words = [
        count
        for word, count in counts.items()
        if word not in common_words
    ]
    if repeated_words and max(repeated_words) >= 10 and max(repeated_words) / float(len(words)) >= 0.25:
        return True

    phrases = [" ".join(words[index:index + 2]) for index in range(len(words) - 1)]
    if len(phrases) >= 12:
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        if max(phrase_counts.values()) >= 6:
            return True

    return False


def _validate_builder_gemma_prompt(text, label):
    if not str(text or "").strip():
        raise ValueError(f"Gemma returned an empty {label} prompt.")
    if _looks_like_gemma_repeat_failure(text):
        hint = "Try again or shorten the notes."
        if str(label or "").lower() != "flux/klein":
            hint = "Try again, shorten the notes, or turn off image reference."
        raise ValueError(
            f"Gemma returned repeated/thought text for the {label} prompt. "
            f"{hint}"
        )


def _generate_builder_t2i_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    ref_image_path = str(payload.get("ref_image_path", "") or "").strip().strip('"')
    user_notes = str(payload.get("user_notes", "") or "").strip()
    theme_style = _read_text_file(payload.get("theme_style_path", ""), "Theme/style file")
    story_idea = _read_text_file(payload.get("story_idea_path", ""), "Story idea file")
    subject_scene = _read_text_file(payload.get("subject_scene_path", ""), "Subject/scene file")
    context_parts = []
    if subject_scene:
        context_parts.append(f"Subject/scene:\n{subject_scene}")
    if theme_style:
        context_parts.append(f"Theme/style:\n{theme_style}")
    if story_idea:
        context_parts.append(f"Story idea:\n{story_idea}")
    if context_parts:
        user_notes = "\n\n".join(context_parts + ([f"Segment notes:\n{user_notes}"] if user_notes else []))
    use_vision = bool(payload.get("use_vision"))
    has_ref_image = bool(use_vision and ref_image_path and os.path.isfile(ref_image_path))
    if not model_file:
        raise ValueError("Choose a Gemma model first.")
    if use_vision and not has_ref_image:
        raise ValueError("Choose a valid reference image path or turn off vision reference.")
    if has_ref_image and not mmproj_file:
        raise ValueError("Choose an mmproj file for the vision model.")
    if not has_ref_image and not user_notes:
        raise ValueError("Enter scene notes or provide a reference image.")

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = llm._resolve_dropdown_path(mmproj_file, llm.MISSING_MMPROJ_OPTION) if has_ref_image else ""
    image = Image.open(ref_image_path).convert("RGB") if has_ref_image else None
    if has_ref_image:
        prompt = _VISUAL_T2I_INSTRUCTIONS
        prompt += f"\n\nUser notes:\n{user_notes or 'Use the reference image as the guide.'}"
    else:
        prompt = f"{_TEXT_ONLY_T2I_INSTRUCTIONS}\n\nUser notes:\n{user_notes}"

    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or (0.25 if has_ref_image else 0.6))
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or (8000 if has_ref_image else 1200))
    unload_after = True

    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            mmproj_path=mmproj_path,
        )
        if has_ref_image:
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[image],
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            text = llm._run_gguf_text_pipeline(
                model=model,
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        text = _clean_visual_gemma_text(text)
        _validate_builder_gemma_prompt(text, "T2I")
        return {
            "prompt": text,
            "used_reference_image": has_ref_image,
            "used_model": model_path,
            "used_mmproj": mmproj_path,
            "unloaded": unload_after,
        }
    finally:
        if unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _generate_builder_i2v_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    t2i_prompt = str(payload.get("t2i_prompt", "") or "").strip()
    image_reference_path = str(payload.get("image_reference_path", "") or "").strip().strip('"')
    image_reference_data = str(payload.get("image_reference_data", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    if not model_file:
        raise ValueError("Choose an I2V Gemma model first.")
    if not model_file.lower().endswith(".gguf"):
        raise ValueError("The I2V model field is not a GGUF model.")

    image = None
    has_image_reference = False
    if image_reference_data:
        image = _image_from_data_url(image_reference_data).convert("RGB")
        has_image_reference = True
    elif image_reference_path:
        image_path = _resolve_existing_file(image_reference_path, "I2V image reference")
        image = Image.open(image_path).convert("RGB")
        has_image_reference = True

    if has_image_reference and not mmproj_file:
        raise ValueError("Choose an mmproj file for the I2V vision model.")
    if not has_image_reference and not t2i_prompt:
        raise ValueError("Create or paste a T2I prompt first, or save/load an image reference.")
    if not has_image_reference:
        theme_style = _read_text_file(payload.get("theme_style_path", ""), "Theme/style file")
        story_idea = _read_text_file(payload.get("story_idea_path", ""), "Story idea file")
        subject_scene = _read_text_file(payload.get("subject_scene_path", ""), "Subject/scene file")
        context_parts = []
        if subject_scene:
            context_parts.append(f"Subject/scene:\n{subject_scene}")
        if theme_style:
            context_parts.append(f"Theme/style:\n{theme_style}")
        if story_idea:
            context_parts.append(f"Story idea:\n{story_idea}")
        if context_parts:
            user_notes = "\n\n".join(context_parts + ([f"Segment motion notes:\n{user_notes}"] if user_notes else []))

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = llm._resolve_dropdown_path(mmproj_file, llm.MISSING_MMPROJ_OPTION) if has_image_reference else ""
    if has_image_reference:
        prompt = (
            f"{_I2V_INSTRUCTIONS}\n\n"
            "Use the provided image as the primary visual reference. Preserve the visible subject, setting, clothing, mood, and scene identity from the image. "
            "Use the text-to-image prompt only as extra scene context. "
            "Use only the provided image, the text-to-image prompt, and the user motion/camera notes. Do not use concept prompts, global story text, theme files, or subject files. "
            "Use the user motion/camera notes as the highest priority when deciding motion, performance, camera movement, and energy.\n\n"
            f"Text-to-image prompt:\n{t2i_prompt or 'Use the image as the main visual reference.'}\n\n"
        )
    else:
        prompt = f"{_I2V_INSTRUCTIONS}\n\nText-to-image prompt:\n{t2i_prompt}\n\n"
    prompt += f"User motion/camera notes:\n{user_notes or 'Create fast cinematic performance motion that fits the scene.'}"

    n_ctx = int(payload.get("n_ctx") or 13000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or (0.25 if has_image_reference else 0.7))
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 4000)
    unload_after = True

    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            mmproj_path=mmproj_path,
        )
        if has_image_reference:
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[image],
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            text = llm._run_gguf_text_pipeline(
                model=model,
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        text = _clean_gemma_prompt_text(text)
        _validate_builder_gemma_prompt(text, "I2V")
        return {"prompt": text, "used_model": model_path, "used_mmproj": mmproj_path, "used_image_reference": has_image_reference, "unloaded": unload_after}
    finally:
        if unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _image_from_prompt_payload(path, data, label):
    raw_data = str(data or "").strip()
    raw_path = str(path or "").strip().strip('"')
    if raw_data:
        return _image_from_data_url(raw_data).convert("RGB")
    if raw_path:
        image_path = _resolve_existing_file(raw_path, label)
        return Image.open(image_path).convert("RGB")
    raise ValueError(f"{label} is required.")


def _combine_subject_location_images(subject_image, location_image):
    max_height = 640
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    resized = []
    for image in (subject_image, location_image):
        scale = min(1.0, max_height / max(1, image.height))
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        resized.append(image.resize((width, height), resample))
    gap = 24
    canvas_width = resized[0].width + resized[1].width + gap
    canvas_height = max(resized[0].height, resized[1].height)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (20, 20, 20))
    canvas.paste(resized[0], (0, (canvas_height - resized[0].height) // 2))
    canvas.paste(resized[1], (resized[0].width + gap, (canvas_height - resized[1].height) // 2))
    return canvas


def _combine_flux_ingredient_images(images):
    if not images:
        raise ValueError("At least one image ingredient is required.")
    cell_size = 384 if len(images) <= 4 else 256
    gap = 24
    columns = 1 if len(images) == 1 else int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / columns))
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)

    canvas_width = (columns * cell_size) + (gap * (columns - 1))
    canvas_height = (rows * cell_size) + (gap * (rows - 1))
    canvas = Image.new("RGB", (canvas_width, canvas_height), (20, 20, 20))

    resized = []
    for image in images:
        scale = min(1.0, cell_size / max(1, image.width), cell_size / max(1, image.height))
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        resized.append(image.resize((width, height), resample))

    for index, image in enumerate(resized):
        column = index % columns
        row = index // columns
        cell_x = column * (cell_size + gap)
        cell_y = row * (cell_size + gap)
        x = cell_x + ((cell_size - image.width) // 2)
        y = cell_y + ((cell_size - image.height) // 2)
        canvas.paste(image, (x, y))
    return canvas


def _clear_comfy_model_memory():
    try:
        import comfy.model_management as model_management

        unload_all_models = getattr(model_management, "unload_all_models", None)
        if callable(unload_all_models):
            unload_all_models()
        soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
        if callable(soft_empty_cache):
            soft_empty_cache()
    except Exception:
        pass


def _generate_flux_klein_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    if not model_file:
        raise ValueError("Choose a Gemma vision model first.")
    if not mmproj_file:
        raise ValueError("Choose an mmproj file for the vision model.")

    ingredients = payload.get("image_ingredients") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("Image ingredients must be a list.")
    images = []
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        images.append(_image_from_prompt_payload(item.get("path", ""), item.get("data", ""), f"Image ingredient {index}"))
    combined_image = _combine_flux_ingredient_images(images)
    instruction = (
        "Create one polished text-to-image prompt for an image generation model.\n\n"
        "The provided reference image is a composite of image ingredients. These may include a character, background, props, style references, or other visual ingredients.\n\n"
        "Use the visible ingredients to create one coherent new scene. Use the user's notes for pose, camera framing, mood, or other requested details, and give user notes priority.\n\n"
        "Rules:\n"
        "- Output one normal text-to-image prompt, not an edit prompt.\n"
        "- Describe only visible concrete details.\n"
        "- Do not mention reference image, composite, source images, ingredient images, or image grid.\n"
        "- Do not include labels, notes, quotes, markdown, or explanations.\n"
        "- Keep it cinematic, detailed, and visually specific.\n"
        "- Keep the prompt under 120 words.\n\n"
        f"User notes:\n{user_notes or 'Create a cinematic image using the provided image ingredients.'}"
    )

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = llm._resolve_dropdown_path(mmproj_file, llm.MISSING_MMPROJ_OPTION)
    # Flux/Klein only needs one short prompt, and vision GGUF context is expensive.
    # Keep this lower than the broader Gemma prompt tools to reduce crash risk.
    n_ctx = int(payload.get("n_ctx") or 2048)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 350)
    clear_before_load = bool(payload.get("clear_before_load", True))
    unload_after = bool(payload.get("unload_after", True))

    try:
        if clear_before_load:
            _clear_comfy_model_memory()
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            mmproj_path=mmproj_path,
        )
        text = llm._run_gguf_vision_pipeline(
            model=model,
            pil_images=[combined_image],
            instruction_text=instruction,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        text = _clean_visual_gemma_text(text)
        _validate_builder_gemma_prompt(text, "Flux/Klein")
        return {"prompt": text, "used_model": model_path, "used_mmproj": mmproj_path, "unloaded": unload_after}
    finally:
        if unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
            _clear_comfy_model_memory()


def _gemma_choices():
    from .LLM import VRGDG_SuperGemmaGGUFChat

    return {
        "models": VRGDG_SuperGemmaGGUFChat._list_local_gemma_gguf_choices(),
        "mmproj": VRGDG_SuperGemmaGGUFChat._list_local_mmproj_choices(),
    }


def _save_builder_session(payload):
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

    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    segments = session.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    overlay_segments = session.get("overlay_segments", [])
    if not isinstance(overlay_segments, list):
        overlay_segments = []
        session["overlay_segments"] = overlay_segments
    audio_path, session = _snapshot_project_assets(project_folder, session, audio_path)
    session = {
        **session,
        "audio_path": audio_path,
        "project_folder": project_folder,
        "updated": time.time(),
        "segments": segments,
    }

    srt_text = _segments_to_srt(segments)
    _backup_session_file(project_folder)
    with open(_session_path(project_folder), "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(_srt_path(project_folder), "w", encoding="utf-8") as handle:
        handle.write(srt_text)

    t2i_lines = []
    i2v_lines = []
    for segment in sorted(list(segments) + list(overlay_segments), key=lambda item: float(item.get("start", 0) or 0)):
        if str(segment.get("t2i_prompt", "")).strip():
            t2i_lines.append(str(segment.get("t2i_prompt", "")).strip())
        if str(segment.get("i2v_prompt", "")).strip():
            i2v_lines.append(str(segment.get("i2v_prompt", "")).strip())
    with open(os.path.join(_prompts_folder(project_folder), "t2i_prompts.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(t2i_lines).strip() + ("\n" if t2i_lines else ""))
    with open(os.path.join(_prompts_folder(project_folder), "i2v_prompts.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(i2v_lines).strip() + ("\n" if i2v_lines else ""))

    return {
        "project_folder": project_folder,
        "session_path": _session_path(project_folder),
        "srt_path": _srt_path(project_folder),
        "images_folder": _images_folder(project_folder),
        "prompts_folder": _prompts_folder(project_folder),
        "context_folder": _context_folder(project_folder),
        "session": session,
    }


def _save_scene_image(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    os.makedirs(_images_folder(project_folder), exist_ok=True)
    scene_number = int(payload.get("scene_number") or 1)

    image_data = str(payload.get("image_data", "") or "").strip()
    if image_data:
        target_path = _scene_image_path(project_folder, scene_number, ".png")
        image = _image_from_data_url(image_data)
        image.save(target_path, format="PNG")
    else:
        source_path = ""
        image_info = payload.get("image")
        if isinstance(image_info, dict):
            source_path = _resolve_comfy_image_path(image_info)
        else:
            source_path = _resolve_existing_file(payload.get("source_path", ""), "Image file")
        ext = os.path.splitext(source_path)[1] or ".png"
        target_path = _scene_image_path(project_folder, scene_number, ext)
        shutil.copy2(source_path, target_path)
    return {
        "saved_path": target_path,
        "images_folder": _images_folder(project_folder),
        "scene_number": scene_number,
    }


def _delete_project_media(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    media_path = os.path.abspath(str(payload.get("path", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    if not media_path:
        raise ValueError("Media path is empty.")
    if not os.path.isfile(media_path):
        return {"deleted": False, "path": media_path, "reason": "File was already missing."}
    try:
        common = os.path.commonpath([project_folder, media_path])
    except ValueError:
        common = ""
    if common != project_folder:
        raise ValueError("This file is outside the current project folder, so it was not deleted.")
    os.remove(media_path)
    return {"deleted": True, "path": media_path}


def _archive_scene_image(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    scene_number = int(payload.get("scene_number") or 1)

    image_data = str(payload.get("image_data", "") or "").strip()
    if image_data:
        target_path = _unique_preview_path(project_folder, scene_number, ".png")
        image = _image_from_data_url(image_data)
        image.save(target_path, format="PNG")
    else:
        image_info = payload.get("image")
        if isinstance(image_info, dict):
            source_path = _resolve_comfy_image_path(image_info)
        else:
            source_path = _resolve_existing_file(payload.get("source_path", ""), "Image file")
        ext = os.path.splitext(source_path)[1] or ".png"
        target_path = _unique_preview_path(project_folder, scene_number, ext)
        shutil.copy2(source_path, target_path)

    return {
        "saved_path": target_path,
        "preview_folder": _scene_preview_folder(project_folder, scene_number),
        "scene_number": scene_number,
    }


def _audio_bytes_from_data_url(audio_data):
    raw = str(audio_data or "").strip()
    if not raw:
        raise ValueError("Audio data is empty.")
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def _save_scene_audio(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    scene_number = int(payload.get("scene_number") or 1)
    os.makedirs(_scene_audio_folder(project_folder), exist_ok=True)

    source_name = str(payload.get("audio_name", "") or "").strip()
    source_ext = os.path.splitext(source_name)[1].lower()
    audio_data = str(payload.get("audio_data", "") or "").strip()
    if audio_data:
        target_path = _scene_audio_path(project_folder, scene_number, source_ext or ".wav")
        with open(target_path, "wb") as handle:
            handle.write(_audio_bytes_from_data_url(audio_data))
    else:
        source_path = _resolve_existing_file(payload.get("source_path", ""), "Audio file")
        target_path = _scene_audio_path(project_folder, scene_number, os.path.splitext(source_path)[1] or ".wav")
        shutil.copy2(source_path, target_path)

    audio_info = _read_audio_peaks(target_path, 600)
    return {
        "saved_path": target_path,
        "audio_folder": _scene_audio_folder(project_folder),
        "scene_number": scene_number,
        **audio_info,
    }


def _save_project_audio(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    os.makedirs(project_folder, exist_ok=True)
    folder = os.path.join(project_folder, "project_audio")
    os.makedirs(folder, exist_ok=True)
    source_name = str(payload.get("audio_name", "") or "").strip() or "project_audio.wav"
    ext = os.path.splitext(source_name)[1].lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        ext = ".wav"
    target_path = os.path.join(folder, f"project_audio{ext}")
    audio_data = str(payload.get("audio_data", "") or "").strip()
    if audio_data:
        with open(target_path, "wb") as handle:
            handle.write(_audio_bytes_from_data_url(audio_data))
    else:
        source_path = _resolve_existing_file(payload.get("source_path", ""), "Audio file")
        shutil.copy2(source_path, target_path)
    audio_info = _read_audio_peaks(target_path, 1600)
    return {"saved_path": target_path, "audio_folder": folder, **audio_info}


def _save_project_srt(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    os.makedirs(project_folder, exist_ok=True)
    srt_text = str(payload.get("srt_text", "") or "")
    if not srt_text.strip():
        raise ValueError("SRT text is empty.")
    path = _srt_path(project_folder)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(srt_text)
    segments = _parse_srt_segments(srt_text)
    return {"srt_path": path, "segments": segments}


def _save_single_scene_srt(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    scene_number = int(payload.get("scene_number") or 1)
    duration = max(0.1, float(payload.get("duration") or 4))
    label = str(payload.get("label") or f"Scene {scene_number}").strip()
    folder = os.path.join(project_folder, "scene_srt")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"scene_{scene_number:04d}.srt")
    text = "\n".join([
        "1",
        f"{_format_srt_time(0)} --> {_format_srt_time(duration)}",
        label,
        "",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return {"srt_path": path, "scene_number": scene_number, "duration": duration}


def _trim_scene_audio(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    source_path = _resolve_existing_file(payload.get("source_path", ""), "Audio file")
    scene_number = int(payload.get("scene_number") or 1)
    start = max(0.0, float(payload.get("start") or 0))
    duration = max(0.05, float(payload.get("duration") or 0))
    folder = os.path.join(project_folder, "scene_audio_trimmed")
    os.makedirs(folder, exist_ok=True)
    target_path = os.path.join(folder, f"scene_audio_{scene_number:04d}.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        source_path,
        "-t",
        str(duration),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        target_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed to trim scene audio.").strip())
    return {"audio_path": target_path, "scene_number": scene_number, "start": start, "duration": duration, "format": "pcm_s16le_wav"}


def _find_ffmpeg_path():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except Exception:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or imageio-ffmpeg to mix scene audio.") from exc


def _concat_file_path(path):
    return os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")


def _scene_audio_mix_folder(project_folder):
    return os.path.join(project_folder, "project_audio")


def _prepare_scene_audio_mix(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    segments = payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError("No scenes were provided for scene audio mix.")
    allow_missing_scene_audio = bool(payload.get("allow_missing_scene_audio", False))

    ffmpeg_path = _find_ffmpeg_path()
    folder = _scene_audio_mix_folder(project_folder)
    os.makedirs(folder, exist_ok=True)
    parts_folder = os.path.join(folder, "_scene_audio_mix_parts")
    if os.path.isdir(parts_folder):
        shutil.rmtree(parts_folder, ignore_errors=True)
    os.makedirs(parts_folder, exist_ok=True)

    timeline_items = []
    missing = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            missing.append(f"Scene {index}: invalid scene data.")
            continue
        path = str(segment.get("custom_audio_path", "") or "").strip().strip('"')
        if not path:
            if allow_missing_scene_audio:
                start = max(0.0, float(segment.get("start", 0) or 0))
                end = max(start + 0.05, float(segment.get("end", start + 4) or start + 4))
                duration = max(0.05, end - start)
                timeline_items.append({
                    "index": index,
                    "path": "",
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "source_start": 0.0,
                    "silent": True,
                })
                continue
            missing.append(f"Scene {index}: custom audio is missing.")
            continue
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            missing.append(f"Scene {index}: custom audio file was not found: {path}")
            continue
        segment_start = max(0.0, float(segment.get("start", 0) or 0))
        segment_end = max(segment_start + 0.05, float(segment.get("end", segment_start + 4) or segment_start + 4))
        start = max(0.0, float(segment.get("custom_audio_timeline_start", segment_start) or segment_start))
        duration = float(segment.get("custom_audio_duration", 0) or 0)
        if duration <= 0:
            duration = segment_end - segment_start
        duration = max(0.05, duration)
        source_start = max(0.0, float(segment.get("custom_audio_source_start", 0) or 0))
        timeline_items.append({
            "index": index,
            "path": path,
            "start": start,
            "end": start + duration,
            "duration": duration,
            "source_start": source_start,
            "silent": False,
        })
    if missing:
        raise ValueError("\n".join(missing))

    timeline_items.sort(key=lambda item: (item["start"], item["index"]))
    concat_file = os.path.join(parts_folder, "scene_audio_mix_list.txt")
    part_paths = []
    cursor = 0.0
    part_index = 1
    for item in timeline_items:
        gap = max(0.0, item["start"] - cursor)
        if gap > 0.01:
            silence_path = os.path.join(parts_folder, f"part_{part_index:04d}_silence.wav")
            part_index += 1
            silence_cmd = [
                ffmpeg_path,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{gap:.6f}",
                "-c:a",
                "pcm_s16le",
                silence_path,
            ]
            result = subprocess.run(silence_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed to create silence.").strip())
            part_paths.append(silence_path)

        clip_path = os.path.join(parts_folder, f"part_{part_index:04d}_scene_{item['index']:04d}.wav")
        part_index += 1
        if item.get("silent"):
            clip_cmd = [
                ffmpeg_path,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{item['duration']:.6f}",
                "-c:a",
                "pcm_s16le",
                clip_path,
            ]
        else:
            clip_cmd = [
                ffmpeg_path,
                "-y",
                "-ss",
                f"{item['source_start']:.6f}",
                "-i",
                item["path"],
                "-t",
                f"{item['duration']:.6f}",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                clip_path,
            ]
        result = subprocess.run(clip_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or f"ffmpeg failed to prepare scene {item['index']} audio.").strip())
        part_paths.append(clip_path)
        cursor = max(cursor, item["start"] + item["duration"])

    if not part_paths:
        raise ValueError("No scene audio parts were created.")

    mix_path = os.path.join(folder, "scene_audio_mix.wav")
    with open(concat_file, "w", encoding="utf-8") as handle:
        for path in part_paths:
            handle.write(f"file '{_concat_file_path(path)}'\n")
    mix_cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c:a",
        "pcm_s16le",
        mix_path,
    ]
    result = subprocess.run(mix_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed to create scene audio mix.").strip())

    srt_path = _srt_path(project_folder)
    with open(srt_path, "w", encoding="utf-8") as handle:
        handle.write(_segments_to_srt(segments))

    shutil.rmtree(parts_folder, ignore_errors=True)
    audio_info = _read_audio_peaks(mix_path, 1600)
    return {
        "audio_path": mix_path,
        "srt_path": srt_path,
        "duration": audio_info.get("duration", cursor),
        "peaks": audio_info.get("peaks", []),
        "beats": _estimate_beats_from_peaks(audio_info.get("peaks", []), audio_info.get("duration", cursor)),
        "scene_count": len(timeline_items),
        "used_scene_audio": True,
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
    return {"project_folder": folder, "session_path": path, "srt_path": _srt_path(folder), "session": session}


def _list_builder_projects():
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    projects = []
    if not os.path.isdir(output_dir):
        return {"projects": projects, "output_dir": output_dir}
    for name in os.listdir(output_dir):
        folder = os.path.join(output_dir, name)
        if not os.path.isdir(folder):
            continue
        session_path = _session_path(folder)
        if not os.path.isfile(session_path):
            continue
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
        projects.append({
            "name": name,
            "project_folder": os.path.abspath(folder),
            "session_path": os.path.abspath(session_path),
            "updated": mtime,
            "scene_count": scene_count,
        })
    projects.sort(key=lambda item: item.get("updated", 0), reverse=True)
    return {"projects": projects, "output_dir": output_dir}


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


def _scan_builder_scene_videos(project_folder):
    folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not folder:
        raise ValueError("Project folder is empty.")
    video_folder = os.path.join(folder, "rendered_scene_videos")
    backup_root = os.path.join(folder, "rendered_scene_videos_backup")
    videos = {}
    video_backups = {}
    if not os.path.isdir(video_folder):
        return {"project_folder": folder, "video_folder": video_folder, "videos": videos, "video_backups": video_backups}
    pattern = re.compile(r"^video_(\d+)-audio\.mp4$", re.IGNORECASE)
    for name in os.listdir(video_folder):
        match = pattern.match(name)
        if not match:
            continue
        path = os.path.join(video_folder, name)
        if os.path.isfile(path):
            videos[str(int(match.group(1)))] = path
    if os.path.isdir(backup_root):
        backup_pattern = re.compile(r"^video_(\d+)-audio_.*\.mp4$", re.IGNORECASE)
        for root, _, names in os.walk(backup_root):
            for name in names:
                match = backup_pattern.match(name)
                if not match:
                    continue
                path = os.path.join(root, name)
                if not os.path.isfile(path):
                    continue
                key = str(int(match.group(1)))
                video_backups.setdefault(key, []).append(path)
        for paths in video_backups.values():
            paths.sort(key=lambda item: os.path.getmtime(item))
    return {"project_folder": folder, "video_folder": video_folder, "videos": videos, "video_backups": video_backups}


def _ensure_music_builder_routes():
    global _VRGDG_MUSIC_BUILDER_ROUTES_REGISTERED
    if _VRGDG_MUSIC_BUILDER_ROUTES_REGISTERED:
        return
    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.post("/vrgdg/music_builder/analyze_audio")
    async def vrgdg_music_builder_analyze_audio(request):
        try:
            payload = await request.json()
            audio_path = _resolve_existing_file(payload.get("audio_path", ""), "Audio file")
            result = _read_audio_peaks(audio_path, payload.get("target_peaks", 1600))
            result["beats"] = _estimate_beats_from_peaks(result.get("peaks", []), result.get("duration", 0))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "audio_path": audio_path, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_session")
    async def vrgdg_music_builder_save_session(request):
        try:
            payload = await request.json()
            result = _save_builder_session(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/new_project")
    async def vrgdg_music_builder_new_project(request):
        try:
            payload = await request.json()
            result = _new_builder_project(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_project_as")
    async def vrgdg_music_builder_save_project_as(request):
        try:
            payload = await request.json()
            result = _save_builder_project_as(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_scene_image")
    async def vrgdg_music_builder_save_scene_image(request):
        try:
            payload = await request.json()
            result = _save_scene_image(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/delete_project_media")
    async def vrgdg_music_builder_delete_project_media(request):
        try:
            payload = await request.json()
            result = _delete_project_media(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/archive_scene_image")
    async def vrgdg_music_builder_archive_scene_image(request):
        try:
            payload = await request.json()
            result = _archive_scene_image(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_scene_audio")
    async def vrgdg_music_builder_save_scene_audio(request):
        try:
            payload = await request.json()
            result = _save_scene_audio(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_project_audio")
    async def vrgdg_music_builder_save_project_audio(request):
        try:
            payload = await request.json()
            result = _save_project_audio(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_project_srt")
    async def vrgdg_music_builder_save_project_srt(request):
        try:
            payload = await request.json()
            result = _save_project_srt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_single_scene_srt")
    async def vrgdg_music_builder_save_single_scene_srt(request):
        try:
            payload = await request.json()
            result = _save_single_scene_srt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/trim_scene_audio")
    async def vrgdg_music_builder_trim_scene_audio(request):
        try:
            payload = await request.json()
            result = _trim_scene_audio(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/prepare_scene_audio_mix")
    async def vrgdg_music_builder_prepare_scene_audio_mix(request):
        try:
            payload = await request.json()
            result = _prepare_scene_audio_mix(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_session")
    async def vrgdg_music_builder_load_session(request):
        try:
            payload = await request.json()
            result = _load_builder_session(payload.get("project_folder", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/list_projects")
    async def vrgdg_music_builder_list_projects(request):
        try:
            result = _list_builder_projects()
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/delete_project")
    async def vrgdg_music_builder_delete_project(request):
        try:
            payload = await request.json()
            result = _delete_builder_project(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/scan_scene_videos")
    async def vrgdg_music_builder_scan_scene_videos(request):
        try:
            payload = await request.json()
            result = _scan_builder_scene_videos(payload.get("project_folder", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_srt")
    async def vrgdg_music_builder_load_srt(request):
        try:
            payload = await request.json()
            result = _load_srt_segments(payload.get("srt_path", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_prompt_json")
    async def vrgdg_music_builder_load_prompt_json(request):
        try:
            payload = await request.json()
            result = _load_prompt_json(payload.get("prompt_json_path", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_text_file")
    async def vrgdg_music_builder_load_text_file(request):
        try:
            payload = await request.json()
            result = _load_editable_text_file(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_text_file")
    async def vrgdg_music_builder_save_text_file(request):
        try:
            payload = await request.json()
            result = _save_editable_text_file(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/default_context_paths")
    async def vrgdg_music_builder_default_context_paths(request):
        return web.json_response({"ok": True, **_default_context_paths()})

    @server_instance.routes.post("/vrgdg/music_builder/project_prompt_creator_paths")
    async def vrgdg_music_builder_project_prompt_creator_paths(request):
        try:
            payload = await request.json()
            result = _project_prompt_creator_paths(payload.get("project_folder", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/default_audio_srt_paths")
    async def vrgdg_music_builder_default_audio_srt_paths(request):
        return web.json_response({"ok": True, **_default_audio_srt_paths()})

    @server_instance.routes.post("/vrgdg/music_builder/pick_path")
    async def vrgdg_music_builder_pick_path(request):
        try:
            payload = await request.json()
            path = _open_native_picker(str(payload.get("kind", "") or ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "path": path})

    @server_instance.routes.get("/vrgdg/music_builder/audio")
    async def vrgdg_music_builder_audio(request):
        raw_path = str(request.query.get("path", "") or "").strip()
        audio_path = os.path.normpath(os.path.abspath(raw_path))
        if not os.path.isfile(audio_path):
            return web.json_response({"ok": False, "error": "Audio file was not found."}, status=404)
        return web.FileResponse(audio_path)

    @server_instance.routes.get("/vrgdg/music_builder/gemma_choices")
    async def vrgdg_music_builder_gemma_choices(request):
        try:
            result = _gemma_choices()
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_t2i")
    async def vrgdg_music_builder_generate_t2i(request):
        try:
            payload = await request.json()
            result = _generate_builder_t2i_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_i2v")
    async def vrgdg_music_builder_generate_i2v(request):
        try:
            payload = await request.json()
            result = _generate_builder_i2v_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_flux_klein_prompt")
    async def vrgdg_music_builder_generate_flux_klein_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_klein_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    _VRGDG_MUSIC_BUILDER_ROUTES_REGISTERED = True


class VRGDG_MusicVideoBuilderUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "project_folder": ("STRING", {"default": ""}),
                "session_path": ("STRING", {"default": ""}),
                "srt_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("project_folder", "session_path", "srt_path")
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Prototype UI for building a music video from audio, timing segments, prompts, and approved ZImage previews."

    def noop(self, audio_path, project_folder, session_path, srt_path):
        return (project_folder, session_path, srt_path)


_ensure_music_builder_routes()


NODE_CLASS_MAPPINGS = {
    "VRGDG_MusicVideoBuilderUI": VRGDG_MusicVideoBuilderUI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MusicVideoBuilderUI": "VRGDG Music Video Builder UI",
}
