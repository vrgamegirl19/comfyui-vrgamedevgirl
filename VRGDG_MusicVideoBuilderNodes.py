import json
import math
import os
import re
import asyncio
import subprocess
import shutil
import sys
import time
import wave
import base64
import array
import gc
import urllib.error
import urllib.request

import folder_paths
from aiohttp import web
from PIL import Image
from server import PromptServer

from .VRGDG_ModelPathSettings import register_custom_model_root
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

_LM_STUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"

_T2V_INSTRUCTIONS = """Convert the user's concept prompt into a dynamic text-to-video prompt.

Use the user's prompt as the full scene foundation. Preserve the original subject, setting, outfit, mood, atmosphere, and scene identity. Infer only the missing video details needed to make the scene feel complete, including time of day, weather, lighting behavior, environmental movement, subject movement, camera movement, and performance energy. Do not add unrelated characters, new locations, major story changes, captions, text overlays, dialogue, or audio instructions.

Add fast, cinematic motion by giving the subject a clear action sequence, expressive facial expressions, strong gestures, and intentional camera movement. Keep the subject visible, centered, and clearly framed throughout. Add lighting only as natural scene behavior, such as flickering stage lights, passing sunlight, glowing streetlights, storm flashes, reflections, or shifting shadows, based on what best fits the user's prompt.

Output one polished paragraph using this structure:

The [Subject] in [setting/environment] during [time/weather]. The subject [dynamic performance action with expressive face, body movement, and strong gestures]. Their clothing/hair [reacts to movement, wind, or performance energy]. The lighting [changes or reacts naturally within the scene]. The camera [Camera Motion] while maintaining [subject visibility and framing]. The environment [reacts dynamically].

Each word in brackets should be chosen based on the user input and what best fits the scene.

Rules:
- This is text-to-video
- Never output square brackets or placeholder text. Replace every bracketed example with concrete scene details.
- Do not force singing. Follow user notes for singing, speaking, narration, instrumental, b-roll, or no-lip-sync behavior.
- Do not invent or quote lyric/dialogue text. Exact vocal/lyric directives are added by the UI after Gemma output.
- Do not add audio, dialogue, captions, text overlays, unrelated characters, new locations, major story changes, color grading, camera photo style, or static image-quality descriptions.
- Keep it vivid, fast, cinematic, dynamic, and video-ready
- Use one location inferred by the user's concept prompt. If one is not listed use one from the location list.
- Must use user input to help create the prompt
- User notes take priority. If user notes ask for no singing, silent b-roll, instrumental motion, no lip movement, or non-performance action, follow the user notes instead.
- Do not use orbit type camera motion, do not use the word "spin", and the subject should never spin.
- Do not mention source prompts, notes, lyrics, segments, JSON, or instructions.
- Do not include markdown, labels, quotes, or explanations."""


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
    legacy_srt_folder = os.path.join(repo_dir, "SRT_Files")
    audio_folder = os.path.join(folder_paths.get_output_directory(), "VRGDG_AudioFiles")
    srt_path = _newest_file(srt_folder, (".srt",)) or _newest_file(legacy_srt_folder, (".srt",))
    return {
        "audio_path": _newest_file(audio_folder, (".wav", ".mp3", ".flac", ".m4a", ".ogg")),
        "srt_path": srt_path,
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
        errors="replace",
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
    overlay_segments = _assign_overlay_scene_numbers(overlay_segments)
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
    scene_notes_path = _write_scene_notes_json(target, segments)
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


def _session_path(project_folder):
    return os.path.join(project_folder, "vrgdg_builder_session.json")


def _scene_notes_path(project_folder):
    return os.path.join(project_folder, "SceneNotes.json")


def _srt_path(project_folder):
    return os.path.join(project_folder, "builder_segments.srt")


def _images_folder(project_folder):
    return os.path.join(project_folder, "zimage_approved")


def _prompts_folder(project_folder):
    return os.path.join(project_folder, "prompts")


def _context_folder(project_folder):
    return os.path.join(project_folder, "project_context")


def _wizard_folder(project_folder):
    return os.path.join(project_folder, "wizard")


def _wizard_draft_path(project_folder):
    return os.path.join(_wizard_folder(project_folder), "wizard_draft.json")


def _wizard_lyrics_path(project_folder):
    return os.path.join(_wizard_folder(project_folder), "lyrics.txt")


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


def _unique_file_path(path):
    base = os.path.abspath(str(path or "").strip().strip('"'))
    folder = os.path.dirname(base)
    stem, ext = os.path.splitext(os.path.basename(base))
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(base):
        return base
    index = 2
    while True:
        candidate = os.path.join(folder, f"{stem}_{index:02d}{ext}")
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


def _convert_audio_to_wav(source_path, target_path):
    source = _resolve_existing_file(source_path, "Audio file")
    target = os.path.abspath(str(target_path or "").strip().strip('"'))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        target,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        error_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Could not convert audio to WAV: {error_text or f'ffmpeg exited with code {result.returncode}'}")
    if not os.path.isfile(target) or os.path.getsize(target) <= 0:
        raise ValueError("Audio conversion finished, but the WAV file was not created.")
    return target


def _copy_or_convert_project_audio(source_path, target_folder, target_name=None):
    source = _resolve_existing_file(source_path, "Audio file")
    name = target_name or os.path.basename(source)
    if os.path.splitext(source)[1].lower() == ".m4a":
        safe_stem = _safe_project_name(os.path.splitext(name)[0])
        return _convert_audio_to_wav(source, os.path.join(target_folder, f"{safe_stem}.wav"))
    return _copy_file_into_folder(source, target_folder, name)


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


def _open_local_file(path):
    target = os.path.abspath(str(path or "").strip().strip('"'))
    if not target or not os.path.isfile(target):
        raise ValueError("Video file was not found.")
    if os.name == "nt":
        os.startfile(target)  # pylint: disable=no-member
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])
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


def _extract_json_object_from_text(text):
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    candidates = [cleaned, _repair_builder_json_like_text(cleaned)]
    last_error = None
    for candidate in candidates:
        if not str(candidate or "").strip():
            continue
        try:
            return json.loads(candidate)
        except Exception as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError("Gemma did not return a JSON object.")


def _repair_builder_json_like_text(text):
    repaired = str(text or "").strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r"//.*?$", "", repaired, flags=re.MULTILINE)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(
        r'([{\[,]\s*)(locations|scene_map|name|description|id|label|prompt|runner|used_model)\s*:',
        r'\1"\2":',
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(r'([{\[,]\s*)(Scene\s*\d+|Scene\d+)\s*:', lambda m: f'{m.group(1)}"{m.group(2).replace(" ", "")}":', repaired, flags=re.IGNORECASE)
    repaired = re.sub(
        r'(^\s*)(locations|scene_map|name|description|id|label|prompt|runner|used_model)\s*:',
        r'\1"\2":',
        repaired,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    repaired = re.sub(r'(^\s*)(Scene\s*\d+|Scene\d+)\s*:', lambda m: f'{m.group(1)}"{m.group(2).replace(" ", "")}":', repaired, flags=re.IGNORECASE | re.MULTILINE)
    # Gemma sometimes omits commas between array objects or object properties.
    repaired = re.sub(r'(["}\]])\s*\n\s*(")', r'\1,\n\2', repaired)
    repaired = re.sub(r'(})\s*\n\s*({)', r'\1,\n\2', repaired)
    repaired = re.sub(r'(})\s*({)', r'\1,\2', repaired)
    repaired = re.sub(r'(\])\s*("scene_map"\s*:)', r'\1,\2', repaired, flags=re.IGNORECASE)
    return repaired


def _parse_flux_location_map_fallback(text, cleaned_scenes, existing_locations=None):
    cleaned = _clean_visual_gemma_text(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]

    locations = []
    seen_names = set()
    location_block_match = re.search(
        r'"?locations"?\s*:\s*\[(.*?)]\s*,?\s*"?scene_map"?\s*:',
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    location_block = location_block_match.group(1) if location_block_match else ""
    for object_text in re.findall(r"\{(.*?)\}", location_block, flags=re.DOTALL):
        name_match = re.search(r'"?name"?\s*:\s*"([^"]+)"', object_text, flags=re.IGNORECASE | re.DOTALL)
        description_match = re.search(r'"?description"?\s*:\s*"([^"]*)"', object_text, flags=re.IGNORECASE | re.DOTALL)
        name = re.sub(r"\s+", " ", (name_match.group(1) if name_match else "").strip())
        description = re.sub(r"\s+", " ", (description_match.group(1) if description_match else "").strip())
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        locations.append({"name": name, "description": description})

    if not locations and isinstance(existing_locations, list):
        for item in existing_locations:
            if not isinstance(item, dict):
                continue
            name = re.sub(r"\s+", " ", str(item.get("name", "") or "").strip())
            description = re.sub(r"\s+", " ", str(item.get("description", "") or "").strip())
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            locations.append({"name": name, "description": description})

    scene_lookup = {}
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_lookup[str(scene["id"]).strip().lower()] = scene["id"]
        scene_lookup[str(scene["label"]).strip().lower()] = scene["id"]
        scene_lookup[f"scene {index}"] = scene["id"]
        scene_lookup[f"scene{index}"] = scene["id"]
        scene_lookup[str(index)] = scene["id"]

    scene_map = {}
    scene_block_match = re.search(r'"?scene_map"?\s*:\s*\{(.*?)\}\s*$', cleaned, flags=re.IGNORECASE | re.DOTALL)
    scene_block = scene_block_match.group(1) if scene_block_match else ""
    for raw_key, raw_location in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', scene_block, flags=re.DOTALL):
        lookup_key = re.sub(r"\s+", " ", raw_key.strip().lower())
        scene_id = scene_lookup.get(lookup_key) or scene_lookup.get(lookup_key.replace(" ", ""))
        location_name = re.sub(r"\s+", " ", raw_location.strip())
        if scene_id and location_name:
            scene_map[scene_id] = location_name

    if locations:
        if not scene_map:
            scene_map = _fallback_location_map_by_overlap(cleaned_scenes, locations)
        else:
            valid_location_names = {item["name"].lower(): item["name"] for item in locations}
            for scene in cleaned_scenes:
                raw_name = re.sub(r"\s+", " ", str(scene_map.get(scene["id"], "") or "").strip())
                if raw_name.lower() not in valid_location_names:
                    scene_map[scene["id"]] = _best_location_for_scene(scene, locations)["name"]
        return {"locations": locations, "scene_map": scene_map}

    raise ValueError("Gemma location map could not be parsed as JSON or recovered from text.")


def _fallback_location_map_by_overlap(cleaned_scenes, locations):
    mapped = {}
    for scene in cleaned_scenes:
        mapped[scene["id"]] = _best_location_for_scene(scene, locations)["name"]
    return mapped


def _best_location_for_scene(scene, locations):
    if not locations:
        return {"name": "Location 1", "description": ""}
    scene_text = f"{scene.get('concept', '')} {scene.get('notes', '')}"
    best_location = locations[0]
    best_score = -1
    for location in locations:
        score = _location_text_overlap_score(
            scene_text,
            f"{location.get('name', '')} {location.get('description', '')}",
        )
        if score > best_score:
            best_location = location
            best_score = score
    return best_location


def _canonical_location_name(name, locations):
    raw = re.sub(r"\s+", " ", str(name or "").strip()).lower()
    for location in locations or []:
        loc_name = re.sub(r"\s+", " ", str(location.get("name", "") or "").strip())
        if loc_name.lower() == raw:
            return loc_name
    return ""


def _location_usage_counts_from_payload(payload, locations):
    names = [re.sub(r"\s+", " ", str(item.get("name", "") or "").strip()) for item in locations or []]
    counts = {name: 0 for name in names if name}
    raw_counts = payload.get("used_location_counts")
    if isinstance(raw_counts, dict):
        for raw_name, raw_count in raw_counts.items():
            name = _canonical_location_name(raw_name, locations)
            if name:
                try:
                    counts[name] = max(0, int(raw_count or 0))
                except Exception:
                    counts[name] = counts.get(name, 0)
    raw_assignments = payload.get("previous_assignments")
    if isinstance(raw_assignments, list):
        for item in raw_assignments:
            if isinstance(item, dict):
                name = _canonical_location_name(item.get("location") or item.get("location_name"), locations)
            else:
                name = _canonical_location_name(item, locations)
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _balance_location_map_by_usage(scene_map, cleaned_scenes, locations, previous_counts=None):
    if not scene_map or not cleaned_scenes or not locations:
        return scene_map
    location_by_name = {}
    for item in locations:
        name = re.sub(r"\s+", " ", str(item.get("name", "") or "").strip())
        if name:
            location_by_name[name] = item
    location_names = list(location_by_name.keys())
    if len(location_names) <= 1:
        return scene_map
    balanced = {}
    fallback = _fallback_location_map_by_overlap(cleaned_scenes, locations)
    for scene in cleaned_scenes:
        name = _canonical_location_name(scene_map.get(scene["id"], ""), locations) or fallback.get(scene["id"], "")
        balanced[scene["id"]] = name

    previous_counts = previous_counts or {}
    current_counts = {name: 0 for name in location_names}
    for name in balanced.values():
        if name in current_counts:
            current_counts[name] += 1

    target_count = min(len(cleaned_scenes), len(location_names))
    desired_locations = sorted(
        location_names,
        key=lambda name: (int(previous_counts.get(name, 0) or 0), current_counts.get(name, 0), location_names.index(name)),
    )[:target_count]

    for desired_name in desired_locations:
        if current_counts.get(desired_name, 0) > 0:
            continue
        desired_location = location_by_name.get(desired_name) or {"name": desired_name, "description": ""}
        best_scene = None
        best_score = None
        for scene in cleaned_scenes:
            current_name = balanced.get(scene["id"], "")
            if current_name == desired_name:
                continue
            if current_counts.get(current_name, 0) <= 1 and any(current_counts.get(name, 0) == 0 for name in desired_locations if name != desired_name):
                continue
            scene_text = f"{scene.get('concept', '')} {scene.get('notes', '')}"
            desired_score = _location_text_overlap_score(scene_text, f"{desired_location.get('name', '')} {desired_location.get('description', '')}")
            current_location = location_by_name.get(current_name, {"name": current_name, "description": ""})
            current_score = _location_text_overlap_score(scene_text, f"{current_location.get('name', '')} {current_location.get('description', '')}")
            repeat_penalty = current_counts.get(current_name, 0) + int(previous_counts.get(current_name, 0) or 0)
            score = (desired_score - current_score) + repeat_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_scene = scene
        if best_scene:
            old_name = balanced.get(best_scene["id"], "")
            if old_name in current_counts:
                current_counts[old_name] = max(0, current_counts[old_name] - 1)
            balanced[best_scene["id"]] = desired_name
            current_counts[desired_name] = current_counts.get(desired_name, 0) + 1
    return balanced


def _location_text_overlap_score(scene_text, location_text):
    stop_words = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into", "is", "it",
        "of", "on", "or", "the", "to", "with", "scene", "shot", "cinematic", "woman", "man",
        "girl", "boy", "subject", "character", "wearing", "light", "lighting",
    }
    scene_tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(scene_text or "").lower())
        if len(token) > 2 and token not in stop_words
    }
    location_tokens = [
        token for token in re.findall(r"[a-z0-9]+", str(location_text or "").lower())
        if len(token) > 2 and token not in stop_words
    ]
    if not scene_tokens or not location_tokens:
        return 0
    score = 0
    for token in location_tokens:
        if token in scene_tokens:
            score += 3
        elif any(scene_token.startswith(token) or token.startswith(scene_token) for scene_token in scene_tokens if len(scene_token) > 4):
            score += 1
    return score


def _parse_location_lines(text):
    locations = []
    seen_names = set()
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().strip("-").strip()
        if not line or line in {"{", "}", "[", "]"}:
            continue
        match = re.match(r"^\s*(?:Location\s*)?(\d+)\s*(?:[|:=\).-])\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if not match:
            continue
        rest = match.group(2).strip().strip('"').rstrip(",")
        parts = [part.strip().strip('"') for part in rest.split("|")]
        if len(parts) >= 2:
            name = parts[0]
            description = " | ".join(parts[1:])
        else:
            name = rest
            description = rest
        name = re.sub(r"^\s*name\s*[:=]\s*", "", name, flags=re.IGNORECASE)
        description = re.sub(r"^\s*description\s*[:=]\s*", "", description, flags=re.IGNORECASE)
        raw_name = name
        name, description = _clean_location_card(name, description, raw_name)
        if not name or _looks_like_location_meta_text(name) or _looks_like_location_meta_text(description) or not _valid_location_card(name, description) or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        locations.append({"name": name, "description": description})
    return locations


def _looks_like_location_meta_text(value):
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    if not text:
        return True
    if len(text) > 140 and not re.search(r"\b(?:room|hall|hallway|corridor|street|road|forest|temple|pool|motel|stage|club|warehouse|desert|beach|shore|city|rooftop|alley|kitchen|bedroom|bathroom|church|chapel|station|train|car|bus|field|garden|vault|cave|lake|river|bridge|tunnel|apartment|house|mansion|hotel|bar|lounge|studio|parking|garage)\b", text):
        return True
    meta_patterns = (
        r"\bsince the provided\b",
        r"\bprovided .*content was not visible\b",
        r"\bprovided .*prompt\b",
        r"\bsubjectsandscenes\.txt\b",
        r"\bhere is (?:a|the)\b",
        r"\breusable location list\b",
        r"\bbased on the structural context\b",
        r"\bscene descriptions imply\b",
        r"\bcohesive project\b",
        r"\bi can(?:not|'t)\b",
        r"\bi(?:'|’)m sorry\b",
        r"\bas an ai\b",
        r"\boutput format\b",
        r"\buser input\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in meta_patterns)


_LOCATION_PLACE_WORDS = {
    "alley", "apartment", "arena", "attic", "ballroom", "bar", "barn", "bathroom", "beach", "bedroom",
    "bridge", "building", "cabin", "cafe", "casino", "cathedral", "cave", "chapel", "church", "city",
    "club", "corridor", "courtyard", "desert", "diner", "dock", "factory", "field", "forest", "foyer",
    "garage", "garden", "greenhouse", "hall", "hallway", "harbor", "hotel", "house", "kitchen", "lab",
    "lake", "lounge", "mansion", "market", "motel", "museum", "office", "palace", "parking", "pier",
    "pool", "quarry", "railway", "road", "rooftop", "room", "school", "set", "shore", "stage", "station",
    "street", "studio", "subway", "temple", "theater", "tower", "train", "tunnel", "vault", "warehouse",
    "workshop",
}

_NON_LOCATION_OBJECT_WORDS = {
    "bottle", "bowl", "bracelet", "camera", "candle", "chair", "collar", "comb", "crown", "dress",
    "frame", "glass", "kit", "knife", "locket", "mirror", "necklace", "perfume", "phone", "pocket",
    "razor", "rice", "ring", "shaving", "shoe", "sink", "sugar", "table", "tooth", "vanity", "veil", "window",
}

_LOCATION_SURFACE_WORDS = {
    "ceiling", "corner", "floor", "partition", "partitions", "wall", "walls",
}

_CHARACTER_WORDS = {
    "bride", "character", "face", "ghost", "girl", "hand", "human", "man", "person", "silhouette",
    "subject", "woman",
}


def _location_place_pattern():
    return r"(?:%s)" % "|".join(sorted((re.escape(word) for word in _LOCATION_PLACE_WORDS), key=len, reverse=True))


def _nice_location_name(value):
    text = re.sub(r"\s+", " ", str(value or "").strip(" ,.-"))
    if not text:
        return ""
    return text[:1].upper() + text[1:]


def _location_text_has_non_place_focus(value):
    tokens = set(re.findall(r"[a-z0-9]+", str(value or "").lower()))
    return bool(tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS))


def _normalize_location_name(name):
    text = re.sub(r"\s+", " ", str(name or "").strip(" ,.-"))
    if not text:
        return ""
    lowered = text.lower()
    place_re = _location_place_pattern()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))

    if re.search(r"\b(?:a|an|the|with|without|under|beneath|behind|beside|inside|outside|near|through|over|heavy|long|thick|deep|wide|large|small|empty|covered)\s*$", lowered):
        place_match = re.search(rf"^(.{{3,80}}?\b{place_re}\b)", text, flags=re.IGNORECASE)
        if place_match:
            return _nice_location_name(place_match.group(1))

    if tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS):
        prep_match = re.search(
            rf"\b(?:in|inside|within|at|on|near|beside|behind|beneath|under)\s+(?:a|an|the)?\s*([^,.;]*?\b{place_re}\b(?:\s+\w+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if prep_match:
            return _nice_location_name(prep_match.group(1))

        surface_match = re.search(
            rf"^(.{{3,80}}?\b{place_re}\b)\s+(?:{'|'.join(sorted(_LOCATION_SURFACE_WORDS | _NON_LOCATION_OBJECT_WORDS, key=len, reverse=True))})\b",
            text,
            flags=re.IGNORECASE,
        )
        if surface_match:
            return _nice_location_name(surface_match.group(1))

        trailing_object_match = re.search(
            rf"\b({place_re})\s+(?:{'|'.join(sorted(_LOCATION_SURFACE_WORDS | _NON_LOCATION_OBJECT_WORDS, key=len, reverse=True))})\b",
            text,
            flags=re.IGNORECASE,
        )
        if trailing_object_match:
            before = text[:trailing_object_match.end(1)].strip()
            return _nice_location_name(before)

    return _nice_location_name(text)


def _clean_location_card(name, description, raw_name=None):
    normalized_name = _normalize_location_name(name)
    cleaned_description = _clean_location_description(normalized_name, description)
    raw_text = str(raw_name if raw_name is not None else name or "")
    if normalized_name.lower() != re.sub(r"\s+", " ", raw_text.strip(" ,.-")).lower() and _location_text_has_non_place_focus(raw_text):
        cleaned_description = normalized_name
    return normalized_name, cleaned_description


def _location_name_fragment_is_complete(fragment):
    text = re.sub(r"\s+", " ", str(fragment or "").strip().lower())
    if not text:
        return False
    if re.search(r"\b(?:a|an|the|with|without|under|beneath|behind|beside|inside|outside|near|through|over|heavy|long|thick|deep|wide|large|small|empty|covered)\s*$", text):
        return False
    return bool(re.search(rf"\b{_location_place_pattern()}\b", text, flags=re.IGNORECASE))


def _location_name_is_place(name):
    text = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if not text:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", text))
    if tokens & _LOCATION_PLACE_WORDS:
        if tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS):
            normalized = _normalize_location_name(name).lower()
            normalized_tokens = set(re.findall(r"[a-z0-9]+", normalized))
            if normalized_tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS):
                return False
        return True
    if tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS):
        return False
    if re.search(r"\b(?:inside|outside|interior|exterior|room|space|area|zone)\b", text):
        return True
    return False


def _clean_location_description(name, description):
    name_text = re.sub(r"\s+", " ", str(name or "").strip())
    desc = re.sub(r"\s+", " ", str(description or "").strip())
    if not desc:
        return name_text
    fragments = [part.strip(" .") for part in re.split(r"\s*,\s*", desc) if part.strip(" .")]
    kept = []
    for index, fragment in enumerate(fragments):
        fragment_lower = fragment.lower()
        if index > 0 and re.search(
            r"\b(?:bride|woman|man|girl|boy|face|hand|human|silhouette|wearing|watching|pressed|resting|trailing|caught|hidden|dissolving|reflecting|smelling)\b",
            fragment_lower,
        ):
            continue
        if index > 0 and re.search(r"\b(?:razor|tooth|dress|veil|bottle|locket|collar|frame|kit|rice|sugar bowl)\b", fragment_lower):
            continue
        kept.append(fragment)
    cleaned = ", ".join(kept).strip()
    if not cleaned:
        cleaned = name_text
    if name_text and not cleaned.lower().startswith(name_text.lower()):
        cleaned = f"{name_text}, {cleaned}"
    return cleaned


def _valid_location_card(name, description):
    if not _location_name_is_place(name):
        return False
    combined = f"{name} {description}".lower()
    if re.search(r"\b(?:character|subject|woman|man|girl|boy|bride|ghostly face|human tooth|hand pressed)\b", combined):
        desc_without_name = str(description or "").lower().replace(str(name or "").lower(), "")
        if not re.search(r"\b(?:room|hall|hallway|corridor|street|road|forest|garden|stage|kitchen|bathroom|bedroom|ballroom|pool|motel|warehouse|studio|city)\b", desc_without_name):
            return False
    return True


def _clean_location_context_text(value):
    text = str(value or "").strip()
    if not text:
        return ""
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.search(r"\bsubjectsandscenes\.txt\b", line, flags=re.IGNORECASE):
            continue
        if re.search(r"\.(?:txt|json|srt)\b", line, flags=re.IGNORECASE) and re.search(r"[A-Za-z]:\\|/|\\", line):
            continue
        if _looks_like_location_meta_text(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if _looks_like_location_meta_text(cleaned):
        return ""
    return cleaned


def _parse_location_idea_lines(text):
    locations = []
    seen_names = set()
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\s*music\s+video\s+locations\s*:?\s*$", line, flags=re.IGNORECASE):
            continue
        line = re.sub(r"^\s*(?:[-*\u2022]+|\d+[\).:-])\s*", "", line).strip()
        line = line.strip('"').strip()
        if not line or line in {"{", "}", "[", "]"} or line.startswith("{") or line.startswith("["):
            continue
        if re.match(r"^(?:location ideas?|output|user input)\s*:?\s*$", line, flags=re.IGNORECASE):
            continue
        name = line
        description = line
        split_match = re.match(r"^(.{4,80}?)(?:\s+-\s+|\s+--\s+|:\s+)(.{4,})$", line)
        if split_match:
            name = split_match.group(1).strip()
            description = split_match.group(2).strip()
        else:
            comma_parts = line.split(",", 1)
            if len(comma_parts) == 2 and 4 <= len(comma_parts[0].strip()) <= 60 and _location_name_fragment_is_complete(comma_parts[0]):
                name = comma_parts[0].strip()
        raw_name = name
        name, description = _clean_location_card(name, description, raw_name)
        if _looks_like_location_meta_text(name) or _looks_like_location_meta_text(description) or not _valid_location_card(name, description) or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        locations.append({"name": name, "description": description})
    return locations


def _parse_location_ideas_flexible(text):
    locations = _parse_location_idea_lines(text)
    if locations:
        return locations
    cleaned = _clean_visual_gemma_text(text)
    try:
        data = _extract_json_object_from_text(cleaned)
        if isinstance(data, dict):
            raw_locations = data.get("locations") or data.get("location_ideas") or data.get("music_video_locations")
        else:
            raw_locations = data
        if isinstance(raw_locations, list):
            parsed = []
            for item in raw_locations:
                if isinstance(item, dict):
                    raw_name = item.get("name") or item.get("location") or item.get("idea") or ""
                    name = _normalize_location_name(raw_name)
                    description = re.sub(r"\s+", " ", str(item.get("description") or item.get("detail") or item.get("visual_detail") or name).strip())
                else:
                    raw_name = item
                    name = _normalize_location_name(raw_name)
                    description = name
                name, description = _clean_location_card(name, description or name, raw_name)
                if name and not _looks_like_location_meta_text(name) and not _looks_like_location_meta_text(description) and _valid_location_card(name, description):
                    parsed.append({"name": name, "description": description or name})
            if parsed:
                return parsed
    except Exception:
        pass
    bulletish = re.split(r"(?:\n+|(?<=\.)\s+(?=[A-Z][A-Za-z ]{4,70}(?:\s+-|:)))", cleaned)
    parsed = _parse_location_idea_lines("\n".join(part.strip() for part in bulletish if part.strip()))
    if parsed:
        return parsed
    candidates = []
    for part in re.split(r";|\n", cleaned):
        part = re.sub(r"^\s*(?:Music Video Locations|Locations|Location ideas)\s*:?\s*", "", part.strip(), flags=re.I)
        if 4 <= len(part) <= 180 and not _looks_like_location_meta_text(part) and not re.search(r"\b(?:sorry|cannot|unable|lyrics|song meaning|summary)\b", part, flags=re.I):
            candidates.append(part)
    return _parse_location_idea_lines("\n".join(f"- {item}" for item in candidates))


def _parse_subject_lines(text):
    subjects = []
    for raw_line in str(text or "").splitlines():
      line = raw_line.strip().strip("-* ")
      if not line:
          continue
      match = re.match(r"^\s*(?:\d+[\).:\-|]\s*)?([^|:]+?)\s*\|\s*(.+)$", line)
      if match:
          name = re.sub(r"\s+", " ", match.group(1).strip())
          description = re.sub(r"\s+", " ", match.group(2).strip())
      else:
          match = re.match(r"^\s*(?:\d+[\).:\-|]\s*)?([^:]+?)\s*:\s*(.+)$", line)
          if not match:
              continue
          name = re.sub(r"\s+", " ", match.group(1).strip())
          description = re.sub(r"\s+", " ", match.group(2).strip())
      if name and description:
          subjects.append({"name": name, "description": description})
    return subjects


def _parse_scene_location_number_map(text, cleaned_scenes, locations):
    scene_map = {}
    scene_by_number = {str(index): scene["id"] for index, scene in enumerate(cleaned_scenes, start=1)}
    scene_by_label = {}
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_by_label[scene["id"].lower()] = scene["id"]
        scene_by_label[scene["label"].lower()] = scene["id"]
        scene_by_label[f"scene {index}"] = scene["id"]
        scene_by_label[f"scene{index}"] = scene["id"]
    location_by_number = {str(index): item["name"] for index, item in enumerate(locations, start=1)}
    location_by_name = {item["name"].lower(): item["name"] for item in locations}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        match = re.match(
            r'^\s*(?:Scene\s*)?("?[^"=:\|]+"?)\s*(?:=|:|\|)\s*(?:Location\s*)?("?[^",\s]+"?)',
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        raw_scene = match.group(1).strip().strip('"').lower()
        raw_location = match.group(2).strip().strip('"').lower()
        scene_id = scene_by_number.get(raw_scene) or scene_by_label.get(raw_scene) or scene_by_label.get(raw_scene.replace(" ", ""))
        location_name = location_by_number.get(raw_location) or location_by_name.get(raw_location)
        if scene_id and location_name:
            scene_map[scene_id] = location_name
    return scene_map


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


def _read_audio_peaks_with_ffmpeg(audio_path, target_peaks=1600):
    sample_rate = 16000
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        error_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(error_text or f"ffmpeg exited with code {result.returncode}")
    samples = array.array("f")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    total_samples = len(samples)
    duration = total_samples / float(sample_rate)
    if total_samples <= 0:
        return {"duration": 0, "sample_rate": sample_rate, "channels": 1, "peaks": []}
    peak_count = max(1, min(int(target_peaks or 1600), total_samples))
    samples_per_peak = max(1, math.ceil(total_samples / peak_count))
    peaks = []
    for start in range(0, total_samples, samples_per_peak):
        chunk = samples[start: start + samples_per_peak]
        if not chunk:
            continue
        total = 0.0
        for value in chunk:
            total += float(value) * float(value)
        peaks.append(min(1.0, math.sqrt(total / len(chunk))))
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": 1,
        "peaks": peaks,
    }


def _read_audio_peaks(audio_path, target_peaks=1600):
    try:
        return _read_audio_peaks_with_torchaudio(audio_path, target_peaks)
    except Exception as torch_exc:
        try:
            return _read_audio_peaks_with_wave(audio_path, target_peaks)
        except Exception as wave_exc:
            try:
                return _read_audio_peaks_with_ffmpeg(audio_path, target_peaks)
            except Exception as ffmpeg_exc:
                raise ValueError(
                    "Could not read audio for waveform. Try a standard WAV, MP3, FLAC, or M4A file. "
                    f"torchaudio error: {torch_exc}; wav fallback error: {wave_exc}; ffmpeg fallback error: {ffmpeg_exc}"
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


def _looks_like_unfilled_prompt_template(text):
    sample = str(text or "").strip()
    if not sample:
        return False
    bracketed = re.findall(r"\[([^\]]{2,80})\]", sample)
    if len(bracketed) >= 2:
        return True
    placeholder_terms = (
        "subject",
        "setting",
        "environment",
        "time",
        "weather",
        "camera motion",
        "dynamic performance",
        "subject visibility",
        "framing",
        "reacts dynamically",
        "clothing",
        "hair",
    )
    lowered = sample.lower()
    if any(f"[{term}]" in lowered for term in placeholder_terms):
        return True
    if re.search(r"\[[^\]]*(?:subject|setting|environment|camera|motion|weather|lighting|dynamic|framing)[^\]]*\]", lowered):
        return True
    return False


def _looks_like_bad_reference_description(text):
    sample = re.sub(r"\s+", " ", str(text or "").strip())
    if not sample:
        return True
    if _looks_like_gemma_repeat_failure(sample) or _looks_like_unfilled_prompt_template(sample):
        return True
    tokens = re.findall(r"[\w']+", sample.lower(), flags=re.UNICODE)
    tokens = [token.strip("_'") for token in tokens if token.strip("_'")]
    if not tokens:
        return True
    if len(tokens) < 6 and len(set(tokens)) <= 2:
        return True
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if counts:
        max_count = max(counts.values())
        if max_count >= 3 and max_count / float(len(tokens)) >= 0.45:
            return True
    for size in (1, 2, 3):
        if len(tokens) < size * 3:
            continue
        for index in range(0, len(tokens) - (size * 3) + 1):
            chunk = tokens[index:index + size]
            if all(tokens[index + offset:index + offset + size] == chunk for offset in range(size, size * 3, size)):
                return True
    alpha_chars = re.findall(r"[a-zA-Z]", sample)
    if len(alpha_chars) < 18:
        return True
    return False


def _validate_reference_description(text, label):
    if _looks_like_bad_reference_description(text):
        raise ValueError(
            f"Gemma returned unusable repeated text for the {label} description. "
            "Try again, or use a clearer reference image."
        )


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
    if _looks_like_unfilled_prompt_template(text):
        raise ValueError(
            f"Gemma returned an unfilled template for the {label} prompt. "
            "Try again or add more specific scene/motion notes."
        )


def _llm_runner_from_payload(payload):
    runner = str(payload.get("text_runner") or payload.get("gemma_runner") or "builtin").strip().lower()
    if runner in {"lmstudio", "lm-studio", "lm_studio"}:
        runner = "lm_studio"
    if runner not in {"builtin", "lm_studio"}:
        runner = "builtin"
    return runner


def _resolve_mmproj_dropdown_path(llm, mmproj_file):
    selected = str(mmproj_file or "").strip()
    try:
        if selected:
            return llm._resolve_dropdown_path(selected, llm.MISSING_MMPROJ_OPTION)
    except Exception:
        pass
    try:
        choices = [
            choice for choice in llm._list_local_mmproj_choices()
            if choice and choice != llm.MISSING_MMPROJ_OPTION
        ]
    except Exception:
        choices = []
    if len(choices) == 1:
        return llm._resolve_dropdown_path(choices[0], llm.MISSING_MMPROJ_OPTION)
    if selected:
        return llm._resolve_dropdown_path(selected, llm.MISSING_MMPROJ_OPTION)
    raise ValueError("Choose an mmproj file for the vision model.")


def _run_lm_studio_text(payload, instruction_text, temperature=0.6, top_p=0.95, max_new_tokens=1200):
    base_url = str(payload.get("lmstudio_base_url") or _LM_STUDIO_DEFAULT_BASE_URL).strip().rstrip("/")
    model = str(payload.get("lmstudio_model") or payload.get("model_file") or "").strip()
    api_key = str(payload.get("lmstudio_api_key") or "").strip()
    if not base_url:
        raise ValueError("LM Studio base URL is empty.")
    if not model:
        raise ValueError("Enter the LM Studio model name shown in LM Studio.")
    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": instruction_text}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_new_tokens),
        "stream": False,
    }
    if payload.get("seed") is not None:
        body["seed"] = int(payload.get("seed"))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(payload.get("lmstudio_timeout") or 180)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"LM Studio request failed ({exc.code}): {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to LM Studio at {base_url}. Make sure LM Studio's local server is running.") from exc
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise ValueError("LM Studio returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    text = message.get("content") if isinstance(message, dict) else ""
    text = str(text or "").strip()
    if not text:
        raise ValueError("LM Studio returned empty text.")
    return text


def _pil_image_to_data_url(image, max_height=512, quality=88):
    if image is None:
        raise ValueError("LM Studio vision image is missing.")
    image = image.convert("RGB")
    if image.height > max_height:
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        width = max(1, int(image.width * (float(max_height) / max(1, image.height))))
        image = image.resize((width, int(max_height)), resample)
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _run_lm_studio_vision(payload, instruction_text, pil_images, temperature=0.25, top_p=0.95, max_new_tokens=1200):
    base_url = str(payload.get("lmstudio_base_url") or _LM_STUDIO_DEFAULT_BASE_URL).strip().rstrip("/")
    model = str(payload.get("lmstudio_model") or payload.get("model_file") or "").strip()
    api_key = str(payload.get("lmstudio_api_key") or "").strip()
    if not base_url:
        raise ValueError("LM Studio base URL is empty.")
    if not model:
        raise ValueError("Enter the LM Studio vision model name shown in LM Studio.")
    images = list(pil_images or [])
    if not images:
        raise ValueError("LM Studio vision needs at least one image.")
    content = [{"type": "text", "text": instruction_text}]
    for image in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": _pil_image_to_data_url(image)},
        })
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_new_tokens),
        "stream": False,
    }
    if payload.get("seed") is not None:
        body["seed"] = int(payload.get("seed"))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(payload.get("lmstudio_timeout") or 300)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"LM Studio vision request failed ({exc.code}): {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to LM Studio at {base_url}. Make sure LM Studio's local server is running.") from exc
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise ValueError("LM Studio vision returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    text = message.get("content") if isinstance(message, dict) else ""
    text = str(text or "").strip()
    if not text:
        raise ValueError("LM Studio vision returned empty text.")
    return text


def _list_lm_studio_models(payload):
    base_url = str(payload.get("lmstudio_base_url") or _LM_STUDIO_DEFAULT_BASE_URL).strip().rstrip("/")
    api_key = str(payload.get("lmstudio_api_key") or "").strip()
    if not base_url:
        raise ValueError("LM Studio base URL is empty.")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url}/models", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=float(payload.get("lmstudio_timeout") or 30)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"LM Studio models request failed ({exc.code}): {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to LM Studio at {base_url}. Make sure LM Studio's local server is running.") from exc
    items = data.get("data") if isinstance(data, dict) else []
    models = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                model_id = str(item.get("id") or "").strip()
                if model_id:
                    models.append(model_id)
    return {"models": models, "raw": data}


def _clean_lm_studio_plain_text(text):
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^\s*```(?:text|json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    cleaned = re.sub(r"^(?:Assistant|Answer|Final answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _run_builder_text_llm(payload, instruction_text, temperature=0.6, top_p=0.95, max_new_tokens=1200, label="Gemma", preserve_paragraphs=False):
    if _llm_runner_from_payload(payload) == "lm_studio":
        text = _run_lm_studio_text(
            payload,
            instruction_text,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        cleaned = _clean_lm_studio_plain_text(text) if preserve_paragraphs else _clean_visual_gemma_text(text)
        return cleaned, {
            "runner": "lm_studio",
            "used_model": str(payload.get("lmstudio_model") or "").strip(),
            "unloaded": False,
        }

    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file:
        raise ValueError(f"Choose a {label} model first.")
    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    unload_after = bool(payload.get("unload_after", True))
    seed = payload.get("seed")
    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
        )
        text = llm._run_gguf_text_pipeline(
            model=model,
            instruction_text=instruction_text,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            seed=int(seed) if seed is not None else None,
        )
        cleaned = _clean_lm_studio_plain_text(text) if preserve_paragraphs else _clean_visual_gemma_text(text)
        return cleaned, {
            "runner": "builtin",
            "used_model": model_path,
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
                mmproj_path="",
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _repair_builder_gemma_prompt(payload, text, label):
    original = str(text or "").strip()
    needs_repair = _looks_like_gemma_repeat_failure(original) or _looks_like_unfilled_prompt_template(original)
    if not original or not needs_repair:
        return original

    repair_payload = dict(payload or {})
    repair_model = str(
        repair_payload.get("repair_model_file")
        or repair_payload.get("text_model_file")
        or repair_payload.get("model_file")
        or ""
    ).strip()
    if repair_model:
        repair_payload["model_file"] = repair_model
    repair_payload["mmproj_file"] = ""
    repair_payload["use_vision"] = False
    broken = original[:5000]
    label_text = str(label or "").strip()
    video_repair = label_text.lower() in {"i2v", "t2v"}
    if video_repair:
        concept_prompt = str(repair_payload.get("t2i_prompt") or "").strip()[:3000]
        motion_notes = str(repair_payload.get("user_notes") or "").strip()[:2000]
        instruction = (
            f"Clean this broken {label_text} video prompt into one usable final video prompt.\n\n"
            "The broken text may contain internal thoughts, repeated tokens, markdown, or unfilled square-bracket placeholders.\n"
            "Use only the concept prompt and motion notes below as context. Do not use project story, lyrics, agent chat, or any other context.\n"
            "Replace placeholders like [Subject], [setting/environment], [time/weather], and [Camera Motion] with concrete details from the concept prompt and motion notes.\n"
            "Do not continue the broken text. Do not explain the repair. Do not mention that it was repaired.\n"
            "Return exactly one normal video prompt paragraph with no square brackets, labels, markdown, or placeholders.\n"
            "Keep it under 120 words.\n\n"
            f"Concept/T2I prompt:\n{concept_prompt or '[none provided]'}\n\n"
            f"I2V/T2V motion notes:\n{motion_notes or '[none provided]'}\n\n"
            f"Broken video prompt:\n{broken}"
        )
    else:
        instruction = (
            f"Clean this broken {label_text} image prompt into one usable final prompt.\n\n"
            "The broken text may contain internal thoughts, analysis, channel tags, repeated tokens, markdown, unfilled square-bracket placeholders, or junk.\n"
            "Do not continue the broken text. Do not explain the repair. Do not mention that it was repaired.\n"
            "Return exactly one normal image prompt paragraph.\n"
            "Remove all thought, analysis, channel, role, markdown, labels, square brackets, placeholder words, and repeated junk.\n"
            "Replace placeholders like [Subject], [setting/environment], [time/weather], and [Camera Motion] with concrete details inferred from the broken text and user notes.\n"
            "Keep only usable visual image-generation content. If usable details are scarce, create a concise cinematic prompt from the usable fragments.\n"
            "Keep it under 120 words.\n\n"
            f"Broken text:\n{broken}"
        )
    try:
        repaired, _run_info = _run_builder_text_llm(
            repair_payload,
            instruction,
            temperature=0.25,
            top_p=0.85,
            max_new_tokens=350,
            label=f"{label_text} repair Gemma",
        )
        repaired = _clean_visual_gemma_text(repaired)
        if repaired and not _looks_like_gemma_repeat_failure(repaired) and not _looks_like_unfilled_prompt_template(repaired):
            return repaired
    except Exception:
        pass
    return original

def _repair_and_validate_builder_gemma_prompt(payload, text, label):
    repaired = _repair_builder_gemma_prompt(payload, text, label)
    _validate_builder_gemma_prompt(repaired, label)
    return repaired


def _generate_builder_agent_reply(payload):
    context = payload.get("context") or {}
    if not isinstance(context, dict):
        context = {}
    auto_apply = bool(payload.get("auto_apply"))
    agent_purpose = str(payload.get("agent_purpose") or "scene_work").strip().lower()
    if agent_purpose not in {"walkthrough", "scene_work", "story_builder", "troubleshoot"}:
        agent_purpose = "scene_work"
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    cleaned_messages = []
    for item in messages[-10:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            cleaned_messages.append({"role": role, "content": content[:3000]})
    latest_user = str(payload.get("message") or "").strip()
    if latest_user:
        cleaned_messages.append({"role": "user", "content": latest_user[:3000]})
    if not cleaned_messages:
        raise ValueError("Type a message for the Builder Agent first.")

    context_text = json.dumps(context, ensure_ascii=False, indent=2)[:9000]
    conversation_text = "\n\n".join(
        f"{item['role'].title()}:\n{item['content']}"
        for item in cleaned_messages
    )
    instruction = (
        "You are the VRGDG Music Video Builder Agent, a local assistant inside a scene-by-scene music video workflow.\n"
        "Help the user think through lyrics, scene concepts, image prompts, video prompts, shot choices, continuity, and troubleshooting.\n"
        "Be practical, concise, and specific to the provided active scene context.\n"
        "Keep the user-facing reply short: usually 1-5 sentences. Do not write strategy essays, option menus, or long explanations unless the user asks.\n"
        "When suggesting prompt text, provide copy-ready text clearly, but keep normal chat friendly.\n"
        "If the context is missing something, say what is missing and make a reasonable suggestion from what is available.\n\n"
        f"Agent purpose: {agent_purpose}.\n"
        "Read project_status and scene_directory first. If project_status.scene_count is greater than 0, scenes already exist; do not say no scenes have been created and do not ask to create initial scenes from timeline markers unless the user explicitly asks for that.\n"
        "If purpose is walkthrough, act like intake plus step-by-step onboarding. First identify what the user is making: music video, short film, ad/social clip, visualizer, or something else. Use project_status before choosing the next step. If project_status.has_audio is false, ask whether they plan to use custom audio and point them to Choose Audio if yes; do not ask for lyrics/SRT before resolving the audio plan. If audio exists, ask how they want scenes created: import Prompt Creator outputs, load SRT/lyrics, manually create scenes, or plan scenes with the agent. Only ask about lyrics when the user is making a music video or lyric-driven project. For short films/ads, ask for script, scene beats, product/message, or visual outline instead. Tell them one small UI step at a time and why. Do not generate prompts or run models unless the user explicitly asks. Prefer one next step over a full tutorial.\n"
        "If purpose is troubleshoot, focus on diagnosing the user's stated problem from project_status, active scene context, and available get actions.\n"
        "If purpose is scene_work, help create/update scene notes, prompts, images, and videos using the supported actions.\n\n"
        "If purpose is story_builder, act like a scene planner for complex multi-character projects. This workflow may not use Prompt Creator, SRT, or existing lyric segments. First check project_status.has_audio and story_source.has_source. For music videos/songs, if no audio exists, ask for global/timeline audio before asking for lyrics. If no story source exists, ask the user to paste lyrics/script/story source and use save_story_source in Auto Apply mode when they provide it. Use get_story_source when you need the saved lyrics/script instead of keeping it in chat memory. Help define characters, assign which character(s) appear in each scene, match the lyrics/story beat, and create compact per-scene planning notes. Prefer structured scene plans over long prose. When Auto Apply is enabled, use create_story_scenes_from_source if the user asks to create starter scenes from saved lyrics/script, then use set_scene_plan for requested scenes. Do not generate final image/video prompts unless the user explicitly asks; write planning notes that the existing prompt generators can use.\n\n"
        "Return only valid JSON with this shape:\n"
        "{\n"
        "  \"reply\": \"short message to show the user\",\n"
        "  \"actions\": []\n"
        "}\n\n"
        "Supported actions are:\n"
        "- {\"type\":\"select_scene\",\"scene_id\":\"...\"} or {\"type\":\"select_scene\",\"scene_number\":2}\n"
        "- {\"type\":\"get_scene_lyrics\",\"scene_id\":\"...\"} or {\"type\":\"get_scene_lyrics\",\"scene_number\":2}\n"
        "- {\"type\":\"get_scene_context\",\"scene_id\":\"...\"} or {\"type\":\"get_scene_context\",\"scene_number\":2}\n"
        "- {\"type\":\"get_story_source\"}\n"
        "- {\"type\":\"get_context_prompts\",\"context_type\":\"all|theme_style|story_idea|subject_scene\"}\n"
        "- {\"type\":\"get_selected_timeline_range\"}\n"
        "- {\"type\":\"save_story_source\",\"text\":\"lyrics/script/story source text\"}\n"
        "- {\"type\":\"set_context_prompt\",\"context_type\":\"theme_style|story_idea|subject_scene\",\"text\":\"...\",\"mode\":\"replace|append\"}\n"
        "- {\"type\":\"create_story_scenes_from_source\",\"scene_count\":12}\n"
        "- {\"type\":\"create_concept_prompts\",\"source_mode\":\"all|director|scene|timeline|director_scene\",\"scope\":\"all|selected|range\",\"batch_size\":5,\"use_story\":true,\"use_theme\":true}\n"
        "- {\"type\":\"create_motion_notes\",\"source_mode\":\"concept|concept_director|concept_timeline|all\",\"scope\":\"all|selected|range\",\"batch_size\":5,\"use_story\":true,\"use_theme\":true}\n"
        "- {\"type\":\"set_active_scene_to_selected_range\",\"scene_id\":\"optional target scene\"}\n"
        "- {\"type\":\"create_scene_from_selected_range\",\"label\":\"...\",\"director_note\":\"...\",\"scene_notes\":\"...\",\"flux_notes\":\"...\",\"nb_notes\":\"...\",\"video_notes\":\"...\"}\n"
        "- {\"type\":\"split_selected_range_into_scenes\",\"scene_count\":3,\"label_prefix\":\"Scene\",\"director_notes\":[\"...\"],\"scene_notes\":[\"...\"]}\n"
        "- {\"type\":\"split_scene_into_subscenes\",\"scene_id\":\"...\",\"scene_number\":4,\"scene_count\":3,\"label_prefix\":\"Scene 4\"}\n"
        "- {\"type\":\"merge_scenes\",\"scene_numbers\":[2,3,4],\"label\":\"Scene 2\"}\n"
        "- {\"type\":\"renumber_scene_labels\"}\n"
        "- {\"type\":\"normalize_dual_vocal_director_notes\",\"replacement\":\"male and female\"}\n"
        "- {\"type\":\"replace_director_note_text\",\"find\":\"male singer and female singer\",\"replace\":\"female and male\"}\n"
        "- {\"type\":\"assign_selected_range_note\",\"label\":\"Female vocal\",\"marker_type\":\"female vocal|male vocal|chorus|verse|beat|note\",\"note\":\"...\"}\n"
        "- {\"type\":\"sync_existing_scenes_to_timeline_markers\",\"create_missing\":true}\n"
        "- {\"type\":\"set_scene_notes\",\"scene_number\":2,\"text\":\"...\"}\n"
        "- {\"type\":\"set_flux_notes\",\"scene_number\":2,\"text\":\"...\"}\n"
        "- {\"type\":\"set_nb_notes\",\"scene_number\":2,\"text\":\"...\"}\n"
        "- {\"type\":\"set_video_notes\",\"scene_number\":2,\"text\":\"...\"}\n"
        "- {\"type\":\"set_scene_plan\",\"scene_number\":2,\"director_note\":\"...\",\"scene_notes\":\"...\",\"flux_notes\":\"...\",\"nb_notes\":\"...\",\"video_notes\":\"...\"}\n"
        "- {\"type\":\"set_image_model_mode\",\"image_mode\":\"zimage|flux_klein|nano_banana|ernie_image\"}\n"
        "- {\"type\":\"set_video_model_mode\",\"video_mode\":\"i2v|t2v\"}\n"
        "- {\"type\":\"request_reference_images\",\"scene_id\":\"...\",\"image_mode\":\"nano_banana|flux_klein\"}\n"
        "- {\"type\":\"generate_image_prompt_for_current_mode\",\"scene_id\":\"...\",\"image_mode\":\"optional zimage|flux_klein|nano_banana|ernie_image\"}\n"
        "- {\"type\":\"run_image_for_current_mode\",\"scene_id\":\"...\",\"image_mode\":\"optional zimage|flux_klein|nano_banana|ernie_image\"}\n"
        "- {\"type\":\"generate_video_prompt_for_current_mode\",\"scene_id\":\"...\",\"video_mode\":\"optional i2v|t2v\"}\n\n"
        "- {\"type\":\"run_video_for_current_mode\",\"scene_id\":\"...\",\"video_mode\":\"optional i2v|t2v\"}\n\n"
        "Use note actions to capture the creative direction you discuss with the user. Use set_scene_plan when planning character assignments, story beats, scene concepts, and motion notes for one or more scenes. Keep each set_scene_plan field short and direct.\n"
        "Use create_concept_prompts when the user asks to create, generate, build, or update concept prompts from director notes, scene notes, timeline notes, story idea, or theme/style. This writes generated concept prompts into scene notes in batches.\n"
        "Use create_motion_notes when the user asks to create, generate, build, or update I2V/T2V motion notes from concept prompts, director notes, timeline notes, story idea, or theme/style. This writes generated motion notes into the I2V motion notes fields/file in batches.\n"
        "Use get_context_prompts to read the global Theme/style, Story idea, and Subject/scene context prompts. Use set_context_prompt when the user asks to update those global context prompt files. Choose context_type theme_style for style/mood/look, story_idea for plot/concept/narrative, and subject_scene for characters, subjects, locations, and recurring scene details.\n"
        "Use selected_timeline_range and timeline_markers when the user is timing vocals, choruses, dialogue, or music sections. If they say this selected range is where someone sings, use assign_selected_range_note. If they ask to make a scene cover it, use set_active_scene_to_selected_range. Only use create_scene_from_selected_range or split_selected_range_into_scenes when selected_timeline_range is not null and the user explicitly asks to create/split the selected in/out range. If the user asks to split one existing scene, such as 'split scene 4 into 3', use split_scene_into_subscenes with that scene number and count. If the user asks to combine, merge, join, or consolidate scenes, use merge_scenes with the scene numbers. If the user asks to update/reword/fix director notes, do note actions only; do not sync or change timings. If they ask to replace director note text, such as 'when a director note says X change it to Y', use replace_director_note_text. If they ask that all both-character or dual-vocal director notes say a specific phrase such as 'male and female' or 'female and male', use normalize_dual_vocal_director_notes with that replacement phrase. If timeline markers already exist and the user asks to update, align, match, sync, or split existing scene segments from those markers, or answers yes after you suggest updating scenes from markers, use sync_existing_scenes_to_timeline_markers. If the user asks to fix missing/skipped/duplicate scene numbers or renumber labels, use renumber_scene_labels. Do not use split_selected_range_into_scenes for marker-based timing, one-scene splits, or merge/combine requests.\n"
        "Use generate actions when the user asks you to make/create/update the actual image or video prompt. Use run_image_for_current_mode only when the user asks to run/create/generate the actual image, not just the text prompt.\n"
        "If the user asks for multiple steps, include all requested actions in order. Example: prompt, then image, then video means generate_image_prompt_for_current_mode, run_image_for_current_mode, generate_video_prompt_for_current_mode, run_video_for_current_mode.\n"
        "Nano B and Flux/Klein can run with or without reference images. If no references exist, use the normal generate/run actions anyway and the app will use text-only prompting. Only request reference images when the user specifically asks to use references or the scene concept requires matching an exact character/location.\n"
        "Do not write final image/video prompts yourself as actions. Do not use unsupported set_image_prompt, set_flux_prompt, set_nb_prompt, or set_video_prompt actions.\n"
        "Never put JSON, action arrays, code fences, or internal notes inside the reply string. The reply must be a short user-facing sentence; all work must be in the actions array.\n"
        "Do not claim which image/video generator ran. The app will report the actual mode after it runs the action.\n"
        f"Agent mode: {'AUTO APPLY. If the user asks you to do/apply/update/fill/write prompts or notes, include supported note/generate actions for the requested scene fields.' if auto_apply else 'MANUAL. Never include actions. Do not say you applied or updated the project; only suggest copy-ready text.'}\n"
        "Prefer scene_number for scene-targeted actions. Use scene_id only when copying an exact internal id from the context JSON. If the target scene is unclear, ask one short clarifying question and return no actions.\n"
        "In Auto Apply mode, the application will apply actions after your response. In Manual mode, actions must always be an empty list.\n\n"
        "Active context JSON:\n"
        f"{context_text}\n\n"
        "Conversation:\n"
        f"{conversation_text}\n\n"
        "Return the JSON now."
    )
    text, info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.55),
        top_p=float(payload.get("top_p") or 0.95),
        max_new_tokens=int(payload.get("max_new_tokens") or 500),
        label="Builder Agent",
        preserve_paragraphs=True,
    )
    raw = _clean_lm_studio_plain_text(text)
    try:
        parsed = _extract_json_object_from_text(raw)
    except Exception:
        parsed = {"reply": raw, "actions": []}
    reply = _clean_lm_studio_plain_text(str(parsed.get("reply") or "")).strip()
    actions = parsed.get("actions") if isinstance(parsed, dict) else []
    if not isinstance(actions, list) or not auto_apply:
        actions = []
    cleaned_actions = []
    allowed_types = {
        "set_scene_notes",
        "set_scene_plan",
        "select_scene",
        "get_scene_lyrics",
        "get_scene_context",
        "get_story_source",
        "get_context_prompts",
        "get_selected_timeline_range",
        "save_story_source",
        "set_context_prompt",
        "create_story_scenes_from_source",
        "create_concept_prompts",
        "create_motion_notes",
        "set_active_scene_to_selected_range",
        "create_scene_from_selected_range",
        "split_selected_range_into_scenes",
        "split_scene_into_subscenes",
        "merge_scenes",
        "renumber_scene_labels",
        "normalize_dual_vocal_director_notes",
        "replace_director_note_text",
        "assign_selected_range_note",
        "sync_existing_scenes_to_timeline_markers",
        "set_flux_notes",
        "set_nb_notes",
        "set_video_notes",
        "set_image_model_mode",
        "set_video_model_mode",
        "request_reference_images",
        "generate_image_prompt_for_current_mode",
        "run_image_for_current_mode",
        "generate_video_prompt_for_current_mode",
        "run_video_for_current_mode",
    }
    allowed_image_modes = {"zimage", "flux_klein", "nano_banana", "ernie_image"}
    allowed_video_modes = {"i2v", "t2v"}
    max_actions = 60 if agent_purpose == "story_builder" else 12
    plan_fields = {"director_note", "scene_notes", "flux_notes", "nb_notes", "video_notes"}
    for action in actions[:max_actions]:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type") or "").strip()
        scene_id = str(action.get("scene_id") or "").strip()
        action_text_raw = str(action.get("text") or "").strip()
        image_mode = str(action.get("image_mode") or "").strip()
        video_mode = str(action.get("video_mode") or "").strip()
        context_type = str(action.get("context_type") or "").strip().lower()
        context_mode = str(action.get("mode") or "").strip().lower()
        scene_count = action.get("scene_count")
        try:
            scene_count = int(scene_count) if scene_count is not None else None
        except Exception:
            scene_count = None
        plan_values = {
            field: str(action.get(field) or "").strip()[:5000]
            for field in plan_fields
            if str(action.get(field) or "").strip()
        }
        scene_number = action.get("scene_number")
        if scene_number is None:
            scene_target_text = " ".join(
                str(action.get(field) or "")
                for field in (
                    "scene_id",
                    "scene",
                    "scene_label",
                    "scene_name",
                    "target_scene",
                    "target",
                    "label",
                    "name",
                    "title",
                )
            )
            scene_match = re.search(r"\bscene\s*(\d+)(?:\b|\s|\.|:|-)", scene_target_text, re.I)
            if scene_match:
                scene_number = scene_match.group(1)
        try:
            scene_number = int(scene_number) if scene_number is not None else None
        except Exception:
            scene_number = None
        no_scene_actions = {
            "set_image_model_mode",
            "set_video_model_mode",
            "get_story_source",
            "get_context_prompts",
            "save_story_source",
            "set_context_prompt",
            "create_story_scenes_from_source",
            "create_concept_prompts",
            "create_motion_notes",
            "get_selected_timeline_range",
            "create_scene_from_selected_range",
            "split_selected_range_into_scenes",
            "merge_scenes",
            "renumber_scene_labels",
            "normalize_dual_vocal_director_notes",
            "replace_director_note_text",
            "assign_selected_range_note",
            "sync_existing_scenes_to_timeline_markers",
        }
        if action_type in allowed_types and (scene_id or scene_number is not None or action_type in no_scene_actions):
            if action_type == "request_reference_images" and not scene_id:
                continue
            cleaned_action = {
                "type": action_type,
            }
            if scene_id:
                cleaned_action["scene_id"] = scene_id[:160]
            if scene_number is not None:
                cleaned_action["scene_number"] = scene_number
            if action_text_raw:
                cleaned_action["text"] = action_text_raw[:50000 if action_type == "save_story_source" else 5000]
            if image_mode in allowed_image_modes:
                cleaned_action["image_mode"] = image_mode
            if video_mode in allowed_video_modes:
                cleaned_action["video_mode"] = video_mode
            if scene_count is not None:
                cleaned_action["scene_count"] = max(1, min(120, scene_count))
            if action_type == "get_context_prompts":
                if context_type not in {"all", "theme_style", "story_idea", "subject_scene"}:
                    context_type = "all"
                cleaned_action["context_type"] = context_type
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "set_context_prompt":
                if context_type not in {"theme_style", "story_idea", "subject_scene"}:
                    continue
                if not action_text_raw:
                    continue
                cleaned_action["context_type"] = context_type
                cleaned_action["text"] = action_text_raw[:50000]
                cleaned_action["mode"] = context_mode if context_mode in {"replace", "append"} else "replace"
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "assign_selected_range_note":
                for field in ("label", "marker_type", "note"):
                    value = str(action.get(field) or "").strip()
                    if value:
                        cleaned_action[field] = value[:1000]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "create_scene_from_selected_range":
                for field in ("label", "director_note", "scene_notes", "flux_notes", "nb_notes", "video_notes", "note"):
                    value = str(action.get(field) or "").strip()
                    if value:
                        cleaned_action[field] = value[:5000]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "split_selected_range_into_scenes":
                label_prefix = str(action.get("label_prefix") or "").strip()
                if label_prefix:
                    cleaned_action["label_prefix"] = label_prefix[:160]
                for field in ("director_notes", "scene_notes"):
                    values = action.get(field)
                    if isinstance(values, list):
                        cleaned_action[field] = [str(value or "").strip()[:5000] for value in values[:120]]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "split_scene_into_subscenes":
                label_prefix = str(action.get("label_prefix") or "").strip()
                if label_prefix:
                    cleaned_action["label_prefix"] = label_prefix[:160]
                for field in ("director_notes", "scene_notes"):
                    values = action.get(field)
                    if isinstance(values, list):
                        cleaned_action[field] = [str(value or "").strip()[:5000] for value in values[:120]]
                if scene_count is None or scene_count < 2:
                    continue
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "merge_scenes":
                numbers = action.get("scene_numbers") or action.get("scenes") or []
                cleaned_numbers = []
                if isinstance(numbers, list):
                    for value in numbers[:120]:
                        try:
                            number = int(value)
                        except Exception:
                            continue
                        if number > 0 and number not in cleaned_numbers:
                            cleaned_numbers.append(number)
                for src, dst in (("start_scene_number", "start_scene_number"), ("end_scene_number", "end_scene_number")):
                    try:
                        value = int(action.get(src))
                    except Exception:
                        value = None
                    if value is not None and value > 0:
                        cleaned_action[dst] = value
                label = str(action.get("label") or "").strip()
                if label:
                    cleaned_action["label"] = label[:160]
                if cleaned_numbers:
                    cleaned_action["scene_numbers"] = cleaned_numbers
                if len(cleaned_numbers) < 2 and not (cleaned_action.get("start_scene_number") and cleaned_action.get("end_scene_number")):
                    continue
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "renumber_scene_labels":
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "normalize_dual_vocal_director_notes":
                replacement = str(action.get("replacement") or "male and female").strip()
                cleaned_action["replacement"] = (replacement or "male and female")[:160]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "replace_director_note_text":
                find_text = str(action.get("find") or action.get("from") or "").strip()
                replace_text = str(action.get("replace") or action.get("to") or "").strip()
                if not find_text or not replace_text:
                    continue
                cleaned_action["find"] = find_text[:500]
                cleaned_action["replace"] = replace_text[:500]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "get_story_source":
                cleaned_actions.append(cleaned_action)
                continue
            if action_type in {"save_story_source", "create_story_scenes_from_source"} and action_type == "save_story_source" and not action_text_raw:
                continue
            if action_type == "save_story_source":
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "create_story_scenes_from_source":
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "create_concept_prompts":
                source_mode = str(action.get("source_mode") or action.get("source") or "all").strip().lower()
                scope = str(action.get("scope") or "all").strip().lower()
                if source_mode not in {"all", "director", "scene", "timeline", "director_scene"}:
                    source_mode = "all"
                if scope not in {"all", "selected", "range"}:
                    scope = "all"
                try:
                    batch_size = int(action.get("batch_size") or action.get("batch") or 5)
                except Exception:
                    batch_size = 5
                cleaned_action["source_mode"] = source_mode
                cleaned_action["scope"] = scope
                cleaned_action["batch_size"] = max(1, min(10, batch_size))
                cleaned_action["use_story"] = action.get("use_story") is not False
                cleaned_action["use_theme"] = action.get("use_theme") is not False
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "create_motion_notes":
                source_mode = str(action.get("source_mode") or action.get("source") or "concept").strip().lower()
                scope = str(action.get("scope") or "all").strip().lower()
                if source_mode not in {"concept", "concept_director", "concept_timeline", "all"}:
                    source_mode = "concept"
                if scope not in {"all", "selected", "range"}:
                    scope = "all"
                try:
                    batch_size = int(action.get("batch_size") or action.get("batch") or 5)
                except Exception:
                    batch_size = 5
                cleaned_action["source_mode"] = source_mode
                cleaned_action["scope"] = scope
                cleaned_action["batch_size"] = max(1, min(10, batch_size))
                cleaned_action["use_story"] = action.get("use_story") is not False
                cleaned_action["use_theme"] = action.get("use_theme") is not False
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "sync_existing_scenes_to_timeline_markers":
                cleaned_action["create_missing"] = action.get("create_missing") is not False
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "set_scene_plan":
                for field, value in plan_values.items():
                    cleaned_action[field] = value
                if not plan_values and not action_text_raw:
                    continue
                cleaned_actions.append(cleaned_action)
                continue
            if action_type in {"set_scene_notes", "set_flux_notes", "set_nb_notes", "set_video_notes"} and not cleaned_action.get("text"):
                alias_fields = {
                    "set_scene_notes": ("scene_notes", "notes", "concept", "concept_prompt", "prompt"),
                    "set_flux_notes": ("flux_notes", "notes", "prompt"),
                    "set_nb_notes": ("nb_notes", "nano_banana_notes", "notes", "prompt"),
                    "set_video_notes": ("video_notes", "i2v_notes", "motion_notes", "notes", "prompt"),
                }[action_type]
                for field in alias_fields:
                    value = str(action.get(field) or "").strip()
                    if value:
                        cleaned_action["text"] = value[:5000]
                        break
            if action_type.startswith("set_") and not action_text_raw:
                if action_type == "set_image_model_mode" and cleaned_action.get("image_mode"):
                    cleaned_actions.append(cleaned_action)
                    continue
                if action_type == "set_video_model_mode" and cleaned_action.get("video_mode"):
                    cleaned_actions.append(cleaned_action)
                    continue
                if action_type in {"set_scene_notes", "set_flux_notes", "set_nb_notes", "set_video_notes"} and cleaned_action.get("text"):
                    cleaned_actions.append(cleaned_action)
                    continue
                continue
            cleaned_actions.append(cleaned_action)
    action_hint_text = "\n\n".join(
        [item.get("content", "") for item in cleaned_messages[-6:]]
        + ([reply] if reply else [])
    )
    latest_user_text = str(latest_user or cleaned_messages[-1].get("content", "") if cleaned_messages else "").strip()
    latest_merge_intent = re.search(r"\b(combine|merge|join|consolidate)\b", latest_user_text, re.I)
    latest_split_intent = re.search(r"\b(split|break|divide|sub[-\s]?scenes?|sub[-\s]?segments?)\b", latest_user_text, re.I)
    latest_director_note_text_intent = re.search(r"\b(director\s+notes?|notes?|reword|rename|text)\b", latest_user_text, re.I)
    latest_dual_vocal_note_intent = (
        latest_director_note_text_intent and
        re.search(r"\b(male\s+and\s+female|both\s+(?:characters|singers|vocals?)|dual[-\s]?vocal)\b", latest_user_text, re.I)
    )
    if auto_apply and latest_director_note_text_intent:
        cleaned_actions = [
            action for action in cleaned_actions
            if action.get("type") not in {
                "sync_existing_scenes_to_timeline_markers",
                "split_selected_range_into_scenes",
                "split_scene_into_subscenes",
                "merge_scenes",
            }
        ]
    if auto_apply and latest_dual_vocal_note_intent and not any(action.get("type") == "normalize_dual_vocal_director_notes" for action in cleaned_actions):
        exact_replace_match = re.search(r"\b(?:director\s+notes?|director\s+note|note)\s+says?\s+[\"']([^\"']+)[\"'].*?\b(?:changed?\s+to|to|instead)\s+[\"']([^\"']+)[\"']", latest_user_text, re.I)
        if exact_replace_match and not any(action.get("type") == "replace_director_note_text" for action in cleaned_actions):
            cleaned_actions.append(
                {
                    "type": "replace_director_note_text",
                    "find": exact_replace_match.group(1).strip(),
                    "replace": exact_replace_match.group(2).strip(),
                }
            )
        replacement_match = re.search(r"\b(?:says?|to|instead|changed?\s+to)\s+[\"']?((?:fe)?male\s+and\s+(?:fe)?male)[\"']?", latest_user_text, re.I)
        replacement = replacement_match.group(1).strip().lower() if replacement_match else "male and female"
        cleaned_actions.append({"type": "normalize_dual_vocal_director_notes", "replacement": replacement})
    if auto_apply and latest_split_intent:
        cleaned_actions = [
            action for action in cleaned_actions
            if action.get("type") != "merge_scenes"
        ]
    if auto_apply and latest_merge_intent:
        cleaned_actions = [
            action for action in cleaned_actions
            if action.get("type") not in {"split_scene_into_subscenes", "split_selected_range_into_scenes"}
        ]
    merge_intent = latest_merge_intent or (
        not latest_split_intent and
        re.fullmatch(r"\s*(yes|yep|yeah|ok|okay|please do it|do it|go ahead|sure)\.?\s*", latest_user_text, re.I) and
        re.search(r"\b(combine|merge|join|consolidate)\b", action_hint_text, re.I)
    )
    if auto_apply and merge_intent:
        cleaned_actions = [
            action for action in cleaned_actions
            if action.get("type") not in {"split_scene_into_subscenes", "split_selected_range_into_scenes"}
        ]
        has_merge_action = any(action.get("type") == "merge_scenes" for action in cleaned_actions)
        if not has_merge_action:
            range_match = re.search(r"\bscenes?\s+(\d+)\s*(?:-|to|through)\s*(\d+)\b", action_hint_text, re.I)
            scene_numbers = []
            if range_match:
                start_num = int(range_match.group(1))
                end_num = int(range_match.group(2))
                if start_num <= end_num:
                    scene_numbers = list(range(start_num, end_num + 1))
            if not scene_numbers:
                sequence_match = re.search(r"\bscenes?\s+((?:\d+\s*(?:,|and)?\s*){2,})", action_hint_text, re.I)
                if sequence_match:
                    scene_numbers = [int(value) for value in re.findall(r"\d+", sequence_match.group(1))]
            if len(scene_numbers) >= 2:
                scene_numbers = list(dict.fromkeys(scene_numbers))
                cleaned_actions.append(
                    {
                        "type": "merge_scenes",
                        "scene_numbers": scene_numbers,
                        "label": f"Scene {scene_numbers[0]}",
                    }
                )
                if not reply:
                    reply = f"Merging scenes {', '.join(str(value) for value in scene_numbers)}."
    if auto_apply and not cleaned_actions:
        split_hint_text = latest_user_text if latest_split_intent else action_hint_text
        if not merge_intent and re.search(r"\b(split|break|divide|sub[-\s]?scenes?|sub[-\s]?segments?)\b", split_hint_text, re.I):
            scene_matches = re.findall(r"\bscene\s+(\d+)\b", split_hint_text, re.I)
            count_matches = []
            count_matches.extend(
                int(match)
                for match in re.findall(r"\b(?:into|in|to)\s+(\d+)\s*(?:sub[-\s]?scenes?|scenes?|segments?)\b", split_hint_text, re.I)
                if str(match).isdigit()
            )
            count_matches.extend(
                int(match)
                for match in re.findall(r"\b(\d+)\s*(?:sub[-\s]?scenes?|sub[-\s]?segments?)\b", split_hint_text, re.I)
                if str(match).isdigit()
            )
            latest_user_number = re.fullmatch(r"\s*(\d{1,2})\s*", latest_user or "")
            if latest_user_number:
                count_matches.append(int(latest_user_number.group(1)))
            if scene_matches and count_matches:
                inferred_scene = int(scene_matches[-1])
                inferred_count = max(2, min(24, int(count_matches[-1])))
                cleaned_actions.append(
                    {
                        "type": "split_scene_into_subscenes",
                        "scene_number": inferred_scene,
                        "scene_count": inferred_count,
                        "label_prefix": f"Scene {inferred_scene}",
                    }
                )
                if not reply:
                    reply = f"Splitting Scene {inferred_scene} into {inferred_count} sub-scenes."
    if not reply:
        reply = "Done." if cleaned_actions else "I can help with that."
    return {"reply": reply, "actions": cleaned_actions, **info}


def _generate_builder_concept_prompts(payload):
    source_mode = str(payload.get("source_mode") or "all").strip().lower()
    story_idea = str(payload.get("story_idea") or "").strip()
    theme_style = str(payload.get("theme_style") or "").strip()
    previous_summary = str(payload.get("previous_summary") or "").strip()
    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for concept prompt creation.")

    cleaned_scenes = []
    for scene in scenes[:20]:
        if not isinstance(scene, dict):
            continue
        try:
            number = int(scene.get("scene_number") or scene.get("number") or len(cleaned_scenes) + 1)
        except Exception:
            number = len(cleaned_scenes) + 1
        timeline_notes = scene.get("timeline_notes") or []
        if not isinstance(timeline_notes, list):
            timeline_notes = []
        cleaned_scenes.append({
            "scene_number": max(1, number),
            "label": str(scene.get("label") or f"Scene {number}").strip()[:160],
            "lyric_text": str(scene.get("lyric_text") or "").strip()[:1200],
            "lyric_singers": [
                str(item or "").strip()[:160]
                for item in (scene.get("lyric_singers") if isinstance(scene.get("lyric_singers"), list) else [])
                if str(item or "").strip()
            ][:8],
            "lyric_instrumental": bool(scene.get("lyric_instrumental")),
            "lyric_no_lip_sync": bool(scene.get("lyric_no_lip_sync")),
            "no_character_present": bool(scene.get("no_character_present") or scene.get("no_subject") or scene.get("no_visible_subject")),
            "mapped_subjects": str(scene.get("mapped_subjects") or "").strip()[:1600],
            "director_note": str(scene.get("director_note") or "").strip()[:1200],
            "scene_note": str(scene.get("scene_note") or "").strip()[:1200],
            "timeline_notes": [
                {
                    "label": str(item.get("label") or item.get("type") or "note").strip()[:160],
                    "note": str(item.get("note") or "").strip()[:800],
                    "start": item.get("start"),
                    "end": item.get("end"),
                }
                for item in timeline_notes[:8]
                if isinstance(item, dict)
            ],
            "subject_reference_mode": str(scene.get("subject_reference_mode") or "none").strip()[:80],
            "location_reference_mode": str(scene.get("location_reference_mode") or "none").strip()[:80],
        })
    if not cleaned_scenes:
        raise ValueError("No valid scenes were provided for concept prompt creation.")

    source_note = {
        "director": "Use director notes as the main scene notes.",
        "scene": "Use raw scene notes as the main scene notes.",
        "timeline": "Use overlapping timeline notes as the main scene notes.",
        "lyrics": "Use lyric notes as the main scene notes.",
        "subjects": "Use subject and singer mapping as the main scene notes.",
        "lyrics_subjects": "Use lyric notes plus subject and singer mapping as the main scene notes.",
        "director_scene": "Use director notes and raw scene notes as the main scene notes.",
        "all": "Use all available notes: director notes, raw scene notes, lyric notes, subject/singer mapping, and overlapping timeline notes.",
    }.get(source_mode, "Use all available notes: director notes, raw scene notes, lyric notes, subject/singer mapping, and overlapping timeline notes.")

    instruction = (
        "You will create concept prompts from scene notes and/or director notes, timeline notes, story idea, and style/theme.\n"
        "Return only valid JSON in this exact flat format:\n"
        "{\n"
        "  \"Scene1\": \"\",\n"
        "  \"Scene2\": \"\"\n"
        "}\n\n"
        f"Source mode: {source_mode}. {source_note}\n\n"
        "The story idea explains the overall music video concept, emotional arc, setting, and visual direction. Use it to make the prompts feel connected and to make each scene flow naturally like a music video.\n"
        "The style/theme explains the visual language, mood, palette, texture, and production design. Use it for consistency.\n"
        "Character descriptions or character reference images may be provided separately. Do not describe characters' hair, face, clothing, makeup, or exact appearance in the concept prompts.\n\n"
        "Create one prompt for each provided scene.\n"
        "Each concept prompt must include who is in the scene, shot type, Location:, and scene details that fit the story world.\n\n"
        "Subject rules:\n"
        "- If lyric_singers contains names, those are the subjects/performers for that scene.\n"
        "- If mapped_subjects contains character reference names, use those as the visible subjects for that scene.\n"
        "- If no_character_present is true, do not include any main character, singer, performer, person, mapped subject, or character reference in that scene. Start with \"No main subject\" and focus on location, props, objects, atmosphere, or environmental action.\n"
        "- If lyric_no_lip_sync is true or lyric_instrumental is true, the scene should not be treated as a singing/lip-sync performance scene.\n"
        "- If the scene note says female, start with \"Female only\".\n"
        "- If the scene note says male, start with \"Male only\".\n"
        "- If the scene note says female and male, start with \"Female and male together in frame\".\n"
        "- If the scene note is instrumental only or has no clear subject, start with \"No main subject\".\n\n"
        "Music video flow rules:\n"
        "- Make the prompts feel like connected shots from the same music video.\n"
        "- Let locations and details progress with the story instead of feeling random.\n"
        "- Instrumental scenes can be establishing shots, transitions, symbolic visuals, or world-building.\n"
        "- Character scenes should feel like performance, duet, reaction, memory, confrontation, longing, or story moments.\n"
        "- Keep the visual world consistent across all prompts.\n\n"
        "Shot rules:\n"
        "- Instrumental scenes should usually be wide establishing shots or wide environment shots.\n"
        "- Female-only or male-only scenes should usually be close-up, upper body, or medium shot.\n"
        "- Female and male together should usually be full body, two-shot, medium-wide, or wide shot.\n"
        "- Make the shot type fit the scene note and story moment.\n\n"
        "Location rules:\n"
        "- Include \"Location:\" in every prompt.\n"
        "- Pick locations that fit the story idea, mood, and style/theme.\n"
        "- If a location reference is available for that scene, use it as the location identity instead of inventing an unrelated location.\n"
        "- If no location reference is available, create the location from the story idea, style/theme, and notes.\n"
        "- Keep locations visually connected so the scenes feel like one continuous music video world.\n\n"
        "Style rules:\n"
        "- Do not write full text-to-image prompts.\n"
        "- Do not include character descriptions.\n"
        "- Do not include camera settings, aspect ratio, render style, or quality tags.\n"
        "- Keep each prompt as one clear sentence or short paragraph.\n"
        "- Make prompts detailed enough for another LLM to expand into image prompts.\n"
        "- Return only the JSON object.\n\n"
        f"Story idea:\n{story_idea or '[none provided]'}\n\n"
        f"Style/theme:\n{theme_style or '[none provided]'}\n\n"
        f"Previous batch visual progression summary:\n{previous_summary or '[none yet]'}\n\n"
        "Scenes:\n"
        f"{json.dumps(cleaned_scenes, ensure_ascii=False, indent=2)}"
    )
    text, info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.95),
        max_new_tokens=int(payload.get("max_new_tokens") or 1200),
        label="Concept Prompt Creator",
        preserve_paragraphs=True,
    )
    raw = _clean_lm_studio_plain_text(text)
    prompts = {}
    try:
        parsed = _extract_json_object_from_text(raw)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                match = re.search(r"scene\s*(\d+)", str(key), re.I)
                if not match:
                    continue
                prompt_text = str(value or "").strip()
                if prompt_text:
                    prompts[f"Scene{int(match.group(1))}"] = prompt_text
    except Exception:
        prompts = {}
    if not prompts:
        for match in re.finditer(r'"?\bScene\s*(\d+)"?\s*:\s*"([^"]+)"', raw, re.I | re.S):
            prompt_text = match.group(2).strip()
            if prompt_text:
                prompts[f"Scene{int(match.group(1))}"] = prompt_text
    if not prompts:
        raise ValueError("Gemma did not return any SceneN concept prompts.")
    summary_lines = []
    for key in sorted(prompts.keys(), key=lambda item: int(re.search(r"\d+", item).group(0))):
        text_value = prompts[key]
        summary_lines.append(f"{key}: {text_value[:180]}")
    return {
        "prompts": prompts,
        "raw": raw,
        "summary": "\n".join(summary_lines)[:1200],
        "run_info": info,
    }


def _generate_builder_motion_notes(payload):
    source_mode = str(payload.get("source_mode") or "concept").strip().lower()
    story_idea = str(payload.get("story_idea") or "").strip()
    theme_style = str(payload.get("theme_style") or "").strip()
    previous_summary = str(payload.get("previous_summary") or "").strip()
    video_mode = str(payload.get("video_mode") or "i2v").strip().lower()
    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for motion note creation.")

    cleaned_scenes = []
    for scene in scenes[:20]:
        if not isinstance(scene, dict):
            continue
        try:
            number = int(scene.get("scene_number") or scene.get("number") or len(cleaned_scenes) + 1)
        except Exception:
            number = len(cleaned_scenes) + 1
        timeline_notes = scene.get("timeline_notes") or []
        if not isinstance(timeline_notes, list):
            timeline_notes = []
        cleaned_scenes.append({
            "scene_number": max(1, number),
            "label": str(scene.get("label") or f"Scene {number}").strip()[:160],
            "concept_prompt": str(scene.get("concept_prompt") or "").strip()[:1600],
            "director_note": str(scene.get("director_note") or "").strip()[:1200],
            "timeline_notes": [
                {
                    "label": str(item.get("label") or item.get("type") or "note").strip()[:160],
                    "note": str(item.get("note") or "").strip()[:800],
                    "start": item.get("start"),
                    "end": item.get("end"),
                }
                for item in timeline_notes[:8]
                if isinstance(item, dict)
            ],
        })
    if not cleaned_scenes:
        raise ValueError("No valid scenes were provided for motion note creation.")

    source_note = {
        "concept": "Use only concept prompts.",
        "concept_director": "Use concept prompts and director notes.",
        "concept_timeline": "Use concept prompts and overlapping timeline notes.",
        "all": "Use concept prompts, director notes, and overlapping timeline notes.",
    }.get(source_mode, "Use concept prompts.")

    instruction = (
        "You will create concise I2V/T2V motion notes for scene-by-scene music video generation.\n"
        "Return only valid JSON in this exact flat format:\n"
        "{\n"
        "  \"Motion1\": \"\",\n"
        "  \"Motion2\": \"\"\n"
        "}\n\n"
        f"Source mode: {source_mode}. {source_note}\n"
        f"Target video mode: {video_mode.upper()}.\n\n"
        "Write one motion note for each provided scene.\n"
        "Each motion note should describe camera motion, subject movement or interaction, environmental motion, and emotional pacing.\n"
        "Use the concept prompt as the main visual action source. Use director notes and timeline notes only when provided by the source mode.\n\n"
        "Rules:\n"
        "- Keep each motion note one sentence or one short paragraph.\n"
        "- Do not write a full video prompt.\n"
        "- Do not repeat character appearance details.\n"
        "- Do not include camera settings, frame rate, aspect ratio, render quality, or model tags.\n"
        "- For instrumental/no-subject scenes, focus on environmental movement, transitions, symbolic motion, light, particles, atmosphere, or camera drift.\n"
        "- For solo character scenes, include performance/reaction movement, facial/emotional energy, body motion, and camera movement.\n"
        "- For two-character scenes, include interaction, blocking, distance, reaction, duet/confrontation energy, and camera movement.\n"
        "- Keep motion connected across the batch like shots from the same music video.\n"
        "- Return only the JSON object.\n\n"
        f"Story idea:\n{story_idea or '[none provided]'}\n\n"
        f"Style/theme:\n{theme_style or '[none provided]'}\n\n"
        f"Previous batch motion summary:\n{previous_summary or '[none yet]'}\n\n"
        "Scenes:\n"
        f"{json.dumps(cleaned_scenes, ensure_ascii=False, indent=2)}"
    )
    text, info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.95),
        max_new_tokens=int(payload.get("max_new_tokens") or 1200),
        label="Motion Note Creator",
        preserve_paragraphs=True,
    )
    raw = _clean_lm_studio_plain_text(text)
    notes = {}
    try:
        parsed = _extract_json_object_from_text(raw)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                match = re.search(r"(?:motion|scene)\s*(\d+)", str(key), re.I)
                if not match:
                    continue
                note_text = str(value or "").strip()
                if note_text:
                    notes[f"Motion{int(match.group(1))}"] = note_text
    except Exception:
        notes = {}
    if not notes:
        for match in re.finditer(r'"?\b(?:Motion|Scene)\s*(\d+)"?\s*:\s*"([^"]+)"', raw, re.I | re.S):
            note_text = match.group(2).strip()
            if note_text:
                notes[f"Motion{int(match.group(1))}"] = note_text
    if not notes:
        raise ValueError("Gemma did not return any MotionN notes.")
    summary_lines = []
    for key in sorted(notes.keys(), key=lambda item: int(re.search(r"\d+", item).group(0))):
        text_value = notes[key]
        summary_lines.append(f"{key}: {text_value[:180]}")
    return {
        "notes": notes,
        "raw": raw,
        "summary": "\n".join(summary_lines)[:1200],
        "run_info": info,
    }


def _generate_builder_t2i_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    ref_image_path = str(payload.get("ref_image_path", "") or "").strip().strip('"')
    ref_image_data = str(payload.get("ref_image_data", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    prompt_mode = str(payload.get("prompt_mode", "") or "").strip().lower()
    reference_context = payload.get("reference_context") or {}
    if not isinstance(reference_context, dict):
        reference_context = {}
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
    has_ref_image = bool(use_vision and ((ref_image_path and os.path.isfile(ref_image_path)) or ref_image_data))
    text_runner = _llm_runner_from_payload(payload)
    if not model_file and text_runner != "lm_studio":
        raise ValueError("Choose a Gemma model first.")
    if use_vision and not has_ref_image:
        raise ValueError("Choose a valid reference image path/data or turn off vision reference.")
    if not has_ref_image and not user_notes:
        raise ValueError("Enter scene notes or provide a reference image.")

    llm = VRGDG_SuperGemmaGGUFChat() if has_ref_image else None
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if has_ref_image else ""
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if has_ref_image else ""
    image = None
    if has_ref_image:
        image = _image_from_data_url(ref_image_data).convert("RGB") if ref_image_data else Image.open(ref_image_path).convert("RGB")
    if has_ref_image:
        prompt = _VISUAL_T2I_INSTRUCTIONS
        prompt += f"\n\nUser notes:\n{user_notes or 'Use the reference image as the guide.'}"
    elif prompt_mode == "flux_klein":
        subject_description = str(reference_context.get("subject_description", "") or "").strip()
        location_name = str(reference_context.get("location_name", "") or "").strip()
        location_description = str(reference_context.get("location_description", "") or "").strip()
        reference_rules = []
        if subject_description:
            reference_rules.append(
                "Use the subject description for the main subject identity, face/body details, outfit, and visible character consistency. "
                f"Subject description: {subject_description}"
            )
        if location_name or location_description:
            location_details = "; ".join(part for part in (location_name, location_description) if part)
            reference_rules.append(
                "Use the mapped location as the required setting/background. Do not replace it with a different location from the concept or notes. "
                f"Mapped location: {location_details}"
            )
        reference_text = ""
        if reference_rules:
            reference_text = (
                "\nReference Builder priorities:\n"
                + "\n".join(f"- {rule}" for rule in reference_rules)
                + "\n- Use the user's notes/concept for action, pose, mood, lighting, story beat, and details after respecting the reference priorities.\n"
            )
        prompt = (
            "Create one polished text-to-image prompt for an image generation model.\n\n"
            "Use the text notes and available reference descriptions to create one coherent new scene. Use the user's notes for pose, camera framing, mood, or other requested details, and give user notes priority.\n"
            f"{reference_text}\n"
            "Rules:\n"
            "- Output one normal text-to-image prompt, not an edit prompt.\n"
            "- Describe only concrete visual details supported by the notes/reference descriptions.\n"
            "- Do not mention reference image, composite, source images, ingredient images, or image grid.\n"
            "- Do not include labels, notes, quotes, markdown, or explanations.\n"
            "- Keep it cinematic, detailed, and visually specific.\n"
            "- Keep the prompt under 120 words.\n\n"
            f"User notes:\n{user_notes or 'Create a cinematic image using the available scene notes.'}"
        )
    elif prompt_mode == "nano_banana":
        subject_description = str(reference_context.get("subject_description", "") or "").strip()
        location_name = str(reference_context.get("location_name", "") or "").strip()
        location_description = str(reference_context.get("location_description", "") or "").strip()
        has_subject_reference = bool(reference_context.get("has_subject_reference") or subject_description)
        has_location_reference = bool(reference_context.get("has_location_reference") or location_name or location_description)
        context_parts = []
        if subject_description:
            context_parts.append(f"Character reference description:\n{subject_description}")
        if location_name or location_description:
            context_parts.append(f"Location reference description:\n{location_name}\n{location_description}".strip())
        if user_notes:
            context_parts.append(f"User input:\n{user_notes}")
        start_rules = [
            "- Start with: Using the provided character reference and location reference, create...",
            "- If only a character reference description is available, start with: Using the provided character reference, create...",
            "- If only a location reference description is available, start with: Using the provided location reference, create...",
        ]
        if not has_subject_reference and not has_location_reference:
            start_rules = ["- Start directly with the shot and subject; do not claim a provided reference exists."]
        prompt = (
            "Create one concise NanoBanana image prompt from the user input and available reference descriptions.\n"
            "Output one normal paragraph, not sections, not markdown, not labels, not explanations.\n\n"
            "Prompt style:\n"
            + "\n".join(start_rules)
            + "\n"
            "- Use a clear cinematic shot type such as close-up, profile close-up, medium close-up, upper body shot, waist-up shot, three-quarter shot, seated shot, over-the-shoulder shot, or low-angle portrait.\n"
            "- Use the user's scene/concept notes as the main creative direction.\n"
            "- Preserve the character identity from the character reference description: face, hair, outfit, makeup, and overall identity.\n"
            "- Preserve the location identity from the location reference description: environment, architecture, layout, atmosphere, and major visible setting details.\n"
            "- Create a new camera angle, new pose, and new composition.\n"
            "- Avoid full-body walking or standing shots unless the user specifically asks for them.\n"
            "- Prefer intimate cinematic compositions when no shot type is specified: close-up, medium close-up, profile, upper body, shallow depth of field, foreground framing, soft bokeh, rim light, atmospheric lighting.\n"
            "- Keep the prompt visually specific and practical for image generation.\n"
            "- Do not include captions, text overlays, dialogue, markdown, labels, bullet points, or section headers.\n\n"
            f"{chr(10).join(context_parts) if context_parts else 'User input: Create a cinematic image using the available scene notes.'}"
        )
    else:
        prompt = f"{_TEXT_ONLY_T2I_INSTRUCTIONS}\n\nUser notes:\n{user_notes}"

    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or (0.25 if has_ref_image else 0.6))
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or (1000 if has_ref_image else 1200))
    unload_after = bool(payload.get("unload_after", True))
    seed = payload.get("seed")

    try:
        if has_ref_image:
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
                pil_images=[image],
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            text, run_info = _run_builder_text_llm(
                payload,
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                label="Gemma",
            )
        text = _clean_visual_gemma_text(text)
        label = "Flux/Klein" if prompt_mode == "flux_klein" else "NanoBanana" if prompt_mode == "nano_banana" else "T2I"
        text = _repair_and_validate_builder_gemma_prompt(payload, text, label)
        return {
            "prompt": text,
            "used_reference_image": has_ref_image,
            "used_model": model_path if has_ref_image else run_info.get("used_model", ""),
            "used_mmproj": mmproj_path,
            "runner": "builtin" if has_ref_image else run_info.get("runner", "builtin"),
            "unloaded": unload_after if has_ref_image else run_info.get("unloaded", unload_after),
        }
    finally:
        if has_ref_image and unload_after:
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
    subject_context = str(payload.get("subject_context", "") or "").strip()
    location_context = str(payload.get("location_context", "") or "").strip()
    no_character_present = bool(payload.get("no_character_present") or payload.get("no_subject") or payload.get("no_visible_subject"))
    text_runner = _llm_runner_from_payload(payload)
    if not model_file and text_runner != "lm_studio":
        raise ValueError("Choose an I2V Gemma model first.")
    if model_file and text_runner != "lm_studio" and not model_file.lower().endswith(".gguf"):
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

    if has_image_reference and text_runner != "lm_studio" and not model_file:
        raise ValueError("Choose an I2V vision Gemma model first.")
    if not has_image_reference and not t2i_prompt:
        raise ValueError("Create or paste a T2I prompt first, or save/load an image reference.")
    if not has_image_reference:
        theme_style = _read_text_file(payload.get("theme_style_path", ""), "Theme/style file")
        story_idea = _read_text_file(payload.get("story_idea_path", ""), "Story idea file")
        subject_scene = _read_text_file(payload.get("subject_scene_path", ""), "Subject/scene file")
        context_parts = []
        if no_character_present:
            context_parts.append("Subject visibility:\nNo main character, singer, performer, person, mapped subject, or character reference is present in this scene. Use location, props, objects, atmosphere, and camera motion instead.")
        elif subject_scene:
            context_parts.append(f"Subject/scene:\n{subject_scene}")
        if subject_context and not no_character_present:
            context_parts.append(f"Mapped scene character(s):\n{subject_context}")
        if location_context:
            context_parts.append(f"Mapped scene location:\n{location_context}")
        if theme_style:
            context_parts.append(f"Theme/style:\n{theme_style}")
        if story_idea:
            context_parts.append(f"Story idea:\n{story_idea}")
        if context_parts:
            user_notes = "\n\n".join(context_parts + ([f"Segment motion notes:\n{user_notes}"] if user_notes else []))

    llm = VRGDG_SuperGemmaGGUFChat() if text_runner != "lm_studio" else None
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else ""
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if has_image_reference and llm else ""
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

    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or (0.25 if has_image_reference else 0.7))
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 4000)
    unload_after = bool(payload.get("unload_after", True))
    seed = payload.get("seed")

    try:
        if has_image_reference and text_runner == "lm_studio":
            text = _run_lm_studio_vision(
                payload,
                prompt,
                [image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            run_info = {
                "runner": "lm_studio_vision",
                "used_model": str(payload.get("lmstudio_model") or "").strip(),
                "unloaded": False,
            }
        else:
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
                    seed=int(seed) if seed is not None else None,
                )
                run_info = {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}
            else:
                text = llm._run_gguf_text_pipeline(
                    model=model,
                    instruction_text=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    seed=int(seed) if seed is not None else None,
                )
                run_info = {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}
        text = _clean_gemma_prompt_text(text)
        text = _repair_and_validate_builder_gemma_prompt(payload, text, "I2V")
        return {"prompt": text, "used_model": run_info.get("used_model", model_path), "used_mmproj": mmproj_path, "used_image_reference": has_image_reference, "runner": run_info.get("runner", "builtin"), "unloaded": run_info.get("unloaded", unload_after)}
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _generate_builder_t2v_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    scene_prompt = str(payload.get("t2i_prompt", "") or payload.get("scene_prompt", "") or "").strip()
    image_reference_path = str(payload.get("image_reference_path", "") or "").strip().strip('"')
    image_reference_data = str(payload.get("image_reference_data", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    subject_context = str(payload.get("subject_context", "") or "").strip()
    location_context = str(payload.get("location_context", "") or "").strip()
    no_character_present = bool(payload.get("no_character_present") or payload.get("no_subject") or payload.get("no_visible_subject"))
    text_runner = _llm_runner_from_payload(payload)
    if not model_file and text_runner != "lm_studio":
        raise ValueError("Choose a T2V Gemma model first.")
    if model_file and text_runner != "lm_studio" and not model_file.lower().endswith(".gguf"):
        raise ValueError("The T2V model field is not a GGUF model.")
    if not scene_prompt:
        raise ValueError("Create or paste a T2I/concept prompt first.")

    image = None
    has_image_reference = False
    if image_reference_data:
        image = _image_from_data_url(image_reference_data).convert("RGB")
        has_image_reference = True
    elif image_reference_path:
        image_path = _resolve_existing_file(image_reference_path, "T2V Gemma image reference")
        image = Image.open(image_path).convert("RGB")
        has_image_reference = True
    if has_image_reference and text_runner != "lm_studio" and not model_file:
        raise ValueError("Choose a T2V vision Gemma model first.")
    if has_image_reference:
        max_height = 512
        if image.height > max_height:
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
            width = max(1, int(image.width * (max_height / max(1, image.height))))
            image = image.resize((width, max_height), resample)

    theme_style = _read_text_file(payload.get("theme_style_path", ""), "Theme/style file")
    story_idea = _read_text_file(payload.get("story_idea_path", ""), "Story idea file")
    subject_scene = _read_text_file(payload.get("subject_scene_path", ""), "Subject/scene file")
    context_parts = []
    if no_character_present:
        context_parts.append("Subject visibility:\nNo main character, singer, performer, person, mapped subject, or character reference is present in this scene. Use location, props, objects, atmosphere, and camera motion instead.")
    elif subject_scene:
        context_parts.append(f"Subject/scene:\n{subject_scene}")
    if subject_context and not no_character_present:
        context_parts.append(f"Mapped scene character(s):\n{subject_context}")
    if location_context:
        context_parts.append(f"Mapped scene location:\n{location_context}")
    if theme_style:
        context_parts.append(f"Theme/style:\n{theme_style}")
    if story_idea:
        context_parts.append(f"Story idea:\n{story_idea}")
    if context_parts:
        user_notes = "\n\n".join(context_parts + ([f"Segment motion notes:\n{user_notes}"] if user_notes else []))

    image_guidance = (
        "Use the provided reference image only to guide pose, framing, composition, mood, visible styling, or other user-requested visual details. "
        "Do not describe it as a reference image in the final prompt.\n\n"
        if has_image_reference else ""
    )
    prompt = (
        f"{_T2V_INSTRUCTIONS}\n\n"
        f"{image_guidance}"
        f"Scene concept:\n{scene_prompt}\n\n"
        f"User motion/camera notes:\n{user_notes or 'Create cinematic camera movement and natural subject/environment motion that fits the scene.'}"
    )

    llm = VRGDG_SuperGemmaGGUFChat() if has_image_reference and text_runner != "lm_studio" else None
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else ""
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.7)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 4000)
    unload_after = bool(payload.get("unload_after", True))

    try:
        if has_image_reference and text_runner == "lm_studio":
            text = _run_lm_studio_vision(
                payload,
                prompt,
                [image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            run_info = {
                "runner": "lm_studio_vision",
                "used_model": str(payload.get("lmstudio_model") or "").strip(),
                "unloaded": False,
            }
        elif has_image_reference:
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
                pil_images=[image],
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            run_info = {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}
        else:
            text, run_info = _run_builder_text_llm(
                payload,
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                label="T2V Gemma",
            )
        text = _clean_gemma_prompt_text(text)
        text = _repair_and_validate_builder_gemma_prompt(payload, text, "T2V")
        return {
            "prompt": text,
            "used_model": run_info.get("used_model", model_path if has_image_reference else ""),
            "used_mmproj": mmproj_path,
            "runner": run_info.get("runner", "builtin"),
            "used_image_reference": has_image_reference,
            "unloaded": run_info.get("unloaded", unload_after),
        }
    finally:
        if llm and has_image_reference and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _video_prompt_enhancement_instructions(payload):
    mode_label = str(payload.get("mode_label") or "I2V").strip().upper()
    draft_prompt = str(payload.get("draft_prompt") or "").strip()
    scene_prompt = str(payload.get("t2i_prompt") or payload.get("scene_prompt") or "").strip()
    user_notes = str(payload.get("user_notes") or "").strip()
    lyric_text = str(payload.get("lyric_text") or "").strip()
    singers_raw = payload.get("singers") or []
    if isinstance(singers_raw, str):
        singers = [item.strip() for item in singers_raw.split(",") if item.strip()]
    elif isinstance(singers_raw, list):
        singers = [str(item or "").strip() for item in singers_raw if str(item or "").strip()]
    else:
        singers = []
    no_vocal = bool(payload.get("no_vocal") or payload.get("instrumental") or payload.get("broll"))
    no_character_present = bool(payload.get("no_character_present") or payload.get("no_subject") or payload.get("no_visible_subject"))
    has_multiple_singers = len(singers) > 1
    has_one_singer = len(singers) == 1
    singer_text = ", ".join(singers)

    if no_character_present:
        template = (
            "Use this final prompt shape:\n"
            "The [main visual focus: location, prop, object, environment, architecture, weather, light, or atmosphere] in [location/environment] during [time/weather/lighting]. "
            "[Main visual focus] [clear visible action or environmental motion]. [Objects, props, fabric, water, plants, particles, or lighting] move naturally if visible. "
            "The camera [camera motion] while keeping the location or main object clearly framed and visible. "
            "The environment [visible motion/reaction].\n\n"
            "No-character rule: do not include, mention, imply, show, or describe any main character, singer, performer, person, mapped subject, or character reference in the final prompt."
        )
    elif no_vocal:
        template = (
            "Use this final prompt shape:\n"
            "The [subject or main visual focus] in [location/environment] during [time/weather/lighting]. "
            "[Subject or main visual focus] [visible action only]. [Clothing, hair, objects, or props] move naturally if visible. "
            "The camera [camera motion] while keeping the main visual focus clearly framed and visible. "
            "The environment [visible motion/reaction].\n\n"
            "No-vocal rule: do not mention singing, lip-syncing, vocals, lyric, mouth movement, instrumental status, or no-vocal status in the final prompt."
        )
    elif has_multiple_singers:
        template = (
            "Use this final prompt shape:\n"
            f"The [subjects: {singer_text}] in [location/environment] during [time/weather/lighting]. "
            f"[Subjects] are singing with passion, clearly moving their mouths in sync with the lyric \"{lyric_text}\", "
            "with expressive facial emotion, head movement, and strong visible performance gestures. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping all singers clearly framed and visible. "
            "The environment [visible motion/reaction]."
        )
    elif has_one_singer:
        template = (
            "Use this final prompt shape:\n"
            f"The [subject: {singer_text}] in [location/environment] during [time/weather/lighting]. "
            f"[Subject] is singing with passion, clearly moving their mouth in sync with the lyric \"{lyric_text}\", "
            "with expressive facial emotion, head movement, and strong visible performance gestures. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping the subject clearly framed and visible. "
            "The environment [visible motion/reaction]."
        )
    elif lyric_text:
        template = (
            "Use this final prompt shape:\n"
            f"The [visible subject] in [location/environment] during [time/weather/lighting]. "
            f"The visible subject is singing with passion, clearly moving their mouth in sync with the lyric \"{lyric_text}\", "
            "with expressive facial emotion, head movement, and strong visible performance gestures. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping the subject clearly framed and visible. "
            "The environment [visible motion/reaction]."
        )
    else:
        template = (
            "Use this final prompt shape:\n"
            "The [subject or main visual focus] in [location/environment] during [time/weather/lighting]. "
            "[Subject or main visual focus] [clear visible action]. [Clothing, hair, objects, or props] move naturally if visible. "
            "The camera [camera motion] while keeping the main visual focus clearly framed and visible. "
            "The environment [visible motion/reaction]."
        )

    return (
        f"Rewrite this draft {mode_label} video prompt into one stronger LTX-ready paragraph.\n\n"
        "Use the requested sentence shape, but replace every bracketed phrase with concrete details. Never output brackets or placeholder words.\n"
        "Preserve the subject, location, outfit, scene identity, and any user-requested camera/motion notes from the inputs.\n"
        "If the no-character rule is present, ignore any subject/character/singer from the draft and preserve only location, props, objects, atmosphere, and camera/motion notes.\n"
        "Only describe visible physical actions or visible scene motion. Do not mention invisible sensations, internal thoughts, symbolism, breath, heartbeat, sound-only details, or audio instructions.\n"
        "Do not use the word lip-sync. Use singing language only when this is a vocal scene.\n"
        "Do not add microphones unless they are visible or explicitly requested.\n"
        "Do not add captions, text overlays, dialogue explanations, unrelated characters, new locations, markdown, labels, or explanations.\n"
        "Output one polished paragraph only.\n\n"
        f"{template}\n\n"
        f"Draft prompt:\n{draft_prompt}\n\n"
        f"Scene/T2I context:\n{scene_prompt or '(none)'}\n\n"
        f"User/video notes:\n{user_notes or '(none)'}\n\n"
        f"Lyric text:\n{lyric_text or '(none)'}\n\n"
        f"Singer(s):\n{singer_text or '(none)'}"
    )


def _enhance_builder_video_prompt(payload):
    draft_prompt = str(payload.get("draft_prompt") or "").strip()
    if not draft_prompt:
        raise ValueError("Draft video prompt is empty.")
    model_file = str(payload.get("model_file") or payload.get("repair_model_file") or "").strip()
    if model_file:
        payload = dict(payload)
        payload["model_file"] = model_file
    instruction = _video_prompt_enhancement_instructions(payload)
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("enhance_temperature") or 0.25),
        top_p=float(payload.get("enhance_top_p") or 0.9),
        max_new_tokens=int(payload.get("enhance_max_new_tokens") or 1200),
        label="I2V prompt enhancement",
    )
    text = _clean_gemma_prompt_text(text)
    text = _repair_and_validate_builder_gemma_prompt(payload, text, str(payload.get("mode_label") or "I2V"))
    return {
        "prompt": text,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", bool(payload.get("unload_after", True))),
    }


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


def _combine_story_reference_batch(images, cell_size=512):
    if not images:
        raise ValueError("At least one Story reference image is required.")
    batch = list(images[:4])
    gap = 16
    columns = 1 if len(batch) == 1 else 2
    rows = int(math.ceil(len(batch) / columns))
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    canvas = Image.new("RGB", ((columns * cell_size) + (gap * (columns - 1)), (rows * cell_size) + (gap * (rows - 1))), (20, 20, 20))
    for index, image in enumerate(batch):
        image = image.convert("RGB")
        scale = min(cell_size / max(1, image.width), cell_size / max(1, image.height))
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        resized = image.resize((width, height), resample)
        column = index % columns
        row = index // columns
        cell_x = column * (cell_size + gap)
        cell_y = row * (cell_size + gap)
        canvas.paste(resized, (cell_x + ((cell_size - width) // 2), cell_y + ((cell_size - height) // 2)))
    return canvas


def _clear_comfy_model_memory():
    result = {
        "comfy_loaded_before": None,
        "comfy_loaded_after": None,
        "comfy_cleanup_calls": [],
        "comfy_cleanup_errors": [],
        "torch_cuda_cache_cleared": False,
    }
    try:
        import comfy.model_management as model_management
        import torch

        loaded_models = getattr(model_management, "loaded_models", None)
        if callable(loaded_models):
            try:
                result["comfy_loaded_before"] = len(loaded_models())
            except Exception as exc:
                result["comfy_cleanup_errors"].append(f"loaded_models before: {exc}")

        unload_all_models = getattr(model_management, "unload_all_models", None)
        if callable(unload_all_models):
            unload_all_models()
            result["comfy_cleanup_calls"].append("unload_all_models")

        cleanup_models_gc = getattr(model_management, "cleanup_models_gc", None)
        if callable(cleanup_models_gc):
            cleanup_models_gc()
            result["comfy_cleanup_calls"].append("cleanup_models_gc")

        cleanup_models = getattr(model_management, "cleanup_models", None)
        if callable(cleanup_models):
            cleanup_models()
            result["comfy_cleanup_calls"].append("cleanup_models")

        soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
        if callable(soft_empty_cache):
            try:
                soft_empty_cache(force=True)
            except TypeError:
                soft_empty_cache()
            result["comfy_cleanup_calls"].append("soft_empty_cache")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
            result["torch_cuda_cache_cleared"] = True

        gc.collect()
        if callable(loaded_models):
            try:
                result["comfy_loaded_after"] = len(loaded_models())
            except Exception as exc:
                result["comfy_cleanup_errors"].append(f"loaded_models after: {exc}")
    except Exception as exc:
        result["comfy_cleanup_errors"].append(str(exc))
    return result


def _clear_builder_memory_direct():
    from .LLM import _clear_vrgdg_llm_caches

    comfy_result = _clear_comfy_model_memory()
    llm_result = _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
    gc.collect()
    cleanup_calls = ", ".join(comfy_result.get("comfy_cleanup_calls") or []) or "none"
    cleanup_errors = "; ".join(comfy_result.get("comfy_cleanup_errors") or [])
    return {
        "message": (
            "Memory cleanup finished.\n"
            f"Comfy cleanup calls: {cleanup_calls}\n"
            f"Comfy loaded models: {comfy_result.get('comfy_loaded_before')} -> {comfy_result.get('comfy_loaded_after')}\n"
            f"GGUF models unloaded: {llm_result.get('gguf_models_unloaded', 0)}\n"
            f"HF pipelines unloaded: {llm_result.get('hf_pipelines_unloaded', 0)}\n"
            f"CUDA cache cleared: {bool(llm_result.get('cuda_cache_cleared') or comfy_result.get('torch_cuda_cache_cleared'))}"
            + (f"\nCleanup warnings: {cleanup_errors}" if cleanup_errors else "")
        ),
        **comfy_result,
        **llm_result,
    }


def _generate_builder_reference_description(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    reference_type = str(payload.get("reference_type") or "subject").strip().lower()
    name_hint = str(payload.get("name") or "").strip()
    subject_label = re.sub(r"\s+", " ", name_hint).strip()
    if subject_label.lower().startswith("the "):
        subject_label = subject_label.lower()
    object_reference_types = {"prop", "object", "vehicle", "creature", "animal", "outfit", "style", "environment", "other"}
    if reference_type not in {"subject", "character", "location", *object_reference_types}:
        reference_type = "subject"
    if not model_file:
        raise ValueError("Choose a Gemma vision model first.")
    image = _image_from_prompt_payload(payload.get("image_path", ""), payload.get("image_data", ""), "Reference image")

    if reference_type in {"subject", "character"}:
        instruction = (
            "Look at the image and write one concise character appearance description.\n"
            "Output only the description, one paragraph, no markdown, no label, no bullet points.\n"
            "Describe only the character's full visible appearance: hair, makeup, accessories, jewelry, clothing, outfit materials, colors, shoes, and distinctive visual identity details.\n"
            "Do not mention skin color, skin tone, ethnicity, race, or complexion.\n"
            "Do not describe the background, location, pose, camera angle, facial expression, mood, action, or what the character is doing.\n"
            "Do not invent hidden or unseen details.\n"
            "Keep it under 100 words."
        )
        if subject_label:
            instruction += (
                f"\nRefer to the subject as {subject_label}. "
                f"Do not call them the character or the subject in the final description."
            )
    elif reference_type == "location":
        instruction = (
            "Look at the image and write one concise location/environment description.\n"
            "Output only the description, one paragraph, no markdown, no label, no bullet points.\n"
            "Describe the place itself: environment, architecture, layout, major objects, materials, colors, lighting, atmosphere, and visible setting details.\n"
            "Do not describe a main character, pose, performance, camera angle, or story action.\n"
            "Do not invent hidden or unseen details.\n"
            "Keep it under 100 words."
        )
        if name_hint:
            instruction += f"\nUse this label only as the location name if needed: {name_hint}"
    else:
        label = reference_type.replace("_", " ")
        instruction = (
            f"Look at the image and write one concise {label} reference description.\n"
            "Output only the description, one paragraph, no markdown, no label, no bullet points.\n"
            "Describe only the visible reference item: shape, form, materials, colors, markings, texture, scale cues, construction, accessories, and distinctive visual identity details.\n"
            "Do not describe the background, location, pose, camera angle, facial expression, mood, action, story meaning, or a person unless the reference item itself is a person.\n"
            "Do not invent hidden or unseen details.\n"
            "Keep it under 100 words."
        )
        if name_hint:
            instruction += f"\nUse this label only as the reference name if needed: {name_hint}"

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file)
    n_ctx = int(payload.get("n_ctx") or 2048)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.2)
    top_p = float(payload.get("top_p") or 0.9)
    max_new_tokens = int(payload.get("max_new_tokens") or 180)
    seed = payload.get("seed")
    clear_before_load = bool(payload.get("clear_before_load", False))
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
        def clean_reference_description(raw_text):
            text = _clean_visual_gemma_text(raw_text)
            text = re.sub(r"^\s*(character|subject|location|description)\s*:\s*", "", text, flags=re.I).strip()
            if reference_type in {"subject", "character"}:
                if subject_label:
                    text = re.sub(r"\bthe character\b", subject_label, text, flags=re.I)
                    text = re.sub(r"\bthe subject\b", subject_label, text, flags=re.I)
                    text = re.sub(r"\ba character\b", subject_label, text, flags=re.I)
                    text = re.sub(r"\ba subject\b", subject_label, text, flags=re.I)
                text = re.sub(
                    r"\b(?:with|has|having|featuring)?\s*(?:very\s+|pale\s+|fair\s+|light\s+|medium\s+|tan\s+|tanned\s+|olive\s+|brown\s+|dark\s+|deep\s+|warm\s+|cool\s+|golden\s+|porcelain\s+|dusky\s+|caramel\s+|bronze\s+|dark-skinned\s+|light-skinned\s+)+(?:skin|skin tone|complexion)\b,?\s*(?:and\s+)?",
                    "",
                    text,
                    flags=re.I,
                )
                text = re.sub(r"\b(?:skin|skin tone|complexion)\s*(?:is|appears|looks)\s+[^,.]+,?\s*(?:and\s+)?", "", text, flags=re.I)
                text = re.sub(r"\s+,", ",", text)
                text = re.sub(r"(?:^|\.\s*)and\s+", "", text, flags=re.I).strip()
                text = re.sub(r"\s{2,}", " ", text).strip(" ,")
            words = text.split()
            if len(words) > 100:
                text = " ".join(words[:100]).rstrip(" ,.;:") + "."
            if not text:
                raise ValueError("Gemma returned an empty reference description.")
            _validate_reference_description(text, reference_type)
            return text

        last_error = None
        text = ""
        for attempt in range(2):
            attempt_instruction = instruction
            if attempt:
                attempt_instruction += (
                    "\n\nPrevious output was rejected because it repeated words or was not a usable description. "
                    "Write a normal visual description with varied concrete nouns. Do not repeat any word more than twice."
                )
            raw_text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[image],
                instruction_text=attempt_instruction,
                temperature=0.05 if attempt else temperature,
                top_p=0.75 if attempt else top_p,
                max_new_tokens=max_new_tokens,
                seed=int(seed) if seed is not None else None,
            )
            try:
                text = clean_reference_description(raw_text)
                last_error = None
                break
            except ValueError as exc:
                last_error = exc
        if last_error:
            raise last_error
        return {"description": text, "used_model": model_path, "used_mmproj": mmproj_path, "unloaded": unload_after}
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


def _generate_flux_klein_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    if not model_file:
        raise ValueError("Choose a Gemma vision model first.")

    ingredients = payload.get("image_ingredients") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("Image ingredients must be a list.")
    reference_context = payload.get("reference_context") or {}
    if not isinstance(reference_context, dict):
        reference_context = {}
    has_subject_reference = bool(reference_context.get("has_subject_reference"))
    has_location_reference = bool(reference_context.get("has_location_reference"))
    subject_description = str(reference_context.get("subject_description", "") or "").strip()
    location_name = str(reference_context.get("location_name", "") or "").strip()
    location_description = str(reference_context.get("location_description", "") or "").strip()
    images = []
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        images.append(_image_from_prompt_payload(item.get("path", ""), item.get("data", ""), f"Image ingredient {index}"))
    combined_image = _combine_flux_ingredient_images(images)
    reference_rules = []
    if has_subject_reference:
        subject_line = "Use the subject reference for the main subject identity, face/body details, outfit, and visible character consistency."
        if subject_description:
            subject_line += f" Subject description: {subject_description}"
        reference_rules.append(subject_line)
    if has_location_reference:
        location_line = "Use the mapped location reference as the required setting/background. Do not replace it with a different location from the concept or notes."
        location_details = "; ".join(part for part in (location_name, location_description) if part)
        if location_details:
            location_line += f" Mapped location: {location_details}"
        reference_rules.append(location_line)
    reference_text = ""
    if reference_rules:
        reference_text = (
            "\nReference Builder priorities:\n"
            + "\n".join(f"- {rule}" for rule in reference_rules)
            + "\n- Use the user's notes/concept for action, pose, mood, lighting, story beat, and details after respecting the reference priorities.\n"
        )
    instruction = (
        "Create one concise Flux/Klein image prompt from the user input and available reference context.\n"
        "Output one normal paragraph, not sections, not markdown, not labels, not explanations.\n\n"
        "The image input contains the available reference images/visual ingredients. These may include a character, background, props, style references, or other visual ingredients.\n"
        f"{reference_text}"
        "Prompt style:\n"
        "- Start with: Using the provided character reference and location reference, create...\n"
        "- If only a character reference is available, start with: Using the provided character reference, create...\n"
        "- If only a location reference is available, start with: Using the provided location reference, create...\n"
        "- Use a clear cinematic shot type such as close-up, profile close-up, medium close-up, upper body shot, waist-up shot, three-quarter shot, seated shot, over-the-shoulder shot, or low-angle portrait.\n"
        "- Use the user's scene/concept notes as the main creative direction.\n"
        "- Preserve the character identity from the character reference: face, hair, outfit, makeup, and overall identity.\n"
        "- Preserve the location identity from the location reference: environment, architecture, layout, atmosphere, and major visible setting details.\n"
        "- Create a new camera angle, new pose, and new composition.\n"
        "- Do not paste the character into the location image.\n"
        "- Do not copy the character reference pose, full-body standing pose, studio background, panel layout, crop, camera angle, or lens distance.\n"
        "- Do not copy the exact location reference camera angle, framing, perspective, or composition.\n"
        "- Avoid full-body walking or standing shots unless the user specifically asks for them.\n"
        "- Prefer intimate cinematic compositions when no shot type is specified: close-up, medium close-up, profile, upper body, shallow depth of field, foreground framing, soft bokeh, rim light, atmospheric lighting.\n"
        "- Keep it cinematic, detailed, and visually specific.\n"
        "- Keep the prompt visually specific and practical for image generation.\n"
        "- Do not include captions, text overlays, dialogue, markdown, labels, bullet points, or section headers.\n"
        "- Keep the prompt under 120 words.\n\n"
        "Good output examples:\n"
        "Using the provided character reference and location reference, create a close-up profile shot of the woman in the misty forest. Focus on her expression and the intricate details of her crown while the pale trees and fog appear softly blurred in the background. Use a cool moody palette, atmospheric haze, shallow depth of field, dramatic rim lighting, and high cinematic detail.\n"
        "Using the provided character reference and location reference, create an intimate upper body shot of the woman framed by gnarled forest branches. Preserve her identity, hair, outfit, and crown from the character reference while using the forest reference for the white fibrous trees, mist, and eerie atmosphere. New pose, new camera angle, soft bokeh, high cinematic quality.\n\n"
        f"User input:\n{user_notes or 'Create a new image using the available reference images.'}"
    )

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file)
    # Flux/Klein only needs one short prompt, and vision GGUF context is expensive.
    # Keep this lower than the broader Gemma prompt tools to reduce crash risk.
    n_ctx = int(payload.get("n_ctx") or 2048)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 350)
    seed = payload.get("seed")
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
            seed=int(seed) if seed is not None else None,
        )
        text = _clean_visual_gemma_text(text)
        text = _repair_and_validate_builder_gemma_prompt(payload, text, "Flux/Klein")
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


def _analyze_builder_story_references(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    if not model_file:
        raise ValueError("Choose a Gemma vision model first.")
    ingredients = payload.get("image_ingredients") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("Story reference images must be a list.")
    images = []
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        images.append(_image_from_prompt_payload(item.get("path", ""), item.get("data", ""), f"Story reference image {index}"))
    if not images:
        raise ValueError("Add at least one Story Builder reference image first.")
    batches = [images[index:index + 4] for index in range(0, len(images), 4)]
    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file)
    n_ctx = int(payload.get("n_ctx") or 4096)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 500)
    unload_after = bool(payload.get("unload_after", True))
    try:
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
        batch_notes = []
        for batch_index, batch in enumerate(batches, start=1):
            combined_image = _combine_story_reference_batch(batch, cell_size=512)
            instruction = (
                "Analyze these reference images for a music video Story Builder. "
                "Each image has been resized into a 512px tile; this batch contains at most four images. "
                "Write compact reusable planning notes for a text-only agent. "
                "Identify likely singers/characters, clothing, faces/hair/body details, location ideas, props, color palette, lighting, mood, genre, and overall aesthetic. "
                "Do not write an image generation prompt. Do not mention image grids or panels. "
                "Use short labeled lines. Keep it under 180 words.\n\n"
                f"Batch {batch_index} of {len(batches)}.\n"
                f"User notes:\n{user_notes or 'Summarize the characters, locations, style, and aesthetic shown in the images.'}"
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
            if _looks_like_gemma_repeat_failure(text):
                raise ValueError(f"Gemma returned repeated/thought junk instead of usable Story reference notes for batch {batch_index}.")
            if text.strip():
                prefix = f"Reference batch {batch_index}: " if len(batches) > 1 else ""
                batch_notes.append(prefix + text.strip())
        return {"notes": "\n\n".join(batch_notes).strip(), "used_model": model_path, "used_mmproj": mmproj_path, "unloaded": unload_after, "batches": len(batches)}
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


def _generate_nb_image_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    text_runner = _llm_runner_from_payload(payload)
    if text_runner != "lm_studio" and not model_file:
        raise ValueError("Choose a NanoBanana Gemma vision model first.")

    ingredients = payload.get("image_ingredients") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("NanoBanana reference images must be a list.")
    images = []
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        images.append(_image_from_prompt_payload(item.get("path", ""), item.get("data", ""), f"NanoBanana reference image {index}"))
    reference_context = payload.get("reference_context") or {}
    if not isinstance(reference_context, dict):
        reference_context = {}

    context_parts = []
    subject_description = str(reference_context.get("subject_description", "") or "").strip()
    location_name = str(reference_context.get("location_name", "") or "").strip()
    location_description = str(reference_context.get("location_description", "") or "").strip()
    has_subject_reference = bool(reference_context.get("has_subject_reference"))
    has_location_reference = bool(reference_context.get("has_location_reference"))

    if subject_description:
        context_parts.append(f"Subject description:\n{subject_description}")
    if location_name or location_description:
        context_parts.append(f"Location reference:\n{location_name}\n{location_description}".strip())
    if user_notes:
        context_parts.append(f"User input:\n{user_notes}")

    reference_flags = []
    if has_subject_reference:
        reference_flags.append("A character reference image is available.")
    if has_location_reference:
        reference_flags.append("A scene/location reference image is available.")
    if reference_flags:
        context_parts.append("\n".join(reference_flags))

    has_images = bool(images)
    has_unmapped_reference_images = has_images and not has_subject_reference and not has_location_reference

    def _cleanup_nb_reference_claims(text):
        text = str(text or "")
        fallback_reference = "Using the provided reference image" if has_images else "Create"
        if not has_subject_reference and not has_location_reference:
            text = re.sub(
                r"\bUsing the provided character reference and location reference\b",
                fallback_reference,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bUsing the provided location reference and character reference\b",
                fallback_reference,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bUsing the provided (?:character|location|scene) reference\b",
                fallback_reference,
                text,
                flags=re.IGNORECASE,
            )
        elif not has_location_reference:
            text = re.sub(
                r"\bUsing the provided character reference and location reference\b",
                "Using the provided character reference",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bUsing the provided location reference and character reference\b",
                "Using the provided character reference",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s+and (?:the\s+)?(?:provided\s+)?(?:location|scene) reference(?: image)?\b", "", text, flags=re.IGNORECASE)
        elif not has_subject_reference:
            text = re.sub(
                r"\bUsing the provided character reference and location reference\b",
                "Using the provided location reference",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bUsing the provided location reference and character reference\b",
                "Using the provided location reference",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s+and (?:the\s+)?(?:provided\s+)?character reference(?: image)?\b", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\bcharacter reference(?: image)?\s+and\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _ensure_nb_reference_opening(text):
        text = str(text or "").strip()
        if not text:
            return text
        if has_subject_reference and has_location_reference:
            opening = "Using the provided character reference and location reference"
        elif has_subject_reference:
            opening = "Using the provided character reference"
        elif has_location_reference:
            opening = "Using the provided location reference"
        else:
            return text
        if re.search(r"\bUsing the provided (?:character|location|scene|reference image)", text, flags=re.IGNORECASE):
            return text
        if re.match(r"^(?:create|make|generate)\b", text, flags=re.IGNORECASE):
            text = re.sub(r"^(?:create|make|generate)\b\s*", "", text, count=1, flags=re.IGNORECASE)
        return f"{opening}, create {text[:1].lower()}{text[1:] if len(text) > 1 else ''}".strip()

    if has_subject_reference and has_location_reference:
        reference_prompt_rules = (
            "- Start by mentioning both the provided character reference and location reference.\n"
            "- Preserve the character identity from the character reference: face, hair, outfit, makeup, and overall identity.\n"
            "- Preserve the location identity from the location reference: environment, architecture, layout, atmosphere, and major visible setting details.\n"
            "- Use the user's scene/concept notes for action, pose, camera, mood, and story beat.\n"
            "- Do not paste the character into the location image.\n"
            "- Do not copy the exact pose, crop, camera angle, perspective, or composition from either reference.\n"
        )
        example_text = (
            "Using the provided character reference and location reference, create a close-up profile shot of the woman in the misty forest. "
            "Preserve her identity, hair, outfit, and crown from the character reference while using the forest reference for the white fibrous trees, mist, and eerie atmosphere. "
            "Use a new pose, new camera angle, soft bokeh, atmospheric haze, and dramatic rim lighting."
        )
    elif has_subject_reference:
        reference_prompt_rules = (
            "- Start by mentioning only the provided character reference. Do not mention a location reference.\n"
            "- Preserve the character identity from the character reference: face, hair, outfit, makeup, and overall identity.\n"
            "- Create the setting, background, atmosphere, and location from the user's scene/concept notes.\n"
            "- Do not copy the character reference pose, studio background, crop, camera angle, or lens distance.\n"
        )
        example_text = (
            "Using the provided character reference, create an intimate upper body shot of the woman in a misty white forest built from the scene concept. "
            "Preserve her identity, blonde hair, lace outfit, and delicate facial details while placing her among pale gnarled trees, soft fog, shallow depth of field, and ethereal rim lighting."
        )
    elif has_location_reference:
        reference_prompt_rules = (
            "- Start by mentioning only the provided location reference. Do not mention a character reference.\n"
            "- Use the location reference for environment, architecture, layout, atmosphere, and major visible setting details.\n"
            "- Create any subject, pose, outfit, camera, and story details from the user's scene/concept notes.\n"
            "- Do not copy the exact location reference camera angle, framing, perspective, or composition.\n"
        )
        example_text = (
            "Using the provided location reference, create a cinematic medium shot of the scene's subject moving through the misty forest. "
            "Preserve the pale fibrous trees, narrow fog-covered path, and eerie atmosphere from the location reference while creating a new camera angle, new subject pose, soft bokeh, and dramatic haze."
        )
    elif has_unmapped_reference_images:
        reference_prompt_rules = (
            "- You may use the provided reference image or images only as loose visual guidance.\n"
            "- Do not call them character references or location references unless the context explicitly says that.\n"
            "- Use the user's scene/concept notes as the main source of subject, setting, action, pose, and mood.\n"
            "- Create a new camera angle, new pose, and new composition.\n"
        )
        example_text = (
            "Create a cinematic medium close-up from the scene concept, using the provided visual reference only as loose style guidance. "
            "Build the subject, setting, lighting, and atmosphere from the notes, with soft bokeh, layered haze, a new composition, and high cinematic detail."
        )
    else:
        reference_prompt_rules = (
            "- Do not mention provided references or reference images.\n"
            "- Treat this as normal text-to-image prompt writing from the user's scene/concept notes.\n"
            "- Create the subject, setting, action, pose, camera, lighting, and atmosphere from the notes.\n"
        )
        example_text = (
            "Create a cinematic medium close-up in a vast cosmic void filled with drifting pearlescent particles and soft ethereal light. "
            "Use a new composition, shallow depth of field, gentle atmospheric haze, luminous highlights, and a quiet dreamlike mood."
        )
    instruction = (
        "Create one concise NanoBanana image prompt from the user input and available reference context.\n"
        "Output one normal paragraph, not sections, not markdown, not labels, not explanations.\n\n"
        "Prompt style:\n"
        f"{reference_prompt_rules}"
        "- Use a clear cinematic shot type such as close-up, profile close-up, medium close-up, upper body shot, waist-up shot, three-quarter shot, seated shot, over-the-shoulder shot, or low-angle portrait.\n"
        "- Use the user's scene/concept notes as the main creative direction.\n"
        "- Create a new camera angle, new pose, and new composition.\n"
        "- Avoid full-body walking or standing shots unless the user specifically asks for them.\n"
        "- Prefer intimate cinematic compositions when no shot type is specified: close-up, medium close-up, profile, upper body, shallow depth of field, foreground framing, soft bokeh, rim light, atmospheric lighting.\n"
        "- Keep the prompt visually specific and practical for image generation.\n"
        "- Do not include captions, text overlays, dialogue, markdown, labels, bullet points, or section headers.\n\n"
        f"Good output example:\n{example_text}\n\n"
        f"{chr(10).join(context_parts) if context_parts else 'User input: Create a new image from the notes.'}"
    )

    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 900)
    seed = payload.get("seed")
    clear_before_load = bool(payload.get("clear_before_load", True))
    unload_after = bool(payload.get("unload_after", True))
    if not has_images:
        text, info = _run_builder_text_llm(
            payload,
            instruction,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            label="NanoBanana text Gemma",
            preserve_paragraphs=False,
        )
        text = _clean_lm_studio_plain_text(text)
        if not text:
            raise ValueError("NanoBanana Gemma returned an empty prompt.")
        text = _cleanup_nb_reference_claims(text)
        text = _ensure_nb_reference_opening(text)
        text = _repair_and_validate_builder_gemma_prompt(payload, text, "NanoBanana")
        return {"prompt": text, **info}

    combined_image = _combine_flux_ingredient_images(images)
    llm = VRGDG_SuperGemmaGGUFChat() if text_runner != "lm_studio" else None
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else str(payload.get("lmstudio_model") or "").strip()
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    try:
        if text_runner == "lm_studio":
            text = _run_lm_studio_vision(
                payload,
                instruction,
                [combined_image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            info = {
                "runner": "lm_studio_vision",
                "used_model": model_path,
                "used_mmproj": "",
                "unloaded": False,
            }
        else:
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
                seed=int(seed) if seed is not None else None,
            )
            info = {
                "runner": "builtin",
                "used_model": model_path,
                "used_mmproj": mmproj_path,
                "unloaded": unload_after,
            }
    finally:
        if llm and unload_after:
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
    text = _clean_lm_studio_plain_text(text)
    text = re.sub(r"(?im)^CAMERA\s+COMPOSI+TION\s*\(PRIORITY\)", "CAMERA COMPOSITION (PRIORITY)", text)
    text = re.sub(r"(?im)^CHARACTER\s+REFEREN[CC]E\b", "CHARACTER REFERENCE", text)
    text = re.sub(r"(?im)^SCE+NE\s+REFEREN[CC]E\b", "SCENE REFERENCE", text)
    text = re.sub(r"\breference\s+imagae\b", "reference image", text, flags=re.IGNORECASE)
    text = re.sub(r"\breference\s+imagaes\b", "reference images", text, flags=re.IGNORECASE)
    text = _cleanup_nb_reference_claims(text)
    text = _ensure_nb_reference_opening(text)
    if not text:
        raise ValueError("NanoBanana Gemma returned an empty prompt.")
    text = _repair_and_validate_builder_gemma_prompt(payload, text, "NanoBanana")
    return {"prompt": text, **info}


def _generate_flux_reference_location_map(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file and _llm_runner_from_payload(payload) != "lm_studio":
        raise ValueError("Choose a non-vision Gemma model first.")

    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for location mapping.")

    cleaned_scenes = []
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("id", "") or f"scene_{index}").strip()
        label = str(scene.get("label", "") or f"Scene {index}").strip()
        concept = str(scene.get("concept", "") or "").strip()
        notes = str(scene.get("notes", "") or "").strip()
        if concept or notes:
            cleaned_scenes.append({
                "id": scene_id,
                "label": label,
                "concept": concept,
                "notes": notes,
            })
    if not cleaned_scenes:
        raise ValueError("Scenes need lyrics, scene notes, concept prompts, or timeline notes before Gemma can map locations.")

    subject_scene = _clean_location_context_text(payload.get("subject_scene_text", ""))
    style_theme = _clean_location_context_text(payload.get("style_theme", ""))
    subject_context = _clean_location_context_text(payload.get("subject_context", ""))
    existing_locations = payload.get("existing_locations") or []
    if not isinstance(existing_locations, list):
        existing_locations = []
    normalized_existing_locations = []
    seen_existing = set()
    existing_location_lines = []
    for item in existing_locations:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        if name:
            key = name.lower()
            if key not in seen_existing:
                seen_existing.add(key)
                normalized_existing_locations.append({"name": re.sub(r"\s+", " ", name), "description": re.sub(r"\s+", " ", description)})
            existing_location_lines.append(f"- {name}" + (f": {description}" if description else ""))
    if not normalized_existing_locations:
        raise ValueError("Auto Map needs locations first. Click Extract Locations or add locations manually, then run Auto Map.")

    scene_lines = []
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_lines.append(
            f"Scene {index}\n"
            f"id: {scene['id']}\n"
            f"label: {scene['label']}\n"
            f"concept: {scene['concept']}\n"
            f"notes: {scene['notes']}"
        )

    numbered_locations = "\n".join(
        f"{index}={item['name']}" + (f" | {item['description']}" if item.get("description") else "")
        for index, item in enumerate(normalized_existing_locations, start=1)
    )
    previous_counts = _location_usage_counts_from_payload(payload, normalized_existing_locations)
    usage_lines = "\n".join(
        f"- {name}: already used {int(previous_counts.get(name, 0) or 0)} time(s)"
        for name in sorted(previous_counts, key=lambda item: (int(previous_counts.get(item, 0) or 0), item.lower()))
    )
    instruction = (
        "You are mapping music-video scenes to an existing location list.\n\n"
        "Choose the best existing location number for each scene using the scene lyric line, concept text, notes, visual mood, objects, and environment. "
        "First use a location clearly named or implied by the scene text. "
        "If the scene has no clear location, choose the closest fit from the location list based on emotional tone and visual atmosphere. "
        "Use locations that have not been used yet before repeating locations that were already used. "
        "Avoid repeating the same location across too many neighboring scenes when another listed location fits equally well.\n\n"
        "Output only simple lines in this exact format:\n"
        "Scene1=1\n"
        "Scene2=3\n"
        "Scene3=3\n\n"
        "Rules:\n"
        "- Every scene must get one line.\n"
        "- Use only location numbers from the list.\n"
        "- Prefer the least-used matching location when multiple locations fit.\n"
        "- If there are enough scenes, use every listed location at least once before heavy repeats.\n"
        "- Do not output JSON, markdown, bullets, explanations, names, or descriptions.\n"
        "- Do not invent new locations.\n\n"
        f"Optional extra context, if any:\n{subject_scene or '(none)'}\n\n"
        f"Already used locations before these scenes:\n{usage_lines or '(none)'}\n\n"
        f"Locations:\n{numbered_locations}\n\n"
        f"Scenes:\n\n{chr(10).join(scene_lines)}"
    )

    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.8)
    max_new_tokens = int(payload.get("max_new_tokens") or 5000)
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        label="Gemma",
    )
    normalized_locations = normalized_existing_locations
    normalized_map = _parse_scene_location_number_map(text, cleaned_scenes, normalized_locations)
    if not normalized_map:
        normalized_map = _fallback_location_map_by_overlap(cleaned_scenes, normalized_locations)
    else:
        fallback_map = _fallback_location_map_by_overlap(cleaned_scenes, normalized_locations)
        for scene in cleaned_scenes:
            normalized_map.setdefault(scene["id"], fallback_map[scene["id"]])
    normalized_map = _balance_location_map_by_usage(normalized_map, cleaned_scenes, normalized_locations, previous_counts)
    return {
        "locations": normalized_locations,
        "scene_map": normalized_map,
        "raw_text": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _generate_flux_reference_locations(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file and _llm_runner_from_payload(payload) != "lm_studio":
        raise ValueError("Choose a non-vision Gemma model first.")
    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for location extraction.")
    cleaned_scenes = []
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        concept = str(scene.get("concept", "") or "").strip()
        notes = str(scene.get("notes", "") or "").strip()
        if concept or notes:
            cleaned_scenes.append({
                "id": str(scene.get("id", "") or f"scene_{index}").strip(),
                "label": str(scene.get("label", "") or f"Scene {index}").strip(),
                "concept": concept,
                "notes": notes,
            })
    if not cleaned_scenes:
        raise ValueError("Scenes need lyrics, scene notes, concept prompts, or timeline notes before Gemma can extract locations.")

    subject_scene = _clean_location_context_text(payload.get("subject_scene_text", ""))
    style_theme = _clean_location_context_text(payload.get("style_theme", ""))
    subject_context = _clean_location_context_text(payload.get("subject_context", ""))
    existing_locations = payload.get("existing_locations") or []
    existing_lines = []
    if isinstance(existing_locations, list):
        for item in existing_locations:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            description = str(item.get("description", "") or "").strip()
            if name:
                existing_lines.append(f"- {name}" + (f": {description}" if description else ""))

    scene_lines = []
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_lines.append(
            f"Scene {index}: {scene['label']}\n"
            f"concept: {scene['concept']}\n"
            f"notes: {scene['notes']}"
        )

    instruction = (
        "Extract a short reusable location list for Flux/Klein or Nano B reference images.\n\n"
        "Use the scene concept prompts and scene notes as the source of truth. "
        "Optional extra context may be empty; if it is empty or missing, ignore it completely. "
        "Use character/reference descriptions and style/theme only to understand the visual world, era, mood, genre, and design language. "
        "Do not turn characters, clothing, props, accessories, or body details into location names. "
        "Find concrete physical places, sets, rooms, buildings, landscapes, or backgrounds that repeat or are useful as references. "
        "If the extra context includes locations not directly named in a concept prompt, include them when they fit the project.\n\n"
        "Output only simple lines in this exact format:\n"
        "1|location name|short visual description for a reference image\n"
        "2|location name|short visual description for a reference image\n\n"
        "Rules:\n"
        "- Do not output JSON, markdown, bullets, headings, or explanations.\n"
        "- Keep names short, like bedroom, foggy white forest, salt-flat desert, ruined opera house.\n"
        "- Every name must be an actual place where the subject could stand, walk, sit, perform, or be filmed.\n"
        "- Do not output props, objects, clothing, accessories, body parts, people, characters, creatures, or symbolic items as locations.\n"
        "- Descriptions must describe only the place/background: architecture, layout, surfaces, lighting, weather, atmosphere, era, and color.\n"
        "- Do not include characters or actions in descriptions. No bride, woman, man, face, hand, tooth, dress, razor, locket, veil, or similar subject/object details.\n"
        "- Reuse broad locations instead of creating one unique location for every scene.\n"
        "- 3 to 8 locations is usually enough unless the project clearly needs more.\n\n"
        f"Optional style/theme guidance:\n{style_theme or '(none)'}\n\n"
        f"Character/reference descriptions for style guidance only:\n{subject_context or '(none)'}\n\n"
        f"Optional extra context:\n{subject_scene or '(none)'}\n\n"
        f"Existing user locations:\n{chr(10).join(existing_lines) if existing_lines else '(none)'}\n\n"
        f"Scenes:\n\n{chr(10).join(scene_lines)}"
    )
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.2),
        top_p=float(payload.get("top_p") or 0.8),
        max_new_tokens=int(payload.get("max_new_tokens") or 2200),
        label="Gemma",
    )
    locations = _parse_location_lines(text)
    if not locations:
        locations = _parse_location_ideas_flexible(text)
    if not locations:
        try:
            data = _extract_json_object_from_text(_clean_visual_gemma_text(text))
            raw_locations = data.get("locations") if isinstance(data, dict) else data
            if isinstance(raw_locations, list):
                for item in raw_locations:
                    if isinstance(item, dict):
                        raw_name = item.get("name", "")
                        name = _normalize_location_name(raw_name)
                        description = re.sub(r"\s+", " ", str(item.get("description", "") or "").strip())
                        if name:
                            name, description = _clean_location_card(name, description, raw_name)
                            if _valid_location_card(name, description):
                                locations.append({"name": name, "description": description})
        except Exception:
            pass
    if not locations:
        retry_instruction = (
            "Return only reusable music video filming locations from the scene text below.\n"
            "Do not explain anything.\n"
            "Do not summarize the scenes.\n"
            "Write 3 to 12 locations.\n"
            "Write one location per line as a short bullet.\n"
            "Each bullet must be a concrete visual place/background where the subject could stand or move.\n"
            "Reject props, objects, clothing, accessories, people, body parts, symbolic items, and actions.\n"
            "Describe only the environment, not characters or objects from the lyrics.\n"
            "If any optional context is missing, invisible, empty, or unavailable, ignore that and use only the scene text.\n"
            "Never mention missing files, missing context, subjectsandscenes.txt, prompts, or instructions.\n"
            "Example format:\n"
            "- Abandoned motel pool, turquoise water under buzzing neon signs\n"
            "- Foggy pine road, wet asphalt and fading headlights\n\n"
            f"Optional style/theme guidance:\n{style_theme or '(none)'}\n\n"
            f"Character/reference descriptions for style guidance only:\n{subject_context or '(none)'}\n\n"
            f"Optional extra context:\n{subject_scene or '(none)'}\n\n"
            f"Existing user locations:\n{chr(10).join(existing_lines) if existing_lines else '(none)'}\n\n"
            f"Scenes:\n\n{chr(10).join(scene_lines)}"
        )
        retry_text, retry_info = _run_builder_text_llm(
            payload,
            retry_instruction,
            temperature=float(payload.get("temperature") or 0.35),
            top_p=float(payload.get("top_p") or 0.9),
            max_new_tokens=int(payload.get("max_new_tokens") or 2200),
            label="Gemma",
        )
        retry_locations = _parse_location_lines(retry_text) or _parse_location_ideas_flexible(retry_text)
        if retry_locations:
            text = retry_text
            run_info = retry_info
            locations = retry_locations
    if not locations:
        try:
            scout_payload = {
                **payload,
                "subject_scene_text": "",
                "style_theme": style_theme,
                "subject_context": subject_context,
                "lyrics_text": "\n\n".join(scene_lines),
                "user_input": "Create reusable location ideas from these music-video scene lines.",
                "temperature": float(payload.get("temperature") or 0.45),
                "top_p": float(payload.get("top_p") or 0.9),
                "max_new_tokens": int(payload.get("max_new_tokens") or 2200),
            }
            scout_result = _generate_wizard_locations_from_lyrics(scout_payload)
            locations = scout_result.get("locations") or []
            if locations:
                text = scout_result.get("raw_text", text)
                run_info = {
                    **run_info,
                    "used_model": scout_result.get("used_model", run_info.get("used_model", "")),
                    "runner": scout_result.get("runner", run_info.get("runner", "builtin")),
                    "unloaded": scout_result.get("unloaded", run_info.get("unloaded", True)),
                }
        except Exception:
            pass
    if not locations:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(preview) > 700:
            preview = preview[:697].rstrip() + "..."
        raise ValueError(f"Gemma did not return any usable locations. Raw response preview: {preview or '(empty)'}")
    deduped = []
    seen = set()
    for item in locations:
        raw_name = item.get("name", "")
        name, description = _clean_location_card(raw_name, item.get("description", ""), raw_name)
        if not _valid_location_card(name, description):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"name": name, "description": description})
    if not deduped:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(preview) > 700:
            preview = preview[:697].rstrip() + "..."
        raise ValueError(f"Gemma returned only non-location items. Raw response preview: {preview or '(empty)'}")
    return {
        "locations": deduped,
        "raw_text": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _generate_wizard_locations_from_lyrics(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file and _llm_runner_from_payload(payload) != "lm_studio":
        raise ValueError("Choose a non-vision Gemma model first.")
    lyrics_text = str(payload.get("lyrics_text", "") or payload.get("lyrics", "") or "").strip()
    user_input = str(payload.get("user_input", "") or payload.get("notes", "") or "").strip()
    style_theme = _clean_location_context_text(payload.get("style_theme", ""))
    subject_context = _clean_location_context_text(payload.get("subject_context", ""))
    if not lyrics_text:
        raise ValueError("Paste or create lyrics before creating locations from lyrics.")

    existing_locations = payload.get("existing_locations") or []
    existing_lines = []
    if isinstance(existing_locations, list):
        for item in existing_locations:
            if not isinstance(item, dict):
                continue
            name = re.sub(r"\s+", " ", str(item.get("name", "") or "").strip())
            description = re.sub(r"\s+", " ", str(item.get("description", "") or "").strip())
            if name:
                existing_lines.append(f"- {name}" + (f": {description}" if description else ""))

    instruction = (
        "You are a music video location scout.\n\n"
        "The user will provide song lyrics.\n\n"
        "Your task is to analyze the mood, imagery, setting clues, themes, and emotional tone of the lyrics, "
        "then generate a list of reusable filming locations where the main subject or character could be placed.\n\n"
        "Return only actual locations, sets, rooms, buildings, outdoor areas, roads, stages, landscapes, or environments.\n\n"
        "Use character/reference descriptions and style/theme only to understand the visual world, era, mood, genre, and design language. "
        "Do not turn characters, clothing, props, accessories, or body details into location names.\n\n"
        "Rules:\n\n"
        "Do not summarize the lyrics.\n"
        "Do not explain the song meaning.\n"
        "Do not quote long lyric sections.\n"
        "Focus on visual places that could realistically appear in a music video and hold the subject.\n"
        "Include literal locations from the lyrics and cinematic locations inspired by the mood, but they must still be real places.\n"
        "Do not output props, objects, clothing, accessories, body parts, people, characters, creatures, or symbolic items as locations.\n"
        "Bad location names: Vintage shaving kit, Ornate sugar bowl, Lace veil, Silver frame, Human tooth, Ghostly face, Tattered dress.\n"
        "Good location names: Grand ballroom, Dark wood study, Steamy bathroom, Overgrown garden, Dimly lit hallway, Empty stage.\n"
        "Descriptions must describe only the place: architecture, layout, surfaces, lighting, weather, atmosphere, era, and color.\n"
        "Do not include characters or actions in descriptions. No bride, woman, man, face, hand, tooth, dress, razor, locket, veil, or similar subject/object details.\n"
        "Test every idea: could the selected subject stand, walk, sit, perform, or be filmed inside this place? If no, reject it.\n"
        "Make the locations specific and cinematic.\n"
        "Output 10-20 locations.\n"
        "Use short bullet points.\n"
        "Each bullet should be a place only, with a brief visual detail if helpful.\n\n"
        "Output format:\n\n"
        "Music Video Locations:\n\n"
        "- [location idea]\n"
        "- [location idea]\n"
        "- [location idea]\n\n"
        f"Existing user locations to avoid duplicating exactly:\n{chr(10).join(existing_lines) if existing_lines else '(none)'}\n\n"
        f"Optional style/theme guidance:\n{style_theme or '(none)'}\n\n"
        f"Character/reference descriptions for style guidance only:\n{subject_context or '(none)'}\n\n"
        f"Optional user input:\n{user_input or '(none)'}\n\n"
        f"User lyrics:\n{lyrics_text}"
    )
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.9),
        max_new_tokens=int(payload.get("max_new_tokens") or 2200),
        label="Gemma",
    )
    locations = _parse_location_ideas_flexible(text)
    if not locations:
        retry_instruction = (
            "Return exactly 12 reusable music video filming locations for the lyrics below.\n"
            "Do not write a heading.\n"
            "Do not explain anything.\n"
            "Do not summarize the lyrics.\n"
            "Write one location per line as a short bullet.\n"
            "Each bullet must be a specific cinematic place where the subject could stand, walk, sit, perform, or be filmed.\n"
            "Reject props, objects, clothing, accessories, body parts, people, characters, creatures, symbolic items, and actions.\n"
            "Descriptions must describe only the place/background, not characters or objects from the lyrics.\n"
            "Example format:\n"
            "- Abandoned motel pool, turquoise water under buzzing neon signs\n"
            "- Foggy pine road, wet asphalt and fading headlights\n\n"
            f"Optional style/theme guidance:\n{style_theme or '(none)'}\n\n"
            f"Character/reference descriptions for style guidance only:\n{subject_context or '(none)'}\n\n"
            f"Optional user input:\n{user_input or '(none)'}\n\n"
            f"Lyrics:\n{lyrics_text}"
        )
        retry_text, retry_info = _run_builder_text_llm(
            payload,
            retry_instruction,
            temperature=float(payload.get("temperature") or 0.55),
            top_p=float(payload.get("top_p") or 0.9),
            max_new_tokens=int(payload.get("max_new_tokens") or 2200),
            label="Gemma",
        )
        retry_locations = _parse_location_ideas_flexible(retry_text)
        if retry_locations:
            text = retry_text
            run_info = retry_info
            locations = retry_locations
    if not locations:
        locations = _parse_location_lines(text)
    if not locations:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(preview) > 700:
            preview = preview[:697].rstrip() + "..."
        raise ValueError(f"Gemma did not return any usable location ideas. Raw response preview: {preview or '(empty)'}")
    deduped = []
    seen = set()
    for item in locations:
        raw_name = item.get("name", "")
        name, description = _clean_location_card(raw_name, item.get("description", ""), raw_name)
        if not _valid_location_card(name, description):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"name": name, "description": description})
    if not deduped:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(preview) > 700:
            preview = preview[:697].rstrip() + "..."
        raise ValueError(f"Gemma returned only non-location items. Raw response preview: {preview or '(empty)'}")
    return {
        "locations": deduped[:20],
        "raw_text": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _generate_flux_reference_subjects(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file and _llm_runner_from_payload(payload) != "lm_studio":
        raise ValueError("Choose a non-vision Gemma model first.")
    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for subject extraction.")
    cleaned_scenes = []
    scene_note_fallbacks = _load_scene_notes_json(payload.get("project_folder", ""))
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        concept = str(scene.get("concept", "") or "").strip()
        notes = str(scene.get("notes", "") or "").strip()
        director_note = str(scene.get("director_note", "") or "").strip() or scene_note_fallbacks.get(index, "")
        if concept or notes or director_note:
            cleaned_scenes.append({
                "id": str(scene.get("id", "") or f"scene_{index}").strip(),
                "label": str(scene.get("label", "") or f"Scene {index}").strip(),
                "concept": concept,
                "notes": notes,
                "director_note": director_note,
            })
    if not cleaned_scenes:
        raise ValueError("Scenes need concept prompt text, notes, or Director Notes before Gemma can extract subjects.")

    requested_count = max(2, min(12, int(payload.get("requested_count") or 2)))
    subject_scene = str(payload.get("subject_scene_text", "") or "").strip()
    existing_subjects = payload.get("existing_subjects") or []
    existing_lines = []
    if isinstance(existing_subjects, list):
        for item in existing_subjects:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            description = str(item.get("description", "") or "").strip()
            if name and description:
                existing_lines.append(f"- {name}: {description}")

    scene_lines = []
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_lines.append(
            f"Scene {index}: {scene['label']}\n"
            f"concept: {scene['concept']}\n"
            f"notes: {scene['notes']}\n"
            f"director_note: {scene['director_note']}"
        )

    instruction = (
        "Extract reusable character/subject identities for reference images.\n\n"
        "Look at the scene concept prompts, notes, Director Notes, and subject/scene context. "
        "Find distinct recurring people, creatures, mascots, or main visual subjects that may need separate reference images. "
        f"The user expects about {requested_count} character references if the project supports that count.\n\n"
        "Output only simple lines in this exact format:\n"
        "1|character name|short visual description for a character reference image\n"
        "2|character name|short visual description for a character reference image\n\n"
        "Rules:\n"
        "- Do not output JSON, markdown, bullets, headings, or explanations.\n"
        "- Keep names short and stable, like blonde woman, masked man, young singer, red android.\n"
        "- Descriptions must describe identity, face/body, hair, outfit, colors, and visual consistency details.\n"
        "- Do not describe locations as characters.\n"
        "- If two characters appear together in a scene, list them as separate characters, not one combined subject.\n\n"
        f"Subject/scene context:\n{subject_scene or '(none)'}\n\n"
        f"Existing user subjects:\n{chr(10).join(existing_lines) if existing_lines else '(none)'}\n\n"
        f"Scenes:\n\n{chr(10).join(scene_lines)}"
    )
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.2),
        top_p=float(payload.get("top_p") or 0.8),
        max_new_tokens=int(payload.get("max_new_tokens") or 2200),
        label="Gemma",
    )
    subjects = _parse_subject_lines(text)
    if not subjects:
        raise ValueError("Gemma did not return any usable subjects.")
    deduped = []
    seen = set()
    for item in subjects:
        key = item["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return {
        "subjects": deduped,
        "raw_text": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _generate_flux_reference_zimage_prompt(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    reference_type = str(payload.get("reference_type", "") or "").strip().lower()
    source_text = str(payload.get("source_text", "") or "").strip()
    style_theme = str(payload.get("style_theme", "") or "").strip()
    if reference_type not in {"subject", "location"}:
        raise ValueError("Reference type must be subject or location.")
    if not model_file and _llm_runner_from_payload(payload) != "lm_studio":
        raise ValueError("Choose a non-vision Gemma model first.")
    if not source_text:
        raise ValueError("Enter a subject or location description first.")

    if reference_type == "subject":
        instruction = (
            "Create one text-to-image prompt for a character reference sheet.\n\n"
            "The image must contain the same character shown three times in one image:\n"
            "1. Left panel: close-up head and face portrait\n"
            "2. Center panel: upper-body view from waist/chest up\n"
            "3. Right panel: full-body standing view\n\n"
            "All three views must show the same person with consistent face, hair, outfit, colors, body type, and identity. "
            "Use a clean neutral studio background. The character should face forward or mostly forward. Keep lighting clear and even. "
            "Do not create a cinematic scene, action pose, environment, story moment, props, text labels, captions, logos, watermarks, or multiple different characters.\n\n"
            "Use this exact output structure:\n\n"
            "A clean three-panel character reference sheet on a neutral studio background, showing the same [subject] in all panels with consistent [face/hair/body/outfit/identity details]. "
            "Left panel: close-up head and face portrait. Center panel: upper-body waist-up view. Right panel: full-body standing view. "
            "Clear even lighting, front-facing pose, consistent outfit colors and body proportions, detailed character design, no text, no labels, no props, no environment.\n\n"
            "Rules:\n"
            "- Output only one polished text-to-image prompt.\n"
            "- Do not include markdown, labels, quotes, explanations, or multiple options.\n"
            "- Keep it as one single image containing three views of the same character.\n"
            "- Preserve the subject identity from the user input.\n\n"
            f"Subject description:\n{source_text}\n\n"
            f"Optional global style/theme:\n{style_theme or '(none)'}"
        )
    else:
        instruction = (
            "Create one text-to-image prompt for a reusable location reference image.\n\n"
            "The image must show only the physical environment/location. Do not include the main character, people, animals, readable text, captions, logos, watermarks, or story action.\n\n"
            "Use this exact output structure:\n\n"
            "A clear cinematic environment reference image of [location/environment], showing [layout/architecture], [important furniture/props/objects], [lighting details], [colors/materials/textures], and [atmosphere]. "
            "Wide enough framing to understand the space layout, [time of day/weather if relevant], no people, no animals, no readable text, no logos, no captions.\n\n"
            "Rules:\n"
            "- Output only one polished text-to-image prompt.\n"
            "- Do not include markdown, labels, quotes, explanations, or multiple options.\n"
            "- Do not include the main character.\n"
            "- Do not describe a music video action.\n"
            "- Keep it as a reusable setting/reference image.\n"
            "- Preserve the location identity from the user input.\n"
            "- Use the optional style/theme only for visual mood, lighting, color, or texture.\n\n"
            f"Location description:\n{source_text}\n\n"
            f"Optional global style/theme:\n{style_theme or '(none)'}"
        )

    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.8)
    max_new_tokens = int(payload.get("max_new_tokens") or 900)

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        label="Gemma",
    )
    text = _clean_visual_gemma_text(text)
    _validate_builder_gemma_prompt(text, f"Flux/Klein {reference_type} reference")
    return {
        "prompt": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _gemma_choices():
    register_custom_model_root()
    from .LLM import VRGDG_SuperGemmaGGUFChat

    return {
        "models": VRGDG_SuperGemmaGGUFChat._list_local_gemma_gguf_choices(),
        "mmproj": VRGDG_SuperGemmaGGUFChat._list_local_mmproj_choices(),
    }


_MODEL_DEFAULT_KEYS = (
    "text_gemma_runner",
    "lm_studio_base_url",
    "lm_studio_model",
    "lm_studio_api_key",
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


def _model_defaults_path():
    defaults_folder = os.path.join(folder_paths.get_output_directory(), "VRGDG_Model_Defaults")
    os.makedirs(defaults_folder, exist_ok=True)
    return os.path.join(defaults_folder, "model_defaults.json")


def _extract_model_defaults(session):
    if not isinstance(session, dict):
        return {}
    defaults = {}
    for key in _MODEL_DEFAULT_KEYS:
        value = session.get(key)
        if value is not None:
            defaults[key] = value
    return defaults


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
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
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
    overlay_segments = _assign_overlay_scene_numbers(overlay_segments)
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
    model_defaults_path = _save_model_defaults(session)
    scene_notes_path = _write_scene_notes_json(project_folder, segments)

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
        "model_defaults_path": model_defaults_path,
        "scene_notes_path": scene_notes_path,
        "session": session,
    }


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


def _save_flux_reference_image(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    reference_type = str(payload.get("reference_type", "") or "").strip().lower()
    if reference_type not in {"subject", "location", "ingredients_sheet"}:
        reference_type = "location"
    raw_name = str(payload.get("name", "") or "").strip() or reference_type
    safe_name = _safe_project_name(raw_name)
    folder_name = "ingredients_sheets" if reference_type == "ingredients_sheet" else f"{reference_type}s"
    target_dir = os.path.join(_context_folder(project_folder), "flux_references", folder_name)
    os.makedirs(target_dir, exist_ok=True)

    image_data = str(payload.get("image_data", "") or "").strip()
    if image_data:
        ext = ".png"
        target_path = _unique_file_path(os.path.join(target_dir, f"{safe_name}{ext}"))
        image = _image_from_data_url(image_data)
        image.save(target_path, format="PNG")
    else:
        image_info = payload.get("image")
        if isinstance(image_info, dict):
            source_path = _resolve_comfy_image_path(image_info)
        else:
            source_path = _resolve_existing_file(payload.get("source_path", ""), "Reference image")
        ext = os.path.splitext(source_path)[1] or ".png"
        target_path = _unique_file_path(os.path.join(target_dir, f"{safe_name}{ext}"))
        shutil.copy2(source_path, target_path)

    return {
        "saved_path": target_path,
        "reference_type": reference_type,
        "folder": target_dir,
    }


def _import_reference_subjects_from_project(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        raise ValueError("Create or load a project first so the subject folder can be found.")

    subject_dir = os.path.join(project_folder, "subject_location", "subject")
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"Subject folder does not exist:\n{subject_dir}")

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    subjects = []
    missing_descriptions = []
    for filename in sorted(os.listdir(subject_dir), key=lambda item: item.lower()):
        image_path = os.path.join(subject_dir, filename)
        if not os.path.isfile(image_path):
            continue
        stem, ext = os.path.splitext(filename)
        if ext.lower() not in image_exts:
            continue
        text_path = os.path.join(subject_dir, f"{stem}.txt")
        description = ""
        if os.path.isfile(text_path):
            with open(text_path, "r", encoding="utf-8", errors="ignore") as handle:
                description = handle.read().strip()
        else:
            missing_descriptions.append(f"{stem}.txt")
        preview_data = ""
        try:
            with Image.open(image_path) as preview_image:
                preview_data = _pil_image_to_data_url(preview_image, max_height=220, quality=72)
        except Exception:
            preview_data = ""
        safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_") or f"subject_{len(subjects) + 1}"
        subjects.append({
            "id": f"subj_import_{len(subjects) + 1}_{safe_id}",
            "name": stem,
            "description": description,
            "image": {
                "path": image_path,
                "data": preview_data,
                "name": filename,
            },
        })

    if not subjects:
        raise ValueError(f"No subject images were found in:\n{subject_dir}")

    return {
        "folder": subject_dir,
        "subjects": subjects,
        "missing_descriptions": missing_descriptions,
    }


def _import_reference_locations_from_project(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        raise ValueError("Create or load a project first so the location folder can be found.")

    base_dir = os.path.join(project_folder, "subject_location")
    location_dir = os.path.join(base_dir, "location")
    typo_location_dir = os.path.join(base_dir, "locaton")
    if not os.path.isdir(location_dir) and os.path.isdir(typo_location_dir):
        location_dir = typo_location_dir
    if not os.path.isdir(location_dir):
        raise FileNotFoundError(
            "Location folder does not exist:\n"
            f"{os.path.join(base_dir, 'location')}\n\n"
            "Expected folder layout:\n"
            "subject_location/location"
        )

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    locations = []
    missing_descriptions = []
    for filename in sorted(os.listdir(location_dir), key=lambda item: item.lower()):
        image_path = os.path.join(location_dir, filename)
        if not os.path.isfile(image_path):
            continue
        stem, ext = os.path.splitext(filename)
        if ext.lower() not in image_exts:
            continue
        text_path = os.path.join(location_dir, f"{stem}.txt")
        description = ""
        if os.path.isfile(text_path):
            with open(text_path, "r", encoding="utf-8", errors="ignore") as handle:
                description = handle.read().strip()
        else:
            missing_descriptions.append(f"{stem}.txt")
        preview_data = ""
        try:
            with Image.open(image_path) as preview_image:
                preview_data = _pil_image_to_data_url(preview_image, max_height=220, quality=72)
        except Exception:
            preview_data = ""
        safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_") or f"location_{len(locations) + 1}"
        locations.append({
            "id": f"loc_import_{len(locations) + 1}_{safe_id}",
            "name": stem,
            "description": description,
            "image": {
                "path": image_path,
                "data": preview_data,
                "name": filename,
            },
        })

    if not locations:
        raise ValueError(f"No location images were found in:\n{location_dir}")

    return {
        "folder": location_dir,
        "locations": locations,
        "missing_descriptions": missing_descriptions,
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
    target_path = os.path.join(folder, f"project_audio{'.wav' if ext == '.m4a' else ext}")
    raw_target_path = os.path.join(folder, f"project_audio_source{ext}") if ext == ".m4a" else target_path
    audio_data = str(payload.get("audio_data", "") or "").strip()
    if audio_data:
        with open(raw_target_path, "wb") as handle:
            handle.write(_audio_bytes_from_data_url(audio_data))
    else:
        source_path = _resolve_existing_file(payload.get("source_path", ""), "Audio file")
        if ext == ".m4a":
            shutil.copy2(source_path, raw_target_path)
        else:
            shutil.copy2(source_path, target_path)
    if ext == ".m4a":
        target_path = _convert_audio_to_wav(raw_target_path, target_path)
        try:
            if os.path.abspath(raw_target_path) != os.path.abspath(target_path):
                os.remove(raw_target_path)
        except Exception:
            pass
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
    start_time = max(0.0, float(payload.get("start_time") or 0))
    end_time = start_time + duration
    label = str(payload.get("label") or f"Scene {scene_number}").strip()
    folder = os.path.join(project_folder, "scene_srt")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"scene_{scene_number:04d}.srt")
    text = "\n".join([
        "1",
        f"{_format_srt_time(start_time)} --> {_format_srt_time(end_time)}",
        label,
        "",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return {"srt_path": path, "scene_number": scene_number, "start_time": start_time, "duration": duration}


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
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
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
            result = subprocess.run(silence_cmd, capture_output=True, text=True, errors="replace", check=False)
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
        result = subprocess.run(clip_cmd, capture_output=True, text=True, errors="replace", check=False)
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
    result = subprocess.run(mix_cmd, capture_output=True, text=True, errors="replace", check=False)
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


def _builder_scene_video_thumbnail_path(video_path):
    root, _ext = os.path.splitext(os.path.abspath(str(video_path or "")))
    return f"{root}.jpg"


def _ensure_builder_scene_video_thumbnail(video_path):
    video_path = os.path.abspath(str(video_path or "").strip().strip('"'))
    if not os.path.isfile(video_path):
        return ""
    thumbnail_path = _builder_scene_video_thumbnail_path(video_path)
    if os.path.isfile(thumbnail_path):
        return thumbnail_path
    try:
        ffmpeg_path = _find_ffmpeg_path()
        for timestamp in ("0.5", "0"):
            cmd = [
                ffmpeg_path,
                "-y",
                "-ss",
                timestamp,
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-vf",
                "scale=480:-2",
                "-q:v",
                "3",
                thumbnail_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
            if result.returncode == 0 and os.path.isfile(thumbnail_path):
                return thumbnail_path
        error_text = (result.stderr or result.stdout or "ffmpeg could not extract a thumbnail.").strip()
        print(f"[VRGDG Music Builder] Could not create scene video thumbnail for '{video_path}': {error_text}")
    except Exception as exc:
        print(f"[VRGDG Music Builder] Could not create scene video thumbnail for '{video_path}': {exc}")
    return ""


def _scan_builder_scene_videos(project_folder):
    folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not folder:
        raise ValueError("Project folder is empty.")
    video_folder = os.path.join(folder, "rendered_scene_videos")
    backup_root = os.path.join(folder, "rendered_scene_videos_backup")
    videos = {}
    video_thumbnails = {}
    video_backups = {}
    video_backup_thumbnails = {}
    if not os.path.isdir(video_folder):
        return {
            "project_folder": folder,
            "video_folder": video_folder,
            "videos": videos,
            "video_thumbnails": video_thumbnails,
            "video_backups": video_backups,
            "video_backup_thumbnails": video_backup_thumbnails,
        }
    pattern = re.compile(r"^video_(\d+)-audio\.mp4$", re.IGNORECASE)
    for name in os.listdir(video_folder):
        match = pattern.match(name)
        if not match:
            continue
        path = os.path.join(video_folder, name)
        if os.path.isfile(path):
            key = str(int(match.group(1)))
            videos[key] = path
            thumb = _ensure_builder_scene_video_thumbnail(path)
            if thumb:
                video_thumbnails[key] = thumb
    if os.path.isdir(backup_root):
        max_backups_per_scene = 12
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
                try:
                    modified = os.path.getmtime(path)
                except OSError:
                    modified = 0
                video_backups.setdefault(key, []).append((path, modified))
        for key, pairs in list(video_backups.items()):
            pairs.sort(key=lambda item: item[1], reverse=True)
            kept = pairs[:max_backups_per_scene]
            kept.reverse()
            video_backups[key] = [item[0] for item in kept]
            video_backup_thumbnails[key] = [_ensure_builder_scene_video_thumbnail(item[0]) for item in kept]
    return {
        "project_folder": folder,
        "video_folder": video_folder,
        "videos": videos,
        "video_thumbnails": video_thumbnails,
        "video_backups": video_backups,
        "video_backup_thumbnails": video_backup_thumbnails,
    }


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
            project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
            if os.path.splitext(audio_path)[1].lower() == ".m4a" and project_folder:
                audio_path = _convert_audio_to_wav(
                    audio_path,
                    os.path.join(project_folder, "project_audio", "project_audio.wav"),
                )
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

    @server_instance.routes.post("/vrgdg/music_builder/save_wizard_draft")
    async def vrgdg_music_builder_save_wizard_draft(request):
        try:
            payload = await request.json()
            result = _save_wizard_draft(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_wizard_draft")
    async def vrgdg_music_builder_load_wizard_draft(request):
        try:
            payload = await request.json()
            result = _load_wizard_draft(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/model_defaults")
    async def vrgdg_music_builder_model_defaults(request):
        try:
            result = _load_model_defaults()
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

    @server_instance.routes.post("/vrgdg/music_builder/save_flux_reference_image")
    async def vrgdg_music_builder_save_flux_reference_image(request):
        try:
            payload = await request.json()
            result = _save_flux_reference_image(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/import_reference_subjects")
    async def vrgdg_music_builder_import_reference_subjects(request):
        try:
            payload = await request.json()
            result = _import_reference_subjects_from_project(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/import_reference_locations")
    async def vrgdg_music_builder_import_reference_locations(request):
        try:
            payload = await request.json()
            result = _import_reference_locations_from_project(payload)
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

    @server_instance.routes.post("/vrgdg/music_builder/open_local_file")
    async def vrgdg_music_builder_open_local_file(request):
        try:
            payload = await request.json()
            path = _open_local_file(payload.get("path", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "path": path})

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

    @server_instance.routes.post("/vrgdg/music_builder/import_latest_prompt_creator_outputs")
    async def vrgdg_music_builder_import_latest_prompt_creator_outputs(request):
        try:
            payload = await request.json()
            result = _copy_latest_prompt_creator_outputs(payload.get("project_folder", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/copy_prompt_creator_outputs")
    async def vrgdg_music_builder_copy_prompt_creator_outputs(request):
        try:
            payload = await request.json()
            result = _copy_prompt_creator_outputs_from_source(
                payload.get("project_folder", ""),
                payload.get("source_project_folder", ""),
            )
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

    @server_instance.routes.post("/vrgdg/music_builder/lm_studio_models")
    async def vrgdg_music_builder_lm_studio_models(request):
        try:
            payload = await request.json()
            result = _list_lm_studio_models(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/clear_memory_direct")
    async def vrgdg_music_builder_clear_memory_direct(request):
        try:
            result = await asyncio.to_thread(_clear_builder_memory_direct)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_t2i")
    async def vrgdg_music_builder_generate_t2i(request):
        try:
            payload = await request.json()
            result = _generate_builder_t2i_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/agent_chat")
    async def vrgdg_music_builder_agent_chat(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_agent_reply, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_concept_prompts")
    async def vrgdg_music_builder_generate_concept_prompts(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_concept_prompts, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_motion_notes")
    async def vrgdg_music_builder_generate_motion_notes(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_motion_notes, payload)
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

    @server_instance.routes.post("/vrgdg/music_builder/generate_t2v")
    async def vrgdg_music_builder_generate_t2v(request):
        try:
            payload = await request.json()
            result = _generate_builder_t2v_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/enhance_video_prompt")
    async def vrgdg_music_builder_enhance_video_prompt(request):
        try:
            payload = await request.json()
            result = _enhance_builder_video_prompt(payload)
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

    @server_instance.routes.post("/vrgdg/music_builder/describe_reference_image")
    async def vrgdg_music_builder_describe_reference_image(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_reference_description, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/analyze_story_references")
    async def vrgdg_music_builder_analyze_story_references(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_analyze_builder_story_references, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_nb_image_prompt")
    async def vrgdg_music_builder_generate_nb_image_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_nb_image_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/flux_reference_location_map")
    async def vrgdg_music_builder_flux_reference_location_map(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_reference_location_map, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/flux_reference_extract_locations")
    async def vrgdg_music_builder_flux_reference_extract_locations(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_reference_locations, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/wizard_locations_from_lyrics")
    async def vrgdg_music_builder_wizard_locations_from_lyrics(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_wizard_locations_from_lyrics, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/flux_reference_extract_subjects")
    async def vrgdg_music_builder_flux_reference_extract_subjects(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_reference_subjects, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/flux_reference_zimage_prompt")
    async def vrgdg_music_builder_flux_reference_zimage_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_reference_zimage_prompt, payload)
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
