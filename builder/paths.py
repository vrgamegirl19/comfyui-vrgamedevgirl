"""Video Builder project paths, file pickers and small file helpers."""

import json
import os
import re
import subprocess
import shutil
import sys
import time
import folder_paths


def _vrgdg_textfile_path(folder_name, file_name):
    return os.path.join(
        folder_paths.get_output_directory(),
        "VRGDG_TEMP",
        "TextFiles",
        folder_name,
        file_name,
    )


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
        elif kind == "video":
            path = filedialog.askopenfilename(
                title="Choose video file",
                filetypes=[
                    ("Video files", "*.mp4 *.mov *.mkv *.webm *.avi"),
                    ("All files", "*.*"),
                ],
            )
        elif kind == "gguf":
            path = filedialog.askopenfilename(
                title="Choose GGUF model file",
                filetypes=[("GGUF model files", "*.gguf"), ("All files", "*.*")],
            )
        elif kind in {"project_folder", "project_root"}:
            title = "Choose projects root folder" if kind == "project_root" else "Choose project folder"
            path = filedialog.askdirectory(title=title)
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
    elif kind == "video":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose video file'
$dialog.Filter = 'Video files (*.mp4;*.mov;*.mkv;*.webm;*.avi)|*.mp4;*.mov;*.mkv;*.webm;*.avi|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write($dialog.FileName) }
"""
    elif kind == "gguf":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose GGUF model file'
$dialog.Filter = 'GGUF model files (*.gguf)|*.gguf|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write($dialog.FileName) }
"""
    elif kind == "project_folder":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose project folder'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write($dialog.SelectedPath) }
"""
    elif kind == "project_root":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose projects root folder'
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


def _session_path(project_folder):
    return os.path.join(project_folder, "vrgdg_builder_session.json")


def _scene_notes_path(project_folder):
    return os.path.join(project_folder, "SceneNotes.json")


def _render_logs_folder(project_folder):
    return os.path.join(project_folder, "render_logs")


def _images_folder(project_folder):
    return os.path.join(project_folder, "zimage_approved")


def _prompts_folder(project_folder):
    return os.path.join(project_folder, "prompts")


def _context_folder(project_folder):
    return os.path.join(project_folder, "project_context")


def _safe_builder_scene_id(value):
    scene_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return scene_id[:120]


def _project_folder_from_builder_payload(payload):
    raw = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not raw:
        raise ValueError("Create or load a Builder project before editing instructions.")
    return os.path.abspath(raw)


def _wizard_folder(project_folder):
    return os.path.join(project_folder, "wizard")


def _wizard_draft_path(project_folder):
    return os.path.join(_wizard_folder(project_folder), "wizard_draft.json")


def _wizard_lyrics_path(project_folder):
    return os.path.join(_wizard_folder(project_folder), "lyrics.txt")


def _is_internal_approved_image_path(path):
    parts = os.path.normpath(str(path or "")).split(os.sep)
    return "zimage_approved" in {part.lower() for part in parts}


def _scene_preview_folder(project_folder, scene_number):
    scene = max(1, int(scene_number or 1))
    return os.path.join(project_folder, "scene_image_previews", f"scene_{scene:04d}")


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
        path = os.path.normcase(os.path.abspath(path))
        folder = os.path.normcase(os.path.abspath(folder))
        return os.path.commonpath([path, folder]) == folder
    except Exception:
        return False


def _looks_like_filesystem_path(text):
    value = str(text or "").strip().strip('"')
    if not value or "\n" in value or len(value) > 4096:
        return False
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    if value.startswith("\\\\") or value.startswith("/"):
        return True
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


# The route hands the path to the OS "open" action, so anything that could run a program is refused.
_OPENABLE_EXTENSIONS = {
    ".mp4", ".mov", ".webm", ".mkv", ".avi",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".wav", ".mp3", ".flac", ".m4a",
    ".txt", ".srt", ".json",
}


def _open_local_file(path):
    target = os.path.realpath(str(path or "").strip().strip('"'))
    if not os.path.isdir(target):
        if not os.path.isfile(target):
            raise ValueError("File was not found.")
        if os.path.splitext(target)[1].lower() not in _OPENABLE_EXTENSIONS:
            raise ValueError(f"Only video, image, audio and text files can be opened: {os.path.basename(target)}")
    if os.name == "nt":
        os.startfile(target)  # pylint: disable=no-member
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])
    return target


def _read_text_file(path, label):
    if not str(path or "").strip():
        return ""
    text_path = _resolve_existing_file(path, label)
    with open(text_path, "r", encoding="utf-8-sig") as handle:
        return handle.read().strip()


def _load_json_file(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _model_defaults_path():
    defaults_folder = os.path.join(folder_paths.get_output_directory(), "VRGDG_Model_Defaults")
    os.makedirs(defaults_folder, exist_ok=True)
    return os.path.join(defaults_folder, "model_defaults.json")
