"""Rendered scene video files: collecting, trimming, color matching, thumbnails, stitching and slideshows."""

import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction

from .paths import _bool_payload, _ffprobe_path_for, _find_ffmpeg_path, _int_payload, _resolve_comfy_image_path, _resolve_save_folder, _unique_copy_path


def _save_generated_image(payload):
    image_info = payload.get("image")
    if not isinstance(image_info, dict):
        raise ValueError("Image info is missing.")
    source_path = _resolve_comfy_image_path(image_info)
    target_dir = _resolve_save_folder(payload.get("save_folder"))
    target_path = _unique_copy_path(target_dir, source_path)
    shutil.copy2(source_path, target_path)
    return {"saved_path": target_path, "save_folder": target_dir}


def _probe_video_size(video_path, ffmpeg_path=None):
    ffprobe_path = _ffprobe_path_for(ffmpeg_path)
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=True)
    text = (result.stdout or "").strip().splitlines()[0]
    width_text, height_text = text.lower().split("x", 1)
    return int(width_text), int(height_text)


def _normalize_video_canvas(ffmpeg_path, source_path, target_path, width, height):
    width = int(width or 0)
    height = int(height or 0)
    if width <= 0 or height <= 0:
        return False
    try:
        source_width, source_height = _probe_video_size(source_path, ffmpeg_path)
        if source_width == width and source_height == height:
            return False
    except Exception as exc:
        print(f"[VRGDG WorkflowRunner] Could not probe video size before final canvas normalization: {exc}")

    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        source_path,
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        target_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=True)
    return True


def _scene_video_thumbnail_path(video_path):
    video_path = os.path.abspath(str(video_path or "").strip().strip('"'))
    root, _ext = os.path.splitext(video_path)
    video_name = os.path.basename(root)
    current = os.path.dirname(video_path)
    while current and current != os.path.dirname(current):
        if os.path.basename(current).lower() in {"rendered_scene_videos", "rendered_scene_videos_backup"}:
            project_folder = os.path.dirname(current)
            return os.path.join(project_folder, "scene_video_thumbnails", f"{video_name}.jpg")
        current = os.path.dirname(current)
    return f"{root}.jpg"


def _legacy_scene_video_thumbnail_path(video_path):
    root, _ext = os.path.splitext(os.path.abspath(str(video_path or "").strip().strip('"')))
    return f"{root}.jpg"


def _create_scene_video_thumbnail(video_path, thumbnail_path=None):
    video_path = os.path.abspath(str(video_path or "").strip().strip('"'))
    if not os.path.isfile(video_path):
        return ""
    thumbnail_path = os.path.abspath(str(thumbnail_path or _scene_video_thumbnail_path(video_path)).strip().strip('"'))
    os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
    ffmpeg_path = _find_ffmpeg_path()

    def _run_extract(timestamp):
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss",
            str(timestamp),
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
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    result = _run_extract(0.5)
    if result.returncode != 0 or not os.path.isfile(thumbnail_path):
        result = _run_extract(0)
    if result.returncode != 0 or not os.path.isfile(thumbnail_path):
        error_text = (result.stderr or result.stdout or "ffmpeg could not extract a thumbnail.").strip()
        print(f"[VRGDG WorkflowRunner] Could not create scene video thumbnail for '{video_path}': {error_text}")
        return ""
    return thumbnail_path


def _safe_project_subfolder(project_folder, folder_name):
    project = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not project:
        raise ValueError("Project folder is empty.")
    target = os.path.abspath(os.path.join(project, folder_name))
    if os.path.commonpath([project, target]) != project:
        raise ValueError("Target folder escapes the project folder.")
    os.makedirs(target, exist_ok=True)
    return project, target


def _unique_final_video_path(project_folder, prefix="FINAL_VIDEO"):
    safe_prefix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(prefix or "FINAL_VIDEO")).strip("_") or "FINAL_VIDEO"
    candidate = os.path.join(project_folder, f"{safe_prefix}.mp4")
    if not os.path.exists(candidate):
        return candidate
    index = 2
    while True:
        candidate = os.path.join(project_folder, f"{safe_prefix}{index}.mp4")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _concat_file_path(path):
    return os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")


def _cleanup_video_scratch_folders(project_folder, keep_folders=None):
    project_folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    keep = {os.path.abspath(path) for path in (keep_folders or []) if path}
    scratch_prefixes = ("image_to_video_clips_", "text_to_video_clips_")
    permanent_folders = {"image_to_video_clips", "text_to_video_clips", "rendered_scene_videos", "rendered_scene_videos_backup"}
    removed_folders = []
    if not os.path.isdir(project_folder):
        return removed_folders
    for name in os.listdir(project_folder):
        path = os.path.abspath(os.path.join(project_folder, name))
        if path in keep or not os.path.isdir(path):
            continue
        if name in permanent_folders or not name.startswith(scratch_prefixes):
            continue
        try:
            if os.path.commonpath([project_folder, path]) != project_folder:
                continue
            shutil.rmtree(path)
            removed_folders.append(path)
        except Exception as exc:
            print(f"[VRGDG WorkflowRunner] Could not delete video scratch folder '{path}': {exc}")
    return removed_folders


def _cleanup_i2v_scratch_folders(project_folder, keep_folders=None):
    return _cleanup_video_scratch_folders(project_folder, keep_folders=keep_folders)


def _retry_file_op(operation, description, attempts=30, delay=0.25):
    last_exc = None
    for attempt in range(max(1, attempts)):
        try:
            return operation()
        except PermissionError as exc:
            last_exc = exc
        except OSError as exc:
            if getattr(exc, "winerror", None) != 32:
                raise
            last_exc = exc
        if attempt < attempts - 1:
            time.sleep(delay)
    raise RuntimeError(f"{description} failed because the file stayed locked: {last_exc}") from last_exc


def _wait_for_stable_readable_file(path, timeout=20.0, interval=0.25):
    deadline = time.time() + max(0.5, float(timeout or 0))
    last_size = -1
    stable_reads = 0
    last_exc = None
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as handle:
                handle.read(1)
            if size > 0 and size == last_size:
                stable_reads += 1
                if stable_reads >= 2:
                    return
            else:
                stable_reads = 0
                last_size = size
        except (OSError, PermissionError) as exc:
            last_exc = exc
            stable_reads = 0
        time.sleep(interval)
    if last_exc:
        raise RuntimeError(f"Scene video is still locked and cannot be read: {path}") from last_exc


def _replace_file_with_retry(source_path, target_path):
    _wait_for_stable_readable_file(source_path)
    temp_target = f"{target_path}.copying"
    index = 2
    while os.path.exists(temp_target):
        temp_target = f"{target_path}.copying_{index:02d}"
        index += 1

    try:
        _retry_file_op(
            lambda: shutil.copy2(source_path, temp_target),
            f"Copying scene video to temporary file '{temp_target}'",
        )
        _retry_file_op(
            lambda: os.replace(temp_target, target_path),
            f"Replacing scene video '{target_path}'",
        )
    finally:
        if os.path.exists(temp_target):
            try:
                os.remove(temp_target)
            except Exception:
                pass

    try:
        _retry_file_op(
            lambda: os.remove(source_path),
            f"Removing scratch scene video '{source_path}'",
            attempts=8,
            delay=0.25,
        )
    except Exception as exc:
        print(f"[VRGDG WorkflowRunner] Copied scene video but could not remove scratch source '{source_path}': {exc}")


def _collect_scene_video(payload):
    source_path = os.path.abspath(str(payload.get("source_path", "") or "").strip().strip('"'))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Scene video was not found: {source_path}")
    project_folder, target_dir = _safe_project_subfolder(payload.get("project_folder", ""), "rendered_scene_videos")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    existing_action = str(payload.get("existing_action", "overwrite") or "overwrite").strip().lower()
    if existing_action not in {"overwrite", "backup"}:
        existing_action = "overwrite"

    source_dir = os.path.abspath(os.path.dirname(source_path))
    if not source_path.lower().endswith("-audio.mp4"):
        candidates = [
            os.path.join(source_dir, name)
            for name in os.listdir(source_dir)
            if name.lower().endswith("-audio.mp4") and os.path.isfile(os.path.join(source_dir, name))
        ]
        candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        if candidates:
            source_path = os.path.abspath(candidates[0])
            source_dir = os.path.abspath(os.path.dirname(source_path))

    target_path = os.path.join(target_dir, f"video_{scene_number:04d}-audio.mp4")
    target_thumbnail_path = _scene_video_thumbnail_path(target_path)
    legacy_target_thumbnail_path = _legacy_scene_video_thumbnail_path(target_path)
    backup_path = ""
    backup_thumbnail_path = ""
    if os.path.abspath(source_path) != os.path.abspath(target_path):
        if os.path.exists(target_path):
            if existing_action == "backup":
                backup_dir = os.path.join(project_folder, "rendered_scene_videos_backup", f"scene_{scene_number:04d}")
                os.makedirs(backup_dir, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"video_{scene_number:04d}-audio_{stamp}.mp4")
                index = 2
                while os.path.exists(backup_path):
                    backup_path = os.path.join(backup_dir, f"video_{scene_number:04d}-audio_{stamp}_{index:02d}.mp4")
                    index += 1
                _retry_file_op(
                    lambda: shutil.move(target_path, backup_path),
                    f"Backing up existing scene video '{target_path}'",
                )
                if os.path.exists(target_thumbnail_path):
                    backup_thumbnail_path = _scene_video_thumbnail_path(backup_path)
                    _retry_file_op(
                        lambda: shutil.move(target_thumbnail_path, backup_thumbnail_path),
                        f"Backing up existing scene video thumbnail '{target_thumbnail_path}'",
                    )
                if os.path.exists(legacy_target_thumbnail_path):
                    _retry_file_op(
                        lambda: os.remove(legacy_target_thumbnail_path),
                        f"Removing legacy scene video thumbnail '{legacy_target_thumbnail_path}'",
                    )
            else:
                _retry_file_op(
                    lambda: os.remove(target_path),
                    f"Removing existing scene video '{target_path}'",
                )
                if os.path.exists(target_thumbnail_path):
                    try:
                        _retry_file_op(
                            lambda: os.remove(target_thumbnail_path),
                            f"Removing existing scene video thumbnail '{target_thumbnail_path}'",
                        )
                    except Exception as exc:
                        print(f"[VRGDG WorkflowRunner] Could not remove old scene video thumbnail '{target_thumbnail_path}': {exc}")
                if os.path.exists(legacy_target_thumbnail_path):
                    try:
                        _retry_file_op(
                            lambda: os.remove(legacy_target_thumbnail_path),
                            f"Removing legacy scene video thumbnail '{legacy_target_thumbnail_path}'",
                        )
                    except Exception as exc:
                        print(f"[VRGDG WorkflowRunner] Could not remove legacy scene video thumbnail '{legacy_target_thumbnail_path}': {exc}")
        _replace_file_with_retry(source_path, target_path)

    if os.path.exists(legacy_target_thumbnail_path):
        try:
            _retry_file_op(
                lambda: os.remove(legacy_target_thumbnail_path),
                f"Removing legacy scene video thumbnail '{legacy_target_thumbnail_path}'",
            )
        except Exception as exc:
            print(f"[VRGDG WorkflowRunner] Could not remove legacy scene video thumbnail '{legacy_target_thumbnail_path}': {exc}")

    thumbnail_path = _create_scene_video_thumbnail(target_path, target_thumbnail_path)
    removed_files = []
    removed_folder = ""
    removed_scratch_folders = []

    return {
        "video_path": target_path,
        "thumbnail_path": thumbnail_path,
        "video_folder": target_dir,
        "backup_path": backup_path,
        "backup_thumbnail_path": backup_thumbnail_path,
        "existing_action": existing_action,
        "source_path": source_path,
        "removed_files": removed_files,
        "removed_folder": removed_folder,
        "removed_scratch_folders": removed_scratch_folders,
    }


def _trim_scene_video(payload):
    source_path = os.path.abspath(str(payload.get("source_path", "") or "").strip().strip('"'))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Scene video was not found: {source_path}")
    if os.path.splitext(source_path)[1].lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise ValueError(f"Scene media is not a supported video file: {source_path}")
    project_folder, target_dir = _safe_project_subfolder(payload.get("project_folder", ""), "rendered_scene_videos")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    start = max(0.0, float(payload.get("start", 0) or 0))
    duration = max(0.05, float(payload.get("duration", 0) or 0))
    frames = _int_payload(payload, "frames", 0, 0, 999999)
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(payload.get("label", "trim") or "trim").strip().lower()).strip("_") or "trim"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    audio_suffix = "-audio" if _bool_payload(payload, "mark_as_audio_video", False) else ""
    target_path = os.path.join(target_dir, f"video_{scene_number:04d}-{label}_{stamp}{audio_suffix}.mp4")
    index = 2
    while os.path.exists(target_path):
        target_path = os.path.join(target_dir, f"video_{scene_number:04d}-{label}_{stamp}_{index:02d}{audio_suffix}.mp4")
        index += 1

    # A scene rendered with a masked Audio Mask mix gets the real scene audio back on the finished clip.
    restore_text = str(payload.get("restore_audio_path", "") or "").strip().strip('"')
    restore_audio = os.path.abspath(restore_text) if restore_text else ""
    if restore_audio and not os.path.isfile(restore_audio):
        raise FileNotFoundError(f"Scene audio to restore was not found: {restore_audio}")
    restore_start = max(0.0, float(payload.get("restore_audio_start_seconds", 0) or 0))

    ffmpeg_path = _find_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        source_path,
    ]
    if restore_audio:
        cmd += ["-ss", f"{restore_start:.6f}", "-i", restore_audio]
    cmd += [
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0" if restore_audio else "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        target_path,
    ]
    if frames:
        # match the frame count the stitcher keeps so the clip's last frame is the one shown
        cmd[-1:-1] = ["-frames:v", str(frames)]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(target_path):
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed to trim scene video.").strip())
    thumbnail_path = _create_scene_video_thumbnail(target_path)
    return {
        "video_path": target_path,
        "thumbnail_path": thumbnail_path,
        "video_folder": target_dir,
        "source_path": source_path,
        "start": start,
        "duration": duration,
    }


def _apply_scene_start_color_match(payload):
    """Match a new clip's opening color to the prior clip, then fade the correction out."""
    from PIL import Image, ImageStat

    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    video_path = os.path.abspath(str(payload.get("video_path", "") or "").strip().strip('"'))
    reference_video_path = os.path.abspath(str(payload.get("reference_video_path", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        raise ValueError("Project folder is empty or does not exist.")
    for label, path in (("Scene video", video_path), ("Previous scene video", reference_video_path)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} was not found: {path}")
        try:
            inside_project = os.path.commonpath([project_folder, path]) == project_folder
        except ValueError:
            inside_project = False
        if not inside_project:
            raise ValueError(f"{label} must be inside the current project folder.")

    fade_seconds = max(0.05, min(30.0, float(payload.get("fade_seconds", 1.0) or 1.0)))
    strength = max(0.0, min(1.0, float(payload.get("strength", 0.85) or 0.85)))
    if strength <= 0.0:
        return {"video_path": video_path, "applied": False, "reason": "strength is zero"}

    ffmpeg_path = _find_ffmpeg_path()
    work_dir = os.path.dirname(video_path)
    token = f"{int(time.time() * 1000)}_{os.getpid()}"
    reference_frame = os.path.join(work_dir, f".vrgdg_color_reference_{token}.png")
    target_frame = os.path.join(work_dir, f".vrgdg_color_target_{token}.png")
    cube_path = os.path.join(work_dir, f".vrgdg_color_match_{token}.cube")
    output_path = os.path.join(work_dir, f".vrgdg_color_matched_{token}.mp4")

    def run_ffmpeg(command, message):
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", cwd=work_dir)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or message).strip())

    try:
        # -update 1 leaves the last decoded frame in the PNG after processing the final second.
        run_ffmpeg([
            ffmpeg_path, "-y", "-sseof", "-1", "-i", reference_video_path,
            "-map", "0:v:0", "-an", "-update", "1", reference_frame,
        ], "FFmpeg could not read the previous clip's final frame.")
        run_ffmpeg([
            ffmpeg_path, "-y", "-i", video_path, "-map", "0:v:0", "-an",
            "-frames:v", "1", target_frame,
        ], "FFmpeg could not read the new clip's first frame.")

        with Image.open(reference_frame) as image:
            reference_stats = ImageStat.Stat(image.convert("RGB"))
        with Image.open(target_frame) as image:
            target_stats = ImageStat.Stat(image.convert("RGB"))
        reference_mean = [float(value) for value in reference_stats.mean[:3]]
        reference_std = [max(1.0, float(value)) for value in reference_stats.stddev[:3]]
        target_mean = [float(value) for value in target_stats.mean[:3]]
        target_std = [max(1.0, float(value)) for value in target_stats.stddev[:3]]
        scales = [max(0.25, min(4.0, reference_std[i] / target_std[i])) for i in range(3)]
        offsets = [reference_mean[i] - target_mean[i] * scales[i] for i in range(3)]

        cube_size = 17
        with open(cube_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write('TITLE "VRGDG opening color match"\n')
            handle.write(f"LUT_3D_SIZE {cube_size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
            for blue in range(cube_size):
                for green in range(cube_size):
                    for red in range(cube_size):
                        values = [red, green, blue]
                        corrected = [
                            max(0.0, min(1.0, ((values[i] / (cube_size - 1)) * 255.0 * scales[i] + offsets[i]) / 255.0))
                            for i in range(3)
                        ]
                        handle.write(f"{corrected[0]:.8f} {corrected[1]:.8f} {corrected[2]:.8f}\n")

        weight = f"max(0\\,min(1\\,{strength:.6f}*(1-T/{fade_seconds:.6f})))"
        filter_graph = (
            f"[0:v]split=2[original][to_match];"
            f"[to_match]lut3d=file='{os.path.basename(cube_path)}'[matched];"
            f"[original][matched]blend=all_expr='A*(1-({weight}))+B*({weight})'[video]"
        )
        run_ffmpeg([
            ffmpeg_path, "-y", "-i", video_path,
            "-filter_complex", filter_graph,
            "-map", "[video]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart", output_path,
        ], "FFmpeg could not apply the opening color match.")
        if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
            raise RuntimeError("Opening color match did not create a valid video.")
        os.replace(output_path, video_path)
        thumbnail_path = _create_scene_video_thumbnail(video_path, _scene_video_thumbnail_path(video_path))
        return {
            "video_path": video_path,
            "thumbnail_path": thumbnail_path,
            "applied": True,
            "fade_seconds": fade_seconds,
            "strength": strength,
            "reference_video_path": reference_video_path,
        }
    finally:
        for temporary_path in (reference_frame, target_frame, cube_path, output_path):
            try:
                if os.path.isfile(temporary_path):
                    os.remove(temporary_path)
            except Exception:
                pass


def _find_scene_video_output(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        raise ValueError("Project folder is empty or does not exist.")
    mode = str(payload.get("video_mode", "") or "").strip().lower()
    if mode == "rtv":
        prefixes = ("reference_to_video_clips", "reference_to_video_clips_")
    elif mode == "t2v":
        prefixes = ("text_to_video_clips", "text_to_video_clips_")
    elif mode == "ingredients":
        prefixes = ("ingredients_to_video_clips", "ingredients_to_video_clips_")
    elif mode == "id_lora":
        prefixes = ("id_lora_i2v_clips", "id_lora_i2v_clips_")
    else:
        prefixes = ("image_to_video_clips", "image_to_video_clips_")

    scene_number = _int_payload(payload, "scene_number", 0, 0, 999999)
    prompt_number = _int_payload(payload, "prompt_number_one_based", scene_number or 0, 0, 999999)
    min_mtime = float(payload.get("min_mtime") or 0)
    output_folder = os.path.abspath(str(payload.get("output_folder", "") or "").strip().strip('"')) if payload.get("output_folder") else ""

    folders = []
    if output_folder and os.path.isdir(output_folder):
        try:
            if os.path.commonpath([project_folder, output_folder]) == project_folder:
                folders.append(output_folder)
        except ValueError:
            pass
    if not bool(payload.get("strict_output_folder")):
        for name in os.listdir(project_folder):
            path = os.path.abspath(os.path.join(project_folder, name))
            if not os.path.isdir(path):
                continue
            if any(name == prefix.rstrip("_") or name.startswith(prefix) for prefix in prefixes):
                folders.append(path)
    folders = list(dict.fromkeys(folders))

    candidates = []
    for folder in folders:
        for root, _dirs, files in os.walk(folder):
            try:
                if os.path.commonpath([project_folder, os.path.abspath(root)]) != project_folder:
                    continue
            except ValueError:
                continue
            for name in files:
                lower = name.lower()
                if not lower.endswith("-audio.mp4"):
                    continue
                path = os.path.abspath(os.path.join(root, name))
                try:
                    mtime = os.path.getmtime(path)
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0 or (min_mtime and mtime + 1 < min_mtime):
                    continue
                score = 0
                if scene_number and re.match(rf"^video_{scene_number:04d}-audio\.mp4$", name, re.IGNORECASE):
                    score += 1000
                if prompt_number and re.match(rf"^video_{prompt_number:04d}(?:_|-)", name, re.IGNORECASE):
                    score += 700
                if scene_number and f"_{scene_number:04d}_" in name:
                    score += 100
                candidates.append((score, mtime, path, folder))
    if not candidates:
        return {"video_path": "", "output_folder": "", "searched_folders": folders}
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _score, _mtime, path, folder = candidates[0]
    _wait_for_stable_readable_file(path, timeout=8.0, interval=0.25)
    return {
        "video_path": path,
        "output_folder": folder,
        "searched_folders": folders,
    }


def _collect_minimax_h3_stage_backup(payload):
    source_path = os.path.abspath(str(payload.get("source_path", "") or "").strip().strip('"'))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"MiniMax H3 stage video was not found: {source_path}")
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        raise ValueError("Project folder is empty or does not exist.")
    if os.path.commonpath([project_folder, source_path]) == project_folder:
        raise ValueError("MiniMax H3 stage backup source must be outside the project folder.")
    stage = str(payload.get("stage", "") or "").strip().lower()
    if stage not in {"stage1", "stage2"}:
        raise ValueError("MiniMax H3 stage backup must be stage1 or stage2.")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    backup_dir = os.path.join(project_folder, "rendered_scene_videos_backup", f"scene_{scene_number:04d}")
    os.makedirs(backup_dir, exist_ok=True)
    _wait_for_stable_readable_file(source_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target_path = os.path.join(backup_dir, f"video_{scene_number:04d}-{stage}_{stamp}.mp4")
    index = 2
    while os.path.exists(target_path):
        target_path = os.path.join(backup_dir, f"video_{scene_number:04d}-{stage}_{stamp}_{index:02d}.mp4")
        index += 1
    _retry_file_op(lambda: shutil.copy2(source_path, target_path), f"Copying MiniMax H3 {stage} backup")
    thumbnail_path = _create_scene_video_thumbnail(target_path)
    return {
        "backup_path": target_path,
        "backup_thumbnail_path": thumbnail_path,
        "stage": stage,
        "scene_number": scene_number,
    }


def _find_minimax_h3_stage_outputs(payload):
    output_folder = os.path.abspath(str(payload.get("output_folder", "") or "").strip().strip('"'))
    if not output_folder or not os.path.isdir(output_folder):
        return {"stage1_path": "", "stage2_path": "", "stage3_path": ""}
    min_mtime = float(payload.get("min_mtime") or 0)
    found = {}
    for root, _dirs, files in os.walk(output_folder):
        for name in files:
            lower = name.lower()
            if not lower.endswith("-audio.mp4"):
                continue
            stage = next((item for item in ("stage1", "stage2", "stage3") if item in lower), "")
            if not stage:
                continue
            path = os.path.abspath(os.path.join(root, name))
            try:
                mtime = os.path.getmtime(path)
                if mtime + 1 < min_mtime or os.path.getsize(path) <= 0:
                    continue
            except OSError:
                continue
            previous = found.get(stage)
            if not previous or mtime > previous[0]:
                found[stage] = (mtime, path)
    return {f"{stage}_path": found.get(stage, (0, ""))[1] for stage in ("stage1", "stage2", "stage3")}


EMBEDDED_AUDIO_SAMPLE_RATE = 48000


def _probe_stream_fields(path, ffmpeg_path, stream, fields, count_packets=False):
    """ffprobe ``fields`` of the first ``stream`` (``v:0`` or ``a:0``) as a dict; empty when there is no such stream."""
    cmd = [_ffprobe_path_for(ffmpeg_path), "-v", "error", "-select_streams", stream]
    if count_packets:
        cmd.append("-count_packets")
    cmd += ["-show_entries", "stream=" + ",".join(fields), "-of", "default=noprint_wrappers=1", path]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        # Fail loudly: reading "no audio stream" from a failed probe would silence the scene.
        raise RuntimeError((result.stderr or f"ffprobe could not read {path}").strip())
    values = {}
    for line in (result.stdout or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _probe_float(values, key, default=0.0):
    try:
        return float(values.get(key))
    except (TypeError, ValueError):
        return default


def _clip_frame_length(path, ffmpeg_path):
    """``(frames, fps)`` of a clip's video stream, so its length is exactly ``frames / fps`` seconds."""
    values = _probe_stream_fields(path, ffmpeg_path, "v:0", ["nb_read_packets", "r_frame_rate", "avg_frame_rate"], count_packets=True)
    fps = Fraction(0)
    for key in ("r_frame_rate", "avg_frame_rate"):
        try:
            fps = Fraction(values.get(key, "0/1"))
        except (ValueError, ZeroDivisionError):
            fps = Fraction(0)
        if fps > 0:
            break
    try:
        frames = int(values.get("nb_read_packets") or 0)
    except ValueError:
        frames = 0
    if frames <= 0 or fps <= 0:
        raise RuntimeError(f"Could not read the frame count and frame rate of scene video: {path}")
    return frames, fps


def _embedded_scene_audio_track(ffmpeg_path, scene_paths, frame_lengths, target_dir, temp_files):
    """Join the scenes' own audio as one PCM track that lines up with the joined video at every scene start.

    Each scene's audio is padded with silence or cut to exactly its clip's ``frames / fps`` (sample counts
    from the running total, so 44.1 kHz rounding cannot build up), a scene without an audio stream becomes
    silence of that length, and the parts are joined as PCM. The caller encodes the result once, so no
    per-scene encoder priming or AAC frame padding lands between scenes.
    """
    sample_rate = EMBEDDED_AUDIO_SAMPLE_RATE
    elapsed = Fraction(0)
    start_sample = 0
    part_paths = []
    for index, (path, (frames, fps)) in enumerate(zip(scene_paths, frame_lengths), start=1):
        elapsed += Fraction(int(frames)) / Fraction(fps)
        end_sample = int(math.floor(elapsed * sample_rate + Fraction(1, 2)))
        samples = max(1, end_sample - start_sample)
        start_sample = end_sample
        part_path = os.path.join(target_dir, f"_temp_scene_audio_{index:04d}.wav")
        temp_files.append(part_path)
        audio_info = _probe_stream_fields(path, ffmpeg_path, "a:0", ["index", "start_time"])
        if audio_info.get("index", "") != "":
            # Keep the audio where it sits against the clip's first frame (normally both start at 0).
            video_info = _probe_stream_fields(path, ffmpeg_path, "v:0", ["start_time"])
            lead = _probe_float(audio_info, "start_time") - _probe_float(video_info, "start_time")
            filters = [f"aresample={sample_rate}", "aformat=sample_fmts=s16:channel_layouts=stereo"]
            if lead > 0.0005:
                filters.append(f"adelay={lead * 1000:.3f}:all=1")
            elif lead < -0.0005:
                filters.append(f"atrim=start={-lead:.6f},asetpts=PTS-STARTPTS")
            filters += [f"apad=whole_len={samples}", f"atrim=end_sample={samples}"]
            cmd = [ffmpeg_path, "-y", "-i", path, "-map", "0:a:0", "-vn", "-af", ",".join(filters), "-c:a", "pcm_s16le", part_path]
        else:
            # No audio stream: silence for the clip's length, so the later scenes do not move up.
            cmd = [
                ffmpeg_path, "-y", "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=stereo",
                "-af", f"atrim=end_sample={samples}", "-c:a", "pcm_s16le", part_path,
            ]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not os.path.isfile(part_path):
            raise RuntimeError((result.stderr or result.stdout or f"FFmpeg failed to prepare scene {index} audio.").strip())
        part_paths.append(part_path)

    list_path = os.path.join(target_dir, "_temp_scene_audio_list.txt")
    joined_path = os.path.join(target_dir, "_temp_scene_audio_joined.wav")
    temp_files.extend([list_path, joined_path])
    with open(list_path, "w", encoding="utf-8") as handle:
        for part_path in part_paths:
            handle.write(f"file '{_concat_file_path(part_path)}'\n")
    subprocess.run(
        [ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c:a", "copy", joined_path],
        capture_output=True, text=True, errors="replace", check=True,
    )
    return joined_path


def _stitch_scene_videos(payload):
    raw_paths = payload.get("scene_paths", [])
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("No scene video paths were provided.")
    project_folder, target_dir = _safe_project_subfolder(payload.get("project_folder", ""), "rendered_scene_videos")
    raw_scene_audio_paths = payload.get("scene_audio_paths", [])
    if not isinstance(raw_scene_audio_paths, list):
        raw_scene_audio_paths = []
    raw_scene_audio_items = payload.get("scene_audio_items", [])
    if not isinstance(raw_scene_audio_items, list):
        raw_scene_audio_items = []
    raw_overlay_items = payload.get("overlay_items", [])
    if not isinstance(raw_overlay_items, list):
        raw_overlay_items = []
    raw_scene_timing_items = payload.get("scene_timing_items", [])
    if not isinstance(raw_scene_timing_items, list):
        raw_scene_timing_items = []
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    preview_audio_start = max(0.0, float(payload.get("audio_start", 0) or 0))
    preview_audio_duration = max(0.0, float(payload.get("audio_duration", 0) or 0))
    target_width = _int_payload(payload, "width", 0, 0, 8192)
    target_height = _int_payload(payload, "height", 0, 0, 8192)
    use_embedded_scene_audio = bool(payload.get("use_embedded_scene_audio"))
    timeline_fps = _int_payload(payload, "timeline_fps", 0, 0, 120)

    scene_paths = []
    for index, raw_path in enumerate(raw_paths, start=1):
        path = os.path.abspath(str(raw_path or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Scene {index} video was not found: {path}")
        if os.path.splitext(path)[1].lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            raise ValueError(f"Scene {index} media is not a supported video file: {path}")
        scene_paths.append(path)

    scene_audio_paths = []
    scene_audio_items = []
    if raw_scene_audio_items and any(str((item or {}).get("path", "") if isinstance(item, dict) else "").strip() for item in raw_scene_audio_items):
        if len(raw_scene_audio_items) != len(scene_paths):
            raise ValueError("Scene audio item count does not match scene video count.")
        for index, item in enumerate(raw_scene_audio_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Scene {index} audio item is invalid.")
            path = os.path.abspath(str(item.get("path", "") or "").strip().strip('"'))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Scene {index} audio was not found: {path}")
            start = max(0.0, float(item.get("start", 0) or 0))
            duration = max(0.05, float(item.get("duration", 0) or 0))
            scene_audio_items.append({"path": path, "start": start, "duration": duration})
            scene_audio_paths.append(path)
    elif raw_scene_audio_paths and any(str(item or "").strip() for item in raw_scene_audio_paths):
        if len(raw_scene_audio_paths) != len(scene_paths):
            raise ValueError("Scene audio path count does not match scene video count.")
        for index, raw_path in enumerate(raw_scene_audio_paths, start=1):
            path = os.path.abspath(str(raw_path or "").strip().strip('"'))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Scene {index} audio was not found: {path}")
            scene_audio_paths.append(path)
            scene_audio_items.append({"path": path, "start": 0.0, "duration": 0.0})
    elif use_embedded_scene_audio:
        for path in scene_paths:
            scene_audio_paths.append(path)
            scene_audio_items.append({"path": path, "start": 0.0, "duration": 0.0, "embedded": True})
    elif not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")

    ffmpeg_path = _find_ffmpeg_path()
    timeline_sync_paths = []
    timeline_sync_frame_count = 0
    timeline_frame_lengths = []
    concat_scene_paths = scene_paths
    if raw_scene_timing_items:
        if timeline_fps <= 0:
            raise ValueError("Timeline FPS is required when scene timing items are provided.")
        if len(raw_scene_timing_items) != len(scene_paths):
            raise ValueError("Scene timing item count does not match scene video count.")

        concat_scene_paths = []
        for index, (path, item) in enumerate(zip(scene_paths, raw_scene_timing_items), start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Scene {index} timing item is invalid.")
            start = max(0.0, float(item.get("start", 0) or 0))
            end = max(start, float(item.get("end", start) or start))
            start_frame = int(start * timeline_fps + 0.5)
            end_frame = int(end * timeline_fps + 0.5)
            target_frames = max(1, end_frame - start_frame)
            timeline_sync_frame_count += target_frames
            timeline_frame_lengths.append((target_frames, Fraction(timeline_fps)))
            sync_path = os.path.join(target_dir, f"_temp_timeline_scene_{index:04d}.mp4")
            sync_filter = (
                f"fps={timeline_fps},"
                "tpad=stop_mode=clone:stop_duration=1,"
                f"trim=start_frame=0:end_frame={target_frames},"
                "setpts=PTS-STARTPTS"
            )
            sync_cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                path,
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                sync_filter,
                "-frames:v",
                str(target_frames),
                "-r",
                str(timeline_fps),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                sync_path,
            ]
            sync_result = subprocess.run(sync_cmd, capture_output=True, text=True, errors="replace")
            if sync_result.returncode != 0 or not os.path.isfile(sync_path):
                raise RuntimeError(
                    (sync_result.stderr or sync_result.stdout or f"FFmpeg failed to align scene {index} to the timeline.").strip()
                )
            timeline_sync_paths.append(sync_path)
            concat_scene_paths.append(sync_path)

    concat_file = os.path.join(target_dir, "concat_list.txt")
    with open(concat_file, "w", encoding="utf-8") as handle:
        for path in concat_scene_paths:
            handle.write(f"file '{_concat_file_path(path)}'\n")

    temp_video = os.path.join(target_dir, "_temp_video_no_audio.mp4")
    normalized_video = os.path.join(target_dir, "_temp_video_normalized_canvas.mp4")
    temp_audio = os.path.join(target_dir, "_temp_scene_audio.m4a")
    temp_global_audio = os.path.join(target_dir, "_temp_global_audio.m4a")
    temp_audio_parts = []
    audio_concat_file = os.path.join(target_dir, "audio_concat_list.txt")
    final_output = _unique_final_video_path(project_folder, payload.get("output_prefix", "FINAL_VIDEO"))
    normalized_canvas = False

    concat_cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-an",
        "-c:v",
        "copy",
        temp_video,
    ]
    subprocess.run(concat_cmd, capture_output=True, text=True, errors="replace", check=True)

    insert_items = []
    for index, item in enumerate(raw_overlay_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Insert {index} item is invalid.")
        path = os.path.abspath(str(item.get("path", "") or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Insert {index} video was not found: {path}")
        if os.path.splitext(path)[1].lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            raise ValueError(f"Insert {index} media is not a supported video file: {path}")
        start = max(0.0, float(item.get("start", 0) or 0))
        end = max(start + 0.05, float(item.get("end", start + 4) or start + 4))
        source_start = max(0.0, float(item.get("source_start", 0) or 0))
        insert_items.append({"path": path, "start": start, "end": end, "duration": end - start, "source_start": source_start})

    if insert_items:
        insert_items.sort(key=lambda item: (item["start"], item["end"]))
        flattened_video = os.path.join(target_dir, "_temp_video_with_inserts.mp4")
        flatten_list = os.path.join(target_dir, "flatten_concat_list.txt")
        flatten_parts = []
        cursor = 0.0
        part_index = 1

        def add_flatten_part(source_path, start=None, duration=None):
            nonlocal part_index
            part_path = os.path.join(target_dir, f"_temp_flatten_part_{part_index:04d}.mp4")
            part_index += 1
            cmd = [ffmpeg_path, "-y"]
            if start is not None:
                cmd.extend(["-ss", f"{max(0.0, float(start)):.6f}"])
            cmd.extend(["-i", source_path])
            if duration is not None:
                cmd.extend(["-t", f"{max(0.05, float(duration)):.6f}"])
            cmd.extend([
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                part_path,
            ])
            subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=True)
            flatten_parts.append(part_path)

        for item in insert_items:
            if item["start"] > cursor + 0.01:
                add_flatten_part(temp_video, cursor, item["start"] - cursor)
            add_flatten_part(item["path"], item.get("source_start", 0.0), item["duration"])
            cursor = max(cursor, item["end"])

        add_flatten_part(temp_video, cursor, None)
        with open(flatten_list, "w", encoding="utf-8") as handle:
            for path in flatten_parts:
                handle.write(f"file '{_concat_file_path(path)}'\n")
        flatten_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            flatten_list,
            "-an",
            "-c:v",
            "copy",
            flattened_video,
        ]
        subprocess.run(flatten_cmd, capture_output=True, text=True, errors="replace", check=True)
        try:
            os.remove(temp_video)
        except Exception:
            pass
        try:
            os.remove(flatten_list)
        except Exception:
            pass
        for part_path in flatten_parts:
            try:
                os.remove(part_path)
            except Exception:
                pass
        temp_video = flattened_video

    if target_width > 0 and target_height > 0:
        normalized_canvas = _normalize_video_canvas(ffmpeg_path, temp_video, normalized_video, target_width, target_height)
        if normalized_canvas:
            try:
                os.remove(temp_video)
            except Exception:
                pass
            temp_video = normalized_video

    mux_audio_path = audio_path
    audio_matches_video = False
    if scene_audio_items and all(item.get("embedded") for item in scene_audio_items):
        # Each clip's own audio, cut or padded to the clip's exact length: the timeline frame count when the
        # clips were synced to the timeline, otherwise the clip's own frames / fps.
        frame_lengths = timeline_frame_lengths if timeline_sync_paths else [
            _clip_frame_length(path, ffmpeg_path) for path in scene_paths
        ]
        mux_audio_path = _embedded_scene_audio_track(ffmpeg_path, scene_paths, frame_lengths, target_dir, temp_audio_parts)
        audio_matches_video = True
    elif scene_audio_paths:
        with open(audio_concat_file, "w", encoding="utf-8") as handle:
            for index, item in enumerate(scene_audio_items, start=1):
                path = item["path"]
                duration = float(item.get("duration", 0) or 0)
                if item.get("embedded") or item.get("start", 0) or duration:
                    part_path = os.path.join(target_dir, f"_temp_scene_audio_{index:04d}.m4a")
                    trim_cmd = [
                        ffmpeg_path,
                        "-y",
                        "-ss",
                        str(float(item.get("start", 0) or 0)),
                        "-i",
                        path,
                    ]
                    if duration:
                        trim_cmd.extend(["-t", str(duration)])
                    trim_cmd.extend(["-vn", "-c:a", "aac", part_path])
                    subprocess.run(trim_cmd, capture_output=True, text=True, errors="replace", check=True)
                    temp_audio_parts.append(part_path)
                    path = part_path
                handle.write(f"file '{_concat_file_path(path)}'\n")
        audio_concat_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            audio_concat_file,
            "-vn",
            "-c:a",
            "aac",
            temp_audio,
        ]
        subprocess.run(audio_concat_cmd, capture_output=True, text=True, errors="replace", check=True)
        mux_audio_path = temp_audio
    elif preview_audio_start or preview_audio_duration:
        trim_audio_cmd = [ffmpeg_path, "-y"]
        if preview_audio_start:
            trim_audio_cmd.extend(["-ss", f"{preview_audio_start:.6f}"])
        trim_audio_cmd.extend(["-i", audio_path])
        if preview_audio_duration:
            trim_audio_cmd.extend(["-t", f"{preview_audio_duration:.6f}"])
        trim_audio_cmd.extend(["-vn", "-c:a", "aac", temp_global_audio])
        subprocess.run(trim_audio_cmd, capture_output=True, text=True, errors="replace", check=True)
        mux_audio_path = temp_global_audio

    mux_cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        temp_video,
        "-i",
        mux_audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
    ]
    if not timeline_sync_paths and not audio_matches_video:
        # -shortest ends the file at the shorter stream. The joined embedded audio is already exactly the
        # video's length, and -shortest would clip its last few ms with the last frame's duration.
        mux_cmd.append("-shortest")
    mux_cmd.append(final_output)
    try:
        subprocess.run(mux_cmd, capture_output=True, text=True, errors="replace", check=True)
    finally:
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except Exception:
            pass
        try:
            if os.path.exists(normalized_video):
                os.remove(normalized_video)
        except Exception:
            pass
        try:
            if os.path.exists(concat_file):
                os.remove(concat_file)
        except Exception:
            pass
        try:
            if os.path.exists(audio_concat_file):
                os.remove(audio_concat_file)
        except Exception:
            pass
        try:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        except Exception:
            pass
        try:
            if os.path.exists(temp_global_audio):
                os.remove(temp_global_audio)
        except Exception:
            pass
        for part_path in temp_audio_parts:
            try:
                if os.path.exists(part_path):
                    os.remove(part_path)
            except Exception:
                pass
        for sync_path in timeline_sync_paths:
            try:
                if os.path.exists(sync_path):
                    os.remove(sync_path)
            except Exception:
                pass
    removed_scratch_folders = _cleanup_video_scratch_folders(project_folder, keep_folders=[target_dir])

    return {
        "final_video_path": final_output,
        "video_folder": target_dir,
        "concat_file": "",
        "scene_count": len(scene_paths),
        "insert_count": len(insert_items),
        "used_scene_audio": bool(scene_audio_paths),
        "used_embedded_scene_audio": bool(use_embedded_scene_audio and scene_audio_paths),
        "normalized_canvas": normalized_canvas,
        "timeline_frame_sync": bool(timeline_sync_paths),
        "timeline_fps": timeline_fps if timeline_sync_paths else 0,
        "timeline_frame_count": timeline_sync_frame_count,
        "output_width": target_width,
        "output_height": target_height,
        "removed_scratch_folders": removed_scratch_folders,
    }


def _render_image_slideshow(payload):
    raw_items = payload.get("image_items", [])
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("No scene images were provided for the slideshow preview.")
    project_folder, target_dir = _safe_project_subfolder(payload.get("project_folder", ""), "slideshow_previews")
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Global audio file was not found: {audio_path}")
    audio_start = max(0.0, float(payload.get("audio_start", 0) or 0))
    target_width = _int_payload(payload, "width", 1920, 64, 8192)
    target_height = _int_payload(payload, "height", 1080, 64, 8192)
    fps = _int_payload(payload, "fps", 24, 1, 120)

    items = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Scene {index} slideshow item is invalid.")
        path = os.path.abspath(str(item.get("path", "") or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Scene {index} image was not found: {path}")
        if os.path.splitext(path)[1].lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            raise ValueError(f"Scene {index} media is not a supported slideshow image: {path}")
        duration = max(0.05, float(item.get("duration", 0) or 0))
        items.append({"path": path, "duration": duration})

    total_duration = sum(item["duration"] for item in items)
    ffmpeg_path = _find_ffmpeg_path()
    scratch = tempfile.mkdtemp(prefix="_slideshow_", dir=target_dir)
    concat_file = os.path.join(scratch, "images.txt")
    video_only = os.path.join(scratch, "video.mp4")
    final_output = _unique_final_video_path(project_folder, payload.get("output_prefix", "IMAGE_SLIDESHOW_PREVIEW"))
    try:
        # The concat demuxer does not safely handle still images whose stream
        # properties change mid-list.  In particular, FFmpeg's fps filter can
        # discard the image immediately before a resolution change while the
        # filter graph is reinitialized.  Normalize every source to one common
        # RGB frame first so a mixed-resolution project cannot lose a scene.
        normalized_items = []
        normalize_filter = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            "setsar=1,format=rgb24"
        )
        for index, item in enumerate(items, start=1):
            normalized_path = os.path.join(scratch, f"image_{index:06d}.png")
            normalize_cmd = [
                ffmpeg_path, "-y", "-i", item["path"],
                "-vf", normalize_filter,
                "-frames:v", "1", normalized_path,
            ]
            try:
                subprocess.run(normalize_cmd, capture_output=True, text=True, errors="replace", check=True)
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr or exc.stdout or str(exc)
                raise RuntimeError(f"Could not normalize slideshow Scene {index}:\n{detail}") from exc
            normalized_items.append({"path": normalized_path, "duration": item["duration"]})

        with open(concat_file, "w", encoding="utf-8") as handle:
            for item in normalized_items:
                handle.write(f"file '{_concat_file_path(item['path'])}'\n")
                handle.write(f"duration {item['duration']:.6f}\n")
            # The concat demuxer only applies the final duration when its last
            # still is repeated once.
            handle.write(f"file '{_concat_file_path(normalized_items[-1]['path'])}'\n")

        filter_graph = f"fps={fps},format=yuv420p"
        slideshow_cmd = [
            ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-vf", filter_graph,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-t", f"{total_duration:.6f}", "-movflags", "+faststart", video_only,
        ]
        subprocess.run(slideshow_cmd, capture_output=True, text=True, errors="replace", check=True)

        mux_cmd = [ffmpeg_path, "-y", "-i", video_only]
        if audio_start:
            mux_cmd.extend(["-ss", f"{audio_start:.6f}"])
        mux_cmd.extend([
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", f"{total_duration:.6f}",
            "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
            final_output,
        ])
        subprocess.run(mux_cmd, capture_output=True, text=True, errors="replace", check=True)
        if not os.path.isfile(final_output) or os.path.getsize(final_output) <= 0:
            raise RuntimeError("FFmpeg did not create the slideshow preview video.")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "final_video_path": final_output,
        "video_folder": target_dir,
        "scene_count": len(items),
        "duration": total_duration,
        "audio_start": audio_start,
        "output_width": target_width,
        "output_height": target_height,
        "fps": fps,
    }
