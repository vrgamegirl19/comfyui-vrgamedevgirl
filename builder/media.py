"""Video Builder media: scene images, reference images, video thumbnails, and scene video scanning and restore."""

import math
import os
import re
import subprocess
import shutil
import time
import base64
from PIL import Image
from .video_editor import _image_from_data_url
from ..runner.paths import _resolve_comfy_image_path

from .paths import _context_folder, _copy_file_if_exists, _images_folder, _resolve_existing_file, _safe_project_name, _scene_preview_folder, _unique_file_path, _unique_preview_path
from .audio import _find_ffmpeg_path


def _scene_image_path(project_folder, scene_number, extension=".png"):
    scene = max(1, int(scene_number or 1))
    ext = str(extension or ".png").lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        ext = ".png"
    return os.path.join(_images_folder(project_folder), f"image_{scene:04d}{ext}")


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


def _extract_video_final_frame_as_scene_image(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    source_path = _resolve_existing_file(payload.get("source_path", ""), "Source video")
    try:
        common = os.path.commonpath([project_folder, source_path])
    except ValueError:
        common = ""
    if common != project_folder:
        raise ValueError("Source video must be inside the current project folder.")

    scene_number = int(payload.get("scene_number") or payload.get("target_scene_number") or 1)
    target_path = _unique_preview_path(project_folder, scene_number, ".png")
    ffmpeg_path = _find_ffmpeg_path()
    attempts = [
        ["-sseof", "-0.04"],
        ["-sseof", "-0.12"],
        ["-sseof", "-0.5"],
    ]
    last_error = ""
    for seek_args in attempts:
        cmd = [
            ffmpeg_path,
            "-y",
            *seek_args,
            "-i",
            source_path,
            "-frames:v",
            "1",
            "-update",
            "1",
            target_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
        if result.returncode == 0 and os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
            return {
                "saved_path": target_path,
                "preview_folder": _scene_preview_folder(project_folder, scene_number),
                "scene_number": scene_number,
                "source_path": source_path,
            }
        last_error = (result.stderr or result.stdout or "ffmpeg could not extract a final frame.").strip()
        try:
            if os.path.isfile(target_path):
                os.remove(target_path)
        except Exception:
            pass
    raise RuntimeError(last_error or "ffmpeg could not extract a final frame.")


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


def _builder_scene_video_thumbnail_path(video_path):
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


def _builder_legacy_scene_video_thumbnail_path(video_path):
    root, _ext = os.path.splitext(os.path.abspath(str(video_path or "").strip().strip('"')))
    return f"{root}.jpg"


def _ensure_builder_scene_video_thumbnail(video_path):
    video_path = os.path.abspath(str(video_path or "").strip().strip('"'))
    if not os.path.isfile(video_path):
        return ""
    thumbnail_path = _builder_scene_video_thumbnail_path(video_path)
    if os.path.isfile(thumbnail_path):
        return thumbnail_path
    try:
        # Older Builder projects predate the dedicated thumbnail directory.
        # Their videos are still valid, but ffmpeg cannot write the recovered
        # thumbnail until its parent directory exists.
        os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
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


def _ffprobe_path_for_builder(ffmpeg_path):
    if not ffmpeg_path or ffmpeg_path == "ffmpeg":
        return "ffprobe"
    folder = os.path.dirname(os.path.abspath(ffmpeg_path))
    exe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    candidate = os.path.join(folder, exe_name)
    return candidate if os.path.isfile(candidate) else "ffprobe"


def _probe_video_duration_seconds(video_path):
    video_path = os.path.abspath(str(video_path or "").strip().strip('"'))
    if not os.path.isfile(video_path):
        return 0.0
    try:
        ffprobe_path = _ffprobe_path_for_builder(_find_ffmpeg_path())
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if result.returncode == 0:
            return max(0.0, float(str(result.stdout or "0").strip() or 0))
    except Exception as exc:
        print(f"[VRGDG Music Builder] Could not probe video duration for '{video_path}': {exc}")
    return 0.0


def _restore_scene_video(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    source_path = os.path.abspath(str(payload.get("source_path", "") or "").strip().strip('"'))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video file was not found: {source_path}")
    if os.path.splitext(source_path)[1].lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        raise ValueError("Choose a supported video file: .mp4, .mov, .mkv, .webm, or .avi")
    scene_number = max(1, int(payload.get("scene_number") or 1))
    duration = _probe_video_duration_seconds(source_path)
    expected_duration = max(0.0, float(payload.get("expected_duration") or 0))
    tolerance = max(0.1, float(payload.get("duration_tolerance") or 0.5))
    duration_delta = abs(duration - expected_duration) if duration and expected_duration else 0.0
    if duration_delta > tolerance and not bool(payload.get("confirm_duration_mismatch")):
        return {
            "needs_confirmation": True,
            "source_path": source_path,
            "scene_number": scene_number,
            "duration": duration,
            "expected_duration": expected_duration,
            "duration_delta": duration_delta,
            "duration_tolerance": tolerance,
        }
    target_dir = os.path.join(project_folder, "rendered_scene_videos")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"video_{scene_number:04d}-audio.mp4")
    thumbnail_path = _builder_scene_video_thumbnail_path(target_path)
    legacy_thumbnail_path = _builder_legacy_scene_video_thumbnail_path(target_path)
    backup_path = ""
    backup_thumbnail_path = ""
    if os.path.isfile(target_path) and os.path.normcase(os.path.abspath(source_path)) != os.path.normcase(os.path.abspath(target_path)):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = os.path.join(project_folder, "rendered_scene_videos_backup", f"scene_{scene_number:04d}")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"video_{scene_number:04d}-audio_manual_restore_{stamp}.mp4")
        shutil.move(target_path, backup_path)
        if os.path.isfile(thumbnail_path):
            backup_thumbnail_path = _builder_scene_video_thumbnail_path(backup_path)
            shutil.move(thumbnail_path, backup_thumbnail_path)
    copied = _copy_file_if_exists(source_path, target_path)
    if not copied:
        raise RuntimeError("Could not copy the selected video into the project.")
    for stale_thumbnail in (thumbnail_path, legacy_thumbnail_path):
        if os.path.isfile(stale_thumbnail):
            try:
                os.remove(stale_thumbnail)
            except OSError:
                pass
    created_thumbnail = _ensure_builder_scene_video_thumbnail(copied)
    return {
        "video_path": copied,
        "video_folder": target_dir,
        "thumbnail_path": created_thumbnail,
        "scene_number": scene_number,
        "source_path": source_path,
        "duration": duration,
        "backup_path": backup_path,
        "backup_thumbnail_path": backup_thumbnail_path,
    }


def _scan_builder_scene_videos(project_folder):
    folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not folder:
        raise ValueError("Project folder is empty.")
    video_folder = os.path.join(folder, "rendered_scene_videos")
    backup_root = os.path.join(folder, "rendered_scene_videos_backup")
    scratch_prefixes = (
        "image_to_video_clips",
        "text_to_video_clips",
        "reference_to_video_clips",
        "ingredients_to_video_clips",
    )
    videos = {}
    video_thumbnails = {}
    video_backups = {}
    video_backup_thumbnails = {}
    recovered_from_scratch = {}
    scene_srt_mtimes = []
    scene_srt_folder = os.path.join(folder, "scene_srt")
    if os.path.isdir(scene_srt_folder):
        srt_pattern = re.compile(r"^scene_(\d+)\.srt$", re.IGNORECASE)
        for srt_name in os.listdir(scene_srt_folder):
            match = srt_pattern.match(srt_name)
            if not match:
                continue
            srt_path = os.path.join(scene_srt_folder, srt_name)
            if not os.path.isfile(srt_path):
                continue
            try:
                scene_srt_mtimes.append((str(int(match.group(1))), os.path.getmtime(srt_path)))
            except OSError:
                continue
    scene_srt_mtimes.sort(key=lambda item: item[1])
    os.makedirs(video_folder, exist_ok=True)
    pattern = re.compile(r"^video_(\d+)-audio\.mp4$", re.IGNORECASE)
    for name in os.listdir(video_folder):
        match = pattern.match(name)
        if not match:
            continue
        path = os.path.join(video_folder, name)
        if os.path.isfile(path):
            key = str(int(match.group(1)))
            videos[key] = path
            legacy_thumbnail = _builder_legacy_scene_video_thumbnail_path(path)
            if os.path.isfile(legacy_thumbnail):
                try:
                    os.remove(legacy_thumbnail)
                except OSError:
                    pass
            thumb = _ensure_builder_scene_video_thumbnail(path)
            if thumb:
                video_thumbnails[key] = thumb
    scratch_candidates = {}
    scratch_pattern = re.compile(r"^video_(\d+)(?:[-_].*)?\.mp4$", re.IGNORECASE)
    scene_folder_pattern = re.compile(r"scene[_-](\d+)", re.IGNORECASE)
    def infer_scratch_scene_key(path, raw_key, modified):
        parts = os.path.abspath(path).split(os.sep)
        for part in reversed(parts):
            match = scene_folder_pattern.search(part)
            if match:
                return str(int(match.group(1)))
        if raw_key != "1" and raw_key not in videos:
            return raw_key
        previous_srts = [(key, mtime) for key, mtime in scene_srt_mtimes if mtime <= modified + 2.0 and key not in videos]
        if previous_srts:
            return max(previous_srts, key=lambda item: item[1])[0]
        return raw_key

    for name in os.listdir(folder):
        scratch_folder = os.path.abspath(os.path.join(folder, name))
        if not os.path.isdir(scratch_folder):
            continue
        if not any(name == prefix or name.startswith(f"{prefix}_") for prefix in scratch_prefixes):
            continue
        for root, _, names in os.walk(scratch_folder):
            try:
                if os.path.commonpath([folder, os.path.abspath(root)]) != folder:
                    continue
            except ValueError:
                continue
            for file_name in names:
                if not file_name.lower().endswith(".mp4"):
                    continue
                match = scratch_pattern.match(file_name)
                if not match:
                    continue
                path = os.path.abspath(os.path.join(root, file_name))
                if not os.path.isfile(path):
                    continue
                try:
                    size = os.path.getsize(path)
                    modified = os.path.getmtime(path)
                except OSError:
                    continue
                if size <= 0:
                    continue
                raw_key = str(int(match.group(1)))
                key = infer_scratch_scene_key(path, raw_key, modified)
                score = 100 if file_name.lower().endswith("-audio.mp4") else 0
                score += 10 if "-audio" in file_name.lower() else 0
                current = scratch_candidates.get(key)
                if not current or (score, modified) > (current[0], current[1]):
                    scratch_candidates[key] = (score, modified, path)
    for key, (_score, _modified, source_path) in scratch_candidates.items():
        if key in videos:
            continue
        try:
            scene_number = int(key)
        except ValueError:
            continue
        target_path = os.path.join(video_folder, f"video_{scene_number:04d}-audio.mp4")
        try:
            copied = _copy_file_if_exists(source_path, target_path)
        except Exception as exc:
            print(f"[VRGDG Music Builder] Could not recover scene video '{source_path}': {exc}")
            copied = ""
        if copied:
            videos[key] = copied
            recovered_from_scratch[key] = source_path
            thumb = _ensure_builder_scene_video_thumbnail(copied)
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
        "recovered_from_scratch": recovered_from_scratch,
    }
