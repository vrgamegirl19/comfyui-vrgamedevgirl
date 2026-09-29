"""Workflow runner paths, payload readers, and file and folder resolving."""

import os
import subprocess
import time
import folder_paths


def _workflow_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "text2image_zimage.json",
    )


def _int_payload(payload, key, default, minimum=1, maximum=16384):
    try:
        value = int(payload.get(key, default))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _float_payload(payload, key, default, minimum=-100.0, maximum=100.0):
    try:
        value = float(payload.get(key, default))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _bool_payload(payload, key, default=False):
    value = payload.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _first_payload_value(payload, *keys, default=None):
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return default


def _resolve_existing_file(raw_path, label="file"):
    text = str(raw_path or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError(f"{label} path is empty.")

    candidates = []
    if os.path.isabs(text):
        candidates.append(text)
    else:
        candidates.extend(
            [
                text,
                os.path.abspath(text),
                os.path.join(folder_paths.get_input_directory(), text),
                os.path.join(folder_paths.get_output_directory(), text),
            ]
        )
        get_temp_directory = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_directory):
            candidates.append(os.path.join(get_temp_directory(), text))

    seen = set()
    for candidate in candidates:
        path = os.path.normpath(os.path.abspath(candidate))
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(f"{label} was not found: {text}")


def _scene_render_output_folder(project_folder, folder_name, payload):
    scene_number = _int_payload(payload, "scene_number", 0, 0, 999999)
    root = os.path.join(project_folder, folder_name)
    if scene_number > 0:
        root = os.path.join(root, f"scene_{scene_number:04d}")
    os.makedirs(root, exist_ok=True)
    return root


def _safe_subfolder_path(base_dir, subfolder):
    base_abs = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.join(base_abs, str(subfolder or "")))
    if os.path.commonpath([base_abs, candidate]) != base_abs:
        raise ValueError("Image subfolder escapes the allowed ComfyUI folder.")
    return candidate


def _resolve_comfy_image_path(image_info):
    filename = os.path.basename(str(image_info.get("filename", "") or ""))
    if not filename:
        raise ValueError("Image filename is empty.")
    image_type = str(image_info.get("type", "output") or "output").lower()
    if image_type == "temp":
        base_dir = folder_paths.get_temp_directory()
    elif image_type == "input":
        base_dir = folder_paths.get_input_directory()
    else:
        base_dir = folder_paths.get_output_directory()
    folder = _safe_subfolder_path(base_dir, image_info.get("subfolder", ""))
    image_path = os.path.abspath(os.path.join(folder, filename))
    if os.path.commonpath([os.path.abspath(base_dir), image_path]) != os.path.abspath(base_dir):
        raise ValueError("Image path escapes the allowed ComfyUI folder.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Generated image was not found: {image_path}")
    return image_path


def _resolve_save_folder(raw_folder):
    text = str(raw_folder or "").strip().strip('"')
    if not text:
        text = "VRGDG_WorkflowRunner_Saved"
    if os.path.isabs(text):
        target = os.path.abspath(text)
    else:
        target = os.path.abspath(os.path.join(folder_paths.get_output_directory(), text))
    os.makedirs(target, exist_ok=True)
    return target


def _unique_copy_path(target_dir, source_path):
    stem, ext = os.path.splitext(os.path.basename(source_path))
    if not ext:
        ext = ".png"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(target_dir, f"{stem}_approved_{timestamp}{ext}")
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(target_dir, f"{stem}_approved_{timestamp}_{counter}{ext}")
        counter += 1
    return candidate


def _find_ffmpeg_path():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except Exception:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError(f"FFmpeg was not found: {exc}") from exc


def _ffprobe_path_for(ffmpeg_path):
    if not ffmpeg_path or ffmpeg_path == "ffmpeg":
        return "ffprobe"
    folder = os.path.dirname(os.path.abspath(ffmpeg_path))
    exe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    candidate = os.path.join(folder, exe_name)
    return candidate if os.path.isfile(candidate) else "ffprobe"
