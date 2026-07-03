import os
import time
from pathlib import Path


def safe_preview_token(value, fallback="media"):
    token = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value or "")).strip("_")
    return token or fallback


def preview_stamp():
    return int(time.time() * 1000)


def preview_output_path(root, scene_id, input_path, suffix, extension=".jpg"):
    source_stem = Path(input_path).stem
    safe_scene = safe_preview_token(scene_id, "scene")
    safe_source = safe_preview_token(source_stem, "media")
    return os.path.join(root, f"{safe_scene}_{safe_source}_{suffix}_{preview_stamp()}{extension}")


def preview_source_frame_path(root, scene_id, input_path, stamp=None):
    source_stem = Path(input_path).stem
    safe_scene = safe_preview_token(scene_id, "scene")
    safe_source = safe_preview_token(source_stem, "media")
    frame_stamp = preview_stamp() if stamp is None else stamp
    return os.path.join(root, f"{safe_scene}_{safe_source}_source_frame_{frame_stamp}.jpg")


def save_rgb_preview_frame(frame, output_path, quality=92):
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(frame).save(output_path, quality=quality)
    return output_path


def source_preview_payload(path, temporary=False):
    return {
        "source_preview_path": path or "",
        "source_preview_temporary": bool(temporary),
    }


def delete_preview_file_quietly(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass
