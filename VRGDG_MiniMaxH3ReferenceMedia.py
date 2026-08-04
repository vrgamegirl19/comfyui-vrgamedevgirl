import json
import os
import re

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps


MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
REFERENCE_VIDEO_FPS = 24
REFERENCE_VIDEO_MAX_FRAMES = 15 * REFERENCE_VIDEO_FPS


def _parse_path_values(raw, collection_keys=()):
    text = str(raw or "").strip()
    if not text:
        return []

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        pass

    if isinstance(parsed, list):
        values = parsed
    elif isinstance(parsed, dict):
        values = None
        for key in collection_keys:
            if isinstance(parsed.get(key), list):
                values = parsed[key]
                break
        if values is None:
            values = list(parsed.values())
    else:
        values = re.split(r"[\r\n]+", text)
    return values


def _clean_path(value):
    if isinstance(value, dict):
        value = value.get("path") or value.get("file") or value.get("image") or value.get("video") or ""
    return str(value or "").strip().strip('"').strip("'")


def _parse_image_paths(raw):
    return [
        path
        for path in (_clean_path(item) for item in _parse_path_values(raw, ("image_paths", "images")))
        if path
    ]


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_nonnegative_float(value, default=0.0):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, float(default))


def _parse_video_references(raw):
    references = []
    for item in _parse_path_values(raw, ("video_references", "videos")):
        if isinstance(item, dict):
            path = _clean_path(item)
            start_seconds = _as_nonnegative_float(
                item.get("start_seconds", item.get("start", item.get("seek_seconds", 0)))
            )
            duration = _as_nonnegative_float(
                item.get("duration_seconds", item.get("duration", 0))
            )
            use_audio = _as_bool(
                item.get("use_audio", item.get("include_audio", item.get("reference_audio", False)))
            )
        else:
            path = _clean_path(item)
            start_seconds = 0.0
            duration = 0.0
            use_audio = False
        if path:
            references.append({
                "path": path,
                "start_seconds": start_seconds,
                "duration": duration,
                "use_audio": use_audio,
            })
    return references


def _resolve_media_path(raw_path):
    path_text = _clean_path(raw_path)
    if not path_text:
        raise FileNotFoundError("MiniMax H3 reference media path was empty.")

    candidates = []
    if os.path.isabs(path_text):
        candidates.append(path_text)
    else:
        candidates.extend([
            path_text,
            os.path.abspath(path_text),
            os.path.join(folder_paths.get_input_directory(), path_text),
            os.path.join(folder_paths.get_output_directory(), path_text),
        ])
        get_temp_directory = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_directory):
            candidates.append(os.path.join(get_temp_directory(), path_text))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(os.path.abspath(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isfile(normalized):
            return normalized
    raise FileNotFoundError(f"MiniMax H3 reference media was not found: {path_text}")


def _load_image_tensor(raw_path):
    resolved = _resolve_media_path(raw_path)
    with Image.open(resolved) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _vhs_video_loader():
    import nodes

    loader_class = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoPath")
    if loader_class is None:
        raise RuntimeError(
            "MiniMax H3 video references require Video Helper Suite's VHS_LoadVideoPath node."
        )
    return loader_class()


def _load_video_reference(reference, slot_index):
    resolved = _resolve_media_path(reference["path"])
    fps = REFERENCE_VIDEO_FPS
    start_seconds = _as_nonnegative_float(reference.get("start_seconds", 0))
    duration = _as_nonnegative_float(reference.get("duration", 0))
    skip_first_frames = max(0, round(start_seconds * fps))
    frame_load_cap = (
        min(REFERENCE_VIDEO_MAX_FRAMES, max(1, round(duration * fps)))
        if duration > 0
        else REFERENCE_VIDEO_MAX_FRAMES
    )

    loader = _vhs_video_loader()
    load_function = getattr(loader, getattr(loader, "FUNCTION", "load_video"))
    frames, _frame_count, audio, _video_info = load_function(
        video=resolved,
        force_rate=fps,
        custom_width=0,
        custom_height=0,
        frame_load_cap=frame_load_cap,
        skip_first_frames=skip_first_frames,
        select_every_nth=1,
        unique_id=f"vrgdg_minimax_h3_reference_video_{slot_index}",
    )
    return frames, audio if reference.get("use_audio") else None


def _pad_slots(values, count):
    values = list(values[:count])
    return values + [None] * (count - len(values))


class VRGDG_MiniMaxH3ReferenceMediaFromPaths:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": (
                            "Ordered MiniMax H3 reference images. Use a JSON list, an object with "
                            "image_paths/images, or one path per line. Supports up to 9 images."
                        ),
                    },
                ),
                "video_references": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": (
                            "Ordered MiniMax H3 reference videos. Use a JSON list of paths or objects "
                            "with path, start_seconds, duration, and use_audio. Supports up to 3 videos."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",) * MAX_REFERENCE_IMAGES + ("IMAGE",) * MAX_REFERENCE_VIDEOS + ("AUDIO",) * MAX_REFERENCE_VIDEOS
    RETURN_NAMES = (
        tuple(f"ref_image_{index}" for index in range(MAX_REFERENCE_IMAGES))
        + tuple(f"ref_video_{index}" for index in range(MAX_REFERENCE_VIDEOS))
        + tuple(f"ref_video_audio_{index}" for index in range(MAX_REFERENCE_VIDEOS))
    )
    FUNCTION = "load_references"
    CATEGORY = "VRGDG/Video/Conditioning"
    DESCRIPTION = (
        "Builder-friendly ordered media loader for MiniMax H3. It accepts project/input/output paths, "
        "keeps each image and video in its own H3 reference slot, and returns None for unused slots."
    )

    def load_references(self, image_paths, video_references):
        paths = _parse_image_paths(image_paths)
        videos = _parse_video_references(video_references)
        if len(paths) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"MiniMax H3 supports at most {MAX_REFERENCE_IMAGES} reference images; received {len(paths)}."
            )
        if len(videos) > MAX_REFERENCE_VIDEOS:
            raise ValueError(
                f"MiniMax H3 supports at most {MAX_REFERENCE_VIDEOS} reference videos; received {len(videos)}."
            )

        image_outputs = _pad_slots([_load_image_tensor(path) for path in paths], MAX_REFERENCE_IMAGES)
        loaded_videos = [
            _load_video_reference(reference, index)
            for index, reference in enumerate(videos)
        ]
        video_outputs = _pad_slots([item[0] for item in loaded_videos], MAX_REFERENCE_VIDEOS)
        video_audio_outputs = _pad_slots([item[1] for item in loaded_videos], MAX_REFERENCE_VIDEOS)
        return tuple(image_outputs + video_outputs + video_audio_outputs)


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3ReferenceMediaFromPaths": VRGDG_MiniMaxH3ReferenceMediaFromPaths,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3ReferenceMediaFromPaths": "VRGDG MiniMax H3 Reference Media From Paths",
}
