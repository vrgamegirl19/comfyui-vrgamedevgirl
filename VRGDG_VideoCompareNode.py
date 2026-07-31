import os

import folder_paths


_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def _video_path_candidates(value):
    candidates = []
    if isinstance(value, str):
        candidates.append(value)
    elif isinstance(value, dict):
        for key in ("fullpath", "path", "video_path", "filename"):
            item = value.get(key)
            if isinstance(item, str):
                candidates.append(item)
        for key in ("files", "filenames", "videos", "gifs"):
            candidates.extend(_video_path_candidates(value.get(key)))
    elif isinstance(value, (list, tuple)):
        for item in value:
            candidates.extend(_video_path_candidates(item))
    return candidates


def _resolve_video_path(value, label):
    roots = [
        "",
        folder_paths.get_output_directory(),
        folder_paths.get_temp_directory(),
        folder_paths.get_input_directory(),
    ]
    candidates = _video_path_candidates(value)
    for raw_path in reversed(candidates):
        text = str(raw_path or "").strip().strip('"')
        if not text or os.path.splitext(text)[1].lower() not in _VIDEO_EXTENSIONS:
            continue
        for root in roots:
            path = text if not root or os.path.isabs(text) else os.path.join(root, text)
            path = os.path.normpath(os.path.abspath(path))
            if os.path.isfile(path):
                return path
    raise ValueError(
        f"{label} video was not found. Connect the Filenames output from a "
        "Video Combine node that has already created a video."
    )


class VRGDG_VideoCompareSlider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before_video": (
                    "VHS_FILENAMES",
                    {"tooltip": "Connect the Filenames output for the original/before video."},
                ),
                "after_video": (
                    "VHS_FILENAMES",
                    {"tooltip": "Connect the Filenames output for the processed/after video."},
                ),
                "slider_position": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Starting position of the before/after wipe.",
                    },
                ),
                "before_label": (
                    "STRING",
                    {"default": "Before", "tooltip": "Label shown over the original video."},
                ),
                "after_label": (
                    "STRING",
                    {"default": "After", "tooltip": "Label shown over the processed video."},
                ),
                "show_labels": ("BOOLEAN", {"default": True}),
                "loop": ("BOOLEAN", {"default": True}),
                "muted": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Mute playback. When unmuted, audio comes from the before video.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("VHS_FILENAMES", "VHS_FILENAMES")
    RETURN_NAMES = ("before_video", "after_video")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "VRGDG/Image"

    def compare(
        self,
        before_video,
        after_video,
        slider_position,
        before_label,
        after_label,
        show_labels,
        loop,
        muted,
    ):
        before_path = _resolve_video_path(before_video, "Before")
        after_path = _resolve_video_path(after_video, "After")
        return {
            "ui": {
                "compare_videos": [
                    {
                        "compare_role": "before",
                        "path": before_path,
                        "name": os.path.basename(before_path),
                    },
                    {
                        "compare_role": "after",
                        "path": after_path,
                        "name": os.path.basename(after_path),
                    },
                ],
                "compare_video_settings": {
                    "slider_position": float(slider_position),
                    "before_label": str(before_label or "Before"),
                    "after_label": str(after_label or "After"),
                    "show_labels": bool(show_labels),
                    "loop": bool(loop),
                    "muted": bool(muted),
                },
            },
            "result": (before_video, after_video),
        }


NODE_CLASS_MAPPINGS = {
    "VRGDG_VideoCompareSlider": VRGDG_VideoCompareSlider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_VideoCompareSlider": "VRGDG Video Compare Slider",
}
