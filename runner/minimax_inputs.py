"""MiniMax H3 inputs: templates, reference media, audio context trimming and output folders."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import wave
import folder_paths

from .paths import _ffprobe_path_for, _find_ffmpeg_path, _first_payload_value, _float_payload


_MINIMAX_H3_ASPECT_RATIOS = {
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
}


_MINIMAX_H3_MAX_REFERENCE_IMAGES = 9


_MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3


def _minimax_h3_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "minimax_audio_driven_builder_api.json",
    )


def _minimax_h3_2pass_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "minimax_audio_driven_builder_latent_upscale_2pass_api.json",
    )


def _minimax_h3_3pass_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "minimax_ref2video_3pass_audio_driven_api.json",
    )


def _minimax_h3_built_in_audio_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "minimax_built_in_audio_builder_api.json",
    )


def _minimax_h3_collection(value, collection_keys=()):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in collection_keys:
            if isinstance(value.get(key), list):
                return value[key]
        return list(value.values())
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if parsed is not None and parsed is not value:
        return _minimax_h3_collection(parsed, collection_keys)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _minimax_h3_media_path(value):
    if isinstance(value, dict):
        value = value.get("path") or value.get("file") or value.get("image") or value.get("video")
    return str(value or "").strip().strip('"').strip("'")


def _minimax_h3_image_paths(payload):
    raw = _first_payload_value(payload, "image_paths", "reference_images", "images", default=[])
    paths = [
        path
        for path in (
            _minimax_h3_media_path(item)
            for item in _minimax_h3_collection(raw, ("image_paths", "images"))
        )
        if path
    ]
    if len(paths) > _MINIMAX_H3_MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"MiniMax H3 supports at most {_MINIMAX_H3_MAX_REFERENCE_IMAGES} reference images; "
            f"received {len(paths)}."
        )
    return paths


def _minimax_h3_video_references(payload):
    raw = _first_payload_value(payload, "video_references", "reference_videos", "videos", default=[])
    references = []
    for item in _minimax_h3_collection(raw, ("video_references", "videos")):
        if isinstance(item, dict):
            path = _minimax_h3_media_path(item)
            try:
                start_seconds = max(0.0, float(_first_payload_value(
                    item, "start_seconds", "start", "seek_seconds", default=0
                ) or 0))
                duration = max(0.0, float(_first_payload_value(
                    item, "duration", "duration_seconds", default=0
                ) or 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("MiniMax H3 video reference timing must be numeric.") from exc
            use_audio_value = _first_payload_value(
                item, "use_audio", "include_audio", "reference_audio", default=False
            )
            use_audio = (
                str(use_audio_value).strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(use_audio_value, str)
                else bool(use_audio_value)
            )
        else:
            path = _minimax_h3_media_path(item)
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
    if len(references) > _MINIMAX_H3_MAX_REFERENCE_VIDEOS:
        raise ValueError(
            f"MiniMax H3 supports at most {_MINIMAX_H3_MAX_REFERENCE_VIDEOS} reference videos; "
            f"received {len(references)}."
        )
    return references


def _patch_minimax_h3_image_to_video_node(prompt, image_paths, include_references=False, has_last_frame=False):
    """Replace the H3 reference node with the native first/last-frame node.

    The standard MiniMax builder graph intentionally keeps the same downstream
    sampler/audio/output chain for both nodes: both emit positive conditioning
    and the combined H3 video/audio latent.  The media loader's first two image
    outputs therefore map directly to the native node's optional frame inputs.
    """
    node = prompt.get("136")
    if not isinstance(node, dict):
        raise KeyError("MiniMax H3 image-to-video node 136 was not found.")

    paths = list(image_paths or [])
    if not paths:
        raise ValueError("MiniMax H3 image-to-video requires a first-frame image.")
    if not include_references and len(paths) > 2:
        raise ValueError("MiniMax H3 image-to-video accepts at most a first and last frame.")

    inputs = {
        "prompt": ["138", 0],
        "width": ["115", 0],
        "height": ["115", 1],
        "length": ["131", 1],
        "clip": ["128", 0],
        "vae": ["119", 0],
        "first_frame": ["180", 0],
    }
    if (include_references and has_last_frame) or (not include_references and len(paths) > 1):
        inputs["last_frame"] = ["180", 1]

    if include_references:
        ref_start = 2 if has_last_frame else 1
        for ref_index, loader_slot in enumerate(range(ref_start, len(paths))):
            inputs[f"ref_images.ref_image_{ref_index}"] = ["180", loader_slot]
        inputs["ref_image_size"] = "max"
        node["class_type"] = "VRGDG_MiniMaxH3ImageReferenceToVideo"
        node["_meta"] = {"title": "MiniMax H3 Image + Reference to Video"}
        node["inputs"] = inputs
        return

    node["class_type"] = "MiniMaxH3ImageToVideo"
    node["_meta"] = {"title": "MiniMax H3 Image to Video"}
    node["inputs"] = inputs


def _probe_media_duration_seconds(path):
    ffprobe_path = _ffprobe_path_for(_find_ffmpeg_path())
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "FFprobe could not read the audio duration.").strip())
    try:
        duration = float((result.stdout or "").strip().splitlines()[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"FFprobe did not return a valid duration for: {path}") from exc
    if duration <= 0:
        raise ValueError(f"Source audio has no usable duration: {path}")
    return duration


def _trim_minimax_h3_audio_context(source_path, project_folder, scene_number, timing):
    target_dir = os.path.join(project_folder, "minimax_h3_scene_audio")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"scene_audio_{scene_number:04d}.wav")
    ffmpeg_path = _find_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{timing.audio_trim_start_seconds:.9f}",
        "-i",
        source_path,
        "-t",
        f"{timing.audio_trim_duration_seconds:.9f}",
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        target_path,
    ]
    if timing.audio_leading_padding_seconds > 0:
        cmd[-1:-1] = ["-af", f"adelay={timing.audio_leading_padding_seconds * 1000:.9f}:all=1"]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(target_path):
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg failed to trim MiniMax H3 scene audio.").strip())
    try:
        with wave.open(target_path, "rb") as handle:
            actual_duration = handle.getnframes() / float(handle.getframerate())
    except Exception as exc:
        raise RuntimeError(f"Could not verify the trimmed MiniMax H3 audio: {target_path}") from exc
    if actual_duration + 0.02 < timing.audio_trim_duration_seconds:
        raise ValueError(
            "The trimmed MiniMax H3 audio ended before the required scene context. "
            f"Needed {timing.audio_trim_duration_seconds:.3f}s; received {actual_duration:.3f}s."
        )
    return {
        "audio_path": target_path,
        "start": timing.audio_trim_start_seconds,
        "duration": actual_duration,
        "requested_duration": timing.audio_trim_duration_seconds,
        "format": "pcm_s16le_wav",
    }


def _prepare_scene_audio_clip(payload):
    source_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not source_path:
        raise ValueError("Audio file path is empty.")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Audio file was not found: {source_path}")
    if not project_folder:
        raise ValueError("Create or load a project before preparing scene audio.")
    os.makedirs(project_folder, exist_ok=True)
    scene_number = int(_float_payload(payload, "scene_number", 1, minimum=1, maximum=9999))
    start = _float_payload(payload, "start_seconds", 0.0, minimum=0.0, maximum=24 * 60 * 60)
    duration = _float_payload(payload, "duration_seconds", 8.0, minimum=0.05, maximum=120.0)
    target_dir = os.path.join(project_folder, "minimax_h3_scene_audio")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"scene_audio_{scene_number:04d}.wav")
    ffmpeg_path = _find_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start:.9f}",
        "-i",
        source_path,
        "-t",
        f"{duration:.9f}",
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        target_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(target_path):
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg failed to prepare scene audio.").strip())
    actual_duration = _probe_media_duration_seconds(target_path)
    return {
        "audio_path": target_path,
        "start": start,
        "duration": actual_duration,
        "requested_duration": duration,
        "format": "pcm_s16le_wav",
    }


def _minimax_h3_output_location(project_folder, scene_number, *, create=True):
    project_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        os.path.basename(os.path.normpath(project_folder)),
    ).strip("_") or "project"
    project_key = hashlib.sha1(os.path.normcase(project_folder).encode("utf-8")).hexdigest()[:8]
    relative_dir = os.path.join(
        "VRGDG_MiniMaxH3",
        f"{project_name}_{project_key}",
        f"scene_{scene_number:04d}",
    )
    output_folder = os.path.join(folder_paths.get_output_directory(), relative_dir)
    if create:
        os.makedirs(output_folder, exist_ok=True)
    filename_prefix = os.path.join(
        relative_dir,
        f"MiniMaxH3_scene_{scene_number:04d}",
    ).replace("\\", "/")
    return output_folder, filename_prefix


def _cleanup_minimax_h3_output_folder(payload):
    output_value = str(payload.get("output_folder") or "").strip().strip('"')
    project_value = str(payload.get("project_folder") or "").strip().strip('"')
    if not output_value or not project_value:
        raise ValueError("Output folder and project folder are required.")
    scene_number = int(payload.get("scene_number", 0))
    if scene_number < 1 or scene_number > 999999:
        raise ValueError("A valid scene number is required.")
    project_folder = os.path.abspath(project_value)
    expected, _ = _minimax_h3_output_location(project_folder, scene_number, create=False)
    output_folder = os.path.abspath(output_value)
    root = os.path.realpath(os.path.join(folder_paths.get_output_directory(), "VRGDG_MiniMaxH3"))
    resolved = os.path.realpath(output_folder)
    try:
        inside_root = os.path.normcase(os.path.commonpath([root, resolved])) == os.path.normcase(root)
    except ValueError:
        inside_root = False
    if (not inside_root or os.path.normcase(resolved) == os.path.normcase(root)
            or os.path.normcase(output_folder) != os.path.normcase(os.path.abspath(expected))
            or os.path.normcase(resolved) != os.path.normcase(output_folder)):
        raise ValueError("Output folder must be the scene's MiniMax H3 scratch directory without links.")
    if not os.path.isdir(output_folder):
        return {"removed": False}
    shutil.rmtree(output_folder)
    return {"removed": True, "output_folder": output_folder}
