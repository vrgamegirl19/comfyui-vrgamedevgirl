"""Builder graphs for ComfyUI partner video nodes (no provider credentials stored here)."""

import base64
import math
import os
import shutil
import subprocess
import uuid

import folder_paths


def _seedance_ffmpeg_path():
    from .paths import _find_ffmpeg_path
    return _find_ffmpeg_path()


def _seedance_audio_source_duration(path):
    from .minimax_inputs import _probe_media_duration_seconds
    return _probe_media_duration_seconds(path)


def _prepare_seedance_scene_audio(payload):
    """Trim only this scene's audio, then pad silence to Seedance's whole-second duration."""
    source = os.path.abspath(str(payload.get("source_path") or "").strip().strip('"'))
    project = str(payload.get("project_folder") or "").strip().strip('"')
    if not os.path.isfile(source):
        raise ValueError("Project/custom scene audio was not found. Load audio before using Seedance scene-audio mode.")
    if not project or not os.path.isdir(project):
        raise ValueError("Set an existing project folder before preparing Seedance audio.")
    start, duration = float(payload.get("start", 0)), float(payload.get("duration", 0))
    if not math.isfinite(start) or start < 0 or not math.isfinite(duration) or not 0 < duration <= 30:
        raise ValueError("Invalid Seedance scene audio timing. Scenes must be longer than zero and at most 30 seconds.")
    source_duration = _seedance_audio_source_duration(source)
    if source_duration <= 0 or start >= source_duration - 0.001:
        raise ValueError("The scene starts after its source audio ends. Check the scene's audio offset.")
    read_duration = float(payload.get("source_duration", duration))
    leading = float(payload.get("leading_silence", 0))
    if not math.isfinite(read_duration) or read_duration <= 0 or read_duration > duration or not math.isfinite(leading) or not 0 <= leading < duration:
        raise ValueError("Invalid custom audio clip placement or duration.")
    generation_duration = max(4, math.ceil(duration))
    folder = os.path.join(os.path.abspath(project), "seedance_scene_audio")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"scene_audio_{uuid.uuid4().hex}.wav")
    command = [_seedance_ffmpeg_path(), "-y", "-ss", f"{start:.9f}", "-i", source,
               "-vn", "-af", f"atrim=duration={read_duration:.9f},asetpts=PTS-STARTPTS,adelay={leading * 1000:.6f}:all=1,apad=whole_dur={generation_duration}",
               "-t", str(generation_duration), "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", path]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(path):
        raise RuntimeError((result.stderr or "Could not prepare Seedance scene audio.").strip())
    # An empty seek must fail before a paid API submission, instead of becoming all-silence guidance.
    import wave
    with wave.open(path, "rb") as handle:
        frames = handle.getnframes()
    if frames == 0:
        raise ValueError("The scene starts after its source audio ends. Check the scene's audio offset.")
    return {"audio_path": path, "source_path": source, "start": start,
            "duration": duration, "generation_duration": generation_duration}


def _seedance_scene_audio_prompt(text, payload):
    """Bind Audio 1 to the prepared clip without claiming a deterministic lip-sync API."""
    if payload.get("audio_mode") != "scene_audio":
        return text
    if len(payload.get("reference_audios") or []) != 1:
        raise ValueError("Scene-audio mode requires exactly one prepared project/custom audio excerpt as Audio 1.")
    image = int(payload.get("lip_sync_image") or 0)
    images = len(payload.get("reference_images") or [])
    if image < 0 or image > images:
        raise ValueError("Select a lip-sync performer image that is actually attached to this scene.")
    if payload.get("no_lip_sync"):
        performance = "This is a non-vocal/instrumental performance shot. Keep every person's lips closed; use the audio only for musical timing and body/instrument movement."
    elif image:
        performance = (f"The performer from @Image {image} lip-syncs to the actual vocals/speech in @Audio 1. "
                       "Match the supplied words, phrasing, phonemes, melody and pauses at their original timestamps. "
                       "Keep the mouth visible during vocal lines. Other people keep their lips closed unless the scene explicitly assigns them a vocal part.")
    else:
        performance = ("Use the existing scene's singer/speaker assignments to lip-sync to the actual vocals/speech in @Audio 1. "
                       "Match the supplied words, phrasing, phonemes and pauses at their original timestamps. "
                       "Keep the assigned singer's face and mouth readable during vocal phrases. "
                       "Keep non-singers' lips closed and preserve all instrumental-only/lips-closed shot instructions.")
    return text + ("\n\nScene audio timing:\n@Audio 1 is the exact project/custom audio excerpt for this scene, starting at scene time 0. "
                   "Use its original timing and tempo; do not rewrite words, invent vocals, or stretch or retime the performance. "
                   "Any trailing silence is padding after the scene ends. " + performance +
                   " The Builder restores the supplied audio to the finished scene and uses the project soundtrack in final assembly.")


def _stage_reference(item, kind):
    """Use unique input names even when several references are staged in one millisecond."""
    if isinstance(item, str):
        item = {"path": item}
    source = str(item.get("path") or "").strip().strip('"')
    if not source and kind == "image" and item.get("data"):
        raw = str(item["data"])
        header, encoded = raw.split(",", 1) if raw.startswith("data:") else ("", raw)
        ext = ".jpg" if "jpeg" in header else ".webp" if "webp" in header else ".png"
        target = f"vrgdg_seedance_{uuid.uuid4().hex}{ext}"
        with open(os.path.join(folder_paths.get_input_directory(), target), "wb") as handle:
            handle.write(base64.b64decode(encoded, validate=True))
        return target
    if not source or not os.path.isfile(source):
        raise ValueError(f"Seedance {kind} reference was not found: {source or '(empty)'}")
    extensions = {
        "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
        "video": {".mp4", ".mov", ".webm", ".mkv", ".m4v"},
        "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4"},
    }
    ext = os.path.splitext(source)[1].lower()
    if ext not in extensions[kind]:
        raise ValueError(f"Unsupported Seedance {kind} reference format: {ext}")
    target = f"vrgdg_seedance_{uuid.uuid4().hex}{ext}"
    shutil.copy2(source, os.path.join(folder_paths.get_input_directory(), target))
    return target


def _build_seedance25_reference_prompt(payload):
    import nodes

    required = {"ByteDance2ReferenceNodeV2", "SaveVideo"}
    missing = required - nodes.NODE_CLASS_MAPPINGS.keys()
    if missing:
        raise ValueError("Update ComfyUI and enable API nodes. Missing nodes: " + ", ".join(sorted(missing)))
    text = str(payload.get("prompt") or "").strip()
    if not text or len(text) > 15000:
        raise ValueError("Seedance needs a prompt between 1 and 15,000 characters.")
    duration = float(payload.get("duration", 5))
    if not math.isfinite(duration) or not 0 < duration <= 30:
        raise ValueError("Seedance scenes must be longer than zero and at most 30 seconds. Split longer scenes first.")
    resolution = payload.get("resolution", "720p")
    ratio = payload.get("ratio", "16:9")
    if resolution not in {"480p", "720p", "1080p"}:
        raise ValueError("Unsupported Seedance resolution.")
    if ratio not in {"16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"}:
        raise ValueError("Unsupported Seedance aspect ratio.")
    groups = {}
    for kind, limit in (("image", 30), ("video", 10), ("audio", 10)):
        items = payload.get(f"reference_{kind}s") or []
        if not isinstance(items, list) or len(items) > limit:
            raise ValueError(f"Seedance accepts at most {limit} {kind} references.")
        groups[kind] = items
    if not any(groups.values()):
        raise ValueError("Reference to Video needs at least one image, video, or audio reference.")
    for kind, loader in (("image", "LoadImage"), ("video", "LoadVideo"), ("audio", "LoadAudio")):
        if groups[kind] and loader not in nodes.NODE_CLASS_MAPPINGS:
            raise ValueError(f"ComfyUI is missing {loader}. Update ComfyUI first.")

    text = _seedance_scene_audio_prompt(text, payload)
    if len(text) > 15000:
        raise ValueError("The prompt plus scene-audio timing instructions exceeds 15,000 characters. Shorten the prompt first.")

    inputs = {
        "model": "Seedance 2.5",
        "model.prompt": text,
        "model.resolution": resolution,
        "model.ratio": ratio,
        "model.duration": max(4, math.ceil(duration)),
        "model.generate_audio": bool(payload.get("generate_audio", True)),
        "model.task_type": "reference",
        "model.output_format": "mp4",
        "model.auto_downscale": True,
        "model.auto_upscale": False,
        "seed": int(payload.get("seed", 0)) % 2147483648,
        "watermark": False,
    }
    prompt = {}
    for kind, loader, field in (("image", "LoadImage", "image"), ("video", "LoadVideo", "file"), ("audio", "LoadAudio", "audio")):
        for index, item in enumerate(groups[kind], 1):
            node_id = f"{kind}_{index}"
            prompt[node_id] = {"class_type": loader, "inputs": {field: _stage_reference(item, kind)}}
            inputs[f"model.reference_{kind}s.{kind}_{index}"] = [node_id, 0]
    prefix = f"vrgdg_api_video/seedance25_{uuid.uuid4().hex}/video"
    prompt["seedance"] = {"class_type": "ByteDance2ReferenceNodeV2", "inputs": inputs}
    prompt["save_video"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ["seedance", 0], "filename_prefix": prefix, "format": "mp4", "format.codec": "auto"},
    }
    return {
        "prompt": prompt,
        "output_root": folder_paths.get_output_directory(),
        "generation_duration": inputs["model.duration"],
        "timeline_duration": duration,
    }
