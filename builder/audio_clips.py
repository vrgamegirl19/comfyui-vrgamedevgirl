"""Render non-destructive timeline audio edits with silence between clips."""

import hashlib
import json
import math
import os
import subprocess
import uuid
from typing import Any

from .audio import _find_ffmpeg_path, _read_audio_peaks


def prepare_audio_clip_mix(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a cached PCM timeline mix without modifying any source file."""
    project = str(payload.get("project_folder") or "").strip()
    clips = payload.get("clips")
    if not project or not isinstance(clips, list):
        raise ValueError("Project folder and an audio clip list are required.")
    duration = float(payload.get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0 or duration > 86400:
        raise ValueError("Audio timeline duration must be between 0 and 86400 seconds.")
    normalized = []
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("Each audio clip must be an object.")
        volume = float(clip.get("volume", 1))
        if not math.isfinite(volume) or not 0 <= volume <= 2:
            raise ValueError("Audio clip volume must be between 0 and 2 (0% to 200%).")
        if clip.get("muted"):
            volume = 0
        included = clip.get("include_in_generation", clip.get("role", "dialogue") == "dialogue")
        path = os.path.abspath(str(clip.get("path") or ""))
        start = float(clip.get("start", 0))
        source_start = float(clip.get("source_start", 0))
        length = float(clip.get("duration", 0))
        if not all(math.isfinite(value) for value in (start, source_start, length)):
            raise ValueError("Audio clip timing must be finite.")
        if start < 0 or source_start < 0 or length <= 0 or start + length > 86400:
            raise ValueError("Audio clip timing is outside the supported range.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio clip source not found: {path}")
        duration = max(duration, start + length)
        if payload.get("generation_only") and not included:
            continue
        normalized.append({"path": path, "start": start, "source_start": source_start,
                           "duration": length, "volume": volume, "mtime": os.stat(path).st_mtime_ns})
    key = hashlib.sha256(json.dumps([normalized, duration], sort_keys=True).encode()).hexdigest()[:24]
    folder = os.path.join(os.path.abspath(project), "project_audio", "clip_edits")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f"audio_edit_{key}.wav")
    if os.path.isfile(target):
        return {"audio_path": target, "duration": duration, "peaks": _read_audio_peaks(target, 1600)["peaks"]}
    temporary = os.path.join(folder, f".{key}_{uuid.uuid4().hex}.wav")
    command = [_find_ffmpeg_path(), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    filters = [f"[0:a]atrim=duration={duration:.9f}[silence]"]
    for index, clip in enumerate(normalized, start=1):
        command.extend(["-ss", str(clip["source_start"]), "-t", str(clip["duration"]),
                        "-i", clip["path"]])
        delay = round(clip["start"] * 44100)
        filters.append(f"[{index}:a]aresample=44100,aformat=channel_layouts=stereo,"
                       f"atrim=duration={clip['duration']:.9f},asetpts=PTS-STARTPTS,"
                       f"volume={clip['volume']:.9f},"
                       f"adelay={delay}S:all=1[c{index}]")
    inputs = "[silence]" + "".join(f"[c{index}]" for index in range(1, len(normalized) + 1))
    filters.append(f"{inputs}amix=inputs={len(normalized) + 1}:duration=first:normalize=0[out]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-t", str(duration),
                    "-c:a", "pcm_s16le", temporary])
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Could not prepare edited timeline audio.")
        os.replace(temporary, target)
    finally:
        if os.path.isfile(temporary):
            os.remove(temporary)
    return {"audio_path": target, "duration": duration, "peaks": _read_audio_peaks(target, 1600)["peaks"]}
