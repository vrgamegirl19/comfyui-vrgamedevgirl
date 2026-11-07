"""Thumbnail intro: keep a generated thumbnail in the project and put it in front of the stitched video.

The Builder's Thumbnail dialog makes the picture (an LLM writes the edit prompt, NanoBanana or Flux/Klein draws it) and
calls ``_save_thumbnail_image`` to keep it in ``<project>/thumbnail``. After the final stitch the Builder calls
``_prepend_thumbnail_intro``, which makes the picture the video's first frame: exactly one frame in front of the first
scene, with the same length of silence. The scene clips, the timeline and the audio sync are never touched: only the
stitched video is rewritten.
"""

import math
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

THUMBNAIL_FOLDER = "thumbnail"
THUMBNAIL_CAPABILITIES_VERSION = 1
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


def _thumbnail_capabilities() -> Dict[str, Any]:
    """Tell the Builder that this build can save thumbnails, prepend the intro, take one exact frame of a video and make an overlay.

    ``overlay`` is the Overlay engine (``thumbnail_overlay.py``): a build without it does not offer that engine.
    """
    return {"version": THUMBNAIL_CAPABILITIES_VERSION, "frame": True, "overlay": True}


def _clean_path(raw: Any) -> str:
    return os.path.abspath(str(raw or "").strip().strip('"'))


def _next_thumbnail_path(folder: str, prefix: str = "thumbnail") -> str:
    """Return ``<prefix>_NNNN.png`` with the first number that is free in ``folder``."""
    taken = set(os.listdir(folder)) if os.path.isdir(folder) else set()
    number = 1
    while f"{prefix}_{number:04d}.png" in taken:
        number += 1
    return os.path.join(folder, f"{prefix}_{number:04d}.png")


def _save_thumbnail_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a generated image (a ComfyUI view descriptor or a file path) into the project's thumbnail folder.

    Every call keeps its own numbered PNG, so earlier tries stay available in the dialog's history.
    """
    from ..runner.paths import _resolve_comfy_image_path
    from ..runner.video_files import _safe_project_subfolder

    # An empty path would resolve to the folder the server runs in, so it is refused before any folder is made.
    if not str(payload.get("project_folder") or "").strip().strip('"'):
        raise ValueError("Project folder is empty.")
    _project, folder = _safe_project_subfolder(payload.get("project_folder", ""), THUMBNAIL_FOLDER)
    image_info = payload.get("image")
    if isinstance(image_info, dict):
        source = _resolve_comfy_image_path(image_info)
    else:
        source = _clean_path(payload.get("source_path"))
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Thumbnail image was not found: {source}")
    target = _next_thumbnail_path(folder)
    with Image.open(source) as image:
        width, height = image.size
        image.convert("RGB").save(target, format="PNG")
    return {"image_path": target, "width": width, "height": height}


def _format_duration(path: str, ffmpeg_path: str) -> float:
    """Length of a media file in seconds, from ffprobe (0 when unknown)."""
    from ..runner.paths import _ffprobe_path_for

    result = subprocess.run(
        [_ffprobe_path_for(ffmpeg_path), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, errors="replace",
    )
    try:
        return float((result.stdout or "").strip())
    except ValueError:
        return 0.0


def _video_packets(path: str, ffmpeg_path: str) -> int:
    """How many video frames a file holds, counted from its packets (no decoding, so it is quick even for a long video)."""
    from ..runner.video_files import _probe_stream_fields

    fields = _probe_stream_fields(path, ffmpeg_path, "v:0", ["nb_read_packets"], count_packets=True)
    try:
        return int(fields.get("nb_read_packets") or 0)
    except ValueError:
        return 0


def _frame_rate(values: Dict[str, str]) -> Fraction:
    for key in ("r_frame_rate", "avg_frame_rate"):
        try:
            rate = Fraction(values.get(key, "0/1"))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return Fraction(24)


def _extract_thumbnail_frame(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Take one exact frame of a video as ``<project>/thumbnail/source_NNNN.png`` (the picture the thumbnail is made from).

    ``seconds`` is the moment in ``video_path``. It is snapped to the nearest frame, and the seek lands half a frame
    before that frame's time, so the frame the Builder showed at that moment is the one that is saved, not its neighbor.
    A moment past the end gives the last frame.
    """
    from ..runner.paths import _find_ffmpeg_path
    from ..runner.video_files import _probe_stream_fields, _safe_project_subfolder

    if not str(payload.get("project_folder") or "").strip().strip('"'):
        raise ValueError("Project folder is empty.")
    video = _clean_path(payload.get("video_path"))
    if not os.path.isfile(video) or os.path.splitext(video)[1].lower() not in _VIDEO_SUFFIXES:
        raise FileNotFoundError(f"The video to take the frame from was not found: {video}")
    try:
        seconds = float(payload.get("seconds"))
    except (TypeError, ValueError):
        raise ValueError("Say which moment of the video to use (seconds).") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("The moment in the video must be 0 seconds or more.")

    ffmpeg_path = _find_ffmpeg_path()
    fields = _probe_stream_fields(video, ffmpeg_path, "v:0", ["width", "height", "r_frame_rate", "avg_frame_rate"])
    rate = _frame_rate(fields)
    duration = _format_duration(video, ffmpeg_path)
    frame = int(math.floor(seconds * float(rate) + 0.5))
    if duration > 0:
        frame = min(frame, max(0, int(math.floor(duration * float(rate) + 0.5)) - 1))
    start = max(0.0, (frame - 0.5) / float(rate)) if frame > 0 else 0.0

    _project, folder = _safe_project_subfolder(payload.get("project_folder", ""), THUMBNAIL_FOLDER)
    target = _next_thumbnail_path(folder, "source")
    try:
        _run_ffmpeg([ffmpeg_path, "-y", "-ss", f"{start:.6f}", "-i", video, "-frames:v", "1", "-an", target])
        if not os.path.isfile(target) or os.path.getsize(target) == 0:
            raise RuntimeError("FFmpeg did not find a frame at that moment of the video.")
        with Image.open(target) as image:
            width, height = image.size
    except Exception:
        if os.path.isfile(target):
            os.remove(target)
        raise
    return {
        "image_path": target, "width": width, "height": height,
        "frame": frame, "seconds": frame / float(rate), "fps": float(rate),
    }


def _run_ffmpeg(command: List[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{(result.stderr or result.stdout or '').strip()[-2000:]}")


def _intro_command(
    ffmpeg_path: str, image_path: str, output: str, seconds: float, size: Tuple[int, int], rate: Fraction,
    audio: Optional[Tuple[int, int]], crf: Any,
) -> List[str]:
    """ffmpeg command that turns the picture into a one frame clip (with a matching silence) in the video's size, frame rate and audio format.

    ``seconds`` is the length of that one frame (1 / frame rate). ``-frames:v 1`` makes it exactly one frame.
    """
    from ..runner.video_files import _scene_audio_encode_args, _scene_video_encode_args

    width, height = size
    fill = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,format=yuv420p"
    command = [ffmpeg_path, "-y", "-loop", "1", "-framerate", str(rate), "-t", f"{seconds:.6f}", "-i", image_path]
    if audio:
        sample_rate, channels = audio
        layout = "mono" if channels == 1 else "stereo"
        command += ["-f", "lavfi", "-t", f"{seconds:.6f}", "-i", f"anullsrc=r={sample_rate}:cl={layout}"]
    command += ["-vf", fill, "-r", str(rate), "-frames:v", "1", *_scene_video_encode_args(crf)]
    if audio:
        command += [*_scene_audio_encode_args(), "-ar", str(audio[0]), "-ac", str(audio[1]), "-shortest"]
    else:
        command += ["-an"]
    return [*command, output]


def _join_by_copy(ffmpeg_path: str, intro: str, source: str, output: str, work: str) -> None:
    """Join without re-encoding the video: the intro was made to match, so the streams can be copied."""
    listing = os.path.join(work, "join.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        for path in (intro, source):
            handle.write("file '{}'\n".format(path.replace("\\", "/").replace("'", "'\\''")))
    _run_ffmpeg([ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", output])


def _join_by_encoding(ffmpeg_path: str, intro: str, source: str, output: str, has_audio: bool, crf: Any) -> None:
    """Join with the concat filter and one new encode, for the rare video the stream-copy join does not fit."""
    from ..runner.video_files import _scene_audio_encode_args, _scene_video_encode_args

    if has_audio:
        graph, maps = "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]", ["-map", "[v]", "-map", "[a]"]
    else:
        graph, maps = "[0:v][1:v]concat=n=2:v=1:a=0[v]", ["-map", "[v]", "-an"]
    command = [ffmpeg_path, "-y", "-i", intro, "-i", source, "-filter_complex", graph, *maps, *_scene_video_encode_args(crf)]
    if has_audio:
        command += _scene_audio_encode_args()
    _run_ffmpeg([*command, "-movflags", "+faststart", output])


def _prepend_thumbnail_intro(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make ``image_path`` the first frame of ``input_path`` (a stitched video): one frame in front of it, in place by default."""
    from ..runner.paths import _find_ffmpeg_path
    from ..runner.video_files import _probe_stream_fields, _source_crf

    source = _clean_path(payload.get("input_path"))
    image_path = _clean_path(payload.get("image_path"))
    if not os.path.isfile(source) or os.path.splitext(source)[1].lower() not in _VIDEO_SUFFIXES:
        raise FileNotFoundError(f"The video to put the thumbnail in front of was not found: {source}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The thumbnail image was not found: {image_path}")
    with Image.open(image_path) as image:
        image.verify()
    replace_source = payload.get("replace_source", True) is not False

    ffmpeg_path = _find_ffmpeg_path()
    video = _probe_stream_fields(source, ffmpeg_path, "v:0", ["width", "height", "r_frame_rate", "avg_frame_rate"])
    if not video.get("width") or not video.get("height"):
        raise RuntimeError(f"Could not read the size of the video: {source}")
    audio_fields = _probe_stream_fields(source, ffmpeg_path, "a:0", ["sample_rate", "channels"])
    audio = None
    if audio_fields.get("sample_rate"):
        audio = (int(audio_fields["sample_rate"]), max(1, int(audio_fields.get("channels") or 2)))
    size = (int(video["width"]), int(video["height"]))
    rate = _frame_rate(video)
    seconds = 1.0 / float(rate)
    crf = _source_crf([source], payload.get("crf"))
    source_frames = _video_packets(source, ffmpeg_path)

    folder = os.path.dirname(source)
    work = tempfile.mkdtemp(prefix=".thumbnail_intro_", dir=folder)
    try:
        intro = os.path.join(work, "intro.mp4")
        _run_ffmpeg(_intro_command(ffmpeg_path, image_path, intro, seconds, size, rate, audio, crf))
        joined = os.path.join(work, "joined.mp4")
        method = "copy"
        try:
            _join_by_copy(ffmpeg_path, intro, source, joined, work)
            # One frame is far shorter than a length check could tell, so the frames are counted: exactly one more than before.
            if _video_packets(joined, ffmpeg_path) != source_frames + 1:
                raise RuntimeError("The joined video does not have exactly one more frame.")
        except RuntimeError:
            method = "reencode"
            _join_by_encoding(ffmpeg_path, intro, source, joined, audio is not None, crf)
        if replace_source:
            destination = source
        else:
            stem, extension = os.path.splitext(source)
            destination = f"{stem}_with_thumbnail{extension}"
        os.replace(joined, destination)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "final_video_path": destination,
        "intro_frames": 1,
        "intro_seconds": seconds,
        "method": method,
        "duration": _format_duration(destination, ffmpeg_path),
    }
