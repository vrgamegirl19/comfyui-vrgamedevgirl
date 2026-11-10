"""Video intro: put the person's own video in front of the stitched video.

The Thumbnail dialog lets the person choose a video file they already have (a logo sting, a title card, an artist intro).
After the final stitch the Builder calls ``_prepend_video_intro``, which makes that clip match the stitched video (size,
frame rate, pixel format, audio format) and joins it in front of the first scene. The whole clip is kept: it is fitted
inside the frame with black bars when its shape differs, never cropped, and its own sound plays (silence when it has
none). The scene clips, the timeline and the audio sync are never touched: only the stitched video is rewritten, so
stitching again gives the plain video.

When the Thumbnail intro is on too, the Builder joins this clip first and the thumbnail picture second, so the picture is
the very first frame of the finished video and the intro follows it.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple

from .thumbnail import (
    _VIDEO_SUFFIXES, _clean_path, _format_duration, _frame_rate, _join_by_copy, _join_by_encoding, _run_ffmpeg, _video_packets,
)


def _intro_clip_command(
    ffmpeg_path: str, intro: str, output: str, seconds: float, size: Tuple[int, int], rate: Any,
    intro_has_audio: bool, audio: Optional[Tuple[int, int]], crf: Any,
) -> list:
    """ffmpeg command that turns the intro into a clip in the stitched video's size, frame rate and audio format.

    The picture is fitted inside the frame (black bars, no cropping). ``seconds`` is the intro's own length: ``-t`` and
    ``apad`` make the sound exactly that long, so the join keeps the video and the sound together. With ``audio`` set the
    clip always has a sound track (the intro's own, or silence), so the stream layout equals the stitched video's.
    """
    from ..runner.video_files import _scene_audio_encode_args, _scene_video_encode_args

    width, height = size
    fit = (
        f"fps={rate},scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p"
    )
    command = [ffmpeg_path, "-y", "-i", intro]
    if audio and not intro_has_audio:
        sample_rate, channels = audio
        layout = "mono" if channels == 1 else "stereo"
        command += ["-f", "lavfi", "-t", f"{seconds:.6f}", "-i", f"anullsrc=r={sample_rate}:cl={layout}"]
    command += ["-map", "0:v:0"]
    if audio:
        command += ["-map", "0:a:0" if intro_has_audio else "1:a:0"]
    command += ["-vf", fit, "-r", str(rate), *_scene_video_encode_args(crf)]
    if audio:
        command += [*_scene_audio_encode_args(), "-ar", str(audio[0]), "-ac", str(audio[1]), "-af", "apad"]
    else:
        command += ["-an"]
    return [*command, "-t", f"{seconds:.6f}", output]


def _prepend_video_intro(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make ``intro_path`` the start of ``input_path`` (a stitched video): joined in front of it, in place by default."""
    from ..runner.paths import _find_ffmpeg_path
    from ..runner.video_files import _probe_stream_fields, _source_crf

    source = _clean_path(payload.get("input_path"))
    intro = _clean_path(payload.get("intro_path"))
    if not os.path.isfile(source) or os.path.splitext(source)[1].lower() not in _VIDEO_SUFFIXES:
        raise FileNotFoundError(f"The video to put the intro in front of was not found: {source}")
    if not os.path.isfile(intro) or os.path.splitext(intro)[1].lower() not in _VIDEO_SUFFIXES:
        raise FileNotFoundError(f"The intro video was not found: {intro}")
    if os.path.samefile(source, intro):
        raise ValueError("The intro video is the final video itself. Choose a different file.")
    replace_source = payload.get("replace_source", True) is not False

    ffmpeg_path = _find_ffmpeg_path()
    video = _probe_stream_fields(source, ffmpeg_path, "v:0", ["width", "height", "r_frame_rate", "avg_frame_rate"])
    if not video.get("width") or not video.get("height"):
        raise RuntimeError(f"Could not read the size of the video: {source}")
    if not _probe_stream_fields(intro, ffmpeg_path, "v:0", ["width"]).get("width"):
        raise RuntimeError(f"The intro file has no video picture: {intro}")
    audio_fields = _probe_stream_fields(source, ffmpeg_path, "a:0", ["sample_rate", "channels"])
    audio = None
    if audio_fields.get("sample_rate"):
        audio = (int(audio_fields["sample_rate"]), max(1, int(audio_fields.get("channels") or 2)))
    intro_has_audio = bool(_probe_stream_fields(intro, ffmpeg_path, "a:0", ["sample_rate"]).get("sample_rate"))
    seconds = _format_duration(intro, ffmpeg_path)
    if seconds <= 0:
        raise RuntimeError(f"Could not read the length of the intro video: {intro}")
    size = (int(video["width"]), int(video["height"]))
    rate = _frame_rate(video)
    crf = _source_crf([source], payload.get("crf"))
    source_frames = _video_packets(source, ffmpeg_path)

    folder = os.path.dirname(source)
    work = tempfile.mkdtemp(prefix=".video_intro_", dir=folder)
    try:
        clip = os.path.join(work, "intro.mp4")
        _run_ffmpeg(_intro_clip_command(ffmpeg_path, intro, clip, seconds, size, rate, intro_has_audio, audio, crf))
        intro_frames = _video_packets(clip, ffmpeg_path)
        if intro_frames < 1:
            raise RuntimeError(f"The intro video has no frames: {intro}")
        joined = os.path.join(work, "joined.mp4")
        method = "copy"
        try:
            _join_by_copy(ffmpeg_path, clip, source, joined, work)
            # The frames are counted, so a join that dropped or repeated one is noticed and redone with a new encode.
            if _video_packets(joined, ffmpeg_path) != source_frames + intro_frames:
                raise RuntimeError("The joined video does not have the intro's frames plus the video's frames.")
        except RuntimeError:
            method = "reencode"
            _join_by_encoding(ffmpeg_path, clip, source, joined, audio is not None, crf)
        if replace_source:
            destination = source
        else:
            stem, extension = os.path.splitext(source)
            destination = f"{stem}_with_intro{extension}"
        os.replace(joined, destination)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "final_video_path": destination,
        "intro_frames": intro_frames,
        "intro_seconds": intro_frames / float(rate),
        "method": method,
        "duration": _format_duration(destination, ffmpeg_path),
    }
