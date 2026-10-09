"""Record a rendered scene video on a timeline segment the way the Video Builder does.

The timeline shows a scene's picture from ``video_thumbnail_path`` and its history from ``video_history``
with the parallel ``video_thumbnail_history`` (``activateSegmentVideoPath`` / ``normalizeSegmentVideoHistory``
in ``timeline_state.mjs``). Setting only ``video_path`` leaves the scene on the timeline with no picture.
"""

import os
import time
from typing import Any, Dict


def _key(path: Any) -> str:
    return os.path.normcase(os.path.abspath(str(path))).replace("\\", "/") if str(path or "").strip() else ""


def _thumbnail_for(video_path: str, thumbnail_path: str) -> str:
    if str(thumbnail_path or "").strip():
        return str(thumbnail_path).strip()
    from ..builder.media import _builder_scene_video_thumbnail_path

    guess = _builder_scene_video_thumbnail_path(video_path)
    return guess if os.path.isfile(guess) else ""


def apply_scene_video(segment: Dict[str, Any], video_path: str, thumbnail_path: str = "") -> Dict[str, Any]:
    """Make ``video_path`` the scene's current video, with its thumbnail and history entry. Returns ``segment``."""
    video_path = str(video_path or "").strip()
    if not video_path:
        return segment
    thumbnail = _thumbnail_for(video_path, thumbnail_path)
    segment["video_path"] = video_path
    segment["rendered_video_path"] = video_path
    segment["video_folder"] = os.path.dirname(video_path)
    segment["video_status"] = "done"
    segment["scene_audio_render_dirty"] = False
    segment["preview_mode"] = "video"
    if thumbnail:
        segment["thumbnail_path"] = thumbnail

    history = [str(p) for p in segment.get("video_history") or [] if str(p).strip()]
    thumbs = [str(t or "") for t in segment.get("video_thumbnail_history") or []]
    thumbs += [""] * (len(history) - len(thumbs))
    by_key = {_key(p): thumbs[i] for i, p in enumerate(history)}
    if _key(video_path) not in by_key:
        history.append(video_path)
    if thumbnail:
        by_key[_key(video_path)] = thumbnail
    segment["video_history"] = history
    segment["video_thumbnail_history"] = [by_key.get(_key(p), "") for p in history]
    index = next(i for i, p in enumerate(history) if _key(p) == _key(video_path))
    segment["video_history_index"] = index
    segment["video_thumbnail_path"] = segment["video_thumbnail_history"][index]
    segment["video_cache_bust"] = int(time.time() * 1000)
    return segment
