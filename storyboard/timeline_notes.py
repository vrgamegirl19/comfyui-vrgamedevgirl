"""Timed user notes used to plan the storyboard's story arc."""

import math
from typing import Any


def _time_seconds(value: Any) -> float | None:
    """Return a finite timestamp, rejecting malformed timing values."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, number) if math.isfinite(number) else None


def story_arc_timeline_notes(payload: dict) -> list[dict]:
    """Preserve note text and times, identifying intersecting scene cards."""
    storyboard = payload.get("storyboard")
    storyboard = storyboard if isinstance(storyboard, dict) else {}
    markers = payload.get("timeline_markers", storyboard.get("timeline_markers", []))
    scenes = payload.get("scenes", storyboard.get("scenes", []))
    if not isinstance(markers, list):
        return []
    scenes = scenes if isinstance(scenes, list) else []
    notes = []
    for marker in markers:
        if not isinstance(marker, dict):
            continue
        note = str(marker.get("note") or "").strip()
        start = _time_seconds(marker.get("start", 0))
        if not note or start is None:
            continue
        end = _time_seconds(marker.get("end"))
        end = end if end is not None and end > start else None
        scene_numbers = []
        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                continue
            scene_start = _time_seconds(scene.get("timeline_start", scene.get("start", 0)))
            scene_end = _time_seconds(scene.get("timeline_end", scene.get("end")))
            if scene_start is None or scene_end is None or scene_end <= scene_start:
                continue
            overlaps = (scene_start <= start < scene_end) if end is None else (
                start < scene_end and end > scene_start
            )
            if overlaps:
                scene_numbers.append(scene.get("scene_number") or index)
        notes.append({
            "start": start,
            "end": end,
            "type": str(marker.get("type") or "note").strip(),
            "label": str(marker.get("label") or "Timeline note").strip(),
            "note": note,
            "scene_numbers": scene_numbers,
        })
    return sorted(notes, key=lambda item: item["start"])
