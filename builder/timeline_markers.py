"""Timed Timeline Notes (the Video Builder's ``+ Timeline Note`` markers).

The notes live in the session's ``timeline_markers`` list. Python twin of ``normalizeTimelineMarkers``
and ``newTimelineMarker`` (``web/music_video_builder/timeline_state.mjs`` and ``segments.mjs``): every
marker keeps its ``id``, ``start``, ``end`` (``None`` for a point note), ``type``, ``label`` and
``note``, and the list is sorted by start time. Story Arc planning reads the saved markers through
``storyboard/timeline_notes.py``.
"""

import math
import random
import time
from typing import Any, Dict, List, Optional

DEFAULT_TYPE = "note"
DEFAULT_LABEL = "Timeline note"
_TEXT_LIMITS = {"type": 120, "label": 500, "note": 20000}


class TimelineMarkerError(ValueError):
    """A timeline note request that cannot be applied."""


def new_marker_id() -> str:
    """An id in the Builder's ``mark_<ms>_<n>`` form."""
    return f"mark_{int(time.time() * 1000)}_{random.randint(0, 9999)}"


def _seconds(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise TimelineMarkerError(f"{key} must be a number of seconds.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TimelineMarkerError(f"{key} must be a number of seconds.") from None
    if not math.isfinite(number) or number < 0:
        raise TimelineMarkerError(f"{key} must be a number of seconds that is 0 or more.")
    return number


def _text(value: Any, key: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise TimelineMarkerError(f"{key} must be text.")
    text = str(value).strip()
    if len(text) > _TEXT_LIMITS[key]:
        raise TimelineMarkerError(f"{key} is longer than {_TEXT_LIMITS[key]} characters.")
    return text


def normalize_marker(marker: Dict[str, Any]) -> Dict[str, Any]:
    """One saved marker in the Builder's shape. A missing or non-increasing end makes a point note."""
    source = marker if isinstance(marker, dict) else {}
    try:
        start = max(0.0, float(source.get("start") or 0))
    except (TypeError, ValueError):
        start = 0.0
    if not math.isfinite(start):
        start = 0.0
    try:
        end = float(source.get("end"))
    except (TypeError, ValueError):
        end = None
    if end is not None and (not math.isfinite(end) or end <= start):
        end = None
    return {
        "id": str(source.get("id") or new_marker_id()),
        "start": start,
        "end": end,
        "type": str(source.get("type") or DEFAULT_TYPE).strip() or DEFAULT_TYPE,
        "label": str(source.get("label") or DEFAULT_LABEL).strip() or DEFAULT_LABEL,
        "note": str(source.get("note") or "").strip(),
    }


def normalize_markers(markers: Any) -> List[Dict[str, Any]]:
    """All saved markers, normalized and sorted by start like the Builder's list."""
    items = [normalize_marker(item) for item in (markers if isinstance(markers, list) else [])]
    return sorted(items, key=lambda item: item["start"])


def _apply_fields(marker: Dict[str, Any], fields: Dict[str, Any]) -> None:
    if "start" in fields:
        marker["start"] = _seconds(fields["start"], "start")
    if "end" in fields:
        # null, "" or an explicit point request turns a range note back into a point note.
        marker["end"] = None if fields["end"] in (None, "") else _seconds(fields["end"], "end")
    for key in ("type", "label", "note"):
        if key in fields:
            marker[key] = _text(fields[key], key)
    if not marker["type"]:
        marker["type"] = DEFAULT_TYPE
    if not marker["label"]:
        marker["label"] = DEFAULT_LABEL
    if marker["end"] is not None and marker["end"] <= marker["start"]:
        raise TimelineMarkerError("end must be later than start. Send end: null for a point note.")


_EDITABLE = ("start", "end", "type", "label", "note")


def _check_keys(fields: Dict[str, Any], allowed: tuple) -> None:
    if not isinstance(fields, dict):
        raise TimelineMarkerError("The timeline note must be a JSON object.")
    unknown = sorted(set(fields) - set(allowed))
    if unknown:
        raise TimelineMarkerError(
            f"Unknown timeline note field{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(allowed)}."
        )


def create_marker(markers: List[Dict[str, Any]], fields: Dict[str, Any]) -> Dict[str, Any]:
    """Add a note to ``markers`` (already normalized) and return it. ``start`` is required."""
    _check_keys(fields, ("id",) + _EDITABLE)
    if "start" not in fields:
        raise TimelineMarkerError("start is required (seconds on the project timeline).")
    marker_id = str(fields.get("id") or "").strip() or new_marker_id()
    if any(item["id"] == marker_id for item in markers):
        raise TimelineMarkerError(f"A timeline note with id '{marker_id}' already exists.")
    marker = {"id": marker_id, "start": 0.0, "end": None, "type": DEFAULT_TYPE, "label": DEFAULT_LABEL, "note": ""}
    _apply_fields(marker, fields)
    markers.append(marker)
    markers.sort(key=lambda item: item["start"])
    return marker


def find_marker(markers: List[Dict[str, Any]], marker_id: str) -> Optional[Dict[str, Any]]:
    return next((item for item in markers if item["id"] == str(marker_id)), None)


def update_marker(markers: List[Dict[str, Any]], marker_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """Change only the given fields of one note. Raises ``KeyError`` when the id is unknown."""
    _check_keys(fields, ("id",) + _EDITABLE)
    if "id" in fields and str(fields["id"]) != str(marker_id):
        raise TimelineMarkerError("A timeline note id cannot change.")
    marker = find_marker(markers, marker_id)
    if marker is None:
        raise KeyError(marker_id)
    updated = dict(marker)
    _apply_fields(updated, {key: value for key, value in fields.items() if key != "id"})
    marker.update(updated)
    markers.sort(key=lambda item: item["start"])
    return marker


def delete_marker(markers: List[Dict[str, Any]], marker_id: str) -> Dict[str, Any]:
    """Remove one note and return it. Raises ``KeyError`` when the id is unknown."""
    marker = find_marker(markers, marker_id)
    if marker is None:
        raise KeyError(marker_id)
    markers.remove(marker)
    return marker


def overlapping_scene_ids(marker: Dict[str, Any], segments: List[Dict[str, Any]]) -> List[str]:
    """Scenes a note applies to, by the same overlap rule Story Arc planning uses."""
    start, end = marker["start"], marker["end"]
    ids = []
    for segment in sorted((s for s in segments if isinstance(s, dict)), key=lambda s: float(s.get("start") or 0)):
        try:
            seg_start, seg_end = float(segment.get("start") or 0), float(segment.get("end") or 0)
        except (TypeError, ValueError):
            continue
        if seg_end <= seg_start:
            continue
        inside = (seg_start <= start < seg_end) if end is None else (start < seg_end and end > seg_start)
        if inside:
            ids.append(str(segment.get("id") or ""))
    return ids
