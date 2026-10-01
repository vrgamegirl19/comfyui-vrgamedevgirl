import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple


_BULK_COMMENT_OR_BLANK = re.compile(r"^\s*(?:#|//|$).*$")
_BULK_PREFIX_STRIP = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s+)")


def shift_segment_timing(segment: Dict[str, Any], delta: float) -> None:
    """Shift a segment's start, end, and custom audio timing by delta seconds."""
    amount = float(delta or 0.0)
    if not segment or abs(amount) < 1e-6:
        return
    segment["start"] = round(float(segment.get("start", 0.0) or 0.0) + amount, 4)
    segment["end"] = round(float(segment.get("end", 0.0) or 0.0) + amount, 4)
    custom_start = segment.get("custom_audio_timeline_start")
    if custom_start is not None:
        try:
            segment["custom_audio_timeline_start"] = round(float(custom_start) + amount, 4)
        except (ValueError, TypeError):
            pass


def sort_segments(segments: List[Dict[str, Any]]) -> None:
    """Sort segments in-place ascending by start time."""
    segments.sort(key=lambda s: float(s.get("start", 0.0) or 0.0))


def renumber_generic_base_scene_labels(segments: List[Dict[str, Any]]) -> None:
    """Keep generic base-scene labels ('Scene N', 'N. Rooftop') synchronized with position."""
    sort_segments(segments)
    for index, segment in enumerate(segments, start=1):
        current = str(segment.get("label", "") or "").strip()
        desc_match = re.match(r"^\d+\.\s*(.+)$", current)
        if desc_match:
            segment["label"] = f"{index}. {desc_match.group(1)}"
        elif not current or re.match(r"^scene(?:\s+\d+(?:\.\d+)?)?$", current, re.IGNORECASE):
            segment["label"] = f"Scene {index}"


def scene_path_key(path: Any) -> str:
    """Normalize a filesystem path for case-insensitive matching."""
    return str(path or "").replace("\\", "/").lower()


def rewrite_renamed_scene_paths(node: Any, renamed: List[Tuple[str, str]]) -> None:
    """Point stored paths at renamed filenames; exact matches win over parent folder matches."""
    if not renamed or not node:
        return

    # Normalize lookup tables
    exact_map = {scene_path_key(orig): repl for orig, repl in renamed}
    parent_map = [(scene_path_key(orig) + "/", orig, repl) for orig, repl in renamed]

    def rename_val(value: str) -> str:
        key = scene_path_key(value)
        if key in exact_map:
            return exact_map[key]
        for parent_key, orig, repl in parent_map:
            if key.startswith(parent_key):
                suffix = value[len(orig):]
                return repl + suffix
        return value

    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, str):
                node[k] = rename_val(v)
            elif isinstance(v, (dict, list)):
                rewrite_renamed_scene_paths(v, renamed)
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            if isinstance(item, str):
                node[idx] = rename_val(item)
            elif isinstance(item, (dict, list)):
                rewrite_renamed_scene_paths(item, renamed)


def has_locked_video(segment: Dict[str, Any]) -> bool:
    """Return True if a scene has a rendered or imported video assigned."""
    if not isinstance(segment, dict):
        return False
    vid = str(segment.get("video_path") or segment.get("video_output") or "").strip()
    return bool(vid)


def normalize_segments(
    segments: List[Dict[str, Any]],
    active_segment: Optional[Dict[str, Any]] = None,
    active_index: Optional[int] = None,
    min_duration: float = 0.1,
) -> None:
    """Enforce rolling edit invariants after modifying a scene's start or end (Section 15.2 P2)."""
    sort_segments(segments)
    if not segments:
        return

    if active_segment is None and active_index is None:
        active_index = 0
    elif active_index is None and active_segment is not None:
        target_id = active_segment.get("id")
        active_index = next((i for i, s in enumerate(segments) if s.get("id") == target_id), -1)

    if active_index is None or active_index < 0 or active_index >= len(segments):
        return

    active = segments[active_index]
    active["start"] = max(0.0, float(active.get("start", 0.0) or 0.0))
    active["end"] = max(active["start"] + min_duration, float(active.get("end", active["start"] + 4.0) or active["start"] + 4.0))

    prev_scene = segments[active_index - 1] if active_index > 0 else None
    next_scene = segments[active_index + 1] if active_index + 1 < len(segments) else None

    if prev_scene is None:
        active["start"] = 0.0
    else:
        if has_locked_video(prev_scene):
            active["start"] = max(active["start"], float(prev_scene.get("end", 0.0) or 0.0))
        else:
            prev_start = float(prev_scene.get("start", 0.0) or 0.0)
            active["start"] = max(active["start"], prev_start + min_duration)
            prev_scene["end"] = active["start"]

    active["end"] = max(active["start"] + min_duration, active["end"])

    if next_scene is not None:
        if has_locked_video(next_scene):
            active["end"] = min(active["end"], float(next_scene.get("start", active["end"]) or active["end"]))
        else:
            next_end = float(next_scene.get("end", active["end"] + min_duration) or active["end"] + min_duration)
            active["end"] = min(active["end"], next_end - min_duration)
        active["end"] = max(active["start"] + min_duration, active["end"])
        if not has_locked_video(next_scene):
            next_scene["start"] = active["end"]
            if float(next_scene.get("end", 0.0) or 0.0) < next_scene["start"] + min_duration:
                next_scene["end"] = next_scene["start"] + min_duration

    if prev_scene is not None and not has_locked_video(prev_scene):
        prev_scene["end"] = active["start"]

    sort_segments(segments)


def close_base_timeline_gap(
    segments: List[Dict[str, Any]],
    overlay_segments: Optional[List[Dict[str, Any]]],
    start_time: float,
    end_time: float,
) -> float:
    """Ripple shift later base and overlay scenes left by duration (Section 15.2 P4)."""
    start = float(start_time or 0.0)
    end = float(end_time or 0.0)
    duration = max(0.0, end - start)
    if duration <= 0.0001:
        return 0.0

    delta = -duration
    threshold = end - 0.001

    for item in segments:
        if float(item.get("start", 0.0) or 0.0) >= threshold:
            shift_segment_timing(item, delta)

    if overlay_segments:
        for item in overlay_segments:
            if float(item.get("start", 0.0) or 0.0) >= threshold:
                shift_segment_timing(item, delta)

    sort_segments(segments)
    if overlay_segments:
        sort_segments(overlay_segments)
    return duration


def close_all_base_timeline_gaps(
    segments: List[Dict[str, Any]],
    overlay_segments: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """Walk base scenes in order, closing all gaps where start > previous end (Section 15.4 T15)."""
    sort_segments(segments)
    removed = 0.0
    cursor = 0.0

    for segment in segments:
        start = float(segment.get("start", 0.0) or 0.0)
        end = max(start + 0.05, float(segment.get("end", start + 0.05) or start + 0.05))

        if start > cursor + 0.001:
            gap = start - cursor
            shift_segment_timing(segment, -gap)
            removed += gap
            if overlay_segments:
                for overlay in overlay_segments:
                    if float(overlay.get("start", 0.0) or 0.0) >= start - 0.001:
                        shift_segment_timing(overlay, -gap)
        elif abs(start - cursor) <= 0.001 and start != cursor:
            shift_segment_timing(segment, cursor - start)

        cursor = max(cursor, float(segment.get("end", 0.0) or 0.0))

    sort_segments(segments)
    if overlay_segments:
        sort_segments(overlay_segments)
    return round(removed, 4)


def snap_to_beat(t: float, beats: List[Any], tolerance: float = 0.14) -> float:
    """Find nearest beat to timestamp t within tolerance seconds."""
    if not beats:
        return t
    time_val = float(t or 0.0)
    best_candidate = time_val
    min_dist = tolerance

    for b in beats:
        b_time = float(b["time"] if isinstance(b, dict) else b)
        dist = abs(b_time - time_val)
        if dist <= min_dist:
            min_dist = dist
            best_candidate = b_time

    return round(best_candidate, 4)


def parse_bulk_time_value(raw: str) -> float:
    """Parse time string in seconds (ss.mmm) or mm:ss or hh:mm:ss format."""
    text = str(raw or "").strip().replace(",", ".")
    if not text:
        raise ValueError("Empty time value")

    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    raise ValueError(f"Invalid time format: '{raw}'")


def parse_bulk_timings(text: str, mode: str = "durations") -> List[Tuple[float, float]]:
    """Parse multiline bulk timing text into a list of (start, end) tuples (Section 15.4 T16).
    
    Modes:
      - 'durations': each line is a scene duration (cursor increments).
      - 'ranges': lines formatted as 'start --> end' or 'start - end' or 'start to end'.
      - 'markers': timestamps in ascending order, consecutive pairs form scenes.
    """
    lines = [
        _BULK_PREFIX_STRIP.sub("", line.strip())
        for line in text.splitlines()
        if not _BULK_COMMENT_OR_BLANK.match(line)
    ]
    if not lines:
        return []

    results: List[Tuple[float, float]] = []

    if mode == "durations":
        cursor = 0.0
        for idx, line in enumerate(lines, start=1):
            try:
                dur = parse_bulk_time_value(line)
                if dur <= 0.05:
                    raise ValueError("Duration too short")
                start = cursor
                end = round(cursor + dur, 4)
                results.append((start, end))
                cursor = end
            except Exception as exc:
                raise ValueError(f"Line {idx} '{line}': invalid duration ({exc})") from exc

    elif mode == "ranges":
        range_split = re.compile(r"\s*(?:-->|->|–|-|to)\s*")
        for idx, line in enumerate(lines, start=1):
            parts = range_split.split(line)
            if len(parts) != 2:
                raise ValueError(f"Line {idx} '{line}': expected 'start - end'")
            try:
                start = parse_bulk_time_value(parts[0])
                end = parse_bulk_time_value(parts[1])
                if end <= start + 0.05:
                    raise ValueError("End time must be after start time")
                results.append((round(start, 4), round(end, 4)))
            except Exception as exc:
                raise ValueError(f"Line {idx} '{line}': invalid range ({exc})") from exc

    elif mode == "markers":
        timestamps: List[float] = []
        for idx, line in enumerate(lines, start=1):
            try:
                timestamps.append(parse_bulk_time_value(line))
            except Exception as exc:
                raise ValueError(f"Line {idx} '{line}': invalid timestamp ({exc})") from exc
        timestamps.sort()
        for i in range(len(timestamps) - 1):
            s, e = timestamps[i], timestamps[i + 1]
            if e > s + 0.05:
                results.append((round(s, 4), round(e, 4)))

    return results


def validate_timeline_consistency(
    segments: List[Dict[str, Any]],
    overlay_segments: Optional[List[Dict[str, Any]]] = None,
    audio_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Check timeline for overlaps, invalid boundaries, and gaps."""
    issues: List[Dict[str, Any]] = []
    if not segments:
        return issues

    sort_segments(segments)
    first = segments[0]
    if float(first.get("start", 0.0) or 0.0) > 0.001:
        issues.append({
            "type": "timeline_gap",
            "message": f"First scene does not start at 0.0 (starts at {first.get('start')}).",
            "fix": "Run close-gaps or set first scene start to 0.",
        })

    for i in range(len(segments) - 1):
        curr_scene = segments[i]
        next_scene = segments[i + 1]
        c_end = float(curr_scene.get("end", 0.0) or 0.0)
        n_start = float(next_scene.get("start", 0.0) or 0.0)

        if n_start < c_end - 0.001:
            issues.append({
                "type": "overlap",
                "message": f"Scene {i + 1} and Scene {i + 2} overlap ({c_end} > {n_start}).",
                "scenes": [curr_scene.get("id"), next_scene.get("id")],
            })
        elif n_start > c_end + 0.001:
            issues.append({
                "type": "gap",
                "message": f"Gap of {round(n_start - c_end, 3)}s between Scene {i + 1} and Scene {i + 2}.",
                "scenes": [curr_scene.get("id"), next_scene.get("id")],
            })

    if audio_duration is not None and audio_duration > 0:
        last_end = float(segments[-1].get("end", 0.0) or 0.0)
        if last_end > audio_duration + 0.05:
            issues.append({
                "type": "exceeds_audio",
                "message": f"Timeline duration ({last_end}s) exceeds audio duration ({audio_duration}s).",
            })

    return issues


def move_scene_timing(
    segments: List[Dict[str, Any]],
    scene_id: str,
    new_start: float,
    ripple: bool = False,
) -> Dict[str, Any]:
    """Move a scene to a new start time, with optional ripple shift for later scenes (Section 15.4 T10)."""
    sort_segments(segments)
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise ValueError(f"Scene {scene_id} not found")

    target = segments[idx]
    old_start = float(target.get("start", 0.0) or 0.0)
    old_end = float(target.get("end", old_start + 4.0) or old_start + 4.0)
    dur = max(0.1, old_end - old_start)
    target_start = max(0.0, round(float(new_start), 4))
    delta = target_start - old_start

    target["start"] = target_start
    target["end"] = round(target_start + dur, 4)

    if ripple:
        for s in segments[idx + 1:]:
            shift_segment_timing(s, delta)
    else:
        normalize_segments(segments, active_index=idx)

    sort_segments(segments)
    renumber_generic_base_scene_labels(segments)
    return {"scene": target, "delta": delta}


def resize_scene_timing(
    segments: List[Dict[str, Any]],
    scene_id: str,
    new_duration: Optional[float] = None,
    new_end: Optional[float] = None,
    ripple: bool = False,
) -> Dict[str, Any]:
    """Resize a scene duration or end boundary, with optional ripple shift (Section 15.4 T10)."""
    sort_segments(segments)
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise ValueError(f"Scene {scene_id} not found")

    target = segments[idx]
    start = float(target.get("start", 0.0) or 0.0)
    old_end = float(target.get("end", start + 4.0) or start + 4.0)

    if new_duration is not None:
        calc_end = max(start + 0.1, start + float(new_duration))
    elif new_end is not None:
        calc_end = max(start + 0.1, float(new_end))
    else:
        calc_end = old_end

    delta = calc_end - old_end
    target["end"] = round(calc_end, 4)

    if ripple:
        for s in segments[idx + 1:]:
            shift_segment_timing(s, delta)
    else:
        normalize_segments(segments, active_index=idx)

    sort_segments(segments)
    renumber_generic_base_scene_labels(segments)
    return {"scene": target, "delta": delta}


JOURNAL_FILENAME = ".timeline_journal.json"


class TimelineJournal:
    """Transactional rename and mutation journal for rolling timeline operations (Section 15.6)."""

    def __init__(self, project_folder: str, operation: str):
        self.project_folder = project_folder
        self.operation = operation
        self.journal_path = os.path.join(project_folder, JOURNAL_FILENAME)
        self.session_path = os.path.join(project_folder, "vrgdg_builder_session.json")
        self.backup_path = os.path.join(project_folder, f".timeline_session_backup_{int(time.time() * 1000)}.json")
        self.renames: List[Tuple[str, str]] = []
        self._active = False

    def start(self) -> None:
        """Create a journal file and backup the session state."""
        if os.path.isfile(self.session_path):
            shutil.copy2(self.session_path, self.backup_path)
        data = {
            "status": "in_progress",
            "operation": self.operation,
            "backup_session": self.backup_path,
            "renames": [],
            "created_at": time.time(),
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._active = True

    def record_renames(self, renames: List[Tuple[str, str]]) -> None:
        """Record executed file renames to the journal."""
        self.renames.extend(renames)
        if os.path.isfile(self.journal_path):
            data = {
                "status": "in_progress",
                "operation": self.operation,
                "backup_session": self.backup_path,
                "renames": [[s, t] for s, t in self.renames],
                "updated_at": time.time(),
            }
            with open(self.journal_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def commit(self) -> None:
        """Delete journal and cleanup backup on successful completion."""
        if os.path.isfile(self.journal_path):
            try:
                os.remove(self.journal_path)
            except Exception:
                pass
        if os.path.isfile(self.backup_path):
            try:
                os.remove(self.backup_path)
            except Exception:
                pass
        self._active = False

    def rollback(self) -> None:
        """Reverse all recorded file renames and restore the session backup."""
        for source, target in reversed(self.renames):
            if os.path.exists(target) and not os.path.exists(source):
                try:
                    os.rename(target, source)
                except Exception:
                    pass
        if os.path.isfile(self.backup_path):
            try:
                shutil.copy2(self.backup_path, self.session_path)
                os.remove(self.backup_path)
            except Exception:
                pass
        if os.path.isfile(self.journal_path):
            try:
                os.remove(self.journal_path)
            except Exception:
                pass
        self._active = False


def recover_pending_journal(project_folder: str) -> bool:
    """Check for dangling .timeline_journal.json from an interrupted operation and roll back."""
    journal_path = os.path.join(project_folder, JOURNAL_FILENAME)
    if not os.path.isfile(journal_path):
        return False
    try:
        with open(journal_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        renames = data.get("renames", [])
        backup_path = data.get("backup_session")
        session_path = os.path.join(project_folder, "vrgdg_builder_session.json")
        for item in reversed(renames):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                source, target = item
                if os.path.exists(target) and not os.path.exists(source):
                    try:
                        os.rename(target, source)
                    except Exception:
                        pass
        if backup_path and os.path.isfile(backup_path):
            shutil.copy2(backup_path, session_path)
            try:
                os.remove(backup_path)
            except Exception:
                pass
        os.remove(journal_path)
        return True
    except Exception:
        return False

