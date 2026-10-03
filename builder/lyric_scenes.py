"""Scenes from timed lyric lines, and scene-length rules (Python twin of the Video Builder's Line Mapping).

Ports ``createSegmentsFromTimestampedLyricsPayload``, ``normalizeTimestampedSceneDurations``,
``mergeTimestampedLyricText`` and ``applyLyricSectionsFromReferenceText`` from
``web/music_video_builder/lyric_transcription.mjs`` / ``lyric_cues.mjs`` so the Agent API makes the same
scenes from the same timed lines. It adds ``next_length_fix`` / ``enforce_scene_lengths``: merge scenes
shorter than a minimum, split scenes longer than a maximum into equal parts, and keep the lyric text
with the right scene while doing it.

Pure Python: no ComfyUI, aiohttp or torch imports.
"""

import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

REFERENCE_UNIT_MODES = ("reference_lines", "exact_reference_lines", "reference_stanzas")
DEFAULT_INSTRUMENTAL_TEXT = "[instrumental]"
_EPSILON = 0.03

_NON_VOCAL_MARKER = re.compile(r"^(instrumental|break|interlude|solo|no vocal|no vocals|no lyrics|silence|b roll|music only)$")
_SECTION_WORDS = (
    r"intro|verse|pre[\s-]?chorus|chorus|post[\s-]?chorus|bridge|outro|refrain|hook|breakdown|drop|interlude|"
    r"instrumental(?:\s+break)?|solo|break|spoken(?:\s+word)?|rap"
)
_STRUCTURAL_SECTION = re.compile(rf"^(?:{_SECTION_WORDS})(?:\s+(?:\d+|[ivxlcdm]+))?$", re.IGNORECASE)
_INSTRUMENTAL_MARKER = re.compile(r"\binstrumental|no vocals?|no singing|no lip\s*-?\s*sync|no lipsync|b-?roll|visual only|silence\b", re.IGNORECASE)
_LABEL_WORDS = (
    r"intro|outro|bridge|verse|chorus|pre-chorus|prechorus|hook|refrain|interlude|break|section|instrumental|music|"
    r"no vocals?|no singing|no lip\s*-?\s*sync|no lipsync|b-?roll|visual only|silence"
)


def new_segment_id() -> str:
    """Scene id in the ``seg_<hex>`` form the rest of the API uses."""
    return f"seg_{uuid.uuid4().hex[:12]}"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def is_instrumental_lyric_text(text: Any) -> bool:
    """Mirror ``isInstrumentalLyricText``: scene text that means "no vocals"."""
    value = re.sub(r"\s+", " ", _text(text).lower())
    if not value:
        return False
    if value in ("instrumental", "[instrumental]", "instrumental section", "instrumental section."):
        return True
    stripped = re.sub(rf"\[(?:{_LABEL_WORDS})\]", " ", value, flags=re.IGNORECASE)
    stripped = re.sub(rf"\b(?:{_LABEL_WORDS})\b", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"[\W_]+", " ", stripped).strip()
    return bool(_INSTRUMENTAL_MARKER.search(value)) and not stripped


def _is_non_vocal_marker_label(label: str) -> bool:
    clean = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", label.lower())).strip()
    return bool(_NON_VOCAL_MARKER.match(clean))


def clean_timestamped_lyric_text(text: Any) -> str:
    """Mirror ``cleanTimestampedLyricText``: drop ``[Verse 1]`` style headers, tidy spacing."""
    stripped = re.sub(
        r"\[([^\]]{2,80})\]",
        lambda match: match.group(0) if _is_non_vocal_marker_label(match.group(1)) else " ",
        str(text or ""),
    )
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = re.sub(r"([(\[{])\s+", r"\1", stripped)
    stripped = re.sub(r"\s+([)\]}])", r"\1", stripped)
    return stripped.strip()


def merge_lyric_text(a: Any, b: Any, instrumental_text: str = DEFAULT_INSTRUMENTAL_TEXT) -> str:
    """Mirror ``mergeTimestampedLyricText``: join two scenes' lyrics, ignoring instrumental markers."""
    values = [_text(value) for value in (a, b) if _text(value)]
    vocal = [value for value in values if not is_instrumental_lyric_text(value)]
    if not vocal:
        return values[0] if values else instrumental_text
    seen: List[str] = []
    for value in vocal:
        if value not in seen:
            seen.append(value)
    return "\n".join(seen)


def split_lyric_text(text: Any, fraction: float, instrumental_text: str = DEFAULT_INSTRUMENTAL_TEXT) -> Tuple[str, str]:
    """Divide a scene's lyrics between its two halves when it is split.

    ``fraction`` is how much of the scene's time the left half takes. Lines are divided first; a
    single line is divided between its words. Instrumental scenes stay instrumental on both sides.
    """
    value = _text(text)
    if not value or is_instrumental_lyric_text(value):
        return (value or instrumental_text, value or instrumental_text)
    fraction = min(0.95, max(0.05, fraction))
    lines = [line.strip() for line in value.replace("\r\n", "\n").split("\n") if line.strip()]
    if len(lines) >= 2:
        left_count = min(len(lines) - 1, max(1, round(len(lines) * fraction)))
        return ("\n".join(lines[:left_count]), "\n".join(lines[left_count:]))
    words = value.split()
    if len(words) < 2:
        return (value, instrumental_text)
    left_count = min(len(words) - 1, max(1, round(len(words) * fraction)))
    return (" ".join(words[:left_count]), " ".join(words[left_count:]))


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def carry_lyrics_over(old_segments: List[Dict[str, Any]], ranges: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """Lyrics for new scene ranges, taken from the scenes they replace.

    Each old scene's lyric goes to the new range it overlaps. When several new ranges overlap one old
    scene its lyric is divided between them in time order (lines first, then words, like splitting a
    scene). When several old scenes land in one new range their lyrics are joined. An old scene that sits
    in a gap goes to the nearest range. Returns one dict per range with ``lyric_text``,
    ``lyric_singers`` and ``lyric_no_lip_sync`` (empty when nothing landed there).
    """
    pieces: List[List[str]] = [[] for _ in ranges]
    singers: List[List[str]] = [[] for _ in ranges]
    ordered = sorted(
        (seg for seg in old_segments if isinstance(seg, dict)),
        key=lambda seg: _number(seg.get("start")),
    )
    for old in ordered:
        text = _text(old.get("lyric_text"))
        if not text or not ranges:
            continue
        old_start = _number(old.get("start"))
        old_end = max(old_start, _number(old.get("end"), old_start))
        weights = [
            (index, _overlap(old_start, old_end, start, end))
            for index, (start, end) in enumerate(ranges)
        ]
        weights = [(index, weight) for index, weight in weights if weight > 0.0]
        if not weights:
            middle = (old_start + old_end) / 2.0
            nearest = min(
                range(len(ranges)),
                key=lambda index: min(abs(middle - ranges[index][0]), abs(middle - ranges[index][1])),
            )
            weights = [(nearest, 1.0)]
        remaining = text
        remaining_weight = sum(weight for _, weight in weights)
        for position, (index, weight) in enumerate(weights):
            if position == len(weights) - 1:
                piece = remaining
            else:
                piece, remaining = split_lyric_text(remaining, weight / remaining_weight)
                remaining_weight -= weight
            if _text(piece):
                pieces[index].append(piece)
            for name in old.get("lyric_singers") or []:
                name = _text(name)
                if name and name not in singers[index]:
                    singers[index].append(name)
    carried: List[Dict[str, Any]] = []
    for index in range(len(ranges)):
        merged = ""
        for piece in pieces[index]:
            merged = merge_lyric_text(merged, piece) if merged else _text(piece)
        entry: Dict[str, Any] = {}
        if merged:
            entry["lyric_text"] = merged
            entry["lyric_no_lip_sync"] = is_instrumental_lyric_text(merged)
            if singers[index]:
                entry["lyric_singers"] = singers[index]
        carried.append(entry)
    return carried


# ---------------------------------------------------------------------------
# Section labels (Verse 1, Bridge ...) from the pasted reference lyrics
# ---------------------------------------------------------------------------

def _section_lookup_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[‘’'`]", "", text)
    text = re.sub(r"[^\w' ]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lyric_section_map_from_reference_text(text: Any) -> Dict[str, str]:
    """Map each lyric line (normalized) to the ``[Section]`` header above it."""
    mapping: Dict[str, str] = {}
    current = ""
    for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        tags = list(re.finditer(r"\[([^\]]{1,80})\]", line))
        prefix_match = re.match(r"^(?:\s*\[[^\]]{1,80}\])+\s*", line)
        tag_prefix = prefix_match.group(0) if tags and tags[0].start() == 0 and prefix_match else ""
        structural = next(
            (re.sub(r"\s+", " ", m.group(1)).strip() for m in tags
             if _STRUCTURAL_SECTION.match(re.sub(r"\s+", " ", m.group(1)).strip())),
            None,
        )
        terminal = any(re.match(r"^(?:end|end of song)$", m.group(1).strip(), re.IGNORECASE) for m in tags)
        if structural and tag_prefix:
            current = structural
            key = _section_lookup_text(line[len(tag_prefix):].strip())
            if key and key not in mapping:
                mapping[key] = current
            continue
        if terminal and tag_prefix and not line[len(tag_prefix):].strip():
            current = ""
            continue
        key = _section_lookup_text(line)
        if key and current and key not in mapping:
            mapping[key] = current
    return mapping


def apply_lyric_sections(segments: List[Dict[str, Any]], reference_text: Any) -> int:
    """Set ``lyric_section`` on scenes from the reference lyrics. Returns how many were set."""
    section_map = lyric_section_map_from_reference_text(reference_text)
    if not section_map:
        return 0
    entries = list(section_map.items())
    applied = 0
    for segment in segments:
        existing = _text(segment.get("lyric_section")).lower()
        lyric = _text(segment.get("lyric_text"))
        if existing and existing != "instrumental":
            continue
        if not lyric or is_instrumental_lyric_text(lyric):
            segment["lyric_section"] = "instrumental"
            continue
        for line in [part.strip() for part in lyric.replace("\r\n", "\n").replace("\r", "\n").split("\n") if part.strip()]:
            header = re.match(r"^\[([^\]]{2,80})\]$", line)
            if header:
                segment["lyric_section"] = header.group(1).strip()
                applied += 1
                break
            key = _section_lookup_text(line)
            if key in section_map:
                segment["lyric_section"] = section_map[key]
                applied += 1
                break
            if len(key) >= 6:
                fuzzy = next((value for ref_key, value in entries if len(ref_key) >= 6 and (ref_key in key or key in ref_key)), None)
                if fuzzy:
                    segment["lyric_section"] = fuzzy
                    applied += 1
                    break
    return applied


# ---------------------------------------------------------------------------
# Timed lines -> scenes
# ---------------------------------------------------------------------------

def _new_scene(start: float, end: float, label: str = "") -> Dict[str, Any]:
    return {
        "id": new_segment_id(),
        "track": "base",
        "start": start,
        "end": end,
        "label": label,
        "notes": "",
        "timeline_note": "",
        "lyric_text": "",
        "lyric_section": "",
        "story_beat": "",
        "no_character_present": False,
        "source": "timestamped_lyrics",
    }


def segments_from_timestamped_payload(
    payload: Dict[str, Any],
    *,
    segment_mode: str = "",
    include_instrumental_gaps: bool = True,
    instrumental_text: str = "",
    min_gap_seconds: Optional[float] = None,
    max_scene_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Mirror ``createSegmentsFromTimestampedLyricsPayload``.

    ``payload`` is the JSON the timestamp workflow returns (``segments`` with ``start``, ``end``,
    ``text``, ``type`` and ``words``, plus ``duration``). Long chunks are split near word
    boundaries, gaps between vocal chunks become instrumental scenes, and the song's tail is filled.
    """
    source_segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    mode = _text(segment_mode or payload.get("segment_mode") or payload.get("segmentMode"))
    preserves_units = mode in REFERENCE_UNIT_MODES
    ordered_source = []
    for item in source_segments:
        start = max(0.0, _number(item.get("start") if isinstance(item, dict) else 0))
        end = max(0.0, _number(item.get("end") if isinstance(item, dict) else 0))
        if isinstance(item, dict) and end > start + 0.01:
            ordered_source.append({"item": item, "start": start, "end": end})
    ordered_source.sort(key=lambda entry: (entry["start"], entry["end"]))

    instrumental = _text(instrumental_text or payload.get("instrumental_text") or payload.get("instrumentalText")) or DEFAULT_INSTRUMENTAL_TEXT
    fill_gaps = (
        include_instrumental_gaps is not False
        and payload.get("include_instrumental_gaps") is not False
        and payload.get("includeInstrumentalGaps") is not False
    )
    min_gap = max(0.0, _number(min_gap_seconds if min_gap_seconds is not None else payload.get("min_gap_seconds", payload.get("minGapSeconds")), 0.25))
    max_scene = max(0.5, _number(max_scene_seconds if max_scene_seconds is not None else payload.get("max_scene_seconds", payload.get("maxSceneSeconds")), 8.0) or 8.0)
    soft_max = max_scene + 2

    ordered: List[Dict[str, Any]] = []
    for source in ordered_source:
        duration = source["end"] - source["start"]
        if preserves_units or duration <= soft_max + _EPSILON:
            ordered.append(source)
            continue
        words = []
        for word in source["item"].get("words") or []:
            start = _number(word.get("start"), float("nan")) if isinstance(word, dict) else float("nan")
            end = _number(word.get("end", word.get("start")), float("nan")) if isinstance(word, dict) else float("nan")
            if math.isfinite(start) and math.isfinite(end) and end > start:
                words.append({**word, "start": start, "end": end})
        words.sort(key=lambda w: (w["start"], w["end"]))
        part_start = source["start"]
        cursor = 0
        while source["end"] - part_start > soft_max + _EPSILON:
            target = part_start + max_scene
            limit = part_start + soft_max
            candidates = [
                (word, index) for index, word in enumerate(words)
                if index >= cursor and word["end"] > part_start + 0.1 and word["end"] <= limit + _EPSILON
            ]
            chosen = min(candidates, key=lambda c: abs(c[0]["end"] - target)) if candidates else None
            part_end = chosen[0]["end"] if chosen else target
            part_item = dict(source["item"])
            if chosen:
                part_words = words[cursor:chosen[1] + 1]
                part_item["words"] = part_words
                joined = " ".join(_text(w.get("text") or w.get("word")) for w in part_words if _text(w.get("text") or w.get("word")))
                part_item["text"] = joined or source["item"].get("text", "")
                cursor = chosen[1] + 1
            else:
                part_item["words"] = []
            part_item["timing_warning"] = f"Long transcription chunk split near a word boundary to honor the {max_scene:.2f} second maximum."
            ordered.append({"item": part_item, "start": part_start, "end": part_end})
            part_start = part_end
        final_item = dict(source["item"])
        final_words = [w for w in words[cursor:] if w["end"] > part_start - _EPSILON]
        if final_words:
            final_item["words"] = final_words
            joined = " ".join(_text(w.get("text") or w.get("word")) for w in final_words if _text(w.get("text") or w.get("word")))
            final_item["text"] = joined or source["item"].get("text", "")
        final_item["timing_warning"] = "Continuation of a long transcription chunk split near word boundaries."
        ordered.append({"item": final_item, "start": part_start, "end": source["end"]})

    created: List[Dict[str, Any]] = []

    def add_segment(start: float, end: float, item: Optional[Dict[str, Any]] = None, forced_text: str = "") -> None:
        clean_start = max(0.0, _number(start))
        clean_end = max(clean_start + 0.05, _number(end, clean_start + 4))
        segment = _new_scene(clean_start, clean_end, f"SCENE {len(created) + 1}")
        segment["timeline_note"] = _text(item.get("timing_warning")) if item else ""
        segment["lyric_text"] = clean_timestamped_lyric_text(forced_text or (item.get("text") if item else "") or "") or instrumental
        segment["lyric_no_lip_sync"] = (
            item is None or _text(item.get("type")).lower() == "instrumental" or is_instrumental_lyric_text(segment["lyric_text"])
        )
        created.append(segment)

    def add_gap(start: float, end: float) -> bool:
        if not fill_gaps:
            return False
        clean_start = max(0.0, _number(start))
        clean_end = max(clean_start, _number(end, clean_start))
        if clean_end - clean_start < max(min_gap, _EPSILON):
            return False
        gap_start = clean_start
        added = False
        while gap_start < clean_end - _EPSILON:
            gap_end = min(clean_end, gap_start + max_scene)
            add_segment(gap_start, gap_end, None, instrumental)
            added = True
            gap_start = gap_end
        return added

    cursor = 0.0
    for entry in ordered:
        segment_start = entry["start"]
        if entry["start"] > cursor + _EPSILON:
            if not add_gap(cursor, entry["start"]):
                if created:
                    created[-1]["end"] = entry["start"]
                else:
                    segment_start = cursor
        add_segment(segment_start, entry["end"], entry["item"])
        cursor = max(cursor, entry["end"])
    duration = _number(payload.get("duration"))
    if duration > cursor + _EPSILON:
        if not add_gap(cursor, duration) and created:
            created[-1]["end"] = duration
    created.sort(key=lambda s: (s["start"], s["end"]))
    for index, segment in enumerate(created, start=1):
        segment["label"] = f"SCENE {index}"
    return created


def normalize_scene_durations(
    segments: List[Dict[str, Any]],
    *,
    min_scene_seconds: float = 1.0,
    max_scene_seconds: float = 8.0,
    segment_mode: str = "",
    instrumental_text: str = DEFAULT_INSTRUMENTAL_TEXT,
) -> List[Dict[str, Any]]:
    """Mirror ``normalizeTimestampedSceneDurations``: merge scenes shorter than the minimum into a neighbor."""
    items = sorted((s for s in segments if s), key=lambda s: (_number(s.get("start")), _number(s.get("end"))))
    if segment_mode in REFERENCE_UNIT_MODES:
        return items
    min_s = max(1.0, _number(min_scene_seconds, 1.0))
    max_s = max(min_s, _number(max_scene_seconds, 8.0))
    soft_max = max_s + 2
    if not items:
        return items

    index = 0
    while index < len(items):
        segment = items[index]
        segment["start"] = max(0.0, _number(segment.get("start")))
        segment["end"] = max(segment["start"] + 0.05, _number(segment.get("end"), segment["start"] + 0.05))
        duration = segment["end"] - segment["start"]
        if duration >= min_s or len(items) <= 1:
            index += 1
            continue
        prev = items[index - 1] if index > 0 else None
        nxt = items[index + 1] if index + 1 < len(items) else None
        prev_duration = max(0.0, _number(prev.get("end")) - _number(prev.get("start"))) if prev else -1
        next_duration = max(0.0, _number(nxt.get("end")) - _number(nxt.get("start"))) if nxt else -1
        segment_instrumental = is_instrumental_lyric_text(segment.get("lyric_text"))
        prev_vocal = bool(prev) and not is_instrumental_lyric_text(prev.get("lyric_text"))
        next_vocal = bool(nxt) and not is_instrumental_lyric_text(nxt.get("lyric_text"))
        nearest = nxt if nxt and (not prev or next_duration <= prev_duration) else prev
        target = nearest if segment_instrumental else (prev if prev_vocal else (nxt if next_vocal else nearest))
        if not target and not segment_instrumental:
            next_start = _number(nxt.get("start")) if nxt else None
            desired_end = segment["start"] + min_s
            segment["end"] = min(desired_end, next_start) if next_start and next_start > segment["start"] else desired_end
            index += 1
            continue
        if not target:
            index += 1
            continue
        merged_start = min(_number(target.get("start")), segment["start"])
        merged_end = max(_number(target.get("end"), _number(target.get("start")) + min_s), segment["end"])
        if merged_end - merged_start > soft_max + 0.001:
            index += 1
            continue
        target_first = _number(target.get("start")) <= segment["start"]
        target["start"] = merged_start
        target["end"] = merged_end
        target["lyric_text"] = (
            merge_lyric_text(target.get("lyric_text"), segment.get("lyric_text"), instrumental_text)
            if target_first else merge_lyric_text(segment.get("lyric_text"), target.get("lyric_text"), instrumental_text)
        )
        target["lyric_no_lip_sync"] = is_instrumental_lyric_text(target["lyric_text"])
        target["timeline_note"] = "\n".join(p for p in (_text(target.get("timeline_note")), _text(segment.get("timeline_note"))) if p)
        items.pop(index)
        index = max(0, index - 2)

    for index, segment in enumerate(items):
        if index > 0:
            segment["start"] = max(_number(items[index - 1].get("end")), _number(segment.get("start")))
            if segment["end"] <= segment["start"] + 0.05:
                segment["end"] = segment["start"] + min_s
        segment["end"] = max(_number(segment.get("start")) + 0.05, _number(segment.get("end")))
    return items


# ---------------------------------------------------------------------------
# Scene-length rule: merge short scenes, split long ones
# ---------------------------------------------------------------------------

def _duration(segment: Dict[str, Any]) -> float:
    return _number(segment.get("end")) - _number(segment.get("start"))


def next_length_fix(
    segments: List[Dict[str, Any]],
    min_scene_seconds: float,
    max_scene_seconds: float,
    locked_ids: Optional[set] = None,
    tolerance: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """Return the next edit that moves the timeline toward ``min <= length <= max``, or ``None``.

    Long scenes are split first, into equal parts (a 12 s scene becomes 6 s + 6 s). Then each short
    scene is merged into the neighbor that gives the shorter result, preferring a vocal neighbor for
    a vocal scene. A merge that overshoots the maximum is split again by the next call, so two
    awkward neighbors end up balanced. Scenes in ``locked_ids`` (rendered video) are never touched.

    The result is ``{"op": "split", "scene_id", "at_time"}`` or ``{"op": "merge", "scene_id", "with"}``.
    """
    min_s = _number(min_scene_seconds)
    max_s = _number(max_scene_seconds)
    if min_s <= 0 or max_s < min_s:
        raise ValueError("min_scene_seconds must be positive and not larger than max_scene_seconds.")
    locked = locked_ids or set()
    items = sorted(segments, key=lambda s: (_number(s.get("start")), _number(s.get("end"))))

    for segment in items:
        if segment.get("id") in locked:
            continue
        duration = _duration(segment)
        if duration > max_s + tolerance:
            parts = math.ceil(duration / max_s - 1e-9)
            return {
                "op": "split",
                "scene_id": segment.get("id"),
                "at_time": round(_number(segment.get("start")) + duration / parts, 4),
            }

    if len(items) < 2:
        return None
    for index, segment in enumerate(items):
        if segment.get("id") in locked or _duration(segment) >= min_s - tolerance:
            continue
        options = []
        for neighbor_index, direction in ((index - 1, "previous"), (index + 1, "next")):
            if 0 <= neighbor_index < len(items) and items[neighbor_index].get("id") not in locked:
                neighbor = items[neighbor_index]
                merged = max(_number(neighbor.get("end")), _number(segment.get("end"))) - min(_number(neighbor.get("start")), _number(segment.get("start")))
                same_kind = is_instrumental_lyric_text(neighbor.get("lyric_text")) == is_instrumental_lyric_text(segment.get("lyric_text"))
                options.append((merged > max_s + tolerance, 0 if same_kind else 1, merged, direction))
        if not options:
            continue
        options.sort()
        direction = options[0][3]
        return {"op": "merge", "scene_id": segment.get("id"), "with": direction}
    return None


def apply_length_fix(segments: List[Dict[str, Any]], fix: Dict[str, Any], instrumental_text: str = DEFAULT_INSTRUMENTAL_TEXT) -> None:
    """Apply one ``next_length_fix`` edit to an in-memory scene list (no files involved)."""
    segments.sort(key=lambda s: (_number(s.get("start")), _number(s.get("end"))))
    index = next(i for i, s in enumerate(segments) if s.get("id") == fix["scene_id"])
    scene = segments[index]
    if fix["op"] == "split":
        start, end, at = _number(scene.get("start")), _number(scene.get("end")), _number(fix["at_time"])
        left_text, right_text = split_lyric_text(scene.get("lyric_text"), (at - start) / max(end - start, 0.01), instrumental_text)
        right = dict(scene)
        right.update({"id": new_segment_id(), "start": round(at, 4), "end": round(end, 4), "lyric_text": right_text, "source": "split"})
        right["lyric_no_lip_sync"] = is_instrumental_lyric_text(right_text)
        scene.update({"end": round(at, 4), "lyric_text": left_text})
        scene["lyric_no_lip_sync"] = is_instrumental_lyric_text(left_text)
        segments.insert(index + 1, right)
        return
    other_index = index - 1 if fix["with"] == "previous" else index + 1
    first, second = (segments[other_index], scene) if other_index < index else (scene, segments[other_index])
    first["start"] = min(_number(first.get("start")), _number(second.get("start")))
    first["end"] = max(_number(first.get("end")), _number(second.get("end")))
    first["lyric_text"] = merge_lyric_text(first.get("lyric_text"), second.get("lyric_text"), instrumental_text)
    first["lyric_no_lip_sync"] = is_instrumental_lyric_text(first["lyric_text"])
    first["timeline_note"] = "\n".join(p for p in (_text(first.get("timeline_note")), _text(second.get("timeline_note"))) if p)
    first["notes"] = "\n\n".join(p for p in (_text(first.get("notes")), _text(second.get("notes"))) if p)
    segments.remove(second)


def enforce_scene_lengths(
    segments: List[Dict[str, Any]],
    min_scene_seconds: float,
    max_scene_seconds: float,
    instrumental_text: str = DEFAULT_INSTRUMENTAL_TEXT,
    locked_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Apply ``next_length_fix`` until every scene fits, on an in-memory list. Returns the list."""
    limit = max(50, len(segments) * 6)
    for _ in range(limit):
        fix = next_length_fix(segments, min_scene_seconds, max_scene_seconds, locked_ids)
        if fix is None:
            break
        apply_length_fix(segments, fix, instrumental_text)
    for index, segment in enumerate(sorted(segments, key=lambda s: _number(s.get("start"))), start=1):
        if re.match(r"^SCENE \d+$", _text(segment.get("label")), re.IGNORECASE):
            segment["label"] = f"SCENE {index}"
    segments.sort(key=lambda s: (_number(s.get("start")), _number(s.get("end"))))
    return segments
