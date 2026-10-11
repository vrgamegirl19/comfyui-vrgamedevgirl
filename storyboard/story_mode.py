"""Story Mode data: the story book, chapter settings, acts and scene plan kept in ``builder_story_layer``.

Pure functions only (no LLM, no disk). The LLM steps live in ``storyboard/story_book.py``.

Shape inside ``builder_story_layer`` (only present when the project has used Story Mode):

- ``story_flow_version``: 2 for projects on the Story Mode flow.
- ``story_book``: ``{title, premise, chapters: [{id, title, setting_id, paragraphs: [str]}]}``.
- ``story_book_text``: the editable plain-text form of the book (see ``story_book_to_text``).
- ``story_settings``: ``[{id, name, look}]``, the general setting each chapter happens in.
- ``story_acts``: ``[{id, chapter_id, title, summary, cast_ids, scene_ids}]``.
- ``story_scene_plan``: ``[{scene_id, act_id, story_slice, spot, mood, performance_mode, note_sources}]``.
- ``story_spot_min_scenes``: the fewest scenes in a row at one spot (the Builder's "scenes per location").

JS twin: ``web/music_video_builder/story_mode_data.mjs``. Both are checked against
``tests/story_mode_cases.json``.
"""

import re

STORY_FLOW_VERSION = 2
STORY_MODE_KEYS = (
    "story_flow_version", "story_book", "story_book_text", "story_settings", "story_acts", "story_scene_plan",
    "story_spot_min_scenes",
)
DEFAULT_SPOT_MIN_SCENES = 3
PLAN_PERFORMANCE_MODES = ("singing", "speaking", "no_lip_sync", "instrumental")

_MAX_CHAPTERS = 40
_MAX_PARAGRAPHS = 200
_CHAPTER_HEADING = re.compile(
    r"^\s*(?:#+\s*)?chapter\s+([0-9]+|[a-z]+(?:[\s-][a-z]+)?)\s*(?:[:.\-–—]\s*(.*?))?\s*$",
    re.IGNORECASE,
)
_META_LINE = re.compile(r"^\s*(title|premise|setting)\s*:\s*(.*?)\s*$", re.IGNORECASE)


def _text(value, limit=4000):
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text[:limit]


def _line(value, limit=200):
    return re.sub(r"\s+", " ", _text(value, limit * 2)).strip()[:limit]


def _id_list(value, limit=400):
    items = value if isinstance(value, list) else []
    return [_line(item, 160) for item in items if _line(item, 160)][:limit]


def slug(value):
    """Lower-case id fragment from a name (``Home - Kitchen`` -> ``home_kitchen``)."""
    return re.sub(r"[^a-z0-9]+", "_", _line(value).lower()).strip("_")[:60] or "item"


def setting_id(name):
    return f"setting_{slug(name)}"


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

def normalize_story_book(value):
    source = value if isinstance(value, dict) else {}
    chapters = []
    for index, chapter in enumerate(source.get("chapters") or [], start=1):
        if not isinstance(chapter, dict) or len(chapters) >= _MAX_CHAPTERS:
            continue
        paragraphs = [_text(p, 6000) for p in chapter.get("paragraphs") or [] if _text(p, 6000)][:_MAX_PARAGRAPHS]
        chapters.append({
            "id": _line(chapter.get("id"), 80) or f"ch_{index}",
            "title": _line(chapter.get("title"), 200),
            "setting_id": _line(chapter.get("setting_id"), 120),
            "paragraphs": paragraphs,
        })
    return {
        "title": _line(source.get("title"), 200),
        "premise": _text(source.get("premise"), 2000),
        "chapters": chapters,
    }


def normalize_story_settings(value):
    settings, seen = [], set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        name = _line(item.get("name"), 160)
        item_id = _line(item.get("id"), 120) or (setting_id(name) if name else "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        settings.append({"id": item_id, "name": name or item_id, "look": _text(item.get("look"), 2000)})
    return settings


def normalize_story_acts(value):
    acts = []
    for index, item in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        acts.append({
            "id": _line(item.get("id"), 80) or f"act_{index}",
            "chapter_id": _line(item.get("chapter_id"), 80),
            "title": _line(item.get("title"), 200),
            "summary": _text(item.get("summary"), 2000),
            "cast_ids": _id_list(item.get("cast_ids"), 40),
            "scene_ids": _id_list(item.get("scene_ids")),
        })
    return acts


def normalize_plan_performance(value):
    text = re.sub(r"[\s-]+", "_", _line(value).lower())
    if text in ("speaking", "short_film", "dialogue", "dialog", "spoken"):
        return "speaking"
    if text in ("no_lip_sync", "nolipsync", "no_lipsync", "story", "story_only", "visual_only", "silent"):
        return "no_lip_sync"
    if text in ("instrumental", "music_only", "no_vocals"):
        return "instrumental"
    return "singing"


def normalize_story_scene_plan(value):
    plan, seen = [], set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        scene_id = _line(item.get("scene_id"), 160)
        if not scene_id or scene_id in seen:
            continue
        seen.add(scene_id)
        plan.append({
            "scene_id": scene_id,
            "act_id": _line(item.get("act_id"), 80),
            "story_slice": _text(item.get("story_slice"), 1200),
            "spot": _line(item.get("spot"), 160),
            "mood": _line(item.get("mood"), 200),
            "performance_mode": normalize_plan_performance(item.get("performance_mode")),
            "note_sources": _id_list(item.get("note_sources"), 12),
        })
    return plan


def normalize_story_mode_fields(source):
    """The Story Mode keys of a story layer, normalized. Keys the source does not have are left out."""
    source = source if isinstance(source, dict) else {}
    result = {}
    if "story_flow_version" in source:
        try:
            version = int(float(source.get("story_flow_version") or 0))
        except (TypeError, ValueError):
            version = 0
        result["story_flow_version"] = STORY_FLOW_VERSION if version >= STORY_FLOW_VERSION else 0
    if "story_book" in source:
        result["story_book"] = normalize_story_book(source.get("story_book"))
    if "story_book_text" in source:
        result["story_book_text"] = _text(source.get("story_book_text"), 120000)
    if "story_settings" in source:
        result["story_settings"] = normalize_story_settings(source.get("story_settings"))
    if "story_acts" in source:
        result["story_acts"] = normalize_story_acts(source.get("story_acts"))
    if "story_scene_plan" in source:
        result["story_scene_plan"] = normalize_story_scene_plan(source.get("story_scene_plan"))
    if "story_spot_min_scenes" in source:
        result["story_spot_min_scenes"] = normalize_spot_min_scenes(source.get("story_spot_min_scenes"))
    return result


def normalize_spot_min_scenes(value):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return DEFAULT_SPOT_MIN_SCENES
    return max(1, min(20, number))


def plan_segment_fields(performance_mode):
    """Timeline segment fields for a scene plan performance. Story-only and instrumental scenes are no lip sync."""
    mode = normalize_plan_performance(performance_mode)
    if mode in ("no_lip_sync", "instrumental"):
        return {"performance_mode": "no_lip_sync", "lyric_no_lip_sync": True}
    return {"performance_mode": mode, "lyric_no_lip_sync": False}


def is_story_mode_layer(layer):
    return int((layer or {}).get("story_flow_version") or 0) >= STORY_FLOW_VERSION


# ---------------------------------------------------------------------------
# Book text
# ---------------------------------------------------------------------------

def split_paragraphs(text):
    """Paragraphs of a story. Blank lines separate paragraphs; text without blank lines is one paragraph per line."""
    body = _text(text, 200000)
    if not body:
        return []
    if re.search(r"\n\s*\n", body):
        blocks = re.split(r"\n\s*\n", body)
    else:
        blocks = body.split("\n")
    return [block.strip() for block in blocks if block.strip()]


def _chapter_title(raw_number, raw_title):
    title = _line(raw_title, 200)
    return title or f"Chapter {raw_number}"


def parse_story_book_text(text):
    """Parse the book text into ``(book, setting_names)``.

    ``book`` follows ``normalize_story_book`` with ``setting_id`` from each chapter's ``Setting:`` line and
    ``setting_names`` maps those ids to the names written. Text with no chapter headings is one chapter.
    """
    lines = _text(text, 200000).split("\n")
    title, premise = "", ""
    chapters = []
    setting_names = {}
    current = None
    buffer = []

    def flush():
        if current is not None:
            current["paragraphs"] = split_paragraphs("\n".join(buffer))
            chapters.append(current)

    preamble = []
    for raw in lines:
        heading = _CHAPTER_HEADING.match(raw)
        if heading:
            flush()
            buffer = []
            number = heading.group(1)
            current = {
                "id": f"ch_{len(chapters) + 1}",
                "title": _chapter_title(number, heading.group(2)),
                "setting_id": "",
                "paragraphs": [],
            }
            continue
        meta = _META_LINE.match(raw)
        if meta:
            key, value = meta.group(1).lower(), meta.group(2)
            if key == "setting" and current is not None and not current["setting_id"] and not "".join(buffer).strip():
                if _line(value):
                    current["setting_id"] = setting_id(value)
                    setting_names.setdefault(current["setting_id"], _line(value, 160))
                continue
            if current is None and key in ("title", "premise"):
                if key == "title" and not title:
                    title = _line(value, 200)
                    continue
                if key == "premise" and not premise:
                    premise = _text(value, 2000)
                    continue
        if current is None:
            preamble.append(raw)
        else:
            buffer.append(raw)
    flush()
    if not chapters:
        paragraphs = split_paragraphs("\n".join(preamble))
        if paragraphs:
            chapters.append({"id": "ch_1", "title": "Chapter 1", "setting_id": "", "paragraphs": paragraphs})
    book = normalize_story_book({"title": title, "premise": premise, "chapters": chapters})
    return book, setting_names


def story_book_to_text(book, settings=None):
    """The editable plain-text form of a book. ``parse_story_book_text`` reads it back unchanged."""
    book = normalize_story_book(book)
    names = {item["id"]: item["name"] for item in normalize_story_settings(settings or [])}
    parts = []
    if book["title"]:
        parts.append(f"Title: {book['title']}")
    if book["premise"]:
        parts.append(f"Premise: {book['premise']}")
    head = "\n".join(parts)
    blocks = [head] if head else []
    for number, chapter in enumerate(book["chapters"], start=1):
        heading = f"Chapter {number}: {chapter['title'] or f'Chapter {number}'}"
        setting_name = names.get(chapter["setting_id"], "")
        if setting_name:
            heading += f"\nSetting: {setting_name}"
        blocks.append("\n\n".join([heading, *chapter["paragraphs"]]))
    return "\n\n".join(blocks)


def sync_story_settings(book, setting_names, existing=None):
    """Settings for every chapter: keeps existing looks, adds settings named in the text, drops unused ones."""
    current = {item["id"]: item for item in normalize_story_settings(existing or [])}
    used = []
    for chapter in normalize_story_book(book)["chapters"]:
        sid = chapter["setting_id"]
        if sid and sid not in used:
            used.append(sid)
    settings = []
    for sid in used:
        item = dict(current.get(sid) or {"id": sid, "name": "", "look": ""})
        item["name"] = (setting_names or {}).get(sid) or item.get("name") or sid
        settings.append(item)
    return settings


def apply_story_book_text(layer, text):
    """Story layer fields after the user edits the book text: book, text and settings updated together."""
    book, names = parse_story_book_text(text)
    settings = sync_story_settings(book, names, (layer or {}).get("story_settings"))
    return {
        "story_flow_version": STORY_FLOW_VERSION,
        "story_book": book,
        "story_book_text": story_book_to_text(book, settings),
        "story_settings": settings,
    }


# ---------------------------------------------------------------------------
# Pasted stories
# ---------------------------------------------------------------------------

def chapters_from_split(paragraphs, split):
    """Build chapters from a paragraph split, keeping every paragraph exactly as written.

    ``split`` is a list of ``{start_paragraph, title, setting}`` with 1-based paragraph numbers. Starts are
    sorted, the first chapter always starts at paragraph 1, and out-of-range or repeated starts are dropped.
    Returns ``(chapters, setting_names)``.
    """
    count = len(paragraphs)
    if not count:
        return [], {}
    entries = []
    for item in split if isinstance(split, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start_paragraph"))
        except (TypeError, ValueError):
            continue
        if 1 <= start <= count:
            entries.append((start, _line(item.get("title"), 200), _line(item.get("setting"), 160)))
    entries.sort(key=lambda entry: entry[0])
    starts, kept = set(), []
    for entry in entries:
        if entry[0] not in starts:
            starts.add(entry[0])
            kept.append(entry)
    if not kept or kept[0][0] != 1:
        kept.insert(0, (1, "", ""))
    chapters, names = [], {}
    for index, (start, title, setting) in enumerate(kept):
        end = kept[index + 1][0] - 1 if index + 1 < len(kept) else count
        sid = setting_id(setting) if setting else ""
        if sid:
            names.setdefault(sid, setting)
        chapters.append({
            "id": f"ch_{index + 1}",
            "title": title or f"Chapter {index + 1}",
            "setting_id": sid,
            "paragraphs": list(paragraphs[start - 1:end]),
        })
    return chapters, names


def chapters_keep_text(paragraphs, chapters):
    """True when the chapters hold exactly the pasted paragraphs, in order."""
    joined = [p for chapter in chapters for p in chapter.get("paragraphs") or []]
    return joined == list(paragraphs)


def story_beat_context(layer, scene_id, scene_number):
    """Story Mode text for one scene's beat, or ``None`` when the scene has no plan entry.

    Returns ``(arc_text, brief_text)``. The arc text holds the act and the previous, current and next scene slices as
    ``Scene N (Spot) - slice`` lines (the beat prompt uses this scene's line as its main story source); the brief text
    holds the book premise and the setting's look.
    """
    layer = layer or {}
    plan = normalize_story_scene_plan(layer.get("story_scene_plan"))
    index = next((i for i, entry in enumerate(plan) if entry["scene_id"] == str(scene_id)), -1)
    if index < 0:
        return None
    entry = plan[index]
    acts = {act["id"]: act for act in normalize_story_acts(layer.get("story_acts"))}
    act = acts.get(entry["act_id"]) or {}
    book = normalize_story_book(layer.get("story_book"))
    chapter = next((c for c in book["chapters"] if c["id"] == act.get("chapter_id")), {})
    settings = {item["id"]: item for item in normalize_story_settings(layer.get("story_settings"))}
    setting = settings.get(chapter.get("setting_id", "")) or {}
    try:
        number = int(scene_number)
    except (TypeError, ValueError):
        number = index + 1
    lines = []
    if act:
        lines.append(f"Act: {act['title']} - {act['summary']}".strip(" -"))
    for offset in (-1, 0, 1):
        position = index + offset
        if 0 <= position < len(plan):
            other = plan[position]
            spot = other["spot"] or setting.get("name", "")
            lines.append(f"Scene {number + offset} ({spot}) - {other['story_slice']}")
    brief = []
    if book["premise"]:
        brief.append(f"Story premise: {book['premise']}")
    if chapter:
        brief.append(f"Chapter: {chapter.get('title', '')}" + (f", set in {setting['name']}" if setting.get("name") else ""))
    if setting.get("look"):
        brief.append(f"Setting look: {setting['look']}")
    if entry["mood"]:
        brief.append(f"Scene mood: {entry['mood']}")
    if layer.get("song_story_brief"):
        brief.append(str(layer["song_story_brief"]))
    return "\n".join(lines), "\n".join(brief)


def fill_scene_cast(subject_scene_map, acts, subject_ids):
    """Characters for scenes that have none: each act's cast goes onto its scenes. User mappings are kept.

    Returns ``(new_map, changed_scene_ids)``. Only ids in ``subject_ids`` are used.
    """
    result = {key: list(value) if isinstance(value, list) else value for key, value in (subject_scene_map or {}).items()}
    valid = set(subject_ids)
    changed = []
    for act in normalize_story_acts(acts):
        cast = [cast_id for cast_id in act["cast_ids"] if cast_id in valid]
        if not cast:
            continue
        for scene_id in act["scene_ids"]:
            current = result.get(scene_id)
            if (isinstance(current, list) and current) or (isinstance(current, str) and current.strip()):
                continue
            result[scene_id] = list(cast)
            changed.append(scene_id)
    return result, changed


# A heading is a bracketed tag ("[Chorus 2]") or a section word followed by a colon ("Verse 1:", "Act 2:").
_ARC_HEADING = re.compile(
    r"^\s*(?:\[[^\]\n]{1,40}\]|(?:intro|outro|verse|chorus|pre[\s-]?chorus|post[\s-]?chorus|bridge|hook|refrain|"
    r"interlude|break|instrumental|breakdown|drop|solo|tag|coda|part|section|act)\b[^:\n]{0,30}:)\s*",
    re.IGNORECASE,
)
_ARC_SCENE_PREFIX = re.compile(r"^\s*scene\s+\d+\s*(?:\([^)]*\))?\s*[\u2014\u2013:-]\s*", re.IGNORECASE)


def story_arc_to_story_text(arc_text):
    """Plain paragraphs from an old song-section story arc, for converting a project to Story Mode.

    Section headings (``Verse 1:``, ``[Chorus 2]``) and ``Scene N (Location) -`` prefixes are removed; each section
    becomes one paragraph. The words themselves are kept.
    """
    paragraphs, current = [], []
    for raw in _text(arc_text, 200000).split("\n"):
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        heading = _ARC_HEADING.match(line)
        if heading:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            line = line[heading.end():].strip()
            if not line:
                continue
        line = _ARC_SCENE_PREFIX.sub("", line).strip()
        if line:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)
