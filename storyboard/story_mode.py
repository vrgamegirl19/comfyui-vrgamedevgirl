"""Story Mode data: the story book, chapter settings, acts and scene plan kept in ``builder_story_layer``.

Pure functions only (no LLM, no disk). The LLM steps live in ``storyboard/story_book.py``.

Shape inside ``builder_story_layer`` (only present when the project has used Story Mode):

- ``story_flow_version``: 2 for projects on the Story Mode flow.
- ``story_book``: ``{title, premise, chapters: [{id, title, setting_id, paragraphs: [str]}]}``.
- ``story_book_text``: the editable plain-text form of the book (see ``story_book_to_text``).
- ``story_settings``: ``[{id, name, look}]``, the general setting each chapter happens in.
- ``story_acts``: ``[{id, chapter_id, title, summary, cast_ids, scene_ids}]``.
- ``story_scene_plan``: ``[{scene_id, act_id, story_slice, spot, mood, performance_mode, note_sources,
  continues_previous, continuation_source, llm_continues, continuation_direction, cast_ids, performer_id, interaction,
  cast_source, user_note}]``. ``cast_ids`` are the characters in the scene (``cast_source`` "manual" when the user
  set them), ``performer_id`` sings or speaks, ``interaction`` is what they do with each other, and ``user_note`` is
  the user's own direction for the scene. ``continues_previous`` makes
  the scene a MiniMax masked continuation of the scene before. ``continuation_source`` says who decided it
  (``auto``, ``llm`` or ``manual``), ``llm_continues`` is the LLM's own answer, and ``continuation_direction`` is
  what happens once the carried-on moment settles.
- ``story_spot_min_scenes``: the fewest scenes in a row at one spot (the Builder's "scenes per location").
- ``story_cast_ids``: subject card ids the story is about. Empty means every character card.
- ``story_allow_new_cast``: kept for old projects only. The story always uses only the cast.
- Levels: the Story Builder shows Story > Acts > Chapters > Scenes. ``story_book.chapters`` are the Acts (with a
  ``purpose`` and a ``presence`` grid) and ``story_acts`` are the Chapters inside them (each with its own
  ``setting_id`` and ``presence``; an empty setting uses its act's). The keys predate the names.
- ``story_act_count``: how many acts the user wants (0 lets the story decide). ``story_chapter_counts``: ``{act id:
  chapters wanted}``.
- ``story_main_cast_id``: the main character, who leads the story and is in most or every scene.
- ``story_cast_roles``: ``{cast id: role}``, role one of ``STORY_CAST_ROLES``.
- ``story_relationships``: ``[{a, b, type, note}]``, how two cast members are connected (type one of
  ``STORY_RELATIONSHIP_TYPES``).
- ``story_user_locations``: the user's own places for the video, free text (one place per line, with its look after a
  colon). The story and the setting looks use these places first.
- ``story_tone_notes``: the tone, genre and world the user wants the story written in.
- ``story_character_notes``: ``{cast id: text}``, the personality and emotions of each character, in the user's words.
- ``story_video_style``: how the story is told on screen: ``music_video`` (the cast performs the song),
  ``spoken`` (the characters talk) or ``story_only`` (no one sings or speaks on camera).
- ``story_continuation_mode``: which scenes continue the scene before: ``off``, ``location`` (touching scenes at the
  same spot in the same chapter) or ``llm`` (the scene plan's answer). A manual choice always wins.
- ``story_stamps``: ``{book: {part: print}, acts|plan|places|beats|prompts: {id: {part: print}}}``, prints of what
  each Story Builder step's output was made from. The Story Builder compares them to mark later steps out of date.

JS twin: ``web/music_video_builder/story_mode_data.mjs``. Both are checked against
``tests/story_mode_cases.json``.
"""

import re

STORY_FLOW_VERSION = 2
STORY_MODE_KEYS = (
    "story_flow_version", "story_book", "story_book_text", "story_settings", "story_acts", "story_scene_plan",
    "story_spot_min_scenes", "story_cast_ids", "story_allow_new_cast", "story_video_style", "story_continuation_mode",
    "story_stamps", "story_main_cast_id", "story_cast_roles", "story_relationships", "story_act_count",
    "story_chapter_counts", "story_user_locations", "story_character_notes", "story_tone_notes",
)
STORY_CAST_ROLES = ("main", "love_interest", "supporting", "antagonist")
STORY_RELATIONSHIP_TYPES = ("love_interest", "partners", "exes", "family", "friends", "rivals", "strangers", "other")
STORY_VIDEO_STYLES = ("music_video", "spoken", "story_only")
STORY_CONTINUATION_MODES = ("off", "location", "llm")
CONTINUATION_SOURCES = ("auto", "llm", "manual")
STORY_STAMP_SECTIONS = ("acts", "plan", "places", "beats", "prompts")
DEFAULT_SPOT_MIN_SCENES = 3
PLAN_PERFORMANCE_MODES = ("singing", "speaking", "no_lip_sync", "instrumental")

_MAX_CHAPTERS = 40
_MAX_PARAGRAPHS = 200
# A book section heading: "Act 1: title" (written since story flow 3) or "Chapter 1: title" (older books).
_CHAPTER_HEADING = re.compile(
    r"^\s*(?:#+\s*)?(?:chapter|act)\s+([0-9]+|[a-z]+(?:[\s-][a-z]+)?)\s*(?:[:.\-–—]\s*(.*?))?\s*$",
    re.IGNORECASE,
)
# Levels: the story is built Story > Acts > Chapters > Scenes. In the data, story_book chapters are the Acts and
# story_acts are the Chapters (the keys predate the names).
STORY_ACT_PURPOSES = ("setup", "turn", "climax", "resolution")
STORY_PRESENCE_LEVELS = ("absent", "present", "featured")


def normalize_story_presence(value):
    """``{cast id: absent | present | featured}``: how much a character is in an act or chapter."""
    presence = {}
    for cast_id, level in list((value if isinstance(value, dict) else {}).items())[:40]:
        key, name = _line(cast_id, 160), _line(level, 20).lower()
        if key and name in STORY_PRESENCE_LEVELS:
            presence[key] = name
    return presence


def normalize_story_count(value, minimum, maximum):
    """A whole count between minimum and maximum; anything else is minimum (0 means "decide for me")."""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, number))


def chapter_setting_id(act, chapter):
    """The setting a chapter (story_acts entry) happens in: its own, or its act's (story_book chapter)."""
    return _line((act or {}).get("setting_id"), 120) or _line((chapter or {}).get("setting_id"), 120)
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
        purpose = _line(chapter.get("purpose"), 20).lower()
        chapters.append({
            "id": _line(chapter.get("id"), 80) or f"ch_{index}",
            "title": _line(chapter.get("title"), 200),
            "setting_id": _line(chapter.get("setting_id"), 120),
            "paragraphs": paragraphs,
            # Shown as an Act: its part in the story and who is in it.
            "purpose": purpose if purpose in STORY_ACT_PURPOSES else "",
            "presence": normalize_story_presence(chapter.get("presence")),
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
        act = {
            "id": _line(item.get("id"), 80) or f"act_{index}",
            "chapter_id": _line(item.get("chapter_id"), 80),
            "title": _line(item.get("title"), 200),
            "summary": _text(item.get("summary"), 2000),
            "cast_ids": _id_list(item.get("cast_ids"), 40),
            "scene_ids": _id_list(item.get("scene_ids")),
            # Shown as a Chapter: the one setting it happens in (empty uses its act's) and who is in it.
            "setting_id": _line(item.get("setting_id"), 120),
            "presence": normalize_story_presence(item.get("presence")),
        }
        # The user's own words from Replan (Scenes tab): where it happens and what must happen. Kept only when set.
        for key in ("location_note", "direction"):
            if _text(item.get(key), 2000):
                act[key] = _text(item.get(key), 2000)
        acts.append(act)
    return acts


def act_user_text(act):
    """The user's location and direction for an act, as lines the plan and beat prompts must follow."""
    lines = []
    if (act or {}).get("location_note"):
        lines.append(f"Where this happens (user, follow exactly): {act['location_note']}")
    if (act or {}).get("direction"):
        lines.append(f"What must happen (user, follow exactly): {act['direction']}")
    return "\n".join(lines)


def normalize_plan_performance(value):
    text = re.sub(r"[\s-]+", "_", _line(value).lower())
    if text in ("speaking", "short_film", "dialogue", "dialog", "spoken"):
        return "speaking"
    if text in ("no_lip_sync", "nolipsync", "no_lipsync", "story", "story_only", "visual_only", "silent"):
        return "no_lip_sync"
    if text in ("instrumental", "music_only", "no_vocals"):
        return "instrumental"
    return "singing"


def _continuation_source(value):
    text = _line(value, 20).lower()
    return text if text in CONTINUATION_SOURCES else ""


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
            "continues_previous": item.get("continues_previous") is True,
            "continuation_source": _continuation_source(item.get("continuation_source")),
            "llm_continues": item.get("llm_continues") is True,
            "continuation_direction": _text(item.get("continuation_direction"), 600),
            # Who is in the scene, who sings or speaks, what they do with each other, and who set the cast
            # ("story" or "manual"). user_note is the user's own direction for the scene.
            "cast_ids": _id_list(item.get("cast_ids"), 12),
            "performer_id": _line(item.get("performer_id"), 160),
            "interaction": _text(item.get("interaction"), 600),
            "cast_source": _line(item.get("cast_source"), 20).lower()
            if _line(item.get("cast_source"), 20).lower() in ("story", "manual") else "",
            "user_note": _text(item.get("user_note"), 1200),
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
    if "story_cast_ids" in source:
        result["story_cast_ids"] = _id_list(source.get("story_cast_ids"), 40)
    if "story_allow_new_cast" in source:
        value = source.get("story_allow_new_cast")
        result["story_allow_new_cast"] = value is True or str(value).strip().lower() == "true"
    if "story_video_style" in source:
        result["story_video_style"] = normalize_story_video_style(source.get("story_video_style"))
    if "story_continuation_mode" in source:
        result["story_continuation_mode"] = normalize_story_continuation_mode(source.get("story_continuation_mode"))
    if "story_stamps" in source:
        result["story_stamps"] = normalize_story_stamps(source.get("story_stamps"))
    if "story_main_cast_id" in source:
        result["story_main_cast_id"] = _line(source.get("story_main_cast_id"), 160)
    if "story_cast_roles" in source:
        result["story_cast_roles"] = normalize_story_cast_roles(source.get("story_cast_roles"))
    if "story_relationships" in source:
        result["story_relationships"] = normalize_story_relationships(source.get("story_relationships"))
    if "story_user_locations" in source:
        result["story_user_locations"] = _text(source.get("story_user_locations"), 3000)
    if "story_tone_notes" in source:
        result["story_tone_notes"] = _text(source.get("story_tone_notes"), 2000)
    if "story_character_notes" in source:
        result["story_character_notes"] = normalize_story_character_notes(source.get("story_character_notes"))
    if "story_act_count" in source:
        result["story_act_count"] = normalize_story_count(source.get("story_act_count"), 0, 12)
    if "story_chapter_counts" in source:
        # {act (story_book chapter) id: how many chapters the user wants in it}.
        counts = source.get("story_chapter_counts") if isinstance(source.get("story_chapter_counts"), dict) else {}
        result["story_chapter_counts"] = {
            _line(key, 80): normalize_story_count(value, 0, 20)
            for key, value in list(counts.items())[:40]
            if _line(key, 80) and normalize_story_count(value, 0, 20) > 0
        }
    return result


def normalize_story_cast_roles(value):
    """``{cast id: role}`` for the story's characters. Unknown roles are dropped."""
    roles = {}
    for cast_id, role in list((value if isinstance(value, dict) else {}).items())[:40]:
        key, name = _line(cast_id, 160), _line(role, 40).lower()
        if key and name in STORY_CAST_ROLES:
            roles[key] = name
    return roles


def normalize_story_character_notes(value):
    """``{cast id: text}``: each character's personality and emotions. Empty notes are dropped."""
    notes = {}
    for cast_id, note in list((value if isinstance(value, dict) else {}).items())[:40]:
        key, text = _line(cast_id, 160), _line(note, 600)
        if key and text:
            notes[key] = text
    return notes


def normalize_story_relationships(value):
    """``[{a, b, type, note}]``: how two characters are connected. A pair with one person or an unknown type is dropped."""
    seen = set()
    result = []
    for item in (value if isinstance(value, list) else [])[:40]:
        item = item if isinstance(item, dict) else {}
        a, b = _line(item.get("a"), 160), _line(item.get("b"), 160)
        kind = _line(item.get("type"), 40).lower()
        pair = "|".join(sorted([a, b]))
        if not a or not b or a == b or kind not in STORY_RELATIONSHIP_TYPES or pair in seen:
            continue
        seen.add(pair)
        result.append({"a": a, "b": b, "type": kind, "note": _line(item.get("note"), 300)})
    return result


def _stamp_parts(value):
    parts = {}
    for key, print_ in list((value if isinstance(value, dict) else {}).items())[:12]:
        name, text = _line(key, 40), _line(print_, 64)
        if name and text:
            parts[name] = text
    return parts


def normalize_story_stamps(value):
    """``{book: {part: print}, acts|plan|places|beats|prompts: {id: {part: print}}}``: what each step was made from."""
    source = value if isinstance(value, dict) else {}
    result = {"book": _stamp_parts(source.get("book"))}
    for section in STORY_STAMP_SECTIONS:
        items = {}
        group = source.get(section)
        for item_id, item in list((group if isinstance(group, dict) else {}).items())[:2000]:
            key, parts = _line(item_id, 160), _stamp_parts(item)
            if key and parts:
                items[key] = parts
        result[section] = items
    return result


def normalize_story_continuation_mode(value):
    """``off``, ``location`` or ``llm``; anything else is ``off``."""
    text = re.sub(r"[\s-]+", "_", _line(value).lower())
    return text if text in STORY_CONTINUATION_MODES else "off"


def resolve_story_continuations(plan, scenes, acts, mode):
    """The scene plan with ``continues_previous`` set for every scene.

    ``scenes`` is the timeline in order (``id``, ``start``, ``end``). A manual choice is kept. Otherwise ``location``
    continues a scene that touches the scene before at the same spot in the same chapter, ``llm`` uses the LLM's
    answer and ``off`` continues nothing. The first scene, and a scene with a gap before it, never continue.
    """
    mode = normalize_story_continuation_mode(mode)
    entries = {entry["scene_id"]: dict(entry) for entry in normalize_story_scene_plan(plan)}
    chapter_of = {sid: act["chapter_id"] for act in normalize_story_acts(acts) for sid in act["scene_ids"]}
    previous = None
    for scene in scenes or []:
        sid = str((scene or {}).get("id") or "")
        entry = entries.get(sid)
        if entry is not None:
            before = entries.get(str(previous.get("id") or "")) if previous else None
            touches = previous is not None and abs(_seconds(scene.get("start")) - _seconds(previous.get("end"))) <= 0.05
            if not touches:
                entry["continues_previous"], entry["continuation_source"] = False, ""
            elif entry["continuation_source"] != "manual":
                same_spot = bool(before and entry["spot"] and entry["spot"].lower() == before["spot"].lower()
                                 and chapter_of.get(sid) and chapter_of.get(sid) == chapter_of.get(before["scene_id"]))
                if mode == "location" and same_spot:
                    entry["continues_previous"], entry["continuation_source"] = True, "auto"
                elif mode == "llm" and entry["llm_continues"]:
                    entry["continues_previous"], entry["continuation_source"] = True, "llm"
                else:
                    entry["continues_previous"], entry["continuation_source"] = False, ""
        previous = scene or {}
    return [entries[entry["scene_id"]] for entry in normalize_story_scene_plan(plan)]


def _seconds(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_story_video_style(value):
    """``music_video``, ``spoken`` or ``story_only``; anything else is ``music_video``."""
    text = re.sub(r"[\s-]+", "_", _line(value).lower())
    return text if text in STORY_VIDEO_STYLES else "music_video"


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
    return title or f"Act {raw_number}"


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
            chapters.append({"id": "ch_1", "title": "Act 1", "setting_id": "", "paragraphs": paragraphs})
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
        heading = f"Act {number}: {chapter['title'] or f'Act {number}'}"
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
            "title": title or f"Act {index + 1}",
            "setting_id": sid,
            "paragraphs": list(paragraphs[start - 1:end]),
        })
    return normalize_story_book({"chapters": chapters})["chapters"], names


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
    if entry["continues_previous"]:
        brief.append(
            f"Continuation: this scene is the very next moment of Scene {number - 1} in one unbroken camera take. It "
            "opens exactly where that scene ends (same place, people, framing and camera move), with no cut or new "
            "setup." + (f" Then: {entry['continuation_direction']}" if entry["continuation_direction"] else "")
        )
    if entry.get("interaction"):
        brief.append(f"What the characters do with each other: {entry['interaction']}")
    if entry.get("user_note"):
        brief.append(f"User direction for this scene (follow it exactly): {entry['user_note']}")
    if act_user_text(act):
        brief.append(act_user_text(act))
    if layer.get("song_story_brief"):
        brief.append(str(layer["song_story_brief"]))
    return "\n".join(lines), "\n".join(brief)


_PERSON_REFERENCE_TYPES = {"", "character", "person", "human", "subject", "singer"}


def is_story_person_card(card):
    """True for a subject card that is a person. Clothing, props, vehicles and other things are not cast members.

    Twin of ``isStoryPersonCard`` in web/music_video_builder/story_mode_data.mjs.
    """
    if not isinstance(card, dict) or card.get("extra_reference_for"):
        return False
    return not is_scene_item_card(card)


def is_scene_item_card(card):
    """True for a clothing, prop, vehicle or other thing card: its type is not a person type."""
    return isinstance(card, dict) and str(card.get("reference_type") or "").strip().lower() not in _PERSON_REFERENCE_TYPES


def scene_items_text(items, people):
    """One line for a scene's prompt or beat that says who wears or uses each clothing or prop card mapped to it.

    ``items`` are the scene's clothing and prop cards (``name``, ``description``, ``reference_type``, ``wears``).
    ``people`` are its people (``id``, ``name``). An item's ``wears`` names who wears or uses it in this scene; when
    that person is not in the scene, the first person does. Empty when the scene has no items.
    """
    names = {str(person.get("id") or ""): str(person.get("name") or "").strip() for person in people if isinstance(person, dict)}
    first = next((name for name in names.values() if name), "")
    lines = []
    for item in items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        description = " ".join(str(item.get("description") or "").split())[:300]
        detail = f" ({description})" if description else ""
        user = names.get(str(item.get("wears") or "")) or first
        kind = str(item.get("reference_type") or "").strip().lower()
        if not user:
            lines.append(f"- {name}{detail} is in the scene.")
        elif kind == "outfit":
            lines.append(f"- {user} wears {name}{detail}. Show {user} wearing it whenever {user} is seen.")
        elif kind == "vehicle":
            lines.append(f"- {name}{detail} is {user}'s vehicle: {user} drives it or stands beside it.")
        else:
            lines.append(f"- {user} holds or uses {name}{detail}.")
    if not lines:
        return ""
    return "\n".join(["WARDROBE AND PROPS IN THIS SCENE (things, not people; the beat must show them as described):", *lines])


def _scene_map_ids(value):
    """The ids a scene mapping holds: a list, or a comma-separated string from older projects."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def fill_scene_cast(subject_scene_map, acts, subject_ids):
    """Characters for scenes that have none: each act's cast goes onto its scenes. User mappings are kept, and
    clothing or props mapped to a scene (ids not in ``subject_ids``) stay beside the characters added.

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
            current = _scene_map_ids(result.get(scene_id))
            if any(item in valid for item in current):
                continue
            result[scene_id] = cast + current
            changed.append(scene_id)
    return result, changed


def apply_scene_cast(subject_scene_map, plan, acts, subject_ids):
    """Each planned scene's characters (``cast_ids``) become the characters of its mapping; clothing and props
    mapped to the scene (ids not in ``subject_ids``) are kept after them. Scenes the plan gives no cast keep theirs,
    or get their act's cast when they have none (``fill_scene_cast``).

    Returns ``(new_map, changed_scene_ids)``. Only ids in ``subject_ids`` are used.
    """
    result = {key: list(value) if isinstance(value, list) else value for key, value in (subject_scene_map or {}).items()}
    valid = set(subject_ids)
    changed = []
    for entry in normalize_story_scene_plan(plan):
        cast = [cast_id for cast_id in entry["cast_ids"] if cast_id in valid]
        if not cast:
            continue
        following = cast + [item for item in _scene_map_ids(result.get(entry["scene_id"])) if item not in valid]
        if result.get(entry["scene_id"]) != following:
            result[entry["scene_id"]] = following
            changed.append(entry["scene_id"])
    result, filled = fill_scene_cast(result, acts, subject_ids)
    return result, changed + [scene_id for scene_id in filled if scene_id not in changed]


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
