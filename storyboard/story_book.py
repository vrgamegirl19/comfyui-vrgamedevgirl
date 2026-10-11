"""Story Mode book steps with the selected LLM runner: draft the book, rewrite a chapter, split a pasted story,
and find story characters that have no cast card.

Each function takes the usual Storyboard payload (runner fields, ``story_layer``, ``scenes``, ``reference_builder``,
``lyrics``) and returns ``{"story_layer": {...changed Story Mode fields...}, ...}``. Saving is the caller's job.
"""

from ..llm.output_checks import _extract_json_object_from_text
from ..llm.prompts.story_mode import (
    STORY_CAST_SCAN_SCHEMA,
    STORY_SPLIT_SCHEMA,
    story_book_instruction,
    story_cast_scan_instruction,
    story_chapter_rewrite_instruction,
    story_mode_lyric_guidance,
    story_split_instruction,
)
from .scene_helpers import _clean_scene_text, _normalize_story_layer, _storyboard_dialogue_reference_catalog
from .story_mode import (
    STORY_FLOW_VERSION,
    _CHAPTER_HEADING,
    chapters_from_split,
    chapters_keep_text,
    is_story_mode_layer,
    normalize_story_book,
    parse_story_book_text,
    split_paragraphs,
    story_arc_to_story_text,
    story_book_to_text,
    sync_story_settings,
)

STORY_MODE_LYRIC_STRENGTH = 5


def _run(payload, instruction, *, label, max_new_tokens, temperature=0.7, json_schema=None):
    from ..llm.builder_runner import _run_builder_text_llm

    return _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or temperature),
        top_p=float(payload.get("top_p") or 0.92),
        max_new_tokens=int(payload.get("max_new_tokens") or max_new_tokens),
        label=label,
        preserve_paragraphs=True,
        json_schema=json_schema,
    )


def _layer(payload):
    return _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})


def _scenes(payload):
    scenes = payload.get("scenes")
    return [scene for scene in scenes if isinstance(scene, dict)] if isinstance(scenes, list) else []


def _song_seconds(payload, scenes):
    try:
        given = float(payload.get("song_seconds") or 0)
    except (TypeError, ValueError):
        given = 0.0
    if given > 0:
        return given
    end = 0.0
    for scene in scenes:
        for key in ("timeline_end", "end"):
            try:
                end = max(end, float(scene.get(key) or 0))
            except (TypeError, ValueError):
                pass
    return end


def _scene_count(payload, scenes):
    try:
        given = int(payload.get("scene_count") or 0)
    except (TypeError, ValueError):
        given = 0
    return given or len(scenes)


def cast_lines(payload):
    """``Name (type): description`` for every character card the story can use."""
    subjects, _locations = _storyboard_dialogue_reference_catalog(payload)
    lines = []
    for subject in subjects:
        kind = subject.get("reference_type") or "character"
        description = subject.get("description") or ""
        lines.append(f"- {subject['name']} ({kind})" + (f": {description}" if description else ""))
    return "\n".join(lines)


def _lyrics(payload, scenes):
    lyrics = _clean_scene_text(payload.get("lyrics") or payload.get("lyrics_text") or "", 12000)
    if lyrics:
        return lyrics
    return "\n".join(_clean_scene_text(scene.get("lyrics") or scene.get("lyric_text") or "", 400) for scene in scenes)[:12000]


def _first_story_mode_fields(layer):
    """Fields a project gets the first time it uses Story Mode."""
    if is_story_mode_layer(layer):
        return {}
    return {"story_flow_version": STORY_FLOW_VERSION, "lyric_story_strength": STORY_MODE_LYRIC_STRENGTH}


def _book_fields(layer, book, setting_names):
    """Story layer fields for a new or changed book. Acts and the scene plan are rebuilt from a new book."""
    settings = sync_story_settings(book, setting_names, layer.get("story_settings"))
    return {
        "story_book": book,
        "story_book_text": story_book_to_text(book, settings),
        "story_settings": settings,
    }


def _runner_fields(run_info):
    return {
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


def draft_story_book(payload):
    """Write the whole book from the premise with the LLM."""
    layer = _layer(payload)
    scenes = _scenes(payload)
    scene_count = _scene_count(payload, scenes)
    if scene_count <= 0:
        raise ValueError("Create the timeline scenes before writing the story.")
    premise = _clean_scene_text(payload.get("story_idea") or payload.get("premise") or layer.get("overall_story_idea") or "", 4000)
    strength = layer["lyric_story_strength"] if is_story_mode_layer(layer) else STORY_MODE_LYRIC_STRENGTH
    instruction = story_book_instruction(
        premise=premise,
        cast_text=cast_lines(payload),
        tone_notes=_clean_scene_text(payload.get("tone_notes") or payload.get("style_theme") or "", 2000),
        scene_count=scene_count,
        song_seconds=_song_seconds(payload, scenes),
        lyric_guidance=story_mode_lyric_guidance(strength),
        lyrics=_lyrics(payload, scenes),
        previous_book=_clean_scene_text(payload.get("previous_book") or "", 8000),
        seed=_clean_scene_text(payload.get("seed") or "", 80),
    )
    text, run_info = _run(payload, instruction, label="Story Mode Book", max_new_tokens=6000)
    book, names = parse_story_book_text(text)
    if not book["chapters"] or not any(chapter["paragraphs"] for chapter in book["chapters"]):
        raise ValueError("The LLM did not return a story with chapters. Try again or write the story yourself.")
    fields = {
        **_first_story_mode_fields(layer),
        **_book_fields(layer, book, names),
        "overall_story_idea": premise or book["premise"],
        "story_acts": [],
        "story_scene_plan": [],
    }
    return {"story_layer": fields, "story_book": book, **_runner_fields(run_info)}


def rewrite_story_chapter(payload):
    """Rewrite one chapter with the LLM. Acts and plan entries of that chapter are cleared."""
    layer = _layer(payload)
    book = normalize_story_book(layer.get("story_book"))
    chapter_id = _clean_scene_text(payload.get("chapter_id") or "", 80)
    index = next((i for i, chapter in enumerate(book["chapters"]) if chapter["id"] == chapter_id), -1)
    if index < 0:
        raise ValueError(f"Chapter '{chapter_id}' is not in the story book.")
    settings = layer.get("story_settings") or []
    book_text = story_book_to_text(book, settings)
    chapter_only = {"title": "", "premise": "", "chapters": [book["chapters"][index]]}
    chapter_text = story_book_to_text(chapter_only, settings)
    strength = layer["lyric_story_strength"] if is_story_mode_layer(layer) else STORY_MODE_LYRIC_STRENGTH
    instruction = story_chapter_rewrite_instruction(
        book_text=book_text,
        chapter_number=index + 1,
        chapter_text=chapter_text,
        direction=_clean_scene_text(payload.get("direction") or "", 2000),
        cast_text=cast_lines(payload),
        lyric_guidance=story_mode_lyric_guidance(strength),
    )
    text, run_info = _run(payload, instruction, label="Story Mode Chapter", max_new_tokens=2400)
    rewritten, names = parse_story_book_text(text)
    if not rewritten["chapters"] or not rewritten["chapters"][0]["paragraphs"]:
        raise ValueError("The LLM did not return the chapter. Try again.")
    new_chapter = rewritten["chapters"][0]
    old_chapter = book["chapters"][index]
    new_chapter["id"] = old_chapter["id"]
    if not new_chapter["setting_id"]:
        new_chapter["setting_id"] = old_chapter["setting_id"]
    book["chapters"][index] = new_chapter
    known = {item.get("id"): item.get("name") for item in settings if isinstance(item, dict)}
    names = {**known, **names}
    acts = [act for act in layer.get("story_acts") or [] if act.get("chapter_id") != chapter_id]
    act_ids = {act["id"] for act in acts}
    plan = [entry for entry in layer.get("story_scene_plan") or [] if entry.get("act_id") in act_ids]
    fields = {
        **_first_story_mode_fields(layer),
        **_book_fields(layer, book, names),
        "story_acts": acts,
        "story_scene_plan": plan,
    }
    return {"story_layer": fields, "chapter": new_chapter, **_runner_fields(run_info)}


def _fixed_chapter_starts(text):
    """1-based paragraph numbers where the pasted text already has ``Chapter N`` headings."""
    starts, count = [], 0
    for block in split_paragraphs(text):
        first_line = block.split("\n", 1)[0]
        if _CHAPTER_HEADING.match(first_line):
            starts.append(count + 1)
        count += 1
    return starts


def split_pasted_story(payload):
    """Turn a story the user wrote into chapters with settings. The words are kept exactly as written.

    A story that already has ``Chapter N`` headings keeps them; the LLM only names the settings. Otherwise the LLM
    picks the chapter starts by paragraph number.
    """
    layer = _layer(payload)
    text = _clean_scene_text(payload.get("text") or payload.get("story_text") or "", 200000)
    if not text:
        raise ValueError("Paste the story text first.")
    parsed, parsed_names = parse_story_book_text(text)
    has_headings = len(parsed["chapters"]) > 1 or bool(_fixed_chapter_starts(text))
    if has_headings:
        paragraphs = [p for chapter in parsed["chapters"] for p in chapter["paragraphs"]]
        fixed_starts, position = [], 1
        for chapter in parsed["chapters"]:
            fixed_starts.append(position)
            position += len(chapter["paragraphs"])
    else:
        paragraphs = split_paragraphs(text)
        fixed_starts = []
    if not paragraphs:
        raise ValueError("The pasted story has no paragraphs.")
    numbered = "\n\n".join(f"[{number}] {paragraph}" for number, paragraph in enumerate(paragraphs, start=1))
    instruction = story_split_instruction(
        numbered_paragraphs=numbered[:60000],
        fixed_starts=fixed_starts,
        scene_count=_scene_count(payload, _scenes(payload)),
    )
    raw, run_info = _run(payload, instruction, label="Story Mode Split", max_new_tokens=1600, temperature=0.2,
                         json_schema=STORY_SPLIT_SCHEMA)
    try:
        split = (_extract_json_object_from_text(raw) or {}).get("chapters") or []
    except Exception:
        split = []
    if fixed_starts:
        # Keep the user's own chapter breaks and titles; take only the settings (and missing titles) from the LLM.
        by_order = list(split) + [{}] * len(fixed_starts)
        split = []
        for index, start in enumerate(fixed_starts):
            suggestion = by_order[index] if isinstance(by_order[index], dict) else {}
            own = parsed["chapters"][index]
            own_title = "" if own["title"] == f"Chapter {index + 1}" else own["title"]
            own_setting = next((name for sid, name in parsed_names.items() if sid == own["setting_id"]), "")
            split.append({
                "start_paragraph": start,
                "title": own_title or suggestion.get("title") or "",
                "setting": own_setting or suggestion.get("setting") or "",
            })
    chapters, names = chapters_from_split(paragraphs, split)
    if not chapters_keep_text(paragraphs, chapters):
        raise ValueError("Splitting changed the story text. Nothing was saved.")
    book = normalize_story_book({
        "title": parsed["title"] or _clean_scene_text(payload.get("title") or "", 200),
        "premise": parsed["premise"],
        "chapters": chapters,
    })
    fields = {
        **_first_story_mode_fields(layer),
        **_book_fields(layer, book, names),
        "story_acts": [],
        "story_scene_plan": [],
    }
    if book["premise"] and not layer.get("overall_story_idea"):
        fields["overall_story_idea"] = book["premise"]
    return {"story_layer": fields, "story_book": book, **_runner_fields(run_info)}


def convert_story_arc(payload):
    """Move an older project to Story Mode: its song-section story arc becomes the story book.

    Section headings and scene prefixes are removed, then the text is split into chapters like a pasted story
    (every word kept). The old arc stays saved in ``user_story_arc``.
    """
    layer = _layer(payload)
    text = story_arc_to_story_text(layer.get("user_story_arc") or "")
    if not text:
        raise ValueError("This project has no story arc to convert. Write a new story instead.")
    return split_pasted_story({**payload, "text": text})


def scan_story_cast(payload):
    """People in the story who have no character card, each with a drafted look."""
    layer = _layer(payload)
    book = normalize_story_book(layer.get("story_book"))
    if not book["chapters"]:
        raise ValueError("Write the story first.")
    subjects, _locations = _storyboard_dialogue_reference_catalog(payload)
    extras = (payload.get("reference_builder") or {}).get("extras") or []
    known = [subject["name"] for subject in subjects] + [
        _clean_scene_text(extra.get("name") or "", 160) for extra in extras if isinstance(extra, dict)
    ]
    known = [name for name in known if name]
    instruction = story_cast_scan_instruction(
        book_text=story_book_to_text(book, layer.get("story_settings"))[:60000],
        known_names="\n".join(f"- {name}" for name in known),
    )
    raw, run_info = _run(payload, instruction, label="Story Mode Cast Scan", max_new_tokens=1600, temperature=0.2,
                         json_schema=STORY_CAST_SCAN_SCHEMA)
    try:
        found = (_extract_json_object_from_text(raw) or {}).get("characters") or []
    except Exception as error:
        raise ValueError(f"The LLM did not return a character list: {error}") from error
    known_lower = {name.lower() for name in known}
    characters, seen = [], set()
    for item in found:
        if not isinstance(item, dict):
            continue
        name = _clean_scene_text(item.get("name") or "", 120)
        key = name.lower()
        if not name or key in known_lower or key in seen:
            continue
        seen.add(key)
        role = _clean_scene_text(item.get("role") or "", 40).lower()
        characters.append({
            "name": name,
            "role": role if role in ("main", "supporting", "background") else "supporting",
            "description": _clean_scene_text(item.get("description") or "", 600),
        })
    return {"characters": characters, **_runner_fields(run_info)}


def story_book_payload_summary(layer):
    """Short JSON-safe summary of a book for logs and API replies."""
    book = normalize_story_book((layer or {}).get("story_book"))
    return {
        "title": book["title"],
        "chapters": [
            {"id": c["id"], "title": c["title"], "setting_id": c["setting_id"], "paragraphs": len(c["paragraphs"])}
            for c in book["chapters"]
        ],
    }
