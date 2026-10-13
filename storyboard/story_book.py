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
    story_remove_people_instruction,
    story_split_instruction,
)
from .scene_helpers import _clean_scene_text, _normalize_story_layer, _storyboard_dialogue_reference_catalog
from .story_mode import (
    STORY_FLOW_VERSION,
    _CHAPTER_HEADING,
    chapters_from_split,
    chapters_keep_text,
    is_story_mode_layer,
    is_story_person_card,
    normalize_story_book,
    normalize_story_video_style,
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


def story_template(payload, key):
    """The project's own text for a Story Mode instruction (``story_mode_*``), or its built-in default."""
    from ..llm.builder_instructions import _BUILDER_INSTRUCTION_DEFAULTS, _effective_builder_instruction

    return _effective_builder_instruction(payload, key, _BUILDER_INSTRUCTION_DEFAULTS[key])


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


def story_cast(payload):
    """The people the story is about: the ones picked in ``story_cast_ids``, or every person card when none are.
    Clothing and prop cards are never part of the cast."""
    subjects, _locations = _storyboard_dialogue_reference_catalog(payload)
    subjects = [subject for subject in subjects if is_story_person_card(subject)]
    chosen = set(_layer(payload).get("story_cast_ids") or [])
    picked = [subject for subject in subjects if subject.get("id") in chosen]
    return picked if chosen and picked else subjects


def allow_new_cast(payload):
    """Always False: the story uses only the cast the user picked (old projects may still carry the setting)."""
    return False


_ROLE_LABELS = {"main": "main character", "love_interest": "love interest", "supporting": "supporting",
                "antagonist": "antagonist"}
_RELATIONSHIP_LABELS = {"love_interest": "love interests", "partners": "partners", "exes": "exes",
                        "family": "family", "friends": "friends", "rivals": "rivals", "strangers": "strangers",
                        "other": "connected"}


def main_cast(payload):
    """The main character card: the one picked in ``story_main_cast_id``, or the first story cast card."""
    cast = story_cast(payload)
    chosen = str(_layer(payload).get("story_main_cast_id") or "")
    return next((subject for subject in cast if subject.get("id") == chosen), cast[0] if cast else None)


def cast_role(payload, subject):
    """The subject's role in the story: the main character is always ``main``; others default to supporting."""
    main = main_cast(payload)
    if main and subject.get("id") == main.get("id"):
        return "main"
    role = (_layer(payload).get("story_cast_roles") or {}).get(subject.get("id"), "supporting")
    return "supporting" if role == "main" else role


def relationship_lines(payload):
    """``- Mara and Joe: love interests (estranged)`` for each relationship between story cast members."""
    names = {subject.get("id"): subject["name"] for subject in story_cast(payload)}
    lines = []
    for item in _layer(payload).get("story_relationships") or []:
        if item["a"] in names and item["b"] in names:
            note = f" ({item['note']})" if item.get("note") else ""
            lines.append(f"- {names[item['a']]} and {names[item['b']]}: {_RELATIONSHIP_LABELS.get(item['type'], 'connected')}{note}")
    return lines


def cast_lines(payload):
    """``Name (type, role): description`` for every character the story uses, then their relationships."""
    lines = []
    for subject in story_cast(payload):
        kind = subject.get("reference_type") or "character"
        role = _ROLE_LABELS.get(cast_role(payload, subject), "supporting")
        description = subject.get("description") or ""
        lines.append(f"- {subject['name']} ({kind}, {role})" + (f": {description}" if description else ""))
    relationships = relationship_lines(payload)
    if relationships:
        lines += ["", "Relationships:", *relationships]
    return "\n".join(lines)


def _clock(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def song_map(scenes):
    """The song in order, one line per section: ``0:12-0:40 Verse 1: lyric / lyric``.

    Neighbouring scenes with the same section label are one section. Scenes without a label are grouped into
    runs of vocal or instrumental scenes.
    """
    groups = []
    for scene in scenes:
        lyric = _clean_scene_text(scene.get("lyrics") or scene.get("lyric_text") or "", 300).replace("\n", " / ")
        instrumental = bool(scene.get("lyric_instrumental")) or not lyric or lyric.strip("[] ").lower() == "instrumental"
        section = _clean_scene_text(scene.get("lyric_section") or "", 60)
        key = section.lower() or ("instrumental" if instrumental else "vocal")
        start = _number(scene.get("timeline_start", scene.get("start")))
        end = _number(scene.get("timeline_end", scene.get("end")))
        if groups and groups[-1]["key"] == key:
            groups[-1]["end"] = max(groups[-1]["end"], end)
            if not instrumental:
                groups[-1]["lyrics"].append(lyric)
            continue
        groups.append({"key": key, "label": section or ("Instrumental" if instrumental else "Vocals"),
                       "start": start, "end": end, "lyrics": [] if instrumental else [lyric]})
    lines = []
    for group in groups:
        words = " / ".join(group["lyrics"])[:600]
        lines.append(f"{_clock(group['start'])}-{_clock(group['end'])} {group['label']}"
                     + (f": {words}" if words else " (no vocals)"))
    return "\n".join(lines)


def video_style(payload):
    """How the story is told on screen (``music_video``, ``spoken`` or ``story_only``)."""
    return normalize_story_video_style(_layer(payload).get("story_video_style"))


def lead_name(payload):
    """The main character's name, used as the performer in the prompts."""
    main = main_cast(payload)
    return main["name"] if main else ""


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
        allow_new_cast=allow_new_cast(payload),
        tone_notes=_clean_scene_text(payload.get("tone_notes") or payload.get("style_theme") or "", 2000),
        scene_count=scene_count,
        song_seconds=_song_seconds(payload, scenes),
        lyric_guidance=story_mode_lyric_guidance(strength),
        lyrics=song_map(scenes) or _lyrics(payload, scenes),
        video_style=video_style(payload),
        lead_name=lead_name(payload),
        previous_book=_clean_scene_text(payload.get("previous_book") or "", 8000),
        seed=_clean_scene_text(payload.get("seed") or "", 80),
        template=story_template(payload, "story_mode_book"),
        act_count=layer.get("story_act_count") or 0,
    )
    text, run_info = _run(payload, instruction, label="Story Mode Book", max_new_tokens=6000)
    book, names = parse_story_book_text(text)
    if not book["chapters"] or not any(chapter["paragraphs"] for chapter in book["chapters"]):
        raise ValueError("The LLM did not return a story with chapters. Try again or write the story yourself.")
    book, names, removed = _keep_book_to_cast(payload, book, names)
    fields = {
        **_first_story_mode_fields(layer),
        **_book_fields(layer, book, names),
        "overall_story_idea": premise or book["premise"],
        "story_acts": [],
        "story_scene_plan": [],
    }
    return {"story_layer": fields, "story_book": book, "removed_people": removed, **_runner_fields(run_info)}


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
        allow_new_cast=allow_new_cast(payload),
        lyric_guidance=story_mode_lyric_guidance(strength),
        video_style=video_style(payload),
        lead_name=lead_name(payload),
        template=story_template(payload, "story_mode_chapter"),
    )
    text, run_info = _run(payload, instruction, label="Story Mode Chapter", max_new_tokens=2400)
    rewritten, names = parse_story_book_text(text)
    if not rewritten["chapters"] or not rewritten["chapters"][0]["paragraphs"]:
        raise ValueError("The LLM did not return the chapter. Try again.")
    rewritten, names, removed = _keep_book_to_cast(payload, rewritten, names)
    new_chapter = rewritten["chapters"][0]
    old_chapter = book["chapters"][index]
    new_chapter["id"] = old_chapter["id"]
    if not new_chapter["setting_id"]:
        new_chapter["setting_id"] = old_chapter["setting_id"]
    # The act keeps its purpose and presence grid.
    new_chapter["purpose"] = old_chapter.get("purpose", "")
    new_chapter["presence"] = old_chapter.get("presence", {})
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
    return {"story_layer": fields, "chapter": new_chapter, "removed_people": removed, **_runner_fields(run_info)}


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


def _known_people(payload):
    """Names the story may use: the story cast (every card when new people are allowed) and the extras."""
    if allow_new_cast(payload):
        subjects, _locations = _storyboard_dialogue_reference_catalog(payload)
    else:
        subjects = story_cast(payload)
    extras = (payload.get("reference_builder") or {}).get("extras") or []
    known = [subject["name"] for subject in subjects] + [
        _clean_scene_text(extra.get("name") or "", 160) for extra in extras if isinstance(extra, dict)
    ]
    return [name for name in known if name]


def _scan_people(payload, text, known, *, label):
    """LLM list of people in ``text`` who are not in ``known``, and the run info."""
    instruction = story_cast_scan_instruction(
        book_text=text[:60000],
        known_names="\n".join(f"- {name}" for name in known),
    )
    raw, run_info = _run(payload, instruction, label=label, max_new_tokens=1600, temperature=0.2,
                         json_schema=STORY_CAST_SCAN_SCHEMA)
    try:
        found = (_extract_json_object_from_text(raw) or {}).get("characters") or []
    except Exception as error:
        raise ValueError(f"The LLM did not return a character list: {error}") from error
    return found, run_info


def _without_people(payload, book, setting_names, people):
    """The book rewritten by the LLM so ``people`` are gone. ``ok`` is False when the reply was unusable."""
    settings = sync_story_settings(book, setting_names, _layer(payload).get("story_settings"))
    instruction = story_remove_people_instruction(
        story_text=story_book_to_text(book, settings), cast_text=cast_lines(payload), names=people,
    )
    size = sum(len(paragraph) for chapter in book["chapters"] for paragraph in chapter["paragraphs"])
    text, _run_info = _run(payload, instruction, label="Story Mode Remove People", temperature=0.5,
                           max_new_tokens=max(1600, min(8000, size // 2)))
    fixed, fixed_names = parse_story_book_text(text)
    if len(fixed["chapters"]) != len(book["chapters"]) or not all(chapter["paragraphs"] for chapter in fixed["chapters"]):
        return book, setting_names, False
    for new, old in zip(fixed["chapters"], book["chapters"]):
        new["id"] = old["id"]
        if not new["setting_id"]:
            new["setting_id"] = old["setting_id"]
    fixed["title"] = fixed["title"] or book["title"]
    fixed["premise"] = fixed["premise"] or book["premise"]
    return fixed, {**setting_names, **fixed_names}, True


def _keep_book_to_cast(payload, book, setting_names):
    """With new people off, rewrite out anyone outside the story cast. Returns the book, setting names, removed."""
    if allow_new_cast(payload) or not story_cast(payload):
        return book, setting_names, []
    settings = sync_story_settings(book, setting_names, _layer(payload).get("story_settings"))
    known = _known_people(payload)
    try:
        found, _run_info = _scan_people(payload, story_book_to_text(book, settings), known, label="Story Mode Cast Check")
    except ValueError as error:
        print(f"[VRGDG Story Mode] Cast check skipped, the reply was not a list: {error}")
        return book, setting_names, []
    known_lower = {name.lower() for name in known}
    people = []
    for item in found:
        name = _clean_scene_text(item.get("name") or "", 120) if isinstance(item, dict) else ""
        if name and name.lower() not in known_lower and name not in people:
            people.append(name)
    if not people:
        return book, setting_names, []
    fixed, names, ok = _without_people(payload, book, setting_names, people)
    return (fixed, names, people) if ok else (book, setting_names, [])


def remove_story_people(payload):
    """Rewrite the book without the named people (``names``), so only the story cast remains."""
    layer = _layer(payload)
    book = normalize_story_book(layer.get("story_book"))
    if not book["chapters"]:
        raise ValueError("Write the story first.")
    names = payload.get("names") if isinstance(payload.get("names"), list) else []
    people = [_clean_scene_text(name, 120) for name in names if _clean_scene_text(name, 120)]
    if not people:
        raise ValueError("Name the people to remove from the story.")
    known = {item.get("id"): item.get("name") for item in layer.get("story_settings") or [] if isinstance(item, dict)}
    fixed, setting_names, ok = _without_people(payload, book, known, people)
    if not ok:
        raise ValueError("The LLM did not return the whole story. Nothing was changed; try again.")
    fields = {
        **_first_story_mode_fields(layer),
        **_book_fields(layer, fixed, setting_names),
        "story_acts": [],
        "story_scene_plan": [],
    }
    return {"story_layer": fields, "story_book": fixed, "removed_people": people}


def scan_story_cast(payload):
    """People in the story with no character card, or, with new people off, who are not in the story cast."""
    layer = _layer(payload)
    book = normalize_story_book(layer.get("story_book"))
    if not book["chapters"]:
        raise ValueError("Write the story first.")
    known = _known_people(payload)
    found, run_info = _scan_people(payload, story_book_to_text(book, layer.get("story_settings")), known,
                                   label="Story Mode Cast Scan")
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
