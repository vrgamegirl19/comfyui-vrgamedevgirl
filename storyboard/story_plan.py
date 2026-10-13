"""Story Mode acts and scene plan with the selected LLM runner.

``build_story_acts`` gives each chapter a run of timeline scenes (proportional to its paragraphs) and asks the LLM to
split that run into acts. ``build_story_scene_plan`` asks, act by act, for each scene's story slice, spot, mood and
performance. Code checks and repairs every reply (``storyboard/story_acts.py``), so a weak reply never leaves a scene
without an act or a plan entry. Both return the story layer fields to save; saving is the caller's job.
"""

from ..llm.output_checks import _extract_json_object_from_text
from ..llm.prompts.story_mode import (
    STORY_ACT_SUMMARY_SCHEMA,
    STORY_ACTS_SCHEMA,
    STORY_SCENE_PLAN_SCHEMA,
    story_act_summary_instruction,
    story_acts_instruction,
    story_mode_lyric_guidance,
    story_scene_plan_instruction,
)
from .scene_helpers import _clean_scene_text, _normalize_story_layer, _storyboard_dialogue_reference_catalog
from .story_acts import (
    DEFAULT_SPOT_MIN_SCENES,
    allocate_chapter_scenes,
    chapter_ranges,
    enforce_spot_runs,
    fallback_act_ranges,
    keep_notes_inside_acts,
    repair_act_ranges,
    validate_story_acts,
)
from .story_book import _run, cast_lines, main_cast, story_cast, story_template
from .story_mode import (
    act_user_text,
    normalize_plan_performance,
    normalize_story_book,
    normalize_story_video_style,
    resolve_story_continuations,
)
from .timeline_notes import story_arc_timeline_notes

PLAN_NOTE_FIELDS = (("director", "timeline_note"), ("video", "motion_summary"), ("planning", "notes"))


def _scenes(payload):
    scenes = payload.get("scenes")
    return [scene for scene in scenes if isinstance(scene, dict)] if isinstance(scenes, list) else []


def _scene_id(scene, index):
    return str(scene.get("id") or f"scene_{index + 1}")


def _seconds(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clock(seconds):
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def _scene_notes(scene):
    notes = {}
    for source, key in PLAN_NOTE_FIELDS:
        text = _clean_scene_text(scene.get(key) or (scene.get("i2v_notes") if source == "video" else "") or "", 800)
        if text:
            notes[source] = text
    return notes


def _scene_line(number, scene, timed_notes, *, with_notes):
    start = _seconds(scene.get("timeline_start", scene.get("start")))
    end = _seconds(scene.get("timeline_end", scene.get("end")))
    lyric = _clean_scene_text(scene.get("lyrics") or scene.get("lyric_text") or "", 300).replace("\n", " / ")
    instrumental = bool(scene.get("lyric_instrumental")) or not lyric
    parts = [f"Scene {number} ({_clock(start)}-{_clock(end)}, {max(0.0, end - start):.1f} s"]
    if scene.get("lyric_section"):
        parts[0] += f", {_clean_scene_text(scene['lyric_section'], 60)}"
    parts[0] += ", instrumental)" if instrumental else ")"
    if lyric and not instrumental:
        parts.append(f"lyric mood: \"{lyric}\"")
    if with_notes:
        for source, text in _scene_notes(scene).items():
            parts.append(f"{source} note: {text}")
        for note in timed_notes:
            if number in note.get("scene_numbers") or []:
                parts.append(f"timed note: {note['note']}")
    return " | ".join(parts)


def _names(value):
    """Lowercased names from an LLM list of names."""
    return [str(name).strip().lower() for name in (value if isinstance(value, list) else []) if str(name).strip()]


def _user_direction(entry, names_by_id):
    """The user's own direction for a scene (its note and a cast they set), added to its scene line."""
    parts = []
    if entry.get("cast_source") == "manual" and entry.get("cast_ids"):
        names = [names_by_id.get(cid, cid) for cid in entry["cast_ids"]]
        parts.append(f"user direction, characters in this scene: {', '.join(names)}")
    if entry.get("user_note"):
        parts.append(f"user direction: {entry['user_note']}")
    return "".join(f" | {part}" for part in parts)


def _note_ranges(timed_notes):
    ranges = []
    for note in timed_notes:
        numbers = sorted(int(n) for n in note.get("scene_numbers") or [] if str(n).isdigit() or isinstance(n, int))
        if numbers:
            ranges.append((numbers[0], numbers[-1]))
    return ranges


def _settings_by_id(layer):
    return {item["id"]: item for item in layer.get("story_settings") or [] if isinstance(item, dict)}


def _cast_ids_by_name(payload):
    subjects, _locations = _storyboard_dialogue_reference_catalog(payload)
    return {subject["name"].lower(): subject["id"] or subject["name"] for subject in subjects if subject.get("name")}


def _parse(raw, key):
    try:
        data = _extract_json_object_from_text(raw) or {}
    except Exception:
        return []
    items = data.get(key) if isinstance(data, dict) else None
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _chapter_scene_ranges(layer, chapters, scene_ids):
    """Each chapter's scene range: from its saved acts when every chapter has some, else by paragraph count."""
    number_by_id = {sid: index + 1 for index, sid in enumerate(scene_ids)}
    by_chapter = {}
    for act in layer.get("story_acts") or []:
        numbers = [number_by_id[sid] for sid in act.get("scene_ids") or [] if sid in number_by_id]
        if numbers:
            low, high = by_chapter.get(act.get("chapter_id"), (min(numbers), max(numbers)))
            by_chapter[act.get("chapter_id")] = (min(low, min(numbers)), max(high, max(numbers)))
    if chapters and all(chapter["id"] in by_chapter for chapter in chapters) and not validate_story_acts(
        layer.get("story_acts") or [], chapters, scene_ids
    ):
        return [by_chapter[chapter["id"]] for chapter in chapters]
    counts = allocate_chapter_scenes([len(chapter["paragraphs"]) for chapter in chapters], len(scene_ids))
    return chapter_ranges(counts)


def build_story_acts(payload, progress=None):
    """Acts for every chapter (or only ``chapter_id``). The scene plan of rebuilt chapters is cleared."""
    layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    chapters = normalize_story_book(layer.get("story_book"))["chapters"]
    if not chapters:
        raise ValueError("Write the story first; acts are made from its chapters.")
    scenes = _scenes(payload)
    if not scenes:
        raise ValueError("Create the timeline scenes first.")
    scene_ids = [_scene_id(scene, index) for index, scene in enumerate(scenes)]
    ranges = _chapter_scene_ranges(layer, chapters, scene_ids)
    only = str(payload.get("chapter_id") or "").strip()
    if only and only not in {chapter["id"] for chapter in chapters}:
        raise ValueError(f"Chapter '{only}' is not in the story book.")
    timed_notes = story_arc_timeline_notes({"scenes": [
        {**scene, "scene_number": index + 1} for index, scene in enumerate(scenes)
    ], "timeline_markers": payload.get("timeline_markers") or []})
    settings = _settings_by_id(layer)
    cast_text = cast_lines(payload)
    cast_ids = _cast_ids_by_name(payload)
    kept = {}
    for act in (layer.get("story_acts") or []) if only else []:  # a full rebuild replaces every chapter's acts
        kept.setdefault(act["chapter_id"], []).append(act)
    built = {}
    run_info = {}
    todo = [i for i, chapter in enumerate(chapters) if ranges[i] and (not only or chapter["id"] == only)]
    for step, index in enumerate(todo):
        chapter = chapters[index]
        first, last = ranges[index]
        if progress:
            progress(step, len(todo), chapter["title"])
        chapter_notes = [
            {**note, "scene_numbers": [n for n in note["scene_numbers"] if first <= n <= last]}
            for note in timed_notes if any(first <= n <= last for n in note["scene_numbers"])
        ]
        proposals = []
        if last > first:
            instruction = story_acts_instruction(
                chapter_number=index + 1,
                chapter_title=chapter["title"],
                setting_name=(settings.get(chapter["setting_id"]) or {}).get("name", ""),
                chapter_text="\n\n".join(chapter["paragraphs"]),
                scenes_text="\n".join(
                    _scene_line(n, scenes[n - 1], chapter_notes, with_notes=False) for n in range(first, last + 1)
                ),
                cast_text=cast_text,
                notes_text="\n".join(
                    f"- scenes {note['scene_numbers'][0]}-{note['scene_numbers'][-1]}: {note['note']}"
                    for note in chapter_notes if note["scene_numbers"]
                ),
                target_acts=max(1, round((last - first + 1) / 4)),
                template=story_template(payload, "story_mode_acts"),
            )
            try:
                raw, run_info = _run(payload, instruction, label="Story Mode Acts", max_new_tokens=2400,
                                     temperature=0.4, json_schema=STORY_ACTS_SCHEMA)
                proposals = _parse(raw, "acts")
            except Exception as error:  # a failed chapter falls back to one act per paragraph
                print(f"[VRGDG Story Mode] Acts for chapter {index + 1} fell back to paragraphs: {error}")
        by_start = {}
        for item in proposals:
            try:
                by_start.setdefault(int(item.get("first_scene")), item)
            except (TypeError, ValueError):
                continue
        if by_start:
            act_ranges = repair_act_ranges([(start, None) for start in by_start], first, last)
        else:
            act_ranges = fallback_act_ranges(len(chapter["paragraphs"]), first, last)
        act_ranges = keep_notes_inside_acts(act_ranges, _note_ranges(chapter_notes))
        acts = []
        for part, (a, b) in enumerate(act_ranges):
            item = by_start.get(a) or {}
            fallback_text = chapter["paragraphs"][min(part, len(chapter["paragraphs"]) - 1)] if chapter["paragraphs"] else ""
            names = [str(name).strip().lower() for name in item.get("cast") or [] if str(name).strip()]
            acts.append({
                "chapter_id": chapter["id"],
                "title": _clean_scene_text(item.get("title") or f"{chapter['title']} - part {part + 1}", 200),
                "summary": _clean_scene_text(item.get("summary") or fallback_text, 2000),
                "cast_ids": [cast_ids[name] for name in names if name in cast_ids],
                "scene_ids": scene_ids[a - 1:b],
            })
        built[chapter["id"]] = acts
    ordered = []
    for chapter in chapters:
        ordered.extend(built.get(chapter["id"], kept.get(chapter["id"], [])))
    story_acts = [{**act, "id": f"act_{index + 1}"} for index, act in enumerate(ordered)]
    rebuilt_scenes = {sid for chapter_id in built for act in built[chapter_id] for sid in act["scene_ids"]}
    plan = [entry for entry in layer.get("story_scene_plan") or [] if entry.get("scene_id") not in rebuilt_scenes]
    act_by_scene = {sid: act["id"] for act in story_acts for sid in act["scene_ids"]}
    plan = [{**entry, "act_id": act_by_scene.get(entry["scene_id"], "")} for entry in plan]
    if progress:
        progress(len(todo), len(todo), "done")
    return {
        "story_layer": {"story_acts": story_acts, "story_scene_plan": plan},
        "runner": run_info.get("runner", ""),
        "used_model": run_info.get("used_model", ""),
    }


def _default_performance(scene):
    lyric = _clean_scene_text(scene.get("lyrics") or scene.get("lyric_text") or "", 300)
    return "instrumental" if scene.get("lyric_instrumental") or not lyric else "singing"


def style_performance(performance, style):
    """A scene performance that fits the video style: spoken never sings, story only never sings or speaks."""
    if style == "spoken" and performance == "singing":
        return "speaking"
    if style == "story_only" and performance in ("singing", "speaking"):
        return "no_lip_sync"
    return performance


def build_story_scene_plan(payload, progress=None):
    """A plan entry for every scene of every act (or only ``act_id``), with spots kept for at least
    ``spot_min_scenes`` scenes in a row."""
    layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    chapters = normalize_story_book(layer.get("story_book"))["chapters"]
    scenes = _scenes(payload)
    scene_ids = [_scene_id(scene, index) for index, scene in enumerate(scenes)]
    acts = layer.get("story_acts") or []
    if not acts:
        raise ValueError("Create the acts first; the scene plan is written act by act.")
    problems = validate_story_acts(acts, chapters, scene_ids)
    if problems:
        raise ValueError("The acts do not match the timeline. Rebuild or fix the acts first: " + " ".join(problems))
    only = str(payload.get("act_id") or "").strip()
    if only and only not in {act["id"] for act in acts}:
        raise ValueError(f"Act '{only}' does not exist.")
    # Replan only these scenes (Scenes tab, scene by scene). The LLM still sees the whole act; other scenes keep their plan.
    only_scenes = {str(item) for item in payload.get("scene_ids") or [] if str(item).strip()}
    try:
        spot_min = max(1, int(payload.get("spot_min_scenes") or layer.get("story_spot_min_scenes")
                              or DEFAULT_SPOT_MIN_SCENES))
    except (TypeError, ValueError):
        spot_min = DEFAULT_SPOT_MIN_SCENES
    number_by_id = {sid: index + 1 for index, sid in enumerate(scene_ids)}
    timed_notes = story_arc_timeline_notes({"scenes": [
        {**scene, "scene_number": index + 1} for index, scene in enumerate(scenes)
    ], "timeline_markers": payload.get("timeline_markers") or []})
    settings = _settings_by_id(layer)
    chapter_by_id = {chapter["id"]: chapter for chapter in chapters}
    guidance = story_mode_lyric_guidance(layer.get("lyric_story_strength", 5))
    cast_text = cast_lines(payload)
    # The scene cast comes only from the story cast; names are matched without case.
    names_by_id = {subject["id"]: subject["name"] for subject in story_cast(payload) if subject.get("id")}
    ids_by_name = {name.lower(): cid for cid, name in names_by_id.items()}
    main = main_cast(payload)
    main_id, main_name = (main.get("id", ""), main.get("name", "")) if main else ("", "")
    style = normalize_story_video_style(layer.get("story_video_style"))
    plan = {entry["scene_id"]: dict(entry) for entry in layer.get("story_scene_plan") or []}
    run_info = {}
    todo = [act for act in acts if not only or act["id"] == only]
    for step, act in enumerate(todo):
        if progress:
            progress(step, len(todo), act["title"] or act["id"])
        chapter = chapter_by_id[act["chapter_id"]]
        setting_name = (settings.get(chapter["setting_id"]) or {}).get("name", "")
        numbers = sorted(number_by_id[sid] for sid in act["scene_ids"])
        act_names = [names_by_id[cid] for cid in act.get("cast_ids") or [] if cid in names_by_id]
        previous_spot = ""
        if numbers[0] > 1:
            previous_spot = (plan.get(scene_ids[numbers[0] - 2]) or {}).get("spot", "")
        chapter_scene_ids = {sid for other in acts if other["chapter_id"] == act["chapter_id"] for sid in other["scene_ids"]}
        known_spots = sorted({entry["spot"] for sid, entry in plan.items() if sid in chapter_scene_ids and entry.get("spot")})
        instruction = story_scene_plan_instruction(
            act_title=act["title"],
            act_summary="\n".join(filter(None, [act["summary"], act_user_text(act)])),
            setting_name=setting_name,
            chapter_text="\n\n".join(chapter["paragraphs"]),
            scenes_text="\n".join(
                _scene_line(n, scenes[n - 1], timed_notes, with_notes=True)
                + _user_direction(plan.get(scene_ids[n - 1]) or {}, names_by_id)
                for n in numbers
            ),
            cast_text=cast_text,
            spot_min_scenes=spot_min,
            previous_spot=previous_spot,
            lyric_guidance=guidance,
            known_spots=", ".join(known_spots),
            video_style=style,
            act_cast=", ".join(act_names),
            main_name=main_name,
            template=story_template(payload, "story_mode_scene_plan"),
        )
        replies = {}
        try:
            raw, run_info = _run(payload, instruction, label="Story Mode Scene Plan", max_new_tokens=3000,
                                 temperature=0.5, json_schema=STORY_SCENE_PLAN_SCHEMA)
            for item in _parse(raw, "scenes"):
                try:
                    replies.setdefault(int(item.get("scene")), item)
                except (TypeError, ValueError):
                    continue
        except Exception as error:  # the act keeps a plan built from its summary
            print(f"[VRGDG Story Mode] Scene plan for {act['id']} fell back to the act summary: {error}")
        for number in numbers:
            if only_scenes and scene_ids[number - 1] not in only_scenes:
                continue
            scene = scenes[number - 1]
            item = replies.get(number) or {}
            performance = normalize_plan_performance(item.get("performance") or _default_performance(scene))
            performance = style_performance(performance, style)
            if _default_performance(scene) == "instrumental":
                performance = "instrumental"
            sources = list(_scene_notes(scene))
            if any(number in note.get("scene_numbers") or [] for note in timed_notes):
                sources.append("timeline")
            # A continuation the user set by hand survives a replan, with its direction.
            old = plan.get(scene_ids[number - 1]) or {}
            manual = old.get("continuation_source") == "manual"
            # So does a cast the user set by hand. Otherwise the story's cast: names from the cast list only.
            if old.get("cast_source") == "manual":
                scene_cast, cast_source = list(old.get("cast_ids") or []), "manual"
            else:
                scene_cast = [ids_by_name[name] for name in _names(item.get("cast")) if name in ids_by_name]
                scene_cast = (list(dict.fromkeys(scene_cast))
                              or [cid for cid in act.get("cast_ids") or [] if cid in names_by_id]
                              or ([main_id] if main_id else []))
                cast_source = "story"
            performer = ids_by_name.get(str(item.get("performer") or "").strip().lower(), "")
            if performer not in scene_cast:
                performer = ""
            plan[scene_ids[number - 1]] = {
                "scene_id": scene_ids[number - 1],
                "act_id": act["id"],
                "story_slice": _clean_scene_text(item.get("story_slice") or act["summary"], 1200),
                "spot": _clean_scene_text(item.get("spot") or "", 160),
                "mood": _clean_scene_text(item.get("mood") or "", 200),
                "performance_mode": performance,
                "note_sources": sources,
                "continues_previous": bool(old.get("continues_previous")) if manual else False,
                "continuation_source": "manual" if manual else "",
                "llm_continues": item.get("continues") is True,
                "continuation_direction": old.get("continuation_direction", "") if manual
                else _clean_scene_text(item.get("continuation") or "", 600),
                "cast_ids": scene_cast,
                "performer_id": performer,
                "interaction": _clean_scene_text(item.get("interaction") or "", 600) if len(scene_cast) > 1 else "",
                "cast_source": cast_source,
                "user_note": old.get("user_note", ""),
            }
    # Spots are kept for a minimum run inside each chapter. Scenes with a user note, and (when only one act was
    # rebuilt) scenes outside that act, keep their spot.
    rebuilt = {sid for act in todo for sid in act["scene_ids"] if not only_scenes or sid in only_scenes}
    for chapter in chapters:
        ids = [sid for act in acts if act["chapter_id"] == chapter["id"] for sid in act["scene_ids"]]
        ids = [sid for sid in sorted(ids, key=number_by_id.get) if sid in plan]
        if not ids:
            continue
        locked = {i for i, sid in enumerate(ids) if plan[sid]["note_sources"] or sid not in rebuilt}
        spots = enforce_spot_runs([plan[sid]["spot"] for sid in ids], spot_min, locked)
        fallback_spot = (settings.get(chapter["setting_id"]) or {}).get("name", "") or chapter["title"]
        for sid, spot in zip(ids, spots):
            plan[sid]["spot"] = spot or fallback_spot
    timeline = [{"id": sid, "start": _seconds(scene.get("timeline_start", scene.get("start"))),
                 "end": _seconds(scene.get("timeline_end", scene.get("end")))} for sid, scene in zip(scene_ids, scenes)]
    ordered = resolve_story_continuations([plan[sid] for sid in scene_ids if sid in plan], timeline, acts,
                                          layer.get("story_continuation_mode"))
    if progress:
        progress(len(todo), len(todo), "done")
    return {
        "story_layer": {"story_scene_plan": ordered},
        "runner": run_info.get("runner", ""),
        "used_model": run_info.get("used_model", ""),
    }


def rewrite_act_summary(payload):
    """Rewrite one act's description (``act_id``) so it matches the scenes planned for it and the user's own words.

    Returns the story layer's ``story_acts`` with only that act's summary changed. Saving is the caller's job.
    """
    layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    acts = layer.get("story_acts") or []
    act_id = str(payload.get("act_id") or "").strip()
    act = next((item for item in acts if item["id"] == act_id), None)
    if not act:
        raise ValueError(f"Act '{act_id}' does not exist.")
    scenes = _scenes(payload)
    scene_ids = [_scene_id(scene, index) for index, scene in enumerate(scenes)]
    number_by_id = {sid: index + 1 for index, sid in enumerate(scene_ids)}
    plan = {entry["scene_id"]: entry for entry in layer.get("story_scene_plan") or []}
    names_by_id = {subject["id"]: subject["name"] for subject in story_cast(payload) if subject.get("id")}
    lines = []
    for sid in sorted((sid for sid in act["scene_ids"] if sid in plan), key=lambda sid: number_by_id.get(sid, 0)):
        entry = plan[sid]
        who = ", ".join(names_by_id[cid] for cid in entry.get("cast_ids") or [] if cid in names_by_id)
        line = f"Scene {number_by_id.get(sid, '?')} ({entry.get('spot') or 'no spot'}"
        line += f"; {who})" if who else ")"
        lines.append(f"{line}: {entry.get('story_slice', '')}" + (f" {entry['interaction']}" if entry.get("interaction") else ""))
    if not lines:
        raise ValueError("Plan the scenes of this act first.")
    chapters = {chapter["id"]: chapter for chapter in normalize_story_book(layer.get("story_book"))["chapters"]}
    setting = (_settings_by_id(layer).get((chapters.get(act["chapter_id"]) or {}).get("setting_id", "")) or {}).get("name", "")
    instruction = story_act_summary_instruction(
        act_title=act["title"], setting_name=setting, scenes_text="\n".join(lines), user_text=act_user_text(act),
        cast_text=cast_lines(payload), old_summary=act["summary"],
    )
    raw, run_info = _run(payload, instruction, label="Story Mode Act Description", max_new_tokens=700,
                         temperature=0.5, json_schema=STORY_ACT_SUMMARY_SCHEMA)
    summary = _clean_scene_text(_extract_summary(raw), 2000)
    if not summary:
        raise ValueError("The LLM did not return a description. Try again.")
    updated = [{**item, "summary": summary} if item["id"] == act_id else item for item in acts]
    return {
        "story_layer": {"story_acts": updated},
        "summary": summary,
        "runner": run_info.get("runner", ""),
        "used_model": run_info.get("used_model", ""),
    }


def _extract_summary(raw):
    """The ``summary`` text from a JSON reply, or the plain reply when the model skipped the JSON."""
    data = _extract_json_object_from_text(raw)
    if isinstance(data, dict) and str(data.get("summary") or "").strip():
        return str(data["summary"])
    return "" if str(raw or "").lstrip().startswith("{") else str(raw or "")
