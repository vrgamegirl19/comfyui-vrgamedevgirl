"""Story Mode: change the story from one scene (Update Story From This Scene).

The user sets a scene's characters and writes what should happen in it. One LLM step rewrites that scene's plan and
adapts its act and chapter to agree, as little as possible, and names the other scenes of the act that no longer fit.
Takes the usual Story Mode payload plus ``scene_id`` and returns the changed story layer fields. Saving is the
caller's job.
"""

from ..llm.output_checks import _extract_json_object_from_text
from ..llm.prompts.story_mode import STORY_REVISE_SCENE_SCHEMA, story_mode_lyric_guidance, story_revise_scene_instruction
from .scene_helpers import _clean_scene_text
from .story_book import _layer, _run, _runner_fields, _scenes, cast_lines, story_cast
from .story_mode import (
    normalize_story_acts,
    normalize_story_book,
    normalize_story_scene_plan,
    normalize_story_settings,
    story_book_to_text,
)


def _scene_id(scene, index):
    return str(scene.get("id") or f"scene_{index + 1}")


def revise_story_from_scene(payload):
    """Rewrite one scene's plan from the user's change and adapt its act and chapter to it."""
    layer = _layer(payload)
    scene_id = _clean_scene_text(payload.get("scene_id") or "", 160)
    scene_ids = [_scene_id(scene, index) for index, scene in enumerate(_scenes(payload))]
    number_by_id = {sid: index + 1 for index, sid in enumerate(scene_ids)}
    plan = normalize_story_scene_plan(layer.get("story_scene_plan"))
    entry = next((item for item in plan if item["scene_id"] == scene_id), None)
    if not entry or scene_id not in number_by_id:
        raise ValueError("Plan this scene first; the story is updated from a planned scene.")
    acts = normalize_story_acts(layer.get("story_acts"))
    act = next((item for item in acts if item["id"] == entry["act_id"]), None)
    book = normalize_story_book(layer.get("story_book"))
    chapter_index = next((i for i, chapter in enumerate(book["chapters"]) if act and chapter["id"] == act["chapter_id"]), -1)
    if not act or chapter_index < 0:
        raise ValueError("This scene's act or chapter is missing. Rebuild the acts first.")
    chapter = book["chapters"][chapter_index]
    settings = normalize_story_settings(layer.get("story_settings"))
    setting_name = next((item["name"] for item in settings if item["id"] == chapter["setting_id"]), "")
    names_by_id = {subject["id"]: subject["name"] for subject in story_cast(payload) if subject.get("id")}
    ids_by_name = {name.lower(): cid for cid, name in names_by_id.items()}
    by_scene = {item["scene_id"]: item for item in plan}
    act_lines = []
    for sid in sorted(act["scene_ids"], key=lambda value: number_by_id.get(value, 0)):
        other = by_scene.get(sid) or {}
        cast = ", ".join(names_by_id.get(cid, cid) for cid in other.get("cast_ids") or []) or "as planned"
        act_lines.append(f"Scene {number_by_id.get(sid, '?')} ({other.get('spot') or setting_name}): "
                         f"{other.get('story_slice') or '[no slice yet]'} [characters: {cast}]")
    instruction = story_revise_scene_instruction(
        book_text=story_book_to_text(book, settings),
        chapter_number=chapter_index + 1,
        chapter_title=chapter["title"],
        setting_name=setting_name,
        chapter_text="\n\n".join(chapter["paragraphs"]),
        act_title=act["title"],
        act_summary=act["summary"],
        act_scenes_text="\n".join(act_lines),
        scene_number=number_by_id[scene_id],
        scene_cast=", ".join(names_by_id.get(cid, cid) for cid in entry["cast_ids"]),
        user_note=entry["user_note"],
        cast_text=cast_lines(payload),
        lyric_guidance=story_mode_lyric_guidance(layer.get("lyric_story_strength", 5)),
    )
    raw, run_info = _run(payload, instruction, label="Story Mode Revise Scene", max_new_tokens=3200,
                         temperature=0.5, json_schema=STORY_REVISE_SCENE_SCHEMA)
    reply = _extract_json_object_from_text(raw) or {}
    scene = reply.get("scene") if isinstance(reply.get("scene"), dict) else {}
    if not _clean_scene_text(scene.get("story_slice") or "", 1200):
        raise ValueError("The LLM did not return the changed scene. Try again.")

    # The user's characters and direction are kept exactly; the LLM fills in the rest of the scene.
    performer = ids_by_name.get(str(scene.get("performer") or "").strip().lower(), "")
    entry.update({
        "story_slice": _clean_scene_text(scene.get("story_slice"), 1200),
        "spot": _clean_scene_text(scene.get("spot") or "", 160) or entry["spot"],
        "mood": _clean_scene_text(scene.get("mood") or "", 200) or entry["mood"],
        "performer_id": performer if performer in entry["cast_ids"] else "",
        "interaction": _clean_scene_text(scene.get("interaction") or "", 600) if len(entry["cast_ids"]) > 1 else "",
    })
    summary = _clean_scene_text(reply.get("act_summary") or "", 2000)
    if summary:
        act["summary"] = summary
    paragraphs = [_clean_scene_text(text, 4000) for text in reply.get("chapter_paragraphs") or [] if isinstance(text, str)]
    paragraphs = [text for text in paragraphs if text]
    if paragraphs:
        chapter["paragraphs"] = paragraphs
    others = []
    for item in reply.get("other_scenes") or []:
        try:
            number = int(item.get("scene"))
        except (AttributeError, TypeError, ValueError):
            continue
        other_id = scene_ids[number - 1] if 1 <= number <= len(scene_ids) else ""
        if other_id and other_id != scene_id and other_id in act["scene_ids"] and other_id not in [o["scene_id"] for o in others]:
            others.append({"scene_id": other_id, "reason": _clean_scene_text(item.get("reason") or "", 300)})
    return {
        "story_layer": {
            "story_scene_plan": plan,
            "story_acts": acts,
            "story_book": book,
            "story_book_text": story_book_to_text(book, settings),
        },
        "scene_id": scene_id,
        "act_id": act["id"],
        "chapter_id": chapter["id"],
        "other_scenes": others,
        **_runner_fields(run_info),
    }
