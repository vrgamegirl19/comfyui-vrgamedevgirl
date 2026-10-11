"""Story Mode locations: a look for each chapter setting and one location card per spot, mapped to its scenes.

Built only from the story (book, settings, scene plan) and the style/theme notes. The lyrics are never sent.
``apply_story_locations`` merges the cards into the Reference Builder data (``flux_reference_builder``); its JS twin is
``web/music_video_builder/story_locations_data.mjs`` (checked against ``tests/story_locations_cases.json``).
"""

import copy

from ..llm.output_checks import _extract_json_object_from_text
from ..llm.prompts.story_mode import (
    STORY_SETTING_LOOKS_SCHEMA,
    STORY_SPOTS_SCHEMA,
    story_setting_looks_instruction,
    story_spots_instruction,
)
from .scene_helpers import _clean_scene_text, _normalize_story_layer
from .story_book import _run, cast_lines
from .story_mode import normalize_story_book, normalize_story_settings


def card_name(setting_name, spot):
    """``Home - Kitchen``; just the setting when the spot is the setting itself."""
    setting_name, spot = str(setting_name or "").strip(), str(spot or "").strip()
    if not spot or spot.lower() == setting_name.lower():
        return setting_name or spot
    return f"{setting_name} - {spot}" if setting_name else spot


def spot_groups(layer, scene_ids):
    """Settings in story order, each with its spots in first-use order: names, scenes, acts and story slices."""
    layer = layer or {}
    chapters = normalize_story_book(layer.get("story_book"))["chapters"]
    settings = {item["id"]: item for item in normalize_story_settings(layer.get("story_settings"))}
    chapter_setting = {chapter["id"]: chapter["setting_id"] for chapter in chapters}
    act_chapter = {act.get("id"): act.get("chapter_id") for act in layer.get("story_acts") or []}
    plan = {entry.get("scene_id"): entry for entry in layer.get("story_scene_plan") or []}
    groups, by_setting = [], {}
    for scene_id in scene_ids:
        entry = plan.get(scene_id)
        if not entry:
            continue
        setting_id = chapter_setting.get(act_chapter.get(entry.get("act_id")), "")
        setting = settings.get(setting_id) or {"id": setting_id, "name": "", "look": ""}
        group = by_setting.get(setting_id)
        if group is None:
            group = {"setting_id": setting_id, "setting_name": setting["name"], "look": setting["look"], "spots": []}
            by_setting[setting_id] = group
            groups.append(group)
        name = card_name(setting["name"], entry.get("spot"))
        spot = next((item for item in group["spots"] if item["card_name"].lower() == name.lower()), None)
        if spot is None:
            spot = {"name": str(entry.get("spot") or setting["name"]).strip(), "card_name": name,
                    "scene_ids": [], "act_ids": [], "slices": []}
            group["spots"].append(spot)
        spot["scene_ids"].append(scene_id)
        if entry.get("act_id") and entry["act_id"] not in spot["act_ids"]:
            spot["act_ids"].append(entry["act_id"])
        if entry.get("story_slice") and len(spot["slices"]) < 6:
            spot["slices"].append(entry["story_slice"])
    return groups


def _parse(raw, key):
    try:
        data = _extract_json_object_from_text(raw) or {}
    except Exception:
        return []
    items = data.get(key) if isinstance(data, dict) else None
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def build_story_locations(payload, progress=None):
    """Looks for settings without one (all with ``regenerate_looks``) and a card for every spot in the scene plan."""
    layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    scenes = payload.get("scenes") if isinstance(payload.get("scenes"), list) else []
    scene_ids = [str(scene.get("id") or f"scene_{i + 1}") for i, scene in enumerate(scenes) if isinstance(scene, dict)]
    groups = spot_groups(layer, scene_ids)
    if not groups:
        raise ValueError("Create the scene plan first; locations are built from its spots.")
    style_theme = _clean_scene_text(payload.get("style_theme") or "", 3000)
    settings = normalize_story_settings(layer.get("story_settings"))
    need_looks = [g for g in groups if g["setting_name"] and (payload.get("regenerate_looks") or not g["look"])]
    run_info = {}
    total = len(groups) + (1 if need_looks else 0)
    if need_looks:
        if progress:
            progress(0, total, "setting looks")
        book = normalize_story_book(layer.get("story_book"))
        summary = "\n".join(
            f"Chapter {i + 1} ({chapter['title']}): " + " ".join(chapter["paragraphs"])[:600]
            for i, chapter in enumerate(book["chapters"])
        )
        instruction = story_setting_looks_instruction(
            settings_text="\n".join(f"- {g['setting_name']}" for g in need_looks),
            book_summary=summary[:12000],
            style_theme=style_theme,
            cast_text=cast_lines(payload),
        )
        try:
            raw, run_info = _run(payload, instruction, label="Story Mode Setting Looks", max_new_tokens=1600,
                                 temperature=0.5, json_schema=STORY_SETTING_LOOKS_SCHEMA)
            looks = {str(item.get("name") or "").strip().lower(): _clean_scene_text(item.get("look") or "", 2000)
                     for item in _parse(raw, "settings")}
        except Exception as error:
            print(f"[VRGDG Story Mode] Setting looks failed, spots are written without them: {error}")
            looks = {}
        for group in need_looks:
            look = looks.get(group["setting_name"].lower())
            if look:
                group["look"] = look
        settings = [
            {**item, "look": next((g["look"] for g in groups if g["setting_id"] == item["id"] and g["look"]), item["look"])}
            for item in settings
        ]
    locations = []
    for step, group in enumerate(groups):
        if progress:
            progress(step + (1 if need_looks else 0), total, group["setting_name"] or "setting")
        spots_text = "\n".join(
            f"- {spot['name']}: " + (" ".join(spot["slices"]) or "[no story yet]") for spot in group["spots"]
        )
        descriptions = {}
        try:
            raw, run_info = _run(
                payload,
                story_spots_instruction(setting_name=group["setting_name"], setting_look=group["look"],
                                        spots_text=spots_text[:8000], style_theme=style_theme),
                label="Story Mode Spots", max_new_tokens=2000, temperature=0.5, json_schema=STORY_SPOTS_SCHEMA,
            )
            descriptions = {str(item.get("name") or "").strip().lower(): _clean_scene_text(item.get("description") or "", 800)
                            for item in _parse(raw, "spots")}
        except Exception as error:
            print(f"[VRGDG Story Mode] Spot descriptions for {group['setting_name']} failed: {error}")
        for spot in group["spots"]:
            description = descriptions.get(spot["name"].lower()) or descriptions.get(spot["card_name"].lower()) or ""
            locations.append({
                "name": spot["card_name"],
                "description": description or group["look"],
                "story_setting_id": group["setting_id"],
                "story_act_ids": list(spot["act_ids"]),
                "scene_ids": list(spot["scene_ids"]),
            })
    if progress:
        progress(total, total, "done")
    return {
        "story_layer": {"story_settings": settings},
        "locations": locations,
        "runner": run_info.get("runner", ""),
        "used_model": run_info.get("used_model", ""),
    }


def apply_story_locations(refs, locations, new_id, replace_descriptions=False):
    """Reference Builder data with the story's location cards merged in and every planned scene mapped.

    A card with the same name is reused: its description is only replaced when it is empty (or with
    ``replace_descriptions``) and its image is never touched. ``new_id(taken)`` makes an id for a new card.
    """
    result = copy.deepcopy(refs if isinstance(refs, dict) else {})
    cards = result.setdefault("locations", [])
    by_name = {str(card.get("name") or "").strip().lower(): card for card in cards if isinstance(card, dict)}
    taken = {card.get("id") for card in cards if isinstance(card, dict)}
    scene_map = result.get("scene_map") if isinstance(result.get("scene_map"), dict) else {}
    result["scene_map"] = scene_map
    added, updated = 0, 0
    for item in locations:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        card = by_name.get(name.lower())
        if card is None:
            card = {"id": new_id(taken), "name": name, "description": item.get("description") or "",
                    "image": {"path": "", "data": "", "name": ""}}
            cards.append(card)
            by_name[name.lower()] = card
            added += 1
        elif item.get("description") and (replace_descriptions or not str(card.get("description") or "").strip()):
            card["description"] = item["description"]
            updated += 1
        card["story_setting_id"] = item.get("story_setting_id") or ""
        card["story_act_ids"] = list(item.get("story_act_ids") or [])
        for scene_id in item.get("scene_ids") or []:
            scene_map[scene_id] = card["id"]
    result["locations_cleared"] = False
    result["use_location_references"] = bool(cards)
    return result, {"added": added, "updated": updated}
