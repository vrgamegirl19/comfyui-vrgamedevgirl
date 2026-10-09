"""Keep the Storyboard Builder's saved copy (``storyboard/storyboard.json``) in step with the Video Builder session.

``scene_cards`` is the Python twin of ``storyboardScenePayload`` (``web/music_video_builder/scene_output.mjs``):
the scene cards the Builder hands the Storyboard, built from the timeline. ``merge_storyboard`` is the twin of
the Storyboard's ``loadExisting`` merge (``web/storyboard_builder/persistence.mjs``): linked cards take their
timeline-owned fields from the session and keep everything else (Summary, triggers, image, speaker plan),
Storyboard-only cards stay, and cards the user deleted from the Storyboard stay deleted.

Pure module: no ComfyUI imports.
"""

import copy
from typing import Any, Dict, Iterable, List, Optional

from ..builder.lyric_scenes import is_instrumental_lyric_text
from .scene_card_fields import VIDEO_PROMPT_TYPES

# Card fields the open Storyboard takes from the live timeline scene (``liveOwned`` and ``sharedFields`` in
# ``loadExisting``, plus timing and lyrics). The saved card keeps every other field.
LINKED_CARD_KEYS = (
    "scene_number", "label", "lyrics", "lyric_section", "story_beat", "timeline_start", "timeline_end",
    "exact_duration", "project_video_engine", "minimax_h3_mode", "performance_mode", "lyric_singers",
    "lyric_no_lip_sync",
    "lyric_instrumental", "no_character_present", "performance_style", "facial_performance",
    "facial_performance_custom", "shot_type", "camera_motion", "character_motion", "image_prompt",
    "video_prompt", "video_prompt_origin", "minimax_h3_pass2_prompt", "notes", "timeline_note",
    "motion_summary", "audio_direction", "continuity", "flf_start_state", "flf_transformation",
    "flf_end_state", "flf_carry_forward", "video_style", "video_style_custom",
    "temporal_world_effect_override", "temporal_world_effect_custom",
)

# Scene default keys saved by the Builder (``builder_storyboard_defaults``) and their Storyboard names.
DEFAULT_KEY_MAP = {
    "global_consistency_phrase": "global_consistency_phrase",
    "camera_motion_speed": "camera_motion_speed",
    "character_motion_speed": "character_motion_speed",
    "minimax_h3_cut_frequency": "minimax_h3_cut_frequency",
    "performance_style": "performance_style_default",
    "short_film_planning_mode": "short_film_planning_mode",
    "camera_flow": "camera_flow",
    "custom_camera_flow_sequence": "custom_camera_flow_sequence",
    "image_shot_flow": "image_shot_flow",
    "image_aesthetic": "image_aesthetic",
    "video_style": "video_style",
    "video_style_custom": "video_style_custom",
    "temporal_world_effect": "temporal_world_effect",
    "temporal_world_effect_custom": "temporal_world_effect_custom",
    "temporal_allow_background_extras": "temporal_allow_background_extras",
    "temporal_background_intensity": "temporal_background_intensity",
    "temporal_environment_time_passage": "temporal_environment_time_passage",
    "temporal_protected_characters": "temporal_protected_characters",
    "temporal_protected_custom": "temporal_protected_custom",
    "fx_preset": "fx_preset",
    "fx_custom_json": "fx_custom_json",
    "story_arc_detail": "story_arc_detail",
}


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _first_present(source: Dict[str, Any], *keys: str) -> Any:
    """JavaScript ``a ?? b``: the first key whose value is not missing or null."""
    for key in keys:
        if source.get(key) is not None:
            return source[key]
    return None


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def video_engine(session: Dict[str, Any]) -> str:
    return "minimax_h3" if _text(session.get("video_engine")).lower() == "minimax_h3" else "ltx"


def image_mode(session: Dict[str, Any]) -> str:
    mode = session.get("image_model_mode") or _dict(session.get("flux_klein_settings")).get("image_model_mode")
    return _text(mode) or "zimage"


def _slim_reference(ref: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(ref, dict):
        return None
    image = _dict(ref.get("image"))
    return {
        "id": _text(ref.get("id")),
        "name": _text(ref.get("name")),
        "description": _text(ref.get("description")),
        "minimax_voice": ref.get("minimax_voice") if isinstance(ref.get("minimax_voice"), dict) else {},
        "trigger_phrase": _text(ref.get("trigger_phrase")),
        "trigger_position": "end" if _text(ref.get("trigger_position")) == "end" else "start",
        "image": {"path": _text(image.get("path")), "name": _text(image.get("name")), "data": ""},
    }


def _id_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    return [part.strip() for part in _text(value).split(",") if part.strip()]


def storyboard_prompt_for_segment(segment: Dict[str, Any], mode: str) -> str:
    """Twin of ``storyboardPromptForSegment``: the image prompt the card shows for the project's image model."""
    candidates = [
        segment.get("flow_gpt_prompt") if mode == "flow_gpt" else "",
        segment.get("flux_klein_prompt") if mode == "flux_klein" else "",
        segment.get("nb_prompt") if mode == "nano_banana" else "",
        segment.get("ernie_t2i_prompt") if mode == "ernie_image" else "",
        segment.get("flow_gpt_prompt"), segment.get("t2i_prompt"), segment.get("flux_klein_prompt"),
        segment.get("nb_prompt"), segment.get("ernie_t2i_prompt"),
    ]
    return next((_text(item) for item in candidates if _text(item)), "")


def _performance_mode(session: Dict[str, Any], segment: Dict[str, Any]) -> str:
    """Twin of ``effectiveVideoPerformanceModeForSegment``: no-lip-sync scenes, else the project's video type."""
    if segment.get("lyric_no_lip_sync"):
        return "no_lip_sync"
    raw = _text(session.get("video_type") or session.get("videoType")).lower().replace("-", "_").replace(" ", "_")
    return "speaking" if raw in {"speaking", "short_film", "dialogue", "dialog"} else "singing"


def scene_cards(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One Storyboard scene card per timeline scene, in timeline order, with every editable field."""
    refs = _dict(session.get("flux_reference_builder"))
    subjects = {_text(s.get("id")): s for s in refs.get("subjects") or [] if isinstance(s, dict)}
    locations = {_text(loc.get("id")): loc for loc in refs.get("locations") or [] if isinstance(loc, dict)}
    subject_map = _dict(refs.get("subject_scene_map"))
    location_map = _dict(refs.get("scene_map"))
    locations_cleared = bool(refs.get("locations_cleared"))
    defaults = _dict(session.get("builder_storyboard_defaults"))
    engine = video_engine(session)
    mode = image_mode(session)
    minimax = engine == "minimax_h3"
    segments = [s for s in session.get("segments") or [] if isinstance(s, dict)]
    segments.sort(key=lambda s: float(s.get("start") or 0))
    cards: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        no_character = bool(segment.get("no_character_present"))
        mapped = [] if no_character else [
            _slim_reference(subjects[i]) for i in _id_list(subject_map.get(segment.get("id"))) if i in subjects
        ]
        location_id = _text(location_map.get(segment.get("id")))
        location = None if locations_cleared else _slim_reference(locations.get(location_id))
        start, end = float(segment.get("start") or 0), float(segment.get("end") or 0)
        lyric = _text(segment.get("lyric_text") or segment.get("lyric_note") or segment.get("lyrics"))
        video_notes = _text(_first_present(segment, "i2v_notes", "video_notes"))
        scene_notes = _text(_first_present(segment, "notes", "director_note"))
        cue_map = segment.get("lyric_cue_map")
        explicit_singers = segment.get("lyric_singers") if isinstance(segment.get("lyric_singers"), list) else []
        singers = [] if no_character else (
            [_text(name) for name in explicit_singers if _text(name)] or [r["name"] for r in mapped if r and r["name"]]
        )
        if minimax:
            raw_prompt = segment.get("minimax_h3_prompt")
        else:
            raw_prompt = segment.get("i2v_prompt") or segment.get("t2v_prompt")
        video_prompt = _text(raw_prompt)
        origin = _text(segment.get("minimax_h3_prompt_origin") if minimax else segment.get("i2v_prompt_origin")).lower()
        stored_type = _text(segment.get("video_prompt_type"))
        summary_parts = [scene_notes, _text(segment.get("timeline_note")), video_notes,
                         _text((location or {}).get("description")),
                         "" if segment.get("lyric_no_lip_sync") else lyric]
        cards.append({
            "id": segment.get("id") or f"scene_{index + 1}",
            "scene_number": index + 1,
            "label": _text(segment.get("label")) or f"Scene {index + 1}",
            "lyrics": lyric,
            "lyric_section": _text(segment.get("lyric_section")),
            "story_beat": _text(segment.get("story_beat")),
            "flf_start_state": _text(segment.get("flf_start_state")),
            "flf_transformation": _text(segment.get("flf_transformation")),
            "flf_end_state": _text(segment.get("flf_end_state")),
            "flf_carry_forward": _text(segment.get("flf_carry_forward")),
            "performance_mode": _performance_mode(session, segment),
            "prompt_summary": next((part for part in summary_parts if part), ""),
            "motion_summary": video_notes,
            "lyric_singers": singers,
            "lyric_cue_map": copy.deepcopy(cue_map) if isinstance(cue_map, list) else [],
            "lyric_shot_word_timing_enabled": bool(segment.get("lyric_shot_word_timing_enabled")),
            "lyric_performance_mode": _text(segment.get("lyric_performance_mode")),
            "speaker_assignments": copy.deepcopy(segment.get("minimax_speaker_assignments"))
            if isinstance(segment.get("minimax_speaker_assignments"), list) else [],
            "lyric_no_lip_sync": bool(segment.get("lyric_no_lip_sync")),
            "lyric_instrumental": is_instrumental_lyric_text(lyric),
            "no_character_present": no_character,
            "subjects": [r["name"] for r in mapped if r],
            "subject_refs": [r for r in mapped if r],
            "setting": _text((location or {}).get("description") or (location or {}).get("name")),
            "location_ref": location,
            "timeline_start": start,
            "timeline_end": end,
            "exact_duration": max(0.0, end - start),
            "shot_type": _text(segment.get("shot_type")),
            "camera_motion": _text(segment.get("camera_motion") or segment.get("motion_preset")),
            "character_motion": _text(segment.get("character_motion")),
            "include_microphone": bool(segment.get("include_microphone")),
            "performance_style": _text(segment.get("performance_style") or defaults.get("performance_style")),
            "facial_performance": _text(segment.get("facial_performance")),
            "facial_performance_custom": _text(segment.get("facial_performance_custom")),
            "project_video_engine": engine,
            "minimax_h3_mode": _text(segment.get("minimax_h3_mode")) or ("reference_to_video" if minimax else ""),
            "video_style": _text(segment.get("minimax_h3_video_style") or defaults.get("video_style")),
            "video_style_custom": _text(
                segment.get("minimax_h3_video_style_custom") or defaults.get("video_style_custom")
            ),
            "temporal_world_effect_override": _text(segment.get("temporal_world_effect_override")) or "global",
            "temporal_world_effect_custom": _text(segment.get("temporal_world_effect_custom")),
            "video_prompt_type": stored_type if stored_type in VIDEO_PROMPT_TYPES else ("rtv" if minimax else "i2v"),
            "image_prompt": storyboard_prompt_for_segment(segment, mode),
            "video_prompt": video_prompt,
            # Like the old API sync: a saved prompt with no recorded origin was written by the LLM.
            "video_prompt_origin": "gemma" if origin == "gemma" or (video_prompt and not origin) else "manual",
            "minimax_h3_pass2_prompt": str(segment.get("minimax_h3_pass2_prompt") or ""),
            "image_path": _text(segment.get("approved_image_path") or segment.get("image_path")),
            "notes": scene_notes,
            "timeline_note": str(segment.get("timeline_note") or ""),
            "audio_direction": _text(segment.get("audio_direction")),
            "continuity": _text(segment.get("continuity")),
            "extra_subjects": [],
        })
    return cards


def scene_card_context(card: Dict[str, Any]) -> Dict[str, Any]:
    """Twin of ``storyboardSceneCardContext`` (``web/storyboard_builder/scenes.mjs``): the complete card for an LLM.

    Inline image data is replaced by ``has_inline_image``; pictures reach the LLM through its vision inputs.
    """
    context = copy.deepcopy(card)
    image_data = context.pop("image_data", "")
    context["has_inline_image"] = bool(image_data)

    def describe(ref: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(ref, dict):
            return None
        image = _dict(ref.get("image"))
        return {**ref, "image": {"path": _text(image.get("path")), "name": _text(image.get("name")),
                                 "has_inline_image": bool(image.get("data"))}}

    context["subject_refs"] = [describe(ref) for ref in context.get("subject_refs") or [] if isinstance(ref, dict)]
    context["location_ref"] = describe(context.get("location_ref"))
    return context


def video_prompt_status(card: Dict[str, Any], previous: str = "") -> str:
    """The Storyboard's green/red status after the card's video prompt is set or cleared."""
    if _text(card.get("video_prompt")):
        return "video_prompt_ready"
    return "draft" if previous in ("", "video_prompt_ready") else previous


def _linked_card(saved: Dict[str, Any], fresh: Dict[str, Any]) -> Dict[str, Any]:
    """A saved card updated with the timeline-owned fields of its live scene (``loadExisting``)."""
    merged = copy.deepcopy(saved)
    for key in LINKED_CARD_KEYS:
        if key in fresh:
            merged[key] = copy.deepcopy(fresh[key])
    # loadExisting keeps the saved card's value when the timeline has none for these.
    for key in ("shot_type", "camera_motion", "character_motion", "minimax_h3_mode"):
        if not fresh.get(key) and saved.get(key):
            merged[key] = saved[key]
    merged["id"] = fresh.get("id") or saved.get("id")
    saved_refs = {_text(ref.get("id")): ref for ref in saved.get("subject_refs") or [] if isinstance(ref, dict)}
    merged["subject_refs"] = [] if fresh.get("no_character_present") else [
        copy.deepcopy(saved_refs.get(_text(ref.get("id"))) or ref) for ref in fresh.get("subject_refs") or []
    ]
    names = [ref.get("name") for ref in merged["subject_refs"] if ref.get("name")]
    merged["subjects"] = names or list(fresh.get("subjects") or [])
    saved_location = saved.get("location_ref") if isinstance(saved.get("location_ref"), dict) else None
    fresh_location = fresh.get("location_ref") if isinstance(fresh.get("location_ref"), dict) else None
    if _text((fresh_location or {}).get("id")) != _text((saved_location or {}).get("id")):
        merged["location_ref"] = copy.deepcopy(fresh_location)
        merged["setting"] = fresh.get("setting", "")
    merged["status"] = video_prompt_status(merged, _text(saved.get("status")))
    return merged


def merge_scene_cards(saved: Optional[Dict[str, Any]], cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The Storyboard's cards after opening it on the current timeline (``loadExisting``)."""
    saved_scenes = [s for s in (saved or {}).get("scenes") or [] if isinstance(s, dict)]
    if not saved_scenes and not isinstance((saved or {}).get("source_scene_ids"), list):
        return [{**copy.deepcopy(card), "status": video_prompt_status(card)} for card in cards]
    incoming = {card["id"]: card for card in cards}
    saved_ids = {s.get("id") for s in saved_scenes}
    raw_source_ids = (saved or {}).get("source_scene_ids")
    source_ids = set(str(i) for i in raw_source_ids) if isinstance(raw_source_ids, list) else None
    if source_ids is not None:
        order = [s for s in saved_scenes if not cards or s.get("id") not in source_ids or s.get("id") in incoming]
        order += [card for card in cards if card["id"] not in source_ids and card["id"] not in saved_ids]
    else:
        order = list(cards) if cards else saved_scenes
    by_number = {int(s.get("scene_number") or 0): s for s in saved_scenes}
    saved_by_id = {s.get("id"): s for s in saved_scenes}
    merged: List[Dict[str, Any]] = []
    for item in order:
        fresh = incoming.get(item.get("id"))
        if fresh is None:  # a Storyboard-only card
            merged.append(copy.deepcopy(item))
            continue
        saved_card = saved_by_id.get(fresh["id"])
        if saved_card is None and source_ids is None:
            saved_card = by_number.get(int(fresh.get("scene_number") or 0))
        if saved_card:
            merged.append(_linked_card(saved_card, fresh))
        else:
            merged.append({**copy.deepcopy(fresh), "status": video_prompt_status(fresh)})
    return merged


def _merge_references(saved_items: Iterable[Any], session_items: Iterable[Any]) -> List[Dict[str, Any]]:
    saved_by_id = {_text(item.get("id")): item for item in saved_items or [] if isinstance(item, dict)}
    merged = []
    for item in session_items or []:
        slim = _slim_reference(item)
        if not slim:
            continue
        merged.append({**copy.deepcopy(saved_by_id.get(slim["id"], {})), **slim})
    return merged


def merge_storyboard(saved: Optional[Dict[str, Any]], session: Dict[str, Any],
                     cards: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The saved storyboard brought up to date with the session, keeping Storyboard-only data."""
    cards = scene_cards(session) if cards is None else cards
    storyboard = copy.deepcopy(saved) if saved else {
        "project_video_engine": video_engine(session),
        "mode": "image_to_video_prep",
        "performance_mode": "speaking" if _performance_mode(session, {}) == "speaking" else "singing",
    }
    storyboard.pop("path", None)
    storyboard.pop("exists", None)
    storyboard["project_video_engine"] = video_engine(session)
    storyboard["scenes"] = merge_scene_cards(saved, cards)
    previous_ids = storyboard.get("source_scene_ids") if isinstance(storyboard.get("source_scene_ids"), list) else []
    known_ids = [str(i) for i in previous_ids] + [str(card["id"]) for card in cards]
    storyboard["source_scene_ids"] = list(dict.fromkeys(known_ids))
    defaults = _dict(session.get("builder_storyboard_defaults"))
    for session_key, storyboard_key in DEFAULT_KEY_MAP.items():
        if session_key in defaults:
            storyboard[storyboard_key] = copy.deepcopy(defaults[session_key])
    layer = session.get("builder_story_layer") or session.get("builderStoryLayer")
    if isinstance(layer, dict):
        storyboard["story_layer"] = {**_dict(storyboard.get("story_layer")), **copy.deepcopy(layer)}
    refs = _dict(session.get("flux_reference_builder"))
    catalog = _dict(storyboard.get("reference_builder"))
    if refs.get("subjects") or not catalog.get("subjects"):
        catalog["subjects"] = _merge_references(catalog.get("subjects"), refs.get("subjects"))
    if refs.get("locations_cleared"):
        catalog["locations"] = []
        catalog["locations_cleared"] = True
    elif refs.get("locations") or not catalog.get("locations"):
        catalog["locations"] = _merge_references(catalog.get("locations"), refs.get("locations"))
    storyboard["reference_builder"] = catalog
    return storyboard
