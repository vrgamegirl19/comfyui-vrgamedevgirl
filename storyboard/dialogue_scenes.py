import json
import re

from ..llm.prompts.storyboard import (
    _id_lora_structured_image_prompt,
    _storyboard_dialogue_json_repair_instruction,
    _storyboard_dialogue_plan_schema,
    _storyboard_dialogue_planner_instruction,
)
from .persistence import _normalize_storyboard_scene
from .scene_helpers import _clean_scene_text, _normalize_story_layer, _storyboard_dialogue_reference_catalog
from .scene_prompts import _camera_motion_for_storyboard_speed
from .story_layer import _authoritative_script_from_payload, _authoritative_script_text


def _normalize_generated_dialogue_scenes(raw_scenes, subjects, locations):
    if not isinstance(raw_scenes, list):
        raise ValueError("Gemma dialogue plan did not include a scenes array.")
    subject_ids = {str(item.get("id") or "") for item in subjects if str(item.get("id") or "")}
    location_ids = {str(item.get("id") or "") for item in locations if str(item.get("id") or "")}
    scenes = []
    for index, item in enumerate(raw_scenes[:80], start=1):
        if not isinstance(item, dict):
            continue
        subject_id = _clean_scene_text(item.get("character_id") or item.get("subject_id") or item.get("speaker_id") or "", 180)
        location_id = _clean_scene_text(item.get("location_id") or "", 180)
        if subject_id and subject_ids and subject_id not in subject_ids:
            subject_id = ""
        if location_id and location_ids and location_id not in location_ids:
            location_id = ""
        subject_refs = []
        if subject_id:
            subject = next((entry for entry in subjects if entry.get("id") == subject_id), None)
            if subject:
                subject_refs = [{
                    "id": subject.get("id", ""),
                    "name": subject.get("name", ""),
                    "description": subject.get("description", ""),
                    "reference_type": subject.get("reference_type", "character"),
                    "image": {**(subject.get("image") or {})},
                }]
        location_ref = None
        if location_id:
            location = next((entry for entry in locations if entry.get("id") == location_id), None)
            if location:
                location_ref = {
                    "id": location.get("id", ""),
                    "name": location.get("name", ""),
                    "description": location.get("description", ""),
                    "image": {**(location.get("image") or {})},
                }
        subject_for_prompt = subject_refs[0] if subject_refs else None
        dialogue = _clean_scene_text(item.get("dialogue") or item.get("line") or item.get("lyrics") or "", 1200)
        label = _clean_scene_text(item.get("label") or item.get("title") or f"Scene {index}", 160)
        scene = _normalize_storyboard_scene({
            "id": _clean_scene_text(item.get("id") or f"id_lora_story_scene_{index}", 160),
            "scene_number": index,
            "label": label or f"Scene {index}",
            "lyrics": dialogue,
            "lyric_singers": [_clean_scene_text(item.get("character_name") or item.get("speaker") or "", 160)] if not subject_refs else [subject_refs[0].get("name", "")],
            "story_beat": _clean_scene_text(item.get("story_beat") or item.get("beat") or "", 1800),
            "prompt_summary": _clean_scene_text(item.get("visual_direction") or item.get("summary") or "", 1800),
            "motion_summary": _clean_scene_text(item.get("motion_summary") or item.get("video_notes") or item.get("camera_motion") or "", 1400),
            "subjects": [subject_refs[0].get("name", "")] if subject_refs else [],
            "subject_refs": subject_refs,
            "setting": _clean_scene_text(item.get("setting") or item.get("location_name") or (location_ref or {}).get("name", ""), 1000),
            "location_ref": location_ref,
            "video_prompt_type": "id_lora",
            "performance_mode": "speaking",
            "shot_type": _clean_scene_text(item.get("shot_type") or "", 160),
            "camera_motion": _clean_scene_text(item.get("camera_motion") or "", 500),
            "facial_performance": _clean_scene_text(item.get("facial_performance") or item.get("emotion") or "", 240),
            "facial_performance_custom": _clean_scene_text(item.get("facial_performance_custom") or item.get("delivery") or "", 800),
            "image_prompt": _id_lora_structured_image_prompt(item, subject_for_prompt, location_ref),
        }, index)
        scene["id_lora_character_id"] = subject_id
        scene["id_lora_location_id"] = location_id
        scenes.append(scene)
    if not scenes:
        raise ValueError("Gemma returned no usable dialogue scenes.")
    return scenes


_MINIMAX_DIALOGUE_NON_INWARD_CAMERA_SEQUENCE = (
    "quiet handheld hold",
    "subtle lateral drift",
    "slow orbit left",
    "gentle pull-back",
    "restrained pan right",
    "rack focus between the speakers",
    "slow orbit right",
    "locked-off reaction hold",
)


def _minimax_camera_motion_family(value):
    text = _clean_scene_text(value or "", 500).lower()
    if re.search(r"\b(push(?:es)?[ -]?in|doll(?:y|ies)[ -]?in|zoom(?:s)?[ -]?in|track(?:s|ing)?[ -]?(?:in|forward)|drift(?:s|ing)?[ -]?(?:closer|forward))\b", text):
        return "inward"
    if re.search(r"\b(pull(?:s)?[ -]?(?:back|out)|doll(?:y|ies)[ -]?out|zoom(?:s)?[ -]?out|track(?:s|ing)?[ -]?backward)\b", text):
        return "outward"
    if re.search(r"\b(orbit|arc|circle|rotate|rotation)\b", text):
        return "orbit"
    if re.search(r"\b(pan|lateral|side|truck)\b", text):
        return "lateral"
    if re.search(r"\b(rack focus|focus pull)\b", text):
        return "focus"
    if re.search(r"\b(hold|locked|static)\b", text):
        return "hold"
    return "other" if text else ""


def _rebalance_generated_minimax_camera_motion(scenes, camera_flow="balanced", camera_motion_speed=4):
    """Prevent an LLM-planned dialogue sequence from collapsing into repeated push-ins.

    This only runs while new guided MiniMax scene cards are being created. Later manual
    edits remain authoritative. Inward moves are allowed as an accent, but no more than
    once in a rolling six-scene window.
    """
    if not isinstance(scenes, list) or str(camera_flow or "").strip().lower() == "off":
        return scenes
    try:
        speed = max(0, min(10, int(round(float(camera_motion_speed)))))
    except Exception:
        speed = 4
    recent_families = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue
        motion = _camera_motion_for_storyboard_speed(scene.get("camera_motion") or "", speed)
        if motion:
            scene["camera_motion"] = motion
        family = _minimax_camera_motion_family(motion)
        if speed <= 0:
            replacement = "locked-off camera"
        else:
            replacement = _MINIMAX_DIALOGUE_NON_INWARD_CAMERA_SEQUENCE[index % len(_MINIMAX_DIALOGUE_NON_INWARD_CAMERA_SEQUENCE)]
        if not motion or (family == "inward" and "inward" in recent_families[-5:]):
            scene["camera_motion"] = replacement
            family = _minimax_camera_motion_family(replacement)
        recent_families.append(family)
    return scenes


def _normalize_generated_minimax_dialogue_scenes(
    raw_scenes,
    subjects,
    locations,
    minimax_h3_mode="text_to_video",
    camera_flow="balanced",
    camera_motion_speed=4,
):
    if not isinstance(raw_scenes, list):
        raise ValueError("MiniMax dialogue plan did not include a scenes array.")
    subject_by_id = {str(item.get("id") or ""): item for item in subjects if str(item.get("id") or "")}
    location_by_id = {str(item.get("id") or ""): item for item in locations if str(item.get("id") or "")}
    mode = str(minimax_h3_mode or "text_to_video").strip().lower().replace("-", "_").replace(" ", "_")
    if mode not in {"text_to_video", "image_to_video", "reference_to_video", "video_to_video"}:
        mode = "text_to_video"
    scenes = []
    for index, item in enumerate(raw_scenes[:80], start=1):
        if not isinstance(item, dict):
            continue
        raw_cues = item.get("dialogue_cues") if isinstance(item.get("dialogue_cues"), list) else []
        if not raw_cues:
            raw_cues = [{
                "character_id": item.get("character_id") or item.get("subject_id") or item.get("speaker_id") or "",
                "speaker": item.get("character_name") or item.get("speaker") or "",
                "dialogue": item.get("dialogue") or item.get("line") or item.get("lyrics") or "",
            }]
        speaker_assignments = []
        subject_refs = []
        seen_subject_ids = set()
        for cue_index, cue in enumerate(raw_cues[:40], start=1):
            if not isinstance(cue, dict):
                continue
            subject_id = _clean_scene_text(cue.get("character_id") or cue.get("subject_id") or cue.get("speaker_id") or "", 180)
            if subject_id and subject_by_id and subject_id not in subject_by_id:
                subject_id = ""
            subject = subject_by_id.get(subject_id) if subject_id else None
            speaker_name = _clean_scene_text(cue.get("speaker") or cue.get("character_name") or (subject or {}).get("name") or "", 160)
            dialogue = _clean_scene_text(cue.get("dialogue") or cue.get("line") or cue.get("text") or "", 1200)
            if not dialogue:
                continue
            speaker_assignments.append({
                "id": f"minimax_dialogue_{index}_{cue_index}",
                "speaker_id": subject_id,
                "speaker_name": speaker_name or "Speaker",
                "text": dialogue,
            })
            if subject and subject_id not in seen_subject_ids:
                subject_refs.append({
                    "id": subject.get("id", ""),
                    "name": subject.get("name", ""),
                    "description": subject.get("description", ""),
                    "reference_type": subject.get("reference_type", "character"),
                    "image": {**(subject.get("image") or {})},
                })
                seen_subject_ids.add(subject_id)
        for participant_id in item.get("participant_ids") or []:
            participant_id = _clean_scene_text(participant_id, 180)
            participant = subject_by_id.get(participant_id) if participant_id else None
            if not participant or participant_id in seen_subject_ids:
                continue
            subject_refs.append({
                "id": participant.get("id", ""),
                "name": participant.get("name", ""),
                "description": participant.get("description", ""),
                "reference_type": participant.get("reference_type", "character"),
                "image": {**(participant.get("image") or {})},
            })
            seen_subject_ids.add(participant_id)
        location_id = _clean_scene_text(item.get("location_id") or "", 180)
        if location_id and location_by_id and location_id not in location_by_id:
            location_id = ""
        location = location_by_id.get(location_id) if location_id else None
        location_ref = ({
            "id": location.get("id", ""),
            "name": location.get("name", ""),
            "description": location.get("description", ""),
            "image": {**(location.get("image") or {})},
        } if location else None)
        dialogue_lines = [f'{cue["speaker_name"]}: "{cue["text"]}"' for cue in speaker_assignments]
        label = _clean_scene_text(item.get("label") or item.get("title") or f"Scene {index}", 160)
        scene = _normalize_storyboard_scene({
            "id": _clean_scene_text(item.get("id") or f"minimax_story_scene_{index}", 160),
            "scene_number": index,
            "label": label or f"Scene {index}",
            "lyrics": "\n".join(dialogue_lines),
            "lyric_singers": [cue["speaker_name"] for cue in speaker_assignments],
            "speaker_assignments": speaker_assignments,
            "story_beat": _clean_scene_text(item.get("story_beat") or item.get("beat") or "", 1800),
            "prompt_summary": _clean_scene_text(item.get("visual_direction") or item.get("summary") or "", 1800),
            "motion_summary": _clean_scene_text(item.get("motion_summary") or item.get("video_notes") or "", 1400),
            "subjects": [subject.get("name", "") for subject in subject_refs],
            "subject_refs": subject_refs,
            "setting": _clean_scene_text(item.get("setting") or item.get("location_name") or (location_ref or {}).get("name", ""), 1000),
            "location_ref": location_ref,
            "video_prompt_type": "i2v",
            "project_video_engine": "minimax_h3",
            "minimax_h3_mode": mode,
            "minimax_h3_audio_mode": "built_in_audio",
            "performance_mode": "speaking",
            "timeline_start": item.get("timeline_start", 0),
            "timeline_end": item.get("timeline_end", 0),
            "exact_duration": item.get("exact_duration") or item.get("duration") or 0,
            "shot_type": _clean_scene_text(item.get("shot_type") or "", 160),
            "camera_motion": _clean_scene_text(item.get("camera_motion") or "", 500),
            "character_motion": _clean_scene_text(item.get("character_motion") or item.get("action") or "", 500),
            "facial_performance": _clean_scene_text(item.get("facial_performance") or item.get("emotion") or "", 240),
            "facial_performance_custom": _clean_scene_text(item.get("facial_performance_custom") or item.get("delivery") or "", 800),
            "image_prompt": _id_lora_structured_image_prompt(item, subject_refs[0] if subject_refs else None, location_ref),
            "audio_direction": _clean_scene_text(item.get("audio_direction") or "", 4000),
            "continuity": _clean_scene_text(item.get("continuity") or "", 4000),
            "notes": _clean_scene_text(item.get("notes") or "", 4000),
        }, index)
        scenes.append(scene)
    if not scenes:
        raise ValueError("The LLM returned no usable MiniMax dialogue scenes.")
    return _rebalance_generated_minimax_camera_motion(scenes, camera_flow, camera_motion_speed)


def _apply_authoritative_script_plan(raw_scenes, script_import):
    generated = raw_scenes if isinstance(raw_scenes, list) else []
    planned_scenes = ((script_import or {}).get("scene_plan") or {}).get("scenes") or []
    locked_scenes = []
    previous_location_id = ""
    for index, planned in enumerate(planned_scenes):
        generated_scene = dict(generated[index]) if index < len(generated) and isinstance(generated[index], dict) else {}
        exact_cues = []
        for cue in planned.get("speaker_assignments") or []:
            exact_cues.append({
                "character_id": cue.get("speaker_id") or "",
                "speaker_id": cue.get("speaker_id") or "",
                "speaker": cue.get("speaker_name") or cue.get("speaker_alias") or "Speaker",
                "dialogue": cue.get("text") or "",
            })
        generated_scene["label"] = generated_scene.get("label") or planned.get("label") or f"Script Segment {index + 1}"
        generated_scene["dialogue_cues"] = exact_cues
        generated_scene["participant_ids"] = list(planned.get("participant_ids") or [])
        generated_scene["participant_names"] = list(planned.get("participant_names") or [])
        current_location_id = _clean_scene_text(generated_scene.get("location_id") or "", 180)
        if planned.get("continuation_of_previous") and previous_location_id:
            generated_scene["location_id"] = previous_location_id
        elif not planned.get("continuation_of_previous"):
            previous_location_id = current_location_id
        elif current_location_id:
            previous_location_id = current_location_id
        generated_scene["exact_duration"] = float(planned.get("duration_seconds") or 0)
        generated_scene["duration"] = float(planned.get("duration_seconds") or 0)
        generated_scene["timeline_start"] = float(planned.get("timeline_start_seconds") or 0)
        generated_scene["timeline_end"] = float(planned.get("timeline_end_seconds") or 0)
        generated_scene["notes"] = _clean_scene_text(
            "\n".join(filter(None, [
                generated_scene.get("notes") or "",
                f"Authoritative Script Mapper segment {index + 1}. Exact dialogue and order are locked.",
                "Continuation of the previous script segment." if planned.get("continuation_of_previous") else "",
            ])),
            4000,
        )
        locked_scenes.append(generated_scene)
    return locked_scenes


def _build_id_lora_dialogue_scenes(payload):
    planner_profile = str(payload.get("_dialogue_planner_profile") or "id_lora").strip().lower()
    is_minimax = planner_profile == "minimax_short_film"
    authoritative_script = _authoritative_script_from_payload(payload) if is_minimax else None
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    storyboard_settings = payload.get("storyboard") if isinstance(payload.get("storyboard"), dict) else {}
    camera_flow = _clean_scene_text(
        payload.get("camera_flow") or payload.get("cameraFlow") or storyboard_settings.get("camera_flow") or "balanced",
        120,
    )
    try:
        camera_motion_speed = max(0, min(10, int(round(float(
            payload.get("camera_motion_speed")
            or payload.get("cameraMotionSpeed")
            or storyboard_settings.get("camera_motion_speed")
            or 4
        )))))
    except Exception:
        camera_motion_speed = 4
    try:
        character_motion_speed = max(0, min(10, int(round(float(
            payload.get("character_motion_speed")
            or payload.get("characterMotionSpeed")
            or storyboard_settings.get("character_motion_speed")
            or 4
        )))))
    except Exception:
        character_motion_speed = 4
    story_source = _clean_scene_text(
        _authoritative_script_text(authoritative_script) if authoritative_script else payload.get("story_source") or payload.get("storySource") or story_layer.get("user_story_arc") or story_layer.get("song_story_brief") or "",
        100000 if authoritative_script else 12000,
    )
    try:
        scene_count = int(float(payload.get("scene_count") or payload.get("sceneCount") or 6))
    except Exception:
        scene_count = 6
    if authoritative_script:
        scene_count = len((authoritative_script.get("scene_plan") or {}).get("scenes") or []) or scene_count
        scene_count = max(1, min(80, scene_count))
    else:
        scene_count = max(1, min(24, scene_count))
    subjects, locations = _storyboard_dialogue_reference_catalog(payload)
    existing_scenes = payload.get("scenes") if isinstance(payload.get("scenes"), list) else []
    compact_existing = []
    for index, scene in enumerate(existing_scenes[:24], start=1):
        if not isinstance(scene, dict):
            continue
        normalized = _normalize_storyboard_scene(scene, index)
        compact_existing.append({
            "scene_number": normalized.get("scene_number", index),
            "label": normalized.get("label", ""),
            "dialogue": normalized.get("lyrics", ""),
            "story_beat": normalized.get("story_beat", ""),
        })
    instruction = _storyboard_dialogue_planner_instruction(
        is_minimax=is_minimax,
        has_authoritative_script=bool(authoritative_script),
        camera_flow=camera_flow,
        camera_motion_speed=camera_motion_speed,
        character_motion_speed=character_motion_speed,
        scene_count=scene_count,
        story_source=story_source,
        script_mapper_plan_json=json.dumps((authoritative_script or {}).get("scene_plan") or {}, ensure_ascii=False, indent=2) if authoritative_script else "[none]",
        story_layer_json=json.dumps(story_layer, ensure_ascii=False, indent=2),
        project_motion_settings_json=json.dumps({"camera_flow": camera_flow, "camera_motion_speed": camera_motion_speed, "character_motion_speed": character_motion_speed}, ensure_ascii=False, indent=2),
        subjects_json=json.dumps(subjects, ensure_ascii=False, indent=2) if subjects else "[none provided]",
        locations_json=json.dumps(locations, ensure_ascii=False, indent=2) if locations else "[none provided]",
        compact_existing_json=json.dumps(compact_existing, ensure_ascii=False, indent=2) if compact_existing else "[none]",
    )
    from ..llm.output_checks import _extract_json_object_from_text
    from ..llm.builder_runner import _run_builder_text_llm, _runner_supports_json_schema

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.55),
        top_p=float(payload.get("top_p") or 0.92),
        max_new_tokens=int(payload.get("max_new_tokens") or max(1400, scene_count * 280)),
        label="MiniMax Short Film Dialogue Scenes LLM" if is_minimax else "ID-LoRA Dialogue Scenes Gemma",
        preserve_paragraphs=True,
    )
    try:
        data = _extract_json_object_from_text(text)
    except Exception as parse_error:
        # Runners that support schemas re-run the plan constrained to valid JSON; the text repair prompt is the fallback.
        # The first call stays unconstrained because schema decoding is much slower on the built-in GGUF runner.
        if _runner_supports_json_schema(payload):
            retry_instruction = instruction
            retry_schema = _storyboard_dialogue_plan_schema(is_minimax)
        else:
            retry_instruction = _storyboard_dialogue_json_repair_instruction(is_minimax, text)
            retry_schema = None
        repaired_text, repair_info = _run_builder_text_llm(
            payload,
            retry_instruction,
            temperature=0.1,
            top_p=0.8,
            max_new_tokens=int(payload.get("max_new_tokens") or max(2200, scene_count * 520)),
            label="MiniMax Short Film Dialogue JSON Repair" if is_minimax else "ID-LoRA Dialogue Scenes JSON Repair",
            preserve_paragraphs=True,
            json_schema=retry_schema,
        )
        try:
            data = _extract_json_object_from_text(repaired_text)
            run_info = {**run_info, "json_repaired": True, "repair_runner": repair_info.get("runner", "")}
        except Exception:
            raise ValueError(f"Gemma returned malformed dialogue-plan JSON and repair failed. Original parse error: {parse_error}")
    generated_scene_rows = _apply_authoritative_script_plan(data.get("scenes"), authoritative_script) if authoritative_script else data.get("scenes")
    scenes = (
        _normalize_generated_minimax_dialogue_scenes(
            generated_scene_rows,
            subjects,
            locations,
            payload.get("minimax_h3_mode"),
            camera_flow,
            camera_motion_speed,
        )
        if is_minimax else
        _normalize_generated_dialogue_scenes(data.get("scenes"), subjects, locations)
    )
    return {
        "title": _clean_scene_text(data.get("title") or "", 200),
        "premise": _clean_scene_text(data.get("premise") or story_source or "", 4000),
        "scenes": scenes,
        "scene_count": len(scenes),
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
        "authoritative_script_used": bool(authoritative_script),
    }


def _build_minimax_dialogue_scenes(payload):
    request_payload = dict(payload or {})
    request_payload["_dialogue_planner_profile"] = "minimax_short_film"
    request_payload["performance_mode"] = "speaking"
    request_payload["short_film_planning_mode"] = "guided_film"
    return _build_id_lora_dialogue_scenes(request_payload)
