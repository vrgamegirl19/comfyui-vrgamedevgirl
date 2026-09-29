import json
import os
import re
from datetime import datetime

from ..core.atomic_write import atomic_write_json
from .scene_helpers import (
    _clean_scene_text,
    _decode_image_data_url,
    _normalize_performance_mode,
    _normalize_reference_catalog,
    _normalize_reference_item,
    _normalize_reference_items,
    _normalize_speaker_assignments,
    _normalize_story_arc_detail,
    _normalize_story_layer,
    _normalize_tags,
    _prompts_folder,
    _safe_file_stem,
    _safe_project_folder,
    _scene_number,
    _speed_value,
    _storyboard_folder,
    _storyboard_path,
)
from .scene_prompts import _enforce_storyboard_video_facial_requirements


def _import_storyboard_reference_image(payload):
    project_folder = _safe_project_folder(payload.get("project_folder", ""))
    kind = str(payload.get("kind") or "subject").strip().lower()
    if kind not in {"subject", "location"}:
        kind = "subject"
    name = _clean_scene_text(payload.get("name") or ("Location" if kind == "location" else "Subject"), 240)
    description = _clean_scene_text(payload.get("description") or "", 4000)
    raw, ext = _decode_image_data_url(payload.get("image_data") or payload.get("data") or "")
    reference_dir = os.path.join(_storyboard_folder(project_folder), "references", "locations" if kind == "location" else "subjects")
    os.makedirs(reference_dir, exist_ok=True)
    stem = _safe_file_stem(name, kind)
    path = os.path.join(reference_dir, f"{stem}.{ext}")
    suffix = 2
    while os.path.exists(path):
        path = os.path.join(reference_dir, f"{stem}_{suffix}.{ext}")
        suffix += 1
    with open(path, "wb") as handle:
        handle.write(raw)
    ref_id = _clean_scene_text(payload.get("id") or f"{kind}_{stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}", 160)
    reference = _normalize_reference_item({
        "id": ref_id,
        "name": name,
        "description": description,
        "image": {
            "path": path,
            "name": os.path.basename(path),
            "data": "",
        },
    }, name, ref_id)
    return {"reference": reference, "path": path}


def _normalize_storyboard_scene(scene, fallback_number=1):
    if not isinstance(scene, dict):
        scene = {}
    number = _scene_number(scene, fallback_number)
    label = _clean_scene_text(scene.get("label") or f"Scene {number}", 180)
    lyrics = _clean_scene_text(scene.get("lyrics") or scene.get("lyric_text") or scene.get("lyricNote") or "", 4000)
    lyric_section = _clean_scene_text(scene.get("lyric_section") or scene.get("section") or scene.get("song_section") or "", 160)
    story_beat = _clean_scene_text(scene.get("story_beat") or scene.get("scene_story_beat") or scene.get("narrative_beat") or "", 1800)
    performance_mode = _normalize_performance_mode(scene.get("performance_mode") or scene.get("performanceMode") or scene.get("video_performance_mode") or scene.get("videoPerformanceMode"))
    image_prompt = _clean_scene_text(scene.get("image_prompt") or scene.get("t2i_prompt") or scene.get("prompt") or "", 12000)
    video_prompt = _clean_scene_text(scene.get("video_prompt") or scene.get("i2v_prompt") or scene.get("t2v_prompt") or "", 100000)
    image_path = _clean_scene_text(scene.get("image_path") or scene.get("approved_image_path") or scene.get("image") or "", 2000)
    image_data = str(scene.get("image_data") or scene.get("image_reference_data") or "").strip()
    image_name = _clean_scene_text(scene.get("image_name") or scene.get("image_reference_name") or "", 260)
    motion_summary = _clean_scene_text(scene.get("motion_summary") or scene.get("video_notes") or scene.get("i2v_notes") or "", 3000)
    prompt_summary = _clean_scene_text(scene.get("prompt_summary") or scene.get("summary") or image_prompt[:260], 1000)
    subjects = _normalize_tags(scene.get("subjects") or scene.get("singers") or scene.get("mapped_subjects"))
    subject_refs = _normalize_reference_items(scene.get("subject_refs"))
    speaker_assignments = _normalize_speaker_assignments(
        scene.get("speaker_assignments") or scene.get("minimax_speaker_assignments") or scene.get("dialogue_cues")
    )
    setting = _clean_scene_text(scene.get("setting") or scene.get("location") or "", 500)
    location_ref = _normalize_reference_item(scene.get("location_ref"), setting or "Location", "location") if isinstance(scene.get("location_ref"), dict) else None
    shot_type = _clean_scene_text(scene.get("shot_type") or scene.get("shot") or "", 200)
    camera_motion = _clean_scene_text(scene.get("camera_motion") or scene.get("motion_preset") or "", 200)
    character_motion = _clean_scene_text(scene.get("character_motion") or scene.get("character_motion_preset") or scene.get("subject_motion") or "", 240)
    performance_style = _clean_scene_text(scene.get("performance_style") or scene.get("song_style") or scene.get("music_style") or "", 120)
    performance_direction = _clean_scene_text(scene.get("performance_direction") or "", 1000)
    facial_performance = _clean_scene_text(scene.get("facial_performance") or scene.get("facialPerformance") or scene.get("facial_expression") or scene.get("facialExpression") or "", 120)
    facial_performance_custom = _clean_scene_text(scene.get("facial_performance_custom") or scene.get("facialPerformanceCustom") or scene.get("facial_expression_custom") or scene.get("facialExpressionCustom") or "", 1200)
    facial_performance_direction = _clean_scene_text(scene.get("facial_performance_direction") or scene.get("facialPerformanceDirection") or facial_performance_custom or "", 1600)
    include_microphone = bool(scene.get("include_microphone") or scene.get("use_microphone") or scene.get("microphone"))
    trigger_position = str(scene.get("trigger_position") or scene.get("triggerPosition") or scene.get("trigger_placement") or "start").strip().lower()
    video_prompt_type = _clean_scene_text(scene.get("video_prompt_type") or scene.get("video_type") or scene.get("mode") or "", 40)
    if video_prompt_type not in {"i2v", "id_lora", "t2v", "rtv", "ingredients"}:
        video_prompt_type = "i2v"
    project_video_engine = "minimax_h3" if str(scene.get("project_video_engine") or scene.get("projectVideoEngine") or "").strip().lower() == "minimax_h3" else "ltx"
    minimax_h3_mode = str(scene.get("minimax_h3_mode") or scene.get("minimaxH3Mode") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if minimax_h3_mode not in {"text_to_video", "image_to_video", "reference_to_video", "video_to_video"}:
        minimax_h3_mode = "text_to_video"
    raw_minimax_audio_mode = str(scene.get("minimax_h3_audio_mode") or scene.get("minimaxH3AudioMode") or "input_audio").strip().lower().replace("-", "_").replace(" ", "_")
    minimax_h3_audio_mode = "built_in_audio" if raw_minimax_audio_mode in {"built_in_audio", "native_audio", "generated_audio"} else "input_audio"
    try:
        timeline_start = float(scene.get("timeline_start", scene.get("start", 0)) or 0)
        timeline_end = float(scene.get("timeline_end", scene.get("end", 0)) or 0)
        exact_duration = max(0.0, float(scene.get("exact_duration", scene.get("duration", 0)) or 0))
    except (TypeError, ValueError):
        timeline_start = 0.0
        timeline_end = 0.0
        exact_duration = 0.0
    raw_extra_subjects = scene.get("extra_subjects") or scene.get("extraSubjects") or []
    extra_subjects = []
    if isinstance(raw_extra_subjects, list):
        for index, item in enumerate(raw_extra_subjects[:100], start=1):
            if not isinstance(item, dict):
                continue
            interaction = str(item.get("interaction") or "background").strip()
            if interaction not in {"background", "background_dancing", "alongside", "dancing_with", "direct"}:
                interaction = "background"
            try:
                count = max(1, min(100, int(round(float(item.get("count") or 1)))))
            except (TypeError, ValueError):
                count = 1
            extra_subjects.append({
                "id": _clean_scene_text(item.get("id") or f"extra_{index}", 180),
                "name": _clean_scene_text(item.get("name") or item.get("title") or f"Extra {index}", 180),
                "count": count,
                "interaction": interaction,
                "identity": _clean_scene_text(item.get("identity") or item.get("description") or "", 240),
            })
    if scene.get("no_character_present") or scene.get("noCharacterPresent") or scene.get("no_visible_subject") or scene.get("no_subject"):
        extra_subjects = []
    if video_prompt and project_video_engine != "minimax_h3":
        video_prompt = _enforce_storyboard_video_facial_requirements(video_prompt, {
            **scene,
            "subjects": subjects,
            "subject_refs": subject_refs,
            "lyrics": lyrics,
            "performance_mode": performance_mode,
        })
    status = _clean_scene_text(scene.get("status") or ("image_ready" if image_path or image_data else "draft"), 80)
    return {
        "id": _clean_scene_text(scene.get("id") or f"storyboard_scene_{number}", 160),
        "scene_number": number,
        "label": label,
        "lyrics": lyrics,
        "lyric_section": lyric_section,
        "story_beat": story_beat,
        "performance_mode": performance_mode,
        "prompt_summary": prompt_summary,
        "motion_summary": motion_summary,
        "subjects": subjects,
        "subject_refs": subject_refs,
        "extra_subjects": extra_subjects,
        "speaker_assignments": speaker_assignments,
        "setting": setting,
        "location_ref": location_ref,
        "shot_type": shot_type,
        "camera_motion": camera_motion,
        "character_motion": character_motion,
        "performance_style": performance_style,
        "performance_direction": performance_direction,
        "facial_performance": facial_performance,
        "facial_performance_custom": facial_performance_custom,
        "facial_performance_direction": facial_performance_direction,
        "include_microphone": include_microphone,
        "trigger_phrase": _clean_scene_text(scene.get("trigger_phrase") or scene.get("trigger") or scene.get("Trigger") or "", 1200),
        "trigger_position": "end" if trigger_position == "end" else "start",
        "video_prompt_type": video_prompt_type,
        "project_video_engine": project_video_engine,
        "minimax_h3_mode": minimax_h3_mode,
        "minimax_h3_audio_mode": minimax_h3_audio_mode,
        "video_style": _clean_scene_text(scene.get("video_style") or scene.get("videoStyle") or "", 160),
        "video_style_custom": _clean_scene_text(scene.get("video_style_custom") or scene.get("videoStyleCustom") or "", 3000),
        "temporal_world_effect_override": _clean_scene_text(scene.get("temporal_world_effect_override") or scene.get("temporalWorldEffectOverride") or "global", 120),
        "temporal_world_effect_custom": _clean_scene_text(scene.get("temporal_world_effect_custom") or scene.get("temporalWorldEffectCustom") or "", 3000),
        "timeline_start": timeline_start,
        "timeline_end": timeline_end,
        "exact_duration": exact_duration,
        "video_prompt_origin": "gemma" if str(scene.get("video_prompt_origin") or scene.get("i2v_prompt_origin") or "").strip().lower() == "gemma" else "manual",
        "minimax_h3_pass2_prompt": _clean_scene_text(scene.get("minimax_h3_pass2_prompt") or scene.get("pass2_prompt") or "", 100000),
        "status": status,
        "image_prompt": image_prompt,
        "video_prompt": video_prompt,
        "image_path": image_path,
        "image_data": image_data,
        "image_name": image_name,
        "notes": _clean_scene_text(scene.get("notes") or "", 4000),
        "audio_direction": _clean_scene_text(scene.get("audio_direction") or scene.get("audioDirection") or "", 4000),
        "continuity": _clean_scene_text(scene.get("continuity") or scene.get("continuity_direction") or scene.get("continuityDirection") or "", 4000),
        "id_lora_character_id": _clean_scene_text(scene.get("id_lora_character_id") or scene.get("character_id") or scene.get("subject_id") or "", 180),
        "id_lora_location_id": _clean_scene_text(scene.get("id_lora_location_id") or scene.get("location_id") or "", 180),
    }


def _normalize_script_import(value):
    source = value if isinstance(value, dict) else {}
    raw_cues = source.get("cues") if isinstance(source.get("cues"), list) else []
    cues = []
    for index, item in enumerate(raw_cues[:1000], start=1):
        if not isinstance(item, dict):
            continue
        speaker_alias = _clean_scene_text(item.get("speaker_alias") or item.get("speaker") or item.get("speaker_name") or "", 240)
        text = _clean_scene_text(item.get("text") or item.get("dialogue") or item.get("line") or "", 4000)
        if not speaker_alias or not text:
            continue
        cues.append({
            "index": int(item.get("index") or index),
            "line_number": int(item.get("line_number") or 0),
            "scene_index": int(item.get("scene_index") or 0),
            "scene_label": _clean_scene_text(item.get("scene_label") or "", 240),
            "speaker": speaker_alias,
            "speaker_alias": speaker_alias,
            "speaker_id": _clean_scene_text(item.get("speaker_id") or item.get("reference_subject_id") or "", 180),
            "speaker_name": _clean_scene_text(item.get("speaker_name") or item.get("reference_subject_name") or speaker_alias, 240),
            "reference_subject_id": _clean_scene_text(item.get("reference_subject_id") or item.get("speaker_id") or "", 180),
            "reference_subject_name": _clean_scene_text(item.get("reference_subject_name") or item.get("speaker_name") or "", 240),
            "speaker_match_method": _clean_scene_text(item.get("speaker_match_method") or "manual", 40),
            "text": text,
            "word_count": int(item.get("word_count") or len(text.split())),
        })
    raw_matches = source.get("speaker_matches") if isinstance(source.get("speaker_matches"), list) else []
    speaker_matches = []
    for item in raw_matches[:180]:
        if not isinstance(item, dict):
            continue
        alias = _clean_scene_text(item.get("speaker_alias") or item.get("speaker") or "", 240)
        if not alias:
            continue
        speaker_matches.append({
            "speaker_alias": alias,
            "reference_subject_id": _clean_scene_text(item.get("reference_subject_id") or item.get("speaker_id") or "", 180),
            "reference_subject_name": _clean_scene_text(item.get("reference_subject_name") or item.get("speaker_name") or "", 240),
            "match_method": _clean_scene_text(item.get("match_method") or "manual", 40),
        })
    try:
        maximum_scene_seconds = float(source.get("maximum_scene_seconds") or source.get("max_scene_seconds") or 8)
    except Exception:
        maximum_scene_seconds = 8.0
    maximum_scene_seconds = max(3.0, min(15.0, maximum_scene_seconds))
    plan_source = source.get("scene_plan") if isinstance(source.get("scene_plan"), dict) else {}
    raw_scenes = plan_source.get("scenes") if isinstance(plan_source.get("scenes"), list) else []
    planned_scenes = []
    for scene_index, scene in enumerate(raw_scenes[:240], start=1):
        if not isinstance(scene, dict):
            continue
        raw_assignments = scene.get("speaker_assignments") if isinstance(scene.get("speaker_assignments"), list) else []
        assignments = []
        for cue_index, cue in enumerate(raw_assignments[:80], start=1):
            if not isinstance(cue, dict):
                continue
            dialogue = _clean_scene_text(cue.get("text") or cue.get("dialogue") or "", 4000)
            if not dialogue:
                continue
            assignments.append({
                "speaker_id": _clean_scene_text(cue.get("speaker_id") or cue.get("reference_subject_id") or "", 180),
                "speaker_name": _clean_scene_text(cue.get("speaker_name") or cue.get("speaker_alias") or "Speaker", 240),
                "speaker_alias": _clean_scene_text(cue.get("speaker_alias") or cue.get("speaker_name") or "Speaker", 240),
                "text": dialogue,
                "source_cue_index": int(cue.get("source_cue_index") or 0),
                "part_index": int(cue.get("part_index") or 1),
                "part_count": int(cue.get("part_count") or 1),
                "planned_start_seconds": float(cue.get("planned_start_seconds") or 0),
                "planned_end_seconds": float(cue.get("planned_end_seconds") or 0),
                "estimated_spoken_seconds": float(cue.get("estimated_spoken_seconds") or 0),
            })
        if not assignments:
            continue
        planned_scenes.append({
            "index": int(scene.get("index") or scene_index),
            "label": _clean_scene_text(scene.get("label") or f"Script Segment {scene_index}", 240),
            "source_scene_index": int(scene.get("source_scene_index") or 0),
            "source_scene_label": _clean_scene_text(scene.get("source_scene_label") or "", 240),
            "continuation_of_previous": bool(scene.get("continuation_of_previous")),
            "duration_seconds": float(scene.get("duration_seconds") or 0),
            "timeline_start_seconds": float(scene.get("timeline_start_seconds") or 0),
            "timeline_end_seconds": float(scene.get("timeline_end_seconds") or 0),
            "participant_ids": [_clean_scene_text(item, 180) for item in (scene.get("participant_ids") or []) if _clean_scene_text(item, 180)],
            "participant_names": [_clean_scene_text(item, 240) for item in (scene.get("participant_names") or []) if _clean_scene_text(item, 240)],
            "speaker_assignments": assignments,
        })
    enabled = bool(source.get("enabled", True)) and bool(cues)
    return {
        "enabled": enabled,
        "authoritative": bool(source.get("authoritative", True)),
        "format": _clean_scene_text(source.get("format") or "text", 40),
        "raw_text": _clean_scene_text(source.get("raw_text") or source.get("rawText") or "", 100000),
        "imported_at": _clean_scene_text(source.get("imported_at") or source.get("importedAt") or "", 80),
        "maximum_scene_seconds": maximum_scene_seconds,
        "cues": cues,
        "speaker_matches": speaker_matches,
        "unmatched_speakers": [_clean_scene_text(item, 240) for item in (source.get("unmatched_speakers") or []) if _clean_scene_text(item, 240)],
        "scene_plan": {
            "maximum_scene_seconds": maximum_scene_seconds,
            "scene_count": len(planned_scenes),
            "estimated_total_seconds": float(plan_source.get("estimated_total_seconds") or 0),
            "split_cue_count": int(plan_source.get("split_cue_count") or 0),
            "scenes": planned_scenes,
        },
    }


def _normalize_short_film_planning_mode(value):
    clean = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "fully_custom" if clean in {"fully_custom", "custom"} else "guided_film"


def _default_storyboard(payload):
    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []
    normalized = [_normalize_storyboard_scene(scene, index + 1) for index, scene in enumerate(scenes)]
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "project_folder": os.path.abspath(str(payload.get("project_folder", "") or "")),
        "project_video_engine": "minimax_h3" if str(payload.get("project_video_engine") or payload.get("projectVideoEngine") or "").strip().lower() == "minimax_h3" else "ltx",
        "mode": "image_to_video_prep" if any(scene.get("image_path") or scene.get("image_data") for scene in normalized) else "storyboard_prompts",
        "performance_mode": _normalize_performance_mode(payload.get("performance_mode") or payload.get("performanceMode") or payload.get("video_type") or payload.get("videoType")),
        "short_film_planning_mode": _normalize_short_film_planning_mode(payload.get("short_film_planning_mode") or payload.get("shortFilmPlanningMode")),
        "camera_flow": _clean_scene_text(payload.get("camera_flow") or "balanced", 80),
        "image_shot_flow": _clean_scene_text(payload.get("image_shot_flow") or "intimate", 80),
        "image_aesthetic": _clean_scene_text(payload.get("image_aesthetic") or "", 120),
        "video_style": _clean_scene_text(payload.get("video_style") or payload.get("videoStyle") or "", 160),
        "video_style_custom": _clean_scene_text(payload.get("video_style_custom") or payload.get("videoStyleCustom") or "", 3000),
        "temporal_world_effect": _clean_scene_text(payload.get("temporal_world_effect") or payload.get("temporalWorldEffect") or "", 160),
        "temporal_world_effect_custom": _clean_scene_text(payload.get("temporal_world_effect_custom") or payload.get("temporalWorldEffectCustom") or "", 3000),
        "temporal_allow_background_extras": (payload.get("temporal_allow_background_extras") if "temporal_allow_background_extras" in payload else payload.get("temporalAllowBackgroundExtras", True)) is not False,
        "temporal_background_intensity": _speed_value(payload.get("temporal_background_intensity") if "temporal_background_intensity" in payload else payload.get("temporalBackgroundIntensity", 8)),
        "temporal_environment_time_passage": (payload.get("temporal_environment_time_passage") if "temporal_environment_time_passage" in payload else payload.get("temporalEnvironmentTimePassage", True)) is not False,
        "temporal_protected_characters": _clean_scene_text(payload.get("temporal_protected_characters") or payload.get("temporalProtectedCharacters") or "all_referenced", 80),
        "temporal_protected_custom": _clean_scene_text(payload.get("temporal_protected_custom") or payload.get("temporalProtectedCustom") or "", 1000),
        "global_consistency_phrase": _clean_scene_text(payload.get("global_consistency_phrase") or "", 1200),
        "camera_motion_speed": _speed_value(payload.get("camera_motion_speed") or payload.get("cameraMotionSpeed")),
        "character_motion_speed": _speed_value(payload.get("character_motion_speed") or payload.get("characterMotionSpeed")),
        "story_arc_detail": _normalize_story_arc_detail(payload.get("story_arc_detail") or payload.get("storyArcDetail")),
        "performance_style_default": _clean_scene_text(payload.get("performance_style_default") or payload.get("performance_style") or payload.get("performanceStyle") or "", 120),
        "facial_performance_default": _clean_scene_text(payload.get("facial_performance_default") or payload.get("facial_performance") or "", 120),
        "facial_performance_custom_default": _clean_scene_text(payload.get("facial_performance_custom_default") or payload.get("facial_performance_custom") or "", 1200),
        "story_layer": _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {}),
        "script_import": _normalize_script_import(payload.get("script_import") or payload.get("scriptImport") or {}),
        "reference_builder": _normalize_reference_catalog(payload.get("reference_builder") or payload.get("referenceBuilder") or {}),
        "scenes": normalized,
    }


def _load_storyboard(payload):
    project_folder = _safe_project_folder(payload.get("project_folder", ""))
    path = _storyboard_path(project_folder)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        scenes = data.get("scenes", [])
        if not isinstance(scenes, list):
            scenes = []
        data["scenes"] = [_normalize_storyboard_scene(scene, index + 1) for index, scene in enumerate(scenes)]
        data["story_layer"] = _normalize_story_layer(data.get("story_layer") or data.get("storyLayer") or {})
        data["script_import"] = _normalize_script_import(data.get("script_import") or data.get("scriptImport") or {})
        data["short_film_planning_mode"] = _normalize_short_film_planning_mode(data.get("short_film_planning_mode") or data.get("shortFilmPlanningMode"))
        data["reference_builder"] = _normalize_reference_catalog(data.get("reference_builder") or data.get("referenceBuilder") or {})
        data["path"] = path
        return data
    data = _default_storyboard(payload)
    data["path"] = path
    return data


def _save_storyboard(payload):
    project_folder = _safe_project_folder(payload.get("project_folder", ""))
    storyboard = payload.get("storyboard", {})
    if not isinstance(storyboard, dict):
        raise ValueError("Storyboard payload is invalid.")
    scenes = storyboard.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []
    data = {
        "version": 1,
        "created_at": storyboard.get("created_at") or datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "project_folder": project_folder,
        "project_video_engine": "minimax_h3" if str(storyboard.get("project_video_engine") or storyboard.get("projectVideoEngine") or "").strip().lower() == "minimax_h3" else "ltx",
        "mode": storyboard.get("mode") or "storyboard_prompts",
        "performance_mode": _normalize_performance_mode(storyboard.get("performance_mode") or storyboard.get("performanceMode") or storyboard.get("video_type") or storyboard.get("videoType")),
        "short_film_planning_mode": _normalize_short_film_planning_mode(storyboard.get("short_film_planning_mode") or storyboard.get("shortFilmPlanningMode")),
        "camera_flow": _clean_scene_text(storyboard.get("camera_flow") or "balanced", 80),
        "image_shot_flow": _clean_scene_text(storyboard.get("image_shot_flow") or "intimate", 80),
        "image_aesthetic": _clean_scene_text(storyboard.get("image_aesthetic") or "", 120),
        "video_style": _clean_scene_text(storyboard.get("video_style") or storyboard.get("videoStyle") or "", 160),
        "video_style_custom": _clean_scene_text(storyboard.get("video_style_custom") or storyboard.get("videoStyleCustom") or "", 3000),
        "temporal_world_effect": _clean_scene_text(storyboard.get("temporal_world_effect") or storyboard.get("temporalWorldEffect") or "", 160),
        "temporal_world_effect_custom": _clean_scene_text(storyboard.get("temporal_world_effect_custom") or storyboard.get("temporalWorldEffectCustom") or "", 3000),
        "temporal_allow_background_extras": (storyboard.get("temporal_allow_background_extras") if "temporal_allow_background_extras" in storyboard else storyboard.get("temporalAllowBackgroundExtras", True)) is not False,
        "temporal_background_intensity": _speed_value(storyboard.get("temporal_background_intensity") if "temporal_background_intensity" in storyboard else storyboard.get("temporalBackgroundIntensity", 8)),
        "temporal_environment_time_passage": (storyboard.get("temporal_environment_time_passage") if "temporal_environment_time_passage" in storyboard else storyboard.get("temporalEnvironmentTimePassage", True)) is not False,
        "temporal_protected_characters": _clean_scene_text(storyboard.get("temporal_protected_characters") or storyboard.get("temporalProtectedCharacters") or "all_referenced", 80),
        "temporal_protected_custom": _clean_scene_text(storyboard.get("temporal_protected_custom") or storyboard.get("temporalProtectedCustom") or "", 1000),
        "global_consistency_phrase": _clean_scene_text(storyboard.get("global_consistency_phrase") or "", 1200),
        "camera_motion_speed": _speed_value(storyboard.get("camera_motion_speed") or storyboard.get("cameraMotionSpeed")),
        "character_motion_speed": _speed_value(storyboard.get("character_motion_speed") or storyboard.get("characterMotionSpeed")),
        "story_arc_detail": _normalize_story_arc_detail(storyboard.get("story_arc_detail") or storyboard.get("storyArcDetail")),
        "performance_style_default": _clean_scene_text(storyboard.get("performance_style_default") or storyboard.get("performance_style") or storyboard.get("performanceStyle") or "", 120),
        "facial_performance_default": _clean_scene_text(storyboard.get("facial_performance_default") or storyboard.get("facial_performance") or "", 120),
        "facial_performance_custom_default": _clean_scene_text(storyboard.get("facial_performance_custom_default") or storyboard.get("facial_performance_custom") or "", 1200),
        "story_layer": _normalize_story_layer(storyboard.get("story_layer") or storyboard.get("storyLayer") or {}),
        "script_import": _normalize_script_import(storyboard.get("script_import") or storyboard.get("scriptImport") or {}),
        "reference_builder": _normalize_reference_catalog(storyboard.get("reference_builder") or storyboard.get("referenceBuilder") or {}),
        "scenes": [_normalize_storyboard_scene(scene, index + 1) for index, scene in enumerate(scenes)],
    }
    path = _storyboard_path(project_folder)
    atomic_write_json(path, data)
    data["path"] = path
    return data


def _write_key_value_file(path, prefix, scenes, field):
    with open(path, "w", encoding="utf-8") as handle:
        for index, scene in enumerate(scenes, start=1):
            text_limit = 100000 if field == "video_prompt" else 12000
            text = _clean_scene_text(scene.get(field) or "", text_limit)
            handle.write(f"{prefix}{index}={text}\n")


def _prompt_json_entry(scene, index, field):
    prompt_limit = 100000 if field == "video_prompt" else 12000
    prompt = _clean_scene_text(scene.get(field) or "", prompt_limit)
    return {
        "scene": index,
        "scene_id": _clean_scene_text(scene.get("id") or "", 120),
        "label": _clean_scene_text(scene.get("label") or f"Scene {index}", 200),
        "lyric_section": _clean_scene_text(scene.get("lyric_section") or "", 160),
        "lyric_line": _clean_scene_text(scene.get("lyrics") or "", 1200),
        "prompt": prompt,
    }


def _export_storyboard_prompts(payload):
    saved = _save_storyboard(payload)
    project_folder = _safe_project_folder(payload.get("project_folder", ""))
    prompts_dir = _prompts_folder(project_folder)
    scenes = saved.get("scenes", [])
    t2i_path = os.path.join(prompts_dir, "t2i_prompts.txt")
    i2v_path = os.path.join(prompts_dir, "i2v_prompts.txt")
    t2i_json_path = os.path.join(prompts_dir, "t2i_prompts.json")
    video_json_path = os.path.join(prompts_dir, "video_prompts.json")
    summary_path = os.path.join(_storyboard_folder(project_folder), "storyboard_export.json")
    _write_key_value_file(t2i_path, "Prompt", scenes, "image_prompt")
    _write_key_value_file(i2v_path, "I2V", scenes, "video_prompt")
    t2i_json = {
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "type": "storyboard_t2i_prompts",
        "scene_count": len(scenes),
        "scenes": [_prompt_json_entry(scene, index, "image_prompt") for index, scene in enumerate(scenes, start=1)],
    }
    existing_pass2 = {}
    if os.path.isfile(video_json_path):
        try:
            with open(video_json_path, "r", encoding="utf-8") as handle:
                existing_video = json.load(handle)
            for item in (existing_video.get("scenes") or []) if isinstance(existing_video, dict) else []:
                if not isinstance(item, dict) or "pass2_prompt" not in item:
                    continue
                pass2 = str(item.get("pass2_prompt") or "")
                scene_id = str(item.get("scene_id") or "")
                if scene_id:
                    existing_pass2[scene_id] = pass2
                try:
                    existing_pass2[int(item.get("scene") or 0)] = pass2
                except (TypeError, ValueError):
                    pass
        except (OSError, ValueError, TypeError):
            existing_pass2 = {}

    def _exported_pass2_prompt(scene, index):
        text = _clean_scene_text(scene.get("minimax_h3_pass2_prompt") or scene.get("pass2_prompt") or "", 100000)
        if text:
            return text
        return existing_pass2.get(str(scene.get("id") or "")) or existing_pass2.get(index) or ""

    video_json = {
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "type": "storyboard_video_prompts",
        "project_video_engine": saved.get("project_video_engine") or "ltx",
        "performance_mode": saved.get("performance_mode") or "singing",
        "scene_count": len(scenes),
        "scenes": [
            {
                **_prompt_json_entry(scene, index, "video_prompt"),
                "video_prompt_type": _clean_scene_text(scene.get("video_prompt_type") or "", 80),
                "minimax_h3_mode": _clean_scene_text(scene.get("minimax_h3_mode") or "", 80),
                "video_style": _clean_scene_text(scene.get("video_style") or "", 160),
                "video_style_custom": _clean_scene_text(scene.get("video_style_custom") or "", 3000),
                "performance_mode": _normalize_performance_mode(scene.get("performance_mode") or saved.get("performance_mode")),
                "pass2_prompt": _exported_pass2_prompt(scene, index),
            }
            for index, scene in enumerate(scenes, start=1)
        ],
    }
    with open(t2i_json_path, "w", encoding="utf-8") as handle:
        json.dump(t2i_json, handle, indent=2, ensure_ascii=False)
    with open(video_json_path, "w", encoding="utf-8") as handle:
        json.dump(video_json, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "version": 1,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "t2i_prompts": t2i_path,
            "i2v_prompts": i2v_path,
            "t2i_prompts_json": t2i_json_path,
            "video_prompts_json": video_json_path,
            "scenes": scenes,
        }, handle, indent=2, ensure_ascii=False)
    return {
        "storyboard_path": saved.get("path", ""),
        "t2i_prompts_path": t2i_path,
        "i2v_prompts_path": i2v_path,
        "t2i_prompts_json_path": t2i_json_path,
        "video_prompts_json_path": video_json_path,
        "export_path": summary_path,
        "scene_count": len(scenes),
    }
