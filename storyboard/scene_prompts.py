import json
import math
import re

from ..llm.text_cleaning import extract_prompt_text_from_gemma_output
from .scene_helpers import (
    _clean_scene_text,
    _fix_single_subject_prompt_pronouns,
    _normalize_performance_mode,
    _selected_storyboard_scene,
    _single_subject_pronouns,
    _storyboard_prompt_mentions_visible_face,
    _storyboard_scene_has_visible_character,
)
from ..llm.prompts.image import _STANDARD_IMAGE_T2I_INSTRUCTIONS
from ..llm.prompts.storyboard import (
    _STORYBOARD_T2I_GEMMA_INSTRUCTIONS,
    _STORYBOARD_T2V_GEMMA_INSTRUCTIONS,
    _storyboard_flf_endpoint_instruction,
    _storyboard_image_world_style_contract,
    _storyboard_video_ltx_one_pass_contract,
    _storyboard_video_prompt_writing_rules,
    _storyboard_video_pronoun_contract,
    _STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTIONS,
    _storyboard_video_vocal_contract,
)


def _storyboard_timed_lyric_contract(scene):
    if not isinstance(scene, dict):
        return ""
    vocal_status = scene.get("vocal_status") if isinstance(scene.get("vocal_status"), dict) else {}
    cues = scene.get("lyric_cue_map") or vocal_status.get("lyric_cue_map") or []
    if not isinstance(cues, list) or not cues:
        return _clean_scene_text(
            scene.get("timed_lyric_cue_contract") or vocal_status.get("timed_lyric_cue_contract") or "",
            12000,
        )
    rows = []
    starts = []
    for index, cue in enumerate(cues, start=1):
        if not isinstance(cue, dict):
            continue
        try:
            start = float(cue.get("start"))
            end = float(cue.get("end"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        cue_type = "instrumental" if str(cue.get("type") or "").strip().lower() == "instrumental" else "vocal"
        starts.append(start)
        if cue_type == "instrumental":
            rows.append(
                f"[Cue {index}] {start:.3f}s-{end:.3f}s: INSTRUMENTAL. No visible singing or lip-sync; mouths remain closed or naturally relaxed."
            )
            continue
        lyric = _clean_scene_text(cue.get("text") or "", 1200)
        if not lyric:
            continue
        singer = _clean_scene_text(cue.get("singer_name") or "the assigned singer", 160)
        rows.append(
            f'[Cue {index}] {start:.3f}s-{end:.3f}s: {singer} sings only <d>[English] {lyric}</d>. Do not add words from any other cue.'
        )
    if not rows:
        return ""
    cut_times = ", ".join(f"{value:.3f}s" for value in starts[1:])
    cut_rule = (
        f"Create exactly {len(rows)} chronological shot intervals with cuts at {cut_times}."
        if cut_times else
        "Create one continuous shot interval for this cue."
    )
    return "\n".join([
        "AUTHORITATIVE WHISPER-TIMED LYRIC CUE CONTRACT — this overrides the general cut-frequency plan and full scene lyric:",
        cut_rule,
        *rows,
        "Never repeat the complete scene lyric in every shot. Each shot may contain only the lyric assigned to its own timed cue; instrumental cues contain no visible singing.",
    ])

def _storyboard_scene_is_visible_singing(scene):
    if not isinstance(scene, dict) or not _storyboard_scene_has_visible_character(scene):
        return False
    vocal_status = scene.get("vocal_status") if isinstance(scene.get("vocal_status"), dict) else {}
    performance_mode = _normalize_performance_mode(
        scene.get("performance_mode")
        or vocal_status.get("performance_mode")
        or scene.get("video_type")
        or scene.get("videoType")
    )
    if performance_mode != "singing":
        return False
    if vocal_status.get("instrumental") or vocal_status.get("no_lip_sync") or vocal_status.get("no_character_present"):
        return False
    if vocal_status.get("should_lip_sync") is False:
        return False
    return bool(_clean_scene_text(vocal_status.get("lyric_text") or scene.get("lyrics") or scene.get("lyric_line") or "", 1200))


def _enforce_storyboard_video_facial_requirements(prompt, scene):
    text = _clean_scene_text(prompt or "", 100000)
    if not text:
        return text
    vocal_status = scene.get("vocal_status") if isinstance(scene, dict) else {}
    no_character = bool(
        (isinstance(vocal_status, dict) and vocal_status.get("no_character_present"))
        or (isinstance(scene, dict) and (scene.get("no_character_present") or scene.get("noCharacterPresent")))
    )
    if no_character:
        return text
    if not (_storyboard_scene_has_visible_character(scene) or _storyboard_prompt_mentions_visible_face(text)):
        return text
    prompt_says_singing = bool(re.search(r"\b(?:sings?|singing|raps?|rapping)\b", text, flags=re.IGNORECASE))
    if _storyboard_scene_is_visible_singing(scene) or prompt_says_singing:
        replacements = [
            (r"\bwith\s+a\s+quiet,\s*internal\s+intensity\b", "with controlled internal intensity"),
            (r"\bwith\s+quiet\s+internal\s+intensity\b", "with controlled internal intensity"),
            (r"\bquiet,\s*internal\s+intensity\b", "controlled internal intensity"),
            (r"\bquiet\s+internal\s+intensity\b", "controlled internal intensity"),
            (r"\bquiet\s+intensity\b", "controlled intensity"),
            (r"\bquiet\s+performance\b", "controlled performance"),
            (r"\bquiet\s+emotion\b", "restrained emotion"),
            (r"\bquiet\s+singing\b", "focused singing"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    has_blink = re.search(r"\bblink\w*\b", text, flags=re.IGNORECASE)
    has_eye_movement = re.search(r"\beye\s+movement\b|\beyes?\s+(?:shift|move|track|glance|flick|dart)\b", text, flags=re.IGNORECASE)
    additions = []
    if not has_eye_movement:
        additions.append("subtle natural eye movement")
    if not has_blink:
        additions.append("occasional natural blinking")
    if additions:
        insert = ", " + ", ".join(additions)
        face_sentence = re.search(
            r"([^.]*(?:face|eyes?|brows?|gaze|expression)[^.]*)(\.)",
            text,
            flags=re.IGNORECASE,
        )
        if face_sentence:
            start, end = face_sentence.span(1)
            sentence = text[start:end]
            sentence = sentence.rstrip() + insert
            text = text[:start] + sentence + text[end:]
        else:
            text = f"{text.rstrip().rstrip('.')} with {', '.join(additions)}."
    return _clean_scene_text(re.sub(r"\s{2,}", " ", text).strip(), 100000)


def _storyboard_speed_value(value, fallback=4):
    try:
        number = float(value)
    except Exception:
        return fallback
    if not math.isfinite(number):
        return fallback
    return max(0, min(10, number))


def _camera_motion_for_storyboard_speed(value, speed_value):
    motion = _clean_scene_text(value or "", 500)
    speed = _storyboard_speed_value(speed_value, 4)
    if not motion or speed < 7:
        return motion
    replacements = [
        (r"\bslow cinematic drift\b", "energetic cinematic tracking drift"),
        (r"\bslow orbit\b", "energetic orbit"),
        (r"\bslow (left|right) orbit\b", r"energetic \1 orbit"),
        (r"\bslow zoom out\b", "brisk pull-back reveal"),
        (r"\bslow (left|right|side|lateral) drift\b", r"brisk \1 tracking drift"),
        (r"\bslow (pan|tilt|track|tracking|pull[ -]?back|drift)\b", r"brisk \1"),
        (r"\bgentle lateral drift\b", "energetic lateral tracking"),
        (r"\bgentle pan reveal\b", "brisk pan reveal"),
        (r"\bgentle (pan|tilt|orbit|drift|camera movement)\b", r"brisk \1"),
        (r"\bsubtle handheld movement\b", "active handheld tracking"),
        (r"\bsubtle handheld camera\b", "active handheld camera"),
        (r"\bsubtle handheld follow\b", "energetic handheld follow"),
        (r"\bsubtle rack focus\b", "quick rack focus"),
        (r"\bsubtle energetic orbit\b", "energetic orbit"),
        (r"\bsubtle settling pause\b", "active reframing beat"),
        (r"\bsubtle orbit movement\b", "energetic orbit movement"),
        (r"\b(?:quiet handheld hold|locked-off reaction hold|locked-off shot)\b", "active handheld reaction tracking"),
        (r"\brestrained pan\b", "brisk pan"),
    ]
    for pattern, replacement in replacements:
        motion = re.sub(pattern, replacement, motion, flags=re.IGNORECASE)
    return _clean_scene_text(re.sub(r"\s{2,}", " ", motion).strip(), 500)


def _enforce_storyboard_high_motion_language(prompt, scene):
    text = _clean_scene_text(prompt or "", 100000)
    if not text or not isinstance(scene, dict):
        return text
    camera_speed = _storyboard_speed_value(scene.get("camera_motion_speed") or scene.get("cameraMotionSpeed"), 4)
    character_speed = _storyboard_speed_value(scene.get("character_motion_speed") or scene.get("characterMotionSpeed"), 4)
    if camera_speed >= 7:
        text = _camera_motion_for_storyboard_speed(text, camera_speed)
        replacements = [
            (r"\bthen\s+holds?\s+on\b", "then continues moving across"),
            (r"\bthen\s+holds?\b", "then continues moving"),
            (r"\bsettles?\s+into\s+a\s+(?:static\s+|steady\s+)?hold\b", "flows into another coordinated camera move"),
            (r"\b(?:static|steady)\s+hold\b", "continued camera motion"),
            (r"\bholds?\s+on\s+her\s+steady,\s*powerful\s+gaze\b", "tracks her powerful gaze while the camera keeps moving"),
            (r"\bholds?\s+on\s+(his|her|their|the)\s+([^,.]+)\b", r"keeps moving around \1 \2"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        camera_terms = re.findall(r"\b(?:tracking|orbit|whip pan|pan|tilt|crane|pullback|pull-back|push|dolly|handheld|reveal)\b", text, flags=re.IGNORECASE)
        if not camera_terms:
            text = f"{text.rstrip().rstrip('.')}, with energetic camera tracking that keeps moving instead of settling into a static hold."
    if character_speed >= 4:
        replacements = [
            (r"\bmoves?\s+with\s+a\s+quiet,\s*poised\s+authority\b", "moves with forceful, physically active authority"),
            (r"\bmoves?\s+with\s+quiet,\s*poised\s+authority\b", "moves with forceful, physically active authority"),
            (r"\bquiet,\s*poised\s+authority\b", "forceful, physically active authority"),
            (r"\bquiet\s+poised\s+authority\b", "forceful physical authority"),
            (r"\bpoised,\s*unyielding\s+head\s+position\b", "forward-driving head posture with sharp turns"),
            (r"\bpoised\s+posture\b", "active, commanding posture"),
            (r"\bsubtle\s+body\s+motion\b", "clear full-body movement"),
            (r"\bstands?\s+still\b", "moves through the space"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if not re.search(r"\b(?:walks?|steps?|strides?|runs?|sprints?|dances?|crosses?|lunges?|reaches?|pushes?|pulls?|climbs?|fights?|brushes?|sweeps?|gestures?|interacts?|grabs?|lifts?|paces?)\b", text, flags=re.IGNORECASE):
            text = f"{text.rstrip().rstrip('.')}, while the subject performs a clear physical action with the body, hands, or surrounding set instead of relying on facial movement alone."
    return _clean_scene_text(re.sub(r"\s{2,}", " ", text).strip(), 100000)


def _storyboard_starting_shot_value(scene):
    if not isinstance(scene, dict):
        return ""
    requirement = scene.get("starting_shot")
    if not isinstance(requirement, dict) or requirement.get("required") is not True:
        return ""
    return _clean_scene_text(
        requirement.get("selected_starting_shot")
        or requirement.get("shot_type")
        or scene.get("shot_type")
        or "",
        240,
    )


def _storyboard_starting_shot_subject(scene):
    if not isinstance(scene, dict):
        return "the subject"
    visible_subjects = scene.get("visible_subjects")
    if isinstance(visible_subjects, list):
        for value in visible_subjects:
            name = _clean_scene_text(value, 160)
            if name:
                return name
    for key in ("subject_refs", "subjects"):
        subjects = scene.get(key)
        if not isinstance(subjects, list):
            continue
        for subject in subjects:
            if isinstance(subject, dict):
                name = _clean_scene_text(subject.get("name") or "", 160)
            else:
                name = _clean_scene_text(subject, 160)
            if name:
                return name
    return "the subject"


def _storyboard_starting_shot_sentence(scene):
    shot = _storyboard_starting_shot_value(scene)
    if not shot:
        return ""
    subject = _storyboard_starting_shot_subject(scene)
    shot_key = re.sub(r"[\s_-]+", " ", shot.lower()).strip()
    if shot_key == "eyes shot":
        return f"The video begins with an extreme close-up of {subject}'s eyes."
    if shot_key == "mouth shot":
        return f"The video begins with an extreme close-up of {subject}'s mouth."
    if shot_key == "hands shot":
        return f"The video begins with a close-up of {subject}'s hands."
    if shot_key == "feet shot":
        return f"The video begins with a close-up of {subject}'s feet."
    article = "an" if shot_key[:1] in "aeiou" else "a"
    target = subject if subject != "the subject" else "the scene"
    return f"The video begins with {article} {shot} of {target}."


def _ensure_storyboard_starting_shot(prompt, scene):
    text = _clean_scene_text(prompt or "", 100000)
    sentence = _storyboard_starting_shot_sentence(scene)
    if not text or not sentence:
        return text
    opening = text[:500]
    has_opening_marker = re.search(
        r"\b(?:the\s+video\s+)?(?:begins?|starts?|opens?)\s+with\b"
        r"|\b(?:opening|first)\s+(?:shot|frame)\b",
        opening,
        flags=re.IGNORECASE,
    )
    shot_key = _storyboard_starting_shot_value(scene).lower()
    if shot_key == "eyes shot":
        has_required_framing = re.search(r"\beyes?\b", opening, flags=re.IGNORECASE)
    else:
        shot_words = [word for word in re.findall(r"[a-z0-9]+", shot_key) if word != "shot"]
        has_required_framing = bool(shot_words) and all(
            re.search(rf"\b{re.escape(word)}\b", opening, flags=re.IGNORECASE)
            for word in shot_words
        )
    if has_opening_marker and has_required_framing:
        return text
    return f"{sentence} {text}".strip()


def _storyboard_reference_opening(scene):
    if not isinstance(scene, dict) or scene.get("no_character_present"):
        subject_count = 0
    else:
        subject_refs = scene.get("subject_refs") if isinstance(scene.get("subject_refs"), list) else []
        subject_count = 0
        for subject in subject_refs:
            if not isinstance(subject, dict):
                continue
            image = subject.get("image") if isinstance(subject.get("image"), dict) else subject
            if image.get("path") or image.get("data") or subject.get("image_path") or subject.get("image_data"):
                subject_count += 1
    location_ref = scene.get("location_ref") if isinstance(scene.get("location_ref"), dict) else {}
    location_image = location_ref.get("image") if isinstance(location_ref.get("image"), dict) else location_ref
    has_location = bool(location_image.get("path") or location_image.get("data") or location_ref.get("image_path") or location_ref.get("image_data"))
    if not subject_count and not has_location:
        return ""
    character_phrase = "character reference images" if subject_count > 1 else "character reference image"
    if subject_count and has_location:
        return f"Using the provided {character_phrase} and location reference image"
    if subject_count:
        return f"Using the provided {character_phrase}"
    return "Using the provided location reference image"


def _ensure_storyboard_reference_opening(prompt, scene):
    text = str(prompt or "").strip()
    opening = _storyboard_reference_opening(scene)
    if not opening or not text:
        return text
    text = re.sub(
        r"^Using the provided\s+"
        r"(?:(?:character|location|scene|reference)\s+)*(?:images?|references?)"
        r"(?:\s+and\s+(?:(?:character|location|scene|reference)\s+)*(?:images?|references?))*"
        r"\s*,?\s*(?:create\s+)?",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"^and\s+(?:(?:character|location|scene|reference)\s+)*(?:images?|references?)\s*,?\s*(?:create\s+)?",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(r"^(?:create|make|generate)\b\s*", "", text, count=1, flags=re.IGNORECASE).strip()
    if not text:
        return f"{opening}, create a cinematic still image."
    return f"{opening}, create {text[:1].lower()}{text[1:] if len(text) > 1 else ''}".strip()


def _storyboard_image_mode_uses_reference_opening(scene_bundle):
    mode = str((scene_bundle or {}).get("image_model_mode") or (scene_bundle or {}).get("imageMode") or "").strip().lower()
    return mode in {"nano_banana", "flux_klein", "flow_gpt"}


def _build_storyboard_image_prompt(payload):
    scene_bundle = payload.get("storyboard_payload") or payload.get("scene_bundle") or payload.get("gpt_payload")
    if not isinstance(scene_bundle, dict):
        raise ValueError("Storyboard scene-card payload is missing.")
    scenes = scene_bundle.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard scene-card payload has no scenes.")
    instruction_text = _STORYBOARD_T2I_GEMMA_INSTRUCTIONS
    instruction_key = str(payload.get("builder_instruction_key") or payload.get("instruction_key") or "").strip()
    if instruction_key:
        from ..llm.builder_instructions import _effective_builder_instruction

        instruction_text = _effective_builder_instruction(payload, instruction_key, _STANDARD_IMAGE_T2I_INSTRUCTIONS)
    selected_scene = _selected_storyboard_scene(scene_bundle)
    instruction_text += "\n\n" + _STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTIONS
    image_world_style = str(payload.get("image_world_style") or "natural").strip().lower()
    image_custom_style_direction = _clean_scene_text(payload.get("image_custom_style_direction") or "", 3000)
    instruction_text += _storyboard_image_world_style_contract(image_world_style, image_custom_style_direction)
    flf_image_target = str(payload.get("flf_image_target") or "").strip().lower()
    if flf_image_target in {"start", "end"}:
        story_layer = selected_scene.get("story_layer") if isinstance(selected_scene.get("story_layer"), dict) else {}
        start_state = _clean_scene_text(story_layer.get("flf_start_state") or selected_scene.get("flf_start_state") or "", 1800)
        transformation = _clean_scene_text(story_layer.get("flf_transformation") or selected_scene.get("flf_transformation") or "", 1800)
        end_state = _clean_scene_text(story_layer.get("flf_end_state") or selected_scene.get("flf_end_state") or "", 1800)
        carry_forward = _clean_scene_text(story_layer.get("flf_carry_forward") or selected_scene.get("flf_carry_forward") or "", 1800)
        target_state = start_state if flf_image_target == "start" else end_state
        instruction_text += _storyboard_flf_endpoint_instruction(flf_image_target, target_state, transformation, carry_forward)
    instruction = (
        instruction_text
        + "\n\nScene-card JSON:\n"
        + json.dumps(scene_bundle, indent=2, ensure_ascii=False)
    )
    from ..llm.builder_runner import _run_builder_text_llm

    prompt, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or 1200),
        label="Storyboard T2I Gemma",
        preserve_paragraphs=True,
    )
    prompt = extract_prompt_text_from_gemma_output(prompt, scene_bundle.get("selected_scene_number"))
    prompt = _clean_scene_text(_fix_single_subject_prompt_pronouns(prompt, scene_bundle), 12000)
    if _storyboard_image_mode_uses_reference_opening(scene_bundle):
        prompt = _ensure_storyboard_reference_opening(prompt, selected_scene)
    if not prompt:
        raise ValueError("Gemma returned an empty Storyboard image prompt.")
    return {
        "prompt": prompt,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


def _build_storyboard_video_prompt(payload):
    scene_bundle = payload.get("storyboard_payload") or payload.get("scene_bundle") or payload.get("gpt_payload")
    if not isinstance(scene_bundle, dict):
        raise ValueError("Storyboard scene-card payload is missing.")
    scenes = scene_bundle.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard scene-card payload has no scenes.")
    selected_scene = _selected_storyboard_scene(scene_bundle)
    image_path = _clean_scene_text(selected_scene.get("image_path") or selected_scene.get("approved_image_path") or "", 2000)
    image_data = str(selected_scene.get("image_data") or selected_scene.get("image_reference_data") or "").strip()
    if image_path or image_data:
        from ..llm.video_prompt_generation import _generate_builder_i2v_prompt

        subject_context = "\n\n".join(
            f"{_clean_scene_text(subject.get('name') or 'Subject', 120)}: {_clean_scene_text(subject.get('description') or '', 1000)}".strip()
            for subject in selected_scene.get("subjects") or []
            if isinstance(subject, dict)
        )
        # The mapped Reference Builder location is authoritative. ``setting``
        # may be only a plain-text scene field, so reading it first can discard
        # a valid location_ref and leave the vision model without labeled
        # location context.
        location_ref = selected_scene.get("location_ref") or selected_scene.get("setting") or {}
        location_context = ""
        if isinstance(location_ref, dict):
            location_context = f"{_clean_scene_text(location_ref.get('name') or 'Location', 120)}: {_clean_scene_text(location_ref.get('description') or '', 1000)}".strip()
        elif isinstance(location_ref, str):
            location_context = _clean_scene_text(location_ref, 1000)
        vocal_status = selected_scene.get("vocal_status") or {}
        performance_mode = _normalize_performance_mode(
            selected_scene.get("performance_mode")
            or vocal_status.get("performance_mode")
            or scene_bundle.get("performance_mode")
            or payload.get("performance_mode")
            or payload.get("performanceMode")
            or payload.get("video_type")
            or payload.get("videoType")
        )
        timed_lyric_contract = _storyboard_timed_lyric_contract(selected_scene)
        pronouns = _single_subject_pronouns(selected_scene)
        pronoun_contract = _storyboard_video_pronoun_contract(pronouns)
        vocal_contract = _storyboard_video_vocal_contract(
            bool(timed_lyric_contract),
            _storyboard_scene_is_visible_singing(selected_scene),
        )
        ltx_scene = str(selected_scene.get("project_video_engine") or scene_bundle.get("project_video_engine") or "").strip().lower() != "minimax_h3"
        ltx_one_pass_contract = _storyboard_video_ltx_one_pass_contract(vocal_contract, pronoun_contract) if ltx_scene else ""
        story_layer = selected_scene.get("story_layer") or {}
        camera_guidance = selected_scene.get("camera_guidance") if isinstance(selected_scene.get("camera_guidance"), dict) else {}
        camera_speed_guidance = (
            selected_scene.get("camera_motion_speed_guidance")
            or camera_guidance.get("camera_motion_speed_guidance")
            or ""
        )
        first_frame_inventory = selected_scene.get("first_frame_visual_inventory")
        if isinstance(first_frame_inventory, dict):
            first_frame_inventory = first_frame_inventory.get("text") or ""
        first_frame_inventory = _clean_scene_text(
            first_frame_inventory
            or selected_scene.get("text_to_image_prompt")
            or selected_scene.get("scene_summary")
            or "",
            12000,
        )
        motion_summary = _clean_scene_text(selected_scene.get("motion_summary") or "", 1200)
        camera_motion = "" if motion_summary else _clean_scene_text(selected_scene.get("camera_motion") or "", 500)
        user_notes = "\n\n".join(
            part for part in [
                ltx_one_pass_contract,
                _STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTIONS,
                "Complete scene_card:\n" + json.dumps(
                    selected_scene.get("scene_card") or selected_scene,
                    indent=2, ensure_ascii=False,
                ),
                timed_lyric_contract,
                f"MANDATORY editing / cut plan:\n{_clean_scene_text((selected_scene.get('cut_plan') or {}).get('instruction') if isinstance(selected_scene.get('cut_plan'), dict) else '', 5000)}" if not timed_lyric_contract else "",
                f"Required starting shot:\n{json.dumps(selected_scene.get('starting_shot'), ensure_ascii=False)}" if _storyboard_starting_shot_value(selected_scene) else "",
                f"Performance mode:\n{performance_mode}",
                f"Scene lyrics:\n{_clean_scene_text(vocal_status.get('lyric_text') or '', 1000)}" if not timed_lyric_contract else "",
                f"Lyric section:\n{_clean_scene_text(vocal_status.get('lyric_section') or story_layer.get('lyric_section') or '', 200)}",
                f"Scene story beat:\n{_clean_scene_text(story_layer.get('scene_story_beat') or '', 1200)}",
                f"Motion/video summary:\n{motion_summary}",
                f"Required video style:\n{_clean_scene_text(selected_scene.get('video_style') or '', 200)}",
                f"MANDATORY exact video style verbiage — copy word-for-word into the final prompt:\n{_clean_scene_text(selected_scene.get('video_style_verbiage') or '', 3000)}",
                f"MANDATORY exact temporal / world effect verbiage — copy word-for-word into the final prompt:\n{_clean_scene_text(selected_scene.get('temporal_world_effect_verbiage') or '', 5000)}",
                f"Camera motion:\n{camera_motion}",
                f"Required camera-flow framing:\n{_clean_scene_text(selected_scene.get('camera_flow_guidance') or '', 1600)}",
                f"Camera motion speed guidance:\n{_clean_scene_text(camera_speed_guidance, 1000)}",
                f"Character motion guidance:\n{_clean_scene_text(selected_scene.get('character_motion_guidance') or '', 1000)}",
                f"Performance direction:\n{_clean_scene_text(selected_scene.get('performance_direction') or selected_scene.get('performance_style') or '', 1000)}",
                f"Facial performance direction:\n{_clean_scene_text(selected_scene.get('facial_performance_direction') or selected_scene.get('facial_performance_custom') or selected_scene.get('facial_performance') or '', 1600)}",
                f"First-frame visual inventory:\n{_clean_scene_text(first_frame_inventory, 1600)}" if first_frame_inventory else "",
                _storyboard_video_prompt_writing_rules(),
            ]
            if part.split(":\n", 1)[-1].strip()
        )
        vision_payload = {
            **payload,
            "model_file": payload.get("vision_model_file") or payload.get("vision_model") or payload.get("model_file") or "",
            "mmproj_file": payload.get("mmproj_file") or payload.get("mmproj") or "",
            "t2i_prompt": "",
            "image_reference_path": image_path,
            "image_reference_data": image_data,
            "user_notes": user_notes,
            "lyric_text": "" if timed_lyric_contract else _clean_scene_text(vocal_status.get("lyric_text") or "", 1200),
            "lyric_cue_map": selected_scene.get("lyric_cue_map") or vocal_status.get("lyric_cue_map") or [],
            "timed_lyric_cue_contract": timed_lyric_contract,
            "performance_mode": performance_mode,
            "subject_context": subject_context,
            "location_context": location_context,
            "no_character_present": bool(vocal_status.get("no_character_present")),
            "max_new_tokens": int(payload.get("max_new_tokens") or 1800),
        }
        result = _generate_builder_i2v_prompt(vision_payload)
        result["prompt"] = _ensure_storyboard_starting_shot(
            _enforce_storyboard_high_motion_language(
                _enforce_storyboard_video_facial_requirements(
                    _fix_single_subject_prompt_pronouns(result.get("prompt") or "", scene_bundle),
                    selected_scene,
                ),
                selected_scene,
            ),
            selected_scene,
        )
        return result

    instruction = (
        _STORYBOARD_T2V_GEMMA_INSTRUCTIONS
        + "\n\n" + _STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTIONS
        + "\n\nScene-card JSON:\n"
        + json.dumps(scene_bundle, indent=2, ensure_ascii=False)
    )
    from ..llm.builder_runner import _run_builder_text_llm

    prompt, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or 1400),
        label="Storyboard Gemma4",
        preserve_paragraphs=True,
    )
    prompt = _ensure_storyboard_starting_shot(
        _enforce_storyboard_high_motion_language(
            _enforce_storyboard_video_facial_requirements(
                _fix_single_subject_prompt_pronouns(prompt, scene_bundle),
                selected_scene,
            ),
            selected_scene,
        ),
        selected_scene,
    )
    if not prompt:
        raise ValueError("Gemma returned an empty Storyboard video prompt.")
    return {
        "prompt": prompt,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }
