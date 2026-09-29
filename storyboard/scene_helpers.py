import base64
import os
import re


def _safe_project_folder(path):
    folder = os.path.abspath(str(path or "").strip().strip('"'))
    if not folder:
        raise ValueError("Project folder is missing.")
    os.makedirs(folder, exist_ok=True)
    return folder


def _storyboard_folder(project_folder):
    folder = os.path.join(_safe_project_folder(project_folder), "storyboard")
    os.makedirs(folder, exist_ok=True)
    return folder


def _storyboard_path(project_folder):
    return os.path.join(_storyboard_folder(project_folder), "storyboard.json")


def _prompts_folder(project_folder):
    folder = os.path.join(_safe_project_folder(project_folder), "prompts")
    os.makedirs(folder, exist_ok=True)
    return folder


def _clean_scene_text(value, limit=12000):
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()[:limit]


def _selected_storyboard_scene(scene_bundle):
    scenes = scene_bundle.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard scene-card payload has no scenes.")
    selected = int(scene_bundle.get("selected_scene_number") or scenes[0].get("scene_number") or 1)
    for scene in scenes:
        if int(scene.get("scene_number") or 0) == selected:
            return scene
    return scenes[0]


def _single_subject_pronouns(scene):
    if not isinstance(scene, dict):
        return None
    subject_count = scene.get("subject_count")
    subjects = scene.get("subjects")
    subject_refs = scene.get("subject_refs")
    if subject_count is None:
        if isinstance(subject_refs, list) and subject_refs:
            subject_count = len(subject_refs)
        elif isinstance(subjects, list):
            subject_count = len(subjects)
    try:
        if int(subject_count or 0) != 1:
            return None
    except Exception:
        return None

    subject = None
    if isinstance(subject_refs, list) and subject_refs:
        subject = subject_refs[0]
    elif isinstance(subjects, list) and subjects:
        subject = subjects[0]

    if isinstance(subject, dict):
        name = _clean_scene_text(subject.get("name") or "the subject", 160)
        desc = _clean_scene_text(subject.get("description") or "", 1200)
    else:
        name = _clean_scene_text(subject or "the subject", 160)
        desc = ""
    probe = f"{name}\n{desc}".lower()
    if re.search(r"\b(woman|girl|female|feminine|she|her)\b", probe):
        return {"subject": name, "they": "she", "them": "her", "their": "her", "theirs": "hers", "are": "is", "sing": "sings", "perform": "performs", "say": "says"}
    if re.search(r"\b(man|boy|male|masculine|he|him|his)\b", probe):
        return {"subject": name, "they": "he", "them": "him", "their": "his", "theirs": "his", "are": "is", "sing": "sings", "perform": "performs", "say": "says"}
    return {"subject": name or "the subject", "they": name or "the subject", "them": name or "the subject", "their": f"{name or 'the subject'}'s", "theirs": f"{name or 'the subject'}'s", "are": "is", "sing": "sings", "perform": "performs", "say": "says"}


def _match_case(replacement, original):
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _fix_single_subject_prompt_pronouns(prompt, scene_bundle):
    scene = _selected_storyboard_scene(scene_bundle)
    pronouns = _single_subject_pronouns(scene)
    if not pronouns:
        return prompt

    text = str(prompt or "")
    phrase_map = [
        (r"\bthey\s+are\b", f"{pronouns['they']} {pronouns['are']}"),
        (r"\bthey\s+sing\b", f"{pronouns['they']} {pronouns['sing']}"),
        (r"\bthey\s+say\b", f"{pronouns['they']} {pronouns['say']}"),
        (r"\bthey\s+perform\b", f"{pronouns['they']} {pronouns['perform']}"),
        (r"\bthey\s+stand\b", f"{pronouns['they']} stands"),
        (r"\bthey\s+move\b", f"{pronouns['they']} moves"),
        (r"\bthey\s+walk\b", f"{pronouns['they']} walks"),
        (r"\bthey\s+glide\b", f"{pronouns['they']} glides"),
        (r"\bthey\s+turn\b", f"{pronouns['they']} turns"),
        (r"\bthey\s+look\b", f"{pronouns['they']} looks"),
        (r"\bthey\s+hold\b", f"{pronouns['they']} holds"),
        (r"\bthey\s+raise\b", f"{pronouns['they']} raises"),
        (r"\bthey\s+tilt\b", f"{pronouns['they']} tilts"),
        (r"\bthey\s+lean\b", f"{pronouns['they']} leans"),
    ]
    for pattern, replacement in phrase_map:
        text = re.sub(pattern, lambda match: _match_case(replacement, match.group(0)), text, flags=re.IGNORECASE)

    word_map = {
        "they": pronouns["they"],
        "them": pronouns["them"],
        "their": pronouns["their"],
        "theirs": pronouns["theirs"],
    }
    text = re.sub(
        r"\b(they|them|their|theirs)\b",
        lambda match: _match_case(word_map[match.group(1).lower()], match.group(0)),
        text,
        flags=re.IGNORECASE,
    )
    return text


def _scene_number(scene, fallback):
    value = scene.get("scene_number", scene.get("number", fallback))
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(fallback or 1))


def _normalize_tags(value):
    if isinstance(value, list):
        return [str(item or "").strip()[:120] for item in value if str(item or "").strip()][:12]
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip()[:120] for item in re.split(r"[,;\n]+", text) if item.strip()][:12]


def _normalize_performance_mode(value):
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    if text in {"speaking", "short_film", "dialogue", "dialog"}:
        return "speaking"
    if text in {"no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"}:
        return "no_lip_sync"
    return "singing"


def _normalize_reference_image(value):
    image = value if isinstance(value, dict) else {}
    return {
        "path": _clean_scene_text(image.get("path") or "", 2000),
        "data": _clean_scene_text(image.get("data") or "", 400000),
        "name": _clean_scene_text(image.get("name") or "", 240),
    }


def _normalize_reference_item(value, fallback_name="Reference", fallback_id="ref"):
    item = value if isinstance(value, dict) else {}
    trigger_position = str(item.get("trigger_position") or item.get("triggerPosition") or item.get("trigger_placement") or "start").strip().lower()
    raw_voice = item.get("minimax_voice") or item.get("miniMaxVoice") or {}
    if not isinstance(raw_voice, dict):
        raw_voice = {}
    minimax_voice = {
        "preset_id": _clean_scene_text(raw_voice.get("preset_id") or raw_voice.get("presetId") or raw_voice.get("preset") or "none", 120),
        "gender": _clean_scene_text(raw_voice.get("gender") or "", 40),
        "preset_name": _clean_scene_text(raw_voice.get("preset_name") or raw_voice.get("presetName") or raw_voice.get("name") or "", 240),
        "description": _clean_scene_text(raw_voice.get("description") or raw_voice.get("voice_description") or raw_voice.get("voiceDescription") or "", 2000),
    }
    return {
        "id": _clean_scene_text(item.get("id") or fallback_id, 160),
        "name": _clean_scene_text(item.get("name") or fallback_name, 240),
        "description": _clean_scene_text(item.get("description") or "", 4000),
        "minimax_voice": minimax_voice,
        "trigger_phrase": _clean_scene_text(item.get("trigger_phrase") or item.get("trigger") or item.get("Trigger") or "", 1200),
        "trigger_position": "end" if trigger_position == "end" else "start",
        "image": _normalize_reference_image(item.get("image") if isinstance(item.get("image"), dict) else {}),
    }


def _normalize_reference_items(value):
    if not isinstance(value, list):
        return []
    refs = []
    for index, item in enumerate(value[:12]):
        if not isinstance(item, dict):
            continue
        refs.append(_normalize_reference_item(item, f"Subject {index + 1}", f"subject_{index + 1}"))
    return refs


def _normalize_speaker_assignments(value):
    if not isinstance(value, list):
        return []
    assignments = []
    for index, item in enumerate(value[:40]):
        if not isinstance(item, dict):
            continue
        assignments.append({
            "id": _clean_scene_text(item.get("id") or item.get("cue_id") or f"speaker_cue_{index + 1}", 160),
            "speaker_id": _clean_scene_text(item.get("speaker_id") or item.get("speakerId") or item.get("subject_id") or "", 160),
            "speaker_name": _clean_scene_text(item.get("speaker_name") or item.get("speakerName") or item.get("speaker") or item.get("character") or "", 240),
            "text": _clean_scene_text(item.get("text") or item.get("dialogue") or item.get("line") or item.get("lyric") or "", 2000),
        })
    return assignments


def _normalize_reference_catalog(value):
    source = value if isinstance(value, dict) else {}

    def normalize_list(items, fallback_name, fallback_id):
        if not isinstance(items, list):
            return []
        refs = []
        for index, item in enumerate(items[:180]):
            if not isinstance(item, dict):
                continue
            refs.append(_normalize_reference_item(item, f"{fallback_name} {index + 1}", f"{fallback_id}_{index + 1}"))
        return refs

    trigger_position = str(source.get("trigger_position") or source.get("triggerPosition") or source.get("trigger_placement") or "start").strip().lower()
    subject_trigger_position = str(source.get("subject_trigger_position") or source.get("subjectTriggerPosition") or source.get("trigger_position") or "start").strip().lower()
    location_trigger_position = str(source.get("location_trigger_position") or source.get("locationTriggerPosition") or source.get("trigger_position") or "start").strip().lower()
    return {
        "subjects": normalize_list(source.get("subjects"), "Subject", "subject"),
        "locations": normalize_list(source.get("locations"), "Location", "location"),
        "trigger_position": "end" if trigger_position == "end" else "start",
        "subject_trigger_position": "end" if subject_trigger_position == "end" else "start",
        "location_trigger_position": "end" if location_trigger_position == "end" else "start",
    }


def _normalize_story_layer(value):
    source = value if isinstance(value, dict) else {}
    try:
        lyric_story_strength = int(float(source.get("lyric_story_strength", source.get("lyricStoryStrength", 7))))
    except Exception:
        lyric_story_strength = 7
    lyric_story_strength = max(0, min(10, lyric_story_strength))
    return {
        "enabled": bool(source.get("enabled", True)),
        "overall_story_idea": _clean_scene_text(source.get("overall_story_idea") or source.get("overallStoryIdea") or source.get("story_idea") or source.get("storyIdea") or "", 4000),
        "user_story_arc": _clean_scene_text(source.get("user_story_arc") or source.get("userStoryArc") or "", 8000),
        "song_story_brief": _clean_scene_text(source.get("song_story_brief") or source.get("songStoryBrief") or "", 4000),
        "lyric_story_strength": lyric_story_strength,
    }


def _lyric_story_strength_guidance(story_layer):
    try:
        strength = int(float((story_layer or {}).get("lyric_story_strength", 7)))
    except Exception:
        strength = 7
    strength = max(0, min(10, strength))
    if strength <= 0:
        guidance = (
            "Ignore the lyrics as story source. Use the story arc, style, subjects, and locations instead. "
            "Do not force lyric objects, actions, or meanings into scenes."
        )
    elif strength <= 3:
        guidance = (
            "Use lyrics lightly as mood and emotional timing only. Avoid literal lyric objects/actions unless they naturally support the story."
        )
    elif strength <= 6:
        guidance = (
            "Balance lyrics with the story arc. Each vocal scene should reflect the lyric's emotional intent, and concrete lyric anchors can appear when they fit."
        )
    elif strength <= 8:
        guidance = (
            "Lyrics strongly shape the story. For each vocal scene, preserve the lyric's main feeling, situation, or image, and include a recognizable lyric anchor when possible."
        )
    else:
        guidance = (
            "Use lyrics as literally as possible while staying cinematic. For every non-instrumental scene, include at least one concrete object, action, emotion, or situation from that exact lyric line unless it would be impossible or unsafe."
        )
    return f"Lyric Story Strength: {strength}/10. {guidance}"


def _speed_value(value, fallback=4):
    try:
        speed = int(float(value))
    except Exception:
        speed = fallback
    return max(0, min(10, speed))


def _safe_file_stem(value, fallback="reference"):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    return (text[:90] or fallback).strip("._") or fallback


def _decode_image_data_url(value):
    text = str(value or "").strip()
    match = re.match(r"^data:image/([A-Za-z0-9.+-]+);base64,(.*)$", text, flags=re.S)
    if match:
        ext = match.group(1).lower()
        payload = match.group(2)
    else:
        ext = "png"
        payload = text
    if ext == "jpeg":
        ext = "jpg"
    if ext not in {"png", "jpg", "webp"}:
        ext = "png"
    try:
        data = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError("Reference image data could not be decoded.") from exc
    if not data:
        raise ValueError("Reference image data is empty.")
    if len(data) > 30 * 1024 * 1024:
        raise ValueError("Reference image is too large.")
    return data, ext


def _storyboard_scene_has_visible_character(scene):
    vocal_status = scene.get("vocal_status") if isinstance(scene, dict) else {}
    if isinstance(vocal_status, dict) and vocal_status.get("no_character_present"):
        return False
    if isinstance(scene, dict):
        if scene.get("no_character_present") or scene.get("noCharacterPresent"):
            return False
        return bool(scene.get("subject_refs") or scene.get("subjects") or scene.get("visible_subjects") or scene.get("visibleSubjects"))
    return False


def _storyboard_prompt_mentions_visible_face(prompt):
    text = _clean_scene_text(prompt or "", 12000).lower()
    if not text:
        return False
    return bool(re.search(
        r"\b(?:woman|man|girl|boy|person|subject|singer|rapper|performer|speaker|character|face|eyes?|brows?|gaze|mouth|jaw|cheeks?|expression|smile|frown|sings?|singing|says|speaks?)\b",
        text,
        flags=re.IGNORECASE,
    ))


def _storyboard_dialogue_reference_catalog(payload):
    reference_builder = payload.get("reference_builder") or payload.get("referenceBuilder") or {}
    if not isinstance(reference_builder, dict):
        reference_builder = {}
    catalog = _normalize_reference_catalog(reference_builder)
    subjects = []
    locations = []
    for subject in catalog.get("subjects") or []:
        if not isinstance(subject, dict):
            continue
        subject_id = _clean_scene_text(subject.get("id") or "", 160)
        name = _clean_scene_text(subject.get("name") or "", 160)
        description = _clean_scene_text(subject.get("description") or "", 1200)
        if subject_id or name or description:
            image = subject.get("image") if isinstance(subject.get("image"), dict) else {}
            subjects.append({
                "id": subject_id,
                "name": name or subject_id or "Character",
                "description": description,
                "reference_type": _clean_scene_text(subject.get("reference_type") or "character", 80),
                "image": {
                    "path": _clean_scene_text(image.get("path") or "", 2000),
                    "data": "",
                    "name": _clean_scene_text(image.get("name") or "", 240),
                },
            })
    for location in catalog.get("locations") or []:
        if not isinstance(location, dict):
            continue
        location_id = _clean_scene_text(location.get("id") or "", 160)
        name = _clean_scene_text(location.get("name") or "", 160)
        description = _clean_scene_text(location.get("description") or "", 1200)
        if location_id or name or description:
            image = location.get("image") if isinstance(location.get("image"), dict) else {}
            locations.append({
                "id": location_id,
                "name": name or location_id or "Location",
                "description": description,
                "image": {
                    "path": _clean_scene_text(image.get("path") or "", 2000),
                    "data": "",
                    "name": _clean_scene_text(image.get("name") or "", 240),
                },
            })
    return subjects, locations
