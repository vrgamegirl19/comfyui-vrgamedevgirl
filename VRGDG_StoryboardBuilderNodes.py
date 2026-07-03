import asyncio
import base64
import json
import os
import re
from datetime import datetime

from aiohttp import web
from server import PromptServer

from .VRGDG_GemmaPromptSanitizer import extract_prompt_text_from_gemma_output


_VRGDG_STORYBOARD_ROUTES_REGISTERED = False

_STORYBOARD_T2V_GEMMA_INSTRUCTIONS = """You are a text-to-video prompt builder.

The user will provide a JSON scene-card bundle. Your job is to read the JSON and create one polished text-to-video prompt for the selected scene.

Use `selected_scene_number` to choose the scene.

Use `performance_mode` to decide which opening structure to use. Read it from the selected scene's `performance_mode` or `vocal_status.performance_mode`.

If `performance_mode` is `singing` and `vocal_status.should_lip_sync` is true, use this structure:

[Shot type] on [singer subject or all visible subjects] as [singer subject sings/performs] with controlled expressive intensity, physically singing "[exact lyric line from vocal_status.lyric_text]" in sync with the music. [Singer subject]'s face shows [specific visible emotion] through [specific eye expression], subtle natural eye movement, occasional natural blinking, [specific brows], [jaw/mouth/cheek detail shaped by the lyric], and [gaze/posture/head detail], with expressive performance energy. [Hair/costume/appearance detail] catches the light or motion. [All non-singing mapped subjects are also visibly present in the same location, reacting, watching, moving, or sharing the scene without singing.]

[Singer subject] [performs a clear motivated action that fits the lyric, vocal intensity, and scene mood] [position/framing], while [each non-singing mapped subject performs a visible non-vocal reaction or action]. [Secondary action or physical interaction with the environment]. The camera [camera movement that follows or reacts to the performance], then [optional secondary camera move or reframing that does not repeat the same inward move]. It then [final visual beat such as a hold, drift, reveal, pass-by, pull-back, lateral move, rack focus, tilt, subject gesture, reflection, silhouette, texture, or emotional detail], capturing [specific facial detail, eye emotion, reflection, silhouette, texture, or emotional beat].

[Background/environment details]. [Lighting description]. [Atmosphere, haze, reflections, motion blur, particles, or texture]. [Mood/style/genre tone].

If `performance_mode` is `speaking` and `vocal_status.should_lip_sync` is true, use this structure:

[Shot type] on [speaker subject or all visible subjects] as [speaker subject/she/he] says "[exact dialogue line from vocal_status.lyric_text]" with [specific visible emotion]. [Speaker subject]'s face shows [specific visible emotion] through [specific eye expression], subtle natural eye movement, occasional natural blinking, [specific brows], [jaw/cheek detail shaped by the dialogue], and [gaze/posture/head detail], with grounded short-film acting energy. [Hair/costume/appearance detail] catches the light or motion. [All non-speaking mapped subjects are also visibly present in the same location, reacting, watching, moving, or sharing the scene silently.]

[Speaker subject] [performs a clear motivated action that fits the dialogue, emotion, and scene mood] [position/framing], while [each non-speaking mapped subject performs a visible silent reaction or action]. [Secondary action or physical interaction with the environment]. The camera [camera movement that follows or reacts to the scene], then [optional secondary camera move or reframing that does not repeat the same inward move]. It then [final visual beat such as a hold, drift, reveal, pass-by, pull-back, lateral move, rack focus, tilt, subject gesture, reflection, silhouette, texture, or emotional detail], capturing [specific facial detail, eye emotion, reflection, silhouette, texture, or emotional beat].

[Background/environment details]. [Lighting description]. [Atmosphere, haze, reflections, motion blur, particles, or texture]. [Mood/style/genre tone].

If `performance_mode` is `no_lip_sync`, `vocal_status.instrumental` is true, `vocal_status.no_lip_sync` is true, or `vocal_status.should_lip_sync` is false, use this structure:

[Shot type] on [all visible mapped subjects] in [location/setting], framed by [key environmental perspective/detail]. [Each mapped subject is visibly present; describe their shared blocking or relationship in the frame.] [Subject faces show specific visible emotion] through [specific eye expression], subtle natural eye movement, occasional natural blinking, [specific brows], [jaw/cheek detail], and [gaze/posture/head detail], with [hair/costume/appearance details] catching the light or motion.

[Each mapped subject performs a clear motivated non-vocal action that fits the scene mood, character status, and environment] [position/framing]. [Secondary action or physical interaction with the environment]. The camera [camera movement that follows or reacts to the action], then [optional secondary camera move or reframing that does not repeat the same inward move]. It then [final visual beat such as a hold, drift, reveal, pass-by, pull-back, lateral move, rack focus, tilt, subject gesture, reflection, silhouette, texture, or emotional detail], capturing [specific facial detail, eye emotion, reflection, silhouette, texture, or emotional beat].

[Background/environment details]. [Lighting description]. [Atmosphere, haze, reflections, motion blur, particles, or texture]. [Mood/style/genre tone].

Rules:

* Pull the visible subject list only from the selected scene's `subject_refs`.
* Never use subjects from the project catalog, another scene, the song story brief, or the user story arc unless that subject is also present in the selected scene's `subject_refs`.
* If a person, singer, partner, lover, husband, wife, or other character appears in the story idea but is not listed in the selected scene's `subject_refs`, treat that person as off-screen, implied, reflected only if explicitly requested, or absent. Do not describe their body, face, clothing, beard, hair, or reference image.
* If `subject_refs` has one subject, the prompt may include only that one visible subject. Secondary characters are not allowed unless there is a second subject object in `subject_refs`.
* If `subject_refs` has more than one subject, every listed subject must be visibly present in the final prompt. Do not drop, merge, hide, imply, or omit any listed subject.
* In `singing` mode, if `vocal_status.singers` lists only one subject while `subject_refs` lists multiple subjects, only the singer should sing the lyric. The other mapped subjects must still be visible as non-singing subjects who react, watch, move, pose, confront, avoid, touch the environment, or otherwise participate silently.
* In `speaking` mode, treat `vocal_status.singers` as the speaker list. If it lists only one subject while `subject_refs` lists multiple subjects, only the speaker should say the line. The other mapped subjects must still be visible as silent subjects who react, watch, move, pose, confront, avoid, touch the environment, or otherwise participate silently.
* If `vocal_status.singers` is empty but `subject_refs` has multiple subjects, include every mapped subject as visible non-singing or non-speaking subjects, depending on `performance_mode`.
* If `subject_refs` contains exactly one subject, treat it as one individual person even if the subject label sounds plural, collective, or awkwardly worded. Do not create extra copies, duplicate singers, a group, or multiple people unless multiple subject objects are provided or the user explicitly asks for a group.
* When there is one subject, use singular phrasing and pronouns that fit the subject description. For example, "The woman sings..." or "The woman says..." rather than plural wording if the provided description is a single feminine character.
* When there is one subject, never use "they", "them", or "their" for that subject. If the subject is a woman/girl/feminine character, use she/her. If the subject is a man/boy/masculine character, use he/him. If gender is unclear, repeat the subject label instead of using plural pronouns.
* When there is one subject in singing mode, write "she sings", "he sings", or "[subject label] sings", never "they sing".
* When there is one subject in speaking mode, write only "she says", "he says", or "[subject label] says", never "they say".
* Pull the location from `location_ref`.
* Use `shot_type` from the scene when available.
* Use `motion_video_summary` or `camera_motion` for camera movement.
* If `camera_motion` is empty, use `motion_video_summary`.
* Follow `camera_guidance` when present. If it says to avoid default inward moves, do not add zoom-in, push-in, dolly-in, crash-zoom, or close-up endings unless the scene explicitly requests that exact motion.
* Do not default to zoom-in, push-in, dolly-in, crash-zoom, or close-up endings. Use those inward camera moves only when `camera_motion`, `shot_type`, or the user notes explicitly ask for them.
* If `camera_motion` names a non-inward move such as pull back, track backward, side-follow, pan, tilt, crane, reveal, orbit, handheld follow, rack focus, or drift, preserve that motion and do not add a zoom-in or push-in afterward.
* Vary camera behavior between scenes. Avoid repeating the same inward camera language across multiple prompts.
* If `global_consistency_phrase` is present, include it in the final video prompt. Preserve its wording as much as possible, but lightly adapt grammar if needed so it fits the scene naturally.
* Use `performance_style` and `performance_direction` to choose body language, gesture intensity, and camera energy. In singing mode, rap/hip-hop may describe rapping with rhythmic energy, hand gestures, head nods, and confident body language instead of soft singing. In speaking mode, remove music-video wording and use grounded short-film acting language.
* Use `facial_performance` and `facial_performance_direction` as the main source for facial emotion, eyes, brows, cheeks, jaw, gaze, mouth behavior, and blinking.
* If `story_layer` exists, use `song_story_brief`, `user_story_arc`, `lyric_section`, and `scene_story_beat` as narrative guidance for emotion, symbolic action, continuity, and visual motivation. Do not quote the story layer or explain it; weave it into the scene naturally.
* If `performance_mode` is `singing` and the scene is singing, use the exact lyric line from `vocal_status.lyric_text`.
* If `performance_mode` is `speaking` and the scene has a line, use the exact line from `vocal_status.lyric_text` only inside "as she says \"...\"", "as he says \"...\"", or "as [subject label] says \"...\"".
* In speaking mode, do not use alternate verbs for the dialogue line or any wording that could be interpreted as a physical handoff action. Use "says" only.
* In speaking mode, do not mention music, singing, rapping, vocals, lyrics, song, beat, performing vocals, or lip-syncing to music.
* If `performance_mode` is `no_lip_sync`, do not quote `vocal_status.lyric_text` and do not mention saying, speaking, dialogue, singing, rapping, lyrics, vocals, mouth movement, lip-syncing, or no-vocal status.
* If the scene is instrumental or no-lip-sync, do not mention singing, speaking, lip-syncing, vocals, dialogue, mouth movement, or no-vocal status.
* Do not mention or add a microphone, mic stand, headset mic, studio mic, or microphone prop unless `microphone.include` is true or the user's scene notes explicitly ask for a microphone.
* If `microphone.include` is true, include a handheld microphone or stand microphone only when it naturally fits the scene, stage, studio, club, or live performance setup.
* Every character-present prompt must include visible facial emotion or facial performance. The subject face sentence itself must include subtle natural eye movement and occasional natural blinking, placed beside the eye/brow/gaze description. Do not append blinking or eye movement to an environment sentence.
* Singing prompts must identify the exact lyric line and include visible emotion, body language, gestures, and performance energy that fit the lyric, such as longing, defiance, grief, joy, awe, fear, tenderness, anger, confidence, or desperation.
* For visible singing prompts, do not use the word "quiet" to describe the singing, performance, intensity, face, or emotion. Use controlled, focused, intimate, restrained, inward, tender, or simmering intensity instead.
* Speaking prompts must identify the exact line with "says" only and include visible emotion, body language, gestures, and grounded acting energy that fit the line, such as longing, defiance, grief, joy, awe, fear, tenderness, anger, confidence, or desperation.
* For singing or speaking prompts, facial performance may include natural jaw movement, expressive vowel/consonant mouth shapes, lips slightly parted, bared teeth, smiles, pouts, or open-mouth intensity when the selected facial_performance_direction calls for it.
* For instrumental, no-lip-sync, or non-speaking prompts, do not describe open mouth, parted lips, mouth shapes, lip movement, mouth position, or mouthing words. Keep mouth relaxed or closed unless the scene notes explicitly ask for a visible non-vocal reaction such as a smile, grimace, or gasp.
* Non-singing and non-speaking prompts must still include visible emotional expression or restrained facial tension. Do not leave the subject blank-faced.
* Do not use "expressionless", "blank expression", "empty face", "emotionless", "unreadable face", "deadpan", or "perfectly still face" unless the user's scene notes explicitly ask for that exact effect.
* If the character is described as calm, silent, stoic, robotic, alien, or controlled, translate that into visible restrained emotion: tense jaw, focused eyes, narrowed gaze, lifted brow, suppressed tears, soft smile, or subtle unease.
* Do not copy reference image composition unless the scene card explicitly asks for it.
* Keep the prompt cinematic, visual, and video-friendly.
* Do not mention JSON, IDs, file paths, image names, or metadata.
* Do not include explanations.
* Output only the final prompt.
* Use natural language, not bracket labels.
* Keep it as one clean paragraph unless the user asks otherwise.

When information is missing, infer a fitting cinematic detail from the available subject, setting, tone, and notes."""

_STORYBOARD_T2I_GEMMA_INSTRUCTIONS = """You are a text-to-image prompt builder for a music-video storyboard.

The user will provide a JSON scene-card bundle. Your job is to read the JSON and create one polished text-to-image prompt for the selected scene.

Use `selected_scene_number` to choose the scene.

Rules:

* Create one cinematic still-frame prompt, not a video prompt.
* Pull the visible subject list only from the selected scene's `subject_refs`.
* Never use subjects from the project catalog, another scene, the song story brief, or the user story arc unless that subject is also present in the selected scene's `subject_refs`.
* If `subject_refs` has more than one subject, every listed subject must be visibly present in the image prompt. Do not drop, merge, hide, imply, or omit any listed subject.
* If `subject_refs` has one subject, describe only that one visible subject. Do not create duplicates, backup singers, crowds, or extra people unless the scene notes explicitly ask for them.
* If `vocal_status.no_character_present` is true, do not include, mention, imply, or describe any mapped character/singer/subject. Use the location, props, environment, objects, atmosphere, and composition instead.
* Pull the setting from `location_ref`.
* Include the mapped subject descriptions and location description when available.
* Use the scene lyrics, lyric section, story beat, song story brief, and user story arc only as visual guidance. Do not quote long lyrics.
* If the scene is a singing scene, show performance energy and emotion. For still images, describe a believable expressive singing expression only when facial_performance_direction calls for it, without mentioning lip sync or audio behavior.
* If the scene is instrumental or no-lip-sync, do not mention singing, lip-syncing, vocals, mouth movement, or no-vocal status.
* Use `shot_type` as the still-frame composition when available.
* If `global_consistency_phrase` is present, include it in the final image prompt. Preserve its wording as much as possible, but lightly adapt grammar if needed so it fits the scene naturally.
* Use `performance_style` and `performance_direction` for body language, wardrobe energy, and genre feel.
* Use `facial_performance` and `facial_performance_direction` for facial emotion, eyes, brows, cheeks, jaw, mouth behavior, gaze, and blinking.
* Do not describe future camera movement, animation, transitions, frame changes, or what happens next.
* Do not mention JSON, IDs, file paths, image names, or metadata.
* Do not include explanations.
* Output only the final image prompt.
* Use natural language, not bracket labels.
* Keep it as one clean paragraph.

When information is missing, infer a fitting cinematic still image from the available subject, setting, tone, and notes."""


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
        return {}
    selected = _scene_number({"scene_number": scene_bundle.get("selected_scene_number")}, 1)
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        if _scene_number(scene, index) == selected:
            return scene
    return scenes[0] if isinstance(scenes[0], dict) else {}


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
    return {
        "id": _clean_scene_text(item.get("id") or fallback_id, 160),
        "name": _clean_scene_text(item.get("name") or fallback_name, 240),
        "description": _clean_scene_text(item.get("description") or "", 4000),
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
    return {
        "enabled": bool(source.get("enabled", True)),
        "user_story_arc": _clean_scene_text(source.get("user_story_arc") or source.get("userStoryArc") or "", 8000),
        "song_story_brief": _clean_scene_text(source.get("song_story_brief") or source.get("songStoryBrief") or "", 4000),
    }


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
    video_prompt = _clean_scene_text(scene.get("video_prompt") or scene.get("i2v_prompt") or scene.get("t2v_prompt") or "", 12000)
    image_path = _clean_scene_text(scene.get("image_path") or scene.get("approved_image_path") or scene.get("image") or "", 2000)
    motion_summary = _clean_scene_text(scene.get("motion_summary") or scene.get("video_notes") or scene.get("i2v_notes") or "", 3000)
    prompt_summary = _clean_scene_text(scene.get("prompt_summary") or scene.get("summary") or image_prompt[:260], 1000)
    subjects = _normalize_tags(scene.get("subjects") or scene.get("singers") or scene.get("mapped_subjects"))
    subject_refs = _normalize_reference_items(scene.get("subject_refs"))
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
    if video_prompt_type not in {"i2v", "t2v", "rtv", "ingredients"}:
        video_prompt_type = "i2v"
    if video_prompt:
        video_prompt = _enforce_storyboard_video_facial_requirements(video_prompt, {
            **scene,
            "subjects": subjects,
            "subject_refs": subject_refs,
            "lyrics": lyrics,
            "performance_mode": performance_mode,
        })
    status = _clean_scene_text(scene.get("status") or ("image_ready" if image_path else "draft"), 80)
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
        "status": status,
        "image_prompt": image_prompt,
        "video_prompt": video_prompt,
        "image_path": image_path,
        "notes": _clean_scene_text(scene.get("notes") or "", 4000),
    }


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
        "mode": "image_to_video_prep" if any(scene.get("image_path") for scene in normalized) else "storyboard_prompts",
        "performance_mode": _normalize_performance_mode(payload.get("performance_mode") or payload.get("performanceMode") or payload.get("video_type") or payload.get("videoType")),
        "camera_flow": _clean_scene_text(payload.get("camera_flow") or "balanced", 80),
        "image_shot_flow": _clean_scene_text(payload.get("image_shot_flow") or "intimate", 80),
        "image_aesthetic": _clean_scene_text(payload.get("image_aesthetic") or "", 120),
        "global_consistency_phrase": _clean_scene_text(payload.get("global_consistency_phrase") or "", 1200),
        "facial_performance_default": _clean_scene_text(payload.get("facial_performance_default") or payload.get("facial_performance") or "", 120),
        "facial_performance_custom_default": _clean_scene_text(payload.get("facial_performance_custom_default") or payload.get("facial_performance_custom") or "", 1200),
        "story_layer": _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {}),
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
        "mode": storyboard.get("mode") or "storyboard_prompts",
        "performance_mode": _normalize_performance_mode(storyboard.get("performance_mode") or storyboard.get("performanceMode") or storyboard.get("video_type") or storyboard.get("videoType")),
        "camera_flow": _clean_scene_text(storyboard.get("camera_flow") or "balanced", 80),
        "image_shot_flow": _clean_scene_text(storyboard.get("image_shot_flow") or "intimate", 80),
        "image_aesthetic": _clean_scene_text(storyboard.get("image_aesthetic") or "", 120),
        "global_consistency_phrase": _clean_scene_text(storyboard.get("global_consistency_phrase") or "", 1200),
        "facial_performance_default": _clean_scene_text(storyboard.get("facial_performance_default") or storyboard.get("facial_performance") or "", 120),
        "facial_performance_custom_default": _clean_scene_text(storyboard.get("facial_performance_custom_default") or storyboard.get("facial_performance_custom") or "", 1200),
        "story_layer": _normalize_story_layer(storyboard.get("story_layer") or storyboard.get("storyLayer") or {}),
        "reference_builder": _normalize_reference_catalog(storyboard.get("reference_builder") or storyboard.get("referenceBuilder") or {}),
        "scenes": [_normalize_storyboard_scene(scene, index + 1) for index, scene in enumerate(scenes)],
    }
    path = _storyboard_path(project_folder)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    data["path"] = path
    return data


def _write_key_value_file(path, prefix, scenes, field):
    with open(path, "w", encoding="utf-8") as handle:
        for index, scene in enumerate(scenes, start=1):
            text = _clean_scene_text(scene.get(field) or "")
            handle.write(f"{prefix}{index}={text}\n")


def _prompt_json_entry(scene, index, field):
    prompt = _clean_scene_text(scene.get(field) or "")
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
    video_json = {
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "type": "storyboard_video_prompts",
        "performance_mode": saved.get("performance_mode") or "singing",
        "scene_count": len(scenes),
        "scenes": [
            {
                **_prompt_json_entry(scene, index, "video_prompt"),
                "video_prompt_type": _clean_scene_text(scene.get("video_prompt_type") or "", 80),
                "performance_mode": _normalize_performance_mode(scene.get("performance_mode") or saved.get("performance_mode")),
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


def _selected_storyboard_scene(scene_bundle):
    scenes = scene_bundle.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard scene-card payload has no scenes.")
    selected = int(scene_bundle.get("selected_scene_number") or scenes[0].get("scene_number") or 1)
    for scene in scenes:
        if int(scene.get("scene_number") or 0) == selected:
            return scene
    return scenes[0]


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
    text = _clean_scene_text(prompt or "", 12000)
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
    return _clean_scene_text(re.sub(r"\s{2,}", " ", text).strip(), 12000)


def _build_storyboard_image_prompt(payload):
    scene_bundle = payload.get("storyboard_payload") or payload.get("scene_bundle") or payload.get("gpt_payload")
    if not isinstance(scene_bundle, dict):
        raise ValueError("Storyboard scene-card payload is missing.")
    scenes = scene_bundle.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard scene-card payload has no scenes.")
    instruction = (
        _STORYBOARD_T2I_GEMMA_INSTRUCTIONS
        + "\n\nScene-card JSON:\n"
        + json.dumps(scene_bundle, indent=2, ensure_ascii=False)
    )
    from .VRGDG_MusicVideoBuilderNodes import _run_builder_text_llm

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
    if image_path:
        from .VRGDG_MusicVideoBuilderNodes import _generate_builder_i2v_prompt

        subject_context = "\n\n".join(
            f"{_clean_scene_text(subject.get('name') or 'Subject', 120)}: {_clean_scene_text(subject.get('description') or '', 1000)}".strip()
            for subject in selected_scene.get("subjects") or []
            if isinstance(subject, dict)
        )
        location_ref = selected_scene.get("setting") or {}
        location_context = ""
        if isinstance(location_ref, dict):
            location_context = f"{_clean_scene_text(location_ref.get('name') or 'Location', 120)}: {_clean_scene_text(location_ref.get('description') or '', 1000)}".strip()
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
        story_layer = selected_scene.get("story_layer") or {}
        user_notes = "\n\n".join(
            part for part in [
                f"Performance mode:\n{performance_mode}",
                f"Scene lyrics:\n{_clean_scene_text(vocal_status.get('lyric_text') or '', 1000)}",
                f"Lyric section:\n{_clean_scene_text(vocal_status.get('lyric_section') or story_layer.get('lyric_section') or '', 200)}",
                f"Scene story beat:\n{_clean_scene_text(story_layer.get('scene_story_beat') or '', 1200)}",
                f"Motion/video summary:\n{_clean_scene_text(selected_scene.get('motion_summary') or '', 1200)}",
                f"Camera motion:\n{_clean_scene_text(selected_scene.get('camera_motion') or '', 500)}",
                f"Performance direction:\n{_clean_scene_text(selected_scene.get('performance_direction') or selected_scene.get('performance_style') or '', 1000)}",
                f"Facial performance direction:\n{_clean_scene_text(selected_scene.get('facial_performance_direction') or selected_scene.get('facial_performance_custom') or selected_scene.get('facial_performance') or '', 1600)}",
            ]
            if part.split(":\n", 1)[-1].strip()
        )
        vision_payload = {
            **payload,
            "model_file": payload.get("vision_model_file") or payload.get("vision_model") or payload.get("model_file") or "",
            "mmproj_file": payload.get("mmproj_file") or payload.get("mmproj") or "",
            "t2i_prompt": _clean_scene_text(selected_scene.get("text_to_image_prompt") or selected_scene.get("scene_summary") or "", 12000),
            "image_reference_path": image_path,
            "user_notes": user_notes,
            "performance_mode": performance_mode,
            "subject_context": subject_context,
            "location_context": location_context,
            "no_character_present": bool(vocal_status.get("no_character_present")),
            "max_new_tokens": int(payload.get("max_new_tokens") or 1800),
        }
        result = _generate_builder_i2v_prompt(vision_payload)
        result["prompt"] = _enforce_storyboard_video_facial_requirements(
            _fix_single_subject_prompt_pronouns(result.get("prompt") or "", scene_bundle),
            selected_scene,
        )
        return result

    instruction = (
        _STORYBOARD_T2V_GEMMA_INSTRUCTIONS
        + "\n\nScene-card JSON:\n"
        + json.dumps(scene_bundle, indent=2, ensure_ascii=False)
    )
    from .VRGDG_MusicVideoBuilderNodes import _run_builder_text_llm

    prompt, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or 1400),
        label="Storyboard Gemma4",
        preserve_paragraphs=True,
    )
    prompt = _enforce_storyboard_video_facial_requirements(
        _fix_single_subject_prompt_pronouns(prompt, scene_bundle),
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


def _build_story_layer_brief(payload):
    lyrics = _clean_scene_text(payload.get("lyrics") or payload.get("lyrics_text") or "", 16000)
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        scenes = []
    compact_scenes = []
    for index, scene in enumerate(scenes[:160], start=1):
        if not isinstance(scene, dict):
            continue
        normalized = _normalize_storyboard_scene(scene, index)
        compact_scenes.append({
            "scene_number": normalized["scene_number"],
            "label": normalized["label"],
            "lyric_section": normalized.get("lyric_section", ""),
            "lyrics": normalized.get("lyrics", "")[:500],
        })
    if not lyrics and not compact_scenes and not story_layer.get("user_story_arc"):
        raise ValueError("Lyrics, scene lyrics, or a user story arc are required to create a story brief.")
    instruction = (
        "You are a music video story planner.\n"
        "Create a compact story brief that can guide per-scene video prompts without sending the full lyrics every time.\n\n"
        "Rules:\n"
        "- Use the user story arc as the strongest direction when it exists.\n"
        "- Use the lyrics and song sections to infer emotional progression, recurring symbols, visual motifs, and character journey.\n"
        "- Do not summarize every lyric line.\n"
        "- Do not quote long lyric sections.\n"
        "- Keep it useful for music-video scene prompting.\n"
        "- Output plain text only, no markdown table.\n"
        "- Keep it under 250 words.\n\n"
        "Include these compact headings exactly:\n"
        "Story premise:\n"
        "Emotional arc:\n"
        "Visual motifs:\n"
        "Scene guidance:\n\n"
        f"User story arc:\n{story_layer.get('user_story_arc') or '[none]'}\n\n"
        f"Full/pasted lyrics:\n{lyrics or '[not provided]'}\n\n"
        f"Scene lyric map:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}"
    )
    from .VRGDG_MusicVideoBuilderNodes import _run_builder_text_llm

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or 800),
        label="Storyboard Story Brief Gemma",
        preserve_paragraphs=True,
    )
    text = _clean_scene_text(text, 4000)
    if not text:
        raise ValueError("Gemma returned an empty story brief.")
    return {
        "story_brief": text,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


def _build_story_layer_arc(payload):
    lyrics = _clean_scene_text(payload.get("lyrics") or payload.get("lyrics_text") or "", 16000)
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    story_idea = _clean_scene_text(payload.get("story_idea") or payload.get("storyIdea") or story_layer.get("user_story_arc") or "", 4000)
    style_theme = _clean_scene_text(payload.get("style_theme") or payload.get("styleTheme") or payload.get("theme") or "", 1600)
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        scenes = []
    compact_scenes = []
    subjects = []
    locations = []
    seen_subjects = set()
    seen_locations = set()
    for index, scene in enumerate(scenes[:160], start=1):
        if not isinstance(scene, dict):
            continue
        normalized = _normalize_storyboard_scene(scene, index)
        compact_scenes.append({
            "scene_number": normalized["scene_number"],
            "label": normalized["label"],
            "lyric_section": normalized.get("lyric_section", ""),
            "lyrics": normalized.get("lyrics", "")[:500],
        })
        for subject in normalized.get("subject_refs") or []:
            if not isinstance(subject, dict):
                continue
            name = _clean_scene_text(subject.get("name") or "", 120)
            description = _clean_scene_text(subject.get("description") or "", 500)
            key = name.lower()
            if key and key not in seen_subjects:
                seen_subjects.add(key)
                subjects.append({"name": name, "description": description})
        location = normalized.get("location_ref")
        if isinstance(location, dict):
            name = _clean_scene_text(location.get("name") or "", 120)
            description = _clean_scene_text(location.get("description") or "", 500)
            key = name.lower()
            if key and key not in seen_locations:
                seen_locations.add(key)
                locations.append({"name": name, "description": description})
    reference_builder = payload.get("reference_builder") or payload.get("referenceBuilder") or {}
    if isinstance(reference_builder, dict):
        for subject in reference_builder.get("subjects") or []:
            if not isinstance(subject, dict):
                continue
            name = _clean_scene_text(subject.get("name") or "", 120)
            description = _clean_scene_text(subject.get("description") or "", 500)
            key = name.lower()
            if key and key not in seen_subjects:
                seen_subjects.add(key)
                subjects.append({"name": name, "description": description})
        for location in reference_builder.get("locations") or []:
            if not isinstance(location, dict):
                continue
            name = _clean_scene_text(location.get("name") or "", 120)
            description = _clean_scene_text(location.get("description") or "", 500)
            key = name.lower()
            if key and key not in seen_locations:
                seen_locations.add(key)
                locations.append({"name": name, "description": description})
    instruction = (
        "You are a music video story arc generator.\n\n"
        "Your job is to take song lyrics and turn them into a simple, short story arc for a music video.\n\n"
        "The user may provide:\n"
        "* Song lyrics\n"
        "* Story idea (optional)\n"
        "* Style/theme (optional)\n"
        "* Character descriptions\n"
        "* Location descriptions\n\n"
        "All inputs are optional. If something is missing, make a strong creative choice and continue.\n\n"
        "Your output should be short, clean, and easy to use.\n"
        "Always break the idea down by song structure.\n\n"
        "Use this format:\n\n"
        "Intro:\n"
        "Super short visual concept.\n\n"
        "Verse 1:\n"
        "Super short story idea.\n\n"
        "Pre-Chorus:\n"
        "Super short build-up idea.\n\n"
        "Chorus:\n"
        "Super short main emotional or visual payoff.\n\n"
        "Verse 2:\n"
        "Super short story development.\n\n"
        "Pre-Chorus 2:\n"
        "Super short escalation idea.\n\n"
        "Chorus 2:\n"
        "Super short bigger version of the payoff.\n\n"
        "Bridge:\n"
        "Super short twist, memory, confrontation, or transformation.\n\n"
        "Final Chorus:\n"
        "Super short climax idea.\n\n"
        "Outro:\n"
        "Super short ending image.\n\n"
        "Rules:\n"
        "* Keep every section brief.\n"
        "* Do not write a full treatment.\n"
        "* Do not over-explain.\n"
        "* Do not summarize the lyrics line by line.\n"
        "* Turn the lyrics into a simple visual story arc.\n"
        "* Each section should be a clear concept, not a long scene.\n"
        "* Use cinematic, visual language.\n"
        "* If the song does not clearly have every section, infer a natural structure.\n"
        "* If only lyrics are provided, build the arc from the lyrics.\n"
        "* If no lyrics are provided, build the arc from the theme or story idea.\n"
        "* Do not ask follow-up questions unless absolutely necessary.\n"
        "* Output only the story arc sections. No intro note, no markdown table, no JSON.\n\n"
        f"Story idea:\n{story_idea or '[not provided]'}\n\n"
        f"Style/theme:\n{style_theme or '[not provided]'}\n\n"
        f"Character descriptions:\n{json.dumps(subjects[:24], ensure_ascii=False, indent=2) if subjects else '[not provided]'}\n\n"
        f"Location descriptions:\n{json.dumps(locations[:40], ensure_ascii=False, indent=2) if locations else '[not provided]'}\n\n"
        f"Full/pasted lyrics:\n{lyrics or '[not provided]'}\n\n"
        f"Scene lyric map:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2) if compact_scenes else '[not provided]'}"
    )
    from .VRGDG_MusicVideoBuilderNodes import _run_builder_text_llm

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.92),
        max_new_tokens=int(payload.get("max_new_tokens") or 900),
        label="Storyboard Story Arc Gemma",
        preserve_paragraphs=True,
    )
    text = _clean_scene_text(text, 5000)
    if not text:
        raise ValueError("Gemma returned an empty story arc.")
    return {
        "story_arc": text,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


def _build_story_layer_scene_beat(payload):
    scene_bundle = payload.get("storyboard_payload") or payload.get("scene_bundle") or payload.get("gpt_payload")
    if not isinstance(scene_bundle, dict):
        raise ValueError("Storyboard scene-card payload is missing.")
    scene = _selected_storyboard_scene(scene_bundle)
    if not scene:
        raise ValueError("Storyboard scene-card payload has no selected scene.")
    story_layer = _normalize_story_layer(payload.get("story_layer") or scene_bundle.get("story_layer") or {})
    previous_beat = _clean_scene_text(payload.get("previous_beat") or "", 1200)
    next_lyrics = _clean_scene_text(payload.get("next_lyrics") or "", 800)
    instruction = (
        "You are a music video scene-story planner.\n"
        "Create one concise scene story beat that tells the video prompt writer what this scene contributes to the larger music-video story.\n\n"
        "Rules:\n"
        "- Use the Song Story Brief and User Story Arc as continuity anchors.\n"
        "- Use the selected scene lyrics, lyric section, subject details, location details, vocal status, and no-character flag.\n"
        "- Describe narrative purpose, emotional state, visual symbolism, and how the scene should feel.\n"
        "- Do not write the final video prompt.\n"
        "- Do not include camera technical instructions unless they are part of the story emotion.\n"
        "- Do not quote long lyric text.\n"
        "- If no character is present, make the beat about location, objects, atmosphere, memory, or symbolism.\n"
        "- Output one short paragraph only, no label, no bullets.\n"
        "- Keep it under 80 words.\n\n"
        f"User Story Arc:\n{story_layer.get('user_story_arc') or '[none]'}\n\n"
        f"Song Story Brief:\n{story_layer.get('song_story_brief') or '[none]'}\n\n"
        f"Previous scene beat:\n{previous_beat or '[none]'}\n\n"
        f"Next scene lyric text:\n{next_lyrics or '[none]'}\n\n"
        "Selected scene JSON:\n"
        + json.dumps(scene, ensure_ascii=False, indent=2)
    )
    from .VRGDG_MusicVideoBuilderNodes import _run_builder_text_llm

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or 360),
        label="Storyboard Scene Beat Gemma",
        preserve_paragraphs=True,
    )
    text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(text, 1800), flags=re.I)
    if not text:
        raise ValueError("Gemma returned an empty scene story beat.")
    return {
        "story_beat": text,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


def _ensure_storyboard_routes():
    global _VRGDG_STORYBOARD_ROUTES_REGISTERED
    if _VRGDG_STORYBOARD_ROUTES_REGISTERED:
        return
    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.post("/vrgdg/storyboard/load")
    async def vrgdg_storyboard_load(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_load_storyboard, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, "storyboard": result})

    @server_instance.routes.post("/vrgdg/storyboard/save")
    async def vrgdg_storyboard_save(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_storyboard, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, "storyboard": result})

    @server_instance.routes.post("/vrgdg/storyboard/import_reference_image")
    async def vrgdg_storyboard_import_reference_image(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_import_storyboard_reference_image, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/export_prompts")
    async def vrgdg_storyboard_export_prompts(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_export_storyboard_prompts, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/gemma_video_prompt")
    async def vrgdg_storyboard_gemma_video_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_storyboard_video_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/gemma_image_prompt")
    async def vrgdg_storyboard_gemma_image_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_storyboard_image_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/story_brief")
    async def vrgdg_storyboard_story_brief(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_story_layer_brief, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/story_arc")
    async def vrgdg_storyboard_story_arc(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_story_layer_arc, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/scene_story_beat")
    async def vrgdg_storyboard_scene_story_beat(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_story_layer_scene_beat, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    _VRGDG_STORYBOARD_ROUTES_REGISTERED = True


class VRGDG_StoryboardBuilderUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_folder": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("project_folder",)
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Storyboard planning UI for organizing scene prompts before image/video creation."

    def noop(self, project_folder):
        return (project_folder,)


_ensure_storyboard_routes()


NODE_CLASS_MAPPINGS = {
    "VRGDG_StoryboardBuilderUI": VRGDG_StoryboardBuilderUI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_StoryboardBuilderUI": "VRGDG Storyboard Builder UI",
}
