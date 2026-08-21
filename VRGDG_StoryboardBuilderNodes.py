import asyncio
import base64
import json
import math
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
* `location_ref` is the required physical set. Do not replace it with a location from `story_layer`, `scene_story_beat`, lyrics, or the user story arc.
* If the story layer mentions a different place, translate only its emotion, conflict, or action into the mapped `location_ref` environment.
* Treat `first_frame_visual_inventory`, `text_to_image_prompt`, `scene_summary`, and existing image prompt text as first-frame visual inventory only. They may identify visible subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition.
* Do not use first-frame visual inventory for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing.
* Build video action from this hierarchy: `character_motion_guidance`, `camera_motion_speed_guidance`, `camera_guidance`, `performance_direction`, `vocal_status`, and scene story beat first; story layer second; first-frame visual inventory last, and only for visible environment/appearance details.
* Each sentence has one job and must add new information. Do not repeat the same mood, trait, motion, authority/defiance language, setting adjective, or descriptive phrase across the face, body, camera, environment, and atmosphere sentences.
* If an emotional idea or trait appears in the face sentence, do not repeat that same idea in the body, camera, environment, or atmosphere sentence. Use a different concrete visual detail instead.
* Do not duplicate adjacent words or descriptors such as "tall, tall", "vast, vast", "steady, steady", or repeated authority/defiance phrases.
* Do not copy still-image pose language, stillness language, gentle/poised/static wording, or photography-only wording from `text_to_image_prompt` into the video motion plan.
* User motion fields, `character_motion_guidance`, `camera_motion_speed_guidance`, `camera_guidance`, `performance_direction`, and scene story beat control animation, body action, camera movement, and performance energy.
* If motion speed guidance is high, it overrides calm, poised, subtle, static, steady, restrained, quiet, or hold wording from the image prompt or first frame.
* For camera speed 7-8, use energetic, visibly active camera movement; do not use slow, gentle, subtle, restrained, locked-off, static, or hold camera wording. For camera speed 9-10, include two or more coordinated camera actions in the same scene when readable.
* For character motion speed 4 or higher, include at least one clear physical body action, gesture, step, or interaction with the set. Facial expression, blinking, breathing, and mouth movement alone do not count. For speed 9-10, prefer clear full-body action such as striding, crossing the space, forceful gestures, dancing, running, fighting, climbing, or interacting with the set.
* Use `shot_type` from the scene when available.
* Follow `camera_flow_guidance` when present. Treat its framing limits as hard constraints for the entire shot, including every camera move and the ending composition.
* If `starting_shot.required` is true, the first sentence must explicitly state that the video begins with `starting_shot.selected_starting_shot`. Do not merely imply this framing or move it to the middle or end of the prompt.
* The selected starting shot describes the literal first generated frame. Do not begin with a wide, distant, establishing, or full-body lead-in and then move into the selected framing.
* For an `eyes shot`, explicitly say that the video begins with an extreme close-up of the subject's eyes.
* Begin the selected `camera_motion` from the required starting-shot framing; it may widen, orbit, track, or otherwise move afterward.
* If `motion_summary` is non-empty, it is the authoritative custom motion and camera direction. Ignore `camera_motion` rather than combining the two.
* Use `camera_motion` only when `motion_summary` is empty.
* Follow `camera_guidance` when present. If it says to avoid default inward moves, do not add zoom-in, push-in, dolly-in, crash-zoom, or close-up endings unless the scene explicitly requests that exact motion.
* Follow `camera_motion_speed_guidance` or `camera_guidance.camera_motion_speed_guidance` when present. Low values mean static/slow camera; high values mean faster or compound camera action with no static hold ending.
* Do not default to zoom-in, push-in, dolly-in, crash-zoom, or close-up endings. Use those inward camera moves only when `camera_motion`, `shot_type`, or the user notes explicitly ask for them.
* If `camera_motion` names a non-inward move such as pull back, track backward, side-follow, pan, tilt, crane, reveal, orbit, handheld follow, rack focus, or drift, preserve that motion and do not add a zoom-in or push-in afterward.
* Vary camera behavior between scenes. Avoid repeating the same inward camera language across multiple prompts.
* Follow `cut_plan.instruction` exactly when present. MiniMax cut plans use timestamped `CUT TO` blocks. LTX cut plans must express the same number of distinct continuity-preserving shots in ordinary chronological language such as "then cut to" and must not use the MiniMax timestamp schema. A continuous-shot plan forbids cuts for either engine.
* If `global_consistency_phrase` is present, include it in the final video prompt. Preserve its wording as much as possible, but lightly adapt grammar if needed so it fits the scene naturally.
* If `video_style_verbiage` is present, copy that exact text word-for-word into the final prompt. Do not paraphrase, shorten, rename, or omit it. Treat it only as the governing visual-appearance contract for lighting, color, texture, materials, production design, grading, and image finish. It must not select, replace, or modify camera motion, character motion, shot timing, editing, or transitions.
* If `temporal_world_effect_verbiage` is present, copy that exact text word-for-word into the final prompt before the first shot description. MiniMax may place it before its first timestamp; LTX must keep it in ordinary natural-language prompt form. Do not paraphrase, shorten, or omit it. It is a hard temporal-layer contract: every protected mapped/reference character, their face, performance, voice, dialogue/singing timing, and lip sync remain natural and stable while only the explicitly unprotected background/world elements receive the temporal effect. Anonymous extras may be added only when the contract allows them, and must fit `location_ref` without replacing or duplicating a mapped character.
* A temporal/world contract must be enacted, not merely copied. Every MiniMax timestamp block or LTX natural-language shot must contain the contract's required number of concrete visible background/world actions. At intensity 7 or higher, subtle flicker, ambience, particles, or vague time-passage language alone is invalid. Use visibly accelerated, frozen, reversed, looping, delayed, season-changing, light-changing, crowd, traffic, weather, reflection, shadow, or location activity appropriate to the selected effect and mapped location.
* When a temporal/world contract permits anonymous extras, wording such as `no people` in `location_ref` describes the source reference image only and is not an output prohibition. In Continuity, prohibit only additional named, principal, mapped, or referenced characters; explicitly preserve the contract's permission for anonymous unreferenced background extras.
* Use `performance_style` and `performance_direction` to choose body language, gesture intensity, and camera energy. In singing mode, rap/hip-hop may describe rapping with rhythmic energy, hand gestures, head nods, and confident body language instead of soft singing. In speaking mode, remove music-video wording and use grounded short-film acting language.
* Follow `character_motion_guidance` when present. Low values mean still/subtle body language; high values mean energetic or fast physical action when it fits the scene.
* Use `facial_performance` and `facial_performance_direction` as the main source for facial emotion, eyes, brows, cheeks, jaw, gaze, mouth behavior, and blinking.
* If `story_layer` exists, use `song_story_brief`, `user_story_arc`, `lyric_section`, and `scene_story_beat` as narrative guidance for emotion, symbolic action, continuity, and visual motivation. Do not quote the story layer or explain it; weave it into the scene naturally.
* If `performance_mode` is `singing` and the scene is singing, use the exact lyric line from `vocal_status.lyric_text`.
* For Input Audio singing, quote the exact lyric only once in the Audio 1 assignment. Do not paste the full lyric or the complete lip-sync boilerplate into every timestamp. Timestamp blocks must say that the assigned singer begins, continues, or completes the currently audible portion of the assigned lyric without quoting it again, restarting it, or extending it into silence.
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
* `location_ref` is the required physical set. Do not replace it with a location from `story_beat`, `song_story_brief`, `user_story_arc`, or lyrics.
* If the story layer mentions a different place, translate only its emotion, symbolism, pose, or action into the mapped `location_ref` environment.
* Include the mapped subject descriptions and location description when available.
* Use the scene lyrics, lyric section, story beat, song story brief, and user story arc only as visual guidance. Do not quote long lyrics.
* If the scene is a singing scene, show performance energy and emotion as a still expression only. Do not mention lip sync, audio behavior, mouth movement, eye movement, blinking, or animation.
* If the scene is instrumental or no-lip-sync, do not mention singing, lip-syncing, vocals, mouth movement, or no-vocal status.
* Use `shot_type` as the still-frame composition when available.
* If `global_consistency_phrase` is present, include it in the final image prompt. Preserve its wording as much as possible, but lightly adapt grammar if needed so it fits the scene naturally.
* Use `performance_style` and `performance_direction` for body language, wardrobe energy, and genre feel.
* Follow `character_motion_guidance` when present, but express it as still-image pose/action/body language only. Do not describe animation or future movement.
* Use `facial_performance` and `facial_performance_direction` only for still-image facial emotion: eye direction, brows, cheeks, jaw tension, mouth expression, gaze, and pose.
* Do not describe future camera movement, animation, transitions, frame changes, blinking, eye movement, mouth movement, or what happens next.
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
        "performance_style_default": _clean_scene_text(storyboard.get("performance_style_default") or storyboard.get("performance_style") or storyboard.get("performanceStyle") or "", 120),
        "facial_performance_default": _clean_scene_text(storyboard.get("facial_performance_default") or storyboard.get("facial_performance") or "", 120),
        "facial_performance_custom_default": _clean_scene_text(storyboard.get("facial_performance_custom_default") or storyboard.get("facial_performance_custom") or "", 1200),
        "story_layer": _normalize_story_layer(storyboard.get("story_layer") or storyboard.get("storyLayer") or {}),
        "script_import": _normalize_script_import(storyboard.get("script_import") or storyboard.get("scriptImport") or {}),
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


def _storyboard_video_prompt_writing_rules():
    return "\n".join([
        "Prompt writing rules:",
        "Use the image reference and text-to-image prompt only for visible first-frame details: subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition.",
        "Do not use the image prompt for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing.",
        "Use the motion/camera notes, performance direction, vocal direction, facial direction, and scene story beat to decide animation, body action, camera movement, and performance energy.",
        "Each sentence has one job and must add new information. Do not repeat the same mood, trait, motion, authority/defiance language, setting adjective, or descriptive phrase across the face, body, camera, environment, and atmosphere sentences.",
        "If an idea appears in the face sentence, do not repeat it in the body, camera, environment, or atmosphere sentence; use a different concrete visual detail instead.",
        "Do not duplicate adjacent words such as tall, tall or vast, vast.",
    ])


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
        from .VRGDG_MusicVideoBuilderNodes import _STANDARD_IMAGE_T2I_INSTRUCTIONS, _effective_builder_instruction

        instruction_text = _effective_builder_instruction(payload, instruction_key, _STANDARD_IMAGE_T2I_INSTRUCTIONS)
    selected_scene = _selected_storyboard_scene(scene_bundle)
    image_world_style = str(payload.get("image_world_style") or "natural").strip().lower()
    image_custom_style_direction = _clean_scene_text(payload.get("image_custom_style_direction") or "", 3000)
    image_style_presets = {
        "natural": "Use a naturalistic, believable visual world. Surreal details may appear only when required by the scene.",
        "surreal_subject": "Keep the environment broadly believable, but render the subject and its materials with unmistakable surreal invention.",
        "balanced_surreal": "Make both subject and environment visibly surreal while retaining enough spatial coherence to read as one designed cinematic world.",
        "full_surreal": "Construct the entire image as an unmistakably surreal world. Environment, architecture, ground, sky, vegetation, furniture, props, lighting, perspective, scale, gravity, subject, anatomy, clothing, and materials must all obey deliberate dream logic. Do not place a surreal subject inside an otherwise ordinary realistic location. Avoid generic cinematic realism, conventional architecture, naturalistic staging, and merely photoreal backgrounds.",
        "abstract": "Create a strongly abstract, nonliteral visual world using impossible space, symbolic forms, transformed materials, unconventional scale, and expressive color and light. Literal realism is not the goal.",
        "custom": "Follow the user's custom style direction as the primary visual-world contract. Apply it to every visible layer of the image, including the environment and background.",
    }
    style_instruction = image_style_presets.get(image_world_style, image_style_presets["natural"])
    instruction_text += (
        "\n\nGLOBAL IMAGE WORLD STYLE CONTRACT:\n"
        f"- {style_instruction}\n"
        + (f"- User's custom visual direction: {image_custom_style_direction}\n" if image_custom_style_direction else "")
        + "- Apply this contract to the complete frame, not only the main subject. Preserve required scene content and endpoint facts while expressing them through this style.\n"
        + "- Do not mention this contract, preset names, or workflow settings in the final prompt."
    )
    flf_image_target = str(payload.get("flf_image_target") or "").strip().lower()
    if flf_image_target in {"start", "end"}:
        story_layer = selected_scene.get("story_layer") if isinstance(selected_scene.get("story_layer"), dict) else {}
        start_state = _clean_scene_text(story_layer.get("flf_start_state") or selected_scene.get("flf_start_state") or "", 1800)
        transformation = _clean_scene_text(story_layer.get("flf_transformation") or selected_scene.get("flf_transformation") or "", 1800)
        end_state = _clean_scene_text(story_layer.get("flf_end_state") or selected_scene.get("flf_end_state") or "", 1800)
        carry_forward = _clean_scene_text(story_layer.get("flf_carry_forward") or selected_scene.get("flf_carry_forward") or "", 1800)
        target_state = start_state if flf_image_target == "start" else end_state
        endpoint_context = (
            "- This is strictly the untouched opening condition before the scene action begins.\n"
            "- Do not include, foreshadow, partially reveal, or imply the later transformation, destination anatomy, destination objects, or completed action.\n"
            "- Ignore transformation, end-state, and carry-forward fields when writing this START image.\n"
            "- Lyrics and the general story beat may guide mood only; do not include any lyric/story action that is not already explicitly visible in the required opening state.\n"
            if flf_image_target == "start" else
            f"- Planned transformation context: {transformation or '[none]'}\n"
            f"- Carry-forward continuity: {carry_forward or '[none]'}\n"
        )
        endpoint_instruction = (
            "\n\nFIRST / LAST FRAME STILL-IMAGE RULES:\n"
            f"- You are writing the {flf_image_target.upper()} endpoint still image, not a video prompt.\n"
            f"- Required visible endpoint state: {target_state or '[use the scene card literally]'}\n"
            f"{endpoint_context}"
            "- Make the required endpoint state visually concrete in one frozen image.\n"
            "- Preserve mapped subject identity, wardrobe, environment, lighting, and established anatomy unless the required endpoint explicitly changes one of them.\n"
            "- Do not describe motion over time, a transition, morphing process, first/last frames, or workflow instructions in the final image prompt.\n"
            "- Output only the image prompt."
        )
        instruction_text += endpoint_instruction
    instruction = (
        instruction_text
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
        from .VRGDG_MusicVideoBuilderNodes import _generate_builder_i2v_prompt

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
        pronouns = _single_subject_pronouns(selected_scene)
        if pronouns:
            pronoun_contract = (
                f"This scene contains exactly one mapped subject. Use {pronouns['they']}/{pronouns['them']}/{pronouns['their']} "
                "consistently for that person. Never use they, them, their, or plural agreement."
            )
        else:
            pronoun_contract = "Use pronouns and singular/plural agreement that exactly match the mapped subject count."
        if _storyboard_scene_is_visible_singing(selected_scene):
            vocal_contract = (
                "This is a visible singing/lip-sync scene. The performer must visibly vocalize the supplied lyric in sync with the audio, "
                "using natural mouth, lip, cheek, and jaw movement. Never describe closed, still, sealed, relaxed-closed, or unmoving lips."
            )
        else:
            vocal_contract = (
                "This is not a visible singing/lip-sync scene. Do not say any subject sings, vocalizes, mouths words, or lip-syncs, "
                "and do not quote the lyric as performed dialogue."
            )
        ltx_scene = str(selected_scene.get("project_video_engine") or scene_bundle.get("project_video_engine") or "").strip().lower() != "minimax_h3"
        ltx_one_pass_contract = (
            "AUTHORITATIVE LTX ONE-PASS OUTPUT CONTRACT — obey every item and output the finished prompt only:\n"
            "- Follow the editing/cut plan exactly. Write every required cut directly into ordinary chronological prose using 'then cut to' or equivalent natural wording; do not use MiniMax timestamps.\n"
            f"- {vocal_contract}\n"
            f"- {pronoun_contract}\n"
            "- Integrate facial and performance guidance naturally. Never print field names, headings, metadata labels, or phrases such as 'Facial performance direction:'.\n"
            "- The first sentence is the sole opening-shot statement. Continue with new action afterward; never restate that the subject is first shown, already shown, framed, introduced, or seen in that opening shot.\n"
            "- Describe each camera action once. Do not repeat the opening framing, reveal, pull-back, or other camera direction.\n"
            "- Use natural possessive anatomy phrasing such as 'the woman's eye' or 'the subject's eye'; never write 'one eye of the woman'.\n"
            "- Write only complete grammatical sentences. Attach short details with wording such as 'with subtle natural eye movement'; never append a bare comma fragment.\n"
            "- Treat first-frame visual inventory as optional visible detail, not a checklist. Mention only details inside the current framing; eye, face, and upper-body shots must not claim shoes, heels, feet, lower-body clothing, or other off-frame details are visible.\n"
            "- Return one polished generation-ready prompt. Do not explain these rules."
        ) if ltx_scene else ""
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
                f"MANDATORY editing / cut plan:\n{_clean_scene_text((selected_scene.get('cut_plan') or {}).get('instruction') if isinstance(selected_scene.get('cut_plan'), dict) else '', 5000)}",
                f"Required starting shot:\n{json.dumps(selected_scene.get('starting_shot'), ensure_ascii=False)}" if _storyboard_starting_shot_value(selected_scene) else "",
                f"Performance mode:\n{performance_mode}",
                f"Scene lyrics:\n{_clean_scene_text(vocal_status.get('lyric_text') or '', 1000)}",
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


def _authoritative_script_from_payload(payload):
    source = payload.get("script_import") or payload.get("scriptImport")
    if not source and isinstance(payload.get("storyboard"), dict):
        source = payload["storyboard"].get("script_import") or payload["storyboard"].get("scriptImport")
    normalized = _normalize_script_import(source or {})
    return normalized if normalized.get("enabled") and normalized.get("cues") else None


def _authoritative_script_text(script_import):
    raw_text = _clean_scene_text((script_import or {}).get("raw_text") or "", 100000)
    if raw_text:
        return raw_text
    return "\n".join(
        f'{cue.get("speaker_alias") or cue.get("speaker_name") or "Speaker"}: {cue.get("text") or ""}'
        for cue in (script_import or {}).get("cues") or []
        if cue.get("text")
    )


def _build_short_film_script_story_text(payload, script_import, purpose="premise"):
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    subjects, locations = _storyboard_dialogue_reference_catalog(payload)
    script_text = _authoritative_script_text(script_import)
    planned_scenes = (script_import.get("scene_plan") or {}).get("scenes") or []
    compact_plan = []
    for scene in planned_scenes[:240]:
        compact_plan.append({
            "segment": scene.get("index"),
            "duration_seconds": scene.get("duration_seconds"),
            "continuation_of_previous": bool(scene.get("continuation_of_previous")),
            "dialogue": [
                {
                    "speaker_id": cue.get("speaker_id", ""),
                    "speaker": cue.get("speaker_name") or cue.get("speaker_alias") or "Speaker",
                    "exact_text": cue.get("text", ""),
                }
                for cue in scene.get("speaker_assignments") or []
            ],
        })
    if purpose == "brief":
        task = (
            "Create a compact short-film production brief that Guided Film Automation can use to design visual scenes around the authoritative script. "
            "Use these headings exactly: Story premise:, Character dynamics:, Visual progression:, Continuity rules:, Scene-direction guidance:. "
            "Keep the complete response under 450 words."
        )
        max_tokens = 1000
        label = "Storyboard Authoritative Script Film Brief"
    else:
        task = (
            "Create one cohesive short-film premise and visual narrative arc from the authoritative script. Explain the dramatic situation, character goals, emotional progression, "
            "and how the film can develop visually across the supplied timed segments. Output plain prose under 500 words with no screenplay rewrite and no dialogue list."
        )
        max_tokens = 1100
        label = "Storyboard Authoritative Script Film Premise"
    instruction = (
        "You are a short-film development director working from a locked screenplay.\n\n"
        f"{task}\n\n"
        "NON-NEGOTIABLE SCRIPT CONTRACT:\n"
        "- The supplied dialogue is authoritative and immutable. Never rewrite, paraphrase, shorten, extend, reorder, merge, or invent spoken words.\n"
        "- Do not add narration, voice-over, replacement dialogue, or extra speakers.\n"
        "- Develop only the visual story around the locked dialogue: actions, reactions, motivations, blocking, locations, props, shots, camera language, atmosphere, and continuity.\n"
        "- Use Reference Builder character names and descriptions as identity authority.\n"
        "- Use only supplied Reference Builder locations when locations are available.\n\n"
        f"Optional user story idea:\n{story_layer.get('overall_story_idea') or '[none]'}\n\n"
        f"Reference Builder characters:\n{json.dumps(subjects, ensure_ascii=False, indent=2) if subjects else '[none]'}\n\n"
        f"Reference Builder locations:\n{json.dumps(locations, ensure_ascii=False, indent=2) if locations else '[none]'}\n\n"
        f"Authoritative exact script:\n{script_text}\n\n"
        f"Authoritative timed segment plan:\n{json.dumps(compact_plan, ensure_ascii=False, indent=2)}"
    )
    from .VRGDG_MusicVideoBuilderNodes import _run_builder_text_llm

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or max_tokens),
        label=label,
        preserve_paragraphs=True,
    )
    text = _clean_scene_text(text, 10000)
    if not text:
        raise ValueError("The LLM returned an empty short-film story result.")
    return text, run_info


def _build_story_layer_brief(payload):
    lyrics = _clean_scene_text(payload.get("lyrics") or payload.get("lyrics_text") or "", 16000)
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    authoritative_script = _authoritative_script_from_payload(payload)
    if authoritative_script:
        text, run_info = _build_short_film_script_story_text(payload, authoritative_script, "brief")
        return {
            "story_brief": text,
            "runner": run_info.get("runner", "builtin"),
            "used_model": run_info.get("used_model", ""),
            "unloaded": run_info.get("unloaded", True),
            "authoritative_script_used": True,
        }
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
            "mapped_extras": [
                {
                    "name": item.get("name", ""),
                    "count": item.get("count", 1),
                    "interaction": item.get("interaction", "background"),
                }
                for item in (normalized.get("extra_subjects") or [])
                if isinstance(item, dict) and item.get("name")
            ],
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
        f"{_lyric_story_strength_guidance(story_layer)}\n\n"
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


def _parse_story_arc_lyric_sections(lyrics, collapse_adjacent=True):
    """Return ordered (display label, body) pairs from bracketed lyric headers."""
    structural_pattern = re.compile(
        r"^(?:intro|verse|pre[\s-]?chorus|chorus|post[\s-]?chorus|bridge|outro|"
        r"refrain|hook|breakdown|drop|interlude|instrumental(?:\s+break)?|solo|break|"
        r"spoken(?:\s+word)?|rap)(?:\s+(?:\d+|[ivxlcdm]+))?$",
        re.IGNORECASE,
    )
    annotation_pattern = re.compile(
        r"^(?:whispered|spoken|sung|dark atmosphere|building energy|high energy|"
        r"emotional climax|explosive|quiet arrangement|falling tension|rising tension|"
        r"silence|soft|loud|gentle|intense|energetic|calm|dramatic|atmospheric)$",
        re.IGNORECASE,
    )

    def parse_header_line(raw_line):
        """Return (section label, lyric remainder, terminal marker)."""
        stripped = str(raw_line or "").strip()
        if not stripped.startswith("["):
            return "", raw_line, False
        labels = []
        position = 0
        while position < len(stripped):
            match = re.match(r"\s*\[([^\]\n]{1,80})\]", stripped[position:])
            if not match:
                break
            labels.append(re.sub(r"\s+", " ", match.group(1)).strip())
            position += match.end()
        if not labels:
            return "", raw_line, False
        remainder = stripped[position:].strip()
        terminal = any(label.casefold() in {"end", "end of song"} for label in labels)
        structural = next((label for label in labels if structural_pattern.fullmatch(label)), "")
        if not structural:
            first = labels[0]
            if not annotation_pattern.fullmatch(first) and first.casefold() not in {"end", "end of song"}:
                # Preserve custom section names such as [Part A], while avoiding
                # common performance/mood annotations used beside real headers.
                structural = first
        return structural, remainder, terminal and not structural

    sections = []
    current_label = ""
    current_lines = []
    for raw_line in str(lyrics or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        header_label, remainder, terminal = parse_header_line(raw_line)
        if header_label:
            if current_label:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = header_label
            current_lines = [remainder] if remainder else []
        elif terminal:
            if current_label:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = ""
            current_lines = []
        elif current_label:
            # Lines containing annotation-only tags may still carry lyric text.
            current_lines.append(remainder if remainder != raw_line else raw_line)
    if current_label:
        sections.append((current_label, "\n".join(current_lines).strip()))
    if not sections:
        return []

    # Timeline/storyboard payloads repeat the scene's section header before every
    # lyric chunk.  Treat adjacent copies as one real song section while keeping
    # later recurrences (for example, a chorus after Verse 2) as separate blocks.
    collapsed = []
    for label, body in sections:
        if collapse_adjacent and collapsed and collapsed[-1][0].casefold() == label.casefold():
            previous_label, previous_body = collapsed[-1]
            merged_body = "\n".join(part for part in (previous_body, body) if part).strip()
            collapsed[-1] = (previous_label, merged_body)
        else:
            collapsed.append((label, body))

    counts = {}
    numbered = []
    for label, body in collapsed:
        key = label.casefold()
        counts[key] = counts.get(key, 0) + 1
        occurrence = counts[key]
        display = label if occurrence == 1 else f"{label} {occurrence}"
        numbered.append((display, body))
    return numbered


def _cap_story_arc_words(text, maximum=100):
    words = re.findall(r"\S+", str(text or ""))
    if len(words) <= maximum:
        return " ".join(words)
    clipped = " ".join(words[:maximum])
    sentence_end = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if sentence_end >= max(80, len(clipped) // 2):
        return clipped[:sentence_end + 1].strip()
    return clipped.rstrip(" ,;:") + "…"


def _story_arc_section_word_limit(section_count):
    """Keep long song structures within the fixed Story Arc output budget."""
    try:
        count = max(0, int(section_count))
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return 100
    return max(30, min(100, 1500 // count))


class StoryArcFormatError(ValueError):
    """Format failure carrying bounded model output for the UI diagnostics panel."""

    def __init__(self, message, *, raw_output="", cleaned_output="", expected_sections=None, runner="LLM"):
        super().__init__(message)
        self.raw_output = str(raw_output or "")[-12000:]
        self.cleaned_output = str(cleaned_output or "")[-12000:]
        self.expected_sections = [str(item) for item in (expected_sections or [])]
        self.runner = str(runner or "LLM")


def _normalize_story_arc_output(text, required_labels, maximum_words=100, runner_label="LLM"):
    """Enforce the detected headings and configured per-section word limit."""
    raw = str(text or "").strip()
    runner_label = str(runner_label or "LLM").strip() or "LLM"
    heading_pattern = re.compile(r"(?m)^\s*([^\n:]{1,80}):\s*(?:\n|$)")
    matches = list(heading_pattern.finditer(raw))
    if not matches:
        if required_labels:
            raise ValueError(f"{runner_label} did not return the required lyric section headings.")
        return _cap_story_arc_words(raw, 100)
    blocks = []
    for index, match in enumerate(matches):
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        bracketed = re.fullmatch(r"\[([^\]\n]{1,80})\]", label)
        if bracketed:
            label = re.sub(r"\s+", " ", bracketed.group(1)).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        blocks.append((label, raw[match.end():end].strip()))
    if required_labels:
        required_keys = {label.casefold() for label in required_labels}
        nonstructural_scene_markers = {
            "instrumental", "instrumental section", "instrumental break",
            "break", "interlude", "solo", "music only", "no vocals",
            "no vocal", "silence", "b roll", "b-roll",
        }
        # The timeline uses values such as [instrumental] as scene-content
        # markers. Gemma can mistakenly promote one into a ninth story heading
        # even when it also returned every required lyric heading in order. Fold
        # that prose into the nearest real section instead of rejecting an
        # otherwise valid arc. Unknown invented headings still fail below.
        folded_blocks = []
        pending_prefix = []
        for label, body in blocks:
            key = label.casefold()
            if key not in required_keys and key in nonstructural_scene_markers:
                if folded_blocks:
                    previous_label, previous_body = folded_blocks[-1]
                    folded_blocks[-1] = (
                        previous_label,
                        "\n".join(part for part in (previous_body, body) if part).strip(),
                    )
                elif body:
                    pending_prefix.append(body)
                continue
            if pending_prefix:
                body = "\n".join([*pending_prefix, body] if body else pending_prefix).strip()
                pending_prefix = []
            folded_blocks.append((label, body))
        if pending_prefix and folded_blocks:
            last_label, last_body = folded_blocks[-1]
            folded_blocks[-1] = (
                last_label,
                "\n".join(part for part in (last_body, *pending_prefix) if part).strip(),
            )
        blocks = folded_blocks
        required = [label.casefold() for label in required_labels]
        meta_heading_pattern = re.compile(
            r"\b(?:user|instruction|requirement|preserve|output|heading|exact sections?|"
            r"requested format|response format|story arc request)\b",
            re.IGNORECASE,
        )
        # Qwen and some server models may echo a short instruction heading
        # before the requested answer. Ignore only clearly meta/instructional
        # preamble blocks; invented story sections remain strict failures.
        while blocks:
            first_key = blocks[0][0].casefold()
            first_matches_required = first_key == required[0] or first_key == f"{required[0]} 1"
            if first_matches_required or not meta_heading_pattern.search(blocks[0][0]):
                break
            blocks.pop(0)
        returned = [label.casefold() for label, _body in blocks]
        # Some local Gemma models add "1" to the first occurrence of a
        # heading even though only repeated occurrences are numbered.
        returned = [
            required[index] if index < len(required) and actual == f"{required[index]} 1" else actual
            for index, actual in enumerate(returned)
        ]
        if len(blocks) > len(required_labels) and returned[:len(required)] == required:
            # Gemma occasionally preserves every required lyric heading, then
            # appends invented sections such as an extra Instrumental or Outro.
            # The required prefix is already a complete valid story arc, so
            # discard only those trailing additions instead of failing the
            # entire generation. Missing, renamed, reordered, or interleaved
            # headings still fail the strict comparison below.
            blocks = blocks[:len(required_labels)]
            returned = returned[:len(required)]
        if returned != required:
            mismatch_index = next(
                (index for index, (expected, actual) in enumerate(zip(required, returned)) if expected != actual),
                min(len(required), len(returned)),
            )
            expected_label = required_labels[mismatch_index] if mismatch_index < len(required_labels) else "[none]"
            returned_label = blocks[mismatch_index][0] if mismatch_index < len(blocks) else "[missing]"
            missing = [label for label in required_labels if label.casefold() not in returned]
            extra = [label for label, _body in blocks if label.casefold() not in required]
            details = [
                f"Expected {len(required_labels)} headings but {runner_label} returned {len(blocks)}.",
                f"First mismatch at section {mismatch_index + 1}: expected '{expected_label}', received '{returned_label}'.",
            ]
            if missing:
                details.append("Missing: " + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "") + ".")
            if extra:
                details.append("Extra: " + ", ".join(extra[:8]) + ("…" if len(extra) > 8 else "") + ".")
            raise ValueError(
                f"{runner_label} changed the lyric structure. "
                + " ".join(details)
            )
        blocks = [
            (required_labels[index], body)
            for index, (_label, body) in enumerate(blocks)
        ]
    return "\n\n".join(
        f"{label}:\n{_cap_story_arc_words(body, maximum_words)}"
        for label, body in blocks
        if body
    )


def _build_story_layer_arc(payload):
    authoritative_script = _authoritative_script_from_payload(payload)
    if authoritative_script:
        text, run_info = _build_short_film_script_story_text(payload, authoritative_script, "premise")
        return {
            "story_arc": text,
            "lyrics_source": "Authoritative Script Mapper import",
            "story_arc_seed": _clean_scene_text(payload.get("story_arc_seed") or payload.get("storyArcSeed") or payload.get("seed") or "", 80),
            "runner": run_info.get("runner", "builtin"),
            "used_model": run_info.get("used_model", ""),
            "unloaded": run_info.get("unloaded", True),
            "authoritative_script_used": True,
        }
    storyboard = payload.get("storyboard") if isinstance(payload.get("storyboard"), dict) else {}
    timeline_lyrics = _clean_scene_text(payload.get("lyrics") or payload.get("lyrics_text") or "", 40000)
    line_mapping_lyrics = _clean_scene_text(payload.get("line_mapping_lyrics") or payload.get("lineMappingLyrics") or "", 40000)
    project_value = payload.get("project_folder") or payload.get("projectFolder") or ""
    project_folder = os.path.abspath(str(project_value).strip().strip('"')) if project_value else ""
    prompt_creator_lyrics = ""
    if project_folder:
        full_lyrics_path = os.path.join(project_folder, "project_context", "full_lyrics.txt")
        if os.path.isfile(full_lyrics_path):
            try:
                with open(full_lyrics_path, "r", encoding="utf-8-sig") as handle:
                    prompt_creator_lyrics = _clean_scene_text(handle.read(), 40000)
            except OSError:
                prompt_creator_lyrics = ""
    # The lyrics currently open in Line Mapping are the live user input and must
    # win over a potentially stale project_context/full_lyrics.txt file.
    lyrics = line_mapping_lyrics or prompt_creator_lyrics or timeline_lyrics
    lyrics_source = "Line Mapping reference lyrics" if line_mapping_lyrics else ("Prompt Creator reference lyrics" if prompt_creator_lyrics else "timeline scene lyrics")
    lyric_sections = _parse_story_arc_lyric_sections(
        lyrics,
        collapse_adjacent=not bool(line_mapping_lyrics or prompt_creator_lyrics),
    )
    required_section_labels = [item[0] for item in lyric_sections]
    section_word_limit = _story_arc_section_word_limit(len(required_section_labels))
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or storyboard.get("story_layer") or {})
    story_idea = _clean_scene_text(payload.get("story_idea") or payload.get("storyIdea") or story_layer.get("overall_story_idea") or "", 4000)
    story_arc_seed = _clean_scene_text(payload.get("story_arc_seed") or payload.get("storyArcSeed") or payload.get("seed") or "", 80)
    previous_story_arc = _clean_scene_text(payload.get("previous_story_arc") or payload.get("previousStoryArc") or "", 5000)
    style_theme = _clean_scene_text(payload.get("style_theme") or payload.get("styleTheme") or payload.get("theme") or "", 1600)
    performance_style = _clean_scene_text(payload.get("performance_style") or payload.get("performanceStyle") or storyboard.get("performance_style_default") or "", 200)
    facial_performance = _clean_scene_text(payload.get("facial_performance") or payload.get("facialPerformance") or storyboard.get("facial_performance_default") or "", 200)
    camera_flow = _clean_scene_text(payload.get("camera_flow") or payload.get("cameraFlow") or storyboard.get("camera_flow") or "", 200)
    try:
        camera_motion_speed = int(float(payload.get("camera_motion_speed", payload.get("cameraMotionSpeed", storyboard.get("camera_motion_speed", 4)))))
    except Exception:
        camera_motion_speed = 4
    camera_motion_speed = max(0, min(10, camera_motion_speed))
    try:
        character_motion = int(float(payload.get(
            "character_motion",
            payload.get("characterMotion", payload.get("character_motion_speed", payload.get("characterMotionSpeed", storyboard.get("character_motion_speed", 7))))
        )))
    except Exception:
        character_motion = 7
    character_motion = max(0, min(10, character_motion))
    if character_motion <= 2:
        motion_guidance = (
            "Character motion level: mostly still. The singer may hold poses, but each section still needs a visible micro-action "
            "such as reaching, turning, touching fabric, shifting weight, raising a hand, or interacting with one object."
        )
    elif character_motion <= 5:
        motion_guidance = (
            "Character motion level: moderate. Give the singer controlled performance movement: steps, turns, gestures, changes in blocking, "
            "and occasional interaction with the set."
        )
    elif character_motion <= 8:
        motion_guidance = (
            "Character motion level: active. The singer should usually move through the location, walk, approach, retreat, touch objects, "
            "use architecture, cross rooms, lean into weather, or physically interact with the environment."
        )
    else:
        motion_guidance = (
            "Character motion level: highly active. Build big physical beats: dancing, running, climbing, struggling, sweeping gestures, "
            "forceful environmental interaction, or kinetic performance movement."
        )
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
    if required_section_labels:
        structure_instruction = (
            "The reference lyrics contain explicit section headers. Preserve exactly this section order and these output headings; "
            "do not add, remove, merge, rename, or invent sections:\n"
            + "\n".join(f"- {label}" for label in required_section_labels)
            + "\nRepeated sections have been numbered by occurrence so every real section has its own summary."
        )
    else:
        structure_instruction = (
            "The reference lyrics do not contain explicit section headers. Infer a sensible compact song structure from lyrical, "
            "emotional, and narrative changes. Do not add an Intro or Outro unless the supplied material clearly supports one."
        )
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
        "Your output should be clean, complete, and easy to use. Break it down by the actual song structure.\n\n"
        f"{structure_instruction}\n\n"
        "Format every section as its heading on one line ending in a colon, followed by one prose paragraph.\n\n"
        "Rules:\n"
        f"* Fully summarize the visual story progression of each section in no more than {section_word_limit} words.\n"
        "* Use the entire section, not only its first lyric line.\n"
        "* Do not summarize the lyrics line by line.\n"
        "* Turn the lyrics into a simple visual story arc.\n"
        "* Each section should be a cohesive visual-story paragraph, not a line-by-line list.\n"
        "* Use cinematic, visual language.\n"
        "* The main character or singer should not default to standing still, standing alone, staring, looking, being framed, or holding a pose.\n"
        "* Unless the character motion level is very low, every section must include a distinct physical action by the singer or main character.\n"
        "* Vary the action between sections. Avoid repeating stand, stare, gaze, look, walk, or turn as the only beat.\n"
        "* Make the location support the action; do not let the location be the whole story beat.\n"
        "* If Location descriptions are provided, use only those locations as the physical settings for the arc.\n"
        "* Do not invent warehouses, loading docks, corridors, steel stairs, metal doors, concrete halls, or other industrial spaces unless those are explicitly present in the provided Location descriptions.\n"
        "* To create variety, change subject actions, camera energy, props, lighting, mood, blocking, and use of the mapped locations instead of inventing unrelated places.\n"
        "* The Scene lyric map may include mapped_extras. Use those exact extras only in the scenes where they are mapped, and use each interaction role to shape the section's larger action progression.\n"
        "* Plan recurring extra relationships across sections when the same named extra returns, especially direct, dancing_with, and alongside roles. Background and background_dancing extras may support group progression without becoming principal singers.\n"
        "* Extras never sing, speak, or receive dialogue unless the scene mapping explicitly identifies them as a vocal subject elsewhere. Do not move an extra into an unmapped scene.\n"
        "* Never force a standard pop-song template over explicit lyric section headers.\n"
        "* When a lyric header line contains several bracketed tags, only the required section heading listed above is structural. Tags such as [Whispered], [High Energy], [Dark Atmosphere], and [Explosive] are performance or mood notes, not output headings. [End] is only an end marker.\n"
        "* Values such as [instrumental], [music only], or [no vocals] inside the Scene lyric map are timing/content markers, not additional song-section headings. Cover their visuals inside the nearest listed required section and never output those markers as separate headings.\n"
        "* If only lyrics are provided, build the arc from the lyrics.\n"
        "* If no lyrics are provided, build the arc from the theme or story idea.\n"
        "* Do not ask follow-up questions unless absolutely necessary.\n"
        "* Output only the story arc sections. No intro note, no markdown table, no JSON.\n\n"
        f"Creative variation seed: {story_arc_seed or '[none]'}\n"
        "Use this seed as a reroll key. If the user regenerates the story arc with a different seed, choose a meaningfully different visual interpretation, section action pattern, and location usage while still respecting the same subjects, locations, lyrics, and style.\n\n"
        f"Scene default style settings:\n"
        f"- Camera flow: {camera_flow or '[not provided]'}\n"
        f"- Camera motion speed: {camera_motion_speed}/10\n"
        f"- Character motion speed: {character_motion}/10\n"
        f"- Performance style: {performance_style or '[not provided]'}\n"
        f"- Facial performance: {facial_performance or '[not provided]'}\n\n"
        f"{motion_guidance}\n\n"
        f"{_lyric_story_strength_guidance(story_layer)}\n\n"
        f"Story idea:\n{story_idea or '[not provided]'}\n\n"
        f"Previous generated story arc to avoid copying:\n{previous_story_arc or '[not provided]'}\n\n"
        "If a previous generated story arc is provided, do not preserve its specific locations, set pieces, or section actions. Use it only as a negative example of what should change on this reroll.\n\n"
        f"Style/theme:\n{style_theme or '[not provided]'}\n\n"
        f"Character descriptions:\n{json.dumps(subjects[:24], ensure_ascii=False, indent=2) if subjects else '[not provided]'}\n\n"
        f"Location descriptions:\n{json.dumps(locations[:40], ensure_ascii=False, indent=2) if locations else '[not provided]'}\n\n"
        f"Authoritative lyric source: {lyrics_source}\n"
        f"Full reference lyrics:\n{lyrics or '[not provided]'}\n\n"
        f"Scene lyric map:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2) if compact_scenes else '[not provided]'}"
    )
    from .VRGDG_MusicVideoBuilderNodes import _llm_runner_display_name, _run_builder_text_llm

    runner_label = _llm_runner_display_name(payload)

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.92),
        max_new_tokens=int(payload.get("max_new_tokens") or 2400),
        label=f"Storyboard Story Arc {runner_label}",
        preserve_paragraphs=True,
    )
    text = _clean_scene_text(text, 14000)
    if not text:
        raise ValueError(f"{runner_label} returned an empty story arc.")
    try:
        text = _normalize_story_arc_output(text, required_section_labels, section_word_limit, runner_label)
    except ValueError as first_error:
        if not required_section_labels:
            raise
        exact_format = "\n\n".join(f"{label}:\n[one visual-story paragraph]" for label in required_section_labels)
        retry_instruction = (
            "CORRECTION: Your previous answer did not follow the required lyric-section output format.\n"
            "Answer the original task again from scratch. Do not discuss, quote, summarize, or acknowledge these instructions.\n"
            f"The very first line must be exactly: {required_section_labels[0]}:\n"
            "Return every required heading exactly once and in this exact order, with no preamble, notes, bullets, or extra headings.\n\n"
            f"Exact output skeleton:\n{exact_format}\n\n"
            f"Original task:\n{instruction}"
        )
        retry_payload = dict(payload or {})
        try:
            retry_payload["seed"] = (int(payload.get("seed") or 0) + 1) % 2147483647
        except (TypeError, ValueError):
            retry_payload["seed"] = 1
        retry_text, retry_info = _run_builder_text_llm(
            retry_payload,
            retry_instruction,
            temperature=0.2,
            top_p=0.85,
            max_new_tokens=int(payload.get("max_new_tokens") or 2400),
            label=f"Storyboard Story Arc {runner_label} format retry",
            preserve_paragraphs=True,
        )
        retry_text = _clean_scene_text(retry_text, 14000)
        try:
            text = _normalize_story_arc_output(
                retry_text,
                required_section_labels,
                section_word_limit,
                runner_label,
            )
            run_info = retry_info
        except ValueError as retry_error:
            raise StoryArcFormatError(
                f"{runner_label} could not preserve the lyric-section structure after an automatic format retry. "
                f"{retry_error}",
                raw_output=retry_text or text,
                cleaned_output=retry_text or text,
                expected_sections=required_section_labels,
                runner=runner_label,
            ) from first_error
    return {
        "story_arc": text,
        "lyrics_source": lyrics_source,
        "story_arc_seed": story_arc_seed,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


_STORYBOARD_DRIFT_LOCATION_PATTERNS = [
    (r"\bwarehouse\b", "warehouse"),
    (r"\bloading\s+dock\b", "loading dock"),
    (r"\bindustrial\b", "industrial"),
    (r"\bbackstage\s+corridor\b", "backstage corridor"),
    (r"\bnarrow,\s*dimly\s*lit\s+corridor\b", "dimly lit corridor"),
    (r"\bdark,\s*narrow\s+corridor\b", "dark corridor"),
    (r"\bheavy\s+(?:metal|steel)\s+door\b", "heavy metal/steel door"),
    (r"\bmassive\s+window\b", "massive window"),
    (r"\bconcrete\b", "concrete"),
    (r"\bmetal\s+pipes?\b", "metal pipes"),
    (r"\bsteel\s+stairs?\b", "steel stairs"),
    (r"\bvast,\s*silent\s+hall\b", "vast hall"),
    (r"\bvast\s+empty\s+space\b", "vast empty space"),
]


def _storyboard_scene_location_context(scene):
    if not isinstance(scene, dict):
        return ""
    location = scene.get("location_ref") if isinstance(scene.get("location_ref"), dict) else {}
    parts = []
    if isinstance(location, dict):
        parts.extend([
            location.get("name"),
            location.get("description"),
            location.get("trigger_phrase") or location.get("trigger") or location.get("Trigger"),
        ])
    parts.extend([scene.get("setting"), scene.get("location")])
    return _clean_scene_text(" ".join(str(part or "") for part in parts if str(part or "").strip()), 2400)


def _storyboard_location_drift_terms(text, location_context):
    text_lower = str(text or "").lower()
    location_lower = str(location_context or "").lower()
    if not text_lower or not location_lower:
        return []
    drift_terms = []
    for pattern, label in _STORYBOARD_DRIFT_LOCATION_PATTERNS:
        if re.search(pattern, text_lower, flags=re.IGNORECASE) and not re.search(pattern, location_lower, flags=re.IGNORECASE):
            drift_terms.append(label)
    return drift_terms


def _parse_flf_endpoint_json(text):
    raw = str(text or "").strip()
    raw = re.sub(
        r"^\s*[^A-Za-z0-9]*(?:(?:user|assistant|model)\b)?[^A-Za-z0-9]*(?:thought|analysis|reasoning)(?=[A-Z]|[^A-Za-z0-9]|$)[^A-Za-z0-9]*",
        "",
        raw,
        flags=re.I,
    ).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    match = re.search(r"\{.*\}", raw, flags=re.S)
    candidate = match.group(0) if match else raw
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)


def _build_story_layer_scene_beat(payload):
    scene_bundle = payload.get("storyboard_payload") or payload.get("scene_bundle") or payload.get("gpt_payload")
    if not isinstance(scene_bundle, dict):
        raise ValueError("Storyboard scene-card payload is missing.")
    scene = _selected_storyboard_scene(scene_bundle)
    if not scene:
        raise ValueError("Storyboard scene-card payload has no selected scene.")
    story_layer = _normalize_story_layer(payload.get("story_layer") or scene_bundle.get("story_layer") or {})
    previous_beat = _clean_scene_text(payload.get("previous_beat") or "", 1200)
    previous_lyrics = _clean_scene_text(payload.get("previous_lyrics") or "", 800)
    previous_end_state = _clean_scene_text(payload.get("previous_end_state") or "", 1800)
    previous_carry_forward = _clean_scene_text(payload.get("previous_carry_forward") or "", 1800)
    current_lyrics = _clean_scene_text(payload.get("current_lyrics") or scene.get("lyrics") or scene.get("lyric_text") or "", 1200)
    next_lyrics = _clean_scene_text(payload.get("next_lyrics") or "", 800)
    flf_mode = bool(payload.get("flf_mode")) or str(scene.get("video_prompt_type") or "").strip().lower() == "flf"
    vocal_status = scene.get("vocal_status") if isinstance(scene.get("vocal_status"), dict) else {}
    # Scene-beat generation is also called from Image Prep, where the normal
    # prompt payload intentionally clears vocal_status.singers. The storyboard
    # scene's lyric_singers is the performer assignment that remains valid in
    # both Image Prep and Video Prep.
    assigned_performers = scene.get("lyric_singers") if isinstance(scene.get("lyric_singers"), list) else []
    if not assigned_performers and isinstance(vocal_status.get("singers"), list):
        assigned_performers = vocal_status.get("singers")
    assigned_performers = [_clean_scene_text(item, 180) for item in assigned_performers if _clean_scene_text(item, 180)]
    performer_assignment = scene.get("performer_assignment") if isinstance(scene.get("performer_assignment"), dict) else {}
    assigned_performers = [
        _clean_scene_text(item, 180)
        for item in (performer_assignment.get("singing") if isinstance(performer_assignment.get("singing"), list) else assigned_performers)
        if _clean_scene_text(item, 180)
    ]
    if not assigned_performers:
        mapped_subject_names = []
        for item in scene.get("subject_refs") or []:
            if isinstance(item, dict):
                name = _clean_scene_text(item.get("name") or "", 180)
                if name:
                    mapped_subject_names.append(name)
        for item in scene.get("subjects") or []:
            name = _clean_scene_text(item.get("name") if isinstance(item, dict) else item, 180)
            if name:
                mapped_subject_names.append(name)
        singer_named_subjects = [
            name for name in dict.fromkeys(mapped_subject_names)
            if re.search(r"\b(?:singer|performer|vocalist|rapper)\b", name, flags=re.IGNORECASE)
        ]
        if len(singer_named_subjects) == 1:
            assigned_performers = singer_named_subjects
    silent_performers = [
        _clean_scene_text(item, 180)
        for item in (performer_assignment.get("silent") if isinstance(performer_assignment.get("silent"), list) else [])
        if _clean_scene_text(item, 180)
    ]
    scene_defaults = {
        "shot_type": _clean_scene_text(scene.get("shot_type") or scene.get("shot") or "", 240),
        "camera_motion": _clean_scene_text(scene.get("camera_motion") or scene.get("camera_motion_preset") or "", 500),
        "camera_flow": _clean_scene_text(scene.get("camera_flow") or scene.get("cameraFlow") or "", 120),
        "camera_flow_guidance": _clean_scene_text(scene.get("camera_flow_guidance") or "", 1200),
        "character_motion": _clean_scene_text(scene.get("character_motion") or scene.get("character_motion_preset") or "", 700),
        "performance_direction": _clean_scene_text(scene.get("performance_direction") or scene.get("performance_style") or "", 1000),
        "facial_performance_direction": _clean_scene_text(scene.get("facial_performance_direction") or scene.get("facial_performance_custom") or scene.get("facial_performance") or "", 1200),
    }
    raw_extra_subjects = scene.get("extra_subjects") or scene.get("extraSubjects") or []
    extra_subjects = []
    if isinstance(raw_extra_subjects, list):
        for index, item in enumerate(raw_extra_subjects[:100], start=1):
            if not isinstance(item, dict):
                continue
            name = _clean_scene_text(item.get("name") or item.get("title") or f"Extra {index}", 180)
            if not name:
                continue
            interaction = str(item.get("interaction") or "background").strip()
            if interaction not in {"background", "background_dancing", "alongside", "dancing_with", "direct"}:
                interaction = "background"
            try:
                count = max(1, min(100, int(round(float(item.get("count") or 1)))))
            except (TypeError, ValueError):
                count = 1
            extra_subjects.append({
                "name": name,
                "count": count,
                "interaction": interaction,
                "identity": _clean_scene_text(item.get("identity") or item.get("description") or "", 240),
            })
    if scene.get("no_character_present") or scene.get("noCharacterPresent") or scene.get("no_visible_subject") or scene.get("no_subject"):
        extra_subjects = []
    beat_word_limit = 100 if extra_subjects else 80
    output_rules = (
        "Return valid JSON only with exactly these string keys: story_beat, flf_start_state, flf_transformation, flf_end_state, flf_carry_forward.\n"
        f"The story_beat is a concise compatibility summary under {beat_word_limit} words.\n"
        "flf_start_state describes the concrete visible opening image. If Previous FLF end state is provided, copy it exactly as flf_start_state; do not reinterpret or redesign it.\n"
        "flf_transformation describes one continuous, progressive visual change that expresses the CURRENT lyric.\n"
        "flf_end_state describes the concrete visible destination image reached by the end of the CURRENT lyric.\n"
        "flf_carry_forward records the subject, anatomy, wardrobe, props, setting, lighting, and transformation state that the next scene must inherit.\n"
        "The current lyric is authoritative. Previous and next lyrics provide continuity only and must not replace or steal this scene's action.\n"
        "Do not include Markdown fences or any text outside the JSON object."
        if flf_mode else
        f"Output one short paragraph only, no label, no bullets.\nKeep it under {beat_word_limit} words."
    )
    instruction = (
        "You are a music video scene-story planner.\n"
        "Create one concise scene story beat that tells the video prompt writer what this scene contributes to the larger music-video story.\n\n"
        "Rules:\n"
        "- Use the Song Story Brief and User Story Arc as continuity anchors.\n"
        "- Use the selected scene lyrics, lyric section, subject details, location details, vocal status, and no-character flag.\n"
        "- Vocal casting is absolute: follow the Performer assignment exactly. Only names in its singing list may sing. Every name in its silent list must remain visibly present but silent. Never write that both/all visible subjects sing unless both/all are explicitly in the singing list. If the singing list has one name, use singular wording: that performer sings; the other subject does not sing.\n"
        "- This request creates a narrative Scene Story Beat, not the final still-image prompt. Ignore any Image Prep instruction saying that subjects must be silent or that singing must not be mentioned. For this beat, use the Performer assignment: if one performer is assigned, that performer visibly sings while every other visible subject remains silent.\n"
        "- The existing scene story beat, if present in the selected scene JSON, is stale draft text being replaced. Do not copy, preserve, or treat it as a fact; the Performer assignment and current scene data override it.\n"
        "- Scene defaults are authoritative when supplied: use the selected shot, camera motion/camera-flow direction, character motion, performance direction, and facial direction to shape the beat. Do not replace them with generic actions.\n"
        "- Treat the selected scene location_ref as the required physical setting for this scene.\n"
        "- Do not invent or import a different place from the story arc, song brief, previous beat, or next lyrics.\n"
        "- If the story arc names a different location, translate only its emotion, tension, symbolism, or action into the selected location_ref.\n"
        "- Describe narrative purpose, emotional state, visual symbolism, and how the scene should feel.\n"
        "- Use every mapped extra listed below in the scene's action or blocking. Keep each exact extra name visible in the beat.\n"
        "- Interaction meanings are exact: background stays present without active choreography; background_dancing performs backup choreography; alongside moves beside the main subject without contact; dancing_with performs partnered or group choreography with the main subject; direct performs an explicit physical or narrative interaction.\n"
        "- Describe direct, dancing_with, and alongside extras individually. Extras sharing background or background_dancing may be combined into one concise named group.\n"
        "- Use an extra's identity only when needed to distinguish people. Do not copy full appearance or wardrobe biographies into the beat.\n"
        "- Extras do not sing, speak, or receive speaker IDs unless the selected scene explicitly supplies them as vocal sources elsewhere.\n"
        "- Do not write the final video prompt.\n"
        "- Do not include camera technical instructions unless they are part of the story emotion.\n"
        "- Do not quote long lyric text.\n"
        "- If no character is present, make the beat about location, objects, atmosphere, memory, or symbolism.\n"
        f"- {output_rules}\n\n"
        f"{_lyric_story_strength_guidance(story_layer)}\n\n"
        f"User Story Arc:\n{story_layer.get('user_story_arc') or '[none]'}\n\n"
        f"Song Story Brief:\n{story_layer.get('song_story_brief') or '[none]'}\n\n"
        f"Previous scene beat:\n{previous_beat or '[none]'}\n\n"
        f"Previous scene lyric text (continuity only):\n{previous_lyrics or '[none]'}\n\n"
        f"Previous FLF end state (required opening state when present):\n{previous_end_state or '[none — this is the first scene]'}\n\n"
        f"Previous FLF carry-forward constraints:\n{previous_carry_forward or '[none]'}\n\n"
        f"CURRENT scene lyric text (main authority):\n{current_lyrics or '[none]'}\n\n"
        f"Next scene lyric text:\n{next_lyrics or '[none]'}\n\n"
        f"Scene defaults and motion direction (authoritative when supplied):\n{json.dumps(scene_defaults, ensure_ascii=False, indent=2)}\n\n"
        f"Assigned singing performers (the only subjects allowed to sing):\n{json.dumps(assigned_performers, ensure_ascii=False)}\n\n"
        f"Performer assignment contract:\n{json.dumps({'singing': assigned_performers, 'silent': silent_performers}, ensure_ascii=False, indent=2)}\n\n"
        f"Mapped extras and exact scene roles:\n{json.dumps(extra_subjects, ensure_ascii=False, indent=2) if extra_subjects else '[none]'}\n\n"
        "Selected scene JSON:\n"
        + json.dumps(scene, ensure_ascii=False, indent=2)
        + "\n\nFINAL SCENE-BEAT OVERRIDE — FOLLOW THIS LAST:\n"
        + f"This is a narrative scene beat, not an Image Prep still-image prompt. Assigned singing performers: {json.dumps(assigned_performers, ensure_ascii=False)}. Assigned silent visible subjects: {json.dumps(silent_performers, ensure_ascii=False)}. "
        + (f"Only {assigned_performers[0]} visibly sings the current lyric; every other visible subject is silent. Do not write that both subjects sing."
           if len(assigned_performers) == 1 else
           "No mapped subject sings; keep all visible subjects silent."
           if not assigned_performers else
           f"Only these performers visibly sing: {', '.join(assigned_performers)}; every other visible subject is silent.")
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
    flf_fields = {}
    if flf_mode:
        try:
            parsed = _parse_flf_endpoint_json(text)
        except Exception as parse_error:
            repair_instruction = (
                "Repair the malformed FLF endpoint response below into valid JSON.\n"
                "Return JSON only: no thought text, prose, Markdown, or code fences.\n"
                "Use exactly these five string keys: story_beat, flf_start_state, flf_transformation, flf_end_state, flf_carry_forward.\n"
                "Preserve the original meaning and wording as closely as possible. Escape quotation marks inside strings and include every comma, colon, quote, and closing brace required by strict JSON.\n"
                "If a field was cut off or omitted, reconstruct it concisely from the other fields and selected scene context.\n\n"
                f"MALFORMED RESPONSE:\n{text}\n\n"
                f"SELECTED SCENE CONTEXT:\n{json.dumps(scene, ensure_ascii=False)}"
            )
            repaired_text, repair_info = _run_builder_text_llm(
                payload,
                repair_instruction,
                temperature=0.05,
                top_p=0.75,
                max_new_tokens=max(900, int(payload.get("max_new_tokens") or 360)),
                label="Storyboard FLF Endpoint JSON Repair",
                preserve_paragraphs=True,
            )
            try:
                parsed = _parse_flf_endpoint_json(repaired_text)
                run_info = {**run_info, "json_repaired": True, "repair_runner": repair_info.get("runner", "")}
            except Exception as repair_error:
                raise ValueError(
                    f"Gemma returned malformed FLF endpoint JSON and automatic repair failed. "
                    f"Original parse error: {parse_error}; repair parse error: {repair_error}"
                ) from repair_error
        flf_fields = {
            key: _clean_scene_text(parsed.get(key) or "", 1800)
            for key in ("flf_start_state", "flf_transformation", "flf_end_state", "flf_carry_forward")
        }
        if previous_end_state:
            flf_fields["flf_start_state"] = previous_end_state
        text = _clean_scene_text(parsed.get("story_beat") or "", 1800)
        missing = [key for key, value in flf_fields.items() if not value]
        if not text or missing:
            raise ValueError("Gemma returned incomplete FLF endpoint fields: " + ", ".join((["story_beat"] if not text else []) + missing))
    else:
        text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(text, 1800), flags=re.I)
    if not text:
        raise ValueError("Gemma returned an empty scene story beat.")
    location_context = _storyboard_scene_location_context(scene)
    drift_terms = _storyboard_location_drift_terms(text, location_context)
    if drift_terms:
        repair_instruction = (
            "Rewrite the scene story beat so it obeys the mapped location.\n\n"
            "Hard rules:\n"
            "- Keep the same emotional purpose and subject energy.\n"
            "- Use only the mapped location as the physical setting.\n"
            "- Remove every incompatible place/object listed below.\n"
            "- Do not mention a warehouse, loading dock, industrial corridor, metal door, concrete hall, steel stairs, pipes, or massive window unless those details are explicitly in the mapped location.\n"
            "- Output one short paragraph only, under 80 words.\n\n"
            f"Mapped location:\n{location_context or '[none]'}\n\n"
            f"Incompatible leaked location terms:\n{', '.join(drift_terms)}\n\n"
            f"Original scene beat:\n{text}"
        )
        repaired_text, repair_info = _run_builder_text_llm(
            payload,
            repair_instruction,
            temperature=0.20,
            top_p=0.85,
            max_new_tokens=300,
            label="Storyboard Scene Beat Location Repair Gemma",
            preserve_paragraphs=True,
        )
        repaired_text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(repaired_text, 1800), flags=re.I)
        repaired_drift_terms = _storyboard_location_drift_terms(repaired_text, location_context)
        if repaired_text and not repaired_drift_terms:
            text = repaired_text
            run_info = {
                **run_info,
                "location_repaired": True,
                "location_repair_terms": drift_terms,
                "location_repair_runner": repair_info.get("runner", ""),
                "location_repair_model": repair_info.get("used_model", ""),
            }
        else:
            subject_hint = "The scene subject" if not scene.get("subject_refs") else _clean_scene_text((scene.get("subject_refs") or [{}])[0].get("name") or "The scene subject", 120)
            text = (
                f"{subject_hint} channels the story arc's defiant, boundary-breaking energy inside {location_context}. "
                "The beat focuses on tension, control, and release through posture, expression, and interaction with the mapped studio environment, without changing the physical location."
            )
            run_info = {
                **run_info,
                "location_repaired": True,
                "location_repair_terms": drift_terms,
                "location_repair_fallback": True,
            }
    missing_extras = [item["name"] for item in extra_subjects if item["name"].casefold() not in text.casefold()]
    if missing_extras:
        repair_instruction = (
            "Rewrite this music-video scene beat so it includes every mapped extra by exact name and gives each the assigned action/blocking role.\n\n"
            "Hard rules:\n"
            "- Preserve the original narrative purpose, mapped location, main subject action, and emotional progression.\n"
            "- Include every exact extra name from the mapping.\n"
            "- Apply each interaction role exactly. Group only extras sharing background or background_dancing.\n"
            "- Extras do not sing, speak, or receive speaker IDs.\n"
            "- Use identity details only when required to distinguish characters; do not copy wardrobe biographies.\n"
            "- Output one concise paragraph only, under 100 words, with no label or bullets.\n\n"
            f"Mapped extras:\n{json.dumps(extra_subjects, ensure_ascii=False, indent=2)}\n\n"
            f"Original scene beat:\n{text}"
        )
        repaired_text, repair_info = _run_builder_text_llm(
            payload,
            repair_instruction,
            temperature=0.15,
            top_p=0.82,
            max_new_tokens=360,
            label="Storyboard Scene Beat Extra Mapping Repair Gemma",
            preserve_paragraphs=True,
        )
        repaired_text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(repaired_text, 1800), flags=re.I)
        still_missing = [item["name"] for item in extra_subjects if item["name"].casefold() not in repaired_text.casefold()]
        if not repaired_text or still_missing:
            raise ValueError("Gemma omitted mapped extras from the scene story beat after repair: " + ", ".join(still_missing or missing_extras))
        text = repaired_text
        run_info = {
            **run_info,
            "extra_mapping_repaired": True,
            "extra_mapping_repair_runner": repair_info.get("runner", ""),
            "extra_mapping_repair_model": repair_info.get("used_model", ""),
        }
    return {
        "story_beat": text,
        **flf_fields,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


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


def _id_lora_structured_image_prompt(item, subject_ref=None, location_ref=None):
    raw_prompt = _clean_scene_text(item.get("image_prompt") or item.get("visual_prompt") or "", 3000)
    words = re.findall(r"[A-Za-z0-9']+", raw_prompt)
    has_rich_prompt = (
        len(words) >= 45
        and re.search(r"\b(close-up|medium close-up|upper body|waist-up|portrait|profile|over-the-shoulder|low-angle|lens|lighting|depth of field|bokeh|palette|texture|cinematic)\b", raw_prompt, re.IGNORECASE)
    )
    if has_rich_prompt:
        return raw_prompt

    subject_ref = subject_ref if isinstance(subject_ref, dict) else {}
    location_ref = location_ref if isinstance(location_ref, dict) else {}
    subject_name = _clean_scene_text(item.get("character_name") or item.get("speaker") or subject_ref.get("name") or "the speaking character", 160)
    subject_description = _clean_scene_text(subject_ref.get("description") or item.get("character_description") or "", 900)
    location_name = _clean_scene_text(item.get("setting") or item.get("location_name") or location_ref.get("name") or "the scene location", 160)
    location_description = _clean_scene_text(location_ref.get("description") or item.get("location_description") or "", 900)
    shot_type = _clean_scene_text(item.get("shot_type") or "cinematic medium close-up", 120)
    visual_direction = _clean_scene_text(item.get("visual_direction") or item.get("summary") or item.get("story_beat") or item.get("beat") or "", 1000)
    facial = _clean_scene_text(item.get("facial_performance_custom") or item.get("facial_performance") or item.get("emotion") or item.get("delivery") or "", 500)

    has_subject_image = bool((subject_ref.get("image") or {}).get("path") or (subject_ref.get("image") or {}).get("name"))
    has_location_image = bool((location_ref.get("image") or {}).get("path") or (location_ref.get("image") or {}).get("name"))
    if has_subject_image and has_location_image:
        opening = "Using the provided character reference and location reference, create"
    elif has_subject_image:
        opening = "Using the provided character reference, create"
    elif has_location_image:
        opening = "Using the provided location reference, create"
    else:
        opening = "Create"

    subject_clause = f"{subject_name}"
    if subject_description:
        subject_clause = f"{subject_clause}, preserving {subject_description}"
    location_clause = f"in {location_name}"
    if location_description:
        location_clause = f"{location_clause}, with {location_description}"
    action_clause = visual_direction or "a tense dialogue-first short-film moment"
    face_clause = f" Give the face/body language {facial}." if facial else ""
    prompt = (
        f"{opening} a {shot_type} of {subject_clause} {location_clause}. "
        f"Stage the still frame around {action_clause}.{face_clause} "
        "Use a new pose and camera angle, shallow depth of field, practical cinematic lighting, textured materials, atmospheric haze or background separation, a deliberate color palette, crisp facial detail, and high cinematic image quality. "
        "No captions, no text overlays, no dialogue printed in the image."
    )
    return _clean_scene_text(re.sub(r"\s+", " ", prompt), 3000)


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
    planner_identity = (
        "You are the dedicated MiniMax H3 short-film scene planner. Create model-ready scene cards for a dialogue-driven MiniMax project."
        if is_minimax else
        "You are a short-film dialogue scene planner for an ID-LoRA image-to-video workflow."
    )
    dialogue_rule = (
        "- A scene may contain one or more ordered dialogue_cues. Use exact character ids from AVAILABLE CHARACTERS. Keep every cue short enough to fit naturally in one generated clip.\n"
        "- When two or more characters speak in one scene, preserve their exact turn order in dialogue_cues and give each character only their own words.\n"
        if is_minimax else
        "- Create exact spoken dialogue lines. Keep each line short enough for a single generated clip.\n"
        "- Prefer one speaking character per scene. Use only character ids from AVAILABLE CHARACTERS when possible.\n"
    )
    authoritative_rule = (
        "AUTHORITATIVE SCRIPT CONTRACT:\n"
        "- The SCRIPT MAPPER SEGMENT PLAN below is immutable. Return exactly one scene per supplied segment, in the same order.\n"
        "- Copy every dialogue cue word-for-word with its exact character_id and speaker. Never rewrite, paraphrase, correct, shorten, extend, merge, reorder, or invent dialogue.\n"
        "- Do not add narration, voice-over, new speakers, or extra spoken words.\n"
        "- Your creative job is only the surrounding visual direction: story beat, actions, reactions, blocking, location choice, shot, camera, facial delivery, ambience, and continuity.\n"
        "- Continuation segments must preserve the prior segment's character identity, wardrobe, location, props, spatial positions, and screen direction unless the locked script plan starts a new source scene.\n\n"
        if authoritative_script else ""
    )
    output_dialogue_shape = (
        '      "dialogue_cues": [{"character_id": "exact character id", "speaker": "character name", "dialogue": "exact spoken words"}],\n'
        if is_minimax else
        '      "character_id": "exact id from available characters or empty",\n'
        '      "dialogue": "exact spoken line",\n'
    )
    instruction = (
        f"{planner_identity}\n\n"
        "Create a preview storyboard plan. The user will review it before anything is applied to the Video Builder timeline.\n\n"
        "Important behavior:\n"
        f"{authoritative_rule}"
        "- If USER STORY / SCRIPT has text, use it as the source. It may be a premise, outline, or pasted script.\n"
        "- If USER STORY / SCRIPT is empty, invent an original short-film premise from the available characters and locations.\n"
        f"{dialogue_rule}"
        "- Use only location ids from AVAILABLE LOCATIONS when possible.\n"
        "- Each scene needs a story beat, visual direction for image prep, a full text-to-image prompt, and optional camera/facial direction.\n"
        f"- Follow the project camera plan: camera flow is {camera_flow!r} and camera motion speed is {camera_motion_speed}/10. "
        "Use controlled cinematic camera variation across the sequence. Do not default every scene to static or locked-off framing when camera speed is above 0. "
        "At camera speed 7-8, every camera_motion value must use energetic, visibly active wording and must not say slow, gentle, subtle, restrained, locked-off, static, or hold. At speed 9-10, prefer two coordinated readable camera actions. "
        "An inward move (push-in, dolly-in, zoom-in, track forward, or drift closer) is a rare accent: use at most one inward move in any six neighboring scenes. "
        "Never assign inward moves to alternating scenes. Prefer lateral drift, restrained pan, orbit, pull-back, rack focus, handheld hold, and intentional locked coverage. "
        "The requested shot_type is the literal first-frame scale: never begin wider or farther away and move inward to reach it. "
        "Reserve a static camera for an intentional dramatic beat and keep neighboring camera-motion families visibly different.\n"
        f"- Character motion speed is {character_motion_speed}/10. At speed 4 or higher, every scene needs at least one clear physical body action, gesture, step, or interaction with the set; facial expression, blinking, breathing, and mouth movement alone do not count. Keep dialogue lip sync practical.\n"
        "- camera_motion must contain the actual concise camera direction. motion_summary is optional and must only contain additional custom motion direction that is not already stated in camera_motion; otherwise leave motion_summary empty.\n"
        "- The image_prompt must follow the existing NanoBanana/Krea-style still-image prompt structure, not a short keyword list.\n"
        "- For image_prompt, write one polished paragraph, about 65-115 words, practical for text-to-image generation.\n"
        "- For image_prompt, include concrete subject identity, wardrobe, hair, makeup or facial detail when known, pose/body language, shot/framing, lens feel, lighting setup, environment, materials, atmosphere, color palette, texture, and cinematic finish.\n"
        "- For image_prompt, create a still frame only. Do not describe animation, camera movement, future action, lip sync, audio, captions, text overlays, or printed dialogue.\n"
        "- For image_prompt, prefer intimate cinematic compositions when no shot is specified: close-up, medium close-up, profile, upper body, shallow depth of field, foreground framing, bokeh, rim light, atmospheric lighting.\n"
        "- For image_prompt, if character or location reference images are available, start naturally with 'Using the provided character reference...' or 'Using the provided character reference and location reference...' and preserve the important identity/setting details without copying the exact pose, crop, or camera angle.\n"
        f"- Do not mention {'MiniMax H3, models' if is_minimax else 'ID-LoRA, LoRA'}, nodes, workflow files, voice cloning, prompts, or metadata in dialogue.\n"
        "- Do not write markdown, explanations, or code fences.\n\n"
        "Return only valid JSON with this exact shape:\n"
        "{\n"
        '  "title": "short title",\n'
        '  "premise": "one paragraph premise",\n'
        '  "scenes": [\n'
        "    {\n"
        '      "label": "Scene 1 title",\n'
        f"{output_dialogue_shape}"
        '      "location_id": "exact id from available locations or empty",\n'
        '      "story_beat": "one concise story beat",\n'
        '      "visual_direction": "short first-frame visual direction for image prep",\n'
        '      "image_prompt": "full NanoBanana/Krea-style still image prompt paragraph for creating the scene image",\n'
        '      "shot_type": "optional shot/framing",\n'
        '      "camera_motion": "optional camera movement",\n'
        '      "character_motion": "visible character blocking or action",\n'
        '      "facial_performance": "optional facial/emotional direction",\n'
        '      "delivery": "optional voice/performance delivery note",\n'
        '      "audio_direction": "ambience, sound effects, silence, breathing, and other non-dialogue audio direction",\n'
        '      "continuity": "identity, wardrobe, props, location, screen direction, and spatial continuity requirements"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Requested scene count: {scene_count}\n\n"
        f"USER STORY / SCRIPT:\n{story_source or '[blank - invent an original short-film premise]'}\n\n"
        f"SCRIPT MAPPER SEGMENT PLAN (authoritative when present):\n{json.dumps((authoritative_script or {}).get('scene_plan') or {}, ensure_ascii=False, indent=2) if authoritative_script else '[none]'}\n\n"
        f"Story layer:\n{json.dumps(story_layer, ensure_ascii=False, indent=2)}\n\n"
        f"Project motion settings:\n{json.dumps({'camera_flow': camera_flow, 'camera_motion_speed': camera_motion_speed, 'character_motion_speed': character_motion_speed}, ensure_ascii=False, indent=2)}\n\n"
        f"Available characters:\n{json.dumps(subjects, ensure_ascii=False, indent=2) if subjects else '[none provided]'}\n\n"
        f"Available locations:\n{json.dumps(locations, ensure_ascii=False, indent=2) if locations else '[none provided]'}\n\n"
        f"Existing starter scenes:\n{json.dumps(compact_existing, ensure_ascii=False, indent=2) if compact_existing else '[none]'}"
    )
    from .VRGDG_MusicVideoBuilderNodes import _extract_json_object_from_text, _run_builder_text_llm

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
        repair_instruction = (
            f"Repair this malformed JSON for a {'MiniMax short-film' if is_minimax else 'ID-LoRA'} dialogue scene plan.\n"
            "Return only valid JSON. Do not add prose, markdown, code fences, comments, or trailing commas.\n"
            "Every property name must be enclosed in double quotes. Every string value must be enclosed in double quotes.\n"
            "Keep the same title, premise, and scenes when possible.\n\n"
            f"MALFORMED JSON:\n{text}"
        )
        repaired_text, repair_info = _run_builder_text_llm(
            payload,
            repair_instruction,
            temperature=0.1,
            top_p=0.8,
            max_new_tokens=int(payload.get("max_new_tokens") or max(2200, scene_count * 520)),
            label="MiniMax Short Film Dialogue JSON Repair" if is_minimax else "ID-LoRA Dialogue Scenes JSON Repair",
            preserve_paragraphs=True,
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
        except StoryArcFormatError as exc:
            return web.json_response({
                "ok": False,
                "error": str(exc),
                "diagnostics": {
                    "kind": "story_arc_format",
                    "runner": exc.runner,
                    "expected_sections": exc.expected_sections,
                    "raw_output": exc.raw_output,
                    "cleaned_output": exc.cleaned_output,
                },
            }, status=500)
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

    @server_instance.routes.post("/vrgdg/storyboard/id_lora_dialogue_scenes")
    async def vrgdg_storyboard_id_lora_dialogue_scenes(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_id_lora_dialogue_scenes, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/minimax_dialogue_scenes")
    async def vrgdg_storyboard_minimax_dialogue_scenes(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_minimax_dialogue_scenes, payload)
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
