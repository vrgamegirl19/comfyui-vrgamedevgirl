import asyncio
import json
import os
import re
from datetime import datetime

from aiohttp import web
from server import PromptServer


_VRGDG_STORYBOARD_ROUTES_REGISTERED = False

_STORYBOARD_T2V_GEMMA_INSTRUCTIONS = """You are a text-to-video prompt builder.

The user will provide a JSON scene-card bundle. Your job is to read the JSON and create one polished text-to-video prompt for the selected scene.

Use `selected_scene_number` to choose the scene.

Use `vocal_status` to decide which opening structure to use.

If `vocal_status.should_lip_sync` is true, use this structure:

[Shot type] on [main subject] as [main subject sings/performs] with passion, physically singing "[exact lyric line from vocal_status.lyric_text]" in sync with the music. [Main subject]'s face shows [specific visible emotion] through [eyes/brows/mouth/jaw/cheeks/gaze], with clear singing mouth shapes and expressive performance energy. [Hair/costume/appearance detail] catches the light or motion. [Main subject] is in [location/setting], surrounded by [key environmental perspective/detail].

[Main subject] [performs a clear motivated action that fits the lyric, vocal intensity, and scene mood] [position/framing]. [Secondary action or physical interaction with the environment]. The camera [camera movement that follows or reacts to the performance], then [optional secondary camera move or reframing that does not repeat the same inward move]. It then [final visual beat such as a hold, drift, reveal, pass-by, pull-back, lateral move, rack focus, tilt, subject gesture, reflection, silhouette, texture, or emotional detail], capturing [specific facial detail, eye emotion, lip movement, reflection, silhouette, texture, or emotional beat].

[Background/environment details]. [Lighting description]. [Atmosphere, haze, reflections, motion blur, particles, or texture]. [Mood/style/genre tone].

If `vocal_status.instrumental` is true, `vocal_status.no_lip_sync` is true, or `vocal_status.should_lip_sync` is false, use this structure:

[Shot type] on [main subject] in [location/setting], framed by [key environmental perspective/detail]. [Main subject]'s face shows [specific visible emotion] through [eyes/brows/mouth/jaw/cheeks/gaze], with [hair/costume/appearance detail] catching the light or motion.

[Main subject] [performs a clear motivated action that fits the scene mood, character status, and environment] [position/framing]. [Secondary action or physical interaction with the environment]. The camera [camera movement that follows or reacts to the action], then [optional secondary camera move or reframing that does not repeat the same inward move]. It then [final visual beat such as a hold, drift, reveal, pass-by, pull-back, lateral move, rack focus, tilt, subject gesture, reflection, silhouette, texture, or emotional detail], capturing [specific facial detail, eye emotion, reflection, silhouette, texture, or emotional beat].

[Background/environment details]. [Lighting description]. [Atmosphere, haze, reflections, motion blur, particles, or texture]. [Mood/style/genre tone].

Rules:

* Pull the subject from `subject_refs`.
* If `subject_refs` contains exactly one subject, treat it as one individual person even if the subject label sounds plural, collective, or awkwardly worded. Do not create extra copies, duplicate singers, a group, or multiple people unless multiple subject objects are provided or the user explicitly asks for a group.
* When there is one subject, use singular phrasing and pronouns that fit the subject description. For example, "The woman sings..." rather than "The women sing..." if the provided description is a single feminine character.
* When there is one subject, never use "they", "them", or "their" for that subject. If the subject is a woman/girl/feminine character, use she/her. If the subject is a man/boy/masculine character, use he/him. If gender is unclear, repeat the subject label instead of using plural pronouns.
* When there is one subject, write "she sings", "he sings", or "[subject label] sings", never "they sing".
* Pull the location from `location_ref`.
* Use `shot_type` from the scene when available.
* Use `motion_video_summary` or `camera_motion` for camera movement.
* If `camera_motion` is empty, use `motion_video_summary`.
* Follow `camera_guidance` when present. If it says to avoid default inward moves, do not add zoom-in, push-in, dolly-in, crash-zoom, or close-up endings unless the scene explicitly requests that exact motion.
* Do not default to zoom-in, push-in, dolly-in, crash-zoom, or close-up endings. Use those inward camera moves only when `camera_motion`, `shot_type`, or the user notes explicitly ask for them.
* If `camera_motion` names a non-inward move such as pull back, track backward, side-follow, pan, tilt, crane, reveal, orbit, handheld follow, rack focus, or drift, preserve that motion and do not add a zoom-in or push-in afterward.
* Vary camera behavior between scenes. Avoid repeating the same inward camera language across multiple prompts.
* Use `performance_style` and `performance_direction` to choose the vocal wording, facial emotion, body language, gesture intensity, and camera energy. For rap/hip-hop, describe rapping or performing the lyric with rhythmic mouth movement and hand gestures instead of soft singing. For rock, punk, or metal, use stronger facial intensity and performance energy.
* If the scene is singing, use the exact lyric line from `vocal_status.lyric_text`.
* If the scene is instrumental or no-lip-sync, do not mention singing, lip-syncing, vocals, mouth movement, or no-vocal status.
* Do not mention or add a microphone, mic stand, headset mic, studio mic, or microphone prop unless `microphone.include` is true or the user's scene notes explicitly ask for a microphone.
* If `microphone.include` is true, include a handheld microphone or stand microphone only when it naturally fits the scene, stage, studio, club, or live performance setup.
* Every prompt must include visible facial emotion or facial performance. Describe what the eyes, brows, mouth, jaw, cheeks, gaze, or expression are doing.
* Singing prompts must include expressive singing mouth shapes plus an emotion that fits the lyric, such as longing, defiance, grief, joy, awe, fear, tenderness, anger, confidence, or desperation.
* Non-singing prompts must still include visible emotional expression or restrained facial tension. Do not leave the subject blank-faced.
* Do not use "expressionless", "blank expression", "empty face", "emotionless", "unreadable face", "deadpan", or "perfectly still face" unless the user's scene notes explicitly ask for that exact effect.
* If the character is described as calm, silent, stoic, robotic, alien, or controlled, translate that into visible restrained emotion: tense jaw, focused eyes, slight parted lips, narrowed gaze, lifted brow, suppressed tears, soft smile, or subtle unease.
* Do not copy reference image composition unless the scene card explicitly asks for it.
* Keep the prompt cinematic, visual, and video-friendly.
* Do not mention JSON, IDs, file paths, image names, or metadata.
* Do not include explanations.
* Output only the final prompt.
* Use natural language, not bracket labels.
* Keep it as one clean paragraph unless the user asks otherwise.

When information is missing, infer a fitting cinematic detail from the available subject, setting, tone, and notes."""


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
        return {"subject": name, "they": "she", "them": "her", "their": "her", "theirs": "hers", "are": "is", "sing": "sings", "perform": "performs"}
    if re.search(r"\b(man|boy|male|masculine|he|him|his)\b", probe):
        return {"subject": name, "they": "he", "them": "him", "their": "his", "theirs": "his", "are": "is", "sing": "sings", "perform": "performs"}
    return {"subject": name or "the subject", "they": name or "the subject", "them": name or "the subject", "their": f"{name or 'the subject'}'s", "theirs": f"{name or 'the subject'}'s", "are": "is", "sing": "sings", "perform": "performs"}


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


def _normalize_reference_image(value):
    image = value if isinstance(value, dict) else {}
    return {
        "path": _clean_scene_text(image.get("path") or "", 2000),
        "data": _clean_scene_text(image.get("data") or "", 400000),
        "name": _clean_scene_text(image.get("name") or "", 240),
    }


def _normalize_reference_item(value, fallback_name="Reference", fallback_id="ref"):
    item = value if isinstance(value, dict) else {}
    return {
        "id": _clean_scene_text(item.get("id") or fallback_id, 160),
        "name": _clean_scene_text(item.get("name") or fallback_name, 240),
        "description": _clean_scene_text(item.get("description") or "", 4000),
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


def _normalize_storyboard_scene(scene, fallback_number=1):
    if not isinstance(scene, dict):
        scene = {}
    number = _scene_number(scene, fallback_number)
    label = _clean_scene_text(scene.get("label") or f"Scene {number}", 180)
    lyrics = _clean_scene_text(scene.get("lyrics") or scene.get("lyric_text") or scene.get("lyricNote") or "", 4000)
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
    include_microphone = bool(scene.get("include_microphone") or scene.get("use_microphone") or scene.get("microphone"))
    video_prompt_type = _clean_scene_text(scene.get("video_prompt_type") or scene.get("video_type") or scene.get("mode") or "", 40)
    if video_prompt_type not in {"i2v", "t2v", "rtv"}:
        video_prompt_type = "i2v"
    status = _clean_scene_text(scene.get("status") or ("image_ready" if image_path else "draft"), 80)
    return {
        "id": _clean_scene_text(scene.get("id") or f"storyboard_scene_{number}", 160),
        "scene_number": number,
        "label": label,
        "lyrics": lyrics,
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
        "include_microphone": include_microphone,
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
        "camera_flow": _clean_scene_text(payload.get("camera_flow") or "balanced", 80),
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
        "camera_flow": _clean_scene_text(storyboard.get("camera_flow") or "balanced", 80),
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


def _export_storyboard_prompts(payload):
    saved = _save_storyboard(payload)
    project_folder = _safe_project_folder(payload.get("project_folder", ""))
    prompts_dir = _prompts_folder(project_folder)
    scenes = saved.get("scenes", [])
    t2i_path = os.path.join(prompts_dir, "t2i_prompts.txt")
    i2v_path = os.path.join(prompts_dir, "i2v_prompts.txt")
    summary_path = os.path.join(_storyboard_folder(project_folder), "storyboard_export.json")
    _write_key_value_file(t2i_path, "Prompt", scenes, "image_prompt")
    _write_key_value_file(i2v_path, "I2V", scenes, "video_prompt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "version": 1,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "t2i_prompts": t2i_path,
            "i2v_prompts": i2v_path,
            "scenes": scenes,
        }, handle, indent=2, ensure_ascii=False)
    return {
        "storyboard_path": saved.get("path", ""),
        "t2i_prompts_path": t2i_path,
        "i2v_prompts_path": i2v_path,
        "export_path": summary_path,
        "scene_count": len(scenes),
    }


def _build_storyboard_video_prompt(payload):
    scene_bundle = payload.get("storyboard_payload") or payload.get("scene_bundle") or payload.get("gpt_payload")
    if not isinstance(scene_bundle, dict):
        raise ValueError("Storyboard scene-card payload is missing.")
    scenes = scene_bundle.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard scene-card payload has no scenes.")
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
    prompt = _clean_scene_text(_fix_single_subject_prompt_pronouns(prompt, scene_bundle), 12000)
    if not prompt:
        raise ValueError("Gemma returned an empty Storyboard video prompt.")
    return {
        "prompt": prompt,
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
