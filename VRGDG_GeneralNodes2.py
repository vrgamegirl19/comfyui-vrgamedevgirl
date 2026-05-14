import json
import numbers
import os
import re
import threading
import torch
import time
import asyncio

import comfy
import folder_paths
from aiohttp import web
from nodes import PreviewImage
from server import PromptServer


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

_VRGDG_TEST_SAVE_ROUTE_REGISTERED = False
_VRGDG_T2I_CONCEPT_PROGRESS = {
    "stage": "idle",
    "message": "T2I concept prompt generator is idle.",
    "current": 0,
    "total": 0,
    "current_key": "",
    "output_path": "",
    "updated": time.time(),
}
_VRGDG_T2V_CONCEPT_PROGRESS = {
    "stage": "idle",
    "message": "T2V concept prompt generator is idle.",
    "current": 0,
    "total": 0,
    "current_key": "",
    "output_path": "",
    "updated": time.time(),
}
_VRGDG_TEST_TEXT_TARGETS = {
    "full_lyrics": ("VRGDG_TEMP", "TextFiles", "fulllyrics", "full_lyrics.txt"),
    "style_theme": ("VRGDG_TEMP", "TextFiles", "themestyle", "themestyle.txt"),
    "story_idea": ("VRGDG_TEMP", "TextFiles", "storyconcept", "storyconcept.txt"),
    "subjects_and_scenes": ("VRGDG_TEMP", "TextFiles", "subjectandscenes", "subjectsandscenes.txt"),
    "text_to_image_notes": ("VRGDG_TEMP", "TextFiles", "t2iNotes", "t2iNotes.txt"),
    "image_to_video_notes": ("VRGDG_TEMP", "TextFiles", "i2vNotes", "i2vNotes.txt"),
}

_VRGDG_GEMMA4_STYLE_INSTRUCTIONS = """send back ONLY  this 3-part block:

STYLE / THEME
1 short sentence describing the overall feeling, tone, and visual direction.

COLOR PALETTE
1 short line describing the main colors and accent colors. Never fade into dark colors.

LIGHTING / MOOD
1 short line describing brightness, contrast, and shadows.

Rules:
Use simple, everyday words.
Keep the full output under 1000 characters.
Do not include camera, lens, framing, composition, or extra detail sections.
Avoid metaphors, symbolism, poetic language, and extra explanation.
Output only the block."""

_VRGDG_GEMMA4_STORY_INSTRUCTIONS = """You turn song lyrics and optional user notes into a short story idea/concept.

Input:
- Lyrics
- Optional notes such as style, genre, mood, setting, characters, themes, or constraints

Task:
Create one concise short story idea inspired by the lyrics and notes.

Rules:
- Keep the final concept under 1000 characters.
- Do not quote or reuse long lyric phrases.
- Capture the emotional core, imagery, conflict, or theme of the lyrics.
- If notes are provided, follow them.
- If notes conflict with the lyrics, blend them creatively.
- Output only the story concept.
- No explanations, titles, bullet points, or extra text.

Style:
- Clear, vivid, and specific.
- Prefer cinematic story hooks.
- Avoid vague concepts like "a person learns about love."
- Make it feel like a story premise, not a summary of the song.

CLICK HERE TO START behavior:
If the user says exactly "CLICK HERE TO START", respond only with:
Please provide me with the full lyrics and optional style/theme and gender of your main character."""

_VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS = """# LLM Instructions: Subject & Location Extractor

You are a structured extraction assistant.

Your task is to extract and organize:

- One simple subject implied by the story idea and optional user notes
- A list of distinct physical locations implied by the story idea and optional user notes

USER NOTES PRIORITY

Optional user notes have priority over inference.
If the user notes provide an exact subject line or say to use a subject line verbatim, copy that subject line exactly as written.
Do not rewrite, shorten, correct, reformat, or replace a verbatim subject line.
Only infer the subject when the user did not provide an exact subject line.

OUTPUT FORMAT (Follow Exactly)

subject: a [gender], with [hair color], wearing [outfit].

Locations:
[one possible location]
[one possible location]
[one possible location]
[one possible location]

RULES FOR SUBJECT

Create one simple subject only.

Infer gender only if clearly implied. If unclear, use:

subject: a person, with [hair color], wearing [outfit].

If hair color is not mentioned, invent a reasonable default that fits the tone.

Do not include eye color.
Do not include hats, headwear, jewelry, or accessories.
Keep the subject concise.
The outfit should reflect the tone, genre, and setting implied by the story idea.
The outfit must be described using specific clothing items.

Do NOT use vague descriptors such as:
sleek
practical
stylish
cool
modern
fashionable

Do NOT include personality traits, emotions, or backstory.
Do NOT describe actions.
Only include gender, hair color, and outfit.

RULES FOR LOCATIONS

List only physical environments or locations.
Locations should be directly mentioned or strongly implied.
If locations are not clear, infer simple locations that fit the tone and imagery.
Locations must be places where a person could realistically be standing and photographed.
Avoid aerial perspectives, drone views, satellite views, or wide landscape shots that imply the camera is far above the environment.
Keep descriptions concise.
Use one short phrase per line.
No camera directions.
No emotional language.
No symbolic explanations.
No actions.
Just the setting.

ADDITIONAL RULES

Never output duplicate locations.
Before producing the final output, remove any repeated lines.
Do not add commentary.
Do not explain your choices.
Do not summarize.
Only output the structured list.
If the user says CLICK HERE TO START, respond: Please provide the lyrics and optional gender of the main character and any other details like hair color and clothing. Otherwise I'll make them up."""

_VRGDG_GEMMA4_LYRICS_INSTRUCTIONS = """You are a professional songwriter creating short, complete lyrics for music generation.

Task:
Turn the user's song idea and optional notes into original song lyrics.

Output only the lyrics.
No title.
No genre summary.
No style prompt.
No explanations.
No commentary.

Choose structure based on requested duration:

If duration is 60 seconds or less:

[Verse 1]
4 lines

[Chorus]
4 lines

[Verse 2]
4 lines

[Chorus]
4 lines

If duration is 61 to 150 seconds:

[Verse 1]
4 lines

[Chorus]
4 lines with a strong repeatable hook

[Verse 2]
4 lines

[Bridge]
4 lines that add contrast or a shift

[Chorus]
Repeat or lightly vary the chorus, 4 lines

If duration is over 150 seconds:

[Intro]
2 lines

[Verse 1]
4 lines

[Chorus]
4 lines

[Verse 2]
4 lines

[Bridge]
4 lines

[Chorus]
4 lines

[Outro]
2 lines

Rules:
- Keep the song suitable for the requested duration.
- Use simple, singable phrasing.
- Keep imagery consistent.
- Make the chorus memorable.
- Do not copy existing songs, artists, or lyrics.
- Do not include section names outside the required bracketed structure.
- Do not include chords, production notes, style prompts, or metadata.
- Output only the finished lyrics."""

_VRGDG_GEMMA4_T2I_FROM_CONCEPT_INSTRUCTIONS = """Create one text-to-image prompt from the user input.

User input includes:
- subject
- one current visual prompt
- a style/theme

Use all parts of the user input together.

Priority:
- Use the current visual prompt as the main scene foundation.
- Keep the main action, subject, and setting from the current visual prompt unless the user clearly changes them.
- Use the style/theme to control the visual aesthetic, color grading, lighting, mood, wardrobe refinement, environment design, and overall cinematic treatment.
- Use the provided subject as the main subject of the image.

Rules:
- Create one polished text-to-image prompt.
- Treat the current visual prompt as the base scene description.
- Expand and improve that scene using the style/theme.
- Keep the image prompt concrete and visual.
- Use the style/theme to influence color palette, tone, texture, lighting style, atmosphere, and production quality.
- If the current visual prompt includes concrete objects, actions, reflections, or setting details, keep them visible in the final prompt.
- Do not use metaphors, abstract symbolic wording, or non-visible language.
- Do not use phrases like "metaphorical thunder," "invisible storm clouds," "lightness of being," or other poetic abstractions.
- Describe only things that can be seen in the final image.
- Keep the result as one strong image prompt, not a summary.
- Correct obvious typos, malformed words, and broken phrases from the current visual prompt before using it.
- Fix spelling errors in character, clothing, objects, and setting details.
- Preserve the intended meaning while cleaning the wording.
- Do not mention that typos were fixed.
- Do not explain your choices.
- Only send the final prompt text.

Use this exact format:

A high resolution cinematic photograph of a [subject], [action or pose based primarily on the current visual prompt], in [environment/location shaped by the current visual prompt], during [time of day]. The subject is wearing [main outfit from the current visual prompt refined by the style/theme], [shoes/accessories from the current visual prompt refined by the style/theme], and [additional visible style details inspired by the style/theme]. Their hair is [hair color], [hair length/style], and [movement or texture]. The environment is [visual style of location from the current visual prompt shaped by the style/theme] with [background details that visibly represent the current visual prompt], [lighting and color grading details that match the style/theme], and [surface/reflection/material details connected to the current visual prompt and style/theme]. Camera is [camera angle] with a [lens type or framing]. The weather is [weather condition appropriate to the scene], with [atmospheric detail influenced by the style/theme], creating a [mood/style] mood.

[subject] = character gender! don't just say "subject"!

Only send the final prompt text. Do not include labels, notes, quotes, or extra text."""

_VRGDG_GEMMA4_T2V_FROM_CONCEPT_INSTRUCTIONS = """Convert the user's concept prompt into a dynamic text-to-video prompt.

Use the user's prompt as the full scene foundation. Preserve the original subject, setting, outfit, mood, atmosphere, and scene identity. Infer only the missing video details needed to make the scene feel complete, including time of day, weather, lighting behavior, environmental movement, subject movement, camera movement, and performance energy. Do not add unrelated characters, new locations, major story changes, captions, text overlays, dialogue, or audio instructions.

Add fast, cinematic motion by giving the subject a clear action sequence, expressive facial expressions, strong gestures, and intentional camera movement. Keep the subject visible, centered, and clearly framed throughout. Add lighting only as natural scene behavior, such as flickering stage lights, passing sunlight, glowing streetlights, storm flashes, reflections, or shifting shadows, based on what best fits the user's prompt.

Output one polished paragraph using this structure:

The [Subject] who is singing with passion and in sync with the audio, in [setting/environment] during [time/weather]. The subject [dynamic performance action with expressive face, body movement, and strong gestures]. Their clothing/hair [reacts to movement, wind, or performance energy]. The lighting [changes or reacts naturally within the scene]. The camera [Camera Motion] while maintaining [subject visibility and framing]. The environment [reacts dynamically].

Each word in brackets should be chosen based on the user input and what best fits the scene.

Rules:
- This is text-to-video
- Subject must be physically singing with passion
- Do not add audio, dialogue, captions, text overlays, unrelated characters, new locations, major story changes, color grading, camera photo style, or static image-quality descriptions.
- Keep it vivid, fast, cinematic, dynamic, and video-ready
- use one location infered by the user's concept prompt. If one is not listed use one from the location list.
- Must use user input to help create the prompt
- Only send the final prompt text. Do not include labels, notes, quotes, or extra text."""

_VRGDG_GEMMA4_ADVANCED_PROMPT_DETAIL_INSTRUCTIONS = """You create one visual prompt detail list for a video workflow.

Input:
- A detail label, such as Camera Motion, Lighting, Weather, Emotion, Facial Expression, Dialogue, or a custom label
- A numbered list of scene prompts

Task:
For each scene prompt, create exactly one matching detail line for the requested label.

Rules:
- Output only the list.
- Return exactly one line per scene prompt.
- Keep the line order exactly the same as the scene prompt order.
- Do not include numbers, bullets, labels, titles, quotes, markdown, or explanations.
- Each line must be short and specific.
- Each line must fit the requested label only.
- Follow the optional user guidance if provided.
- If optional guidance conflicts with the requested label, keep the label as the main category and use the guidance only for tone, speed, mood, intensity, or style.
- Do not combine multiple categories in one line.
- Do not repeat the full prompt.
- Avoid vague words like cinematic, beautiful, cool, stylish, dramatic, or interesting unless the label specifically asks for mood.
- If the prompt does not clearly imply a value, invent a simple value that fits the scene.
- For Camera Motion, output only camera movement phrases.
- For Dialogue, output only one short spoken line with no quotation marks.
- For Lighting, output only lighting descriptions.
- For Weather, output only weather descriptions.
- For Time of Day, output only time-of-day phrases.
- For Emotion or Facial Expression, output only the emotion or expression."""


def _get_default_comfy_output_directory():
    base_path = getattr(folder_paths, "base_path", None)
    if base_path:
        return os.path.normpath(os.path.join(base_path, "output"))

    custom_nodes_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root_dir = os.path.normpath(os.path.join(custom_nodes_dir, "..", ".."))
    return os.path.normpath(os.path.join(comfy_root_dir, "output"))


def _get_output_directory_candidates():
    candidates = []
    try:
        configured_output = folder_paths.get_output_directory()
        if configured_output:
            candidates.append(os.path.normpath(configured_output))
    except Exception:
        pass

    candidates.append(_get_default_comfy_output_directory())

    unique = []
    seen = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _first_existing_nonempty_path(relative_parts):
    fallback = None
    for output_dir in _get_output_directory_candidates():
        path = os.path.normpath(os.path.join(output_dir, *relative_parts))
        if fallback is None:
            fallback = path
        if os.path.isfile(path):
            try:
                if os.path.getsize(path) > 0:
                    return path
            except OSError:
                return path
    return fallback


def _apply_backend_node_action(node_id, action):
    action = str(action or "mute").lower()
    if action == "active":
        PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": True})
    elif action == "bypass":
        PromptServer.instance.send_sync(
            "impact-bridge-continue",
            {"node_id": str(node_id), "bypasses": [str(node_id)], "mutes": [], "actives": []},
        )
    else:
        PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": False})


def _get_workflow_from_extra_pnginfo(extra_pnginfo):
    if not isinstance(extra_pnginfo, list) or not extra_pnginfo:
        return None
    first = extra_pnginfo[0]
    if not isinstance(first, dict):
        return None
    workflow = first.get("workflow")
    return workflow if isinstance(workflow, dict) else None


def _workflow_groups_sorted_alpha(workflow):
    groups = workflow.get("groups", []) if isinstance(workflow, dict) else []
    if not isinstance(groups, list):
        return []
    filtered = [g for g in groups if isinstance(g, dict) and str(g.get("title", "")).strip()]
    filtered.sort(key=lambda g: str(g.get("title", "")).strip().lower())
    return filtered


def _workflow_nodes_for_group(workflow, group):
    if not isinstance(workflow, dict) or not isinstance(group, dict):
        return []
    bounds = group.get("bounding")
    if not isinstance(bounds, list) or len(bounds) < 4:
        return []
    try:
        gx, gy, gw, gh = [float(v) for v in bounds[:4]]
    except Exception:
        return []

    resolved = []
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        try:
            node_id = int(node.get("id"))
        except Exception:
            continue
        pos = node.get("pos") or [0, 0]
        size = node.get("size") or [140, 80]
        try:
            center_x = float(pos[0]) + float(size[0]) * 0.5
            center_y = float(pos[1]) + float(size[1]) * 0.5
        except Exception:
            continue
        if gx <= center_x < gx + gw and gy <= center_y < gy + gh:
            resolved.append(node_id)
    return resolved


def _resolve_preset_targets(extra_pnginfo, target_specs):
    workflow = _get_workflow_from_extra_pnginfo(extra_pnginfo)
    if workflow is None:
        return []

    groups_sorted = _workflow_groups_sorted_alpha(workflow)
    resolved_targets = []
    for spec in target_specs:
        if not isinstance(spec, dict):
            continue
        slot = spec.get("slot")
        title = str(spec.get("title", "")).strip()
        matched_group = None

        if title:
            matched_group = next(
                (group for group in groups_sorted if str(group.get("title", "")).strip() == title),
                None,
            )

        if matched_group is None:
            try:
                slot_index = int(slot) - 1
            except Exception:
                slot_index = -1
            if 0 <= slot_index < len(groups_sorted):
                matched_group = groups_sorted[slot_index]

        node_ids = _workflow_nodes_for_group(workflow, matched_group) if matched_group is not None else []
        resolved_targets.append(
            {
                "slot": slot,
                "title": title,
                "action": spec.get("action", "mute"),
                "node_ids": node_ids,
            }
        )
    return resolved_targets


def _run_preset_group_state(signal, extra_pnginfo, target_specs, queue_delay=5.0):
    targets = _resolve_preset_targets(extra_pnginfo, target_specs)
    for target in targets:
        for node_id in target.get("node_ids", []):
            _apply_backend_node_action(node_id, target.get("action", "mute"))

    if targets:
        PromptServer.instance.send_sync("vrgdg-apply-node-modes", {"targets": targets})

    def _delayed_queue():
        time.sleep(max(0.0, float(queue_delay or 0.0)))
        PromptServer.instance.send_sync("impact-add-queue", {})

    threading.Thread(target=_delayed_queue, daemon=True).start()
    return (signal,)


def _get_test_popup_audio_dir():
    return os.path.normpath(os.path.join(folder_paths.get_output_directory(), "VRGDG_AudioFiles"))


def _set_t2i_concept_progress(stage, message, current=None, total=None, current_key=None, output_path=None):
    _VRGDG_T2I_CONCEPT_PROGRESS["stage"] = str(stage or "")
    _VRGDG_T2I_CONCEPT_PROGRESS["message"] = str(message or "")
    _VRGDG_T2I_CONCEPT_PROGRESS["updated"] = time.time()
    if current is not None:
        _VRGDG_T2I_CONCEPT_PROGRESS["current"] = int(current)
    if total is not None:
        _VRGDG_T2I_CONCEPT_PROGRESS["total"] = int(total)
    if current_key is not None:
        _VRGDG_T2I_CONCEPT_PROGRESS["current_key"] = str(current_key)
    if output_path is not None:
        _VRGDG_T2I_CONCEPT_PROGRESS["output_path"] = str(output_path)


def _get_t2i_concept_progress():
    return dict(_VRGDG_T2I_CONCEPT_PROGRESS)


def _set_t2v_concept_progress(stage, message, current=None, total=None, current_key=None, output_path=None):
    _VRGDG_T2V_CONCEPT_PROGRESS["stage"] = str(stage or "")
    _VRGDG_T2V_CONCEPT_PROGRESS["message"] = str(message or "")
    _VRGDG_T2V_CONCEPT_PROGRESS["updated"] = time.time()
    if current is not None:
        _VRGDG_T2V_CONCEPT_PROGRESS["current"] = int(current)
    if total is not None:
        _VRGDG_T2V_CONCEPT_PROGRESS["total"] = int(total)
    if current_key is not None:
        _VRGDG_T2V_CONCEPT_PROGRESS["current_key"] = str(current_key)
    if output_path is not None:
        _VRGDG_T2V_CONCEPT_PROGRESS["output_path"] = str(output_path)


def _get_t2v_concept_progress():
    return dict(_VRGDG_T2V_CONCEPT_PROGRESS)



def _get_test_popup_text_path(field_name):
    parts = _VRGDG_TEST_TEXT_TARGETS[field_name]
    return os.path.normpath(os.path.join(folder_paths.get_output_directory(), *parts))


def _get_part2_concept_prompts_path():
    return _first_existing_nonempty_path(
        (
            "VRGDG_TEMP",
            "TextFiles",
            "ConceptPrompts",
            "ConceptPrompts.txt",
        )
    )


def _get_vrgdg_text_file_path(folder_name, file_name):
    return os.path.normpath(
        os.path.join(
            folder_paths.get_output_directory(),
            "VRGDG_TEMP",
            "TextFiles",
            folder_name,
            file_name,
        )
    )


def _get_t2i_prompts_output_path():
    return _get_vrgdg_text_file_path("t2i_Prompts", "t2i_Prompts.txt")


def _get_t2v_prompts_output_path():
    return _get_vrgdg_text_file_path("t2v_Prompts", "t2v_Prompts.txt")


def _read_text_file_if_exists(path):
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        return handle.read().strip()


def _strip_json_fence(text):
    value = str(text or "").strip()
    value = re.sub(r"^\s*```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```\s*$", "", value)
    return value.strip()


def _parse_concept_prompt_items(text):
    cleaned = _strip_json_fence(text)
    if not cleaned:
        raise ValueError("ConceptPrompts.txt is empty.")

    try:
        data = json.loads(cleaned, object_pairs_hook=list)
    except json.JSONDecodeError as exc:
        blocks = [block.strip() for block in re.split(r"(?:\r?\n){2,}", cleaned) if block.strip()]
        if not blocks:
            raise ValueError(
                f"ConceptPrompts.txt is not valid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            )
        return [(f"prompt_{index}", block) for index, block in enumerate(blocks, start=1)]

    if isinstance(data, list):
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in data):
            pairs = data
        else:
            pairs = [(f"prompt_{index}", item) for index, item in enumerate(data, start=1)]
    elif isinstance(data, dict):
        pairs = list(data.items())
    else:
        raise ValueError("ConceptPrompts.txt must contain a JSON object or array.")

    items = []
    for index, (key, value) in enumerate(pairs, start=1):
        if isinstance(value, str):
            prompt_text = value.strip()
        else:
            prompt_text = json.dumps(value, ensure_ascii=False)
        if not prompt_text:
            continue
        items.append((str(key), prompt_text))

    if not items:
        raise ValueError("ConceptPrompts.txt did not contain any usable prompt rows.")

    return items


def _clean_gemma4_text(value):
    text = str(value or "").strip()
    text = re.sub(r"^\s*```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _first_clean_gemma4_line(value):
    for line in _clean_gemma4_text(value).splitlines():
        text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if text:
            return text
    return ""


def _build_gemma4_prompt(target, payload):
    target = str(target or "").strip()
    notes = str(payload.get("notes", "") or "").strip()
    lyrics = str(payload.get("lyrics", "") or "").strip()
    style_theme = str(payload.get("style_theme", "") or "").strip()
    story_idea = str(payload.get("story_idea", "") or "").strip()

    if target == "style_theme":
        prompt = f"{_VRGDG_GEMMA4_STYLE_INSTRUCTIONS}\n\nfull lyrics:\n{lyrics}"
        if notes:
            prompt += f"\n\nother notes:\n{notes}"
        return prompt

    if target == "story_idea":
        prompt = f"{_VRGDG_GEMMA4_STORY_INSTRUCTIONS}\n\nLyrics:\n{lyrics}"
        if style_theme:
            prompt += f"\n\nStyle/theme:\n{style_theme}"
        if notes:
            prompt += f"\n\nOptional notes:\n{notes}"
        return prompt

    if target == "subjects_and_scenes":
        prompt = f"{_VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS}"
        if notes:
            prompt += f"\n\nUser notes - highest priority:\n{notes}"
        prompt += f"\n\nStory idea:\n{story_idea}"
        return prompt

    if target == "song_lyrics":
        duration = str(payload.get("duration", "") or "").strip()
        prompt = f"{_VRGDG_GEMMA4_LYRICS_INSTRUCTIONS}"
        if duration:
            prompt += f"\n\nRequested duration seconds:\n{duration}"
        prompt += f"\n\nSong idea and notes:\n{notes}"
        if not notes:
            prompt += "\nCreate an original short song about refusing to give up."
        return prompt

    if target == "text_to_image_from_concept":
        concept_prompt = str(payload.get("concept_prompt", "") or "").strip()
        if not concept_prompt:
            raise ValueError("No concept prompt was provided for text-to-image generation.")
        extra_user_input = str(payload.get("extra_user_input", "") or "").strip()
        prompt = (
            f"{_VRGDG_GEMMA4_T2I_FROM_CONCEPT_INSTRUCTIONS}\n\n"
            f"Story idea:\n{story_idea}\n\n"
            f"Style/theme:\n{style_theme}\n\n"
            f"Current visual prompt:\n{concept_prompt}"
        )
        if extra_user_input:
            prompt += f"\n\nExtra user input:\n{extra_user_input}"
        return prompt

    if target == "text_to_video_from_concept":
        concept_prompt = str(payload.get("concept_prompt", "") or "").strip()
        if not concept_prompt:
            raise ValueError("No concept prompt was provided for text-to-video generation.")
        subjects_and_scenes = str(payload.get("subjects_and_scenes", "") or "").strip()
        extra_user_input = str(payload.get("extra_user_input", "") or "").strip()
        prompt = (
            f"{_VRGDG_GEMMA4_T2V_FROM_CONCEPT_INSTRUCTIONS}\n\n"
            f"Subject and location list:\n{subjects_and_scenes}\n\n"
            f"Style/theme:\n{style_theme}\n\n"
            f"Concept prompt:\n{concept_prompt}"
        )
        if extra_user_input:
            prompt += f"\n\nUser input:\n{extra_user_input}"
        return prompt

    if target == "advanced_prompt_detail":
        label = str(payload.get("label", "") or "").strip() or "Custom"
        prompts = payload.get("prompts") or []
        if not isinstance(prompts, list):
            prompts = []
        prompt_lines = []
        for index, item in enumerate(prompts, start=1):
            text = str(item or "").strip()
            if text:
                prompt_lines.append(f"{index}. {text}")
        if not prompt_lines:
            raise ValueError("No scene prompts were provided for Gemma4 advanced list generation.")
        prompt = (
            f"{_VRGDG_GEMMA4_ADVANCED_PROMPT_DETAIL_INSTRUCTIONS}\n\n"
            f"Detail label:\n{label}\n\n"
        )
        if notes:
            prompt += f"Optional user guidance for all lists:\n{notes}\n\n"
        prompt += f"Scene prompts:\n" + "\n".join(prompt_lines)
        return prompt

    raise ValueError(f"Unsupported Gemma4 target: {target}")


def _run_gemma4_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat

    target = str(payload.get("target", "") or "").strip()
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file:
        raise ValueError("No Gemma4 model_file was selected.")

    prompt = _build_gemma4_prompt(target, payload)
    if not prompt.strip():
        raise ValueError("Gemma4 prompt was empty.")

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    # Match the known-good SuperGemma settings used in the workflow node.
    n_ctx = int(payload.get("n_ctx") or 13000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.75)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 32000)
    unload_after = bool(payload.get("unload_after"))

    mmproj_path = ""
    model = None
    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            mmproj_path=mmproj_path,
        )
        text = llm._run_gguf_text_pipeline(
            model=model,
            instruction_text=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        text = _clean_gemma4_text(text)
        if not text:
            raise ValueError("Gemma4 returned an empty response.")
        return {
            "text": text,
            "used_model": model_path,
            "unloaded": unload_after,
        }
    finally:
        if unload_after and model_path:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )


def _unload_gemma4_prompt_model(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat

    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file:
        return {"unloaded": False, "reason": "No model_file provided."}
    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    n_ctx = int(payload.get("n_ctx") or 13000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    llm._unload_gguf_model(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        chat_format=chat_format,
        mmproj_path="",
    )
    return {"unloaded": True, "used_model": model_path}


def _generate_t2i_prompts_from_concepts(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file:
        raise ValueError("Choose a Gemma4 model first.")
    extra_user_input = str(payload.get("extra_user_input", "") or "").strip()

    _set_t2i_concept_progress(
        "starting",
        "Reading ConceptPrompts.txt, themestyle.txt, and storyconcept.txt...",
        current=0,
        total=0,
        current_key="",
        output_path="",
    )

    concept_path = _get_part2_concept_prompts_path()
    style_path = _get_vrgdg_text_file_path("themestyle", "themestyle.txt")
    story_path = _get_vrgdg_text_file_path("storyconcept", "storyconcept.txt")
    output_path = _get_t2i_prompts_output_path()

    concept_text = _read_text_file_if_exists(concept_path)
    style_theme = _read_text_file_if_exists(style_path)
    story_idea = _read_text_file_if_exists(story_path)

    if not os.path.isfile(concept_path):
        raise FileNotFoundError(f"ConceptPrompts.txt was not found: {concept_path}")
    if not style_theme:
        raise ValueError(f"Style/theme text file is empty or missing: {style_path}")
    if not story_idea:
        raise ValueError(f"Story concept text file is empty or missing: {story_path}")

    concept_items = _parse_concept_prompt_items(concept_text)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_steps = len(concept_items) * 3
    _set_t2i_concept_progress(
        "running",
        f"Loaded {len(concept_items)} concept prompt row(s). Starting Gemma...",
        current=0,
        total=total_steps,
        output_path=output_path,
    )

    generated = {}
    used_model = ""
    current_step = 0
    for index, (key, concept_prompt) in enumerate(concept_items, start=1):
        current_step += 1
        _set_t2i_concept_progress(
            "running",
            f"Generating camera motion for prompt {index} of {len(concept_items)}...",
            current=current_step,
            total=total_steps,
            current_key=key,
            output_path=output_path,
        )
        camera_result = _run_gemma4_prompt(
            {
                "target": "advanced_prompt_detail",
                "model_file": model_file,
                "label": "Camera Motion",
                "prompts": [concept_prompt],
                "notes": extra_user_input,
                "n_ctx": int(payload.get("n_ctx") or 13000),
                "max_new_tokens": 512,
                "temperature": float(payload.get("temperature") or 0.75),
                "top_p": float(payload.get("top_p") or 0.95),
                "unload_after": False,
            }
        )
        used_model = str(camera_result.get("used_model") or used_model)
        camera_motion = _first_clean_gemma4_line(camera_result.get("text") or "")

        current_step += 1
        _set_t2i_concept_progress(
            "running",
            f"Generating character motion for prompt {index} of {len(concept_items)}...",
            current=current_step,
            total=total_steps,
            current_key=key,
            output_path=output_path,
        )
        character_result = _run_gemma4_prompt(
            {
                "target": "advanced_prompt_detail",
                "model_file": model_file,
                "label": "Character Movement/Motion",
                "prompts": [concept_prompt],
                "notes": extra_user_input,
                "n_ctx": int(payload.get("n_ctx") or 13000),
                "max_new_tokens": 512,
                "temperature": float(payload.get("temperature") or 0.75),
                "top_p": float(payload.get("top_p") or 0.95),
                "unload_after": False,
            }
        )
        used_model = str(character_result.get("used_model") or used_model)
        character_motion = _first_clean_gemma4_line(character_result.get("text") or "")

        t2i_user_input = []
        if camera_motion:
            t2i_user_input.append(f"camera motion = {camera_motion}")
        if character_motion:
            t2i_user_input.append(f"character motion = {character_motion}")
        if extra_user_input:
            t2i_user_input.append(f"extra user input = {extra_user_input}")

        current_step += 1
        _set_t2i_concept_progress(
            "running",
            f"Creating final text-to-image prompt {index} of {len(concept_items)}...",
            current=current_step,
            total=total_steps,
            current_key=key,
            output_path=output_path,
        )
        result = _run_gemma4_prompt(
            {
                "target": "text_to_image_from_concept",
                "model_file": model_file,
                "style_theme": style_theme,
                "story_idea": story_idea,
                "concept_prompt": concept_prompt,
                "extra_user_input": "\n".join(t2i_user_input),
                "n_ctx": int(payload.get("n_ctx") or 13000),
                "max_new_tokens": int(payload.get("max_new_tokens") or 1600),
                "temperature": float(payload.get("temperature") or 0.75),
                "top_p": float(payload.get("top_p") or 0.95),
                "unload_after": index == len(concept_items),
            }
        )
        used_model = str(result.get("used_model") or used_model)
        generated[key] = _clean_gemma4_text(result.get("text") or "")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(generated, handle, indent=2, ensure_ascii=False)
        _set_t2i_concept_progress(
            "running",
            f"Saved prompt {index} of {len(concept_items)} to t2i_Prompts.txt.",
            current=current_step,
            total=total_steps,
            current_key=key,
            output_path=output_path,
        )

    _set_t2i_concept_progress(
        "done",
        f"Finished {len(generated)} text-to-image prompt(s). Gemma was unloaded.",
        current=total_steps,
        total=total_steps,
        current_key="",
        output_path=output_path,
    )
    return {
        "ok": True,
        "count": len(generated),
        "output_path": output_path,
        "concept_path": concept_path,
        "style_path": style_path,
        "story_path": story_path,
        "used_model": used_model,
    }


def _generate_t2v_prompts_from_concepts(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file:
        raise ValueError("Choose a Gemma4 model first.")
    extra_user_input = str(payload.get("extra_user_input", "") or "").strip()

    _set_t2v_concept_progress(
        "starting",
        "Reading ConceptPrompts.txt, themestyle.txt, and subjectsandscenes.txt...",
        current=0,
        total=0,
        current_key="",
        output_path="",
    )

    concept_path = _get_part2_concept_prompts_path()
    style_path = _get_vrgdg_text_file_path("themestyle", "themestyle.txt")
    subjects_path = _get_vrgdg_text_file_path("subjectandscenes", "subjectsandscenes.txt")
    output_path = _get_t2v_prompts_output_path()

    concept_text = _read_text_file_if_exists(concept_path)
    style_theme = _read_text_file_if_exists(style_path)
    subjects_and_scenes = _read_text_file_if_exists(subjects_path)

    if not os.path.isfile(concept_path):
        raise FileNotFoundError(f"ConceptPrompts.txt was not found: {concept_path}")
    if not style_theme:
        raise ValueError(f"Style/theme text file is empty or missing: {style_path}")
    if not subjects_and_scenes:
        raise ValueError(f"Subject/location text file is empty or missing: {subjects_path}")

    concept_items = _parse_concept_prompt_items(concept_text)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _set_t2v_concept_progress(
        "running",
        f"Loaded {len(concept_items)} concept prompt row(s). Starting Gemma...",
        current=0,
        total=len(concept_items),
        output_path=output_path,
    )

    generated = {}
    used_model = ""
    for index, (key, concept_prompt) in enumerate(concept_items, start=1):
        _set_t2v_concept_progress(
            "running",
            f"Creating text-to-video prompt {index} of {len(concept_items)}...",
            current=index,
            total=len(concept_items),
            current_key=key,
            output_path=output_path,
        )
        result = _run_gemma4_prompt(
            {
                "target": "text_to_video_from_concept",
                "model_file": model_file,
                "style_theme": style_theme,
                "subjects_and_scenes": subjects_and_scenes,
                "concept_prompt": concept_prompt,
                "extra_user_input": extra_user_input,
                "n_ctx": int(payload.get("n_ctx") or 13000),
                "max_new_tokens": int(payload.get("max_new_tokens") or 1200),
                "temperature": float(payload.get("temperature") or 0.75),
                "top_p": float(payload.get("top_p") or 0.95),
                "unload_after": index == len(concept_items),
            }
        )
        used_model = str(result.get("used_model") or used_model)
        generated[key] = _clean_gemma4_text(result.get("text") or "")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(generated, handle, indent=2, ensure_ascii=False)
        _set_t2v_concept_progress(
            "running",
            f"Saved prompt {index} of {len(concept_items)} to t2v_Prompts.txt.",
            current=index,
            total=len(concept_items),
            current_key=key,
            output_path=output_path,
        )

    _set_t2v_concept_progress(
        "done",
        f"Finished {len(generated)} text-to-video prompt(s). Gemma was unloaded.",
        current=len(concept_items),
        total=len(concept_items),
        current_key="",
        output_path=output_path,
    )
    return {
        "ok": True,
        "count": len(generated),
        "output_path": output_path,
        "concept_path": concept_path,
        "style_path": style_path,
        "subjects_path": subjects_path,
        "used_model": used_model,
    }


def _ensure_test_save_route_registered():
    global _VRGDG_TEST_SAVE_ROUTE_REGISTERED
    if _VRGDG_TEST_SAVE_ROUTE_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.get("/vrgdg/test_popup/config")
    async def vrgdg_test_popup_config(request):
        text_targets = {
            field_name: _get_test_popup_text_path(field_name)
            for field_name in _VRGDG_TEST_TEXT_TARGETS
        }
        return web.json_response(
            {
                "ok": True,
                "audio_dir": _get_test_popup_audio_dir(),
                "text_targets": text_targets,
                "concept_prompts_path": _get_part2_concept_prompts_path(),
            }
        )

    @server_instance.routes.get("/vrgdg/part2/load_concept_prompts")
    async def vrgdg_part2_load_concept_prompts(request):
        target_path = _get_part2_concept_prompts_path()
        if not os.path.isfile(target_path):
            return web.json_response(
                {
                    "ok": False,
                    "error": "ConceptPrompts.txt was not found. Run Step 1 first or paste the prompt JSON manually.",
                    "path": target_path,
                },
                status=404,
            )

        try:
            with open(target_path, "r", encoding="utf-8-sig") as handle:
                text = handle.read()
        except Exception as exc:
            return web.json_response(
                {
                    "ok": False,
                    "error": f"Could not read ConceptPrompts.txt: {exc}",
                    "path": target_path,
                },
                status=500,
            )

        return web.json_response({"ok": True, "text": text, "path": target_path})

    @server_instance.routes.post("/vrgdg/test_popup/save_text")
    async def vrgdg_test_popup_save_text(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        saved_paths = {}
        for field_name in _VRGDG_TEST_TEXT_TARGETS:
            target_path = _get_test_popup_text_path(field_name)
            folder_path = os.path.dirname(target_path)
            text_value = str(payload.get(field_name, "") or "")

            try:
                os.makedirs(folder_path, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as handle:
                    handle.write(text_value)
            except Exception as exc:
                return web.json_response(
                    {
                        "ok": False,
                        "error": f"Could not write {field_name}: {exc}",
                        "field": field_name,
                    },
                    status=400,
                )

            saved_paths[field_name] = target_path

        return web.json_response({"ok": True, "saved_paths": saved_paths})

    @server_instance.routes.post("/vrgdg/test_popup/upload_audio")
    async def vrgdg_test_popup_upload_audio(request):
        post = await request.post()
        audio = post.get("audio")
        if audio is None or not getattr(audio, "file", None):
            return web.json_response({"ok": False, "error": "Missing audio file."}, status=400)

        filename = os.path.basename(str(getattr(audio, "filename", "") or "").strip())
        if not filename:
            return web.json_response({"ok": False, "error": "Invalid audio filename."}, status=400)

        target_dir = _get_test_popup_audio_dir()
        target_path = os.path.join(target_dir, filename)

        try:
            os.makedirs(target_dir, exist_ok=True)
            for existing_name in os.listdir(target_dir):
                existing_path = os.path.join(target_dir, existing_name)
                if os.path.isfile(existing_path):
                    os.remove(existing_path)
            with open(target_path, "wb") as handle:
                handle.write(audio.file.read())
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": f"Could not write audio file: {exc}"},
                status=400,
            )

        return web.json_response({"ok": True, "path": target_path, "filename": filename})

    @server_instance.routes.post("/vrgdg/gemma4/generate")
    async def vrgdg_gemma4_generate(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        if not isinstance(payload, dict):
            return web.json_response({"ok": False, "error": "JSON body must be an object."}, status=400)

        try:
            result = await asyncio.to_thread(_run_gemma4_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/gemma4/unload")
    async def vrgdg_gemma4_unload(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        try:
            result = await asyncio.to_thread(_unload_gemma4_prompt_model, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/t2i_from_concepts/generate")
    async def vrgdg_t2i_from_concepts_generate(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        if not isinstance(payload, dict):
            return web.json_response({"ok": False, "error": "JSON body must be an object."}, status=400)

        try:
            result = await asyncio.to_thread(_generate_t2i_prompts_from_concepts, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        return web.json_response(result)

    @server_instance.routes.get("/vrgdg/t2i_from_concepts/progress")
    async def vrgdg_t2i_from_concepts_progress(request):
        return web.json_response({"ok": True, **_get_t2i_concept_progress()})

    @server_instance.routes.post("/vrgdg/t2v_from_concepts/generate")
    async def vrgdg_t2v_from_concepts_generate(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        if not isinstance(payload, dict):
            return web.json_response({"ok": False, "error": "JSON body must be an object."}, status=400)

        try:
            result = await asyncio.to_thread(_generate_t2v_prompts_from_concepts, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        return web.json_response(result)

    @server_instance.routes.get("/vrgdg/t2v_from_concepts/progress")
    async def vrgdg_t2v_from_concepts_progress(request):
        return web.json_response({"ok": True, **_get_t2v_concept_progress()})
    @server_instance.routes.post("/vrgdg/apply_node_modes")
    async def vrgdg_apply_node_modes(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        targets = payload.get("targets", [])
        if not isinstance(targets, list):
            return web.json_response({"ok": False, "error": "targets must be a list."}, status=400)

        applied = 0
        for target in targets:
            if not isinstance(target, dict):
                continue
            action = target.get("action", "mute")
            node_ids = target.get("node_ids", [])
            if not isinstance(node_ids, list):
                continue
            for raw_id in node_ids:
                try:
                    node_id = int(raw_id)
                except Exception:
                    continue
                if node_id < 0:
                    continue
                _apply_backend_node_action(node_id, action)
                applied += 1

        return web.json_response({"ok": True, "applied": applied})

    _VRGDG_TEST_SAVE_ROUTE_REGISTERED = True





class VRGDG_ShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "VRGDG/General"

    def notify(self, text=None, unique_id=None, extra_pnginfo=None):
        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("VRGDG_ShowText: extra_pnginfo is not a list")
            elif not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]:
                print("VRGDG_ShowText: extra_pnginfo[0] is not a dict or missing 'workflow'")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow.get("nodes", []) if str(x.get("id")) == str(unique_id[0])),
                    None,
                )
                if node is not None:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}


class VRGDG_ShowAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "value": (any_typ, {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "VRGDG/General"

    def _format_value(self, value):
        if isinstance(value, str):
            return value

        try:
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def notify(self, value=None, unique_id=None, extra_pnginfo=None):
        if value is None:
            value = [None]
        elif not isinstance(value, list):
            value = [value]

        text = [self._format_value(item) for item in value]

        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("VRGDG_ShowAny: extra_pnginfo is not a list")
            elif not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]:
                print("VRGDG_ShowAny: extra_pnginfo[0] is not a dict or missing 'workflow'")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow.get("nodes", []) if str(x.get("id")) == str(unique_id[0])),
                    None,
                )
                if node is not None:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}


class VRGDG_TextBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "output_mode": (["string", "json"], {"default": "string"}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON")
    RETURN_NAMES = ("text_output", "json_output")
    FUNCTION = "output_text"
    CATEGORY = "VRGDG/General"

    def output_text(self, text, output_mode):
        if output_mode == "json":
            try:
                json_output = json.loads(text)
            except Exception as e:
                raise ValueError(f"VRGDG_TextBox: output_mode is 'json' but input is not valid JSON: {e}")
            return (text, json_output)

        return (text, {})


class VRGDG_String2Json:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
                "auto_fix": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("json_output",)
    FUNCTION = "to_json"
    CATEGORY = "VRGDG/General"

    def _basic_cleanup(self, text):
        cleaned = str(text).strip()
        cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "")
        cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        return cleaned

    def _escape_unescaped_inner_quotes(self, s):
        chars = []
        in_string = False
        escaped = False
        n = len(s)
        i = 0

        while i < n:
            ch = s[i]

            if not in_string:
                chars.append(ch)
                if ch == '"':
                    in_string = True
                    escaped = False
                i += 1
                continue

            # in string
            if escaped:
                chars.append(ch)
                escaped = False
                i += 1
                continue

            if ch == "\\":
                chars.append(ch)
                escaped = True
                i += 1
                continue

            if ch == '"':
                # If this quote is not followed by a valid JSON token boundary,
                # treat it as an inner quote and escape it.
                j = i + 1
                while j < n and s[j].isspace():
                    j += 1
                next_ch = s[j] if j < n else ""
                if next_ch not in [",", "}", "]", ":", ""]:
                    chars.append("\\")
                    chars.append('"')
                    i += 1
                    continue

                chars.append(ch)
                in_string = False
                i += 1
                continue

            chars.append(ch)
            i += 1

        return "".join(chars)

    def _remove_trailing_commas(self, s):
        import re
        return re.sub(r",(\s*[}\]])", r"\1", s)

    def _auto_fix_json_text(self, text):
        cleaned = self._basic_cleanup(text)
        cleaned = self._escape_unescaped_inner_quotes(cleaned)
        cleaned = self._remove_trailing_commas(cleaned)
        return cleaned

    def to_json(self, text, auto_fix=True):
        raw = self._basic_cleanup(text)

        try:
            return (json.loads(raw),)
        except Exception as e:
            if not auto_fix:
                raise ValueError(f"VRGDG_String2Json: invalid JSON input: {e}")

            fixed = self._auto_fix_json_text(raw)
            try:
                return (json.loads(fixed),)
            except Exception as e2:
                raise ValueError(
                    "VRGDG_String2Json: invalid JSON input after auto-fix attempt: "
                    f"{e2}"
                )


class VRGDG_Json2String:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("JSON", {"forceInput": True}),
                "pretty": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "to_string"
    CATEGORY = "VRGDG/General"

    def to_string(self, json_input, pretty=True):
        try:
            if pretty:
                text_output = json.dumps(json_input, indent=2, ensure_ascii=False, default=str)
            else:
                text_output = json.dumps(json_input, separators=(",", ":"), ensure_ascii=False, default=str)
        except Exception:
            text_output = str(json_input)

        return (text_output,)


class VRGDG_ShowImage(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show_image"
    OUTPUT_NODE = True
    CATEGORY = "VRGDG/General"

    def _is_empty_image(self, image):
        if image is None:
            return True

        if isinstance(image, numbers.Number):
            return image == 0

        if isinstance(image, (list, tuple)):
            return len(image) == 0

        if hasattr(image, "numel"):
            try:
                return image.numel() == 0
            except Exception:
                pass

        if hasattr(image, "shape"):
            try:
                if len(image.shape) == 0:
                    return False
                return image.shape[0] == 0
            except Exception:
                pass

        return False

    def show_image(self, image=None, prompt=None, extra_pnginfo=None):
        if self._is_empty_image(image):
            return {"ui": {"images": []}}

        return self.save_images(
            image,
            filename_prefix="VRGDG_ShowImage",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )


class VRGDG_BoxIT:
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "BoxIT", "multiline": False}),
            }
        }

    def run(self, label):
        return ()


class VRGDG_IntToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "convert"
    CATEGORY = "VRGDG/General"

    def convert(self, value):
        return (float(value),)


class VRGDG_ImageIndex0HUMOEDIT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_index": ("STRING", {"default": "0", "multiline": False}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "build_image"
    CATEGORY = "VRGDG/General"
    DESCRIPTION = "Returns an EmptyImage-style image when the comma-separated index string contains 0."

    def _parse_indices(self, image_index):
        indices = []
        parts = [part.strip() for part in str(image_index or "").replace(";", ",").split(",") if part.strip()]
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                continue
            if value not in indices:
                indices.append(value)
        return indices

    def build_image(self, image_index, width, height):
        indices = self._parse_indices(image_index)
        if 0 not in indices:
            return (None,)

        empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
        return (empty_image,)


class VRGDG_OptionalMultiLoraModelOnly:
    MAX_LORA_SLOTS = 20
    NONE_LORA = "[none]"

    def __init__(self):
        self._loaded_loras = {}

    @staticmethod
    def _lora_stem(lora_name):
        if not lora_name:
            return ""
        return os.path.splitext(os.path.basename(str(lora_name)))[0]

    @classmethod
    def _lora_choices(cls):
        loras = folder_paths.get_filename_list("loras")
        return [cls.NONE_LORA] + [name for name in loras if str(name or "").strip() != cls.NONE_LORA]

    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = cls._lora_choices()
        required = {
            "model": ("MODEL",),
            "use_custom_loras": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Off passes the model through unchanged and ignores all LoRA slots.",
                },
            ),
            "lora_count": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": cls.MAX_LORA_SLOTS,
                    "step": 1,
                    "tooltip": "How many LoRA slots to show and apply. Zero applies none.",
                },
            ),
            "ltx_two_pass_mode": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "When enabled, first_pass_model uses half strength and second_pass_model uses full strength.",
                },
            ),
        }

        for i in range(1, cls.MAX_LORA_SLOTS + 1):
            required[f"lora_{i}"] = (
                lora_choices,
                {
                    "default": cls.NONE_LORA,
                    "tooltip": "Choose [none] to leave this slot unused.",
                },
            )
            required[f"strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Full-strength value. In LTX two-pass mode, first pass uses half of this.",
                },
            )

        return {"required": required}

    RETURN_TYPES = ("MODEL", "MODEL", "STRING")
    RETURN_NAMES = ("first_pass_model", "second_pass_model", "lora_names")
    FUNCTION = "apply_loras"
    CATEGORY = "VRGDG/Loaders"
    DESCRIPTION = "Safely applies optional model-only LoRAs. Defaults to [none], so shared workflows do not warn about missing LoRA files."

    def _is_none_lora(self, lora_name):
        value = str(lora_name or "").strip()
        return not value or value == self.NONE_LORA

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def _get_lora_data(self, lora_name):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if lora_path not in self._loaded_loras:
            self._loaded_loras[lora_path] = comfy.utils.load_torch_file(lora_path, safe_load=True)
        return self._loaded_loras[lora_path]

    def _collect_lora_specs(self, lora_count, kwargs):
        try:
            count = int(lora_count)
        except Exception:
            count = 0
        count = max(0, min(self.MAX_LORA_SLOTS, count))

        specs = []
        for slot in range(1, count + 1):
            lora_name = kwargs.get(f"lora_{slot}", self.NONE_LORA)
            if self._is_none_lora(lora_name):
                continue

            try:
                strength = float(kwargs.get(f"strength_{slot}", 1.0))
            except Exception:
                strength = 1.0
            if strength == 0:
                continue

            specs.append((str(lora_name), strength))
        return specs

    def _apply_specs(self, model, specs, multiplier):
        output_model = model
        for lora_name, strength in specs:
            effective_strength = float(strength) * float(multiplier)
            if effective_strength == 0:
                continue
            lora = self._get_lora_data(lora_name)
            output_model, _ = comfy.sd.load_lora_for_models(output_model, None, lora, effective_strength, 0)
        return output_model

    def apply_loras(self, model, use_custom_loras=False, lora_count=0, ltx_two_pass_mode=True, **kwargs):
        if not self._as_bool(use_custom_loras):
            return (model, model, "")

        specs = self._collect_lora_specs(lora_count, kwargs)
        if not specs:
            return (model, model, "")

        first_multiplier = 0.5 if self._as_bool(ltx_two_pass_mode) else 1.0
        second_multiplier = 1.0
        first_pass_model = self._apply_specs(model, specs, first_multiplier)
        second_pass_model = self._apply_specs(model, specs, second_multiplier)
        lora_names = ", ".join(self._lora_stem(name) for name, _ in specs)
        return (first_pass_model, second_pass_model, lora_names)


class VRGDG_NoteBox:
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": ("STRING", {"default": "Note", "multiline": False}),
                "note": (
                    "STRING",
                    {
                        "default": "Write your workflow notes here.",
                        "multiline": True,
                    },
                ),
                "font_size": ("INT", {"default": 18, "min": 12, "max": 120, "step": 1}),
            }
        }

    def run(self, title, note, font_size):
        return ()

class VRGDG_MultiStringConcat:
    MAX_STRING_SLOTS = 20

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "string_count": ("INT", {"default": 2, "min": 1, "max": cls.MAX_STRING_SLOTS, "step": 1}),
            "delimiter": (
                "STRING",
                {
                    "default": "\\n\\n",
                    "multiline": False,
                    "placeholder": "Text inserted between strings. Use \\n for newline.",
                    "tooltip": "Text inserted between each non-empty string. Use \\n for newline, \\t for tab, or clear for no delimiter.",
                },
            ),
        }
        for i in range(1, cls.MAX_STRING_SLOTS + 1):
            required[f"string_{i}"] = ("STRING", {"default": "", "multiline": True})
        return {"required": required}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concat"
    CATEGORY = "VRGDG/General"
    DESCRIPTION = "Concatenates multiple multiline string widgets with an optional delimiter."

    @staticmethod
    def _normalize_delimiter(delimiter):
        return str(delimiter or "").replace("\\r\\n", "\r\n").replace("\\n", "\n").replace("\\t", "\t")

    def concat(self, string_count, delimiter, **kwargs):
        count = max(1, min(self.MAX_STRING_SLOTS, int(string_count or 1)))
        parts = []
        for i in range(1, count + 1):
            value = kwargs.get(f"string_{i}", "")
            if value is None:
                continue
            text = str(value)
            if text == "":
                continue
            parts.append(text)
        return (self._normalize_delimiter(delimiter).join(parts),)




class VRGDG_SetMuteStateMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (any_typ,),
                "node_ids": ("STRING", {"default": "", "multiline": False}),
                "set_state": ("BOOLEAN", {"default": True, "label_on": "active", "label_off": "mute"}),
                "off_mode": (["mute", "bypass"], {"default": "mute"}),
            }
        }

    FUNCTION = "doit"
    CATEGORY = "VRGDG/General"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def _parse_node_ids(self, node_ids):
        parsed = []
        parts = [part.strip() for part in str(node_ids or "").replace(";", ",").split(",") if part.strip()]
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                continue
            if value < 0:
                continue
            if value not in parsed:
                parsed.append(value)
        return parsed

    def doit(self, signal, node_ids, set_state, off_mode):
        for node_id in self._parse_node_ids(node_ids):
            if set_state:
                PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": True})
            elif off_mode == "bypass":
                # Reuse Impact Pack bridge event to set a single node to bypass (mode=4).
                PromptServer.instance.send_sync(
                    "impact-bridge-continue",
                    {"node_id": str(node_id), "bypasses": [str(node_id)], "mutes": [], "actives": []},
                )
            else:
                PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": False})
        return (signal,)


class VRGDG_SetGroupStateMulti:
    MAX_GROUP_SLOTS = 12
    NONE_OPTION = "<none>"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "signal": (any_typ,),
            "group_count": ("INT", {"default": 1, "min": 1, "max": cls.MAX_GROUP_SLOTS, "step": 1}),
            # Legacy global fallback action (used only if per-group target map is empty).
            "group_action": (["active", "mute", "bypass"], {"default": "mute"}),
            "auto_queue_next": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
            "queue_delay_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            # Populated by frontend helper from selected group dropdowns (legacy fallback).
            "node_ids_csv": ("STRING", {"default": ""}),
            # Populated by frontend helper with per-group actions and node ids.
            "group_targets_json": ("STRING", {"default": "[]"}),
        }
        for i in range(1, cls.MAX_GROUP_SLOTS + 1):
            # Must be STRING on backend so dynamic frontend dropdown values validate.
            required[f"group_{i}"] = ("STRING", {"default": cls.NONE_OPTION})
            required[f"group_{i}_action"] = (["active", "mute", "bypass"], {"default": "mute"})
        return {"required": required}

    FUNCTION = "doit"
    CATEGORY = "VRGDG/General"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def _parse_node_ids(self, node_ids_csv):
        parsed = []
        parts = [part.strip() for part in str(node_ids_csv or "").replace(";", ",").split(",") if part.strip()]
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                continue
            if value < 0:
                continue
            if value not in parsed:
                parsed.append(value)
        return parsed

    def _apply_action(self, node_id, action):
        action = str(action or "mute").lower()
        if action == "active":
            PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": True})
        elif action == "bypass":
            PromptServer.instance.send_sync(
                "impact-bridge-continue",
                {"node_id": str(node_id), "bypasses": [str(node_id)], "mutes": [], "actives": []},
            )
        else:
            PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": False})

    def doit(
        self,
        signal,
        group_count,
        group_action,
        auto_queue_next,
        queue_delay_seconds,
        node_ids_csv,
        group_targets_json,
        **kwargs,
    ):
        applied = False

        # Preferred path: explicit per-group action + node id targets generated by frontend helper.
        try:
            targets = json.loads(str(group_targets_json or "[]"))
        except Exception:
            targets = []

        if isinstance(targets, list):
            for target in targets:
                if not isinstance(target, dict):
                    continue
                action = target.get("action", "mute")
                node_ids = target.get("node_ids", [])
                if not isinstance(node_ids, list):
                    continue
                for raw_id in node_ids:
                    try:
                        node_id = int(raw_id)
                    except Exception:
                        continue
                    if node_id < 0:
                        continue
                    self._apply_action(node_id, action)
                    applied = True

        # Backward fallback: one action for all collected ids.
        if not applied:
            node_ids = self._parse_node_ids(node_ids_csv)
            for node_id in node_ids:
                self._apply_action(node_id, group_action)
                applied = True

        # Apply mode changes on frontend for root+subgraphs (including subgraph contents).
        # Impact Pack events can miss some subgraph internals depending on graph context.
        if isinstance(targets, list) and len(targets) > 0:
            PromptServer.instance.send_sync("vrgdg-apply-node-modes", {"targets": targets})

        # Important: mode changes affect future runs; queue one more run to continue the chain.
        if applied and bool(auto_queue_next):
            delay = max(0.0, float(queue_delay_seconds or 0.0))
            if delay <= 0:
                PromptServer.instance.send_sync("impact-add-queue", {})
            else:
                # Non-blocking delayed queue to allow mode changes to settle first.
                def _delayed_queue():
                    time.sleep(delay)
                    PromptServer.instance.send_sync("impact-add-queue", {})

                threading.Thread(target=_delayed_queue, daemon=True).start()
        return (signal,)






class VRGDG_MuteUnmute4PromptCreatorWF_1(VRGDG_SetGroupStateMulti):
    pass


class VRGDG_MuteUnmute4PromptCreatorWF_2(VRGDG_SetGroupStateMulti):
    pass


_ensure_test_save_route_registered()


class VRGDG_MuteUnmute4PromptCreatorWF_0(VRGDG_SetGroupStateMulti):
    pass


_ensure_test_save_route_registered()



class VRGDG_LyricSegmentJsonFixer:
    OUTPUT_KEY_PATTERN = "lyricSegment"
    ACCEPTED_KEY_PREFIXES = ("lyricSegment", "segment")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON", "BOOLEAN", "STRING")
    RETURN_NAMES = ("fixed_text", "json_output", "was_fixed", "notes")
    FUNCTION = "fix_json"
    CATEGORY = "VRGDG/General"

    def _strip_markdown_json_fence(self, text):
        value = str(text or "").strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines:
                first = lines[0].strip().lower()
                if first == "```" or first.startswith("```json"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                value = "\n".join(lines).strip()
        return value

    def _basic_cleanup(self, text):
        cleaned = self._strip_markdown_json_fence(text)
        cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "")
        cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        return cleaned.strip()

    def _escape_unescaped_inner_quotes(self, s):
        chars = []
        in_string = False
        escaped = False
        n = len(s)
        i = 0

        while i < n:
            ch = s[i]

            if not in_string:
                chars.append(ch)
                if ch == '"':
                    in_string = True
                    escaped = False
                i += 1
                continue

            if escaped:
                chars.append(ch)
                escaped = False
                i += 1
                continue

            if ch == "\\":
                chars.append(ch)
                escaped = True
                i += 1
                continue

            if ch == '"':
                j = i + 1
                while j < n and s[j].isspace():
                    j += 1
                next_ch = s[j] if j < n else ""
                if next_ch not in [",", "}", "]", ":", ""]:
                    chars.append("\\")
                    chars.append('"')
                    i += 1
                    continue

                chars.append(ch)
                in_string = False
                i += 1
                continue

            chars.append(ch)
            i += 1

        return "".join(chars)

    def _json_object_slices(self, text):
        slices = []
        stack = []
        start = None
        in_string = False
        escaped = False

        for i, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                escaped = False
                continue

            if ch == "{":
                if not stack:
                    start = i
                stack.append(ch)
                continue

            if ch == "}" and stack:
                stack.pop()
                if not stack and start is not None:
                    slices.append(text[start : i + 1])
                    start = None

        return slices

    def _extract_json_slice(self, text):
        slices = self._json_object_slices(text)
        if slices:
            return slices[-1]
        start = text.find("{")
        if start < 0:
            return text
        end = text.rfind("}")
        if end >= start:
            return text[start : end + 1]
        return text[start:]

    def _remove_duplicate_open_braces(self, text):
        chars = []
        in_string = False
        escaped = False
        i = 0
        changes = 0

        while i < len(text):
            ch = text[i]
            if in_string:
                chars.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            if ch == '"':
                in_string = True
                chars.append(ch)
                i += 1
                continue

            if ch == "{":
                chars.append(ch)
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == "{":
                    changes += 1
                    i = j
                    continue
                i += 1
                continue

            chars.append(ch)
            i += 1

        return "".join(chars), changes

    def _remove_trailing_commas(self, text):
        import re

        updated = re.sub(r",(\s*[}\]])", r"\1", text)
        return updated, int(updated != text)

    def _insert_missing_key_commas(self, text):
        import re

        updated = re.sub(
            r'("(?:(?:[A-Za-z]*segment[A-Za-z]*)|(?:segment))\d+"\s*:\s*"((?:\\.|[^"\\])*)")(\s*)"(?=(?:(?:[A-Za-z]*segment[A-Za-z]*)|(?:segment))\d+"\s*:)',
            r'\1,\3"',
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return updated, int(updated != text)

    def _remove_loose_text_before_keys(self, text):
        updated = re.sub(
            r'([,{]\s*)[^"{}\[\],:\r\n]+(?="[^"\r\n]*segment[^"\r\n]*\d+"\s*:)',
            r'\1',
            text,
            flags=re.IGNORECASE,
        )
        return updated, int(updated != text)

    def _balance_outer_structure(self, text):
        stripped = text.strip()
        changes = 0
        if stripped.startswith("{") and stripped.count("{") > stripped.count("}"):
            text += "}" * (stripped.count("{") - stripped.count("}"))
            changes += 1
        return text, changes

    def _format_json_error(self, exc, text, label):
        if not isinstance(exc, json.JSONDecodeError):
            return f"{label}: {exc}"

        lines = str(text or "").splitlines()
        context = ""
        if 1 <= exc.lineno <= len(lines):
            line = lines[exc.lineno - 1]
            pointer = " " * max(0, exc.colno - 1) + "^"
            context = f" Line {exc.lineno}, column {exc.colno}:\n{line}\n{pointer}"
        return f"{label}: {exc.msg}.{context}"

    def _split_key(self, key):
        if not isinstance(key, str):
            return None, None
        stripped = key.strip()
        lowered = stripped.lower()
        for prefix in self.ACCEPTED_KEY_PREFIXES:
            if lowered.startswith(prefix.lower()):
                suffix = stripped[len(prefix) :]
                if str(suffix).isdigit():
                    return prefix, suffix
        match = re.fullmatch(r"(?i)([A-Za-z]*segment[A-Za-z]*)(\d+)", stripped)
        if match:
            return self.OUTPUT_KEY_PATTERN, match.group(2)
        compact = re.sub(r"[^A-Za-z0-9]", "", stripped)
        match = re.fullmatch(r"(?i)([A-Za-z]*segment[A-Za-z]*)(\d+)", compact)
        if match:
            return self.OUTPUT_KEY_PATTERN, match.group(2)
        match = re.fullmatch(r"(?i)((?:lyric|segment)[A-Za-z]*)(\d+)", compact)
        if match:
            return self.OUTPUT_KEY_PATTERN, match.group(2)
        match = re.fullmatch(r"(?i)([ls][A-Za-z0-9]*?)(\d+)", compact)
        if match:
            return self.OUTPUT_KEY_PATTERN, match.group(2)
        return None, None

    def _payload_items(self, data):
        if isinstance(data, dict):
            return list(data.items())
        if isinstance(data, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in data):
            return data
        return None

    def _payload_numbers(self, data):
        numbers = []
        items = self._payload_items(data) or []
        for key, _ in items:
            _, suffix = self._split_key(key)
            try:
                numbers.append(int(str(suffix)))
            except Exception:
                pass
        return numbers

    def _parse_json_preserving_order(self, text):
        return json.loads(text, object_pairs_hook=list)

    def _validate_payload(self, data):
        errors = []
        items = self._payload_items(data)
        if items is None:
            return ["Top-level JSON must be an object of lyricSegment/segment keys."]

        if not items:
            return ["At least one lyricSegment or segment key is required."]

        valid_item_count = 0
        found_prefixes = set()
        for key, value in items:
            prefix, suffix = self._split_key(key)
            if prefix is None:
                errors.append(f"Invalid key '{key}'. Expected keys like lyricSegment1 or segment1.")
                continue
            found_prefixes.add(prefix)
            try:
                segment_number = int(suffix)
            except Exception:
                errors.append(f"Invalid key '{key}'. Expected numeric suffix, e.g. lyricSegment1 or segment1.")
                continue
            if segment_number <= 0:
                errors.append(f"Invalid segment number in '{key}'. It must be greater than 0.")
                continue
            valid_item_count += 1
            if not isinstance(value, str):
                errors.append(f"{key} must be a string.")

        if not valid_item_count:
            errors.append("No valid lyricSegment/segment keys were found.")

        return errors

    def _normalize_payload(self, data):
        validation_errors = self._validate_payload(data)
        if validation_errors:
            raise ValueError(" ".join(validation_errors))

        normalized = {}
        for idx, (key, value) in enumerate(self._payload_items(data), start=1):
            normalized[f"{self.OUTPUT_KEY_PATTERN}{idx}"] = "" if value is None else str(value)
        return normalized

    def _repair_schema_text(self, text):
        notes = []
        working = self._basic_cleanup(text)
        sliced = self._extract_json_slice(working)
        if sliced != working:
            notes.append("trimmed extra text outside JSON")
            working = sliced

        working, duplicate_count = self._remove_duplicate_open_braces(working)
        if duplicate_count:
            notes.append(f"removed duplicate '{{' x{duplicate_count}")

        escaped_quotes = self._escape_unescaped_inner_quotes(working)
        if escaped_quotes != working:
            working = escaped_quotes
            notes.append("escaped inner quotes inside segment text")

        working, comma_cleanup = self._remove_trailing_commas(working)
        if comma_cleanup:
            notes.append("removed trailing commas")

        working, inserted_commas = self._insert_missing_key_commas(working)
        if inserted_commas:
            notes.append(f"inserted missing commas between lyric segments x{inserted_commas}")

        working, loose_key_prefixes = self._remove_loose_text_before_keys(working)
        if loose_key_prefixes:
            notes.append(f"removed loose text before segment keys x{loose_key_prefixes}")

        working, balance_changes = self._balance_outer_structure(working)
        if balance_changes:
            notes.append("balanced closing braces")

        return working, notes

    def fix_json(self, text):
        original = self._basic_cleanup(text)
        notes = []

        try:
            parsed = self._parse_json_preserving_order(original)
        except json.JSONDecodeError as exc:
            repaired_text, notes = self._repair_schema_text(text)
            try:
                parsed = self._parse_json_preserving_order(repaired_text)
            except json.JSONDecodeError as repaired_exc:
                original_error = self._format_json_error(exc, original, "Original JSON parse failed")
                repaired_error = self._format_json_error(repaired_exc, repaired_text, "Repair attempt still invalid")
                raise ValueError(f"VRGDG_LyricSegmentJsonFixer: {original_error}\n{repaired_error}")
        else:
            repaired_text = original

        original_numbers = self._payload_numbers(parsed)
        expected_numbers = list(range(1, len(original_numbers) + 1))
        if original_numbers and original_numbers != expected_numbers:
            notes.append("renumbered lyricSegment keys sequentially")

        try:
            normalized = self._normalize_payload(parsed)
        except ValueError as exc:
            raise ValueError(f"VRGDG_LyricSegmentJsonFixer schema error: {exc}")

        fixed_text = json.dumps(normalized, indent=2, ensure_ascii=False)
        was_fixed = bool(notes) or fixed_text.strip() != original.strip()
        note_text = "; ".join(notes) if notes else ("normalized formatting" if was_fixed else "")
        return (fixed_text, normalized, was_fixed, note_text)


class VRGDG_LyricSegmentTextCleaner:
    FILLER_WORDS = {"oh", "you"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics_text": ("STRING", {"multiline": True, "default": ""}),
                "repeat_output_count": ("INT", {"default": 3, "min": 2, "max": 8, "step": 1}),
                "min_repeats_to_collapse": ("INT", {"default": 4, "min": 2, "max": 50, "step": 1}),
                "bridge_single_word_segments": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When a segment has one non-filler word, blend it with neighboring lyric words.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("cleaned_lyrics_text", "changed_count", "notes")
    FUNCTION = "clean"
    CATEGORY = "VRGDG/General"
    DESCRIPTION = "Cleans extracted lyricSegmentN text by shortening repeated filler lyrics and smoothing one-word fragments."

    @staticmethod
    def _parse_segment_line(line):
        match = re.match(r"^(\s*lyricSegment)(\d+)(\s*=\s*)(.*)$", str(line or ""), re.IGNORECASE)
        if not match:
            return None
        return {
            "prefix": match.group(1),
            "number": int(match.group(2)),
            "separator": match.group(3),
            "text": match.group(4).strip(),
        }

    @staticmethod
    def _word_tokens(text):
        return re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", str(text or ""))

    @staticmethod
    def _clean_word(word, title_case=False):
        value = str(word or "").strip()
        if not value:
            return ""
        value = value[0].upper() + value[1:].lower()
        return value if title_case else value

    @classmethod
    def _is_filler_word(cls, text):
        words = cls._word_tokens(text)
        return len(words) == 1 and words[0].lower() in cls.FILLER_WORDS

    @classmethod
    def _collapse_repeated_words(cls, text, repeat_output_count, min_repeats_to_collapse):
        words = cls._word_tokens(text)
        if not words:
            return None

        lowered = [word.lower() for word in words]
        if len(set(lowered)) != 1:
            return None

        word = lowered[0]
        if len(words) < int(min_repeats_to_collapse) and word not in cls.FILLER_WORDS:
            return None

        display_word = "Oh" if word in cls.FILLER_WORDS else cls._clean_word(words[0])
        return ", ".join([display_word] * int(repeat_output_count)) + "."

    @staticmethod
    def _last_word_before(segments, current_index):
        for index in range(current_index - 1, -1, -1):
            words = VRGDG_LyricSegmentTextCleaner._word_tokens(segments[index].get("original_text", segments[index]["text"]))
            if words:
                return words[-1], len(words) > 1
        return "", False

    @staticmethod
    def _first_words_after(segments, current_index):
        for index in range(current_index + 1, len(segments)):
            words = VRGDG_LyricSegmentTextCleaner._word_tokens(segments[index].get("original_text", segments[index]["text"]))
            if words:
                if words[0].lower() == "the" and len(words) > 1:
                    return words[:2]
                return words[:1]
        return []

    @classmethod
    def _bridge_single_word(cls, segments, current_index):
        current_words = cls._word_tokens(segments[current_index]["text"])
        if len(current_words) != 1:
            return None

        current = current_words[0]
        previous, previous_from_phrase = cls._last_word_before(segments, current_index)
        next_words = cls._first_words_after(segments, current_index)

        parts = []
        if previous and previous.lower() != current.lower():
            parts.append(cls._clean_word(previous) if previous_from_phrase else previous.lower())

        parts.append(current.lower())

        if next_words:
            first_next = next_words[0]
            if first_next.lower() != current.lower():
                if first_next.lower() == "the":
                    tail = " ".join(cls._clean_word(word) for word in next_words)
                    if len(parts) > 1:
                        return f"{parts[0]}, {parts[1]}. {tail}."
                    return f"{parts[0]}. {tail}."
                parts.append(first_next.lower())

        if len(parts) <= 1:
            return None
        return ", ".join(parts) + "."

    def clean(self, lyrics_text, repeat_output_count=3, min_repeats_to_collapse=4, bridge_single_word_segments=True):
        lines = str(lyrics_text or "").splitlines()
        parsed_by_line = {}
        segments = []

        for line_index, line in enumerate(lines):
            parsed = self._parse_segment_line(line)
            if parsed is None:
                continue
            parsed["line_index"] = line_index
            parsed["original_text"] = parsed["text"]
            parsed_by_line[line_index] = parsed
            segments.append(parsed)

        changed_count = 0
        notes = []

        for segment_index, segment in enumerate(segments):
            original_text = segment["text"]
            replacement = self._collapse_repeated_words(
                original_text,
                repeat_output_count,
                min_repeats_to_collapse,
            )
            if replacement is None and self._is_filler_word(original_text):
                replacement = ", ".join(["Oh"] * int(repeat_output_count)) + "."
            if replacement is None and bool(bridge_single_word_segments):
                replacement = self._bridge_single_word(segments, segment_index)

            if replacement and replacement != original_text:
                segment["text"] = replacement
                changed_count += 1
                notes.append(f"lyricSegment{segment['number']}")

        output_lines = list(lines)
        for line_index, segment in parsed_by_line.items():
            output_lines[line_index] = f"{segment['prefix']}{segment['number']}{segment['separator']}{segment['text']}"

        note_text = "Cleaned " + ", ".join(notes) if notes else "No lyric cleanup needed"
        return ("\n".join(output_lines), changed_count, note_text)


class VRGDG_PromptMapJsonFixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "use_srt_file": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, count scene entries in the connected SRT file/text and require it to match the prompt count.",
                    },
                ),
            },
            "optional": {
                "srt_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "forceInput": True,
                        "tooltip": "Connect an SRT file path, or raw SRT text. Ignored when Use SRT File is off.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "JSON", "BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("fixed_text", "json_output", "was_fixed", "notes", "prompt_count")
    FUNCTION = "fix_json"
    CATEGORY = "VRGDG/General"

    def _strip_markdown_json_fence(self, text):
        value = str(text or "").strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines:
                first = lines[0].strip().lower()
                if first == "```" or first.startswith("```json"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                value = "\n".join(lines).strip()
        return value

    def _basic_cleanup(self, text):
        cleaned = self._strip_markdown_json_fence(text)
        cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "")
        cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        return cleaned.strip()

    def _extract_json_slice(self, text):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return text[start : end + 1]
        if start >= 0:
            return text[start:]
        return text

    def _remove_trailing_commas(self, text):
        return re.sub(r",(\s*[}\]])", r"\1", text)

    def _normalize_prompt_text(self, value):
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = str(value)
        return " ".join(value.replace("\r", " ").replace("\n", " ").split())

    def _normalize_from_mapping(self, data):
        prompts = {}
        notes = []

        for key, value in data.items():
            key_text = str(key)
            match = re.search(r"(\d+)", key_text)
            if not match:
                continue
            index = int(match.group(1))
            if index <= 0:
                continue
            if not re.fullmatch(r"Prompt\d+", key_text):
                notes.append(f"renamed {key_text} to Prompt{index}")
            if index in prompts:
                notes.append(f"duplicate Prompt{index}; kept last value")
            prompts[index] = self._normalize_prompt_text(value)

        if not prompts and data:
            for index, value in enumerate(data.values(), start=1):
                prompts[index] = self._normalize_prompt_text(value)
            notes.append("no numbered prompt keys found; used object order")

        return prompts, notes

    def _extract_prompt_entries(self, text):
        entries = {}
        notes = ["rebuilt object from Prompt entries"]
        pattern = re.compile(
            r'(?i)(?:^|[,{]\s*|[\r\n]\s*)[A-Za-z]*"?Prompt[A-Za-z]*(\d+)"?\s*:\s*"((?:\\.|[^"\\])*)"',
            re.DOTALL,
        )

        for match in pattern.finditer(text):
            index = int(match.group(1))
            if index <= 0:
                continue
            raw_value = match.group(2)
            try:
                value = json.loads(f'"{raw_value}"')
            except Exception:
                value = raw_value.replace('\\"', '"')
            if index in entries:
                notes.append(f"duplicate Prompt{index}; kept last value")
            entries[index] = self._normalize_prompt_text(value)

        return entries, notes

    def _build_output(self, prompts):
        normalized = {
            f"Prompt{index}": prompts[index]
            for index in sorted(prompts)
        }
        return normalized

    def _read_srt_source(self, srt_file):
        value = str(srt_file or "").strip().strip("\"'")
        if not value:
            raise ValueError("VRGDG_PromptMapJsonFixer: Use SRT File is enabled, but no SRT file/text was connected.")

        if os.path.isfile(value):
            with open(value, "r", encoding="utf-8-sig") as file_obj:
                return file_obj.read(), value

        if "-->" in value:
            return value, "connected SRT text"

        raise ValueError(
            "VRGDG_PromptMapJsonFixer: connected SRT value is not an existing file path and does not look like SRT text."
        )

    def _count_srt_scenes(self, srt_file):
        srt_text, source_label = self._read_srt_source(srt_file)
        timestamp_lines = re.findall(
            r"(?m)^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}.*$",
            str(srt_text or ""),
        )
        if not timestamp_lines:
            raise ValueError(
                f"VRGDG_PromptMapJsonFixer: no SRT timestamp lines were found in {source_label}."
            )
        return len(timestamp_lines), source_label

    def fix_json(self, text, use_srt_file=False, srt_file=""):
        original = str(text or "")
        cleaned = self._basic_cleanup(original)
        sliced = self._extract_json_slice(cleaned)
        candidate = self._remove_trailing_commas(sliced)
        notes = []

        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("top-level JSON is not an object")
            prompts, normalize_notes = self._normalize_from_mapping(parsed)
            notes.extend(normalize_notes)
        except Exception:
            prompts, extract_notes = self._extract_prompt_entries(candidate)
            notes.extend(extract_notes)

        normalized = self._build_output(prompts)
        prompt_count = len(normalized)

        if bool(use_srt_file):
            srt_scene_count, source_label = self._count_srt_scenes(srt_file)
            if prompt_count != srt_scene_count:
                raise ValueError(
                    "VRGDG_PromptMapJsonFixer: prompt count does not match SRT scene count. "
                    f"Prompts: {prompt_count}, SRT scenes: {srt_scene_count}. Source: {source_label}."
                )
            notes.append(f"SRT scene count matched prompt count ({prompt_count})")

        fixed_text = json.dumps(normalized, indent=2, ensure_ascii=False)
        was_fixed = fixed_text.strip() != cleaned.strip()
        if cleaned.startswith("```"):
            notes.append("removed markdown code fence")
        if candidate != cleaned:
            notes.append("trimmed text outside JSON or removed trailing commas")
        if was_fixed and not notes:
            notes.append("normalized formatting")

        return (fixed_text, normalized, was_fixed, "; ".join(notes), prompt_count)


class VRGDG_PromptJsonSubjectPrepender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "a female wearing a red dress",
                    },
                ),
                "prompt_json": (any_typ, {"multiline": True, "default": "{}"}),
                "separator": (
                    "STRING",
                    {
                        "default": ", ",
                        "multiline": False,
                        "tooltip": "Text inserted between the subject and each prompt.",
                    },
                ),
                "skip_if_already_starts_with_subject": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Avoids adding the subject twice when a prompt already starts with it.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "JSON", "INT")
    RETURN_NAMES = ("json_text", "json_output", "prompt_count")
    FUNCTION = "prepend_subject"
    CATEGORY = "VRGDG/General"
    DESCRIPTION = "Prepends the same subject text to every Prompt value in prompt-map JSON."

    @staticmethod
    def _strip_markdown_json_fence(text):
        value = str(text or "").strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines:
                first = lines[0].strip().lower()
                if first == "```" or first.startswith("```json"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                value = "\n".join(lines).strip()
        return value

    @staticmethod
    def _extract_json_slice(text):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return text[start : end + 1]
        return text

    @staticmethod
    def _normalize_prompt_text(value):
        if value is None:
            return ""
        return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def _load_prompt_map(self, prompt_json):
        if isinstance(prompt_json, dict):
            return prompt_json

        cleaned = self._strip_markdown_json_fence(prompt_json)
        cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "")
        candidate = self._extract_json_slice(cleaned)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VRGDG_PromptJsonSubjectPrepender: invalid prompt JSON: {exc}")
        if not isinstance(parsed, dict):
            raise ValueError("VRGDG_PromptJsonSubjectPrepender: prompt JSON must be an object.")
        return parsed

    def prepend_subject(self, subject, prompt_json, separator=", ", skip_if_already_starts_with_subject=True):
        subject_text = self._normalize_prompt_text(subject)
        separator_text = str(separator or "")
        prompt_map = self._load_prompt_map(prompt_json)
        skip_existing = self._as_bool(skip_if_already_starts_with_subject)

        output = {}
        for key, value in prompt_map.items():
            prompt_text = self._normalize_prompt_text(value)
            if subject_text and not (skip_existing and prompt_text.lower().startswith(subject_text.lower())):
                prompt_text = f"{subject_text}{separator_text}{prompt_text}" if prompt_text else subject_text
            output[str(key)] = prompt_text

        json_text = json.dumps(output, indent=2, ensure_ascii=False)
        return (json_text, output, len(output))


class VRGDG_LyricSegmentDurationMerger:
    ACCEPTED_KEY_PREFIXES = ("lyricSegment", "segment")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_text": ("STRING", {"multiline": True, "default": ""}),
                "segments_json": ("STRING", {"multiline": True, "default": "{}"}),
                "strict_count_match": ("BOOLEAN", {"default": True}),
                "decimal_places": ("INT", {"default": 3, "min": 0, "max": 6, "step": 1}),
                "use_srt_durations": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON", "INT", "INT")
    RETURN_NAMES = ("merged_text", "merged_json", "segment_count", "duration_count")
    FUNCTION = "merge"
    CATEGORY = "VRGDG/General"

    def _strip_markdown_json_fence(self, text):
        value = str(text or "").strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines:
                first = lines[0].strip().lower()
                if first == "```" or first.startswith("```json"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                value = "\n".join(lines).strip()
        return value

    def _split_key(self, key):
        if not isinstance(key, str):
            return None, None
        for prefix in self.ACCEPTED_KEY_PREFIXES:
            if key.startswith(prefix):
                return prefix, key[len(prefix) :]
        return None, None

    def _parse_segments(self, segments_json):
        cleaned = self._strip_markdown_json_fence(segments_json)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"VRGDG_LyricSegmentDurationMerger: segment JSON is invalid at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            )

        if not isinstance(data, dict):
            raise ValueError("VRGDG_LyricSegmentDurationMerger: segment JSON must be an object.")

        found_prefixes = set()
        ordered_segments = []
        for key, value in data.items():
            prefix, suffix = self._split_key(key)
            if prefix is None:
                raise ValueError(
                    f"VRGDG_LyricSegmentDurationMerger: invalid key '{key}'. "
                    "Expected keys like lyricSegment1 or segment1."
                )
            found_prefixes.add(prefix)
            try:
                index = int(suffix)
            except Exception:
                raise ValueError(
                    f"VRGDG_LyricSegmentDurationMerger: invalid key '{key}'. "
                    "Numeric suffix is required."
                )
            if index <= 0:
                raise ValueError(
                    f"VRGDG_LyricSegmentDurationMerger: invalid key '{key}'. Index must be greater than 0."
                )
            if not isinstance(value, str):
                raise ValueError(f"VRGDG_LyricSegmentDurationMerger: {key} must map to a string.")
            ordered_segments.append((index, key, value))

        if not ordered_segments:
            raise ValueError("VRGDG_LyricSegmentDurationMerger: no segment keys were found.")

        if len(found_prefixes) > 1:
            raise ValueError(
                "VRGDG_LyricSegmentDurationMerger: do not mix 'segmentN' and 'lyricSegmentN' keys."
            )

        ordered_segments.sort(key=lambda item: item[0])
        expected = list(range(1, len(ordered_segments) + 1))
        actual = [item[0] for item in ordered_segments]
        if actual != expected:
            raise ValueError(
                "VRGDG_LyricSegmentDurationMerger: segment keys must be sequential starting at 1. "
                f"Found: {', '.join(str(v) for v in actual)}."
            )
        return ordered_segments

    def _to_seconds(self, timestamp):
        hours, minutes, seconds_ms = timestamp.split(":")
        seconds, milliseconds = seconds_ms.split(",")
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(milliseconds) / 1000.0
        )

    def _parse_durations(self, srt_text):
        import re

        matches = re.findall(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            str(srt_text or ""),
        )
        if not matches:
            raise ValueError("VRGDG_LyricSegmentDurationMerger: no SRT timestamps were found.")

        durations = []
        for start, end in matches:
            duration = self._to_seconds(end) - self._to_seconds(start)
            if duration < 0:
                raise ValueError(
                    "VRGDG_LyricSegmentDurationMerger: found a subtitle end time earlier than its start time."
                )
            durations.append(duration)
        return durations

    def _format_duration(self, value, decimal_places):
        rounded = round(float(value), int(decimal_places))
        text = f"{rounded:.{int(decimal_places)}f}" if int(decimal_places) > 0 else str(int(round(rounded)))
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    def merge(self, srt_text, segments_json, strict_count_match=True, decimal_places=3, use_srt_durations=True):
        ordered_segments = self._parse_segments(segments_json)
        durations = self._parse_durations(srt_text) if use_srt_durations else []

        if use_srt_durations and strict_count_match and len(ordered_segments) != len(durations):
            raise ValueError(
                "VRGDG_LyricSegmentDurationMerger: segment count does not match SRT duration count. "
                f"Segments: {len(ordered_segments)}, durations: {len(durations)}."
            )

        merged = {}
        for idx, (_, original_key, value) in enumerate(ordered_segments):
            if not use_srt_durations:
                merged[original_key] = value
                continue
            duration_value = durations[idx] if idx < len(durations) else 0.0
            duration_text = self._format_duration(duration_value, decimal_places)
            merged[f"{original_key}_duration_{duration_text}"] = value

        merged_text = json.dumps(merged, indent=2, ensure_ascii=False)
        return (merged_text, merged, len(ordered_segments), len(durations))


class VRGDG_PromptCreatorUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "VRGDG/General"

    def noop(self):
        return ()

class VRGDG_PromptCreatorUI_V2(VRGDG_PromptCreatorUI):
    pass

class VRGDG_Part2WorkflowUI(VRGDG_PromptCreatorUI):
    pass


class VRGDG_Part3WorkflowUI(VRGDG_PromptCreatorUI):
    pass


class VRGDG_T2IPromptsFromConcepts:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .LLM import VRGDG_SuperGemmaGGUFChat

            gemma_choices = VRGDG_SuperGemmaGGUFChat._list_local_gemma_gguf_choices()
        except Exception:
            gemma_choices = ["[No Gemma GGUF found in models/LLM]"]

        return {
            "required": {
                "model_file": (
                    gemma_choices,
                    {
                        "default": gemma_choices[0],
                        "tooltip": "Gemma GGUF model used to create t2i prompts from ConceptPrompts.txt.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "VRGDG/General"

    def noop(self, model_file):
        return ()


class VRGDG_T2VPromptsFromConcepts:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .LLM import VRGDG_SuperGemmaGGUFChat

            gemma_choices = VRGDG_SuperGemmaGGUFChat._list_local_gemma_gguf_choices()
        except Exception:
            gemma_choices = ["[No Gemma GGUF found in models/LLM]"]

        return {
            "required": {
                "model_file": (
                    gemma_choices,
                    {
                        "default": gemma_choices[0],
                        "tooltip": "Gemma GGUF model used to create t2v prompts from ConceptPrompts.txt.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "VRGDG/General"

    def noop(self, model_file):
        return ()


class VRGDG_StoryGroupJsonFixer:
    REQUIRED_GROUP_KEYS = ("index", "subject", "camera", "scene_and_lighting", "frame")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON", "BOOLEAN", "STRING")
    RETURN_NAMES = ("fixed_text", "json_output", "was_fixed", "notes")
    FUNCTION = "fix_json"
    CATEGORY = "VRGDG/General"

    def _strip_markdown_json_fence(self, text):
        value = str(text or "").strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines:
                first = lines[0].strip().lower()
                if first == "```" or first.startswith("```json"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                value = "\n".join(lines).strip()
        return value

    def _basic_cleanup(self, text):
        cleaned = self._strip_markdown_json_fence(text)
        cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "")
        cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        return cleaned.strip()

    def _extract_json_slice(self, text):
        start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
        if not start_candidates:
            return text
        start = min(start_candidates)
        end_obj = text.rfind("}")
        end_arr = text.rfind("]")
        end = max(end_obj, end_arr)
        if end >= start:
            return text[start : end + 1]
        return text[start:]

    def _remove_duplicate_open_braces(self, text):
        chars = []
        in_string = False
        escaped = False
        i = 0
        changes = 0

        while i < len(text):
            ch = text[i]
            if in_string:
                chars.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            if ch == '"':
                in_string = True
                chars.append(ch)
                i += 1
                continue

            if ch == "{":
                chars.append(ch)
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == "{":
                    changes += 1
                    i = j
                    continue
                i += 1
                continue

            chars.append(ch)
            i += 1

        return "".join(chars), changes

    def _remove_trailing_commas(self, text):
        import re

        updated = re.sub(r",(\s*[}\]])", r"\1", text)
        return updated, int(updated != text)

    def _insert_missing_object_commas(self, text):
        chars = []
        in_string = False
        escaped = False
        changes = 0
        i = 0

        while i < len(text):
            ch = text[i]
            chars.append(ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            if ch == '"':
                in_string = True
                i += 1
                continue

            if ch == "}":
                j = i + 1
                whitespace = []
                while j < len(text) and text[j].isspace():
                    whitespace.append(text[j])
                    j += 1
                if j < len(text) and text[j] == "{":
                    chars.extend(whitespace)
                    chars.append(",")
                    changes += 1
                    i = j
                    continue
            i += 1

        return "".join(chars), changes

    def _balance_outer_structure(self, text):
        stripped = text.strip()
        changes = 0
        if stripped.startswith("{") and stripped.count("{") > stripped.count("}"):
            text += "}" * (stripped.count("{") - stripped.count("}"))
            changes += 1
        if stripped.startswith("[") and stripped.count("[") > stripped.count("]"):
            text += "]" * (stripped.count("[") - stripped.count("]"))
            changes += 1
        if '"groups"' in text:
            prefix = text.split('"groups"', 1)[0]
            if prefix.count("[") > prefix.count("]") + 1:
                text += "]" * (prefix.count("[") - prefix.count("]") - 1)
                changes += 1
            if text.count("[") > text.count("]"):
                text += "]" * (text.count("[") - text.count("]"))
                changes += 1
        return text, changes

    def _format_json_error(self, exc, text, label):
        if not isinstance(exc, json.JSONDecodeError):
            return f"{label}: {exc}"

        lines = str(text or "").splitlines()
        context = ""
        if 1 <= exc.lineno <= len(lines):
            line = lines[exc.lineno - 1]
            pointer = " " * max(0, exc.colno - 1) + "^"
            context = f" Line {exc.lineno}, column {exc.colno}:\n{line}\n{pointer}"
        return f"{label}: {exc.msg}.{context}"

    def _validate_story_payload(self, data):
        errors = []
        if not isinstance(data, dict):
            return ["Top-level JSON must be an object with 'story_summary' and 'groups'."]

        if "story_summary" not in data:
            errors.append("Missing top-level key 'story_summary'.")
        elif not isinstance(data.get("story_summary"), str):
            errors.append("'story_summary' must be a string.")

        if "groups" not in data:
            errors.append("Missing top-level key 'groups'.")
            return errors

        groups = data.get("groups")
        if not isinstance(groups, list):
            errors.append("'groups' must be a list.")
            return errors

        seen_indexes = set()
        for pos, group in enumerate(groups, start=1):
            if not isinstance(group, dict):
                errors.append(f"groups[{pos}] must be an object.")
                continue

            missing = [key for key in self.REQUIRED_GROUP_KEYS if key not in group]
            if missing:
                errors.append(f"groups[{pos}] is missing keys: {', '.join(missing)}.")

            if "index" in group:
                try:
                    index_value = int(group.get("index"))
                    if index_value <= 0:
                        errors.append(f"groups[{pos}].index must be greater than 0.")
                    elif index_value in seen_indexes:
                        errors.append(f"Duplicate group index {index_value}.")
                    else:
                        seen_indexes.add(index_value)
                except Exception:
                    errors.append(f"groups[{pos}].index must be an integer.")

            for key in self.REQUIRED_GROUP_KEYS[1:]:
                if key in group and not isinstance(group.get(key), str):
                    errors.append(f"groups[{pos}].{key} must be a string.")

        return errors

    def _normalize_group(self, item, fallback_index):
        if not isinstance(item, dict):
            item = {}

        normalized = {}
        raw_index = item.get("index", fallback_index)
        try:
            normalized["index"] = int(raw_index)
        except Exception:
            normalized["index"] = fallback_index

        for key in self.REQUIRED_GROUP_KEYS[1:]:
            value = item.get(key, "")
            if value is None:
                value = ""
            elif not isinstance(value, str):
                value = str(value)
            normalized[key] = value
        return normalized

    def _normalize_story_payload(self, data):
        validation_errors = self._validate_story_payload(data)
        if validation_errors:
            raise ValueError(" ".join(validation_errors))

        story_summary = data.get("story_summary", "")
        groups = data.get("groups", [])

        normalized_groups = []
        for idx, group in enumerate(groups, start=1):
            normalized_groups.append(self._normalize_group(group, idx))

        normalized_groups.sort(key=lambda item: item.get("index", 0))
        for idx, group in enumerate(normalized_groups, start=1):
            if group.get("index") <= 0:
                group["index"] = idx

        return {
            "story_summary": story_summary,
            "groups": normalized_groups,
        }

    def _parse_json_preserving_order(self, text):
        return json.loads(text)

    def _repair_schema_text(self, text):
        notes = []
        working = self._basic_cleanup(text)
        sliced = self._extract_json_slice(working)
        if sliced != working:
            notes.append("trimmed extra text outside JSON")
            working = sliced

        working, duplicate_count = self._remove_duplicate_open_braces(working)
        if duplicate_count:
            notes.append(f"removed duplicate '{{' x{duplicate_count}")

        working, comma_cleanup = self._remove_trailing_commas(working)
        if comma_cleanup:
            notes.append("removed trailing commas")

        working, inserted_commas = self._insert_missing_object_commas(working)
        if inserted_commas:
            notes.append(f"inserted missing commas between objects x{inserted_commas}")

        working, balance_changes = self._balance_outer_structure(working)
        if balance_changes:
            notes.append("balanced closing brackets/braces")

        return working, notes

    def fix_json(self, text):
        original = self._basic_cleanup(text)
        notes = []

        try:
            parsed = self._parse_json_preserving_order(original)
        except json.JSONDecodeError as exc:
            repaired_text, notes = self._repair_schema_text(text)
            try:
                parsed = self._parse_json_preserving_order(repaired_text)
            except json.JSONDecodeError as repaired_exc:
                original_error = self._format_json_error(exc, original, "Original JSON parse failed")
                repaired_error = self._format_json_error(repaired_exc, repaired_text, "Repair attempt still invalid")
                raise ValueError(f"VRGDG_StoryGroupJsonFixer: {original_error}\n{repaired_error}")
        else:
            repaired_text = original

        try:
            normalized = self._normalize_story_payload(parsed)
        except ValueError as exc:
            raise ValueError(f"VRGDG_StoryGroupJsonFixer schema error: {exc}")
        fixed_text = json.dumps(normalized, indent=2, ensure_ascii=False)
        was_fixed = bool(notes) or fixed_text.strip() != original.strip()
        note_text = "; ".join(notes) if notes else ("normalized formatting" if was_fixed else "")
        return (fixed_text, normalized, was_fixed, note_text)

NODE_CLASS_MAPPINGS = {
    "VRGDG_ShowText": VRGDG_ShowText,
    "VRGDG_ShowAny": VRGDG_ShowAny,
    "VRGDG_TextBox": VRGDG_TextBox,
    "VRGDG_String2Json": VRGDG_String2Json,
    "VRGDG_Json2String": VRGDG_Json2String,
    "VRGDG_ShowImage": VRGDG_ShowImage,
    "VRGDG_BoxIT": VRGDG_BoxIT,
    "VRGDG_IntToFloat": VRGDG_IntToFloat,
    "VRGDG_ImageIndex0HUMOEDIT": VRGDG_ImageIndex0HUMOEDIT,
    "VRGDG_OptionalMultiLoraModelOnly": VRGDG_OptionalMultiLoraModelOnly,
    "VRGDG_NoteBox": VRGDG_NoteBox,
    "VRGDG_SetMuteStateMulti": VRGDG_SetMuteStateMulti,
    "VRGDG_SetGroupStateMulti": VRGDG_SetGroupStateMulti,
    "VRGDG_MuteUnmute4PromptCreatorWF_1": VRGDG_MuteUnmute4PromptCreatorWF_1,
    "VRGDG_MuteUnmute4PromptCreatorWF_2": VRGDG_MuteUnmute4PromptCreatorWF_2,
    "VRGDG_MuteUnmute4PromptCreatorWF_0": VRGDG_MuteUnmute4PromptCreatorWF_0,
    "VRGDG_PromptCreatorUI": VRGDG_PromptCreatorUI,
    "VRGDG_MultiStringConcat": VRGDG_MultiStringConcat,
    "VRGDG_StoryGroupJsonFixer": VRGDG_StoryGroupJsonFixer,
    "VRGDG_LyricSegmentJsonFixer": VRGDG_LyricSegmentJsonFixer,
    "VRGDG_LyricSegmentTextCleaner": VRGDG_LyricSegmentTextCleaner,
    "VRGDG_PromptMapJsonFixer": VRGDG_PromptMapJsonFixer,    
    "VRGDG_PromptJsonSubjectPrepender": VRGDG_PromptJsonSubjectPrepender,
    "VRGDG_LyricSegmentDurationMerger": VRGDG_LyricSegmentDurationMerger,
    "VRGDG_PromptCreatorUI_V2": VRGDG_PromptCreatorUI_V2,
    "VRGDG_Part2WorkflowUI": VRGDG_Part2WorkflowUI,
    "VRGDG_Part3WorkflowUI": VRGDG_Part3WorkflowUI,
    "VRGDG_T2IPromptsFromConcepts": VRGDG_T2IPromptsFromConcepts,
    "VRGDG_T2VPromptsFromConcepts": VRGDG_T2VPromptsFromConcepts,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ShowText": "VRGDG_ShowText",
    "VRGDG_ShowAny": "VRGDG_ShowAny",
    "VRGDG_TextBox": "VRGDG_TextBox",
    "VRGDG_String2Json": "VRGDG_String2Json",
    "VRGDG_Json2String": "VRGDG_Json2String",
    "VRGDG_ShowImage": "VRGDG_ShowImage",
    "VRGDG_BoxIT": "VRGDG_BoxIT",
    "VRGDG_IntToFloat": "VRGDG_IntToFloat",
    "VRGDG_ImageIndex0HUMOEDIT": "VRGDG_ImageIndex0HUMOEDIT",
    "VRGDG_OptionalMultiLoraModelOnly": "VRGDG Optional Multi LoRA Model Only",
    "VRGDG_NoteBox": "VRGDG_NoteBox",
    "VRGDG_SetMuteStateMulti": "VRGDG_SetMuteStateMulti",
    "VRGDG_SetGroupStateMulti": "VRGDG_SetGroupStateMulti",
    "VRGDG_MuteUnmute4PromptCreatorWF_1": "VRGDG_MuteUnmute4PromptCreatorWF_1",
    "VRGDG_MuteUnmute4PromptCreatorWF_2": "VRGDG_MuteUnmute4PromptCreatorWF_2",
    "VRGDG_MuteUnmute4PromptCreatorWF_0": "VRGDG_MuteUnmute4PromptCreatorWF_0",
    "VRGDG_PromptCreatorUI": "VRGDG_PromptCreatorUI",
    "VRGDG_MultiStringConcat": "VRGDG_MultiStringConcat",
    "VRGDG_StoryGroupJsonFixer": "VRGDG_StoryGroupJsonFixer",
    "VRGDG_LyricSegmentJsonFixer": "VRGDG_LyricSegmentJsonFixer",
    "VRGDG_LyricSegmentTextCleaner": "VRGDG Lyric Segment Text Cleaner",
    "VRGDG_PromptMapJsonFixer": "VRGDG_PromptMapJsonFixer",
    "VRGDG_PromptJsonSubjectPrepender": "VRGDG Prompt JSON Subject Prepender",
    "VRGDG_LyricSegmentDurationMerger": "VRGDG_LyricSegmentDurationMerger",
    "VRGDG_PromptCreatorUI_V2": "VRGDG_PromptCreatorUI_V2",
    "VRGDG_Part2WorkflowUI": "VRGDG Part 2 Workflow UI",
    "VRGDG_Part3WorkflowUI": "VRGDG Workflow 3 UI",
    "VRGDG_T2IPromptsFromConcepts": "Text to image prompts from concepts",
    "VRGDG_T2VPromptsFromConcepts": "Text to video prompts from concepts",
    
}
