import json

import os

import re

import time

import asyncio


import folder_paths

from aiohttp import web

from server import PromptServer


from .llm_helpers import _llm_runner_from_payload, _run_builder_text_llm


_VRGDG_TEST_SAVE_ROUTE_REGISTERED = False

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

OPTIONAL STYLE/THEME PRIORITY

The optional style/theme for location extraction has priority over the generic location rules below. If it specifies crowds, passengers, people, characters, activity, or other visible details, preserve those details.

Only reject details that conflict with the optional style/theme. Do not explain or critique the choice; output the final location description only.

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


_VRGDG_GEMMA4_LOCATION_DETAIL_INSTRUCTIONS = """Expand a short location description into one detailed, standalone environment description.

Use only the supplied location label and short description as your source. Preserve the same place and do not invent a different setting. Add concrete visual detail about architecture or terrain, spatial layout, materials, textures, color palette, lighting, atmosphere, weather, and background elements that would help a visual artist reproduce the location.

Return only the expanded location description as one cohesive paragraph. Do not mention reference images, pictures, prompts, image generation, LLMs, subjects, characters, lyrics, or camera shots. Do not add a title, bullets, or labels."""


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


def _get_test_popup_audio_dir():
    return os.path.normpath(os.path.join(folder_paths.get_output_directory(), "VRGDG_AudioFiles"))


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


def _prompt_creator_custom_instruction(payload, key, default_text):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        return str(default_text or "")
    safe_key = re.sub(r"[^a-z0-9_]+", "_", str(key or "").strip().lower()).strip("_")
    if not safe_key:
        return str(default_text or "")
    path = os.path.join(project_folder, "project_context", "custom_llm_instructions", f"{safe_key}.txt")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                text = handle.read().strip()
            if text:
                return text
        except Exception:
            pass
    return str(default_text or "")


def _build_gemma4_prompt(target, payload):
    target = str(target or "").strip()
    notes = str(payload.get("notes", "") or "").strip()
    lyrics = str(payload.get("lyrics", "") or "").strip()
    style_theme = str(payload.get("style_theme", "") or "").strip()
    story_idea = str(payload.get("story_idea", "") or "").strip()

    if target == "builder_style_theme":
        idea = notes or lyrics or style_theme or story_idea
        instructions = _prompt_creator_custom_instruction(payload, "style_theme", _VRGDG_GEMMA4_STYLE_INSTRUCTIONS)
        prompt = (
            f"{instructions}\n\n"
            "Create the style/theme block from normal user ideas instead of lyrics.\n"
            "Use the user's rough idea as the full creative direction.\n"
            "If the idea is short, infer a useful cinematic visual style without adding extra sections.\n\n"
            f"User idea:\n{idea}"
        )
        return prompt

    if target == "builder_story_idea":
        idea = notes or story_idea or lyrics
        instructions = _prompt_creator_custom_instruction(payload, "story_idea", _VRGDG_GEMMA4_STORY_INSTRUCTIONS)
        prompt = (
            f"{instructions}\n\n"
            "Create the story idea from normal user ideas instead of lyrics.\n"
            "Treat the user idea as the full creative foundation.\n"
            "Do not ask for lyrics. Output only the story concept.\n\n"
            f"User idea:\n{idea}"
        )
        if style_theme:
            prompt += f"\n\nStyle/theme:\n{style_theme}"
        return prompt

    if target == "builder_subjects_and_scenes":
        idea = notes or story_idea or lyrics
        instructions = _prompt_creator_custom_instruction(payload, "subject_locations", _VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS)
        prompt = (
            f"{instructions}\n\n"
            "Create the subject and location list from normal user ideas instead of lyrics.\n"
            "Use the user idea as the highest priority creative direction.\n"
        )
        if style_theme:
            prompt += f"\n\nStyle/theme:\n{style_theme}"
        prompt += f"\n\nStory or user idea:\n{idea}"
        return prompt

    if target == "style_theme":
        instructions = _prompt_creator_custom_instruction(payload, "style_theme", _VRGDG_GEMMA4_STYLE_INSTRUCTIONS)
        prompt = f"{instructions}\n\nfull lyrics:\n{lyrics}"
        if notes:
            prompt += f"\n\nother notes:\n{notes}"
        return prompt

    if target == "story_idea":
        instructions = _prompt_creator_custom_instruction(payload, "story_idea", _VRGDG_GEMMA4_STORY_INSTRUCTIONS)
        prompt = f"{instructions}\n\nLyrics:\n{lyrics}"
        if style_theme:
            prompt += f"\n\nStyle/theme:\n{style_theme}"
        if notes:
            prompt += f"\n\nOptional notes:\n{notes}"
        return prompt

    if target == "subjects_and_scenes":
        instructions = _prompt_creator_custom_instruction(payload, "subject_locations", _VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS)
        prompt = f"{instructions}"
        if notes:
            prompt += f"\n\nUser notes - highest priority:\n{notes}"
        prompt += f"\n\nStory idea:\n{story_idea}"
        return prompt

    if target == "song_lyrics":
        duration = str(payload.get("duration", "") or "").strip()
        instructions = _prompt_creator_custom_instruction(payload, "full_lyrics", _VRGDG_GEMMA4_LYRICS_INSTRUCTIONS)
        prompt = f"{instructions}"
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

    if target == "location_description_detail":
        location_name = str(payload.get("location_name", "") or "").strip()
        location_description = str(payload.get("location_description", "") or notes).strip()
        if not location_name or not location_description:
            raise ValueError("A location label and short description are required.")
        instructions = _prompt_creator_custom_instruction(
            payload,
            "location_description_detail",
            _VRGDG_GEMMA4_LOCATION_DETAIL_INSTRUCTIONS,
        )
        return (
            f"{instructions}\n\n"
            f"Location label:\n{location_name}\n\n"
            f"Short location description:\n{location_description}"
        )

    raise ValueError(f"Unsupported Gemma4 target: {target}")


def _run_gemma4_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat

    target = str(payload.get("target", "") or "").strip()
    model_file = str(payload.get("model_file", "") or "").strip()
    runner = _llm_runner_from_payload(payload)
    if not model_file and runner == "builtin":
        raise ValueError("No Gemma4 model_file was selected.")

    prompt = _build_gemma4_prompt(target, payload)
    if not prompt.strip():
        raise ValueError("Gemma4 prompt was empty.")

    # Match the known-good SuperGemma settings used in the workflow node.
    n_ctx = int(payload.get("n_ctx") or 13000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.75)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 32000)
    unload_after = bool(payload.get("unload_after"))

    if runner != "builtin":
        text, run_info = _run_builder_text_llm(
            payload,
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            label="Gemma4",
            preserve_paragraphs=target in {
                "builder_style_theme",
                "style_theme",
                "builder_story_idea",
                "story_idea",
                "builder_subjects_and_scenes",
                "subjects_and_scenes",
                "song_lyrics",
            },
        )
        text = _clean_gemma4_text(text)
        if not text:
            raise ValueError("Gemma4 returned an empty response.")
        return {
            "text": text,
            "used_model": run_info.get("used_model", ""),
            "runner": run_info.get("runner", runner),
            "unloaded": False,
        }

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
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

    @server_instance.routes.post("/vrgdg/prompt_creator/gemma4/generate")
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

    @server_instance.routes.post("/vrgdg/prompt_creator/gemma4/unload")
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
    _VRGDG_TEST_SAVE_ROUTE_REGISTERED = True


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


_ensure_test_save_route_registered()


NODE_CLASS_MAPPINGS = {
    "VRGDG_PromptCreatorUI": VRGDG_PromptCreatorUI,
    "VRGDG_PromptCreatorUI_V2": VRGDG_PromptCreatorUI_V2,
    "VRGDG_Part2WorkflowUI": VRGDG_Part2WorkflowUI,
    "VRGDG_Part3WorkflowUI": VRGDG_Part3WorkflowUI,
    "VRGDG_T2VPromptsFromConcepts": VRGDG_T2VPromptsFromConcepts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_PromptCreatorUI": "VRGDG_PromptCreatorUI",
    "VRGDG_PromptCreatorUI_V2": "VRGDG_PromptCreatorUI_V2",
    "VRGDG_Part2WorkflowUI": "VRGDG Part 2 Workflow UI",
    "VRGDG_Part3WorkflowUI": "VRGDG Workflow 3 UI",
    "VRGDG_T2VPromptsFromConcepts": "Text to video prompts from concepts",
}
