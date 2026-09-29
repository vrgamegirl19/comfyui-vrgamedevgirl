"""Video Builder image prompt generation: concepts, T2I, NanoBanana, Flux Klein, reference descriptions, locations and subjects."""

import json
import os
import re
from PIL import Image
from ..builder.video_editor import _image_from_data_url
from .text_cleaning import _clean_visual_gemma_text, extract_prompt_text_from_gemma_output
from .prompts.image import _FLOW_GPT_T2I_INSTRUCTIONS, _FLUX_KLEIN_T2I_INSTRUCTIONS, _NANO_B_T2I_INSTRUCTIONS, _TEXT_ONLY_T2I_INSTRUCTIONS, _VISUAL_T2I_INSTRUCTIONS

from .prompts.image import _image_prompt_edit_instructions
from ..builder.paths import _read_text_file, _resolve_existing_file
from .output_checks import _balance_location_map_by_usage, _clean_location_card, _clean_location_context_text, _extract_json_object_from_text, _fallback_location_map_by_overlap, _location_usage_counts_from_payload, _looks_like_gemma_repeat_failure, _normalize_location_name, _parse_location_ideas_flexible, _parse_location_lines, _parse_scene_location_number_map, _parse_subject_lines, _valid_location_card, _validate_builder_gemma_prompt, _validate_reference_description
from ..builder.media import _combine_flux_ingredient_images, _combine_story_reference_batch, _image_from_prompt_payload
from ..builder.project import _load_scene_notes_json
from .builder_instructions import _effective_builder_instruction
from .builder_runner import _EXTERNAL_LLM_RUNNERS, _builder_local_llm, _builder_local_mmproj_file, _builder_local_model_file, _clean_lm_studio_plain_text, _clear_comfy_model_memory, _llm_runner_from_payload, _repair_and_validate_builder_gemma_prompt, _resolve_mmproj_dropdown_path, _run_builder_text_llm, _runner_output_token_limit, _try_run_remote_vision


def _generate_builder_concept_prompts(payload):
    source_mode = str(payload.get("source_mode") or "all").strip().lower()
    story_idea = str(payload.get("story_idea") or "").strip()
    theme_style = str(payload.get("theme_style") or "").strip()
    previous_summary = str(payload.get("previous_summary") or "").strip()
    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for concept prompt creation.")

    cleaned_scenes = []
    for scene in scenes[:20]:
        if not isinstance(scene, dict):
            continue
        try:
            number = int(scene.get("scene_number") or scene.get("number") or len(cleaned_scenes) + 1)
        except Exception:
            number = len(cleaned_scenes) + 1
        timeline_notes = scene.get("timeline_notes") or []
        if not isinstance(timeline_notes, list):
            timeline_notes = []
        cleaned_scenes.append({
            "scene_number": max(1, number),
            "label": str(scene.get("label") or f"Scene {number}").strip()[:160],
            "lyric_text": str(scene.get("lyric_text") or "").strip()[:1200],
            "lyric_singers": [
                str(item or "").strip()[:160]
                for item in (scene.get("lyric_singers") if isinstance(scene.get("lyric_singers"), list) else [])
                if str(item or "").strip()
            ][:8],
            "lyric_instrumental": bool(scene.get("lyric_instrumental")),
            "lyric_no_lip_sync": bool(scene.get("lyric_no_lip_sync")),
            "no_character_present": bool(scene.get("no_character_present") or scene.get("no_subject") or scene.get("no_visible_subject")),
            "mapped_subjects": str(scene.get("mapped_subjects") or "").strip()[:1600],
            "director_note": str(scene.get("director_note") or "").strip()[:1200],
            "scene_note": str(scene.get("scene_note") or "").strip()[:1200],
            "timeline_notes": [
                {
                    "label": str(item.get("label") or item.get("type") or "note").strip()[:160],
                    "note": str(item.get("note") or "").strip()[:800],
                    "start": item.get("start"),
                    "end": item.get("end"),
                }
                for item in timeline_notes[:8]
                if isinstance(item, dict)
            ],
            "subject_reference_mode": str(scene.get("subject_reference_mode") or "none").strip()[:80],
            "location_reference_mode": str(scene.get("location_reference_mode") or "none").strip()[:80],
        })
    if not cleaned_scenes:
        raise ValueError("No valid scenes were provided for concept prompt creation.")

    source_note = {
        "director": "Use director notes as the main scene notes.",
        "scene": "Use raw scene notes as the main scene notes.",
        "timeline": "Use overlapping timeline notes as the main scene notes.",
        "lyrics": "Use lyric notes as the main scene notes.",
        "subjects": "Use subject and singer mapping as the main scene notes.",
        "lyrics_subjects": "Use lyric notes plus subject and singer mapping as the main scene notes.",
        "director_scene": "Use director notes and raw scene notes as the main scene notes.",
        "all": "Use all available notes: director notes, raw scene notes, lyric notes, subject/singer mapping, and overlapping timeline notes.",
    }.get(source_mode, "Use all available notes: director notes, raw scene notes, lyric notes, subject/singer mapping, and overlapping timeline notes.")

    instruction = (
        "You will create concept prompts from scene notes and/or director notes, timeline notes, story idea, and style/theme.\n"
        "Return only valid JSON in this exact flat format:\n"
        "{\n"
        "  \"Scene1\": \"\",\n"
        "  \"Scene2\": \"\"\n"
        "}\n\n"
        f"Source mode: {source_mode}. {source_note}\n\n"
        "The story idea explains the overall music video concept, emotional arc, setting, and visual direction. Use it to make the prompts feel connected and to make each scene flow naturally like a music video.\n"
        "The style/theme explains the visual language, mood, palette, texture, and production design. Use it for consistency.\n"
        "Character descriptions or character reference images may be provided separately. Do not describe characters' hair, face, clothing, makeup, or exact appearance in the concept prompts.\n\n"
        "Create one prompt for each provided scene.\n"
        "Each concept prompt must include who is in the scene, shot type, Location:, and scene details that fit the story world.\n\n"
        "Subject rules:\n"
        "- If lyric_singers contains names, those are the subjects/performers for that scene.\n"
        "- If mapped_subjects contains character reference names, use those as the visible subjects for that scene.\n"
        "- If no_character_present is true, write only the location/background, environment, architecture, props, objects, weather, lighting, atmosphere, environmental motion, and camera. Do not include, mention, imply, describe, silhouette, reflect, or refer to any person, character, subject, singer, performer, band member, crowd, body part, face, voice, or character reference.\n"
        "- If lyric_no_lip_sync is true or lyric_instrumental is true, the scene should not be treated as a singing/lip-sync performance scene.\n"
        "- If the scene note says female, start with \"Female only\".\n"
        "- If the scene note says male, start with \"Male only\".\n"
        "- If the scene note says female and male, start with \"Female and male together in frame\".\n"
        "- If the scene note is instrumental only or has no clear subject, start with \"No main subject\".\n\n"
        "Music video flow rules:\n"
        "- Make the prompts feel like connected shots from the same music video.\n"
        "- Let locations and details progress with the story instead of feeling random.\n"
        "- Instrumental scenes can be establishing shots, transitions, symbolic visuals, or world-building.\n"
        "- Character scenes should feel like performance, duet, reaction, memory, confrontation, longing, or story moments.\n"
        "- Keep the visual world consistent across all prompts.\n\n"
        "Shot rules:\n"
        "- Instrumental scenes should usually be wide establishing shots or wide environment shots.\n"
        "- Female-only or male-only scenes should vary between wide, medium-wide, medium, waist-up, and upper body shots. Use a close-up only when the scene note calls for it.\n"
        "- Female and male together should usually be full body, two-shot, medium-wide, or wide shot.\n"
        "- Make the shot type fit the scene note and story moment.\n\n"
        "Location rules:\n"
        "- Include \"Location:\" in every prompt.\n"
        "- Pick locations that fit the story idea, mood, and style/theme.\n"
        "- If a location reference is available for that scene, use it as the location identity instead of inventing an unrelated location.\n"
        "- If no location reference is available, create the location from the story idea, style/theme, and notes.\n"
        "- Keep locations visually connected so the scenes feel like one continuous music video world.\n\n"
        "Style rules:\n"
        "- Do not write full text-to-image prompts.\n"
        "- Do not include character descriptions.\n"
        "- Do not include camera settings, aspect ratio, render style, or quality tags.\n"
        "- Keep each prompt as one clear sentence or short paragraph.\n"
        "- Make prompts detailed enough for another LLM to expand into image prompts.\n"
        "- Return only the JSON object.\n\n"
        f"Story idea:\n{story_idea or '[none provided]'}\n\n"
        f"Style/theme:\n{theme_style or '[none provided]'}\n\n"
        f"Previous batch visual progression summary:\n{previous_summary or '[none yet]'}\n\n"
        "Scenes:\n"
        f"{json.dumps(cleaned_scenes, ensure_ascii=False, indent=2)}"
    )
    text, info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.95),
        max_new_tokens=int(payload.get("max_new_tokens") or 1200),
        label="Concept Prompt Creator",
        preserve_paragraphs=True,
    )
    raw = _clean_lm_studio_plain_text(text)
    prompts = {}
    try:
        parsed = _extract_json_object_from_text(raw)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                match = re.search(r"scene\s*(\d+)", str(key), re.I)
                if not match:
                    continue
                prompt_text = str(value or "").strip()
                if prompt_text:
                    prompts[f"Scene{int(match.group(1))}"] = prompt_text
    except Exception:
        prompts = {}
    if not prompts:
        for match in re.finditer(r'"?\bScene\s*(\d+)"?\s*:\s*"([^"]+)"', raw, re.I | re.S):
            prompt_text = match.group(2).strip()
            if prompt_text:
                prompts[f"Scene{int(match.group(1))}"] = prompt_text
    if not prompts:
        raise ValueError("Gemma did not return any SceneN concept prompts.")
    summary_lines = []
    for key in sorted(prompts.keys(), key=lambda item: int(re.search(r"\d+", item).group(0))):
        text_value = prompts[key]
        summary_lines.append(f"{key}: {text_value[:180]}")
    return {
        "prompts": prompts,
        "raw": raw,
        "summary": "\n".join(summary_lines)[:1200],
        "run_info": info,
    }


def _generate_builder_t2i_prompt(payload):
    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    ref_image_path = str(payload.get("ref_image_path", "") or "").strip().strip('"')
    ref_image_data = str(payload.get("ref_image_data", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    prompt_mode = str(payload.get("prompt_mode", "") or "").strip().lower()
    builder_instruction_key = str(payload.get("builder_instruction_key") or payload.get("instruction_key") or "").strip()
    reference_context = payload.get("reference_context") or {}
    if not isinstance(reference_context, dict):
        reference_context = {}
    theme_style = _read_text_file(payload.get("theme_style_path", ""), "Theme/style file")
    story_idea = _read_text_file(payload.get("story_idea_path", ""), "Story idea file")
    subject_scene = _read_text_file(payload.get("subject_scene_path", ""), "Subject/scene file")
    context_parts = []
    if subject_scene:
        context_parts.append(f"Subject/scene:\n{subject_scene}")
    if theme_style:
        context_parts.append(f"Theme/style:\n{theme_style}")
    if story_idea:
        context_parts.append(f"Story idea:\n{story_idea}")
    if context_parts:
        user_notes = "\n\n".join(context_parts + ([f"Segment notes:\n{user_notes}"] if user_notes else []))
    use_vision = bool(payload.get("use_vision"))
    has_ref_image = bool(use_vision and ((ref_image_path and os.path.isfile(ref_image_path)) or ref_image_data))
    text_runner = _llm_runner_from_payload(payload)
    if not model_file and text_runner not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose a Gemma model first.")
    if use_vision and not has_ref_image:
        raise ValueError("Choose a valid reference image path/data or turn off vision reference.")
    if not has_ref_image and not user_notes:
        raise ValueError("Enter scene notes or provide a reference image.")

    llm = _builder_local_llm(payload) if has_ref_image and text_runner not in _EXTERNAL_LLM_RUNNERS else None
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else ""
    mmproj_file = _builder_local_mmproj_file(payload, mmproj_file)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    image = None
    if has_ref_image:
        image = _image_from_data_url(ref_image_data).convert("RGB") if ref_image_data else Image.open(ref_image_path).convert("RGB")
    if has_ref_image:
        prompt = _VISUAL_T2I_INSTRUCTIONS
        prompt += f"\n\nUser notes:\n{user_notes or 'Use the reference image as the guide.'}"
    elif prompt_mode == "flux_klein":
        subject_description = str(reference_context.get("subject_description", "") or "").strip()
        location_name = str(reference_context.get("location_name", "") or "").strip()
        location_description = str(reference_context.get("location_description", "") or "").strip()
        reference_rules = []
        if subject_description:
            reference_rules.append(
                "Use the subject description for the main subject identity, face/body details, outfit, and visible character consistency. "
                f"Subject description: {subject_description}"
            )
        if location_name or location_description:
            location_details = "; ".join(part for part in (location_name, location_description) if part)
            reference_rules.append(
                "Use the mapped location as the required setting/background. Do not replace it with a different location from the concept or notes. "
                f"Mapped location: {location_details}"
            )
        reference_text = ""
        if reference_rules:
            reference_text = (
                "\nReference Builder priorities:\n"
                + "\n".join(f"- {rule}" for rule in reference_rules)
                + "\n- Use the user's notes/concept for action, pose, mood, lighting, story beat, and details after respecting the reference priorities.\n"
            )
        base_instructions = _FLUX_KLEIN_T2I_INSTRUCTIONS
        if builder_instruction_key:
            base_instructions = _effective_builder_instruction(payload, builder_instruction_key, _FLUX_KLEIN_T2I_INSTRUCTIONS)
        prompt = (
            f"{base_instructions}\n"
            f"{reference_text}\n"
            "\n"
            f"User notes:\n{user_notes or 'Create a cinematic image using the available scene notes.'}"
        )
    elif prompt_mode == "nano_banana":
        subject_description = str(reference_context.get("subject_description", "") or "").strip()
        location_name = str(reference_context.get("location_name", "") or "").strip()
        location_description = str(reference_context.get("location_description", "") or "").strip()
        has_subject_reference = bool(reference_context.get("has_subject_reference") or subject_description)
        has_location_reference = bool(reference_context.get("has_location_reference") or location_name or location_description)
        context_parts = []
        if subject_description:
            context_parts.append(f"Character reference description:\n{subject_description}")
        if location_name or location_description:
            context_parts.append(f"Location reference description:\n{location_name}\n{location_description}".strip())
        if user_notes:
            context_parts.append(f"User input:\n{user_notes}")
        start_rules = [
            "- Start with: Using the provided character reference and location reference, create...",
            "- If only a character reference description is available, start with: Using the provided character reference, create...",
            "- If only a location reference description is available, start with: Using the provided location reference, create...",
        ]
        if not has_subject_reference and not has_location_reference:
            start_rules = ["- Start directly with the shot and subject; do not claim a provided reference exists."]
        base_instructions = _NANO_B_T2I_INSTRUCTIONS
        if builder_instruction_key:
            base_instructions = _effective_builder_instruction(payload, builder_instruction_key, _NANO_B_T2I_INSTRUCTIONS)
        prompt = (
            f"{base_instructions}\n\n"
            "Reference wording rules:\n"
            + "\n".join(start_rules)
            + "\n"
            "- Preserve the character identity from the character reference description: face, hair, outfit, makeup, and overall identity.\n"
            "- Preserve the location identity from the location reference description: environment, architecture, layout, atmosphere, and major visible setting details.\n"
            "\n"
            f"{chr(10).join(context_parts) if context_parts else 'User input: Create a cinematic image using the available scene notes.'}"
        )
    else:
        base_instructions = _TEXT_ONLY_T2I_INSTRUCTIONS
        if builder_instruction_key:
            base_instructions = _effective_builder_instruction(payload, builder_instruction_key, _TEXT_ONLY_T2I_INSTRUCTIONS)
        prompt = f"{base_instructions}\n\nUser notes:\n{user_notes}"

    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or (0.25 if has_ref_image else 0.6))
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or (1000 if has_ref_image else 1200)))
    unload_after = bool(payload.get("unload_after", True))

    try:
        if has_ref_image and text_runner in _EXTERNAL_LLM_RUNNERS:
            text, run_info = _try_run_remote_vision(
                payload,
                prompt,
                [image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        elif has_ref_image:
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[image],
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            text, run_info = _run_builder_text_llm(
                payload,
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                label="Gemma",
            )
        text = _clean_visual_gemma_text(text)
        text = extract_prompt_text_from_gemma_output(text, payload.get("scene_number"))
        label = "Flux/Klein" if prompt_mode == "flux_klein" else "NanoBanana" if prompt_mode == "nano_banana" else "Flow/GPT" if prompt_mode == "flow_gpt" else "T2I"
        text = _repair_and_validate_builder_gemma_prompt(payload, text, label)
        return {
            "prompt": text,
            "used_reference_image": has_ref_image,
            "used_model": run_info.get("used_model", model_path) if has_ref_image and text_runner in _EXTERNAL_LLM_RUNNERS else model_path if has_ref_image else run_info.get("used_model", ""),
            "used_mmproj": mmproj_path,
            "runner": run_info.get("runner", "builtin") if has_ref_image and text_runner in _EXTERNAL_LLM_RUNNERS else "builtin" if has_ref_image else run_info.get("runner", "builtin"),
            "unloaded": run_info.get("unloaded", unload_after) if has_ref_image and text_runner in _EXTERNAL_LLM_RUNNERS else unload_after if has_ref_image else run_info.get("unloaded", unload_after),
        }
    finally:
        if llm and has_ref_image and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _ensure_reference_opening_for_image_edit(prompt, payload):
    text = str(prompt or "").strip()
    prompt_mode = str(payload.get("prompt_mode") or "").strip().lower()
    if prompt_mode not in {"nano_banana", "flow_gpt"} or not text:
        return text
    reference_context = payload.get("reference_context") if isinstance(payload.get("reference_context"), dict) else {}
    has_subject_reference = bool(reference_context.get("has_subject_reference"))
    has_location_reference = bool(reference_context.get("has_location_reference"))
    try:
        subject_reference_count = int(float(reference_context.get("subject_reference_count") or 1)) if has_subject_reference else 0
    except (TypeError, ValueError):
        subject_reference_count = 1 if has_subject_reference else 0
    subject_reference_count = max(0, min(99, subject_reference_count))
    if not has_subject_reference and not has_location_reference:
        return text
    character_reference_phrase = "character reference images" if subject_reference_count > 1 else "character reference image"
    if has_subject_reference and has_location_reference:
        opening = f"Using the provided {character_reference_phrase} and location reference image"
    elif has_subject_reference:
        opening = f"Using the provided {character_reference_phrase}"
    else:
        opening = "Using the provided location reference image"
    text = re.sub(
        r"^Using the provided\s+(?:(?:character|location|scene|reference)\s+)+(?:images?|references?)\s*,?\s*(?:create\s+)?",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    if re.match(r"^(?:create|make|generate)\b", text, flags=re.IGNORECASE):
        text = re.sub(r"^(?:create|make|generate)\b\s*", "", text, count=1, flags=re.IGNORECASE).strip()
    if not text:
        return f"{opening}, create a cinematic still image."
    return f"{opening}, create {text[:1].lower()}{text[1:] if len(text) > 1 else ''}".strip()


def _edit_builder_image_prompt(payload):
    from .cache import _clear_vrgdg_llm_caches

    current_prompt = str(payload.get("current_prompt") or "").strip()
    if not current_prompt:
        raise ValueError("Current image prompt is empty.")
    model_file = str(payload.get("model_file") or payload.get("repair_model_file") or "").strip()
    mmproj_file = str(payload.get("mmproj_file") or "").strip()
    ref_image_path = str(payload.get("ref_image_path") or payload.get("image_reference_path") or "").strip().strip('"')
    ref_image_data = str(payload.get("ref_image_data") or payload.get("image_reference_data") or "").strip()
    use_vision_reference = bool(payload.get("use_vision_reference"))
    text_runner = _llm_runner_from_payload(payload)
    if model_file:
        payload = dict(payload)
        payload["model_file"] = model_file
    instruction = _image_prompt_edit_instructions(payload)
    image = None
    if use_vision_reference:
        if ref_image_data:
            image = _image_from_data_url(ref_image_data).convert("RGB")
        elif ref_image_path:
            image_path = _resolve_existing_file(ref_image_path, "Image prompt edit reference image")
            image = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("Prompt edit requested a reference image, but no image reference was provided.")

    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.9)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 1200))
    unload_after = bool(payload.get("unload_after", True))
    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format") or "").strip()
    seed = payload.get("seed")
    llm = None
    model_path = ""
    mmproj_path = ""
    try:
        if use_vision_reference and text_runner in _EXTERNAL_LLM_RUNNERS:
            text, run_info = _try_run_remote_vision(
                payload,
                instruction,
                [image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        elif use_vision_reference:
            if not model_file:
                raise ValueError("Choose a vision Gemma model first.")
            if not model_file.lower().endswith(".gguf"):
                raise ValueError("The vision Gemma model field is not a GGUF model.")
            llm = _builder_local_llm(payload)
            model_file = _builder_local_model_file(payload, model_file)
            model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
            mmproj_file = _builder_local_mmproj_file(payload, mmproj_file)
            mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file)
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[image],
                instruction_text=instruction,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=int(seed) if seed is not None else None,
            )
            run_info = {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}
        else:
            text, run_info = _run_builder_text_llm(
                payload,
                instruction,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                label="image prompt edit",
            )
        text = _clean_visual_gemma_text(text)
        text = extract_prompt_text_from_gemma_output(text, payload.get("scene_number"))
        text = _repair_and_validate_builder_gemma_prompt(payload, text, str(payload.get("mode_label") or "Image"))
        text = _ensure_reference_opening_for_image_edit(text, payload)
        return {
            "prompt": text,
            "runner": run_info.get("runner", "builtin"),
            "used_model": run_info.get("used_model", model_path),
            "used_mmproj": mmproj_path,
            "used_image_reference": use_vision_reference,
            "unloaded": run_info.get("unloaded", unload_after),
        }
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _generate_builder_reference_description(payload):
    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    text_runner = _llm_runner_from_payload(payload)
    reference_type = str(payload.get("reference_type") or "subject").strip().lower()
    name_hint = str(payload.get("name") or "").strip()
    subject_label = re.sub(r"\s+", " ", name_hint).strip()
    if subject_label.lower().startswith("the "):
        subject_label = subject_label.lower()
    object_reference_types = {"prop", "object", "vehicle", "creature", "animal", "outfit", "style", "environment", "other"}
    if reference_type not in {"subject", "character", "face", "location", "extra", *object_reference_types}:
        reference_type = "subject"
    if text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file:
        raise ValueError("Choose a Gemma vision model first.")
    image = _image_from_prompt_payload(payload.get("image_path", ""), payload.get("image_data", ""), "Reference image")

    if reference_type == "face":
        instruction = (
            "Study the visible person's face and write one precise face-identity description for an image enhancement prompt.\n"
            "Return only one plain-text paragraph. It must begin exactly with: a photo of\n"
            "Describe stable facial identity details only: apparent age range, face shape, eyes, eyebrows, nose, lips, jawline, cheeks, distinctive facial features, hair color, hairline, and face-framing hair.\n"
            "Include makeup or facial accessories only when clearly visible and useful for preserving identity.\n"
            "Do not mention the background, location, clothing, body, pose, camera angle, action, mood, or facial expression.\n"
            "Do not describe smiling, frowning, an open mouth, gaze direction, or head direction.\n"
            "Do not add markdown, a label, quotation marks, instructions, or commentary.\n"
            "Do not invent details that are not visible. Keep it under 100 words."
        )
    elif reference_type in {"subject", "character"}:
        instruction = (
            "Look at the image and write one concise character appearance description.\n"
            "Output only the description, one paragraph, no markdown, no label, no bullet points.\n"
            "Describe only the character's full visible appearance: hair, makeup, accessories, jewelry, clothing, outfit materials, colors, shoes, and distinctive visual identity details.\n"
            "Do not mention skin color, skin tone, ethnicity, race, or complexion.\n"
            "Do not describe the background, location, pose, camera angle, facial expression, mood, action, or what the character is doing.\n"
            "Do not invent hidden or unseen details.\n"
            "Keep it under 100 words."
        )
        if subject_label:
            instruction += (
                f"\nRefer to the subject as {subject_label}. "
                f"Do not call them the character or the subject in the final description."
            )
    elif reference_type == "extra":
        try:
            count = max(1, min(100, int(round(float(payload.get("count") or 1)))))
        except (TypeError, ValueError, OverflowError):
            count = 1
        style_hint = re.sub(r"\s+", " ", str(payload.get("style") or "")).strip()
        group_hint = f"one of a group of {count} background performers"
        if style_hint:
            group_hint += f" in a {style_hint}-appropriate wardrobe"
        instruction = (
            "Look at the image and write one very short background-performer identity description.\n"
            "Output only the description, one paragraph, no markdown, no label, no bullet points.\n"
            "Include apparent gender presentation, hair color and hairstyle, a concise outfit description with its main colors, and at most one clearly visible distinguishing trait when available, such as build, face shape, facial hair, or one prominent accessory.\n"
            "Use one complete grammatical sentence of no more than 35 words.\n"
            "Do not mention skin color, skin tone, ethnicity, race, or complexion.\n"
            "Do not describe the background, location, pose, camera angle, facial expression, mood, action, or what the performer is doing.\n"
            "Do not list minor jewelry, every accessory, fabric micro-details, or unseen details.\n"
            f"Treat the pictured person as {group_hint}."
        )
        if subject_label:
            instruction += f"\nUse this label only as the performer name if needed: {subject_label}"
    elif reference_type == "location":
        instruction = (
            "Look at the image and write one concise location/environment description.\n"
            "Output only the description, one paragraph, no markdown, no label, no bullet points.\n"
            "Describe the place itself: environment, architecture, layout, major objects, materials, colors, lighting, atmosphere, and visible setting details.\n"
            "Do not describe a main character, pose, performance, camera angle, or story action.\n"
            "Do not invent hidden or unseen details.\n"
            "Keep it under 100 words."
        )
        if name_hint:
            instruction += f"\nUse this label only as the location name if needed: {name_hint}"
    else:
        label = reference_type.replace("_", " ")
        instruction = (
            f"Look at the image and write one concise {label} reference description.\n"
            "Output only the description, one paragraph, no markdown, no label, no bullet points.\n"
            "Describe only the visible reference item: shape, form, materials, colors, markings, texture, scale cues, construction, accessories, and distinctive visual identity details.\n"
            "Do not describe the background, location, pose, camera angle, facial expression, mood, action, story meaning, or a person unless the reference item itself is a person.\n"
            "Do not invent hidden or unseen details.\n"
            "Keep it under 100 words."
        )
        if name_hint:
            instruction += f"\nUse this label only as the reference name if needed: {name_hint}"

    llm = _builder_local_llm(payload) if text_runner not in _EXTERNAL_LLM_RUNNERS else None
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else str(payload.get("lmstudio_model") or "").strip()
    mmproj_file = _builder_local_mmproj_file(payload, mmproj_file)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    n_ctx = int(payload.get("n_ctx") or 2048)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.2)
    top_p = float(payload.get("top_p") or 0.9)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 180))
    seed = payload.get("seed")
    clear_before_load = bool(payload.get("clear_before_load", False))
    unload_after = bool(payload.get("unload_after", True))

    try:
        if clear_before_load and llm:
            _clear_comfy_model_memory()
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
        model = None
        if llm:
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
        def clean_reference_description(raw_text):
            text = _clean_visual_gemma_text(raw_text)
            text = re.sub(r"^\s*(character|subject|location|description)\s*:\s*", "", text, flags=re.I).strip()
            if reference_type == "face":
                text = text.strip().strip('"').strip("'").strip()
                # Vision models occasionally leak a short thought and then emit the
                # requested prompt, e.g. "a photo of me thoughta photo of a woman...".
                # The final occurrence is the actual answer, even when the leaked
                # text omitted whitespace before it.
                markers = list(re.finditer(r"a\s+photo\s+of", text, flags=re.I))
                if markers:
                    text = text[markers[-1].start():]
                else:
                    text = re.sub(r"^\s*(?:face prompt|prompt|face description)\s*:\s*", "", text, flags=re.I)
                    text = f"a photo of {text.lstrip(' ,.-')}"
                text = re.sub(r"^a\s+photo\s+of\b", "a photo of", text, count=1, flags=re.I)
                text = re.sub(r"\s+", " ", text).strip()
                if len(re.findall(r"a\s+photo\s+of", text, flags=re.I)) != 1:
                    raise ValueError("The face description repeated its required opening phrase.")
                if re.search(r"\b(?:assistant|analysis|reasoning|let me|i think|i thought|my thought)\b", text, flags=re.I):
                    raise ValueError("The face description leaked model reasoning.")
            elif reference_type in {"subject", "character"}:
                if subject_label:
                    text = re.sub(r"\bthe character\b", subject_label, text, flags=re.I)
                    text = re.sub(r"\bthe subject\b", subject_label, text, flags=re.I)
                    text = re.sub(r"\ba character\b", subject_label, text, flags=re.I)
                    text = re.sub(r"\ba subject\b", subject_label, text, flags=re.I)
                text = re.sub(
                    r"\b(?:with|has|having|featuring)?\s*(?:very\s+|pale\s+|fair\s+|light\s+|medium\s+|tan\s+|tanned\s+|olive\s+|brown\s+|dark\s+|deep\s+|warm\s+|cool\s+|golden\s+|porcelain\s+|dusky\s+|caramel\s+|bronze\s+|dark-skinned\s+|light-skinned\s+)+(?:skin|skin tone|complexion)\b,?\s*(?:and\s+)?",
                    "",
                    text,
                    flags=re.I,
                )
                text = re.sub(r"\b(?:skin|skin tone|complexion)\s*(?:is|appears|looks)\s+[^,.]+,?\s*(?:and\s+)?", "", text, flags=re.I)
                text = re.sub(r"\s+,", ",", text)
                text = re.sub(r"(?:^|\.\s*)and\s+", "", text, flags=re.I).strip()
                text = re.sub(r"\s{2,}", " ", text).strip(" ,")
            words = text.split()
            if len(words) > 100:
                text = " ".join(words[:100]).rstrip(" ,.;:") + "."
            if not text:
                raise ValueError("Gemma returned an empty reference description.")
            _validate_reference_description(text, reference_type)
            return text

        last_error = None
        text = ""
        for attempt in range(2):
            attempt_instruction = instruction
            if attempt:
                attempt_instruction += (
                    "\n\nPrevious output was rejected because it repeated words or was not a usable description. "
                    "Write a normal visual description with varied concrete nouns. Do not repeat any word more than twice."
                )
            if text_runner in _EXTERNAL_LLM_RUNNERS:
                raw_text, _run_info = _try_run_remote_vision(
                    payload,
                    attempt_instruction,
                    [image],
                    temperature=0.05 if attempt else temperature,
                    top_p=0.75 if attempt else top_p,
                    max_new_tokens=max_new_tokens,
                )
            else:
                raw_text = llm._run_gguf_vision_pipeline(
                    model=model,
                    pil_images=[image],
                    instruction_text=attempt_instruction,
                    temperature=0.05 if attempt else temperature,
                    top_p=0.75 if attempt else top_p,
                    max_new_tokens=max_new_tokens,
                    seed=int(seed) if seed is not None else None,
                )
            try:
                text = clean_reference_description(raw_text)
                last_error = None
                break
            except ValueError as exc:
                last_error = exc
        if last_error:
            raise last_error
        return {"description": text, "used_model": model_path, "used_mmproj": mmproj_path, "runner": "llm_api_vision" if text_runner == "llm_api" else "own_server_vision" if text_runner == "own_server" else "lm_studio_vision" if text_runner == "lm_studio" else "builtin", "unloaded": False if text_runner in _EXTERNAL_LLM_RUNNERS else unload_after}
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _generate_flux_klein_prompt(payload):
    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    builder_instruction_key = str(payload.get("builder_instruction_key") or payload.get("instruction_key") or "flux_klein_t2i").strip()
    text_runner = _llm_runner_from_payload(payload)
    if text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file:
        raise ValueError("Choose a Gemma vision model first.")

    ingredients = payload.get("image_ingredients") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("Image ingredients must be a list.")
    reference_context = payload.get("reference_context") or {}
    if not isinstance(reference_context, dict):
        reference_context = {}
    has_subject_reference = bool(reference_context.get("has_subject_reference"))
    has_location_reference = bool(reference_context.get("has_location_reference"))
    subject_description = str(reference_context.get("subject_description", "") or "").strip()
    location_name = str(reference_context.get("location_name", "") or "").strip()
    location_description = str(reference_context.get("location_description", "") or "").strip()
    images = []
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        images.append(_image_from_prompt_payload(item.get("path", ""), item.get("data", ""), f"Image ingredient {index}"))
    combined_image = _combine_flux_ingredient_images(images)
    reference_rules = []
    if has_subject_reference:
        subject_line = "Use the subject reference for the main subject identity, face/body details, outfit, and visible character consistency."
        if subject_description:
            subject_line += f" Subject description: {subject_description}"
        reference_rules.append(subject_line)
    if has_location_reference:
        location_line = "Use the mapped location reference as the required setting/background. Do not replace it with a different location from the concept or notes."
        location_details = "; ".join(part for part in (location_name, location_description) if part)
        if location_details:
            location_line += f" Mapped location: {location_details}"
        reference_rules.append(location_line)
    reference_text = ""
    if reference_rules:
        reference_text = (
            "\nReference Builder priorities:\n"
            + "\n".join(f"- {rule}" for rule in reference_rules)
            + "\n- Use the user's notes/concept for action, pose, mood, lighting, story beat, and details after respecting the reference priorities.\n"
        )
    base_instructions = _effective_builder_instruction(payload, builder_instruction_key, _FLUX_KLEIN_T2I_INSTRUCTIONS)
    instruction = (
        f"{base_instructions}\n\n"
        "The image input contains the available reference images/visual ingredients. These may include a character, background, props, style references, or other visual ingredients.\n"
        f"{reference_text}"
        "\n"
        f"User input:\n{user_notes or 'Create a new image using the available reference images.'}"
    )

    llm = _builder_local_llm(payload) if text_runner not in _EXTERNAL_LLM_RUNNERS else None
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else str(payload.get("lmstudio_model") or "").strip()
    mmproj_file = _builder_local_mmproj_file(payload, mmproj_file)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    # Flux/Klein only needs one short prompt, and vision GGUF context is expensive.
    # Keep this lower than the broader Gemma prompt tools to reduce crash risk.
    n_ctx = int(payload.get("n_ctx") or 2048)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 350))
    seed = payload.get("seed")
    clear_before_load = bool(payload.get("clear_before_load", True))
    unload_after = bool(payload.get("unload_after", True))

    try:
        if clear_before_load and llm:
            _clear_comfy_model_memory()
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
        if text_runner in _EXTERNAL_LLM_RUNNERS:
            text, run_info = _try_run_remote_vision(
                payload,
                instruction,
                [combined_image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            run_info = {}
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[combined_image],
                instruction_text=instruction,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=int(seed) if seed is not None else None,
            )
        text = _clean_visual_gemma_text(text)
        text = _repair_and_validate_builder_gemma_prompt(payload, text, "Flux/Klein")
        return {"prompt": text, "used_model": run_info.get("used_model", model_path) if text_runner in _EXTERNAL_LLM_RUNNERS else model_path, "used_mmproj": mmproj_path, "runner": "llm_api_vision" if text_runner == "llm_api" else "own_server_vision" if text_runner == "own_server" else "lm_studio_vision" if text_runner == "lm_studio" else "builtin", "unloaded": False if text_runner in _EXTERNAL_LLM_RUNNERS else unload_after}
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
            _clear_comfy_model_memory()


def _analyze_builder_story_references(payload):
    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    text_runner = _llm_runner_from_payload(payload)
    if text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file:
        raise ValueError("Choose a Gemma vision model first.")
    ingredients = payload.get("image_ingredients") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("Story reference images must be a list.")
    images = []
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        images.append(_image_from_prompt_payload(item.get("path", ""), item.get("data", ""), f"Story reference image {index}"))
    if not images:
        raise ValueError("Add at least one Story Builder reference image first.")
    batches = [images[index:index + 4] for index in range(0, len(images), 4)]
    llm = _builder_local_llm(payload) if text_runner not in _EXTERNAL_LLM_RUNNERS else None
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else str(payload.get("lmstudio_model") or "").strip()
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    n_ctx = int(payload.get("n_ctx") or 4096)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 500))
    unload_after = bool(payload.get("unload_after", True))
    used_model = model_path
    try:
        model = None
        if llm:
            _clear_comfy_model_memory()
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
        batch_notes = []
        for batch_index, batch in enumerate(batches, start=1):
            combined_image = _combine_story_reference_batch(batch, cell_size=512)
            instruction = (
                "Analyze these reference images for a music video Story Builder. "
                "Each image has been resized into a 512px tile; this batch contains at most four images. "
                "Write compact reusable planning notes for a text-only agent. "
                "Identify likely singers/characters, clothing, faces/hair/body details, location ideas, props, color palette, lighting, mood, genre, and overall aesthetic. "
                "Do not write an image generation prompt. Do not mention image grids or panels. "
                "Use short labeled lines. Keep it under 180 words.\n\n"
                f"Batch {batch_index} of {len(batches)}.\n"
                f"User notes:\n{user_notes or 'Summarize the characters, locations, style, and aesthetic shown in the images.'}"
            )
            if text_runner in _EXTERNAL_LLM_RUNNERS:
                text, _run_info = _try_run_remote_vision(
                    payload,
                    instruction,
                    [combined_image],
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                used_model = _run_info.get("used_model", used_model)
            else:
                text = llm._run_gguf_vision_pipeline(
                    model=model,
                    pil_images=[combined_image],
                    instruction_text=instruction,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            text = _clean_visual_gemma_text(text)
            if _looks_like_gemma_repeat_failure(text):
                raise ValueError(f"Gemma returned repeated/thought junk instead of usable Story reference notes for batch {batch_index}.")
            if text.strip():
                prefix = f"Reference batch {batch_index}: " if len(batches) > 1 else ""
                batch_notes.append(prefix + text.strip())
        return {"notes": "\n\n".join(batch_notes).strip(), "used_model": used_model, "used_mmproj": mmproj_path, "runner": "llm_api_vision" if text_runner == "llm_api" else "own_server_vision" if text_runner == "own_server" else "lm_studio_vision" if text_runner == "lm_studio" else "builtin", "unloaded": False if text_runner in _EXTERNAL_LLM_RUNNERS else unload_after, "batches": len(batches)}
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
            _clear_comfy_model_memory()


def _generate_nb_image_prompt(payload):
    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    prompt_mode = str(payload.get("prompt_mode", "nano_banana") or "nano_banana").strip().lower()
    is_flow_gpt = prompt_mode == "flow_gpt"
    default_instruction_key = "flow_gpt_t2i" if is_flow_gpt else "nano_b_t2i"
    default_instruction_text = _FLOW_GPT_T2I_INSTRUCTIONS if is_flow_gpt else _NANO_B_T2I_INSTRUCTIONS
    prompt_label = "Flow/GPT" if is_flow_gpt else "NanoBanana"
    builder_instruction_key = str(payload.get("builder_instruction_key") or payload.get("instruction_key") or default_instruction_key).strip()
    text_runner = _llm_runner_from_payload(payload)
    if text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file:
        raise ValueError(f"Choose a {prompt_label} Gemma vision model first.")

    ingredients = payload.get("image_ingredients") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError(f"{prompt_label} reference images must be a list.")
    images = []
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        images.append(_image_from_prompt_payload(item.get("path", ""), item.get("data", ""), f"{prompt_label} reference image {index}"))
    reference_context = payload.get("reference_context") or {}
    if not isinstance(reference_context, dict):
        reference_context = {}

    context_parts = []
    subject_description = str(reference_context.get("subject_description", "") or "").strip()
    location_name = str(reference_context.get("location_name", "") or "").strip()
    location_description = str(reference_context.get("location_description", "") or "").strip()
    has_subject_reference = bool(reference_context.get("has_subject_reference"))
    has_location_reference = bool(reference_context.get("has_location_reference"))
    try:
        subject_reference_count = int(float(reference_context.get("subject_reference_count") or 1)) if has_subject_reference else 0
    except (TypeError, ValueError):
        subject_reference_count = 1 if has_subject_reference else 0
    subject_reference_count = max(0, min(99, subject_reference_count))
    character_reference_phrase = "character reference images" if subject_reference_count > 1 else "character reference image"
    character_identity_phrase = "character identities from the character references" if subject_reference_count > 1 else "character identity from the character reference"

    if subject_description:
        context_parts.append(f"Subject description:\n{subject_description}")
    if location_name or location_description:
        context_parts.append(f"Location reference:\n{location_name}\n{location_description}".strip())
    if user_notes:
        context_parts.append(f"User input:\n{user_notes}")

    reference_flags = []
    if has_subject_reference:
        reference_flags.append(f"{subject_reference_count or 1} character reference image{'s are' if (subject_reference_count or 1) != 1 else ' is'} available.")
    if has_location_reference:
        reference_flags.append("A scene/location reference image is available.")
    if reference_flags:
        context_parts.append("\n".join(reference_flags))

    has_images = bool(images)
    has_unmapped_reference_images = has_images and not has_subject_reference and not has_location_reference

    def _cleanup_nb_reference_claims(text):
        text = str(text or "")
        fallback_reference = "Using the provided reference image" if has_images else "Create"
        if not has_subject_reference and not has_location_reference:
            text = re.sub(
                r"\bUsing the provided character reference and location reference\b",
                fallback_reference,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bUsing the provided location reference and character reference\b",
                fallback_reference,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bUsing the provided (?:character|location|scene) reference\b",
                fallback_reference,
                text,
                flags=re.IGNORECASE,
            )
        elif not has_location_reference:
            text = re.sub(
                r"\bUsing the provided character reference and location reference\b",
                f"Using the provided {character_reference_phrase}",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bUsing the provided location reference and character reference\b",
                f"Using the provided {character_reference_phrase}",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s+and (?:the\s+)?(?:provided\s+)?(?:location|scene) reference(?: image)?\b", "", text, flags=re.IGNORECASE)
        elif not has_subject_reference:
            text = re.sub(
                r"\bUsing the provided character reference and location reference\b",
                "Using the provided location reference",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bUsing the provided location reference and character reference\b",
                "Using the provided location reference",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s+and (?:the\s+)?(?:provided\s+)?character reference(?: image)?\b", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\bcharacter reference(?: image)?\s+and\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _ensure_nb_reference_opening(text):
        text = str(text or "").strip()
        if not text:
            return text
        if has_subject_reference and has_location_reference:
            opening = f"Using the provided {character_reference_phrase} and location reference image"
        elif has_subject_reference:
            opening = f"Using the provided {character_reference_phrase}"
        elif has_location_reference:
            opening = "Using the provided location reference image"
        else:
            return text
        if re.search(r"\bUsing the provided (?:character|location|scene|reference image)", text, flags=re.IGNORECASE):
            return text
        if re.match(r"^(?:create|make|generate)\b", text, flags=re.IGNORECASE):
            text = re.sub(r"^(?:create|make|generate)\b\s*", "", text, count=1, flags=re.IGNORECASE)
        return f"{opening}, create {text[:1].lower()}{text[1:] if len(text) > 1 else ''}".strip()

    if has_subject_reference and has_location_reference:
        reference_prompt_rules = (
            f"- Start by mentioning both the provided {character_reference_phrase} and location reference image.\n"
            f"- Preserve the {character_identity_phrase}: face, hair, outfit, makeup, and overall identity.\n"
            "- Preserve the location identity from the location reference: environment, architecture, layout, atmosphere, and major visible setting details.\n"
            "- Use the user's scene/concept notes for action, pose, camera, mood, and story beat.\n"
            "- Do not paste the character into the location image.\n"
            "- Do not copy the exact pose, crop, camera angle, perspective, or composition from either reference.\n"
        )
        example_text = (
            f"Using the provided {character_reference_phrase} and location reference image, create a close-up profile shot of the woman in the misty forest. "
            f"Preserve {'the subjects identities, hair, outfits, makeup, and overall identity details from the character references' if subject_reference_count > 1 else 'her identity, hair, outfit, and crown from the character reference'} while using the forest reference for the white fibrous trees, mist, and eerie atmosphere. "
            "Use a new pose, new camera angle, soft bokeh, atmospheric haze, and dramatic rim lighting."
        )
    elif has_subject_reference:
        reference_prompt_rules = (
            f"- Start by mentioning only the provided {character_reference_phrase}. Do not mention a location reference image.\n"
            f"- Preserve the {character_identity_phrase}: face, hair, outfit, makeup, and overall identity.\n"
            "- Create the setting, background, atmosphere, and location from the user's scene/concept notes.\n"
            f"- Do not copy the {'character reference poses, studio backgrounds, crops, camera angles, or lens distances' if subject_reference_count > 1 else 'character reference pose, studio background, crop, camera angle, or lens distance'}.\n"
        )
        example_text = (
            f"Using the provided {character_reference_phrase}, create an intimate upper body shot of the woman in a misty white forest built from the scene concept. "
            f"Preserve {'the subjects identities, hair, outfits, and facial details' if subject_reference_count > 1 else 'her identity, blonde hair, lace outfit, and delicate facial details'} while placing the scene among pale gnarled trees, soft fog, shallow depth of field, and ethereal rim lighting."
        )
    elif has_location_reference:
        reference_prompt_rules = (
            "- Start by mentioning only the provided location reference. Do not mention a character reference.\n"
            "- Use the location reference for environment, architecture, layout, atmosphere, and major visible setting details.\n"
            "- Create any subject, pose, outfit, camera, and story details from the user's scene/concept notes.\n"
            "- Do not copy the exact location reference camera angle, framing, perspective, or composition.\n"
        )
        example_text = (
            "Using the provided location reference, create a cinematic medium shot of the scene's subject moving through the misty forest. "
            "Preserve the pale fibrous trees, narrow fog-covered path, and eerie atmosphere from the location reference while creating a new camera angle, new subject pose, soft bokeh, and dramatic haze."
        )
    elif has_unmapped_reference_images:
        reference_prompt_rules = (
            "- You may use the provided reference image or images only as loose visual guidance.\n"
            "- Do not call them character references or location references unless the context explicitly says that.\n"
            "- Use the user's scene/concept notes as the main source of subject, setting, action, pose, and mood.\n"
            "- Create a new camera angle, new pose, and new composition.\n"
        )
        example_text = (
            "Create a cinematic medium close-up from the scene concept, using the provided visual reference only as loose style guidance. "
            "Build the subject, setting, lighting, and atmosphere from the notes, with soft bokeh, layered haze, a new composition, and high cinematic detail."
        )
    else:
        reference_prompt_rules = (
            "- Do not mention provided references or reference images.\n"
            "- Treat this as normal text-to-image prompt writing from the user's scene/concept notes.\n"
            "- Create the subject, setting, action, pose, camera, lighting, and atmosphere from the notes.\n"
        )
        example_text = (
            "Create a cinematic medium close-up in a vast cosmic void filled with drifting pearlescent particles and soft ethereal light. "
            "Use a new composition, shallow depth of field, gentle atmospheric haze, luminous highlights, and a quiet dreamlike mood."
        )
    base_instructions = _effective_builder_instruction(payload, builder_instruction_key, default_instruction_text)
    instruction = (
        f"{base_instructions}\n\n"
        "Reference wording rules:\n"
        f"{reference_prompt_rules}"
        "\n"
        f"Good output example:\n{example_text}\n\n"
        f"{chr(10).join(context_parts) if context_parts else 'User input: Create a new image from the notes.'}"
    )

    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 900))
    seed = payload.get("seed")
    clear_before_load = bool(payload.get("clear_before_load", True))
    unload_after = bool(payload.get("unload_after", True))
    if not has_images:
        text, info = _run_builder_text_llm(
            payload,
            instruction,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            label=f"{prompt_label} text Gemma",
            preserve_paragraphs=False,
        )
        text = _clean_lm_studio_plain_text(text)
        if not text:
            raise ValueError(f"{prompt_label} Gemma returned an empty prompt.")
        text = _cleanup_nb_reference_claims(text)
        text = _ensure_nb_reference_opening(text)
        text = _repair_and_validate_builder_gemma_prompt(payload, text, prompt_label)
        return {"prompt": text, **info}

    combined_image = _combine_flux_ingredient_images(images)
    llm = _builder_local_llm(payload) if text_runner not in _EXTERNAL_LLM_RUNNERS else None
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else str(payload.get("lmstudio_model") or "").strip()
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    try:
        if text_runner in _EXTERNAL_LLM_RUNNERS:
            text, info = _try_run_remote_vision(
                payload,
                instruction,
                [combined_image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            if clear_before_load:
                _clear_comfy_model_memory()
                _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[combined_image],
                instruction_text=instruction,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=int(seed) if seed is not None else None,
            )
            info = {
                "runner": "builtin",
                "used_model": model_path,
                "used_mmproj": mmproj_path,
                "unloaded": unload_after,
            }
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
            _clear_comfy_model_memory()
    text = _clean_lm_studio_plain_text(text)
    text = re.sub(r"(?im)^CAMERA\s+COMPOSI+TION\s*\(PRIORITY\)", "CAMERA COMPOSITION (PRIORITY)", text)
    text = re.sub(r"(?im)^CHARACTER\s+REFEREN[CC]E\b", "CHARACTER REFERENCE", text)
    text = re.sub(r"(?im)^SCE+NE\s+REFEREN[CC]E\b", "SCENE REFERENCE", text)
    text = re.sub(r"\breference\s+imagae\b", "reference image", text, flags=re.IGNORECASE)
    text = re.sub(r"\breference\s+imagaes\b", "reference images", text, flags=re.IGNORECASE)
    text = _cleanup_nb_reference_claims(text)
    text = _ensure_nb_reference_opening(text)
    if not text:
        raise ValueError(f"{prompt_label} Gemma returned an empty prompt.")
    text = _repair_and_validate_builder_gemma_prompt(payload, text, prompt_label)
    return {"prompt": text, **info}


def _generate_flux_reference_location_map(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file and _llm_runner_from_payload(payload) not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose a non-vision Gemma model first.")

    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for location mapping.")

    cleaned_scenes = []
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("id", "") or f"scene_{index}").strip()
        label = str(scene.get("label", "") or f"Scene {index}").strip()
        concept = str(scene.get("concept", "") or "").strip()
        notes = str(scene.get("notes", "") or "").strip()
        if concept or notes:
            cleaned_scenes.append({
                "id": scene_id,
                "label": label,
                "concept": concept,
                "notes": notes,
            })
    if not cleaned_scenes:
        raise ValueError("Scenes need lyrics, scene notes, concept prompts, or timeline notes before Gemma can map locations.")

    subject_scene = _clean_location_context_text(payload.get("subject_scene_text", ""))
    existing_locations = payload.get("existing_locations") or []
    if not isinstance(existing_locations, list):
        existing_locations = []
    normalized_existing_locations = []
    seen_existing = set()
    existing_location_lines = []
    for item in existing_locations:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        if name:
            key = name.lower()
            if key not in seen_existing:
                seen_existing.add(key)
                normalized_existing_locations.append({"name": re.sub(r"\s+", " ", name), "description": re.sub(r"\s+", " ", description)})
            existing_location_lines.append(f"- {name}" + (f": {description}" if description else ""))
    if not normalized_existing_locations:
        raise ValueError("Auto Map needs locations first. Click Extract Locations or add locations manually, then run Auto Map.")

    scene_lines = []
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_lines.append(
            f"Scene {index}\n"
            f"id: {scene['id']}\n"
            f"label: {scene['label']}\n"
            f"concept: {scene['concept']}\n"
            f"notes: {scene['notes']}"
        )

    numbered_locations = "\n".join(
        f"{index}={item['name']}" + (f" | {item['description']}" if item.get("description") else "")
        for index, item in enumerate(normalized_existing_locations, start=1)
    )
    previous_counts = _location_usage_counts_from_payload(payload, normalized_existing_locations)
    usage_lines = "\n".join(
        f"- {name}: already used {int(previous_counts.get(name, 0) or 0)} time(s)"
        for name in sorted(previous_counts, key=lambda item: (int(previous_counts.get(item, 0) or 0), item.lower()))
    )
    instruction = (
        "You are mapping music-video scenes to an existing location list.\n\n"
        "Choose the best existing location number for each scene using the scene lyric line, concept text, notes, visual mood, objects, and environment. "
        "First use a location clearly named or implied by the scene text. "
        "If the scene has no clear location, choose the closest fit from the location list based on emotional tone and visual atmosphere. "
        "Use locations that have not been used yet before repeating locations that were already used. "
        "Avoid repeating the same location across too many neighboring scenes when another listed location fits equally well.\n\n"
        "Output only simple lines in this exact format:\n"
        "Scene1=1\n"
        "Scene2=3\n"
        "Scene3=3\n\n"
        "Rules:\n"
        "- Every scene must get one line.\n"
        "- Use only location numbers from the list.\n"
        "- Prefer the least-used matching location when multiple locations fit.\n"
        "- If there are enough scenes, use every listed location at least once before heavy repeats.\n"
        "- Do not output JSON, markdown, bullets, explanations, names, or descriptions.\n"
        "- Do not invent new locations.\n\n"
        f"Optional extra context, if any:\n{subject_scene or '(none)'}\n\n"
        f"Already used locations before these scenes:\n{usage_lines or '(none)'}\n\n"
        f"Locations:\n{numbered_locations}\n\n"
        f"Scenes:\n\n{chr(10).join(scene_lines)}"
    )

    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.8)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 5000))
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        label="Gemma",
    )
    normalized_locations = normalized_existing_locations
    normalized_map = _parse_scene_location_number_map(text, cleaned_scenes, normalized_locations)
    if not normalized_map:
        normalized_map = _fallback_location_map_by_overlap(cleaned_scenes, normalized_locations)
    else:
        fallback_map = _fallback_location_map_by_overlap(cleaned_scenes, normalized_locations)
        for scene in cleaned_scenes:
            normalized_map.setdefault(scene["id"], fallback_map[scene["id"]])
    normalized_map = _balance_location_map_by_usage(normalized_map, cleaned_scenes, normalized_locations, previous_counts)
    return {
        "locations": normalized_locations,
        "scene_map": normalized_map,
        "raw_text": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _generate_flux_reference_locations(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file and _llm_runner_from_payload(payload) not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose a non-vision Gemma model first.")
    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for location extraction.")
    cleaned_scenes = []
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        concept = str(scene.get("concept", "") or "").strip()
        notes = str(scene.get("notes", "") or "").strip()
        if concept or notes:
            cleaned_scenes.append({
                "id": str(scene.get("id", "") or f"scene_{index}").strip(),
                "label": str(scene.get("label", "") or f"Scene {index}").strip(),
                "concept": concept,
                "notes": notes,
            })
    if not cleaned_scenes:
        raise ValueError("Scenes need lyrics, scene notes, concept prompts, or timeline notes before Gemma can extract locations.")

    subject_scene = _clean_location_context_text(payload.get("subject_scene_text", ""))
    style_theme = _clean_location_context_text(payload.get("style_theme", ""))
    subject_context = _clean_location_context_text(payload.get("subject_context", ""))
    existing_locations = payload.get("existing_locations") or []
    max_locations = max(1, min(50, int(payload.get("max_locations") or 8)))
    existing_lines = []
    if isinstance(existing_locations, list):
        for item in existing_locations:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            description = str(item.get("description", "") or "").strip()
            if name:
                existing_lines.append(f"- {name}" + (f": {description}" if description else ""))

    scene_lines = []
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_lines.append(
            f"Scene {index}: {scene['label']}\n"
            f"concept: {scene['concept']}\n"
            f"notes: {scene['notes']}"
        )

    instruction = (
        "Extract a short reusable location list for Flux/Klein or Nano B reference images.\n\n"
        "Use the scene concept prompts and scene notes as the source of truth. "
        "Optional extra context may be empty; if it is empty or missing, ignore it completely. "
        "Use character/reference descriptions and style/theme only to understand the visual world, era, mood, genre, and design language. "
        "Do not turn characters, clothing, props, accessories, or body details into location names. "
        "Find concrete physical places, sets, rooms, buildings, landscapes, or backgrounds that repeat or are useful as references. "
        "If the extra context includes locations not directly named in a concept prompt, include them when they fit the project.\n\n"
        "Output only simple lines in this exact format:\n"
        "1|location name|short visual description for a reference image\n"
        "2|location name|short visual description for a reference image\n\n"
        "Rules:\n"
        "- Do not output JSON, markdown, bullets, headings, or explanations.\n"
        "- Keep names short, like bedroom, foggy white forest, salt-flat desert, ruined opera house.\n"
        "- Every name must be an actual place where the subject could stand, walk, sit, perform, or be filmed.\n"
        "- Do not output props, objects, clothing, accessories, body parts, people, characters, creatures, or symbolic items as locations.\n"
        "- Descriptions must describe only the place/background: architecture, layout, surfaces, lighting, weather, atmosphere, era, and color.\n"
        "- Do not include characters or actions in descriptions. No bride, woman, man, face, hand, tooth, dress, razor, locket, veil, or similar subject/object details.\n"
        "- Reuse broad locations instead of creating one unique location for every scene.\n"
        f"- Return no more than {max_locations} locations. Reuse broad locations instead of exceeding this maximum.\n\n"
        f"Optional style/theme guidance:\n{style_theme or '(none)'}\n\n"
        f"Character/reference descriptions for style guidance only:\n{subject_context or '(none)'}\n\n"
        f"Optional extra context:\n{subject_scene or '(none)'}\n\n"
        f"Existing user locations:\n{chr(10).join(existing_lines) if existing_lines else '(none)'}\n\n"
        f"Scenes:\n\n{chr(10).join(scene_lines)}"
    )
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.2),
        top_p=float(payload.get("top_p") or 0.8),
        max_new_tokens=int(payload.get("max_new_tokens") or 2200),
        label="Gemma",
    )
    locations = _parse_location_lines(text)
    if not locations:
        locations = _parse_location_ideas_flexible(text)
    if not locations:
        try:
            data = _extract_json_object_from_text(_clean_visual_gemma_text(text))
            raw_locations = data.get("locations") if isinstance(data, dict) else data
            if isinstance(raw_locations, list):
                for item in raw_locations:
                    if isinstance(item, dict):
                        raw_name = item.get("name", "")
                        name = _normalize_location_name(raw_name)
                        description = re.sub(r"\s+", " ", str(item.get("description", "") or "").strip())
                        if name:
                            name, description = _clean_location_card(name, description, raw_name)
                            if _valid_location_card(name, description):
                                locations.append({"name": name, "description": description})
        except Exception:
            pass
    if not locations:
        retry_instruction = (
            "Return only reusable music video filming locations from the scene text below.\n"
            "Do not explain anything.\n"
            "Do not summarize the scenes.\n"
            f"Write no more than {max_locations} locations.\n"
            "Write one location per line as a short bullet.\n"
            "Each bullet must be a concrete visual place/background where the subject could stand or move.\n"
            "Reject props, objects, clothing, accessories, people, body parts, symbolic items, and actions.\n"
            "Describe only the environment, not characters or objects from the lyrics.\n"
            "If any optional context is missing, invisible, empty, or unavailable, ignore that and use only the scene text.\n"
            "Never mention missing files, missing context, subjectsandscenes.txt, prompts, or instructions.\n"
            "Example format:\n"
            "- Abandoned motel pool, turquoise water under buzzing neon signs\n"
            "- Foggy pine road, wet asphalt and fading headlights\n\n"
            f"Optional style/theme guidance:\n{style_theme or '(none)'}\n\n"
            f"Character/reference descriptions for style guidance only:\n{subject_context or '(none)'}\n\n"
            f"Optional extra context:\n{subject_scene or '(none)'}\n\n"
            f"Existing user locations:\n{chr(10).join(existing_lines) if existing_lines else '(none)'}\n\n"
            f"Scenes:\n\n{chr(10).join(scene_lines)}"
        )
        retry_text, retry_info = _run_builder_text_llm(
            payload,
            retry_instruction,
            temperature=float(payload.get("temperature") or 0.35),
            top_p=float(payload.get("top_p") or 0.9),
            max_new_tokens=int(payload.get("max_new_tokens") or 2200),
            label="Gemma",
        )
        retry_locations = _parse_location_lines(retry_text) or _parse_location_ideas_flexible(retry_text)
        if retry_locations:
            text = retry_text
            run_info = retry_info
            locations = retry_locations
    if not locations:
        try:
            scout_payload = {
                **payload,
                "subject_scene_text": "",
                "style_theme": style_theme,
                "subject_context": subject_context,
                "lyrics_text": "\n\n".join(scene_lines),
                "user_input": "Create reusable location ideas from these music-video scene lines.",
                "temperature": float(payload.get("temperature") or 0.45),
                "top_p": float(payload.get("top_p") or 0.9),
                "max_new_tokens": int(payload.get("max_new_tokens") or 2200),
            }
            scout_result = _generate_wizard_locations_from_lyrics(scout_payload)
            locations = scout_result.get("locations") or []
            if locations:
                text = scout_result.get("raw_text", text)
                run_info = {
                    **run_info,
                    "used_model": scout_result.get("used_model", run_info.get("used_model", "")),
                    "runner": scout_result.get("runner", run_info.get("runner", "builtin")),
                    "unloaded": scout_result.get("unloaded", run_info.get("unloaded", True)),
                }
        except Exception:
            pass
    if not locations:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(preview) > 700:
            preview = preview[:697].rstrip() + "..."
        raise ValueError(f"Gemma did not return any usable locations. Raw response preview: {preview or '(empty)'}")
    deduped = []
    seen = set()
    for item in locations:
        raw_name = item.get("name", "")
        name, description = _clean_location_card(raw_name, item.get("description", ""), raw_name)
        if not _valid_location_card(name, description):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"name": name, "description": description})
    if not deduped:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(preview) > 700:
            preview = preview[:697].rstrip() + "..."
        raise ValueError(f"Gemma returned only non-location items. Raw response preview: {preview or '(empty)'}")
    return {
        "locations": deduped[:max_locations],
        "raw_text": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _generate_wizard_locations_from_lyrics(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file and _llm_runner_from_payload(payload) not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose a non-vision Gemma model first.")
    lyrics_text = str(payload.get("lyrics_text", "") or payload.get("lyrics", "") or "").strip()
    user_input = str(payload.get("user_input", "") or payload.get("notes", "") or "").strip()
    style_theme = _clean_location_context_text(payload.get("style_theme", ""))
    subject_context = _clean_location_context_text(payload.get("subject_context", ""))
    max_locations = max(1, min(50, int(payload.get("max_locations") or 8)))
    if not lyrics_text:
        raise ValueError("Paste or create lyrics before creating locations from lyrics.")

    existing_locations = payload.get("existing_locations") or []
    existing_lines = []
    if isinstance(existing_locations, list):
        for item in existing_locations:
            if not isinstance(item, dict):
                continue
            name = re.sub(r"\s+", " ", str(item.get("name", "") or "").strip())
            description = re.sub(r"\s+", " ", str(item.get("description", "") or "").strip())
            if name:
                existing_lines.append(f"- {name}" + (f": {description}" if description else ""))

    instruction = (
        "You are a music video location scout.\n\n"
        "The user will provide song lyrics.\n\n"
        "Your task is to analyze the mood, imagery, setting clues, themes, and emotional tone of the lyrics, "
        "then generate a list of reusable filming locations where the main subject or character could be placed.\n\n"
        "Return only actual locations, sets, rooms, buildings, outdoor areas, roads, stages, landscapes, or environments.\n\n"
        "Use character/reference descriptions and style/theme only to understand the visual world, era, mood, genre, and design language. "
        "Do not turn characters, clothing, props, accessories, or body details into location names.\n\n"
        "Rules:\n\n"
        "Do not summarize the lyrics.\n"
        "Do not explain the song meaning.\n"
        "Do not quote long lyric sections.\n"
        "Focus on visual places that could realistically appear in a music video and hold the subject.\n"
        "Include literal locations from the lyrics and cinematic locations inspired by the mood, but they must still be real places.\n"
        "Do not output props, objects, clothing, accessories, body parts, people, characters, creatures, or symbolic items as locations.\n"
        "Bad location names: Vintage shaving kit, Ornate sugar bowl, Lace veil, Silver frame, Human tooth, Ghostly face, Tattered dress.\n"
        "Good location names: Grand ballroom, Dark wood study, Steamy bathroom, Overgrown garden, Dimly lit hallway, Empty stage.\n"
        "Descriptions must describe only the place: architecture, layout, surfaces, lighting, weather, atmosphere, era, and color.\n"
        "Do not include characters or actions in descriptions. No bride, woman, man, face, hand, tooth, dress, razor, locket, veil, or similar subject/object details.\n"
        "Test every idea: could the selected subject stand, walk, sit, perform, or be filmed inside this place? If no, reject it.\n"
        "Make the locations specific and cinematic.\n"
        f"Output no more than {max_locations} locations. Prefer reusable locations rather than one location per scene.\n"
        "Use short bullet points.\n"
        "Each bullet should be a place only, with a brief visual detail if helpful.\n\n"
        "Output format:\n\n"
        "Music Video Locations:\n\n"
        "- [location idea]\n"
        "- [location idea]\n"
        "- [location idea]\n\n"
        f"Existing user locations to avoid duplicating exactly:\n{chr(10).join(existing_lines) if existing_lines else '(none)'}\n\n"
        f"Optional style/theme guidance:\n{style_theme or '(none)'}\n\n"
        f"Character/reference descriptions for style guidance only:\n{subject_context or '(none)'}\n\n"
        f"Optional user input:\n{user_input or '(none)'}\n\n"
        f"User lyrics:\n{lyrics_text}"
    )
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.9),
        max_new_tokens=int(payload.get("max_new_tokens") or 2200),
        label="Gemma",
    )
    locations = _parse_location_ideas_flexible(text)
    if not locations:
        retry_instruction = (
            f"Return no more than {max_locations} reusable music video filming locations for the lyrics below.\n"
            "Do not write a heading.\n"
            "Do not explain anything.\n"
            "Do not summarize the lyrics.\n"
            "Write one location per line as a short bullet.\n"
            "Each bullet must be a specific cinematic place where the subject could stand, walk, sit, perform, or be filmed.\n"
            "Reject props, objects, clothing, accessories, body parts, people, characters, creatures, symbolic items, and actions.\n"
            "Descriptions must describe only the place/background, not characters or objects from the lyrics.\n"
            "Example format:\n"
            "- Abandoned motel pool, turquoise water under buzzing neon signs\n"
            "- Foggy pine road, wet asphalt and fading headlights\n\n"
            f"Optional style/theme guidance:\n{style_theme or '(none)'}\n\n"
            f"Character/reference descriptions for style guidance only:\n{subject_context or '(none)'}\n\n"
            f"Optional user input:\n{user_input or '(none)'}\n\n"
            f"Lyrics:\n{lyrics_text}"
        )
        retry_text, retry_info = _run_builder_text_llm(
            payload,
            retry_instruction,
            temperature=float(payload.get("temperature") or 0.55),
            top_p=float(payload.get("top_p") or 0.9),
            max_new_tokens=int(payload.get("max_new_tokens") or 2200),
            label="Gemma",
        )
        retry_locations = _parse_location_ideas_flexible(retry_text)
        if retry_locations:
            text = retry_text
            run_info = retry_info
            locations = retry_locations
    if not locations:
        locations = _parse_location_lines(text)
    if not locations:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(preview) > 700:
            preview = preview[:697].rstrip() + "..."
        raise ValueError(f"Gemma did not return any usable location ideas. Raw response preview: {preview or '(empty)'}")
    deduped = []
    seen = set()
    for item in locations:
        raw_name = item.get("name", "")
        name, description = _clean_location_card(raw_name, item.get("description", ""), raw_name)
        if not _valid_location_card(name, description):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"name": name, "description": description})
    if not deduped:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(preview) > 700:
            preview = preview[:697].rstrip() + "..."
        raise ValueError(f"Gemma returned only non-location items. Raw response preview: {preview or '(empty)'}")
    return {
        "locations": deduped[:max_locations],
        "raw_text": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _generate_flux_reference_subjects(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file and _llm_runner_from_payload(payload) not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose a non-vision Gemma model first.")
    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for subject extraction.")
    cleaned_scenes = []
    scene_note_fallbacks = _load_scene_notes_json(payload.get("project_folder", ""))
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        concept = str(scene.get("concept", "") or "").strip()
        notes = str(scene.get("notes", "") or "").strip()
        director_note = str(scene.get("director_note", "") or "").strip() or scene_note_fallbacks.get(index, "")
        if concept or notes or director_note:
            cleaned_scenes.append({
                "id": str(scene.get("id", "") or f"scene_{index}").strip(),
                "label": str(scene.get("label", "") or f"Scene {index}").strip(),
                "concept": concept,
                "notes": notes,
                "director_note": director_note,
            })
    if not cleaned_scenes:
        raise ValueError("Scenes need concept prompt text, notes, or Director Notes before Gemma can extract subjects.")

    requested_count = max(2, min(12, int(payload.get("requested_count") or 2)))
    subject_scene = str(payload.get("subject_scene_text", "") or "").strip()
    existing_subjects = payload.get("existing_subjects") or []
    existing_lines = []
    if isinstance(existing_subjects, list):
        for item in existing_subjects:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            description = str(item.get("description", "") or "").strip()
            if name and description:
                existing_lines.append(f"- {name}: {description}")

    scene_lines = []
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_lines.append(
            f"Scene {index}: {scene['label']}\n"
            f"concept: {scene['concept']}\n"
            f"notes: {scene['notes']}\n"
            f"director_note: {scene['director_note']}"
        )

    instruction = (
        "Extract reusable character/subject identities for reference images.\n\n"
        "Look at the scene concept prompts, notes, Director Notes, and subject/scene context. "
        "Find distinct recurring people, creatures, mascots, or main visual subjects that may need separate reference images. "
        f"The user expects about {requested_count} character references if the project supports that count.\n\n"
        "Output only simple lines in this exact format:\n"
        "1|character name|short visual description for a character reference image\n"
        "2|character name|short visual description for a character reference image\n\n"
        "Rules:\n"
        "- Do not output JSON, markdown, bullets, headings, or explanations.\n"
        "- Keep names short and stable, like blonde woman, masked man, young singer, red android.\n"
        "- Descriptions must describe identity, face/body, hair, outfit, colors, and visual consistency details.\n"
        "- Do not describe locations as characters.\n"
        "- If two characters appear together in a scene, list them as separate characters, not one combined subject.\n\n"
        f"Subject/scene context:\n{subject_scene or '(none)'}\n\n"
        f"Existing user subjects:\n{chr(10).join(existing_lines) if existing_lines else '(none)'}\n\n"
        f"Scenes:\n\n{chr(10).join(scene_lines)}"
    )
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.2),
        top_p=float(payload.get("top_p") or 0.8),
        max_new_tokens=int(payload.get("max_new_tokens") or 2200),
        label="Gemma",
    )
    subjects = _parse_subject_lines(text)
    if not subjects:
        raise ValueError("Gemma did not return any usable subjects.")
    deduped = []
    seen = set()
    for item in subjects:
        key = item["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return {
        "subjects": deduped,
        "raw_text": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }


def _generate_flux_reference_zimage_prompt(payload):
    model_file = str(payload.get("model_file", "") or "").strip()
    reference_type = str(payload.get("reference_type", "") or "").strip().lower()
    source_text = str(payload.get("source_text", "") or "").strip()
    style_theme = str(payload.get("style_theme", "") or "").strip()
    if reference_type not in {"subject", "location"}:
        raise ValueError("Reference type must be subject or location.")
    if not model_file and _llm_runner_from_payload(payload) not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose a non-vision Gemma model first.")
    if not source_text:
        raise ValueError("Enter a subject or location description first.")

    if reference_type == "subject":
        instruction = (
            "Create one text-to-image prompt for a character reference sheet.\n\n"
            "The user input may be a simple subject description or a structured character-creation brief with sections such as generation mode, subject label, reference type, existing description, lyrics, song style, gender/role, reference image notes, and extra user direction. "
            "When lyrics or song style are provided, use them to invent or refine the character's identity, wardrobe, era, mood, expression, and visual design. Do not quote lyrics or make the final prompt a lyric scene. "
            "When an existing description or reference image notes are provided, preserve those explicit identity details and use lyrics/style only as supporting visual influence.\n\n"
            "The image must contain the same character shown three times in one image, with clearly different camera distances:\n"
            "1. Left panel: extreme close-up face portrait only, from top of hair to just below the chin. The face fills 80-90% of the panel. No shoulders, chest, torso, hands, or outfit details visible except maybe a tiny neckline.\n"
            "2. Center panel: upper-body waist-up portrait, from head to waist. Face, shoulders, chest, arms, and main outfit details visible.\n"
            "3. Right panel: full-body standing view, from head to shoes. Entire outfit, body proportions, legs, and feet visible.\n\n"
            "All three views must show the same person with consistent face, hair, outfit, colors, body type, and identity. "
            "Use a clean neutral studio background. The character should face forward or mostly forward. Keep lighting clear and even. "
            "Do not make the left and center panels the same crop. Do not create a cinematic scene, action pose, environment, story moment, props, text labels, captions, logos, watermarks, or multiple different characters.\n\n"
            "Use this exact output structure:\n\n"
            "A clean three-panel character reference sheet on a neutral studio background, showing the same [subject] in all panels with consistent [face/hair/body/outfit/identity details]. "
            "Left panel: extreme close-up face-only portrait, top of hair to just below chin, face fills 80-90% of the panel, no shoulders, no chest, no torso. "
            "Center panel: upper-body waist-up portrait from head to waist, showing shoulders, chest, arms, and outfit details. "
            "Right panel: full-body standing view from head to shoes, entire outfit and feet visible. "
            "Each panel uses a clearly different camera distance: face-only close-up, waist-up portrait, full-body standing. "
            "Clear even lighting, front-facing pose, consistent outfit colors and body proportions, detailed character design, no text, no labels, no props, no environment.\n\n"
            "Rules:\n"
            "- Output only one polished text-to-image prompt.\n"
            "- Do not include markdown, labels, quotes, explanations, or multiple options.\n"
            "- Keep it as one single image containing three views of the same character.\n"
            "- The left panel must be face-only, not another upper-body portrait.\n"
            "- Preserve the subject identity from the user input.\n\n"
            f"Subject creation brief:\n{source_text}\n\n"
            f"Optional global style/theme:\n{style_theme or '(none)'}"
        )
    else:
        instruction = (
            "Create one text-to-image prompt for a reusable location reference image.\n\n"
            "The image must show only the physical environment/location. Do not include the main character, people, animals, readable text, captions, logos, watermarks, or story action.\n\n"
            "Important: if the input, style/theme, or surrounding context mentions a character reference sheet, white/neutral studio background, plain backdrop, panel layout, close-up portrait, upper-body portrait, or full-body character view, treat those as character-reference-sheet artifacts only. "
            "Do not turn those artifacts into the location. Do not create a white studio, neutral studio, seamless photo backdrop, blank room, or character-sheet background unless the user explicitly names that as the intended location.\n\n"
            "Use this exact output structure:\n\n"
            "A clear cinematic environment reference image of [location/environment], showing [layout/architecture], [important furniture/props/objects], [lighting details], [colors/materials/textures], and [atmosphere]. "
            "Wide enough framing to understand the space layout, [time of day/weather if relevant], no people, no animals, no readable text, no logos, no captions.\n\n"
            "Rules:\n"
            "- Output only one polished text-to-image prompt.\n"
            "- Do not include markdown, labels, quotes, explanations, or multiple options.\n"
            "- Do not include the main character.\n"
            "- Do not describe a music video action.\n"
            "- Keep it as a reusable setting/reference image.\n"
            "- Preserve the location identity from the user input.\n"
            "- Ignore character-sheet backgrounds, panel layouts, and portrait crops when inventing the setting.\n"
            "- Use the optional style/theme only for visual mood, lighting, color, or texture.\n\n"
            f"Location description:\n{source_text}\n\n"
            f"Optional global style/theme:\n{style_theme or '(none)'}"
        )

    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.8)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 900))

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        label="Gemma",
    )
    text = _clean_visual_gemma_text(text)
    _validate_builder_gemma_prompt(text, f"Flux/Klein {reference_type} reference")
    return {
        "prompt": text,
        "used_model": run_info.get("used_model", ""),
        "runner": run_info.get("runner", "builtin"),
        "unloaded": run_info.get("unloaded", True),
    }
