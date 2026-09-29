"""Text-to-image prompts for the Video Builder's image models."""


_VISUAL_T2I_INSTRUCTIONS = """Create one text-to-image prompt from the provided image and user input.

User input includes:
- a reference image
- optional user notes

Use all parts of the input together.

Priority:
- Use the provided image as the main visual foundation.
- Preserve the visible subject, setting, outfit, mood, lighting, and scene identity from the image unless the user clearly asks to change them.
- Use the user notes to adjust framing, pose, camera distance, lighting, mood, action, wardrobe refinement, or environment details.

Rules:
- Create one polished text-to-image prompt.
- Treat the provided image as the base scene reference.
- Describe only what should be visible in the final generated image.
- If the user asks for a closer shot, wider shot, different pose, different lighting, different camera angle, or different mood, apply that change while keeping the same core subject and scene identity.
- Keep the image prompt concrete and visual.
- Do not use metaphors, abstract symbolic wording, or non-visible language.
- Do not explain your choices.
- Only send the final prompt text.

Use this exact format:

A high resolution cinematic photograph of a [subject], [action or pose based on the reference image and user notes], in [environment/location based on the reference image], during [time of day]. The subject is wearing [main outfit visible in the reference image refined by the user notes], [shoes/accessories visible in the reference image refined by the user notes], and [additional visible style details inspired by the user notes]. Their hair is [hair color], [hair length/style], and [movement or texture]. The environment is [visual style of location from the reference image shaped by the user notes] with [background details visible in the reference image], [lighting and color details based on the reference image and user notes], and [surface/reflection/material details connected to the reference image]. Camera is [camera angle/framing requested by the user or inferred from the image] with a [lens type or framing]. The weather is [weather condition appropriate to the scene], with [atmospheric detail influenced by the reference image and user notes], creating a [mood] mood.

[subject] = character gender! don't just say "subject"!

Only send the final prompt text. Do not include labels, notes, quotes, or extra text.

User Input:"""


_TEXT_ONLY_T2I_INSTRUCTIONS = """Create one text-to-image prompt from the user input.

User input includes:
- user notes describing the desired image

Use the user notes as the full scene foundation. Preserve all concrete visible details from the user notes, including subject, setting, outfit, pose, mood, lighting, camera framing, and environment details. If details are missing, infer only the visual details needed to make one complete image prompt.

Rules:
- Create one polished text-to-image prompt.
- Keep the image prompt concrete and visual.
- Do not use metaphors, abstract symbolic wording, or non-visible language.
- Do not explain your choices.
- Only send the final prompt text.

Use this exact format:

A high resolution cinematic photograph of a [subject], [action or pose based on the user notes], in [environment/location based on the user notes], during [time of day]. The subject is wearing [main outfit based on the user notes], [shoes/accessories based on the user notes], and [additional visible style details based on the user notes]. Their hair is [hair color], [hair length/style], and [movement or texture]. The environment is [visual style of location from the user notes] with [background details], [lighting and color details], and [surface/reflection/material details]. Camera is [camera angle/framing requested by the user or inferred from the notes] with a [lens type or framing]. The weather is [weather condition appropriate to the scene], with [atmospheric detail], creating a [mood] mood.

[subject] = character gender! don't just say "subject"!

Only send the final prompt text. Do not include labels, notes, quotes, or extra text.

User Input:"""


_FLUX_KLEIN_T2I_INSTRUCTIONS = """Create one concise Flux/Klein image prompt from the user input and any available reference context or image ingredients.

Output one normal paragraph, not sections, not markdown, not labels, not explanations.

Prompt style:
- If character and location references are available, start with: Using the provided character reference and location reference, create...
- If only a character reference is available, start with: Using the provided character reference, create...
- If only a location reference is available, start with: Using the provided location reference, create...
- If no visual reference is available, start directly with the shot and subject; do not claim a provided reference exists.
- Use a clear cinematic shot type such as close-up, profile close-up, medium close-up, upper body shot, waist-up shot, three-quarter shot, seated shot, over-the-shoulder shot, or low-angle portrait.
- Use the user's scene/concept notes as the main creative direction.
- Preserve character identity from character references when provided: face, hair, outfit, makeup, and overall identity.
- Preserve location identity from location references when provided: environment, architecture, layout, atmosphere, and major visible setting details.
- Create a new camera angle, new pose, and new composition.
- Do not paste the character into the location image.
- Do not copy the character reference pose, full-body standing pose, studio background, panel layout, crop, camera angle, or lens distance.
- Do not copy the exact location reference camera angle, framing, perspective, or composition.
- Avoid full-body walking or standing shots unless the user specifically asks for them.
- Prefer intimate cinematic compositions when no shot type is specified: close-up, medium close-up, profile, upper body, shallow depth of field, foreground framing, soft bokeh, rim light, atmospheric lighting.
- Keep it cinematic, detailed, visually specific, and practical for image generation.
- Do not include captions, text overlays, dialogue, markdown, labels, bullet points, or section headers.
- Keep the prompt under 120 words.

Good output examples:
Using the provided character reference and location reference, create a close-up profile shot of the woman in the misty forest. Focus on her expression and the intricate details of her crown while the pale trees and fog appear softly blurred in the background. Use a cool moody palette, atmospheric haze, shallow depth of field, dramatic rim lighting, and high cinematic detail.
Using the provided character reference and location reference, create an intimate upper body shot of the woman framed by gnarled forest branches. Preserve her identity, hair, outfit, and crown from the character reference while using the forest reference for the white fibrous trees, mist, and eerie atmosphere. New pose, new camera angle, soft bokeh, high cinematic quality."""


_NANO_B_T2I_INSTRUCTIONS = """Create one concise NanoBanana image prompt from the user input and any available reference context.

Output one normal paragraph, not sections, not markdown, not labels, not explanations.

Prompt style:
- Use advanced Krea 2-style image prompting: concrete subject identity, wardrobe, hair, makeup, pose, camera framing, lens feel, lighting setup, environment, materials, atmosphere, color palette, texture, and cinematic finish.
- Use a clear cinematic shot type such as close-up, profile close-up, medium close-up, upper body shot, waist-up shot, three-quarter shot, seated shot, over-the-shoulder shot, or low-angle portrait.
- Use the user's scene/concept notes as the main creative direction.
- Create a new camera angle, new pose, and new composition.
- Avoid full-body walking or standing shots unless the user specifically asks for them.
- Prefer intimate cinematic compositions when no shot type is specified: close-up, medium close-up, profile, upper body, shallow depth of field, foreground framing, soft bokeh, rim light, atmospheric lighting.
- Keep the prompt visually specific and practical for image generation.
- Do not include captions, text overlays, dialogue, markdown, labels, bullet points, or section headers."""


_FLOW_GPT_T2I_INSTRUCTIONS = """Create one concise browser image prompt for Flow/GPT from the user input and any available reference context.

Output one normal paragraph, not sections, not markdown, not labels, not explanations.

Prompt style:
- Write a direct image-generation prompt that can be pasted into Flow or GPT Image.
- Use the user's scene/concept notes as the main creative direction.
- If character or location references are available, preserve their important identity and setting details without overexplaining the reference system.
- Use concrete subject identity, wardrobe, hair, makeup, pose, camera framing, lens feel, lighting setup, environment, materials, atmosphere, color palette, texture, and cinematic finish.
- Create a clear still-image composition, not a video prompt.
- Create a new camera angle, new pose, and new composition.
- Avoid full-body walking or standing shots unless the user specifically asks for them.
- Prefer intimate cinematic compositions when no shot type is specified: close-up, medium close-up, profile, upper body, shallow depth of field, foreground framing, soft bokeh, rim light, atmospheric lighting.
- Do not include captions, text overlays, dialogue, markdown, labels, bullet points, or section headers.
- Do not include aspect ratio text; GPT Image aspect ratio is appended separately by the browser runner."""


_STANDARD_IMAGE_T2I_INSTRUCTIONS = """You are a text-to-image prompt builder for a music-video storyboard.

The user will provide a JSON scene-card bundle. Your job is to read the JSON and create one polished text-to-image prompt for the selected scene.

Use `selected_scene_number` to choose the scene.

Rules:

* Create one cinematic still-frame prompt, not a video prompt.
* Use advanced Krea 2-style image prompting: concrete subject identity, wardrobe, hair, makeup, pose, camera framing, lens feel, lighting setup, environment, materials, atmosphere, color palette, texture, and cinematic finish.
* Pull the visible subject list only from the selected scene's `subject_refs`.
* Never use subjects from the project catalog, another scene, the song story brief, or the user story arc unless that subject is also present in the selected scene's `subject_refs`.
* If `subject_refs` has more than one subject, every listed subject must be visibly present in the image prompt. Do not drop, merge, hide, imply, or omit any listed subject.
* If `subject_refs` has one subject, describe only that one visible subject. Do not create duplicates, backup singers, crowds, or extra people unless the scene notes explicitly ask for them.
* If `vocal_status.no_character_present` is true, do not include, mention, imply, or describe any mapped character/singer/subject. Use the location, props, environment, objects, atmosphere, and composition instead.
* Pull the setting from `location_ref`.
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


def _image_prompt_edit_instructions(payload):
    current_prompt = str(payload.get("current_prompt") or "").strip()
    edit_request = str(payload.get("edit_request") or "").strip()
    mode_label = str(payload.get("mode_label") or "image").strip() or "image"
    prompt_mode = str(payload.get("prompt_mode") or "").strip().lower()
    use_full_scene_context = bool(payload.get("use_full_scene_context"))
    use_vision_reference = bool(payload.get("use_vision_reference"))
    scene_context = payload.get("scene_context") if isinstance(payload.get("scene_context"), dict) else {}
    reference_context = payload.get("reference_context") if isinstance(payload.get("reference_context"), dict) else {}
    if not current_prompt:
        raise ValueError("Current image prompt is empty.")
    if not edit_request:
        raise ValueError("Edit request is empty.")
    has_subject_reference = bool(reference_context.get("has_subject_reference"))
    has_location_reference = bool(reference_context.get("has_location_reference"))
    try:
        subject_reference_count = int(float(reference_context.get("subject_reference_count") or 1)) if has_subject_reference else 0
    except (TypeError, ValueError):
        subject_reference_count = 1 if has_subject_reference else 0
    subject_reference_count = max(0, min(99, subject_reference_count))
    character_reference_phrase = "character reference images" if subject_reference_count > 1 else "character reference image"
    reference_opening = ""
    if prompt_mode in {"nano_banana", "flow_gpt"}:
        if has_subject_reference and has_location_reference:
            reference_opening = f"Using the provided {character_reference_phrase} and location reference image"
        elif has_subject_reference:
            reference_opening = f"Using the provided {character_reference_phrase}"
        elif has_location_reference:
            reference_opening = "Using the provided location reference image"
    mode_rules = ""
    if prompt_mode in {"nano_banana", "flow_gpt"}:
        mode_rules = (
            f"This is a {'Flow/GPT browser image' if prompt_mode == 'flow_gpt' else 'NanoBanana'} prompt. Preserve any required wording about provided character or location reference images unless the user explicitly asks to change reference usage. "
            "Use advanced Krea 2-style image prompting: concrete subject identity, wardrobe, hair, makeup, pose, camera framing, lens feel, lighting setup, environment, materials, atmosphere, color palette, texture, and cinematic finish. "
            "Keep it practical as one image-generation prompt paragraph."
        )
        if reference_opening:
            mode_rules += f" The revised prompt must start with exactly this reference opening before the rest of the prompt: \"{reference_opening}, create\"."
    elif prompt_mode == "flux_klein":
        mode_rules = (
            "This is a Flux/Klein image prompt. Preserve mapped subject and location identity from the prompt/reference context unless the user explicitly asks to change them. "
            "Do not mention image indexes or internal reference labels."
        )
    elif prompt_mode == "krea2_2pass":
        mode_rules = "This is a Krea 2 image prompt. Keep enough concrete detail for pose, wardrobe, lighting, camera framing, environment, materials, and atmosphere."
    else:
        mode_rules = "This is a text-to-image prompt. Keep it visually specific and usable directly by an image generation model."

    context_text = ""
    if use_full_scene_context:
        context_text = (
            "\n\nFull scene context is enabled. Use this context only when it helps satisfy the requested edit. "
            "Do not rewrite unrelated parts just because context is present.\n"
            f"Scene label: {str(scene_context.get('label') or '').strip() or '(none)'}\n"
            f"Scene notes: {str(scene_context.get('scene_notes') or '').strip() or '(none)'}\n"
            f"Director note: {str(scene_context.get('director_note') or '').strip() or '(none)'}\n"
            f"Lyric section: {str(scene_context.get('lyric_section') or '').strip() or '(none)'}\n"
            f"Lyric text: {str(scene_context.get('lyric_text') or '').strip() or '(none)'}\n"
            f"Subject context: {str(scene_context.get('subject_context') or '').strip() or '(none)'}\n"
            f"Location context: {str(scene_context.get('location_context') or '').strip() or '(none)'}\n"
            f"No character present: {bool(scene_context.get('no_character_present'))}"
        )
    ref_bits = []
    for label, key in (
        ("Reference subject description", "subject_description"),
        ("Reference location name", "location_name"),
        ("Reference location description", "location_description"),
    ):
        value = str(reference_context.get(key) or "").strip()
        if value:
            ref_bits.append(f"{label}: {value}")
    reference_text = f"\n\nReference Builder context:\n{chr(10).join(ref_bits)}" if ref_bits else ""
    return (
        f"You are editing an existing {mode_label} image generation prompt.\n"
        "Make the smallest useful edit that satisfies the user's request.\n"
        "Preserve the subject, setting, identity, style, lighting, wardrobe, camera framing, mood, and continuity unless the user explicitly asks to change them.\n"
        "If a reference image is provided, use it only as visual reference for subject, setting, composition, color, and concrete visible details.\n"
        "When full scene context is disabled, use only the current prompt and user requested change.\n"
        "When full scene context is enabled, you may use the supplied scene context for a larger but still controlled rewrite.\n"
        f"{mode_rules}\n"
        "Do not add captions, text overlays, markdown, labels, bullet points, explanations, or commentary.\n"
        "Do not mention this edit request. Do not describe what changed.\n"
        "Return only one clean revised image prompt paragraph.\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"User requested change:\n{edit_request}\n\n"
        f"Reference image provided: {use_vision_reference}"
        f"{context_text}"
        f"{reference_text}"
    )
