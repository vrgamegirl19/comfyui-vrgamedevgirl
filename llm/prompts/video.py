"""Video prompts: text-to-video, image-to-video, ID-LoRA, and I2V motion notes."""


_T2V_INSTRUCTIONS = """Convert the user's concept prompt into a dynamic text-to-video prompt.

Use the user's prompt as the full scene foundation. Preserve the original subject, setting, outfit, mood, atmosphere, and scene identity. Infer only the missing video details needed to make the scene feel complete, including time of day, weather, lighting behavior, environmental movement, subject movement, camera movement, and performance energy. Do not add unrelated characters, new locations, major story changes, captions, text overlays, dialogue, or audio instructions.

Add fast, cinematic motion by giving the subject a clear action sequence, expressive facial expressions, strong gestures, and intentional camera movement. Keep the subject visible, centered, and clearly framed throughout. Add lighting only as natural scene behavior, such as flickering stage lights, passing sunlight, glowing streetlights, storm flashes, reflections, or shifting shadows, based on what best fits the user's prompt.

Output one polished paragraph using this structure:

The [Subject] in [setting/environment] during [time/weather]. The subject [dynamic performance action with expressive face, body movement, and strong gestures]. Their clothing/hair [reacts to movement, wind, or performance energy]. The lighting [changes or reacts naturally within the scene]. The camera [Camera Motion] while maintaining [subject visibility and framing]. The environment [reacts dynamically].

Each word in brackets should be chosen based on the user input and what best fits the scene.

Rules:
- This is text-to-video
- Never output square brackets or placeholder text. Replace every bracketed example with concrete scene details.
- Do not force singing. Follow user notes for singing, speaking, narration, instrumental, b-roll, or no-lip-sync behavior.
- Do not invent lyric/dialogue text. If exact lyric or dialogue text is provided in the user notes, use only that exact text when the performance direction calls for singing, speaking, or lip-sync. If no exact text is provided, describe the visible performance without inventing words.
- Do not add audio, dialogue, captions, text overlays, unrelated characters, new locations, major story changes, color grading, camera photo style, or static image-quality descriptions.
- Keep it vivid, fast, cinematic, dynamic, and video-ready
- Use one location inferred by the user's concept prompt. If one is not listed use one from the location list.
- Must use user input to help create the prompt
- User notes take priority. If user notes ask for no singing, silent b-roll, instrumental motion, no lip movement, or non-performance action, follow the user notes instead.
- Do not mention source prompts, notes, lyrics, segments, JSON, or instructions.
- Do not include markdown, labels, quotes, or explanations."""


_ID_LORA_INSTRUCTIONS = """Create one short-film ID-LoRA prompt for LTX 2.3 ID-LoRA.

Use the user's scene concept, story context, mapped character/location notes, and motion notes to write a compact in-context video script. The workflow receives a reference image and a reference voice sample separately, so the prompt should describe the visible person, action, camera, and spoken vocal content without mentioning files, inputs, samples, LoRAs, or cloning.

Output exactly three labeled sections:

[VISUAL]: One cinematic paragraph describing the visible subject identity, setting, action, facial expression, camera movement, lighting, and environment motion.
[SPEECH]: One or two short spoken lines for the character to say, or a concise narration line if the scene calls for narration. This section is required. If user notes include an exact required [SPEECH] line, copy that line into this section. If no words are provided, write natural short dialogue that fits the scene.
[SOUNDS]: Brief non-speech sound cues such as room tone, footsteps, ambience, music bed, or silence.

Rules:
- Keep the whole prompt short enough for a single 3-8 second shot.
- Preserve the subject, location, mood, and story intent from the user input.
- Do not add unrelated characters, captions, subtitles, text overlays, credits, or UI text.
- Do not mention reference images, voice samples, cloning, ID-LoRA, LoRA files, workflow nodes, source prompts, segments, JSON, or instructions.
- Do not output markdown fences, bullets, explanations, or extra labels.
- Never omit [SPEECH] or [SOUNDS].
- The final answer must contain only [VISUAL], [SPEECH], and [SOUNDS]."""


_I2V_INSTRUCTIONS = """Convert the user's image reference, text-to-image prompt, and motion notes into a dynamic image-to-video prompt.

Use the image reference and text-to-image prompt only as first-frame visual inventory: subject identity, setting, outfit, props, visible mood, atmosphere, composition, and scene identity. Do not use the image prompt to decide body action, camera motion, performance energy, lyric action, story action, or animation pacing.

Use the user motion notes, camera notes, performance direction, facial direction, lyric context, and scene story beat to decide animation, body action, camera movement, and performance energy. Give the subject a clear motivated action, expressive facial performance, body movement, strong gestures, and intentional camera movement when those notes call for it. Keep the subject visible and framed throughout.

Output one polished paragraph using this structure:

The [Subject] in [setting/environment] during [time/weather]. The subject [dynamic performance action]. Their clothing/hair [reacts to movement]. The camera [Camera Motion] while maintaining [subject visibility]. The environment [reacts dynamically].

Each word in brackets should be chosen based on user input that would best fit the scene.

Do not invent lyric/dialogue text. If exact lyric or dialogue text is provided in the user notes, use only that exact text when the performance direction calls for singing, speaking, or lip-sync. If no exact text is provided, describe the visible performance without inventing words.

Do not add audio, dialogue, captions, text overlays, unrelated characters, new locations, major story changes, color style, lighting style, or image-quality descriptions. Keep it vivid, fast, cinematic, dynamic, and video-ready.

User input always takes priority over the text-to-image prompt when the user asks for specific camera motion, character movement, performance direction, or scene changes.

User Input, must follow:"""


_I2V_MOTION_NOTES_INSTRUCTIONS = r"""You are an image-to-video motion note writer.

INPUTS
You will receive:
1. CONCEPT_PROMPT_JSON: one visual concept per scene.
2. STORY: the overall story arc.
3. THEME_STYLE: visual style, mood, genre, world, and atmosphere.
4. SUBJECT: the main subject details, for character movement and performance only.

TASK
Create one short image-to-video motion note for each concept prompt.
These notes will be placed into per-scene I2V motion notes in the video builder.

RULES
- Write camera motion, character/performance motion, environmental motion, and mood pacing.
- Use the concept prompt as the main source for each scene.
- Use SUBJECT only for broad performance or body motion when useful.
- Do not rewrite the image prompt.
- Do not mention text, captions, lyrics, prompts, JSON, source images, or reference images.
- Keep each note practical for image-to-video generation.
- Keep each value one sentence, under 45 words.
- Avoid impossible object transformations unless the concept already implies surreal motion.
- If a scene is quiet or instrumental, use subtle camera/environment movement.

OUTPUT KEYS
Return one key for every input prompt.
Use keys named "Motion1", "Motion2", "Motion3", etc.
Never use Prompt keys.
Never skip, merge, split, or reorder notes.

OUTPUT
Return valid JSON only.
No markdown.
No explanation.
Use double quotes.
No trailing commas.
No line breaks inside string values.

FORMAT
{
  "Motion1": "Slow dolly toward the subject as background light drifts and small environmental details move gently.",
  "Motion2": "Wide lateral camera drift through the setting with subtle character movement and atmospheric motion."
}"""


def _video_prompt_enhancement_instructions(payload):
    mode_label = str(payload.get("mode_label") or "I2V").strip().upper()
    draft_prompt = str(payload.get("draft_prompt") or "").strip()
    scene_prompt = str(payload.get("t2i_prompt") or payload.get("scene_prompt") or "").strip()
    user_notes = str(payload.get("user_notes") or "").strip()
    lyric_text = str(payload.get("lyric_text") or "").strip()
    performance_mode = str(
        payload.get("performance_mode")
        or payload.get("performanceMode")
        or payload.get("video_type")
        or payload.get("videoType")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if performance_mode in {"speaking", "short_film", "dialogue", "dialog"}:
        performance_mode = "speaking"
    elif performance_mode in {"no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"}:
        performance_mode = "no_lip_sync"
    else:
        performance_mode = "singing"
    singers_raw = payload.get("singers") or []
    if isinstance(singers_raw, str):
        singers = [item.strip() for item in singers_raw.split(",") if item.strip()]
    elif isinstance(singers_raw, list):
        singers = [str(item or "").strip() for item in singers_raw if str(item or "").strip()]
    else:
        singers = []
    no_vocal = bool(payload.get("no_vocal") or payload.get("instrumental") or payload.get("broll") or performance_mode == "no_lip_sync")
    no_character_present = bool(payload.get("no_character_present") or payload.get("no_subject") or payload.get("no_visible_subject"))
    has_multiple_singers = len(singers) > 1
    has_one_singer = len(singers) == 1
    singer_text = ", ".join(singers)

    if no_character_present:
        template = (
            "Use this final prompt shape:\n"
            "The [main visual focus: location, prop, object, environment, architecture, weather, light, or atmosphere] in [location/environment] during [time/weather/lighting]. "
            "[Main visual focus] [clear visible action or environmental motion]. [Objects, props, fabric, water, plants, particles, or lighting] move naturally if visible. "
            "The camera [camera motion] while keeping the location or main object clearly framed and visible. "
            "The environment [visible motion/reaction].\n\n"
            "No-character rule: do not include, mention, imply, show, or describe any main character, singer, performer, person, mapped subject, or character reference in the final prompt."
        )
    elif no_vocal:
        template = (
            "Use this final prompt shape:\n"
            "The [subject or main visual focus] in [location/environment] during [time/weather/lighting]. "
            "[Subject or main visual focus] [visible action only]. [Clothing, hair, objects, or props] move naturally if visible. "
            "The camera [camera motion] while keeping the main visual focus clearly framed and visible. "
            "The environment [visible motion/reaction].\n\n"
            "Visual-only rule: do not quote or mention the lyric line. Do not mention singing, speaking, saying, dialogue, lip-syncing, vocals, lyric, mouth movement, instrumental status, or no-vocal status in the final prompt."
        )
    elif performance_mode == "speaking" and has_multiple_singers:
        template = (
            "Use this final prompt shape:\n"
            f"The [speakers: {singer_text}] in [location/environment] during [time/weather/lighting]. "
            f"[Speakers] say the dialogue line \"{lyric_text}\" with expressive facial emotion, intentional gestures, and natural body movement. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping all speakers clearly framed and visible. "
            "The environment [visible motion/reaction].\n\n"
            "Speaking-mode rule: use says/say wording only. Do not use singing, rapping, vocals, lyric, lip-sync, or music-performance wording."
        )
    elif performance_mode == "speaking" and has_one_singer:
        template = (
            "Use this final prompt shape:\n"
            f"The [speaker: {singer_text}] in [location/environment] during [time/weather/lighting]. "
            f"[Speaker] says the dialogue line \"{lyric_text}\" with expressive facial emotion, intentional gestures, and natural body movement. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping the speaker clearly framed and visible. "
            "The environment [visible motion/reaction].\n\n"
            "Speaking-mode rule: use says wording only. Do not use singing, rapping, vocals, lyric, lip-sync, or music-performance wording."
        )
    elif performance_mode == "speaking" and lyric_text:
        template = (
            "Use this final prompt shape:\n"
            "The [visible speaker] in [location/environment] during [time/weather/lighting]. "
            f"The visible speaker says the dialogue line \"{lyric_text}\" with expressive facial emotion, intentional gestures, and natural body movement. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping the speaker clearly framed and visible. "
            "The environment [visible motion/reaction].\n\n"
            "Speaking-mode rule: use says wording only. Do not use singing, rapping, vocals, lyric, lip-sync, or music-performance wording."
        )
    elif has_multiple_singers:
        template = (
            "Use this final prompt shape:\n"
            f"The [subjects: {singer_text}] in [location/environment] during [time/weather/lighting]. "
            f"[Subjects] are singing with passion, clearly moving their mouths in sync with the lyric \"{lyric_text}\", "
            "with expressive facial emotion, head movement, and strong visible performance gestures. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping all singers clearly framed and visible. "
            "The environment [visible motion/reaction]."
        )
    elif has_one_singer:
        template = (
            "Use this final prompt shape:\n"
            f"The [subject: {singer_text}] in [location/environment] during [time/weather/lighting]. "
            f"[Subject] is singing with passion, clearly moving their mouth in sync with the lyric \"{lyric_text}\", "
            "with expressive facial emotion, head movement, and strong visible performance gestures. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping the subject clearly framed and visible. "
            "The environment [visible motion/reaction]."
        )
    elif lyric_text:
        template = (
            "Use this final prompt shape:\n"
            f"The [visible subject] in [location/environment] during [time/weather/lighting]. "
            f"The visible subject is singing with passion, clearly moving their mouth in sync with the lyric \"{lyric_text}\", "
            "with expressive facial emotion, head movement, and strong visible performance gestures. "
            "[Clothing/hair] moves naturally with their body motion. "
            "The camera [camera motion] while keeping the subject clearly framed and visible. "
            "The environment [visible motion/reaction]."
        )
    else:
        template = (
            "Use this final prompt shape:\n"
            "The [subject or main visual focus] in [location/environment] during [time/weather/lighting]. "
            "[Subject or main visual focus] [clear visible action]. [Clothing, hair, objects, or props] move naturally if visible. "
            "The camera [camera motion] while keeping the main visual focus clearly framed and visible. "
            "The environment [visible motion/reaction]."
        )

    return (
        f"Rewrite this draft {mode_label} video prompt into one stronger LTX-ready paragraph.\n\n"
        "Use the requested sentence shape, but replace every bracketed phrase with concrete details. Never output brackets or placeholder words.\n"
        "Preserve the subject, location, outfit, scene identity, and any user-requested camera/motion notes from the inputs.\n"
        "If the no-character rule is present, ignore any subject/character/singer from the draft and preserve only location, props, objects, atmosphere, and camera/motion notes.\n"
        "Only describe visible physical actions or visible scene motion. Do not mention invisible sensations, internal thoughts, symbolism, breath, heartbeat, sound-only details, or audio instructions.\n"
        "Do not use the word lip-sync. Use singing language only in singing mode. Use saying/dialogue language only in speaking mode. Use neither in no-lip-sync mode.\n"
        "Do not add microphones unless they are visible or explicitly requested.\n"
        "Do not add captions, text overlays, dialogue explanations, unrelated characters, new locations, markdown, labels, or explanations.\n"
        "Output one polished paragraph only.\n\n"
        f"{template}\n\n"
        f"Video Type / performance mode:\n{performance_mode}\n\n"
        f"Draft prompt:\n{draft_prompt}\n\n"
        f"Scene/T2I context:\n{scene_prompt or '(none)'}\n\n"
        f"User/video notes:\n{user_notes or '(none)'}\n\n"
        f"Lyric text:\n{lyric_text or '(none)'}\n\n"
        f"Singer(s):\n{singer_text or '(none)'}"
    )


def _video_prompt_edit_instructions(payload):
    current_prompt = str(payload.get("current_prompt") or "").strip()
    edit_request = str(payload.get("edit_request") or "").strip()
    mode_label = str(payload.get("mode_label") or "video").strip() or "video"
    performance_mode = str(
        payload.get("performance_mode")
        or payload.get("performanceMode")
        or payload.get("video_type")
        or payload.get("videoType")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if performance_mode in {"speaking", "short_film", "dialogue", "dialog"}:
        performance_mode = "speaking"
    elif performance_mode in {"no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"}:
        performance_mode = "no_lip_sync"
    else:
        performance_mode = "singing"
    if performance_mode == "speaking":
        performance_rule = (
            "Video Type is speaking / short film. Preserve speaking/dialogue behavior. "
            "Use says/say wording for dialogue and do not introduce singing, rapping, vocals, lyric, lip-sync, or music-performance wording."
        )
    elif performance_mode == "no_lip_sync":
        performance_rule = (
            "Video Type is no lip sync / visual-only. Do not quote lyric text. "
            "Do not introduce saying, speaking, dialogue, singing, rapping, vocals, lyric, lip-sync, mouth movement, or no-vocal status. "
            "Keep the edit focused on visible action, camera motion, environment, mood, and physical movement."
        )
    else:
        performance_rule = (
            "Video Type is singing / music video. Preserve singing behavior only when the current prompt or edit request calls for a vocal performance."
        )
    use_full_scene_context = bool(payload.get("use_full_scene_context"))
    use_vision_reference = bool(payload.get("use_vision_reference"))
    scene_context = payload.get("scene_context") if isinstance(payload.get("scene_context"), dict) else {}
    if not current_prompt:
        raise ValueError("Current video prompt is empty.")
    if not edit_request:
        raise ValueError("Edit request is empty.")
    context_text = ""
    if use_full_scene_context:
        singers = scene_context.get("singers")
        if isinstance(singers, list):
            singers_text = ", ".join(str(item or "").strip() for item in singers if str(item or "").strip())
        else:
            singers_text = str(singers or "").strip()
        context_text = (
            "\n\nFull scene context is enabled. Use this context only when it helps satisfy the requested edit. "
            "Do not rewrite unrelated parts just because context is present.\n"
            f"Scene label: {str(scene_context.get('label') or '').strip() or '(none)'}\n"
            f"Image/concept prompt: {str(scene_context.get('image_prompt') or '').strip() or '(none)'}\n"
            f"Scene notes: {str(scene_context.get('scene_notes') or '').strip() or '(none)'}\n"
            f"Director note: {str(scene_context.get('director_note') or '').strip() or '(none)'}\n"
            f"Motion notes: {str(scene_context.get('motion_notes') or '').strip() or '(none)'}\n"
            f"Lyric section: {str(scene_context.get('lyric_section') or '').strip() or '(none)'}\n"
            f"Lyric text: {str(scene_context.get('lyric_text') or '').strip() or '(none)'}\n"
            f"Performance mode: {str(scene_context.get('performance_mode') or performance_mode).strip() or performance_mode}\n"
            f"Singer(s): {singers_text or '(none)'}\n"
            f"Subject context: {str(scene_context.get('subject_context') or '').strip() or '(none)'}\n"
            f"Location context: {str(scene_context.get('location_context') or '').strip() or '(none)'}\n"
            f"No character present: {bool(scene_context.get('no_character_present'))}"
        )
    return (
        f"You are editing an existing {mode_label} generation prompt.\n"
        "Make the smallest useful edit that satisfies the user's request.\n"
        "Preserve the subject, setting, style, lighting, wardrobe, identity, mood, and continuity unless the user explicitly asks to change them.\n"
        "If a starting image is provided, use it only as visual reference for subject, setting, framing, and visible motion/camera feasibility.\n"
        "If the user asks for a camera or motion change, replace conflicting camera/motion language instead of stacking contradictions.\n"
        "When full scene context is disabled, use only the current prompt and user requested change.\n"
        "When full scene context is enabled, you may use the supplied scene context for a larger but still controlled rewrite.\n"
        f"{performance_rule}\n"
        "Do not add new characters, new locations, unrelated actions, captions, text overlays, markdown, labels, or explanations.\n"
        "Do not mention this edit request. Do not describe what changed.\n"
        "Return only one clean revised prompt paragraph.\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"User requested change:\n{edit_request}\n\n"
        f"Starting image provided: {use_vision_reference}"
        f"{context_text}"
    )
