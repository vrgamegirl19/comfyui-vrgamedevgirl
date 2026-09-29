"""Storyboard Builder prompts: story arc, scene beats, dialogue scenes, and scene image/video prompts."""

import re

from ...storyboard.scene_helpers import _clean_scene_text


# ============================================================================
# Storyboard Builder
# Every LLM prompt used by the Storyboard Builder, in one place.
#
# Organized by pipeline stage. Each item's docstring says which Storyboard
# route/button calls it and what output it controls. Edit wording here; the
# files in VRGDG_Storyboard*.py only gather scene data and call these.
# ============================================================================

# ============================================================================
# Video prompt (T2V fallback + LTX/i2v contract pieces)
# Used by: storyboard.scene_prompts._build_storyboard_video_prompt
# Route: /vrgdg/storyboard/gemma_video_prompt ("Generate Video Prompt" / the
# "LM Studio Video All" batch button in Storyboard Builder)
# Controls: the wording of the scene's final video-generation prompt.
# ============================================================================

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


def _storyboard_video_prompt_writing_rules():
    """Used inside the i2v (image-reference) video-prompt `user_notes` block.
    Controls: the same anti-repetition / first-frame-vs-motion rules as the
    T2V fallback above, restated for the vision-model (i2v) prompt path.
    """
    return "\n".join([
        "Prompt writing rules:",
        "Use the image reference and text-to-image prompt only for visible first-frame details: subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition.",
        "Do not use the image prompt for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing.",
        "Use the motion/camera notes, performance direction, vocal direction, facial direction, and scene story beat to decide animation, body action, camera movement, and performance energy.",
        "Each sentence has one job and must add new information. Do not repeat the same mood, trait, motion, authority/defiance language, setting adjective, or descriptive phrase across the face, body, camera, environment, and atmosphere sentences.",
        "If an idea appears in the face sentence, do not repeat it in the body, camera, environment, or atmosphere sentence; use a different concrete visual detail instead.",
        "Do not duplicate adjacent words such as tall, tall or vast, vast.",
    ])


def _storyboard_video_vocal_contract(is_timed_lyric_scene, is_visible_singing_scene):
    """One line of the i2v LTX one-pass contract.
    Controls: whether the video prompt is told to depict visible singing/lip-sync,
    and whether it must respect a Whisper-timed lyric cue map instead of the
    full scene lyric.
    """
    if is_timed_lyric_scene:
        return (
            "This scene uses an authoritative Whisper-timed lyric cue map. Visible singing occurs only inside vocal cue intervals, "
            "using only the words assigned to that interval. Instrumental intervals have no visible singing or lip-sync. "
            "Never repeat or restart the complete scene lyric in each shot."
        )
    if is_visible_singing_scene:
        return (
            "This is a visible singing/lip-sync scene. The performer must visibly vocalize the supplied lyric in sync with the audio, "
            "using natural mouth, lip, cheek, and jaw movement. Never describe closed, still, sealed, relaxed-closed, or unmoving lips."
        )
    return (
        "This is not a visible singing/lip-sync scene. Do not say any subject sings, vocalizes, mouths words, or lip-syncs, "
        "and do not quote the lyric as performed dialogue."
    )


def _storyboard_video_pronoun_contract(pronouns):
    """One line of the i2v LTX one-pass contract.
    Controls: singular pronoun enforcement for single-subject scenes.
    """
    if pronouns:
        return (
            f"This scene contains exactly one mapped subject. Use {pronouns['they']}/{pronouns['them']}/{pronouns['their']} "
            "consistently for that person. Never use they, them, their, or plural agreement."
        )
    return "Use pronouns and singular/plural agreement that exactly match the mapped subject count."


def _storyboard_video_ltx_one_pass_contract(vocal_contract, pronoun_contract):
    """Used only for LTX (non-MiniMax H3) i2v scenes.
    Controls: cut-plan formatting, opening-shot phrasing, camera-repetition,
    anatomy phrasing, and off-frame-detail rules for the final LTX video prompt.
    """
    return (
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
    )


# ============================================================================
# Image prompt (T2I)
# Used by: storyboard.scene_prompts._build_storyboard_image_prompt
# Route: /vrgdg/storyboard/gemma_image_prompt ("Generate Image Prompt")
# Controls: the wording of the scene's still-frame (T2I) prompt.
# ============================================================================

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


_STORYBOARD_IMAGE_WORLD_STYLE_PRESETS = {
    "natural": "Use a naturalistic, believable visual world. Surreal details may appear only when required by the scene.",
    "surreal_subject": "Keep the environment broadly believable, but render the subject and its materials with unmistakable surreal invention.",
    "balanced_surreal": "Make both subject and environment visibly surreal while retaining enough spatial coherence to read as one designed cinematic world.",
    "full_surreal": "Construct the entire image as an unmistakably surreal world. Environment, architecture, ground, sky, vegetation, furniture, props, lighting, perspective, scale, gravity, subject, anatomy, clothing, and materials must all obey deliberate dream logic. Do not place a surreal subject inside an otherwise ordinary realistic location. Avoid generic cinematic realism, conventional architecture, naturalistic staging, and merely photoreal backgrounds.",
    "abstract": "Create a strongly abstract, nonliteral visual world using impossible space, symbolic forms, transformed materials, unconventional scale, and expressive color and light. Literal realism is not the goal.",
    "custom": "Follow the user's custom style direction as the primary visual-world contract. Apply it to every visible layer of the image, including the environment and background.",
}


def _storyboard_image_world_style_contract(image_world_style, image_custom_style_direction):
    """Appended to the T2I instructions for every scene.
    Controls: which of the "Image World Style" presets (natural / surreal /
    abstract / custom) governs the still-image prompt's visual world.
    """
    style_instruction = _STORYBOARD_IMAGE_WORLD_STYLE_PRESETS.get(image_world_style, _STORYBOARD_IMAGE_WORLD_STYLE_PRESETS["natural"])
    return (
        "\n\nGLOBAL IMAGE WORLD STYLE CONTRACT:\n"
        f"- {style_instruction}\n"
        + (f"- User's custom visual direction: {image_custom_style_direction}\n" if image_custom_style_direction else "")
        + "- Apply this contract to the complete frame, not only the main subject. Preserve required scene content and endpoint facts while expressing them through this style.\n"
        + "- Do not mention this contract, preset names, or workflow settings in the final prompt."
    )


def _storyboard_flf_endpoint_instruction(flf_image_target, target_state, transformation, carry_forward):
    """Appended to the T2I instructions only when generating a First/Last-Frame
    endpoint still image (not a normal scene image).
    Controls: whether the still image is written as the untouched opening
    state or as the transformed end state of a First/Last-Frame pair.
    """
    endpoint_context = (
        "- This is strictly the untouched opening condition before the scene action begins.\n"
        "- Do not include, foreshadow, partially reveal, or imply the later transformation, destination anatomy, destination objects, or completed action.\n"
        "- Ignore transformation, end-state, and carry-forward fields when writing this START image.\n"
        "- Lyrics and the general story beat may guide mood only; do not include any lyric/story action that is not already explicitly visible in the required opening state.\n"
        if flf_image_target == "start" else
        f"- Planned transformation context: {transformation or '[none]'}\n"
        f"- Carry-forward continuity: {carry_forward or '[none]'}\n"
    )
    return (
        "\n\nFIRST / LAST FRAME STILL-IMAGE RULES:\n"
        f"- You are writing the {flf_image_target.upper()} endpoint still image, not a video prompt.\n"
        f"- Required visible endpoint state: {target_state or '[use the scene card literally]'}\n"
        f"{endpoint_context}"
        "- Make the required endpoint state visually concrete in one frozen image.\n"
        "- Preserve mapped subject identity, wardrobe, environment, lighting, and established anatomy unless the required endpoint explicitly changes one of them.\n"
        "- Do not describe motion over time, a transition, morphing process, first/last frames, or workflow instructions in the final image prompt.\n"
        "- Output only the image prompt."
    )


# ============================================================================
# Story Brief
# Used by: storyboard.story_layer._build_story_layer_brief /
# _build_short_film_script_story_text
# Route: /vrgdg/storyboard/story_brief ("Generate Story Brief")
# Controls: the compact story brief shown in the Story Layer panel.
# ============================================================================

def _storyboard_script_story_instruction(purpose, story_idea, subjects_json, locations_json, script_text, compact_plan_json):
    """Used when an Authoritative Script Mapper import is active, for both the
    Story Brief and the Story Arc (the same locked-script text feeds both).
    `purpose` is "brief" (compact production brief) or "premise" (narrative arc).
    Controls: how the locked script's surrounding visual story is developed
    without ever altering the authoritative dialogue.
    """
    if purpose == "brief":
        task = (
            "Create a compact short-film production brief that Guided Film Automation can use to design visual scenes around the authoritative script. "
            "Use these headings exactly: Story premise:, Character dynamics:, Visual progression:, Continuity rules:, Scene-direction guidance:. "
            "Keep the complete response under 450 words."
        )
    else:
        task = (
            "Create one cohesive short-film premise and visual narrative arc from the authoritative script. Explain the dramatic situation, character goals, emotional progression, "
            "and how the film can develop visually across the supplied timed segments. Output plain prose under 500 words with no screenplay rewrite and no dialogue list."
        )
    return (
        "You are a short-film development director working from a locked screenplay.\n\n"
        f"{task}\n\n"
        "NON-NEGOTIABLE SCRIPT CONTRACT:\n"
        "- The supplied dialogue is authoritative and immutable. Never rewrite, paraphrase, shorten, extend, reorder, merge, or invent spoken words.\n"
        "- Do not add narration, voice-over, replacement dialogue, or extra speakers.\n"
        "- Develop only the visual story around the locked dialogue: actions, reactions, motivations, blocking, locations, props, shots, camera language, atmosphere, and continuity.\n"
        "- Use Reference Builder character names and descriptions as identity authority.\n"
        "- Use only supplied Reference Builder locations when locations are available.\n\n"
        f"Optional user story idea:\n{story_idea or '[none]'}\n\n"
        f"Reference Builder characters:\n{subjects_json}\n\n"
        f"Reference Builder locations:\n{locations_json}\n\n"
        f"Authoritative exact script:\n{script_text}\n\n"
        f"Authoritative timed segment plan:\n{compact_plan_json}"
    )


def _storyboard_story_brief_instruction(lyric_story_strength_guidance_text, user_story_arc, lyrics, compact_scenes_json):
    """Used when there is no Authoritative Script Mapper import.
    Controls: the compact story brief built from lyrics/story arc/scene map.
    """
    return (
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
        f"{lyric_story_strength_guidance_text}\n\n"
        "Include these compact headings exactly:\n"
        "Story premise:\n"
        "Emotional arc:\n"
        "Visual motifs:\n"
        "Scene guidance:\n\n"
        f"User story arc:\n{user_story_arc or '[none]'}\n\n"
        f"Full/pasted lyrics:\n{lyrics or '[not provided]'}\n\n"
        f"Scene lyric map:\n{compact_scenes_json}"
    )


# ============================================================================
# Story Arc
# Used by: storyboard.story_layer._build_story_layer_arc
# Route: /vrgdg/storyboard/story_arc ("Generate Story Arc")
# Controls: the per-lyric-section visual story arc shown in the Story Layer panel.
# ============================================================================

def _storyboard_story_arc_structure_instruction(required_section_labels):
    """Controls: whether the Story Arc must follow explicit lyric section
    headers exactly, or is free to invent its own song structure.
    """
    if required_section_labels:
        formatted_labels = "\n".join(f"- {label}" for label in required_section_labels)
        return (
            "STRUCTURAL RIGIDITY (MANDATORY):\n"
            "The song structure is fixed. Output exactly these section headings in this exact sequence. "
            "Do not merge, omit, rename, reorder, or add any headings:\n"
            f"{formatted_labels}\n"
            "Every heading represents an independent, continuous scene beat. "
            "Write exactly one cohesive visual paragraph under each heading."
        )
    return (
        "STRUCTURAL INFERENCE:\n"
        "The lyrics lack explicit structural markers. Establish a compact, standard song structure "
        "(e.g., Verse 1, Chorus 1, Verse 2, Chorus 2, Bridge, Chorus 3) dictated strictly by musical "
        "energy and narrative turning points. Do not invent gratuitous Intro or Outro sections unless "
        "explicitly supported by instrumental cues."
    )


def _storyboard_story_arc_motion_guidance(character_motion_speed):
    """Controls: character kinetics tiered by motion speed, calibrated to prevent
    diffusion artifacts (melting limbs, morphing objects) in MiniMax H3.
    """
    if character_motion_speed <= 2:
        return (
            "Character Kinetic Profile [Low / 1-2]:\n"
            "- Movement: Micro-kinetics only. Avoid rapid extremity travel.\n"
            "- Actions: Subtle weight transfers, deliberate head turns, slow postural settling, "
            "gradual eye-line shifts, breathing motion, or fingertips grazing a static surface.\n"
            "- Diffusion Guard: Keep limbs anchored; avoid rapid hand gestures or complex prop manipulations."
        )
    if character_motion_speed <= 5:
        return (
            "Character Kinetic Profile [Moderate / 3-5]:\n"
            "- Movement: Single-vector, controlled linear action.\n"
            "- Actions: Paced steady stride in one direction, lifting or placing a single grounded prop, "
            "deliberate pivot/turn of the torso, or a measured seated-to-standing transition.\n"
            "- Diffusion Guard: Every movement must follow a single physical trajectory without abrupt reversals."
        )
    if character_motion_speed <= 8:
        return (
            "Character Kinetic Profile [Active / 6-8]:\n"
            "- Movement: Dynamic environmental traversal.\n"
            "- Actions: Purposeful transit across depth planes (moving directly toward or away from the camera), "
            "hurried strides, pushing through physical barriers (doors, curtains, weather), or rhythmic performance choreography.\n"
            "- Diffusion Guard: Keep the subject's center of gravity readable; avoid multi-limb acrobatics."
        )
    return (
        "Character Kinetic Profile [High Kinetic / 9-10]:\n"
        "- Movement: High-momentum, full-body kinetic staging.\n"
        "- Actions: Full sprint, athletic leaps, rapid choreography with sweeping arm sweeps, forceful interactions with "
        "the set (shoving, slamming, breaking into a run), or high-velocity rotational turns.\n"
        "- Diffusion Guard: Stage the momentum along a sustained path so motion blur remains coherent across frames."
    )


def _storyboard_story_arc_instruction(
    *,
    required_section_labels,
    section_word_limit,
    story_arc_seed,
    camera_flow,
    camera_motion_speed,
    character_motion_speed,
    performance_style,
    facial_performance,
    lyric_story_strength_guidance_text,
    story_idea,
    previous_story_arc,
    style_theme,
    lyrics_source,
    lyrics,
    compact_scenes_json,
    subjects_json,
    locations_json,
):
    """The primary Story Arc generation prompt.
    Directs the LLM as a cinematographic continuity supervisor optimized for MiniMax H3.
    """
    structure_instruction = _storyboard_story_arc_structure_instruction(required_section_labels)
    motion_guidance = _storyboard_story_arc_motion_guidance(character_motion_speed)

    return (
        "You are an expert cinematic director and visual continuity supervisor designing a scene-by-scene "
        "music video storyboard optimized for video diffusion generation (MiniMax H3).\n\n"
        "Your objective: Translate song lyrics, narrative themes, and performer profiles into a physically grounded, "
        "visually continuous storyboard structured strictly by musical sections.\n\n"
        "OUTPUT FORMAT REQUIREMENTS:\n"
        "- Output ONLY the final story arc sections.\n"
        "- Do not include markdown code fences, JSON blocks, conversational intros, or closing summaries.\n"
        "- Format every section as its exact heading line ending in a colon, followed by exactly one prose paragraph:\n"
        "  [Section Name]:\n"
        "  [Continuous visual prose block]\n\n"
        f"{structure_instruction}\n\n"
        "DIRECTORIAL & DIFFUSION CONTINUITY RULES:\n"
        f"1. Section Length: Write each section's visual summary between {max(20, int(section_word_limit * 0.8))} and {section_word_limit} words. Never exceed {section_word_limit} words.\n"
        "2. Grounded Physical Action (Anti-Statue / Anti-Morph):\n"
        "   - Never leave principal subjects in static poses, generic staring, or passive poses.\n"
        "   - Anchor every section around ONE dominant, continuous physical action.\n"
        "   - Avoid compound temporal sequences (e.g., do NOT write 'she enters, sits down, drinks coffee, then runs out'). "
        "Stage a single unbroken continuous physical beat suitable for short AI video clips.\n"
        "3. Spatial Blocking & Depth Planes:\n"
        "   - Explicitly establish depth: specify what occupies the foreground, midground, and background.\n"
        "   - Specify practical lighting sources (neon tubes, shafts of sunlight, rim lighting, lens flares) and atmospheric textures.\n"
        "4. Somatic Emotion (No Abstract Poetry):\n"
        "   - Never write internal emotional states (e.g., 'he remembers his grief', 'she feels isolated').\n"
        "   - Render emotional subtext purely through physical cues: jaw clenching, rapid shallow breathing, averted gaze, rigid posture, or trembling hands.\n"
        "5. Environmental & Extra Discipline:\n"
        "   - Confine all scenes strictly to the environments described in 'Allowed Locations'. Do not invent unlisted locations or architectural transitions.\n"
        "   - Extras must only appear in scenes where explicitly mapped, performing their designated support roles (direct, dancing_with, alongside, background). Extras never sing or speak.\n"
        "6. Metadata Stripping:\n"
        "   - Ignore performance tags ([Whispered], [Explosive], [Guitar Solo]) as potential structural headings. Incorporate their energy directly into the scene's lighting, tempo, and action.\n\n"
        f"Creative Seed: {story_arc_seed or '[none]'}\n"
        "If this seed differs from previous runs, establish distinct blocking, alternative camera elevations, and different prop interactions while maintaining identical world rules.\n\n"
        "DIRECTORIAL PARAMETERS:\n"
        f"- Camera Flow: {camera_flow or '[Smooth Cinematic Panning]'}\n"
        f"- Camera Speed: {camera_motion_speed}/10\n"
        f"- Character Kinetic Speed: {character_motion_speed}/10\n"
        f"- Performance Style: {performance_style or '[Method Dramatic]'}\n"
        f"- Facial Intensity: {facial_performance or '[Subtle Realism]'}\n\n"
        f"{motion_guidance}\n\n"
        f"{lyric_story_strength_guidance_text}\n\n"
        f"Story Premise:\n{story_idea or '[Derive organic premise directly from lyrics]'}\n\n"
        f"Negative Baseline (Do Not Replicate Previous Run):\n{previous_story_arc or '[none]'}\n\n"
        f"Visual Theme & Palette:\n{style_theme or '[Cinematic Photorealism, 35mm film stock, high contrast]'}\n\n"
        f"Character Profiles:\n{subjects_json}\n\n"
        f"Allowed Locations:\n{locations_json}\n\n"
        f"Authoritative Lyrics ({lyrics_source}):\n{lyrics or '[not provided]'}\n\n"
        f"Scene Alignment Map:\n{compact_scenes_json}"
    )


def _storyboard_story_arc_format_retry_instruction(required_section_labels, original_instruction):
    """Sent when the initial response fails exact heading validation.
    Enforces a strict heading skeleton and forbids markdown/conversational output.
    """
    exact_format = "\n\n".join(f"{label}:\n[Continuous visual-story paragraph]" for label in required_section_labels)
    return (
        "CRITICAL FORMAT VALIDATION FAILURE:\n"
        "Your previous response violated the structural section heading contract.\n\n"
        "RULES FOR THIS REGENERATION:\n"
        f"1. Your response must begin immediately on line 1 with: '{required_section_labels[0]}:'\n"
        "2. Include EVERY section heading listed below exactly once, in this exact sequence, followed by a colon.\n"
        "3. Provide exactly ONE descriptive prose block per section.\n"
        "4. Absolutely no markdown lists, code fences, introductory sentences, or summary commentary.\n\n"
        f"REQUIRED OUTPUT SKELETON:\n{exact_format}\n\n"
        f"ORIGINAL CONTEXT & PARAMETERS:\n{original_instruction}"
    )


def _storyboard_story_arc_schema(required_section_labels):
    """JSON schema for strict structured-output validation.
    Guarantees that every section label is an object property with no hallucinated extras.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            label: {
                "type": "string",
                "description": f"One continuous, physically grounded visual paragraph for {label}."
            }
            for label in required_section_labels
        },
        "required": list(required_section_labels),
        "additionalProperties": False,
    }


def _storyboard_story_arc_json_retry_instruction(original_instruction):
    """Sent with _storyboard_story_arc_schema when structured output / JSON mode is available."""
    return (
        f"{original_instruction}\n\n"
        "OUTPUT FORMAT ENFORCEMENT:\n"
        "Return a raw, valid JSON object matching the provided schema. "
        "Keys must be the exact section headings in their prescribed order. "
        "Values must be single, continuous cinematic prose paragraphs describing the scene's blocking and action. "
        "Do not wrap in markdown ```json code fences. Do not repeat the heading name inside the value string."
    )


# ============================================================================
# Scene Story Beat
# Used by: storyboard.story_layer._build_story_layer_scene_beat
# Route: /vrgdg/storyboard/scene_story_beat ("Generate Scene Story Beat")
# Controls: the per-scene narrative beat (and, in FLF mode, the FLF endpoint
# fields) shown in the scene card.
# ============================================================================

def _storyboard_scene_beat_output_rules(flf_mode, beat_word_limit):
    """Controls: plain-paragraph output vs. the 5-field FLF endpoint JSON shape."""
    if flf_mode:
        return (
            "Return valid JSON only with exactly these string keys: story_beat, flf_start_state, flf_transformation, flf_end_state, flf_carry_forward.\n"
            f"The story_beat is a concise compatibility summary under {beat_word_limit} words.\n"
            "flf_start_state describes the concrete visible opening image. If Previous FLF end state is provided, copy it exactly as flf_start_state; do not reinterpret or redesign it.\n"
            "flf_transformation describes one continuous, progressive visual change that expresses the CURRENT lyric.\n"
            "flf_end_state describes the concrete visible destination image reached by the end of the CURRENT lyric.\n"
            "flf_carry_forward records the subject, anatomy, wardrobe, props, setting, lighting, and transformation state that the next scene must inherit.\n"
            "The current lyric is authoritative. Previous and next lyrics provide continuity only and must not replace or steal this scene's action.\n"
            "Do not include Markdown fences or any text outside the JSON object."
        )
    return f"Output one short paragraph only, no label, no bullets.\nKeep it under {beat_word_limit} words."


def _storyboard_scene_beat_instruction(
    *,
    flf_mode,
    beat_word_limit,
    lyric_story_strength_guidance_text,
    user_story_arc,
    song_story_brief,
    previous_beat,
    previous_lyrics,
    previous_end_state,
    previous_carry_forward,
    current_lyrics,
    next_lyrics,
    scene_defaults_json,
    performance_assignment_json,
    extra_subjects_json,
    scene_json,
):
    """The main Scene Story Beat prompt.
    Controls: keeping the beat purely visual (no audio/vocal language),
    respecting scene defaults, mapped extras, and the mapped location.
    """
    output_rules = _storyboard_scene_beat_output_rules(flf_mode, beat_word_limit)
    return (
        "You are a music video scene-story planner.\n"
        "Create one concise scene story beat that tells the video prompt writer what this scene contributes to the larger music-video story.\n\n"
        "Rules:\n"
        "- Use the Song Story Brief and User Story Arc as continuity anchors.\n"
        "- Use the selected scene lyrics, lyric section, subject details, location details, vocal status, and no-character flag.\n"
        "- This request creates a visual narrative Scene Story Beat only. Do not mention singing, lyrics, vocals, rapping, lip-sync, music, instrumental sections, dialogue delivery, or any other audio/performance metadata. Do not describe whether a subject is silent or vocal. Show the story through visible action, posture, expression, blocking, props, environment, and emotional stakes.\n"
        "- Performer and vocal assignments are downstream video-prompt metadata, not content for this beat; never copy those assignments into the story_beat.\n"
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
        f"{lyric_story_strength_guidance_text}\n\n"
        f"User Story Arc:\n{user_story_arc or '[none]'}\n\n"
        f"Song Story Brief:\n{song_story_brief or '[none]'}\n\n"
        f"Previous scene beat:\n{previous_beat or '[none]'}\n\n"
        f"Previous scene lyric text (continuity only):\n{previous_lyrics or '[none]'}\n\n"
        f"Previous FLF end state (required opening state when present):\n{previous_end_state or '[none — this is the first scene]'}\n\n"
        f"Previous FLF carry-forward constraints:\n{previous_carry_forward or '[none]'}\n\n"
        f"CURRENT scene lyric text (main authority):\n{current_lyrics or '[none]'}\n\n"
        f"Next scene lyric text:\n{next_lyrics or '[none]'}\n\n"
        f"Scene defaults and motion direction (authoritative when supplied):\n{scene_defaults_json}\n\n"
        "Performance assignment metadata (do not mention this in the story beat):\n"
        f"{performance_assignment_json}\n\n"
        f"Mapped extras and exact scene roles:\n{extra_subjects_json}\n\n"
        "Selected scene JSON:\n"
        + scene_json
        + "\n\nFINAL SCENE-BEAT OVERRIDE — FOLLOW THIS LAST:\n"
        + "Return only visual story information: setting, visible actions, blocking, props, facial emotion, atmosphere, symbolism, and continuity. Exclude all audio, lyric, vocal, singing, lip-sync, and performance-delivery language."
    )


def _storyboard_flf_endpoint_schema():
    """JSON schema for the FLF-mode Scene Story Beat reply, matching the five keys the FLF output rules ask for."""
    keys = ("story_beat", "flf_start_state", "flf_transformation", "flf_end_state", "flf_carry_forward")
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": list(keys),
        "additionalProperties": False,
    }


def _storyboard_flf_endpoint_repair_instruction(malformed_text, scene_json):
    """Sent only when the Scene Story Beat's FLF-mode JSON response fails to parse.
    Controls: recovering valid 5-field FLF endpoint JSON from a malformed response.
    """
    return (
        "Repair the malformed FLF endpoint response below into valid JSON.\n"
        "Return JSON only: no thought text, prose, Markdown, or code fences.\n"
        "Use exactly these five string keys: story_beat, flf_start_state, flf_transformation, flf_end_state, flf_carry_forward.\n"
        "Preserve the original meaning and wording as closely as possible. Escape quotation marks inside strings and include every comma, colon, quote, and closing brace required by strict JSON.\n"
        "If a field was cut off or omitted, reconstruct it concisely from the other fields and selected scene context.\n\n"
        f"MALFORMED RESPONSE:\n{malformed_text}\n\n"
        f"SELECTED SCENE CONTEXT:\n{scene_json}"
    )


def _storyboard_scene_beat_audio_language_repair_instruction(original_text):
    """Sent only when the Scene Story Beat leaked singing/lyric/vocal wording.
    Controls: rewriting the beat as visual-only narrative guidance.
    """
    return (
        "Rewrite this scene story beat as visual narrative guidance only. Preserve its setting, visible actions, props, emotion, symbolism, and continuity. "
        "Remove every reference to singing, lyrics, vocals, rapping, lip-sync, music, instrumental sections, dialogue delivery, or audio/performance metadata. "
        "Do not replace those references with a statement that the subject is silent; simply describe the visible action. Output one concise paragraph only, with no label or bullets.\n\n"
        f"Original scene beat:\n{original_text}"
    )


def _storyboard_scene_beat_location_repair_instruction(location_context, drift_terms, original_text):
    """Sent only when the Scene Story Beat drifted onto an unmapped location
    (e.g. an invented warehouse/corridor not present in location_ref).
    Controls: rewriting the beat to use only the mapped location.
    """
    return (
        "Rewrite the scene story beat so it obeys the mapped location.\n\n"
        "Hard rules:\n"
        "- Keep the same emotional purpose and subject energy.\n"
        "- Use only the mapped location as the physical setting.\n"
        "- Remove every incompatible place/object listed below.\n"
        "- Do not mention a warehouse, loading dock, industrial corridor, metal door, concrete hall, steel stairs, pipes, or massive window unless those details are explicitly in the mapped location.\n"
        "- Output one short paragraph only, under 80 words.\n\n"
        f"Mapped location:\n{location_context or '[none]'}\n\n"
        f"Incompatible leaked location terms:\n{', '.join(drift_terms)}\n\n"
        f"Original scene beat:\n{original_text}"
    )


def _storyboard_scene_beat_extra_mapping_repair_instruction(extra_subjects_json, original_text):
    """Sent only when the Scene Story Beat omitted one or more mapped extras.
    Controls: rewriting the beat so every mapped extra appears by exact name.
    """
    return (
        "Rewrite this music-video scene beat so it includes every mapped extra by exact name and gives each the assigned action/blocking role.\n\n"
        "Hard rules:\n"
        "- Preserve the original narrative purpose, mapped location, main subject action, and emotional progression.\n"
        "- Include every exact extra name from the mapping.\n"
        "- Apply each interaction role exactly. Group only extras sharing background or background_dancing.\n"
        "- Extras do not sing, speak, or receive speaker IDs.\n"
        "- Use identity details only when required to distinguish characters; do not copy wardrobe biographies.\n"
        "- Output one concise paragraph only, under 100 words, with no label or bullets.\n\n"
        f"Mapped extras:\n{extra_subjects_json}\n\n"
        f"Original scene beat:\n{original_text}"
    )


# ============================================================================
# Dialogue Scene Generation (ID-LoRA / MiniMax short film)
# Used by: storyboard.dialogue_scenes._build_id_lora_dialogue_scenes
# Route: /vrgdg/storyboard/id_lora_dialogue_scenes and
# /vrgdg/storyboard/minimax_dialogue_scenes ("Auto-Generate Scenes")
# Controls: the auto-generated batch of dialogue scene cards (label, dialogue,
# story beat, image prompt, camera/facial direction) for a whole short film.
# ============================================================================

def _id_lora_structured_image_prompt(item, subject_ref=None, location_ref=None):
    """Deterministic fallback still-image prompt used when the LLM's own
    `image_prompt` field for a generated dialogue scene is too short/generic.
    Controls: the NanoBanana/Krea-style still-frame prompt text for ID-LoRA
    and MiniMax auto-generated scenes.
    """
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


def _storyboard_dialogue_planner_instruction(
    *,
    is_minimax,
    has_authoritative_script,
    camera_flow,
    camera_motion_speed,
    character_motion_speed,
    scene_count,
    story_source,
    script_mapper_plan_json,
    story_layer_json,
    project_motion_settings_json,
    subjects_json,
    locations_json,
    compact_existing_json,
):
    """The main dialogue/scene-plan generation prompt, shared by the ID-LoRA
    and MiniMax Short Film planners.
    Controls: planner persona, whether dialogue is locked to an Authoritative
    Script Mapper import, dialogue-cue shape, camera/character motion rules,
    and the required JSON scene shape.
    """
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
        if has_authoritative_script else ""
    )
    output_dialogue_shape = (
        '      "dialogue_cues": [{"character_id": "exact character id", "speaker": "character name", "dialogue": "exact spoken words"}],\n'
        if is_minimax else
        '      "character_id": "exact id from available characters or empty",\n'
        '      "dialogue": "exact spoken line",\n'
    )
    return (
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
        f"SCRIPT MAPPER SEGMENT PLAN (authoritative when present):\n{script_mapper_plan_json}\n\n"
        f"Story layer:\n{story_layer_json}\n\n"
        f"Project motion settings:\n{project_motion_settings_json}\n\n"
        f"Available characters:\n{subjects_json}\n\n"
        f"Available locations:\n{locations_json}\n\n"
        f"Existing starter scenes:\n{compact_existing_json}"
    )


def _storyboard_dialogue_plan_schema(is_minimax):
    """JSON schema for the dialogue planner reply, matching the shape _storyboard_dialogue_planner_instruction asks for.

    Runners that support schemas constrain the reply to it; optional fields are still required keys and may be empty.
    """
    text_fields = (
        "label", "location_id", "story_beat", "visual_direction", "image_prompt", "shot_type", "camera_motion",
        "character_motion", "facial_performance", "delivery", "audio_direction", "continuity",
    )
    scene_properties = {name: {"type": "string"} for name in text_fields}
    if is_minimax:
        cue_properties = {name: {"type": "string"} for name in ("character_id", "speaker", "dialogue")}
        scene_properties["dialogue_cues"] = {
            "type": "array",
            "items": {"type": "object", "properties": cue_properties, "required": list(cue_properties), "additionalProperties": False},
        }
    else:
        scene_properties["character_id"] = {"type": "string"}
        scene_properties["dialogue"] = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "premise": {"type": "string"},
            "scenes": {
                "type": "array",
                "items": {"type": "object", "properties": scene_properties, "required": list(scene_properties), "additionalProperties": False},
            },
        },
        "required": ["title", "premise", "scenes"],
        "additionalProperties": False,
    }


def _storyboard_dialogue_json_repair_instruction(is_minimax, malformed_text):
    """Sent only when the dialogue planner's JSON response fails to parse.
    Controls: recovering valid scene-plan JSON from a malformed response.
    """
    return (
        f"Repair this malformed JSON for a {'MiniMax short-film' if is_minimax else 'ID-LoRA'} dialogue scene plan.\n"
        "Return only valid JSON. Do not add prose, markdown, code fences, comments, or trailing commas.\n"
        "Every property name must be enclosed in double quotes. Every string value must be enclosed in double quotes.\n"
        "Keep the same title, premise, and scenes when possible.\n\n"
        f"MALFORMED JSON:\n{malformed_text}"
    )
