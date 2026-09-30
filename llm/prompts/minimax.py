"""MiniMax H3 prompt creation: per-mode instructions, frame continuity and short-film contracts."""


# ============================================================================
# MiniMax H3 prompt creation
# Default LLM instructions for MiniMax H3 Builder prompt creation.
#
# The Builder resolves the current scene's mode, audio, performance, timing,
# reference, and continuity choices before calling the LLM. These presets define
# only the shared response shape and the one active generation task.
# ============================================================================

MINIMAX_H3_PROMPT_DIRECTOR_CORE = """Write the creative shot descriptions for the active MiniMax H3 scene.

The Scene concept already contains the Builder's resolved mode, performance, audio, timing, reference, and continuity rules. Follow only those active rules; do not invent alternate mode behavior.

Return plain valid JSON only:
{"shots":[{"description":"..."},{"description":"..."}]}

- Return exactly the requested number of descriptions as complete cinematic prose.
- Use only the keys `shots` and `description`.
- Do not add shot labels, timestamps, fixed prompt sections, markdown, analysis, notes, or text outside the JSON object. The Builder adds the final structure.

SHOT FORMAT
Write each shot like a music video director's shot note, 2 to 4 sentences, in this order:
1. Camera: the opening framing, one named move (direction and speed), and the ending framing.
2. Action: what each person in the cast physically does, by label, as one continuous motion.
3. Life detail: one or two small human movements inside the action, such as a breath catching, a glance away and back, a step and a stop.
4. Light and set: one line using only the mapped location's own light and objects.
Describe only what the camera sees. Show emotion only as visible movement, never with feeling words, and keep it strong: people should look alive and expressive. Keep everything realistic: natural physical movement, real-world light, believable weight and timing. Match the energy to the camera and character speed values in the Scene concept. Use the labels from the cast list, such as <Subject 1> (name) on first mention in each shot, never only a name, "he", or "she". Show only the people in the cast list: no other person, hand, arm, shadow, reflection, or crowd.
Character appearance comes from the reference images. Do not describe clothing, hair, accessories, jewelry, or facial features, except one brief mention when a garment or feature moves in the action. Every shot should read differently from the others in framing and movement.
"""


_MINIMAX_H3_TEXT_TO_VIDEO_MODE = """MODE: TEXT TO VIDEO
Use the resolved text context. Do not invent picture or video labels.
"""


_MINIMAX_H3_IMAGE_TO_VIDEO_MODE = """MODE: IMAGE TO VIDEO
Use the resolved starting-picture assignment as the visual anchor. Animate it naturally.
"""


_MINIMAX_H3_REFERENCE_TO_VIDEO_MODE = """MODE: REFERENCE TO VIDEO
Use the resolved <Subject N> and <Picture N> assignments directly in the cinematic shot descriptions.
"""


_MINIMAX_H3_IMAGE_REFERENCE_TO_VIDEO_MODE = """MODE: IMAGE + REFERENCE TO VIDEO
Follow the resolved start, end, and supporting-picture assignments directly in the cinematic shot descriptions.
"""


_MINIMAX_H3_VIDEO_TO_VIDEO_MODE = """MODE: VIDEO TO VIDEO
Use the resolved <Video N>, <Picture N>, and <Subject N> assignments directly in the cinematic shot descriptions.
"""


MINIMAX_H3_TEXT_TO_VIDEO_INSTRUCTIONS = (
    MINIMAX_H3_PROMPT_DIRECTOR_CORE + "\n" + _MINIMAX_H3_TEXT_TO_VIDEO_MODE
)

MINIMAX_H3_IMAGE_TO_VIDEO_INSTRUCTIONS = (
    MINIMAX_H3_PROMPT_DIRECTOR_CORE + "\n" + _MINIMAX_H3_IMAGE_TO_VIDEO_MODE
)

MINIMAX_H3_REFERENCE_TO_VIDEO_INSTRUCTIONS = (
    MINIMAX_H3_PROMPT_DIRECTOR_CORE + "\n" + _MINIMAX_H3_REFERENCE_TO_VIDEO_MODE
)

MINIMAX_H3_IMAGE_REFERENCE_TO_VIDEO_INSTRUCTIONS = (
    MINIMAX_H3_PROMPT_DIRECTOR_CORE + "\n" + _MINIMAX_H3_IMAGE_REFERENCE_TO_VIDEO_MODE
)

MINIMAX_H3_VIDEO_TO_VIDEO_INSTRUCTIONS = (
    MINIMAX_H3_PROMPT_DIRECTOR_CORE + "\n" + _MINIMAX_H3_VIDEO_TO_VIDEO_MODE
)


MINIMAX_H3_FRAME_CONTINUITY_INSTRUCTIONS = MINIMAX_H3_PROMPT_DIRECTOR_CORE + """

ACTIVE TASK: FRAME-TO-FRAME CONTINUITY
Use the previous render's attached final frame as the authoritative opening state. Follow the resolved FRAME-TO-FRAME CONTINUITY contract in the Scene concept and write one seamless uninterrupted continuation entirely through positive descriptions of desired visual action.
"""


MINIMAX_H3_INSTRUCTIONS_BY_MODE = {
    "text_to_video": MINIMAX_H3_TEXT_TO_VIDEO_INSTRUCTIONS,
    "image_to_video": MINIMAX_H3_IMAGE_TO_VIDEO_INSTRUCTIONS,
    "reference_to_video": MINIMAX_H3_REFERENCE_TO_VIDEO_INSTRUCTIONS,
    "image_reference_to_video": MINIMAX_H3_IMAGE_REFERENCE_TO_VIDEO_INSTRUCTIONS,
    "video_to_video": MINIMAX_H3_VIDEO_TO_VIDEO_INSTRUCTIONS,
}


MINIMAX_H3_INSTRUCTION_KEYS_BY_MODE = {
    "text_to_video": "minimax_h3_text_to_video",
    "image_to_video": "minimax_h3_image_to_video",
    "reference_to_video": "minimax_h3_reference_to_video",
    "image_reference_to_video": "minimax_h3_image_reference_to_video",
    "video_to_video": "minimax_h3_video_to_video",
}


MINIMAX_H3_SHORT_FILM_GUIDED_CONTRACT = """SHORT FILM — GUIDED FILM AUTOMATION
- Treat this as a dialogue-first narrative film scene, not a music-video performance unless the scene card explicitly says otherwise.
- Use the supplied premise, scene story beat, ordered speaker assignments, character references, location reference, acting direction, shot, camera direction, audio direction, and continuity to construct a clear dramatic beat.
- Preserve every supplied dialogue cue verbatim and in its exact speaker order. Never transfer words between characters, merge turns, repeat a line, or add dialogue.
- The ordered speaker assignments are authoritative for who speaks. Other visible characters remain silent unless they have their own cue, and should react naturally.
- The planner may fill only genuinely missing visual connective details needed to stage a coherent shot. It must not replace the user's plot, casting, dialogue, setting, or requested action.
- Use natural short-film vocabulary: blocking, eyelines, reaction shots, motivated camera movement, screen direction, physical business, subtext, and continuity where useful.
- Do not output ID-LoRA sections, LTX syntax, [VISUAL]/[SPEECH]/[SOUNDS] labels, workflow terminology, or model metadata.
"""


MINIMAX_H3_SHORT_FILM_CUSTOM_CONTRACT = """SHORT FILM — FULLY CUSTOM / MANUAL SOURCE CONTRACT
- Every populated scene-card field is locked, authoritative source material: ordered dialogue and speakers, story beat, action, performance, facial direction, shot/framing, camera movement, setting, character and location references, audio direction, continuity, and notes.
- A supplied `EDITING / CUT PLAN — MANDATORY` contract is also locked, authoritative source material. Applying its exact scheduled cuts and choosing the continuity-preserving coverage needed for those shots is required formatting, not an invented camera or story choice. A continuous-shot plan still prohibits every cut or transition.
- Your only creative task is to format those exact instructions into the required MiniMax H3 prompt structure and timestamp them across the exact supplied duration.
- Do not invent, rewrite, polish, paraphrase, reorder, merge, omit, replace, or contradict any supplied dialogue, speaker, action, story beat, camera choice, setting, sound, or continuity detail.
- Never add dialogue, narration, lyrics, a new speaker, a new character, a new action, a new plot beat, a new location, or an unrequested camera move.
- If the user leaves a field blank, leave that decision unspecified. Do not fill the blank with an inferred story choice.
- Preserve ordered dialogue cues word-for-word and assign each cue only to its named speaker. Use silence, breathing, reactions, held expressions, existing action, and requested ambience—not invented speech—when dialogue finishes before the clip ends.
- Do not output ID-LoRA sections, LTX syntax, [VISUAL]/[SPEECH]/[SOUNDS] labels, workflow terminology, or model metadata.
"""


MINIMAX_H3_SHORT_FILM_GUIDED_INSTRUCTIONS_BY_MODE = {
    mode: instructions + "\n" + MINIMAX_H3_SHORT_FILM_GUIDED_CONTRACT
    for mode, instructions in MINIMAX_H3_INSTRUCTIONS_BY_MODE.items()
}


MINIMAX_H3_SHORT_FILM_CUSTOM_INSTRUCTIONS_BY_MODE = {
    mode: instructions + "\n" + MINIMAX_H3_SHORT_FILM_CUSTOM_CONTRACT
    for mode, instructions in MINIMAX_H3_INSTRUCTIONS_BY_MODE.items()
}
