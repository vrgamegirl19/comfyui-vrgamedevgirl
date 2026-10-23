"""MiniMax H3 prompt creation: per-mode instructions, frame continuity and short-film contracts."""

MINIMAX_H3_SUPPORTING_FACE_DIRECTION = """SUPPORTING CHARACTER FACIAL ACTING — MANDATORY
Every established character whose face is visible gets an individual, readable facial reaction in every shot, including silent supporting performers and people behind the lead. Identify each by their existing subject label; do not replace their acting with 'the others watch', 'lips closed', 'eyes bright' or one shared expression for the group.
Give each visible supporting face a specific reaction and a small progression suited to the available time: for example one person's brows pinch before one brow arches toward the singer, another gives a sideways glance that becomes a closed-mouth grin, while another's eyes widen before a skeptical head tilt. Choose different reactions from each person's role, the lyric or dialogue meaning, and the scene beat instead of copying those examples or making everyone raise their brows and smile together. Make the change clear enough to read at the actual framing, scaled to the requested facial direction, video style and scene energy; keep explicitly restrained acting restrained.
Silence restricts vocal performance, not facial expression. Supporting performers remain silent outside their assigned cues while their eyes, brows, cheeks, closed-mouth expressions and small head movements stay alive. Do not add words, lip-sync, audible laughter or another vocal cue. Preserve supplied audio and existing mouth-movement restrictions.
Carry each person's expression and eyeline naturally across camera cuts along with their position and ongoing action, then let the reaction develop during the next shot. Keep established camera coverage and staging; describe facial details only for faces visible and readable in that framing. Offscreen people and faces hidden by a close-up need no invented reaction or extra shot. Environment and prop labels are not characters. Before returning, check each shot for individual acting from every readable supporting face without adding cast, props or story events. Keep each reaction concise.
"""


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

LITERAL SCENE CLARITY — MANDATORY: Write natural, fluent shot prose that H3 can follow literally. Lead physical action with the person doing it; use their possessive for a body part or worn item and clear pronouns afterward. For example: '<Subject 1> steps past the ground-level camera, her boot briefly crossing the foreground.' Describe how the camera films that same action instead of independently placing a body part in the foreground and its owner elsewhere. A close-up identifies whose body part is framed; describe cropping only when needed to understand the shot. Keep ownership and spatial checks in your reasoning; finished sentences describe visible action, not explanations of anatomy or routine clothing attachment. Rewrite ambiguous scene-card or image-prompt wording into ordinary action-led sentences rather than copying it. Keep character, camera and prop positions consistent from the opening through the ending. Place loose props explicitly and use only objects supported by this scene's directions or mapped references. Carry an object from an adjacent scene only when the current scene explicitly does so. Before returning, read the shot by itself: every action has an actor, every item has a clear owner or placement, and the camera movement fits the same physical action. Resolve uncertainty with clear verbs and possessives, not explanatory qualifiers. Return only the requested shot JSON with natural visual prose.

SCENE TIMING
Follow the exact scene duration and per-shot time windows in the Scene concept. Use the scene beat and storyboard directions together to stage only what can physically finish within each shot's available time. Reserve time for the required singing or speaking and any supplied entrance, reveal, or transition. For a brief shot, express the essential beat through one economical continuous action and the requested camera move; let performance and camera movement happen together where physically possible. Keep optional gestures and reactions only when time remains. Do not invent sequential steps, pivots, glances, or framing changes to fill a checklist. Preserve explicitly requested action and exact vocal words; simplify optional choreography rather than rushing required performance. Motion speed sets the energy of the chosen movement, not the number of actions. Before returning, check that the opening, action, camera travel, and ending fit the available time without an extra cut, rushed final beat, or action continuing beyond the shot.

MULTI-SHOT CONTINUITY — MANDATORY: When a scene has multiple shots, write them as consecutive camera views of one uninterrupted physical performance in the same place and moment. Shot 1 establishes staging. In every later description, write a complete sentence explicitly continuing directly from the previous shot's final moment. At the instant of the cut, preserve the previous ending position, posture, body orientation, eyeline and prop contact, then carry the ongoing movement forward at its existing pace and direction. For a scene with active singing, for example: 'Continuing directly from the previous shot's final moment, <Subject 1> remains seated in the same position on the curb, her torso still angled toward the doorway, and continues the same uninterrupted singing performance in sync with the assigned audio.' Use the actual preceding physical state, not a generic continuity phrase. Do not restart an action, reset a pose, or replace continued performance with an unrelated beauty shot. An ongoing action may span scheduled cuts. Only camera angle, framing and camera movement change at a cut unless the scene explicitly requests a new action or transition. Show any requested physical change developing naturally from the inherited state. During every shot's active vocal interval, explicitly state that the assigned performer continues singing or speaking in sync with the assigned audio. Preserve cue timing and speaker assignments; instrumental, no-lip-sync and silent intervals remain non-vocal. Follow supplied lyric/dialogue allocations without duplicating an already quoted cue or restarting the scene's lyrics after a cut. Use complete grammatical sentences rather than fragments such as 'Still seated on the curb.' Self-contained shot prose and varied camera coverage must preserve the shared staging. Before returning, check every adjacent pair for a matching end-to-start physical state and continuous performance.

SHOT FORMAT
Write each shot like a music video director's shot note in concise complete sentences suited to its available time, in this order:
1. Camera: the opening framing, one named move (direction and speed), and the ending framing.
2. Action: what each person in the cast physically does, by label, as one continuous motion.
3. Acting detail: each visible face's concrete expression for this moment, taken from the lyric, story beat and song and sized to the video style, as the Scene concept's PERFORMANCE CRAFT rules describe.
4. Light and set: one line using only the mapped location's own light and objects.
Describe only what the camera sees. Show emotion only as visible movement, never with feeling words, and keep it strong: people should look alive and expressive. Keep everything realistic: natural physical movement, real-world light, believable weight and timing. Match the energy to the camera and character speed values in the Scene concept. Identify each character by its subject label on first mention in each shot, then use unambiguous pronouns and possessives. Repeat a label when the actor changes or ownership would be unclear; do not append names or picture origins in parentheses. Show only the people in the cast list: no other person, hand, arm, shadow, reflection, or crowd.
Keep every person upright: standing, walking, sitting on a seat, kneeling or leaning. Never write lying down, side-lying, reclining, sprawled or resting on the floor or a bed, unless the scene concept explicitly asks for it. Never press, lean, rest or lay a cheek, face, forehead, head or temple against a wall, glass, window, door, table, pillow, surface, object or another person. Faces stay free in the air. Use ordinary natural poses and gestures instead.
Character appearance comes from the reference images. Do not describe clothing, hair, accessories, jewelry, or facial features, except one brief mention when a garment or feature moves in the action. Every shot should read differently from the others in framing and movement.
"""


_MINIMAX_H3_TEXT_TO_VIDEO_MODE = """MODE: TEXT TO VIDEO
Use the resolved text context. Do not invent picture or video labels.
"""


_MINIMAX_H3_IMAGE_TO_VIDEO_MODE = """MODE: IMAGE TO VIDEO
Use the resolved starting-picture assignment as the visual anchor. Animate it naturally.
"""


_MINIMAX_H3_REFERENCE_TO_VIDEO_MODE = """MODE: REFERENCE TO VIDEO
Use the resolved <Subject N> and <Picture N> assignments directly in the cinematic shot descriptions. Identify each character by its subject label on first mention in each shot, then use natural pronouns when unambiguous; do not append names or picture origins in parentheses.
Generate a complete new scene from those character and environment references. Character pictures supply identity and appearance; environment pictures supply the set. The prompt determines opening framing, camera angle, staging, pose, composition, and action. Never copy a reference picture's composition, framing, camera angle, or pose as the opening shot. A separately enabled continuation task follows its own previous-frame rules.
A saved image prompt is a proposed scene idea, not a visual inventory of a supplied picture. Use props supported by the mapped references or current explicit scene directions; omit unsupported carryover props.
When a requested action needs a new prop, introduce its appearance and physical placement before referring to it as established. Do not write unexplained references such as 'the rusted worktable' when no supplied reference or earlier sentence establishes that table.
Use natural pronouns after establishing the character by label. Describe the location reference as the physical setting and derive staging and camera direction from the scene beat and storyboard details. Do not add a standalone reference-definition paragraph to the shot description.
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


# The "Enhance Prompts" pass: the LLM receives only these instructions and one finished prompt. The builder keeps
# everything outside the [Shot N] blocks itself, so the rules below are about the shot writing only.
MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT = MINIMAX_H3_SUPPORTING_FACE_DIRECTION + "\n" + """MULTI-SHOT CONTINUITY — REQUIRED ENHANCEMENT CONTRACT
For multiple [Shot N] blocks, all shots are consecutive camera views of one uninterrupted performance in the same location and moment. This contract takes priority over any instruction to give each shot a separate action or held ending.
The builder places this scene-wide paragraph after the detailed_description opening and before [Shot 1]: All shots follow one continuous performance, preserving every person's exact position, body orientation, posture, and ongoing movement across every camera cut; each new shot continues directly from the final physical moment of the previous shot, with only the camera angle and framing changing.
Keep that paragraph outside every shot block. Shot 1 starts with its actual camera framing and action. If returning only shot blocks inside JSON, omit the global paragraph; the builder adds it.
Start every later shot immediately after its original [Shot N] label and optional At timestamp with exactly: continuing directly from the previous shot's final moment,
Then state the actual inherited physical state from the preceding shot's ending: the same characters in the same location, their positions, posture, body orientation, eyelines, what each hand holds or touches, and the movement already in progress. Follow with the new camera angle or framing and the shot's existing action developing naturally from that state. Use the actual subject and location labels already assigned; their identities, clothing, props, set and lighting remain consistent, including people outside the new framing.
Only the camera angle, framing or camera movement changes at the cut. Continue ongoing action at its existing pace and direction; do not restart a gesture, replay an entrance, reset a pose, move a character to a new position, swap hands or props, or establish a new place at a cut. An action may continue across several shots without a held ending. Any physical change already requested by the original prompt must develop visibly from the inherited state during the shot.
Keep singing or speaking continuous across cuts during the assigned active vocal intervals, using the same performer and audio reference, with every original cue, word and timing preserved. Keep instrumental and silent intervals as supplied.
Example of a later opening: [Shot 2] At 00:03.150, continuing directly from the previous shot's final moment, <Subject 1> remains at the front centre of the same stage, the microphone still in her right hand and the band still playing behind her; the camera cuts to a low close-up as her ongoing performance continues.
Use the current prompt's actual staging, props and timings rather than copying the example. Before returning, check every adjacent pair for a matching physical end-to-start state. For a single shot, preserve its single-shot format without adding multi-shot language.
CAMERA PATH CLARITY: Check the opening framing, camera move and final framing as one physically coherent view. A cut directly to someone's face already establishes that close-up; hold it or use a motivated move from that view. Never open on a face, whip-pan away, then claim to settle on that same face without a physically explained return. For example, replace 'cuts to a high-angle 85mm close-up of the singer, then whip-pans right and settles on her face' with 'cuts directly to a high-angle 85mm close-up of the singer and holds her face in frame.' Use a pan only when its different starting view and destination are clear. Keep camera travel plausible within the mapped space and shot duration.

ALL REFERENCE IMAGES AND EXACT TAGS — MANDATORY:
Every assigned reference image must remain represented in the completed prompt by its exact <Picture N> label and its original subject association and reference role. Include every mapped character, extra, location, outfit, object and frame reference; none is optional or removable as repetitive. When images are attached, inspect every mapped image and return every required mapped description.
Before rewriting each shot, inventory every <Subject N>, <Picture N>, <Audio N> and <Video N> tag in that original shot. Include every one of those exact tags in that same rewritten shot. A definition elsewhere in the prompt does not replace a tag required inside a shot. Preserve the original image-to-subject numbering; never renumber, swap, merge or drop references.
A mapped subject label and its picture label are not interchangeable: if the original shot contains <Picture 5>, writing only <Subject 5>, 'the same location' or 'the car interior' is insufficient. Keep <Picture 5> explicitly in that shot, even when <Subject 5> is also present. For an assigned location image, include its exact picture label in every shot, for example 'The action continues in the environment from <Picture 5>.' Use the actual mapped location number.
Keep references in their assigned roles. Retaining an offscreen character's reference does not require bringing them into frame; retaining start/end-frame references does not turn them into new scenes or change their anchor timing. Preserve the existing camera coverage, cast and continuity.
FINAL REFERENCE CHECK: Compare each rewritten shot against its original tag inventory and restore every omitted tag before returning. Confirm that all assigned images remain represented in the complete prompt. Shorten optional descriptive prose to meet the length limit; never shorten by removing a reference tag, mapped image, lyric or timing.
"""

MINIMAX_H3_PROMPT_ENHANCE_INSTRUCTIONS = """You improve one finished MiniMax H3 music video prompt. Rewrite the text of every [Shot N] block so the camera work, the movement and the facial acting are stronger, while the scene stays the same scene.

Return the complete prompt in the same format, with every [Shot N] block in the same order. Return only the prompt: no notes, lists, headings, markdown or explanations.

KEEP EXACTLY
- Every [Shot N] label and every "At 00:00.000," time at the start of a shot.
- Every <Subject N>, <Picture N>, <Audio N> and <Video N> label, used for the same person or thing.
- Every sung or spoken word inside <d>...</d> and inside double quotes, in the same order. You may improve the emotion words in the [ ] tag at the start of a <d> cue.
- Every "from X s to Y s" timing.
- Who is in the scene, where it happens and what happens. Add no new people, props, places or story events.
- Clothing, hair and props only as the prompt already names them.

CAMERA
- After the required continuity opening, describe a specific framing and lens: for example a wide-angle close-up with the face large in the frame, a low wide-angle shot from the floor, a medium shot from the side, a Dutch-tilted medium close-up, a shot tracking backward in front of the singer.
- Every new shot changes the framing or angle in a way the viewer can see.
- Name a camera move only inside the shot where it happens (cuts to, whip-pans to, pushes in, circles, tracks backward). One move per shot.
- Keep the main singer at full body or closer; use medium shots and close-ups while she or he sings.

MOVEMENT
- Refine the existing physical action, timed with the line being sung, preserving its progression across camera cuts and using only established props and set contact.
- Show the feeling of the line through action; do not act out the words literally.
- Carry the physical state into the next shot: where the character stands, what they hold, what just happened ("still at the table, the glass in her hand").
- Let an ongoing action carry through a cut; use a held reaction or pose only when it naturally belongs to the existing action.

FACE AND EMOTION
- Write faces as visible actions: a brow lifting, eyes widening or narrowing, a smile cracking into a laugh, a jaw setting, eyes dropping, a slow blink.
- Let the expression change from line to line, sized to the mood the prompt already has.
- Give every sung cue two fitting emotion words in its [ ] tag, keep <i> on one or two key words, and keep or add <breath> only at a real pause between phrases.
- Give every readable supporting face its own evolving reaction by subject label in every shot where it is visible. Silent performers keep expressive eyes, brows, cheeks and closed-mouth expressions; carry these reactions across cuts. Use distinct reactions motivated by the scene rather than a collective 'the others watch' or identical raised brows for everyone.

WRITING
- Describe only what the camera sees, in complete, positive sentences. Write what happens, never what does not happen.
- Do not add sentences about lighting, color grading, film look or general camera style.
- Keep the prompt about the same length as the original. Make shots tighter, not longer.
""" + "\n" + MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT


MINIMAX_H3_PROMPT_ENHANCE_WITH_REFERENCES_INSTRUCTIONS = """Improve the supplied finished MiniMax H3 prompt using the attached, explicitly mapped reference images.

Inspect every mapped image. For each entry marked describe=true, return a brief factual description, about 15–35 words. For a character include visible hair colour/style and the outfit type and main colours, including a distinctive accessory only if clear. For a location describe the type of place, a few defining set features and its visible lighting. For other reference roles describe only their assigned purpose. Multi-view panels show one identity, not extra people. Describe only visible facts; omit uncertain or hidden details.
Follow each reference's purpose and authority: face/hair-only images do not supply wardrobe; outfit images do not supply identity. Start/end-frame images remain timeline anchors. Use their existing subject/picture labels exactly and keep the image-to-label assignment unchanged. Reference appearance corrects conflicting hair, clothing or environment wording in the shot prose, without changing staging, action, audio, timing or cast. Brief character/location definitions will be assembled separately by the builder; keep the shot prose focused on action and acting rather than repeating appearance in every shot.

Enhance camera coverage and individual visible facial acting while preserving the scene's exact actions, subjects, props and setting. Preserve every [Shot N] label, opening At timestamp, reference label, from X s to Y s timing, sung/spoken word and quoted lyric. Emotion headers may improve, but never change who performs a cue or its words. Add no people, props, places, vocal cues or story events. Keep the complete final prompt, including brief definitions, under 7,000 characters. Everything outside the shots is retained by the builder except the mapped visual definitions.

Return only valid JSON with exactly this shape:
{"reference_descriptions":[{"label":"<Subject 1>","description":"A woman with a cherry-red bob, wearing an ivory ruffled top, a black corset belt and a pleated skirt."}],"prompt":"[Shot 1] ...\\n\\n[Shot 2] At 00:03.150, ..."}
Return one description for every mapped entry marked describe=true, using its actual label, and none for entries marked describe=false. Return all enhanced shot blocks in prompt, in the original order. The example appearance is illustrative; use only the actual images. No markdown or commentary.
""" + "\n" + MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT
