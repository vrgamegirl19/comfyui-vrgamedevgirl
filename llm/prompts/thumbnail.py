"""Instructions for the LLM that writes the YouTube-style thumbnail prompt.

Every kind except ``qwen_edit`` starts with the same creative brief, ``THUMBNAIL_ROLE_INSTRUCTIONS`` (the project owner's own text, kept
word for word). ``qwen_edit`` has its own instruction, because Qwen Image Edit needs a prompt about the change, not the picture.
It is written to an image editor that enhances a picture, but this step has to return a prompt, so what follows it is a short
"Your Answer" section: write the one prompt the image workflow will receive, in the form that workflow needs, with the title rule.

Three kinds, because the image workflows use the chosen frame in different ways. An "edit" workflow (NanoBanana, Flux/Klein) takes the
frame as a reference picture and follows an instruction such as "recrop the attached image". "moss_edit" (MOSS-Image Edit) also
follows an instruction, but a short one in the form its edit LoRA was trained on: what to add or change, then a "Keep ... unchanged"
sentence that names what stays. A "restyle" workflow (Krea 2, Z-Image, Ernie, MOSS-Image) samples from the frame as an image-to-image
source and has no idea what "the attached image" is, so its prompt has to describe the finished picture.

The title text is mandatory in every kind. The rules here ask for it, and the Thumbnail dialog makes certain of it afterwards
(``ensureThumbnailTitle`` in ``thumbnail_state.mjs``), because an LLM can still leave it out.
"""

# The brief. Word for word as the project owner wrote it (only the spaces at the end of two lines were dropped): change it only when they ask.
THUMBNAIL_ROLE_INSTRUCTIONS = """# Role and Objective
You are an expert AI image editor and creative director specializing in YouTube thumbnails for music videos across genres like hip-hop, pop, R&B, and electronic. Your task is to take an input image of an artist or scene and enhance it into a high-impact, click-worthy, professional thumbnail.

# Core Enhancement Guidelines
1. **Subject Isolation & Depth:** Separate the main artist/subject sharply from the background. Apply a strong depth of field with a subtle cinematic blur to the background to ensure the subject pops immediately.
2. **Lighting & Color Grading:**
   - Apply dramatic, high-contrast cinematic lighting.
   - Introduce strong **rim lighting** or colored edge-glows (e.g., contrasting neon cyan and magenta, electric blue, or warm sunset gold) that complement the mood of the music.
   - Deepen shadows and boost mid-tone clarity on the artist’s face and clothing to create a high-end editorial look.
3. **Composition & Framing:**
   - Crop or frame tightly on the subject (chest-up or waist-up) with intense, engaging eye contact toward the camera.
   - Maintain intentional **negative space** on one side of the frame (left or right) so a title text overlay can be added later.
4. **Texture & Finish:**
   - Enhance fine details (skin texture, clothing fabric, jewelry shine, hair strands) without making it look overly plastic or AI-smoothed.
   - Apply a subtle, premium digital film grain or grade to unify the subject and background.

# Output Constraints
- Keep the face, likeness, and key identity of the subject from the input image recognizable.
- Avoid cluttered or distracting background elements; keep the focus entirely on the star power and emotion of the artist.
- Deliver a clean, crisp, 16:9 aspect ratio final image optimized for small mobile screens."""

# What the image workflow wants as its prompt, one form for each kind.
_EDIT_FORM = (
    'Write it as ONE image-edit prompt: direct instructions to an image editor ("Recrop the attached image ..."), as one paragraph of '
    "80 to 170 words, in concrete visual words only."
)
_RESTYLE_FORM = (
    "Write it as ONE image prompt that describes the finished thumbnail, because the image-to-image workflow starts from this frame and "
    'does not know what "the attached image" is. Describe the finished picture as one paragraph of 60 to 140 words, in concrete '
    'visual words ("A close-up of ..."). Do not write instructions, and do not mention the attached image or an edit.'
)
_MOSS_EDIT_FORM = (
    "Write it as ONE short image-edit instruction in the form MOSS-Image Edit was trained on: what to add or change, then one sentence "
    'that starts with "Keep" and names what must not change, for example "Keep the man, his black beanie and black hoodie, his face '
    'and beard, his upward gaze, the painted mural, the candles and the composition unchanged." That model keeps the frame\'s '
    "composition, so leave out any cropping or reframing, and always name the headwear, hair and clothes in the Keep sentence. "
    "Write one paragraph of 50 to 110 words."
)


def _with_answer(form: str) -> str:
    """The brief, then the section that turns it into the one prompt this step returns (``{text_rule}`` is filled in later)."""
    return (
        THUMBNAIL_ROLE_INSTRUCTIONS + "\n\n"
        "# Your Answer\n"
        "The image workflow cannot read the instructions above. Look at the input image and write the one prompt it will receive. "
        + form + "\n\n"
        "The prompt must follow this rule:\n"
        "{text_rule}\n\n"
        "Match the mood of the song and the notes below.\n"
        "Output only the prompt: no title, no quotes around it, no explanation, no markdown.\n"
    )


THUMBNAIL_EDIT_PROMPT_INSTRUCTIONS = _with_answer(_EDIT_FORM)
THUMBNAIL_RESTYLE_PROMPT_INSTRUCTIONS = _with_answer(_RESTYLE_FORM)
THUMBNAIL_MOSS_EDIT_PROMPT_INSTRUCTIONS = _with_answer(_MOSS_EDIT_FORM)

# Qwen Image Edit does not use the brief above. It already sees the frame, so a prompt that describes the picture (the person, the
# scene, the camera, the grain) only buries the one thing that has to be made: the title lettering. On a real frame, a prompt that
# spent its words on the picture gave a plain headline, and one that spent them on the lettering gave a full graffiti piece. The shape
# is the one Qwen's own prompt rewriter asks for: direct and specific, the change first (all text in double quotes, the style by its key
# visual features), then one short "Keep ... unchanged." (https://github.com/QwenLM/Qwen-Image, prompt_utils.py).
THUMBNAIL_QWEN_EDIT_PROMPT_INSTRUCTIONS = """You write ONE edit prompt for Qwen Image Edit, an instruction-following image editor, to turn the attached video frame into a YouTube thumbnail. The model already sees the frame, so the prompt describes only what to ADD or CHANGE, never the picture itself: do not describe the person, the scene, the camera or the lighting that are already there.

The title lettering is the star of the thumbnail, so most of your words go there. Write the prompt as plain, direct sentences, 50 to 100 words in total, in this order:
1. Start with: Add the text "<the title>" as ... Describe the lettering by its key visual features: the lettering style (follow the style notes below and make them vivid and specific), the fill colors or gradient, the outline and glow, effects such as drips, splatter, sparks or shine, and decorative lines, swirls or shapes around it and their colors. Say where it goes: pick an empty or calm area you can see in the frame (for example "in the dark sky at the upper right"), say that it sits fully inside the frame, and say how big it is (for example "about a third of the image width"). It never covers the person's face.
2. Optionally, one short sentence of overall enhancement, for example "Add a thin warm rim light along the person's profile and richer contrast." Nothing else: no recropping or reframing, no blur or depth of field, no film grain, no new objects or people.
3. End with one short sentence that starts with "Keep" and names what must stay as it is, taken from what you see: the person, their headwear and clothes, and the main background, "unchanged". For example "Keep the man, his beanie and hoodie, the mural and the street unchanged."

{text_rule}

An example of the level of detail wanted. Use your own title, colors and style from the notes below, not this one: Add the text "HEAT" as a graffiti piece in the dark sky at the upper right, fully inside the frame and about a third of the image width: wildstyle graffiti lettering with an orange-to-red gradient fill, a glowing cyan neon outline, dripping paint, spray-paint splatter, and flowing calligraphic swirl lines in cyan, magenta and yellow around the letters. Add a thin warm rim light along his profile. Keep the man, his beanie and hoodie, the mural and the street unchanged.

Match the mood of the song and the notes below.
Output only the prompt: no title, no quotes around it, no explanation, no markdown.
"""

THUMBNAIL_QWEN_TEXT_RULE = (
    'The title text is mandatory: the prompt must contain the text "{title}" in double quotes, spelled exactly as given, as the '
    "lettering of step 1. If the style notes do not describe the lettering, use bold, high-contrast lettering with a thick outline and a glow."
)

THUMBNAIL_QWEN_NO_TEXT_RULE = (
    "There is no title text: skip step 1, do not add any text, letters or numbers, and write only the enhancement and the Keep sentence. "
    "Ignore the example below: it is about a title."
)

THUMBNAIL_TEXT_RULE = (
    '- The title text is mandatory: the prompt must contain "{title}" in double quotes, exactly as written and spelled, in huge bold '
    "sans-serif capital letters with a thick dark outline and a slight glow, placed in the calmer third of the frame, large enough "
    "to read on a phone and never covering the face. Follow any lettering style in the style notes. Add no other text."
)

# For MOSS-Image Edit, which keeps the frame as it is: the text goes where the frame already has room for it.
THUMBNAIL_MOSS_TEXT_RULE = (
    '- First, the title text, which is mandatory: add "{title}" in double quotes, exactly as written and spelled, in huge bold '
    "sans-serif capital letters with a thick dark outline and a slight glow, large enough to read on a phone, in the calmest part "
    "of the frame (the sky, a blurred wall or empty space you can see) and never covering the face. Follow any lettering style in "
    "the style notes. Add no other text."
)

THUMBNAIL_NO_TEXT_RULE = "- Do not add any text, letters or numbers to the image."


def thumbnail_instruction(title: str = "", kind: str = "edit") -> str:
    """Return the instruction for ``kind`` (``"edit"``, ``"moss_edit"``, ``"qwen_edit"`` or ``"restyle"``; anything else is ``"edit"``).

    The rule for the title text is filled in, or the no-text rule when there is no title.
    """
    title = " ".join(str(title or "").split()).replace('"', "'")
    kind = str(kind or "").strip().lower()
    no_text_rule = THUMBNAIL_NO_TEXT_RULE
    if kind == "restyle":
        template, text_rule = THUMBNAIL_RESTYLE_PROMPT_INSTRUCTIONS, THUMBNAIL_TEXT_RULE
    elif kind == "moss_edit":
        template, text_rule = THUMBNAIL_MOSS_EDIT_PROMPT_INSTRUCTIONS, THUMBNAIL_MOSS_TEXT_RULE
    elif kind == "qwen_edit":
        template, text_rule, no_text_rule = THUMBNAIL_QWEN_EDIT_PROMPT_INSTRUCTIONS, THUMBNAIL_QWEN_TEXT_RULE, THUMBNAIL_QWEN_NO_TEXT_RULE
    else:
        template, text_rule = THUMBNAIL_EDIT_PROMPT_INSTRUCTIONS, THUMBNAIL_TEXT_RULE
    rule = text_rule.format(title=title) if title else no_text_rule
    return template.replace("{text_rule}", rule)
