"""Instructions for the LLM that writes the YouTube-style thumbnail prompt.

Every kind starts with the same creative brief, ``THUMBNAIL_ROLE_INSTRUCTIONS`` (the project owner's own text, kept word for word).
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
    """Return the instruction for ``kind`` (``"edit"``, ``"moss_edit"`` or ``"restyle"``; anything else is ``"edit"``).

    The rule for the title text is filled in, or the no-text rule when there is no title.
    """
    title = " ".join(str(title or "").split()).replace('"', "'")
    kind = str(kind or "").strip().lower()
    if kind == "restyle":
        template, text_rule = THUMBNAIL_RESTYLE_PROMPT_INSTRUCTIONS, THUMBNAIL_TEXT_RULE
    elif kind == "moss_edit":
        template, text_rule = THUMBNAIL_MOSS_EDIT_PROMPT_INSTRUCTIONS, THUMBNAIL_MOSS_TEXT_RULE
    else:
        template, text_rule = THUMBNAIL_EDIT_PROMPT_INSTRUCTIONS, THUMBNAIL_TEXT_RULE
    rule = text_rule.format(title=title) if title else THUMBNAIL_NO_TEXT_RULE
    return template.replace("{text_rule}", rule)
