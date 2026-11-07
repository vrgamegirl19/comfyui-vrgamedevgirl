"""Instructions for the LLM that writes the YouTube-style thumbnail prompt.

Three kinds, because the image workflows use the chosen frame in different ways. An "edit" workflow (NanoBanana, Flux/Klein) takes the
frame as a reference picture and follows an instruction such as "recrop the attached image". "moss_edit" (MOSS-Image Edit) also
follows an instruction, but a short one in the form its edit LoRA was trained on: what to add or change, then a "Keep ... unchanged"
list that names what stays. A "restyle" workflow (Krea 2, Z-Image, Ernie, MOSS-Image) samples from the frame as an image-to-image
source and has no idea what "the attached image" is, so its prompt has to describe the finished picture.

The title text is mandatory in every kind. The rules here ask for it, and the Thumbnail dialog makes certain of it afterwards
(``ensureThumbnailTitle`` in ``thumbnail_state.mjs``), because an LLM can still leave it out.
"""

THUMBNAIL_EDIT_PROMPT_INSTRUCTIONS = """You write ONE image-edit prompt that turns an attached video frame into a YouTube-style video thumbnail.

The attached image is the base. Keep who and what is in it: the same character or characters, face, hair, outfit, location and art style stay recognizable. Do not invent new people or change who they are.

Edit the frame into a thumbnail that stops a viewer from scrolling:
- Crop closer on the main subject so the face and upper body fill the frame, eyes sharp, expression clear and slightly heightened.
- Strong separation: a bright rim light on the subject, a simplified background with depth blur, and one bold accent color that contrasts the subject.
- A punchy grade: higher contrast and saturation, a soft glow on the light sources, a clean sharp subject. Keep it cinematic. Stay with illustration if the source image is an illustration.
- A thin colored outline or glow around the subject, a soft vignette, and light streaks or particles only when they fit the scene.
- A 16:9 composition: the subject fills one side and the opposite third stays calmer, with nothing important at the edges.
{text_rule}
- No watermarks, logos, platform buttons, timestamps or borders.

Match the mood of the song and the style notes below. Write the prompt as direct instructions to an image editor ("Recrop the attached image ..."), as one paragraph of 80 to 170 words, in concrete visual words only.
Output only the prompt: no title, no quotes around it, no explanation, no markdown.
"""

THUMBNAIL_RESTYLE_PROMPT_INSTRUCTIONS = """You write ONE image prompt that describes a finished YouTube-style video thumbnail. The attached image is the video frame it is made from: an image-to-image workflow starts from that frame, so describe what the thumbnail looks like, not an edit.

Keep what is in the frame: the same character or characters, face, hair, outfit, location, pose and art style, so the result still looks like the frame. Do not invent new people or change who they are.

Describe a thumbnail that stops a viewer from scrolling:
- A close framing of the main subject, the face and upper body large in the frame, eyes sharp, an expression that is clear and slightly heightened.
- Strong separation: a bright rim light on the subject, a simplified background with depth blur, and one bold accent color that contrasts the subject.
- A punchy grade: high contrast and saturation, a soft glow on the light sources, a clean sharp subject. Keep it cinematic. Stay with illustration if the frame is an illustration.
- A thin colored outline or glow around the subject, a soft vignette, and light streaks or particles only when they fit the scene.
- A 16:9 composition: the subject fills one side and the opposite third stays calmer, with nothing important at the edges.
{text_rule}
- No watermarks, logos, platform buttons, timestamps or borders.

Match the mood of the song and the style notes below. Write one paragraph of 60 to 140 words that describes the finished picture in concrete visual words ("A close-up of ..."). Do not write instructions, and do not mention the attached image or an edit.
Output only the prompt: no title, no quotes around it, no explanation, no markdown.
"""

THUMBNAIL_MOSS_EDIT_PROMPT_INSTRUCTIONS = """You write ONE short image-edit instruction for MOSS-Image Edit, an edit model that redraws the attached video frame with only the changes you name. The result is a YouTube-style video thumbnail.

The model was trained on instructions that say what to add or change and then what to keep, in this shape: "Add ... Change ... Keep the <named things> unchanged." It keeps the frame's composition, so do not ask it to crop, reframe, move or resize anything.

Write the instruction in this order:
{text_rule}
- Add a bright rim light on the subject in one bold accent color that contrasts the scene, and a thin matching glow around the subject's outline.
- Give the picture a punchy grade: higher contrast and saturation, a soft glow on the light sources, a clean sharp subject. Stay with illustration if the frame is an illustration.
- Last, one sentence that starts with "Keep" and names what must not change, from what you see in the frame: the person or people, their headwear or hair, clothing, face and facial hair, pose and gaze, and the important objects and the background. Name each one specifically, for example "Keep the man, his black beanie and black hoodie, his face and beard, his upward gaze, the painted mural, the candles and the composition unchanged." Anything you do not name can be redrawn, so always name the headwear, hair and clothes.
- No watermarks, logos, platform buttons, timestamps or borders.

Match the mood of the song and the style notes below. Write one paragraph of 50 to 110 words of direct instructions in concrete visual words.
Output only the instruction: no title, no quotes around it, no explanation, no markdown.
"""

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
