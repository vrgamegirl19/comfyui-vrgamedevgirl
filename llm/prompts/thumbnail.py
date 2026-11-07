"""Instructions for the LLM that writes the YouTube-style thumbnail prompt.

Two kinds, because the image workflows use the chosen frame in two ways. An "edit" workflow (NanoBanana, Flux/Klein) takes the frame
as a reference picture and follows an instruction such as "recrop the attached image". A "restyle" workflow (Krea 2, Z-Image, Ernie,
MOSS-Image) samples from the frame as an image-to-image source and has no idea what "the attached image" is, so its prompt has to
describe the finished picture.
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

THUMBNAIL_TEXT_RULE = (
    '- Add the title text "{title}" exactly as written and spelled, in huge bold sans-serif capital letters with a thick '
    "dark outline and a slight glow, placed in the calmer third of the frame, large enough to read on a phone and never "
    "covering the face. Add no other text."
)

THUMBNAIL_NO_TEXT_RULE = "- Do not add any text, letters or numbers to the image."


def thumbnail_instruction(title: str = "", kind: str = "edit") -> str:
    """Return the instruction for ``kind`` (``"edit"`` or ``"restyle"``; anything else is ``"edit"``).

    The rule for the title text is filled in, or the no-text rule when there is no title.
    """
    title = " ".join(str(title or "").split())
    rule = THUMBNAIL_TEXT_RULE.format(title=title.replace('"', "'")) if title else THUMBNAIL_NO_TEXT_RULE
    template = THUMBNAIL_RESTYLE_PROMPT_INSTRUCTIONS if str(kind or "").strip().lower() == "restyle" else THUMBNAIL_EDIT_PROMPT_INSTRUCTIONS
    return template.replace("{text_rule}", rule)
