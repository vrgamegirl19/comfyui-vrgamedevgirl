"""Instruction for the LLM that writes the YouTube-style thumbnail edit prompt."""

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

THUMBNAIL_TEXT_RULE = (
    '- Add the title text "{title}" exactly as written and spelled, in huge bold sans-serif capital letters with a thick '
    "dark outline and a slight glow, placed in the calmer third of the frame, large enough to read on a phone and never "
    "covering the face. Add no other text."
)

THUMBNAIL_NO_TEXT_RULE = "- Do not add any text, letters or numbers to the image."


def thumbnail_instruction(title: str = "") -> str:
    """Return the instruction, with the rule for the title text filled in (or the no-text rule when there is none)."""
    title = " ".join(str(title or "").split())
    rule = THUMBNAIL_TEXT_RULE.format(title=title.replace('"', "'")) if title else THUMBNAIL_NO_TEXT_RULE
    return THUMBNAIL_EDIT_PROMPT_INSTRUCTIONS.replace("{text_rule}", rule)
