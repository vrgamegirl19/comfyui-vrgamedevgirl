"""Prompts and JSON schemas for Story Mode (the story book, acts, scene plan and story locations).

Used by: storyboard.story_book (book, chapter rewrite, pasted-story split, cast scan).
Rule shared by every prompt here: lyrics set mood and emotion only. They can be literal or metaphorical, so they
never decide a story event or a place.
"""

LYRICS_ARE_MOOD_ONLY = (
    "Lyrics rule: the song lyrics are literal or metaphorical. Use them only for mood and emotion. "
    "Never take a story event, a character, an object or a place from the lyrics."
)

_BOOK_FORMAT = (
    "Title: <story title>\n"
    "Premise: <one or two sentences>\n\n"
    "Chapter 1: <chapter title>\n"
    "Setting: <the one general setting of this chapter, 1-4 words, e.g. Home, The Night Train, Grandmother's Farm>\n\n"
    "<paragraph>\n\n"
    "<paragraph>\n\n"
    "Chapter 2: <chapter title>\n"
    "Setting: <setting>\n\n"
    "<paragraph>"
)


def story_mode_lyric_guidance(strength):
    """Lyric Story Strength in Story Mode: how strongly a scene's emotion follows its lyric line (never events or places)."""
    try:
        value = int(float(strength))
    except (TypeError, ValueError):
        value = 5
    value = max(0, min(10, value))
    if value <= 0:
        guidance = "Ignore the lyrics entirely. The story alone sets the mood."
    elif value <= 3:
        guidance = "Let the lyrics tint the mood lightly. The story sets the emotion."
    elif value <= 6:
        guidance = "Balance the story's emotion with the feeling of the lyrics."
    elif value <= 8:
        guidance = "Let each scene's emotion follow the feeling of its lyric line closely."
    else:
        guidance = "Match each scene's emotion to its lyric line as closely as the story allows."
    return f"Lyric Story Strength: {value}/10 (emotion only). {guidance} {LYRICS_ARE_MOOD_ONLY}"


def story_book_size(scene_count):
    """(chapters, paragraphs) a book should have for a timeline of ``scene_count`` scenes."""
    count = max(1, int(scene_count or 1))
    chapters = max(3, min(7, round(count / 8) or 3))
    paragraphs = max(chapters, round(count / 2.5))
    return chapters, paragraphs


def story_book_instruction(*, premise, cast_text, tone_notes, scene_count, song_seconds, lyric_guidance, lyrics,
                           previous_book, seed):
    """Draft the whole story book from the user's premise."""
    chapters, paragraphs = story_book_size(scene_count)
    return (
        "You are a novelist writing the story that a music video will tell.\n\n"
        "Write a short book: a title, a premise, and chapters of plain prose. It must read like a book, not a shot "
        "list or a song breakdown.\n\n"
        "Rules:\n"
        f"- About {chapters} chapters and about {paragraphs} paragraphs in total. The video has {scene_count} scenes "
        f"and runs {song_seconds:.0f} seconds, so the story must fit that length.\n"
        "- Every chapter happens in ONE general setting, named on its Setting line. A new setting starts a new chapter. "
        "Inside a setting the story can move between specific spots (kitchen, bedroom, porch).\n"
        "- Each paragraph is 2-5 sentences of things that visibly happen: actions, choices, meetings, discoveries.\n"
        "- Use the cast names below. Main characters drive the story. You may add a few new minor people when the "
        "story needs them; give each a name.\n"
        "- Third person. Present or past tense, but keep the same tense throughout.\n"
        "- No camera directions, no shot language, no song-section words (Intro, Verse, Chorus, Bridge, Outro), "
        "no scene numbers.\n"
        "- The story has a beginning, a turn and an ending.\n"
        f"- {lyric_guidance}\n"
        "- Output only the book in exactly this format, with no notes before or after it:\n\n"
        f"{_BOOK_FORMAT}\n\n"
        f"Premise from the user:\n{premise or '[none, invent one from the cast and tone]'}\n\n"
        f"Cast:\n{cast_text or '[none, invent a small cast]'}\n\n"
        f"Tone, genre and world notes:\n{tone_notes or '[none]'}\n\n"
        f"Song lyrics (mood and emotion only):\n{lyrics or '[none]'}\n\n"
        f"Previous draft to move away from on this reroll:\n{previous_book or '[none]'}\n\n"
        f"Variation seed: {seed or '[none]'}"
    )


def story_chapter_rewrite_instruction(*, book_text, chapter_number, chapter_text, direction, cast_text, lyric_guidance):
    """Rewrite or expand one chapter, keeping the rest of the book as context."""
    return (
        "You are a novelist revising one chapter of a story that a music video will tell.\n\n"
        f"Rewrite chapter {chapter_number} following the user's direction. Keep it consistent with the chapters "
        "around it, keep its setting unless the direction says otherwise, and keep the cast names.\n\n"
        "Rules:\n"
        "- Plain prose paragraphs of things that visibly happen. No camera or song-section words.\n"
        f"- {lyric_guidance}\n"
        "- Output only the chapter in exactly this format:\n\n"
        f"Chapter {chapter_number}: <chapter title>\n"
        "Setting: <setting>\n\n"
        "<paragraph>\n\n"
        "<paragraph>\n\n"
        f"User direction:\n{direction or 'Improve and expand this chapter.'}\n\n"
        f"Cast:\n{cast_text or '[none]'}\n\n"
        f"Chapter to rewrite:\n{chapter_text}\n\n"
        f"Whole book for context:\n{book_text}"
    )


def story_split_instruction(*, numbered_paragraphs, fixed_starts, scene_count):
    """Group a pasted story's paragraphs into chapters, each with one setting. The text itself is never rewritten."""
    fixed = (
        "The story already has chapter headings. Keep exactly these chapter starts, in order: "
        + ", ".join(str(start) for start in fixed_starts)
        + ". Only give each one a title (keep a written title) and a setting.\n"
        if fixed_starts else
        "Choose where each chapter starts. A chapter starts where the story moves to a new general setting or a "
        "clear new part begins. Aim for 3-7 chapters.\n"
    )
    return (
        "You are an editor dividing a finished story into chapters for a music video. Do not rewrite any text.\n\n"
        f"The video has {scene_count} scenes.\n"
        "Every chapter happens in ONE general setting (1-4 words, e.g. Home, The Night Train). A new setting needs a "
        "new chapter.\n"
        f"{fixed}"
        "Return JSON only: {\"chapters\": [{\"start_paragraph\": 1, \"title\": \"...\", \"setting\": \"...\"}]}. "
        "start_paragraph is the number of the first paragraph of the chapter. The first chapter starts at 1.\n\n"
        f"Paragraphs:\n{numbered_paragraphs}"
    )


STORY_SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_paragraph": {"type": "integer"},
                    "title": {"type": "string"},
                    "setting": {"type": "string"},
                },
                "required": ["start_paragraph", "title", "setting"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["chapters"],
    "additionalProperties": False,
}


def story_cast_scan_instruction(*, book_text, known_names):
    """List people in the story who have no cast card yet, with a visual description each."""
    return (
        "You are a casting director reading a story for a music video.\n\n"
        "List every person in the story who is NOT in the known cast list. Only people who appear or act in the "
        "story, not people only mentioned in passing. Do not list groups or crowds.\n\n"
        "For each person give:\n"
        "- name: as written in the story (invent a short fitting name only if the story names them by role).\n"
        "- role: main, supporting or background.\n"
        "- description: one sentence of how they look (age range, build, hair, clothing) that fits the story. "
        "No actions, no personality.\n\n"
        "Return JSON only: {\"characters\": [{\"name\": \"...\", \"role\": \"...\", \"description\": \"...\"}]}. "
        "Return an empty list when everyone is already cast.\n\n"
        f"Known cast:\n{known_names or '[none]'}\n\n"
        f"Story:\n{book_text}"
    )


STORY_CAST_SCAN_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "role", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["characters"],
    "additionalProperties": False,
}
