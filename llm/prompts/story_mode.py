"""Prompts and JSON schemas for Story Mode (the story book, acts, scene plan and story locations).

Used by: storyboard.story_book (book, chapter rewrite, pasted-story split, cast scan).
Rule shared by every prompt here: lyrics set mood and emotion only. They can be literal or metaphorical, so they
never decide a story event or a place.
"""

LYRICS_ARE_MOOD_ONLY = (
    "Lyrics rule: the song lyrics are literal or metaphorical. Use them only for mood and emotion. "
    "Never take a story event, a character, an object or a place from the lyrics."
)

# The story is written in Acts (big parts of the story). Chapters, each in one setting, are made from them later.
_BOOK_FORMAT = (
    "Title: <story title>\n"
    "Premise: <one or two sentences>\n\n"
    "Act 1: <act title>\n\n"
    "<paragraph>\n\n"
    "<paragraph>\n\n"
    "Act 2: <act title>\n\n"
    "<paragraph>"
)


def render_story_template(template, tokens):
    """A Story Mode instruction template with its ``{{name}}`` tokens filled in. Unknown tokens stay as written,
    so an edited template can never break the step."""
    text = str(template or "")
    for name, value in tokens.items():
        text = text.replace("{{" + name + "}}", str(value))
    # A saved instruction loses its last line break; the step's own text always starts on a new line.
    return text if not text or text.endswith("\n") else f"{text}\n"


# The editable part of each Story Mode instruction (role and rules). The step adds the data and the reply format
# after it. Saved per project as the story_mode_* Builder instructions (llm/builder_instructions.py).
STORY_BOOK_TEMPLATE = (
    "You are a novelist writing the story that a music video will tell, so a director can read it and see the "
    "video play out with the song.\n\n"
    "Write a short book: a title, a premise, and acts of plain prose. It must read like a book, not a shot "
    "list or a song breakdown, and it must unfold in step with the song.\n\n"
    "How it is told: {{medium_rule}}\n\n"
    "Rules:\n"
    "- Exactly {{acts}} acts and about {{paragraphs}} paragraphs in total. The video has {{scene_count}} scenes "
    "and runs {{song_seconds}} seconds, so the story must fit that length.\n"
    "- An act is one big part of the story (the setup, the turn, the climax, the ending). It goes deep into its part "
    "of the story and its characters, and it can move between places. Name each place plainly in the prose (the "
    "kitchen, the night train), because each act is later split into chapters that each happen in one place.\n"
    "- Each paragraph is 2-5 sentences of things that visibly happen: actions, choices, meetings, discoveries.\n"
    "- {{cast_rule}}\n"
    "- Third person. Present or past tense, but keep the same tense throughout.\n"
    "- No camera directions, no shot language, no song-section words (Intro, Verse, Chorus, Bridge, Outro), "
    "no scene numbers.\n"
    "- The story has a beginning, a turn and an ending.\n"
    "- Follow the song map from start to finish: the story moves in the same order and at the same pace as the "
    "song. Quiet or instrumental parts are quiet moments, rising parts build, the chorus and the loudest parts are "
    "the story's peaks, and the last act ends when the song ends. Do not name the sections in the book.\n"
    "- {{lyric_guidance}}\n"
)

STORY_CHAPTER_TEMPLATE = (
    "You are a novelist revising one act of a story that a music video will tell.\n\n"
    "Rewrite act {{chapter_number}} following the user's direction. Keep it consistent with the acts "
    "around it, keep its places unless the direction says otherwise, and keep the cast names.\n\n"
    "Rules:\n"
    "- Plain prose paragraphs of things that visibly happen. No camera or song-section words. Name each place "
    "plainly in the prose.\n"
    "- {{cast_rule}}\n"
    "- {{medium_rule}}\n"
    "- {{lyric_guidance}}\n"
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


def story_book_size(scene_count, act_count=0):
    """(acts, paragraphs) a book should have for a timeline of ``scene_count`` scenes. ``act_count`` is the user's
    choice; 0 lets the scene count decide (3-5 acts)."""
    count = max(1, int(scene_count or 1))
    acts = int(act_count or 0) or max(3, min(5, round(count / 10) or 3))
    paragraphs = max(acts * 2, round(count / 2.5))
    return acts, paragraphs


def story_cast_rule(cast_text, allow_new_cast):
    """The cast rule line for the book and chapter prompts."""
    if allow_new_cast or not cast_text:
        return ("Use the cast names below. Main characters drive the story. You may add a few new minor people when "
                "the story needs them; give each a name.")
    people_lines = cast_text.split("\nRelationships:")[0]
    count = len([line for line in people_lines.splitlines() if line.strip().startswith("- ")])
    people = "1 character" if count == 1 else f"{count} characters"
    return (f"The story has exactly {people}: the cast below, and no one else. Do not add, name or describe any other "
            "person, not even a minor one (no friends, partners, strangers, staff or family who are not in the cast). "
            "Places that would hold people (a club, a street, a restaurant) stay empty or have only a blurred, unnamed "
            "crowd that never speaks, acts or matters to the story. Every event happens to or through the cast. "
            f"{CAST_PRESENCE_RULE}")


# Keeps every picked character in the story to the end, the main character at its centre.
CAST_PRESENCE_RULE = (
    "The main character leads the story and is in nearly every part of it. Every other character has a clear part "
    "that carries through to the ending: they return in later chapters, and nobody drops out without the story "
    "showing why. The relationships listed with the cast decide how the characters treat each other, and the "
    "story shows them on screen through what they do together."
)


def story_medium_rule(video_style, lead_name=""):
    """How the story is told on screen, for the book and chapter prompts."""
    lead = lead_name or "the main character"
    if video_style == "spoken":
        return (f"This is a spoken story set to music. The characters talk: write the lines they say in quotes, and "
                f"let the story move through what {lead} and the others say and do. No one sings.")
    if video_style == "story_only":
        return ("This story plays out to the song with no one singing or speaking on camera. Everything is told "
                "through action, faces, places and objects.")
    return (f"This is the story of a music video. {lead} performs the song inside the story: write the moments where "
            f"{lead} sings, alone, to the camera, or to someone in the scene, woven into what is "
            "happening (singing in the rain on the street, singing alone in the empty room). Put the biggest of these "
            "moments on the chorus.")


def story_book_instruction(*, premise, cast_text, tone_notes, scene_count, song_seconds, lyric_guidance, lyrics,
                           previous_book, seed, allow_new_cast=False, video_style="music_video", lead_name="",
                           template=None, act_count=0):
    """Draft the whole story book from the user's premise, in ``act_count`` acts (0 lets the scene count decide)."""
    acts, paragraphs = story_book_size(scene_count, act_count)
    return (
        render_story_template(template or STORY_BOOK_TEMPLATE, {
            "medium_rule": story_medium_rule(video_style, lead_name), "acts": acts, "chapters": acts,
            "paragraphs": paragraphs,
            "scene_count": scene_count, "song_seconds": f"{song_seconds:.0f}",
            "cast_rule": story_cast_rule(cast_text, allow_new_cast), "lyric_guidance": lyric_guidance,
        })
        + "- Output only the book in exactly this format, with no notes before or after it:\n\n"
        f"{_BOOK_FORMAT}\n\n"
        f"Premise from the user:\n{premise or '[none, invent one from the cast and tone]'}\n\n"
        f"Cast:\n{cast_text or '[none, invent a small cast]'}\n\n"
        f"Tone, genre and world notes:\n{tone_notes or '[none]'}\n\n"
        "Song map in order (time, section, lyric). The timing and energy set the story's pace; the lyric words are "
        f"for mood and emotion only:\n{lyrics or '[none]'}\n\n"
        "Previous draft to move away from on this reroll (do not reuse anyone from it who is not in the cast):\n"
        f"{previous_book or '[none]'}\n\n"
        f"Variation seed: {seed or '[none]'}"
    )


def story_chapter_rewrite_instruction(*, book_text, chapter_number, chapter_text, direction, cast_text, lyric_guidance,
                                      allow_new_cast=False, video_style="music_video", lead_name="", template=None):
    """Rewrite or expand one chapter, keeping the rest of the book as context."""
    return (
        render_story_template(template or STORY_CHAPTER_TEMPLATE, {
            "chapter_number": chapter_number, "cast_rule": story_cast_rule(cast_text, allow_new_cast),
            "medium_rule": story_medium_rule(video_style, lead_name), "lyric_guidance": lyric_guidance,
        })
        + "- Output only the act in exactly this format:\n\n"
        f"Act {chapter_number}: <act title>\n\n"
        "<paragraph>\n\n"
        "<paragraph>\n\n"
        f"User direction:\n{direction or 'Improve and expand this act.'}\n\n"
        f"Cast:\n{cast_text or '[none]'}\n\n"
        f"Act to rewrite:\n{chapter_text}\n\n"
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


def story_remove_people_instruction(*, story_text, cast_text, names):
    """Rewrite a story (or one chapter) so the named people are gone and only the cast remains."""
    return (
        "You are a novelist editing a story that a music video will tell.\n\n"
        f"Rewrite the text so these people are gone: {', '.join(names)}. Afterwards only the cast below appears, acts, "
        "speaks or is named.\n\n"
        "Rules:\n"
        "- Keep every chapter in the same order, with its 'Chapter N: title' line and its 'Setting:' line unchanged. "
        "Keep the Title and Premise lines when the text has them; rewrite the premise only to drop removed people.\n"
        "- Keep the cast's actions and about the same length.\n"
        "- Where a removed person did something, the cast does it alone, or it shows through things: a message on a "
        "phone, an empty seat, a door closing, a voice off screen with no name.\n"
        "- No new people. A crowd may only be a blurred, unnamed background that never acts.\n"
        "- Output only the rewritten text in the same format, starting with its first line, with no notes.\n\n"
        f"Cast:\n{cast_text or '[none]'}\n\n"
        f"Text:\n{story_text}"
    )


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


STORY_ACTS_TEMPLATE = (
    "You are a music video director breaking one chapter of a story into acts.\n\n"
    "An act is one event inside the chapter (an argument, a phone call, leaving the house). It is a run of "
    "scenes that follow each other on the timeline. Acts stay in the chapter's setting; they may move between "
    "spots inside it.\n\n"
    "Rules:\n"
    "- About {{target_acts}} acts. Follow the chapter's own events in order. Every scene listed belongs to exactly one act.\n"
    "- first_scene is the scene number where the act starts. The first act starts at the chapter's first scene.\n"
    "- A timed user note must fall inside one act; never start an act in the middle of a note's scenes.\n"
    "- Bigger story moments can land on louder or longer scenes, but never change the story's order.\n"
    "- summary: 1-3 sentences of what visibly happens in the act.\n"
    "- cast: names from the cast list who appear in the act. The main character is in nearly every act. Keep "
    "the other characters in the acts where the chapter has them, so each one stays in the story to the end.\n"
    "- {{lyrics_rule}}\n"
)


def story_acts_instruction(*, chapter_number, chapter_title, setting_name, chapter_text, scenes_text, cast_text,
                           notes_text, target_acts, template=None):
    """Split one chapter into acts, each a contiguous run of the chapter's scenes."""
    return (
        render_story_template(template or STORY_ACTS_TEMPLATE,
                              {"target_acts": target_acts, "lyrics_rule": LYRICS_ARE_MOOD_ONLY})
        + "Return JSON only: {\"acts\": [{\"first_scene\": 1, \"title\": \"...\", \"summary\": \"...\", \"cast\": [\"...\"]}]}\n\n"
        f"Chapter {chapter_number}: {chapter_title}\n"
        f"Setting: {setting_name or '[not named]'}\n\n"
        f"{chapter_text}\n\n"
        f"Cast:\n{cast_text or '[none]'}\n\n"
        f"Timed user notes for these scenes:\n{notes_text or '[none]'}\n\n"
        f"Scenes of this chapter (time, length, song section, lyric mood):\n{scenes_text}"
    )


STORY_ACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "acts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "first_scene": {"type": "integer"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "cast": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["first_scene", "title", "summary", "cast"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["acts"],
    "additionalProperties": False,
}


_PLAN_PERFORMANCE_RULES = {
    "music_video": "Use singing for most scenes with vocals, so the performer sings through the video; use story "
                   "where the action needs the screen to itself.",
    "spoken": "Use speaking for scenes with vocals: a character says a line. Never singing.",
    "story_only": "Use story for every scene with vocals, and instrumental where there are none. Never singing or "
                  "speaking.",
}


STORY_SCENE_PLAN_TEMPLATE = (
    "You are a music video director planning each scene of one act.\n\n"
    "For every scene listed, write what happens in it, where in the setting it happens, its mood, how the "
    "performer appears, and who is in it and what they do with each other.\n\n"
    "Rules:\n"
    "- story_slice: 1-2 sentences of what visibly happens in this scene, continuing from the scene before. "
    "Together the slices tell the act in order.\n"
    "- cast: names from the cast list who are visibly in this scene. "
    "{{main_character_line}}"
    "This act's characters are {{act_cast}}; give each of them scenes in this act. "
    "Never name anyone outside the cast list.\n"
    "- performer: the one name from cast who sings or speaks in this scene, or an empty string when nobody "
    "does.\n"
    "- interaction: one sentence of what the characters in this scene do with or toward each other, shaped by "
    "their relationship (a glance, a touch, a turn away). Empty when only one character is in the scene.\n"
    "- A scene with a user note (\"user direction\") follows it exactly, including any character it adds and what "
    "they do.\n"
    "- spot: the specific place inside the setting \"{{setting_name}}\" (e.g. Kitchen, Bedroom, "
    "Front Porch). 1-3 words, no setting name. The spot comes from the story action, never from the lyrics. "
    "Stay at one spot for at least {{spot_min_scenes}} scenes in a row unless a user note moves it. "
    "Reuse these spot names when the place is the same: {{known_spots}}.\n"
    "- mood: 2-5 words of emotion.\n"
    "- performance: singing (the performer sings the lyric to the camera or in the moment), speaking (a "
    "character says a line), story (no lip sync, the story plays out) or instrumental (no vocals in this scene). "
    "Scenes marked instrumental must be instrumental. "
    "{{performance_rule}}\n"
    "- User notes on a scene are the user's direction and win over the story. Follow them exactly and write "
    "the slice to agree with them.\n"
    "- continues: true when this scene is the very next moment of the scene before it in one unbroken camera "
    "take: same spot, same people, the action carries straight on with no time jump. False when time jumps, "
    "the spot changes or a new moment starts.\n"
    "- continuation: one sentence, the physical action the performer does in this scene once the carried-on "
    "moment settles (e.g. \"She sets the letter down and walks to the window.\"). Write it for every scene.\n"
    "- {{lyric_guidance}}\n"
)


def story_scene_plan_instruction(*, act_title, act_summary, setting_name, chapter_text, scenes_text, cast_text,
                                 spot_min_scenes, previous_spot, lyric_guidance, known_spots, video_style="music_video",
                                 act_cast="", main_name="", template=None):
    """Give every scene of one act its slice of the story, a spot, a mood, a performance and its cast."""
    return (
        render_story_template(template or STORY_SCENE_PLAN_TEMPLATE, {
            "main_character_line": f"{main_name} is the main character and is in most scenes. " if main_name else "",
            "act_cast": act_cast or "the cast below", "setting_name": setting_name or "the setting",
            "spot_min_scenes": spot_min_scenes, "known_spots": known_spots or "[none yet]",
            "performance_rule": _PLAN_PERFORMANCE_RULES.get(video_style, _PLAN_PERFORMANCE_RULES["music_video"]),
            "lyric_guidance": lyric_guidance,
        })
        + "Return JSON only: {\"scenes\": [{\"scene\": 1, \"story_slice\": \"...\", \"spot\": \"...\", \"mood\": \"...\", "
        "\"performance\": \"singing\", \"continues\": false, \"continuation\": \"...\", \"cast\": [\"...\"], "
        "\"performer\": \"...\", \"interaction\": \"...\"}]}\n\n"
        f"Act: {act_title}\n{act_summary}\n\n"
        f"Spot of the scene before this act: {previous_spot or '[none]'}\n\n"
        f"Cast:\n{cast_text or '[none]'}\n\n"
        f"Chapter text for context:\n{chapter_text}\n\n"
        f"Scenes of this act:\n{scenes_text}"
    )


STORY_SCENE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer"},
                    "story_slice": {"type": "string"},
                    "spot": {"type": "string"},
                    "mood": {"type": "string"},
                    "performance": {"type": "string", "enum": ["singing", "speaking", "story", "instrumental"]},
                    "continues": {"type": "boolean"},
                    "continuation": {"type": "string"},
                    "cast": {"type": "array", "items": {"type": "string"}},
                    "performer": {"type": "string"},
                    "interaction": {"type": "string"},
                },
                "required": ["scene", "story_slice", "spot", "mood", "performance", "continues", "continuation",
                             "cast", "performer", "interaction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scenes"],
    "additionalProperties": False,
}


STORY_SETTING_LOOKS_TEMPLATE = (
    "You are a production designer for a music video that tells the story below.\n\n"
    "For each general setting, write its look: the one visual world every spot inside it shares.\n\n"
    "Rules:\n"
    "- look: 2-3 sentences of physical design only: era, region, architecture, materials, colour palette, "
    "light sources and quality, weather. No people, no actions, no story events.\n"
    "- The look must fit the story's world and the style notes. Obey every ban in the style notes.\n"
    "- Settings in the same story should feel like the same world.\n"
)


def story_setting_looks_instruction(*, settings_text, book_summary, style_theme, cast_text, template=None):
    """One shared look per general setting, so every spot inside it matches. No lyrics are sent."""
    return (
        render_story_template(template or STORY_SETTING_LOOKS_TEMPLATE, {})
        + "Return JSON only: {\"settings\": [{\"name\": \"...\", \"look\": \"...\"}]} with every setting listed below, "
        "names unchanged.\n\n"
        f"Style and theme notes (look and banned places):\n{style_theme or '[none]'}\n\n"
        f"Cast:\n{cast_text or '[none]'}\n\n"
        f"Story:\n{book_summary}\n\n"
        f"Settings:\n{settings_text}"
    )


STORY_SETTING_LOOKS_SCHEMA = {
    "type": "object",
    "properties": {
        "settings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "look": {"type": "string"}},
                "required": ["name", "look"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["settings"],
    "additionalProperties": False,
}


STORY_SPOTS_TEMPLATE = (
    "You are a music video location scout. Every spot below is a place inside the same setting, so all of them "
    "share the setting's look.\n\n"
    "For each spot write a description for its location reference card.\n\n"
    "Rules:\n"
    "- Exactly one sentence of 20-38 words.\n"
    "- Physical only: architecture, fixtures, surfaces, light sources, colours, weather, atmosphere. "
    "No people, clothing, actions or story events.\n"
    "- Start from the setting's look so every spot matches, then add what is specific to this spot.\n"
    "- The spot must suit what happens there in the story. Obey every ban in the style notes.\n"
)


def story_spots_instruction(*, setting_name, setting_look, spots_text, style_theme, template=None):
    """One location card per spot inside a setting, built on the setting's look. No lyrics are sent."""
    return (
        render_story_template(template or STORY_SPOTS_TEMPLATE, {})
        + "Return JSON only: {\"spots\": [{\"name\": \"...\", \"description\": \"...\"}]} with every spot listed below, "
        "names unchanged.\n\n"
        f"Setting: {setting_name}\n"
        f"Setting look: {setting_look or '[none yet]'}\n\n"
        f"Style and theme notes:\n{style_theme or '[none]'}\n\n"
        f"Spots and what happens there:\n{spots_text}"
    )


STORY_SPOTS_SCHEMA = {
    "type": "object",
    "properties": {
        "spots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                "required": ["name", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["spots"],
    "additionalProperties": False,
}


STORY_REVISE_SCENE_TEMPLATE = (
    "You are a story editor adapting a music video's story, the way a book is adapted into a film: the user has "
    "changed Scene {{scene_number}}, and the story around it must now agree with that change.\n\n"
    "Rules:\n"
    "- The user's change to Scene {{scene_number}} is fixed. Keep the characters they put in it and do exactly what "
    "their direction says. Write the scene to match it.\n"
    "- Change the act and the chapter only as much as the new scene needs. Keep every other event, the order of "
    "events, the setting and the tone. A character the user added gets a reason to be there that fits the story.\n"
    "- chapter_paragraphs: the whole chapter again, as plain prose paragraphs, with the change woven in. Same "
    "rules as the book: things that visibly happen, no camera or song-section words.\n"
    "- other_scenes: the other scenes of this act whose plan no longer fits after the change (a character who "
    "must now be there, an event that moved). Give each a short reason. Leave out scenes that still fit.\n"
    "- {{cast_rule}}\n"
    "- {{lyric_guidance}}\n"
)


def story_revise_scene_instruction(*, book_text, chapter_number, chapter_title, setting_name, chapter_text, act_title,
                                   act_summary, act_scenes_text, scene_number, scene_cast, user_note, cast_text,
                                   lyric_guidance, template=None):
    """Change one scene the way the user wants, and adapt the act and chapter around it as little as possible."""
    return (
        render_story_template(template or STORY_REVISE_SCENE_TEMPLATE, {
            "scene_number": scene_number, "cast_rule": story_cast_rule(cast_text, False), "lyric_guidance": lyric_guidance,
        })
        + "Return JSON only: {\"scene\": {\"story_slice\": \"...\", \"spot\": \"...\", \"mood\": \"...\", \"performer\": "
        "\"...\", \"interaction\": \"...\"}, \"act_summary\": \"...\", \"chapter_paragraphs\": [\"...\"], "
        "\"other_scenes\": [{\"scene\": 3, \"reason\": \"...\"}]}\n\n"
        f"User's change to Scene {scene_number}:\n"
        f"- Characters in the scene: {scene_cast or '[as planned]'}\n"
        f"- Direction: {user_note or '[none, only the characters changed]'}\n\n"
        f"Cast:\n{cast_text or '[none]'}\n\n"
        f"Chapter {chapter_number}: {chapter_title}\nSetting: {setting_name or '[not named]'}\n\n{chapter_text}\n\n"
        f"Act: {act_title}\n{act_summary}\n\n"
        f"Scenes of this act (scene, spot, story slice, characters):\n{act_scenes_text}\n\n"
        f"Whole story for context:\n{book_text}"
    )


STORY_REVISE_SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "object",
            "properties": {
                "story_slice": {"type": "string"},
                "spot": {"type": "string"},
                "mood": {"type": "string"},
                "performer": {"type": "string"},
                "interaction": {"type": "string"},
            },
            "required": ["story_slice", "spot", "mood", "performer", "interaction"],
            "additionalProperties": False,
        },
        "act_summary": {"type": "string"},
        "chapter_paragraphs": {"type": "array", "items": {"type": "string"}},
        "other_scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"scene": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["scene", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scene", "act_summary", "chapter_paragraphs", "other_scenes"],
    "additionalProperties": False,
}
