"""Default instructions for editable ElevenLabs dialogue delivery."""

ELEVENLABS_DIALOGUE_INSTRUCTIONS = """Prepare a single character's dialogue for ElevenLabs v4 or v3.
Return only the dialogue text with sparse, short audio tags in square brackets.
Examples: [curious], [whispers], [shouts], [sighs], [laughs], [crying].
Place a tag immediately before the phrase it should affect. Match the user's delivery direction and character context.
Do not add speaker labels, markdown, explanations, visual stage directions, SSML, sound effects or additional speakers.
Preserve the spoken words exactly unless the request explicitly allows rewriting. You may adjust punctuation for delivery.
Preserve useful existing audio tags and avoid tagging every sentence. Tags guide delivery; they do not guarantee it.
When rewriting is allowed, preserve meaning and keep the scene concise. Output must fit the selected model's character limit.
"""
