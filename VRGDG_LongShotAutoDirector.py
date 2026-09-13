"""Plan a complete LongShot story once, then extract consistent chunk instructions."""

from __future__ import annotations

import json
import math
import re


STORY_MODES = (
    "random cinematic story",
    "user-directed story",
    "reference-inspired story",
    "movie trailer",
    "advertisement",
    "cartoon short",
    "music video",
    "horror sequence",
    "comedy sketch",
    "action sequence",
    "documentary teaser",
    "surreal art film",
)

GENRES = (
    "automatic",
    "drama",
    "thriller",
    "horror",
    "science fiction",
    "fantasy",
    "romance",
    "comedy",
    "action",
    "mystery",
)

VISUAL_STYLES = (
    "automatic",
    "live action",
    "2D cartoon",
    "3D animation",
    "anime",
    "stop motion",
    "graphic novel",
)

REFERENCE_SETUPS = (
    "no reference images",
    "image 1 only",
    "images 1 and 2",
)

REFERENCE_ROLES = (
    "protagonist",
    "second character",
    "multi-character cast sheet",
    "location",
    "product",
    "creature or prop",
    "visual style inspiration",
)

DIALOGUE_AMOUNTS = (
    "none",
    "light",
    "normal",
    "heavy",
)

AUDIO_MODES = (
    "built-in audio",
    "custom audio",
)


MODE_BRIEFS = {
    "random cinematic story": "Invent an original compact cinematic story with a clear turn and ending.",
    "user-directed story": "Treat the user's idea as binding and complete any unspecified details creatively.",
    "reference-inspired story": "Let the visible reference subjects or setting inspire an original story while preserving their recognizable details.",
    "movie trailer": "Create a miniature trailer arc with escalation, memorable trailer moments, and a final hook; avoid on-screen title cards unless requested.",
    "advertisement": "Build a polished product-centered commercial with a problem, benefit demonstration, and memorable finish; avoid unsupported factual claims and visible text unless requested.",
    "cartoon short": "Create an expressive animated short with readable silhouettes, visual comedy or emotion, and consistent stylization.",
    "music video": "Create a performance or narrative music-video concept driven by rhythm and evolving visual choreography.",
    "horror sequence": "Build dread through environment, staging, sound, and escalation before a strong reveal or unresolved final beat.",
    "comedy sketch": "Set up a clear comic premise, escalate it visually, and land a final reaction or punch line without overcrowding the timing.",
    "action sequence": "Create spatially readable action with a clear objective, escalating obstacles, grounded movement, and continuous geography.",
    "documentary teaser": "Create an observational or investigative teaser with concrete imagery, restrained narration or interview dialogue, and a compelling question.",
    "surreal art film": "Create a visually coherent dream logic with recurring motifs and transformations that still preserve chunk-to-chunk continuity.",
}


def _clean(value) -> str:
    return str(value or "").strip()


def dialogue_word_budget(chunk_duration: float, amount: str) -> int:
    rates = {"none": 0.0, "light": 1.1, "normal": 1.8, "heavy": 2.2}
    if amount not in rates:
        raise ValueError(f"Unknown dialogue_amount: {amount}")
    return max(0, math.floor(float(chunk_duration) * rates[amount]))


def _reference_contract(setup: str, role_1: str, role_2: str) -> str:
    if setup == "no reference images":
        return (
            "No images are attached. Invent original characters, wardrobe, locations, props, "
            "and visual style, then describe them precisely enough to remain stable."
        )
    lines = [
        f"Image 1 is attached as: {role_1}. Analyze only visible details and preserve them in the plan."
    ]
    if setup == "images 1 and 2":
        lines.append(
            f"Image 2 is attached as: {role_2}. Analyze only visible details and preserve them in the plan."
        )
    lines.append(
        "These are planning references. Do not claim to see details that are not visible. The two "
        "H3 reference slots may later receive these same images. If multiple characters are needed "
        "with a location reference, prefer one multi-character cast sheet plus one location image."
    )
    return "\n".join(lines)


class VRGDG_LongShotAutoDirectorContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_mode": (list(STORY_MODES), {"default": "random cinematic story"}),
                "user_idea": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "genre": (list(GENRES), {"default": "automatic"}),
                "visual_style": (list(VISUAL_STYLES), {"default": "automatic"}),
                "tone": (
                    "STRING",
                    {"default": "cinematic, emotionally engaging", "multiline": False},
                ),
                "dialogue_amount": (list(DIALOGUE_AMOUNTS), {"default": "normal"}),
                "audio_mode": (list(AUDIO_MODES), {"default": "built-in audio"}),
                "allow_cuts": ("BOOLEAN", {"default": False}),
                "total_chunks": ("INT", {"default": 4, "min": 1, "max": 16}),
                "chunk_duration": (
                    "FLOAT",
                    {"default": 7.5, "min": 1.0, "max": 60.0, "step": 0.1},
                ),
                "reference_setup": (
                    list(REFERENCE_SETUPS),
                    {"default": "no reference images"},
                ),
                "image_1_role": (list(REFERENCE_ROLES), {"default": "protagonist"}),
                "image_2_role": (list(REFERENCE_ROLES), {"default": "second character"}),
                "creative_seed": ("INT", {"default": 1, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("planner_instructions",)
    FUNCTION = "build"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = (
        "Creates one multimodal LLM request for a complete LongShot story plan. Connect its "
        "output to VRGDG LLM Multi and attach up to two planning reference images there."
    )

    def build(
        self,
        story_mode,
        user_idea,
        genre,
        visual_style,
        tone,
        dialogue_amount,
        audio_mode,
        allow_cuts,
        total_chunks,
        chunk_duration,
        reference_setup,
        image_1_role,
        image_2_role,
        creative_seed,
    ):
        if story_mode not in STORY_MODES:
            raise ValueError(f"Unknown story_mode: {story_mode}")
        if genre not in GENRES or visual_style not in VISUAL_STYLES:
            raise ValueError("Unknown genre or visual_style.")
        if reference_setup not in REFERENCE_SETUPS:
            raise ValueError(f"Unknown reference_setup: {reference_setup}")
        if audio_mode not in AUDIO_MODES:
            raise ValueError(f"Unknown audio_mode: {audio_mode}")

        total_chunks = int(total_chunks)
        chunk_duration = float(chunk_duration)
        total_duration = total_chunks * chunk_duration
        word_budget = dialogue_word_budget(chunk_duration, dialogue_amount)
        idea = _clean(user_idea) or "No user premise was supplied; invent the concept completely."
        edit_rule = (
            "Motivated cuts are allowed inside chunks."
            if allow_cuts
            else "Every chunk must be one uninterrupted physical camera take with no cuts or hidden edits."
        )
        audio_rule = (
            "The final videos use H3 built-in audio. Invent exact dialogue and sound appropriate "
            "to each chunk; the later chunk writer will use the three-field built-in-audio format."
            if audio_mode == "built-in audio"
            else "The final videos use a separately supplied continuous soundtrack. Plan exact words "
            "now, but they must later be recorded or generated into that soundtrack before video generation."
        )

        prompt = f"""You are the Auto Director for a connected MiniMax H3 LongShot video.

Create ONE complete story plan before any chunk prompts are written. All chunks must belong to the same story, preserve the same character IDs, voices, wardrobe, world geography, visual style, and emotional progression, and hand moving action naturally from one chunk to the next.

FORMAT MODE
{story_mode}: {MODE_BRIEFS[story_mode]}

USER IDEA
{idea}

PROJECT SETTINGS
- Genre: {genre}
- Visual style: {visual_style}
- Tone: {_clean(tone) or 'automatic'}
- Creative seed: {int(creative_seed)}. Use it only to choose a repeatable creative direction; do not mention the number in the story.
- {total_chunks} chunks × {chunk_duration:.3f} seconds = {total_duration:.3f} seconds total.
- Audio mode: {audio_mode}. {audio_rule}
- Editing: {edit_rule}
- Dialogue amount: {dialogue_amount}. Each chunk may contain at most {word_budget} newly spoken words TOTAL across every speaker. Count contractions as one word. Leave enough time for pauses, reactions, movement, and breathing. Never place more dialogue than can be performed naturally within {chunk_duration:.3f} seconds.

REFERENCE IMAGES
{_reference_contract(reference_setup, image_1_role, image_2_role)}

STORY RULES
- Give the video a beginning, escalation, turn, and satisfying final beat across the available duration.
- Prefer two principal speaking characters at most. Additional people may appear but should remain silent unless timing clearly permits.
- Assign permanent IDs S1, S2, and so on. Give every speaking character a distinctive visible description and voice. Never exchange their lines or voices.
- Use adult characters unless the user explicitly requests otherwise. Keep content suitable for a mainstream film or commercial unless the user requests a different rating.
- Write exact dialogue once, assigned to its speaker. No captions, subtitles, title cards, logos, or readable text unless the user explicitly requests them.
- Plan concrete acting, interaction, environmental events, and camera movement. Avoid four chunks of people merely standing and talking.
- For no-image generation, Chunk 1 starts from text only and later chunks continue from generated motion history. For reference-led generation, describe how the references are used without treating them as start/end keyframes.
- Every chunk ending_state must be a precise visual and moving handoff for the following chunk. Do not use a cut as a substitute for continuity when cuts are disabled.

OUTPUT
Return only valid JSON. Do not use Markdown fences or add commentary. Use exactly this schema and produce exactly {total_chunks} chunk objects:
{{
  "title": "short title",
  "logline": "one-sentence premise",
  "format_mode": "{story_mode}",
  "visual_bible": "concise medium, palette, lighting, lens and world rules",
  "setting_bible": "concise stable geography and important props",
  "characters": [
    {{"id": "S1", "name": "name or role", "visual_description": "stable visible traits", "voice_description": "stable voice"}}
  ],
  "audio_bible": "stable ambience, voice and music continuity",
  "chunks": [
    {{
      "chunk_number": 1,
      "purpose": "story function",
      "opening_state": "exact opening visual and ongoing motion",
      "action": "chronological performance and environmental action",
      "camera": "one clear physical camera trajectory",
      "dialogue": [{{"speaker": "S1", "text": "exact words"}}],
      "sound": "new synchronized ambience and physical sounds",
      "ending_state": "precise pose, geography, emotion and camera motion handed forward"
    }}
  ]
}}

Before returning JSON, verify the chunk count, stable speaker IDs, dialogue word count for every chunk, chronological continuity, and valid JSON syntax."""
        return (prompt,)


def _json_object(text: str) -> dict:
    raw = _clean(text)
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The Auto Director response does not contain a JSON object.")
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The Auto Director returned invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("The Auto Director plan must be a JSON object.")
    return value


def _characters(plan: dict) -> tuple[str, dict[str, str]]:
    lines = []
    names = {}
    for character in plan.get("characters", []) or []:
        if not isinstance(character, dict):
            continue
        speaker_id = _clean(character.get("id")).upper()
        if not speaker_id:
            continue
        name = _clean(character.get("name")) or speaker_id
        names[speaker_id] = name
        visual = _clean(character.get("visual_description")) or "appearance unspecified"
        voice = _clean(character.get("voice_description")) or "voice unspecified"
        lines.append(f"{name} ({speaker_id}): {visual}. Voice: {voice}.")
    return "\n".join(lines) or "No character bible supplied.", names


def _dialogue_text(dialogue, names: dict[str, str]) -> str:
    if not dialogue:
        return ""
    if isinstance(dialogue, str):
        return dialogue.strip()
    lines = []
    for entry in dialogue:
        if not isinstance(entry, dict):
            continue
        speaker = _clean(entry.get("speaker")).upper()
        words = _clean(entry.get("text"))
        if not words:
            continue
        label = names.get(speaker, speaker or "SPEAKER")
        lines.append(f"{label} ({speaker}): {words}" if speaker else f"{label}: {words}")
    return "\n".join(lines)


class VRGDG_LongShotPlanExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "director_plan": ("STRING", {"forceInput": True, "multiline": True}),
                "chunk_number": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "video_idea",
        "chunk_direction",
        "dialogue_this_chunk",
        "previous_dialogue_carryover",
        "character_bible",
        "plan_status",
    )
    FUNCTION = "extract"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = "Extracts one chunk's existing LongShot LLM inputs from an Auto Director JSON plan."

    def extract(self, director_plan, chunk_number):
        plan = _json_object(director_plan)
        chunk_number = int(chunk_number)
        chunks = plan.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("The Auto Director plan has no chunks array.")
        by_number = {
            int(chunk.get("chunk_number", index + 1)): chunk
            for index, chunk in enumerate(chunks)
            if isinstance(chunk, dict)
        }
        if chunk_number not in by_number:
            raise ValueError(f"The Auto Director plan does not contain chunk {chunk_number}.")

        character_bible, names = _characters(plan)
        chunk = by_number[chunk_number]
        previous_chunk = by_number.get(chunk_number - 1)
        dialogue = _dialogue_text(chunk.get("dialogue"), names)
        previous_dialogue = ""
        if previous_chunk:
            previous_lines = _dialogue_text(previous_chunk.get("dialogue"), names).splitlines()
            previous_dialogue = previous_lines[-1] if previous_lines else ""

        video_idea = "\n".join(filter(None, (
            f"Title: {_clean(plan.get('title'))}",
            f"Premise: {_clean(plan.get('logline'))}",
            f"Format: {_clean(plan.get('format_mode'))}",
            f"Visual bible: {_clean(plan.get('visual_bible'))}",
            f"Setting bible: {_clean(plan.get('setting_bible'))}",
            f"Audio bible: {_clean(plan.get('audio_bible'))}",
            "Characters:\n" + character_bible,
        )))
        chunk_direction = "\n".join(filter(None, (
            f"Purpose: {_clean(chunk.get('purpose'))}",
            f"Opening state: {_clean(chunk.get('opening_state'))}",
            f"Action: {_clean(chunk.get('action'))}",
            f"Camera: {_clean(chunk.get('camera'))}",
            f"Sound: {_clean(chunk.get('sound'))}",
            f"Ending state: {_clean(chunk.get('ending_state'))}",
        )))
        status = (
            f"Extracted chunk {chunk_number} of {len(chunks)}: "
            f"{len(dialogue.split())} dialogue tokens in formatted text."
        )
        return video_idea, chunk_direction, dialogue, previous_dialogue, character_bible, status


NODE_CLASS_MAPPINGS = {
    "VRGDG_LongShotAutoDirectorContext": VRGDG_LongShotAutoDirectorContext,
    "VRGDG_LongShotPlanExtractor": VRGDG_LongShotPlanExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LongShotAutoDirectorContext": "VRGDG LongShot Auto Director Context",
    "VRGDG_LongShotPlanExtractor": "VRGDG LongShot Plan Extractor",
}

