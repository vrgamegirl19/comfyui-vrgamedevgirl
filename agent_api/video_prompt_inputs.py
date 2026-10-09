"""Scene inputs for the Agent API's video prompt jobs.

Like the image writers, the video prompt writers (``llm/video_prompt_generation.py``) take the scene's
T2I prompt, motion notes and reference text in the request. This is the Python twin of what the Builder
sends (``videoGemmaNotesForSegment`` and the payloads in ``web/music_video_builder/batch_prompts.mjs``), in
its text-only form: no reference image is attached unless the caller sends ``image_reference_path``.
"""

from typing import Any, Dict

from ..builder.lyric_scenes import is_instrumental_lyric_text
from .errors import ValidationError
from .image_prompt_inputs import _find_segment, _reference_lines, _text, scene_card_for
from .scene_card_context import storyboard_video_context

_T2I_FIELDS = ("t2i_prompt", "flux_prompt", "nb_prompt", "flow_gpt_prompt", "ernie_t2i_prompt")
_NO_CHARACTER_NOTE = (
    "Subject visibility: no main character is present in this scene. Do not include, mention, show, imply, "
    "or describe the mapped character/subject/performer. Build the shot from the location, props, "
    "environment, objects, atmosphere, and camera motion instead."
)
_VISUAL_ONLY_NOTE = (
    "Video Type: no lip sync / visual-only. Do not make any visible subject sing, speak, say dialogue, "
    "lip-sync, or move their mouth to the lyric. Use the lyric only as hidden mood/story context, and focus "
    "on visual acting, camera motion, environmental motion, dancing, posing, walking, or atmosphere."
)
_INSTRUMENTAL_NOTE = (
    "Lyric/performance status: instrumental / no sung lyrics. In the final prompt, do not mention singing, "
    "lip-syncing, mouth movement, instrumental status, or no-vocal status. Use visual acting, camera motion, "
    "environmental motion, dancing, posing, walking, or atmosphere instead."
)


def normalize_video_mode(mode: Any) -> str:
    """``t2v`` stays ``t2v``. Every other value (``i2v``, ``minimax_*``) is written like an I2V prompt."""
    return "t2v" if _text(mode).lower() == "t2v" else "i2v"


def _video_type(session: Dict[str, Any]) -> str:
    raw = _text(session.get("video_type") or session.get("videoType")).lower().replace("-", "_").replace(" ", "_")
    return "speaking" if raw in {"speaking", "short_film", "dialogue", "dialog"} else "singing"


def _performance_mode(session: Dict[str, Any], segment: Dict[str, Any]) -> str:
    style = _text(segment.get("performance_style") or segment.get("song_style")).lower().replace("-", " ").replace("_", " ")
    visual_only = bool(segment.get("lyric_no_lip_sync")) or any(
        word in style for word in ("no vocal", "no lip sync", "b roll", "broll", "visual only")
    )
    return "no_lip_sync" if visual_only else _video_type(session)


def _performance_note(segment: Dict[str, Any], performance: str) -> str:
    if segment.get("no_character_present"):
        return ""
    lyric = _text(segment.get("lyric_text") or segment.get("lyrics"))
    singers = [_text(name) for name in segment.get("lyric_singers") or [] if _text(name)]
    if performance == "no_lip_sync" or not lyric or is_instrumental_lyric_text(lyric):
        return ""
    who = ", ".join(singers) if singers else "the visible performer"
    if performance == "speaking":
        return (
            f"Video Type: speaking / short film. {who} should say the dialogue line naturally. "
            "Do not use singing, rapping, vocals, lyrics, or music-performance wording."
        )
    return (
        f"Vocal/performance direction: {who} should perform as if singing in sync with the audio. The exact "
        "lyric text will be inserted into the final prompt automatically. Do not describe visible singing as "
        "quiet; use controlled, focused, intimate, restrained, inward, tender, or simmering intensity instead."
    )


def _scene_text_prompt(segment: Dict[str, Any], subject: str, location_name: str, location_text: str) -> str:
    """Scene text standing in for a T2I prompt (``sceneVideoConceptPromptText``) when none is saved."""
    parts = []

    def add(title: str, value: Any) -> None:
        if _text(value):
            parts.append(f"{title}:\n{_text(value)}")

    add("Prompt summary", segment.get("prompt_summary") or segment.get("summary"))
    add("Scene notes", segment.get("notes") or segment.get("director_note"))
    add("Scene story beat", segment.get("story_beat"))
    if not segment.get("lyric_no_lip_sync"):
        add("Lyrics / scene text", segment.get("lyric_text") or segment.get("lyrics"))
    add("Mapped subject / character", subject)
    add("Mapped location", location_name)
    add("Shot type", segment.get("shot_type"))
    add("Reference location description", location_text)
    return "\n\n".join(parts)


def scene_video_prompt_inputs(session: Dict[str, Any], scene_id: Any, mode: Any = "i2v") -> Dict[str, Any]:
    """Request fields the video prompt writers need for one scene, built from the saved project."""
    from .orchestrator.storyboard_orchestrator import scene_cards  # imported late: it imports the job layer

    video_mode = normalize_video_mode(mode)
    segment = _find_segment(session, scene_id)
    card = scene_card_for(session, segment, scene_cards)
    no_character = bool(segment.get("no_character_present"))
    subject = "" if no_character else _reference_lines(card.get("subject_refs") or [])
    location = card.get("location_ref") or {}
    location_name, location_text = _text(location.get("name")), _text(location.get("description"))
    location_context = "\n".join(part for part in (location_name, location_text) if part)

    t2i_prompt = next((_text(segment.get(field)) for field in _T2I_FIELDS if _text(segment.get(field))), "")
    if not t2i_prompt:
        t2i_prompt = _scene_text_prompt(segment, subject, location_name, location_text)
    if not t2i_prompt:
        raise ValidationError(
            f"Scene '{scene_id}' has no T2I prompt, notes, lyric or mapped references to write a video prompt from. "
            "Write its image prompt first or add scene notes."
        )

    performance = _performance_mode(session, segment)
    performance_note = _performance_note(segment, performance)
    instrumental = (
        not no_character and performance != "no_lip_sync" and not performance_note
        and is_instrumental_lyric_text(_text(segment.get("lyric_text") or segment.get("lyrics")))
    )
    notes = "\n\n".join(part for part in (
        _NO_CHARACTER_NOTE if no_character else "",
        _VISUAL_ONLY_NOTE if performance == "no_lip_sync" else "",
        performance_note,
        _INSTRUMENTAL_NOTE if instrumental else "",
        _text(segment.get("i2v_notes")),
        # The scene card's directions and the complete card, as the Storyboard sends them with a video prompt.
        storyboard_video_context(card) if card else "",
    ) if part)
    return {
        "t2i_prompt": t2i_prompt,
        "user_notes": notes,
        "builder_instruction_key": video_mode,
        "performance_mode": performance,
        "subject_context": subject,
        "location_context": location_context,
        "no_character_present": no_character,
        "lyric_text": _text(segment.get("lyric_text") or segment.get("lyrics")),
        "lyric_section": _text(segment.get("lyric_section")),
        "story_beat": _text(segment.get("story_beat")),
        "scene_notes": _text(segment.get("notes")),
        "director_note": _text(segment.get("timeline_note") or segment.get("director_note")),
        "image_reference_path": "",
        "image_reference_data": "",
        "use_vision": False,
    }
