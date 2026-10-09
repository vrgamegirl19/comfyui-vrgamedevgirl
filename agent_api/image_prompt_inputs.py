"""Scene inputs for the Agent API's image prompt jobs.

The image prompt writers (``llm/image_prompt_generation.py``) do not read the project: they take the
scene's notes and reference text in the request, and the Video Builder composes those from the scene before
it calls them. This module is the Python twin of that step (``textOnlyFallbackNotesForSegment`` and the
payload around it in ``web/music_video_builder/image_prompts.mjs``), so an API job sends the same inputs the
Builder does instead of an empty request.
"""

from typing import Any, Dict, List

from ..builder.lyric_scenes import is_instrumental_lyric_text
from .errors import ValidationError
from .scene_card_context import scene_card_block

# Same mapping as builderImageInstructionKey in image_prompts.mjs, so a saved custom instruction applies.
IMAGE_INSTRUCTION_KEYS = {
    "zimage": "zimage_t2i",
    "ernie_image": "ernie_t2i",
    "krea2_2pass": "krea2_t2i",
    "flux_klein": "flux_klein_t2i",
    "nano_banana": "nano_b_t2i",
    "flow_gpt": "flow_gpt_t2i",
}
_MODE_ALIASES = {"flux": "flux_klein", "nb": "nano_banana", "z_image": "zimage"}

IMAGE_PREP_RULE = (
    "Create a still text-to-image prompt. The final answer must be a visual image prompt, not the lyric line. "
    "Use lyrics only for mood, symbolism, emotion, styling, and visual direction. Do not quote or return the "
    "lyrics as the prompt. Do not say the subject is singing, lip-syncing, performing vocals, or singing the "
    "lyric unless scene notes explicitly request a live singing image."
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def normalize_image_mode(mode: Any) -> str:
    """The Builder's image model key for an API ``mode`` value (default ``zimage``)."""
    key = _text(mode).lower().replace("-", "_") or "zimage"
    key = _MODE_ALIASES.get(key, key)
    return key if key in IMAGE_INSTRUCTION_KEYS else "zimage"


def _find_segment(session: Dict[str, Any], scene_id: Any) -> Dict[str, Any]:
    segments = [s for s in session.get("segments") or [] if isinstance(s, dict)]
    segments.sort(key=lambda s: float(s.get("start") or 0.0))
    for index, segment in enumerate(segments):
        if _text(segment.get("id")) == _text(scene_id) or str(index + 1) == _text(scene_id):
            return segment
    raise ValidationError(f"Scene '{scene_id}' was not found in the project.")


def scene_card_for(session: Dict[str, Any], segment: Dict[str, Any], scene_cards: Any) -> Dict[str, Any]:
    """The segment's complete scene card, including what only the saved Storyboard keeps."""
    folder = _text(session.get("project_folder"))
    try:
        cards = scene_cards(session, folder)
    except (OSError, ValueError):  # an unreadable storyboard.json: the timeline fields are still known
        cards = scene_cards(session)
    return next((item for item in cards if _text(item.get("id")) == _text(segment.get("id"))), {})


def _reference_lines(refs: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{_text(ref.get('name'))}: {_text(ref.get('description'))}" if _text(ref.get("description")) else _text(ref.get("name"))
        for ref in refs if _text(ref.get("name"))
    )


def scene_image_notes(session: Dict[str, Any], segment: Dict[str, Any], card: Dict[str, Any], image_mode: str) -> str:
    """The notes text the Builder sends for a still image prompt (``textOnlyFallbackNotesForSegment``)."""
    parts: List[str] = []

    def add(title: str, value: Any) -> None:
        text = _text(value)
        if text:
            parts.append(f"{title}:\n{text}")

    refs = card.get("subject_refs") or []
    location = card.get("location_ref") or {}
    lyric = _text(segment.get("lyric_text") or segment.get("lyrics"))
    add("Scene notes", segment.get("notes"))
    add("Director note (timeline)", segment.get("timeline_note"))
    add("Flux/Klein notes", segment.get("flux_notes"))
    add("NanoBanana notes", segment.get("nb_notes"))
    add("Mapped subject / character", ", ".join(_text(ref.get("name")) for ref in refs if _text(ref.get("name"))))
    add("Mapped location", _text(location.get("name")))
    add("Lyric line as still-image mood context", "" if is_instrumental_lyric_text(lyric) else lyric)
    add("Lyric section", segment.get("lyric_section"))
    add("Scene story beat", segment.get("story_beat"))
    add("Still shot direction", segment.get("shot_type"))
    add("Reference subject description", _reference_lines(refs) if not segment.get("no_character_present") else "")
    add("Reference location name", location.get("name"))
    add("Reference location description", location.get("description"))
    layer = session.get("builder_story_layer") or session.get("builderStoryLayer")
    if isinstance(layer, dict) and layer.get("enabled") is not False:
        add("User story arc", layer.get("user_story_arc"))
        add("Song story brief", layer.get("song_story_brief"))
        add("Lyric story strength", f"{layer.get('lyric_story_strength', 7)}/10")
    if not parts:
        parts.append(f"Scene:\n{_text(card.get('label')) or _text(segment.get('label')) or 'This scene'}")
        parts.append("Direction:\nCreate a cinematic image prompt that fits this scene.")
    if card:
        # The complete scene card, as the Storyboard sends it; the instruction keeps it a still image.
        parts.append(scene_card_block(card))
    parts.append(f"Image Prep rule:\n{IMAGE_PREP_RULE}")
    return "\n\n".join(parts)


def scene_image_prompt_inputs(session: Dict[str, Any], scene_id: Any, mode: Any = "zimage") -> Dict[str, Any]:
    """Request fields the image prompt writers need for one scene, built from the saved project."""
    from .orchestrator.storyboard_orchestrator import scene_cards  # imported late: it imports the job layer

    image_mode = normalize_image_mode(mode)
    segment = _find_segment(session, scene_id)
    card = scene_card_for(session, segment, scene_cards)
    refs = card.get("subject_refs") or []
    location = card.get("location_ref") or {}
    subject_description = _reference_lines(refs) if not segment.get("no_character_present") else ""
    reference_context = {
        "subject_description": subject_description,
        "location_name": _text(location.get("name")),
        "location_description": _text(location.get("description")),
    }
    if image_mode == "nano_banana":
        reference_context["has_subject_reference"] = bool(subject_description)
        reference_context["has_location_reference"] = bool(reference_context["location_name"] or reference_context["location_description"])
    return {
        "user_notes": scene_image_notes(session, segment, card, image_mode),
        "lyric_text": _text(segment.get("lyric_text") or segment.get("lyrics")),
        "scene_number": int(card.get("scene_number") or 0),
        "prompt_mode": image_mode,
        "builder_instruction_key": IMAGE_INSTRUCTION_KEYS[image_mode],
        "reference_context": reference_context,
        "no_character_present": bool(segment.get("no_character_present")),
        "use_vision": False,
        "ref_image_path": "",
    }


def merge_request_overrides(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """The scene inputs, with any of the same fields the caller sent in the request winning."""
    merged = dict(inputs)
    for key in inputs:
        value = (params or {}).get(key)
        if value not in (None, ""):
            merged[key] = value
    return merged
