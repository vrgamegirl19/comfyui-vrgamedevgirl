"""Editable scene-card fields and where each one is saved.

A scene card is shown in two places: the Video Builder timeline (the session's ``segments``) and the
Storyboard Builder (``storyboard/storyboard.json``). The Storyboard's Save copies a card onto its segment in
``applyStoryboardPrompts`` (``web/music_video_builder/storyboard_bridge.mjs``) and reads linked cards back
from the timeline in ``loadExisting`` (``web/storyboard_builder/persistence.mjs``). ``SCENE_CARD_FIELDS`` is
the Python twin of that mapping, so an API edit writes the same keys a UI edit writes:

- Director Notes -> ``timeline_note`` on both
- Video Notes -> ``i2v_notes`` (and ``video_notes``) on the segment, ``motion_summary`` on the card
- Planning Notes -> ``notes`` on both
- Lyrics -> ``lyric_text`` on the segment, ``lyrics`` on the card

Values are kept exactly as sent, so an empty string, ``false`` or ``0`` clears or disables a field instead
of falling back to a default. Pure module: no ComfyUI imports.
"""

import copy
import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

PERFORMANCE_MODES = ("singing", "speaking", "no_lip_sync")
VIDEO_PROMPT_TYPES = ("i2v", "id_lora", "t2v", "rtv", "ingredients", "flf")
VIDEO_PROMPT_ORIGINS = ("gemma", "manual")
TRIGGER_POSITIONS = ("start", "end")
SPEAKER_CUE_TYPES = ("dialogue", "instrumental")


class SceneCardField(NamedTuple):
    """One editable field. ``segment_keys`` may be empty (card only) and ``card_key`` may be None."""

    name: str
    kind: str
    segment_keys: Tuple[str, ...]
    card_key: Optional[str]
    aliases: Tuple[str, ...] = ()
    choices: Tuple[str, ...] = ()
    limit: int = 20000


def _text_field(name: str, card_key: Optional[str] = None, segment_keys: Optional[Tuple[str, ...]] = None,
                aliases: Tuple[str, ...] = (), limit: int = 20000) -> SceneCardField:
    return SceneCardField(name, "text", segment_keys if segment_keys is not None else (name,),
                          card_key if card_key is not None else name, aliases, (), limit)


def _flag(name: str, card_key: Optional[str] = None) -> SceneCardField:
    return SceneCardField(name, "bool", (name,), card_key or name)


SCENE_CARD_FIELDS: Tuple[SceneCardField, ...] = (
    _text_field("label", limit=180),
    _text_field("lyric_text", "lyrics", aliases=("lyrics",), limit=4000),
    _text_field("lyric_section", limit=160),
    _text_field("story_beat", limit=1800),
    # Director Notes (the Scene Note shown on the timeline; the Builder's agent calls it director_note).
    _text_field("timeline_note", aliases=("director_note", "director_notes"), limit=4000),
    # Video Notes / motion direction.
    _text_field("i2v_notes", "motion_summary", ("i2v_notes", "video_notes"),
                aliases=("video_notes", "motion_summary"), limit=3000),
    # Planning Notes.
    _text_field("notes", aliases=("planning_notes",), limit=4000),
    # The card's Summary (Video Prep). Saved on the Storyboard card only, like the UI.
    SceneCardField("prompt_summary", "text", (), "prompt_summary", ("summary",), (), 1000),
    # Camera and character.
    _text_field("shot_type", limit=200),
    _text_field("camera_motion", limit=200),
    _text_field("character_motion", limit=240),
    # Performance and facial direction.
    SceneCardField("performance_mode", "choice", ("performance_mode",), "performance_mode", (), PERFORMANCE_MODES),
    _text_field("performance_style", limit=120),
    _text_field("facial_performance", limit=120),
    _text_field("facial_performance_custom", limit=1200),
    _flag("include_microphone"),
    # Audio and continuity.
    _text_field("audio_direction", limit=4000),
    _text_field("continuity", aliases=("continuity_direction",), limit=4000),
    _text_field("flf_start_state", limit=4000),
    _text_field("flf_transformation", limit=4000),
    _text_field("flf_end_state", limit=4000),
    _text_field("flf_carry_forward", limit=4000),
    # Look.
    _text_field("video_style", "video_style", ("minimax_h3_video_style",), limit=160),
    _text_field("video_style_custom", "video_style_custom", ("minimax_h3_video_style_custom",), limit=3000),
    SceneCardField("temporal_world_effect_override", "text", ("temporal_world_effect_override",),
                   "temporal_world_effect_override", (), (), 120),
    _text_field("temporal_world_effect_custom", limit=3000),
    # Card-only trigger for the scene's still prompt.
    SceneCardField("trigger_phrase", "text", (), "trigger_phrase", (), (), 1200),
    SceneCardField("trigger_position", "choice", (), "trigger_position", (), TRIGGER_POSITIONS),
    # Characters, lyrics and dialogue.
    _flag("no_character_present"),
    _flag("lyric_no_lip_sync"),
    _flag("lyric_instrumental"),
    SceneCardField("lyric_singers", "strings", ("lyric_singers",), "lyric_singers"),
    SceneCardField("lyric_cue_map", "objects", ("lyric_cue_map",), "lyric_cue_map"),
    _flag("lyric_shot_word_timing_enabled"),
    _text_field("lyric_performance_mode", limit=40),
    SceneCardField("speaker_assignments", "speakers", ("minimax_speaker_assignments",), "speaker_assignments",
                   ("minimax_speaker_assignments",)),
    SceneCardField("video_prompt_type", "choice", ("video_prompt_type",), "video_prompt_type", (), VIDEO_PROMPT_TYPES),
    # Prompts. ``image_prompt`` and ``video_prompt`` follow the Builder's prompt editing rules
    # (``setSegmentPromptForEdit``); the older single-field names stay available below.
    SceneCardField("image_prompt", "image_prompt", ("t2i_prompt",), "image_prompt", (), (), 12000),
    SceneCardField("video_prompt", "video_prompt", (), "video_prompt", (), (), 100000),
    SceneCardField("video_prompt_origin", "choice", (), "video_prompt_origin", (), VIDEO_PROMPT_ORIGINS),
    _text_field("minimax_h3_pass2_prompt", limit=100000),
    # References: the scene's mapped characters and location (the Reference Builder's scene maps).
    SceneCardField("subject_ids", "references", (), "subject_refs"),
    SceneCardField("location_id", "reference", (), "location_ref"),
)

FIELDS_BY_NAME: Dict[str, SceneCardField] = {field.name: field for field in SCENE_CARD_FIELDS}
_ALIASES: Dict[str, str] = {alias: field.name for field in SCENE_CARD_FIELDS for alias in field.aliases}


class SceneCardFieldError(ValueError):
    """A scene-card value with the wrong type or an unknown choice."""


def canonical_name(key: str) -> Optional[str]:
    """The field a request key names (its own name or an alias), or None."""
    if key in FIELDS_BY_NAME:
        return key
    return _ALIASES.get(key)


def field_names() -> List[str]:
    """Every accepted field name and alias, sorted."""
    return sorted(set(FIELDS_BY_NAME) | set(_ALIASES))


def _clean_text(field: SceneCardField, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise SceneCardFieldError(f"{field.name} must be text.")
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > field.limit:
        raise SceneCardFieldError(f"{field.name} is longer than {field.limit} characters.")
    return text


def _normalize_choice(field: SceneCardField, value: Any) -> str:
    text = str(value if value is not None else "").strip().lower().replace("-", "_").replace(" ", "_")
    if text not in field.choices:
        raise SceneCardFieldError(f"{field.name} must be one of: {', '.join(field.choices)}.")
    return text


def normalize_speaker_assignments(value: Any) -> List[Dict[str, Any]]:
    """Twin of ``normalizeMiniMaxSpeakerAssignments`` (``minimax_h3.mjs``) without random ids."""
    if not isinstance(value, list):
        raise SceneCardFieldError("speaker_assignments must be a list of {speaker_id, speaker_name, text} objects.")
    cues = []
    for index, item in enumerate(value[:40]):
        if not isinstance(item, dict):
            raise SceneCardFieldError("Each speaker assignment must be an object.")

        def seconds(raw: Any) -> Optional[float]:
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return None
            return max(0.0, number) if math.isfinite(number) else None

        cue_type = str(item.get("type") or item.get("kind") or "").strip().lower()
        cues.append({
            "id": str(item.get("id") or item.get("cue_id") or item.get("cueId") or f"speaker_cue_{index + 1}"),
            "type": "instrumental" if cue_type == "instrumental" else "dialogue",
            "speaker_id": str(item.get("speaker_id") or item.get("speakerId") or item.get("subject_id") or ""),
            "speaker_name": str(
                item.get("speaker_name") or item.get("speakerName") or item.get("speaker") or ""
            ).strip(),
            "text": str(item.get("text") or item.get("dialogue") or item.get("line") or "").strip(),
            "action_note": str(item.get("action_note") or item.get("actionNote") or "").strip(),
            "start": seconds(item.get("start")),
            "end": seconds(item.get("end")),
        })
    return cues


def normalize_value(field: SceneCardField, value: Any) -> Any:
    """The value saved for ``field``. Empty strings, ``False`` and empty lists are kept as given."""
    if field.kind in ("text", "image_prompt", "video_prompt", "reference"):
        text = _clean_text(field, value)
        if field.name == "temporal_world_effect_override" and not text:
            return "global"  # the editor saves an empty override as "global"
        return text
    if field.kind == "choice":
        return _normalize_choice(field, value)
    if field.kind == "bool":
        if not isinstance(value, bool):
            raise SceneCardFieldError(f"{field.name} must be true or false.")
        return value
    if field.kind in ("strings", "references"):
        items = [value] if isinstance(value, str) else value
        if items is None:
            items = []
        if not isinstance(items, list) or any(
            not isinstance(item, (str, int)) or isinstance(item, bool) for item in items
        ):
            raise SceneCardFieldError(f"{field.name} must be a list of names or ids.")
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return list(dict.fromkeys(cleaned))
    if field.kind == "objects":
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise SceneCardFieldError(f"{field.name} must be a list of objects.")
        return copy.deepcopy(value)
    if field.kind == "speakers":
        return normalize_speaker_assignments(value)
    raise SceneCardFieldError(f"{field.name} cannot be edited.")  # pragma: no cover - table is fixed


def parse_scene_card_patch(patch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a scene patch into normalized scene-card values and the keys this table does not know.

    Raises ``SceneCardFieldError`` for a bad value or when a field and its alias disagree.
    """
    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    rest: Dict[str, Any] = {}
    for key, raw in patch.items():
        name = canonical_name(key)
        if name is None:
            rest[key] = raw
            continue
        value = normalize_value(FIELDS_BY_NAME[name], raw)
        if name in values and values[name] != value:
            raise SceneCardFieldError(f"{sources[name]} and {key} set the same field to different values.")
        values[name] = value
        sources.setdefault(name, key)
    return values, rest


def image_prompt_keys(image_mode: str) -> Tuple[str, ...]:
    """Segment keys ``setSegmentPromptForEdit(segment, "t2i", ...)`` writes for the project's image model."""
    mode = str(image_mode or "zimage").strip().lower()
    if mode == "flow_gpt":
        return ("t2i_prompt", "flow_gpt_prompt", "nb_prompt")
    if mode == "nano_banana":
        return ("t2i_prompt", "nb_prompt")
    if mode == "flux_klein":
        return ("t2i_prompt", "flux_prompt")
    return ("t2i_prompt", "flux_prompt", "nb_prompt", "flow_gpt_prompt")


def video_prompt_keys(video_engine: str) -> Tuple[str, str]:
    """The segment's video prompt key and its origin key for the project's video engine."""
    if str(video_engine or "").strip().lower() == "minimax_h3":
        return ("minimax_h3_prompt", "minimax_h3_prompt_origin")
    return ("i2v_prompt", "i2v_prompt_origin")


def apply_to_segment(segment: Dict[str, Any], values: Dict[str, Any], *, video_engine: str = "",
                     image_mode: str = "zimage") -> None:
    """Write scene-card values onto a timeline segment with the keys the Builder uses."""
    for name, value in values.items():
        field = FIELDS_BY_NAME[name]
        if field.kind == "image_prompt":
            for key in image_prompt_keys(image_mode):
                segment[key] = value
        elif field.kind == "video_prompt":
            prompt_key, origin_key = video_prompt_keys(video_engine)
            segment[prompt_key] = value
            segment[origin_key] = values.get("video_prompt_origin", "manual")
        elif name == "video_prompt_origin":
            if "video_prompt" not in values:
                segment[video_prompt_keys(video_engine)[1]] = value
        elif field.kind == "speakers":
            segment["minimax_speaker_assignments"] = copy.deepcopy(value)
            # syncMiniMaxSpeakerAssignmentLegacyFields: a filled dialogue plan becomes the lyric and singers.
            filled = [cue for cue in value if cue["text"]]
            if filled and "lyric_text" not in values:
                segment["lyric_text"] = "\n".join(cue["text"] for cue in filled)
            if filled and "lyric_singers" not in values:
                names = (cue["speaker_name"] for cue in filled if cue["speaker_name"])
                segment["lyric_singers"] = list(dict.fromkeys(names))
        else:
            for key in field.segment_keys:
                segment[key] = copy.deepcopy(value)


def apply_to_card(card: Dict[str, Any], values: Dict[str, Any]) -> None:
    """Write scene-card values onto a saved Storyboard card. References are resolved by the caller."""
    for name, value in values.items():
        field = FIELDS_BY_NAME[name]
        if field.kind in ("references", "reference") or field.card_key is None:
            continue
        card[field.card_key] = copy.deepcopy(value)
    if "video_prompt" in values and "video_prompt_origin" not in values:
        card["video_prompt_origin"] = "manual"
    if "speaker_assignments" in values:
        filled = [cue for cue in values["speaker_assignments"] if cue["text"]]
        if filled and "lyric_text" not in values:
            card["lyrics"] = "\n".join(cue["text"] for cue in filled)
        if filled and "lyric_singers" not in values:
            card["lyric_singers"] = list(dict.fromkeys(cue["speaker_name"] for cue in filled if cue["speaker_name"]))


def card_only_names(values: Dict[str, Any]) -> List[str]:
    """Fields in ``values`` that only the Storyboard card stores."""
    return [name for name in values
            if not FIELDS_BY_NAME[name].segment_keys
            and FIELDS_BY_NAME[name].kind not in ("video_prompt", "references", "reference")
            and name != "video_prompt_origin"]
