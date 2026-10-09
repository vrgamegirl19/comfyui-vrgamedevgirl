"""Storyboard story layer for agents: scene defaults, story arc, story brief and scene beats.

These are the Storyboard Builder's "Create Story Arc", "Create Story Brief" and "Create Missing Scene
Beats" buttons. The story idea comes from the caller (the agent). Everything else is written by the
project's LLM through the same functions the Storyboard uses (``storyboard.story_layer``). For LM Studio
the model that is already loaded is used; the API never loads or switches a model (see ``llm_runtime``).

State is saved where the UI keeps it: ``builder_story_layer``, ``builder_storyboard_defaults`` and the
``story_beat`` field of each timeline segment.
"""

import math
import re
from typing import Any, Callable, Dict, List, Optional

from ...storyboard import persistence as storyboard_store
from ...storyboard import session_sync
from ...storyboard import story_layer as story_funcs
from ..errors import RevisionConflictError, ValidationError
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..llm_runtime import llm_payload_from_session, prepare_llm_payload
from ..mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session, scene_field_change
from ..project_events import notify_project_changed
from ..scene_card_context import beat_scene

# Twins of the keys the Builder saves: normalizeBuilderStoryLayer and normalizeBuilderStoryboardDefaults in
# web/music_video_builder/model_settings.mjs. tests/test_agent_api_story_settings_fields.py fails when they drift.
STORY_LAYER_KEYS = ("enabled", "overall_story_idea", "user_story_arc", "song_story_brief", "lyric_story_strength",
                    "image_world_style", "image_custom_style_direction")
IMAGE_WORLD_STYLES = ("natural", "surreal_subject", "balanced_surreal", "full_surreal", "abstract", "custom")
DEFAULT_KEYS = (
    "global_consistency_phrase", "camera_motion_speed", "character_motion_speed", "minimax_h3_cut_frequency",
    "camera_guidance", "character_guidance", "performance_style", "short_film_planning_mode", "camera_flow",
    "custom_camera_flow_sequence", "image_shot_flow", "image_aesthetic", "video_style", "video_style_custom",
    "temporal_world_effect", "temporal_world_effect_custom", "temporal_allow_background_extras",
    "temporal_background_intensity", "temporal_environment_time_passage", "temporal_protected_characters",
    "temporal_protected_custom", "fx_preset", "fx_custom_json",
)
SPEED_KEYS = ("camera_motion_speed", "character_motion_speed", "temporal_background_intensity")
BOOLEAN_DEFAULT_KEYS = ("temporal_allow_background_extras", "temporal_environment_time_passage")
TEMPORAL_PROTECTED_CHARACTERS = ("all_referenced", "lead_only", "custom")
SHORT_FILM_PLANNING_MODES = ("guided_film", "fully_custom")
STORY_ARC_DETAILS = ("compact", "standard", "detailed", "rich")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _story_layer(session: Dict[str, Any]) -> Dict[str, Any]:
    stored = session.get("builder_story_layer") or session.get("builderStoryLayer")
    layer = dict(stored) if isinstance(stored, dict) else {}
    layer.setdefault("enabled", True)
    for key in ("overall_story_idea", "user_story_arc", "song_story_brief", "image_custom_style_direction"):
        layer[key] = _text(layer.get(key))
    layer["lyric_story_strength"] = layer.get("lyric_story_strength", 7)
    layer["image_world_style"] = layer.get("image_world_style") or "natural"
    return layer


def _save_story_layer(session: Dict[str, Any], layer: Dict[str, Any]) -> None:
    session["builder_story_layer"] = layer
    session.pop("builderStoryLayer", None)


def _number_0_to_10(value: Any, key: str) -> float:
    """A 0-10 slider value (the Builder's speed, intensity and strength sliders). Out of range is an error."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not 0 <= number <= 10:  # also false for NaN
        raise ValidationError(f"{key} must be a number from 0 to 10.")
    return number


def _defaults(session: Dict[str, Any]) -> Dict[str, Any]:
    stored = session.get("builder_storyboard_defaults")
    return stored if isinstance(stored, dict) else {}


# ---------------------------------------------------------------------------
# Scene cards (the Storyboard's view of the timeline)
# ---------------------------------------------------------------------------

def scene_cards(session: Dict[str, Any], folder: str = "") -> List[Dict[str, Any]]:
    """One complete Storyboard scene card per timeline scene.

    With ``folder`` the cards also carry what only the Storyboard saves (Summary, triggers, speaker plan),
    merged the way the Storyboard merges them when it opens.
    """
    cards = session_sync.scene_cards(session)
    if not folder:
        return cards
    saved = storyboard_store._load_storyboard({"project_folder": folder})
    if not saved.get("exists"):
        return cards
    merged = {card.get("id"): card for card in session_sync.merge_scene_cards(saved, cards)}
    return [merged.get(card["id"], card) for card in cards]


def _storyboard_summary(session: Dict[str, Any], cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The slim storyboard block the arc and brief generators read (``slimStoryboardForRequest``)."""
    defaults = _defaults(session)
    refs = session.get("flux_reference_builder") if isinstance(session.get("flux_reference_builder"), dict) else {}
    return {
        "mode": "video",
        "project_video_engine": "minimax_h3" if _text(session.get("video_engine")) == "minimax_h3" else _text(session.get("video_engine")),
        "camera_flow": defaults.get("camera_flow") or "balanced",
        "video_style": defaults.get("video_style") or "",
        "video_style_custom": defaults.get("video_style_custom") or "",
        "global_consistency_phrase": defaults.get("global_consistency_phrase") or "",
        "camera_motion_speed": defaults.get("camera_motion_speed", 4),
        "character_motion_speed": defaults.get("character_motion_speed", 4),
        "story_arc_detail": defaults.get("story_arc_detail") or "standard",
        "minimax_h3_cut_frequency": defaults.get("minimax_h3_cut_frequency", 0),
        "performance_style_default": defaults.get("performance_style") or "",
        "story_layer": _story_layer(session),
        "reference_builder": {
            "subjects": [session_sync._slim_reference(s) for s in refs.get("subjects") or [] if isinstance(s, dict)],
            "locations": [session_sync._slim_reference(l) for l in refs.get("locations") or [] if isinstance(l, dict)],
        },
        "scenes": cards,
    }


def _lyrics_blocks(session: Dict[str, Any], cards: List[Dict[str, Any]]) -> Dict[str, str]:
    mapper = session.get("lyric_mapper") if isinstance(session.get("lyric_mapper"), dict) else {}
    return {
        "lyrics": "\n".join(c["lyrics"] for c in cards if c["lyrics"]),
        "line_mapping_lyrics": _text(mapper.get("source_text")),
    }


def _require_scenes(cards: List[Dict[str, Any]]) -> None:
    if not cards:
        raise ValidationError("The project has no scenes. Create the timeline first.")


def sync_storyboard_files(project_id: str) -> Dict[str, Any]:
    """Save the Storyboard Builder's copy of the project and export the prompt files, like "Save Storyboard".

    The Storyboard shows its video prompts, story beats and status from ``storyboard/storyboard.json``
    (not from the session), and the Video Builder writes ``prompts/*.txt|json`` from it. Without this
    the Storyboard lists every scene as having no video prompt. Failures are reported, not raised:
    the story or prompt step itself has already succeeded.
    """
    try:
        with _BUILDER_SAVE_LOCK:
            folder, session = _get_active_session_and_folder(project_id)
            cards = session_sync.scene_cards(session)
            settings = session.get("minimax_h3_settings")
            settings = settings if isinstance(settings, dict) else {}
            for card in cards:
                card["minimax_h3_audio_mode"] = _text(settings.get("audio_mode")) or "input_audio"
            saved = storyboard_store._load_storyboard({"project_folder": folder})
            # Merge into the saved copy: Storyboard-only cards, deleted cards, Summary, triggers and
            # storyboard settings stay as the user left them.
            storyboard = session_sync.merge_storyboard(saved if saved.get("exists") else None, session, cards)
            storyboard.setdefault("story_layer", _story_layer(session))
            result = storyboard_store._export_storyboard_prompts({"project_folder": folder, "storyboard": storyboard})
        scenes = storyboard["scenes"]
        notify_project_changed(folder, session, {"kind": "storyboard", "storyboard": True})
        with_prompt = sum(1 for scene in scenes if _text(scene.get("video_prompt")))
        return {"saved": True, "scenes": len(scenes), "with_video_prompt": with_prompt,
                "path": result.get("storyboard_path") if isinstance(result, dict) else ""}
    except Exception as exc:  # the step that called this has already succeeded
        print(f"[VRGDG API] Storyboard sync failed: {exc}")
        return {"saved": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Scene defaults and the story idea
# ---------------------------------------------------------------------------

def set_story_settings(
    project_id: str,
    params: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Save Storyboard scene defaults (video style, camera flow, motion speeds) and the story idea.

    ``defaults`` holds scene-default fields, ``story`` holds story-layer fields. Both are optional.
    ``if_match_revision`` (the If-Match header) must equal the saved revision when given.
    """
    defaults_in = params.get("defaults") if isinstance(params.get("defaults"), dict) else {}
    story_in = params.get("story") if isinstance(params.get("story"), dict) else {}
    unknown = sorted(set(defaults_in) - set(DEFAULT_KEYS) - {"story_arc_detail"})
    if unknown:
        raise ValidationError(f"Unknown scene default(s): {', '.join(unknown)}. Allowed: {', '.join(DEFAULT_KEYS)}, story_arc_detail.")
    unknown_story = sorted(set(story_in) - set(STORY_LAYER_KEYS))
    if unknown_story:
        raise ValidationError(f"Unknown story field(s): {', '.join(unknown_story)}. Allowed: {', '.join(STORY_LAYER_KEYS)}.")
    for key in SPEED_KEYS:
        if key in defaults_in:
            defaults_in[key] = _number_0_to_10(defaults_in[key], key)
    for key in BOOLEAN_DEFAULT_KEYS:
        if key in defaults_in and not isinstance(defaults_in[key], bool):
            raise ValidationError(f"{key} must be true or false.")
    if "enabled" in story_in and not isinstance(story_in["enabled"], bool):
        raise ValidationError("enabled must be true or false.")
    if defaults_in.get("temporal_protected_characters") not in (None, *TEMPORAL_PROTECTED_CHARACTERS):
        raise ValidationError(f"temporal_protected_characters must be one of: {', '.join(TEMPORAL_PROTECTED_CHARACTERS)}.")
    if "short_film_planning_mode" in defaults_in:
        # Same spelling rules as normalizeMiniMaxShortFilmPlanningMode (minimax_h3.mjs); "custom" means fully_custom.
        mode = re.sub(r"[\s-]+", "_", _text(defaults_in["short_film_planning_mode"]).lower())
        mode = "fully_custom" if mode == "custom" else mode
        if mode not in SHORT_FILM_PLANNING_MODES:
            raise ValidationError(f"short_film_planning_mode must be one of: {', '.join(SHORT_FILM_PLANNING_MODES)}.")
        defaults_in["short_film_planning_mode"] = mode
    if defaults_in.get("story_arc_detail") not in (None, *STORY_ARC_DETAILS):
        raise ValidationError(f"story_arc_detail must be one of: {', '.join(STORY_ARC_DETAILS)}.")
    if story_in.get("image_world_style") not in (None, *IMAGE_WORLD_STYLES):
        raise ValidationError(f"image_world_style must be one of: {', '.join(IMAGE_WORLD_STYLES)}.")
    if "lyric_story_strength" in story_in:
        story_in["lyric_story_strength"] = _number_0_to_10(story_in["lyric_story_strength"], "lyric_story_strength")

    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)
        defaults = dict(_defaults(session))
        defaults.update(defaults_in)
        session["builder_storyboard_defaults"] = defaults
        layer = _story_layer(session)
        layer.update(story_in)
        _save_story_layer(session, layer)
        saved = _persist_session(folder, session)
    return {"defaults": defaults, "story": layer, "revision": saved.get("revision")}


# ---------------------------------------------------------------------------
# LLM steps
# ---------------------------------------------------------------------------

def _llm_request(session: Dict[str, Any], params: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    passthrough = {k: v for k, v in params.items() if k in ("temperature", "top_p", "max_new_tokens", "seed")}
    request = {**llm_payload_from_session(session), **extra, **passthrough, "unload_after": False}
    return prepare_llm_payload(request)


def create_story_arc(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create the story arc with the LLM from the story idea and the lyrics, and save it."""
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    _require_scenes(cards)
    layer = _story_layer(session)
    idea = _text(params.get("story_idea")) or layer["overall_story_idea"]
    if not idea:
        raise ValidationError("A story idea is required. Pass story_idea or save one with the story settings.")
    layer["overall_story_idea"] = idea
    storyboard = _storyboard_summary(session, cards)
    storyboard["story_layer"] = {**layer, "user_story_arc": ""}
    defaults = _defaults(session)
    request = _llm_request(
        session, params,
        n_ctx=max(16384, int(llm_payload_from_session(session).get("n_ctx") or 0)),
        max_new_tokens=int(params.get("max_new_tokens") or 2400),
        story_layer=storyboard["story_layer"],
        storyboard=storyboard,
        story_idea=idea,
        previous_story_arc=_text(params.get("previous_story_arc")),
        timeline_markers=session.get("timeline_markers") or [],
        project_folder=folder,
        scenes=cards,
        camera_flow=defaults.get("camera_flow") or "balanced",
        camera_motion_speed=defaults.get("camera_motion_speed", 4),
        character_motion_speed=defaults.get("character_motion_speed", 4),
        story_arc_detail=params.get("story_arc_detail") or defaults.get("story_arc_detail") or "standard",
        performance_style=defaults.get("performance_style") or "",
        **_lyrics_blocks(session, cards),
    )
    result = story_funcs._build_story_layer_arc(request)
    arc = _text(result.get("story_arc"))
    if not arc:
        raise ValidationError("The LLM returned an empty story arc.")
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        layer = _story_layer(session)
        layer.update({"overall_story_idea": idea, "user_story_arc": arc})
        _save_story_layer(session, layer)
        saved = _persist_session(folder, session)
    return {"story_arc": arc, "model": result.get("used_model") or request.get("lmstudio_model"), "storyboard": sync_storyboard_files(project_id),
            "revision": saved.get("revision")}


def create_story_brief(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create the song story brief with the LLM from the lyrics and the story arc, and save it."""
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    _require_scenes(cards)
    layer = _story_layer(session)
    request = _llm_request(
        session, params,
        max_new_tokens=int(params.get("max_new_tokens") or 1200),
        story_layer=layer,
        storyboard=_storyboard_summary(session, cards),
        reference_builder=_storyboard_summary(session, cards)["reference_builder"],
        scenes=cards,
        lyrics=_lyrics_blocks(session, cards)["lyrics"],
    )
    result = story_funcs._build_story_layer_brief(request)
    brief = _text(result.get("story_brief"))
    if not brief:
        raise ValidationError("The LLM returned an empty story brief.")
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        layer = _story_layer(session)
        layer["song_story_brief"] = brief
        _save_story_layer(session, layer)
        saved = _persist_session(folder, session)
    return {"story_brief": brief, "model": result.get("used_model") or request.get("lmstudio_model"), "storyboard": sync_storyboard_files(project_id),
            "revision": saved.get("revision")}


def _adjacent_line(text: str, last: bool) -> str:
    lines = [line.strip() for line in _text(text).splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1] if last else lines[0]


def create_scene_beats(
    project_id: str,
    params: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Write a story beat for each scene with the LLM, in timeline order, saving after every scene.

    By default only scenes with no beat are written. ``replace_existing`` rewrites all of them and
    ``scene_ids`` or ``limit`` narrow the set (``limit`` counts from the first scene that needs a beat).
    """
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session, folder)
    _require_scenes(cards)
    layer = _story_layer(session)
    if not (layer["user_story_arc"] or layer["song_story_brief"] or layer["overall_story_idea"]):
        raise ValidationError("Create the story arc (and brief) before scene beats.")
    wanted = {str(x) for x in (params.get("scene_ids") or [])}
    replace = bool(params.get("replace_existing"))
    targets = [
        c for c in cards
        if (not wanted or str(c["id"]) in wanted or str(c["scene_number"]) in wanted) and (replace or wanted or not c["story_beat"])
    ]
    limit = int(params.get("limit") or 0)
    if limit > 0:
        targets = targets[:limit]
    if not targets:
        return {"created": 0, "skipped": len(cards), "beats": [], "revision": session.get("revision")}

    refs = session.get("flux_reference_builder") if isinstance(session.get("flux_reference_builder"), dict) else {}
    all_subjects = [{"name": _text(s.get("name")), "description": _text(s.get("description"))} for s in refs.get("subjects") or [] if isinstance(s, dict)]
    by_number = {c["scene_number"]: c for c in cards}
    beats: List[Dict[str, Any]] = []
    revision = session.get("revision")
    for done, card in enumerate(targets):
        number = card["scene_number"]
        previous, following = by_number.get(number - 1), by_number.get(number + 1)
        if progress:
            progress(done, len(targets), f"Scene {number}")
        request = _llm_request(
            session, params,
            max_new_tokens=int(params.get("max_new_tokens") or 360),
            story_layer=layer,
            all_subjects=all_subjects,
            # The complete scene card, as the Storyboard sends it (Director Note, notes, camera, performance...).
            storyboard_payload={"scenes": [beat_scene(card)], "selected_scene_number": number, "story_layer": layer},
            previous_beat=(previous or {}).get("story_beat", ""),
            previous_lyrics=_adjacent_line((previous or {}).get("lyrics", ""), True),
            current_lyrics=card["lyrics"],
            next_lyrics=_adjacent_line((following or {}).get("lyrics", ""), False),
            temperature=float(params.get("temperature") or 0.35),
            top_p=float(params.get("top_p") or 0.90),
        )
        result = story_funcs._build_story_layer_scene_beat(request)
        beat = _text(result.get("story_beat"))
        if not beat:
            raise ValidationError(f"The LLM returned an empty story beat for scene {number}.")
        card["story_beat"] = beat
        with _BUILDER_SAVE_LOCK:
            folder, session = _get_active_session_and_folder(project_id)
            for segment in session.get("segments") or []:
                if isinstance(segment, dict) and segment.get("id") == card["id"]:
                    segment["story_beat"] = beat
            revision = _persist_session(folder, session, scene_field_change(card["id"], ["story_beat"])).get("revision")
        beats.append({"scene_id": card["id"], "scene_number": number, "story_beat": beat})
    if progress:
        progress(len(targets), len(targets), "done")
    return {"created": len(beats), "skipped": len(cards) - len(beats), "beats": beats, "storyboard": sync_storyboard_files(project_id),
            "revision": revision}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def run_story_arc_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "story_arc", message="Writing the story arc with the LLM...")
    return create_story_arc(job.project_id, job.params)


def run_story_brief_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "story_brief", message="Writing the story brief with the LLM...")
    return create_story_brief(job.project_id, job.params)


def run_scene_beats_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    def report(done: int, total: int, label: str) -> None:
        manager.update_progress(job.id, 5.0 + 90.0 * done / max(1, total), "scene_beats", message=f"Scene beats {done}/{total} ({label})")
    return create_scene_beats(job.project_id, job.params, progress=report)


def register_storyboard_orchestrator_handlers(manager: Optional[JobManager] = None) -> None:
    manager = manager or get_job_manager()
    manager.register_handler("storyboard.story_arc", run_story_arc_job)
    manager.register_handler("storyboard.story_brief", run_story_brief_job)
    manager.register_handler("storyboard.scene_beats", run_scene_beats_job)
