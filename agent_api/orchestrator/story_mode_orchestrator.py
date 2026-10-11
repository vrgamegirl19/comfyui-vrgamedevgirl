"""Story Mode steps for agents: the story book, chapter rewrites, pasted stories and the cast scan.

Same functions the Story Builder uses (``storyboard.story_book``). The LLM is the one already loaded; the API
never loads or switches a model. Results are saved in ``builder_story_layer`` like the Builder saves them.
"""

from typing import Any, Callable, Dict, List, Optional

from ...storyboard import story_book as book_funcs
from ...storyboard import story_locations as location_funcs
from ...storyboard import story_plan as plan_funcs
from ...storyboard.story_acts import validate_story_acts
from ...storyboard.story_mode import (
    apply_story_book_text,
    fill_scene_cast,
    normalize_story_book,
    normalize_story_mode_fields,
    plan_segment_fields,
    story_book_to_text,
)
from ..errors import ValidationError
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session, scene_field_change
from .reference_orchestrator import _new_location_id, location_style_theme_text
from .storyboard_orchestrator import (
    _llm_request,
    _lyrics_blocks,
    _require_scenes,
    _save_story_layer,
    _story_layer,
    _storyboard_summary,
    scene_cards,
    sync_storyboard_files,
)

STORY_MODE_STEPS = ("book", "chapter", "split", "convert", "cast", "acts", "plan", "locations")
# Story Mode fields an agent may send to PUT /story/settings. Each is checked before it is saved.
STORY_MODE_EDIT_KEYS = ("story_book_text", "story_settings", "story_acts", "story_scene_plan", "story_spot_min_scenes")


def _request(session: Dict[str, Any], params: Dict[str, Any], cards, **extra: Any) -> Dict[str, Any]:
    summary = _storyboard_summary(session, cards)
    refs = session.get("flux_reference_builder") if isinstance(session.get("flux_reference_builder"), dict) else {}
    return _llm_request(
        session, params,
        story_layer=_story_layer(session),
        scenes=cards,
        reference_builder={**summary["reference_builder"], "extras": refs.get("extras") or []},
        lyrics=_lyrics_blocks(session, cards)["lyrics"],
        **extra,
    )


def apply_plan_to_segments(session: Dict[str, Any], plan: List[Dict[str, Any]]) -> List[str]:
    """Write each planned scene's performance to its timeline segment. Returns the changed scene ids."""
    fields_by_scene = {entry["scene_id"]: plan_segment_fields(entry.get("performance_mode")) for entry in plan}
    changed = []
    for segment in session.get("segments") or []:
        if not isinstance(segment, dict) or segment.get("id") not in fields_by_scene:
            continue
        fields = fields_by_scene[segment["id"]]
        if any(segment.get(key) != value for key, value in fields.items()):
            segment.update(fields)
            changed.append(segment["id"])
    return changed


def fill_cast_from_acts(session: Dict[str, Any], acts: List[Dict[str, Any]]) -> List[str]:
    """Map each act's cast onto its scenes that have no characters yet (the Reference Builder's subject_scene_map)."""
    refs = session.setdefault("flux_reference_builder", {})
    subject_ids = [s.get("id") for s in refs.get("subjects") or [] if isinstance(s, dict) and s.get("id")]
    mapped, changed = fill_scene_cast(refs.get("subject_scene_map") or {}, acts, subject_ids)
    if changed:
        refs["subject_scene_map"] = mapped
        refs["use_subject_reference"] = True
    return changed


def _save_fields(project_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        layer = _story_layer(session)
        layer.update(fields)
        _save_story_layer(session, layer)
        changed = apply_plan_to_segments(session, fields["story_scene_plan"]) if fields.get("story_scene_plan") else []
        if fields.get("story_scene_plan"):
            fill_cast_from_acts(session, layer.get("story_acts") or [])
        if len(changed) == 1:
            saved = _persist_session(folder, session, scene_field_change(changed[0], ["performance_mode", "lyric_no_lip_sync"]))
        else:
            saved = _persist_session(folder, session)
    return {"revision": saved.get("revision"), "storyboard": sync_storyboard_files(project_id)}


def create_story_book(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write the story book from the premise (``story_idea``, else the saved idea) and save it."""
    params = dict(params or {})
    _folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    _require_scenes(cards)
    request = _request(
        session, params, cards,
        story_idea=params.get("story_idea") or "",
        tone_notes=params.get("tone_notes") or "",
        previous_book=params.get("previous_book") or _story_layer(session).get("story_book_text") or "",
        max_new_tokens=int(params.get("max_new_tokens") or 6000),
    )
    result = book_funcs.draft_story_book(request)
    saved = _save_fields(project_id, result["story_layer"])
    return {"story_book": result["story_book"], "story_book_text": result["story_layer"]["story_book_text"],
            "story_settings": result["story_layer"]["story_settings"], "model": result.get("used_model"), **saved}


def rewrite_story_chapter(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rewrite one chapter (``chapter_id``) following ``direction`` and save it."""
    params = dict(params or {})
    if not str(params.get("chapter_id") or "").strip():
        raise ValidationError("chapter_id is required (see story_book.chapters[].id from GET /projects/{pid}/story).")
    _folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    request = _request(session, params, cards, chapter_id=params["chapter_id"], direction=params.get("direction") or "",
                       max_new_tokens=int(params.get("max_new_tokens") or 2400))
    try:
        result = book_funcs.rewrite_story_chapter(request)
    except ValueError as error:
        raise ValidationError(str(error)) from error
    saved = _save_fields(project_id, result["story_layer"])
    return {"chapter": result["chapter"], "story_book_text": result["story_layer"]["story_book_text"],
            "model": result.get("used_model"), **saved}


def split_story(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Turn a story the user wrote (``text``) into chapters with settings, keeping every word, and save it."""
    params = dict(params or {})
    if not str(params.get("text") or "").strip():
        raise ValidationError("text is required: the full story to split into chapters.")
    _folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    request = _request(session, params, cards, text=params["text"], title=params.get("title") or "")
    try:
        result = book_funcs.split_pasted_story(request)
    except ValueError as error:
        raise ValidationError(str(error)) from error
    saved = _save_fields(project_id, result["story_layer"])
    return {"story_book": result["story_book"], "story_book_text": result["story_layer"]["story_book_text"],
            "story_settings": result["story_layer"]["story_settings"], "model": result.get("used_model"), **saved}


def convert_story_arc(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Turn the project's old song-section story arc into a Story Mode book and save it."""
    params = dict(params or {})
    _folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    request = _request(session, params, cards)
    try:
        result = book_funcs.convert_story_arc(request)
    except ValueError as error:
        raise ValidationError(str(error)) from error
    saved = _save_fields(project_id, result["story_layer"])
    return {"story_book": result["story_book"], "story_book_text": result["story_layer"]["story_book_text"],
            "story_settings": result["story_layer"]["story_settings"], "model": result.get("used_model"), **saved}


def scan_story_cast(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """People in the story who have no character card. Nothing is saved; add them with the reference tools."""
    params = dict(params or {})
    _folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    request = _request(session, params, cards)
    try:
        result = book_funcs.scan_story_cast(request)
    except ValueError as error:
        raise ValidationError(str(error)) from error
    return {"characters": result["characters"], "model": result.get("used_model")}


def create_story_acts(project_id: str, params: Optional[Dict[str, Any]] = None,
                      progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """Split every chapter (or ``chapter_id``) into acts with the LLM and save them."""
    params = dict(params or {})
    _folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    _require_scenes(cards)
    request = _request(session, params, cards, chapter_id=params.get("chapter_id") or "",
                       timeline_markers=session.get("timeline_markers") or [])
    try:
        result = plan_funcs.build_story_acts(request, progress=progress)
    except ValueError as error:
        raise ValidationError(str(error)) from error
    saved = _save_fields(project_id, result["story_layer"])
    return {"story_acts": result["story_layer"]["story_acts"], "model": result.get("used_model"), **saved}


def create_story_scene_plan(project_id: str, params: Optional[Dict[str, Any]] = None,
                            progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """Write every scene's story slice, spot, mood and performance (or only ``act_id``'s) and save them.

    Each scene's performance is also written to its timeline segment (``performance_mode``, ``lyric_no_lip_sync``).
    """
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session, folder)
    _require_scenes(cards)
    request = _request(session, params, cards, act_id=params.get("act_id") or "",
                       spot_min_scenes=params.get("spot_min_scenes"),
                       timeline_markers=session.get("timeline_markers") or [])
    try:
        result = plan_funcs.build_story_scene_plan(request, progress=progress)
    except ValueError as error:
        raise ValidationError(str(error)) from error
    saved = _save_fields(project_id, result["story_layer"])
    return {"story_scene_plan": result["story_layer"]["story_scene_plan"], "model": result.get("used_model"), **saved}


def create_story_locations(project_id: str, params: Optional[Dict[str, Any]] = None,
                           progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """Give each chapter setting a look and each planned spot a location card, then map every scene to its spot.

    Uses the story, the scene plan and the style/theme notes only; the lyrics are not sent. Existing cards with the
    same name keep their image (and their description unless ``replace_descriptions``).
    """
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    _require_scenes(cards)
    refs = session.get("flux_reference_builder") if isinstance(session.get("flux_reference_builder"), dict) else {}
    local_notes = str(params.get("style_theme") or refs.get("location_style_theme") or "").strip()
    request = _request(session, params, cards, style_theme=location_style_theme_text(folder, session, local_notes),
                       regenerate_looks=bool(params.get("regenerate_looks")))
    try:
        result = location_funcs.build_story_locations(request, progress=progress)
    except ValueError as error:
        raise ValidationError(str(error)) from error
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        merged, counts = location_funcs.apply_story_locations(
            session.get("flux_reference_builder"), result["locations"], _new_location_id,
            replace_descriptions=bool(params.get("replace_descriptions")),
        )
        session["flux_reference_builder"] = merged
        layer = _story_layer(session)
        layer.update(result["story_layer"])
        _save_story_layer(session, layer)
        saved = _persist_session(folder, session)
    names = {card["id"]: card["name"] for card in merged.get("locations") or []}
    return {
        "locations": [{"name": item["name"], "scenes": len(item["scene_ids"])} for item in result["locations"]],
        "added": counts["added"],
        "updated": counts["updated"],
        "scene_map": {sid: names.get(lid, lid) for sid, lid in (merged.get("scene_map") or {}).items()},
        "model": result.get("used_model"),
        "revision": saved.get("revision"),
        "storyboard": sync_storyboard_files(project_id),
    }


def story_mode_edit_fields(session: Dict[str, Any], layer: Dict[str, Any], story_in: Dict[str, Any]) -> Dict[str, Any]:
    """Checked Story Mode fields for PUT /story/settings. Raises ValidationError on acts that break the rules."""
    fields = normalize_story_mode_fields({key: story_in[key] for key in STORY_MODE_EDIT_KEYS if key in story_in})
    if "story_book_text" in story_in:
        fields.update(apply_story_book_text({**layer, **fields}, story_in.get("story_book_text") or ""))
    if "story_acts" in fields:
        chapters = normalize_story_book(fields.get("story_book") or layer.get("story_book"))["chapters"]
        scene_ids = [str(s.get("id")) for s in session.get("segments") or [] if isinstance(s, dict) and s.get("id")]
        problems = validate_story_acts(fields["story_acts"], chapters, scene_ids)
        if problems:
            raise ValidationError("story_acts are not valid: " + " ".join(problems))
    return fields


def story_mode_text_fields(layer: Dict[str, Any], story_in: Dict[str, Any]) -> Dict[str, Any]:
    """Story layer fields for a direct edit of ``story_book_text``."""
    return apply_story_book_text(layer, story_in.get("story_book_text") or "")


def story_mode_view(layer: Dict[str, Any]) -> Dict[str, Any]:
    """The Story Mode fields of a story layer, normalized, with the book text rebuilt when it is missing."""
    fields = normalize_story_mode_fields(layer)
    if fields.get("story_book") and not fields.get("story_book_text"):
        fields["story_book_text"] = story_book_to_text(fields["story_book"], fields.get("story_settings"))
    return fields


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def run_story_book_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "story_book", message="Writing the story book with the LLM...")
    return create_story_book(job.project_id, job.params)


def run_story_chapter_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "story_chapter", message="Rewriting the chapter with the LLM...")
    return rewrite_story_chapter(job.project_id, job.params)


def run_story_split_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "story_split", message="Splitting the story into chapters...")
    return split_story(job.project_id, job.params)


def run_story_convert_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "story_convert", message="Converting the story arc into a story book...")
    return convert_story_arc(job.project_id, job.params)


def run_story_cast_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "story_cast", message="Finding story characters without a card...")
    return scan_story_cast(job.project_id, job.params)


def _progress(job: Job, manager: JobManager, stage: str, noun: str):
    def report(done: int, total: int, label: str) -> None:
        manager.update_progress(job.id, 5.0 + 90.0 * done / max(1, total), stage, message=f"{noun} {done}/{total} ({label})")
    return report


def run_story_acts_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    return create_story_acts(job.project_id, job.params, progress=_progress(job, manager, "story_acts", "Chapters"))


def run_story_plan_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    return create_story_scene_plan(job.project_id, job.params, progress=_progress(job, manager, "story_plan", "Acts"))


def run_story_locations_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    return create_story_locations(job.project_id, job.params, progress=_progress(job, manager, "story_locations", "Settings"))


STORY_MODE_JOB_TYPES = {
    "book": "storyboard.story_book",
    "chapter": "storyboard.story_chapter",
    "split": "storyboard.story_split",
    "convert": "storyboard.story_convert",
    "cast": "storyboard.story_cast",
    "acts": "storyboard.story_acts",
    "plan": "storyboard.story_plan",
    "locations": "storyboard.story_locations",
}


def register_story_mode_handlers(manager: Optional[JobManager] = None) -> None:
    manager = manager or get_job_manager()
    manager.register_handler("storyboard.story_book", run_story_book_job)
    manager.register_handler("storyboard.story_chapter", run_story_chapter_job)
    manager.register_handler("storyboard.story_split", run_story_split_job)
    manager.register_handler("storyboard.story_convert", run_story_convert_job)
    manager.register_handler("storyboard.story_cast", run_story_cast_job)
    manager.register_handler("storyboard.story_acts", run_story_acts_job)
    manager.register_handler("storyboard.story_plan", run_story_plan_job)
    manager.register_handler("storyboard.story_locations", run_story_locations_job)
