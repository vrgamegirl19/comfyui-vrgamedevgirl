"""Story Mode steps for agents: the story book, chapter rewrites, pasted stories and the cast scan.

Same functions the Story Builder uses (``storyboard.story_book``). The LLM is the one already loaded; the API
never loads or switches a model. Results are saved in ``builder_story_layer`` like the Builder saves them.
"""

from typing import Any, Dict, Optional

from ...storyboard import story_book as book_funcs
from ...storyboard.story_mode import apply_story_book_text, normalize_story_mode_fields, story_book_to_text
from ..errors import ValidationError
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session
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

STORY_MODE_STEPS = ("book", "chapter", "split", "cast")


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


def _save_fields(project_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        layer = _story_layer(session)
        layer.update(fields)
        _save_story_layer(session, layer)
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


def story_mode_text_fields(layer: Dict[str, Any], story_in: Dict[str, Any]) -> Dict[str, Any]:
    """Story layer fields for a direct edit of ``story_book_text`` (PUT /story/settings)."""
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


def run_story_cast_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "story_cast", message="Finding story characters without a card...")
    return scan_story_cast(job.project_id, job.params)


STORY_MODE_JOB_TYPES = {
    "book": "storyboard.story_book",
    "chapter": "storyboard.story_chapter",
    "split": "storyboard.story_split",
    "cast": "storyboard.story_cast",
}


def register_story_mode_handlers(manager: Optional[JobManager] = None) -> None:
    manager = manager or get_job_manager()
    manager.register_handler("storyboard.story_book", run_story_book_job)
    manager.register_handler("storyboard.story_chapter", run_story_chapter_job)
    manager.register_handler("storyboard.story_split", run_story_split_job)
    manager.register_handler("storyboard.story_cast", run_story_cast_job)
