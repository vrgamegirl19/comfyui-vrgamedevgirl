"""Project credentials and account voices shared with the Speaking UI and MCP."""

from typing import Any

from ..builder.elevenlabs import list_voices, validate_api_key
from ..builder.elevenlabs_voice_design import (
    create_designed_voice, design_voice, required_text,
)
from ..builder.project import _BUILDER_SAVE_LOCK
from ..builder.scene_audio_settings import require_speaking
from ..llm.voice_design import generate_voice_description
from .errors import RevisionConflictError, ValidationError
from .llm_runtime import llm_payload_from_session, prepare_llm_payload
from .mutations import _get_active_session_and_folder, _persist_session


def project_credentials(project_id: str, api_key: Any = None,
                        if_match_revision: int | None = None) -> dict[str, Any]:
    """Read configured status or explicitly save/clear the project key atomically."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        revision = int(session.get("revision") or session.get("builder_save_revision") or 0)
        try:
            require_speaking(session)
            if api_key is not None:
                if if_match_revision is not None and revision != if_match_revision:
                    raise RevisionConflictError(revision, if_match_revision)
                session["elevenlabs_api_key_project"] = validate_api_key(api_key, allow_empty=True)
                revision = _persist_session(folder, session).get("revision", revision + 1)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return {"configured": bool(session.get("elevenlabs_api_key_project")), "revision": revision}


def project_voices(project_id: str, next_page_token: str = "", test: bool = False) -> dict[str, Any]:
    """Discover voices or test access using the explicitly saved project key."""
    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        try:
            require_speaking(session)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        key = session.get("elevenlabs_api_key_project", "")
    try:
        result = list_voices(key, next_page_token)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return {"connected": True} if test else result


def project_voice_design(
    project_id: str, payload: dict[str, Any], save: bool = False,
) -> dict[str, Any]:
    """Use the saved project key for previews or explicit account creation; no project mutation."""
    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        try:
            require_speaking(session)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        key = session.get("elevenlabs_api_key_project", "")
    try:
        if save:
            return {"voice": create_designed_voice(key, payload)}
        return design_voice(key, payload)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def voice_description_request(
    project_id: str, subject_id: str, user_input: str,
) -> dict[str, Any]:
    """Build a text-only LLM task using the saved runner and character context."""
    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        try:
            require_speaking(session)
            user_input = required_text(user_input, "Voice idea", 1, 4000)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        subjects = session.get("flux_reference_builder", {}).get("subjects", [])
        subject = next((s for s in subjects if s.get("id") == subject_id), None)
        if (
            not subject or subject.get("reference_type", "character") != "character"
            or subject.get("extra_reference_for")
        ):
            raise ValidationError("Voice Design requires an existing primary character reference.")
        return {
            **llm_payload_from_session(session), "user_input": user_input,
            "character_name": subject.get("name", ""),
            "character_description": subject.get("description", ""),
            "unload_after": False, "clear_before_load": False,
        }


def run_voice_description_job(job: Any, manager: Any) -> dict[str, Any]:
    """Generate a reviewable description without persisting it or calling ElevenLabs."""
    params = job.params or {}
    request = voice_description_request(
        job.project_id, params.get("subject_id", ""), params.get("user_input", ""),
    )
    manager.update_progress(
        job.id, 10, "voice_description",
        message="Writing a voice description with the project LLM...",
    )
    prepared = prepare_llm_payload(request)
    try:
        return generate_voice_description(prepared)
    except Exception as exc:
        message = str(exc)
        for field in ("llm_api_key_project", "own_server_api_key", "own_server_api_key_project", "lmstudio_api_key"):
            secret = str(prepared.get(field) or "")
            if secret:
                message = message.replace(secret, "[redacted]")
        raise ValidationError(message) from None
