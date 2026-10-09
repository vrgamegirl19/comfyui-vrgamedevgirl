"""Saved character voices and selected LLM Runner for scene dialogue API jobs."""

from typing import Any

from ..builder.elevenlabs_speech import (
    dialogue_speaker, generate_speech, validate_dialogue,
)
from ..builder.project import _BUILDER_SAVE_LOCK
from ..builder.scene_audio_settings import require_speaking
from ..llm.scene_dialogue import craft_dialogue
from .errors import SceneNotFoundError, ValidationError
from .llm_runtime import llm_payload_from_session, prepare_llm_payload
from .mutations import _get_active_session_and_folder


def dialogue_request(
    project_id: str, scene_id: str, dialogue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a saved scene, draft, voice and runner without mutating state."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        scene = next((s for s in session.get("segments", [])
                      if s["id"] == scene_id), None)
        if scene is None:
            raise SceneNotFoundError(scene_id, project_id)
        try:
            require_speaking(session)
            draft = validate_dialogue(
                scene.get("scene_dialogue", {}) if dialogue is None else dialogue,
                require_text=True,
            )
            speaker = dialogue_speaker(
                session.get("flux_reference_builder"), draft["speaker_id"],
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return {
            **llm_payload_from_session(session), **speaker, "dialogue": draft,
            "project_folder": folder, "scene_id": scene_id,
            "api_key": session.get("elevenlabs_api_key_project", ""),
            "unload_after": False, "clear_before_load": False,
        }


def run_craft_dialogue_job(job: Any, manager: Any) -> dict[str, Any]:
    """Return editable dialogue using the saved runner and an already loaded LM Studio model."""
    request = dialogue_request(
        job.project_id, job.params["scene_id"], job.params.get("dialogue"),
    )
    manager.update_progress(
        job.id, 10, "dialogue", message="Preparing dialogue with the project LLM...",
    )
    request.pop("api_key", None)
    prepared = prepare_llm_payload(request)
    try:
        return craft_dialogue(prepared)
    except Exception as exc:
        message = str(exc)
        for field in ("llm_api_key_project", "own_server_api_key", "own_server_api_key_project", "lmstudio_api_key"):
            secret = str(prepared.get(field) or "")
            if secret:
                message = message.replace(secret, "[redacted]")
        raise ValidationError(message) from None


def generate_project_speech(
    project_id: str, scene_id: str, dialogue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one take using the saved project key; return audio for explicit import."""
    request = dialogue_request(project_id, scene_id, dialogue)
    try:
        return generate_speech(request["api_key"], request["voice_id"], request["dialogue"])
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
