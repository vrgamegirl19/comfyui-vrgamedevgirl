"""Revision-checked Speaking audio operations shared with the Video Builder."""

from typing import Any

from ..builder.audio import _save_scene_audio
from ..builder.project import _BUILDER_SAVE_LOCK
from ..builder.elevenlabs_speech import dialogue_speaker, validate_dialogue
from ..builder.scene_audio_settings import (
    DEFAULTS, audio_settings_view, require_speaking, update_audio_settings, validate_settings,
)
from .errors import RevisionConflictError, SceneNotFoundError, ValidationError
from .mutations import _get_active_session_and_folder, _persist_session


def get_audio_settings(project_id: str, scene_id: str | None = None) -> dict[str, Any]:
    """Read project defaults or one scene's effective audio settings."""
    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        try:
            require_speaking(session)
            if scene_id is None:
                view = {**DEFAULTS, **(session.get("speaking_audio_defaults") or {})}
            else:
                scene = next((s for s in session.get("segments", []) if s["id"] == scene_id), None)
                if scene is None:
                    raise SceneNotFoundError(scene_id, project_id)
                view = audio_settings_view(session, scene)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return {"settings": view, "revision": session.get("revision", 0)}


def patch_audio_settings(
    project_id: str, settings: dict[str, Any], scene_id: str | None = None,
    if_match_revision: int | None = None, audio_data: str | None = None,
    audio_name: str = "scene_audio.wav", clear_audio: bool = False,
    dialogue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save defaults or scene overrides, optionally importing/removing scene dialogue."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        revision = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and revision != if_match_revision:
            raise RevisionConflictError(revision, if_match_revision)
        scene = next((s for s in session.get("segments", []) if s["id"] == scene_id), None)
        if scene_id is not None and scene is None:
            raise SceneNotFoundError(scene_id, project_id)
        if session.get("timing_frozen") or any(s.get("video_status") == "running" for s in session.get("segments", [])):
            raise ValidationError("Unfreeze timing and wait for scene rendering to finish before editing audio.")
        try:
            require_speaking(session)
            validate_settings(settings, scene_id is not None)
            if dialogue is not None:
                if scene is None:
                    raise ValueError("Dialogue drafts require an existing scene.")
                dialogue = validate_dialogue(dialogue)
                if dialogue["speaker_id"]:
                    dialogue_speaker(session.get("flux_reference_builder"), dialogue["speaker_id"])
            if not isinstance(clear_audio, bool):
                raise ValueError("clear_audio must be a boolean.")
            if (clear_audio or audio_data is not None) and scene_id is None:
                raise ValueError("Audio import/removal requires an existing scene.")
            if audio_data is not None and clear_audio:
                raise ValueError("Choose either audio_data or clear_audio.")
            attachment = {} if clear_audio else None
            if audio_data is not None:
                if scene is None or not isinstance(audio_data, str) or not audio_data.strip():
                    raise ValueError("Non-empty audio_data and an existing scene are required.")
                attachment = _save_scene_audio({
                    "project_folder": folder, "scene_number": session["segments"].index(scene) + 1,
                    "audio_data": audio_data, "audio_name": audio_name, "preserve_source": True,
                })
                attachment["audio_name"] = audio_name
                if not float(attachment.get("duration") or 0) > 0:
                    raise ValueError("The imported audio has no usable duration.")
            updated = update_audio_settings(session, settings, scene_id, attachment, dialogue)
        except (ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc
        saved = _persist_session(folder, updated)
        view = (audio_settings_view(updated, next(s for s in updated["segments"] if s["id"] == scene_id))
                if scene_id is not None else updated["speaking_audio_defaults"])
        return {"settings": view, "revision": saved.get("revision", revision + 1)}
