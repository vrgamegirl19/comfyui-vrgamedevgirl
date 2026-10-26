"""Files an agent makes for a render: a saved RefMod from pictures (``POST /refmods``) and the song split into stems
(``POST /projects/{pid}/audio/stems``). Both run as GPU jobs. The work is the same code the Builder's own buttons call
(``minimax/refmod_studio.py`` and ``builder/audio_stems.py``).
"""

import os
from typing import Any, Dict

from ..builder.audio_stems import separate_project_audio
from ..minimax import refmod_studio
from .errors import ValidationError
from .jobs.manager import JobManager
from .jobs.models import Job
from .mutations import _get_active_session_and_folder
from .paths import session_audio_path


def check_refmod_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The request a RefMod is made from, checked, so a bad one is a 400 before any job exists."""
    try:
        return refmod_studio.validate_request(dict(payload or {}))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def run_refmod_create_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "refmod", message="Encoding the pictures into a RefMod...")
    try:
        return refmod_studio.create_refmod(dict(job.params or {}))
    except refmod_studio.RefModExistsError as exc:
        raise ValidationError(f"{exc} Send overwrite: true to replace it.", details={"exists": True, "path": exc.path}) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise ValidationError(str(exc)) from exc


def stems_request(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """What the song split is made from: the project folder, the song (the project's own unless ``audio_path`` is sent) and the settings."""
    folder, session = _get_active_session_and_folder(project_id)
    audio_path = str((payload or {}).get("audio_path") or "").strip().strip('"') or session_audio_path(session)
    if not audio_path or not os.path.isfile(audio_path):
        raise ValidationError("The project has no song to split. Attach one with PUT /audio, or send audio_path.",
                              details={"audio_path": audio_path})
    request = {"project_folder": folder, "audio_path": audio_path}
    for key in ("model_name", "device", "input_gain_db"):
        if (payload or {}).get(key) is not None:
            request[key] = payload[key]
    return request


def run_audio_stems_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "stems", message="Splitting the song into stems with Demucs...")
    request = stems_request(job.project_id, dict(job.params or {}))
    try:
        result = separate_project_audio(request)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    manager.update_progress(job.id, 100.0, "stems", message="The song is split.")
    return result


def register_refmod_stems_handlers(manager: JobManager) -> None:
    manager.register_handler("refmod.create", run_refmod_create_job)
    manager.register_handler("audio.stems", run_audio_stems_job)
