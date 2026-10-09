"""Structured error codes and exception classes for the VRGDG Agent API (C6)."""

from typing import Any, Dict, Optional


# Error Code Constants
VALIDATION_ERROR = "VALIDATION_ERROR"
AUTH_REQUIRED = "AUTH_REQUIRED"
AUTH_DISABLED = "AUTH_DISABLED"
PATH_OUTSIDE_ROOT = "PATH_OUTSIDE_ROOT"
PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
SCENE_NOT_FOUND = "SCENE_NOT_FOUND"
TIMELINE_NOTE_NOT_FOUND = "TIMELINE_NOTE_NOT_FOUND"
ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
REVISION_CONFLICT = "REVISION_CONFLICT"
PROJECT_BUSY = "PROJECT_BUSY"
GPU_BUSY = "GPU_BUSY"
JOB_NOT_FOUND = "JOB_NOT_FOUND"
JOB_CANCELLED = "JOB_CANCELLED"
COMFY_QUEUE_REJECTED = "COMFY_QUEUE_REJECTED"
COMFY_NODE_ERRORS = "COMFY_NODE_ERRORS"
COMFY_TIMEOUT = "COMFY_TIMEOUT"
COMFY_NO_OUTPUT = "COMFY_NO_OUTPUT"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
LLM_RECOVERABLE = "LLM_RECOVERABLE"
LLM_BAD_OUTPUT = "LLM_BAD_OUTPUT"
FFMPEG_FAILED = "FFMPEG_FAILED"
AUDIO_REQUIRED = "AUDIO_REQUIRED"
LATENT_STALE = "LATENT_STALE"
PREDECESSOR_MISSING = "PREDECESSOR_MISSING"
DURATION_MISMATCH = "DURATION_MISMATCH"
NOT_SUPPORTED_FOR_MODE = "NOT_SUPPORTED_FOR_MODE"
HUMAN_REQUIRED = "HUMAN_REQUIRED"
SETTINGS_INVALID = "SETTINGS_INVALID"
INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentApiError(Exception):
    """Base exception for all VRGDG Agent API errors."""

    def __init__(
        self,
        message: str,
        code: str = INTERNAL_ERROR,
        status: int = 500,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.details = details or {}
        self.retryable = retryable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class ValidationError(AgentApiError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=VALIDATION_ERROR, status=400, details=details, retryable=False)


class AuthError(AgentApiError):
    def __init__(self, message: str = "Authentication required or token invalid.", code: str = AUTH_REQUIRED, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=code, status=401, details=details, retryable=False)


class PathOutsideRootError(AgentApiError):
    def __init__(self, path: str):
        super().__init__(
            f"Path '{path}' escapes allowed project roots.",
            code=PATH_OUTSIDE_ROOT,
            status=403,
            details={"path": path},
            retryable=False,
        )


class NotFoundError(AgentApiError):
    def __init__(self, message: str, code: str = "NOT_FOUND", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=code, status=404, details=details, retryable=False)


class ProjectNotFoundError(NotFoundError):
    def __init__(self, project_id: str):
        super().__init__(
            f"Project '{project_id}' was not found in any allowed root.",
            code=PROJECT_NOT_FOUND,
            details={"project_id": project_id},
        )


class SceneNotFoundError(NotFoundError):
    def __init__(self, scene_id: str, project_id: str = ""):
        super().__init__(
            f"Scene '{scene_id}' was not found in project '{project_id}'.",
            code=SCENE_NOT_FOUND,
            details={"scene_id": scene_id, "project_id": project_id},
        )


class TimelineNoteNotFoundError(NotFoundError):
    def __init__(self, note_id: str, project_id: str = ""):
        super().__init__(
            f"Timeline note '{note_id}' was not found in project '{project_id}'.",
            code=TIMELINE_NOTE_NOT_FOUND,
            details={"note_id": note_id, "project_id": project_id},
        )


class AssetNotFoundError(NotFoundError):
    def __init__(self, asset_id: str, project_id: str = ""):
        super().__init__(
            f"Asset '{asset_id}' was not found.",
            code=ASSET_NOT_FOUND,
            details={"asset_id": asset_id, "project_id": project_id},
        )


class RevisionConflictError(AgentApiError):
    def __init__(self, current_revision: int, incoming_revision: int):
        super().__init__(
            f"Project revision conflict: incoming revision {incoming_revision} < saved revision {current_revision}.",
            code=REVISION_CONFLICT,
            status=409,
            details={"current_revision": current_revision, "incoming_revision": incoming_revision},
            retryable=True,
        )


class SettingsInvalidError(AgentApiError):
    def __init__(self, message: str, invalid_fields: Optional[Dict[str, str]] = None):
        super().__init__(
            message,
            code=SETTINGS_INVALID,
            status=400,
            details={"invalid_fields": invalid_fields or {}},
            retryable=False,
        )


class NotSupportedForModeError(AgentApiError):
    def __init__(self, message: str, mode: str, feature: str = ""):
        super().__init__(
            message,
            code=NOT_SUPPORTED_FOR_MODE,
            status=422,
            details={"mode": mode, "feature": feature},
            retryable=False,
        )


class JobNotFoundError(NotFoundError):
    def __init__(self, job_id: str):
        super().__init__(
            f"Job '{job_id}' was not found.",
            code=JOB_NOT_FOUND,
            details={"job_id": job_id},
        )


class JobCancelledError(AgentApiError):
    def __init__(self, job_id: str = "", message: str = "Job execution was cancelled."):
        super().__init__(
            message,
            code=JOB_CANCELLED,
            status=409,
            details={"job_id": job_id} if job_id else {},
            retryable=False,
        )


class ComfyQueueError(AgentApiError):
    def __init__(self, message: str, node_errors: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            code=COMFY_QUEUE_REJECTED,
            status=502,
            details={"node_errors": node_errors or {}},
            retryable=False,
        )


class ComfyTimeoutError(AgentApiError):
    def __init__(self, prompt_id: str, timeout_seconds: float):
        super().__init__(
            f"Timed out waiting for ComfyUI prompt '{prompt_id}' after {timeout_seconds}s.",
            code=COMFY_TIMEOUT,
            status=504,
            details={"prompt_id": prompt_id, "timeout_seconds": timeout_seconds},
            retryable=True,
        )


class ComfyExecutionError(AgentApiError):
    def __init__(self, message: str, prompt_id: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            code=COMFY_NODE_ERRORS,
            status=502,
            details=details or {"prompt_id": prompt_id},
            retryable=True,
        )


class PredecessorMissingError(AgentApiError):
    def __init__(self, scene_number: int, predecessor_scene: int, message: Optional[str] = None):
        msg = message or f"Latent continuation for Scene {scene_number:03d} requires Scene {predecessor_scene:03d} latent, but it was not found. Render Scene {predecessor_scene:03d} first."
        super().__init__(
            msg,
            code=PREDECESSOR_MISSING,
            status=409,
            details={"scene_number": scene_number, "predecessor_scene": predecessor_scene, "required_scene": predecessor_scene},
            retryable=False,
        )


class LatentStaleError(AgentApiError):
    def __init__(self, scene_number: int, stale_scene: int, message: Optional[str] = None):
        msg = message or f"Latent for Scene {stale_scene:03d} is stale (dirty) and must be re-rendered before Scene {scene_number:03d}."
        super().__init__(
            msg,
            code=LATENT_STALE,
            status=409,
            details={"scene_number": scene_number, "stale_scene": stale_scene, "required_scene": stale_scene},
            retryable=False,
        )


