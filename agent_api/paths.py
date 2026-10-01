"""Project ID resolution and allowed roots management (C8, D7, R1)."""

import os
import re
from typing import List
import folder_paths

from .auth import load_auth_config
from .errors import PathOutsideRootError, ProjectNotFoundError, ValidationError


_INVALID_ID_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def get_allowed_project_roots() -> List[str]:
    """Return all allowlisted project root directories."""
    roots: List[str] = []
    output_dir = folder_paths.get_output_directory()
    if output_dir and os.path.isdir(output_dir):
        roots.append(os.path.abspath(output_dir))

    # Additional roots from agent_api.json config
    auth_config = load_auth_config()
    for root in auth_config.get("allowed_roots", []):
        text = str(root or "").strip()
        if text and os.path.isdir(text):
            abs_root = os.path.abspath(text)
            if abs_root not in roots:
                roots.append(abs_root)

    # Roots from environment variable VRGDG_PROJECT_ROOTS
    env_roots = os.environ.get("VRGDG_PROJECT_ROOTS", "")
    if env_roots:
        for item in env_roots.split(os.pathsep):
            text = item.strip()
            if text and os.path.isdir(text):
                abs_root = os.path.abspath(text)
                if abs_root not in roots:
                    roots.append(abs_root)

    return roots


def validate_project_id(project_id: str) -> str:
    """Validate that a project_id does not contain path traversal or invalid characters."""
    pid = str(project_id or "").strip()
    if not pid:
        raise ValidationError("Project ID cannot be empty.")
    if ".." in pid or _INVALID_ID_CHARS.search(pid):
        raise ValidationError(f"Invalid characters or traversal sequence in Project ID '{pid}'.")
    return pid


def is_path_inside_root(path: str, root: str) -> bool:
    """Check if target path is strictly within the root directory.

    Both paths are resolved through symlinks and junctions first, so a link inside the
    root that points elsewhere does not count as inside it.
    """
    try:
        norm_path = os.path.realpath(os.path.abspath(os.path.normpath(path)))
        norm_root = os.path.realpath(os.path.abspath(os.path.normpath(root)))
        return os.path.normcase(os.path.commonpath([norm_path, norm_root])) == os.path.normcase(norm_root)
    except Exception:
        return False


def resolve_project_folder(project_id: str) -> str:
    """Resolve a project_id to an absolute filesystem path within allowed roots.
    
    Raises:
        ValidationError: If project_id is malformed.
        PathOutsideRootError: If resolved path escapes allowed roots.
        ProjectNotFoundError: If project folder does not exist.
    """
    clean_id = validate_project_id(project_id)
    roots = get_allowed_project_roots()

    for root in roots:
        candidate = os.path.abspath(os.path.join(root, clean_id))
        if not is_path_inside_root(candidate, root):
            raise PathOutsideRootError(clean_id)
        if os.path.isdir(candidate):
            return candidate

    raise ProjectNotFoundError(clean_id)


def get_project_id(folder_path: str) -> str:
    """Derive project ID from an absolute folder path."""
    return os.path.basename(os.path.normpath(folder_path))


def session_audio_path(session) -> str:
    """Project audio path from a session dict.

    The Video Builder saves it as ``audio_path``. ``audio_file`` is accepted for
    sessions written by older agent code. Prefers a path that exists on disk.
    """
    if not isinstance(session, dict):
        return ""
    candidates = [str(session.get(key) or "").strip() for key in ("audio_path", "audio_file")]
    candidates = [path for path in candidates if path]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0] if candidates else ""


def session_video_mode(session) -> str:
    """Default video render mode for a project.

    A MiniMax project renders with the MiniMax graphs whatever ``video_model_mode`` says (that key holds
    the LTX mode and stays set to an old value). Other engines use their saved LTX mode.
    """
    if not isinstance(session, dict):
        return "i2v"
    if str(session.get("video_engine") or "").strip().lower() == "minimax_h3":
        return "minimax_h3"
    return str(session.get("video_model_mode") or session.get("video_mode") or "i2v").strip().lower()

