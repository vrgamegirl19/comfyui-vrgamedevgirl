"""Local, append-only error reports owned by a saved Builder project."""

import json
import logging
import os
import re
import threading
from datetime import datetime

from ..core.atomic_write import atomic_write_text


_LOCK = threading.RLock()
_LOGGER = logging.getLogger(__name__)
_SECRET_KEY = re.compile(r"api.?key|token|password|secret|authorization|credential", re.I)


def secret_values(value: object) -> list[str]:
    """Collect credential values without including unrelated project data."""
    values = []
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_KEY.search(str(key)) and isinstance(item, str) and item:
                values.append(item)
            elif isinstance(item, (dict, list)):
                values.extend(secret_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(secret_values(item))
    return values


def redact(text: object, secrets: list[str]) -> str:
    """Remove known credentials and common credential-bearing error text."""
    result = str(text or "")
    for secret in sorted(set(secrets), key=len, reverse=True):
        result = result.replace(secret, "[redacted]")
    result = re.sub(r"(?i)(Bearer\s+)[^\s,'\"<>]+", r"\1[redacted]", result)
    result = re.sub(
        r"(?i)((?:api[_-]?key|access[_-]?token|token|password|secret|authorization)"
        r"[\"']?\s*[:=]\s*[\"']?)[^\s,\"'&<>]+", r"\1[redacted]", result,
    )
    return re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[redacted]@", result)


def _project_log(project_folder: str) -> tuple[str, dict]:
    """Resolve a fixed log path inside an existing saved project."""
    if not str(project_folder or "").strip():
        raise ValueError("Save or open a project before using its error log.")
    folder = os.path.realpath(project_folder)
    with open(os.path.join(folder, "vrgdg_builder_session.json"), encoding="utf-8-sig") as handle:
        session = json.load(handle)
    if not isinstance(session, dict):
        raise ValueError("Builder session must be an object.")
    path = os.path.realpath(os.path.join(folder, "logs", "error_log.txt"))
    if os.path.commonpath((folder, path)) != folder:
        raise ValueError("The error log must stay inside the project folder.")
    return path, session


def _header(project_folder: str, session: dict) -> str:
    title = session.get("project_name") or os.path.basename(os.path.realpath(project_folder))
    return "=" * 50 + "\nVIDEO BUILDER - PROJECT ERROR LOG\nProject: " + str(title) + "\n" + "=" * 50 + "\n"


def project_error_log(project_folder: str, clear: bool = False, read: bool = True) -> dict:
    """Create/read the report, or explicitly clear it while retaining its header."""
    with _LOCK:
        path, session = _project_log(project_folder)
        if clear or not os.path.exists(path):
            atomic_write_text(path, redact(_header(project_folder, session), secret_values(session)))
        if not read:
            return {"path": path}
        with open(path, encoding="utf-8") as handle:
            return {"path": path, "text": handle.read()}


def append_project_error(project_folder: str, entry: dict, secrets: list[str] | None = None) -> dict:
    """Append one complete UTF-8 record without rewriting previous entries."""
    with _LOCK:
        if not isinstance(entry, dict) or not entry.get("message"):
            raise ValueError("An error-log entry must contain a message.")
        path, session = _project_log(project_folder)
        credentials = secret_values(session) + list(secrets or []) + secret_values(dict(os.environ))
        severity = str(entry.get("severity") or "ERROR").upper()
        if severity not in {"ERROR", "WARNING", "CRITICAL"}:
            raise ValueError("Unsupported error-log severity.")
        timestamp = datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")
        lines = ["", f"[{timestamp}] [{severity}]"]
        for key, label in (("scene", "Scene"), ("stage", "Stage"), ("mode", "Mode"),
                           ("source", "Source"), ("component", "Component"), ("message", "Message")):
            if entry.get(key) is not None and str(entry[key]):
                lines.append(f"{label}: {entry[key]}")
        if entry.get("details"):
            lines.extend(["", "Error Details:", str(entry["details"])])
        if entry.get("stack"):
            lines.extend(["", "Stack Trace:", str(entry["stack"])])
        lines.append("-" * 50)
        text = redact("\n".join(lines) + "\n", credentials)
        if not os.path.exists(path):
            atomic_write_text(path, redact(_header(project_folder, session), credentials))
        # This diagnostic file is an append-only journal, not a replaceable state snapshot.
        with open(path, "a", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
        message = redact(entry.get("message"), credentials)
    _LOGGER.log(getattr(logging, severity), "[VRGDG Builder] %s", message)
    return {"path": path, "message": message}
