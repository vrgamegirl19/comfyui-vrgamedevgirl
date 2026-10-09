"""Tell open Video Builder and Storyboard windows that the Agent API changed a project.

Every API save goes through ``mutations._persist_session``, which calls ``notify_project_changed``. The event
goes to every browser tab over ComfyUI's websocket as ``vrgdg.project_changed``;
``web/music_video_builder/external_changes.mjs`` applies it to the open project (or asks first when the
user has unsaved edits). ``change`` says what changed so the UI can merge just that:

- ``{"kind": "scene_fields", "scenes": {scene_id: {"segment": [...keys], "card": [...keys], "references": [...]}}}``
- ``{"kind": "timeline_markers", "marker_ids": [...]}``
- ``{"kind": "storyboard"}`` (the Storyboard's saved copy only)
- ``{"kind": "project"}`` (anything else: the UI reloads the project when it is safe to)
"""

import sys
from typing import Any, Dict, Optional

EVENT_NAME = "vrgdg.project_changed"


def _prompt_server() -> Any:
    """ComfyUI's running server, or None outside ComfyUI (tests, scripts). Never imports ComfyUI."""
    module = sys.modules.get("server")
    return getattr(getattr(module, "PromptServer", None), "instance", None)


def notify_project_changed(folder: str, session: Dict[str, Any], change: Optional[Dict[str, Any]] = None) -> bool:
    """Broadcast that ``folder`` was saved by the Agent API. Returns False when no UI can be told."""
    server = _prompt_server()
    if server is None or not hasattr(server, "send_sync"):
        return False
    data = {
        "project_folder": folder,
        "revision": int(session.get("revision") or 0),
        "builder_save_revision": int(session.get("builder_save_revision") or 0),
        "source": "agent_api",
        "change": change or {"kind": "project"},
    }
    try:
        server.send_sync(EVENT_NAME, data)
    except Exception as exc:  # a closed event loop must never fail the save that already happened
        print(f"[VRGDG API] Could not notify open windows of the project change: {exc}")
        return False
    return True
