"""Theatrical prompt directions shared with the browser through a generated snapshot."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def theatrical_presets() -> Dict[str, Any]:
    """Read the small bundled preset snapshot; no project files or models are changed."""
    path = Path(__file__).with_name("theatrical_presets.json")
    return json.loads(path.read_text(encoding="utf-8"))


def theatrical_project_defaults(session: Dict[str, Any]) -> Dict[str, Any]:
    """Read the Builder's existing project keys, including its separate facial defaults."""
    stored = session.get("builder_storyboard_defaults")
    return {
        "facial_performance": session.get("default_facial_performance"),
        "facial_performance_custom": session.get("default_facial_performance_custom"),
        **(stored if isinstance(stored, dict) else {}),
    }


def theatrical_direction_lines(
    card: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Resolve only selected Theatrical presets, honoring scene and custom directions."""
    scoped = card.get("scene_defaults")
    effective = {**(defaults or {}), **(scoped if isinstance(scoped, dict) else {})}
    presets = theatrical_presets()
    lines: List[str] = []
    camera = card.get("camera_flow") or effective.get("camera_flow")
    if camera == presets["camera_key"] and not card.get("motion_summary"):
        flow = presets["camera"]
        candidates = "\n".join(
            f"- {entry['shot']}; camera: {entry['camera']}" for entry in flow["sequence"]
        )
        lines.append(
            f"Theatrical camera flow:\n{flow['guidance']}\n"
            "Use the scene card's authored framing and camera path where supplied. "
            "For a continued take, follow its inherited opening frame and continuation contract. "
            "Choose suitable coverage from these candidates for the remaining planned shots:\n"
            + candidates
        )
    performance = card.get("performance_style") or effective.get("performance_style")
    if performance == "theatrical":
        lines.append("Theatrical performance direction:\n" + presets["performance"]["direction"])
    facial = card.get("facial_performance") or effective.get("facial_performance")
    custom = card.get("facial_performance_custom") or effective.get("facial_performance_custom")
    if facial == "theatrical" and not custom:
        lines.append("Theatrical facial performance direction:\n" + presets["facial"]["direction"])
    return lines
