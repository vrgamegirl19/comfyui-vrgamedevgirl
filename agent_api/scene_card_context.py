"""Scene-card context for the Agent API's LLM steps, the same context the Storyboard UI sends.

The Storyboard sends every populated scene-card field with its prompt and beat requests
(``storyboardSceneCardContext`` and ``STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTION`` in
``web/storyboard_builder/scenes.mjs``; the scene-card block of ``storyboardVideoNotes`` in
``web/music_video_builder/storyboard_bridge.mjs``). These helpers build the same block for API jobs.

Each task still reads the card for its own purpose: a scene beat is a visual story summary, an image prompt
describes one still frame, a video prompt adds motion and the audio/performance directions. Timed Timeline
Notes are separate: only Story Arc planning reads them (``storyboard/timeline_notes.py``); a card's own
Director Note is its ``timeline_note``.
"""

import json
from typing import Any, Dict, List

from ..llm.prompts.storyboard import _STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTIONS
from ..storyboard.session_sync import scene_card_context

SCENE_CARD_INSTRUCTION = _STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTIONS


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def scene_card_block(card: Dict[str, Any]) -> str:
    """The instruction plus the complete card as JSON, as the Storyboard appends it to a request."""
    return f"{SCENE_CARD_INSTRUCTION}\nComplete scene_card:\n" + json.dumps(
        scene_card_context(card), indent=2, ensure_ascii=False
    )


def beat_scene(card: Dict[str, Any]) -> Dict[str, Any]:
    """The selected scene of a scene-beat request (``storyboardScenesForGpt`` keys for the card context)."""
    scene = {**card, "story_beat": ""}
    return {
        **scene,
        "scene_card": scene_card_context(scene),
        "scene_card_instruction": SCENE_CARD_INSTRUCTION,
        "director_note": _text(card.get("timeline_note")),
    }


def video_direction_lines(card: Dict[str, Any]) -> List[str]:
    """Card directions a video prompt uses (the ``storyboardVideoNotes`` order), as "Title:\\nvalue" blocks."""
    lines: List[str] = []

    def add(title: str, value: Any) -> None:
        if _text(value):
            lines.append(f"{title}:\n{_text(value)}")

    add("Director note (timeline)", card.get("timeline_note"))
    add("Storyboard scene story beat", card.get("story_beat"))
    motion = _text(card.get("motion_summary"))
    add("Storyboard motion/video summary", motion)
    if not motion:
        add("Storyboard camera motion", card.get("camera_motion"))
    add("Storyboard shot type", card.get("shot_type"))
    add("Storyboard character motion guidance", card.get("character_motion"))
    add("Storyboard performance direction", card.get("performance_style"))
    add("Storyboard facial performance direction",
        card.get("facial_performance_custom") or card.get("facial_performance"))
    add("Storyboard lyric section", card.get("lyric_section"))
    add("Exact manual audio / sound direction", card.get("audio_direction"))
    add("Exact manual continuity requirements", card.get("continuity"))
    return lines


def storyboard_video_context(card: Dict[str, Any]) -> str:
    """The "Storyboard Builder context" a video or MiniMax prompt request carries for one card."""
    return "\n\n".join(video_direction_lines(card) + [scene_card_block(card)])
