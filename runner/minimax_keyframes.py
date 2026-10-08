"""Wire I2V endpoint conditioning to the range kept by exact scene trimming."""

from typing import Any, Dict

from ..minimax.latent_manager import H3_FPS


def patch_i2v_keyframe_timing(prompt: Dict[str, Any], timing: Any) -> None:
    """Keep endpoint anchors at each pass's spatial resolution and at final scene boundaries."""
    first = round(timing.final_trim_start_seconds * H3_FPS)
    last = first + timing.final_frame_count - 1
    for source in ("136", "214"):
        node = prompt.get(source, {})
        if node.get("class_type") != "MiniMaxH3ImageToVideo" or "last_frame" not in node.get("inputs", {}):
            continue
        target = f"i2v_keyframe_timing_{source}"
        for consumer in prompt.values():
            for key, value in list(consumer.get("inputs", {}).items()):
                if value == [source, 0]:
                    consumer["inputs"][key] = [target, 0]
        prompt[target] = {
            "class_type": "VRGDG_MiniMaxH3KeyframeTiming",
            "inputs": {"conditioning": [source, 0], "first_frame_index": first, "last_frame_index": last},
            "_meta": {"title": "I2V First / Last Frame — Exact Scene Range"},
        }
