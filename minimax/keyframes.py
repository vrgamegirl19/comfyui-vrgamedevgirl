"""Keep per-scene I2V keyframes inside the exact video range retained by the Builder."""

from typing import Any, List


def align_i2v_keyframes(conditioning: List[Any], first_index: int, last_index: int) -> List[Any]:
    """Copy keyframe metadata without copying tensors; preserve all other conditioning values."""
    if first_index < 0 or last_index < first_index:
        raise ValueError("First/last frame indices must describe a valid scene range.")
    result = []
    for embedding, metadata in conditioning:
        values = dict(metadata)
        frames = metadata.get("minimax_keyframes")
        if frames:
            frames = [dict(frame) for frame in frames]
            frames[0]["resolved_frame_index"] = first_index
            if len(frames) > 1:
                frames[-1]["resolved_frame_index"] = last_index
            values["minimax_keyframes"] = frames
        result.append([embedding, values])
    return result
