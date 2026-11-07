"""Writes the thumbnail edit prompt with the Builder's own LLM runner, looking at the chosen scene image."""

import re
from typing import Any, Dict

from ..builder.media import _image_from_prompt_payload
from .minimax_reference_enhance import run_reference_vision
from .prompts.thumbnail import thumbnail_instruction

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_LABEL = re.compile(r"^(?:here(?:'s| is)[^:\n]*:|prompt\s*:|edit prompt\s*:)\s*", re.IGNORECASE)


def clean_thumbnail_prompt(text: Any) -> str:
    """Strip code fences, a leading label and wrapping quotes from the model's reply."""
    value = _FENCE.sub("", str(text or "").strip()).strip()
    value = _LABEL.sub("", value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return " ".join(value.split())


def _user_block(payload: Dict[str, Any]) -> str:
    lines = []
    for label, key in (
        ("THUMBNAIL TITLE", "title"), ("STYLE NOTES", "style_notes"),
        ("PROJECT", "project_name"), ("SCENE LYRICS", "scene_lyrics"),
    ):
        value = " ".join(str(payload.get(key) or "").split())
        if value:
            lines.append(f"{label}: {value[:600]}")
    return "\n".join(lines) or "No extra notes. Pick a strong, fitting look from the image."


def _generate_thumbnail_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Look at the scene image and write one edit prompt that makes it a thumbnail."""
    path = str(payload.get("image_path") or "").strip()
    data = str(payload.get("image_data") or "").strip()
    if not (path or data):
        raise ValueError("Choose a scene with an image first. The LLM needs the picture to write the thumbnail prompt.")
    image = _image_from_prompt_payload(path, data, "Thumbnail source image").convert("RGB")
    instruction = f"{thumbnail_instruction(payload.get('title', ''))}\n\n{_user_block(payload)}"
    text, info = run_reference_vision(
        {**payload, "max_new_tokens": 900, "temperature": payload.get("temperature") or 0.75, "top_p": 0.95},
        instruction, [image],
    )
    prompt = clean_thumbnail_prompt(text)
    if not prompt:
        raise ValueError("The LLM returned no thumbnail prompt. Try again or write the prompt yourself.")
    return {
        "prompt": prompt,
        "characters": len(prompt),
        "runner": info.get("runner", "builtin"),
        "used_model": info.get("used_model", ""),
        "unloaded": info.get("unloaded", False),
    }
