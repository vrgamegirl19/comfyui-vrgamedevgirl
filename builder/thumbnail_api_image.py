"""Thumbnail "API image" engine: the chosen frame edited by Gemini, Grok or OpenAI with the key from the LLM Runner.

The result is kept as ``<project>/thumbnail/thumbnail_NNNN.png`` like every other thumbnail engine's.
"""

import os
import re
from typing import Any, Dict

from PIL import Image

_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,119}$")


def _make_thumbnail_api_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    """``POST thumbnail/api_image``: edit the frame with the image API of the LLM Runner's provider.

    ``source_path`` is the frame, ``prompt`` the edit prompt, ``provider`` the LLM Runner's provider (``google``, ``grok``,
    ``openai``), ``api_key`` its key and ``model`` the image model (empty: the provider's first one).
    """
    from ..llm.api_image import ApiImageError, api_image_provider, default_api_image_model, edit_with_api
    from ..runner.video_files import _safe_project_subfolder
    from .thumbnail import THUMBNAIL_FOLDER, _clean_path, _next_thumbnail_path

    if not str(payload.get("project_folder") or "").strip().strip('"'):
        raise ValueError("Project folder is empty.")
    if not str(payload.get("source_path") or "").strip().strip('"'):
        raise ValueError("Choose the frame first: scrub the video to the moment you want and click Use this frame.")
    provider = api_image_provider(payload.get("provider"))
    model = str(payload.get("model") or "").strip() or default_api_image_model(provider)
    if not _MODEL_PATTERN.match(model):
        raise ValueError(f"'{model}' is not a valid image model name.")
    source = _clean_path(payload.get("source_path"))
    if not os.path.isfile(source):
        raise FileNotFoundError(f"The frame was not found: {source}")
    _project, folder = _safe_project_subfolder(payload.get("project_folder", ""), THUMBNAIL_FOLDER)
    with Image.open(source) as opened:
        frame = opened.convert("RGB")
    try:
        picture = edit_with_api(provider, str(payload.get("api_key") or ""), model, str(payload.get("prompt") or ""), frame)
    except ApiImageError as exc:
        raise RuntimeError(str(exc)) from None
    target = _next_thumbnail_path(folder)
    picture.save(target, format="PNG")
    return {"image_path": target, "width": picture.width, "height": picture.height, "provider": provider, "model": model}
