"""Edit one picture through a hosted image API: Gemini, Grok (xAI) or OpenAI.

The Thumbnail dialog's "API image" engine uses this. The provider and the key are the ones set in the LLM Runner, the image
model is chosen next to the engine. Only the standard library and Pillow are used. The key travels in a header, never in the
URL, so it cannot end up in an error message. These models are kept apart from the chat models in ``llm/api.py`` on purpose.
"""

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from io import BytesIO
from typing import Any, Dict, List, Tuple

from PIL import Image

from .http_tls import urlopen_verified

API_IMAGE_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "gemini": {"label": "Gemini", "models": ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"]},
    "grok": {"label": "Grok", "models": ["grok-imagine-image-2.0", "grok-imagine-image"]},
    "openai": {"label": "OpenAI", "models": ["gpt-image-2", "gpt-image-1"]},
}
# LLM Runner provider names -> the provider here. "xai" is the old name of Grok.
_PROVIDER_ALIASES = {"google": "gemini", "gemini": "gemini", "grok": "grok", "xai": "grok", "openai": "openai"}
_USER_AGENT = "VRGDG-LLM-Multi/1.0 (+ComfyUI)"
_TIMEOUT_SECONDS = 240
_MAX_SOURCE_SIDE = 2048
_WIDE = 16 / 9
_WIDE_TOLERANCE = 0.03
_OPENAI_SIZE = "1536x1024"


class ApiImageError(Exception):
    """A provider refused the request or answered without a picture. The message is shown to the user."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def api_image_capabilities() -> Dict[str, Any]:
    """The providers and their image models, for the Thumbnail dialog's choice (the single source of the lists)."""
    return {
        "providers": {
            name: {"label": info["label"], "models": list(info["models"]), "default": info["models"][0]}
            for name, info in API_IMAGE_PROVIDERS.items()
        }
    }


def api_image_provider(name: Any) -> str:
    """The provider behind an LLM Runner provider name (``google`` is Gemini). Raises ``ValueError`` for one that cannot draw."""
    clean = str(name or "").strip().lower()
    if not clean:
        raise ValueError("The LLM Runner has no API provider. Pick Gemini (google), Grok or OpenAI there and add its key.")
    provider = _PROVIDER_ALIASES.get(clean)
    if provider is None:
        raise ValueError(
            f"The LLM Runner provider '{clean}' cannot draw pictures here. Pick Gemini (google), Grok or OpenAI in the LLM Runner."
        )
    return provider


def default_api_image_model(provider: str) -> str:
    return API_IMAGE_PROVIDERS[provider]["models"][0]


def _png_bytes(image: Image.Image) -> bytes:
    """The source frame as PNG, shrunk to ``_MAX_SOURCE_SIDE`` on its long side so the upload stays small."""
    image = image.convert("RGB")
    longest = max(image.size)
    if longest > _MAX_SOURCE_SIDE:
        scale = _MAX_SOURCE_SIDE / longest
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _read(request: urllib.request.Request, provider: str) -> bytes:
    try:
        with urlopen_verified(request, timeout=_TIMEOUT_SECONDS) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        hint = ""
        if provider == "Grok" and exc.code == 403 and "1010" in body:
            hint = " xAI blocked the request (code 1010): check the account, region or firewall."
        raise ApiImageError(f"{provider} HTTP {exc.code}: {body[:600]}{hint}", exc.code) from None
    except urllib.error.URLError as exc:
        raise ApiImageError(f"{provider} network error: {exc.reason}") from None


def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], provider: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": _USER_AGENT, **headers},
        method="POST",
    )
    return _parse_json(_read(request, provider), provider)


def _post_multipart(url: str, headers: Dict[str, str], fields: Dict[str, str], files: List[Tuple[str, str, str, bytes]], provider: str) -> Dict[str, Any]:
    boundary = f"----VRGDG{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
    for name, filename, mime, data in files:
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode("utf-8")
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json", "User-Agent": _USER_AGENT, **headers},
        method="POST",
    )
    return _parse_json(_read(request, provider), provider)


def _parse_json(raw: bytes, provider: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        raise ApiImageError(f"{provider} answered with something that is not JSON.") from None
    if not isinstance(data, dict):
        raise ApiImageError(f"{provider} answered with an unexpected reply.")
    return data


def _post_json_trimmed(url: str, headers: Dict[str, str], body: Dict[str, Any], trimmed: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """Post ``body``. When the provider answers 400 (an option it does not know) post ``trimmed``, the plain request, once."""
    try:
        return _post_json(url, headers, body, provider)
    except ApiImageError as exc:
        if exc.status != 400:
            raise
        return _post_json(url, headers, trimmed, provider)


def _open_image(raw: bytes, provider: str) -> Image.Image:
    try:
        with Image.open(BytesIO(raw)) as opened:
            return opened.convert("RGB")
    except Exception:
        raise ApiImageError(f"{provider} sent data that is not a picture.") from None


def _picture_from_data(reply: Dict[str, Any], provider: str) -> Image.Image:
    """The first picture of an OpenAI-style ``{"data": [{"b64_json" | "url": ...}]}`` reply."""
    items = reply.get("data")
    first = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else {}
    encoded = first.get("b64_json")
    if isinstance(encoded, str) and encoded.strip():
        try:
            return _open_image(base64.b64decode(encoded), provider)
        except ValueError:
            raise ApiImageError(f"{provider} sent a picture that could not be decoded.") from None
    link = first.get("url")
    if isinstance(link, str) and link.startswith("https://"):
        request = urllib.request.Request(link, headers={"User-Agent": _USER_AGENT}, method="GET")
        return _open_image(_read(request, provider), provider)
    error = reply.get("error")
    detail = error.get("message") if isinstance(error, dict) else error
    raise ApiImageError(f"{provider} returned no picture." + (f" {str(detail)[:300]}" if detail else ""))


def _gemini_image(api_key: str, model: str, prompt: str, source: Image.Image) -> Image.Image:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model, safe='-_.~')}:generateContent"
    headers = {"x-goog-api-key": api_key}
    parts = [
        {"text": prompt},
        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(_png_bytes(source)).decode("ascii")}},
    ]
    plain = {"contents": [{"role": "user", "parts": parts}]}
    wide = {**plain, "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": "16:9"}}}
    reply = _post_json_trimmed(url, headers, wide, plain, "Gemini")
    said: List[str] = []
    for candidate in reply.get("candidates") or []:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        for part in (content or {}).get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                return _open_image(base64.b64decode(inline["data"]), "Gemini")
            if isinstance(part.get("text"), str):
                said.append(part["text"].strip())
    feedback = reply.get("promptFeedback")
    reason = feedback.get("blockReason") if isinstance(feedback, dict) else ""
    detail = f" Blocked: {reason}." if reason else ""
    text = " ".join(item for item in said if item)[:300]
    raise ApiImageError(f"Gemini returned no picture.{detail}" + (f" It said: {text}" if text else ""))


def _grok_image(api_key: str, model: str, prompt: str, source: Image.Image) -> Image.Image:
    data_uri = "data:image/png;base64," + base64.b64encode(_png_bytes(source)).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}"}
    plain = {"model": model, "prompt": prompt, "image": {"url": data_uri, "type": "image_url"}}
    body = {**plain, "response_format": "b64_json"}
    reply = _post_json_trimmed("https://api.x.ai/v1/images/edits", headers, body, plain, "Grok")
    return _picture_from_data(reply, "Grok")


def _openai_image(api_key: str, model: str, prompt: str, source: Image.Image) -> Image.Image:
    fields = {"model": model, "prompt": prompt, "size": _OPENAI_SIZE}
    files = [("image", "frame.png", "image/png", _png_bytes(source))]
    reply = _post_multipart("https://api.openai.com/v1/images/edits", {"Authorization": f"Bearer {api_key}"}, fields, files, "OpenAI")
    return _picture_from_data(reply, "OpenAI")


_DRAWERS = {"gemini": _gemini_image, "grok": _grok_image, "openai": _openai_image}


def fit_wide(image: Image.Image) -> Image.Image:
    """Centre-crop to 16:9 when a provider answered with another shape (OpenAI has no 16:9 size). 16:9 stays as it is."""
    width, height = image.size
    ratio = width / height
    if abs(ratio - _WIDE) / _WIDE <= _WIDE_TOLERANCE:
        return image
    if ratio < _WIDE:
        crop_height = round(width / _WIDE)
        top = (height - crop_height) // 2
        return image.crop((0, top, width, top + crop_height))
    crop_width = round(height * _WIDE)
    left = (width - crop_width) // 2
    return image.crop((left, 0, left + crop_width, height))


def edit_with_api(provider: str, api_key: str, model: str, prompt: str, source: Image.Image) -> Image.Image:
    """Send the frame and the prompt to the provider and return the picture it drew, 16:9.

    ``provider`` is one of ``API_IMAGE_PROVIDERS`` (see ``api_image_provider``). Raises ``ApiImageError`` with a readable
    message when the provider refuses or sends no picture.
    """
    if provider not in _DRAWERS:
        raise ValueError(f"Unknown image provider: {provider}")
    if not str(api_key or "").strip():
        raise ValueError(f"Add your {API_IMAGE_PROVIDERS[provider]['label']} API key in the LLM Runner first.")
    if not str(prompt or "").strip():
        raise ValueError("Write the prompt first.")
    return fit_wide(_DRAWERS[provider](str(api_key).strip(), model, str(prompt).strip(), source))
