"""Validated ElevenLabs voice previews and explicit account voice creation."""

import base64
import binascii
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .elevenlabs import _voice_request_error, validate_api_key, validate_voice

DESIGN_MODELS = ("eleven_ttv_v3", "eleven_multilingual_ttv_v2")


def required_text(value: Any, label: str, minimum: int, maximum: int) -> str:
    """Validate a text field using the provider's character limits."""
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ValueError(f"{label} must be {minimum}–{maximum} characters.")
    return value.strip()


def normalize_design_draft(value: Any = None) -> dict[str, str]:
    """Normalize durable editing fields; never persist preview audio or temporary IDs."""
    draft = value if isinstance(value, dict) else {}
    limits = {
        "user_input": 4000, "voice_description": 1000,
        "preview_text": 1000, "voice_name": 256,
    }
    result = {key: str(draft.get(key) or "")[:limit] for key, limit in limits.items()}
    model = draft.get("model_id")
    result["model_id"] = model if model in DESIGN_MODELS else DESIGN_MODELS[0]
    return result


def validate_design_draft(value: Any) -> dict[str, str]:
    """Reject malformed API drafts and keep only durable editor fields."""
    if not isinstance(value, dict):
        raise ValueError("Voice Design settings must be an object.")
    limits = {
        "user_input": 4000, "voice_description": 1000,
        "preview_text": 1000, "voice_name": 256,
    }
    for key, maximum in limits.items():
        field = value.get(key, "")
        if not isinstance(field, str) or len(field) > maximum:
            raise ValueError(f"{key} must be text with at most {maximum} characters.")
    if value.get("model_id", DESIGN_MODELS[0]) not in DESIGN_MODELS:
        raise ValueError("Choose Voice Design v3 or v2.")
    return normalize_design_draft(value)


def _post_voice_design(
    api_key: str, path: str, body: dict[str, Any], saving: bool = False,
) -> dict[str, Any]:
    """Make one explicit request; never retry account mutations automatically."""
    request = Request(
        "https://api.elevenlabs.io" + path,
        data=json.dumps(body).encode("utf-8"), method="POST",
        headers={
            "xi-api-key": validate_api_key(api_key),
            "Content-Type": "application/json", "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("ElevenLabs returned an oversized voice-design response.")
        data = json.loads(raw)
    except HTTPError as exc:
        message = _voice_request_error(exc, voice_design=True)
        if saving and (exc.code >= 500 or exc.code == 408):
            message += (
                " The voice may already have been created. "
                "Refresh account voices before trying Save again."
            )
        raise ValueError(message) from None
    except (URLError, TimeoutError, OSError):
        message = "Could not complete the ElevenLabs voice-design request. Check your connection."
        if saving:
            message += (
                " The voice may already have been created. "
                "Refresh account voices before trying Save again."
            )
        raise ValueError(message) from None
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError(
            "ElevenLabs returned an invalid voice-design response. "
            "Refresh account voices if you were saving."
        ) from None
    if not isinstance(data, dict):
        raise ValueError("ElevenLabs returned an invalid voice-design response.")
    return data


def design_voice(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Generate preview candidates from a reviewed description, without creating a saved voice."""
    description = required_text(
        payload.get("voice_description"), "Voice description", 20, 1000,
    )
    model = payload.get("model_id", DESIGN_MODELS[0])
    if model not in DESIGN_MODELS:
        raise ValueError("Choose Voice Design v3 or v2.")
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Preview text must be text.")
    body = {
        "voice_description": description, "model_id": model,
        "stream_previews": False, "should_enhance": False,
    }
    if text.strip():
        body["text"] = required_text(text, "Preview text", 100, 1000)
        body["auto_generate_text"] = False
    else:
        body["auto_generate_text"] = True
    data = _post_voice_design(api_key, "/v1/text-to-voice/design", body)
    candidates = data.get("previews")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 10:
        raise ValueError("ElevenLabs returned no usable voice previews.")
    previews = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("ElevenLabs returned an invalid voice preview.")
        voice = validate_voice({
            "enabled": True, "voice_id": candidate.get("generated_voice_id"),
        })
        audio = candidate.get("audio_base_64")
        media = candidate.get("media_type")
        if (
            media not in ("audio/mpeg", "audio/mp3")
            or not isinstance(audio, str) or len(audio) > 12 * 1024 * 1024
        ):
            raise ValueError("ElevenLabs returned an unsupported voice preview.")
        try:
            if not base64.b64decode(audio, validate=True):
                raise ValueError("Empty preview.")
        except (ValueError, binascii.Error):
            raise ValueError("ElevenLabs returned invalid preview audio.") from None
        previews.append({
            "generated_voice_id": voice["voice_id"],
            "audio_base_64": audio, "media_type": "audio/mpeg",
        })
    return {
        "previews": previews, "text": str(data.get("text") or text),
        "voice_description": description, "model_id": model,
    }


def create_designed_voice(
    api_key: str, payload: dict[str, Any],
) -> dict[str, str]:
    """Explicitly save one chosen generated voice to the ElevenLabs account."""
    name = required_text(payload.get("voice_name"), "Voice name", 1, 256)
    description = required_text(payload.get("voice_description"), "Voice description", 20, 1000)
    voice = validate_voice({"enabled": True, "voice_id": payload.get("generated_voice_id")})
    data = _post_voice_design(api_key, "/v1/text-to-voice", {
        "voice_name": name, "voice_description": description, "generated_voice_id": voice["voice_id"],
    }, saving=True)
    saved = validate_voice({"enabled": True, "voice_id": data.get("voice_id"), "name": name})
    return {
        "voice_id": saved["voice_id"], "name": name,
        "category": "generated", "voice_description": description,
    }
