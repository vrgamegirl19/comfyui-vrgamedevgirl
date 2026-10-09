"""ElevenLabs account voice discovery, without speech generation or SDK dependencies."""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


def validate_api_key(value: Any, allow_empty: bool = False) -> str:
    """Validate a credential without including it in error messages."""
    if not isinstance(value, str):
        raise ValueError("ElevenLabs API key must be text.")
    key = value.strip()
    if (not key and not allow_empty) or len(key) > 512 or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ValueError("Enter a valid ElevenLabs API key in Builder Settings.")
    return key


def normalize_voice(value: Any = None) -> dict[str, Any]:
    """Normalize the durable subject assignment shared with the Reference Builder."""
    value = value if isinstance(value, dict) else {}
    return {
        "enabled": value.get("enabled") is True,
        "voice_id": str(value.get("voice_id") or "").strip(),
        "name": str(value.get("name") or "").strip(),
    }


def validate_voice(value: Any) -> dict[str, Any]:
    """Reject incomplete enabled assignments and unknown fields."""
    if not isinstance(value, dict) or set(value) - {"enabled", "voice_id", "name"}:
        raise ValueError("elevenlabs_voice must contain enabled, voice_id and name only.")
    if "enabled" in value and not isinstance(value["enabled"], bool):
        raise ValueError("elevenlabs_voice.enabled must be a boolean.")
    if any(not isinstance(value.get(k, ""), str) for k in ("voice_id", "name")):
        raise ValueError("Voice ID and name must be text.")
    voice = normalize_voice(value)
    voice_id = voice["voice_id"]
    if len(voice_id) > 128 or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in voice_id):
        raise ValueError("Invalid ElevenLabs voice ID.")
    if voice["enabled"] and not voice_id:
        raise ValueError("Choose a voice before enabling ElevenLabs for this character.")
    if len(voice["name"]) > 256:
        raise ValueError("Voice name is too long.")
    return voice


def _voice_request_error(error: HTTPError, voice_design: bool = False) -> str:
    """Translate known provider error codes without exposing response messages or secrets."""
    status = ""
    try:
        raw = error.read(8193)
        if len(raw) <= 8192:
            data = json.loads(raw)
            detail = data.get("detail") if isinstance(data, dict) else None
            if isinstance(detail, dict):
                status = detail.get("status")
    except (OSError, ValueError, UnicodeError):
        pass
    finally:
        error.close()
    if status == "missing_permissions":
        if voice_design:
            return "ElevenLabs reports missing voice-design permission. Enable Voice Generation and Voices write access for this key, then try again."
        return (
            "ElevenLabs reports missing permission to list voices. In ElevenLabs, open "
            "Developers → API Keys → Edit this key and enable Voices → Read (voices_read), "
            "then test again. A key used for text to speech may still lack voice-list access."
        )
    if status == "invalid_api_key":
        return (
            "ElevenLabs reports invalid_api_key. Enter the full API key in Builder Settings, "
            "rather than the key ID, name or masked value."
        )
    if voice_design:
        messages = {
            401: "ElevenLabs refused voice-design access (HTTP 401). Check your full key and voice-design permissions.",
            403: "ElevenLabs denied voice-design access (HTTP 403). Check key permissions and IP restrictions.",
            409: "ElevenLabs could not save this preview. It may already have been saved; refresh account voices before trying again.",
            422: "ElevenLabs rejected the voice-design fields. Check the description, preview text and chosen preview.",
            429: "ElevenLabs rate limit reached. Wait before trying voice design again.",
        }
        return messages.get(error.code, "ElevenLabs could not complete voice design. Check your account limits and try again later.")
    messages = {
        401: (
            "ElevenLabs refused voice access (HTTP 401). Check this key's Voices → Read "
            "permission and that the full API key was entered. This response alone does "
            "not confirm the key is invalid."
        ),
        403: (
            "ElevenLabs denied voice access (HTTP 403). Check this key's Voices → Read "
            "permission and any IP allowlist restrictions."
        ),
        429: "ElevenLabs rate limit reached. Wait and refresh voices again.",
    }
    return messages.get(error.code, "ElevenLabs could not list voices. Try again later.")


def list_voices(api_key: str, next_page_token: str = "") -> dict[str, Any]:
    """Fetch one page of available account voices. Never expose upstream error bodies."""
    key = validate_api_key(api_key)
    if not isinstance(next_page_token, str) or len(next_page_token) > 2048:
        raise ValueError("Invalid voice page token.")
    query = {"page_size": "100", "include_total_count": "false", "sort": "name"}
    if next_page_token:
        query["next_page_token"] = next_page_token
    request = Request(
        "https://api.elevenlabs.io/v2/voices?" + urlencode(query),
        headers={"xi-api-key": key, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=25) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("ElevenLabs returned an oversized voice list.")
        data = json.loads(raw)
    except HTTPError as exc:
        raise ValueError(_voice_request_error(exc)) from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("Could not connect to ElevenLabs. Check your connection and try again.") from None
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("ElevenLabs returned an invalid voice list.") from None
    if not isinstance(data, dict) or not isinstance(data.get("voices"), list):
        raise ValueError("ElevenLabs returned an invalid voice list.")
    voices = []
    for item in data["voices"]:
        if not isinstance(item, dict) or not item.get("voice_id"):
            continue
        preview = str(item.get("preview_url") or "")
        try:
            parsed = urlsplit(preview)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                preview = ""
        except ValueError:
            preview = ""
        voices.append({
            "voice_id": str(item["voice_id"]), "name": str(item.get("name") or item["voice_id"]),
            "category": str(item.get("category") or ""), "preview_url": preview,
        })
    token = data.get("next_page_token")
    has_more = bool(data.get("has_more"))
    if has_more and (not isinstance(token, str) or not token or token == next_page_token):
        raise ValueError("ElevenLabs returned an invalid voice page token. Refresh voices again.")
    return {"voices": voices, "has_more": has_more, "next_page_token": token if has_more else ""}
