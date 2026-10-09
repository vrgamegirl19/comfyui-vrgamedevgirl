"""Speaking dialogue drafts and explicit ElevenLabs speech generation."""

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .elevenlabs import validate_api_key, validate_voice
from .elevenlabs_voice_design import required_text

SPEECH_MODELS = {"eleven_v4": 10000, "eleven_v3": 5000}


def normalize_dialogue(value: Any = None) -> dict[str, Any]:
    """Keep only durable per-scene editor fields, never generated audio or keys."""
    draft = value if isinstance(value, dict) else {}
    model = draft.get("model_id", "eleven_v4")
    if not isinstance(model, str) or model not in SPEECH_MODELS:
        model = "eleven_v4"
    return {
        "speaker_id": str(draft.get("speaker_id") or "")[:256],
        "text": str(draft.get("text") or "")[:SPEECH_MODELS[model]],
        "delivery": str(draft.get("delivery") or "")[:4000],
        "allow_rewrite": draft.get("allow_rewrite") is True,
        "model_id": model,
    }


def validate_dialogue(value: Any, require_text: bool = False) -> dict[str, Any]:
    """Validate an editor draft or a generation request without coercion."""
    if not isinstance(value, dict):
        raise ValueError("Scene dialogue must be an object.")
    model = value.get("model_id", "eleven_v4")
    if not isinstance(model, str) or model not in SPEECH_MODELS:
        raise ValueError("Choose Eleven v4 or Eleven v3 for tagged dialogue.")
    for key, limit in {
        "speaker_id": 256, "text": SPEECH_MODELS[model], "delivery": 4000,
    }.items():
        field = value.get(key, "")
        if not isinstance(field, str) or len(field) > limit:
            raise ValueError(f"{key} must be text with at most {limit} characters.")
    if not isinstance(value.get("allow_rewrite", False), bool):
        raise ValueError("allow_rewrite must be a boolean.")
    result = normalize_dialogue(value)
    if require_text:
        result["text"] = required_text(result["text"], "Dialogue", 1, SPEECH_MODELS[model])
    return result


def dialogue_speaker(references: Any, speaker_id: str) -> dict[str, Any]:
    """Resolve a primary character's enabled permanent voice assignment."""
    refs = references if isinstance(references, dict) else {}
    subject = next((item for item in refs.get("subjects", [])
                    if item.get("id") == speaker_id), None)
    if (
        not subject or subject.get("reference_type", "character") != "character"
        or subject.get("extra_reference_for")
    ):
        raise ValueError("Choose a primary character from Reference Builder.")
    voice = validate_voice(subject.get("elevenlabs_voice", {}))
    if not voice["enabled"] or not voice["voice_id"]:
        raise ValueError("Assign and enable an ElevenLabs voice for this character in Reference Builder.")
    return {
        **voice, "speaker_id": speaker_id,
        "character_name": subject.get("name", ""),
        "character_description": subject.get("description", ""),
    }


def generate_speech(
    api_key: str, voice_id: str, dialogue: dict[str, Any],
) -> dict[str, Any]:
    """Generate one MP3 take; no retries, timeline mutation or project writes."""
    draft = validate_dialogue(dialogue, require_text=True)
    voice = validate_voice({"enabled": True, "voice_id": voice_id})
    request = Request(
        "https://api.elevenlabs.io/v1/text-to-speech/"
        + quote(voice["voice_id"], safe="") + "?output_format=mp3_44100_128",
        data=json.dumps({
            "text": draft["text"], "model_id": draft["model_id"],
        }).encode("utf-8"),
        method="POST", headers={
            "xi-api-key": validate_api_key(api_key),
            "Content-Type": "application/json", "Accept": "audio/mpeg",
        },
    )
    try:
        with urlopen(request, timeout=180) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
            media = response.headers.get("Content-Type", "audio/mpeg").split(";")[0]
        if (
            not raw or len(raw) > 32 * 1024 * 1024
            or media not in ("audio/mpeg", "audio/mp3", "application/octet-stream")
        ):
            raise ValueError("ElevenLabs returned empty, oversized or unsupported speech audio.")
    except HTTPError as exc:
        code = exc.code
        exc.close()
        messages = {
            401: "ElevenLabs refused speech generation. Check the full key and Text to Speech permission.",
            403: "ElevenLabs denied speech generation. Check Text to Speech access and key IP restrictions.",
            404: "This voice or speech model is unavailable. Refresh voices or choose another model.",
            422: "ElevenLabs rejected this speech request. Check the dialogue, voice and model.",
            429: "ElevenLabs rate limit reached. Wait before generating another take.",
        }
        raise ValueError(messages.get(
            code, "ElevenLabs could not generate speech. "
            "Check your account credits and model access.",
        )) from None
    except (URLError, TimeoutError, OSError):
        raise ValueError(
            "Speech generation did not complete. Check your connection. "
            "The request may have used credits; it was not retried.",
        ) from None
    return {
        "audio_data": "data:audio/mpeg;base64," + base64.b64encode(raw).decode("ascii"),
        "audio_name": "elevenlabs_dialogue.mp3", "voice_id": voice["voice_id"],
        "dialogue": draft,
    }
