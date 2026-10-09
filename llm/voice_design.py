"""Voice description writing through the Video Builder's selected LLM runner."""

from typing import Any

from ..builder.elevenlabs_voice_design import required_text
from .builder_runner import _builder_local_model_file, _run_builder_text_llm


def generate_voice_description(payload: dict[str, Any]) -> dict[str, Any]:
    """Return an editable voice description without calling ElevenLabs or saving a voice."""
    brief = required_text(payload.get("user_input"), "Voice brief", 1, 4000)
    character = str(payload.get("character_description") or "")[:4000]
    name = str(payload.get("character_name") or "")[:256]
    request = dict(payload)
    request["model_file"] = _builder_local_model_file(request, request.get("model_file", ""))
    instruction = (
        "Write one voice description for ElevenLabs Voice Design using the user's brief. "
        "Return only a single paragraph of 20 to 1000 characters, preferably 300–700. "
        "Describe audible qualities: perceived age and vocal range, accent when specified, "
        "timbre, texture, resonance, rhythm, cadence, articulation and delivery. "
        "Preserve the user's intent. Use character context only to support the brief; "
        "do not include appearance, clothing, locations, stage directions or dialogue. "
        "Do not add a name, title, markdown, quotation marks, explanations or multiple options.\n\n"
        f"User voice brief:\n{brief}\n\nCharacter name: {name}\nCharacter context:\n{character}"
    )
    text, info = _run_builder_text_llm(
        request, instruction, temperature=0.5, max_new_tokens=500,
        label="Voice Description", preserve_paragraphs=True,
    )
    description = required_text(text, "LLM voice description", 20, 1000)
    return {
        "voice_description": description, "runner": info.get("runner", ""),
        "used_model": info.get("used_model", ""),
    }
