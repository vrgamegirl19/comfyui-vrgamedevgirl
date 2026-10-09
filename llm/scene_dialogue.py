"""Reviewable dialogue preparation through the selected LLM Runner."""

import re
from typing import Any

from ..builder.elevenlabs_speech import SPEECH_MODELS, validate_dialogue
from .builder_instructions import _effective_builder_instruction
from .builder_runner import _builder_local_model_file, _run_builder_text_llm
from .prompts.speech import ELEVENLABS_DIALOGUE_INSTRUCTIONS


def spoken_words(text: str) -> list[str]:
    """Compare actual words while allowing audio tags and punctuation changes."""
    return re.findall(r"\w+(?:['’]\w+)*", re.sub(r"\[[^\]]*\]", "", text))


def craft_dialogue(payload: dict[str, Any]) -> dict[str, Any]:
    """Return editable tagged dialogue, without calling ElevenLabs or persisting it."""
    draft = validate_dialogue(payload.get("dialogue"), require_text=True)
    request = dict(payload)
    request["model_file"] = _builder_local_model_file(request, request.get("model_file", ""))
    instructions = _effective_builder_instruction(
        request, "elevenlabs_dialogue", ELEVENLABS_DIALOGUE_INSTRUCTIONS,
    )
    prompt = (
        f"{instructions}\n\nRewrite allowed: {draft['allow_rewrite']}\n"
        f"Maximum output characters: {SPEECH_MODELS[draft['model_id']]}\n"
        f"Character: {str(payload.get('character_name') or '')[:256]}\n"
        f"Character context: {str(payload.get('character_description') or '')[:4000]}\n"
        f"Delivery direction: {draft['delivery']}\nDialogue:\n{draft['text']}"
    )
    text, info = _run_builder_text_llm(
        request, prompt, temperature=0.5, max_new_tokens=1800,
        label="Dialogue", preserve_paragraphs=True,
    )
    result = validate_dialogue({**draft, "text": text}, require_text=True)
    if not draft["allow_rewrite"] and spoken_words(result["text"]) != spoken_words(draft["text"]):
        raise ValueError("The LLM changed the spoken words. Try again, add tags manually, or enable Allow rewriting.")
    return {
        "dialogue": result, "runner": info.get("runner", ""),
        "used_model": info.get("used_model", ""),
    }
