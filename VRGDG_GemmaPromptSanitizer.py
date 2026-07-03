import json
import re


_PROMPT_KEYS = (
    "image_prompt",
    "t2i_prompt",
    "text_to_image_prompt",
    "prompt",
    "flux_prompt",
    "nb_prompt",
    "nano_banana_prompt",
    "ernie_prompt",
    "enhance_prompt",
)


def _strip_json_fence(text):
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _json_candidates(text):
    cleaned = _strip_json_fence(text)
    yield cleaned
    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
    if starts:
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end > start:
            yield cleaned[start:end + 1]


def _scene_number(value):
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    try:
        number = int(match.group(0))
        return number if number > 0 else None
    except Exception:
        return None


def _walk_prompt_values(value):
    if isinstance(value, dict):
        for key in _PROMPT_KEYS:
            text = str(value.get(key) or "").strip()
            if text:
                yield text
        for child in value.values():
            yield from _walk_prompt_values(child)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_prompt_values(item)


def _items_for_scene(parsed, target_scene_number):
    if isinstance(parsed, list):
        items = [item for item in parsed if isinstance(item, dict)]
    elif isinstance(parsed, dict):
        for key in ("scenes", "prompts", "items", "results"):
            if isinstance(parsed.get(key), list):
                items = [item for item in parsed[key] if isinstance(item, dict)]
                break
        else:
            items = [parsed]
    else:
        items = []
    if target_scene_number:
        matched = [
            item for item in items
            if _scene_number(item.get("scene_number") or item.get("sceneNumber") or item.get("scene") or item.get("number")) == target_scene_number
        ]
        if matched:
            return matched
    return items


def extract_prompt_text_from_gemma_output(text, scene_number=None):
    original = str(text or "").strip()
    if not original:
        return original
    target_scene_number = _scene_number(scene_number)
    for candidate in _json_candidates(original):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        for item in _items_for_scene(parsed, target_scene_number):
            for prompt in _walk_prompt_values(item):
                return prompt
        for prompt in _walk_prompt_values(parsed):
            return prompt
    return original
