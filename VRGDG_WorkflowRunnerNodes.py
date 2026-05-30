import copy
import base64
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import time

import folder_paths
from aiohttp import web
from server import PromptServer


_VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED = False
_MAX_LORA_SLOTS = 20
_NONE_LORA = "[none]"
_I2V_UNET_ALIASES = {
    "LTX-2.3-22B-distilled-11-Q6_K.gguf": "LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
}
_PLACEHOLDER_I2I_IMAGE_NAME = "vrgdg_placeholder_i2i.png"
_PLACEHOLDER_I2I_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _workflow_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "text2image_zimage.json",
    )


def _zimage_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "text2image_zimage_API.json",
    )


def _flux_klein_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "fluxKleinMultiImage_API.json",
    )


def _ernie_image_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "image_ernie_image_turbo_API.json",
    )


def _nb_image_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "NB_API.json",
    )


def _z_upscale_enhance_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "z_upscaleEnhance.json",
    )


def _z_upscale_enhance_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "z_upscaleEnhance_API.json",
    )


def _i2v_workflow_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Singlei2vForUI.json",
    )


def _i2v_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Singlei2vForUI_API.json",
    )


def _t2v_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Singlet2vForUI_API.json",
    )


def _clear_memory_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "ClearMemory_API.json",
    )


def _lora_choices():
    try:
        loras = folder_paths.get_filename_list("loras")
    except Exception:
        loras = []
    return [_NONE_LORA] + [name for name in loras if str(name or "").strip() != _NONE_LORA]


def _folder_choices(category):
    if isinstance(category, (list, tuple)):
        values = []
        for item in category:
            values.extend(_folder_choices(item))
        seen = set()
        unique = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return unique
    try:
        return folder_paths.get_filename_list(category)
    except Exception:
        return []


def _clean_i2v_unet_name(value):
    text = str(value or "").strip()
    return _I2V_UNET_ALIASES.get(text, text)


def _load_workflow_template(path=None):
    raw_path = str(path or "").strip()
    if raw_path and not os.path.isabs(raw_path):
        raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), raw_path)
    workflow_path = os.path.abspath(raw_path or _workflow_template_path())
    if not os.path.isfile(workflow_path):
        raise FileNotFoundError(f"Workflow template was not found: {workflow_path}")
    with open(workflow_path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        raise ValueError("Workflow template is not a valid ComfyUI workflow JSON.")
    return workflow_path, workflow


def _load_api_template(path):
    api_path = os.path.abspath(path)
    if not os.path.isfile(api_path):
        raise FileNotFoundError(f"Workflow API template was not found: {api_path}")
    with open(api_path, "r", encoding="utf-8") as handle:
        prompt = json.load(handle)
    if not isinstance(prompt, dict) or not prompt:
        raise ValueError("Workflow API template is not a valid ComfyUI API prompt JSON.")
    return api_path, prompt


def _node_by_id(workflow, node_id):
    target = str(node_id)
    for node in workflow.get("nodes", []):
        if str(node.get("id")) == target:
            return node
    raise KeyError(f"Workflow node {node_id} was not found.")


def _set_widget(workflow, node_id, widget_index, value):
    node = _node_by_id(workflow, node_id)
    widgets = node.setdefault("widgets_values", [])
    if isinstance(widgets, dict):
        widgets[str(widget_index)] = value
        return
    while len(widgets) <= widget_index:
        widgets.append(None)
    widgets[widget_index] = value


def _set_widget_key(workflow, node_id, key, value):
    node = _node_by_id(workflow, node_id)
    widgets = node.setdefault("widgets_values", {})
    if not isinstance(widgets, dict):
        raise TypeError(f"Workflow node {node_id} does not use keyed widget values.")
    widgets[key] = value


def _workflow_node_id_by_class(workflow, class_type, fallback=None):
    for node in workflow.get("nodes", []):
        if node.get("type") == class_type or node.get("class_type") == class_type:
            return str(node.get("id"))
    if fallback is not None:
        _node_by_id(workflow, fallback)
        return str(fallback)
    raise KeyError(f"Workflow node class {class_type} was not found.")


def _api_node_id_by_class(prompt, class_type, fallback=None):
    for node_id, node in prompt.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return str(node_id)
    if fallback is not None and str(fallback) in prompt:
        return str(fallback)
    raise KeyError(f"API prompt node class {class_type} was not found.")


def _int_payload(payload, key, default, minimum=1, maximum=16384):
    try:
        value = int(payload.get(key, default))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _float_payload(payload, key, default, minimum=-100.0, maximum=100.0):
    try:
        value = float(payload.get(key, default))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _bool_payload(payload, key, default=False):
    value = payload.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clean_lora_name(value):
    text = str(value or _NONE_LORA).strip()
    choices = set(_lora_choices())
    if text not in choices:
        return _NONE_LORA
    return text


def _patch_zimage_workflow(workflow, payload):
    workflow = copy.deepcopy(workflow)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    if not prompt_text:
        raise ValueError("Prompt text is empty.")

    first_width = _int_payload(payload, "first_pass_width", 1280, 64, 4096)
    first_height = _int_payload(payload, "first_pass_height", 720, 64, 4096)
    second_width = _int_payload(payload, "second_pass_width", 1920, 64, 4096)
    second_height = _int_payload(payload, "second_pass_height", 1080, 64, 4096)
    batch_size = _int_payload(payload, "batch_size", 1, 1, 16)
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)

    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    ltx_two_pass_mode = _bool_payload(payload, "ltx_two_pass_mode", False)

    _set_widget(workflow, 971, 0, prompt_text)
    _set_widget(workflow, 960, 0, str(payload.get("clip_name", "") or ""))
    _set_widget(workflow, 961, 0, str(payload.get("vae_name", "") or ""))
    _set_widget(workflow, 972, 0, str(payload.get("unet_name", "") or ""))
    _set_widget(workflow, 965, 0, first_width)
    _set_widget(workflow, 965, 1, first_height)
    _set_widget(workflow, 965, 2, batch_size)
    _set_widget(workflow, 967, 1, second_width)
    _set_widget(workflow, 967, 2, second_height)
    _set_widget(workflow, 964, 1, seed)
    _set_widget(workflow, 966, 1, seed)

    lora_node_id = _workflow_node_id_by_class(workflow, "VRGDG_OptionalMultiLoraTwoPassStrengths", fallback=974)
    lora_node = _node_by_id(workflow, lora_node_id)
    is_two_pass_lora = lora_node.get("type") == "VRGDG_OptionalMultiLoraTwoPassStrengths" or lora_node.get("class_type") == "VRGDG_OptionalMultiLoraTwoPassStrengths"
    _set_widget(workflow, lora_node_id, 0, use_custom_loras)
    _set_widget(workflow, lora_node_id, 1, lora_count)
    if is_two_pass_lora:
        for slot in range(1, _MAX_LORA_SLOTS + 1):
            lora_name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA))
            legacy_strength = _float_payload(payload, f"strength_{slot}", 1.0)
            first_pass_strength = _float_payload(payload, f"first_pass_strength_{slot}", legacy_strength)
            second_pass_strength = _float_payload(payload, f"second_pass_strength_{slot}", legacy_strength)
            base_index = 2 + ((slot - 1) * 3)
            _set_widget(workflow, lora_node_id, base_index, lora_name)
            _set_widget(workflow, lora_node_id, base_index + 1, first_pass_strength)
            _set_widget(workflow, lora_node_id, base_index + 2, second_pass_strength)
    else:
        _set_widget(workflow, lora_node_id, 2, ltx_two_pass_mode)
        for slot in range(1, _MAX_LORA_SLOTS + 1):
            lora_name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA))
            strength = _float_payload(payload, f"strength_{slot}", 1.0)
            base_index = 3 + ((slot - 1) * 2)
            _set_widget(workflow, lora_node_id, base_index, lora_name)
            _set_widget(workflow, lora_node_id, base_index + 1, strength)

    return workflow


def _prepare_load_image_name(path="", data="", name="image.png"):
    raw_path = str(path or "").strip().strip('"')
    if raw_path:
        source_path = os.path.abspath(raw_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Image-to-image source was not found: {source_path}")
        ext = os.path.splitext(source_path)[1].lower() or ".png"
        input_dir = folder_paths.get_input_directory()
        target_name = f"vrgdg_i2i_{int(time.time() * 1000)}{ext}"
        shutil.copy2(source_path, os.path.join(input_dir, target_name))
        return target_name

    raw_data = str(data or "").strip()
    if raw_data:
        if "," in raw_data and raw_data.lower().startswith("data:"):
            header, encoded = raw_data.split(",", 1)
            ext = ".png"
            if "jpeg" in header.lower() or "jpg" in header.lower():
                ext = ".jpg"
            elif "webp" in header.lower():
                ext = ".webp"
        else:
            encoded = raw_data
            ext = os.path.splitext(str(name or ""))[1].lower() or ".png"
        input_dir = folder_paths.get_input_directory()
        target_name = f"vrgdg_i2i_{int(time.time() * 1000)}{ext}"
        with open(os.path.join(input_dir, target_name), "wb") as handle:
            handle.write(base64.b64decode(encoded))
        return target_name

    return ""


def _resolve_existing_file(raw_path, label="file"):
    text = str(raw_path or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError(f"{label} path is empty.")

    candidates = []
    if os.path.isabs(text):
        candidates.append(text)
    else:
        candidates.extend(
            [
                text,
                os.path.abspath(text),
                os.path.join(folder_paths.get_input_directory(), text),
                os.path.join(folder_paths.get_output_directory(), text),
            ]
        )
        get_temp_directory = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_directory):
            candidates.append(os.path.join(get_temp_directory(), text))

    seen = set()
    for candidate in candidates:
        path = os.path.normpath(os.path.abspath(candidate))
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(f"{label} was not found: {text}")


def _ensure_placeholder_load_image():
    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)
    target_path = os.path.join(input_dir, _PLACEHOLDER_I2I_IMAGE_NAME)
    if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
        return _PLACEHOLDER_I2I_IMAGE_NAME

    source_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "images",
        _PLACEHOLDER_I2I_IMAGE_NAME,
    )
    if os.path.isfile(source_path) and os.path.getsize(source_path) > 0:
        shutil.copy2(source_path, target_path)
    else:
        with open(target_path, "wb") as handle:
            handle.write(base64.b64decode(_PLACEHOLDER_I2I_IMAGE_BASE64))
    return _PLACEHOLDER_I2I_IMAGE_NAME


def _patch_zimage_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    if not prompt_text:
        raise ValueError("Prompt text is empty.")

    first_width = _int_payload(payload, "first_pass_width", 1280, 64, 4096)
    first_height = _int_payload(payload, "first_pass_height", 720, 64, 4096)
    second_width = _int_payload(payload, "second_pass_width", 1920, 64, 4096)
    second_height = _int_payload(payload, "second_pass_height", 1080, 64, 4096)
    batch_size = _int_payload(payload, "batch_size", 1, 1, 16)
    seed_mode = str(payload.get("seed_mode", "fixed") or "fixed").strip().lower()
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    if seed_mode in {"random", "randomize"}:
        seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    use_i2i = _bool_payload(payload, "use_image_to_image", False)
    start_at_step = _int_payload(payload, "image_to_image_start_at_step", 5, 1, 8)

    _set_api_input(prompt, "971", "text", prompt_text)
    _set_api_input(prompt, "960", "clip_name", str(payload.get("clip_name", "") or ""))
    _set_api_input(prompt, "961", "vae_name", str(payload.get("vae_name", "") or ""))
    _set_api_input(prompt, "972", "unet_name", str(payload.get("unet_name", "") or ""))
    _set_api_input(prompt, "965", "width", first_width)
    _set_api_input(prompt, "965", "height", first_height)
    _set_api_input(prompt, "965", "batch_size", batch_size)
    _set_api_input(prompt, "967", "width", second_width)
    _set_api_input(prompt, "967", "height", second_height)
    _set_api_input(prompt, "964", "noise_seed", seed)
    _set_api_input(prompt, "966", "noise_seed", seed)

    _set_api_input(prompt, "978", "switch", use_i2i)
    _set_api_input(prompt, "981", "switch", use_i2i)
    _set_api_input(prompt, "983", "value", start_at_step)
    _set_api_input(prompt, "979", "image", _ensure_placeholder_load_image())
    if use_i2i:
        image_name = _prepare_load_image_name(
            payload.get("image_to_image_path", ""),
            payload.get("image_to_image_data", ""),
            payload.get("image_to_image_name", "image.png"),
        )
        if not image_name:
            raise ValueError("Image-to-image is enabled, but no source image was provided.")
        _set_api_input(prompt, "979", "image", image_name)

    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    ltx_two_pass_mode = _bool_payload(payload, "ltx_two_pass_mode", False)
    lora_node_id = _api_node_id_by_class(prompt, "VRGDG_OptionalMultiLoraTwoPassStrengths", fallback=974)
    is_two_pass_lora = prompt.get(str(lora_node_id), {}).get("class_type") == "VRGDG_OptionalMultiLoraTwoPassStrengths"
    _set_api_input(prompt, lora_node_id, "use_custom_loras", use_custom_loras)
    _set_api_input(prompt, lora_node_id, "lora_count", lora_count)
    if is_two_pass_lora:
        for slot in range(1, _MAX_LORA_SLOTS + 1):
            legacy_strength = _float_payload(payload, f"strength_{slot}", 1.0)
            first_pass_strength = _float_payload(payload, f"first_pass_strength_{slot}", legacy_strength)
            second_pass_strength = _float_payload(payload, f"second_pass_strength_{slot}", legacy_strength)
            _set_api_input(prompt, lora_node_id, f"lora_{slot}", _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)))
            _set_api_input(prompt, lora_node_id, f"first_pass_strength_{slot}", first_pass_strength)
            _set_api_input(prompt, lora_node_id, f"second_pass_strength_{slot}", second_pass_strength)
    else:
        _set_api_input(prompt, lora_node_id, "ltx_two_pass_mode", ltx_two_pass_mode)
        for slot in range(1, _MAX_LORA_SLOTS + 1):
            _set_api_input(prompt, lora_node_id, f"lora_{slot}", _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)))
            _set_api_input(prompt, lora_node_id, f"strength_{slot}", _float_payload(payload, f"strength_{slot}", 1.0))
    return prompt, seed


def _patch_ernie_image_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    if not prompt_text:
        raise ValueError("Prompt text is empty.")

    width = _int_payload(payload, "width", 1280, 64, 4096)
    height = _int_payload(payload, "height", 720, 64, 4096)
    batch_size = _int_payload(payload, "batch_size", 1, 1, 16)
    seed_mode = str(payload.get("seed_mode", "fixed") or "fixed").strip().lower()
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    if seed_mode in {"random", "randomize"}:
        seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    use_i2i = _bool_payload(payload, "use_image_to_image", False)
    start_at_step = _int_payload(payload, "image_to_image_start_at_step", 5, 1, 8)

    _set_api_input(prompt, "111", "text", prompt_text)
    _set_api_input(prompt, "105", "unet_name", str(payload.get("unet_name", "") or ""))
    _set_api_input(prompt, "108", "clip_name", str(payload.get("clip_name", "") or ""))
    _set_api_input(prompt, "109", "vae_name", str(payload.get("vae_name", "") or ""))
    for node_id in ("104", "120"):
        _set_api_input(prompt, node_id, "width", width)
        _set_api_input(prompt, node_id, "height", height)
        _set_api_input(prompt, node_id, "batch_size", batch_size)
    _set_api_input(prompt, "121", "noise_seed", seed)

    _set_api_input(prompt, "114", "switch", use_i2i)
    _set_api_input(prompt, "117", "switch", use_i2i)
    _set_api_input(prompt, "115", "value", start_at_step)
    _set_api_input(prompt, "118", "image", _ensure_placeholder_load_image())
    if use_i2i:
        image_name = _prepare_load_image_name(
            payload.get("image_to_image_path", ""),
            payload.get("image_to_image_data", ""),
            payload.get("image_to_image_name", "image.png"),
        )
        if not image_name:
            raise ValueError("Image-to-image is enabled, but no source image was provided.")
        _set_api_input(prompt, "118", "image", image_name)

    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    _set_api_input(prompt, "113", "use_custom_loras", use_custom_loras)
    _set_api_input(prompt, "113", "lora_count", lora_count)
    _set_api_input(prompt, "113", "ltx_two_pass_mode", False)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        _set_api_input(prompt, "113", f"lora_{slot}", _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)))
        _set_api_input(prompt, "113", f"strength_{slot}", _float_payload(payload, f"strength_{slot}", 1.0))
    return prompt, seed


def _patch_flux_klein_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    if not prompt_text:
        raise ValueError("Flux/Klein prompt text is empty.")

    ingredients = payload.get("image_ingredients") or payload.get("images") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("Flux/Klein image ingredients must be a list.")

    image_paths = []
    input_dir = folder_paths.get_input_directory()
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path", "") or "").strip()
        raw_data = str(item.get("data", "") or "").strip()
        raw_name = str(item.get("name", "") or f"ingredient_{index}.png").strip() or f"ingredient_{index}.png"
        if raw_data:
            load_image_name = _prepare_load_image_name("", raw_data, raw_name)
            image_paths.append(os.path.abspath(os.path.join(input_dir, load_image_name)))
        elif raw_path:
            image_paths.append(os.path.abspath(_resolve_existing_file(raw_path, f"Flux/Klein ingredient image {index}")))

    if not image_paths:
        raise ValueError("Flux/Klein needs at least one image ingredient.")

    width = _int_payload(payload, "width", 1024, 64, 4096)
    height = _int_payload(payload, "height", 576, 64, 4096)
    seed = _int_payload(payload, "seed", 100, 0, 0xFFFFFFFFFFFFFFFF)

    _set_api_input(prompt, "1067", "text", prompt_text)
    if "1065" in prompt:
        _set_api_input(prompt, "1065", "width", width)
        _set_api_input(prompt, "1065", "height", height)
    if "1052" in prompt:
        _set_api_input(prompt, "1052", "width", width)
        _set_api_input(prompt, "1052", "height", height)
    if "1057" in prompt:
        _set_api_input(prompt, "1057", "width", width)
        _set_api_input(prompt, "1057", "height", height)
        _set_api_input(prompt, "1057", "batch_size", 1)
    _set_api_input(prompt, "1056", "noise_seed", seed)
    _set_api_input(prompt, "1068", "unet_name", str(payload.get("unet_name", "") or ""))
    _set_api_input(prompt, "1066", "clip_name", str(payload.get("clip_name", "") or ""))
    _set_api_input(prompt, "1064", "vae_name", str(payload.get("vae_name", "") or ""))
    lora_node_id = _api_node_id_by_class(prompt, "VRGDG_OptionalMultiLoraModelOnly", fallback=1075)
    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    _set_api_input(prompt, lora_node_id, "use_custom_loras", use_custom_loras)
    _set_api_input(prompt, lora_node_id, "lora_count", lora_count)
    if "ltx_two_pass_mode" in prompt[lora_node_id].get("inputs", {}):
        _set_api_input(prompt, lora_node_id, "ltx_two_pass_mode", False)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        _set_api_input(prompt, lora_node_id, f"lora_{slot}", _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)))
        _set_api_input(prompt, lora_node_id, f"strength_{slot}", _float_payload(payload, f"strength_{slot}", 1.0))
    _set_api_input(prompt, "1072", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    return prompt


def _image_paths_from_payload_ingredients(payload, label="image ingredient"):
    ingredients = payload.get("image_ingredients") or payload.get("images") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError(f"{label.title()}s must be a list.")

    image_paths = []
    input_dir = folder_paths.get_input_directory()
    for index, item in enumerate(ingredients, start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path", "") or "").strip()
        raw_data = str(item.get("data", "") or "").strip()
        raw_name = str(item.get("name", "") or f"{label}_{index}.png").strip() or f"{label}_{index}.png"
        if raw_data:
            load_image_name = _prepare_load_image_name("", raw_data, raw_name)
            image_paths.append(os.path.abspath(os.path.join(input_dir, load_image_name)))
        elif raw_path:
            image_paths.append(os.path.abspath(_resolve_existing_file(raw_path, f"{label.title()} {index}")))
    return image_paths


def _patch_nb_image_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    api_key = str(payload.get("api_key", "") or "").strip()
    if not prompt_text:
        raise ValueError("NanoBanana prompt text is empty.")
    if not api_key:
        raise ValueError("NanoBanana needs an API key.")

    image_paths = _image_paths_from_payload_ingredients(payload, "NanoBanana reference image")
    if not image_paths:
        raise ValueError("NanoBanana needs at least one reference image.")

    _set_api_input(prompt, "1", "api_key", api_key)
    _set_api_input(prompt, "1", "prompt", prompt_text)
    _set_api_input(prompt, "1", "model", str(payload.get("model", "") or "gemini-3-pro-image-preview"))
    _set_api_input(prompt, "3", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    return prompt


def _patch_z_upscale_enhance_workflow(workflow, payload):
    workflow = copy.deepcopy(workflow)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    seed_mode = str(payload.get("seed_mode", "fixed") or "fixed").strip().lower()
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    if seed_mode in {"random", "randomize"}:
        seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    enhance_amount = _int_payload(payload, "enhance_amount", 8, 1, 20)

    image_name = _prepare_load_image_name(
        payload.get("source_image_path", ""),
        payload.get("source_image_data", ""),
        payload.get("source_image_name", "source.png"),
    )
    if not image_name:
        raise ValueError("Upscale/enhance needs a source image.")

    _set_widget(workflow, 960, 0, str(payload.get("clip_name", "") or ""))
    _set_widget(workflow, 961, 0, str(payload.get("vae_name", "") or ""))
    _set_widget(workflow, 972, 0, str(payload.get("unet_name", "") or ""))
    _set_widget(workflow, 971, 0, prompt_text)
    _set_widget(workflow, 967, 1, width)
    _set_widget(workflow, 967, 2, height)
    _set_widget(workflow, 979, 0, image_name)
    _set_widget(workflow, 983, 0, enhance_amount)
    _set_widget(workflow, 983, 1, "fixed")
    _set_widget(workflow, 964, 1, seed)
    _set_widget(workflow, 964, 2, "fixed")

    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    _set_widget(workflow, 974, 0, use_custom_loras)
    _set_widget(workflow, 974, 1, lora_count)
    _set_widget(workflow, 974, 2, False)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        lora_name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA))
        strength = _float_payload(payload, f"strength_{slot}", 1.0)
        base_index = 3 + (slot - 1) * 2
        _set_widget(workflow, 974, base_index, lora_name)
        _set_widget(workflow, 974, base_index + 1, strength)

    return workflow, seed


def _patch_z_upscale_enhance_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    seed_mode = str(payload.get("seed_mode", "fixed") or "fixed").strip().lower()
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    if seed_mode in {"random", "randomize"}:
        seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    enhance_amount = _int_payload(payload, "enhance_amount", 8, 1, 20)

    image_name = _prepare_load_image_name(
        payload.get("source_image_path", ""),
        payload.get("source_image_data", ""),
        payload.get("source_image_name", "source.png"),
    )
    if not image_name:
        raise ValueError("Upscale/enhance needs a source image.")

    _set_api_input(prompt, "960", "clip_name", str(payload.get("clip_name", "") or ""))
    _set_api_input(prompt, "961", "vae_name", str(payload.get("vae_name", "") or ""))
    _set_api_input(prompt, "972", "unet_name", str(payload.get("unet_name", "") or ""))
    _set_api_input(prompt, "971", "text", prompt_text)
    _set_api_input(prompt, "967", "width", width)
    _set_api_input(prompt, "967", "height", height)
    _set_api_input(prompt, "979", "image", image_name)
    _set_api_input(prompt, "983", "value", enhance_amount)
    _set_api_input(prompt, "964", "noise_seed", seed)

    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    _set_api_input(prompt, "974", "use_custom_loras", use_custom_loras)
    _set_api_input(prompt, "974", "lora_count", lora_count)
    _set_api_input(prompt, "974", "ltx_two_pass_mode", False)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        _set_api_input(prompt, "974", f"lora_{slot}", _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)))
        _set_api_input(prompt, "974", f"strength_{slot}", _float_payload(payload, f"strength_{slot}", 1.0))

    return prompt, seed


def _patch_i2v_workflow(workflow, payload):
    workflow = copy.deepcopy(workflow)
    i2v_prompt = str(payload.get("i2v_prompt", "") or "").strip()
    if not i2v_prompt:
        raise ValueError("I2V prompt is empty.")

    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    image_folder = os.path.abspath(str(payload.get("image_folder", "") or "").strip().strip('"'))
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Image folder was not found: {image_folder}")
    srt_path = os.path.abspath(str(payload.get("srt_path", "") or "").strip().strip('"'))
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file was not found: {srt_path}")

    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    output_folder = os.path.join(project_folder, "image_to_video_clips")
    os.makedirs(output_folder, exist_ok=True)

    image_index = _int_payload(payload, "image_index_zero_based", 0, 0, 999999)
    prompt_number = _int_payload(payload, "prompt_number_one_based", 1, 1, 999999)
    fps = _int_payload(payload, "fps", 24, 1, 120)
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)

    _set_widget(workflow, 271, 0, _clean_i2v_unet_name(payload.get("unet_name", "")))
    _set_widget(workflow, 271, 1, str(payload.get("vae_name", "") or ""))
    _set_widget(workflow, 271, 2, str(payload.get("clip_name1", "") or ""))
    _set_widget(workflow, 271, 3, str(payload.get("clip_name2", "") or ""))
    _set_widget(workflow, 271, 4, str(payload.get("upscale_model_name", "") or ""))
    _set_widget(workflow, 271, 5, str(payload.get("audio_vae_name", "") or ""))

    _set_widget(workflow, 736, 0, fps)
    _set_widget(workflow, 736, 1, width)
    _set_widget(workflow, 736, 2, height)
    _set_widget(workflow, 736, 3, seed)
    _set_widget(workflow, 736, 4, 0)

    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    _set_widget(workflow, 842, 0, use_custom_loras)
    _set_widget(workflow, 842, 1, lora_count)
    _set_widget(workflow, 842, 2, True)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        lora_name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA))
        strength = _float_payload(payload, f"strength_{slot}", 1.0)
        base_index = 3 + ((slot - 1) * 2)
        _set_widget(workflow, 842, base_index, lora_name)
        _set_widget(workflow, 842, base_index + 1, strength)

    _set_widget_key(workflow, 927, "audio_file", audio_path)
    _set_widget_key(workflow, 927, "seek_seconds", 0)
    _set_widget_key(workflow, 927, "duration", 0)
    _set_widget(workflow, 925, 0, image_folder)
    _set_widget(workflow, 929, 0, image_index)
    _set_widget(workflow, 929, 1, "fixed")
    _set_widget(workflow, 930, 0, prompt_number)
    _set_widget(workflow, 930, 1, "fixed")
    _set_widget(workflow, 933, 0, i2v_prompt)
    _set_widget(workflow, 933, 1, "string")
    _set_widget(workflow, 935, 0, srt_path)
    _set_widget(workflow, 437, 0, output_folder)
    return workflow, output_folder


def _set_api_input(prompt, node_id, input_name, value):
    node = prompt.get(str(node_id))
    if not isinstance(node, dict):
        raise KeyError(f"API prompt node {node_id} was not found.")
    inputs = node.setdefault("inputs", {})
    inputs[input_name] = value


def _patch_i2v_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    i2v_prompt = str(payload.get("i2v_prompt", "") or "").strip()
    if not i2v_prompt:
        raise ValueError("I2V prompt is empty.")

    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    image_folder = os.path.abspath(str(payload.get("image_folder", "") or "").strip().strip('"'))
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Image folder was not found: {image_folder}")
    srt_path = os.path.abspath(str(payload.get("srt_path", "") or "").strip().strip('"'))
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file was not found: {srt_path}")
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    output_folder = os.path.join(project_folder, "image_to_video_clips")
    os.makedirs(output_folder, exist_ok=True)

    image_index = _int_payload(payload, "image_index_zero_based", 0, 0, 999999)
    prompt_number = _int_payload(payload, "prompt_number_one_based", 1, 1, 999999)
    fps = _int_payload(payload, "fps", 24, 1, 120)
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)

    _set_api_input(prompt, "271:215", "unet_name", _clean_i2v_unet_name(payload.get("unet_name", "")))
    _set_api_input(prompt, "271:256", "vae_name", str(payload.get("vae_name", "") or ""))
    _set_api_input(prompt, "271:216", "clip_name1", str(payload.get("clip_name1", "") or ""))
    _set_api_input(prompt, "271:216", "clip_name2", str(payload.get("clip_name2", "") or ""))
    _set_api_input(prompt, "271:211", "model_name", str(payload.get("upscale_model_name", "") or ""))
    _set_api_input(prompt, "271:254", "vae_name", str(payload.get("audio_vae_name", "") or ""))

    _set_api_input(prompt, "736:424", "value", fps)
    _set_api_input(prompt, "736:425", "value", width)
    _set_api_input(prompt, "736:426", "value", height)
    _set_api_input(prompt, "736:449", "value", seed)
    _set_api_input(prompt, "736:551", "value", 0)

    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    _set_api_input(prompt, "937", "use_custom_loras", use_custom_loras)
    _set_api_input(prompt, "937", "lora_count", lora_count)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        legacy_strength = _float_payload(payload, f"strength_{slot}", 1.0)
        first_pass_strength = _float_payload(payload, f"first_pass_strength_{slot}", legacy_strength)
        second_pass_strength = _float_payload(payload, f"second_pass_strength_{slot}", legacy_strength)
        _set_api_input(prompt, "937", f"lora_{slot}", _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)))
        _set_api_input(prompt, "937", f"first_pass_strength_{slot}", first_pass_strength)
        _set_api_input(prompt, "937", f"second_pass_strength_{slot}", second_pass_strength)

    _set_api_input(prompt, "927", "audio_file", audio_path)
    _set_api_input(prompt, "927", "seek_seconds", 0)
    _set_api_input(prompt, "927", "duration", 0)
    _set_api_input(prompt, "925", "folder_path", image_folder)
    _set_api_input(prompt, "929", "value", image_index)
    _set_api_input(prompt, "930", "value", prompt_number)
    _set_api_input(prompt, "933", "text", i2v_prompt)
    _set_api_input(prompt, "933", "output_mode", "string")
    _set_api_input(prompt, "935", "value", srt_path)
    _set_api_input(prompt, "218:287", "overwrite_mode", "overwrite")
    _set_api_input(prompt, "437", "value", output_folder)
    return prompt, output_folder


def _patch_t2v_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    t2v_prompt = str(payload.get("t2v_prompt", payload.get("i2v_prompt", "")) or "").strip()
    if not t2v_prompt:
        raise ValueError("T2V prompt is empty.")

    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    srt_path = os.path.abspath(str(payload.get("srt_path", "") or "").strip().strip('"'))
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file was not found: {srt_path}")
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    output_folder = os.path.join(project_folder, "text_to_video_clips")
    os.makedirs(output_folder, exist_ok=True)

    prompt_number = _int_payload(payload, "prompt_number_one_based", 1, 1, 999999)
    fps = _int_payload(payload, "fps", 24, 1, 120)
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)

    _set_api_input(prompt, "271:215", "unet_name", _clean_i2v_unet_name(payload.get("unet_name", "")))
    _set_api_input(prompt, "271:256", "vae_name", str(payload.get("vae_name", "") or ""))
    _set_api_input(prompt, "271:216", "clip_name1", str(payload.get("clip_name1", "") or ""))
    _set_api_input(prompt, "271:216", "clip_name2", str(payload.get("clip_name2", "") or ""))
    _set_api_input(prompt, "271:211", "model_name", str(payload.get("upscale_model_name", "") or ""))
    _set_api_input(prompt, "271:254", "vae_name", str(payload.get("audio_vae_name", "") or ""))

    _set_api_input(prompt, "736:424", "value", fps)
    _set_api_input(prompt, "736:425", "value", width)
    _set_api_input(prompt, "736:426", "value", height)
    _set_api_input(prompt, "736:449", "value", seed)
    _set_api_input(prompt, "736:551", "value", 0)

    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    _set_api_input(prompt, "937", "use_custom_loras", use_custom_loras)
    _set_api_input(prompt, "937", "lora_count", lora_count)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        legacy_strength = _float_payload(payload, f"strength_{slot}", 1.0)
        first_pass_strength = _float_payload(payload, f"first_pass_strength_{slot}", legacy_strength)
        second_pass_strength = _float_payload(payload, f"second_pass_strength_{slot}", legacy_strength)
        _set_api_input(prompt, "937", f"lora_{slot}", _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)))
        _set_api_input(prompt, "937", f"first_pass_strength_{slot}", first_pass_strength)
        _set_api_input(prompt, "937", f"second_pass_strength_{slot}", second_pass_strength)

    _set_api_input(prompt, "927", "audio_file", audio_path)
    _set_api_input(prompt, "927", "seek_seconds", 0)
    _set_api_input(prompt, "927", "duration", 0)
    _set_api_input(prompt, "930", "value", prompt_number)
    _set_api_input(prompt, "933", "text", t2v_prompt)
    _set_api_input(prompt, "933", "output_mode", "string")
    _set_api_input(prompt, "935", "value", srt_path)
    _set_api_input(prompt, "218:287", "overwrite_mode", "overwrite")
    _set_api_input(prompt, "437", "value", output_folder)
    return prompt, output_folder


def _get_comfy_node_mappings():
    comfy_nodes = sys.modules.get("nodes")
    if comfy_nodes is None or not hasattr(comfy_nodes, "NODE_CLASS_MAPPINGS"):
        comfy_nodes = importlib.import_module("nodes")
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, dict):
        raise RuntimeError("ComfyUI node mappings are not available yet.")
    return mappings


def _input_names_for_node(class_type, mappings):
    node_class = mappings.get(class_type)
    if node_class is None:
        raise KeyError(f"Node class is not loaded in ComfyUI: {class_type}")
    input_types = node_class.INPUT_TYPES()
    names = []
    for section in ("required", "optional"):
        values = input_types.get(section, {})
        if isinstance(values, dict):
            names.extend(values.keys())
    return names


def _api_widget_values(class_type, widget_values):
    values = list(widget_values or [])
    if class_type == "SamplerCustom" and len(values) >= 4:
        # ComfyUI stores seed control mode ("fixed", "randomize", etc.) in the
        # workflow widgets, but it is not an API input. The next real input is cfg.
        if str(values[2]).lower() in {"fixed", "randomize", "increment", "decrement"}:
            values.pop(2)
    return values


def _workflow_to_api_prompt(workflow):
    workflow = _expand_subgraphs(workflow)
    mappings = _get_comfy_node_mappings()
    links = {}
    for raw_link in workflow.get("links", []):
        if not isinstance(raw_link, list) or len(raw_link) < 6:
            continue
        link_id, origin_id, origin_slot = raw_link[0], raw_link[1], raw_link[2]
        links[int(link_id)] = [str(origin_id), int(origin_slot)]

    set_values = {}
    get_nodes = {}
    for node in workflow.get("nodes", []):
        node_id = str(node.get("id"))
        class_type = node.get("type")
        widgets = node.get("widgets_values", [])
        if class_type == "SetNode" and isinstance(widgets, list) and widgets:
            input_link = None
            for input_info in node.get("inputs", []) or []:
                if input_info.get("link") is not None:
                    input_link = int(input_info.get("link"))
                    break
            if input_link is not None and input_link in links:
                set_values[str(widgets[0])] = links[input_link]
        elif class_type == "GetNode" and isinstance(widgets, list) and widgets:
            get_nodes[node_id] = str(widgets[0])

    prompt = {}
    for node in workflow.get("nodes", []):
        node_id = str(node.get("id"))
        class_type = node.get("type")
        if not node_id or not class_type:
            continue
        if class_type in {"SetNode", "GetNode", "MarkdownNote"}:
            continue

        linked_inputs = {}
        for input_info in node.get("inputs", []) or []:
            link_id = input_info.get("link")
            input_name = input_info.get("name")
            if link_id is not None and input_name and int(link_id) in links:
                source = links[int(link_id)]
                source_node_id = str(source[0])
                if source_node_id in get_nodes and get_nodes[source_node_id] in set_values:
                    source = set_values[get_nodes[source_node_id]]
                linked_inputs[input_name] = source

        inputs = dict(linked_inputs)
        raw_widget_values = node.get("widgets_values", [])
        keyed_widget_values = raw_widget_values if isinstance(raw_widget_values, dict) else None
        widget_values = [] if keyed_widget_values is not None else _api_widget_values(class_type, raw_widget_values)
        widget_index = 0
        for input_name in _input_names_for_node(class_type, mappings):
            if input_name in linked_inputs:
                continue
            if keyed_widget_values is not None:
                if input_name in keyed_widget_values and not isinstance(keyed_widget_values[input_name], dict):
                    inputs[input_name] = keyed_widget_values[input_name]
                continue
            if widget_index >= len(widget_values):
                break
            inputs[input_name] = widget_values[widget_index]
            widget_index += 1

        prompt[node_id] = {"class_type": class_type, "inputs": inputs}

    return prompt


def _expand_subgraphs(workflow, depth=0):
    definitions = {item.get("id"): item for item in workflow.get("definitions", {}).get("subgraphs", []) if isinstance(item, dict)}
    if not definitions or depth > 12:
        return workflow
    if not any(node.get("type") in definitions for node in workflow.get("nodes", [])):
        return workflow

    workflow = copy.deepcopy(workflow)
    outer_links = {}
    max_link_id = 0
    for raw_link in workflow.get("links", []):
        if isinstance(raw_link, list) and len(raw_link) >= 6:
            link_id = int(raw_link[0])
            max_link_id = max(max_link_id, link_id)
            outer_links[link_id] = [str(raw_link[1]), int(raw_link[2])]
        elif isinstance(raw_link, dict):
            link_id = int(raw_link.get("id", 0) or 0)
            max_link_id = max(max_link_id, link_id)
            outer_links[link_id] = [str(raw_link.get("origin_id")), int(raw_link.get("origin_slot", 0) or 0)]

    def new_link_id():
        nonlocal max_link_id
        max_link_id += 1
        return max_link_id

    def link_tuple(link_id, origin_id, origin_slot, target_id, target_slot, link_type):
        return [link_id, origin_id, origin_slot, target_id, target_slot, link_type]

    subgraph_node_ids = {str(node.get("id")) for node in workflow.get("nodes", []) if node.get("type") in definitions}
    expanded_nodes = []
    expanded_links = [
        link for link in workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 6 and str(link[1]) not in subgraph_node_ids and str(link[3]) not in subgraph_node_ids
    ]
    link_assignments = []
    subgraph_output_sources = {}

    for node in workflow.get("nodes", []):
        subgraph = definitions.get(node.get("type"))
        if not subgraph:
            expanded_nodes.append(node)
            continue

        node_id = str(node.get("id"))
        id_map = {str(inner.get("id")): f"{node_id}_{inner.get('id')}" for inner in subgraph.get("nodes", [])}
        external_inputs = node.get("inputs", []) or []
        external_widgets = list(node.get("widgets_values", []) or [])
        input_target_links = {}
        output_sources = {}

        for raw_link in subgraph.get("links", []) or []:
            if isinstance(raw_link, dict):
                link = {
                    "id": int(raw_link.get("id", 0) or 0),
                    "origin_id": raw_link.get("origin_id"),
                    "origin_slot": int(raw_link.get("origin_slot", 0) or 0),
                    "target_id": raw_link.get("target_id"),
                    "target_slot": int(raw_link.get("target_slot", 0) or 0),
                    "type": raw_link.get("type", "*"),
                }
            elif isinstance(raw_link, list) and len(raw_link) >= 6:
                link = {
                    "id": int(raw_link[0]),
                    "origin_id": raw_link[1],
                    "origin_slot": int(raw_link[2]),
                    "target_id": raw_link[3],
                    "target_slot": int(raw_link[4]),
                    "type": raw_link[5],
                }
            else:
                continue

            origin_id = str(link["origin_id"])
            target_id = str(link["target_id"])
            if origin_id == "-10":
                slot = int(link["origin_slot"])
                input_target_links.setdefault(slot, []).append(link)
                continue
            if target_id == "-20":
                output_sources[int(link["target_slot"])] = [id_map.get(origin_id, origin_id), int(link["origin_slot"])]
                continue

            if origin_id in id_map and target_id in id_map:
                new_id = new_link_id()
                expanded_links.append(link_tuple(new_id, id_map[origin_id], int(link["origin_slot"]), id_map[target_id], int(link["target_slot"]), link["type"]))
                link_assignments.append((id_map[target_id], int(link["target_slot"]), new_id))

        inner_nodes = []
        for inner in subgraph.get("nodes", []) or []:
            cloned = copy.deepcopy(inner)
            cloned["id"] = id_map[str(inner.get("id"))]
            for input_info in cloned.get("inputs", []) or []:
                if input_info.get("link") is not None:
                    input_info["link"] = None
            inner_nodes.append(cloned)

        inner_by_id = {str(inner.get("id")): inner for inner in inner_nodes}
        for slot, links_for_slot in input_target_links.items():
            outer_input = external_inputs[slot] if slot < len(external_inputs) else {}
            outer_link_id = outer_input.get("link")
            if outer_link_id is not None and int(outer_link_id) in outer_links:
                source = outer_links[int(outer_link_id)]
                for link in links_for_slot:
                    target = id_map.get(str(link["target_id"]))
                    if not target:
                        continue
                    new_id = new_link_id()
                    expanded_links.append(link_tuple(new_id, source[0], source[1], target, int(link["target_slot"]), link["type"]))
                    link_assignments.append((target, int(link["target_slot"]), new_id))
            else:
                value = external_widgets[slot] if slot < len(external_widgets) else None
                for link in links_for_slot:
                    target = id_map.get(str(link["target_id"]))
                    if not target or value is None:
                        continue
                    target_node = inner_by_id.get(str(target))
                    if not target_node:
                        continue
                    widgets = target_node.setdefault("widgets_values", [])
                    while len(widgets) <= int(link["target_slot"]):
                        widgets.append(None)
                    widgets[int(link["target_slot"])] = value

        subgraph_output_sources[node_id] = output_sources
        expanded_nodes.extend(inner_nodes)

    for raw_link in workflow.get("links", []) or []:
        if not isinstance(raw_link, list) or len(raw_link) < 6:
            continue
        link_id, origin_id, origin_slot, target_id, target_slot, link_type = raw_link[:6]
        output_sources = subgraph_output_sources.get(str(origin_id))
        if not output_sources:
            continue
        source = output_sources.get(int(origin_slot))
        if not source:
            continue
        new_id = new_link_id()
        expanded_links.append(link_tuple(new_id, source[0], source[1], target_id, target_slot, link_type))
        link_assignments.append((str(target_id), int(target_slot), new_id))

    workflow["nodes"] = expanded_nodes
    workflow["links"] = expanded_links
    nodes_by_id = {str(node.get("id")): node for node in workflow.get("nodes", [])}
    for target_id, target_slot, link_id in link_assignments:
        target_node = nodes_by_id.get(str(target_id))
        if not target_node:
            continue
        inputs = target_node.get("inputs", []) or []
        if 0 <= int(target_slot) < len(inputs):
            inputs[int(target_slot)]["link"] = link_id
    if any(node.get("type") in definitions for node in workflow.get("nodes", [])):
        return _expand_subgraphs(workflow, depth + 1)
    return workflow


def _build_zimage_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_zimage_api_template_path())
    patched_prompt, used_seed = _patch_zimage_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
        "used_seed": used_seed,
    }


def _build_ernie_image_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_ernie_image_api_template_path())
    patched_prompt, used_seed = _patch_ernie_image_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
        "used_seed": used_seed,
    }


def _build_i2v_api_prompt(payload):
    api_template = _i2v_api_template_path()
    if os.path.isfile(api_template) and not payload.get("workflow_path"):
        workflow_path, prompt = _load_api_template(api_template)
        patched_prompt, output_folder = _patch_i2v_api_prompt(prompt, payload)
        return {
            "workflow_path": workflow_path,
            "output_folder": output_folder,
            "prompt": patched_prompt,
        }
    workflow_path, workflow = _load_workflow_template(payload.get("workflow_path") or _i2v_workflow_template_path())
    patched, output_folder = _patch_i2v_workflow(workflow, payload)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": _workflow_to_api_prompt(patched),
    }


def _build_t2v_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_t2v_api_template_path())
    patched_prompt, output_folder = _patch_t2v_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": patched_prompt,
    }


def _build_flux_klein_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_flux_klein_api_template_path())
    patched_prompt = _patch_flux_klein_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
    }


def _build_nb_image_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_nb_image_api_template_path())
    patched_prompt = _patch_nb_image_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
    }


def _build_z_upscale_enhance_prompt(payload):
    api_template = _z_upscale_enhance_api_template_path()
    if os.path.isfile(api_template):
        workflow_path, prompt = _load_api_template(api_template)
        patched_prompt, used_seed = _patch_z_upscale_enhance_api_prompt(prompt, payload)
        return {
            "workflow_path": workflow_path,
            "prompt": patched_prompt,
            "used_seed": used_seed,
        }
    workflow_path, workflow = _load_workflow_template(_z_upscale_enhance_template_path())
    patched_workflow, used_seed = _patch_z_upscale_enhance_workflow(workflow, payload)
    expanded = _expand_subgraphs(patched_workflow)
    return {
        "workflow_path": workflow_path,
        "prompt": _workflow_to_api_prompt(expanded),
        "used_seed": used_seed,
    }


def _build_clear_memory_prompt():
    workflow_path, prompt = _load_api_template(_clear_memory_api_template_path())
    return {
        "workflow_path": workflow_path,
        "prompt": prompt,
    }


def _safe_subfolder_path(base_dir, subfolder):
    base_abs = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.join(base_abs, str(subfolder or "")))
    if os.path.commonpath([base_abs, candidate]) != base_abs:
        raise ValueError("Image subfolder escapes the allowed ComfyUI folder.")
    return candidate


def _resolve_comfy_image_path(image_info):
    filename = os.path.basename(str(image_info.get("filename", "") or ""))
    if not filename:
        raise ValueError("Image filename is empty.")
    image_type = str(image_info.get("type", "output") or "output").lower()
    if image_type == "temp":
        base_dir = folder_paths.get_temp_directory()
    elif image_type == "input":
        base_dir = folder_paths.get_input_directory()
    else:
        base_dir = folder_paths.get_output_directory()
    folder = _safe_subfolder_path(base_dir, image_info.get("subfolder", ""))
    image_path = os.path.abspath(os.path.join(folder, filename))
    if os.path.commonpath([os.path.abspath(base_dir), image_path]) != os.path.abspath(base_dir):
        raise ValueError("Image path escapes the allowed ComfyUI folder.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Generated image was not found: {image_path}")
    return image_path


def _resolve_save_folder(raw_folder):
    text = str(raw_folder or "").strip().strip('"')
    if not text:
        text = "VRGDG_WorkflowRunner_Saved"
    if os.path.isabs(text):
        target = os.path.abspath(text)
    else:
        target = os.path.abspath(os.path.join(folder_paths.get_output_directory(), text))
    os.makedirs(target, exist_ok=True)
    return target


def _unique_copy_path(target_dir, source_path):
    stem, ext = os.path.splitext(os.path.basename(source_path))
    if not ext:
        ext = ".png"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(target_dir, f"{stem}_approved_{timestamp}{ext}")
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(target_dir, f"{stem}_approved_{timestamp}_{counter}{ext}")
        counter += 1
    return candidate


def _save_generated_image(payload):
    image_info = payload.get("image")
    if not isinstance(image_info, dict):
        raise ValueError("Image info is missing.")
    source_path = _resolve_comfy_image_path(image_info)
    target_dir = _resolve_save_folder(payload.get("save_folder"))
    target_path = _unique_copy_path(target_dir, source_path)
    shutil.copy2(source_path, target_path)
    return {"saved_path": target_path, "save_folder": target_dir}


def _find_ffmpeg_path():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except Exception:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError(f"FFmpeg was not found: {exc}") from exc


def _safe_project_subfolder(project_folder, folder_name):
    project = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not project:
        raise ValueError("Project folder is empty.")
    target = os.path.abspath(os.path.join(project, folder_name))
    if os.path.commonpath([project, target]) != project:
        raise ValueError("Target folder escapes the project folder.")
    os.makedirs(target, exist_ok=True)
    return project, target


def _unique_final_video_path(project_folder, prefix="FINAL_VIDEO"):
    safe_prefix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(prefix or "FINAL_VIDEO")).strip("_") or "FINAL_VIDEO"
    candidate = os.path.join(project_folder, f"{safe_prefix}.mp4")
    if not os.path.exists(candidate):
        return candidate
    index = 2
    while True:
        candidate = os.path.join(project_folder, f"{safe_prefix}{index}.mp4")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _concat_file_path(path):
    return os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")


def _cleanup_video_scratch_folders(project_folder, keep_folders=None):
    project_folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    keep = {os.path.abspath(path) for path in (keep_folders or []) if path}
    scratch_prefixes = ("image_to_video_clips_", "text_to_video_clips_")
    permanent_folders = {"image_to_video_clips", "text_to_video_clips", "rendered_scene_videos", "rendered_scene_videos_backup"}
    removed_folders = []
    if not os.path.isdir(project_folder):
        return removed_folders
    for name in os.listdir(project_folder):
        path = os.path.abspath(os.path.join(project_folder, name))
        if path in keep or not os.path.isdir(path):
            continue
        if name in permanent_folders or not name.startswith(scratch_prefixes):
            continue
        try:
            if os.path.commonpath([project_folder, path]) != project_folder:
                continue
            shutil.rmtree(path)
            removed_folders.append(path)
        except Exception as exc:
            print(f"[VRGDG WorkflowRunner] Could not delete video scratch folder '{path}': {exc}")
    return removed_folders


def _cleanup_i2v_scratch_folders(project_folder, keep_folders=None):
    return _cleanup_video_scratch_folders(project_folder, keep_folders=keep_folders)


def _retry_file_op(operation, description, attempts=30, delay=0.25):
    last_exc = None
    for attempt in range(max(1, attempts)):
        try:
            return operation()
        except PermissionError as exc:
            last_exc = exc
        except OSError as exc:
            if getattr(exc, "winerror", None) != 32:
                raise
            last_exc = exc
        if attempt < attempts - 1:
            time.sleep(delay)
    raise RuntimeError(f"{description} failed because the file stayed locked: {last_exc}") from last_exc


def _wait_for_stable_readable_file(path, timeout=20.0, interval=0.25):
    deadline = time.time() + max(0.5, float(timeout or 0))
    last_size = -1
    stable_reads = 0
    last_exc = None
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as handle:
                handle.read(1)
            if size > 0 and size == last_size:
                stable_reads += 1
                if stable_reads >= 2:
                    return
            else:
                stable_reads = 0
                last_size = size
        except (OSError, PermissionError) as exc:
            last_exc = exc
            stable_reads = 0
        time.sleep(interval)
    if last_exc:
        raise RuntimeError(f"Scene video is still locked and cannot be read: {path}") from last_exc


def _replace_file_with_retry(source_path, target_path):
    _wait_for_stable_readable_file(source_path)
    temp_target = f"{target_path}.copying"
    index = 2
    while os.path.exists(temp_target):
        temp_target = f"{target_path}.copying_{index:02d}"
        index += 1

    try:
        _retry_file_op(
            lambda: shutil.copy2(source_path, temp_target),
            f"Copying scene video to temporary file '{temp_target}'",
        )
        _retry_file_op(
            lambda: os.replace(temp_target, target_path),
            f"Replacing scene video '{target_path}'",
        )
    finally:
        if os.path.exists(temp_target):
            try:
                os.remove(temp_target)
            except Exception:
                pass

    try:
        _retry_file_op(
            lambda: os.remove(source_path),
            f"Removing scratch scene video '{source_path}'",
            attempts=8,
            delay=0.25,
        )
    except Exception as exc:
        print(f"[VRGDG WorkflowRunner] Copied scene video but could not remove scratch source '{source_path}': {exc}")


def _collect_scene_video(payload):
    source_path = os.path.abspath(str(payload.get("source_path", "") or "").strip().strip('"'))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Scene video was not found: {source_path}")
    project_folder, target_dir = _safe_project_subfolder(payload.get("project_folder", ""), "rendered_scene_videos")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    existing_action = str(payload.get("existing_action", "overwrite") or "overwrite").strip().lower()
    if existing_action not in {"overwrite", "backup"}:
        existing_action = "overwrite"

    source_dir = os.path.abspath(os.path.dirname(source_path))
    if not source_path.lower().endswith("-audio.mp4"):
        candidates = [
            os.path.join(source_dir, name)
            for name in os.listdir(source_dir)
            if name.lower().endswith("-audio.mp4") and os.path.isfile(os.path.join(source_dir, name))
        ]
        candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        if candidates:
            source_path = os.path.abspath(candidates[0])
            source_dir = os.path.abspath(os.path.dirname(source_path))

    target_path = os.path.join(target_dir, f"video_{scene_number:04d}-audio.mp4")
    backup_path = ""
    if os.path.abspath(source_path) != os.path.abspath(target_path):
        if os.path.exists(target_path):
            if existing_action == "backup":
                backup_dir = os.path.join(project_folder, "rendered_scene_videos_backup", f"scene_{scene_number:04d}")
                os.makedirs(backup_dir, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"video_{scene_number:04d}-audio_{stamp}.mp4")
                index = 2
                while os.path.exists(backup_path):
                    backup_path = os.path.join(backup_dir, f"video_{scene_number:04d}-audio_{stamp}_{index:02d}.mp4")
                    index += 1
                _retry_file_op(
                    lambda: shutil.move(target_path, backup_path),
                    f"Backing up existing scene video '{target_path}'",
                )
            else:
                _retry_file_op(
                    lambda: os.remove(target_path),
                    f"Removing existing scene video '{target_path}'",
                )
        _replace_file_with_retry(source_path, target_path)

    removed_files = []
    removed_folder = ""
    removed_scratch_folders = []

    return {
        "video_path": target_path,
        "video_folder": target_dir,
        "backup_path": backup_path,
        "existing_action": existing_action,
        "source_path": source_path,
        "removed_files": removed_files,
        "removed_folder": removed_folder,
        "removed_scratch_folders": removed_scratch_folders,
    }


def _stitch_scene_videos(payload):
    raw_paths = payload.get("scene_paths", [])
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("No scene video paths were provided.")
    project_folder, target_dir = _safe_project_subfolder(payload.get("project_folder", ""), "rendered_scene_videos")
    raw_scene_audio_paths = payload.get("scene_audio_paths", [])
    if not isinstance(raw_scene_audio_paths, list):
        raw_scene_audio_paths = []
    raw_scene_audio_items = payload.get("scene_audio_items", [])
    if not isinstance(raw_scene_audio_items, list):
        raw_scene_audio_items = []
    raw_overlay_items = payload.get("overlay_items", [])
    if not isinstance(raw_overlay_items, list):
        raw_overlay_items = []
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    preview_audio_start = max(0.0, float(payload.get("audio_start", 0) or 0))
    preview_audio_duration = max(0.0, float(payload.get("audio_duration", 0) or 0))

    scene_paths = []
    for index, raw_path in enumerate(raw_paths, start=1):
        path = os.path.abspath(str(raw_path or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Scene {index} video was not found: {path}")
        scene_paths.append(path)

    scene_audio_paths = []
    scene_audio_items = []
    if raw_scene_audio_items and any(str((item or {}).get("path", "") if isinstance(item, dict) else "").strip() for item in raw_scene_audio_items):
        if len(raw_scene_audio_items) != len(scene_paths):
            raise ValueError("Scene audio item count does not match scene video count.")
        for index, item in enumerate(raw_scene_audio_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Scene {index} audio item is invalid.")
            path = os.path.abspath(str(item.get("path", "") or "").strip().strip('"'))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Scene {index} audio was not found: {path}")
            start = max(0.0, float(item.get("start", 0) or 0))
            duration = max(0.05, float(item.get("duration", 0) or 0))
            scene_audio_items.append({"path": path, "start": start, "duration": duration})
            scene_audio_paths.append(path)
    elif raw_scene_audio_paths and any(str(item or "").strip() for item in raw_scene_audio_paths):
        if len(raw_scene_audio_paths) != len(scene_paths):
            raise ValueError("Scene audio path count does not match scene video count.")
        for index, raw_path in enumerate(raw_scene_audio_paths, start=1):
            path = os.path.abspath(str(raw_path or "").strip().strip('"'))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Scene {index} audio was not found: {path}")
            scene_audio_paths.append(path)
            scene_audio_items.append({"path": path, "start": 0.0, "duration": 0.0})
    elif not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")

    ffmpeg_path = _find_ffmpeg_path()
    concat_file = os.path.join(target_dir, "concat_list.txt")
    with open(concat_file, "w", encoding="utf-8") as handle:
        for path in scene_paths:
            handle.write(f"file '{_concat_file_path(path)}'\n")

    temp_video = os.path.join(target_dir, "_temp_video_no_audio.mp4")
    temp_audio = os.path.join(target_dir, "_temp_scene_audio.m4a")
    temp_global_audio = os.path.join(target_dir, "_temp_global_audio.m4a")
    temp_audio_parts = []
    audio_concat_file = os.path.join(target_dir, "audio_concat_list.txt")
    final_output = _unique_final_video_path(project_folder, payload.get("output_prefix", "FINAL_VIDEO"))

    concat_cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-an",
        "-c:v",
        "copy",
        temp_video,
    ]
    subprocess.run(concat_cmd, capture_output=True, text=True, check=True)

    insert_items = []
    for index, item in enumerate(raw_overlay_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Insert {index} item is invalid.")
        path = os.path.abspath(str(item.get("path", "") or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Insert {index} video was not found: {path}")
        start = max(0.0, float(item.get("start", 0) or 0))
        end = max(start + 0.05, float(item.get("end", start + 4) or start + 4))
        source_start = max(0.0, float(item.get("source_start", 0) or 0))
        insert_items.append({"path": path, "start": start, "end": end, "duration": end - start, "source_start": source_start})

    if insert_items:
        insert_items.sort(key=lambda item: (item["start"], item["end"]))
        flattened_video = os.path.join(target_dir, "_temp_video_with_inserts.mp4")
        flatten_list = os.path.join(target_dir, "flatten_concat_list.txt")
        flatten_parts = []
        cursor = 0.0
        part_index = 1

        def add_flatten_part(source_path, start=None, duration=None):
            nonlocal part_index
            part_path = os.path.join(target_dir, f"_temp_flatten_part_{part_index:04d}.mp4")
            part_index += 1
            cmd = [ffmpeg_path, "-y"]
            if start is not None:
                cmd.extend(["-ss", f"{max(0.0, float(start)):.6f}"])
            cmd.extend(["-i", source_path])
            if duration is not None:
                cmd.extend(["-t", f"{max(0.05, float(duration)):.6f}"])
            cmd.extend([
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                part_path,
            ])
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            flatten_parts.append(part_path)

        for item in insert_items:
            if item["start"] > cursor + 0.01:
                add_flatten_part(temp_video, cursor, item["start"] - cursor)
            add_flatten_part(item["path"], item.get("source_start", 0.0), item["duration"])
            cursor = max(cursor, item["end"])

        add_flatten_part(temp_video, cursor, None)
        with open(flatten_list, "w", encoding="utf-8") as handle:
            for path in flatten_parts:
                handle.write(f"file '{_concat_file_path(path)}'\n")
        flatten_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            flatten_list,
            "-an",
            "-c:v",
            "copy",
            flattened_video,
        ]
        subprocess.run(flatten_cmd, capture_output=True, text=True, check=True)
        try:
            os.remove(temp_video)
        except Exception:
            pass
        try:
            os.remove(flatten_list)
        except Exception:
            pass
        for part_path in flatten_parts:
            try:
                os.remove(part_path)
            except Exception:
                pass
        temp_video = flattened_video

    mux_audio_path = audio_path
    if scene_audio_paths:
        with open(audio_concat_file, "w", encoding="utf-8") as handle:
            for index, item in enumerate(scene_audio_items, start=1):
                path = item["path"]
                duration = float(item.get("duration", 0) or 0)
                if item.get("start", 0) or duration:
                    part_path = os.path.join(target_dir, f"_temp_scene_audio_{index:04d}.m4a")
                    trim_cmd = [
                        ffmpeg_path,
                        "-y",
                        "-ss",
                        str(float(item.get("start", 0) or 0)),
                        "-i",
                        path,
                    ]
                    if duration:
                        trim_cmd.extend(["-t", str(duration)])
                    trim_cmd.extend(["-c:a", "aac", part_path])
                    subprocess.run(trim_cmd, capture_output=True, text=True, check=True)
                    temp_audio_parts.append(part_path)
                    path = part_path
                handle.write(f"file '{_concat_file_path(path)}'\n")
        audio_concat_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            audio_concat_file,
            "-c:a",
            "aac",
            temp_audio,
        ]
        subprocess.run(audio_concat_cmd, capture_output=True, text=True, check=True)
        mux_audio_path = temp_audio
    elif preview_audio_start or preview_audio_duration:
        trim_audio_cmd = [ffmpeg_path, "-y"]
        if preview_audio_start:
            trim_audio_cmd.extend(["-ss", f"{preview_audio_start:.6f}"])
        trim_audio_cmd.extend(["-i", audio_path])
        if preview_audio_duration:
            trim_audio_cmd.extend(["-t", f"{preview_audio_duration:.6f}"])
        trim_audio_cmd.extend(["-c:a", "aac", temp_global_audio])
        subprocess.run(trim_audio_cmd, capture_output=True, text=True, check=True)
        mux_audio_path = temp_global_audio

    mux_cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        temp_video,
        "-i",
        mux_audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        final_output,
    ]
    try:
        subprocess.run(mux_cmd, capture_output=True, text=True, check=True)
    finally:
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except Exception:
            pass
        try:
            if os.path.exists(concat_file):
                os.remove(concat_file)
        except Exception:
            pass
        try:
            if os.path.exists(audio_concat_file):
                os.remove(audio_concat_file)
        except Exception:
            pass
        try:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        except Exception:
            pass
        try:
            if os.path.exists(temp_global_audio):
                os.remove(temp_global_audio)
        except Exception:
            pass
        for part_path in temp_audio_parts:
            try:
                if os.path.exists(part_path):
                    os.remove(part_path)
            except Exception:
                pass
    removed_scratch_folders = _cleanup_video_scratch_folders(project_folder, keep_folders=[target_dir])

    return {
        "final_video_path": final_output,
        "video_folder": target_dir,
        "concat_file": "",
        "scene_count": len(scene_paths),
        "insert_count": len(insert_items),
        "used_scene_audio": bool(scene_audio_paths),
        "removed_scratch_folders": removed_scratch_folders,
    }


def _ensure_workflow_runner_routes():
    global _VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED
    if _VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    try:
        _ensure_placeholder_load_image()
    except Exception as exc:
        print(f"[VRGDG] Could not prepare placeholder image for LoadImage validation: {exc}")

    @server_instance.routes.get("/vrgdg/workflow_runner/lora_list")
    async def vrgdg_workflow_runner_lora_list(request):
        return web.json_response({"ok": True, "loras": _lora_choices()})

    @server_instance.routes.get("/vrgdg/workflow_runner/i2v_choices")
    async def vrgdg_workflow_runner_i2v_choices(request):
        return web.json_response({
            "ok": True,
            "unets": _folder_choices(("unet", "diffusion_models")),
            "vae": _folder_choices("vae"),
            "clip": _folder_choices(("clip", "text_encoders")),
            "upscale_models": _folder_choices("upscale_models"),
        })

    @server_instance.routes.post("/vrgdg/workflow_runner/build_zimage_prompt")
    async def vrgdg_workflow_runner_build_zimage_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_zimage_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_ernie_image_prompt")
    async def vrgdg_workflow_runner_build_ernie_image_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_ernie_image_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_i2v_prompt")
    async def vrgdg_workflow_runner_build_i2v_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_i2v_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_t2v_prompt")
    async def vrgdg_workflow_runner_build_t2v_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_t2v_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_flux_klein_prompt")
    async def vrgdg_workflow_runner_build_flux_klein_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_flux_klein_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_nb_image_prompt")
    async def vrgdg_workflow_runner_build_nb_image_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_nb_image_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_z_upscale_enhance_prompt")
    async def vrgdg_workflow_runner_build_z_upscale_enhance_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_z_upscale_enhance_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_clear_memory_prompt")
    async def vrgdg_workflow_runner_build_clear_memory_prompt(request):
        try:
            result = _build_clear_memory_prompt()
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/save_image")
    async def vrgdg_workflow_runner_save_image(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _save_generated_image(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/collect_scene_video")
    async def vrgdg_workflow_runner_collect_scene_video(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _collect_scene_video(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/stitch_scene_videos")
    async def vrgdg_workflow_runner_stitch_scene_videos(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _stitch_scene_videos(payload)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr or exc.stdout or str(exc)
            return web.json_response({"ok": False, "error": f"FFmpeg failed:\n{error}"}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    _VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED = True


class VRGDG_ZImageWorkflowRunnerUI:
    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = _lora_choices()
        required = {
            "workflow_path": ("STRING", {"default": _zimage_api_template_path()}),
            "save_folder": ("STRING", {"default": "VRGDG_WorkflowRunner_Saved"}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "first_pass_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
            "first_pass_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
            "second_pass_width": ("INT", {"default": 1920, "min": 64, "max": 4096, "step": 8}),
            "second_pass_height": ("INT", {"default": 1080, "min": 64, "max": 4096, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            "use_custom_loras": ("BOOLEAN", {"default": False}),
            "lora_count": ("INT", {"default": 0, "min": 0, "max": _MAX_LORA_SLOTS, "step": 1}),
            "ltx_two_pass_mode": ("BOOLEAN", {"default": False}),
        }
        for slot in range(1, _MAX_LORA_SLOTS + 1):
            required[f"lora_{slot}"] = (lora_choices, {"default": _NONE_LORA})
            required[f"first_pass_strength_{slot}"] = (
                "FLOAT",
                {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01},
            )
            required[f"second_pass_strength_{slot}"] = (
                "FLOAT",
                {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
            )
        return {"required": required}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Canvas UI for running the bundled Z-Image text-to-image workflow template without opening it."

    def noop(self, **kwargs):
        return ("Open the Z-Image workflow runner UI and press Run Image Workflow.",)


class VRGDG_ClearMemoryButtonUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Small canvas button that queues the bundled ClearMemory_API workflow."

    def noop(self):
        return ("Press Clear Memory to run the bundled ClearMemory_API workflow.",)


_ensure_workflow_runner_routes()


NODE_CLASS_MAPPINGS = {
    "VRGDG_ZImageWorkflowRunnerUI": VRGDG_ZImageWorkflowRunnerUI,
    "VRGDG_ClearMemoryButtonUI": VRGDG_ClearMemoryButtonUI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ZImageWorkflowRunnerUI": "VRGDG Z-Image Workflow Runner UI",
    "VRGDG_ClearMemoryButtonUI": "VRGDG Clear Memory Button",
}
