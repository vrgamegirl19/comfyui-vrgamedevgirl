"""Image workflow graphs: Z-Image, Krea2, Ernie, Flux Klein, NanoBanana and Z upscale enhance."""

import copy
import json
import os
import random
import folder_paths

from .paths import _bool_payload, _float_payload, _int_payload, _resolve_existing_file
from .models import _MAX_LORA_SLOTS, _NONE_LORA, _clean_lora_name, _require_model_choice
from .api_graph import _api_node_id_by_class, _ensure_placeholder_load_image, _expand_subgraphs, _load_api_template, _load_workflow_template, _node_by_id, _prepare_load_image_name, _set_api_input, _set_widget, _workflow_node_id_by_class, _workflow_to_api_prompt


def _zimage_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "text2image_zimage_API.json",
    )


def _krea2_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Krea2_TextToImage_API.json",
    )


def _krea2_2pass_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Krea2_API_2Pass.json",
    )


def _flux_klein_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "fluxKleinMultiImage_API.json",
    )


def _ernie_image_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "image_ernie_image_turbo_API.json",
    )


def _nb_image_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "NB_API.json",
    )


def _z_upscale_enhance_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "z_upscaleEnhance.json",
    )


def _z_upscale_enhance_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "z_upscaleEnhance_API.json",
    )


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


def _patch_krea2_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    if not prompt_text:
        raise ValueError("Prompt text is empty.")

    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    first_width = _int_payload(payload, "first_pass_width", 1024, 64, 4096)
    first_height = _int_payload(payload, "first_pass_height", 576, 64, 4096)
    seed_mode = str(payload.get("seed_mode", "fixed") or "fixed").strip().lower()
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    if seed_mode in {"random", "randomize"}:
        seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)

    use_zimage_enhance = _bool_payload(payload, "use_zimage_enhance", True)
    enhance_strength = max(0.1, min(1.0, _float_payload(payload, "zimage_enhance_strength", 0.5)))

    krea_unet = str(payload.get("krea_unet_name") or payload.get("unet_name") or "krea2_turbo_fp8_scaled.safetensors").strip()
    krea_clip = str(payload.get("krea_clip_name") or payload.get("clip_name") or "qwen3vl_4b_fp8_scaled.safetensors").strip()
    krea_vae = str(payload.get("krea_vae_name") or payload.get("vae_name") or "qwen_image_vae.safetensors").strip()
    z_unet = str(payload.get("z_unet_name") or payload.get("enhance_unet_name") or "z_image_turbo_bf16.safetensors").strip()
    z_clip = str(payload.get("z_clip_name") or payload.get("enhance_clip_name") or "qwen_3_4b.safetensors").strip()
    z_vae = str(payload.get("z_vae_name") or payload.get("enhance_vae_name") or "ae.safetensors").strip()

    _require_model_choice(("diffusion_models", "unet"), krea_unet, "Krea2 diffusion model")
    _require_model_choice(("text_encoders", "clip"), krea_clip, "Krea2 text encoder")
    _require_model_choice("vae", krea_vae, "Krea2 VAE")
    if use_zimage_enhance:
        _require_model_choice(("unet", "diffusion_models"), z_unet, "ZImage enhancer diffusion model")
        _require_model_choice(("clip", "text_encoders"), z_clip, "ZImage enhancer text encoder")
        _require_model_choice("vae", z_vae, "ZImage enhancer VAE")

    _set_api_input(prompt, "200", "text", prompt_text)
    _set_api_input(prompt, "30:10", "unet_name", krea_unet)
    _set_api_input(prompt, "30:11", "clip_name", krea_clip)
    _set_api_input(prompt, "30:12", "vae_name", krea_vae)
    _set_api_input(prompt, "30:3", "seed", seed)
    _set_api_input(prompt, "30:5", "batch_size", _int_payload(payload, "batch_size", 1, 1, 16))
    _set_api_input(prompt, "201", "width", first_width)
    _set_api_input(prompt, "201", "height", first_height)

    _set_api_input(prompt, "193:16", "unet_name", z_unet)
    _set_api_input(prompt, "193:18", "clip_name", z_clip)
    _set_api_input(prompt, "193:17", "vae_name", z_vae)
    _set_api_input(prompt, "193:86", "noise_seed", seed)
    _set_api_input(prompt, "193:98", "width", width)
    _set_api_input(prompt, "193:98", "height", height)

    # The ZImage branch uses a 10-step partial-denoise schedule. A larger
    # strength begins earlier and therefore allows ZImage to change/add more.
    enhance_steps = 10
    enhance_start = max(0, min(enhance_steps - 1, round(enhance_steps * (1.0 - enhance_strength))))
    _set_api_input(prompt, "193:82", "steps", enhance_steps)
    _set_api_input(prompt, "193:82", "start_at_step", enhance_start)
    _set_api_input(prompt, "193:82", "end_at_step", enhance_steps)

    if not use_zimage_enhance:
        # PreviewImage is the workflow output. Pointing it at the Krea decode
        # removes the unreferenced ZImage branch from ComfyUI execution.
        _set_api_input(prompt, "199", "images", ["30:8", 0])

    aspect_node = prompt.get("49")
    if isinstance(aspect_node, dict):
        inputs = aspect_node.setdefault("inputs", {})
        ratio = width / max(1, height)
        if abs(ratio - (16 / 9)) < 0.04:
            inputs["aspect_ratio"] = "16:9 (Widescreen)"
        elif abs(ratio - 1) < 0.04:
            inputs["aspect_ratio"] = "1:1 (Square)"
        elif ratio < 1:
            inputs["aspect_ratio"] = "9:16 (Portrait)"
        inputs["megapixels"] = max(0.25, round((first_width * first_height) / 1000000, 2))
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


def _patch_krea2_2pass_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    if not prompt_text:
        raise ValueError("Krea 2 prompt text is empty.")

    aspect_ratio = str(payload.get("aspect_ratio") or "16:9 (Widescreen)").strip()
    batch_size = _int_payload(payload, "batch_size", 1, 1, 16)
    seed_mode = str(payload.get("seed_mode", "fixed") or "fixed").strip().lower()
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    if seed_mode in {"random", "randomize"}:
        seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    cfg = max(1.0, min(1.2, _float_payload(payload, "cfg", 1.2)))
    sampler_name = str(payload.get("sampler_name") or "euler_ancestral_cfg_pp").strip()
    use_i2i = _bool_payload(payload, "use_image_to_image", False)
    creativity = _int_payload(payload, "image_to_image_creativity", 5, 0, 10)

    unet_name = str(payload.get("unet_name") or "krea2_turbo_fp8_scaled.safetensors").strip()
    clip_name = str(payload.get("clip_name") or "qwen3vl_4b_fp8_scaled.safetensors").strip()
    vae_name = str(payload.get("vae_name") or "qwen_image_vae.safetensors").strip()
    use_loras = _bool_payload(payload, "use_custom_loras", _bool_payload(payload, "use_loras", False))
    lora_count = _int_payload(payload, "lora_count", 0, 0, 20) if use_loras else 0

    _require_model_choice(("diffusion_models", "unet"), unet_name, "Krea 2 diffusion model")
    _require_model_choice(("text_encoders", "clip"), clip_name, "Krea 2 text encoder")
    _require_model_choice("vae", vae_name, "Krea 2 VAE")
    for slot in range(1, lora_count + 1):
        lora_name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA))
        if lora_name != _NONE_LORA:
            _require_model_choice("loras", lora_name, f"Krea 2 LoRA {slot}")

    _set_api_input(prompt, "228", "text", prompt_text)
    _set_api_input(prompt, "236", "unet_name", unet_name)
    _set_api_input(prompt, "233", "clip_name", clip_name)
    _set_api_input(prompt, "234", "vae_name", vae_name)
    _set_api_input(prompt, "248", "use_custom_loras", bool(use_loras and lora_count > 0))
    _set_api_input(prompt, "248", "lora_count", lora_count if use_loras else 0)
    for slot in range(1, 21):
        lora_name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA))
        legacy_strength = _float_payload(payload, f"strength_{slot}", 1.0)
        first_pass_strength = _float_payload(payload, f"first_pass_strength_{slot}", legacy_strength)
        second_pass_strength = _float_payload(payload, f"second_pass_strength_{slot}", legacy_strength)
        if not use_loras or slot > lora_count:
            lora_name = _NONE_LORA
        _set_api_input(prompt, "248", f"lora_{slot}", lora_name)
        _set_api_input(prompt, "248", f"first_pass_strength_{slot}", first_pass_strength)
        _set_api_input(prompt, "248", f"second_pass_strength_{slot}", second_pass_strength)
    _set_api_input(prompt, "238", "aspect_ratio", aspect_ratio)
    _set_api_input(prompt, "49", "aspect_ratio", aspect_ratio)
    _set_api_input(prompt, "240", "batch_size", batch_size)
    _set_api_input(prompt, "245", "value", creativity)
    _set_api_input(prompt, "242", "switch", use_i2i)
    _set_api_input(prompt, "243", "switch", use_i2i)
    _set_api_input(prompt, "235", "sampler_name", sampler_name)
    for node_id in ("230", "231"):
        _set_api_input(prompt, node_id, "noise_seed", seed)
        _set_api_input(prompt, node_id, "cfg", cfg)

    if use_i2i:
        image_name = _prepare_load_image_name(
            payload.get("image_to_image_path", ""),
            payload.get("image_to_image_data", ""),
            payload.get("image_to_image_name", "image.png"),
        )
        if not image_name:
            raise ValueError("Krea 2 image-to-image is enabled, but no source image was provided.")
        _set_api_input(prompt, "249", "image", image_name)
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
    if image_paths:
        _set_api_input(prompt, "1072", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    else:
        if "1053" in prompt:
            _set_api_input(prompt, "1053", "positive", ["1067", 0])
            _set_api_input(prompt, "1053", "negative", ["1058", 0])
        prompt.pop("1072", None)
        prompt.pop("1059", None)
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


def _looks_like_prompt_text(value):
    text = str(value or "").strip()
    return len(text) > 20 and any(ch.isspace() for ch in text)


def _looks_like_api_key(value):
    text = str(value or "").strip()
    return len(text) >= 20 and not any(ch.isspace() for ch in text)


def _patch_nb_image_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    prompt_text = str(payload.get("prompt", "") or "").strip()
    api_key = str(payload.get("api_key", "") or "").strip()
    if _looks_like_prompt_text(api_key) and _looks_like_api_key(prompt_text):
        api_key, prompt_text = prompt_text, api_key
    if not prompt_text:
        raise ValueError("NanoBanana prompt text is empty.")
    if not api_key:
        raise ValueError("NanoBanana needs an API key.")
    if any(ch.isspace() for ch in api_key):
        raise ValueError("NanoBanana API key looks invalid. It appears to contain prompt text; paste the Google API key into the NanoBanana API key field.")

    image_paths = _image_paths_from_payload_ingredients(payload, "NanoBanana reference image")

    nb_node_id = _api_node_id_by_class(prompt, "VRGDG_NanoBananaPro", fallback=1)
    image_loader_id = _api_node_id_by_class(prompt, "VRGDG_ImageBatchMultiFromPaths", fallback=3)
    _set_api_input(prompt, nb_node_id, "api_key", api_key)
    _set_api_input(prompt, nb_node_id, "prompt", prompt_text)
    _set_api_input(prompt, nb_node_id, "model", str(payload.get("model", "") or "gemini-3-pro-image-preview"))
    if image_paths:
        _set_api_input(prompt, image_loader_id, "image_paths", json.dumps(image_paths, ensure_ascii=False))
    else:
        prompt.get(str(nb_node_id), {}).get("inputs", {}).pop("image1", None)
        prompt.pop(str(image_loader_id), None)
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


def _build_zimage_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_zimage_api_template_path())
    patched_prompt, used_seed = _patch_zimage_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
        "used_seed": used_seed,
    }


def _build_krea2_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_krea2_api_template_path())
    patched_prompt, used_seed = _patch_krea2_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
        "used_seed": used_seed,
    }


def _build_krea2_2pass_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_krea2_2pass_api_template_path())
    patched_prompt, used_seed = _patch_krea2_2pass_api_prompt(prompt, payload)
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
