"""LTX video workflow graphs: I2V, T2V, RTV, ingredients, ID-LoRA, FLF and LTX 2.5."""

import copy
import json
import math
import os
import random
import folder_paths

from .paths import _bool_payload, _float_payload, _int_payload, _scene_render_output_folder
from .models import _MAX_LORA_SLOTS, _NONE_LORA, _REQUIRED_LTX_MSR_LORA, _clean_lora_name, _clean_msr_lora_name, _lora_choices
from .api_graph import _clean_i2v_unet_name, _ensure_placeholder_load_image, _load_api_template, _load_workflow_template, _ltx25_diffusion_loader_node, _patch_ltx_video_model_loader, _prepare_load_image_name, _prepare_optional_input_image_name, _replace_api_input_refs, _set_api_input, _set_optional_api_input, _set_widget, _set_widget_key, _workflow_to_api_prompt


_REQUIRED_LTX25_MSR_LORA = "LTX-2.5-Licon-MSR-V1.safetensors"


_REQUIRED_LTX_INGREDIENTS_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"


_REQUIRED_LTX_ID_LORA = "lora_weights.safetensors"


_MIN_LTX_INGREDIENTS_FRAMES = 121


_DEFAULT_I2V_PASS1_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


_DEFAULT_I2V_PASS2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"


_DEFAULT_INGREDIENTS_SAMPLER = "euler_ancestral_cfg_pp"


def _i2v_workflow_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Singlei2vForUI.json",
    )


def _i2v_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Singlei2vForUI_API.json",
    )


def _t2v_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Singlet2vForUI_API.json",
    )


def _rtv_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "SingleRef2VidForUI_API.json",
    )


def _ingredients_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "SingleIngredients2Video_ForUI_API.json",
    )


def _id_lora_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "LTX2.3_ID_lora_API.json",
    )


def _flf_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows", "UsedForUIDoNotTouch", "LTX2.3_FLF_API.json",
    )


def _rtv_25_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "SingleRef2VidForUI_LTX25_API.json",
    )


def _t2v_25_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "video_ltx2_5_t2v_bult_In_Audio_API.json",
    )


def _clean_required_id_lora_name(value):
    text = str(value or _REQUIRED_LTX_ID_LORA).strip()
    choices = set(_lora_choices())
    candidates = [
        text,
        text.replace("/", "\\"),
        text.replace("\\", "/"),
        _REQUIRED_LTX_ID_LORA,
        _REQUIRED_LTX_ID_LORA.replace("\\", "/"),
    ]
    text_base = os.path.basename(text.replace("\\", "/"))
    if text_base and text_base not in candidates:
        candidates.append(text_base)
    for candidate in candidates:
        if candidate in choices:
            return candidate
    raise ValueError(
        "Required ID-LoRA was not found in ComfyUI/models/loras. "
        "Download AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K and select the LoRA file."
    )


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
    output_folder = _scene_render_output_folder(project_folder, "image_to_video_clips", payload)

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


def _normalize_sigma_list_text(value, default):
    text = str(value or "").strip()
    if not text:
        return default
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return default
    try:
        for part in parts:
            float(part)
    except Exception:
        return default
    return ", ".join(parts)


def _patch_ltx_two_pass_sampler_overrides(prompt, payload):
    _set_api_input(prompt, "218:186", "sampler_name", str(payload.get("pass1_sampler_name") or "euler_ancestral").strip() or "euler_ancestral")
    _set_api_input(prompt, "218:209", "sigmas", _normalize_sigma_list_text(payload.get("pass1_sigmas"), _DEFAULT_I2V_PASS1_SIGMAS))
    _set_api_input(prompt, "219:187", "sampler_name", str(payload.get("pass2_sampler_name") or "euler_ancestral").strip() or "euler_ancestral")
    _set_api_input(prompt, "219:208", "sigmas", _normalize_sigma_list_text(payload.get("pass2_sigmas"), _DEFAULT_I2V_PASS2_SIGMAS))


def _patch_ltx_ingredients_sampler_overrides(prompt, payload):
    _set_api_input(prompt, "218:186", "sampler_name", str(payload.get("pass1_sampler_name") or _DEFAULT_INGREDIENTS_SAMPLER).strip() or _DEFAULT_INGREDIENTS_SAMPLER)
    _set_api_input(prompt, "218:209", "sigmas", _normalize_sigma_list_text(payload.get("pass1_sigmas"), _DEFAULT_I2V_PASS1_SIGMAS))
    _set_api_input(prompt, "219:187", "sampler_name", str(payload.get("pass2_sampler_name") or _DEFAULT_INGREDIENTS_SAMPLER).strip() or _DEFAULT_INGREDIENTS_SAMPLER)
    _set_api_input(prompt, "219:208", "sigmas", _normalize_sigma_list_text(payload.get("pass2_sigmas"), _DEFAULT_I2V_PASS2_SIGMAS))


def _patch_ltx_single_pass_sampler_overrides(prompt, payload):
    _set_api_input(prompt, "218:186", "sampler_name", str(payload.get("pass1_sampler_name") or "euler_ancestral").strip() or "euler_ancestral")
    _set_api_input(prompt, "218:209", "sigmas", _normalize_sigma_list_text(payload.get("pass1_sigmas"), _DEFAULT_I2V_PASS1_SIGMAS))


def _patch_i2v_node_overrides(prompt, payload):
    _patch_ltx_two_pass_sampler_overrides(prompt, payload)
    _set_api_input(prompt, "218:222", "strength", _float_payload(payload, "pass1_inplace_strength", 1.0, 0.0, 1.0))
    _set_api_input(prompt, "218:222", "bypass", _bool_payload(payload, "pass1_inplace_bypass", False))
    _set_api_input(prompt, "219:221", "strength", _float_payload(payload, "pass2_inplace_strength", 1.0, 0.0, 1.0))
    _set_api_input(prompt, "219:221", "bypass", _bool_payload(payload, "pass2_inplace_bypass", False))


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
    output_folder = _scene_render_output_folder(project_folder, "image_to_video_clips", payload)

    image_index = _int_payload(payload, "image_index_zero_based", 0, 0, 999999)
    prompt_number = _int_payload(payload, "prompt_number_one_based", 1, 1, 999999)
    fps = _int_payload(payload, "fps", 24, 1, 120)
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)

    is_ltx25 = str(payload.get("ltx_version", "2.3") or "2.3").strip() == "2.5"
    if is_ltx25:
        prompt["938"] = _ltx25_diffusion_loader_node(payload)
        prompt["937"]["inputs"]["model"] = ["938", 0]
        prompt.pop("939", None)
        prompt.pop("271:215", None)
        prompt["271:216"] = {
            "inputs": {
                "clip_name": str(payload.get("clip_name1", "") or ""),
                "type": "ltxv",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        }
    else:
        _patch_ltx_video_model_loader(prompt, payload)
    _set_api_input(prompt, "271:256", "vae_name", str(payload.get("vae_name", "") or ""))
    if not is_ltx25:
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
    tail_loss_frames = _int_payload(payload, "tail_loss_frames", 25, 0, 10000)
    pre_frames = _int_payload(payload, "pre_frames", 50, 0, 10000)

    _set_api_input(prompt, "925", "folder_path", image_folder)
    _set_api_input(prompt, "929", "value", image_index)
    _set_api_input(prompt, "930", "value", prompt_number)
    _set_api_input(prompt, "933", "text", i2v_prompt)
    _set_api_input(prompt, "933", "output_mode", "string")
    _set_api_input(prompt, "935", "value", srt_path)
    _set_api_input(prompt, "218:287", "overwrite_mode", "overwrite")
    _set_api_input(prompt, "218:287", "tail_loss_frames", tail_loss_frames)
    _set_api_input(prompt, "218:287", "pre_frames", pre_frames)
    _patch_i2v_node_overrides(prompt, payload)
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
    output_folder = _scene_render_output_folder(project_folder, "text_to_video_clips", payload)

    prompt_number = _int_payload(payload, "prompt_number_one_based", 1, 1, 999999)
    fps = _int_payload(payload, "fps", 24, 1, 120)
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    tail_loss_frames = _int_payload(payload, "tail_loss_frames", 25, 0, 10000)
    pre_frames = _int_payload(payload, "pre_frames", 50, 0, 10000)

    is_ltx25 = str(payload.get("ltx_version", "2.3") or "2.3").strip() == "2.5"
    if is_ltx25:
        # Keep the proven custom-audio T2V graph and replace only the legacy
        # diffusion/CLIP pieces, matching the LTX 2.5 I2V and RTV adapters.
        prompt["938"] = _ltx25_diffusion_loader_node(payload)
        prompt["937"]["inputs"]["model"] = ["938", 0]
        prompt.pop("939", None)
        prompt.pop("271:215", None)
        prompt["271:216"] = {
            "inputs": {
                "clip_name": str(payload.get("clip_name1", "") or ""),
                "type": "ltxv",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load LTX 2.5 CLIP"},
        }
    else:
        _patch_ltx_video_model_loader(prompt, payload)
    _set_api_input(prompt, "271:256", "vae_name", str(payload.get("vae_name", "") or ""))
    if not is_ltx25:
        _set_api_input(prompt, "271:216", "clip_name1", str(payload.get("clip_name1", "") or ""))
        _set_api_input(prompt, "271:216", "clip_name2", str(payload.get("clip_name2", "") or ""))
    _set_api_input(prompt, "271:211", "model_name", str(payload.get("upscale_model_name", "") or ""))
    _set_api_input(prompt, "271:254", "vae_name", str(payload.get("audio_vae_name", "") or ""))

    _set_api_input(prompt, "736:424", "value", fps)
    generation_width = int(math.ceil(width / 64.0) * 64) if is_ltx25 else width
    generation_height = int(math.ceil(height / 64.0) * 64) if is_ltx25 else height
    _set_api_input(prompt, "736:425", "value", generation_width)
    _set_api_input(prompt, "736:426", "value", generation_height)
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
    _set_api_input(prompt, "218:287", "tail_loss_frames", tail_loss_frames)
    _set_api_input(prompt, "218:287", "pre_frames", pre_frames)
    if is_ltx25:
        final_resize_id = "vrgdg_ltx25_t2v_final_center_crop"
        _replace_api_input_refs(prompt, ("936", 0), (final_resize_id, 0))
        prompt[final_resize_id] = {
            "class_type": "ImageScale",
            "inputs": {"image": ["936", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"},
            "_meta": {"title": "Center crop LTX 2.5 T2V to requested output resolution"},
        }
    _patch_ltx_two_pass_sampler_overrides(prompt, payload)
    _set_api_input(prompt, "437", "value", output_folder)
    return prompt, output_folder


def _rtv_reference_strength(value):
    text = str(value or "").strip().lower()
    if text.startswith("17"):
        return "17 - light"
    if text.startswith("25"):
        return "25 - balanced"
    if text.startswith("33"):
        return "33 - strong"
    if text.startswith("41"):
        return "41 - strongest"
    return "auto - based on subject count"


def _rtv_background_mode(value, has_background, is_ltx25=False):
    text = str(value or "").strip().lower()
    if is_ltx25 and (text in {"no", "false", "off", "no_background"} or "no background" in text):
        return "no_background"
    if "neutral" in text or "placeholder" in text:
        return "neutral_placeholder_wip"
    if has_background:
        return "use_uploaded_background"
    return "no_background" if is_ltx25 else "neutral_placeholder_wip"


def _srt_time_to_seconds(value):
    text = str(value or "").strip().replace(".", ",")
    hours, minutes, rest = text.split(":", 2)
    seconds, millis = (rest.split(",", 1) + ["0"])[:2]
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int((millis + "000")[:3]) / 1000.0


def _srt_segment_frame_count(path, prompt_number, fps):
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            blocks = handle.read().replace("\r\n", "\n").replace("\r", "\n").strip().split("\n\n")
        segments = []
        for block in blocks:
            for line in block.splitlines():
                if "-->" not in line:
                    continue
                start_text, end_text = line.split("-->", 1)
                segments.append((_srt_time_to_seconds(start_text), _srt_time_to_seconds(end_text)))
                break
        index = max(0, int(prompt_number) - 1)
        if index >= len(segments):
            return 0
        start_sec, end_sec = segments[index]
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        return max(1, end_frame - start_frame)
    except Exception:
        return 0


def _pad_ingredients_preroll_tail(srt_path, prompt_number, fps, pre_frames, tail_loss_frames):
    scene_frames = _srt_segment_frame_count(srt_path, prompt_number, fps)
    original_pre_frames = pre_frames
    original_tail_loss_frames = tail_loss_frames
    if scene_frames <= 0:
        print(
            "[VRGDG Ingredients] Padding check skipped: "
            f"prompt={prompt_number}, fps={fps}, scene_frames={scene_frames}, "
            f"pre_frames={pre_frames}, tail_loss_frames={tail_loss_frames}",
            flush=True,
        )
        return pre_frames, tail_loss_frames
    current_total = scene_frames + pre_frames + tail_loss_frames
    shortfall = max(0, _MIN_LTX_INGREDIENTS_FRAMES - current_total)
    if shortfall <= 0:
        print(
            "[VRGDG Ingredients] Padding check: "
            f"prompt={prompt_number}, fps={fps}, scene_frames={scene_frames}, "
            f"original_pre={original_pre_frames}, original_tail={original_tail_loss_frames}, "
            f"total_frames={current_total}, min_frames={_MIN_LTX_INGREDIENTS_FRAMES}, "
            "added_pre=0, added_tail=0, "
            f"final_pre={pre_frames}, final_tail={tail_loss_frames}",
            flush=True,
        )
        return pre_frames, tail_loss_frames
    add_pre = shortfall // 2
    add_tail = shortfall - add_pre
    final_pre = pre_frames + add_pre
    final_tail = tail_loss_frames + add_tail
    print(
        "[VRGDG Ingredients] Padding applied: "
        f"prompt={prompt_number}, fps={fps}, scene_frames={scene_frames}, "
        f"original_pre={original_pre_frames}, original_tail={original_tail_loss_frames}, "
        f"total_before={current_total}, min_frames={_MIN_LTX_INGREDIENTS_FRAMES}, "
        f"shortfall={shortfall}, added_pre={add_pre}, added_tail={add_tail}, "
        f"final_pre={final_pre}, final_tail={final_tail}, "
        f"total_after={scene_frames + final_pre + final_tail}",
        flush=True,
    )
    return final_pre, final_tail


def _patch_rtv_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    rtv_prompt = str(payload.get("t2v_prompt", payload.get("i2v_prompt", "")) or "").strip()
    if not rtv_prompt:
        raise ValueError("Reference-to-video prompt is empty.")

    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    srt_path = os.path.abspath(str(payload.get("srt_path", "") or "").strip().strip('"'))
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file was not found: {srt_path}")
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    output_folder = _scene_render_output_folder(project_folder, "reference_to_video_clips", payload)

    prompt_number = _int_payload(payload, "prompt_number_one_based", 1, 1, 999999)
    fps = _int_payload(payload, "fps", 24, 1, 120)
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    tail_loss_frames = _int_payload(payload, "tail_loss_frames", 25, 0, 10000)
    pre_frames = _int_payload(payload, "pre_frames", 50, 0, 10000)

    is_ltx25 = str(payload.get("ltx_version", "2.3") or "2.3").strip() == "2.5"
    if is_ltx25:
        prompt["956"] = _ltx25_diffusion_loader_node(payload)
    else:
        _patch_ltx_video_model_loader(prompt, payload)
    _set_api_input(prompt, "271:256", "vae_name", str(payload.get("vae_name", "") or ""))
    if is_ltx25:
        _set_api_input(prompt, "271:216", "clip_name", str(payload.get("clip_name1", "") or ""))
    else:
        _set_api_input(prompt, "271:216", "clip_name1", str(payload.get("clip_name1", "") or ""))
        _set_api_input(prompt, "271:216", "clip_name2", str(payload.get("clip_name2", "") or ""))
    _set_optional_api_input(prompt, "271:211", "model_name", str(payload.get("upscale_model_name", "") or ""))
    _set_api_input(prompt, "271:254", "vae_name", str(payload.get("audio_vae_name", "") or ""))

    _set_api_input(prompt, "736:424", "value", fps)
    stage1_width = max(64, int(round((width / 2) / 32.0)) * 32) if is_ltx25 else width
    stage1_height = max(64, int(round((height / 2) / 32.0)) * 32) if is_ltx25 else height
    _set_api_input(prompt, "736:425", "value", stage1_width)
    _set_api_input(prompt, "736:426", "value", stage1_height)
    _set_api_input(prompt, "736:449", "value", seed)
    _set_api_input(prompt, "736:551", "value", 0)

    msr_default = _REQUIRED_LTX25_MSR_LORA if is_ltx25 else _REQUIRED_LTX_MSR_LORA
    msr_lora_name = _clean_msr_lora_name(payload.get("msr_lora_name", msr_default))
    use_user_loras = _bool_payload(payload, "use_custom_loras", False)
    user_lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS)
    _set_api_input(prompt, "937", "use_custom_loras", use_user_loras)
    _set_api_input(prompt, "937", "lora_count", user_lora_count if use_user_loras else 0)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        if use_user_loras and slot <= user_lora_count:
            legacy_strength = _float_payload(payload, f"strength_{slot}", 1.0)
            first_pass_strength = _float_payload(payload, f"first_pass_strength_{slot}", legacy_strength)
            second_pass_strength = (
                _float_payload(payload, f"second_pass_strength_{slot}", legacy_strength)
                if is_ltx25 else 0.0
            )
            lora_name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA))
        else:
            first_pass_strength = 1.0
            second_pass_strength = 0.0
            lora_name = _NONE_LORA
        _set_api_input(prompt, "937", f"lora_{slot}", lora_name)
        _set_api_input(prompt, "937", f"first_pass_strength_{slot}", first_pass_strength)
        _set_api_input(prompt, "937", f"second_pass_strength_{slot}", second_pass_strength)
    _set_api_input(prompt, "953", "lora_name", msr_lora_name)
    _set_api_input(prompt, "953", "strength_model", _float_payload(payload, "msr_first_pass_strength", 1.0))

    references = payload.get("rtv_references") if isinstance(payload.get("rtv_references"), dict) else {}
    subjects = references.get("subjects") if isinstance(references.get("subjects"), list) else []
    subject_images = [_prepare_optional_input_image_name(item) for item in subjects[:4]]
    if references.get("use_subject_placeholder") and not any(image != "(none)" for image in subject_images):
        subject_images = [_ensure_placeholder_load_image()]
    while len(subject_images) < 4:
        subject_images.append("(none)")
    background_image = _prepare_optional_input_image_name(references.get("background"))
    has_background = background_image != "(none)"

    for index, image_name in enumerate(subject_images, start=1):
        _set_api_input(prompt, "951", f"subject_{index}", image_name)
    _set_api_input(prompt, "951", "background_image", background_image)
    _set_api_input(prompt, "951", "background_mode", _rtv_background_mode(payload.get("msr_background_mode"), has_background, is_ltx25))
    if is_ltx25:
        requested_strength = str(payload.get("msr_reference_strength", "33") or "33").strip()
        reference_frames = "25" if requested_strength.startswith("25") else "33"
        _set_api_input(prompt, "939", "reference_frames", reference_frames)
        _set_api_input(prompt, "961", "reference_frames", reference_frames)
        _set_api_input(prompt, "959", "model_name", str(payload.get("upscale_model_name", "") or ""))
    else:
        _set_api_input(prompt, "951", "reference_strength", _rtv_reference_strength(payload.get("msr_reference_strength")))

    _set_api_input(prompt, "927", "audio_file", audio_path)
    _set_api_input(prompt, "927", "seek_seconds", 0)
    _set_api_input(prompt, "927", "duration", 0)
    _set_api_input(prompt, "930", "value", prompt_number)
    _set_api_input(prompt, "933", "text", rtv_prompt)
    _set_api_input(prompt, "933", "output_mode", "string")
    _set_api_input(prompt, "935", "value", srt_path)
    _set_api_input(prompt, "218:287", "overwrite_mode", "overwrite")
    _set_api_input(prompt, "218:287", "tail_loss_frames", tail_loss_frames)
    _set_api_input(prompt, "218:287", "pre_frames", pre_frames)
    _patch_ltx_single_pass_sampler_overrides(prompt, payload)
    if is_ltx25:
        _set_api_input(prompt, "964", "sampler_name", str(payload.get("pass2_sampler_name") or "euler_ancestral"))
        _set_api_input(prompt, "965", "sigmas", str(payload.get("pass2_sigmas") or _DEFAULT_I2V_PASS2_SIGMAS))
        _set_api_input(prompt, "963", "noise_seed", seed)
        final_resize_id = "vrgdg_ltx25_rtv_final_resize"
        _replace_api_input_refs(prompt, ("954", 0), (final_resize_id, 0))
        prompt[final_resize_id] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["954", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
            "_meta": {"title": "Resize LTX 2.5 RTV to requested final resolution"},
        }
    _set_api_input(prompt, "437", "value", output_folder)
    return prompt, output_folder


def _patch_ingredients_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    ingredients_prompt = str(payload.get("t2v_prompt", payload.get("i2v_prompt", "")) or "").strip()
    if not ingredients_prompt:
        raise ValueError("Ingredients-to-video prompt is empty.")

    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    srt_path = os.path.abspath(str(payload.get("srt_path", "") or "").strip().strip('"'))
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file was not found: {srt_path}")
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    output_folder = _scene_render_output_folder(project_folder, "ingredients_to_video_clips", payload)

    image_path = os.path.abspath(str(payload.get("ingredients_image_path", "") or "").strip().strip('"'))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Ingredients reference image was not found: {image_path}")

    prompt_number = _int_payload(payload, "prompt_number_one_based", 1, 1, 999999)
    fps = _int_payload(payload, "fps", 24, 1, 120)
    width = _int_payload(payload, "width", 768, 64, 4096)
    height = _int_payload(payload, "height", 448, 64, 4096)
    shorter_size = min(width, height)
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    tail_loss_frames = _int_payload(payload, "tail_loss_frames", 25, 0, 10000)
    pre_frames = _int_payload(payload, "pre_frames", 50, 0, 10000)
    pre_frames, tail_loss_frames = _pad_ingredients_preroll_tail(
        srt_path,
        prompt_number,
        fps,
        pre_frames,
        tail_loss_frames,
    )

    is_ltx25 = str(payload.get("ltx_version", "2.3") or "2.3").strip() == "2.5"
    if is_ltx25:
        prompt["958"] = _ltx25_diffusion_loader_node(payload)
        prompt["937"]["inputs"]["model"] = ["958", 0]
        prompt.pop("959", None)
        prompt.pop("271:215", None)
        prompt["271:216"] = {
            "inputs": {
                "clip_name": str(payload.get("clip_name1", "") or ""),
                "type": "ltxv",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        }
    else:
        _patch_ltx_video_model_loader(prompt, payload)
    _set_api_input(prompt, "271:256", "vae_name", str(payload.get("vae_name", "") or ""))
    if not is_ltx25:
        _set_api_input(prompt, "271:216", "clip_name1", str(payload.get("clip_name1", "") or ""))
        _set_api_input(prompt, "271:216", "clip_name2", str(payload.get("clip_name2", "") or ""))
    _set_api_input(prompt, "271:211", "model_name", str(payload.get("upscale_model_name", "") or ""))
    _set_api_input(prompt, "271:254", "vae_name", str(payload.get("audio_vae_name", "") or ""))

    _set_api_input(prompt, "736:424", "value", fps)
    _set_api_input(prompt, "736:449", "value", seed)
    _set_api_input(prompt, "736:551", "value", 0)
    _set_optional_api_input(prompt, "940", "width", width)
    _set_optional_api_input(prompt, "940", "height", height)
    _set_optional_api_input(prompt, "943", "resize_type.shorter_size", shorter_size)

    required_lora = _clean_lora_name(payload.get("ingredients_lora_name", _REQUIRED_LTX_INGREDIENTS_LORA))
    required_strength = _float_payload(payload, "ingredients_first_pass_strength", 1.0)
    use_user_loras = _bool_payload(payload, "use_custom_loras", False)
    user_lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS - 1)
    total_lora_count = 1 + (user_lora_count if use_user_loras else 0)
    _set_api_input(prompt, "937", "use_custom_loras", True)
    _set_api_input(prompt, "937", "lora_count", total_lora_count)
    _set_api_input(prompt, "937", "lora_1", required_lora)
    _set_api_input(prompt, "937", "first_pass_strength_1", required_strength)
    _set_api_input(prompt, "937", "second_pass_strength_1", 0.0)
    for slot in range(2, _MAX_LORA_SLOTS + 1):
        user_slot = slot - 1
        if use_user_loras and user_slot <= user_lora_count:
            legacy_strength = _float_payload(payload, f"strength_{user_slot}", 1.0)
            lora_name = _clean_lora_name(payload.get(f"lora_{user_slot}", _NONE_LORA))
            first_pass_strength = _float_payload(payload, f"first_pass_strength_{user_slot}", legacy_strength)
            second_pass_strength = _float_payload(payload, f"second_pass_strength_{user_slot}", legacy_strength)
        else:
            lora_name = _NONE_LORA
            first_pass_strength = 1.0
            second_pass_strength = 1.0
        _set_api_input(prompt, "937", f"lora_{slot}", lora_name)
        _set_api_input(prompt, "937", f"first_pass_strength_{slot}", first_pass_strength)
        _set_api_input(prompt, "937", f"second_pass_strength_{slot}", second_pass_strength)

    _set_api_input(prompt, "957", "image", image_path)
    _set_api_input(prompt, "957", "custom_width", 0)
    _set_api_input(prompt, "957", "custom_height", 0)
    _set_api_input(prompt, "927", "audio_file", audio_path)
    _set_api_input(prompt, "927", "seek_seconds", 0)
    _set_api_input(prompt, "927", "duration", 0)
    _set_api_input(prompt, "930", "value", prompt_number)
    _set_api_input(prompt, "933", "text", ingredients_prompt)
    _set_api_input(prompt, "933", "output_mode", "string")
    _set_api_input(prompt, "935", "value", srt_path)
    _set_api_input(prompt, "218:287", "overwrite_mode", "overwrite")
    _set_api_input(prompt, "218:287", "tail_loss_frames", tail_loss_frames)
    _set_api_input(prompt, "218:287", "pre_frames", pre_frames)
    _patch_ltx_ingredients_sampler_overrides(prompt, payload)
    _set_api_input(prompt, "437", "value", output_folder)
    return prompt, output_folder


def _id_lora_source_image_path(payload):
    raw_path = str(
        payload.get("source_image_path")
        or payload.get("image_path")
        or payload.get("first_frame_path")
        or payload.get("approved_image_path")
        or ""
    ).strip().strip('"')
    if raw_path:
        image_path = os.path.abspath(raw_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"ID-LoRA image input was not found: {image_path}")
        return image_path
    image_name = _prepare_load_image_name(
        "",
        payload.get("source_image_data", "") or payload.get("image_data", ""),
        payload.get("source_image_name", "") or payload.get("image_name", "id_lora_image.png"),
    )
    if image_name:
        return os.path.join(folder_paths.get_input_directory(), image_name)
    raise ValueError("ID-LoRA needs an image input.")


def _id_lora_reference_audio_path(payload):
    raw_path = str(
        payload.get("id_reference_audio_path")
        or payload.get("reference_audio_path")
        or payload.get("voice_reference_audio_path")
        or payload.get("voice_sample_path")
        or payload.get("audio_path")
        or ""
    ).strip().strip('"')
    if not raw_path:
        raise ValueError("ID-LoRA needs a reference voice audio sample.")
    audio_path = os.path.abspath(raw_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"ID-LoRA reference voice audio was not found: {audio_path}")
    return audio_path


def _patch_id_lora_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    id_prompt = str(payload.get("id_lora_prompt", payload.get("i2v_prompt", payload.get("prompt", ""))) or "").strip()
    if not id_prompt:
        raise ValueError("ID-LoRA prompt is empty.")

    image_path = _id_lora_source_image_path(payload)
    reference_audio_path = _id_lora_reference_audio_path(payload)
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    output_folder = _scene_render_output_folder(project_folder, "id_lora_i2v_clips", payload)

    fps = _int_payload(payload, "fps", 24, 1, 120)
    width = _int_payload(payload, "width", 1920, 64, 4096)
    height = _int_payload(payload, "height", 1080, 64, 4096)
    duration = _float_payload(payload, "duration", 5.0, 0.25, 120.0)
    seed_mode = str(payload.get("seed_mode", "fixed") or "fixed").strip().lower()
    pass1_seed = _int_payload(payload, "pass1_seed", _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF), 0, 0xFFFFFFFFFFFFFFFF)
    pass2_seed = _int_payload(payload, "pass2_seed", _int_payload(payload, "seed_2", 42, 0, 0xFFFFFFFFFFFFFFFF), 0, 0xFFFFFFFFFFFFFFFF)
    if seed_mode in {"random", "randomize"}:
        pass1_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        pass2_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)

    is_ltx25 = str(payload.get("ltx_version", "2.3") or "2.3").strip() == "2.5"
    if is_ltx25:
        prompt["971"] = _ltx25_diffusion_loader_node(payload)
        prompt["972"]["inputs"]["model"] = ["971", 0]
        prompt.pop("970", None)
        prompt.pop("969", None)
        prompt["968"] = {
            "inputs": {
                "clip_name": str(payload.get("clip_name1", "") or ""),
                "type": "ltxv",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        }
    else:
        _patch_ltx_video_model_loader(prompt, payload)
        _set_optional_api_input(prompt, "969", "unet_name", _clean_i2v_unet_name(payload.get("unet_name", "")))
        _set_optional_api_input(prompt, "971", "model_name", str(payload.get("diffusion_model_name") or payload.get("model_name") or ""))
    _set_api_input(prompt, "966", "vae_name", str(payload.get("audio_vae_name", "") or ""))
    _set_api_input(prompt, "967", "vae_name", str(payload.get("vae_name", "") or ""))
    if not is_ltx25:
        _set_api_input(prompt, "968", "clip_name1", str(payload.get("clip_name1", "") or ""))
        _set_api_input(prompt, "968", "clip_name2", str(payload.get("clip_name2", "") or ""))
    _set_api_input(prompt, "951", "model_name", str(payload.get("upscale_model_name", "") or ""))

    _set_api_input(prompt, "957", "value", id_prompt)
    _set_api_input(prompt, "963", "image", image_path)
    _set_api_input(prompt, "963", "custom_width", 0)
    _set_api_input(prompt, "963", "custom_height", 0)
    _set_api_input(prompt, "964", "audio_file", reference_audio_path)
    _set_api_input(prompt, "964", "seek_seconds", _float_payload(payload, "reference_audio_seek_seconds", 0.0, 0.0, 36000.0))
    _set_api_input(prompt, "964", "duration", _float_payload(payload, "reference_audio_duration", 0.0, 0.0, 36000.0))

    _set_api_input(prompt, "937", "value", width)
    _set_api_input(prompt, "949", "value", height)
    _set_api_input(prompt, "945", "value", duration)
    _set_api_input(prompt, "946", "value", fps)
    _set_api_input(prompt, "939", "longer_edge", width)

    _set_api_input(prompt, "954", "identity_guidance_scale", _float_payload(payload, "identity_guidance_scale", 3.0, 0.0, 20.0))
    _set_api_input(prompt, "954", "start_percent", 0.0)
    _set_api_input(prompt, "954", "end_percent", 1.0)

    _set_api_input(prompt, "924", "sampler_name", str(payload.get("pass1_sampler_name") or "euler_ancestral").strip() or "euler_ancestral")
    _set_api_input(prompt, "929", "sigmas", _normalize_sigma_list_text(payload.get("pass1_sigmas"), _DEFAULT_I2V_PASS1_SIGMAS))
    _set_api_input(prompt, "915", "noise_seed", pass1_seed)
    _set_api_input(prompt, "936", "strength", _float_payload(payload, "pass1_inplace_strength", 0.7, 0.0, 1.0))
    _set_api_input(prompt, "936", "bypass", _bool_payload(payload, "pass1_inplace_bypass", False))
    _set_api_input(prompt, "917", "sampler_name", str(payload.get("pass2_sampler_name") or "euler_ancestral").strip() or "euler_ancestral")
    _set_api_input(prompt, "918", "sigmas", _normalize_sigma_list_text(payload.get("pass2_sigmas"), _DEFAULT_I2V_PASS2_SIGMAS))
    _set_api_input(prompt, "914", "noise_seed", pass2_seed)
    _set_api_input(prompt, "923", "strength", _float_payload(payload, "pass2_inplace_strength", 1.0, 0.0, 1.0))
    _set_api_input(prompt, "923", "bypass", _bool_payload(payload, "pass2_inplace_bypass", False))

    required_lora = _clean_required_id_lora_name(payload.get("id_lora_name") or payload.get("required_id_lora_name"))
    use_user_loras = _bool_payload(payload, "use_custom_loras", False)
    user_lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS - 1)
    total_lora_count = 1 + (user_lora_count if use_user_loras else 0)
    _set_api_input(prompt, "972", "use_custom_loras", True)
    _set_api_input(prompt, "972", "lora_count", total_lora_count)
    _set_api_input(prompt, "972", "lora_1", required_lora)
    _set_api_input(prompt, "972", "first_pass_strength_1", _float_payload(payload, "id_lora_first_pass_strength", 1.0))
    _set_api_input(prompt, "972", "second_pass_strength_1", _float_payload(payload, "id_lora_second_pass_strength", 1.0))
    for slot in range(2, _MAX_LORA_SLOTS + 1):
        user_slot = slot - 1
        if use_user_loras and user_slot <= user_lora_count:
            legacy_strength = _float_payload(payload, f"strength_{user_slot}", 1.0)
            lora_name = _clean_lora_name(payload.get(f"lora_{user_slot}", _NONE_LORA))
            first_pass_strength = _float_payload(payload, f"first_pass_strength_{user_slot}", legacy_strength)
            second_pass_strength = _float_payload(payload, f"second_pass_strength_{user_slot}", legacy_strength)
        else:
            lora_name = _NONE_LORA
            first_pass_strength = 1.0
            second_pass_strength = 1.0
        _set_api_input(prompt, "972", f"lora_{slot}", lora_name)
        _set_api_input(prompt, "972", f"first_pass_strength_{slot}", first_pass_strength)
        _set_api_input(prompt, "972", f"second_pass_strength_{slot}", second_pass_strength)

    _set_api_input(prompt, "958", "filename_prefix", os.path.join(output_folder, "id_lora_i2v"))
    _set_api_input(prompt, "958", "frame_rate", fps)
    _set_api_input(prompt, "958", "crf", _int_payload(payload, "crf", 19, 0, 51))
    return prompt, output_folder


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
    # Both versions use the custom-audio T2V graph. LTX 2.5 is adapted in the
    # patcher by swapping the diffusion and CLIP nodes; the native-audio graph
    # is intentionally not used by the Video Builder.
    workflow_path, prompt = _load_api_template(_t2v_api_template_path())
    patched_prompt, output_folder = _patch_t2v_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": patched_prompt,
    }


def _build_rtv_api_prompt(payload):
    version = str(payload.get("ltx_version", "2.3") or "2.3").strip()
    workflow_path, prompt = _load_api_template(
        _rtv_25_api_template_path() if version == "2.5" else _rtv_api_template_path()
    )
    patched_prompt, output_folder = _patch_rtv_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": patched_prompt,
    }


def _build_ingredients_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_ingredients_api_template_path())
    patched_prompt, output_folder = _patch_ingredients_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": patched_prompt,
    }


def _patch_flf_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    video_prompt = str(payload.get("i2v_prompt", "") or "").strip()
    if not video_prompt:
        raise ValueError("First Last Frame prompt is empty.")
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    srt_path = os.path.abspath(str(payload.get("srt_path", "") or "").strip().strip('"'))
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not os.path.isfile(audio_path): raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    if not os.path.isfile(srt_path): raise FileNotFoundError(f"SRT file was not found: {srt_path}")
    if not project_folder: raise ValueError("Project folder is empty.")
    first = payload.get("first_frame") if isinstance(payload.get("first_frame"), dict) else {}
    last = payload.get("last_frame") if isinstance(payload.get("last_frame"), dict) else {}
    first_name = _prepare_optional_input_image_name(first)
    last_name = _prepare_optional_input_image_name(last)
    if first_name == "(none)": raise ValueError("First Last Frame needs a first-frame image.")
    if last_name == "(none)": raise ValueError("First Last Frame needs a last-frame image.")
    if os.path.normcase(first_name) == os.path.normcase(last_name):
        raise ValueError(f"First Last Frame resolved both inputs to the same image: {first_name}")
    output_folder = _scene_render_output_folder(project_folder, "first_last_frame_clips", payload)
    fps = _int_payload(payload, "fps", 24, 1, 120)
    is_ltx25 = str(payload.get("ltx_version", "2.3") or "2.3").strip() == "2.5"
    if is_ltx25:
        prompt["938"] = _ltx25_diffusion_loader_node(payload)
        prompt["937"]["inputs"]["model"] = ["938", 0]
        prompt.pop("939", None)
        prompt.pop("271:215", None)
        prompt["271:216"] = {
            "inputs": {
                "clip_name": str(payload.get("clip_name1", "") or ""),
                "type": "ltxv",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        }
    else:
        _patch_ltx_video_model_loader(prompt, payload)
    model_inputs = [
        ("271:256", "vae_name", payload.get("vae_name", "")),
        ("271:211", "model_name", payload.get("upscale_model_name", "")),
        ("271:254", "vae_name", payload.get("audio_vae_name", "")),
    ]
    if not is_ltx25:
        model_inputs.extend([
            ("271:216", "clip_name1", payload.get("clip_name1", "")),
            ("271:216", "clip_name2", payload.get("clip_name2", "")),
        ])
    for node_id, key, value in model_inputs:
        _set_api_input(prompt, node_id, key, str(value or ""))
    _set_api_input(prompt, "736:424", "value", fps)
    _set_api_input(prompt, "736:425", "value", _int_payload(payload,"width",1920,64,4096))
    _set_api_input(prompt, "736:426", "value", _int_payload(payload,"height",1080,64,4096))
    _set_api_input(prompt, "736:449", "value", _int_payload(payload,"seed",69,0,0xFFFFFFFFFFFFFFFF))
    _set_api_input(prompt, "736:551", "value", 0)

    # FLF is a single-pass render, but its hidden workflow deliberately uses the
    # shared two-pass LoRA loader and consumes output 0 (the first-pass model).
    # Patch that loader here as well; otherwise the UI can send enabled LoRAs
    # while the template remains at lora_count=0 and silently applies none.
    use_custom_loras = _bool_payload(payload, "use_custom_loras", False)
    lora_count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS) if use_custom_loras else 0
    _set_api_input(prompt, "937", "use_custom_loras", use_custom_loras)
    _set_api_input(prompt, "937", "lora_count", lora_count)
    for slot in range(1, _MAX_LORA_SLOTS + 1):
        lora_name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)) if slot <= lora_count else _NONE_LORA
        first_pass_strength = _float_payload(payload, f"first_pass_strength_{slot}", _float_payload(payload, f"strength_{slot}", 1.0))
        _set_api_input(prompt, "937", f"lora_{slot}", lora_name)
        _set_api_input(prompt, "937", f"first_pass_strength_{slot}", first_pass_strength)
        _set_api_input(prompt, "937", f"second_pass_strength_{slot}", 0.0)

    _set_api_input(prompt, "950", "image", first_name)
    _set_api_input(prompt, "945", "image", last_name)
    for node_id, prefix, defaults in (("958", "first", (0, 0.7, 29, 1, 0.9)), ("959", "last", (-1, 0.7, 29, 1, 1.0))):
        frame_idx, strength, crf, blur_radius, attention_strength = defaults
        _set_api_input(prompt, node_id, "frame_idx", _int_payload(payload, f"{prefix}_guide_frame_idx", frame_idx, -9999, 9999))
        _set_api_input(prompt, node_id, "strength", _float_payload(payload, f"{prefix}_guide_strength", strength, 0.0, 1.0))
        _set_api_input(prompt, node_id, "crf", _int_payload(payload, f"{prefix}_guide_crf", crf, 0, 51))
        _set_api_input(prompt, node_id, "blur_radius", _int_payload(payload, f"{prefix}_guide_blur_radius", blur_radius, 0, 7))
        interpolation = str(payload.get(f"{prefix}_guide_interpolation") or "lanczos")
        if interpolation not in {"lanczos", "bislerp", "nearest", "bilinear", "bicubic", "area", "nearest-exact"}:
            interpolation = "lanczos"
        crop = str(payload.get(f"{prefix}_guide_crop") or "center")
        if crop not in {"center", "disabled"}:
            crop = "center"
        _set_api_input(prompt, node_id, "interpolation", interpolation)
        _set_api_input(prompt, node_id, "crop", crop)
        _set_api_input(prompt, node_id, "attention_strength", _float_payload(payload, f"{prefix}_attention_strength", attention_strength, 0.0, 1.0))
    _set_api_input(prompt, "927", "audio_file", audio_path)
    _set_api_input(prompt, "927", "seek_seconds", 0); _set_api_input(prompt, "927", "duration", 0)
    _set_api_input(prompt, "930", "value", _int_payload(payload,"prompt_number_one_based",1,1,999999))
    _set_api_input(prompt, "933", "text", video_prompt); _set_api_input(prompt, "935", "value", srt_path)
    _set_api_input(prompt, "218:287", "overwrite_mode", "overwrite")
    _set_api_input(prompt, "218:287", "tail_loss_frames", _int_payload(payload, "tail_loss_frames", 25, 0, 10000))
    _set_api_input(prompt, "218:287", "pre_frames", _int_payload(payload, "pre_frames", 0, 0, 10000))
    _set_api_input(prompt, "437", "value", output_folder)
    _patch_ltx_single_pass_sampler_overrides(prompt, payload)
    return prompt, output_folder


def _build_flf_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_flf_api_template_path())
    patched, output_folder = _patch_flf_api_prompt(prompt, payload)
    first_name = str(patched.get("950", {}).get("inputs", {}).get("image", "") or "")
    last_name = str(patched.get("945", {}).get("inputs", {}).get("image", "") or "")
    first_source = payload.get("first_frame") if isinstance(payload.get("first_frame"), dict) else {}
    last_source = payload.get("last_frame") if isinstance(payload.get("last_frame"), dict) else {}
    flf_inputs = {
        "first_node": "950",
        "last_node": "945",
        "first_load_image": first_name,
        "last_load_image": last_name,
        "first_source": str(first_source.get("path") or first_source.get("name") or "embedded image data"),
        "last_source": str(last_source.get("path") or last_source.get("name") or "embedded image data"),
        "inputs_are_different": os.path.normcase(first_name) != os.path.normcase(last_name),
        "lora_node": "937",
        "loras_enabled": bool(patched.get("937", {}).get("inputs", {}).get("use_custom_loras", False)),
        "lora_count": int(patched.get("937", {}).get("inputs", {}).get("lora_count", 0) or 0),
        "loras": [
            {
                "name": str(patched.get("937", {}).get("inputs", {}).get(f"lora_{slot}", _NONE_LORA)),
                "strength": float(patched.get("937", {}).get("inputs", {}).get(f"first_pass_strength_{slot}", 1.0) or 0.0),
            }
            for slot in range(1, int(patched.get("937", {}).get("inputs", {}).get("lora_count", 0) or 0) + 1)
        ],
    }
    print(f"[VRGDG FLF] Verified inputs: {json.dumps(flf_inputs, ensure_ascii=False)}", flush=True)
    return {"workflow_path": workflow_path, "output_folder": output_folder, "prompt": patched, "flf_inputs": flf_inputs}


def _build_id_lora_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_id_lora_api_template_path())
    patched_prompt, output_folder = _patch_id_lora_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": patched_prompt,
    }


def _patch_t2v_25_api_prompt(prompt, payload):
    """Patch the native-audio LTX 2.5 T2V graph.

    LoRAs are deliberately expanded into ordinary model-only loader nodes so
    this workflow has no dependency on the legacy multi-LoRA custom node.
    """
    prompt = copy.deepcopy(prompt)
    text = str(payload.get("t2v_prompt", payload.get("i2v_prompt", "")) or "").strip()
    if not text:
        raise ValueError("T2V prompt is empty.")
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    output_folder = _scene_render_output_folder(project_folder, "text_to_video_clips", payload)

    fps = _int_payload(payload, "fps", 24, 1, 120)
    duration = max(1, int(round(_float_payload(payload, "duration", 5.0, 0.25, 3600.0))))
    seed = _int_payload(payload, "seed", 1, 0, 0xFFFFFFFFFFFFFFFF)
    _set_api_input(prompt, "405:361", "value", fps)
    _set_api_input(prompt, "405:362", "value", duration)
    _set_api_input(prompt, "405:376", "value", text)
    _set_api_input(prompt, "405:338", "noise_seed", seed)
    _set_api_input(prompt, "405:339", "noise_seed", seed)
    prompt["405:384"] = _ltx25_diffusion_loader_node(payload)
    _set_api_input(prompt, "405:385", "vae_name", str(payload.get("vae_name") or ""))
    _set_api_input(prompt, "405:386", "vae_name", str(payload.get("audio_vae_name") or ""))
    _set_api_input(prompt, "405:387", "clip_name", str(payload.get("clip_name1") or ""))
    _set_api_input(prompt, "405:371", "model_name", str(payload.get("upscale_model_name") or ""))
    _set_api_input(prompt, "409", "aspect_ratio", str(payload.get("resolution_aspect_ratio") or "16:9 (Widescreen)"))
    _set_api_input(prompt, "409", "megapixels", _float_payload(payload, "resolution_megapixels", 1.2, 0.1, 16.0))
    _set_api_input(prompt, "409", "multiple", 32)

    model_ref = ["405:384", 0]
    count = _int_payload(payload, "lora_count", 0, 0, _MAX_LORA_SLOTS) if _bool_payload(payload, "use_custom_loras", False) else 0
    for slot in range(1, count + 1):
        name = _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA))
        if name == _NONE_LORA:
            continue
        node_id = f"vrgdg_ltx25_lora_{slot}"
        prompt[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(model_ref),
                "lora_name": name,
                "strength_model": _float_payload(payload, f"first_pass_strength_{slot}", 1.0),
            },
            "_meta": {"title": f"LTX 2.5 LoRA {slot}"},
        }
        model_ref = [node_id, 0]
    _set_api_input(prompt, "405:388", "model", list(model_ref))
    _set_api_input(prompt, "405:391", "model", list(model_ref))
    prefix = os.path.join(output_folder, f"scene_{_int_payload(payload, 'prompt_number_one_based', 1, 1, 999999):04d}")
    # The supplied graph also contains an unconnected native SaveVideo node.
    # VHS_VideoCombine is the actual connected A/V output used by the Builder.
    prompt.pop("75", None)
    _set_api_input(prompt, "405:416", "frame_rate", fps)
    _set_api_input(prompt, "405:416", "filename_prefix", prefix)
    return prompt, output_folder
