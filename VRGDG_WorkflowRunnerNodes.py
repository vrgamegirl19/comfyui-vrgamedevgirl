import copy
import base64
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import folder_paths
from aiohttp import web
from server import PromptServer

from .VRGDG_ModelPathSettings import (
    custom_model_root_subfolders,
    load_custom_model_root,
    register_custom_model_root,
    save_custom_model_root,
)
from .VRGDG_MiniMaxH3Timing import calculate_minimax_h3_timing


_VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED = False
_MAX_LORA_SLOTS = 20
_NONE_LORA = "[none]"
_REQUIRED_LTX_MSR_LORA = "licon\\LTX-2.3-Licon-MSR-V1.safetensors"
_REQUIRED_LTX25_MSR_LORA = "LTX-2.5-Licon-MSR-V1.safetensors"
_REQUIRED_LTX_INGREDIENTS_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
_REQUIRED_LTX_ID_LORA = "lora_weights.safetensors"
_MIN_LTX_INGREDIENTS_FRAMES = 121
_DEFAULT_I2V_PASS1_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_DEFAULT_I2V_PASS2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"
_DEFAULT_INGREDIENTS_SAMPLER = "euler_ancestral_cfg_pp"
_MINIMAX_H3_ASPECT_RATIOS = {
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
}
_MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
_MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
_I2V_UNET_ALIASES = {
    "LTX-2.3-22B-distilled-11-Q6_K.gguf": "LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
}
_PLACEHOLDER_I2I_IMAGE_NAME = "vrgdg_placeholder_i2i.png"
_PLACEHOLDER_I2I_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
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


def _krea2_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Krea2_TextToImage_API.json",
    )


def _krea2_2pass_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "Krea2_API_2Pass.json",
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


def _rtv_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "SingleRef2VidForUI_API.json",
    )


def _ingredients_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "SingleIngredients2Video_ForUI_API.json",
    )


def _id_lora_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "LTX2.3_ID_lora_API.json",
    )


def _flf_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows", "UsedForUIDoNotTouch", "LTX2.3_FLF_API.json",
    )


def _minimax_h3_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "minimax_audio_driven_builder_api.json",
    )


def _rtv_25_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "SingleRef2VidForUI_LTX25_API.json",
    )


def _t2v_25_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "video_ltx2_5_t2v_bult_In_Audio_API.json",
    )


def _minimax_h3_2pass_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "minimax_audio_driven_builder_latent_upscale_2pass_api.json",
    )


def _minimax_h3_3pass_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "minimax_ref2video_3pass_audio_driven_api.json",
    )


def _minimax_h3_built_in_audio_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "minimax_built_in_audio_builder_api.json",
    )


def _clear_memory_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "ClearMemory_API.json",
    )


def _transcribe_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "LTX2.3_Transcribe_API.json",
    )


def _timestamped_transcribe_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "LTX2.3_Transcribe_2_API.json",
    )


def _lora_choices():
    register_custom_model_root()
    try:
        loras = folder_paths.get_filename_list("loras")
    except Exception:
        loras = []
    return [_NONE_LORA] + [name for name in loras if str(name or "").strip() != _NONE_LORA]


def _folder_choices(category):
    register_custom_model_root()
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
    values = []
    try:
        values = list(folder_paths.get_filename_list(category) or [])
    except Exception:
        values = []
    values.extend(_manual_model_folder_choices(category))
    seen = set()
    unique = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _ltx_video_model_choices():
    choices = _folder_choices(("unet", "diffusion_models"))
    gguf = []
    diffusion = []
    for choice in choices:
        text = str(choice or "").strip()
        if not text:
            continue
        if text.lower().endswith(".gguf"):
            gguf.append(text)
        else:
            diffusion.append(text)
    return gguf, diffusion


def _model_choice_exists(category, value):
    requested = str(value or "").strip()
    if not requested:
        return False
    requested_base = os.path.basename(requested.replace("\\", "/"))
    for choice in _folder_choices(category):
        text = str(choice or "").strip()
        if not text:
            continue
        if text == requested:
            return True
        if os.path.basename(text.replace("\\", "/")) == requested_base:
            return True
    return False


def _require_model_choice(category, value, label):
    if _model_choice_exists(category, value):
        return
    folder_hint = category[0] if isinstance(category, (list, tuple)) else category
    raise ValueError(
        f"{label} '{value}' was not found in ComfyUI/models/{folder_hint}. "
        "Install the model there, refresh/restart ComfyUI, then try Krea2 again."
    )


def _manual_model_folder_choices(category):
    category = str(category or "").strip()
    if not category:
        return []
    extensions = {
        "unet": {".safetensors", ".ckpt", ".pt", ".bin", ".gguf"},
        "diffusion_models": {".safetensors", ".ckpt", ".pt", ".bin", ".gguf"},
        "clip": {".safetensors", ".ckpt", ".pt", ".bin"},
        "text_encoders": {".safetensors", ".ckpt", ".pt", ".bin"},
        "vae": {".safetensors", ".ckpt", ".pt", ".bin"},
        "upscale_models": {".safetensors", ".ckpt", ".pt", ".bin"},
        "latent_upscale_models": {".safetensors", ".ckpt", ".pt", ".bin"},
    }.get(category, {".safetensors", ".ckpt", ".pt", ".bin", ".gguf"})
    roots = []
    try:
        roots.extend(folder_paths.get_folder_paths(category) or [])
    except Exception:
        pass
    roots.extend(custom_model_root_subfolders(category))
    base = getattr(folder_paths, "models_dir", None)
    if base:
        roots.append(os.path.join(base, category))
    choices = []
    seen_roots = set()
    for root in roots:
        root = os.path.abspath(str(root or ""))
        if not root or root in seen_roots or not os.path.isdir(root):
            continue
        seen_roots.add(root)
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() not in extensions:
                    continue
                rel = os.path.relpath(os.path.join(dirpath, filename), root)
                choices.append(rel.replace("/", os.sep).replace("\\", os.sep))
    return choices


def _clean_i2v_unet_name(value):
    text = str(value or "").strip()
    return _I2V_UNET_ALIASES.get(text, text)


def _replace_api_input_refs(prompt, old_ref, new_ref):
    old = [str(old_ref[0]), int(old_ref[1])]
    new = [str(new_ref[0]), int(new_ref[1])]
    replaced = 0
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == old[0] and int(value[1] or 0) == old[1]:
                inputs[key] = list(new)
                replaced += 1
    return replaced


def _collapse_ltx_video_model_switch(prompt, switch_id, selected_loader_id, unused_loader_id):
    switch_key = str(switch_id or "").strip()
    selected_key = str(selected_loader_id or "").strip()
    unused_key = str(unused_loader_id or "").strip()
    if not switch_key or not selected_key:
        return False
    if switch_key not in prompt or selected_key not in prompt:
        return False
    _replace_api_input_refs(prompt, (switch_key, 0), (selected_key, 0))
    prompt.pop(switch_key, None)
    if unused_key and unused_key != selected_key:
        prompt.pop(unused_key, None)
    return True


def _patch_ltx_video_model_loader(prompt, payload):
    use_gguf = _bool_payload(payload, "use_gguf_model", True)
    gguf_name = _clean_i2v_unet_name(payload.get("unet_name", ""))
    diffusion_name = str(payload.get("diffusion_model_name") or payload.get("model_name") or "").strip()
    if not diffusion_name:
        diffusion_name = gguf_name
    switch_id = _optional_api_node_id_by_class(prompt, "ComfySwitchNode", "Switch-use GGUF", fallback_ids=("955", "939", "959"))
    gguf_loader_id = _optional_api_node_id_by_class(prompt, "UnetLoaderGGUF", fallback_ids=("271:215", "969"))
    diffusion_loader_id = _optional_api_node_id_by_class(prompt, "DiffusionModelLoaderKJ", fallback_ids=("956", "938", "958"))
    if switch_id:
        _set_optional_api_input(prompt, switch_id, "switch", use_gguf)
    if gguf_loader_id:
        _set_optional_api_input(prompt, gguf_loader_id, "unet_name", gguf_name)
    if diffusion_loader_id:
        _set_optional_api_input(prompt, diffusion_loader_id, "model_name", diffusion_name)
    if switch_id and gguf_loader_id and diffusion_loader_id:
        if use_gguf:
            _collapse_ltx_video_model_switch(prompt, switch_id, gguf_loader_id, diffusion_loader_id)
        else:
            _collapse_ltx_video_model_switch(prompt, switch_id, diffusion_loader_id, gguf_loader_id)


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


def _ltx25_diffusion_loader_node(payload):
    return {
        "inputs": {
            "model_name": str(payload.get("diffusion_model_name") or payload.get("model_name") or ""),
            "weight_dtype": "default",
            "compute_dtype": "default",
            "patch_cublaslinear": False,
            "sage_attention": "auto" if _bool_payload(payload, "use_sage_attention", False) else "disabled",
            "enable_fp16_accumulation": _bool_payload(payload, "enable_fp16_accumulation", False),
        },
        "class_type": "DiffusionModelLoaderKJ",
        "_meta": {"title": "LTX 2.5 Diffusion Model Loader"},
    }


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


def _first_payload_value(payload, *keys, default=None):
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return default


def _minimax_h3_collection(value, collection_keys=()):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in collection_keys:
            if isinstance(value.get(key), list):
                return value[key]
        return list(value.values())
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if parsed is not None and parsed is not value:
        return _minimax_h3_collection(parsed, collection_keys)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _minimax_h3_media_path(value):
    if isinstance(value, dict):
        value = value.get("path") or value.get("file") or value.get("image") or value.get("video")
    return str(value or "").strip().strip('"').strip("'")


def _minimax_h3_image_paths(payload):
    raw = _first_payload_value(payload, "image_paths", "reference_images", "images", default=[])
    paths = [
        path
        for path in (
            _minimax_h3_media_path(item)
            for item in _minimax_h3_collection(raw, ("image_paths", "images"))
        )
        if path
    ]
    if len(paths) > _MINIMAX_H3_MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"MiniMax H3 supports at most {_MINIMAX_H3_MAX_REFERENCE_IMAGES} reference images; "
            f"received {len(paths)}."
        )
    return paths


def _minimax_h3_video_references(payload):
    raw = _first_payload_value(payload, "video_references", "reference_videos", "videos", default=[])
    references = []
    for item in _minimax_h3_collection(raw, ("video_references", "videos")):
        if isinstance(item, dict):
            path = _minimax_h3_media_path(item)
            try:
                start_seconds = max(0.0, float(_first_payload_value(
                    item, "start_seconds", "start", "seek_seconds", default=0
                ) or 0))
                duration = max(0.0, float(_first_payload_value(
                    item, "duration", "duration_seconds", default=0
                ) or 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("MiniMax H3 video reference timing must be numeric.") from exc
            use_audio_value = _first_payload_value(
                item, "use_audio", "include_audio", "reference_audio", default=False
            )
            use_audio = (
                str(use_audio_value).strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(use_audio_value, str)
                else bool(use_audio_value)
            )
        else:
            path = _minimax_h3_media_path(item)
            start_seconds = 0.0
            duration = 0.0
            use_audio = False
        if path:
            references.append({
                "path": path,
                "start_seconds": start_seconds,
                "duration": duration,
                "use_audio": use_audio,
            })
    if len(references) > _MINIMAX_H3_MAX_REFERENCE_VIDEOS:
        raise ValueError(
            f"MiniMax H3 supports at most {_MINIMAX_H3_MAX_REFERENCE_VIDEOS} reference videos; "
            f"received {len(references)}."
        )
    return references


def _probe_media_duration_seconds(path):
    ffprobe_path = _ffprobe_path_for(_find_ffmpeg_path())
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "FFprobe could not read the audio duration.").strip())
    try:
        duration = float((result.stdout or "").strip().splitlines()[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"FFprobe did not return a valid duration for: {path}") from exc
    if duration <= 0:
        raise ValueError(f"Source audio has no usable duration: {path}")
    return duration


def _trim_minimax_h3_audio_context(source_path, project_folder, scene_number, timing):
    target_dir = os.path.join(project_folder, "minimax_h3_scene_audio")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"scene_audio_{scene_number:04d}.wav")
    ffmpeg_path = _find_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{timing.audio_trim_start_seconds:.9f}",
        "-i",
        source_path,
        "-t",
        f"{timing.audio_trim_duration_seconds:.9f}",
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        target_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(target_path):
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg failed to trim MiniMax H3 scene audio.").strip())
    try:
        with wave.open(target_path, "rb") as handle:
            actual_duration = handle.getnframes() / float(handle.getframerate())
    except Exception as exc:
        raise RuntimeError(f"Could not verify the trimmed MiniMax H3 audio: {target_path}") from exc
    if actual_duration + 0.02 < timing.audio_trim_duration_seconds:
        raise ValueError(
            "The trimmed MiniMax H3 audio ended before the required scene context. "
            f"Needed {timing.audio_trim_duration_seconds:.3f}s; received {actual_duration:.3f}s."
        )
    return {
        "audio_path": target_path,
        "start": timing.audio_trim_start_seconds,
        "duration": actual_duration,
        "requested_duration": timing.audio_trim_duration_seconds,
        "format": "pcm_s16le_wav",
    }


def _prepare_scene_audio_clip(payload):
    source_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not source_path:
        raise ValueError("Audio file path is empty.")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Audio file was not found: {source_path}")
    if not project_folder:
        raise ValueError("Create or load a project before preparing scene audio.")
    os.makedirs(project_folder, exist_ok=True)
    scene_number = int(_float_payload(payload, "scene_number", 1, minimum=1, maximum=9999))
    start = _float_payload(payload, "start_seconds", 0.0, minimum=0.0, maximum=24 * 60 * 60)
    duration = _float_payload(payload, "duration_seconds", 8.0, minimum=0.05, maximum=120.0)
    target_dir = os.path.join(project_folder, "minimax_h3_scene_audio")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"scene_audio_{scene_number:04d}.wav")
    ffmpeg_path = _find_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start:.9f}",
        "-i",
        source_path,
        "-t",
        f"{duration:.9f}",
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        target_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(target_path):
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg failed to prepare scene audio.").strip())
    actual_duration = _probe_media_duration_seconds(target_path)
    return {
        "audio_path": target_path,
        "start": start,
        "duration": actual_duration,
        "requested_duration": duration,
        "format": "pcm_s16le_wav",
    }


def _minimax_h3_output_location(project_folder, scene_number):
    project_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        os.path.basename(os.path.normpath(project_folder)),
    ).strip("_") or "project"
    project_key = hashlib.sha1(os.path.normcase(project_folder).encode("utf-8")).hexdigest()[:8]
    relative_dir = os.path.join(
        "VRGDG_MiniMaxH3",
        f"{project_name}_{project_key}",
        f"scene_{scene_number:04d}",
    )
    output_folder = os.path.join(folder_paths.get_output_directory(), relative_dir)
    os.makedirs(output_folder, exist_ok=True)
    filename_prefix = os.path.join(
        relative_dir,
        f"MiniMaxH3_scene_{scene_number:04d}",
    ).replace("\\", "/")
    return output_folder, filename_prefix


def _clean_lora_name(value):
    text = str(value or _NONE_LORA).strip()
    choices = set(_lora_choices())
    if text not in choices:
        return _NONE_LORA
    return text


def _clean_msr_lora_name(value):
    text = str(value or _REQUIRED_LTX_MSR_LORA).strip()
    choices = set(_lora_choices())
    candidates = [
        text,
        text.replace("/", "\\"),
        text.replace("\\", "/"),
        _REQUIRED_LTX_MSR_LORA,
        _REQUIRED_LTX_MSR_LORA.replace("\\", "/"),
        "LTX-2.3-Licon-MSR-V1.safetensors",
    ]
    for candidate in candidates:
        if candidate in choices:
            return candidate
    return _clean_lora_name(text)


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


def _prepare_optional_input_image_name(image_info):
    if not isinstance(image_info, dict):
        return "(none)"

    raw_path = str(image_info.get("path") or image_info.get("filename") or "").strip().strip('"')
    if raw_path:
        if os.path.isabs(raw_path):
            return _prepare_load_image_name(raw_path, "", image_info.get("name") or "reference.png") or "(none)"
        clean_path = raw_path.replace("\\", "/")
        if "/" not in clean_path:
            return clean_path
        candidate_bases = [folder_paths.get_input_directory(), folder_paths.get_output_directory()]
        get_temp_directory = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_directory):
            candidate_bases.append(get_temp_directory())
        for base_dir in candidate_bases:
            candidate_path = os.path.abspath(os.path.join(base_dir, clean_path))
            try:
                if os.path.commonpath([os.path.abspath(base_dir), candidate_path]) != os.path.abspath(base_dir):
                    continue
            except ValueError:
                continue
            if os.path.isfile(candidate_path):
                return _prepare_load_image_name(candidate_path, "", image_info.get("name") or os.path.basename(clean_path)) or "(none)"

    image_name = str(image_info.get("name") or "reference.png")
    prepared = _prepare_load_image_name("", image_info.get("data") or "", image_name)
    return prepared or "(none)"


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
        try:
            from PIL import Image
            with Image.open(target_path) as image:
                image.verify()
            return _PLACEHOLDER_I2I_IMAGE_NAME
        except Exception:
            try:
                os.remove(target_path)
            except OSError:
                pass

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


def _set_api_input(prompt, node_id, input_name, value):
    node = prompt.get(str(node_id))
    if not isinstance(node, dict):
        raise KeyError(f"API prompt node {node_id} was not found.")
    inputs = node.setdefault("inputs", {})
    inputs[input_name] = value


def _scene_render_output_folder(project_folder, folder_name, payload):
    scene_number = _int_payload(payload, "scene_number", 0, 0, 999999)
    root = os.path.join(project_folder, folder_name)
    if scene_number > 0:
        root = os.path.join(root, f"scene_{scene_number:04d}")
    os.makedirs(root, exist_ok=True)
    return root


def _set_optional_api_input(prompt, node_id, input_name, value):
    node = prompt.get(str(node_id))
    if not isinstance(node, dict):
        return False
    inputs = node.setdefault("inputs", {})
    inputs[input_name] = value
    return True


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


def _api_node_title(node):
    meta = node.get("_meta") if isinstance(node, dict) else {}
    return str(meta.get("title", "") if isinstance(meta, dict) else "").strip()


def _optional_api_node_id_by_class(prompt, class_type, title="", fallback_ids=()):
    wanted_class = str(class_type or "").strip()
    wanted_title = str(title or "").strip()
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "") or "").strip() != wanted_class:
            continue
        if wanted_title and _api_node_title(node) != wanted_title:
            continue
        return str(node_id)
    for node_id in fallback_ids:
        node = prompt.get(str(node_id))
        if isinstance(node, dict) and str(node.get("class_type", "") or "").strip() == wanted_class:
            return str(node_id)
    return ""


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
    _set_api_input(prompt, "218:287", "tail_loss_frames", tail_loss_frames)
    _set_api_input(prompt, "218:287", "pre_frames", pre_frames)
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
    image_name = str(payload.get("ingredients_image_name", "") or "ingredients_reference.png")
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

    # Reroute is a canvas-only node. Resolve every link originating from one
    # directly to the reroute's incoming source before converting the graph.
    reroute_ids = {str(node.get("id")) for node in workflow.get("nodes", []) if node.get("type") == "Reroute"}
    reroute_inputs = {}
    for node in workflow.get("nodes", []):
        node_id = str(node.get("id"))
        if node_id not in reroute_ids:
            continue
        incoming = next((item.get("link") for item in node.get("inputs", []) or [] if item.get("link") is not None), None)
        if incoming is not None and int(incoming) in links:
            reroute_inputs[node_id] = list(links[int(incoming)])

    def resolve_source(source):
        seen = set()
        current = list(source)
        while str(current[0]) in reroute_inputs and str(current[0]) not in seen:
            seen.add(str(current[0]))
            current = list(reroute_inputs[str(current[0])])
        return current

    for link_id, source in list(links.items()):
        links[link_id] = resolve_source(source)

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
        if class_type in {"SetNode", "GetNode", "MarkdownNote", "Reroute"}:
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


_MINIMAX_H3_SAGE_ATTENTION_MODES = {
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
}


def _require_minimax_h3_memory_efficient_sage_attention():
    try:
        from importlib import metadata

        try:
            installed_version = metadata.version("sageattention")
        except metadata.PackageNotFoundError:
            installed_version = "not installed"
        core = importlib.import_module("sageattention.core")
        get_arch_versions = getattr(core, "get_cuda_arch_versions", None)
        if not callable(get_arch_versions):
            raise RuntimeError("sageattention.core.get_cuda_arch_versions is missing")
        arch_versions = get_arch_versions()
        if not arch_versions:
            raise RuntimeError("CUDA architecture detection returned no architectures")
        supported_arches = {"sm75", "sm80", "sm86", "sm89", "sm90", "sm120", "sm121"}
        detected_arch = str(arch_versions[0])
        if detected_arch not in supported_arches:
            raise RuntimeError(f"detected GPU architecture {detected_arch} is not supported by this KJNodes patch")
    except Exception as exc:
        python_executable = sys.executable
        raise RuntimeError(
            "MiniMax H3 memory-efficient Sage Attention is enabled, but SageAttention is missing or incompatible.\n"
            f"Detected SageAttention version: {installed_version if 'installed_version' in locals() else 'unknown'}.\n"
            "KJNodes requires SageAttention 2.2.0 or newer with CUDA architecture detection.\n\n"
            "Install it with ComfyUI's embedded Python, then restart ComfyUI:\n"
            f'\"{python_executable}\" -m pip install sageattention==2.2.0 --no-build-isolation\n\n'
            "To continue without this optimization, turn off the MiniMax H3 memory-efficient Sage Attention checkbox.\n"
            f"Technical detail: {exc}"
        ) from exc


def _patch_minimax_h3_advanced_settings(prompt, payload):
    sampler_id = _api_node_id_by_class(prompt, "KSamplerSelect", fallback="123")
    scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
    loader_id = _api_node_id_by_class(prompt, "DiffusionModelLoaderKJ", fallback="141")
    easy_cache_id = _optional_api_node_id_by_class(prompt, "EasyCache", fallback_ids=("174",))

    sampler_name = str(payload.get("sampler_name") or "res_multistep").strip() or "res_multistep"
    scheduler = str(payload.get("scheduler") or "simple").strip() or "simple"
    steps = _int_payload(payload, "steps", 20, 1, 1000)
    denoise = _float_payload(payload, "denoise", 1.0, 0.0, 1.0)
    easy_cache_bypass = _bool_payload(payload, "easy_cache_bypass", False)
    easy_cache_reuse_threshold = _float_payload(payload, "easy_cache_reuse_threshold", 0.3, 0.0, 1.0)
    easy_cache_start_percent = _float_payload(payload, "easy_cache_start_percent", 0.2, 0.0, 1.0)
    easy_cache_end_percent = _float_payload(payload, "easy_cache_end_percent", 0.9, 0.0, 1.0)
    easy_cache_verbose = _bool_payload(payload, "easy_cache_verbose", False)
    sage_attention = str(payload.get("sage_attention") or "auto").strip()
    if sage_attention not in _MINIMAX_H3_SAGE_ATTENTION_MODES:
        sage_attention = "auto"
    enable_fp16_accumulation = _bool_payload(payload, "enable_fp16_accumulation", True)

    _set_api_input(prompt, sampler_id, "sampler_name", sampler_name)
    _set_api_input(prompt, scheduler_id, "scheduler", scheduler)
    _set_api_input(prompt, scheduler_id, "steps", steps)
    _set_api_input(prompt, scheduler_id, "denoise", denoise)
    _set_api_input(prompt, loader_id, "sage_attention", sage_attention)
    _set_api_input(prompt, loader_id, "enable_fp16_accumulation", enable_fp16_accumulation)

    if easy_cache_id:
        _set_api_input(prompt, easy_cache_id, "reuse_threshold", easy_cache_reuse_threshold)
        _set_api_input(prompt, easy_cache_id, "start_percent", easy_cache_start_percent)
        _set_api_input(prompt, easy_cache_id, "end_percent", easy_cache_end_percent)
        _set_api_input(prompt, easy_cache_id, "verbose", easy_cache_verbose)
        if easy_cache_bypass:
            _replace_api_input_refs(prompt, (easy_cache_id, 0), (loader_id, 0))
            prompt.pop(easy_cache_id, None)

    return {
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "steps": steps,
        "denoise": denoise,
        "easy_cache_bypass": easy_cache_bypass,
        "easy_cache_reuse_threshold": easy_cache_reuse_threshold,
        "easy_cache_start_percent": easy_cache_start_percent,
        "easy_cache_end_percent": easy_cache_end_percent,
        "easy_cache_verbose": easy_cache_verbose,
        "sage_attention": sage_attention,
        "enable_fp16_accumulation": enable_fp16_accumulation,
    }


def _patch_minimax_h3_turbo(prompt, payload):
    enabled = _bool_payload(payload, "use_turbo_lora", False)
    if not enabled:
        return {
            "enabled": False,
            "lora_name": "",
            "strength": 0.0,
            "scheduler": "",
            "steps": 0,
        }

    try:
        import nodes as comfy_nodes
        mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    except Exception as exc:
        raise ValueError(
            "MiniMax-H3 Turbo could not inspect ComfyUI custom-node registrations. "
            "Restart ComfyUI after installing ComfyUI-MiniMax-H3-Turbo."
        ) from exc
    required_nodes = ("MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler")
    missing_nodes = [name for name in required_nodes if name not in mappings]
    if missing_nodes:
        raise ValueError(
            "MiniMax-H3 Turbo is enabled, but the required custom nodes are not registered: "
            + ", ".join(missing_nodes)
            + ". Install or update ComfyUI-MiniMax-H3-Turbo, then restart ComfyUI."
        )

    lora_name = str(
        payload.get("turbo_lora_name") or "minimax_h3_turbo_4step_ema_ckpt850.safetensors"
    ).strip()
    if not lora_name:
        raise ValueError("MiniMax-H3 Turbo is enabled, but no Turbo LoRA file is selected.")
    if not _model_choice_exists("loras", lora_name):
        raise ValueError(
            f"MiniMax-H3 Turbo LoRA '{lora_name}' was not found in ComfyUI/models/loras. "
            "Download the LoRA, refresh/restart ComfyUI, and select it in MiniMax Video Settings."
        )
    strength = _float_payload(payload, "turbo_lora_strength", 1.0, -10.0, 10.0)
    turbo_steps = _int_payload(payload, "steps", 4, 1, 1000)

    scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
    guider_id = _api_node_id_by_class(prompt, "BasicGuider", fallback="126")
    sampler_advanced_id = _api_node_id_by_class(prompt, "SamplerCustomAdvanced", fallback="125")
    stock_sampler_id = _optional_api_node_id_by_class(prompt, "KSamplerSelect", fallback_ids=("123",))
    scheduler_inputs = prompt.get(scheduler_id, {}).get("inputs", {})
    model_ref = scheduler_inputs.get("model")
    if not isinstance(model_ref, list) or len(model_ref) != 2:
        raise ValueError("MiniMax-H3 Turbo could not find the current model connection feeding BasicScheduler.")

    turbo_lora_id = "9001"
    while turbo_lora_id in prompt:
        turbo_lora_id = str(int(turbo_lora_id) + 1)
    turbo_sampler_id = str(int(turbo_lora_id) + 1)
    while turbo_sampler_id in prompt:
        turbo_sampler_id = str(int(turbo_sampler_id) + 1)
    prompt[turbo_lora_id] = {
        "class_type": "VRGDG_MiniMaxH3TurboLoRACompat",
        "inputs": {
            "model": list(model_ref),
            "lora_name": lora_name,
            "strength": strength,
        },
    }
    prompt[turbo_sampler_id] = {
        "class_type": "MiniMaxH3TurboSampler",
        "inputs": {},
    }
    _set_api_input(prompt, scheduler_id, "model", [turbo_lora_id, 0])
    _set_api_input(prompt, scheduler_id, "scheduler", "simple")
    _set_api_input(prompt, scheduler_id, "steps", turbo_steps)
    _set_api_input(prompt, guider_id, "model", [turbo_lora_id, 0])
    _set_api_input(prompt, sampler_advanced_id, "sampler", [turbo_sampler_id, 0])
    if stock_sampler_id:
        prompt.pop(stock_sampler_id, None)

    return {
        "enabled": True,
        "lora_name": lora_name,
        "strength": strength,
        "scheduler": "simple",
        "steps": turbo_steps,
        "lora_node": "VRGDG_MiniMaxH3TurboLoRACompat",
        "sampler_node": "MiniMaxH3TurboSampler",
    }


def _patch_minimax_h3_memory_efficient_sage_attention(prompt, payload):
    enabled = _bool_payload(payload, "use_memory_efficient_sage_attention", False)
    if not enabled:
        return {"enabled": False, "node": ""}

    scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
    guider_id = _api_node_id_by_class(prompt, "BasicGuider", fallback="126")
    model_ref = prompt.get(scheduler_id, {}).get("inputs", {}).get("model")
    if not isinstance(model_ref, list) or len(model_ref) != 2:
        raise ValueError("MiniMax H3 memory-efficient Sage Attention patch could not find the current model connection.")

    node_id = "9201"
    while node_id in prompt:
        node_id = str(int(node_id) + 1)
    prompt[node_id] = {
        "class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "inputs": {"model": list(model_ref)},
        "_meta": {"title": "MiniMax H3 Mem Eff Sage Attention Patch"},
    }
    patched_ref = [node_id, 0]
    _set_api_input(prompt, scheduler_id, "model", patched_ref)
    _set_api_input(prompt, guider_id, "model", patched_ref)
    return {"enabled": True, "node": node_id}


def _patch_minimax_h3_loras(prompt, payload):
    enabled = _bool_payload(payload, "use_loras", False) or _bool_payload(payload, "use_custom_loras", False)
    if not enabled:
        return {
            "enabled": False,
            "count": 0,
            "loras": [],
        }
    if _bool_payload(payload, "use_turbo_lora", False):
        raise ValueError("MiniMax normal LoRAs and MiniMax-H3 Turbo LoRA cannot be enabled at the same time.")

    raw_loras = payload.get("loras")
    configured = []
    if isinstance(raw_loras, list):
        for item in raw_loras:
            if not isinstance(item, dict):
                continue
            configured.append({
                "name": _clean_lora_name(item.get("name") or item.get("lora_name") or item.get("loraName") or _NONE_LORA),
                "strength": _float_payload(item, "strength", 1.0, -10.0, 10.0),
            })
    lora_count = _int_payload(payload, "lora_count", len(configured), 0, 4)
    if not configured:
        for slot in range(1, lora_count + 1):
            configured.append({
                "name": _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)),
                "strength": _float_payload(payload, f"lora_{slot}_strength", 1.0, -10.0, 10.0),
            })
    configured = [
        item for item in configured[:lora_count]
        if item["name"] and item["name"] != _NONE_LORA
    ]
    if not configured:
        return {
            "enabled": False,
            "count": 0,
            "loras": [],
        }
    for item in configured:
        if not _model_choice_exists("loras", item["name"]):
            raise ValueError(
                f"MiniMax LoRA '{item['name']}' was not found in ComfyUI/models/loras. "
                "Download the LoRA, refresh/restart ComfyUI, and select it in MiniMax Video Settings."
            )

    scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
    guider_id = _api_node_id_by_class(prompt, "BasicGuider", fallback="126")
    scheduler_inputs = prompt.get(scheduler_id, {}).get("inputs", {})
    model_ref = scheduler_inputs.get("model")
    if not isinstance(model_ref, list) or len(model_ref) != 2:
        raise ValueError("MiniMax LoRA patch could not find the current model connection feeding BasicScheduler.")

    next_id = 9101
    current_ref = list(model_ref)
    applied = []
    for index, item in enumerate(configured, start=1):
        while str(next_id) in prompt:
            next_id += 1
        node_id = str(next_id)
        next_id += 1
        prompt[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(current_ref),
                "lora_name": item["name"],
                "strength_model": item["strength"],
            },
            "_meta": {
                "title": f"MiniMax LoRA {index}",
            },
        }
        current_ref = [node_id, 0]
        applied.append({
            "name": item["name"],
            "strength": item["strength"],
            "node": node_id,
        })

    _set_api_input(prompt, scheduler_id, "model", list(current_ref))
    _set_api_input(prompt, guider_id, "model", list(current_ref))
    return {
        "enabled": True,
        "count": len(applied),
        "loras": applied,
    }


def _build_minimax_h3_api_prompt(payload):
    if _bool_payload(payload, "use_memory_efficient_sage_attention", False):
        _require_minimax_h3_memory_efficient_sage_attention()
    raw_audio_mode = str(payload.get("audio_mode") or payload.get("audioMode") or "input_audio").strip().lower().replace("-", "_").replace(" ", "_")
    audio_mode = "built_in_audio" if raw_audio_mode in {"built_in_audio", "native_audio", "generated_audio"} else "input_audio"
    workflow_template = _minimax_h3_built_in_audio_api_template_path() if audio_mode == "built_in_audio" else _minimax_h3_api_template_path()
    workflow_path, prompt = _load_api_template(workflow_template)
    prompt = copy.deepcopy(prompt)

    video_prompt = str(_first_payload_value(
        payload, "prompt", "video_prompt", "i2v_prompt", "t2v_prompt", default=""
    ) or "").strip()
    if not video_prompt:
        raise ValueError("MiniMax H3 video prompt is empty.")

    audio_path = ""
    if audio_mode == "input_audio":
        audio_text = str(_first_payload_value(
            payload, "audio_path", "source_audio_path", default=""
        ) or "").strip().strip('"')
        if not audio_text:
            raise ValueError("MiniMax H3 source audio path is empty.")
        audio_path = os.path.abspath(audio_text)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"MiniMax H3 source audio was not found: {audio_path}")

    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_text:
        raise ValueError("Project folder is empty.")
    project_folder = os.path.abspath(project_text)
    if not os.path.isdir(project_folder):
        raise FileNotFoundError(f"Project folder was not found: {project_folder}")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)

    timeline_start = _first_payload_value(
        payload, "timeline_start_seconds", "scene_start_seconds", "start", default=0
    )
    timeline_end = _first_payload_value(
        payload, "timeline_end_seconds", "scene_end_seconds", "end", default=None
    )
    if timeline_end is None:
        scene_duration = _first_payload_value(
            payload, "scene_duration_seconds", "scene_duration", "duration", default=None
        )
        if scene_duration is None:
            raise ValueError("MiniMax H3 needs timeline_end_seconds or scene_duration_seconds.")
        try:
            timeline_end = float(timeline_start) + float(scene_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("MiniMax H3 timeline timing must be numeric.") from exc

    source_duration = _first_payload_value(
        payload, "source_duration_seconds", "audio_duration_seconds", default=None
    )
    if source_duration is None and audio_mode == "input_audio":
        source_duration = _probe_media_duration_seconds(audio_path)
    source_start = _first_payload_value(
        payload, "source_start_seconds", "audio_start_seconds", default=None
    )
    warmup_frames = _first_payload_value(
        payload, "warmup_frames", "pre_frames", default=0
    )
    cooldown_frames = _first_payload_value(
        payload, "cooldown_frames", "tail_loss_frames", default=0
    )
    timing = calculate_minimax_h3_timing(
        timeline_start,
        timeline_end,
        warmup_frames,
        cooldown_frames,
        source_start_seconds=source_start,
        source_duration_seconds=source_duration,
    )
    prepared_audio = None
    if audio_mode == "input_audio":
        prepared_audio = _trim_minimax_h3_audio_context(
            audio_path,
            project_folder,
            scene_number,
            timing,
        )

    image_paths = _minimax_h3_image_paths(payload)
    video_references = _minimax_h3_video_references(payload)
    aspect_ratio = str(payload.get("aspect_ratio") or "16:9 (Widescreen)").strip()
    if aspect_ratio not in _MINIMAX_H3_ASPECT_RATIOS:
        raise ValueError(f"Unsupported MiniMax H3 aspect ratio: {aspect_ratio}")
    megapixels = _float_payload(payload, "megapixels", 0.9, 0.1, 16.0)
    diffusion_model_name = str(
        payload.get("diffusion_model_name") or "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    ).strip()
    clip_name = str(
        payload.get("clip_name") or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    ).strip()
    video_vae_name = str(
        payload.get("video_vae_name") or "minimax_h3_video_vae_fp16.safetensors"
    ).strip()
    audio_vae_name = str(
        payload.get("audio_vae_name") or "minimax_h3_audio_vae_fp32.safetensors"
    ).strip()
    if diffusion_model_name.lower().endswith(".gguf"):
        raise ValueError("MiniMax H3 GGUF loading is not enabled yet. Choose a non-GGUF diffusion model.")
    _require_model_choice(("diffusion_models", "unet"), diffusion_model_name, "MiniMax H3 diffusion model")
    _require_model_choice(("text_encoders", "clip"), clip_name, "MiniMax H3 text encoder")
    _require_model_choice("vae", video_vae_name, "MiniMax H3 video VAE")
    _require_model_choice("vae", audio_vae_name, "MiniMax H3 audio VAE")

    seed_value = payload.get("seed", 69)
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        seed = 69
    if seed < 0:
        seed = random.randrange(0, 0xFFFFFFFFFFFFFFFF + 1)
    seed = min(seed, 0xFFFFFFFFFFFFFFFF)

    output_folder, filename_prefix = _minimax_h3_output_location(project_folder, scene_number)
    _set_api_input(prompt, "132", "value", timing.workflow_duration_input_seconds)
    _set_api_input(prompt, "138", "value", video_prompt)
    _set_api_input(prompt, "129", "noise_seed", seed)
    _set_api_input(prompt, "115", "aspect_ratio", aspect_ratio)
    _set_api_input(prompt, "115", "megapixels", megapixels)
    _set_api_input(prompt, "115", "multiple", 32)
    _set_api_input(prompt, "141", "model_name", diffusion_model_name)
    _set_api_input(prompt, "128", "clip_name", clip_name)
    _set_api_input(prompt, "119", "vae_name", video_vae_name)
    _set_api_input(prompt, "120", "vae_name", audio_vae_name)
    if audio_mode == "input_audio":
        _set_api_input(prompt, "171", "audio_file", prepared_audio["audio_path"])
        _set_api_input(prompt, "171", "seek_seconds", 0)
        _set_api_input(prompt, "171", "duration", 0)
    _set_api_input(prompt, "180", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    _set_api_input(prompt, "180", "video_references", json.dumps(video_references, ensure_ascii=False))
    _set_api_input(prompt, "142", "frame_rate", 24)
    _set_api_input(prompt, "142", "filename_prefix", filename_prefix)
    # Keep every aligned H3 frame. VHS trim_to_audio muxes with -shortest while
    # stream-copying H.264, which can discard final video packets before our
    # exact scene trimmer receives them.
    _set_api_input(prompt, "142", "trim_to_audio", False)
    advanced_settings = _patch_minimax_h3_advanced_settings(prompt, payload)
    if _bool_payload(payload, "use_memory_efficient_sage_attention", False):
        _set_api_input(prompt, "141", "sage_attention", "disabled")
        advanced_settings["sage_attention"] = "disabled"
    lora_settings = _patch_minimax_h3_loras(prompt, payload)
    turbo_settings = _patch_minimax_h3_turbo(prompt, payload)
    memory_efficient_sage_attention = _patch_minimax_h3_memory_efficient_sage_attention(prompt, payload)
    if turbo_settings["enabled"]:
        advanced_settings = {
            **advanced_settings,
            "effective_sampler_name": "MiniMaxH3TurboSampler",
            "effective_scheduler": "simple",
            "effective_steps": turbo_settings["steps"],
        }

    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": prompt,
        "used_seed": seed,
        "audio_mode": audio_mode,
        "timing": timing.to_dict(),
        "prepared_audio": prepared_audio,
        "post_render_trim": {
            "start": timing.final_trim_start_seconds,
            "duration": timing.final_trim_duration_seconds,
        },
        "reference_inputs": {
            "image_count": len(image_paths),
            "video_count": len(video_references),
            "video_audio_count": sum(1 for item in video_references if item.get("use_audio")),
        },
        "model_settings": {
            "diffusion_model_name": diffusion_model_name,
            "clip_name": clip_name,
            "video_vae_name": video_vae_name,
            "audio_vae_name": audio_vae_name,
        },
        "advanced_settings": advanced_settings,
        "lora_settings": lora_settings,
        "turbo_settings": turbo_settings,
        "memory_efficient_sage_attention": memory_efficient_sage_attention,
    }


def _build_minimax_h3_2pass_api_prompt(payload):
    """Build the cleaned external-audio MiniMax H3 two-pass API prompt.

    This intentionally has its own adapter instead of reusing the one-pass
    node IDs or mutating the existing MiniMax template path.
    """
    workflow_path, prompt = _load_api_template(_minimax_h3_2pass_api_template_path())
    prompt = copy.deepcopy(prompt)
    video_prompt = str(_first_payload_value(payload, "prompt", "video_prompt", default="") or "").strip()
    if not video_prompt:
        raise ValueError("MiniMax H3 two-pass video prompt is empty.")

    audio_path = str(_first_payload_value(payload, "audio_path", "source_audio_path", default="") or "").strip().strip('"')
    if not audio_path:
        raise ValueError("MiniMax H3 two-pass source audio path is empty.")
    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"MiniMax H3 two-pass source audio was not found: {audio_path}")

    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_text:
        raise ValueError("Project folder is empty.")
    project_folder = os.path.abspath(project_text)
    if not os.path.isdir(project_folder):
        raise FileNotFoundError(f"Project folder was not found: {project_folder}")

    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    timeline_start = _first_payload_value(payload, "timeline_start_seconds", "scene_start_seconds", "start", default=0)
    timeline_end = _first_payload_value(payload, "timeline_end_seconds", "scene_end_seconds", "end", default=None)
    if timeline_end is None:
        duration = _first_payload_value(payload, "scene_duration_seconds", "scene_duration", "duration", default=None)
        if duration is None:
            raise ValueError("MiniMax H3 two-pass needs timeline_end_seconds or scene_duration_seconds.")
        timeline_end = float(timeline_start) + float(duration)

    source_duration = _first_payload_value(payload, "source_duration_seconds", "audio_duration_seconds", default=None)
    if source_duration is None:
        source_duration = _probe_media_duration_seconds(audio_path)
    source_start = _first_payload_value(payload, "source_start_seconds", "audio_start_seconds", default=None)
    timing = calculate_minimax_h3_timing(
        timeline_start,
        timeline_end,
        _first_payload_value(payload, "warmup_frames", "pre_frames", default=0),
        _first_payload_value(payload, "cooldown_frames", "tail_loss_frames", default=0),
        source_start_seconds=source_start,
        source_duration_seconds=source_duration,
    )
    prepared_audio = _trim_minimax_h3_audio_context(audio_path, project_folder, scene_number, timing)

    image_paths = _minimax_h3_image_paths(payload)
    video_references = _minimax_h3_video_references(payload)
    diffusion_model_name = str(payload.get("diffusion_model_name") or "minimax_h3_ref2va_pruned_int8_convrot.safetensors").strip()
    clip_name = str(payload.get("clip_name") or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors").strip()
    video_vae_name = str(payload.get("video_vae_name") or "minimax_h3_video_vae_fp16.safetensors").strip()
    audio_vae_name = str(payload.get("audio_vae_name") or "minimax_h3_audio_vae_fp32.safetensors").strip()
    latent_upscaler_name = str(payload.get("latent_upscaler_name") or "minimax_h3_latent_upscaler_3d_bf16.safetensors").strip()
    _require_model_choice(("diffusion_models", "unet"), diffusion_model_name, "MiniMax H3 two-pass diffusion model")
    _require_model_choice(("text_encoders", "clip"), clip_name, "MiniMax H3 two-pass text encoder")
    _require_model_choice("vae", video_vae_name, "MiniMax H3 two-pass video VAE")
    _require_model_choice("vae", audio_vae_name, "MiniMax H3 two-pass audio VAE")
    _require_model_choice("latent_upscale_models", latent_upscaler_name, "MiniMax H3 learned latent upscaler")

    seed = _int_payload(payload, "seed", 69, 0, 0xFFFFFFFFFFFFFFFF)
    pass1_seed = _int_payload(payload, "pass1_seed", seed, 0, 0xFFFFFFFFFFFFFFFF)
    pass2_seed = _int_payload(payload, "pass2_seed", seed, 0, 0xFFFFFFFFFFFFFFFF)
    final_width = _int_payload(payload, "final_width", 1920, 64, 16384)
    final_height = _int_payload(payload, "final_height", 1080, 64, 16384)
    latent_scale = _float_payload(payload, "latent_upscale_scale", 2.0, 1.0, 8.0)

    _set_api_input(prompt, "138", "value", video_prompt)
    _set_api_input(prompt, "132", "value", float(timing.workflow_duration_input_seconds))
    _set_api_input(prompt, "171", "audio_file", prepared_audio["audio_path"])
    _set_api_input(prompt, "171", "seek_seconds", 0)
    _set_api_input(prompt, "171", "duration", 0)
    _set_api_input(prompt, "180", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    _set_api_input(prompt, "180", "video_references", json.dumps(video_references, ensure_ascii=False))
    ref_image_size = str(payload.get("ref_image_size") or "max").strip().lower()
    _set_api_input(prompt, "136", "ref_image_size", ref_image_size if ref_image_size in {"match", "max"} else "max")

    _set_api_input(prompt, "115", "value", final_width)
    _set_api_input(prompt, "184", "value", final_height)
    _set_api_input(prompt, "185", "value", latent_scale)
    _set_api_input(prompt, "188", "model_name", latent_upscaler_name)
    _set_api_input(prompt, "188", "device", str(payload.get("latent_upscaler_device") or "cuda"))
    _set_api_input(prompt, "188", "precision", str(payload.get("latent_upscaler_precision") or "bf16"))

    _set_api_input(prompt, "129", "noise_seed", pass1_seed)
    _set_api_input(prompt, "211", "noise_seed", pass2_seed)
    _set_api_input(prompt, "123", "sampler_name", str(payload.get("pass1_sampler_name") or "res_multistep").strip() or "res_multistep")
    _set_api_input(prompt, "210", "sampler_name", str(payload.get("pass2_sampler_name") or "res_multistep").strip() or "res_multistep")
    _set_api_input(prompt, "124", "scheduler", str(payload.get("pass1_scheduler") or "simple").strip() or "simple")
    _set_api_input(prompt, "124", "steps", _int_payload(payload, "pass1_steps", 20, 1, 1000))
    _set_api_input(prompt, "124", "denoise", _float_payload(payload, "pass1_denoise", 1.0, 0.0, 1.0))
    _set_api_input(prompt, "192", "scheduler", str(payload.get("pass2_scheduler") or "simple").strip() or "simple")
    _set_api_input(prompt, "190", "value", _int_payload(payload, "pass2_steps", 5, 1, 1000))
    _set_api_input(prompt, "191", "value", _float_payload(payload, "pass2_denoise", 0.2, 0.0, 1.0))

    _set_api_input(prompt, "141", "model_name", diffusion_model_name)
    _set_api_input(prompt, "141", "sage_attention", str(payload.get("sage_attention") or "auto"))
    _set_api_input(prompt, "141", "enable_fp16_accumulation", _bool_payload(payload, "enable_fp16_accumulation", True))
    _set_api_input(prompt, "128", "clip_name", clip_name)
    _set_api_input(prompt, "119", "vae_name", video_vae_name)
    _set_api_input(prompt, "120", "vae_name", audio_vae_name)

    turbo_lora_name = _clean_lora_name(payload.get("two_pass_lora_name", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"))
    if turbo_lora_name == _NONE_LORA:
        raise ValueError("Select the MiniMax H3 two-pass Turbo LoRA; it is required by this fast workflow.")
    _require_model_choice("loras", turbo_lora_name, "MiniMax H3 two-pass Turbo LoRA")
    _set_api_input(prompt, "207", "lora_name", turbo_lora_name)
    _set_api_input(prompt, "207", "strength_model", _float_payload(payload, "two_pass_lora_strength", 1.0, -10.0, 10.0))

    use_te_speed = _bool_payload(payload, "two_pass_use_te_speed", True)
    # Bypassing is done by routing the LoRA loader directly from the diffusion
    # model, so the TE-Speed node is not part of the executed dependency graph.
    _set_api_input(prompt, "207", "model", ["208", 0] if use_te_speed else ["141", 0])
    _set_api_input(prompt, "208", "model", ["141", 0])
    _set_api_input(prompt, "208", "processing_control_value", _float_payload(payload, "te_speed_processing_control", 0.07, 0.0, 1.0))
    _set_api_input(prompt, "208", "processing_percent_1", _float_payload(payload, "te_speed_start_percent", 0.1, 0.0, 1.0))
    _set_api_input(prompt, "208", "processing_percent_2", _float_payload(payload, "te_speed_end_percent", 0.9, 0.0, 1.0))
    _set_api_input(prompt, "208", "mcs", _int_payload(payload, "te_speed_mcs", 2, 1, 64))
    _set_api_input(prompt, "208", "cache_depth", _float_payload(payload, "te_speed_cache_depth", 0.75, 0.0, 1.0))
    _set_api_input(prompt, "208", "device", str(payload.get("te_speed_device") or "auto"))
    pass1_model = ["208", 0] if use_te_speed else ["141", 0]
    pass2_model = ["207", 0]

    extra_loras = []
    if _bool_payload(payload, "use_loras", False) or _bool_payload(payload, "use_custom_loras", False):
        raw_loras = payload.get("loras") if isinstance(payload.get("loras"), list) else []
        lora_count = _int_payload(payload, "lora_count", len(raw_loras), 0, 4)
        for item in raw_loras[:lora_count]:
            if not isinstance(item, dict):
                continue
            name = _clean_lora_name(item.get("name") or item.get("lora_name") or item.get("loraName") or _NONE_LORA)
            if not name or name == _NONE_LORA:
                continue
            if not _model_choice_exists("loras", name):
                raise ValueError(
                    f"MiniMax extra LoRA '{name}' was not found in ComfyUI/models/loras. "
                    "Download it, refresh/restart ComfyUI, and select it in MiniMax Video Settings."
                )
            apply_to = str(item.get("apply_to") or item.get("applyTo") or "both").strip().lower()
            if apply_to not in {"both", "pass1", "pass2"}:
                apply_to = "both"
            extra_loras.append({
                "name": name,
                "strength": _float_payload(item, "strength", 1.0, -10.0, 10.0),
                "apply_to": apply_to,
            })

    next_lora_node_id = 9201
    applied_extra_loras = []

    def add_extra_lora(model_ref, item, target, index):
        nonlocal next_lora_node_id
        while str(next_lora_node_id) in prompt:
            next_lora_node_id += 1
        node_id = str(next_lora_node_id)
        next_lora_node_id += 1
        prompt[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(model_ref),
                "lora_name": item["name"],
                "strength_model": item["strength"],
            },
            "_meta": {"title": f"Extra MiniMax LoRA {index} - {target}"},
        }
        applied_extra_loras.append({**item, "target": target, "node": node_id})
        return [node_id, 0]

    for index, item in enumerate(extra_loras, start=1):
        if item["apply_to"] in {"both", "pass1"}:
            pass1_model = add_extra_lora(pass1_model, item, "pass1", index)
        if item["apply_to"] in {"both", "pass2"}:
            pass2_model = add_extra_lora(pass2_model, item, "pass2", index)

    for node_id in ("124", "126"):
        _set_api_input(prompt, node_id, "model", list(pass1_model))
    for node_id in ("192", "193"):
        _set_api_input(prompt, node_id, "model", list(pass2_model))

    _set_api_input(prompt, "183", "upscale_method", str(payload.get("final_resize_method") or "nvidia_rtx_vsr"))
    _set_api_input(prompt, "142", "crf", _int_payload(payload, "output_crf", 19, 0, 100))
    output_folder, filename_prefix = _minimax_h3_output_location(project_folder, scene_number)
    _set_api_input(prompt, "142", "filename_prefix", f"{filename_prefix}_stage2")
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": prompt,
        "used_seed": seed,
        "audio_mode": "input_audio",
        "timing": timing.to_dict(),
        "prepared_audio": prepared_audio,
        "post_render_trim": {"start": timing.final_trim_start_seconds, "duration": timing.final_trim_duration_seconds},
        "reference_inputs": {"image_count": len(image_paths), "video_count": len(video_references)},
        "two_pass": {
            "pass1_steps": prompt["124"]["inputs"]["steps"],
            "pass2_steps": prompt["190"]["inputs"]["value"],
            "final_width": final_width,
            "final_height": final_height,
            "latent_upscale_scale": latent_scale,
            "te_speed_enabled": use_te_speed,
            "extra_loras": applied_extra_loras,
        },
    }


def _remap_api_prompt_references(prompt, prefix):
    """Copy an API prompt under collision-free IDs and rewrite socket links."""
    mapping = {str(node_id): f"{prefix}{str(node_id).replace(':', '_')}" for node_id in prompt}

    def remap(value):
        if isinstance(value, list):
            if len(value) == 2 and str(value[0]) in mapping and isinstance(value[1], int):
                return [mapping[str(value[0])], value[1]]
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value

    return {mapping[str(node_id)]: remap(copy.deepcopy(node)) for node_id, node in prompt.items()}, mapping


def _prune_api_prompt_to_roots(prompt, roots):
    """Keep only API nodes required to produce the requested output roots."""
    required = set()

    def visit(node_id):
        node_id = str(node_id)
        if node_id in required or node_id not in prompt:
            return
        required.add(node_id)
        node = prompt.get(node_id) or {}
        inputs = node.get("inputs") or {}
        for value in inputs.values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) and str(value[0]) in prompt:
                visit(value[0])

    for root in roots:
        visit(root)
    return {node_id: node for node_id, node in prompt.items() if node_id in required}


def _build_minimax_h3_3pass_api_prompt(payload):
    """Build the experimental external-audio MiniMax H3 three-pass prompt."""
    workflow_path, prompt = _load_api_template(_minimax_h3_3pass_api_template_path())
    prompt = copy.deepcopy(prompt)
    video_prompt = str(_first_payload_value(payload, "prompt", "video_prompt", default="") or "").strip()
    if not video_prompt:
        raise ValueError("MiniMax H3 three-pass video prompt is empty.")
    audio_path = str(_first_payload_value(payload, "audio_path", "source_audio_path", default="") or "").strip().strip('"')
    if not audio_path:
        raise ValueError("MiniMax H3 three-pass source audio path is empty.")
    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"MiniMax H3 three-pass source audio was not found: {audio_path}")
    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_text:
        raise ValueError("Project folder is empty.")
    project_folder = os.path.abspath(project_text)
    if not os.path.isdir(project_folder):
        raise FileNotFoundError(f"Project folder was not found: {project_folder}")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    timeline_start = _first_payload_value(payload, "timeline_start_seconds", "scene_start_seconds", "start", default=0)
    timeline_end = _first_payload_value(payload, "timeline_end_seconds", "scene_end_seconds", "end", default=None)
    if timeline_end is None:
        duration = _first_payload_value(payload, "scene_duration_seconds", "scene_duration", "duration", default=None)
        if duration is None:
            raise ValueError("MiniMax H3 three-pass needs timeline_end_seconds or scene_duration_seconds.")
        timeline_end = float(timeline_start) + float(duration)
    source_duration = _first_payload_value(payload, "source_duration_seconds", "audio_duration_seconds", default=None)
    if source_duration is None:
        source_duration = _probe_media_duration_seconds(audio_path)
    source_start = _first_payload_value(payload, "source_start_seconds", "audio_start_seconds", default=None)
    timing = calculate_minimax_h3_timing(
        timeline_start,
        timeline_end,
        _first_payload_value(payload, "warmup_frames", "pre_frames", default=0),
        _first_payload_value(payload, "cooldown_frames", "tail_loss_frames", default=0),
        source_start_seconds=source_start,
        source_duration_seconds=source_duration,
    )
    prepared_audio = _trim_minimax_h3_audio_context(audio_path, project_folder, scene_number, timing)
    image_paths = _minimax_h3_image_paths(payload)
    video_references = _minimax_h3_video_references(payload)
    diffusion_model_name = str(payload.get("diffusion_model_name") or "minimax_h3_ref2va_pruned_int8_convrot.safetensors").strip()
    clip_name = str(payload.get("clip_name") or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors").strip()
    video_vae_name = str(payload.get("video_vae_name") or "minimax_h3_video_vae_fp16.safetensors").strip()
    audio_vae_name = str(payload.get("audio_vae_name") or "minimax_h3_audio_vae_fp32.safetensors").strip()
    _require_model_choice(("diffusion_models", "unet"), diffusion_model_name, "MiniMax H3 three-pass diffusion model")
    _require_model_choice(("text_encoders", "clip"), clip_name, "MiniMax H3 three-pass text encoder")
    _require_model_choice("vae", video_vae_name, "MiniMax H3 three-pass video VAE")
    _require_model_choice("vae", audio_vae_name, "MiniMax H3 three-pass audio VAE")

    seed = _int_payload(payload, "seed", 69, 0, 0xFFFFFFFFFFFFFFFF)
    aspect_ratio = str(payload.get("aspect_ratio") or "16:9 (Widescreen)").strip()
    if aspect_ratio not in _MINIMAX_H3_ASPECT_RATIOS:
        raise ValueError(f"Unsupported MiniMax H3 three-pass aspect ratio: {aspect_ratio}")
    _set_api_input(prompt, "329", "text", video_prompt)
    _set_api_input(prompt, "84", "value", float(timing.workflow_duration_input_seconds))
    _set_api_input(prompt, "9001", "audio_file", prepared_audio["audio_path"])
    _set_api_input(prompt, "9001", "seek_seconds", 0)
    _set_api_input(prompt, "9001", "duration", 0)
    _set_api_input(prompt, "9000", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    _set_api_input(prompt, "9000", "video_references", json.dumps(video_references, ensure_ascii=False))
    ref_image_size = str(payload.get("ref_image_size") or "max").strip().lower()
    if ref_image_size not in {"match", "max"}:
        ref_image_size = "max"
    _set_api_input(prompt, "108", "ref_image_size", ref_image_size)
    _set_api_input(prompt, "330", "model_name", diffusion_model_name)
    _set_api_input(prompt, "4", "clip_name", clip_name)
    _set_api_input(prompt, "5", "vae_name", video_vae_name)
    _set_api_input(prompt, "6", "vae_name", audio_vae_name)
    pass_specs = [
        (1, "105", "248", "249", "243", "328", "246", 0.4, 20, 1.0, True),
        (2, "297", "290", "289", "300", "187", "294", 1.0, 5, 0.2, False),
        (3, "334", "344", "341", "345", "187", "340", 2.0, 5, 0.2, False),
    ]
    for pass_number, resolution_id, scheduler_id, sampler_id, noise_id, default_model_id, guider_id, default_mp, default_steps, default_denoise, default_speed in pass_specs:
        prefix = f"three_pass_pass{pass_number}_"
        _set_api_input(prompt, resolution_id, "aspect_ratio", str(payload.get(f"{prefix}aspect_ratio") or aspect_ratio))
        _set_api_input(prompt, resolution_id, "megapixels", _float_payload(payload, f"{prefix}megapixels", default_mp, 0.1, 16.0))
        _set_api_input(prompt, scheduler_id, "steps", _int_payload(payload, f"{prefix}steps", default_steps, 1, 1000))
        _set_api_input(prompt, scheduler_id, "denoise", _float_payload(payload, f"{prefix}denoise", default_denoise, 0.0, 1.0))
        _set_api_input(prompt, scheduler_id, "scheduler", str(payload.get(f"{prefix}scheduler") or "beta").strip() or "beta")
        _set_api_input(prompt, sampler_id, "sampler_name", str(payload.get(f"{prefix}sampler") or "euler").strip() or "euler")
        _set_api_input(prompt, noise_id, "noise_seed", _int_payload(payload, f"{prefix}seed", seed, 0, 0xFFFFFFFFFFFFFFFF))

    pass1_speed = _bool_payload(payload, "three_pass_pass1_te_speed", True)
    pass2_speed = _bool_payload(payload, "three_pass_pass2_te_speed", False)
    pass3_speed = _bool_payload(payload, "three_pass_pass3_te_speed", False)
    lora_name = _clean_lora_name(payload.get("three_pass_lightx_lora_name", "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"))
    if lora_name == _NONE_LORA:
        raise ValueError("A valid Multi-Pass LightX2V LoRA must be selected.")
    _require_model_choice("loras", lora_name, "MiniMax H3 three-pass LightX2V LoRA")
    lora_strength = _float_payload(payload, "three_pass_lightx_lora_strength", 0.5, -10.0, 10.0)
    _set_api_input(prompt, "187", "lora_name", lora_name)
    _set_api_input(prompt, "187", "strength_model", lora_strength)
    pass1_model_ref = ["328", 0] if pass1_speed else ["330", 0]
    _set_api_input(prompt, "328", "model", ["330", 0])
    _set_api_input(prompt, "248", "model", list(pass1_model_ref))
    _set_api_input(prompt, "246", "model", list(pass1_model_ref))

    pass_model_refs = []
    for pass_number, enabled, scheduler_id, guider_id, speed_node_id, lora_node_id in (
        (2, pass2_speed, "290", "294", "9202", "9204"),
        (3, pass3_speed, "344", "340", "9203", "9205"),
    ):
        pass_base_ref = [speed_node_id, 0] if enabled else ["330", 0]
        if enabled:
            prompt[speed_node_id] = {
                "class_type": "TESpeedMiniMaxH3",
                "inputs": {
                    "processing_control_value": 0.07,
                    "processing_percent_1": 0.1,
                    "processing_percent_2": 0.9,
                    "mcs": 2,
                    "device": "auto",
                    "cache_depth": 0.75,
                    "model": ["330", 0],
                },
                "_meta": {"title": f"TE-Speed-MiniMaxH3 Pass {pass_number}"},
            }
        prompt[lora_node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(pass_base_ref),
                "lora_name": lora_name,
                "strength_model": lora_strength,
            },
            "_meta": {"title": f"MiniMax H3 Pass {pass_number} LoRA"},
        }
        pass_model_refs.append((scheduler_id, guider_id, [lora_node_id, 0]))
    for scheduler_id, guider_id, model_ref in pass_model_refs:
        _set_api_input(prompt, scheduler_id, "model", list(model_ref))
        _set_api_input(prompt, guider_id, "model", list(model_ref))
    output_folder, filename_prefix = _minimax_h3_output_location(project_folder, scene_number)
    _set_api_input(prompt, "91", "filename_prefix", f"{filename_prefix}_stage1")
    _set_api_input(prompt, "299", "filename_prefix", f"{filename_prefix}_stage2")
    _set_api_input(prompt, "353", "filename_prefix", f"{filename_prefix}_stage3")
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": prompt,
        "used_seed": seed,
        "audio_mode": "input_audio",
        "timing": timing.to_dict(),
        "prepared_audio": prepared_audio,
        "post_render_trim": {"start": timing.final_trim_start_seconds, "duration": timing.final_trim_duration_seconds},
        "reference_inputs": {"image_count": len(image_paths), "video_count": len(video_references)},
        "three_pass": {"pass1_steps": prompt["248"]["inputs"]["steps"], "pass2_steps": prompt["290"]["inputs"]["steps"], "pass3_steps": prompt["344"]["inputs"]["steps"]},
        "te_speed": {"pass1": pass1_speed, "pass2": pass2_speed, "pass3": pass3_speed},
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


def _patch_transcribe_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    srt_path = os.path.abspath(str(payload.get("srt_path", "") or "").strip().strip('"'))
    if not audio_path:
        raise ValueError("Audio file path is empty.")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    if not srt_path:
        raise ValueError("SRT file path is empty.")
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file was not found: {srt_path}")

    extractor_id = _api_node_id_by_class(prompt, "VRGDG_ManualLyricsExtractor_SRT_Advanced", "960")
    stems_id = _api_node_id_by_class(prompt, "VRGDG_GetStems", "28:114")
    _set_api_input(prompt, stems_id, "audio_file_path", audio_path)
    _set_api_input(prompt, extractor_id, "srt_path", srt_path)
    _set_api_input(prompt, extractor_id, "reference_lyrics", str(payload.get("reference_lyrics", "") or ""))
    _set_api_input(prompt, extractor_id, "language", str(payload.get("language", "") or "english"))
    _set_api_input(prompt, extractor_id, "strict_reference_text", bool(payload.get("strict_reference_text", True)))
    _set_api_input(prompt, extractor_id, "fill_aggressiveness", _int_payload(payload, "fill_aggressiveness", 1, minimum=0, maximum=3))
    _set_api_input(prompt, extractor_id, "preserve_nonvocal_segments", bool(payload.get("preserve_nonvocal_segments", True)))
    _set_api_input(prompt, extractor_id, "alignment_min_words", _int_payload(payload, "alignment_min_words", 1, minimum=1, maximum=10))
    model_name = str(payload.get("model_name", "") or "large-v3").strip()
    if model_name:
        _set_api_input(prompt, extractor_id, "model_name", model_name)
    return prompt


def _build_transcribe_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_transcribe_api_template_path())
    patched_prompt = _patch_transcribe_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
    }


def _patch_timestamped_transcribe_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not audio_path:
        raise ValueError("Audio file path is empty.")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")

    extractor_id = _api_node_id_by_class(prompt, "VRGDG_TimestampedLyricsExtractor", "962")
    stems_id = _api_node_id_by_class(prompt, "VRGDG_GetStems", "28:114")
    _set_api_input(prompt, stems_id, "audio_file_path", audio_path)
    _set_api_input(prompt, extractor_id, "reference_lyrics", str(payload.get("reference_lyrics", "") or ""))
    _set_api_input(prompt, extractor_id, "language", str(payload.get("language", "") or "english"))
    segment_mode = str(payload.get("segment_mode", "") or "reference_lines").strip()
    if segment_mode not in {"whisper_chunks", "reference_lines", "exact_reference_lines", "reference_stanzas", "reference_scene_words"}:
        segment_mode = "reference_lines"
    _set_api_input(prompt, extractor_id, "segment_mode", segment_mode)
    _set_api_input(prompt, extractor_id, "include_instrumental_gaps", _bool_payload(payload, "include_instrumental_gaps", True))
    _set_api_input(prompt, extractor_id, "instrumental_text", str(payload.get("instrumental_text", "") or "[instrumental]"))
    _set_api_input(prompt, extractor_id, "min_gap_seconds", _float_payload(payload, "min_gap_seconds", 1.0, minimum=0.0, maximum=30.0))
    _set_api_input(prompt, extractor_id, "min_scene_seconds", _float_payload(payload, "min_scene_seconds", 1.0, minimum=1.0, maximum=30.0))
    _set_api_input(prompt, extractor_id, "max_scene_seconds", _float_payload(payload, "max_scene_seconds", 8.0, minimum=1.0, maximum=60.0))
    _set_api_input(prompt, extractor_id, "vocal_tail_padding_seconds", _float_payload(payload, "vocal_tail_padding_seconds", 0.6, minimum=0.0, maximum=3.0))
    model_name = str(payload.get("model_name", "") or "large-v3").strip()
    if model_name:
        _set_api_input(prompt, extractor_id, "model_name", model_name)
    return prompt


def _build_timestamped_transcribe_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_timestamped_transcribe_api_template_path())
    patched_prompt = _patch_timestamped_transcribe_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
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


def _ffprobe_path_for(ffmpeg_path):
    if not ffmpeg_path or ffmpeg_path == "ffmpeg":
        return "ffprobe"
    folder = os.path.dirname(os.path.abspath(ffmpeg_path))
    exe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    candidate = os.path.join(folder, exe_name)
    return candidate if os.path.isfile(candidate) else "ffprobe"


def _probe_video_size(video_path, ffmpeg_path=None):
    ffprobe_path = _ffprobe_path_for(ffmpeg_path)
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=True)
    text = (result.stdout or "").strip().splitlines()[0]
    width_text, height_text = text.lower().split("x", 1)
    return int(width_text), int(height_text)


def _normalize_video_canvas(ffmpeg_path, source_path, target_path, width, height):
    width = int(width or 0)
    height = int(height or 0)
    if width <= 0 or height <= 0:
        return False
    try:
        source_width, source_height = _probe_video_size(source_path, ffmpeg_path)
        if source_width == width and source_height == height:
            return False
    except Exception as exc:
        print(f"[VRGDG WorkflowRunner] Could not probe video size before final canvas normalization: {exc}")

    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        source_path,
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        target_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=True)
    return True


def _scene_video_thumbnail_path(video_path):
    video_path = os.path.abspath(str(video_path or "").strip().strip('"'))
    root, _ext = os.path.splitext(video_path)
    video_name = os.path.basename(root)
    current = os.path.dirname(video_path)
    while current and current != os.path.dirname(current):
        if os.path.basename(current).lower() in {"rendered_scene_videos", "rendered_scene_videos_backup"}:
            project_folder = os.path.dirname(current)
            return os.path.join(project_folder, "scene_video_thumbnails", f"{video_name}.jpg")
        current = os.path.dirname(current)
    return f"{root}.jpg"


def _legacy_scene_video_thumbnail_path(video_path):
    root, _ext = os.path.splitext(os.path.abspath(str(video_path or "").strip().strip('"')))
    return f"{root}.jpg"


def _create_scene_video_thumbnail(video_path, thumbnail_path=None):
    video_path = os.path.abspath(str(video_path or "").strip().strip('"'))
    if not os.path.isfile(video_path):
        return ""
    thumbnail_path = os.path.abspath(str(thumbnail_path or _scene_video_thumbnail_path(video_path)).strip().strip('"'))
    os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
    ffmpeg_path = _find_ffmpeg_path()

    def _run_extract(timestamp):
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            "-q:v",
            "3",
            thumbnail_path,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    result = _run_extract(0.5)
    if result.returncode != 0 or not os.path.isfile(thumbnail_path):
        result = _run_extract(0)
    if result.returncode != 0 or not os.path.isfile(thumbnail_path):
        error_text = (result.stderr or result.stdout or "ffmpeg could not extract a thumbnail.").strip()
        print(f"[VRGDG WorkflowRunner] Could not create scene video thumbnail for '{video_path}': {error_text}")
        return ""
    return thumbnail_path


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
    target_thumbnail_path = _scene_video_thumbnail_path(target_path)
    legacy_target_thumbnail_path = _legacy_scene_video_thumbnail_path(target_path)
    backup_path = ""
    backup_thumbnail_path = ""
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
                if os.path.exists(target_thumbnail_path):
                    backup_thumbnail_path = _scene_video_thumbnail_path(backup_path)
                    _retry_file_op(
                        lambda: shutil.move(target_thumbnail_path, backup_thumbnail_path),
                        f"Backing up existing scene video thumbnail '{target_thumbnail_path}'",
                    )
                if os.path.exists(legacy_target_thumbnail_path):
                    _retry_file_op(
                        lambda: os.remove(legacy_target_thumbnail_path),
                        f"Removing legacy scene video thumbnail '{legacy_target_thumbnail_path}'",
                    )
            else:
                _retry_file_op(
                    lambda: os.remove(target_path),
                    f"Removing existing scene video '{target_path}'",
                )
                if os.path.exists(target_thumbnail_path):
                    try:
                        _retry_file_op(
                            lambda: os.remove(target_thumbnail_path),
                            f"Removing existing scene video thumbnail '{target_thumbnail_path}'",
                        )
                    except Exception as exc:
                        print(f"[VRGDG WorkflowRunner] Could not remove old scene video thumbnail '{target_thumbnail_path}': {exc}")
                if os.path.exists(legacy_target_thumbnail_path):
                    try:
                        _retry_file_op(
                            lambda: os.remove(legacy_target_thumbnail_path),
                            f"Removing legacy scene video thumbnail '{legacy_target_thumbnail_path}'",
                        )
                    except Exception as exc:
                        print(f"[VRGDG WorkflowRunner] Could not remove legacy scene video thumbnail '{legacy_target_thumbnail_path}': {exc}")
        _replace_file_with_retry(source_path, target_path)

    if os.path.exists(legacy_target_thumbnail_path):
        try:
            _retry_file_op(
                lambda: os.remove(legacy_target_thumbnail_path),
                f"Removing legacy scene video thumbnail '{legacy_target_thumbnail_path}'",
            )
        except Exception as exc:
            print(f"[VRGDG WorkflowRunner] Could not remove legacy scene video thumbnail '{legacy_target_thumbnail_path}': {exc}")

    thumbnail_path = _create_scene_video_thumbnail(target_path, target_thumbnail_path)
    removed_files = []
    removed_folder = ""
    removed_scratch_folders = []

    return {
        "video_path": target_path,
        "thumbnail_path": thumbnail_path,
        "video_folder": target_dir,
        "backup_path": backup_path,
        "backup_thumbnail_path": backup_thumbnail_path,
        "existing_action": existing_action,
        "source_path": source_path,
        "removed_files": removed_files,
        "removed_folder": removed_folder,
        "removed_scratch_folders": removed_scratch_folders,
    }


def _trim_scene_video(payload):
    source_path = os.path.abspath(str(payload.get("source_path", "") or "").strip().strip('"'))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Scene video was not found: {source_path}")
    if os.path.splitext(source_path)[1].lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise ValueError(f"Scene media is not a supported video file: {source_path}")
    project_folder, target_dir = _safe_project_subfolder(payload.get("project_folder", ""), "rendered_scene_videos")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    start = max(0.0, float(payload.get("start", 0) or 0))
    duration = max(0.05, float(payload.get("duration", 0) or 0))
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(payload.get("label", "trim") or "trim").strip().lower()).strip("_") or "trim"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    audio_suffix = "-audio" if _bool_payload(payload, "mark_as_audio_video", False) else ""
    target_path = os.path.join(target_dir, f"video_{scene_number:04d}-{label}_{stamp}{audio_suffix}.mp4")
    index = 2
    while os.path.exists(target_path):
        target_path = os.path.join(target_dir, f"video_{scene_number:04d}-{label}_{stamp}_{index:02d}{audio_suffix}.mp4")
        index += 1

    ffmpeg_path = _find_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        source_path,
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        target_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(target_path):
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed to trim scene video.").strip())
    thumbnail_path = _create_scene_video_thumbnail(target_path)
    return {
        "video_path": target_path,
        "thumbnail_path": thumbnail_path,
        "video_folder": target_dir,
        "source_path": source_path,
        "start": start,
        "duration": duration,
    }


def _apply_scene_start_color_match(payload):
    """Match a new clip's opening color to the prior clip, then fade the correction out."""
    from PIL import Image, ImageStat

    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    video_path = os.path.abspath(str(payload.get("video_path", "") or "").strip().strip('"'))
    reference_video_path = os.path.abspath(str(payload.get("reference_video_path", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        raise ValueError("Project folder is empty or does not exist.")
    for label, path in (("Scene video", video_path), ("Previous scene video", reference_video_path)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} was not found: {path}")
        try:
            inside_project = os.path.commonpath([project_folder, path]) == project_folder
        except ValueError:
            inside_project = False
        if not inside_project:
            raise ValueError(f"{label} must be inside the current project folder.")

    fade_seconds = max(0.05, min(30.0, float(payload.get("fade_seconds", 1.0) or 1.0)))
    strength = max(0.0, min(1.0, float(payload.get("strength", 0.85) or 0.85)))
    if strength <= 0.0:
        return {"video_path": video_path, "applied": False, "reason": "strength is zero"}

    ffmpeg_path = _find_ffmpeg_path()
    work_dir = os.path.dirname(video_path)
    token = f"{int(time.time() * 1000)}_{os.getpid()}"
    reference_frame = os.path.join(work_dir, f".vrgdg_color_reference_{token}.png")
    target_frame = os.path.join(work_dir, f".vrgdg_color_target_{token}.png")
    cube_path = os.path.join(work_dir, f".vrgdg_color_match_{token}.cube")
    output_path = os.path.join(work_dir, f".vrgdg_color_matched_{token}.mp4")

    def run_ffmpeg(command, message):
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", cwd=work_dir)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or message).strip())

    try:
        # -update 1 leaves the last decoded frame in the PNG after processing the final second.
        run_ffmpeg([
            ffmpeg_path, "-y", "-sseof", "-1", "-i", reference_video_path,
            "-map", "0:v:0", "-an", "-update", "1", reference_frame,
        ], "FFmpeg could not read the previous clip's final frame.")
        run_ffmpeg([
            ffmpeg_path, "-y", "-i", video_path, "-map", "0:v:0", "-an",
            "-frames:v", "1", target_frame,
        ], "FFmpeg could not read the new clip's first frame.")

        with Image.open(reference_frame) as image:
            reference_stats = ImageStat.Stat(image.convert("RGB"))
        with Image.open(target_frame) as image:
            target_stats = ImageStat.Stat(image.convert("RGB"))
        reference_mean = [float(value) for value in reference_stats.mean[:3]]
        reference_std = [max(1.0, float(value)) for value in reference_stats.stddev[:3]]
        target_mean = [float(value) for value in target_stats.mean[:3]]
        target_std = [max(1.0, float(value)) for value in target_stats.stddev[:3]]
        scales = [max(0.25, min(4.0, reference_std[i] / target_std[i])) for i in range(3)]
        offsets = [reference_mean[i] - target_mean[i] * scales[i] for i in range(3)]

        cube_size = 17
        with open(cube_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write('TITLE "VRGDG opening color match"\n')
            handle.write(f"LUT_3D_SIZE {cube_size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
            for blue in range(cube_size):
                for green in range(cube_size):
                    for red in range(cube_size):
                        values = [red, green, blue]
                        corrected = [
                            max(0.0, min(1.0, ((values[i] / (cube_size - 1)) * 255.0 * scales[i] + offsets[i]) / 255.0))
                            for i in range(3)
                        ]
                        handle.write(f"{corrected[0]:.8f} {corrected[1]:.8f} {corrected[2]:.8f}\n")

        weight = f"max(0\\,min(1\\,{strength:.6f}*(1-T/{fade_seconds:.6f})))"
        filter_graph = (
            f"[0:v]split=2[original][to_match];"
            f"[to_match]lut3d=file='{os.path.basename(cube_path)}'[matched];"
            f"[original][matched]blend=all_expr='A*(1-({weight}))+B*({weight})'[video]"
        )
        run_ffmpeg([
            ffmpeg_path, "-y", "-i", video_path,
            "-filter_complex", filter_graph,
            "-map", "[video]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart", output_path,
        ], "FFmpeg could not apply the opening color match.")
        if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
            raise RuntimeError("Opening color match did not create a valid video.")
        os.replace(output_path, video_path)
        thumbnail_path = _create_scene_video_thumbnail(video_path, _scene_video_thumbnail_path(video_path))
        return {
            "video_path": video_path,
            "thumbnail_path": thumbnail_path,
            "applied": True,
            "fade_seconds": fade_seconds,
            "strength": strength,
            "reference_video_path": reference_video_path,
        }
    finally:
        for temporary_path in (reference_frame, target_frame, cube_path, output_path):
            try:
                if os.path.isfile(temporary_path):
                    os.remove(temporary_path)
            except Exception:
                pass


def _find_scene_video_output(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        raise ValueError("Project folder is empty or does not exist.")
    mode = str(payload.get("video_mode", "") or "").strip().lower()
    if mode == "rtv":
        prefixes = ("reference_to_video_clips", "reference_to_video_clips_")
    elif mode == "t2v":
        prefixes = ("text_to_video_clips", "text_to_video_clips_")
    elif mode == "ingredients":
        prefixes = ("ingredients_to_video_clips", "ingredients_to_video_clips_")
    elif mode == "id_lora":
        prefixes = ("id_lora_i2v_clips", "id_lora_i2v_clips_")
    else:
        prefixes = ("image_to_video_clips", "image_to_video_clips_")

    scene_number = _int_payload(payload, "scene_number", 0, 0, 999999)
    prompt_number = _int_payload(payload, "prompt_number_one_based", scene_number or 0, 0, 999999)
    min_mtime = float(payload.get("min_mtime") or 0)
    output_folder = os.path.abspath(str(payload.get("output_folder", "") or "").strip().strip('"')) if payload.get("output_folder") else ""

    folders = []
    if output_folder and os.path.isdir(output_folder):
        try:
            if os.path.commonpath([project_folder, output_folder]) == project_folder:
                folders.append(output_folder)
        except ValueError:
            pass
    for name in os.listdir(project_folder):
        path = os.path.abspath(os.path.join(project_folder, name))
        if not os.path.isdir(path):
            continue
        if any(name == prefix.rstrip("_") or name.startswith(prefix) for prefix in prefixes):
            folders.append(path)
    folders = list(dict.fromkeys(folders))

    candidates = []
    for folder in folders:
        for root, _dirs, files in os.walk(folder):
            try:
                if os.path.commonpath([project_folder, os.path.abspath(root)]) != project_folder:
                    continue
            except ValueError:
                continue
            for name in files:
                lower = name.lower()
                if not lower.endswith("-audio.mp4"):
                    continue
                path = os.path.abspath(os.path.join(root, name))
                try:
                    mtime = os.path.getmtime(path)
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0 or (min_mtime and mtime + 1 < min_mtime):
                    continue
                score = 0
                if scene_number and re.match(rf"^video_{scene_number:04d}-audio\.mp4$", name, re.IGNORECASE):
                    score += 1000
                if prompt_number and re.match(rf"^video_{prompt_number:04d}(?:_|-)", name, re.IGNORECASE):
                    score += 700
                if scene_number and f"_{scene_number:04d}_" in name:
                    score += 100
                candidates.append((score, mtime, path, folder))
    if not candidates:
        return {"video_path": "", "output_folder": "", "searched_folders": folders}
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _score, _mtime, path, folder = candidates[0]
    _wait_for_stable_readable_file(path, timeout=8.0, interval=0.25)
    return {
        "video_path": path,
        "output_folder": folder,
        "searched_folders": folders,
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


def _collect_minimax_h3_stage_backup(payload):
    source_path = os.path.abspath(str(payload.get("source_path", "") or "").strip().strip('"'))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"MiniMax H3 stage video was not found: {source_path}")
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder or not os.path.isdir(project_folder):
        raise ValueError("Project folder is empty or does not exist.")
    if os.path.commonpath([project_folder, source_path]) == project_folder:
        raise ValueError("MiniMax H3 stage backup source must be outside the project folder.")
    stage = str(payload.get("stage", "") or "").strip().lower()
    if stage not in {"stage1", "stage2"}:
        raise ValueError("MiniMax H3 stage backup must be stage1 or stage2.")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    backup_dir = os.path.join(project_folder, "rendered_scene_videos_backup", f"scene_{scene_number:04d}")
    os.makedirs(backup_dir, exist_ok=True)
    _wait_for_stable_readable_file(source_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target_path = os.path.join(backup_dir, f"video_{scene_number:04d}-{stage}_{stamp}.mp4")
    index = 2
    while os.path.exists(target_path):
        target_path = os.path.join(backup_dir, f"video_{scene_number:04d}-{stage}_{stamp}_{index:02d}.mp4")
        index += 1
    _retry_file_op(lambda: shutil.copy2(source_path, target_path), f"Copying MiniMax H3 {stage} backup")
    thumbnail_path = _create_scene_video_thumbnail(target_path)
    return {
        "backup_path": target_path,
        "backup_thumbnail_path": thumbnail_path,
        "stage": stage,
        "scene_number": scene_number,
    }


def _find_minimax_h3_stage_outputs(payload):
    output_folder = os.path.abspath(str(payload.get("output_folder", "") or "").strip().strip('"'))
    if not output_folder or not os.path.isdir(output_folder):
        return {"stage1_path": "", "stage2_path": "", "stage3_path": ""}
    min_mtime = float(payload.get("min_mtime") or 0)
    found = {}
    for root, _dirs, files in os.walk(output_folder):
        for name in files:
            lower = name.lower()
            if not lower.endswith("-audio.mp4"):
                continue
            stage = next((item for item in ("stage1", "stage2", "stage3") if item in lower), "")
            if not stage:
                continue
            path = os.path.abspath(os.path.join(root, name))
            try:
                mtime = os.path.getmtime(path)
                if mtime + 1 < min_mtime or os.path.getsize(path) <= 0:
                    continue
            except OSError:
                continue
            previous = found.get(stage)
            if not previous or mtime > previous[0]:
                found[stage] = (mtime, path)
    return {f"{stage}_path": found.get(stage, (0, ""))[1] for stage in ("stage1", "stage2", "stage3")}


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
    raw_scene_timing_items = payload.get("scene_timing_items", [])
    if not isinstance(raw_scene_timing_items, list):
        raw_scene_timing_items = []
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    preview_audio_start = max(0.0, float(payload.get("audio_start", 0) or 0))
    preview_audio_duration = max(0.0, float(payload.get("audio_duration", 0) or 0))
    target_width = _int_payload(payload, "width", 0, 0, 8192)
    target_height = _int_payload(payload, "height", 0, 0, 8192)
    use_embedded_scene_audio = bool(payload.get("use_embedded_scene_audio"))
    timeline_fps = _int_payload(payload, "timeline_fps", 0, 0, 120)

    scene_paths = []
    for index, raw_path in enumerate(raw_paths, start=1):
        path = os.path.abspath(str(raw_path or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Scene {index} video was not found: {path}")
        if os.path.splitext(path)[1].lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            raise ValueError(f"Scene {index} media is not a supported video file: {path}")
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
    elif use_embedded_scene_audio:
        for path in scene_paths:
            scene_audio_paths.append(path)
            scene_audio_items.append({"path": path, "start": 0.0, "duration": 0.0, "embedded": True})
    elif not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")

    ffmpeg_path = _find_ffmpeg_path()
    timeline_sync_paths = []
    timeline_sync_frame_count = 0
    concat_scene_paths = scene_paths
    if raw_scene_timing_items:
        if timeline_fps <= 0:
            raise ValueError("Timeline FPS is required when scene timing items are provided.")
        if len(raw_scene_timing_items) != len(scene_paths):
            raise ValueError("Scene timing item count does not match scene video count.")

        concat_scene_paths = []
        for index, (path, item) in enumerate(zip(scene_paths, raw_scene_timing_items), start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Scene {index} timing item is invalid.")
            start = max(0.0, float(item.get("start", 0) or 0))
            end = max(start, float(item.get("end", start) or start))
            start_frame = int(start * timeline_fps + 0.5)
            end_frame = int(end * timeline_fps + 0.5)
            target_frames = max(1, end_frame - start_frame)
            timeline_sync_frame_count += target_frames
            sync_path = os.path.join(target_dir, f"_temp_timeline_scene_{index:04d}.mp4")
            sync_filter = (
                f"fps={timeline_fps},"
                "tpad=stop_mode=clone:stop_duration=1,"
                f"trim=start_frame=0:end_frame={target_frames},"
                "setpts=PTS-STARTPTS"
            )
            sync_cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                path,
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                sync_filter,
                "-frames:v",
                str(target_frames),
                "-r",
                str(timeline_fps),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                sync_path,
            ]
            sync_result = subprocess.run(sync_cmd, capture_output=True, text=True, errors="replace")
            if sync_result.returncode != 0 or not os.path.isfile(sync_path):
                raise RuntimeError(
                    (sync_result.stderr or sync_result.stdout or f"FFmpeg failed to align scene {index} to the timeline.").strip()
                )
            timeline_sync_paths.append(sync_path)
            concat_scene_paths.append(sync_path)

    concat_file = os.path.join(target_dir, "concat_list.txt")
    with open(concat_file, "w", encoding="utf-8") as handle:
        for path in concat_scene_paths:
            handle.write(f"file '{_concat_file_path(path)}'\n")

    temp_video = os.path.join(target_dir, "_temp_video_no_audio.mp4")
    normalized_video = os.path.join(target_dir, "_temp_video_normalized_canvas.mp4")
    temp_audio = os.path.join(target_dir, "_temp_scene_audio.m4a")
    temp_global_audio = os.path.join(target_dir, "_temp_global_audio.m4a")
    temp_audio_parts = []
    audio_concat_file = os.path.join(target_dir, "audio_concat_list.txt")
    final_output = _unique_final_video_path(project_folder, payload.get("output_prefix", "FINAL_VIDEO"))
    normalized_canvas = False

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
    subprocess.run(concat_cmd, capture_output=True, text=True, errors="replace", check=True)

    insert_items = []
    for index, item in enumerate(raw_overlay_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Insert {index} item is invalid.")
        path = os.path.abspath(str(item.get("path", "") or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Insert {index} video was not found: {path}")
        if os.path.splitext(path)[1].lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            raise ValueError(f"Insert {index} media is not a supported video file: {path}")
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
            subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=True)
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
        subprocess.run(flatten_cmd, capture_output=True, text=True, errors="replace", check=True)
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

    if target_width > 0 and target_height > 0:
        normalized_canvas = _normalize_video_canvas(ffmpeg_path, temp_video, normalized_video, target_width, target_height)
        if normalized_canvas:
            try:
                os.remove(temp_video)
            except Exception:
                pass
            temp_video = normalized_video

    mux_audio_path = audio_path
    if scene_audio_paths:
        with open(audio_concat_file, "w", encoding="utf-8") as handle:
            for index, item in enumerate(scene_audio_items, start=1):
                path = item["path"]
                duration = float(item.get("duration", 0) or 0)
                if item.get("embedded") or item.get("start", 0) or duration:
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
                    trim_cmd.extend(["-vn", "-c:a", "aac", part_path])
                    subprocess.run(trim_cmd, capture_output=True, text=True, errors="replace", check=True)
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
            "-vn",
            "-c:a",
            "aac",
            temp_audio,
        ]
        subprocess.run(audio_concat_cmd, capture_output=True, text=True, errors="replace", check=True)
        mux_audio_path = temp_audio
    elif preview_audio_start or preview_audio_duration:
        trim_audio_cmd = [ffmpeg_path, "-y"]
        if preview_audio_start:
            trim_audio_cmd.extend(["-ss", f"{preview_audio_start:.6f}"])
        trim_audio_cmd.extend(["-i", audio_path])
        if preview_audio_duration:
            trim_audio_cmd.extend(["-t", f"{preview_audio_duration:.6f}"])
        trim_audio_cmd.extend(["-vn", "-c:a", "aac", temp_global_audio])
        subprocess.run(trim_audio_cmd, capture_output=True, text=True, errors="replace", check=True)
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
    ]
    if not timeline_sync_paths:
        mux_cmd.append("-shortest")
    mux_cmd.append(final_output)
    try:
        subprocess.run(mux_cmd, capture_output=True, text=True, errors="replace", check=True)
    finally:
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except Exception:
            pass
        try:
            if os.path.exists(normalized_video):
                os.remove(normalized_video)
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
        for sync_path in timeline_sync_paths:
            try:
                if os.path.exists(sync_path):
                    os.remove(sync_path)
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
        "used_embedded_scene_audio": bool(use_embedded_scene_audio and scene_audio_paths),
        "normalized_canvas": normalized_canvas,
        "timeline_frame_sync": bool(timeline_sync_paths),
        "timeline_fps": timeline_fps if timeline_sync_paths else 0,
        "timeline_frame_count": timeline_sync_frame_count,
        "output_width": target_width,
        "output_height": target_height,
        "removed_scratch_folders": removed_scratch_folders,
    }


def _render_image_slideshow(payload):
    raw_items = payload.get("image_items", [])
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("No scene images were provided for the slideshow preview.")
    project_folder, target_dir = _safe_project_subfolder(payload.get("project_folder", ""), "slideshow_previews")
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Global audio file was not found: {audio_path}")
    audio_start = max(0.0, float(payload.get("audio_start", 0) or 0))
    target_width = _int_payload(payload, "width", 1920, 64, 8192)
    target_height = _int_payload(payload, "height", 1080, 64, 8192)
    fps = _int_payload(payload, "fps", 24, 1, 120)

    items = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Scene {index} slideshow item is invalid.")
        path = os.path.abspath(str(item.get("path", "") or "").strip().strip('"'))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Scene {index} image was not found: {path}")
        if os.path.splitext(path)[1].lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            raise ValueError(f"Scene {index} media is not a supported slideshow image: {path}")
        duration = max(0.05, float(item.get("duration", 0) or 0))
        items.append({"path": path, "duration": duration})

    total_duration = sum(item["duration"] for item in items)
    ffmpeg_path = _find_ffmpeg_path()
    scratch = tempfile.mkdtemp(prefix="_slideshow_", dir=target_dir)
    concat_file = os.path.join(scratch, "images.txt")
    video_only = os.path.join(scratch, "video.mp4")
    final_output = _unique_final_video_path(project_folder, payload.get("output_prefix", "IMAGE_SLIDESHOW_PREVIEW"))
    try:
        # The concat demuxer does not safely handle still images whose stream
        # properties change mid-list.  In particular, FFmpeg's fps filter can
        # discard the image immediately before a resolution change while the
        # filter graph is reinitialized.  Normalize every source to one common
        # RGB frame first so a mixed-resolution project cannot lose a scene.
        normalized_items = []
        normalize_filter = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            "setsar=1,format=rgb24"
        )
        for index, item in enumerate(items, start=1):
            normalized_path = os.path.join(scratch, f"image_{index:06d}.png")
            normalize_cmd = [
                ffmpeg_path, "-y", "-i", item["path"],
                "-vf", normalize_filter,
                "-frames:v", "1", normalized_path,
            ]
            try:
                subprocess.run(normalize_cmd, capture_output=True, text=True, errors="replace", check=True)
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr or exc.stdout or str(exc)
                raise RuntimeError(f"Could not normalize slideshow Scene {index}:\n{detail}") from exc
            normalized_items.append({"path": normalized_path, "duration": item["duration"]})

        with open(concat_file, "w", encoding="utf-8") as handle:
            for item in normalized_items:
                handle.write(f"file '{_concat_file_path(item['path'])}'\n")
                handle.write(f"duration {item['duration']:.6f}\n")
            # The concat demuxer only applies the final duration when its last
            # still is repeated once.
            handle.write(f"file '{_concat_file_path(normalized_items[-1]['path'])}'\n")

        filter_graph = f"fps={fps},format=yuv420p"
        slideshow_cmd = [
            ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-vf", filter_graph,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-t", f"{total_duration:.6f}", "-movflags", "+faststart", video_only,
        ]
        subprocess.run(slideshow_cmd, capture_output=True, text=True, errors="replace", check=True)

        mux_cmd = [ffmpeg_path, "-y", "-i", video_only]
        if audio_start:
            mux_cmd.extend(["-ss", f"{audio_start:.6f}"])
        mux_cmd.extend([
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", f"{total_duration:.6f}",
            "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
            final_output,
        ])
        subprocess.run(mux_cmd, capture_output=True, text=True, errors="replace", check=True)
        if not os.path.isfile(final_output) or os.path.getsize(final_output) <= 0:
            raise RuntimeError("FFmpeg did not create the slideshow preview video.")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "final_video_path": final_output,
        "video_folder": target_dir,
        "scene_count": len(items),
        "duration": total_duration,
        "audio_start": audio_start,
        "output_width": target_width,
        "output_height": target_height,
        "fps": fps,
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
        video_gguf_unets, video_diffusion_models = _ltx_video_model_choices()
        return web.json_response({
            "ok": True,
            "unets": _folder_choices(("unet", "diffusion_models")),
            "video_gguf_unets": video_gguf_unets,
            "video_diffusion_models": video_diffusion_models,
            "vae": _folder_choices("vae"),
            "clip": _folder_choices(("clip", "text_encoders")),
            # LatentUpscaleModelLoader validates against latent_upscale_models,
            # not the ESRGAN/image upscale_models category.
            "upscale_models": _folder_choices("latent_upscale_models"),
        })

    @server_instance.routes.get("/vrgdg/workflow_runner/model_root")
    async def vrgdg_workflow_runner_model_root(request):
        result = load_custom_model_root()
        result["registered"] = register_custom_model_root(result.get("models_root", ""))
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/model_root")
    async def vrgdg_workflow_runner_save_model_root(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = save_custom_model_root(payload.get("models_root", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

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

    @server_instance.routes.post("/vrgdg/workflow_runner/build_krea2_prompt")
    async def vrgdg_workflow_runner_build_krea2_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_krea2_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_krea2_2pass_prompt")
    async def vrgdg_workflow_runner_build_krea2_2pass_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_krea2_2pass_api_prompt(payload)
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

    @server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_prompt")
    async def vrgdg_workflow_runner_build_minimax_h3_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_minimax_h3_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_2pass_prompt")
    async def vrgdg_workflow_runner_build_minimax_h3_2pass_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_minimax_h3_2pass_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_3pass_prompt")
    async def vrgdg_workflow_runner_build_minimax_h3_3pass_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_minimax_h3_3pass_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_rtv_prompt")
    async def vrgdg_workflow_runner_build_rtv_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_rtv_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_ingredients_prompt")
    async def vrgdg_workflow_runner_build_ingredients_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_ingredients_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_flf_prompt")
    async def vrgdg_workflow_runner_build_flf_prompt(request):
        try:
            payload = await request.json()
            result = _build_flf_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_id_lora_prompt")
    async def vrgdg_workflow_runner_build_id_lora_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_id_lora_api_prompt(payload)
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

    @server_instance.routes.post("/vrgdg/workflow_runner/build_transcribe_prompt")
    async def vrgdg_workflow_runner_build_transcribe_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_transcribe_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_timestamped_transcribe_prompt")
    async def vrgdg_workflow_runner_build_timestamped_transcribe_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_timestamped_transcribe_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/prepare_scene_audio_clip")
    async def vrgdg_workflow_runner_prepare_scene_audio_clip(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _prepare_scene_audio_clip(payload)
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

    @server_instance.routes.post("/vrgdg/workflow_runner/match_scene_video_start_color")
    async def vrgdg_workflow_runner_match_scene_video_start_color(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _apply_scene_start_color_match(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/trim_scene_video")
    async def vrgdg_workflow_runner_trim_scene_video(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _trim_scene_video(payload)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr or exc.stdout or str(exc)
            return web.json_response({"ok": False, "error": f"FFmpeg failed:\n{error}"}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/find_scene_video_output")
    async def vrgdg_workflow_runner_find_scene_video_output(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _find_scene_video_output(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/collect_minimax_h3_stage_backup")
    async def vrgdg_workflow_runner_collect_minimax_h3_stage_backup(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _collect_minimax_h3_stage_backup(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/find_minimax_h3_stage_outputs")
    async def vrgdg_workflow_runner_find_minimax_h3_stage_outputs(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _find_minimax_h3_stage_outputs(payload)
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

    @server_instance.routes.post("/vrgdg/workflow_runner/render_image_slideshow")
    async def vrgdg_workflow_runner_render_image_slideshow(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _render_image_slideshow(payload)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr or exc.stdout or str(exc)
            return web.json_response({"ok": False, "error": f"FFmpeg failed:\n{error}"}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    _VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED = True


class VRGDG_MiniMaxH3TurboLoRACompat:
    """Apply the upstream Turbo LoRA with its missing pruned Ref2VA audio-row fix.

    The upstream v1.2.1 adapter derives only video/audio and visual-reference
    time rows for pruned MiniMax checkpoints. Reference audio adds another row
    in ComfyUI core, causing AdaLN's base projection to have three rows while
    the LoRA delta has two. Delegate whenever upstream includes audio-row
    support; otherwise reproduce its pruned path with the complete core layout.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "VRGDG/Compatibility"
    DESCRIPTION = (
        "MiniMax-H3 Turbo LoRA adapter with pruned-model reference-audio "
        "conditioning compatibility."
    )

    @staticmethod
    def _condition_times(upstream, timestep, payload, shift_v, shift_a):
        sigma_v = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
        t_video = 1.0 - sigma_v
        t_audio = 1.0 - upstream._time_shift_sigma(sigma_v, shift_v, shift_a)
        layout = payload.get("layout")
        if layout is not None:
            segments = getattr(layout, "segments", ()) or ()
            has_visual_condition = any(
                kind in ("cond", "ref_img") for _, _, kind in segments
            )
            has_audio_condition = any(
                kind == "ref_audio" for _, _, kind in segments
            )
        else:
            refs = payload.get("refs") or ()
            ref_kinds = {
                str(item.get("kind") or "")
                for item in refs
                if isinstance(item, dict)
            }
            has_visual_condition = bool(payload.get("keyframes")) or bool(
                ref_kinds.intersection({"image", "video", "video_audio"})
            )
            has_audio_condition = bool(
                ref_kinds.intersection({"audio", "video_audio"})
            )
        visual_aug = float(payload.get("visual_cond_noise_aug", 0.999))
        audio_aug = float(payload.get("audio_cond_noise_aug", 1.0))
        times = {t_video, t_audio}
        if has_visual_condition:
            times.add(max(t_video, visual_aug))
        if has_audio_condition:
            times.add(max(t_audio, audio_aug))
        return sorted(times)

    def apply_lora(self, model, lora_name, strength):
        import inspect
        import nodes as comfy_nodes

        upstream_class = (getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}).get(
            "MiniMaxH3TurboLoRA"
        )
        if upstream_class is None:
            raise RuntimeError(
                "MiniMaxH3TurboLoRA is not registered. Install or update "
                "ComfyUI-MiniMax-H3-Turbo, then restart ComfyUI."
            )
        upstream = sys.modules.get(upstream_class.__module__)
        if upstream is None:
            upstream = importlib.import_module(upstream_class.__module__)
        upstream_node = upstream_class()
        unique_t = getattr(upstream, "_unique_t", None)
        upstream_supports_audio = False
        if callable(unique_t):
            try:
                upstream_supports_audio = "has_aud_cond" in inspect.signature(unique_t).parameters
            except (TypeError, ValueError):
                upstream_supports_audio = False

        diffusion_model = model.model.diffusion_model
        pruned = bool(getattr(diffusion_model, "use_adaln_curves", False))
        if not pruned or upstream_supports_audio:
            return upstream_node.apply_lora(model, lora_name, strength)

        required_helpers = (
            "_apply_bypass_lora",
            "_egrid",
            "_interp_egrid",
            "_add_dbg_wrapper",
            "_time_shift_sigma",
        )
        missing_helpers = [name for name in required_helpers if not hasattr(upstream, name)]
        make_adaln_forward = getattr(upstream, "_make_adaln_forward", None)
        legacy_adaln_delta = getattr(upstream, "_AdalnDelta", None)
        if not callable(make_adaln_forward) and legacy_adaln_delta is None:
            missing_helpers.append("_make_adaln_forward (or legacy _AdalnDelta)")
        if missing_helpers:
            raise RuntimeError(
                "The installed ComfyUI-MiniMax-H3-Turbo version is incompatible "
                "with Builder reference-audio support. Missing helpers: "
                + ", ".join(missing_helpers)
                + ". Update the Turbo extension and restart ComfyUI."
            )

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise RuntimeError(f"MiniMax-H3 Turbo LoRA was not found: {lora_name}")
        lora = upstream.comfy.utils.load_torch_file(lora_path, safe_load=True)
        modules = sorted({key.rsplit(".lora_", 1)[0] for key in lora})
        new_model = model.clone()
        backbone = [name for name in modules if "adaln_proj" not in name]
        adaln = [name for name in modules if "adaln_proj" in name]
        bound = upstream._apply_bypass_lora(new_model, lora, backbone, strength)

        embedding_grid = upstream._egrid()
        shared = {"silu_temb": None}
        shift_v = float(getattr(diffusion_model, "sigma_shift_video", upstream.SHIFT_V))
        shift_a = float(getattr(diffusion_model, "sigma_shift_audio", upstream.SHIFT_A))

        def wrap(executor, *args, **kwargs):
            timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
            context = args[2] if len(args) > 2 else kwargs.get("context")
            payload = kwargs.get("minimax_payload") or {}
            times = self._condition_times(upstream, timestep, payload, shift_v, shift_a)
            shared["silu_temb"] = upstream._interp_egrid(
                times,
                embedding_grid,
                context.device,
                context.dtype,
            )
            return executor(*args, **kwargs)

        new_model.add_wrapper_with_key(
            upstream.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "vrgdg_h3turbo_ref_audio",
            wrap,
        )
        for name in adaln:
            lora_a = lora[name + ".lora_A.weight"]
            lora_b = lora[name + ".lora_B.weight"] * strength
            model_key = "diffusion_model." + name.rsplit(".linear", 1)[0]
            base_adaln = new_model.get_model_object(model_key)
            if callable(make_adaln_forward):
                # Turbo v1.2.2+ patches only the forward attribute so ComfyUI's
                # dynamic-VRAM unload keeps the original module tree intact.
                new_model.add_object_patch(
                    model_key + ".forward",
                    make_adaln_forward(base_adaln, lora_a, lora_b, shared),
                )
            else:
                # Retain compatibility with the earlier upstream API that
                # exposed a whole-module AdaLN wrapper.
                new_model.add_object_patch(
                    model_key,
                    legacy_adaln_delta(base_adaln, lora_a, lora_b, shared),
                )
        print(
            "[VRGDG MiniMaxH3TurboLoRACompat] pruned base: "
            f"{bound} backbone adapters + {len(adaln)} AdaLN adapters; "
            "reference-audio time rows enabled",
            flush=True,
        )
        dbg_wrapper = upstream._add_dbg_wrapper
        try:
            dbg_has_mode = "mode" in inspect.signature(dbg_wrapper).parameters
        except (TypeError, ValueError):
            dbg_has_mode = False
        if dbg_has_mode:
            dbg_wrapper(
                new_model,
                diffusion_model,
                "pruned-ref-audio-compat",
                "bypass",
            )
        else:
            dbg_wrapper(new_model, diffusion_model, "pruned-ref-audio-compat")
        return (new_model,)


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
    "VRGDG_MiniMaxH3TurboLoRACompat": VRGDG_MiniMaxH3TurboLoRACompat,
    "VRGDG_ZImageWorkflowRunnerUI": VRGDG_ZImageWorkflowRunnerUI,
    "VRGDG_ClearMemoryButtonUI": VRGDG_ClearMemoryButtonUI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3TurboLoRACompat": "VRGDG MiniMax-H3 Turbo LoRA Compatibility",
    "VRGDG_ZImageWorkflowRunnerUI": "VRGDG Z-Image Workflow Runner UI",
    "VRGDG_ClearMemoryButtonUI": "VRGDG Clear Memory Button",
}
