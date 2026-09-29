"""Model, LoRA and folder choices for the workflow runner."""

import os
import folder_paths
from ..core.model_paths import custom_model_root_subfolders, register_custom_model_root


_MAX_LORA_SLOTS = 20


_NONE_LORA = "[none]"


_REQUIRED_LTX_MSR_LORA = "licon\\LTX-2.3-Licon-MSR-V1.safetensors"


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
