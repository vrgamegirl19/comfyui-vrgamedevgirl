"""VRGDG MiniMax H3 learned latent-upscaler loader and apply nodes.

Uses ComfyUI's registered ``latent_upscale_models`` paths, matching the model
dropdown behavior of ComfyUI's native latent-upscale loaders.
"""

import importlib.util
import os
import sys

import torch

import folder_paths
import comfy.nested_tensor


MODEL_TYPE = "VRGDG_MINIMAX_H3_LATENT_UPSCALE_MODEL"
_BACKEND_MODULE_NAME = "_vrgdg_minimax_h3_learned_upscale_backend"
_MODEL_CACHE = {}
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"

if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER),
    )


def _backend_path():
    comfy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        comfy_root,
        "Comfyui_Minimax_h3_latent_Upscaler",
        "nodes",
        "minimax_h3_latent_upscaler_3d.py",
    )


def _load_backend():
    cached = sys.modules.get(_BACKEND_MODULE_NAME)
    if cached is not None:
        return cached

    path = _backend_path()
    if not os.path.isfile(path):
        raise RuntimeError(
            "The LBH MiniMax H3 latent-upscaler backend is not installed. "
            f"Expected: {path}"
        )
    spec = importlib.util.spec_from_file_location(_BACKEND_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MiniMax H3 upscaler backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BACKEND_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _model_choices():
    names = folder_paths.get_filename_list("latent_upscale_models")
    return [
        name
        for name in names
        if "minimax_h3_latent_upscaler" in os.path.basename(name).lower()
        and name.lower().endswith((".safetensors", ".pth"))
    ]


def _resolve_model_path(model_name):
    path = folder_paths.get_full_path("latent_upscale_models", model_name)
    if not path or not os.path.isfile(path):
        searched = folder_paths.get_folder_paths("latent_upscale_models")
        raise FileNotFoundError(
            f"MiniMax H3 latent-upscale model '{model_name}' was not found. "
            f"Registered model folders: {searched}"
        )
    return os.path.abspath(path)


def _resolve_registered_model_path(model_name):
    """Resolve an H3 checkpoint across every registered model directory."""
    requested = str(model_name or "").strip()
    if not requested or requested.startswith("["):
        raise FileNotFoundError("No MiniMax H3 latent-upscale model was selected.")
    for root in folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER):
        candidate = os.path.abspath(os.path.join(str(root), requested))
        if os.path.isfile(candidate):
            return candidate
    # ComfyUI's resolver also understands configured secondary paths and
    # subfolder names.  Keep this fallback for custom folder-path providers.
    resolved = folder_paths.get_full_path(_LATENT_UPSCALE_FOLDER, requested)
    if resolved and os.path.isfile(resolved):
        return os.path.abspath(resolved)
    raise FileNotFoundError(
        f"MiniMax H3 latent-upscale model '{requested}' was not found in registered "
        f"latent_upscale_models folders: {folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)}"
    )


def _load_model(model_name, device_name, precision):
    path = _resolve_model_path(model_name)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cache_key = (path, str(device), precision)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    backend = _load_backend()
    raw_state = backend._load_raw_sd(path)
    state = backend._extract_upscaler_sd(raw_state)
    config = backend._detect_arch(state)
    model = backend.LatentResizer3D(
        in_channels=config["in_channels"],
        in_blocks=config["in_blocks"],
        out_blocks=config["out_blocks"],
        channels=config["channels"],
        dropout=config["dropout"],
        attn=config["attn"],
        temporal_every=config["temporal_every"],
        temporal_kernel=config["temporal_kernel"],
    )
    model.load_state_dict(state, strict=True)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    model = model.to(device=device, dtype=dtype).eval()
    loaded = {
        "model": model,
        "device": device,
        "dtype": dtype,
        "precision": precision,
        "model_name": model_name,
        "path": path,
        "backend": backend,
    }
    _MODEL_CACHE[cache_key] = loaded
    print(f"[VRGDG MiniMax H3 Upscaler] Loaded: {path} ({precision}, {device})")
    return loaded


class VRGDG_MiniMaxH3LatentUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _model_choices()
        if not models:
            models = ["[no models found in registered latent_upscale_models folders]"]
        return {
            "required": {
                "model_name": (models,),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("upscale_model",)
    FUNCTION = "load"
    CATEGORY = "VRGDG/Video/MiniMax H3"

    def load(self, model_name, device, precision):
        if model_name.startswith("["):
            raise FileNotFoundError(
                "No models were found in ComfyUI's registered latent_upscale_models folders."
            )
        return (_load_model(model_name, device, precision),)


class VRGDG_MiniMaxH3UltimateUpscaleParams:
    """Reliable H3 parameter node for the upstream MMH3 Ultimate Upscale node.

    The upstream parameter node builds its Combo options from only one model
    directory at import time.  This node resolves all registered directories
    at execution time and emits the absolute checkpoint path in the shared
    parameter dictionary, so the upstream processor can load the chosen file
    regardless of which ComfyUI model path contains it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _model_choices()
        if not models:
            models = ["[no MiniMax H3 models found in registered latent_upscale_models folders]"]
        return {
            "required": {
                "model_name": (models,),
                "width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 704, "min": 64, "max": 4096, "step": 32}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("H3_UPSCALE_PARAM",)
    RETURN_NAMES = ("latent_upscale_param",)
    FUNCTION = "create"
    CATEGORY = "VRGDG/Video/MiniMax H3"

    def create(self, model_name, width, height, device, precision):
        path = _resolve_registered_model_path(model_name)
        return ({
            "model_name": path,
            "width": int(round(int(width) / 32.0)) * 32,
            "height": int(round(int(height) / 32.0)) * 32,
            "device": device,
            "precision": precision,
        },)


class VRGDG_MiniMaxH3LearnedLatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "upscale_model": (MODEL_TYPE,),
                "scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "VRGDG/Video/MiniMax H3"

    def upscale(self, latent, upscale_model, scale):
        if scale < 1.0:
            raise ValueError("MiniMax H3 learned latent upscale requires scale >= 1.0.")
        if abs(scale - 1.0) < 1e-6:
            return (latent,)
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("Expected a LATENT dictionary containing 'samples'.")

        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            raise ValueError(
                "The learned upscaler accepts the video latent only. "
                "Separate the MiniMax H3 AV latent before this node."
            )
        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"Expected latent samples tensor, got {type(samples).__name__}.")
        if samples.ndim not in (4, 5):
            raise ValueError(f"Expected 4D or 5D video latent, got shape {tuple(samples.shape)}.")
        if samples.shape[1] != 24:
            raise ValueError(
                f"Expected a 24-channel MiniMax H3 video latent, got {samples.shape[1]} channels."
            )

        model = upscale_model["model"]
        device = upscale_model["device"]
        dtype = upscale_model["dtype"]
        backend = upscale_model["backend"]
        original_dtype = samples.dtype
        was_4d = samples.ndim == 4
        work = samples.unsqueeze(2) if was_4d else samples
        work = work.to(device=device, dtype=dtype)
        mean, std = backend._make_norm_tensors(device, dtype)
        work = (work - mean) / std

        with torch.no_grad():
            time, height, width = work.shape[2:]
            target = (
                time,
                int(round(height * scale)),
                int(round(width * scale)),
            )
            output = model(work, scale=scale, target_size=target)

        output = output * std + mean
        if was_4d:
            output = output.squeeze(2)
        output = output.to(device="cpu", dtype=original_dtype)
        result = latent.copy()
        result["samples"] = output
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return (result,)


class VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent:
    """Replace only the video member of a joint H3 AV latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_av_latent": ("LATENT",),
                "upscaled_video_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("upscaled_av_latent",)
    FUNCTION = "replace"
    CATEGORY = "VRGDG/Video/MiniMax H3"

    def replace(self, original_av_latent, upscaled_video_latent):
        if not isinstance(original_av_latent, dict) or "samples" not in original_av_latent:
            raise ValueError("Expected the original joint MiniMax H3 AV latent.")
        samples = original_av_latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("original_av_latent must contain joint video+audio NestedTensor samples.")
        parts = list(samples.unbind())
        if len(parts) < 2:
            raise ValueError("The joint MiniMax H3 latent does not contain an audio member.")

        video = upscaled_video_latent.get("samples")
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != 24:
            shape = tuple(video.shape) if isinstance(video, torch.Tensor) else type(video).__name__
            raise ValueError(f"Expected an upscaled 24-channel 5D video latent, got {shape}.")

        result = original_av_latent.copy()
        result["samples"] = comfy.nested_tensor.NestedTensor((video, parts[1]))

        masks = original_av_latent.get("noise_mask")
        if getattr(masks, "is_nested", False):
            mask_parts = list(masks.unbind())
            audio_mask = mask_parts[1] if len(mask_parts) > 1 else torch.zeros_like(parts[1])
        else:
            audio_mask = torch.zeros_like(parts[1])
        result["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video), audio_mask)
        )
        return (result,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3LatentUpscaleModelLoader": VRGDG_MiniMaxH3LatentUpscaleModelLoader,
    "VRGDG_MiniMaxH3UltimateUpscaleParams": VRGDG_MiniMaxH3UltimateUpscaleParams,
    "VRGDG_MiniMaxH3LearnedLatentUpscale": VRGDG_MiniMaxH3LearnedLatentUpscale,
    "VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent": VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3LatentUpscaleModelLoader": "Load MiniMax H3 Learned Latent Upscaler",
    "VRGDG_MiniMaxH3UltimateUpscaleParams": "MiniMax H3 Ultimate Upscale Params (VRGDG)",
    "VRGDG_MiniMaxH3LearnedLatentUpscale": "MiniMax H3 Learned Latent Upscale",
    "VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent": "MiniMax H3 Replace with Upscaled Video Latent",
}
