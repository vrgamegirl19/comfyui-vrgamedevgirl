import json
import os

import folder_paths


_CUSTOM_MODEL_ROOT_FILE = "custom_model_root.json"
_REGISTERED_MODEL_FOLDERS = set()

_CATEGORY_SUBFOLDERS = {
    "diffusion_models": ("diffusion_models", "unet"),
    "unet": ("diffusion_models", "unet"),
    "text_encoders": ("text_encoders", "clip"),
    "clip": ("text_encoders", "clip"),
    "vae": ("vae",),
    "loras": ("loras",),
    "upscale_models": ("upscale_models", "latent_upscale_models"),
    "latent_upscale_models": ("latent_upscale_models", "upscale_models"),
    "LLM": ("LLM", "llm", "LLMs", "llms"),
}


def _settings_folder():
    folder = os.path.join(folder_paths.get_output_directory(), "VRGDG_Model_Defaults")
    os.makedirs(folder, exist_ok=True)
    return folder


def custom_model_root_path():
    return os.path.join(_settings_folder(), _CUSTOM_MODEL_ROOT_FILE)


def normalize_custom_model_root(value):
    text = str(value or "").strip().strip('"')
    if not text:
        return ""
    return os.path.abspath(os.path.expanduser(text))


def load_custom_model_root():
    path = custom_model_root_path()
    if not os.path.isfile(path):
        return {"models_root": "", "path": path, "exists": False, "registered": []}
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    root = normalize_custom_model_root(payload.get("models_root", ""))
    return {
        "models_root": root,
        "path": path,
        "exists": bool(root and os.path.isdir(root)),
        "registered": [],
    }


def save_custom_model_root(value):
    root = normalize_custom_model_root(value)
    path = custom_model_root_path()
    payload = {"models_root": root}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    result = load_custom_model_root()
    result["registered"] = register_custom_model_root(root)
    return result


def custom_model_root_subfolders(category):
    root = load_custom_model_root().get("models_root", "")
    if not root or not os.path.isdir(root):
        return []
    paths = []
    for subfolder in _CATEGORY_SUBFOLDERS.get(str(category or ""), (str(category or ""),)):
        candidate = os.path.abspath(os.path.join(root, subfolder))
        if os.path.isdir(candidate):
            paths.append(candidate)
    return paths


def register_custom_model_root(root=None):
    root = normalize_custom_model_root(root) or load_custom_model_root().get("models_root", "")
    if not root or not os.path.isdir(root):
        return []
    registered = []
    for category, subfolders in _CATEGORY_SUBFOLDERS.items():
        for subfolder in subfolders:
            candidate = os.path.abspath(os.path.join(root, subfolder))
            if not os.path.isdir(candidate):
                continue
            key = (category, candidate)
            if key in _REGISTERED_MODEL_FOLDERS:
                registered.append({"category": category, "path": candidate})
                continue
            try:
                folder_paths.add_model_folder_path(category, candidate)
                _REGISTERED_MODEL_FOLDERS.add(key)
                registered.append({"category": category, "path": candidate})
            except Exception:
                pass
    return registered
