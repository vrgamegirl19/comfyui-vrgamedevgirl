import os


from .VRGDG_FlowBrowserNodes import (
    DEFAULT_FLOW_DIR,
    DEFAULT_NODE_VERSION,
    MAX_FLOW_IMAGES,
    _chrome_exe,
    _ensure_portable_node,
    _find_local_node_exe,
    _find_local_npm_cmd,
    _is_debug_chrome_ready,
    _npm_command,
    _node_command,
    _start_debug_chrome,
)


_PROVIDERS = {
    "flow_nano_banana": {
        "label": "Flow Nano Banana",
        "class_type": "VRGDG_FlowBrowserImageEdit",
        "url": "https://labs.google/fx/tools/flow",
        "debug_port": 9222,
        "profile_name": "chrome-flow-profile",
        "timeout_seconds": 420,
    },
    "gpt_image": {
        "label": "GPT Image",
        "class_type": "VRGDG_ChatGPTImagesBrowser",
        "url": "https://chatgpt.com/images",
        "debug_port": 9223,
        "profile_name": "chrome-chatgpt-profile",
        "timeout_seconds": 600,
    },
    "meta_ai": {
        "label": "Meta AI",
        "class_type": "VRGDG_MetaAIBrowserImage",
        "url": "https://www.meta.ai/",
        "debug_port": 9224,
        "profile_name": "chrome-meta-profile",
        "timeout_seconds": 600,
    },
}


_PROVIDER_ALIASES = {
    "flow": "flow_nano_banana",
    "flow_browser": "flow_nano_banana",
    "flow_nano": "flow_nano_banana",
    "flow_nanobanana": "flow_nano_banana",
    "flow_nano_banana": "flow_nano_banana",
    "chatgpt": "gpt_image",
    "chatgpt_image": "gpt_image",
    "chatgpt_images": "gpt_image",
    "gpt": "gpt_image",
    "gpt_image": "gpt_image",
    "gpt_image_2": "gpt_image",
    "gpt_images": "gpt_image",
    "meta": "meta_ai",
    "meta_ai": "meta_ai",
    "metaai": "meta_ai",
    "meta_image": "meta_ai",
    "meta_images": "meta_ai",
}


def _normalize_provider(value):
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    provider = _PROVIDER_ALIASES.get(key, key)
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown browser image provider: {value or '(empty)'}")
    return provider


def _newest_manual_download(provider):
    provider = _normalize_provider(provider)
    provider_folder = os.path.join(DEFAULT_FLOW_DIR, "manual_downloads", provider)
    download_folders = [provider_folder]
    user_profile = str(os.environ.get("USERPROFILE", "") or "").strip()
    home_folder = os.path.expanduser("~")
    for folder in [
        os.path.join(user_profile, "Downloads") if user_profile else "",
        os.path.join(home_folder, "Downloads") if home_folder else "",
    ]:
        normalized = os.path.normcase(os.path.abspath(folder)) if folder else ""
        if folder and normalized not in {
            os.path.normcase(os.path.abspath(item)) for item in download_folders
        }:
            download_folders.append(folder)
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".avif"}
    candidates = []
    searched_folders = []
    for folder in download_folders:
        if not os.path.isdir(folder):
            continue
        searched_folders.append(folder)
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            lower_name = filename.lower()
            if lower_name.endswith((".crdownload", ".part", ".tmp")):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_exts:
                continue
            try:
                stat = os.stat(path)
            except OSError:
                continue
            if stat.st_size <= 0:
                continue
            candidates.append((stat.st_mtime, path))
    candidates.sort(reverse=True)
    if not candidates:
        searched = "\n".join(searched_folders or download_folders)
        raise FileNotFoundError(f"No manual browser image downloads were found in:\n{searched}")
    return candidates[0][1]
