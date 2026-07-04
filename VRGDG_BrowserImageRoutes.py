import json
import os
import subprocess
import time

import folder_paths
from aiohttp import web
from server import PromptServer

from .VRGDG_FlowBrowserNodes import (
    DEFAULT_FLOW_DIR,
    DEFAULT_NODE_VERSION,
    MAX_FLOW_IMAGES,
    _chrome_exe,
    _ensure_portable_node,
    _find_local_node_exe,
    _find_local_npm_cmd,
    _npm_command,
    _start_debug_chrome,
)
from .VRGDG_WorkflowRunnerNodes import _prepare_load_image_name, _resolve_existing_file


_VRGDG_BROWSER_IMAGE_ROUTES_REGISTERED = False

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
}


def _normalize_provider(value):
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    provider = _PROVIDER_ALIASES.get(key, key)
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown browser image provider: {value or '(empty)'}")
    return provider


def _coerce_int(value, default, minimum=None, maximum=None):
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _browser_image_status():
    flow_dir = DEFAULT_FLOW_DIR
    package_path = os.path.join(flow_dir, "package.json")
    playwright_dir = os.path.join(flow_dir, "node_modules", "playwright")
    node_exe = _find_local_node_exe(flow_dir)
    npm_cmd = _find_local_npm_cmd(flow_dir)
    chrome_path = ""
    chrome_error = ""
    try:
        chrome_path = _chrome_exe()
    except Exception as exc:
        chrome_error = str(exc)
    return {
        "flow_dir": flow_dir,
        "package_json": package_path,
        "package_json_exists": os.path.isfile(package_path),
        "node_exe": node_exe or "",
        "node_ready": bool(node_exe and os.path.isfile(node_exe)),
        "npm_cmd": npm_cmd or "",
        "npm_ready": bool(npm_cmd and os.path.isfile(npm_cmd)),
        "playwright_dir": playwright_dir,
        "playwright_ready": os.path.isdir(playwright_dir),
        "chrome_exe": chrome_path,
        "chrome_ready": bool(chrome_path),
        "chrome_error": chrome_error,
        "providers": {
            key: {
                "label": config["label"],
                "url": config["url"],
                "debug_port": config["debug_port"],
                "profile_name": config["profile_name"],
            }
            for key, config in _PROVIDERS.items()
        },
    }


def _install_browser_image_deps(payload):
    flow_dir = DEFAULT_FLOW_DIR
    package_path = os.path.join(flow_dir, "package.json")
    if not os.path.isfile(package_path):
        raise FileNotFoundError(f"Flow automation package.json not found: {package_path}")

    install_portable_node = _coerce_bool(payload.get("install_portable_node"), True)
    install_if_missing = _coerce_bool(payload.get("install_if_missing"), True)
    strict_ssl = _coerce_bool(payload.get("strict_ssl"), False)
    timeout_seconds = _coerce_int(payload.get("timeout_seconds"), 600, 30, 1800)
    node_version = str(payload.get("node_version") or DEFAULT_NODE_VERSION).strip() or DEFAULT_NODE_VERSION

    lines = ["VRGDG Browser Image setup", f"flow_dir: {flow_dir}"]
    node_exe = _find_local_node_exe(flow_dir)
    npm_cmd = _find_local_npm_cmd(flow_dir)
    if node_exe and npm_cmd:
        lines.append(f"Portable Node.js is ready: {node_exe}")
    elif install_portable_node:
        lines.append(f"Installing portable Node.js {node_version}...")
        node_exe = _ensure_portable_node(flow_dir, node_version, timeout_seconds)
        npm_cmd = _find_local_npm_cmd(flow_dir)
        lines.append(f"Portable Node.js installed: {node_exe}")
    else:
        lines.append("Portable Node.js is missing; setup will try system npm.")

    playwright_dir = os.path.join(flow_dir, "node_modules", "playwright")
    if os.path.isdir(playwright_dir):
        lines.append("Playwright dependency is already installed.")
        return {"status": "\n".join(lines), **_browser_image_status()}

    if not install_if_missing:
        lines.append("Playwright dependency is missing.")
        lines.append("Enable install_if_missing and run setup again.")
        return {"status": "\n".join(lines), **_browser_image_status()}

    command = [_npm_command(flow_dir), "install"]
    if not strict_ssl:
        command.append("--strict-ssl=false")
    lines.append(f"Running: {' '.join(command)}")
    process = subprocess.run(
        command,
        cwd=flow_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    stdout = process.stdout or ""
    stderr = process.stderr or ""
    if process.returncode != 0:
        raise RuntimeError(f"npm install failed.\n\nSTDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}")
    if not os.path.isdir(playwright_dir):
        raise RuntimeError(
            "npm install completed, but node_modules/playwright was not found.\n\n"
            f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
        )
    lines.append("Install complete.")
    return {"status": "\n".join(lines), **_browser_image_status()}


def _ingredient_load_image_names(payload):
    ingredients = payload.get("image_ingredients") or payload.get("images") or []
    if isinstance(ingredients, str):
        try:
            ingredients = json.loads(ingredients)
        except Exception:
            ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("Browser image ingredients must be a list.")

    image_names = []
    for index, item in enumerate(ingredients[:MAX_FLOW_IMAGES], start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path", "") or "").strip()
        raw_data = str(item.get("data", "") or "").strip()
        raw_name = str(item.get("name", "") or f"browser_ref_{index}.png").strip() or f"browser_ref_{index}.png"
        if raw_data:
            image_name = _prepare_load_image_name("", raw_data, raw_name)
        elif raw_path:
            resolved = _resolve_existing_file(raw_path, f"Browser image reference {index}")
            image_name = _prepare_load_image_name(resolved, "", os.path.basename(resolved))
        else:
            continue
        if image_name:
            image_names.append(image_name)
        time.sleep(0.002)
    return image_names


def _prompt_for_provider(prompt_text, provider, payload):
    prompt_text = str(prompt_text or "").strip()
    if provider != "gpt_image":
        return prompt_text
    aspect_ratio = str(payload.get("aspect_ratio") or "").strip()
    if not aspect_ratio:
        return prompt_text
    if "aspect ratio" in prompt_text.lower() and aspect_ratio in prompt_text:
        return prompt_text
    return f"{prompt_text}\n\nAspect ratio: {aspect_ratio}.".strip()


def _build_browser_image_prompt(payload):
    provider = _normalize_provider(payload.get("provider"))
    config = _PROVIDERS[provider]
    prompt_text = _prompt_for_provider(payload.get("prompt", ""), provider, payload)
    if not prompt_text:
        raise ValueError(f"{config['label']} prompt text is empty.")

    debug_port = _coerce_int(payload.get("debug_port"), config["debug_port"], 1, 65535)
    timeout_seconds = _coerce_int(payload.get("timeout_seconds"), config["timeout_seconds"], 60, 2400)
    image_names = _ingredient_load_image_names(payload)

    browser_node_id = "1"
    preview_node_id = "900"
    prompt = {
        browser_node_id: {
            "inputs": {
                "prompt": prompt_text,
                "image_count": len(image_names),
                "debug_port": debug_port,
                "timeout_seconds": timeout_seconds,
            },
            "class_type": config["class_type"],
            "_meta": {"title": config["label"]},
        },
        preview_node_id: {
            "inputs": {"images": [browser_node_id, 0]},
            "class_type": "PreviewImage",
            "_meta": {"title": "Browser Image Preview"},
        },
    }

    for index, image_name in enumerate(image_names, start=1):
        node_id = str(100 + index)
        prompt[node_id] = {
            "inputs": {"image": image_name, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": f"Browser Reference {index}"},
        }
        prompt[browser_node_id]["inputs"][f"image{index}"] = [node_id, 0]

    return {
        "provider": provider,
        "provider_label": config["label"],
        "prompt": prompt,
        "used_prompt": prompt_text,
        "image_count": len(image_names),
        "debug_port": debug_port,
        "timeout_seconds": timeout_seconds,
    }


def _ensure_browser_image_routes():
    global _VRGDG_BROWSER_IMAGE_ROUTES_REGISTERED
    if _VRGDG_BROWSER_IMAGE_ROUTES_REGISTERED:
        return
    server_instance = PromptServer.instance

    @server_instance.routes.get("/vrgdg/browser_image/status")
    async def vrgdg_browser_image_status(request):
        return web.json_response({"ok": True, **_browser_image_status()})

    @server_instance.routes.post("/vrgdg/browser_image/setup")
    async def vrgdg_browser_image_setup(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        try:
            result = _install_browser_image_deps(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc), **_browser_image_status()}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/browser_image/open_login")
    async def vrgdg_browser_image_open_login(request):
        try:
            payload = await request.json()
            provider = _normalize_provider(payload.get("provider"))
            config = _PROVIDERS[provider]
            port = _coerce_int(payload.get("debug_port"), config["debug_port"], 1, 65535)
            _start_debug_chrome(DEFAULT_FLOW_DIR, port, config["url"], profile_name=config["profile_name"])
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({
            "ok": True,
            "provider": provider,
            "provider_label": config["label"],
            "url": config["url"],
            "debug_port": port,
        })

    @server_instance.routes.post("/vrgdg/workflow_runner/build_flow_gpt_image_prompt")
    async def vrgdg_workflow_runner_build_flow_gpt_image_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_browser_image_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    _VRGDG_BROWSER_IMAGE_ROUTES_REGISTERED = True


_ensure_browser_image_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
