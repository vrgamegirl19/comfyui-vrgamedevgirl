import asyncio
import base64
import json
import os
import re
import shutil
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
    _node_command,
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
    node_exe = _find_local_node_exe(flow_dir) or shutil.which("node")
    npm_cmd = _find_local_npm_cmd(flow_dir) or shutil.which("npm")
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
        "node_ready": bool(node_exe),
        "npm_cmd": npm_cmd or "",
        "npm_ready": bool(npm_cmd),
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
    elif install_portable_node and os.name == "nt":
        lines.append(f"Installing portable Node.js {node_version}...")
        node_exe = _ensure_portable_node(flow_dir, node_version, timeout_seconds)
        npm_cmd = _find_local_npm_cmd(flow_dir)
        lines.append(f"Portable Node.js installed: {node_exe}")
    else:
        if os.name == "nt":
            lines.append("Portable Node.js is missing; setup will try system npm.")
        else:
            node_exe = shutil.which("node")
            npm_cmd = shutil.which("npm")
            if not node_exe or not npm_cmd:
                raise RuntimeError("System Node.js/npm was not found. Install Node.js and npm, or set PATH before starting ComfyUI.")
            lines.append(f"System Node.js is ready: {node_exe}")
            lines.append(f"System npm is ready: {npm_cmd}")

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


def _safe_manual_ref_name(value, fallback):
    name = os.path.basename(str(value or "").strip()) or fallback
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "manual_ref"
    ext = ext if ext.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
    return f"{stem[:90]}{ext}"


def _save_manual_data_url(flow_dir, data_url, name):
    text = str(data_url or "").strip()
    if not text:
        return ""
    match = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", text, re.DOTALL)
    if not match:
        raise ValueError("Manual reference image data must be an image data URL.")
    raw = base64.b64decode(match.group(2), validate=False)
    folder = os.path.join(flow_dir, "manual_refs")
    os.makedirs(folder, exist_ok=True)
    ext = f".{match.group(1).lower().replace('jpeg', 'jpg')}"
    safe_name = _safe_manual_ref_name(name, f"manual_ref_{int(time.time())}{ext}")
    if not os.path.splitext(safe_name)[1]:
        safe_name += ext
    path = os.path.join(folder, safe_name)
    if os.path.exists(path):
        stem, file_ext = os.path.splitext(safe_name)
        path = os.path.join(folder, f"{stem}_{int(time.time() * 1000)}{file_ext or ext}")
    with open(path, "wb") as handle:
        handle.write(raw)
    return path


def _manual_image_paths(payload):
    flow_dir = DEFAULT_FLOW_DIR
    ingredients = payload.get("image_ingredients")
    if ingredients is None:
        ingredients = payload.get("images")
    if isinstance(ingredients, str):
        ingredients = [{"path": line.strip()} for line in ingredients.splitlines() if line.strip()]
    if not isinstance(ingredients, list):
        raise ValueError("Manual browser image references must be a list.")

    image_paths = []
    for index, item in enumerate(ingredients[:MAX_FLOW_IMAGES], start=1):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path", "") or "").strip()
        raw_data = str(item.get("data", "") or "").strip()
        raw_name = str(item.get("name", "") or f"manual_ref_{index}.png").strip()
        if raw_path:
            image_paths.append(_resolve_existing_file(raw_path, f"Manual browser reference {index}"))
        elif raw_data:
            image_paths.append(_save_manual_data_url(flow_dir, raw_data, raw_name))
    return [path for path in image_paths if path]


def _extract_manual_saved_path(stdout):
    for line in reversed((stdout or "").splitlines()):
        text = line.strip()
        if text.lower().startswith("saved:"):
            return text.split(":", 1)[1].strip()
    return ""


def _run_manual_bridge(payload, action):
    provider = _normalize_provider(payload.get("provider"))
    config = _PROVIDERS[provider]
    flow_dir = DEFAULT_FLOW_DIR
    script_path = os.path.join(flow_dir, "manual-bridge.mjs")
    if not os.path.isfile(script_path):
        raise RuntimeError(f"Manual browser bridge script not found: {script_path}")
    playwright_dir = os.path.join(flow_dir, "node_modules", "playwright")
    if not os.path.isdir(playwright_dir):
        raise RuntimeError("Browser automation dependencies are not installed. Run Install Browser Automation first.")

    port = _coerce_int(payload.get("debug_port"), config["debug_port"], 1, 65535)
    timeout_seconds = _coerce_int(payload.get("timeout_seconds"), config["timeout_seconds"], 15, 2400)
    # Keep browser downloads where the user expects them. Import Latest already
    # scans both this normal Downloads folder and the legacy provider capture
    # folders, so changing the CDP download target does not break importing.
    user_profile = str(os.environ.get("USERPROFILE", "") or "").strip()
    normal_downloads = os.path.join(user_profile, "Downloads") if user_profile else os.path.join(os.path.expanduser("~"), "Downloads")
    output_dir = normal_downloads if os.path.isdir(normal_downloads) else os.path.join(flow_dir, "manual_downloads", provider)
    os.makedirs(output_dir, exist_ok=True)
    _start_debug_chrome(flow_dir, port, config["url"], profile_name=config["profile_name"])

    command = [
        _node_command(flow_dir),
        script_path,
        "--provider",
        provider,
        "--action",
        action,
        "--url",
        config["url"],
        "--out",
        output_dir,
        "--timeout",
        str(timeout_seconds * 1000),
        "--connect-cdp",
        f"http://127.0.0.1:{port}",
    ]
    if action == "upload":
        for image_path in _manual_image_paths(payload):
            command.extend(["--image", image_path])
        prompt = str(payload.get("prompt", "") or "").strip()
        if prompt:
            command.extend(["--prompt", prompt])

    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    try:
        process = subprocess.run(
            command,
            cwd=flow_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 20,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        if action == "upload":
            raise RuntimeError(
                f"{config['label']} did not finish attaching the reference image(s) and prompt within "
                f"{timeout_seconds + 20} seconds. The browser may still be open; check whether the attachments "
                "appeared, then retry the send if needed. No downloaded images were searched or imported."
            ) from exc
        raise RuntimeError(
            f"Timed out waiting for a completed {config['label']} image download after "
            f"{timeout_seconds + 20} seconds."
        ) from exc
    stdout = process.stdout or ""
    stderr = process.stderr or ""
    if process.returncode != 0:
        raise RuntimeError((stderr or stdout or f"Manual browser bridge failed with exit code {process.returncode}.").strip())
    return {
        "provider": provider,
        "provider_label": config["label"],
        "debug_port": port,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "saved_path": _extract_manual_saved_path(stdout),
    }


def _manual_wait_download(payload):
    result = _run_manual_bridge(payload, "wait-download")
    saved_path = result.get("saved_path", "")
    if not saved_path:
        raise RuntimeError("Manual browser download completed, but no saved file path was reported.")
    project_folder = str(payload.get("project_folder", "") or "").strip()
    scene_number = payload.get("scene_number")
    if project_folder and scene_number:
        from .VRGDG_MusicVideoBuilderNodes import _save_scene_image
        scene_result = _save_scene_image({
            "project_folder": project_folder,
            "scene_number": scene_number,
            "source_path": saved_path,
        })
        result["scene_image"] = scene_result
    return result


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


def _manual_import_latest(payload):
    provider = _normalize_provider(payload.get("provider"))
    saved_path = _newest_manual_download(provider)
    result = {
        "provider": provider,
        "provider_label": _PROVIDERS[provider]["label"],
        "saved_path": saved_path,
    }
    project_folder = str(payload.get("project_folder", "") or "").strip()
    scene_number = payload.get("scene_number")
    if project_folder and scene_number:
        from .VRGDG_MusicVideoBuilderNodes import _save_scene_image
        scene_result = _save_scene_image({
            "project_folder": project_folder,
            "scene_number": scene_number,
            "source_path": saved_path,
        })
        result["scene_image"] = scene_result
    return result


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
                "reuse_open_project": _coerce_bool(payload.get("reuse_open_project"), True),
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

    @server_instance.routes.post("/vrgdg/browser_image/manual_open")
    async def vrgdg_browser_image_manual_open(request):
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

    @server_instance.routes.post("/vrgdg/browser_image/manual_upload")
    async def vrgdg_browser_image_manual_upload(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = await asyncio.to_thread(_run_manual_bridge, payload, "upload")
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/browser_image/manual_wait_download")
    async def vrgdg_browser_image_manual_wait_download(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = await asyncio.to_thread(_manual_wait_download, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/browser_image/manual_import_latest")
    async def vrgdg_browser_image_manual_import_latest(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _manual_import_latest(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

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
