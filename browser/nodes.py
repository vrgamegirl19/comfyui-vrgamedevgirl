import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from datetime import datetime
from typing import List, Optional

import torch
from aiohttp import web
from PIL import Image
from server import PromptServer

from ..runner.paths import _resolve_existing_file
from ..runner.api_graph import _prepare_load_image_name


DEFAULT_FLOW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow_automation")
DEFAULT_NODE_VERSION = "v20.15.1"
MAX_FLOW_IMAGES = 50


def _is_debug_chrome_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def _chrome_exe() -> str:
    override = (os.environ.get("VRGDG_CHROME_PATH") or os.environ.get("CHROME_PATH") or "").strip()
    if override:
        override = os.path.expandvars(os.path.expanduser(override))
        if os.path.isfile(override):
            return override
        raise RuntimeError(f"Configured Chrome/Chromium executable was not found: {override}")

    candidates = []
    command_names = []
    if os.name == "nt":
        candidates.extend([
            os.path.join(os.environ.get("ProgramFiles", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
        ])
    elif sys.platform == "darwin":
        candidates.extend([
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ])
        command_names.extend(["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"])
    else:
        candidates.extend([
            "/usr/bin/google-chrome-stable", "/usr/bin/google-chrome", "/usr/bin/chromium",
            "/usr/bin/chromium-browser", "/snap/bin/chromium",
        ])
        command_names.extend(["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"])
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    for command_name in command_names:
        candidate = shutil.which(command_name)
        if candidate:
            return candidate
    raise RuntimeError("Could not find Chrome/Chromium. Install Google Chrome or Chromium, or set VRGDG_CHROME_PATH to the browser executable.")


def _start_debug_chrome(flow_dir: str, port: int, url: str, profile_name: str = "chrome-flow-profile") -> None:
    if _is_debug_chrome_ready(port):
        return

    profile_dir = os.path.join(flow_dir, profile_name)
    os.makedirs(profile_dir, exist_ok=True)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(
        [
            _chrome_exe(),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--window-size=1600,950",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.time() + 25
    while time.time() < deadline:
        if _is_debug_chrome_ready(port):
            return
        time.sleep(0.5)

    raise RuntimeError(f"Chrome debug port {port} did not become ready.")


def _runtime_dir(flow_dir: str) -> str:
    return os.path.join(flow_dir, "runtime")


def _local_node_root(flow_dir: str) -> str:
    return os.path.join(_runtime_dir(flow_dir), "node")


def _find_local_node_exe(flow_dir: str) -> Optional[str]:
    root = _local_node_root(flow_dir)
    if not os.path.isdir(root):
        return None

    direct = os.path.join(root, "node.exe")
    if os.path.isfile(direct):
        return direct

    for name in sorted(os.listdir(root)):
        candidate = os.path.join(root, name, "node.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _find_local_npm_cmd(flow_dir: str) -> Optional[str]:
    node_exe = _find_local_node_exe(flow_dir)
    if not node_exe:
        return None

    npm_cmd = os.path.join(os.path.dirname(node_exe), "npm.cmd")
    if os.path.isfile(npm_cmd):
        return npm_cmd
    return None


def _node_command(flow_dir: str) -> str:
    return _find_local_node_exe(flow_dir) or "node"


def _npm_command(flow_dir: str) -> str:
    return _find_local_npm_cmd(flow_dir) or "npm"


def _ensure_portable_node(flow_dir: str, node_version: str, timeout_seconds: int) -> str:
    existing = _find_local_node_exe(flow_dir)
    if existing:
        return existing

    if os.name != "nt":
        raise RuntimeError(
            "Portable Node auto-install is only implemented for Windows. "
            "Install Node.js on this machine, then run the setup node again."
        )

    version = node_version if isinstance(node_version, str) else DEFAULT_NODE_VERSION
    version = (version or DEFAULT_NODE_VERSION).strip()
    if not version.startswith("v"):
        version = f"v{version}"

    archive_name = f"node-{version}-win-x64.zip"
    url = f"https://nodejs.org/dist/{version}/{archive_name}"
    downloads_dir = os.path.join(_runtime_dir(flow_dir), "downloads")
    node_root = _local_node_root(flow_dir)
    archive_path = os.path.join(downloads_dir, archive_name)

    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(node_root, exist_ok=True)

    try:
        with urllib.request.urlopen(url, timeout=int(timeout_seconds)) as response:
            with open(archive_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        raise RuntimeError(
            "Could not download portable Node.js.\n\n"
            f"URL: {url}\n\n"
            "Check internet access, or install Node.js manually and run setup again."
        ) from exc

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(node_root)
    except Exception as exc:
        raise RuntimeError(f"Could not extract portable Node.js archive: {archive_path}") from exc

    node_exe = _find_local_node_exe(flow_dir)
    npm_cmd = _find_local_npm_cmd(flow_dir)
    if not node_exe or not npm_cmd:
        raise RuntimeError("Portable Node.js extracted, but node.exe/npm.cmd could not be found.")
    return node_exe


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image is None:
        raise ValueError("Cannot save a missing image.")

    if len(image.shape) == 4:
        image = image[0]

    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA image tensor, got shape {tuple(image.shape)}")

    image = (image * 255.0).round().to(torch.uint8).contiguous()
    height = int(image.shape[0])
    width = int(image.shape[1])
    return Image.frombytes("RGB", (width, height), bytes(image.reshape(-1).tolist()))


def _save_input_images(flow_dir: str, images: List[torch.Tensor], prefix: str = "comfy_flow_input") -> List[str]:
    input_dir = os.path.join(flow_dir, "inputs")
    os.makedirs(input_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    for index, image in enumerate(images, start=1):
        path = os.path.join(input_dir, f"{prefix}_{stamp}_{index:02d}.png")
        _tensor_to_pil(image).save(path)
        paths.append(path)
    return paths


def _load_image_as_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        data = torch.tensor(list(image.tobytes()), dtype=torch.float32)
    return (data.reshape(height, width, 3) / 255.0).unsqueeze(0)


def _newest_image_file(output_dir: str, after_time: float) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None

    candidates = []
    for name in os.listdir(output_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        path = os.path.join(output_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime >= after_time:
            candidates.append((mtime, path))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_saved_path(stdout: str) -> Optional[str]:
    for line in reversed((stdout or "").splitlines()):
        marker = "Saved:"
        if marker in line:
            path = line.split(marker, 1)[1].strip()
            if path and os.path.isfile(path):
                return path
    return None


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_int(value, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = int(default)
    if min_value is not None:
        result = max(int(min_value), result)
    if max_value is not None:
        result = min(int(max_value), result)
    return result


def _looks_like_path(value) -> bool:
    if not isinstance(value, str):
        return False
    return ":\\" in value or "/" in value or "\\" in value


class VRGDG_FlowBrowserImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"image{i}": (
                "IMAGE",
                {
                    "tooltip": (
                        f"Optional reference/edit input image #{i}. Set image_count high enough to include this socket. "
                        "Images are uploaded to Flow one at a time and added to the prompt before the text prompt is submitted."
                    )
                },
            )
            for i in range(1, MAX_FLOW_IMAGES + 1)
        }
        optional["reuse_open_project"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "When enabled, non-manual automation reuses the currently open Flow project/tab instead of navigating "
                    "back to the Flow home page and clicking New project for every image."
                ),
            },
        )
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "edit these images into a cinematic final image",
                        "multiline": True,
                        "tooltip": (
                            "Text prompt sent to Google Flow after any connected input images are uploaded and added to the prompt. "
                            "For edits, describe the final image you want Flow to create."
                        ),
                    },
                ),
                "image_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": MAX_FLOW_IMAGES,
                        "step": 1,
                        "tooltip": (
                            "How many image sockets this run should read, starting at image1. "
                            "Use 0 for text-only Flow generation. Increase this when connecting multiple reference images."
                        ),
                    },
                ),
                "debug_port": (
                    "INT",
                    {
                        "default": 9222,
                        "min": 1,
                        "max": 65535,
                        "step": 1,
                        "tooltip": (
                            "Local Chrome remote-debugging port used by the automation. "
                            "Leave 9222 unless another app is already using that port."
                        ),
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 420,
                        "min": 60,
                        "max": 1800,
                        "step": 10,
                        "tooltip": (
                            "Maximum time to let Flow run before ComfyUI gives up. "
                            "Use a larger value for slow generations, uploads, or 2K downloads."
                        ),
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run_flow"
    CATEGORY = "VRGDG/Flow"
    DESCRIPTION = "Automates Google Flow in Chrome. Optional dynamic image inputs are uploaded and added to the prompt; the downloaded result is returned as a ComfyUI IMAGE."

    def run_flow(
        self,
        prompt: str,
        image_count: int,
        debug_port: int,
        timeout_seconds: int,
        **kwargs,
    ):
        flow_dir = DEFAULT_FLOW_DIR
        output_dir = os.path.join(flow_dir, "outputs")
        if _looks_like_path(debug_port):
            debug_port = timeout_seconds
            timeout_seconds = 420
        debug_port = _coerce_int(debug_port, 9222, 1, 65535)
        timeout_seconds = _coerce_int(timeout_seconds, 420, 60, 1800)

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        script_path = os.path.join(flow_dir, "flow-poc.mjs")
        if not os.path.isfile(script_path):
            raise RuntimeError(f"Flow automation script not found: {script_path}")

        playwright_dir = os.path.join(flow_dir, "node_modules", "playwright")
        if not os.path.isdir(playwright_dir):
            raise RuntimeError(
                "Flow automation dependencies are not installed.\n\n"
                "Add and run this ComfyUI node once:\n"
                "VRGDG Flow Browser Setup\n\n"
                "That setup node can install portable Node.js and Playwright automatically."
            )

        count = _coerce_int(image_count, 1, 0, MAX_FLOW_IMAGES)
        reuse_open_project = _coerce_bool(kwargs.get("reuse_open_project"), True)
        input_images = []
        for index in range(1, count + 1):
            image = kwargs.get(f"image{index}")
            if image is not None:
                input_images.append(image)

        image_paths = _save_input_images(flow_dir, input_images) if input_images else []
        url = "https://labs.google/fx/tools/flow"
        _start_debug_chrome(flow_dir, debug_port, url)

        started_at = time.time()
        command = [
            _node_command(flow_dir),
            script_path,
            "--url",
            url,
            "--prompt",
            prompt or "",
            "--out",
            output_dir,
            "--connect-cdp",
            f"http://127.0.0.1:{debug_port}",
        ]
        if reuse_open_project:
            command.append("--no-navigate")
        for image_path in image_paths:
            command.extend(["--image", image_path])

        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        process = subprocess.run(
            command,
            cwd=flow_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 120,
            env=env,
        )

        stdout = process.stdout or ""
        stderr = process.stderr or ""
        saved_path = _extract_saved_path(stdout) or _newest_image_file(output_dir, started_at)

        if process.returncode != 0 and not saved_path:
            raise RuntimeError(
                "Flow automation failed.\n\n"
                f"Command: {' '.join(command)}\n\n"
                f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
            )

        if not saved_path:
            raise RuntimeError(
                "Flow automation completed, but no output image was found.\n\n"
                f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
            )

        return (_load_image_as_tensor(saved_path),)


class VRGDG_FlowBrowserSetup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "install_portable_node": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When enabled, setup downloads a private portable Node.js into this custom node folder. "
                            "This lets users install without opening a command prompt or installing Node globally."
                        ),
                    },
                ),
                "node_version": (
                    "STRING",
                    {
                        "default": DEFAULT_NODE_VERSION,
                        "multiline": False,
                        "tooltip": (
                            "Portable Node.js version to download. Leave this alone unless you specifically need another Node 18+ version."
                        ),
                    },
                ),
                "install_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When enabled, setup installs the JavaScript dependencies used by the Flow browser automation. "
                            "Usually leave this on and run the setup node once."
                        ),
                    },
                ),
                "strict_ssl": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Controls npm certificate checking while installing dependencies. "
                            "False is more forgiving on machines with certificate issues; true is stricter."
                        ),
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 600,
                        "min": 30,
                        "max": 1800,
                        "step": 30,
                        "tooltip": (
                            "Maximum time allowed for setup downloads and installs. "
                            "Increase this on slower internet connections."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "setup"
    CATEGORY = "VRGDG/Flow"
    DESCRIPTION = "Checks and optionally installs portable Node.js plus the Playwright dependencies used by the VRGDG Flow Browser nodes."

    def setup(
        self,
        install_portable_node: bool,
        node_version: str,
        install_if_missing: bool,
        strict_ssl: bool,
        timeout_seconds: int,
    ):
        flow_dir = DEFAULT_FLOW_DIR
        if _looks_like_path(install_portable_node) and isinstance(node_version, bool):
            legacy_install_portable_node = node_version
            legacy_node_version = install_if_missing
            legacy_install_if_missing = strict_ssl
            install_portable_node = legacy_install_portable_node
            node_version = legacy_node_version if isinstance(legacy_node_version, str) else DEFAULT_NODE_VERSION
            install_if_missing = _coerce_bool(legacy_install_if_missing, True)
            strict_ssl = False

        install_portable_node = _coerce_bool(install_portable_node, True)
        install_if_missing = _coerce_bool(install_if_missing, True)
        strict_ssl = _coerce_bool(strict_ssl, False)
        timeout_seconds = _coerce_int(timeout_seconds, 600, 30, 1800)

        package_path = os.path.join(flow_dir, "package.json")
        playwright_dir = os.path.join(flow_dir, "node_modules", "playwright")

        lines = [
            "VRGDG Flow Browser Setup",
            f"flow_dir: {flow_dir}",
        ]

        if not os.path.isfile(package_path):
            raise RuntimeError(f"Flow automation package.json not found: {package_path}")

        node_exe = _find_local_node_exe(flow_dir)
        npm_cmd = _find_local_npm_cmd(flow_dir)
        if node_exe and npm_cmd:
            lines.append(f"Portable Node.js is ready: {node_exe}")
        elif install_portable_node and os.name == "nt":
            lines.append(f"Installing portable Node.js {node_version or DEFAULT_NODE_VERSION}...")
            node_exe = _ensure_portable_node(flow_dir, node_version, timeout_seconds)
            npm_cmd = _find_local_npm_cmd(flow_dir)
            lines.append(f"Portable Node.js installed: {node_exe}")
        else:
            if os.name == "nt":
                lines.append("Portable Node.js is missing.")
                lines.append("Setup will try system npm instead.")
            else:
                lines.append("Linux/macOS uses system Node.js and npm.")

        if os.path.isdir(playwright_dir):
            lines.append("Playwright dependency is already installed.")
            return ("\n".join(lines),)

        if not install_if_missing:
            lines.append("Playwright dependency is missing.")
            lines.append("Enable install_if_missing and run this node once.")
            return ("\n".join(lines),)

        command = [_npm_command(flow_dir), "install"]
        if not strict_ssl:
            command.append("--strict-ssl=false")

        lines.append(f"Running: {' '.join(command)}")
        try:
            process = subprocess.run(
                command,
                cwd=flow_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            message = ("npm was not found. Enable install_portable_node and run this setup node again." if os.name == "nt" else "System Node.js/npm was not found. Install Node.js and npm, or set PATH before starting ComfyUI.")
            raise RuntimeError(message) from exc

        stdout = process.stdout or ""
        stderr = process.stderr or ""
        if process.returncode != 0:
            raise RuntimeError(
                "npm install failed.\n\n"
                f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
            )

        if not os.path.isdir(playwright_dir):
            raise RuntimeError(
                "npm install completed, but node_modules/playwright was not found.\n\n"
                f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
            )

        lines.append("Install complete.")
        if stdout.strip():
            lines.append("")
            lines.append(stdout.strip()[-2000:])
        return ("\n".join(lines),)


class VRGDG_ChatGPTImagesBrowser:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"image{i}": (
                "IMAGE",
                {
                    "tooltip": (
                        f"Optional image attachment #{i}. Set image_count high enough to include this socket. "
                        "Images are attached to ChatGPT through the paperclip before the prompt is submitted."
                    )
                },
            )
            for i in range(1, MAX_FLOW_IMAGES + 1)
        }
        optional["reuse_open_project"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "When enabled, non-manual automation reuses the current ChatGPT Images tab/conversation instead of "
                    "navigating back to the images start page for every image."
                ),
            },
        )
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "create a cinematic image",
                        "multiline": True,
                        "tooltip": (
                            "Prompt to paste into ChatGPT Images. Connected images are attached first, then this prompt is submitted."
                        ),
                    },
                ),
                "image_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_FLOW_IMAGES,
                        "step": 1,
                        "tooltip": (
                            "How many image sockets this run should attach, starting at image1. Use 0 for text-only generation."
                        ),
                    },
                ),
                "debug_port": (
                    "INT",
                    {
                        "default": 9223,
                        "min": 1,
                        "max": 65535,
                        "step": 1,
                        "tooltip": (
                            "Local Chrome remote-debugging port for ChatGPT Images automation. "
                            "Default 9223 keeps it separate from the Flow node's 9222 port."
                        ),
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 600,
                        "min": 60,
                        "max": 2400,
                        "step": 10,
                        "tooltip": (
                            "Maximum time to let ChatGPT create and download the image. Increase for slow generations."
                        ),
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run_chatgpt_images"
    CATEGORY = "VRGDG/ChatGPT"
    DESCRIPTION = "Automates ChatGPT Images in Chrome. Optional dynamic image inputs are attached before the prompt; the downloaded image is returned as a ComfyUI IMAGE."

    def run_chatgpt_images(
        self,
        prompt: str,
        image_count: int,
        debug_port: int,
        timeout_seconds: int,
        **kwargs,
    ):
        flow_dir = DEFAULT_FLOW_DIR
        if _looks_like_path(debug_port):
            debug_port = timeout_seconds
            timeout_seconds = 600
        debug_port = _coerce_int(debug_port, 9223, 1, 65535)
        timeout_seconds = _coerce_int(timeout_seconds, 600, 60, 2400)
        output_dir = os.path.abspath(os.path.join(flow_dir, "chatgpt_outputs"))
        os.makedirs(output_dir, exist_ok=True)

        script_path = os.path.join(flow_dir, "chatgpt-images-poc.mjs")
        if not os.path.isfile(script_path):
            raise RuntimeError(f"ChatGPT Images automation script not found: {script_path}")

        playwright_dir = os.path.join(flow_dir, "node_modules", "playwright")
        if not os.path.isdir(playwright_dir):
            raise RuntimeError(
                "Browser automation dependencies are not installed.\n\n"
                "Add and run this ComfyUI node once:\n"
                "VRGDG Flow Browser Setup\n\n"
                "That setup node installs the shared portable Node.js and Playwright dependencies."
            )

        count = _coerce_int(image_count, 0, 0, MAX_FLOW_IMAGES)
        reuse_open_project = _coerce_bool(kwargs.get("reuse_open_project"), True)
        input_images = []
        for index in range(1, count + 1):
            image = kwargs.get(f"image{index}")
            if image is not None:
                input_images.append(image)

        image_paths = _save_input_images(flow_dir, input_images, prefix="comfy_chatgpt_input") if input_images else []
        url = "https://chatgpt.com/images"
        _start_debug_chrome(flow_dir, debug_port, url, profile_name="chrome-chatgpt-profile")

        started_at = time.time()
        command = [
            _node_command(flow_dir),
            script_path,
            "--url",
            url,
            "--prompt",
            prompt or "",
            "--out",
            output_dir,
            "--connect-cdp",
            f"http://127.0.0.1:{debug_port}",
            "--timeout",
            str(max(30_000, (timeout_seconds - 30) * 1000)),
        ]
        if reuse_open_project:
            command.append("--no-navigate")
        for image_path in image_paths:
            command.extend(["--image", image_path])

        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        process = subprocess.run(
            command,
            cwd=flow_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )

        stdout = process.stdout or ""
        stderr = process.stderr or ""
        saved_path = _extract_saved_path(stdout) or _newest_image_file(output_dir, started_at)

        if process.returncode != 0 and not saved_path:
            raise RuntimeError(
                "ChatGPT Images automation failed.\n\n"
                f"Command: {' '.join(command)}\n\n"
                f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
            )

        if not saved_path:
            raise RuntimeError(
                "ChatGPT Images automation completed, but no output image was found.\n\n"
                f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
            )

        return (_load_image_as_tensor(saved_path),)


class VRGDG_MetaAIBrowserImage:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"image{i}": (
                "IMAGE",
                {
                    "tooltip": (
                        f"Optional image attachment #{i}. Set image_count high enough to include this socket. "
                        "Images are attached to Meta AI with the plus/upload control before the prompt is submitted."
                    )
                },
            )
            for i in range(1, MAX_FLOW_IMAGES + 1)
        }
        optional["reuse_open_project"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "When enabled, automation reuses the current Meta AI tab/conversation instead of navigating "
                    "back to the Meta AI start page for every image."
                ),
            },
        )
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "create a cinematic image",
                        "multiline": True,
                        "tooltip": (
                            "Prompt to paste into Meta AI. Connected images are attached first, then this prompt is submitted."
                        ),
                    },
                ),
                "image_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_FLOW_IMAGES,
                        "step": 1,
                        "tooltip": (
                            "How many image sockets this run should attach, starting at image1. Use 0 for text-only generation."
                        ),
                    },
                ),
                "debug_port": (
                    "INT",
                    {
                        "default": 9224,
                        "min": 1,
                        "max": 65535,
                        "step": 1,
                        "tooltip": (
                            "Local Chrome remote-debugging port for Meta AI automation. "
                            "Default 9224 keeps it separate from Flow and GPT Image."
                        ),
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 600,
                        "min": 60,
                        "max": 2400,
                        "step": 10,
                        "tooltip": (
                            "Maximum time to let Meta AI create and download the image. Increase for slow generations."
                        ),
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run_meta_ai"
    CATEGORY = "VRGDG/Meta AI"
    DESCRIPTION = "Automates Meta AI image generation in Chrome. Optional dynamic image inputs are attached before the prompt; the downloaded image is returned as a ComfyUI IMAGE."

    def run_meta_ai(
        self,
        prompt: str,
        image_count: int,
        debug_port: int,
        timeout_seconds: int,
        **kwargs,
    ):
        flow_dir = DEFAULT_FLOW_DIR
        if _looks_like_path(debug_port):
            debug_port = timeout_seconds
            timeout_seconds = 600
        debug_port = _coerce_int(debug_port, 9224, 1, 65535)
        timeout_seconds = _coerce_int(timeout_seconds, 600, 60, 2400)
        output_dir = os.path.abspath(os.path.join(flow_dir, "meta_outputs"))
        os.makedirs(output_dir, exist_ok=True)

        script_path = os.path.join(flow_dir, "meta-ai-poc.mjs")
        if not os.path.isfile(script_path):
            raise RuntimeError(f"Meta AI automation script not found: {script_path}")

        playwright_dir = os.path.join(flow_dir, "node_modules", "playwright")
        if not os.path.isdir(playwright_dir):
            raise RuntimeError(
                "Browser automation dependencies are not installed.\n\n"
                "Add and run this ComfyUI node once:\n"
                "VRGDG Flow Browser Setup\n\n"
                "That setup node installs the shared portable Node.js and Playwright dependencies."
            )

        count = _coerce_int(image_count, 0, 0, MAX_FLOW_IMAGES)
        reuse_open_project = _coerce_bool(kwargs.get("reuse_open_project"), True)
        input_images = []
        for index in range(1, count + 1):
            image = kwargs.get(f"image{index}")
            if image is not None:
                input_images.append(image)

        image_paths = _save_input_images(flow_dir, input_images, prefix="comfy_meta_input") if input_images else []
        url = "https://www.meta.ai/"
        _start_debug_chrome(flow_dir, debug_port, url, profile_name="chrome-meta-profile")

        started_at = time.time()
        command = [
            _node_command(flow_dir),
            script_path,
            "--url",
            url,
            "--prompt",
            prompt or "",
            "--out",
            output_dir,
            "--timeout",
            str(timeout_seconds * 1000),
            "--connect-cdp",
            f"http://127.0.0.1:{debug_port}",
        ]
        if reuse_open_project:
            command.append("--no-navigate")
        for image_path in image_paths:
            command.extend(["--image", image_path])

        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        process = subprocess.run(
            command,
            cwd=flow_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )

        stdout = process.stdout or ""
        stderr = process.stderr or ""
        saved_path = _extract_saved_path(stdout) or _newest_image_file(output_dir, started_at)

        if process.returncode != 0 and not saved_path:
            raise RuntimeError(
                "Meta AI automation failed.\n\n"
                f"Command: {' '.join(command)}\n\n"
                f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
            )

        if not saved_path:
            raise RuntimeError(
                "Meta AI automation completed, but no output image was found.\n\n"
                f"STDOUT:\n{stdout[-4000:]}\n\nSTDERR:\n{stderr[-4000:]}"
            )

        return (_load_image_as_tensor(saved_path),)


_VRGDG_BROWSER_IMAGE_ROUTES_REGISTERED = False
_DOWNLOAD_KEEPERS = {}
_DOWNLOAD_KEEPERS_LOCK = threading.Lock()

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


def _save_image_data_url(folder, data_url, name):
    text = str(data_url or "").strip()
    if not text:
        return ""
    match = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", text, re.DOTALL)
    if not match:
        raise ValueError("Manual reference image data must be an image data URL.")
    raw = base64.b64decode(match.group(2), validate=False)
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


def _save_manual_data_url(flow_dir, data_url, name):
    return _save_image_data_url(os.path.join(flow_dir, "manual_refs"), data_url, name)


def _safe_browser_group_name(value, fallback="Group"):
    text = re.sub(r"[^A-Za-z0-9 +_.-]+", "_", str(value or "").strip()).strip(" ._-")
    return (text or fallback)[:90]


def _project_browser_folder(payload, category, include_group=True):
    project_folder = str(payload.get("project_folder", "") or "").strip()
    if not project_folder:
        raise ValueError("Create or load a Video Builder project before using a project Browser AI folder.")
    project_folder = os.path.abspath(os.path.expanduser(project_folder))
    folder = os.path.join(project_folder, category)
    if include_group:
        folder = os.path.join(folder, _safe_browser_group_name(payload.get("group_name"), "Group"))
    os.makedirs(folder, exist_ok=True)
    return folder


def _store_project_browser_reference(payload):
    data_url = str(payload.get("image_data", "") or "").strip()
    if not data_url:
        raise ValueError("Browser AI reference image data is missing.")
    reference_type = str(payload.get("reference_type", "group") or "group").strip().lower()
    if reference_type == "location":
        folder = _project_browser_folder(payload, "Browser AI References", include_group=False)
        folder = os.path.join(folder, "Locations")
        os.makedirs(folder, exist_ok=True)
    else:
        folder = _project_browser_folder(payload, os.path.join("Browser AI References", "Groups"))
    saved_path = _save_image_data_url(folder, data_url, payload.get("name") or "reference.png")
    return {
        "saved_path": saved_path,
        "name": os.path.basename(saved_path),
        "reference_type": reference_type,
    }


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


def _manual_output_dir(payload, provider, action):
    redirect_downloads = action == "submit" and _coerce_bool(payload.get("redirect_downloads_to_project"), False)
    if redirect_downloads:
        folder = _project_browser_folder(payload, "Browser AI Images", include_group=False)
        set_name = payload.get("download_set_name") or payload.get("group_name")
        folder = os.path.join(folder, _safe_browser_group_name(set_name, "Group"))
        location_name = os.path.splitext(os.path.basename(str(payload.get("download_location_name", "") or "")))[0]
        folder = os.path.join(folder, _safe_browser_group_name(location_name, "No Location"))
        os.makedirs(folder, exist_ok=True)
        return folder, True
    user_profile = str(os.environ.get("USERPROFILE", "") or "").strip()
    normal_downloads = os.path.join(user_profile, "Downloads") if user_profile else os.path.join(os.path.expanduser("~"), "Downloads")
    output_dir = normal_downloads if os.path.isdir(normal_downloads) else os.path.join(DEFAULT_FLOW_DIR, "manual_downloads", provider)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, False


def _stop_download_keeper_locked(port):
    keeper = _DOWNLOAD_KEEPERS.pop(port, None)
    if not keeper:
        return ""
    process = keeper["process"]
    if process.poll() is not None:
        return ""
    try:
        process.stdin.write("stop\n")
        process.stdin.flush()
        stdout, _ = process.communicate(timeout=15)
        return (stdout or "").strip()
    except (BrokenPipeError, OSError):
        return ""
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return ""


def _stop_download_keeper(port):
    with _DOWNLOAD_KEEPERS_LOCK:
        return _stop_download_keeper_locked(port)


def _start_download_keeper(flow_dir, script_path, provider, config, port, output_dir, env):
    normalized_output_dir = os.path.normcase(os.path.abspath(output_dir))
    with _DOWNLOAD_KEEPERS_LOCK:
        existing = _DOWNLOAD_KEEPERS.get(port)
        if existing and existing["process"].poll() is None:
            if existing["output_dir"] == normalized_output_dir:
                return
            _stop_download_keeper_locked(port)
        elif existing:
            _DOWNLOAD_KEEPERS.pop(port, None)

        command = [
            _node_command(flow_dir),
            script_path,
            "--provider",
            provider,
            "--action",
            "hold-downloads",
            "--url",
            config["url"],
            "--out",
            output_dir,
            "--connect-cdp",
            f"http://127.0.0.1:{port}",
        ]
        process = subprocess.Popen(
            command,
            cwd=flow_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        ready = threading.Event()
        output = []

        def read_until_ready():
            try:
                for line in process.stdout:
                    output.append(line)
                    if line.strip().startswith("DOWNLOAD_OVERRIDE_READY:"):
                        ready.set()
                        return
            finally:
                ready.set()

        reader = threading.Thread(target=read_until_ready, daemon=True)
        reader.start()
        if not ready.wait(timeout=30) or process.poll() is not None or not any(
            line.strip().startswith("DOWNLOAD_OVERRIDE_READY:") for line in output
        ):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            detail = "".join(output).strip()
            raise RuntimeError(detail or f"Could not keep browser downloads redirected to {output_dir}.")
        _DOWNLOAD_KEEPERS[port] = {
            "process": process,
            "output_dir": normalized_output_dir,
        }


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
    output_dir, redirect_downloads = _manual_output_dir(payload, provider, action)
    if action == "finish":
        _stop_download_keeper(port)
    if action == "finish" and not _is_debug_chrome_ready(port):
        return {
            "provider": provider,
            "provider_label": config["label"],
            "debug_port": port,
            "stdout": "Browser was already closed; its temporary download override is no longer active.",
            "stderr": "",
            "saved_path": "",
            "download_path": "",
            "redirect_downloads_to_project": False,
        }
    browser_was_ready = _is_debug_chrome_ready(port)
    _start_debug_chrome(flow_dir, port, config["url"], profile_name=config["profile_name"])

    # Starting the controlled browser already creates its first provider tab.
    # Callers can explicitly request another tab, but the Builder's group
    # sequence reuses one tab so providers never receive parallel requests.
    requested_new_tab = action == "submit" and _coerce_bool(payload.get("open_new_tab"), False)
    open_new_tab = requested_new_tab and browser_was_ready
    fresh_request = action == "submit" and _coerce_bool(payload.get("fresh_request"), False)

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
    if action in {"upload", "submit"}:
        for image_path in _manual_image_paths(payload):
            command.extend(["--image", image_path])
        prompt = str(payload.get("prompt", "") or "").strip()
        if prompt:
            command.extend(["--prompt", prompt])
    if action == "submit":
        command.extend(["--redirect-downloads", "true" if redirect_downloads else "false"])
        command.extend(["--new-tab", "true" if open_new_tab else "false"])
        command.extend(["--fresh-request", "true" if fresh_request else "false"])

    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    if action == "submit":
        # A short-lived Playwright/CDP submission connection can reset Chrome's
        # download behavior when it disconnects. Stop the previous keeper now,
        # then apply the new destination only after that helper has fully exited.
        _stop_download_keeper(port)
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
        if action in {"upload", "submit"}:
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
    if action == "submit" and redirect_downloads:
        _start_download_keeper(flow_dir, script_path, provider, config, port, output_dir, env)
    return {
        "provider": provider,
        "provider_label": config["label"],
        "debug_port": port,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "saved_path": _extract_manual_saved_path(stdout),
        "download_path": output_dir if redirect_downloads else "",
        "redirect_downloads_to_project": redirect_downloads,
        "opened_new_tab": open_new_tab,
        "reused_provider_tab": action == "submit" and not open_new_tab,
    }


def _manual_wait_download(payload):
    result = _run_manual_bridge(payload, "wait-download")
    saved_path = result.get("saved_path", "")
    if not saved_path:
        raise RuntimeError("Manual browser download completed, but no saved file path was reported.")
    project_folder = str(payload.get("project_folder", "") or "").strip()
    scene_number = payload.get("scene_number")
    if project_folder and scene_number:
        from ..builder.media import _save_scene_image
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
        from ..builder.media import _save_scene_image
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
            result = await asyncio.to_thread(_install_browser_image_deps, payload)
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
            await asyncio.to_thread(_start_debug_chrome, DEFAULT_FLOW_DIR, port, config["url"], profile_name=config["profile_name"])
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
            await asyncio.to_thread(_start_debug_chrome, DEFAULT_FLOW_DIR, port, config["url"], profile_name=config["profile_name"])
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

    @server_instance.routes.post("/vrgdg/browser_image/manual_submit")
    async def vrgdg_browser_image_manual_submit(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = await asyncio.to_thread(_run_manual_bridge, payload, "submit")
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/browser_image/manual_finish")
    async def vrgdg_browser_image_manual_finish(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = await asyncio.to_thread(_run_manual_bridge, payload, "finish")
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/browser_image/store_reference")
    async def vrgdg_browser_image_store_reference(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = await asyncio.to_thread(_store_project_browser_reference, payload)
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


NODE_CLASS_MAPPINGS = {
    "VRGDG_FlowBrowserImageEdit": VRGDG_FlowBrowserImageEdit,
    "VRGDG_FlowBrowserSetup": VRGDG_FlowBrowserSetup,
    "VRGDG_ChatGPTImagesBrowser": VRGDG_ChatGPTImagesBrowser,
    "VRGDG_MetaAIBrowserImage": VRGDG_MetaAIBrowserImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_FlowBrowserImageEdit": "VRGDG Flow Browser Image Edit",
    "VRGDG_FlowBrowserSetup": "VRGDG Flow Browser Setup",
    "VRGDG_ChatGPTImagesBrowser": "VRGDG ChatGPT Images Browser",
    "VRGDG_MetaAIBrowserImage": "VRGDG Meta AI Browser Image",
}
