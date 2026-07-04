import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from datetime import datetime
from typing import List, Optional

import torch
from PIL import Image


DEFAULT_FLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow_automation")
DEFAULT_NODE_VERSION = "v20.15.1"
MAX_FLOW_IMAGES = 50


def _is_debug_chrome_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def _chrome_exe() -> str:
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise RuntimeError("Could not find chrome.exe in the standard Program Files locations.")


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
        elif install_portable_node:
            lines.append(f"Installing portable Node.js {node_version or DEFAULT_NODE_VERSION}...")
            node_exe = _ensure_portable_node(flow_dir, node_version, timeout_seconds)
            npm_cmd = _find_local_npm_cmd(flow_dir)
            lines.append(f"Portable Node.js installed: {node_exe}")
        else:
            lines.append("Portable Node.js is missing.")
            lines.append("Setup will try system npm instead.")

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
            raise RuntimeError(
                "npm was not found. Enable install_portable_node and run this setup node again."
            ) from exc

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
        ]
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


NODE_CLASS_MAPPINGS = {
    "VRGDG_FlowBrowserImageEdit": VRGDG_FlowBrowserImageEdit,
    "VRGDG_FlowBrowserSetup": VRGDG_FlowBrowserSetup,
    "VRGDG_ChatGPTImagesBrowser": VRGDG_ChatGPTImagesBrowser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_FlowBrowserImageEdit": "VRGDG Flow Browser Image Edit",
    "VRGDG_FlowBrowserSetup": "VRGDG Flow Browser Setup",
    "VRGDG_ChatGPTImagesBrowser": "VRGDG ChatGPT Images Browser",
}
