import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from typing import Optional



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
