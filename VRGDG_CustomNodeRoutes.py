"""Safe, allowlisted installer/status routes for Video Builder dependencies."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from aiohttp import web
from server import PromptServer


_NODE_DIR = Path(__file__).resolve().parent
_CUSTOM_NODES_DIR = _NODE_DIR.parent
_ROUTES_REGISTERED = False

# Keep this list deliberately allowlisted. The browser can request an id, never
# an arbitrary URL or filesystem path.
VIDEO_BUILDER_CUSTOM_NODES = {
    "videohelpersuite": {
        "folder": "ComfyUI-VideoHelperSuite",
        "folders": ("ComfyUI-VideoHelperSuite", "comfyui-videohelpersuite"),
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    },
    "kjnodes": {
        "folder": "ComfyUI-KJNodes",
        "folders": ("ComfyUI-KJNodes", "comfyui-kjnodes"),
        "url": "https://github.com/kijai/ComfyUI-KJNodes.git",
    },
    "gguf": {
        "folder": "ComfyUI-GGUF",
        "folders": ("ComfyUI-GGUF", "comfyui-gguf"),
        "url": "https://github.com/city96/ComfyUI-GGUF.git",
    },
    "ltxvideo": {
        "folder": "ComfyUI-LTXVideo",
        "folders": ("ComfyUI-LTXVideo", "comfyui-ltxvideo"),
        "url": "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
    },
    "te_speed_minimax_h3": {
        "folder": "TE-Speed-MiniMaxH3-OSS",
        "url": "https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS.git",
    },
    "mmh3_ultimate_upscale": {
        "folder": "Comfyui-MMH3-UltimateUpscale",
        "url": "https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale.git",
    },
}


def _node_status(item):
    candidates = [item["folder"], *item.get("folders", ())]
    path = next((_CUSTOM_NODES_DIR / name for name in candidates if (_CUSTOM_NODES_DIR / name).is_dir()), _CUSTOM_NODES_DIR / item["folder"])
    return {"id": item["id"], "folder": path.name, "installed": path.is_dir(), "path": str(path)}


def _run(command, cwd, timeout=900):
    try:
        result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True,
                                errors="replace", timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("Git was not found. Install Git or use ComfyUI Manager.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out: {' '.join(command)}") from exc
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{output or 'No output.'}")
    return output


def _install(ids):
    requested = [VIDEO_BUILDER_CUSTOM_NODES[item_id] | {"id": item_id}
                 for item_id in ids if item_id in VIDEO_BUILDER_CUSTOM_NODES]
    if not requested:
        raise RuntimeError("No recognized Video Builder custom nodes were selected.")
    results = []
    for item in requested:
        existing = next((_CUSTOM_NODES_DIR / name for name in (item["folder"], *item.get("folders", ())) if (_CUSTOM_NODES_DIR / name).is_dir()), None)
        destination = existing or (_CUSTOM_NODES_DIR / item["folder"])
        if not destination.is_dir():
            _run(["git", "clone", "--depth", "1", item["url"], str(destination)], _CUSTOM_NODES_DIR)
            action = "installed"
        else:
            action = "already installed"
        requirements = destination / "requirements.txt"
        requirement_output = ""
        if requirements.is_file():
            requirement_output = _run([os.fspath(sys.executable), "-m", "pip", "install", "-r", str(requirements)], destination)
        results.append({"id": item["id"], "folder": item["folder"], "action": action,
                        "requirements_installed": requirements.is_file(), "output": requirement_output[-1200:]})
    return results


def _register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    @PromptServer.instance.routes.get("/vrgdg/video_builder/custom_nodes/status")
    async def custom_nodes_status(request):
        return web.json_response({"ok": True, "custom_nodes_dir": str(_CUSTOM_NODES_DIR), "nodes": [
            _node_status(item | {"id": item_id})
            for item_id, item in VIDEO_BUILDER_CUSTOM_NODES.items()
        ]})

    @PromptServer.instance.routes.post("/vrgdg/video_builder/custom_nodes/install")
    async def custom_nodes_install(request):
        try:
            payload = await request.json()
            ids = payload.get("ids", []) if isinstance(payload, dict) else []
            if not isinstance(ids, list):
                raise RuntimeError("ids must be a list.")
            result = await asyncio.to_thread(_install, ids)
            return web.json_response({"ok": True, "results": result, "restart_required": True})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    _ROUTES_REGISTERED = True


_register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
