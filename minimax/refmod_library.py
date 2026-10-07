"""The saved RefMods on disk: list them with their metadata and previews, without loading any tensors.

The folder a RefMod lives in is its type (``models/refmods/identity/darrel.safetensors`` is an ``identity`` RefMod), so a
mod's name is its path below the RefMod folder without the extension (``identity/darrel``). Previews are small PNG
files stored next to the mod as ``<name>.preview.png``.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from .refmod_picker import _read_safetensors_header, _refmod_dirs

PREVIEW_SUFFIX = ".preview.png"
PREVIEW_MAX_SIDE = 384
_SKIP_DIRS = {"graph_presets", ".git", "__pycache__"}


def _meta_from_header(header: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = header.get("__metadata__") or {}
    for key in ("refmod_meta", "audio_refmod_meta"):
        if key in raw:
            try:
                return json.loads(raw[key])
            except ValueError:
                return None
    return None


def _entry(name: str, path: str, directory: str) -> Optional[Dict[str, Any]]:
    try:
        header = _read_safetensors_header(path)
    except (OSError, ValueError):
        return None
    meta = _meta_from_header(header)
    if not isinstance(meta, dict) or meta.get("kind") not in ("image", "video"):
        return None
    try:
        frames = int(meta.get("latent_t") or 1)
        height, width = int(meta.get("latent_h") or 0), int(meta.get("latent_w") or 0)
    except (TypeError, ValueError):
        return None
    folder = name.split("/")[0] if "/" in name else ""
    preview = os.path.splitext(path)[0] + PREVIEW_SUFFIX
    return {
        "name": name,
        "folder": folder,
        "type": str(meta.get("concept_type") or folder or "generic"),
        "kind": meta["kind"],
        "frames": frames,
        "canvas": [width * 16, height * 16],
        "tokens": frames * (height // 2) * (width // 2),
        "description": str(meta.get("description") or ""),
        "tags": [str(tag) for tag in (meta.get("tags") or [])],
        "mode": str(meta.get("mode") or ""),
        "has_preview": os.path.isfile(preview),
        "path": path,
        "size": os.path.getsize(path),
        "directory": directory,
    }


def list_refmods(folder: str = "") -> List[Dict[str, Any]]:
    """Every readable visual RefMod, optionally only those in one folder (type). Sorted by name."""
    seen = set()
    entries: List[Dict[str, Any]] = []
    for directory in _refmod_dirs():
        for root, subdirs, files in os.walk(directory):
            subdirs[:] = sorted(name for name in subdirs if name not in _SKIP_DIRS)
            for filename in sorted(files):
                if not filename.endswith(".safetensors"):
                    continue
                path = os.path.join(root, filename)
                name = os.path.relpath(path[: -len(".safetensors")], directory).replace("\\", "/")
                if name in seen:
                    continue
                if folder and name.split("/")[0] != folder:
                    continue
                entry = _entry(name, path, directory)
                if entry:
                    seen.add(name)
                    entries.append(entry)
    return sorted(entries, key=lambda item: item["name"].lower())


def find_refmod(name: str) -> Optional[Dict[str, Any]]:
    """One RefMod by name, or None. Rejects names that could leave the RefMod folders."""
    text = str(name or "").replace("\\", "/").strip()
    if not text or text.startswith("/") or ":" in text or ".." in text.split("/"):
        return None
    for directory in _refmod_dirs():
        path = os.path.join(directory, text + ".safetensors")
        if os.path.isfile(path):
            return _entry(text, path, directory)
    return None


def preview_path(name: str) -> Optional[str]:
    """Where the preview PNG is (or would be) for a RefMod, or None for an unknown name."""
    entry = find_refmod(name)
    return os.path.splitext(entry["path"])[0] + PREVIEW_SUFFIX if entry else None


def save_preview(image, destination: str) -> None:
    """Write a PIL image as a preview PNG no larger than ``PREVIEW_MAX_SIDE`` on its longest side."""
    from PIL import Image

    preview = image.convert("RGB")
    scale = min(1.0, PREVIEW_MAX_SIDE / max(preview.size))
    if scale < 1.0:
        preview = preview.resize((max(1, round(preview.width * scale)), max(1, round(preview.height * scale))), Image.LANCZOS)
    preview.save(destination, format="PNG", optimize=True)


def _register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return

    @instance.routes.get("/vrgdg/refmod/library")
    async def vrgdg_refmod_library(request):
        try:
            entries = await asyncio.to_thread(list_refmods, str(request.query.get("folder", "") or ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        public = [{key: value for key, value in entry.items() if key not in ("path", "directory")} for entry in entries]
        return web.json_response({"ok": True, "refmods": public})

    @instance.routes.get("/vrgdg/refmod/preview")
    async def vrgdg_refmod_preview(request):
        path = await asyncio.to_thread(preview_path, str(request.query.get("name", "") or ""))
        if not path or not os.path.isfile(path):
            return web.json_response({"ok": False, "error": "No preview for this RefMod."}, status=404)
        return web.FileResponse(path, headers={"Cache-Control": "no-cache"})


_register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
