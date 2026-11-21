"""Allowlisted model downloads into ComfyUI's model folders, for the Models dialog.

The catalog of what can be installed is ``model_catalog.json`` (MiniMax H3, image models, LLM models and the DMAD and
MOSS-Image LoRAs). The browser names a model by id. It never sends a URL or a path, so it cannot make this module fetch
anything else or write anywhere else. A download goes to ``<name>.part`` first, is checked, and only then gets its real
name, so ComfyUI never lists a half-written file and a failed download leaves nothing behind. Files that already exist
are never replaced.
"""

import json
import os
import shutil
import struct
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import folder_paths

from ..llm.http_tls import urlopen_verified
from .hf_token import load_token
from .model_paths import register_custom_model_root

_CHUNK_BYTES = 1024 * 1024
# A download needs this much free space on top of the file itself.
_FREE_SPACE_MARGIN_BYTES = 256 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024

_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_catalog.json")
# GGUF files start with these four bytes.
_GGUF_MAGIC = b"GGUF"
_HUGGING_FACE_URL = "https://huggingface.co/"
# What a gated file says while no Hugging Face token is saved, and what a download says when Hugging Face turns the saved one down.
_NEEDS_TOKEN = (
    "Hugging Face asks for a login before it gives this file out. Paste your Hugging Face token in the box at the top of this tab, "
    "accept the model's licence on its page, then try again."
)
_TOKEN_REFUSED = (
    "Hugging Face refused your token for this file. Accept the model's licence on its page with the same account, "
    "and check that the token is allowed to read gated repositories."
)


def _token_for(url: str, token: str) -> str:
    """The token to send with a download from ``url``: only an https address on huggingface.co gets it, and only when one is saved."""
    return token if token and url.startswith(_HUGGING_FACE_URL) else ""


def _load_catalog() -> Dict[str, Dict[str, Any]]:
    """id -> where a model comes from and where it goes, from ``model_catalog.json``.

    ``folder`` is a ComfyUI model folder name (``loras``, ``diffusion_models``, ``LLM``). ``filename`` is the name the file is
    saved as, which need not be the name in the url. ``category`` (``minimax``, ``image``, ``llm``, ``dmad``) says which tab
    of the Models dialog shows it, ``groups`` which cards, ``default`` whether it is part of the recommended set, ``format`` how it
    is checked (``safetensors`` or ``gguf``) and ``gated`` that Hugging Face wants a login for it, so it can be installed only while a Hugging Face token is saved (``hf_token.py``).
    The urls of the DMAD and MOSS-Image LoRAs are also in ``web/music_video_builder/models.mjs`` (tests/test_model_downloads.py checks it).
    """
    with open(_CATALOG_PATH, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    catalog: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        item = dict(entry)
        # ``group`` is the older name of the category, kept for the DMAD (``dmad``) and MOSS-Image (``moss``) LoRAs.
        item.setdefault("group", item["category"])
        item.setdefault("gated", False)
        if item["id"] in catalog:
            raise ValueError(f"model_catalog.json lists {item['id']} twice.")
        catalog[item["id"]] = item
    return catalog


DOWNLOADABLE_MODELS: Dict[str, Dict[str, Any]] = _load_catalog()

_lock = threading.Lock()
# id -> {"state": downloading | done | error | cancelled, "downloaded": int, "total": int, "error": str, "cancel": Event}
_jobs: Dict[str, Dict[str, Any]] = {}


def _folder_dirs(folder: str) -> List[str]:
    register_custom_model_root()
    try:
        dirs = [path for path in folder_paths.get_folder_paths(folder) if path]
    except Exception:
        dirs = []
    if not dirs and folder == "LLM":
        # ComfyUI has no LLM folder of its own. The local LLM nodes read ``models/LLM``.
        models_dir = getattr(folder_paths, "models_dir", "")
        if models_dir:
            dirs = [os.path.join(models_dir, "LLM")]
    return dirs


# ComfyUI lists these older folder names first for diffusion_models and text_encoders. They still work, but files belong in the current ones.
_LEGACY_FOLDER_NAMES = {"diffusion_models": "unet", "text_encoders": "clip"}


def _preferred_dir(folder: str, dirs: List[str]) -> str:
    """ComfyUI's first folder for ``folder``, skipping its legacy name (``models/unet`` for diffusion_models) when there is another."""
    legacy = _LEGACY_FOLDER_NAMES.get(folder)
    for path in dirs:
        if os.path.basename(os.path.normpath(path)).lower() != legacy:
            return path
    return dirs[0]


def destination_dir(folder: str) -> str:
    """The folder a model of this kind is installed into: ComfyUI's first (default) one, created when missing."""
    dirs = _folder_dirs(folder)
    if not dirs:
        raise RuntimeError(f"ComfyUI has no '{folder}' model folder.")
    directory = _preferred_dir(folder, dirs)
    os.makedirs(directory, exist_ok=True)
    return directory


def _folder_index(folder: str) -> Dict[str, str]:
    """File name -> path of every file in ComfyUI's folders for ``folder`` (subfolders included). The first folder wins."""
    index: Dict[str, str] = {}
    for directory in _folder_dirs(folder):
        for root, _dirs, files in os.walk(directory):
            for name in files:
                index.setdefault(name, os.path.join(root, name))
    return index


def installed_path(folder: str, filename: str) -> Optional[str]:
    """Where this file already is in any of ComfyUI's folders for ``folder`` (subfolders included), else ``None``."""
    return _folder_index(folder).get(filename)


def _job_view(job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not job:
        return {"state": "idle", "downloaded": 0, "total": 0, "error": ""}
    return {key: job[key] for key in ("state", "downloaded", "total", "error")}


def model_status() -> List[Dict[str, Any]]:
    """Every downloadable model with whether it is installed and how its download is going.

    A gated file says whether it can be installed (``token_saved``) and what is missing until it can (``gated_message``).
    """
    result = []
    has_token = bool(load_token())
    # Every folder is read once, not once per model.
    indexes: Dict[str, Dict[str, str]] = {}
    destinations: Dict[str, str] = {}
    for model_id, item in DOWNLOADABLE_MODELS.items():
        folder = item["folder"]
        if folder not in indexes:
            indexes[folder] = _folder_index(folder)
            try:
                destinations[folder] = destination_dir(folder)
            except RuntimeError:
                destinations[folder] = ""
        path = indexes[folder].get(item["filename"])
        destination = destinations[folder]
        with _lock:
            job = _job_view(_jobs.get(model_id))
        result.append({
            "id": model_id,
            "label": item["label"],
            "group": item["group"],
            "category": item["category"],
            "groups": list(item["groups"]),
            "default": bool(item["default"]),
            "gated": bool(item["gated"]),
            "token_saved": has_token,
            "gated_message": _NEEDS_TOKEN if item["gated"] and not has_token else None,
            "format": item["format"],
            "download_size": int(item.get("download_size") or 0),
            "folder": item["folder"],
            "filename": item["filename"],
            "url": item["url"],
            "page": item["page"],
            "note": item["note"],
            "installed": path is not None,
            "path": path or "",
            "size": os.path.getsize(path) if path else 0,
            "destination": destination,
            "job": job,
        })
    return result


def _looks_like_safetensors(path: str) -> bool:
    """True when the file starts with a safetensors header (rejects an HTML error page saved as a model)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return False
            (header_bytes,) = struct.unpack("<Q", raw)
            if header_bytes <= 0 or header_bytes > _MAX_SAFETENSORS_HEADER_BYTES or 8 + header_bytes > size:
                return False
            header = json.loads(handle.read(header_bytes))
    except (OSError, ValueError):
        return False
    return isinstance(header, dict)


def _looks_like_gguf(path: str) -> bool:
    """True when the file starts with the GGUF magic (rejects an HTML error page saved as a model)."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_GGUF_MAGIC)) == _GGUF_MAGIC
    except OSError:
        return False


def _set_job(model_id: str, **changes: Any) -> None:
    with _lock:
        if model_id in _jobs:
            _jobs[model_id].update(changes)


def _open_download(item: Dict[str, Any], token: str):
    """Open the file's url. The token goes in an Authorization header of that first request only: urllib does not copy it to a redirect,
    and Hugging Face sends the file itself to a storage server of its own."""

    def attempt(with_token: bool):
        request = urllib.request.Request(item["url"], headers={"User-Agent": "comfyui-vrgamedevgirl-model-installer"})
        if with_token:
            request.add_unredirected_header("Authorization", f"Bearer {token}")
        return urlopen_verified(request, timeout=60)

    try:
        return attempt(bool(token))
    except urllib.error.HTTPError as exc:
        if not token:
            raise
        if item["gated"] and exc.code in (401, 403):
            raise RuntimeError(_TOKEN_REFUSED) from None
        if exc.code == 401:
            # A token Hugging Face does not accept fails public files too. Try again without it.
            return attempt(False)
        raise


def _download(model_id: str) -> None:
    """Download one model. Runs on a worker thread and reports through ``_jobs``."""
    item = DOWNLOADABLE_MODELS[model_id]
    with _lock:
        cancel = _jobs[model_id]["cancel"]
    part_path = ""
    try:
        directory = destination_dir(item["folder"])
        target = os.path.join(directory, item["filename"])
        part_path = f"{target}.part"
        with _open_download(item, _token_for(item["url"], load_token())) as response:
            total = int(response.headers.get("Content-Length") or 0)
            _set_job(model_id, total=total)
            if total and shutil.disk_usage(directory).free < total + _FREE_SPACE_MARGIN_BYTES:
                raise RuntimeError(
                    f"Not enough free disk space in {directory}: {total / 1024 ** 3:.1f} GB needed plus some room to spare."
                )
            downloaded = 0
            with open(part_path, "wb") as handle:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    _set_job(model_id, downloaded=downloaded)
                    if cancel.is_set():
                        raise InterruptedError("Cancelled.")
        if total and downloaded != total:
            raise RuntimeError(f"The download stopped early ({downloaded} of {total} bytes). Try again.")
        if item["format"] == "gguf":
            if not _looks_like_gguf(part_path):
                raise RuntimeError("The downloaded file is not a valid .gguf file. Try again, or download it from the model page.")
        elif not _looks_like_safetensors(part_path):
            raise RuntimeError("The downloaded file is not a valid .safetensors file. Try again, or download it from the model page.")
        if os.path.exists(target):
            # Someone put the file there while this ran. Keep theirs.
            os.remove(part_path)
        else:
            os.replace(part_path, target)
        print(f"[VRGDG Models] Installed {item['filename']} into {directory}")
        _set_job(model_id, state="done")
    except InterruptedError:
        _set_job(model_id, state="cancelled", error="")
    except urllib.error.HTTPError as exc:
        _set_job(model_id, state="error", error=f"The download server answered {exc.code} {exc.reason}.")
    except Exception as exc:
        _set_job(model_id, state="error", error=str(exc) or exc.__class__.__name__)
    finally:
        if part_path and os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass


def _run_downloads(model_ids: List[str]) -> None:
    for model_id in model_ids:
        with _lock:
            job = _jobs.get(model_id)
            skip = bool(job and job["cancel"].is_set())
        if skip:
            _set_job(model_id, state="cancelled")
            continue
        _download(model_id)


def start_install(model_ids: List[str]) -> List[str]:
    """Start downloading the named models one after another. Returns the ids that were started.

    Unknown ids are an error, and so is a gated one while no Hugging Face token is saved. A model that is already installed or already
    downloading is skipped.
    """
    unknown = [model_id for model_id in model_ids if model_id not in DOWNLOADABLE_MODELS]
    if unknown:
        raise ValueError(f"Unknown model: {', '.join(str(item) for item in unknown)}.")
    gated = [model_id for model_id in model_ids if DOWNLOADABLE_MODELS[model_id]["gated"]]
    if gated and not load_token():
        names = ", ".join(DOWNLOADABLE_MODELS[model_id]["label"] for model_id in gated)
        raise ValueError(f"{names}: {_NEEDS_TOKEN}")
    started = []
    for model_id in dict.fromkeys(model_ids):
        item = DOWNLOADABLE_MODELS[model_id]
        if installed_path(item["folder"], item["filename"]):
            continue
        with _lock:
            current = _jobs.get(model_id)
            if current and current["state"] == "downloading":
                continue
            _jobs[model_id] = {
                "state": "downloading", "downloaded": 0, "total": 0, "error": "",
                "cancel": threading.Event(), "started": time.time(),
            }
        started.append(model_id)
    if started:
        threading.Thread(target=_run_downloads, args=(started,), name="vrgdg-model-download", daemon=True).start()
    return started


def cancel_install(model_ids: List[str]) -> List[str]:
    """Stop the named downloads. Returns the ids that were running."""
    cancelled = []
    with _lock:
        for model_id in model_ids:
            job = _jobs.get(model_id)
            if job and job["state"] == "downloading":
                job["cancel"].set()
                cancelled.append(model_id)
    return cancelled
