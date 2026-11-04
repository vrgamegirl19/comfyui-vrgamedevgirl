"""Allowlisted model downloads into ComfyUI's model folders, for the Models dialog.

The browser names a model by id. It never sends a URL or a path, so it cannot make this module fetch anything else
or write anywhere else. A download goes to ``<name>.part`` first, is checked, and only then gets its real name, so
ComfyUI never lists a half-written file and a failed download leaves nothing behind. Files that already exist are
never replaced.
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
from .model_paths import register_custom_model_root

_CHUNK_BYTES = 1024 * 1024
# A download needs this much free space on top of the file itself.
_FREE_SPACE_MARGIN_BYTES = 256 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024

# id -> where it comes from and where it goes. ``folder`` is a ComfyUI model folder name (``loras``). Keep the urls
# in step with DMAD_MODEL_FILES in web/music_video_builder/models.mjs (tests/test_model_downloads.py checks it).
DOWNLOADABLE_MODELS: Dict[str, Dict[str, str]] = {
    "dmad_lora": {
        "label": "DMAD 4-step LoRA",
        "group": "dmad",
        "folder": "loras",
        "filename": "dmad_minimax_h3_4step_full_critic_comfyui.safetensors",
        "url": "https://huggingface.co/ZhengmingYu/DMAD/resolve/main/minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors",
        "page": "https://huggingface.co/ZhengmingYu/DMAD/tree/main/minimax_h3",
        "note": "Pass 1 LoRA. Use the ComfyUI file (name ends in _comfyui). The Diffusers file does not load in ComfyUI.",
    },
    "dmad_refiner": {
        "label": "Refiner LoRA",
        "group": "dmad",
        "folder": "loras",
        "filename": "minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors",
        "page": "https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors",
        "note": "Pass 2 LoRA. Restores detail in 2 to 3 steps.",
    },
}

_lock = threading.Lock()
# id -> {"state": downloading | done | error | cancelled, "downloaded": int, "total": int, "error": str, "cancel": Event}
_jobs: Dict[str, Dict[str, Any]] = {}


def _folder_dirs(folder: str) -> List[str]:
    register_custom_model_root()
    try:
        return [path for path in folder_paths.get_folder_paths(folder) if path]
    except Exception:
        return []


def destination_dir(folder: str) -> str:
    """The folder a model of this kind is installed into: ComfyUI's first (default) one, created when missing."""
    dirs = _folder_dirs(folder)
    if not dirs:
        raise RuntimeError(f"ComfyUI has no '{folder}' model folder.")
    os.makedirs(dirs[0], exist_ok=True)
    return dirs[0]


def installed_path(folder: str, filename: str) -> Optional[str]:
    """Where this file already is in any of ComfyUI's folders for ``folder`` (subfolders included), else ``None``."""
    for directory in _folder_dirs(folder):
        for root, _dirs, files in os.walk(directory):
            if filename in files:
                return os.path.join(root, filename)
    return None


def _job_view(job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not job:
        return {"state": "idle", "downloaded": 0, "total": 0, "error": ""}
    return {key: job[key] for key in ("state", "downloaded", "total", "error")}


def model_status() -> List[Dict[str, Any]]:
    """Every downloadable model with whether it is installed and how its download is going."""
    result = []
    for model_id, item in DOWNLOADABLE_MODELS.items():
        path = installed_path(item["folder"], item["filename"])
        try:
            destination = destination_dir(item["folder"])
        except RuntimeError:
            destination = ""
        with _lock:
            job = _job_view(_jobs.get(model_id))
        result.append({
            "id": model_id,
            "label": item["label"],
            "group": item["group"],
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


def _set_job(model_id: str, **changes: Any) -> None:
    with _lock:
        if model_id in _jobs:
            _jobs[model_id].update(changes)


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
        request = urllib.request.Request(item["url"], headers={"User-Agent": "comfyui-vrgamedevgirl-model-installer"})
        with urlopen_verified(request, timeout=60) as response:
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
        if not _looks_like_safetensors(part_path):
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

    Unknown ids are an error. A model that is already installed or already downloading is skipped.
    """
    unknown = [model_id for model_id in model_ids if model_id not in DOWNLOADABLE_MODELS]
    if unknown:
        raise ValueError(f"Unknown model: {', '.join(str(item) for item in unknown)}.")
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
