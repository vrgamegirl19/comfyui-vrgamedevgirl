"""System routes for the Video Builder: host resource readings and memory cleanup, the update button, and
allowlisted installs of the custom nodes the builder depends on."""

import asyncio
import csv
import importlib.util
import io
import itertools
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import folder_paths
import psutil
from aiohttp import web
from server import PromptServer


_lock = asyncio.Lock()
_cached = None
_cached_at = 0.0
_cleanup_ids = itertools.count(1)
_cleanup_reports = set()
_log = logging.getLogger(__name__)
_FIELDS = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,fan.speed,clocks.gr,power.draw,power.limit"


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except ValueError:
        return None


def _read_resources():
    memory = psutil.virtual_memory()
    result = {
        "ram": {"used": memory.total - memory.available, "total": memory.total},
        "gpus": [],
        "gpu_status": "unavailable",
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=2, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    if completed.returncode != 0:
        return result
    for row in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True):
        if len(row) != 10:
            continue
        index, name, *values = row
        keys = ("load", "used", "total", "temperature", "fan", "clock", "power", "power_limit")
        gpu = dict(zip(keys, map(_number, values)))
        gpu.update(index=index.strip(), name=name.strip())
        for key in ("used", "total"):
            if gpu[key] is not None:
                gpu[key] *= 1024 ** 2
        result["gpus"].append(gpu)
    if result["gpus"]:
        result["gpu_status"] = "available"
    return result


@PromptServer.instance.routes.get("/vrgdg/resource-monitor")
async def resource_monitor(request):
    global _cached, _cached_at
    async with _lock:
        if _cached is None or time.monotonic() - _cached_at >= 1:
            _cached = await asyncio.to_thread(_read_resources)
            _cached_at = time.monotonic()
        return web.json_response(_cached, headers={"Cache-Control": "no-store"})


def _memory_snapshot():
    snapshot = _read_resources()
    snapshot["process_ram"] = psutil.Process().memory_info().rss
    return snapshot


def _log_memory(cleanup_id, label, snapshot, before=None):
    def reading(name, value, previous=None):
        if value is None:
            _log.info("[VRGDG Clear Memory #%s] %s %s: unavailable", cleanup_id, label, name)
            return
        delta = "" if previous is None else f" (decrease: {(previous - value) / 2**30:+.2f} GiB)"
        _log.info("[VRGDG Clear Memory #%s] %s %s: %.2f GiB used%s", cleanup_id, label, name, value / 2**30, delta)

    reading("System RAM", snapshot["ram"]["used"], before["ram"]["used"] if before else None)
    reading("ComfyUI process RAM (RSS)", snapshot["process_ram"], before["process_ram"] if before else None)
    previous_gpus = {gpu["index"]: gpu for gpu in before["gpus"]} if before else {}
    for gpu in snapshot["gpus"]:
        reading(f"GPU {gpu['index']} VRAM ({gpu['name']})", gpu["used"], previous_gpus.get(gpu["index"], {}).get("used"))
    if not snapshot["gpus"]:
        reading("GPU VRAM", None)


async def _report_memory_after(cleanup_id, before):
    try:
        await asyncio.sleep(3)
        after = await asyncio.to_thread(_memory_snapshot)
        _log_memory(cleanup_id, "After request", after, before)
        _log.info("[VRGDG Clear Memory #%s] Follow-up sample, not a worker completion acknowledgement. System/GPU totals include other apps; negative decrease means usage increased.", cleanup_id)
    except Exception:
        _log.exception("[VRGDG Clear Memory #%s] Could not read follow-up memory usage", cleanup_id)


def _request_memory_cleanup():
    queue = PromptServer.instance.prompt_queue
    # Keep the idle check and cleanup atomic with respect to prompt submission/start.
    with queue.mutex:
        if queue.get_tasks_remaining():
            _log.info("[VRGDG Clear Memory] Skipped: running or queued jobs; nothing cleared.")
            return {"ok": False, "error": "Wait for the running and queued jobs to finish."}
        llm = sys.modules.get(f"{__package__}.llm.cache")
        # Builder and editor routes run GGUF chat outside the queue; don't wait on them while holding the queue mutex.
        if llm is not None and not llm._GGUF_LOCK.acquire(blocking=False):
            _log.info("[VRGDG Clear Memory] Skipped: an LLM request is running; nothing cleared.")
            return {"ok": False, "error": "Wait for the running LLM request to finish."}
        cleanup_id = next(_cleanup_ids)
        before = _memory_snapshot()
        _log.info("[VRGDG Clear Memory #%s] Cleanup requested", cleanup_id)
        _log_memory(cleanup_id, "Before", before)
        gguf_unloaded = 0
        if llm is not None:
            try:
                result = llm._clear_vrgdg_llm_caches(clear_cuda_cache=False, clear_hf_pipeline_cache=False)
            finally:
                llm._GGUF_LOCK.release()
            gguf_unloaded = result["gguf_models_unloaded"]
            _log.info("[VRGDG Clear Memory #%s] Gemma/GGUF cache cleanup returned: %s models removed from cache; HF pipeline cache retained.", cleanup_id, gguf_unloaded)
        else:
            _log.info("[VRGDG Clear Memory #%s] Gemma/GGUF cache: module not loaded; skipped.", cleanup_id)
        # The execution worker owns model unloading, executor caches, GC and device caches.
        queue.set_flag("unload_models", True)
        queue.set_flag("free_memory", True)
        _log.info("[VRGDG Clear Memory #%s] Requested native model unloading, execution-cache reset, garbage collection and device-cache cleanup. Follow-up memory sample in about 3 seconds.", cleanup_id)
    return {"ok": True, "status": "requested", "gguf_models_unloaded": gguf_unloaded, "cleanup_id": cleanup_id, "before": before}


@PromptServer.instance.routes.post("/vrgdg/resource-monitor/clear-memory")
async def clear_memory(request):
    try:
        result = await asyncio.to_thread(_request_memory_cleanup)
    except Exception:
        _log.exception("[VRGDG Clear Memory] Cleanup request failed; cleanup may be incomplete.")
        return web.json_response({"ok": False, "error": "Cleanup request failed; see the ComfyUI console."}, status=500)
    if result["ok"]:
        task = asyncio.create_task(_report_memory_after(result["cleanup_id"], result.pop("before")))
        _cleanup_reports.add(task)
        task.add_done_callback(_cleanup_reports.discard)
    return web.json_response(result, status=202 if result["ok"] else 409)

UPDATE_BRANCH = "Beta2.0"
_NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RELEASE_NOTES_FILE = "update_notes.json"


def _run_git(*args, timeout=300):
    try:
        result = subprocess.run(
            ["git", *args], cwd=_NODE_DIR, capture_output=True, text=True,
            errors="replace", timeout=timeout, check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git was not found. Install Git, then try again.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Git timed out while updating. Check your internet connection and try again.") from exc

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        command = "git " + " ".join(args)
        raise RuntimeError(f"{command} failed:\n{output or 'Git returned an unknown error.'}")
    return output


def _install_requirements(timeout=900):
    requirements_path = os.path.join(_NODE_DIR, "requirements.txt")
    if not os.path.isfile(requirements_path):
        raise RuntimeError(f"Updated requirements file was not found: {requirements_path}")

    command = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
    try:
        result = subprocess.run(
            command, cwd=_NODE_DIR, capture_output=True, text=True,
            errors="replace", timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Installing updated Python requirements timed out. "
            f"Run this command manually:\n{subprocess.list2cmdline(command)}"
        ) from exc

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(
            "The code update completed, but installing updated Python requirements failed.\n"
            f"Run this command manually:\n{subprocess.list2cmdline(command)}\n\n"
            f"{output or 'pip returned an unknown error.'}"
        )
    return {
        "command": subprocess.list2cmdline(command),
        "output": output,
    }


def _load_release_notes(ref=""):
    """Load notes from a fetched Git ref, falling back to the installed checkout."""
    raw = ""
    source = "local"
    if ref:
        try:
            raw = _run_git("show", f"{ref}:{_RELEASE_NOTES_FILE}", timeout=20)
            source = ref
        except Exception:
            raw = ""

    if not raw:
        notes_path = os.path.join(_NODE_DIR, _RELEASE_NOTES_FILE)
        if not os.path.isfile(notes_path):
            return {"schema_version": 1, "product": "AI Video Builder", "releases": []}, "none"
        with open(notes_path, "r", encoding="utf-8") as handle:
            raw = handle.read()

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_RELEASE_NOTES_FILE} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise RuntimeError(f"{_RELEASE_NOTES_FILE} must contain a JSON object.")
    releases = document.get("releases")
    if not isinstance(releases, list):
        document["releases"] = []
    return document, source


def _git_is_ancestor(commit, ref):
    commit = str(commit or "").strip()
    ref = str(ref or "").strip()
    if not commit or not ref:
        return False
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, ref],
            cwd=_NODE_DIR, capture_output=True, text=True,
            errors="replace", timeout=20, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _git_commit_tree(commit):
    """Return the tree ID for a commit, or an empty string if it is unavailable."""
    commit = str(commit or "").strip()
    if not commit:
        return ""
    try:
        return _run_git("rev-parse", f"{commit}^{{tree}}", timeout=20).strip()
    except Exception:
        return ""


def _git_history_contains_tree(commit, ref):
    """Recognize equivalent content when a PR was squash-merged to another commit ID."""
    target_tree = _git_commit_tree(commit)
    ref = str(ref or "").strip()
    if not target_tree or not ref:
        return False
    try:
        history_trees = _run_git("log", "--format=%T", ref, timeout=30).splitlines()
    except Exception:
        return False
    return target_tree in {tree.strip() for tree in history_trees if tree.strip()}


def _git_contains_release(commit, ref):
    return _git_is_ancestor(commit, ref) or _git_history_contains_tree(commit, ref)


def _release_note_status(document, local_commit, latest_commit):
    available_release_ids = []
    current_release_id = ""
    for release in document.get("releases", []):
        if not isinstance(release, dict):
            continue
        release_id = str(release.get("id") or "").strip()
        commit = str(release.get("commit") or "").strip()
        if not release_id or not commit:
            continue
        installed = _git_contains_release(commit, local_commit)
        published = _git_contains_release(commit, latest_commit)
        if not current_release_id and installed:
            current_release_id = release_id
        if published and not installed:
            available_release_ids.append(release_id)
    return {
        "available_release_ids": available_release_ids,
        "current_release_id": current_release_id,
    }


def _update_to_main():
    if not os.path.isdir(os.path.join(_NODE_DIR, ".git")):
        raise RuntimeError("This installation is not a Git checkout, so the normal Git update commands cannot run.")

    logs = []
    before_commit = _run_git("rev-parse", "HEAD").strip()
    for args in (
        ("fetch", "origin", UPDATE_BRANCH),
        ("switch", UPDATE_BRANCH),
        ("pull", "--ff-only", "origin", UPDATE_BRANCH),
    ):
        output = _run_git(*args)
        logs.append({"command": "git " + " ".join(args), "output": output})

    branch = _run_git("branch", "--show-current").strip()
    if branch != UPDATE_BRANCH:
        raise RuntimeError(f"Git finished on '{branch or '(detached HEAD)'}' instead of '{UPDATE_BRANCH}'.")

    after_commit = _run_git("rev-parse", "HEAD").strip()
    changed_paths = _run_git(
        "diff", "--name-only", before_commit, after_commit, "--", "requirements.txt"
    ).splitlines()
    requirements_changed = "requirements.txt" in {path.strip() for path in changed_paths}
    requirements_installed = False
    requirements_error = ""
    requirements_command = ""

    if requirements_changed:
        try:
            requirements_result = _install_requirements()
            requirements_installed = True
            requirements_command = requirements_result["command"]
            logs.append({
                "command": requirements_result["command"],
                "output": requirements_result["output"],
            })
        except Exception as exc:
            requirements_error = str(exc)

    release_notes, release_notes_source = _load_release_notes()
    return {
        "branch": branch,
        "directory": _NODE_DIR,
        "before_commit": before_commit,
        "after_commit": after_commit,
        "requirements_changed": requirements_changed,
        "requirements_installed": requirements_installed,
        "requirements_error": requirements_error,
        "requirements_command": requirements_command,
        "restart_required": True,
        "release_notes": release_notes,
        "release_notes_source": release_notes_source,
        "logs": logs,
    }


def _main_status():
    """Compare the installed checkout with the production main branch without changing files."""
    if not os.path.isdir(os.path.join(_NODE_DIR, ".git")):
        raise RuntimeError("This installation is not a Git checkout, so its update status cannot be checked.")

    _run_git("fetch", "origin", UPDATE_BRANCH, timeout=20)
    local_commit = _run_git("rev-parse", "HEAD").strip()
    remote_ref = f"origin/{UPDATE_BRANCH}"
    latest_commit = _run_git("rev-parse", remote_ref).strip()
    branch = _run_git("branch", "--show-current").strip()
    behind = int(_run_git("rev-list", "--count", f"HEAD..{remote_ref}").strip() or "0")
    ahead = int(_run_git("rev-list", "--count", f"{remote_ref}..HEAD").strip() or "0")
    tracked_changes = bool(_run_git("status", "--porcelain", "--untracked-files=no").strip())
    release_notes, release_notes_source = _load_release_notes(remote_ref)
    release_status = _release_note_status(release_notes, local_commit, latest_commit)
    latest_content_installed = _git_contains_release(latest_commit, local_commit)

    return {
        "branch": branch,
        "expected_branch": UPDATE_BRANCH,
        "installed_commit": local_commit,
        "latest_commit": latest_commit,
        "behind": behind,
        "ahead": ahead,
        "outdated": behind > 0 and not latest_content_installed,
        "latest_content_installed": latest_content_installed,
        "tracked_changes": tracked_changes,
        "release_notes": release_notes,
        "release_notes_source": release_notes_source,
        **release_status,
    }

@PromptServer.instance.routes.get("/vrgdg/update/v10/status")
async def vrgdg_update_v10_status(request):
    try:
        result = await asyncio.to_thread(_main_status)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)})
    return web.json_response({"ok": True, **result})

@PromptServer.instance.routes.post("/vrgdg/update/v10")
async def vrgdg_update_v10(request):
    try:
        result = await asyncio.to_thread(_update_to_main)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    return web.json_response({"ok": True, **result})

_CUSTOM_NODES_DIR = Path(_NODE_DIR).parent

# Keep this list deliberately allowlisted. The browser can request an id, never an arbitrary URL or filesystem path.
# "install" is what ComfyUI-Manager's cm_cli installs: a Comfy Registry id, or the GitHub URL for packs that are not
# in the registry. "folders" are the folder names an existing install may use; the first is where a GitHub clone lands.
VIDEO_BUILDER_CUSTOM_NODES = {
    "videohelpersuite": {"install": "comfyui-videohelpersuite", "folders": ("ComfyUI-VideoHelperSuite", "comfyui-videohelpersuite")},
    "kjnodes": {"install": "comfyui-kjnodes", "folders": ("ComfyUI-KJNodes", "comfyui-kjnodes")},
    "gguf": {"install": "ComfyUI-GGUF", "folders": ("ComfyUI-GGUF",)},
    "ltxvideo": {"install": "ComfyUI-LTXVideo", "folders": ("ComfyUI-LTXVideo",)},
    "te_speed_minimax_h3": {"install": "https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS.git", "folders": ("TE-Speed-MiniMaxH3-OSS",)},
    "mmh3_ultimate_upscale": {"install": "Comfyui-MMH3-UltimateUpscale", "folders": ("Comfyui-MMH3-UltimateUpscale",)},
    "minimax_h3_audio_t8": {"install": "https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git", "folders": ("comfyui-minimax-h3-audio-T8",)},
    "minimax_h3_latent_upscaler": {"install": "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git", "folders": ("Comfyui_Minimax_h3_latent_Upscaler",)},
}


def _installed_node_path(item):
    return next((_CUSTOM_NODES_DIR / name for name in item["folders"] if (_CUSTOM_NODES_DIR / name).is_dir()), None)


def _node_status(item_id, item):
    path = _installed_node_path(item)
    return {
        "id": item_id,
        "folder": path.name if path else item["folders"][0],
        "installed": path is not None,
        "path": str(path or _CUSTOM_NODES_DIR / item["folders"][0]),
        "source": "github" if item["install"].startswith("https://") else "registry",
        "install": item["install"],
    }


def _manager_available():
    return importlib.util.find_spec("cm_cli") is not None


def _run_cm_cli(*args, timeout=1800):
    command = [sys.executable, "-m", "cm_cli", *args]
    env = {**os.environ, "COMFYUI_PATH": folder_paths.base_path, "PYTHONIOENCODING": "utf-8"}
    try:
        result = subprocess.run(command, cwd=folder_paths.base_path, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ComfyUI-Manager timed out: {subprocess.list2cmdline(command)}") from exc
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(f"{subprocess.list2cmdline(command)} failed:\n{output[-2000:] or 'No output.'}")
    return output


def _install(ids):
    if not _manager_available():
        raise RuntimeError("ComfyUI-Manager is not installed in ComfyUI's Python. Start ComfyUI with --enable-manager "
                           "or run pip install -r manager_requirements.txt from the ComfyUI folder, then try again.")
    requested = [(item_id, VIDEO_BUILDER_CUSTOM_NODES[item_id]) for item_id in ids if item_id in VIDEO_BUILDER_CUSTOM_NODES]
    if not requested:
        raise RuntimeError("No recognized Video Builder custom nodes were selected.")
    results = []
    for item_id, item in requested:
        path = _installed_node_path(item)
        if path:
            output = _run_cm_cli("post-install", str(path))
            action = "requirements reinstalled"
        else:
            output = _run_cm_cli("install", item["install"], "--exit-on-fail")
            if _installed_node_path(item) is None:
                raise RuntimeError(f"ComfyUI-Manager finished, but no {item['folders'][0]} folder was created:\n{output[-2000:]}")
            action = "installed"
        results.append({"id": item_id, "folder": item["folders"][0], "action": action, "output": output[-1200:]})
    return results

@PromptServer.instance.routes.get("/vrgdg/video_builder/custom_nodes/status")
async def custom_nodes_status(request):
    return web.json_response({
        "ok": True,
        "custom_nodes_dir": str(_CUSTOM_NODES_DIR),
        "manager_available": _manager_available(),
        "nodes": [_node_status(item_id, item) for item_id, item in VIDEO_BUILDER_CUSTOM_NODES.items()],
    })

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

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
