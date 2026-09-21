"""On-demand host resource readings for the canvas overlay."""

import asyncio
import csv
import io
import itertools
import logging
import math
import subprocess
import sys
import time

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
        cleanup_id = next(_cleanup_ids)
        before = _memory_snapshot()
        _log.info("[VRGDG Clear Memory #%s] Cleanup requested", cleanup_id)
        _log_memory(cleanup_id, "Before", before)
        llm = sys.modules.get(f"{__package__}.LLM") or sys.modules.get("_vrgdg_custom_LLM")
        gguf_unloaded = 0
        if llm is not None:
            result = llm._clear_vrgdg_llm_caches(clear_cuda_cache=False, clear_hf_pipeline_cache=False)
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
