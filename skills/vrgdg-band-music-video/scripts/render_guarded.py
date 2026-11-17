"""Render scenes one at a time like render_queue.py, as a draft, a finalize or a full render, with a memory guard.

usage: render_guarded.py --work W [--project P] [--phase draft|finalize|full] [--tag T]
                         [--max-ram-gb 60] [--max-commit-gb 95] 6 4 8 ...

--phase draft     pass 1 only; the latent is saved and the scene holds a DRAFT the user reviews (the default way to work)
--phase finalize  pass 2 from the scene's LATEST draft (the API cannot finalize an older draft; see builder-api.md)
--phase full      both passes in one render

Before each scene: wait for ComfyUI to be idle, POST /vrgdg/resource-monitor/clear-memory, wait, then log system RAM,
Windows commit charge and VRAM to logs/memory.log. If RAM or commit stays above its limit after the clear (Windows is
holding memory), the queue stops BEFORE the next render and logs "RESTART RECOMMENDED". A vocal-gated copy of the song
(audio/gated/scene_NN.wav, from vocal_gate.py) is used automatically for that scene. Every draft's id is appended to
logs/draft_index.json, so the take that belongs to a draft latent can always be found.
"""
import ctypes
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, load_config, log, parser, read_json, wpath, write_json  # noqa: E402
from render_queue import render  # noqa: E402

GB = 1024 ** 3


class MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def commit_gb():
    if not hasattr(ctypes, "windll"):
        return 0.0, 0.0
    st = MemStatus()
    st.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return (st.ullTotalPageFile - st.ullAvailPageFile) / GB, st.ullTotalPageFile / GB


def comfy_url(cfg):
    return str(cfg.get("comfy_url") or str(cfg.get("api_url") or "http://127.0.0.1:8188").split("/vrgdg/")[0]).rstrip("/")


def http(cfg, method, path):
    req = urllib.request.Request(comfy_url(cfg) + path, method=method, data=b"{}" if method == "POST" else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def sample(cfg):
    mon = http(cfg, "GET", "/vrgdg/resource-monitor")
    ram = mon["ram"]["used"] / GB
    gpu = (mon.get("gpus") or [{}])[0]
    vram = gpu.get("used", 0) / GB
    commit, commit_limit = commit_gb()
    return ram, vram, commit, commit_limit


def clear_memory(cfg, label=""):
    """Clear ComfyUI's RAM/VRAM, wait, and return (ram, vram, commit, commit_limit)."""
    try:
        http(cfg, "POST", "/vrgdg/resource-monitor/clear-memory")
    except Exception as exc:  # the render can still run; just record it
        log(cfg, f"{label}clear-memory call failed: {exc}", name="memory.log")
    time.sleep(8)
    return sample(cfg)


def project_folder(cfg, api):
    """The target project's folder (the main project from config, a copy found next to it by its id)."""
    main = cfg.get("project_folder") or ""
    if api.pid == cfg.get("project_id") or not main:
        return main
    return os.path.join(os.path.dirname(os.path.abspath(main)), str(api.pid))


def segment(cfg, api, n):
    """Scene n's saved segment, read-only from vrgdg_builder_session.json (the API's scene GET omits the draft)."""
    path = os.path.join(project_folder(cfg, api), "vrgdg_builder_session.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["segments"][n - 1]


def is_draft(cfg, api, n):
    return bool(segment(cfg, api, n).get("minimax_h3_draft"))


def with_phase(api, phase):
    """Make render_queue.render() send render_phase on its one POST /video/render."""
    if phase == "full":
        return api
    call = api.call

    def call_with_phase(method, path, body=None, *rest, **kw):
        if method == "POST" and path.endswith("/video/render") and isinstance(body, dict):
            body = {**body, "render_phase": phase}
        return call(method, path, body, *rest, **kw)

    api.call = call_with_phase
    return api


def main():
    p = parser(__doc__)
    p.add_argument("scenes", nargs="+", type=int)
    p.add_argument("--tag", default=None)
    p.add_argument("--max-ram-gb", type=float, default=60.0)
    p.add_argument("--max-commit-gb", type=float, default=95.0)
    p.add_argument("--phase", choices=["full", "draft", "finalize"], default="draft")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = with_phase(Api(cfg, args.project), args.phase)
    plan = read_json(wpath(cfg, "render_plan.json"), {}) or {}
    tag = args.tag or args.phase[0]
    for n in args.scenes:
        api.wait_idle()
        ram, vram, commit, limit = clear_memory(cfg, f"scene {n}: ")
        log(cfg, f"scene {n}: BEFORE (after clear) RAM {ram:.1f} GB, commit {commit:.1f}/{limit:.0f} GB, VRAM {vram:.1f} GB",
            name="memory.log")
        if ram > args.max_ram_gb or (limit and commit > args.max_commit_gb):
            msg = (f"RESTART RECOMMENDED: after Clear Memory RAM is {ram:.1f} GB (limit {args.max_ram_gb}) and commit "
                   f"{commit:.1f} GB (limit {args.max_commit_gb}). Stopped before scene {n}.")
            log(cfg, msg, name="memory.log")
            log(cfg, msg)
            return
        scene_plan = plan
        gated = wpath(cfg, os.path.join("audio", "gated", f"scene_{n:02d}.wav"))
        if os.path.isfile(gated):
            scene_plan = {"no_vocals_path": os.path.abspath(gated), "scenes": {str(n): {"audio": "no_vocals"}}}
            log(cfg, f"scene {n}: rendering with the vocal-gated track {gated}", name="memory.log")
        ok = render(cfg, api, n, scene_plan, tag, "" if api.pid == cfg.get("project_id") else f"[{api.pid}] ")
        if ok and args.phase == "draft":
            draft = segment(cfg, api, n).get("minimax_h3_draft") or {}
            idx_path = wpath(cfg, os.path.join("logs", "draft_index.json"))
            idx = read_json(idx_path, {}) or {}
            idx.setdefault(f"{api.pid}:{n}", []).append({"tag": tag, "draft_id": draft.get("id", ""),
                                                        "time": time.strftime("%Y-%m-%d %H:%M")})
            write_json(idx_path, idx)
        api.wait_idle()
        ram, vram, commit, limit = clear_memory(cfg, f"scene {n} after: ")   # clear after every render too (user rule)
        log(cfg, f"scene {n}: AFTER render ({'ok' if ok else 'failed'}, after clear) RAM {ram:.1f} GB, commit {commit:.1f}/{limit:.0f} GB, "
                 f"VRAM {vram:.1f} GB", name="memory.log")
        print(f"scene {n}: AFTER render ({'ok' if ok else 'failed'})", flush=True)
    log(cfg, f"queue done ({args.phase}): " + " ".join(str(n) for n in args.scenes))


if __name__ == "__main__":
    main()
