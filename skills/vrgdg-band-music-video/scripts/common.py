"""Shared helpers for the vrgdg-band-music-video (copied from vrgdg-h3-music-video) scripts. Python 3.9+, standard library only.

Every script takes --work <folder>: the music video's working folder (default: the current folder). It holds
config.json, spec.py, refs.json, timeline.json, lyrics_timed.json and the audio/, prompts/, qa/, logs/ and takes/
folders. --project overrides the Builder project id from config.json (used for an alternates copy).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

FINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted"}
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--work", default=os.getcwd(), help="music video working folder that holds config.json")
    p.add_argument("--project", default=None, help="Builder project id; overrides config.json (e.g. an alternates copy)")
    return p


def load_config(work):
    work = os.path.abspath(work)
    path = os.path.join(work, "config.json")
    if not os.path.isfile(path):
        sys.exit(f"[VRGDG MV] No config.json in {work}. Copy assets/config_example.json there and fill it in.")
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_work"] = work
    return cfg


def save_config(cfg):
    data = {k: v for k, v in cfg.items() if not k.startswith("_")}
    write_json(os.path.join(cfg["_work"], "config.json"), data)


def write_json(path, data):
    """Write JSON through a temp file and rename, so an interrupted write never leaves half a file."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def wpath(cfg, rel):
    """A path from config.json: absolute as given, otherwise relative to the work folder."""
    rel = str(rel or "")
    return rel if os.path.isabs(rel) else os.path.join(cfg["_work"], rel)


def log(cfg, message, name="render.log"):
    line = time.strftime("%H:%M:%S ") + message
    print(line, flush=True)
    folder = os.path.join(cfg["_work"], "logs")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def ffprobe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


class Api:
    """Client for the VRGDG Agent API (/vrgdg/api/v1). Paths may contain {pid}; it is URL-encoded (ids can have spaces)."""

    def __init__(self, cfg, project=None):
        self.base = str(cfg.get("api_url") or "http://127.0.0.1:8188/vrgdg/api/v1").rstrip("/")
        self.comfy = str(cfg.get("comfy_url") or self.base.split("/vrgdg/")[0]).rstrip("/")
        self.pid = project or cfg.get("project_id")

    def url(self, path):
        return self.base + path.replace("{pid}", urllib.parse.quote(str(self.pid), safe=""))

    def call(self, method, path, body=None, if_match=None, timeout=300):
        data = None if body is None else json.dumps(body).encode("utf-8")
        for attempt in range(3):
            req = urllib.request.Request(self.url(path), data=data, method=method.upper())
            req.add_header("Accept", "application/json")
            if data is not None:
                req.add_header("Content-Type", "application/json")
            if if_match is not None:
                req.add_header("If-Match", str(if_match))
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw, status = resp.read().decode("utf-8", "replace"), resp.status
            except urllib.error.HTTPError as exc:
                raw, status = exc.read().decode("utf-8", "replace"), exc.code
            except (ConnectionResetError, urllib.error.URLError) as exc:
                if attempt < 2:
                    time.sleep(3)
                    continue
                return 0, {"ok": False, "error": {"code": "UNREACHABLE", "message": f"{exc} - is ComfyUI running?"}}
            try:
                return status, (json.loads(raw) if raw else {})
            except ValueError:
                return status, {"ok": False, "error": {"code": "NOT_JSON", "message": raw[:500]}}
        return 0, {"ok": False, "error": {"code": "UNREACHABLE", "message": "no answer"}}

    def revision(self):
        _, env = self.call("GET", "/projects/{pid}/summary")
        rev = env.get("revision")
        if rev is None and isinstance(env.get("data"), dict):
            rev = env["data"].get("revision")
        return rev

    def mutate(self, method, path, body=None):
        """Send a write with If-Match; on 409 (someone saved in between, often the open Builder page) re-read and retry."""
        status, env = 0, {}
        for _ in range(4):
            status, env = self.call(method, path, body, if_match=self.revision())
            if status == 409:
                time.sleep(1.5)
                continue
            break
        return status, env

    def wait(self, job_id, poll=15.0, quiet=True):
        since = 0
        while True:
            _, logs = self.call("GET", f"/jobs/{job_id}/log?since={since}")
            for entry in (logs.get("data") or {}).get("logs") or []:
                since = max(since, int(entry.get("seq") or since))
                if not quiet:
                    print(f"  [{entry.get('level', 'info')}] {entry.get('message', '')}", flush=True)
            _, env = self.call("GET", f"/jobs/{job_id}")
            job = env.get("data") or {}
            if job.get("status") in FINAL_STATES:
                return job
            time.sleep(poll)

    def run_job(self, method, path, body=None, poll=10.0, quiet=True):
        status, env = self.call(method, path, body)
        job_id = ((env.get("data") or {}) if isinstance(env.get("data"), dict) else {}).get("job_id")
        if not job_id:
            return {"status": "failed", "error": env.get("error") or env, "http": status}
        return self.wait(job_id, poll=poll, quiet=quiet)

    def scenes(self):
        _, env = self.call("GET", "/projects/{pid}/scenes")
        data = env.get("data") or {}
        return data.get("items") if isinstance(data, dict) else (data or [])

    def scene(self, number):
        _, env = self.call("GET", f"/projects/{{pid}}/scenes/{number}")
        return env.get("data") or {}

    def busy(self):
        """Reasons the GPU is busy (ComfyUI queue or API GPU jobs); empty list when idle."""
        reasons = []
        try:
            with urllib.request.urlopen(self.comfy + "/queue", timeout=10) as resp:
                queue = json.loads(resp.read())
            if queue.get("queue_running") or queue.get("queue_pending"):
                reasons.append("ComfyUI queue is not empty")
        except Exception as exc:  # unreachable counts as busy so nothing gets queued blindly
            reasons.append(f"ComfyUI unreachable: {exc}")
        _, env = self.call("GET", "/queue")
        q = env.get("data") or {}
        if q.get("gpu_running") or q.get("gpu_queued"):
            reasons.append("API GPU job running or queued")
        return reasons

    def wait_idle(self, poll=20.0):
        while self.busy():
            time.sleep(poll)


def video_path(scene):
    """Absolute path of a scene's active video, from the rendered_video URL (it carries the full path)."""
    video = scene.get("rendered_video")
    if not isinstance(video, dict):
        return str(video or "")
    url = str(video.get("url") or "")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if query.get("path"):
        return query["path"][0]
    return str(video.get("path") or "")


def load_spec(cfg, name="spec.py"):
    """Import the scene spec (a Python file in the work folder) as a module."""
    import importlib.util

    path = name if os.path.isabs(name) else os.path.join(cfg["_work"], name)
    if not os.path.isfile(path):
        sys.exit(f"[VRGDG MV] Spec file not found: {path}")
    sys.path.insert(0, os.path.dirname(path))
    module_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module
