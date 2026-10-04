"""Shared helpers for the vrgdg-h3-film skill: settings, ComfyUI HTTP client, project paths, ffmpeg.

Only the Python standard library is used here, so check_env.py and the render scripts run with any Python 3.9+.
Every script takes the project folder as its first argument (or reads $VRGDG_FILM_PROJECT).
"""
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_ID = "vrgdg-h3-film-" + uuid.uuid4().hex[:8]


def project_dir(argv=None):
    """Project folder from argv[1] or $VRGDG_FILM_PROJECT; the remaining args are returned."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and os.path.isdir(argv[0]):
        return os.path.abspath(argv[0]), argv[1:]
    env = os.environ.get("VRGDG_FILM_PROJECT")
    if env and os.path.isdir(env):
        return os.path.abspath(env), argv
    raise SystemExit("Pass the project folder as the first argument (or set VRGDG_FILM_PROJECT).")


def load_settings(project):
    """settings.json in the project, created by new_project.py from check_env.py's findings."""
    path = os.path.join(project, "settings.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing. Run new_project.py first.")
    return json.load(open(path, encoding="utf-8"))


class Comfy:
    def __init__(self, url="http://127.0.0.1:8188", log_path=None):
        self.url = url.rstrip("/")
        self.log_path = log_path

    def _req(self, method, path, body=None, timeout=600):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                out = json.loads(raw)
                if isinstance(out, dict):
                    out.setdefault("_http_status", exc.code)
                return out
            except ValueError:
                raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {raw[:500]!r}")
        return json.loads(raw) if raw else {}

    def get(self, path, timeout=120):
        return self._req("GET", path, None, timeout)

    def post(self, path, body, timeout=600):
        return self._req("POST", path, body, timeout)

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def run(self, prompt, label="job", timeout=4 * 3600, poll=10):
        res = self.post("/prompt", {"prompt": prompt, "client_id": CLIENT_ID})
        if "prompt_id" not in res:
            raise RuntimeError(f"queue failed for {label}: {json.dumps(res)[:3000]}")
        pid, start, note = res["prompt_id"], time.time(), 0
        self.log(f"queued {label}: {pid}")
        while True:
            hist = self.get(f"/history/{pid}")
            if pid in hist:
                status = hist[pid].get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    if any(m[0] == "execution_interrupted" for m in msgs):
                        raise RuntimeError(f"{label} was interrupted in ComfyUI (Stop/Cancel or /interrupt); re-run it")
                    errs = [m for m in msgs if m[0] == "execution_error"]
                    raise RuntimeError(f"{label} failed: {json.dumps(errs)[:3000]}")
                self.log(f"done {label} in {time.time() - start:.0f}s")
                return hist[pid]
            if time.time() - start > timeout:
                raise TimeoutError(f"{label} timed out")
            if time.time() - note > 300:
                self.log(f"... still running {label} ({time.time() - start:.0f}s)")
                note = time.time()
            time.sleep(poll)

    def free(self, unload_models=False):
        try:
            self.post("/free", {"unload_models": unload_models, "free_memory": True}, timeout=60)
        except Exception as exc:  # /free is a convenience; a failure must not stop a render batch
            self.log(f"/free failed: {exc}")

    def fetch_output(self, item, dst):
        """Download one history output file (filename/subfolder/type) through /view."""
        q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        with urllib.request.urlopen(f"{self.url}/view?{q}", timeout=600) as resp, open(dst, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        return dst


def output_files(entry):
    files = []
    for node_id, out in entry.get("outputs", {}).items():
        for key, items in out.items():
            if isinstance(items, list):
                files += [{"node": node_id, "key": key, **i} for i in items if isinstance(i, dict) and "filename" in i]
    return files


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return path


def find_ffmpeg(hint=""):
    for cand in (hint, shutil.which("ffmpeg")):
        if cand and os.path.isfile(cand):
            return cand
    try:
        import imageio_ffmpeg  # bundled with many ComfyUI installs (VideoHelperSuite)
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def ffprobe_duration(ffmpeg, path):
    """Clip duration via ffmpeg itself (ffprobe is not always shipped next to it)."""
    import re
    import subprocess
    err = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace").stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not m:
        raise RuntimeError(f"could not read duration of {path}")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
