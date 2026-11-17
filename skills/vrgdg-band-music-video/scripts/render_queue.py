"""Render scenes one at a time through the Agent API, backing up each previous take and making a QA sheet per take.

usage: render_queue.py --work W [--project P] [--tag T] [--no-retry] [--dry-run] 3 21 13 1 ...
       (run it in the background: nohup python render_queue.py --work W 1 2 3 > logs/queue.out 2>&1 &)

- Waits until ComfyUI and the API queue are idle before every scene (never queue behind the user's own render).
- Scenes marked "no_vocals" in render_plan.json (build_prompts.py writes it) render against
  audio/full_track_no_vocals.wav, so nobody on screen lip-syncs the singer. The stitch still uses the real song.
- Before a scene re-renders, its current video is copied to takes/<project>/video_NNNN_takeK.mp4.
- Logs "scene N: started" / "scene N: saved <path> (x s, scene y s) in m min" / "ERROR" / "queue done" lines to
  logs/render.log (watch them with the Monitor tool) and writes qa/scene_NN_<tag>.jpg after each take.
- A failed scene is retried once (unless --no-retry), then left for the report.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, ffprobe_duration, load_config, log, parser, read_json, video_path, wpath  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))[:60]


def backup(cfg, api, n):
    path = video_path(api.scene(n))
    if not path or not os.path.isfile(path):
        return None
    folder = wpath(cfg, os.path.join("takes", safe(api.pid)))
    os.makedirs(folder, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(folder, f"video_{n:04d}_take{k}.mp4")):
        k += 1
    dest = os.path.join(folder, f"video_{n:04d}_take{k}.mp4")
    shutil.copy2(path, dest)
    return dest


def render(cfg, api, n, plan, tag, label):
    body = {"mode": "minimax_h3"}
    entry = (plan.get("scenes") or {}).get(str(n)) or {}
    if entry.get("audio") == "no_vocals":
        body["audio_path"] = plan.get("no_vocals_path") or os.path.abspath(wpath(cfg, "audio/full_track_no_vocals.wav"))
        if not os.path.isfile(body["audio_path"]):
            log(cfg, f"{label}scene {n}: ERROR vocal-free track missing: {body['audio_path']}")
            return False
    api.wait_idle()
    saved = backup(cfg, api, n)
    if saved:
        log(cfg, f"{label}scene {n}: previous take backed up to {saved}")
    t0 = time.time()
    status, env = api.call("POST", f"/projects/{{pid}}/scenes/{n}/video/render", body)
    job_id = (env.get("data") or {}).get("job_id") if isinstance(env.get("data"), dict) else None
    if not job_id:
        log(cfg, f"{label}scene {n}: ERROR could not start ({status}): {json.dumps(env.get('error') or env)[:400]}")
        return False
    log(cfg, f"{label}scene {n}: started job {job_id} (audio={entry.get('audio', 'full')})")
    job = api.wait(job_id)
    minutes = (time.time() - t0) / 60
    if job.get("status") != "succeeded":
        log(cfg, f"{label}scene {n}: ERROR {job.get('status')} after {minutes:.1f} min: {json.dumps(job.get('error'))[:400]}")
        return False
    scene = api.scene(n)
    path = video_path(scene)
    want = float(scene.get("end", 0)) - float(scene.get("start", 0))
    got = ffprobe_duration(path) if path else 0.0
    log(cfg, f"{label}scene {n}: saved {path} ({got:.2f}s, scene {want:.2f}s) in {minutes:.1f} min")
    subprocess.run([sys.executable, os.path.join(HERE, "qa_sheet.py"), "--work", cfg["_work"], str(n), path,
                    "--tag", tag or time.strftime("%H%M%S")], check=False)
    return True


def main():
    p = parser(__doc__)
    p.add_argument("scenes", nargs="+", type=int)
    p.add_argument("--tag", default=None, help="QA sheet tag (default: the time)")
    p.add_argument("--no-retry", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    plan = read_json(wpath(cfg, "render_plan.json"), {}) or {}
    label = "" if api.pid == cfg.get("project_id") else f"[{api.pid}] "
    if args.dry_run:
        for n in args.scenes:
            entry = (plan.get("scenes") or {}).get(str(n)) or {}
            print(f"scene {n}: audio={entry.get('audio', 'full (not in render_plan.json)')} project={api.pid}")
        print("busy:", api.busy() or "idle")
        return
    for n in args.scenes:
        if not render(cfg, api, n, plan, args.tag, label) and not args.no_retry:
            log(cfg, f"{label}scene {n}: retrying once")
            render(cfg, api, n, plan, args.tag, label)
    log(cfg, f"{label}queue done: " + " ".join(str(n) for n in args.scenes))


if __name__ == "__main__":
    main()
