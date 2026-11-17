"""Unattended overnight runner: draft + finalize every scene, never waits for a person.

usage:
  night_run.py --work W parent 3 4 8 1 2 ...   drive the whole list, then stitch; reads logs/extra_jobs.txt for retakes
  night_run.py --work W retake 5 6              forced retakes (new draft + finalize) with the same watchdog
  night_run.py --work W child N                 one scene: draft, then finalize (used by parent)

Options (before the command):
  --recover        allow the runner to start ComfyUI with the user's launcher when it stops answering
                   (only when the user said so, e.g. "check ComfyUI every few minutes; if it crashed, recover")
  --launcher PATH  the launcher .bat (default: config "comfy_launcher")
  --held-limit N   a finished take with N or more repeated frames gets one automatic re-roll (default 12)
  --prefix NAME    stitch prefix (default: config "title" upper-cased + "_auto")

What it does per scene (run in a child process with a health watchdog):
  - draft (pass 1), then finalize (pass 2); RAM/VRAM cleared before and after every render (render_guarded);
  - out of memory: 1 try + 3 retries, then ref_image_size "match", then a re-draft at match;
  - ComfyUI not answering for 3 minutes -> child killed, (with --recover) ComfyUI restarted, scene retried;
  - a scene running over SCENE_TIMEOUT -> interrupted and retried once;
  - finished take checked for repeated frames (Singularity V2 pass 1 doubles frames); one re-roll when choppy, and the
    smoother of the two takes is kept (first take copied to takes/keep);
  - already-finalized scenes are skipped, so it can be restarted any time.
Append scene numbers (one per line) to logs/extra_jobs.txt to queue retakes while it runs. Log: logs/night.log.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Api, load_config, log, read_json, wpath  # noqa: E402
import drive_queue as dq  # noqa: E402
from render_guarded import is_draft, segment  # noqa: E402
from held_frames import held_count  # noqa: E402

SCENE_TIMEOUT = 70 * 60
OPT = argparse.Namespace(work=os.getcwd(), recover=False, launcher=None, held_limit=12, prefix=None)


def say(cfg, msg):
    log(cfg, "night: " + msg, name="night.log")


def healthy(cfg, timeout=10):
    try:
        with urllib.request.urlopen(cfg["api_url"].rstrip("/") + "/health", timeout=timeout) as resp:
            return bool(json.loads(resp.read()).get("ok"))
    except Exception:
        return False


def start_comfy(cfg):
    """Start ComfyUI with the user's own launcher (only with --recover) and wait until the Agent API answers."""
    if not OPT.recover:
        say(cfg, "ComfyUI is not answering and --recover is off; waiting for it (checked every minute)")
        for _ in range(60):
            time.sleep(60)
            if healthy(cfg):
                return True
        return False
    launcher = OPT.launcher or cfg.get("comfy_launcher")
    if not launcher or not os.path.isfile(launcher):
        say(cfg, f"no launcher found ({launcher}); cannot restart ComfyUI")
        return False
    say(cfg, "ComfyUI is not answering; starting it with the user's launcher")
    subprocess.run(["taskkill", "/F", "/T", "/FI", "WINDOWTITLE eq VRGDG_NIGHT_COMFY*"], capture_output=True)
    subprocess.Popen(f'start "VRGDG_NIGHT_COMFY" /MIN cmd /c "{launcher}"', cwd=os.path.dirname(launcher), shell=True)
    for _ in range(90):
        time.sleep(10)
        if healthy(cfg):
            say(cfg, "ComfyUI is back")
            time.sleep(20)
            return True
    say(cfg, "ComfyUI did not come back after 15 minutes")
    return False


def comfy_pids():
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*ComfyUI*main.py*' } | ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout
    return [x.strip() for x in out.split() if x.strip().isdigit()]


def kill_comfy(cfg):
    say(cfg, "killing the hung ComfyUI process")
    for pid in comfy_pids():
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    time.sleep(8)


def finalized(cfg, api, n):
    """True when scene n has a rendered video that is not a pass-1 draft."""
    try:
        seg = segment(cfg, api, n)
    except Exception:
        return False
    if seg.get("minimax_h3_draft"):
        return False
    return bool(api.scene(n).get("rendered_video"))


def take_path(cfg, n):
    return os.path.join(cfg["project_folder"], "rendered_scene_videos", f"video_{n:04d}-audio.mp4")


def child(n):
    cfg = load_config(OPT.work)
    api = Api(cfg)
    plan = read_json(wpath(cfg, "render_plan.json"), {}) or {}
    tag = "d" + time.strftime("%H%M")
    api.wait_idle()
    if not is_draft(cfg, api, n):
        if not dq.draft(cfg, api, n, tag, plan):
            say(cfg, f"scene {n}: draft failed twice")
            return 2
    for attempt in range(1, 9):
        if attempt == 5:
            dq.set_ref_size(cfg, api, n, "match")
            say(cfg, f"scene {n}: still failing after 3 retries -> ref_image_size match")
        if attempt == 7:
            say(cfg, f"scene {n}: re-draft at match")
            if not dq.draft(cfg, api, n, tag + "m", plan):
                break
        dq.guarded(cfg, api, n, "finalize", f"f{attempt}", plan)
        if not is_draft(cfg, api, n):
            say(cfg, f"scene {n} FINALIZED (try {attempt})")
            return 0
        say(cfg, f"scene {n}: finalize try {attempt} failed")
    say(cfg, f"scene {n} stays a DRAFT")
    return 3


def run_scene(cfg, n):
    """Run one scene in a child process with a health watchdog. Returns True when the scene ended finalized."""
    api = Api(cfg)
    for round_no in (1, 2, 3):
        if not healthy(cfg) and not start_comfy(cfg):
            return False
        t0 = time.time()
        args = [sys.executable, os.path.abspath(__file__), "--work", OPT.work, "child", str(n)]
        proc = subprocess.Popen(args)
        down, reason = 0, ""
        while proc.poll() is None:
            time.sleep(60)
            down = 0 if healthy(cfg, timeout=20) else down + 1
            if down >= 3:
                reason = "ComfyUI stopped answering"
            elif time.time() - t0 > SCENE_TIMEOUT:
                reason = f"scene ran over {SCENE_TIMEOUT // 60} minutes"
            if reason:
                say(cfg, f"scene {n}: {reason}; stopping the child (round {round_no})")
                proc.kill()
                break
        if not reason:
            if finalized(cfg, api, n):
                return True
            say(cfg, f"scene {n}: child ended (code {proc.returncode}) without a finalized take")
            if healthy(cfg):
                return False
        if "ran over" in reason and healthy(cfg):
            try:
                urllib.request.urlopen(urllib.request.Request(cfg["comfy_url"] + "/interrupt", data=b"{}", method="POST"), timeout=20)
                urllib.request.urlopen(urllib.request.Request(cfg["comfy_url"] + "/queue", data=b'{"clear": true}',
                                                              headers={"Content-Type": "application/json"}, method="POST"), timeout=20)
            except Exception as exc:
                say(cfg, f"interrupt failed: {exc}")
            time.sleep(30)
        elif not healthy(cfg):
            if OPT.recover and comfy_pids():
                kill_comfy(cfg)
            if not start_comfy(cfg):
                return False
    return finalized(cfg, api, n)


def stitch(cfg):
    prefix = OPT.prefix or (str(cfg.get("title") or "FINAL_VIDEO").upper().replace(" ", "_") + "_auto")
    out = subprocess.run([sys.executable, os.path.join(HERE, "stitch.py"), "--work", OPT.work, "--prefix", prefix],
                         capture_output=True, text=True)
    say(cfg, "stitch: " + (out.stdout or out.stderr or "")[-1500:].replace("\n", " | "))


def parent(scenes):
    cfg = load_config(OPT.work)
    api = Api(cfg)
    extra_path = wpath(cfg, "logs/extra_jobs.txt")
    os.makedirs(wpath(cfg, "takes/keep"), exist_ok=True)
    consumed = 0
    queue = [(n, False) for n in scenes]
    failed, rolls, best, retried = [], {}, {}, False
    say(cfg, "START " + " ".join(map(str, scenes)))
    while queue:
        n, force = queue.pop(0)
        if not force and healthy(cfg) and finalized(cfg, api, n):
            say(cfg, f"scene {n}: already finalized, skipped")
            continue
        t0 = time.time()
        ok = run_scene(cfg, n)
        say(cfg, f"scene {n}: {'DONE' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min; {len(queue)} left")
        if not ok:
            failed.append(n)
        if ok:
            try:
                h = held_count(take_path(cfg, n))
                say(cfg, f"scene {n}: {h} repeated frames")
                if n in best and h > best[n][0]:
                    st, env = api.mutate("POST", f"/projects/{{pid}}/scenes/{n}/video/select", {"source_path": best[n][1]})
                    say(cfg, f"scene {n}: retake is choppier ({h} vs {best[n][0]}), first take selected again: {st} {env.get('ok')}")
                elif h >= OPT.held_limit and rolls.get(n, 0) < 1:
                    keep = wpath(cfg, f"takes/keep/scene{n:02d}_take{rolls.get(n, 0) + 1}_held{h}.mp4")
                    shutil.copy(take_path(cfg, n), keep)
                    best[n] = (h, keep)
                    rolls[n] = rolls.get(n, 0) + 1
                    queue.append((n, True))
                    say(cfg, f"scene {n}: choppy, one retake queued")
            except Exception as exc:
                say(cfg, f"scene {n}: repeated-frame check failed: {exc}")
        if os.path.isfile(extra_path):
            lines = [l.strip() for l in open(extra_path, encoding="utf-8").read().splitlines() if l.strip()]
            for line in lines[consumed:]:
                if line.isdigit():
                    queue.append((int(line), True))
                    say(cfg, f"retake queued: scene {line}")
            consumed = len(lines)
        if not queue and failed and not retried:
            retried = True
            queue, failed = [(m, False) for m in failed], []
            say(cfg, "second round for failed scenes: " + " ".join(str(m) for m, _ in queue))
    say(cfg, f"QUEUE DONE; failed: {failed or 'none'}")
    if healthy(cfg) or start_comfy(cfg):
        stitch(cfg)
    say(cfg, "NIGHT RUN END")


def retake(scenes):
    cfg = load_config(OPT.work)
    for n in scenes:
        t0 = time.time()
        ok = run_scene(cfg, n)
        say(cfg, f"retake scene {n}: {'DONE' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min")
    say(cfg, "RETAKES END")


def main():
    global OPT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=os.getcwd())
    p.add_argument("--recover", action="store_true")
    p.add_argument("--launcher", default=None)
    p.add_argument("--held-limit", type=int, default=12)
    p.add_argument("--prefix", default=None)
    p.add_argument("command", choices=["parent", "child", "retake"])
    p.add_argument("scenes", nargs="+", type=int)
    OPT = p.parse_args()
    OPT.work = os.path.abspath(OPT.work)
    if OPT.command == "child":
        sys.exit(child(OPT.scenes[0]))
    if OPT.command == "retake":
        retake(OPT.scenes)
    else:
        parent(OPT.scenes)


if __name__ == "__main__":
    main()
