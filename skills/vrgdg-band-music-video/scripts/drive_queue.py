"""Self-driving render queue for unattended runs: never waits for a person and never stops on its own.

usage: drive_queue.py --work W [--project P] --phase draft|finalize|both [--tag T] 3 5 9 ...

draft     draft every scene (one retry each). The user reviews the drafts afterwards. The normal mode while they are here.
finalize  finalize every scene's latest draft.
both      draft then finalize each scene before moving on (only when the user said to run everything unattended).

Finalize out-of-memory ladder (the pass-2 latent upscaler on 10-12 s scenes, random): 2 tries as set, then the
scene's ref_image_size -> "match" and 2 more tries, then a re-draft at "match" and 2 more. Memory guard trips: wait 3
minutes, clear again, retry; after two trips run anyway with the guard relaxed. Run it with run_in_background so a
notification arrives when it ends; log lines go to logs/driver.log and logs/render.log.
Never edit this queue's inputs while it runs; start a new queue instead.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, load_config, log, parser, read_json, wpath  # noqa: E402
from render_guarded import clear_memory, is_draft, segment, with_phase  # noqa: E402
from render_queue import render  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def say(cfg, msg):
    log(cfg, "driver: " + msg, name="driver.log")


def guarded(cfg, api_base, n, phase, tag, plan):
    """One render with memory clearing and the relaxed guard on repeat trips. True when the render saved."""
    limits = (60.0, 95.0)
    for trip in range(3):
        ram, vram, commit, limit = clear_memory(cfg, f"scene {n}: ")
        if (ram > limits[0] or (limit and commit > limits[1])) and trip < 2:
            say(cfg, f"scene {n} {phase}: memory guard tripped (RAM {ram:.1f} GB, commit {commit:.1f} GB); waiting 3 min")
            time.sleep(180)
            continue
        api = with_phase(Api(cfg, api_base.pid), phase)
        scene_plan = plan
        gated = wpath(cfg, os.path.join("audio", "gated", f"scene_{n:02d}.wav"))
        if os.path.isfile(gated):
            scene_plan = {"no_vocals_path": os.path.abspath(gated), "scenes": {str(n): {"audio": "no_vocals"}}}
        ok = render(cfg, api, n, scene_plan, tag, "")
        api_base.wait_idle()
        ram, vram, commit, limit = clear_memory(cfg, f"scene {n} after {phase}: ")   # clear after every render too (user rule)
        say(cfg, f"scene {n} {phase}: after-render clear, RAM {ram:.1f} GB, VRAM {vram:.1f} GB")
        return ok
    return False


def set_ref_size(cfg, api, n, size):
    subprocess.run([sys.executable, os.path.join(HERE, "set_h3.py"), "--work", cfg["_work"], "--project", str(api.pid),
                    str(n), f"ref_image_size={size}"], check=False)


def draft(cfg, api, n, tag, plan):
    for attempt in (1, 2):
        if guarded(cfg, api, n, "draft", tag, plan):
            say(cfg, f"scene {n} draft {tag} done")
            return True
        say(cfg, f"scene {n} draft try {attempt} failed")
    return False


def finalize(cfg, api, n, tag, plan):
    for attempt in range(1, 7):
        if attempt == 3:
            set_ref_size(cfg, api, n, "match")
            say(cfg, f"scene {n}: ref_image_size -> match")
        if attempt == 5:
            say(cfg, f"scene {n}: re-draft at match before the last finalize tries")
            if not draft(cfg, api, n, tag + "m", plan):
                break
        guarded(cfg, api, n, "finalize", f"f{attempt}", plan)
        if not is_draft(cfg, api, n):
            say(cfg, f"scene {n} FINALIZED (try {attempt})")
            return True
        say(cfg, f"scene {n} finalize try {attempt} failed")
    say(cfg, f"scene {n} stays a DRAFT")
    return False


def main():
    p = parser(__doc__)
    p.add_argument("scenes", nargs="+", type=int)
    p.add_argument("--phase", choices=["draft", "finalize", "both"], required=True)
    p.add_argument("--tag", default="d1")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    plan = read_json(wpath(cfg, "render_plan.json"), {}) or {}
    say(cfg, f"START {args.phase} {args.tag}: " + " ".join(map(str, args.scenes)))
    for n in args.scenes:
        api.wait_idle()
        if args.phase in ("draft", "both"):
            if not draft(cfg, api, n, args.tag, plan):
                continue
        if args.phase in ("finalize", "both"):
            if not is_draft(cfg, api, n):
                say(cfg, f"scene {n} has no draft to finalize")
                continue
            finalize(cfg, api, n, args.tag, plan)
    say(cfg, f"QUEUE DONE {args.phase} {args.tag}")


if __name__ == "__main__":
    main()
