"""Check everything a music-video run needs and print what is missing, before the user walks away.

usage: check_inputs.py --work W [--refs FOLDER ...]

Checks: the song and lyrics files, ComfyUI and the Agent API, the H3 model and LoRA named in config.json, ffmpeg,
Demucs, the ElevenLabs key (tier, characters left, voice slots), reference folders, and whether Windows will sleep.
Prints OK / WARN / MISSING lines and exits 1 when anything is MISSING.
"""
import json
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, load_config, parser, wpath  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
missing = []


def report(level, text):
    print(f"{level:8s}{text}")
    if level == "MISSING":
        missing.append(text)


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    p = parser(__doc__)
    p.add_argument("--refs", nargs="*", default=[], help="reference image folders to count")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)

    for key in ("song", "lyrics"):
        path = wpath(cfg, cfg.get(key, ""))
        if not cfg.get(key) or not os.path.isfile(path):
            report("MISSING", f"{key}: {path or '(not set in config.json)'}")
            continue
        if key == "song":
            info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate,channels",
                                   "-of", "default=nw=1", path], capture_output=True, text=True).stdout.split()
            report("OK", f"song: {path} ({', '.join(info)})")
        else:
            lines = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
            sections = sum(1 for l in lines if l.strip().startswith("["))
            report("OK", f"lyrics: {len(lines) - sections} sung lines, {sections} section tags")

    try:
        stats = get_json(api.comfy + "/system_stats")
        dev = (stats.get("devices") or [{}])[0]
        report("OK", f"ComfyUI {stats.get('system', {}).get('comfyui_version')} on {dev.get('name')} "
                     f"({round((dev.get('vram_total') or 0) / 2**30)} GB)")
        queue = get_json(api.comfy + "/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            report("WARN", "ComfyUI is busy right now; renders wait until it is idle")
    except Exception as exc:
        report("MISSING", f"ComfyUI not reachable at {api.comfy} ({exc}); the user has to start it")
        stats = None
    _, health = api.call("GET", "/health")
    if health.get("ok"):
        report("OK", f"Agent API: {health['data'].get('status')}")
    else:
        report("MISSING", f"Agent API /vrgdg/api/v1/health: {health.get('error')}")

    h3 = cfg.get("minimax_h3") or {}
    if stats:
        for folder, key in (("diffusion_models", "diffusion_model_name"), ("loras", "two_pass_lora_name")):
            want = h3.get(key)
            if not want:
                continue
            try:
                have = get_json(f"{api.comfy}/models/{folder}")
            except Exception:
                have = []
            if want in have:
                report("OK", f"{key}: {want}")
            else:
                close = [m for m in have if any(w in m.lower() for w in want.lower().split("_")[:2])][:5]
                report("MISSING", f"{key} '{want}' not installed. Similar: {close}")

    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    report("OK" if ff.returncode == 0 else "MISSING", "ffmpeg on PATH" if ff.returncode == 0 else "ffmpeg not on PATH")
    demucs = subprocess.run([sys.executable, "-c", "import demucs, soundfile; print(demucs.__version__)"],
                            capture_output=True, text=True)
    report("OK" if demucs.returncode == 0 else "WARN",
           f"Demucs {demucs.stdout.strip()} in this Python" if demucs.returncode == 0
           else "Demucs/soundfile not importable in this Python; run split_stems.py with ComfyUI's python_embeded")

    if cfg.get("elevenlabs_key_file"):
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from elevenlabs import request as el_request
            status, body, _ = el_request(cfg, "GET", "/v1/user/subscription")
            if status == 200:
                sub = json.loads(body)
                left = sub.get("character_limit", 0) - sub.get("character_count", 0)
                report("OK", f"ElevenLabs: tier {sub.get('tier')}, {left} characters left, voice slots "
                             f"{sub.get('voice_slots_used')}/{sub.get('voice_limit')}")
                if sub.get("voice_slots_used", 0) >= sub.get("voice_limit", 1):
                    report("WARN", "ElevenLabs voice slots are full: use Voice Library voices by id; never delete the user's voices")
            else:
                report("MISSING", f"ElevenLabs key rejected (HTTP {status})")
        except Exception as exc:
            report("MISSING", f"ElevenLabs check failed: {exc}")

    for folder in args.refs:
        if not os.path.isdir(folder):
            report("MISSING", f"reference folder {folder}")
            continue
        n = sum(1 for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))
        report("OK" if n else "WARN", f"{n} images in {folder}")

    if os.name == "nt":
        out = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
                             capture_output=True, text=True).stdout
        ac = [l for l in out.splitlines() if "Current AC Power Setting Index" in l]
        if ac and not ac[0].strip().endswith("0x00000000"):
            report("WARN", "Windows sleeps on AC power after a timeout; an overnight render would stop")
        elif ac:
            report("OK", "Windows sleep is off on AC power")

    print()
    print("READY" if not missing else f"NOT READY: {len(missing)} missing item(s)")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
