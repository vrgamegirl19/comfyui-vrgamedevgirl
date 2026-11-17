"""Smooth a choppy take (repeated frames): keep every 2nd frame (12 fps) and interpolate back to 24 fps with RIFE
in ComfyUI, then put the real frame back at every hard cut and copy the original audio.

usage: rife_fix.py IN.mp4 OUT.mp4 [--ckpt rife49|rife47] [--comfy http://127.0.0.1:8188] [--output COMFY_OUTPUT_DIR]

Needs the custom nodes FL_RIFE (ComfyUI_Fill-Nodes) and VHS_LoadVideoPath / VHS_VideoCombine
(ComfyUI-VideoHelperSuite). Check they exist first: GET /object_info/FL_RIFE. One GPU job; run it only when the
ComfyUI queue is idle. About a minute per 7 s scene on an RTX 5090.

The result has the same frame count, frame rate and audio stream as IN, so it drops into the timeline with
POST /projects/{pid}/scenes/{n}/video/select {"source_path": OUT} and keeps 0 ms sync.
Why the cut fix: RIFE blends the two shots in the one in-between frame that spans a hard cut (a ghost frame).
Tested on NO PERMISSION TO EXIST scenes 5 and 6: repeated frames 47 -> 0 and 29 -> 0. Watch hands and flying
objects for smearing afterwards.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def post(comfy, path, body):
    req = urllib.request.Request(comfy + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return int(out.strip())


def cuts(path):
    """Frame numbers where a hard cut starts (a frame that differs sharply from the one before)."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=192:108,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, 108, 192).astype(np.float32)
    d = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2))
    return [i + 1 for i, x in enumerate(d) if x > 25 and x > 2.5 * np.median(d[d > 1.0])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--ckpt", default="rife49")
    p.add_argument("--comfy", default="http://127.0.0.1:8188")
    p.add_argument("--output", default=None, help="ComfyUI output folder (only needed when the history has no fullpath)")
    a = p.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    total = frames(src)
    prefix = "VRGDG_rife_fix/" + os.path.splitext(os.path.basename(dst))[0]
    graph = {
        "1": {"class_type": "VHS_LoadVideoPath", "inputs": {"video": src, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                                                            "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 2}},
        "2": {"class_type": "FL_RIFE", "inputs": {"images": ["1", 0], "ckpt_name": a.ckpt, "multiplier": 2, "ensemble": True}},
        "3": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["2", 0], "frame_rate": 24.0, "loop_count": 0,
                                                           "filename_prefix": prefix, "format": "video/h264-mp4",
                                                           "pix_fmt": "yuv420p", "crf": 8, "save_metadata": False,
                                                           "trim_to_audio": False, "pingpong": False, "save_output": True}},
    }
    pid = post(a.comfy, "/prompt", {"prompt": graph})["prompt_id"]
    while True:
        time.sleep(3)
        hist = json.loads(urllib.request.urlopen(f"{a.comfy}/history/{pid}", timeout=60).read())
        if pid in hist:
            break
    entry = hist[pid]
    if entry.get("status", {}).get("status_str") != "success":
        sys.exit("[VRGDG] RIFE job failed: " + json.dumps(entry.get("status"))[:800])
    gifs = [g for out in entry["outputs"].values() for g in out.get("gifs", [])]
    last = gifs[-1]
    video = last.get("fullpath") or os.path.join(a.output or "", last["subfolder"], last["filename"])
    if not os.path.isfile(video):
        sys.exit(f"[VRGDG] RIFE output not found: {video} (pass --output <ComfyUI output folder>)")
    got = frames(video)
    pad = max(0, total - got)
    fix = [c if c % 2 else c - 1 for c in cuts(src)]
    enable = "+".join(f"eq(n\\,{k})" for k in fix) or "0"
    fc = (f"[0:v]tpad=stop_mode=clone:stop={pad},setpts=N/24/TB[r];[1:v]setpts=N/24/TB[s];"
          f"[r][s]overlay=enable='{enable}':eof_action=pass[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", src, "-filter_complex", fc, "-map", "[v]",
                    "-map", "1:a:0", "-frames:v", str(total), "-r", "24", "-c:v", "libx264", "-crf", "15",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", dst], check=True)
    print(f"[VRGDG] {os.path.basename(src)}: {total} frames -> RIFE {got} (+{pad} padded), cut frames restored {fix} "
          f"-> {dst} ({frames(dst)} frames)")


if __name__ == "__main__":
    main()
