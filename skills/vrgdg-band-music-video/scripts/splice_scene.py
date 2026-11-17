"""Replace scenes in a finished film with edited clips, frame-exactly, keeping the film's audio untouched.

usage: splice_scene.py --work W --film FINAL.mp4 --out FINAL_v2.mp4 SCENE=CLIP.mp4 [SCENE=CLIP.mp4 ...]
       splice_scene.py --work W --film FINAL.mp4 --out FINAL_v2.mp4 --append CLIP.mp4 [--pad-audio-seconds 5]

Use this instead of the stitch route's `overlays`: the Builder stitch adds ONE extra frame per overlay, so everything
after the insert runs a frame late against the song (seen 2026-10-25: v4 +1 frame, v5 +2).
A scene occupies frames round(start*24) up to round(end*24) (the stitch's own rule; start/end from timeline.json, or
the project's session for scenes added later). Each clip must have at least that many frames; extra frames at the end
(render padding) are dropped. --append adds a clip after the last frame (an ending after the song) and pads the audio
with silence to the new length. The video is re-encoded once (x264 CRF 14); the audio is copied unless padding is
needed. Prints a frame check per spliced scene (PSNR ~45+ = the same frame).
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, parser, wpath  # noqa: E402

FPS = 24


def frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return int(out.strip())


def psnr(a, fa, b, fb):
    tmp = tempfile.mkdtemp()
    x, y = os.path.join(tmp, "x.png"), os.path.join(tmp, "y.png")
    for src, f, dst in ((a, fa, x), (b, fb, y)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"select=eq(n\\,{f})", "-vsync", "0", dst],
                       check=True)
    out = subprocess.run(["ffmpeg", "-i", x, "-i", y, "-lavfi", "psnr", "-f", "null", "-"], capture_output=True,
                         text=True).stderr
    return out.split("average:")[-1].split()[0] if "average:" in out else "?"


def main():
    p = parser(__doc__)
    p.add_argument("--film", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--append", default=None)
    p.add_argument("--pad-audio-seconds", type=float, default=0.0)
    p.add_argument("swaps", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.work)
    scenes = json.load(open(wpath(cfg, "timeline.json"), encoding="utf-8"))["scenes"]
    total = frames(args.film)
    inputs, parts, cursor, checks = ["-i", args.film], [], 0, []
    swaps = []
    for item in args.swaps:
        n, clip = item.split("=", 1)
        n = int(n)
        start, end = round(float(scenes[n - 1][0]) * FPS), round(float(scenes[n - 1][1]) * FPS)
        if frames(clip) < end - start:
            sys.exit(f"scene {n}: clip has {frames(clip)} frames, the slot needs {end - start}")
        swaps.append((start, end, clip, n))
    swaps.sort()
    filters = []
    for i, (start, end, clip, n) in enumerate(swaps, 1):
        inputs += ["-i", clip]
        filters.append(f"[0:v]trim=start_frame={cursor}:end_frame={start},setpts=PTS-STARTPTS[k{i}]")
        filters.append(f"[{i}:v]trim=end_frame={end - start},setpts=PTS-STARTPTS[s{i}]")
        parts += [f"[k{i}]", f"[s{i}]"]
        checks.append((n, start + 2, clip, 2))
        cursor = end
    filters.append(f"[0:v]trim=start_frame={cursor},setpts=PTS-STARTPTS[tail]")
    parts.append("[tail]")
    if args.append:
        k = len(swaps) + 1
        inputs += ["-i", args.append]
        filters.append(f"[{k}:v]setpts=PTS-STARTPTS[app]")
        parts.append("[app]")
    filters.append("".join(parts) + f"concat=n={len(parts)}:v=1:a=0,format=yuv420p[v]")
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]"]
    pad = args.pad_audio_seconds or (frames(args.append) / FPS if args.append else 0.0)
    if pad:
        dur = total / FPS + pad
        cmd += ["-map", "0:a", "-af", f"apad=whole_dur={dur}", "-c:a", "aac", "-b:a", "320k"]
    else:
        cmd += ["-map", "0:a", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-r", str(FPS), args.out]
    subprocess.run(cmd, check=True)
    print(f"{args.out}: {frames(args.out)} frames (film had {total})")
    for n, f, clip, cf in checks:
        print(f"  scene {n}: out frame {f} vs clip frame {cf}: PSNR {psnr(args.out, f, clip, cf)}")


if __name__ == "__main__":
    main()
