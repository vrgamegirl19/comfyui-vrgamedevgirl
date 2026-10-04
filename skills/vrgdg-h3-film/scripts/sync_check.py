"""Audio/video sync check of the finished film, every scene (run with qa_python).

  python sync_check.py <project> [film.mp4]      default film: final/<TITLE>.mp4

For each scene in play order:
  audio: the loudest 2 s of the clip's own audio is cross-correlated against the final's audio near where the scene
         should start -> where the scene's sound really lands (ms precision);
  video: a run of the clip's frames (small, grey, normalised) is matched against the final's frames -> where the
         scene's picture really lands (1-frame precision).
Prints both offsets and their difference (lip-sync error). Fails (exit 1) on any |audio - video| over 40 ms, any
placement error over 80 ms, or a weak match.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ffprobe_duration, load_settings, project_dir  # noqa: E402

SR, W, H = 16000, 64, 36
LIMIT_MS, PLACE_MS = 40, 80


def audio(ffmpeg, path, ss=0.0, t=None):
    cmd = [ffmpeg, "-v", "error", "-ss", f"{max(0.0, ss):.3f}", "-i", path] + (["-t", f"{t:.3f}"] if t else []) + \
          ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)


def frames(ffmpeg, path, fps, first, n):
    """n decoded frames starting at frame index `first`; seeking half a frame early makes the first one exact."""
    cmd = [ffmpeg, "-v", "error", "-ss", f"{max(0.0, (first - 0.5) / fps):.4f}", "-i", path, "-frames:v", str(n),
           "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    raw = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.uint8).astype(np.float32)
    f = raw[: len(raw) // (W * H) * W * H].reshape(-1, W * H)
    return (f - f.mean(1, keepdims=True)) / (f.std(1, keepdims=True) + 1e-3)


def audio_offset(ffmpeg, clip, film, start, dur):
    a = audio(ffmpeg, clip)
    win = 2 * SR
    if len(a) < win:
        return None, 0.0
    energy = np.convolve(a ** 2, np.ones(SR // 10), "valid")[::SR // 10]
    if 10 * np.log10(energy.max() / (SR // 10) + 1e-12) < -35:   # ambience only: nothing to lip-sync
        return None, 0.0
    k = int(np.argmax([energy[i:i + 20].sum() for i in range(max(1, len(energy) - 20))])) * (SR // 10)
    k = min(k, len(a) - win)
    probe = a[k:k + win]
    seg = audio(ffmpeg, film, start + k / SR - 0.5, 3.0)
    if len(seg) < win:
        return None, 0.0
    c = np.correlate(seg - seg.mean(), probe - probe.mean(), "valid")
    best = int(np.argmax(c))
    strength = c[best] / (np.linalg.norm(probe - probe.mean()) * np.linalg.norm(seg[best:best + win] - seg[best:best + win].mean()) + 1e-9)
    return (best / SR - 0.5) * 1000, float(strength)


def video_offset(ffmpeg, clip, film, start, dur, fps, n=24, search=6):
    """Picture placement measured at 25/50/75% of the clip; the best-contrast measurement wins (still shots match
    ambiguously, a real dropped/duplicated frame shows at every point after it)."""
    found = []
    for frac in (0.25, 0.5, 0.75):
        m = max(0, round(dur * fps * frac) - n // 2)
        ref = frames(ffmpeg, clip, fps, m, n)
        seg = frames(ffmpeg, film, fps, round(start * fps) + m - search, n + 2 * search)
        if len(ref) < n or len(seg) < n + 2 * search:
            continue
        errs = [np.abs(seg[s:s + n] - ref).mean() for s in range(2 * search + 1)]
        best = int(np.argmin(errs))
        found.append(((np.median(errs) - errs[best]) / (np.median(errs) + 1e-6), (best - search) / fps * 1000))
    if not found:
        return None, 0.0
    contrast, ms = max(found)
    return ms, float(contrast)


def main():
    project, args = project_dir()
    settings = load_settings(project)
    ffmpeg, fps = settings["ffmpeg"], settings["fps"]
    sp = json.load(open(os.path.join(project, "screenplay.json"), encoding="utf-8"))
    score = json.load(open(os.path.join(project, "score.json"), encoding="utf-8"))
    board = [s["scene_number"] for s in json.load(open(os.path.join(project, "storyboard", "scenes.json"), encoding="utf-8"))["scenes"]]
    order = score.get("play_order") or board
    film = args[0] if args else os.path.join(project, "final", f"{sp['title']}.mp4")
    bad, worst, t = [], 0.0, 0.0
    for n in order:
        clip = os.path.join(project, "video_clips", f"scene_{n:03d}.mp4")
        dur = round(ffprobe_duration(ffmpeg, clip) * fps) / fps
        a, a_str = audio_offset(ffmpeg, clip, film, t, dur)
        v, v_str = video_offset(ffmpeg, clip, film, t, dur, fps)
        line = (f"scene {n:3d} at {t:7.2f}s  audio {'n/a' if a is None else round(a):>5} ms ({a_str:.2f})  "
                f"video {'n/a' if v is None else round(v):>5} ms ({v_str:.2f})")
        if a is not None and v is not None:
            diff = a - v
            worst = max(worst, abs(diff))
            line += f"  a-v {diff:+.0f} ms"
            if abs(diff) > LIMIT_MS or abs(a) > PLACE_MS or abs(v) > PLACE_MS:
                bad.append(n)
                line += "  <-- OUT OF SYNC"
        if (a is not None and a_str < 0.5) or (v is not None and v_str < 0.05):
            bad.append(n)
            line += "  <-- weak match, check by eye/ear"
        print(line)
        t += dur
    print(f"worst audio-vs-picture offset: {worst:.0f} ms (one video frame = {1000 / fps:.0f} ms)")
    if bad:
        print(f"SYNC CHECK FAILED for scenes {sorted(set(bad))}")
        raise SystemExit(1)
    print("sync ok")


if __name__ == "__main__":
    main()
