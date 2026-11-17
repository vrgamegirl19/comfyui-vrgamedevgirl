"""Check every scene's audio sync inside the finished video.

usage: sync_check.py --work W [--project P] FINAL.mp4

For each scene, a 3 s slice of the clip's own audio is cross-correlated with the final's audio around the scene's
timeline position. Scenes rendered against the vocal-free track still match (the music is identical). Prints the
offset per scene in ms and exits 1 when a scene is over 40 ms or matches weakly (< 0.35). A scene that starts at
0 s is probed later in the clip so the search window never runs off the start of the file.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, load_config, parser, video_path  # noqa: E402

SR = 8000
SEARCH = 0.25


def pcm(path, start, dur):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-i", path, "-t", f"{dur:.3f}",
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.float32)


def best_offset(clip, ref):
    clip = clip - clip.mean()
    norm = np.linalg.norm(clip) + 1e-9
    best, best_k = -1.0, 0
    for step, lo, hi in ((4, 0, len(ref) - len(clip)), (1, None, None)):
        if lo is None:
            lo, hi = max(0, best_k - 4), min(len(ref) - len(clip), best_k + 4)
        for k in range(lo, hi + 1, step):
            seg = ref[k:k + len(clip)]
            seg = seg - seg.mean()
            c = float(np.dot(seg, clip) / ((np.linalg.norm(seg) + 1e-9) * norm))
            if c > best:
                best, best_k = c, k
    return best_k, best


def main():
    p = parser(__doc__)
    p.add_argument("final")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    worst, flagged = 0.0, []
    for scene in api.scenes():
        n = int(scene["number"])
        path = video_path(scene)
        if not path or not os.path.isfile(path):
            print(f"scene {n:2d}: no video")
            flagged.append(n)
            continue
        start, end = float(scene["start"]), float(scene["end"])
        probe_at = max(0.2, SEARCH + 0.05 - start)
        probe = min(3.0, end - start - probe_at - 0.05)
        if probe < 0.5:
            print(f"scene {n:2d}: too short to check")
            continue
        clip = pcm(path, probe_at, probe)
        ref = pcm(args.final, start + probe_at - SEARCH, probe + 2 * SEARCH)
        if len(clip) < SR // 2 or len(ref) < len(clip):
            print(f"scene {n:2d}: could not read audio")
            flagged.append(n)
            continue
        k, match = best_offset(clip, ref)
        offset_ms = (k / SR - SEARCH) * 1000
        worst = max(worst, abs(offset_ms))
        flag = ""
        if abs(offset_ms) > 40 or match < 0.35:
            flag = "  <-- CHECK"
            flagged.append(n)
        print(f"scene {n:2d}: offset {offset_ms:+6.1f} ms  match {match:.2f}{flag}")
    print(f"worst offset {worst:.1f} ms; flagged: {flagged or 'none'}")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
