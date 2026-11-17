"""Doubled-frame check for finished takes (a clip that is really 12 fps shown at 24 fps looks choppy).

usage: frame_check.py CLIP.mp4 [CLIP2.mp4 ...] [--threshold 0.5] [--max-dups 3]

For each clip it measures the mean absolute difference between neighbouring frames (grayscale, 192x108) and counts
near-duplicates (difference below the threshold). A healthy H3 take has 0-2; DEAD OF THE PARTY v3 scene 12 had 65 at the
default threshold (86 of 172 below 1.0, all odd frames), which the user saw as a choppy frame rate across the whole scene. Stills QA sheets can't show
this, so run it on every finalized take. Exit code 1 when any clip has more than --max-dups near-duplicates.
To hand such a take to Topaz for interpolation instead of re-rendering:
  ffmpeg -i take.mp4 -vf "select='not(mod(n\\,2))',setpts=N/12/TB" -r 12 -c:v libx264 -crf 10 -c:a copy take_12fps.mp4
"""
import argparse
import subprocess
import sys

import numpy as np

W, H = 192, 108


def frame_diffs(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-dups", type=int, default=3)
    a = ap.parse_args()
    bad = False
    for clip in a.clips:
        d = frame_diffs(clip)
        dups = np.where(d < a.threshold)[0] + 1
        odd = len(dups) > 4 and all(i % 2 == 1 for i in dups)
        flag = "DOUBLED" if len(dups) > a.max_dups else "ok"
        bad |= flag != "ok"
        note = " (every other frame: effectively 12 fps)" if odd else ""
        print(f"{flag:7} {clip}: {len(d) + 1} frames, {len(dups)} near-duplicates{note}, mean diff {d.mean():.1f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
