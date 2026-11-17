"""Count repeated (held) frames in finished takes. Singularity V2 pass 1 sometimes repeats every 2nd or 4th frame,
which looks choppy (12 fps shown as 24) and is invisible on still QA sheets.

usage: held_frames.py CLIP.mp4 [CLIP2.mp4 ...]
Prints per clip: frames, held frames, median frame difference and the runs where holds cluster (start-end s, count).
Rule of thumb from two videos: 0-6 clean, 7-11 mild, 12+ choppy (re-roll once, then smooth with rife_fix.py).
Needs numpy (ComfyUI's embedded Python has it) and ffmpeg on PATH.
"""
import subprocess
import sys

W, H = 192, 108


def _diffs(path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
    return f, np.abs(f[1:] - f[:-1]).mean(axis=(1, 2))


def held_index(path):
    """Frame numbers that repeat the frame before (tiny difference while both neighbours move)."""
    _, d = _diffs(path)
    out = []
    for i in range(1, len(d) - 1):
        nb = max(d[i - 1], d[i + 1])
        if d[i] < 0.5 and nb > 3 * max(d[i], 0.05) and nb > 0.8:
            out.append(i + 1)
    return out


def held_count(path):
    return len(held_index(path))


def main():
    import numpy as np
    for path in sys.argv[1:]:
        f, d = _diffs(path)
        idx = held_index(path)
        runs, cur = [], []
        for i in idx:
            if cur and i - cur[-1] <= 3:
                cur.append(i)
            else:
                if cur:
                    runs.append(cur)
                cur = [i]
        if cur:
            runs.append(cur)
        txt = ", ".join(f"{r[0] / 24:.2f}-{(r[-1] + 1) / 24:.2f}s({len(r)})" for r in runs if len(r) >= 3)
        print(f"{path}: frames {len(f)}, held {len(idx)}, median diff {np.median(d):.2f} | runs: {txt or '-'}")


if __name__ == "__main__":
    main()
