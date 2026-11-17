"""Which instrument plays in which scene: per-scene activity of every stem, from the timeline.

usage: stem_scenes.py --work W STEMS_DIR [--timeline timeline.json] [--threshold -40] [--offset 0]
       stem_scenes.py --work W STEMS_DIR --grid 1.0      the same as a time grid (one row per second)

Prints one row per scene: for every stem the share of 0.25 s windows above the threshold (dBFS), e.g. "drums 100%",
"bass 40% (from 3.2 s)". Use it to write each scene: a player is shown playing only where their stem plays (the user
rule: "only describe a musician actively playing when their instrument is audible"), the intro shows only the
instruments that are in, backing vocals only where that stem is active, and solos go to the stem that leads.
STEMS_DIR: the user's stems (any *.wav) or the Builder's (audio/builder_stems.json "stems").
--offset: seconds of intro added before the song in the full track (config intro_seconds).
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_config, parser, read_json, wpath  # noqa: E402

WIN = 0.25


def levels(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    step = int(WIN * sr)
    n = len(mono) // step
    rms = np.sqrt((mono[: n * step].reshape(n, step) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms)


def main():
    p = parser(__doc__)
    p.add_argument("stems_dir")
    p.add_argument("--timeline", default="timeline.json")
    p.add_argument("--threshold", type=float, default=-40.0)
    p.add_argument("--offset", type=float, default=None)
    p.add_argument("--grid", type=float, default=None)
    a = p.parse_args()
    cfg = load_config(a.work)
    offset = float(cfg.get("intro_seconds", 0.0) if a.offset is None else a.offset)
    files = [f for f in sorted(glob.glob(os.path.join(a.stems_dir, "*.wav"))) if not any(k in os.path.basename(f).lower() for k in ("no_vocals", "mix", "song_"))]
    if not files:
        sys.exit(f"[VRGDG MV] no .wav stems in {a.stems_dir}")
    names = [os.path.splitext(os.path.basename(f))[0][:10] for f in files]
    lv = [levels(f) for f in files]
    if a.grid:
        k = max(1, int(round(a.grid / WIN)))
        print("time    " + " ".join(f"{n:>10}" for n in names))
        for i in range(0, max(len(x) for x in lv), k):
            cells = ["#" if i < len(x) and x[i:i + k].max() > a.threshold else "." for x in lv]
            print(f"{i * WIN + offset:6.1f}  " + " ".join(f"{c:>10}" for c in cells))
        return
    tl = read_json(wpath(cfg, a.timeline), {}) or {}
    for n, sc in enumerate(tl.get("scenes") or [], start=1):
        s, e = float(sc[0]) - offset, float(sc[1]) - offset
        parts = []
        for name, x in zip(names, lv):
            seg = x[max(0, int(s / WIN)): max(0, int(e / WIN))]
            if not len(seg):
                continue
            on = seg > a.threshold
            share = on.mean()
            if share >= 0.05:
                first = s + WIN * int(np.argmax(on))
                parts.append(f"{name} {share:.0%}" + (f" (from {first - s:.1f} s)" if first - s > 0.5 else ""))
        print(f"scene {n:2d} {sc[0]:7.2f}-{sc[1]:7.2f}: " + (", ".join(parts) or "silence"))


if __name__ == "__main__":
    main()
