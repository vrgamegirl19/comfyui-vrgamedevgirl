"""Mix the user's own stems without the vocals into the vocal-free song (cleaner than a Demucs split).

usage: mix_stems.py --work W STEMS_DIR [--exclude vocal] [--out audio/stems/song_no_vocals.wav]

Every *.wav in STEMS_DIR whose name contains none of the --exclude words (default: "vocal", which drops lead and
backing vocals; full mixes are always skipped) is summed at its own level. The result keeps the length of the
longest stem, is clipped to [-1, 1] and written 16-bit at the stems' sample rate. Sets config "song_no_vocals" so
build_track.py writes audio/full_track_no_vocals.wav from it. NO PERMISSION TO EXIST used stems 2-7 (drums, bass,
guitar, strings, other, woodwinds).
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_config, parser, save_config, wpath  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("stems_dir")
    p.add_argument("--exclude", nargs="*", default=["vocal"])
    p.add_argument("--out", default="audio/stems/song_no_vocals.wav")
    a = p.parse_args()
    cfg = load_config(a.work)
    files = []
    for f in sorted(glob.glob(os.path.join(a.stems_dir, "*.wav"))):
        name = os.path.basename(f).lower()
        if "mix" in name or any(x.lower() in name for x in a.exclude):
            continue
        files.append(f)
    if not files:
        sys.exit("[VRGDG MV] no stems left to mix")
    mix, sr = None, None
    for f in files:
        data, rate = sf.read(f, dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        if sr is None:
            sr = rate
        elif rate != sr:
            sys.exit(f"[VRGDG MV] {f} is {rate} Hz, the others {sr} Hz")
        if mix is None:
            mix = data[:, :2].copy()
        else:
            if len(data) > len(mix):
                mix = np.pad(mix, ((0, len(data) - len(mix)), (0, 0)))
            mix[: len(data)] += data[:, :2]
        print("  +", os.path.basename(f))
    out = wpath(cfg, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    peak = float(np.abs(mix).max())
    sf.write(out, np.clip(mix, -1, 1), sr, subtype="PCM_16")
    cfg["song_no_vocals"] = a.out
    save_config(cfg)
    print(f"[VRGDG MV] {len(files)} stems -> {out} ({len(mix) / sr:.2f} s, peak {peak:.2f}"
          + (", clipped: lower the stems" if peak > 1.0 else "") + ")")


if __name__ == "__main__":
    main()
