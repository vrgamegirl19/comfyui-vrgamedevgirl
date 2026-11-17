"""Print where the isolated vocal stem is active, to find instrumental gaps and ad-libs the lyrics file lacks.

usage: vocal_activity.py VOCALS.wav [--offset SECONDS] [--threshold -38] [--detail START END]

--offset adds the intro length so times match the full track. --detail prints the level every 0.25 s between two
song-relative times (useful when the transcript shows a gap that the stem says is sung).
"""
import argparse

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("vocals")
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=-38.0, help="dBFS RMS level that counts as singing")
    p.add_argument("--detail", nargs=2, type=float)
    a = p.parse_args()
    audio, sr = sf.read(a.vocals, always_2d=True)
    mono = audio.mean(1)
    hop = int(sr * 0.1)
    db = np.array([20 * np.log10(np.sqrt(np.mean(mono[i:i + hop] ** 2)) + 1e-9) for i in range(0, len(mono) - hop, hop)])
    regions, start = [], None
    for i, active in enumerate(db > a.threshold):
        t = i * 0.1
        if active and start is None:
            start = t
        if not active and start is not None:
            if t - start >= 0.15:
                regions.append([start, t])
            start = None
    if start is not None:
        regions.append([start, len(db) * 0.1])
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] < 0.35:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    print(f"vocal-active regions (threshold {a.threshold} dB, times +{a.offset} s):")
    prev = 0.0
    for s, e in merged:
        gap = s - prev
        print(f"  {s + a.offset:7.2f} - {e + a.offset:7.2f}   ({e - s:5.2f} s sung, gap before {gap:5.2f} s)")
        prev = e
    if a.detail:
        q = int(sr * 0.25)
        print(f"\nlevel every 0.25 s from {a.detail[0]} to {a.detail[1]} s (song time):")
        row = []
        for t in np.arange(a.detail[0], a.detail[1], 0.25):
            seg = mono[int(t * sr):int(t * sr) + q]
            row.append(f"{20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9):4.0f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
