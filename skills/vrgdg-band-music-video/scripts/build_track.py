"""Build the project audio: optional spoken intro + the song + optional outro, and the same with the vocal-free song.

usage: build_track.py --work W [--plan track_plan.json]
       -> audio/intro.wav, audio/outro.wav, audio/full_track.wav, audio/full_track_no_vocals.wav

track_plan.json (see assets/track_plan_example.json):
  {"intro_seconds": 16.0, "outro_seconds": 17.0,
   "intro": [{"file": "audio/sfx/street.mp3", "at": 0.0, "lufs": -27, "fade_in": 0.05, "fade_out": 1.0},
             {"file": "audio/voice/i_drummer.mp3", "at": 1.0, "lufs": -16}, ...],
   "outro": [...]}
Each item is loudness-normalised with ffmpeg loudnorm (voices about -16 LUFS like a -16 LUFS song, ambience -27 to
-30), optionally trimmed ("trim": [a, b] seconds), faded and placed at "at" seconds. The song keeps its own level.
Everything is written at the song's sample rate, 16-bit stereo. The no-vocals song comes from config
"song_no_vocals" (default audio/stems/song_no_vocals.wav, made by split_stems.py).
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, parser, read_json, wpath  # noqa: E402


def load(path, lufs, sr):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-af", f"loudnorm=I={lufs}:TP=-2:LRA=20",
           "-ar", str(sr), "-ac", "2", "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def place(buf, clip, at, sr, fade_in=0.0, fade_out=0.0, gain_db=0.0):
    clip = clip * (10 ** (gain_db / 20))
    n = len(clip)
    if fade_in > 0:
        k = min(n, int(fade_in * sr))
        clip[:k] *= np.linspace(0, 1, k)[:, None]
    if fade_out > 0:
        k = min(n, int(fade_out * sr))
        clip[n - k:] *= np.linspace(1, 0, k)[:, None]
    s = int(round(at * sr))
    e = min(len(buf), s + n)
    if e > s:
        buf[s:e] += clip[: e - s]


def mix(cfg, items, seconds, sr):
    buf = np.zeros((int(round(seconds * sr)), 2), dtype=np.float32)
    for item in items:
        clip = load(wpath(cfg, item["file"]), item.get("lufs", -16), sr)
        if "trim" in item:
            a, b = item["trim"]
            clip = clip[int(a * sr): int(b * sr)]
        place(buf, clip, item["at"], sr, item.get("fade_in", 0.0), item.get("fade_out", 0.0), item.get("gain_db", 0.0))
    peak = float(np.abs(buf).max()) if len(buf) else 0.0
    if peak > 0.95:
        buf *= 0.95 / peak
    return buf


def stereo(path):
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2], sr


def main():
    p = parser(__doc__)
    p.add_argument("--plan", default="track_plan.json")
    args = p.parse_args()
    cfg = load_config(args.work)
    plan = read_json(wpath(cfg, args.plan), {"intro_seconds": 0, "outro_seconds": 0, "intro": [], "outro": []})
    song, sr = stereo(wpath(cfg, cfg["song"]))
    inst_path = wpath(cfg, cfg.get("song_no_vocals", "audio/stems/song_no_vocals.wav"))
    inst = stereo(inst_path)[0][: len(song)] if os.path.isfile(inst_path) else None
    os.makedirs(wpath(cfg, "audio"), exist_ok=True)
    intro = mix(cfg, plan.get("intro", []), float(plan.get("intro_seconds", 0)), sr)
    outro = mix(cfg, plan.get("outro", []), float(plan.get("outro_seconds", 0)), sr)
    if len(intro):
        sf.write(wpath(cfg, "audio/intro.wav"), intro, sr, subtype="PCM_16")
    if len(outro):
        sf.write(wpath(cfg, "audio/outro.wav"), outro, sr, subtype="PCM_16")
    full = np.concatenate([intro, song, outro])
    sf.write(wpath(cfg, "audio/full_track.wav"), np.clip(full, -1, 1), sr, subtype="PCM_16")
    if inst is not None:
        if len(inst) < len(song):
            inst = np.concatenate([inst, np.zeros((len(song) - len(inst), 2), dtype=np.float32)])
        full_inst = np.concatenate([intro, inst, outro])
        sf.write(wpath(cfg, "audio/full_track_no_vocals.wav"), np.clip(full_inst, -1, 1), sr, subtype="PCM_16")
    else:
        print("WARN no vocal-free song yet (run split_stems.py); full_track_no_vocals.wav not written")
    print(f"intro {len(intro) / sr:.2f} s + song {len(song) / sr:.2f} s + outro {len(outro) / sr:.2f} s "
          f"= {len(full) / sr:.2f} s at {sr} Hz")
    if cfg.get("intro_seconds") not in (None, plan.get("intro_seconds", 0)):
        print(f"WARN config intro_seconds ({cfg.get('intro_seconds')}) differs from the plan ({plan.get('intro_seconds')}); "
              "lyric offsets use config intro_seconds")


if __name__ == "__main__":
    main()
