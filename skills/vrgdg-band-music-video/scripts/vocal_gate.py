"""A render-only copy of the song with the lead vocal removed inside given windows of one scene.

usage: vocal_gate.py --work W SCENE START-END [START-END ...]     (times are scene-relative seconds)
       -> audio/gated/scene_NN.wav  (full song length, so the Builder slices it exactly like the project audio)

Use it when the camera is on someone who must NOT lip-sync the lyric playing at that moment (the singer off screen,
a guest in close-up during "He called her darling"). With no lead vocal in the window nobody has words to sync to.
render_guarded.py and drive_queue.py pick the file up automatically, for drafts and finalizes. The final stitch uses
the real song, so the audience still hears every word. Pair it with a silent mouth in the prompt ("his lips pressed
together, his mouth closed and still for the whole shot"): talking verbs (murmuring, whispering) still move the mouth.

Inputs (config.json, relative to the work folder): "song_path" (default source/song.wav) and "lead_stem_path"
(default "source/stems/0 Lead Vocals.wav"); the lead stem is subtracted from the mix (gain 0.95) with 60 ms fades.
Scene start times come from timeline.json.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, parser, wpath  # noqa: E402

GAIN = 0.95
FADE = 0.06


def main():
    p = parser(__doc__)
    p.add_argument("scene", type=int)
    p.add_argument("windows", nargs="+")
    args = p.parse_args()
    cfg = load_config(args.work)
    windows = [tuple(float(x) for x in w.split("-")) for w in args.windows]
    timeline = json.load(open(wpath(cfg, "timeline.json"), encoding="utf-8"))["scenes"]
    offset = float(timeline[args.scene - 1][0])
    mix, sr = sf.read(wpath(cfg, cfg.get("song_path") or "source/song.wav"), dtype="float32", always_2d=True)
    lead, lsr = sf.read(wpath(cfg, cfg.get("lead_stem_path") or "source/stems/0 Lead Vocals.wav"), dtype="float32",
                        always_2d=True)
    if lsr != sr:
        sys.exit(f"sample rates differ: song {sr}, lead stem {lsr}")
    gate = np.zeros(len(mix), dtype=np.float32)
    k = int(FADE * sr)
    for a, b in windows:
        s, e = int((offset + a) * sr), int((offset + b) * sr)
        gate[s:e] = 1.0
        gate[max(0, s - k):s] = np.linspace(0, 1, s - max(0, s - k))
        gate[e:e + k] = np.linspace(1, 0, len(gate[e:e + k]))
    n = min(len(mix), len(lead))
    out = mix.copy()
    out[:n] -= GAIN * lead[:n, :mix.shape[1]] * gate[:n, None]
    folder = wpath(cfg, os.path.join("audio", "gated"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"scene_{args.scene:02d}.wav")
    sf.write(path, np.clip(out, -1, 1), sr, subtype="PCM_16")
    print(f"scene {args.scene}: lead vocal removed in {windows} (scene-relative) -> {path}")


if __name__ == "__main__":
    main()
