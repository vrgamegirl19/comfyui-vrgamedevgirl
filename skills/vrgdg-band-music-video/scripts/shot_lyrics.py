"""Show, for every scene in the spec, which lyric words fall in each shot and when (scene-relative seconds).

usage: shot_lyrics.py --work W [--spec spec.py] [--scenes 3 5 8] [--draft]

Use it while writing shots: each shot's <d> cue holds exactly the words printed for that shot, with the window
"from <first word start> s to <last word end> s". A word cut by a shot change is flagged with the nearest gap
where the cut could move instead. --draft prints a ready-to-edit cue line per shot.
Times come from lyrics_timed.json (align_lyrics.py) and scene windows from timeline.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_spec, parser, read_json, wpath  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--spec", default="spec.py")
    p.add_argument("--scenes", nargs="*", type=int)
    p.add_argument("--draft", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.work)
    spec = load_spec(cfg, args.spec)
    timeline = (read_json(wpath(cfg, "timeline.json"), {}) or {}).get("scenes") or []
    words = (read_json(wpath(cfg, "lyrics_timed.json"), {}) or {}).get("words") or []
    for scene in spec.SCENES:
        n = scene["n"]
        if args.scenes and n not in args.scenes:
            continue
        if len(timeline) < n:
            print(f"scene {n}: not in timeline.json")
            continue
        start, end = timeline[n - 1][:2]
        cuts = [t for t, _ in scene["shots"]] + [end - start]
        print(f"scene {n} ({end - start:.2f} s, audio {scene['audio']})")
        for k in range(len(scene["shots"])):
            s0, s1 = start + cuts[k], start + cuts[k + 1]
            inside = [w for w in words if s0 - 0.05 <= w["start"] < s1]
            if inside:
                a, b = max(0.0, inside[0]["start"] - start), inside[-1]["end"] - start
                text = " ".join(w["word"] for w in inside)
                print(f"  shot {k + 1} [{cuts[k]:.2f}-{cuts[k + 1]:.2f}] sung {a:.1f}-{b:.1f} s: {text}")
                if args.draft:
                    print(f"      {{sung}} from {a:.1f} s to {b:.1f} s, <d>[English, EMOTION, EMOTION, singing] {text}</d>, FACIAL ACTING")
            else:
                print(f"  shot {k + 1} [{cuts[k]:.2f}-{cuts[k + 1]:.2f}] no vocals")
            if k + 1 < len(scene["shots"]):
                cut = start + cuts[k + 1]
                hit = [w for w in words if w["start"] < cut < w["end"] - 0.05]
                if hit:
                    gaps = [(words[i]["end"], words[i + 1]["start"]) for i in range(len(words) - 1)
                            if words[i + 1]["start"] - words[i]["end"] > 0.15 and abs(words[i]["end"] - cut) < 2.0]
                    best = min(gaps, key=lambda g: abs((g[0] + g[1]) / 2 - cut)) if gaps else None
                    tip = f"; nearest gap {best[0] - start:.2f}-{best[1] - start:.2f} s" if best else ""
                    print(f"      ! cut at {cuts[k + 1]:.2f} s splits '{hit[0]['word']}'{tip}")


if __name__ == "__main__":
    main()
