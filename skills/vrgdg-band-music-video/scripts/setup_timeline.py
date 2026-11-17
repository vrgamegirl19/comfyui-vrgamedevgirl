"""Lay out the Builder timeline from timeline.json and load the timed lyrics as an SRT.

usage: setup_timeline.py --work W [--timeline timeline.json] [--lyrics lyrics_timed.json] [--dry-run] [--force]

timeline.json (see assets/timeline_example.json): {"scenes": [[start, end, "label or words"], ...]} in full-track
seconds. Scenes must be contiguous from 0 to the end of audio/full_track.wav. The table it prints shows each scene's
length and the words sung inside it. Read it before planning shots: every scene boundary should sit in a vocal gap.

Replacing the timeline clears scene media, so a project that already has rendered videos is refused unless you
pass --force (ask the user first).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, ffprobe_duration, load_config, parser, read_json, wpath  # noqa: E402


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    p = parser(__doc__)
    p.add_argument("--timeline", default="timeline.json")
    p.add_argument("--lyrics", default="lyrics_timed.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    scenes = read_json(wpath(cfg, args.timeline), {}).get("scenes") or []
    timed = read_json(wpath(cfg, args.lyrics), {}) or {}
    words = timed.get("words") or []
    track = ffprobe_duration(wpath(cfg, "audio/full_track.wav"))

    problems = []
    if not scenes:
        sys.exit("timeline.json has no scenes")
    if abs(scenes[0][0]) > 1e-6:
        problems.append(f"first scene starts at {scenes[0][0]}, not 0")
    for (a1, b1, _), (a2, b2, _) in zip(scenes, scenes[1:]):
        if abs(b1 - a2) > 1e-6:
            problems.append(f"gap/overlap between {b1} and {a2}")
    if track and abs(scenes[-1][1] - track) > 0.05:
        problems.append(f"last scene ends at {scenes[-1][1]} but the track is {track:.2f} s")
    print(f" #   start     end    len  words sung inside the scene")
    for n, (a, b, label) in enumerate(scenes, 1):
        inside = [w["word"] for w in words if a <= w["start"] < b]
        straddle = [w["word"] for w in words if w["start"] < a < w["end"] - 0.05]
        flag = "" if 2.0 <= b - a <= 12.0 else "  <-- length"
        if straddle:
            flag += f"  <-- boundary cuts the word '{straddle[0]}'"
        print(f"{n:2d} {a:7.2f} {b:7.2f} {b - a:6.2f}  {' '.join(inside)[:90]}{flag}")
    for prob in problems:
        print("PROBLEM", prob)
    if problems:
        sys.exit(1)

    srt = "\n".join(f"{i}\n{srt_time(l['start'])} --> {srt_time(l['end'])}\n{l['text']}\n"
                    for i, l in enumerate(timed.get("lines") or [], 1))
    text = "\n".join(f"{a:.2f} --> {b:.2f} {label}" for a, b, label in scenes)
    if args.dry_run:
        print(f"\nwould PUT {len(timed.get('lines') or [])} SRT lines and replace the timeline with {len(scenes)} scenes")
        return
    existing = api.scenes()
    if any(s.get("rendered_video") for s in existing) and not args.force:
        sys.exit("this project already has rendered videos; replacing the timeline clears them. Use --force only with the user's OK.")
    if srt:
        status, env = api.mutate("PUT", "/projects/{pid}/lyrics", {"srt_text": srt})
        print("lyrics:", status, env.get("ok"), env.get("error") or "")
    status, env = api.call("POST", "/projects/{pid}/timeline/bulk",
                           {"mode": "ranges", "action": "replace", "text": text, "clear_media": True})
    print("timeline:", status, env.get("data") or env.get("error"))


if __name__ == "__main__":
    main()
