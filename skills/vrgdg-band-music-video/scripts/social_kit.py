"""Social media kit: SocialMediaSharing/ with the poster reference images and the facts the posts quote.

usage: social_kit.py --work W [--project P] --final FINAL.mp4 [--out FOLDER]
       -> FOLDER/poster_refs/<n>_<card>.<ext>  (attach to the image model in this order)
          FOLDER/facts.json                      (scenes, runtime, renders, GPU hours, re-shoots, locations, voices)

FOLDER defaults to SocialMediaSharing/ inside the project folder. Poster cards come from spec.POSTER_CARDS (default:
each character's stage card or main card). Render count and GPU hours come from logs/render.log (started/saved pairs,
any prefix, across midnight). Claude then writes poster_prompt.txt, youtube_description.txt, reddit_post.txt and
short_post.txt from these facts (references/social-kit.md); never invent numbers.
"""
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, ffprobe_duration, load_config, load_spec, parser, read_json, video_path, wpath, write_json  # noqa: E402

LINE = re.compile(r"(\d\d:\d\d:\d\d) (?:\[(.+?)\] |(ALT|JACKET|FIX) )?scene (\d+): (started|saved)")


def render_stats(log_path):
    if not os.path.isfile(log_path):
        return 0, 0.0
    starts, total, count, prev, day = {}, 0.0, 0, None, 0
    for line in open(log_path, encoding="utf-8"):
        m = LINE.match(line)
        if not m:
            continue
        h, mi, s = map(int, m.group(1).split(":"))
        t = h * 3600 + mi * 60 + s
        if prev is not None and t < prev - 3600:
            day += 86400
        prev = t
        t += day
        key = (m.group(2) or m.group(3) or "", m.group(4))
        if m.group(5) == "started":
            starts[key] = t
        elif key in starts:
            total += t - starts.pop(key)
            count += 1
    return count, total / 3600


def main():
    p = parser(__doc__)
    p.add_argument("--final", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--spec", default="spec.py")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    spec = load_spec(cfg, args.spec)
    scenes = api.scenes()
    folder = args.out
    if not folder:
        first = next((video_path(s) for s in scenes if video_path(s)), "")
        folder = os.path.join(os.path.dirname(os.path.dirname(first)) if first else cfg["_work"], "SocialMediaSharing")
    refs_dir = os.path.join(folder, "poster_refs")
    os.makedirs(refs_dir, exist_ok=True)

    _, env = api.call("GET", "/projects/{pid}/references")
    data = env.get("data") or {}
    subjects = data.get("subjects") or (data.get("flux_reference_builder") or {}).get("subjects") or []
    images = {s.get("id"): (s.get("image") or {}).get("path") for s in subjects}
    cards = getattr(spec, "POSTER_CARDS", None) or [
        ((c.get("variants") or {}).get("stage") or {}).get("card") or c["card"] for c in spec.CHARACTERS.values()]
    order = []
    for i, card in enumerate(cards, 1):
        src = images.get(card)
        if src and os.path.isfile(src):
            name = f"{i}_{card}{os.path.splitext(src)[1]}"
            shutil.copy2(src, os.path.join(refs_dir, name))
            order.append(name)
        else:
            print(f"WARN no image for card {card}")

    renders, hours = render_stats(wpath(cfg, "logs/render.log"))
    reshoots = [l.strip() for l in open(wpath(cfg, "logs/reroll_list.txt"), encoding="utf-8")] if os.path.isfile(wpath(cfg, "logs/reroll_list.txt")) else []
    lines = read_json(wpath(cfg, "audio/voice/lines.json"), {}) or {}
    runtime = ffprobe_duration(args.final) if os.path.isfile(args.final) else None
    facts = {
        "title": cfg.get("title") or api.pid,
        "scenes": len(scenes),
        "runtime_seconds": round(runtime, 2) if runtime else None,
        "runtime": f"{int(runtime // 60)}:{int(round(runtime % 60)):02d}" if runtime else None,
        "renders": renders,
        "render_hours_gpu": round(hours, 1),
        "model": (cfg.get("minimax_h3") or {}).get("diffusion_model_name"),
        "reshoot_notes": [l for l in reshoots if l],
        "poster_refs_in_order": order,
        "locations": [loc.get("name", key) for key, loc in spec.LOCATIONS.items()],
        "voices": {k: v.get("name") for k, v in (lines.get("voices") or {}).items()},
    }
    write_json(os.path.join(folder, "facts.json"), facts)
    print(f"{folder}\n  {len(order)} poster refs, {facts['scenes']} scenes, {facts['runtime']} runtime, "
          f"{renders} renders, {facts['render_hours_gpu']} GPU hours, {len(facts['reshoot_notes'])} re-shoot notes")


if __name__ == "__main__":
    main()
