"""Social media kit: SocialMediaSharing/ with the poster reference images and the film's facts (run after assemble.py).

  python social_kit.py <project>
  -> SocialMediaSharing/poster_refs/<n>_<name>.png   cast refs in screenplay order (attach to the image model in this order)
     SocialMediaSharing/facts.json                    scenes, runtime, render hours, re-shoots: the numbers the posts quote
Claude then writes the text files next to them (see SKILL.md step 11).
"""
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ffprobe_duration, load_settings, project_dir, save_json  # noqa: E402


def main():
    project, _ = project_dir()
    settings = load_settings(project)
    sp = json.load(open(os.path.join(project, "screenplay.json"), encoding="utf-8"))
    out = os.path.join(project, "SocialMediaSharing")
    refs = os.path.join(out, "poster_refs")
    os.makedirs(refs, exist_ok=True)
    order = []
    for i, (key, c) in enumerate(sp["cast"].items(), start=1):
        src = os.path.join(project, c["ref"])
        if os.path.isfile(src):
            dst = f"{i}_{key}{os.path.splitext(src)[1]}"
            shutil.copy2(src, os.path.join(refs, dst))
            order.append({"image": dst, "name": c["name"], "desc": c["desc"]})
    seconds = 0.0
    log = os.path.join(project, "logs", "render.log")
    if os.path.isfile(log):
        seconds = sum(int(s) for s in re.findall(r"done scene \d+ H3 2-pass in (\d+)s", open(log, encoding="utf-8").read()))
    rerolls = os.path.join(project, "logs", "reroll_list.txt")
    final = os.path.join(project, "final", f"{sp['title']}.mp4")
    facts = {
        "title": sp["title"], "genre": sp.get("genre", ""), "scenes": len(sp["scenes"]),
        "runtime_seconds": round(ffprobe_duration(settings["ffmpeg"], final), 1) if os.path.isfile(final) else None,
        "render_hours": round(seconds / 3600, 1),
        "reshoots": open(rerolls, encoding="utf-8").read().splitlines() if os.path.isfile(rerolls) else [],
        "poster_refs_in_order": order,
        "locations": [loc["name"] for loc in sp["locations"].values()],
        "voices": sp["voices"],
    }
    save_json(os.path.join(out, "facts.json"), facts)
    print(f"{out}\n  {len(order)} poster refs, {facts['scenes']} scenes, {facts['runtime_seconds']} s, "
          f"{facts['render_hours']} h rendering, {len(facts['reshoots'])} re-shoot notes")


if __name__ == "__main__":
    main()
