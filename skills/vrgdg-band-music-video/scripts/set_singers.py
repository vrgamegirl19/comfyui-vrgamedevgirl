"""Assign the singer on every scene that has sung words (the Builder's singer assignment, `lyric_singers`), so the
project opens in the Video Builder with the right performer per scene and the Builder's own multi-shot cue timing
("auto-time singer cues", Whisper on the scene audio) can re-time any scene from the UI.

usage: set_singers.py --work W [--singer singer] [--dry-run]

Who sings: a spec scene can carry singers=[card ids]; otherwise the lead singer card (--singer, default config
"singer_card" or "singer") on every scene where a word of lyrics_timed.json starts. Other scenes get an empty list.
Run after setup_timeline.py and build_prompts.py --apply (it reads the spec for per-scene overrides).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Api, load_config, load_spec, parser, read_json, wpath  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--singer", default=None)
    p.add_argument("--spec", default="spec.py")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.work)
    api = Api(cfg, a.project)
    lead = a.singer or cfg.get("singer_card") or "singer"
    _, env = api.call("GET", "/projects/{pid}/references")
    data = env.get("data") or {}
    cards = data.get("subjects") or (data.get("flux_reference_builder") or {}).get("subjects") or []
    names = {c.get("id"): c.get("name") or c.get("id") for c in cards}
    overrides = {}
    try:
        spec = load_spec(cfg, a.spec)
        for sc in getattr(spec, "SCENES", []):
            if isinstance(sc, dict) and sc.get("singers") is not None:
                overrides[int(sc["n"])] = list(sc["singers"])
    except SystemExit:
        pass
    words = (read_json(wpath(cfg, "lyrics_timed.json"), {}) or {}).get("words") or []
    for i, scene in enumerate(api.scenes(), start=1):
        lyric = str(scene.get("lyric_text") or scene.get("lyrics") or "").strip()
        start, end = float(scene.get("start") or 0), float(scene.get("end") or 0)
        if words:  # sung = a timed word starts inside the scene (scene labels like "intro: bass" are not lyrics)
            sung = any(start - 0.05 <= float(w["start"]) < end for w in words)
        else:
            sung = bool(lyric) and not lyric.startswith("[") and ":" not in lyric.split(" ")[0]
        singers = overrides.get(i, [lead] if sung else [])
        singers = [names.get(x, x) for x in singers]  # the Builder stores the card name, e.g. "Vex (lead singer)"
        print(f"scene {i:2d}: {singers or '-'}  {lyric[:60]}")
        if not a.dry_run:
            status, env = api.mutate("PATCH", f"/projects/{{pid}}/scenes/{i}", {"lyric_singers": singers})
            if not env.get("ok"):
                print(f"   PATCH failed: {status} {str(env)[:300]}")


if __name__ == "__main__":
    main()
