"""Create Reference Builder cards and generate their pictures with the Builder's Krea 2 two-pass route
(Ref Builder > Create Character), seed by seed, so the best candidate can be picked.

usage: make_refs.py --work W cards                     create/update every card (name + description; keeps its picture)
       make_refs.py --work W gen ID SEED [SEED ...]    generate ID's picture once per seed -> refs/ID_seedN.png
       make_refs.py --work W pick ID SEED              put refs/ID_seedN.png on the card
       make_refs.py --work W show ID                   print the full Krea prompt for ID

The characters and locations live in <work>/refs_spec.py (copy assets/refs_spec_example.py):
  SUBJECTS  = {id: (card name, look)}   the look is dropped into SHEET
  LOCATIONS = {id: (card name, place)}  the place is dropped into LOC
  SHEET / LOC (optional) override the default wording below.
Every generate is one GPU job; the script waits for each. Run only when ComfyUI is idle.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Api, load_config, load_spec, parser, wpath  # noqa: E402

# Borderless three-panel sheet (face close-up | upper body | full body) on solid grey. This exact wording gave clean
# sheets with no divider lines in NO PERMISSION TO EXIST; "LEFT PHOTO / CENTER PHOTO / RIGHT PHOTO" wording gave
# thin divider lines between the panels.
SHEET = ("A character reference sheet, 16:9: one single continuous image on a plain seamless solid mid-grey studio "
         "backdrop that shows the same person three times, side by side, at three very different zoom levels. On the "
         "left third: an extreme close-up of the face only, so close that the face fills that third from the forehead at "
         "the top to the chin at the bottom. In the middle third: an upper-body view from the head to the waist. On the "
         "right third: a full-body view from head to shoes, standing. The person is {look}. Photorealistic studio "
         "photograph, clear even lighting, facing the camera, a cool confident expression with closed lips, arms relaxed "
         "at the sides with empty hands, the same face, hair and outfit in all three views. The solid mid-grey backdrop "
         "is one unbroken flat colour across the whole image, the three views blending into it with no borders, no "
         "frames, no dividing lines and no text.")

LOC = ("A clear cinematic environment reference photograph of {place}. Wide eye-level framing that shows the layout of "
       "the whole space, photorealistic, empty and still, a place with nobody in it.")


def load_refs(cfg):
    mod = load_spec(cfg, "refs_spec.py")
    return (getattr(mod, "SUBJECTS", {}), getattr(mod, "LOCATIONS", {}),
            getattr(mod, "SHEET", SHEET), getattr(mod, "LOC", LOC))


def main():
    p = parser(__doc__)
    p.add_argument("command", choices=["cards", "gen", "pick", "show"])
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    cfg = load_config(a.work)
    api = Api(cfg, a.project)
    subjects, locations, sheet, loc = load_refs(cfg)

    def kind_of(rid):
        if rid in subjects:
            return "subjects"
        if rid in locations:
            return "locations"
        sys.exit(f"[VRGDG MV] {rid} is not in refs_spec.py")

    def prompt_for(rid):
        return sheet.format(look=subjects[rid][1]) if kind_of(rid) == "subjects" else loc.format(place=locations[rid][1])

    if a.command == "cards":
        for rid, (name, look) in subjects.items():
            body = {"name": name, "description": look[0].upper() + look[1:] + ".", "reference_type": "character"}
            status, env = api.mutate("PUT", f"/projects/{{pid}}/references/subjects/{rid}", body)
            print("subject", rid, status, env.get("ok"), (env.get("error") or {}).get("message", ""))
        for rid, (name, place) in locations.items():
            body = {"name": name, "description": place[0].upper() + place[1:] + "."}
            status, env = api.mutate("PUT", f"/projects/{{pid}}/references/locations/{rid}", body)
            print("location", rid, status, env.get("ok"), (env.get("error") or {}).get("message", ""))
    elif a.command == "show":
        print(prompt_for(a.args[0]))
    elif a.command == "gen":
        rid = a.args[0]
        os.makedirs(wpath(cfg, "refs"), exist_ok=True)
        for seed in [int(s) for s in a.args[1:]]:
            api.wait_idle()
            body = {"prompt": prompt_for(rid), "settings": {"seed": seed, "seed_mode": "fixed"}}
            status, env = api.call("POST", f"/projects/{{pid}}/references/{kind_of(rid)}/{rid}/image/generate", body)
            job = (env.get("data") or {}).get("job_id")
            if not job:
                print(rid, seed, "NOT QUEUED", status, str(env)[:300])
                continue
            res = api.wait(job, poll=5.0)
            data = (res.get("result") or res.get("data") or res) if isinstance(res, dict) else {}
            saved = data.get("saved_path") or ((data.get("result") or {}).get("saved_path"))
            if saved and os.path.isfile(saved):
                dst = wpath(cfg, f"refs/{rid}_seed{seed}.png")
                shutil.copy(saved, dst)
                print(rid, seed, "->", dst)
            else:
                print(rid, seed, "no saved_path:", str(res)[:400])
    elif a.command == "pick":
        rid, seed = a.args[0], a.args[1]
        path = os.path.abspath(wpath(cfg, f"refs/{rid}_seed{seed}.png"))
        _, env = api.call("GET", "/projects/{pid}/references")
        data = env.get("data") or {}
        cards = data.get(kind_of(rid)) or (data.get("flux_reference_builder") or {}).get(kind_of(rid)) or []
        card = next(c for c in cards if c.get("id") == rid)
        body = {"name": card.get("name"), "description": card.get("description", ""),
                "image": {"path": path, "name": os.path.basename(path)}}
        if kind_of(rid) == "subjects":
            body["reference_type"] = card.get("reference_type", "character")
        status, env = api.mutate("PUT", f"/projects/{{pid}}/references/{kind_of(rid)}/{rid}", body)
        print("pick", rid, seed, status, env.get("ok"))


if __name__ == "__main__":
    main()
