"""Create or update the Reference Builder cards (characters, outfit/prop cards and locations) from refs.json.

usage: setup_refs.py --work W [--refs refs.json] [--dry-run]

refs.json (see assets/refs_example.json):
  {"subjects": [{"id": "singer", "name": "Singer", "description": "...", "reference_type": "character",
                 "image": "source/characters/singer.jpg"}, ...],
   "locations": [{"id": "kitchen", "name": "Kitchen", "description": "...", "image": "source/scenes/kitchen.png"}]}
reference_type is "character" for people and "outfit" for a garment or prop card (for example a jacket held in a scene).
A PUT resets every field it leaves out, so this always sends the whole card. Image paths must exist; copy the
user's files into the work folder first so the project keeps working if their originals move.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, load_config, parser, read_json, wpath  # noqa: E402


def card(cfg, item):
    path = os.path.abspath(wpath(cfg, item["image"]))
    if not os.path.isfile(path):
        sys.exit(f"image not found for {item['id']}: {path}")
    body = {"name": item.get("name") or item["id"], "description": item.get("description", ""),
            "image": {"path": path, "name": os.path.basename(path)}}
    if "reference_type" in item or "trigger_phrase" in item:
        body["reference_type"] = item.get("reference_type", "character")
    return body


def main():
    p = parser(__doc__)
    p.add_argument("--refs", default="refs.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    refs = read_json(wpath(cfg, args.refs))
    if not refs:
        sys.exit(f"no {args.refs} in the work folder")
    for kind in ("subjects", "locations"):
        for item in refs.get(kind, []):
            body = card(cfg, item)
            if kind == "subjects":
                body.setdefault("reference_type", "character")
            if args.dry_run:
                print(kind, item["id"], json.dumps(body)[:200])
                continue
            status, env = api.mutate("PUT", f"/projects/{{pid}}/references/{kind}/{item['id']}", body)
            print(kind, item["id"], status, env.get("ok"), (env.get("error") or {}).get("message", ""))
    if not args.dry_run:
        _, env = api.call("GET", "/projects/{pid}/references")
        data = env.get("data") or {}
        subs = (data.get("subjects") or (data.get("flux_reference_builder") or {}).get("subjects") or [])
        print(f"\nproject now has {len(subs)} subject cards")


if __name__ == "__main__":
    main()
