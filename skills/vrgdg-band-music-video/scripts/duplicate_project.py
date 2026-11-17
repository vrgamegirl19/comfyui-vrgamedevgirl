"""Copy the Builder project for experiments (continuation alternates) without touching the original.

usage: duplicate_project.py --work W --name "My Song - continuation alternates" [--dry-run]

Works around two quirks of POST /projects/{pid}/duplicate (as of 2026-10):
  1. the route builds the copy's session from a "session" field and writes an EMPTY project without it, so this
     script sends the source project's saved session (vrgdg_builder_session.json, read-only) in options.session;
  2. the copy has no audio attached, so this script attaches the same full track.
The whole folder is copied (videos, latents, and the _claude_work folder too), so the copy can render
continuations from the original takes. Prints the copy's id; pass it as --project to the other scripts.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, load_config, parser, video_path, wpath  # noqa: E402


def source_folder(cfg, api):
    if cfg.get("project_folder") and os.path.isdir(cfg["project_folder"]):
        return cfg["project_folder"]
    for scene in api.scenes():
        path = video_path(scene)
        if path:
            return os.path.dirname(os.path.dirname(path))
    sys.exit("cannot find the source project folder; set project_folder in config.json")


def main():
    p = parser(__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    folder = source_folder(cfg, api)
    session_path = os.path.join(folder, "vrgdg_builder_session.json")
    session = json.load(open(session_path, encoding="utf-8"))
    print(f"source: {api.pid} ({len(session.get('segments') or [])} scenes) in {folder}")
    if args.dry_run:
        print(f"would duplicate to '{args.name}' and attach {wpath(cfg, 'audio/full_track.wav')}")
        return
    status, env = api.call("POST", "/projects/{pid}/duplicate", {"new_name": args.name, "options": {"session": session}}, timeout=1800)
    if not env.get("ok"):
        sys.exit(f"duplicate failed ({status}): {env.get('error')}")
    copy = Api(cfg, env["data"]["project_id"])
    print("copy:", copy.pid, "->", env["data"].get("project_folder"))
    status, env = copy.mutate("PUT", "/projects/{pid}/audio", {"audio_path": os.path.abspath(wpath(cfg, "audio/full_track.wav"))})
    print("audio attached:", status, env.get("ok"))
    _, env = copy.call("GET", "/projects/{pid}/summary")
    print("summary:", json.dumps(env.get("data"))[:400])
    _, env = copy.call("GET", "/projects/{pid}/latents")
    items = (env.get("data") or {}).get("items") or []
    print(f"latents: {sum(1 for i in items if i.get('exists'))} present, dirty: {[i['scene_number'] for i in items if i.get('dirty')]}")
    print(f"\nUse --project \"{copy.pid}\" with build_prompts.py / render_queue.py / stitch.py for the copy.")


if __name__ == "__main__":
    main()
