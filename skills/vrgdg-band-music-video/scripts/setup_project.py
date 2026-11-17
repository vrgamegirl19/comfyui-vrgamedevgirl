"""Create the Builder project (if needed), apply the render settings from config.json and attach the full track.

usage: setup_project.py --work W [--audio audio/full_track.wav] [--dry-run]

Writes the real project id and project_folder back into config.json. The settings go in exactly as given in
config.json "minimax_h3" (key names: GET /settings/minimax-h3/schema) and "project_settings". Afterwards it reads the
effective settings back and prints the ones that matter, so a wrong model or LoRA name shows up before rendering.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, load_config, parser, save_config, wpath  # noqa: E402

SHOW = ["video_mode", "render_pass", "resolution_preset", "audio_mode", "continuity_mode", "diffusion_model_name",
        "two_pass_pass1_steps", "two_pass_lora_name", "two_pass_lora_preset", "two_pass_pass2_steps",
        "two_pass_pass2_denoise", "warmup_frames", "cooldown_frames", "pass1_use_feedforward", "pass2_use_feedforward",
        "two_pass_use_te_speed", "two_pass_use_block_sparse_attention", "two_pass_use_fast_vae_decode"]


def main():
    p = parser(__doc__)
    p.add_argument("--audio", default="audio/full_track.wav")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    settings = {"project": cfg.get("project_settings") or {"video_engine": "minimax_h3"},
                "minimax_h3": cfg.get("minimax_h3") or {}}
    audio = wpath(cfg, args.audio)
    if args.dry_run:
        print("would create/reuse project", api.pid)
        print("PATCH settings", json.dumps(settings, indent=1))
        print("PUT audio", audio, "exists:", os.path.isfile(audio))
        return

    status, env = api.call("GET", "/projects/{pid}/summary")
    if status != 200:
        status, env = api.call("POST", "/projects", {"name": api.pid})
        if not env.get("ok"):
            sys.exit(f"create failed: {env.get('error')}")
        cfg["project_id"] = env["data"]["project_id"]
        cfg["project_folder"] = env["data"]["project_folder"]
        api.pid = cfg["project_id"]
        save_config(cfg)
        print("created project", cfg["project_id"], "->", cfg["project_folder"])
    else:
        print("project exists:", api.pid)

    status, env = api.mutate("PATCH", "/projects/{pid}/settings", settings)
    print("settings:", status, env.get("ok"), env.get("error") or "")
    if os.path.isfile(audio):
        status, env = api.mutate("PUT", "/projects/{pid}/audio", {"audio_path": audio})
        data = env.get("data") or {}
        print("audio:", status, env.get("ok"), data.get("duration"), "s, tempo", data.get("tempo_bpm"), env.get("error") or "")
    else:
        print("WARN audio not attached; build it first:", audio)

    _, env = api.call("GET", "/projects/{pid}/settings")
    h3 = (env.get("data") or {}).get("minimax_h3") or {}
    print("\neffective MiniMax H3 settings:")
    for key in SHOW:
        print(f"  {key}: {h3.get(key)}")


if __name__ == "__main__":
    main()
