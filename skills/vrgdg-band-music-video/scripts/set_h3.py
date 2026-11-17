"""Set keys in one scene's own MiniMax H3 settings, keeping all the others (the API replaces per-scene settings whole).

usage: set_h3.py --work W [--project P] SCENE key=value [key=value ...]     (values are parsed as JSON when possible)
examples:
  set_h3.py --work W 2 continuity_mode=latent_continuation_masked latent_context_frames=39 continuity_prompt_from_last_frame=false
  set_h3.py --work W 19 ref_image_size=match                     (pass-2 out-of-memory fallback)
  set_h3.py --work W 7 seed=1001 two_pass_pass1_seed=1001 two_pass_pass2_seed=1001
Starts from the scene's saved settings, or the project settings when the scene has none.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, load_config, parser  # noqa: E402
from render_guarded import segment  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("scene", type=int)
    p.add_argument("pairs", nargs="+")
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    h3 = dict(segment(cfg, api, args.scene).get("minimax_h3_settings") or {})
    if not h3:
        _, env = api.call("GET", "/projects/{pid}/settings")
        h3 = dict(env["data"]["minimax_h3"])
    for pair in args.pairs:
        key, raw = pair.split("=", 1)
        try:
            h3[key] = json.loads(raw)
        except ValueError:
            h3[key] = raw
    st, env = api.mutate("PATCH", f"/projects/{{pid}}/scenes/{args.scene}",
                         {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": h3})
    if not env.get("ok"):
        sys.exit(f"patch failed: {st} {env}")
    print(f"scene {args.scene}: " + ", ".join(args.pairs))


if __name__ == "__main__":
    main()
