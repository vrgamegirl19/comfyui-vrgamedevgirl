"""Score cues from <project>/score.json, with whichever music engine check_env.py found:

  music3  local MiniMax Music 3 (core ComfyUI nodes; 36 steps, cfg 1.8, euler/simple)
  yue2    VRGDG YuE2 (isolated YuE2 install made by the VRGDG YuE2 Installer node; pass its folder to check_env --yue2-root)

  python gen_music.py <project> [cue_name ...] [--engine music3|yue2]   -> <project>/audio/score/<cue>_<seed>.mp3
YuE2 ignores a cue's "seconds": the song's length follows its lyrics (about 10-15 s per sung line plus intro/outro).

YuE2 is a song model: instrumental cues can still come back with vocals, so check every cue (qa.py --music) and prefer
YuE2 for songs (an end-credits song with lyrics). MiniMax Music 3 handles instrumentals well.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Comfy, load_settings, output_files, project_dir, save_json  # noqa: E402


def node_defaults(comfy, cls, **overrides):
    """An API node with every widget at its default (combos: first option), then the overrides."""
    spec = list(comfy.get(f"/object_info/{cls}").values())[0]["input"]
    inputs = {}
    for group in ("required", "optional"):
        for name, val in spec.get(group, {}).items():
            kind, opts = val[0], (val[1] if len(val) > 1 else {})
            if isinstance(kind, list):
                inputs[name] = opts.get("default", kind[0] if kind else "")
            elif kind == "COMBO":
                inputs[name] = opts.get("default", (opts.get("options") or [""])[0])
            elif "default" in opts:
                inputs[name] = opts["default"]
    inputs.update(overrides)
    return {"class_type": cls, "inputs": inputs}


def music3_graph(m, cue, seed, prefix):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["music3_dit"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["music3_clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["music3_vae"]}},
        "7": {"class_type": "MiniMaxMusic3TextEncode", "inputs": {"clip": ["2", 0], "caption": cue["caption"], "lyrics": cue["lyrics"],
                                                                  "seed": seed, "max_duration": cue["seconds"], "cfg_scale": 1.5, "top_k": 50}},
        "9": {"class_type": "EmptyMiniMaxMusic3LatentAudio", "inputs": {"seconds": ["7", 1], "batch_size": 1}},
        "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "11": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["10", 0], "latent_image": ["9", 0],
                                                    "seed": seed, "steps": 36, "cfg": 1.8, "sampler_name": "euler", "scheduler": "simple",
                                                    "denoise": 1.0}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveAudioMP3", "inputs": {"audio": ["12", 0], "filename_prefix": prefix, "quality": "V0"}},
    }


def yue2_graph(comfy, root, cue, seed, prefix):
    """Installer node used as settings only (it installs nothing when queued), then Generate Song -> MP3."""
    return {
        "1": node_defaults(comfy, "VRGDG_YuE2Installer", target_root=root, local_files_only=True,
                           memory_budget_gib=float(cue.get("yue2_memory_gib", 24.0))),
        "2": node_defaults(comfy, "VRGDG_YuE2Generate", yue2_config=["1", 0], style=cue["caption"], lyrics=cue["lyrics"],
                           seed=seed, filename_prefix=prefix.replace("/", "_")),
        "3": {"class_type": "SaveAudioMP3", "inputs": {"audio": ["2", 0], "filename_prefix": prefix, "quality": "V0"}},
    }


def main():
    project, args = project_dir()
    settings = load_settings(project)
    engine = args[args.index("--engine") + 1] if "--engine" in args else settings["music_engine"]
    names = [a for i, a in enumerate(args) if a != "--engine" and (i == 0 or args[i - 1] != "--engine")]
    if engine not in (settings.get("music_engines") or [settings["music_engine"]]):
        raise SystemExit(f"music engine '{engine}' is not available; check_env.py found {settings.get('music_engines')}")
    if engine == "none":
        raise SystemExit("No music engine was found by check_env.py. Put your own audio files in audio/score/ instead.")
    comfy = Comfy(settings["comfy_url"], os.path.join(project, "logs", "run.log"))
    cues = json.load(open(os.path.join(project, "score.json"), encoding="utf-8"))["cues"]
    out_dir = os.path.join(project, "audio", "score")
    for name, cue in cues.items():
        if names and name not in names:
            continue
        for seed in cue["seeds"]:
            prefix = f"vrgdg_film/{os.path.basename(project)}_{name}"
            graph = music3_graph(settings["models"], cue, seed, prefix) if engine == "music3" else \
                yue2_graph(comfy, settings["yue2_root"], cue, seed, prefix)
            save_json(os.path.join(project, "workflows", f"music_{name}_{seed}_api.json"), graph)
            entry = comfy.run(graph, f"{engine} {name} seed {seed}", timeout=3 * 3600)
            mp3 = [i for i in output_files(entry) if i["filename"].lower().endswith(".mp3")]
            if not mp3:
                raise SystemExit(f"{name} seed {seed}: no mp3 output")
            comfy.log(f"saved {comfy.fetch_output(mp3[-1], os.path.join(out_dir, f'{name}_{seed}.mp3'))}")
    comfy.free(unload_models=True)   # music models are big; give the VRAM back before video renders


if __name__ == "__main__":
    main()
