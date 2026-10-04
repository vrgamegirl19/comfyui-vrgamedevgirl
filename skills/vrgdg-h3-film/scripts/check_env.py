"""Inspect a ComfyUI install for everything the vrgdg-h3-film pipeline needs and pick models automatically.

  python check_env.py [--url http://127.0.0.1:8188] [--comfy-root <ComfyUI folder>] [--yue2-root <YuE2 install folder>]
  -> prints a report and writes env.json next to this skill (new_project.py copies it into each project)

Nothing is installed, downloaded or changed: it only reads /object_info, /models, probes the Builder routes with an
empty request, and tries imports in the chosen Python.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SKILL_DIR, Comfy, find_ffmpeg, save_json  # noqa: E402

ROUTES = {
    "zimage": "/vrgdg/workflow_runner/build_zimage_prompt",
    "h3_2pass": "/vrgdg/workflow_runner/build_minimax_h3_2pass_prompt",
    "trim": "/vrgdg/workflow_runner/trim_scene_video",
    "stitch": "/vrgdg/workflow_runner/stitch_scene_videos",
}
NODES = {
    "h3": ["MiniMaxH3ReferenceToVideo", "VRGDG_MiniMaxH3AudioDrive", "LTXVAudioVAEDecode", "VHS_VideoCombine",
           "VRGDG_MiniMaxH3LearnedLatentUpscale", "VRGDG_MiniMaxH3ReferenceMediaFromPaths"],
    "music3": ["MiniMaxMusic3TextEncode", "EmptyMiniMaxMusic3LatentAudio", "VAEDecodeAudio", "SaveAudioMP3"],
    "yue2": ["VRGDG_YuE2Installer", "VRGDG_YuE2Generate", "SaveAudioMP3"],
}
# (folder, required keywords, preferred keywords) -> best match by score
MODEL_RULES = {
    "h3_diffusion": ("diffusion_models", ["minimax", "h3", "ref2va"], ["pruned", "int8"]),
    "h3_clip": ("text_encoders", ["qwen3vl", "minimax"], []),
    "h3_video_vae": ("vae", ["minimax_h3_video_vae"], ["fp16"]),
    "h3_audio_vae": ("vae", ["minimax_h3_audio_vae"], []),
    "h3_latent_upscaler": ("latent_upscale_models", ["minimax_h3"], ["latent_upscaler"]),
    "h3_turbo_lora": ("loras", ["minimax_h3", "turbo"], ["768p", "4step", "fl2v", "v1.1"]),
    "zimage_unet": ("diffusion_models", ["z_image"], ["turbo"]),
    "zimage_clip": ("text_encoders", ["qwen_3_4b"], []),
    "zimage_vae": ("vae", ["ae."], ["safetensors"]),
    "music3_dit": ("diffusion_models", ["minimax_music3"], ["dit"]),
    "music3_clip": ("text_encoders", ["minimax_music3"], ["text_encoder"]),
    "music3_vae": ("vae", ["minimax_music3"], []),
}
QA_IMPORTS = ["PIL", "numpy", "librosa", "whisper", "torch", "transformers"]


def pick(files, required, preferred):
    hits = [f for f in files if all(k in f.lower() for k in required)]
    if not hits:
        return None
    return sorted(hits, key=lambda f: (-sum(k in f.lower() for k in preferred), len(f)))[0]


def probe_python(python):
    code = ("import importlib,json;r={}\n"
            f"for m in {QA_IMPORTS!r}:\n"
            "  try: importlib.import_module(m); r[m]=True\n"
            "  except Exception: r[m]=False\n"
            "print(json.dumps(r))")
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=180)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"error": str(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-root", default="", help="ComfyUI install folder (finds its Python and ffmpeg)")
    ap.add_argument("--yue2-root", default="", help="Folder YuE2 was installed into by the VRGDG YuE2 Installer node")
    ap.add_argument("--qa-python", default="", help="Python with whisper/librosa/PIL (default: ComfyUI's Python)")
    a = ap.parse_args()
    comfy = Comfy(a.url)
    env = {"comfy_url": a.url, "problems": [], "warnings": []}
    try:
        stats = comfy.get("/system_stats")["system"]
    except Exception as exc:
        raise SystemExit(f"Cannot reach ComfyUI at {a.url}: {exc}")
    env["comfyui_version"] = stats.get("comfyui_version")

    # one class at a time: the full /object_info of a big install can be too large to fetch in one response
    wanted = {n for names in NODES.values() for n in names} | {"VRGDG_MusicVideoBuilderUI"}
    info = {n for n in wanted if comfy.get(f"/object_info/{n}", timeout=60)}
    env["nodes"] = {group: [n for n in names if n not in info] for group, names in NODES.items()}
    if env["nodes"]["h3"]:
        env["problems"].append(f"missing H3/VRGDG nodes: {env['nodes']['h3']} (update the VRGDG nodes / ComfyUI)")
    if "VRGDG_MusicVideoBuilderUI" not in info:
        env["problems"].append("the VRGDG Video Builder node is not loaded")

    env["routes"] = {}
    for key, route in ROUTES.items():
        res = comfy.post(route, {}, timeout=60)
        present = isinstance(res, dict) and "ok" in res
        env["routes"][key] = present
        if not present:
            env["problems"].append(f"Builder route missing: {route} (update the VRGDG nodes)")

    # The Builder registers its custom model root (Builder settings) lazily; asking for its LoRA list does that, so
    # models kept in a custom root show up in /models below exactly as they do in the Builder.
    try:
        comfy.get("/vrgdg/workflow_runner/lora_list", timeout=60)
    except Exception as exc:
        env["warnings"].append(f"could not register the Builder's custom model root: {exc}")
    folders = {rule[0] for rule in MODEL_RULES.values()}
    listing = {}
    for folder in folders:
        res = comfy.get(f"/models/{folder}")
        listing[folder] = res if isinstance(res, list) else []
    env["models"] = {key: pick(listing[f], req, pref) for key, (f, req, pref) in MODEL_RULES.items()}
    for key in ("h3_diffusion", "h3_clip", "h3_video_vae", "h3_audio_vae", "h3_latent_upscaler", "h3_turbo_lora"):
        if not env["models"][key]:
            env["problems"].append(f"no model found for {key} in models/{MODEL_RULES[key][0]}")
    for key in ("zimage_unet", "zimage_clip", "zimage_vae"):
        if not env["models"][key]:
            env["warnings"].append(f"no {key} model: reference images must be supplied by hand")
    lora = env["models"]["h3_turbo_lora"] or ""
    if lora and "768p" not in lora:
        env["warnings"].append(f"turbo LoRA {lora} is not the 768p fl2v v1.1 one; other turbo LoRAs leaked the reference "
                               "sheet into 2-pass renders in testing")

    music3 = not env["nodes"]["music3"] and all(env["models"][k] for k in ("music3_dit", "music3_clip", "music3_vae"))
    yue2 = not env["nodes"]["yue2"] and bool(a.yue2_root) and os.path.isdir(a.yue2_root)
    env["music_engines"] = [e for e, ok in (("music3", music3), ("yue2", yue2)) if ok]
    env["music_engine"] = env["music_engines"][0] if env["music_engines"] else "none"
    env["yue2_root"] = os.path.abspath(a.yue2_root) if yue2 else ""
    if env["music_engine"] == "none":
        env["warnings"].append("no music engine: install MiniMax Music 3 models, or pass --yue2-root, or supply your own "
                               "score files (assemble.py takes any audio)")

    root = os.path.abspath(a.comfy_root) if a.comfy_root else ""
    candidates = [a.qa_python] if a.qa_python else []
    if root:
        candidates += [os.path.join(root, "python_embeded", "python.exe"), os.path.join(root, "..", "python_embeded", "python.exe"),
                       os.path.join(root, "venv", "Scripts", "python.exe"), os.path.join(root, ".venv", "bin", "python"),
                       os.path.join(root, "venv", "bin", "python")]
    candidates.append(sys.executable)
    qa_python = next((os.path.abspath(c) for c in candidates if c and os.path.isfile(c)), sys.executable)
    env["qa_python"] = qa_python
    env["qa_modules"] = probe_python(qa_python)
    missing = [m for m, ok in env["qa_modules"].items() if ok is False]
    if missing:
        env["warnings"].append(f"QA Python {qa_python} lacks {missing}: transcript/voice checks are skipped "
                               "(pip install openai-whisper librosa into it to enable them)")

    ff_hints = [os.path.join(root, "ffmpeg", "bin", "ffmpeg.exe"), os.path.join(root, "..", "ffmpeg", "bin", "ffmpeg.exe")] if root else []
    env["ffmpeg"] = next((f for f in (find_ffmpeg(h) for h in ff_hints + [""]) if f), None)
    if not env["ffmpeg"]:
        env["problems"].append("ffmpeg not found (put it on PATH or pass --comfy-root)")

    path = save_json(os.path.join(SKILL_DIR, "env.json"), env)
    print(json.dumps({k: env[k] for k in ("comfyui_version", "models", "music_engines", "qa_python", "ffmpeg")}, indent=1))
    for p in env["problems"]:
        print("PROBLEM:", p)
    for w in env["warnings"]:
        print("warning:", w)
    print(f"{'READY' if not env['problems'] else 'NOT READY'} -> {path}")
    raise SystemExit(1 if env["problems"] else 0)


if __name__ == "__main__":
    main()
