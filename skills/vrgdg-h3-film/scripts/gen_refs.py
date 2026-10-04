"""Character and location reference images with Z-Image Turbo through the Video Builder's build_zimage_prompt route
(1280x720 first pass, latent upscale to 1920x1080 refine), one image per seed.

  python gen_refs.py <project> [spec_name ...]     (specs come from <project>/image_specs.json)
  -> <project>/<folder>/<name>_<seed>.png ; pick the best one and copy it to the REF_*.png the screenplay names
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Comfy, load_settings, output_files, project_dir, save_json  # noqa: E402


def strip_cleanup_nodes(prompt):
    """Mirror the Builder UI's default (automatic memory cleanup off): bypass RAMCleanup/VRAMCleanup pass-throughs."""
    cleanup = {k for k, v in prompt.items() if v["class_type"] in ("RAMCleanup", "VRAMCleanup")}
    for node in prompt.values():
        for name, value in list(node["inputs"].items()):
            while isinstance(value, list) and len(value) == 2 and str(value[0]) in cleanup:
                value = prompt[str(value[0])]["inputs"].get("anything")
            node["inputs"][name] = value
    for k in cleanup:
        prompt.pop(k)
    return prompt


def main():
    project, names = project_dir()
    settings = load_settings(project)
    m = settings["models"]
    if not (m.get("zimage_unet") and m.get("zimage_clip") and m.get("zimage_vae")):
        raise SystemExit("Z-Image models were not found by check_env.py; add reference images by hand instead.")
    comfy = Comfy(settings["comfy_url"], os.path.join(project, "logs", "run.log"))
    specs = json.load(open(os.path.join(project, "image_specs.json"), encoding="utf-8"))
    for spec in specs:
        if names and spec["name"] not in names:
            continue
        out_dir = os.path.join(project, spec.get("folder", "images"))
        os.makedirs(out_dir, exist_ok=True)
        for seed in spec["seeds"]:
            built = comfy.post("/vrgdg/workflow_runner/build_zimage_prompt", {
                "unet_name": m["zimage_unet"], "clip_name": m["zimage_clip"], "vae_name": m["zimage_vae"],
                "prompt": spec["prompt"], "first_pass_width": spec.get("w1", 1280), "first_pass_height": spec.get("h1", 720),
                "second_pass_width": spec.get("w2", 1920), "second_pass_height": spec.get("h2", 1080),
                "seed": seed, "seed_mode": "fixed",
            })
            if not built.get("ok"):
                raise SystemExit(f"build_zimage_prompt failed: {built.get('error')}")
            prompt = strip_cleanup_nodes(built["prompt"])
            save_json(os.path.join(project, "workflows", f"zimage_{spec['name']}_{seed}_api.json"), prompt)
            entry = comfy.run(prompt, f"zimage {spec['name']} seed {seed}", timeout=1800)
            images = [i for i in output_files(entry) if i["filename"].lower().endswith(".png")]
            if not images:
                raise SystemExit(f"{spec['name']} seed {seed}: no image in the outputs")
            dst = comfy.fetch_output(images[-1], os.path.join(out_dir, f"{spec['name']}_{seed}.png"))
            comfy.log(f"saved {dst}")


if __name__ == "__main__":
    main()
