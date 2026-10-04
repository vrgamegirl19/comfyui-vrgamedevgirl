"""Render scenes with the Video Builder's MiniMax H3 "Reference to Video - 2 Pass" route, switched to H3 built-in audio.

  python render_scenes.py <project> [scene_number ...] [--force]

The Builder's 2-pass route only takes input audio, so it is given a silent timing track, and the returned API prompt is
rewired here (in the project copy only) so pass 1 samples H3's native joint video+audio latent:
  - the reference-audio inputs on MiniMaxH3ReferenceToVideo, VRGDG_MiniMaxH3AudioDrive and its VHS_LoadAudio are removed;
  - everything that read the audio-locked latent reads MiniMaxH3ReferenceToVideo's joint latent instead;
  - pass-1 audio is decoded with LTXVAudioVAEDecode and becomes the VHS_VideoCombine audio.
Nodes are found by class type, not by ID, so the rewire survives template renumbering.
Run it in the background for long batches; each finished scene is logged as "scene N: saved".
"""
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Comfy, ffprobe_duration, load_settings, output_files, project_dir, save_json  # noqa: E402

TWO_PASS = {  # the Builder's MiniMax H3 2-pass defaults (Ref to Video, ref_pass_mode = two_pass)
    "audio_mode": "input_audio", "video_mode": "reference_to_video", "aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.9,
    "ref_image_size": "max", "two_pass_lora_strength": 1, "latent_upscale_scale": 2,
    "two_pass_use_feedforward": False, "two_pass_use_block_sparse_attention": False, "two_pass_use_fast_vae_decode": False,
    "final_resize_method": "nvidia_rtx_vsr", "output_crf": 19,
    "pass1_steps": 20, "pass1_denoise": 1, "pass1_sampler_name": "res_multistep", "pass1_scheduler": "simple",
    "pass2_steps": 2, "pass2_denoise": 0.2, "pass2_sampler_name": "res_multistep", "pass2_scheduler": "simple",
    "sage_attention": "auto", "enable_fp16_accumulation": True, "tail_loss_frames": 0,
}


def by_class(prompt, cls):
    return [k for k, v in prompt.items() if v["class_type"] == cls]


def use_native_audio(prompt):
    (r2v,) = by_class(prompt, "MiniMaxH3ReferenceToVideo")
    drives = by_class(prompt, "VRGDG_MiniMaxH3AudioDrive")
    loaders = {str(prompt[d]["inputs"]["source_audio"][0]) for d in drives if isinstance(prompt[d]["inputs"].get("source_audio"), list)}
    for name in [n for n in prompt[r2v]["inputs"] if n.startswith("ref_audios.")]:
        del prompt[r2v]["inputs"][name]
    for node in prompt.values():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in drives and value[1] == 0:
                node["inputs"][name] = [r2v, 1]
    samplers = [k for k in by_class(prompt, "SamplerCustomAdvanced") if prompt[k]["inputs"].get("latent_image") == [r2v, 1]]
    if len(samplers) != 1:
        raise RuntimeError(f"expected one pass-1 sampler reading the H3 latent, found {samplers}; the Builder template changed")
    for k in drives + sorted(loaders):
        prompt.pop(k, None)
    decode = str(max(int(k) for k in prompt if k.isdigit()) + 1)
    prompt[decode] = {"class_type": "LTXVAudioVAEDecode", "_meta": {"title": "Decode MiniMax H3 native audio (pass 1)"},
                      "inputs": {"samples": [samplers[0], 0], "audio_vae": prompt[r2v]["inputs"]["audio_vae"]}}
    for k in by_class(prompt, "VHS_VideoCombine"):
        prompt[k]["inputs"]["audio"] = [decode, 0]
    dangling = [(k, n) for k, node in prompt.items() for n, v in node["inputs"].items()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] not in prompt]
    if dangling:
        raise RuntimeError(f"native-audio rewire left dangling links: {dangling}")
    return prompt


def render_scene(comfy, project, settings, scene, force):
    num = int(scene["scene_number"])
    clips = os.path.join(project, "video_clips")
    final_path = os.path.join(clips, f"scene_{num:03d}.mp4")
    if os.path.isfile(final_path) and not force:
        comfy.log(f"scene {num}: already rendered, skipping (use --force)")
        return
    if scene["continuity_mode"].endswith("exact_frame") and not os.path.isfile(scene["latent_exact_frame_path"]):
        raise RuntimeError(f"scene {num}: exact-frame continuation needs {scene['latent_exact_frame_path']} "
                           f"(extract the last frame of scene {num - 1} with continuity_frame.py)")
    m, seed = settings["models"], int(scene.get("seed", settings["seed"]))
    payload = {
        **TWO_PASS,
        "diffusion_model_name": m["h3_diffusion"], "clip_name": m["h3_clip"], "video_vae_name": m["h3_video_vae"],
        "audio_vae_name": m["h3_audio_vae"], "latent_upscaler_name": m["h3_latent_upscaler"],
        "two_pass_lora_name": m["h3_turbo_lora"],
        "final_width": settings["final_width"], "final_height": settings["final_height"],
        "project_folder": os.path.join(project, "builder_project"), "scene_number": num,
        "audio_path": os.path.join(project, "audio", "silent_timing_track.wav"),
        "prompt": scene["prompt"], "image_paths": scene["image_paths"], "video_references": [],
        "timeline_start_seconds": scene["start"], "timeline_end_seconds": scene["end"], "source_start_seconds": scene["start"],
        "seed": seed, "pass1_seed": seed, "pass2_seed": seed,
        "pre_frames": settings["warmup_frames"], "continuity_mode": scene["continuity_mode"],
        "latent_context_frames": settings["latent_context_frames"],
        "latent_exact_frame_path": scene["latent_exact_frame_path"],
    }
    built = comfy.post("/vrgdg/workflow_runner/build_minimax_h3_2pass_prompt", payload, timeout=300)
    if not built.get("ok"):
        raise RuntimeError(f"scene {num} build failed: {built.get('error')}")
    prompt = use_native_audio(built["prompt"])
    save_json(os.path.join(project, "workflows", f"scene_{num:03d}_h3_2pass_builtin_audio_api.json"), prompt)
    save_json(os.path.join(project, "logs", f"scene_{num:03d}_build_meta.json"), {k: v for k, v in built.items() if k != "prompt"})
    started = time.time()
    entry = comfy.run(prompt, f"scene {num} H3 2-pass", timeout=3 * 3600)
    videos = [i for i in output_files(entry) if i["filename"].lower().endswith("-audio.mp4")]
    raw = os.path.join(clips, "raw", f"scene_{num:03d}_raw.mp4")
    if videos:
        comfy.fetch_output(videos[-1], raw)
    else:
        found = [p for p in glob.glob(os.path.join(built["output_folder"], "*-audio.mp4")) if os.path.getmtime(p) >= started - 2]
        if not found:
            raise RuntimeError(f"scene {num}: no -audio.mp4 output")
        shutil.copy2(max(found, key=os.path.getmtime), raw)
    # cut only the warm-up off the front and keep everything to the raw take's last frame: H3 renders a few frames
    # past the scene length, and trimming to the exact length clipped the last word of many lines
    start = built["post_render_trim"]["start"]
    duration = ffprobe_duration(settings["ffmpeg"], raw) - start - 0.01
    trimmed = comfy.post("/vrgdg/workflow_runner/trim_scene_video", {
        "project_folder": os.path.join(project, "builder_project"), "scene_number": num, "source_path": raw,
        "start": start, "duration": duration, "frames": int(duration * settings["fps"]), "label": "final_trim",
        "mark_as_audio_video": True,
    })
    if not trimmed.get("ok"):
        raise RuntimeError(f"scene {num} trim failed: {trimmed.get('error')}")
    if os.path.isfile(final_path):
        shutil.move(final_path, os.path.join(clips, "rejected", f"scene_{num:03d}_{time.strftime('%m%d_%H%M%S')}.mp4"))
    shutil.copy2(trimmed["video_path"], final_path)
    comfy.log(f"scene {num}: saved {final_path}")
    comfy.free()   # a GPU left full of cached memory made renders 2x slower


def main():
    project, args = project_dir()
    force = "--force" in args
    wanted = {int(a) for a in args if a.isdigit()}
    settings = load_settings(project)
    comfy = Comfy(settings["comfy_url"], os.path.join(project, "logs", "render.log"))
    board = json.load(open(os.path.join(project, "storyboard", "scenes.json"), encoding="utf-8"))["scenes"]
    for scene in board:
        if wanted and scene["scene_number"] not in wanted:
            continue
        try:
            render_scene(comfy, project, settings, scene, force)
        except Exception as exc:
            comfy.log(f"scene {scene['scene_number']} ERROR: {exc}")
    comfy.log("batch finished")


if __name__ == "__main__":
    main()
