"""Build (and optionally run) a two-scene Latent Continuation Masked test workflow.

Scene 1 is a normal single pass H3 render that saves its latent. Scene 2 continues it with
``latent_continuation_masked`` in the same graph. Both scenes come from the Builder's own graph compiler, so the graph
matches what the Video Builder queues. The two scenes share one model, text encoder and VAE loader.

Writes ``Workflows/masked_continuation_test/MaskedContinuation_2Scene_Test_API.json`` (load it in ComfyUI, then Run).
With ``--run`` it also queues the graph on the running ComfyUI and prints the output videos::

    ..\\..\\..\\python_embeded\\python.exe scripts/run_masked_continuation_test.py [--run]

The raw outputs keep the warm-up: scene 2's video begins with the end of scene 1 (the protected head plus the frames
after it), then continues. Do not join the raw clips, the head repeats. ``--stitch`` trims both clips exactly like the
Video Builder (``post_render_trim``) and joins them into ``output/MaskedContinuationTest/MaskedContinuation_FINAL.mp4``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
import uuid
from typing import Any

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACKAGE_ROOT))
WORKFLOW_PATH = os.path.join(PACKAGE_ROOT, "Workflows", "masked_continuation_test", "MaskedContinuation_2Scene_Test_API.json")
TRIM_PLAN_PATH = os.path.join(os.path.dirname(WORKFLOW_PATH), "trim_plan.json")
FINAL_PATH = os.path.join(COMFY_ROOT, "output", "MaskedContinuationTest", "MaskedContinuation_FINAL.mp4")
PROJECT_NAME = "MaskedContinuationTest"
SHARED_LOADER_CLASSES = ("DiffusionModelLoaderKJ", "CLIPLoader", "VAELoader")
SCENE_2_ID_OFFSET = 1000

DEFAULT_IMAGE = os.path.join(os.path.expanduser("~"), "Pictures", "darrel", "darrel.png")
SUBJECT_NAME = "the man Darrel"
SUBJECT_DESCRIPTION = "a Black man with short twists under a black paisley bandana, a gold chain, and a black and silver patterned hoodie"
STYLE = "photorealistic cinematic"

# Shot descriptions only. The reference definitions and soundscape around them come from the Builder's own assembler.
SCENE_1_SHOT = (
    "Darrel walks steadily down a rain-soaked neon alley at night. The camera tracks backward in front of him at "
    "walking pace, framing him from the chest up. He looks ahead, rain runs off his shoulders, and he keeps a steady "
    "stride past glowing signs reflected in the puddles."
)
SCENE_2_SHOT = (
    "Darrel keeps walking down the same rain-soaked neon alley at the same steady pace. The camera keeps tracking "
    "backward in front of him from the chest up, in the same framing. He lifts his head slightly and glances to his "
    "left at a glowing sign as he walks on, his stride unbroken, rain still running off his hoodie."
)

# Every key the Builder would send for a MiniMax H3 reference-to-video render, single pass, one reference image.
BASE_PAYLOAD: dict[str, Any] = {
    "video_mode": "reference_to_video",
    "audio_mode": "input_audio",
    "aspect_ratio": "16:9 (Widescreen)",
    "megapixels": 0.5625,
    "seed": 69,
    "video_references": [],
}


def _load_builder():
    """Import the runner without starting ComfyUI's custom node loader."""
    sys.path.insert(0, COMFY_ROOT)
    package = types.ModuleType("vrgdg_pkg")
    package.__path__ = [PACKAGE_ROOT]
    sys.modules["vrgdg_pkg"] = package
    from vrgdg_pkg.minimax import prompt_assembly, shot_prompt
    from vrgdg_pkg.minimax.latent_manager import SceneLatentManager
    from vrgdg_pkg.runner import minimax_workflows

    return minimax_workflows, SceneLatentManager, prompt_assembly, shot_prompt


def _shift_scene_2(prompt: dict[str, Any], shared_ids: set[str]) -> dict[str, Any]:
    """Give scene 2's nodes their own ids. Shared loaders keep scene 1's ids, so links to them stay valid."""
    def new_id(node_id: str) -> str:
        return str(node_id) if str(node_id) in shared_ids else str(int(node_id) + SCENE_2_ID_OFFSET)

    def remap(value: Any) -> Any:
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in prompt:
            return [new_id(value[0]), value[1]]
        return value

    shifted = {}
    for node_id, node in prompt.items():
        if str(node_id) in shared_ids:
            continue
        copy = json.loads(json.dumps(node))
        copy["inputs"] = {name: remap(value) for name, value in copy["inputs"].items()}
        title = copy.get("_meta", {}).get("title", copy["class_type"])
        copy["_meta"] = {"title": f"Scene 2 · {title}"}
        shifted[new_id(node_id)] = copy
    return shifted


def build_workflow(audio: str, image: str, start: float, scene_seconds: float, context_frames: int) -> tuple[dict[str, Any], dict[str, Any]]:
    builder, latent_manager, prompt_assembly, shot_prompt = _load_builder()
    project = os.path.join(COMFY_ROOT, "output", PROJECT_NAME)
    os.makedirs(project, exist_ok=True)

    cut_plan = prompt_assembly.storyboard_cut_plan_for_duration(scene_seconds, 0)
    items = [{"kind": "subject", "label": SUBJECT_NAME, "description": SUBJECT_DESCRIPTION, "image": {"path": image}}]
    frame = shot_prompt.reference_frame(items, cut_plan, STYLE, "input_audio", "")

    def scene_prompt(shot: str) -> str:
        return shot_prompt.wrap_reference_prompt(shot_prompt.assemble_prompt([shot], cut_plan, STYLE), frame)

    def payload(number: int, shot: str, scene_start: float, continuity: str) -> dict[str, Any]:
        return {
            **BASE_PAYLOAD,
            "audio_path": audio,
            "image_paths": [image],
            "project_folder": project,
            "scene_number": number,
            "prompt": scene_prompt(shot),
            "timeline_start_seconds": scene_start,
            "timeline_end_seconds": scene_start + scene_seconds,
            "continuity_mode": continuity,
            "latent_context_frames": context_frames,
        }

    first = builder._build_minimax_h3_api_prompt(payload(1, SCENE_1_SHOT, start, "off"))

    # Building scene 2 checks that scene 1's latent exists. It is only written once scene 1 renders, so a stand-in
    # file is used for the build and removed again. The graph loads the real file after scene 1 has saved it.
    latent_path = latent_manager.get_path(project, 1)
    standin = not os.path.isfile(latent_path)
    if standin:
        import torch

        latent_manager.save_latent(
            project, 1,
            {"video": torch.zeros(1, 24, 47, 4, 4), "audio": torch.zeros(1, 32, 2, 264)},
            metadata={"tail_padding_frames": 0},
        )
    try:
        second = builder._build_minimax_h3_api_prompt(payload(2, SCENE_2_SHOT, start + scene_seconds, "latent_continuation_masked"))
    finally:
        if standin:
            for path in (latent_path, latent_path + ".json"):
                if os.path.isfile(path):
                    os.remove(path)

    masked = second["latent_continuation_settings"]
    if not masked.get("enabled"):
        raise RuntimeError(f"Masked continuation was not applied to scene 2: {masked}")

    shared = {
        str(node_id) for node_id, node in first["prompt"].items()
        if node["class_type"] in SHARED_LOADER_CLASSES
    }
    for node_id, node in first["prompt"].items():
        node["_meta"] = {"title": f"Scene 1 · {node.get('_meta', {}).get('title', node['class_type'])}"} \
            if node_id not in shared else node.get("_meta", {})
    scene_2 = _shift_scene_2(second["prompt"], shared)

    # Scene 2 may only read scene 1's latent after scene 1 has saved it.
    save_1 = first["save_latent_settings"]["save_node_id"]
    scene_2[str(int(masked["load_node_id"]) + SCENE_2_ID_OFFSET)]["inputs"]["run_after"] = [save_1, 0]
    trims = {"scene_1": first["post_render_trim"], "scene_2": second["post_render_trim"]}
    return {**first["prompt"], **scene_2}, trims


def _find_ffmpeg() -> str:
    portable = os.path.join(os.path.dirname(COMFY_ROOT), "python_embeded", "ffmpeg.exe")
    path = shutil.which("ffmpeg") or (portable if os.path.isfile(portable) else "")
    if not path:
        raise RuntimeError("ffmpeg was not found. Install it or put it on PATH.")
    return path


def _latest_scene_clip(scene_number: int) -> str:
    pattern = os.path.join(COMFY_ROOT, "output", "VRGDG_MiniMaxH3", "MaskedContinuationTest_*", f"scene_{scene_number:04d}", "*-audio.mp4")
    clips = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not clips:
        raise RuntimeError(f"No rendered clip for scene {scene_number} found ({pattern}). Render the workflow first.")
    return clips[-1]


def stitch() -> str:
    """Trim both raw clips to their scene (head and padding removed, same as the Builder) and join them."""
    with open(TRIM_PLAN_PATH, encoding="utf-8") as handle:
        trims = json.load(handle)
    ffmpeg = _find_ffmpeg()
    work = os.path.dirname(FINAL_PATH)
    os.makedirs(work, exist_ok=True)
    parts = []
    for number in (1, 2):
        trim = trims[f"scene_{number}"]
        source = _latest_scene_clip(number)
        target = os.path.join(work, f"scene_{number}_trimmed.mkv")
        # -ss before -i seeks both streams from the clip start, -frames:v keeps exactly the scene frame count.
        # The audio stays PCM here: AAC adds encoder padding that would shift the second clip when they are joined.
        command = [
            ffmpeg, "-y", "-ss", f"{trim['start']:.6f}", "-i", source, "-t", f"{trim['duration']:.6f}",
            "-frames:v", str(trim["frames"]), "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-preset", "veryfast", "-c:a", "pcm_s16le", target,
        ]
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])
        print(f"Scene {number}: {source}\n  trimmed to start {trim['start']:.3f}s, {trim['frames']} frames -> {target}")
        parts.append(target)
    result = subprocess.run([
        ffmpeg, "-y", "-i", parts[0], "-i", parts[1],
        "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-c:a", "aac",
        FINAL_PATH,
    ], capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    print(f"Stitched video: {FINAL_PATH}")
    return FINAL_PATH


def _request(server: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(f"http://{server}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ComfyUI rejected the request: {exc.read().decode('utf-8', 'replace')[:3000]}") from exc


def run_workflow(server: str, prompt: dict[str, Any]) -> list[str]:
    queued = _request(server, "/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})
    prompt_id = queued["prompt_id"]
    print(f"Queued ({prompt_id}). Rendering scene 1, then scene 2...", flush=True)
    started = time.time()
    while True:
        entry = _request(server, f"/history/{prompt_id}").get(prompt_id)
        if entry and entry.get("status", {}).get("completed") is not None:
            status = entry["status"]
            if status.get("status_str") == "error":
                errors = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                raise RuntimeError(f"Render failed: {json.dumps(errors[-1][1] if errors else status)[:3000]}")
            print(f"Done in {time.time() - started:.0f}s", flush=True)
            files = []
            for output in entry.get("outputs", {}).values():
                for key in ("gifs", "videos", "images"):
                    files.extend(item.get("fullpath") or item.get("filename", "") for item in output.get(key, []))
            return [path for path in files if path]
        time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--audio", default=os.path.join(COMFY_ROOT, "input", "02 In Bloom.mp3"))
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="reference image of the character")
    parser.add_argument("--start", type=float, default=30.0, help="seconds into the audio where scene 1 starts")
    parser.add_argument("--scene-seconds", type=float, default=5.0)
    parser.add_argument("--context-frames", type=int, default=39, choices=(39, 90, 141, 192))
    parser.add_argument("--run", action="store_true", help="also queue the workflow on the running ComfyUI, then stitch")
    parser.add_argument("--stitch", action="store_true", help="only trim and join the clips already rendered")
    args = parser.parse_args()

    if args.stitch:
        stitch()
        return 0

    if not os.path.isfile(args.audio):
        print(f"Audio file not found: {args.audio}\nPass another with --audio.")
        return 1

    if not os.path.isfile(args.image):
        print(f"Reference image not found: {args.image}\nPass another with --image.")
        return 1

    workflow, trims = build_workflow(
        os.path.abspath(args.audio), os.path.abspath(args.image), args.start, args.scene_seconds, args.context_frames
    )
    os.makedirs(os.path.dirname(WORKFLOW_PATH), exist_ok=True)
    with open(WORKFLOW_PATH, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, indent=2)
    with open(TRIM_PLAN_PATH, "w", encoding="utf-8") as handle:
        json.dump(trims, handle, indent=2)
    print(f"Workflow written: {WORKFLOW_PATH}")
    if not args.run:
        return 0

    files = run_workflow(args.server, workflow)
    print("Raw outputs (scene 2 starts with the warm-up, do not join these):", *files, sep="\n  ")
    stitch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
