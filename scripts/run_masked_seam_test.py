"""Masked continuation seam test: render a scene and its predecessor, then join them under the song.

Checks the cut between two scenes where singing runs across it and the second scene uses
``latent_continuation_masked``. The second scene's first visible frames are the predecessor's protected cool-down
frames, so they must have been rendered over the real song (not silence) for the lip sync to hold through the cut.
In 2 Pass the second scene's first pass continues the predecessor's first-pass latent, so its contrast and black
level do not step after the cut.

Both scenes are built from a real project, read only: their saved settings, prompts and references. The
predecessor is rendered first, then the scene continues from that render. Everything goes to a separate test project
folder, so the real project is not touched.

    ..\\..\\..\\python_embeded\\python.exe scripts/run_masked_seam_test.py              # write the workflows
    ..\\..\\..\\python_embeded\\python.exe scripts/run_masked_seam_test.py --run        # also render and stitch
    ..\\..\\..\\python_embeded\\python.exe scripts/run_masked_seam_test.py --stitch     # stitch renders made on the canvas

Workflows go to ``Workflows/masked_seam_test/`` (run the predecessor one first, then write the workflows again so the
continued scene picks up the predecessor's first-pass latent; ``--run`` does both). The stitched video (both scenes
trimmed like the Builder, under the song) goes to ``output/MaskedSeamTest/``.
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
WORKFLOW_DIR = os.path.join(PACKAGE_ROOT, "Workflows", "masked_seam_test")
TEST_NAME = "MaskedSeamTest"
TEST_PROJECT = os.path.join(COMFY_ROOT, "output", TEST_NAME)
PLAN_PATH = os.path.join(WORKFLOW_DIR, "seam_test_plan.json")
DEFAULT_PROJECT = os.path.join(COMFY_ROOT, "output", "Dance With You In It")


def _load_package():
    """Import the Builder modules without starting ComfyUI's custom node loader."""
    sys.path.insert(0, COMFY_ROOT)
    package = types.ModuleType("vrgdg_pkg")
    package.__path__ = [PACKAGE_ROOT]
    sys.modules["vrgdg_pkg"] = package
    from vrgdg_pkg.minimax import settings_payload
    from vrgdg_pkg.minimax.latent_manager import SceneLatentManager
    from vrgdg_pkg.minimax.scene_inputs import resolve_scene_inputs
    from vrgdg_pkg.runner import minimax_workflows

    return settings_payload, SceneLatentManager, resolve_scene_inputs, minimax_workflows


def _no_frame_extract(*_args) -> str:
    raise RuntimeError("This test only supports continuity off or latent_continuation_masked.")


def _scene_payload(session: dict[str, Any], index: int, settings_payload, resolve_scene_inputs) -> tuple[str, dict[str, Any]]:
    """The payload the Agent API builds for this scene (agent_api/orchestrator/video_orchestrator.py), minus the queue."""
    segments = session["segments"]
    seg = segments[index]
    scene_number = index + 1
    settings = settings_payload.minimax_h3_settings_for_scene(session, seg)
    mode = settings_payload.minimax_workflow_key(settings)
    payload = settings_payload.build_minimax_render_payload(settings)
    pass2_prompt = str(seg.get("minimax_h3_pass2_prompt") or "").strip()
    if pass2_prompt:
        payload["pass2_prompt"] = pass2_prompt
    inputs = resolve_scene_inputs(
        session, seg, settings["video_mode"], index,
        continuity_mode=str(settings["continuity_mode"] or "off"),
        previous_segment=segments[index - 1] if index > 0 else None,
        project_folder=TEST_PROJECT,
        scene_number=scene_number,
        extract_final_frame=_no_frame_extract,
        render_pass=str(settings.get("render_pass") or "single"),
    )
    payload["continuity_mode"] = inputs["continuity_mode"]
    payload["image_paths"] = inputs["image_paths"]
    payload["video_references"] = inputs["video_references"]
    prompt = str(seg.get("minimax_h3_prompt") or seg.get("i2v_prompt") or "").strip()
    if not prompt:
        raise RuntimeError(f"Scene {scene_number} has no MiniMax H3 prompt.")
    start, end = float(seg["start"]), float(seg["end"])
    payload.update({
        "i2v_prompt": prompt, "t2v_prompt": prompt, "prompt": prompt,
        "audio_path": session["audio_path"],
        "project_folder": TEST_PROJECT,
        "scene_number": scene_number,
        "start": start, "end": end, "duration": end - start,
    })
    return mode, payload


def _repoint_audio(prompt: dict[str, Any], old_path: str, new_path: str) -> None:
    old = os.path.normcase(os.path.abspath(old_path))
    for node in prompt.values():
        for name, value in node.get("inputs", {}).items():
            if isinstance(value, str) and value and os.path.normcase(os.path.abspath(value)) == old:
                node["inputs"][name] = new_path


def _build_scene(builder, mode: str, payload: dict[str, Any], suffix: str, path: str) -> dict[str, Any]:
    """Build one scene's graph, give it its own scene audio copy and output name, and save it as an API workflow."""
    build_graph = {
        "minimax_h3": builder._build_minimax_h3_api_prompt,
        "minimax_h3_2pass": builder._build_minimax_h3_2pass_api_prompt,
    }.get(mode)
    if build_graph is None:
        raise RuntimeError(f"Render pass {mode} is not supported here. Masked continuation runs in Single pass and 2 Pass.")
    result = build_graph(payload)
    prompt = result["prompt"]
    audio = result["prepared_audio"]["audio_path"]
    scene_audio = audio.replace(".wav", f"_{suffix}.wav")
    shutil.copy2(audio, scene_audio)
    _repoint_audio(prompt, audio, scene_audio)
    for node in prompt.values():
        prefix = node.get("inputs", {}).get("filename_prefix")
        if isinstance(prefix, str) and "scene_" in prefix:
            node["inputs"]["filename_prefix"] = f"{prefix}_{suffix}"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(prompt, handle, indent=2)
    timing = result["timing"]
    print(
        f"Scene {payload['scene_number']} (continuity {payload['continuity_mode']}): "
        f"{timing['h3_frame_count']} frames, scene audio {timing['audio_trim_duration_seconds']:.3f}s of "
        f"{timing['h3_render_duration_seconds']:.3f}s rendered -> {path}"
    )
    return {"scene_number": payload["scene_number"], "suffix": suffix, "workflow": path, "trim": result["post_render_trim"]}


def build(project: str, scene_number: int, copy_latents: bool = True) -> dict[str, Any]:
    """Write the workflows. ``copy_latents=False`` keeps the test folder's latents (after the predecessor rendered)."""
    settings_payload, latent_manager, resolve_scene_inputs, builder = _load_package()
    with open(os.path.join(project, "vrgdg_builder_session.json"), encoding="utf-8") as handle:
        session = json.load(handle)
    index = scene_number - 1
    if index < 1 or index >= len(session["segments"]):
        raise RuntimeError(f"Scene {scene_number} has no predecessor in this project.")

    # Building a masked scene checks its predecessor's latent, and the real files have the shape the test renders
    # will have. Copy them; the renders overwrite the copies in the test folder, never the real project.
    for number in (scene_number - 2, scene_number - 1) if copy_latents else ():
        source = latent_manager.get_path(project, number) if number >= 1 else ""
        if source and os.path.isfile(source):
            target = latent_manager.get_path(TEST_PROJECT, number)
            for suffix in ("", ".json"):
                shutil.copy2(source + suffix, target + suffix)
            # a first-pass latent left by an earlier test belongs to another render
            pass1 = latent_manager.get_pass1_path(TEST_PROJECT, number)
            for path in (pass1, pass1 + ".json"):
                if os.path.isfile(path):
                    os.remove(path)

    pred_mode, pred_payload = _scene_payload(session, index - 1, settings_payload, resolve_scene_inputs)
    mode, payload = _scene_payload(session, index, settings_payload, resolve_scene_inputs)
    if payload["continuity_mode"] != "latent_continuation_masked":
        raise RuntimeError(f"Scene {scene_number} is not set to latent_continuation_masked ({payload['continuity_mode']}).")

    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    tag = f"Scene{scene_number - 1:02d}to{scene_number:02d}"
    scenes = [
        _build_scene(builder, pred_mode, pred_payload, "pred", os.path.join(WORKFLOW_DIR, f"MaskedSeam_{tag}_1_predecessor_API.json")),
        _build_scene(builder, mode, payload, "cont", os.path.join(WORKFLOW_DIR, f"MaskedSeam_{tag}_2_continued_API.json")),
    ]
    previous, current = session["segments"][index - 1], session["segments"][index]
    plan = {
        "project": project,
        "tag": tag,
        "audio_path": session["audio_path"],
        "audio_start": float(previous["start"]),
        "audio_end": float(current["end"]),
        "scenes": scenes,
    }
    with open(PLAN_PATH, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    return plan


def _find_ffmpeg() -> str:
    portable = os.path.join(os.path.dirname(COMFY_ROOT), "python_embeded", "ffmpeg.exe")
    path = shutil.which("ffmpeg") or (portable if os.path.isfile(portable) else "")
    if not path:
        raise RuntimeError("ffmpeg was not found. Install it or put it on PATH.")
    return path


def _ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


def _latest_clip(scene_number: int, suffix: str) -> str:
    pattern = os.path.join(
        COMFY_ROOT, "output", "VRGDG_MiniMaxH3", f"{TEST_NAME}_*", f"scene_{scene_number:04d}", f"*_{suffix}_*-audio.mp4"
    )
    clips = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not clips:
        raise RuntimeError(f"No render of scene {scene_number} found ({pattern}). Run its workflow first.")
    return clips[-1]


def stitch() -> list[str]:
    """Trim the scenes exactly like the Builder and join the predecessor with each continued render under the song."""
    with open(PLAN_PATH, encoding="utf-8") as handle:
        plan = json.load(handle)
    ffmpeg = _find_ffmpeg()
    parts = []
    for scene in plan["scenes"]:
        try:
            source = _latest_clip(scene["scene_number"], scene["suffix"])
        except RuntimeError as exc:
            print(f"Skipped: {exc}")
            continue
        trim = scene["trim"]
        target = os.path.join(TEST_PROJECT, f"{plan['tag']}_{scene['suffix']}_trimmed.mp4")
        _ffmpeg([
            ffmpeg, "-y", "-ss", f"{trim['start']:.6f}", "-i", source, "-frames:v", str(trim["frames"]), "-an",
            "-vf", "setsar=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "veryfast", target,
        ])
        print(f"Scene {scene['scene_number']}: {source}\n  trimmed from {trim['start']:.3f}s, {trim['frames']} frames")
        parts.append((scene["suffix"], target))
    if not parts or parts[0][0] != "pred":
        raise RuntimeError("The predecessor has not been rendered yet.")
    outputs = []
    for suffix, continued in parts[1:]:
        final = os.path.join(TEST_PROJECT, f"{plan['tag']}_seam_{suffix}.mp4")
        # The song itself, not the clips' audio, so any lip-sync offset is against the real track.
        _ffmpeg([
            ffmpeg, "-y", "-i", parts[0][1], "-i", continued,
            "-ss", f"{plan['audio_start']:.6f}", "-t", f"{plan['audio_end'] - plan['audio_start']:.6f}", "-i", plan["audio_path"],
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "2:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "256k", "-shortest", final,
        ])
        print(f"Stitched video: {final}")
        outputs.append(final)
    return outputs


def _request(server: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(f"http://{server}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ComfyUI rejected the request: {exc.read().decode('utf-8', 'replace')[:3000]}") from exc


def run_workflow(server: str, path: str) -> None:
    """Queue one workflow and wait for it, so only one render uses the GPU at a time."""
    with open(path, encoding="utf-8") as handle:
        prompt = json.load(handle)
    prompt_id = _request(server, "/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})["prompt_id"]
    print(f"Queued {os.path.basename(path)} ({prompt_id})", flush=True)
    started = time.time()
    while True:
        entry = _request(server, f"/history/{prompt_id}").get(prompt_id)
        if entry and entry.get("status", {}).get("completed") is not None:
            status = entry["status"]
            if status.get("status_str") == "error":
                errors = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                raise RuntimeError(f"Render failed: {json.dumps(errors[-1][1] if errors else status)[:3000]}")
            print(f"  done in {time.time() - started:.0f}s", flush=True)
            return
        time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="project folder with vrgdg_builder_session.json")
    parser.add_argument("--scene", type=int, default=9, help="continued scene; the scene before it is the predecessor")
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--run", action="store_true", help="render both scenes on the running ComfyUI, then stitch")
    parser.add_argument("--stitch", action="store_true", help="only stitch renders that already exist")
    args = parser.parse_args()

    if args.stitch:
        stitch()
        return 0
    plan = build(os.path.abspath(args.project), args.scene)
    print(f"Plan written: {PLAN_PATH}")
    if not args.run:
        return 0
    run_workflow(args.server, plan["scenes"][0]["workflow"])
    # Built again now that the predecessor's render saved its latents, so the continued scene uses them.
    plan = build(os.path.abspath(args.project), args.scene, copy_latents=False)
    for scene in plan["scenes"][1:]:
        run_workflow(args.server, scene["workflow"])
    stitch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
