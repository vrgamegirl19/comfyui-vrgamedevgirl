"""Finish a scene whose render completed in ComfyUI after render_scenes.py was stopped (timeout, closed terminal):
take the newest stage-2 -audio.mp4 in the scene's Builder output folder, trim it with the saved build plan and install it.

  python adopt_render.py <project> <scene>
"""
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Comfy, load_settings, project_dir  # noqa: E402


def main():
    project, args = project_dir()
    settings = load_settings(project)
    num = int(args[0])
    meta = json.load(open(os.path.join(project, "logs", f"scene_{num:03d}_build_meta.json"), encoding="utf-8"))
    found = glob.glob(os.path.join(meta["output_folder"], "*-audio.mp4"))
    if not found:
        raise SystemExit(f"no finished render in {meta['output_folder']}")
    source = max(found, key=os.path.getmtime)
    raw = os.path.join(project, "video_clips", "raw", f"scene_{num:03d}_raw.mp4")
    shutil.copy2(source, raw)
    trim = meta["post_render_trim"]
    comfy = Comfy(settings["comfy_url"], os.path.join(project, "logs", "render.log"))
    trimmed = comfy.post("/vrgdg/workflow_runner/trim_scene_video", {
        "project_folder": os.path.join(project, "builder_project"), "scene_number": num, "source_path": raw,
        "start": trim["start"], "duration": trim["duration"], "frames": trim["frames"], "label": "final_trim",
        "mark_as_audio_video": True,
    })
    if not trimmed.get("ok"):
        raise SystemExit(f"trim failed: {trimmed.get('error')}")
    final = os.path.join(project, "video_clips", f"scene_{num:03d}.mp4")
    if os.path.isfile(final):
        shutil.move(final, os.path.join(project, "video_clips", "rejected", f"scene_{num:03d}_{time.strftime('%m%d_%H%M%S')}.mp4"))
    shutil.copy2(trimmed["video_path"], final)
    comfy.log(f"scene {num}: saved {final} (adopted {os.path.basename(source)})")


if __name__ == "__main__":
    main()
