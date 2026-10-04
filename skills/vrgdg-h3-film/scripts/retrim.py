"""Re-trim a scene from its raw render through the Builder's trim_scene_video route, e.g. so the clip ends on the raw
take's last frame instead of cutting into a line, or starts after a stray sound.

  python retrim.py <project> <scene> <start_seconds> [duration_seconds]
  (default duration: the rest of the raw take, i.e. end on its last frame)
"""
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Comfy, ffprobe_duration, load_settings, project_dir  # noqa: E402


def main():
    project, args = project_dir()
    settings = load_settings(project)
    num, start = int(args[0]), float(args[1])
    raw = os.path.join(project, "video_clips", "raw", f"scene_{num:03d}_raw.mp4")
    duration = float(args[2]) if len(args) > 2 else ffprobe_duration(settings["ffmpeg"], raw) - start - 0.01
    comfy = Comfy(settings["comfy_url"], os.path.join(project, "logs", "run.log"))
    fps = settings["fps"]
    frames = int(duration * fps)
    trimmed = comfy.post("/vrgdg/workflow_runner/trim_scene_video", {
        "project_folder": os.path.join(project, "builder_project"), "scene_number": num, "source_path": raw,
        "start": start, "duration": frames / fps, "frames": frames, "label": f"retrim_{start:.3f}", "mark_as_audio_video": True,
    })
    if not trimmed.get("ok"):
        raise SystemExit(f"trim failed: {trimmed.get('error')}")
    final = os.path.join(project, "video_clips", f"scene_{num:03d}.mp4")
    if os.path.isfile(final):
        shutil.move(final, os.path.join(project, "video_clips", "rejected", f"scene_{num:03d}_{time.strftime('%m%d_%H%M%S')}_pre_retrim.mp4"))
    shutil.copy2(trimmed["video_path"], final)
    comfy.log(f"scene {num}: re-trimmed from {start:.3f}s ({frames / fps:.2f}s)")


if __name__ == "__main__":
    main()
