"""Save a scene's final visible frame for the next scene's exact-frame latent continuation.

  python continuity_frame.py <project> <scene>
  -> builder_project/continuity_frames/scene_XXX_last.png   (run it again after any re-trim of that scene)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_settings, project_dir  # noqa: E402


def main():
    project, args = project_dir()
    settings = load_settings(project)
    num = int(args[0])
    out_dir = os.path.join(project, "builder_project", "continuity_frames")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"scene_{num:03d}_last.png")
    clip = os.path.join(project, "video_clips", f"scene_{num:03d}.mp4")
    subprocess.run([settings["ffmpeg"], "-v", "error", "-y", "-sseof", "-0.05", "-i", clip, "-frames:v", "1", out], check=True)
    print(out)


if __name__ == "__main__":
    main()
