"""Cut review: for every scene-to-scene join in play order, the last frame of scene A beside the first frame of scene B,
six joins per image (run with qa_python).   python cut_sheet.py <project>  -> logs/qa/cuts_XX.jpg  (view them!)
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_settings, project_dir  # noqa: E402

W = 400


def frame(ffmpeg, project, num, last):
    out = os.path.join(project, "logs", "qa", "_cut.png")
    clip = os.path.join(project, "video_clips", f"scene_{num:03d}.mp4")
    args = ["-sseof", "-0.1", "-i", clip] if last else ["-i", clip]
    subprocess.run([ffmpeg, "-v", "error", "-y", *args, "-frames:v", "1", "-vf", f"scale={W}:-1", out], check=True)
    return Image.open(out).convert("RGB")


def main():
    project, _ = project_dir()
    settings = load_settings(project)
    score_path = os.path.join(project, "score.json")
    score = json.load(open(score_path, encoding="utf-8")) if os.path.isfile(score_path) else {}
    board = [s["scene_number"] for s in json.load(open(os.path.join(project, "storyboard", "scenes.json"), encoding="utf-8"))["scenes"]]
    order = score.get("play_order") or board
    order = [n for n in order if os.path.isfile(os.path.join(project, "video_clips", f"scene_{n:03d}.mp4"))]
    pairs = list(zip(order, order[1:]))
    for g in range(0, len(pairs), 6):
        rows = []
        for a, b in pairs[g:g + 6]:
            fa, fb = frame(settings["ffmpeg"], project, a, True), frame(settings["ffmpeg"], project, b, False)
            row = Image.new("RGB", (W * 2 + 8, fa.height + 18), (30, 30, 30))
            row.paste(fa, (0, 18))
            row.paste(fb, (W + 8, 18))
            ImageDraw.Draw(row).text((4, 3), f"{a} end  |  {b} start", fill=(255, 255, 0))
            rows.append(row)
        sheet = Image.new("RGB", (rows[0].width * 2, rows[0].height * ((len(rows) + 1) // 2)))
        for i, r in enumerate(rows):
            sheet.paste(r, ((i % 2) * r.width, (i // 2) * r.height))
        path = os.path.join(project, "logs", "qa", f"cuts_{g // 6:02d}.jpg")
        sheet.save(path, quality=88)
        print(path)


if __name__ == "__main__":
    main()
