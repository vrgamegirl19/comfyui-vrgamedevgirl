"""QA contact sheet for one take: 4 frames from the first second, 7 across the clip and the last frame (12 tiles).

usage: qa_sheet.py --work W SCENE_NUMBER VIDEO.mp4 [--tag TAG]   -> qa/scene_NN_TAG.jpg

The first-second frames catch lens stares and warm-up leftovers (a one-frame flash of another shot); the spread
catches wrong props, clones, small faces and cuts; the last frame is what the next scene must match.
Read the sheet with the Read tool after every take. For lip-sync or a fast action, extract a denser strip:
  ffmpeg -ss 4.0 -t 1.2 -i VIDEO -vf "fps=10,scale=320:-2,tile=6x2" -frames:v 1 strip.jpg
"""
import os
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ffprobe_duration, load_config, parser, wpath  # noqa: E402


def font(size):
    for name in ("C:/Windows/Fonts/arialbd.ttf", "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    p = parser(__doc__)
    p.add_argument("scene", type=int)
    p.add_argument("video")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    cfg = load_config(args.work)
    d = ffprobe_duration(args.video)
    times = [0.0, 0.25, 0.6, 1.0] + [d * k / 9 for k in range(1, 8)] + [max(0.0, d - 0.06)]
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(times):
            dest = os.path.join(tmp, f"f{i:02d}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", args.video, "-frames:v", "1",
                            "-vf", "scale=480:-2", dest], check=False)
            if os.path.isfile(dest):
                tiles.append((t, Image.open(dest).convert("RGB")))
    if not tiles:
        sys.exit("no frames extracted")
    tw, th = tiles[0][1].size
    rows = (len(tiles) + 3) // 4
    sheet = Image.new("RGB", (4 * tw, rows * (th + 24)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    f = font(18)
    for i, (t, im) in enumerate(tiles):
        x, y = (i % 4) * tw, (i // 4) * (th + 24)
        sheet.paste(im, (x, y + 24))
        draw.text((x + 6, y + 3), f"scene {args.scene}  t={t:.2f}s", fill=(255, 220, 0), font=f)
    out_dir = wpath(cfg, "qa")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"scene_{args.scene:02d}_{args.tag or time.strftime('%H%M%S')}.jpg")
    sheet.save(out, quality=85)
    print(out)


if __name__ == "__main__":
    main()
