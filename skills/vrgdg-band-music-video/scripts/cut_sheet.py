"""Cut sheets: the last frame of scene N beside the first frame of scene N+1, four cuts per image.

usage: cut_sheet.py --work W [--project P] [--range FIRST LAST]   -> qa/cuts_AA_BB.jpg

Check every cut, not just every scene: who is present, positions and screen direction, props (does the stolen item
stay with the right person?), wardrobe, lighting, and text-only extras whose look must not change.
"""
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, ffprobe_duration, load_config, parser, video_path, wpath  # noqa: E402
from qa_sheet import font  # noqa: E402


def frame(path, t, dest):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", path, "-frames:v", "1",
                    "-vf", "scale=480:-2", dest], check=False)


def main():
    p = parser(__doc__)
    p.add_argument("--range", nargs=2, type=int)
    args = p.parse_args()
    cfg = load_config(args.work)
    api = Api(cfg, args.project)
    videos = {}
    for scene in api.scenes():
        path = video_path(scene)
        if path and os.path.isfile(path):
            videos[int(scene["number"])] = path
    if not videos:
        sys.exit("no rendered scenes")
    lo, hi = args.range or (min(videos), max(videos))
    pairs = [(n, n + 1) for n in range(lo, hi) if n in videos and n + 1 in videos]
    out_dir = wpath(cfg, "qa")
    os.makedirs(out_dir, exist_ok=True)
    f = font(18)
    with tempfile.TemporaryDirectory() as tmp:
        for page in range(0, len(pairs), 4):
            chunk = pairs[page:page + 4]
            tiles = []
            for a, b in chunk:
                fa, fb = os.path.join(tmp, f"{a}a.png"), os.path.join(tmp, f"{b}b.png")
                frame(videos[a], ffprobe_duration(videos[a]) - 0.06, fa)
                frame(videos[b], 0.0, fb)
                tiles.append((a, b, Image.open(fa).convert("RGB"), Image.open(fb).convert("RGB")))
            tw, th = tiles[0][2].size
            sheet = Image.new("RGB", (2 * tw + 10, len(tiles) * (th + 26)), (20, 20, 20))
            draw = ImageDraw.Draw(sheet)
            for i, (a, b, ia, ib) in enumerate(tiles):
                y = i * (th + 26)
                sheet.paste(ia, (0, y + 26))
                sheet.paste(ib, (tw + 10, y + 26))
                draw.text((6, y + 4), f"end of scene {a}", fill=(255, 220, 0), font=f)
                draw.text((tw + 16, y + 4), f"start of scene {b}", fill=(255, 220, 0), font=f)
            out = os.path.join(out_dir, f"cuts_{chunk[0][0]:02d}_{chunk[-1][1]:02d}.jpg")
            sheet.save(out, quality=85)
            print(out)


if __name__ == "__main__":
    main()
