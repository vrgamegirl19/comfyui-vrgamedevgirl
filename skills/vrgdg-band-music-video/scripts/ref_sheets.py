"""Labelled contact sheets of a folder of images, so you can look at many references in a few Read calls.

usage: ref_sheets.py FOLDER OUT_PREFIX [--cols 2] [--width 760] [--per 6] [--filter .png]
       -> OUT_PREFIX_1.jpg, OUT_PREFIX_2.jpg, ... (each tile labelled with its number and file name)

Typical: character sheets 2 per row at 760 px; location plates 3 per row at 520 px, 9 per sheet. Put the output
in the scratchpad, not in the project.
"""
import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder")
    p.add_argument("out_prefix")
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--width", type=int, default=760)
    p.add_argument("--per", type=int, default=6)
    p.add_argument("--filter", default="")
    a = p.parse_args()
    files = sorted(f for f in os.listdir(a.folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and a.filter.lower() in f.lower())
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    for start in range(0, len(files), a.per):
        chunk = files[start:start + a.per]
        tiles = []
        for f in chunk:
            im = Image.open(os.path.join(a.folder, f)).convert("RGB")
            tiles.append((f, im.resize((a.width, int(im.height * a.width / im.width)))))
        th = max(t[1].height for t in tiles) + 30
        sheet = Image.new("RGB", (a.cols * a.width, math.ceil(len(tiles) / a.cols) * th), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        for i, (f, im) in enumerate(tiles):
            x, y = (i % a.cols) * a.width, (i // a.cols) * th
            sheet.paste(im, (x, y + 30))
            draw.text((x + 6, y + 4), f"{start + i + 1}: {f[:60]}", fill=(255, 255, 0), font=font)
        out = f"{a.out_prefix}_{start // a.per + 1}.jpg"
        sheet.save(out, quality=88)
        print(out, len(chunk))


if __name__ == "__main__":
    main()
