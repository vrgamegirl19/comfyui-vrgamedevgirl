"""Thumbnail overlay: the YouTube look, made by layers on top of the chosen frame, with no image model.

An image workflow that redraws the frame (Krea 2, MOSS-Image, NanoBanana...) redraws the person too: the face, the skin tone and
the scene drift, whatever the prompt says. This module never redraws anything. It keeps every pixel of the frame and puts layers on
it, like a thumbnail made in an image editor: a color grade (contrast, saturation, warm highlights over cool shadows), a glow on the
brightest parts, a vignette, and the title in bold letters with an outline, a drop shadow and a glow, placed in the calmest part of
the frame. With strength 0 and no title the result is byte for byte the frame.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

OVERLAY_DEFAULT_STRENGTH = 60
OVERLAY_POSITIONS = ("auto", "top", "bottom", "left", "right")
# The look of the title: its fill, the outline around it and the glow behind it.
OVERLAY_COLORS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "yellow": {"fill": (255, 214, 10), "outline": (22, 10, 0), "glow": (255, 120, 20)},
    "white": {"fill": (255, 255, 255), "outline": (10, 10, 24), "glow": (90, 160, 255)},
    "orange": {"fill": (255, 128, 24), "outline": (26, 8, 0), "glow": (255, 64, 0)},
    "cyan": {"fill": (36, 240, 255), "outline": (0, 16, 38), "glow": (0, 170, 255)},
    "red": {"fill": (255, 52, 42), "outline": (22, 0, 0), "glow": (255, 0, 64)},
}
OVERLAY_DEFAULT_COLOR = "yellow"
# Bold display fonts, first one found wins. Windows has Impact; the rest are fallbacks for other systems.
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\impact.ttf", r"C:\Windows\Fonts\ariblk.ttf", r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf", "/Library/Fonts/Impact.ttf", "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_WARM = np.array([1.06, 1.0, 0.92], dtype=np.float32)
_COOL = np.array([0.95, 1.0, 1.08], dtype=np.float32)
# The look at the default strength (60) is the one the grade numbers below are written for.
_STRENGTH_UNIT = 0.6
_MAX_TITLE_CHARACTERS = 80
_MAX_LINES = 3


def overlay_strength(value: Any) -> int:
    """The strength as a whole number from 0 to 100; a missing or wrong value is the default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return OVERLAY_DEFAULT_STRENGTH
    if not math.isfinite(number):
        return OVERLAY_DEFAULT_STRENGTH
    return int(max(0, min(100, round(number))))


def _blur(array: np.ndarray, sigma: float) -> np.ndarray:
    image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(sigma))).astype(np.float32) / 255.0


def grade_frame(frame: Image.Image, strength: int) -> Image.Image:
    """The frame with the grade, the glow on its brightest parts and the vignette. Strength 0 returns the frame itself."""
    factor = overlay_strength(strength) / 100.0 / _STRENGTH_UNIT
    if factor <= 0:
        return frame.convert("RGB")
    width, height = frame.size
    base = np.asarray(frame.convert("RGB")).astype(np.float32) / 255.0
    luma = (base * _LUMA).sum(axis=2, keepdims=True)
    graded = base + 0.35 * factor * (base - luma)                                   # more color
    curve = 0.5 + 0.5 * np.tanh(2.2 * (graded - 0.5)) / np.tanh(1.1)                # an S curve: more contrast
    blend = min(1.0, 0.6 * factor)
    graded = blend * curve + (1.0 - blend) * graded
    warmth = np.clip(luma * 1.6, 0.0, 1.0)                                          # warm highlights over cool shadows
    graded = graded * (1.0 + factor * (_COOL * (1.0 - warmth) + _WARM * warmth - 1.0))
    bright = np.clip((graded - 0.62) * 2.6, 0.0, 1.0)                               # the brightest parts glow
    glow = _blur(bright, max(width, height) * 0.012)
    graded = 1.0 - (1.0 - np.clip(graded, 0.0, 1.0)) * (1.0 - glow * min(0.9, 0.55 * factor))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)                         # a vignette
    distance = np.sqrt(((xs - width / 2) / (width / 2)) ** 2 + ((ys - height / 2) / (height / 2)) ** 2)
    graded = graded * (1.0 - min(0.7, 0.38 * factor) * np.clip(distance - 0.55, 0.0, 1.0) ** 1.6)[..., None]
    return Image.fromarray(np.rint(np.clip(graded, 0.0, 1.0) * 255).astype(np.uint8))


def _font(size: int) -> Tuple[ImageFont.FreeTypeFont, str]:
    """The first bold display font that is installed at ``size``, and its name."""
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size), os.path.basename(path)
            except OSError:
                continue
    return ImageFont.load_default(size), "default"


def _split_lines(words: List[str], count: int) -> List[str]:
    """``words`` in ``count`` lines whose longest line is as short as it can be."""
    if count <= 1 or len(words) < count:
        return [" ".join(words)]
    best: Optional[List[str]] = None
    best_length = None

    def consider(cuts: Tuple[int, ...]) -> None:
        nonlocal best, best_length
        edges = (0, *cuts, len(words))
        lines = [" ".join(words[edges[i]:edges[i + 1]]) for i in range(len(edges) - 1)]
        longest = max(len(line) for line in lines)
        if best_length is None or longest < best_length:
            best, best_length = lines, longest

    if count == 2:
        for cut in range(1, len(words)):
            consider((cut,))
    else:
        for first in range(1, len(words) - 1):
            for second in range(first + 1, len(words)):
                consider((first, second))
    return best or [" ".join(words)]


class _TitleLayout:
    """The lines of the title at one font size: where each character goes and how big the whole block is."""

    def __init__(self, lines: List[str], size: int) -> None:
        self.size = size
        self.font, self.font_name = _font(size)
        self.stroke = max(2, round(size * 0.07))
        self.tracking = size * 0.035
        left, top, _right, _bottom = self.font.getbbox("H", anchor="ls")
        self.cap_height = -top
        self.line_step = self.cap_height * 1.2 + self.stroke
        self.lines = lines
        self.advances = [[self.font.getlength(character) + self.tracking for character in line] for line in lines]
        self.line_widths = [sum(advances) - self.tracking for advances in self.advances]
        self.width = int(math.ceil(max(self.line_widths))) + 2 * self.stroke
        self.height = int(math.ceil(self.cap_height + (len(lines) - 1) * self.line_step + size * 0.08)) + 2 * self.stroke


def _layout_title(text: str, frame_width: int, frame_height: int) -> _TitleLayout:
    """The title in the fewest lines that keep it big: one line unless two or three make it at least a quarter bigger."""
    words = text.split()
    largest = max(12, int(frame_height * 0.26))
    chosen: Optional[_TitleLayout] = None
    for count in range(1, min(_MAX_LINES, len(words)) + 1):
        lines = _split_lines(words, count)
        probe = _TitleLayout(lines, largest)
        fit = min(1.0, frame_width * 0.9 / probe.width, frame_height * 0.34 / probe.height)
        layout = _TitleLayout(lines, max(12, int(largest * fit)))
        if chosen is None or layout.size > chosen.size * 1.25:
            chosen = layout
    assert chosen is not None
    return chosen


def _calm_position(frame: Image.Image, box_width: int, box_height: int, where: str) -> Tuple[int, int]:
    """The top left corner of the box of this size that has the fewest edges in it (the calmest part of the frame).

    ``where`` limits the search (``top``, ``bottom``, ``left``, ``right``). ``auto`` searches the whole frame with a slight
    preference for the upper part, where a title is usually read first.
    """
    width, height = frame.size
    gray = np.asarray(frame.convert("L")).astype(np.float32)
    energy = np.abs(np.diff(gray, axis=0))[:, :-1] + np.abs(np.diff(gray, axis=1))[:-1, :]
    integral = np.zeros((energy.shape[0] + 1, energy.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = energy.cumsum(axis=0).cumsum(axis=1)
    margin = max(4, int(min(width, height) * 0.035))
    x_low, x_high = margin, max(margin, width - box_width - margin)
    y_low, y_high = margin, max(margin, height - box_height - margin)
    if where == "top":
        y_high = max(y_low, min(y_high, int(height * 0.42) - box_height))
    elif where == "bottom":
        y_low = min(y_high, max(y_low, int(height * 0.58)))
    elif where == "left":
        x_high = max(x_low, min(x_high, int(width * 0.45) - box_width))
    elif where == "right":
        x_low = min(x_high, max(x_low, int(width * 0.55)))
    step = max(4, min(width, height) // 96)
    xs = np.arange(x_low, x_high + 1, step)
    ys = np.arange(y_low, y_high + 1, step)
    # The box sums, for every candidate corner at once, from the integral image. The box is kept inside the energy map.
    right = np.minimum(xs + box_width, energy.shape[1])
    bottom = np.minimum(ys + box_height, energy.shape[0])
    sums = (integral[bottom[:, None], right[None, :]] - integral[ys[:, None], right[None, :]]
            - integral[bottom[:, None], xs[None, :]] + integral[ys[:, None], xs[None, :]])
    area = np.maximum(1, (right - xs)[None, :] * (bottom - ys)[:, None])
    score = sums / area / max(float(energy.mean()), 1e-6)
    if where == "auto":
        score = score + 0.25 * (ys[:, None] / height)
    row, column = np.unravel_index(int(np.argmin(score)), score.shape)
    return int(xs[column]), int(ys[row])


def _draw_title(canvas: Image.Image, layout: _TitleLayout, origin: Tuple[int, int], look: Dict[str, Tuple[int, int, int]]) -> None:
    """Draw the title block with its top left corner at ``origin``: a drop shadow, a glow, the outline and the filled letters."""
    pad = int(layout.size * 0.4)
    left, top = origin[0] - pad, origin[1] - pad
    region_size = (layout.width + 2 * pad, layout.height + 2 * pad)

    def mask(stroke: int) -> Image.Image:
        image = Image.new("L", region_size, 0)
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(layout.lines):
            x = pad + layout.stroke + (layout.width - 2 * layout.stroke - layout.line_widths[index]) / 2
            baseline = pad + layout.stroke + layout.cap_height + index * layout.line_step
            for character, advance in zip(line, layout.advances[index]):
                draw.text((x, baseline), character, font=layout.font, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
                x += advance
        return image

    outline = mask(layout.stroke)
    fill = mask(0)
    glow = mask(layout.stroke + round(layout.size * 0.07)).filter(ImageFilter.GaussianBlur(layout.size * 0.09))
    shadow = ImageChops.offset(outline.filter(ImageFilter.GaussianBlur(layout.size * 0.035)), round(layout.size * 0.04), round(layout.size * 0.05))
    # The letters are a little darker toward the bottom.
    rows = np.linspace(1.0, 0.8, region_size[1], dtype=np.float32)[:, None, None]
    gradient = np.clip(np.array(look["fill"], dtype=np.float32)[None, None, :] * rows, 0, 255).astype(np.uint8)
    gradient = Image.fromarray(np.broadcast_to(gradient, (region_size[1], region_size[0], 3)).copy())
    region = canvas.crop((left, top, left + region_size[0], top + region_size[1])).convert("RGBA")
    for color, alpha_mask, opacity in ((look["glow"], glow, 0.9), ((0, 0, 0), shadow, 0.8)):
        layer = Image.new("RGBA", region_size, color + (0,))
        layer.putalpha(alpha_mask.point(lambda value, opacity=opacity: int(value * opacity)))
        region = Image.alpha_composite(region, layer)
    region = Image.alpha_composite(region, Image.merge("RGBA", (*Image.new("RGB", region_size, look["outline"]).split(), outline)))
    region = Image.alpha_composite(region, Image.merge("RGBA", (*gradient.split(), fill)))
    canvas.paste(region.convert("RGB"), (left, top))


def render_overlay(
    frame: Image.Image, title: str = "", color: str = OVERLAY_DEFAULT_COLOR, position: str = "auto", strength: Any = OVERLAY_DEFAULT_STRENGTH,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """The frame with the grade, the glow, the vignette and the title on top. Returns the picture and what was done."""
    frame = frame.convert("RGB")
    width, height = frame.size
    color = color if color in OVERLAY_COLORS else OVERLAY_DEFAULT_COLOR
    position = position if position in OVERLAY_POSITIONS else "auto"
    strength = overlay_strength(strength)
    result = grade_frame(frame, strength)
    text = re.sub(r"\s+", " ", str(title or "")).strip()[:_MAX_TITLE_CHARACTERS].upper()
    info: Dict[str, Any] = {"strength": strength, "color": color, "position": position, "title_box": None, "lines": 0, "font": ""}
    if text:
        layout = _layout_title(text, width, height)
        # The calmest place is looked for on the frame as it was, before any glow of the grade.
        x, y = _calm_position(frame, layout.width, layout.height, position)
        _draw_title(result, layout, (x, y), OVERLAY_COLORS[color])
        info.update({"title_box": {"x": x, "y": y, "width": layout.width, "height": layout.height}, "lines": len(layout.lines), "font": layout.font_name})
    return result, info


def _make_thumbnail_overlay(payload: Dict[str, Any]) -> Dict[str, Any]:
    """``POST thumbnail/overlay``: the chosen frame with the overlay on it, kept as ``<project>/thumbnail/thumbnail_NNNN.png``.

    ``source_path`` is the frame, ``title`` the text (optional: without one only the grade is applied), ``color`` one of
    ``OVERLAY_COLORS``, ``position`` one of ``OVERLAY_POSITIONS`` and ``strength`` 0 to 100 (how strong the grade, the glow and the
    vignette are; 0 leaves the frame alone).
    """
    from ..runner.video_files import _safe_project_subfolder
    from .thumbnail import THUMBNAIL_FOLDER, _clean_path, _next_thumbnail_path

    if not str(payload.get("project_folder") or "").strip().strip('"'):
        raise ValueError("Project folder is empty.")
    if not str(payload.get("source_path") or "").strip().strip('"'):
        raise ValueError("Choose the frame first: scrub the video to the moment you want and click Use this frame.")
    source = _clean_path(payload.get("source_path"))
    if not os.path.isfile(source):
        raise FileNotFoundError(f"The frame was not found: {source}")
    _project, folder = _safe_project_subfolder(payload.get("project_folder", ""), THUMBNAIL_FOLDER)
    with Image.open(source) as opened:
        frame = opened.convert("RGB")
    result, info = render_overlay(frame, payload.get("title", ""), str(payload.get("color") or ""), str(payload.get("position") or ""), payload.get("strength"))
    target = _next_thumbnail_path(folder)
    result.save(target, format="PNG")
    return {"image_path": target, "width": frame.width, "height": frame.height, **info}
