import math

import numpy as np
import torch
from PIL import Image, ImageDraw


def _clamp_int(value, minimum, maximum):
    return max(minimum, min(maximum, int(value)))


def _parse_color(value, fallback):
    text = str(value or "").strip()
    if not text:
        text = fallback
    named = {
        "black": "#000000",
        "white": "#ffffff",
        "gray": "#808080",
        "grey": "#808080",
        "neutral_gray": "#b8b8b8",
        "neutral_grey": "#b8b8b8",
    }
    text = named.get(text.lower(), text)
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        text = fallback.lstrip("#")
    try:
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return _parse_color(fallback, "#000000")


def _image_tensor_to_pil_images(value, batch_mode):
    if value is None or not isinstance(value, torch.Tensor):
        return []

    tensor = value.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or int(tensor.shape[0]) <= 0:
        return []

    if batch_mode == "first_image_only":
        tensor = tensor[:1]

    images = []
    for item in tensor:
        array = item.numpy()
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] > 3:
            array = array[..., :3]
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(array, mode="RGB"))
    return images


def _pil_to_image_tensor(image):
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _resize_to_panel(image, width, height, fit_mode, fill_color):
    width = max(1, int(width))
    height = max(1, int(height))
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        return Image.new("RGB", (width, height), fill_color)

    scale_x = width / source_width
    scale_y = height / source_height
    scale = max(scale_x, scale_y) if fit_mode == "cover_crop" else min(scale_x, scale_y)
    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if fit_mode == "cover_crop":
        left = max(0, (new_width - width) // 2)
        top = max(0, (new_height - height) // 2)
        return resized.crop((left, top, left + width, top + height))

    panel = Image.new("RGB", (width, height), fill_color)
    panel.paste(resized, ((width - new_width) // 2, (height - new_height) // 2))
    return panel


def _rounded_mask(size, radius):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=max(0, int(radius)), fill=255)
    return mask


def _grid_rects(count, columns=None):
    if count <= 0:
        return []
    if not columns or columns <= 0:
        columns = int(math.ceil(math.sqrt(count * 16 / 9)))
    columns = max(1, min(count, int(columns)))
    rows = int(math.ceil(count / columns))
    rects = []
    for index in range(count):
        row = index // columns
        col = index % columns
        rects.append((col / columns, row / rows, 1 / columns, 1 / rows))
    return rects


def _image_aspect(image):
    width, height = image.size
    if width <= 0 or height <= 0:
        return 1.0
    return max(0.05, min(20.0, width / height))


def _row_partitions(count, rows):
    if rows <= 1:
        yield [count]
        return
    if rows >= count:
        yield [1] * count
        return

    def walk(remaining, row_count, prefix):
        if row_count == 1:
            if remaining >= 1:
                yield prefix + [remaining]
            return
        max_first = remaining - row_count + 1
        for first in range(1, max_first + 1):
            yield from walk(remaining - first, row_count - 1, prefix + [first])

    yield from walk(count, rows, [])


def _aspect_row_rects(images, canvas_width, canvas_height):
    count = len(images)
    if count <= 0:
        return []
    if count == 1:
        return [(0.0, 0.0, 1.0, 1.0)]

    target_aspect = max(0.05, canvas_width / max(1, canvas_height))
    aspects = [_image_aspect(image) for image in images]
    max_rows = min(count, 4)
    best = None

    for rows in range(1, max_rows + 1):
        for partition in _row_partitions(count, rows):
            start = 0
            raw_heights = []
            row_sums = []
            for row_count in partition:
                row_sum = sum(aspects[start : start + row_count])
                row_sums.append(row_sum)
                raw_heights.append(target_aspect / max(0.05, row_sum))
                start += row_count

            raw_total = sum(raw_heights)
            if raw_total > 1.02:
                # This partition would need horizontal padding after scaling down.
                score = (raw_total - 1.0) * 10.0 + rows * 0.05
            else:
                # Prefer using most of the canvas while leaving no internal image padding.
                score = (1.0 - raw_total) + rows * 0.035

            height_spread = max(raw_heights) - min(raw_heights)
            score += height_spread * 0.08

            if best is None or score < best[0]:
                best = (score, partition, raw_heights, row_sums, raw_total)

    _, partition, raw_heights, row_sums, raw_total = best
    row_gap = 0.0
    y = max(0.0, (1.0 - raw_total) / 2.0) if raw_total <= 1.0 else 0.0
    if raw_total < 0.98 and len(partition) > 1:
        row_gap = (1.0 - raw_total) / (len(partition) + 1)
        y = row_gap

    rects = []
    index = 0
    for row_index, row_count in enumerate(partition):
        row_height = raw_heights[row_index]
        row_sum = row_sums[row_index]
        if raw_total > 1.0:
            row_height = row_height / raw_total
        row_width = row_height * row_sum / target_aspect
        x = max(0.0, (1.0 - row_width) / 2.0)

        for _ in range(row_count):
            panel_width = row_height * aspects[index] / target_aspect
            rects.append((x, y, panel_width, row_height))
            x += panel_width
            index += 1
        y += row_height + row_gap

    return rects


def _layout_rects(preset, count, columns):
    if count <= 0:
        return []

    if preset == "horizontal_strip":
        return [(index / count, 0.0, 1 / count, 1.0) for index in range(count)]

    if preset == "vertical_strip":
        return [(0.0, index / count, 1.0, 1 / count) for index in range(count)]

    if preset == "wide_bottom" and count >= 3:
        bottom_count = 1
        top_count = count - bottom_count
        top_rows = 2 if top_count > 4 else 1
        top_height = 0.68 if top_rows == 2 else 0.56
        top_rects = _grid_rects(top_count, columns if columns > 0 else None)
        rects = [
            (x, y * top_height, w, h * top_height)
            for x, y, w, h in top_rects
        ]
        rects.append((0.0, top_height, 1.0, 1.0 - top_height))
        return rects[:count]

    if preset == "six_panel_story" and count >= 6:
        if count > 7:
            return _layout_rects("three_row_reference", count, columns)
        rects = [
            (0.0, 0.0, 0.235, 0.52),
            (0.235, 0.0, 0.385, 0.52),
            (0.62, 0.0, 0.38, 0.52),
            (0.0, 0.52, 0.37, 0.23),
            (0.37, 0.52, 0.63, 0.23),
            (0.0, 0.75, 0.37, 0.25),
            (0.37, 0.75, 0.63, 0.25),
        ]
        return rects[:count]

    if preset == "three_row_reference" and count >= 5:
        if count <= 6:
            top = count // 2
            mid = count - top - 1
            rects = []
            for index in range(top):
                rects.append((index / top, 0.0, 1 / top, 0.42))
            for index in range(mid):
                rects.append((index / mid, 0.42, 1 / mid, 0.28))
            rects.append((0.0, 0.70, 1.0, 0.30))
            return rects

        top = min(3, count)
        mid = min(3, count - top)
        bottom = count - top - mid
        rects = []
        for index in range(top):
            rects.append((index / top, 0.0, 1 / top, 0.40))
        for index in range(mid):
            rects.append((index / mid, 0.40, 1 / mid, 0.28))
        for index in range(bottom):
            rects.append((index / bottom, 0.68, 1 / bottom, 0.32))
        return rects

    if preset == "auto_ltx":
        if 6 <= count <= 7:
            return _layout_rects("six_panel_story", count, columns)
        if count >= 5:
            return _layout_rects("three_row_reference", count, columns)

    return _grid_rects(count, columns if columns > 0 else None)


class VRGDG_LTXICIngredientsGrid:
    MAX_IMAGES = 24
    LAYOUTS = [
        "auto_ltx",
        "aspect_rows",
        "six_panel_story",
        "three_row_reference",
        "wide_bottom",
        "uniform_grid",
        "horizontal_strip",
        "vertical_strip",
    ]
    FIT_MODES = ["contain_pad", "cover_crop"]
    BATCH_MODES = ["first_image_only", "all_images"]

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"image{i}": ("IMAGE", {"forceInput": True, "tooltip": f"Ingredient image {i}."})
            for i in range(1, cls.MAX_IMAGES + 1)
        }
        return {
            "required": {
                "image_count": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": cls.MAX_IMAGES,
                        "step": 1,
                        "tooltip": "How many dynamic image inputs to show and compose.",
                    },
                ),
                "layout": (cls.LAYOUTS, {"default": "auto_ltx"}),
                "output_width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "output_height": ("INT", {"default": 448, "min": 64, "max": 8192, "step": 8}),
                "columns": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 12,
                        "step": 1,
                        "tooltip": "Uniform grid columns. Use 0 for auto. Some presets ignore this.",
                    },
                ),
                "gutter": ("INT", {"default": 4, "min": 0, "max": 128, "step": 1}),
                "outer_padding": ("INT", {"default": 4, "min": 0, "max": 128, "step": 1}),
                "corner_radius": ("INT", {"default": 3, "min": 0, "max": 96, "step": 1}),
                "fit_mode": (cls.FIT_MODES, {"default": "contain_pad"}),
                "batch_mode": (cls.BATCH_MODES, {"default": "first_image_only"}),
                "background_color": ("STRING", {"default": "#000000", "multiline": False}),
                "cell_background_color": ("STRING", {"default": "#b8b8b8", "multiline": False}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reference_sheet",)
    FUNCTION = "build"
    CATEGORY = "VRGDG/LTX"
    DESCRIPTION = "Builds an LTX IC-LoRA Ingredients-style reference sheet from dynamic image inputs."

    def build(
        self,
        image_count,
        layout,
        output_width,
        output_height,
        columns,
        gutter,
        outer_padding,
        corner_radius,
        fit_mode,
        batch_mode,
        background_color,
        cell_background_color,
        **kwargs,
    ):
        count = _clamp_int(image_count, 1, self.MAX_IMAGES)
        images = []
        for index in range(1, count + 1):
            images.extend(_image_tensor_to_pil_images(kwargs.get(f"image{index}"), batch_mode))

        if not images:
            raise ValueError("VRGDG LTX IC Ingredients Grid needs at least one connected image input.")

        width = _clamp_int(output_width, 64, 8192)
        height = _clamp_int(output_height, 64, 8192)
        gutter = _clamp_int(gutter, 0, 128)
        padding = _clamp_int(outer_padding, 0, 128)
        radius = _clamp_int(corner_radius, 0, 96)
        columns = _clamp_int(columns, 0, 12)
        background = _parse_color(background_color, "#000000")
        cell_background = _parse_color(cell_background_color, "#b8b8b8")

        if layout == "aspect_rows":
            rects = _aspect_row_rects(images, width, height)
        else:
            rects = _layout_rects(layout, len(images), columns)
        canvas = Image.new("RGB", (width, height), background)
        usable_width = max(1, width - 2 * padding)
        usable_height = max(1, height - 2 * padding)

        for image, rect in zip(images, rects):
            x, y, w, h = rect
            left = padding + int(round(x * usable_width))
            top = padding + int(round(y * usable_height))
            right = padding + int(round((x + w) * usable_width))
            bottom = padding + int(round((y + h) * usable_height))

            inset = gutter // 2
            left += inset
            top += inset
            right -= inset
            bottom -= inset
            panel_width = max(1, right - left)
            panel_height = max(1, bottom - top)

            panel = _resize_to_panel(image.convert("RGB"), panel_width, panel_height, fit_mode, cell_background)
            if radius > 0:
                mask = _rounded_mask((panel_width, panel_height), min(radius, panel_width // 2, panel_height // 2))
                canvas.paste(panel, (left, top), mask)
            else:
                canvas.paste(panel, (left, top))

        return (_pil_to_image_tensor(canvas),)


NODE_CLASS_MAPPINGS = {
    "VRGDG_LTXICIngredientsGrid": VRGDG_LTXICIngredientsGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LTXICIngredientsGrid": "VRGDG LTX IC Ingredients Grid",
}
