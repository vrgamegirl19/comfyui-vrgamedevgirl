import torch
from typing import Dict, List, Optional, Tuple


def _ensure_compat(images: List[torch.Tensor]) -> None:
    if not images:
        return
    base = images[0]
    for img in images[1:]:
        if img.shape[1:] != base.shape[1:] or img.dtype != base.dtype:
            raise Exception("Selected images must have the same shape and dtype to combine.")


def _combine_images(images: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    _ensure_compat(images)
    return torch.cat(images, dim=0)


def _parse_spec(spec: str) -> List[int]:
    spec = (spec or "").strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "all":
        return [1, 2, 3, 4]

    indices: List[int] = []
    parts = [p.strip() for p in spec.replace(";", ",").split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(left.strip())
                end = int(right.strip())
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                if i not in indices:
                    indices.append(i)
        else:
            try:
                i = int(part)
            except ValueError:
                continue
            if i not in indices:
                indices.append(i)
    return indices


def _parse_map(map_text: str) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = {}
    if not map_text:
        return mapping

    lines = []
    for chunk in map_text.replace(";", "\n").splitlines():
        line = chunk.strip()
        if line:
            lines.append(line)

    for line in lines:
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        try:
            key = int(left.strip())
        except ValueError:
            continue
        mapping[key] = _parse_spec(right)
    return mapping


def _select_images(indices: List[int], images: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
    selected: List[torch.Tensor] = []
    for idx in indices:
        if idx < 1 or idx > 4:
            continue
        img = images[idx - 1]
        if img is None:
            continue
        selected.append(img)
    return selected


class VRGDG_ImageSwitch4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("STRING", {"default": "1", "multiline": False}),
            },
            "optional": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "select"
    CATEGORY = "VRGDG/Switch"
    DESCRIPTION = "Select one or more images by index string (e.g., 1,2,4). Returns nothing for none/0."

    def select(
        self,
        index: str,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
    ):
        images = [image1, image2, image3, image4]
        indices = _parse_spec(index)
        selected = _select_images(indices, images)
        output = _combine_images(selected)
        if output is None:
            return (None,)
        return (output,)

class VRGDG_ImageSwitchMultiDynamic:
    @classmethod
    def INPUT_TYPES(cls):
        max_inputs = 50
        optional = {f"image{i}": ("IMAGE", {}) for i in range(1, max_inputs + 1)}
        return {
            "required": {
                "index": ("STRING", {"default": "1", "multiline": False}),
                "image_count": ("INT", {"default": 4, "min": 1, "max": max_inputs, "step": 1}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "select"
    CATEGORY = "VRGDG/Switch"
    DESCRIPTION = "Dynamic image switch. Set image_count, click Refresh Inputs, then select image indices (e.g. 1,2,5 or all)."

    def select(self, index: str, image_count: int, **kwargs):
        count = max(1, min(50, int(image_count)))

        index_text = (index or "").strip().lower()
        if index_text in ("", "none", "0"):
            return (None,)

        if index_text == "all":
            indices = list(range(1, count + 1))
        else:
            indices = _parse_spec(index)

        selected: List[torch.Tensor] = []
        for idx in indices:
            if idx < 1 or idx > count:
                continue
            img = kwargs.get(f"image{idx}")
            if img is not None:
                selected.append(img)

        output = _combine_images(selected)
        if output is None:
            return (None,)
        return (output,)

class VRGDG_ImageIndexMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 0, "max": 999, "step": 1}),
                "map": (
                    "STRING",
                    {
                        "default": "1=1\n2=2\n10=1,4\n20=all\n6=none",
                        "multiline": True,
                    },
                ),
                "fallback": ("STRING", {"default": "same", "multiline": False}),
            },
            "optional": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "select"
    CATEGORY = "VRGDG/Switch"
    DESCRIPTION = "Select one or more images using a map. Supports none/all/1-4, ranges, and commas."

    def select(
        self,
        index: int,
        map: str,
        fallback: str,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
    ):
        images = [image1, image2, image3, image4]
        mapping = _parse_map(map)

        if index in mapping:
            indices = mapping[index]
        else:
            if (fallback or "").strip().lower() == "same":
                indices = [index]
            else:
                indices = _parse_spec(fallback)

        selected = _select_images(indices, images)
        output = _combine_images(selected)
        if output is None:
            return (None,)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_ImageSwitch4": VRGDG_ImageSwitch4,
    "VRGDG_ImageSwitchMultiDynamic": VRGDG_ImageSwitchMultiDynamic,
    "VRGDG_ImageIndexMap": VRGDG_ImageIndexMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ImageSwitch4": "VRGDG Image Switch (1-4)",
    "VRGDG_ImageSwitchMultiDynamic": "VRGDG Image Switch (Multi Dynamic)",
    "VRGDG_ImageIndexMap": "VRGDG Image Switch (Index Map)",
}
