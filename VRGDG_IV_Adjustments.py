import os

import numpy as np
import torch


LUTS_DIR = os.path.join(os.path.dirname(__file__), "LUTS")
SUPPORTED_LUT_EXTENSIONS = (".cube",)
NAMED_COLORS = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "orange": "#ffa500",
    "purple": "#800080",
    "pink": "#ffc0cb",
    "teal": "#008080",
}


def _list_lut_files():
    if not os.path.isdir(LUTS_DIR):
        return ["No LUT files found"]

    files = [
        name
        for name in os.listdir(LUTS_DIR)
        if os.path.isfile(os.path.join(LUTS_DIR, name))
        and name.lower().endswith(SUPPORTED_LUT_EXTENSIONS)
    ]
    files.sort(key=str.lower)
    return files or ["No LUT files found"]


def _sanitize_filename_part(value):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "custom"


def _parse_hex_color(token):
    token = str(token or "").strip().lower()
    if token in NAMED_COLORS:
        token = NAMED_COLORS[token]
    if token.startswith("#"):
        token = token[1:]

    if len(token) == 3:
        token = "".join(ch * 2 for ch in token)

    if len(token) != 6 or any(ch not in "0123456789abcdef" for ch in token):
        raise ValueError(f"Invalid color '{token}'. Use hex like #ff8800 or a basic color name.")

    return np.array(
        [
            int(token[0:2], 16) / 255.0,
            int(token[2:4], 16) / 255.0,
            int(token[4:6], 16) / 255.0,
        ],
        dtype=np.float32,
    )


def _parse_color_list(colors_text):
    parts = [part.strip() for part in str(colors_text or "").split(",") if part.strip()]
    if not parts:
        raise ValueError("Provide one or more colors separated by commas.")
    return np.stack([_parse_hex_color(part) for part in parts], axis=0)


def _interpolate_palette(luma, palette):
    if palette.shape[0] == 1:
        target = np.empty(luma.shape + (3,), dtype=np.float32)
        target[...] = palette[0]
        return target

    positions = np.linspace(0.0, 1.0, palette.shape[0], dtype=np.float32)
    flat_luma = luma.reshape(-1)
    target = np.stack(
        [np.interp(flat_luma, positions, palette[:, channel]) for channel in range(3)],
        axis=-1,
    )
    return target.reshape(luma.shape + (3,)).astype(np.float32)


def _build_palette_lut(colors_text, lut_size):
    palette = _parse_color_list(colors_text)
    axis = np.linspace(0.0, 1.0, int(lut_size), dtype=np.float32)
    blue, green, red = np.meshgrid(axis, axis, axis, indexing="ij")
    source = np.stack([red, green, blue], axis=-1)

    luma = (0.2126 * source[..., 0]) + (0.7152 * source[..., 1]) + (0.0722 * source[..., 2])
    target = _interpolate_palette(luma, palette)

    target_luma = (0.2126 * target[..., 0]) + (0.7152 * target[..., 1]) + (0.0722 * target[..., 2])
    scale = luma / np.maximum(target_luma, 1e-6)
    target = np.clip(target * scale[..., None], 0.0, 1.0)

    source_chroma = source - luma[..., None]
    output = np.clip((target * 0.82) + ((target + source_chroma) * 0.18), 0.0, 1.0)
    return torch.from_numpy(output.astype(np.float32))


def _write_cube_file(lut_tensor, lut_path):
    size = int(lut_tensor.shape[0])
    lut_np = lut_tensor.detach().cpu().numpy()

    os.makedirs(os.path.dirname(lut_path), exist_ok=True)
    with open(lut_path, "w", encoding="utf-8") as handle:
        handle.write(f'TITLE "{os.path.basename(lut_path)}"\n')
        handle.write(f"LUT_3D_SIZE {size}\n")
        handle.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        handle.write("DOMAIN_MAX 1.0 1.0 1.0\n")

        for blue_index in range(size):
            for green_index in range(size):
                for red_index in range(size):
                    red, green, blue = lut_np[blue_index, green_index, red_index]
                    handle.write(f"{red:.6f} {green:.6f} {blue:.6f}\n")


def _next_available_lut_path(base_name):
    os.makedirs(LUTS_DIR, exist_ok=True)
    candidate = os.path.join(LUTS_DIR, f"{base_name}.cube")
    if not os.path.exists(candidate):
        return candidate

    index = 2
    while True:
        candidate = os.path.join(LUTS_DIR, f"{base_name}_{index}.cube")
        if not os.path.exists(candidate):
            return candidate
        index += 1


class VRGDG_LUTS:
    CATEGORY = "VRGDG/IV Adjustments"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_lut"

    _LUT_CACHE = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_name": (_list_lut_files(),),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, image, lut_name, device, strength):
        if lut_name == "No LUT files found":
            return f"missing|{device}|{strength}"

        folder_state = cls._get_luts_folder_state()
        lut_path = os.path.join(LUTS_DIR, lut_name)
        if not os.path.isfile(lut_path):
            return f"{folder_state}|missing|{lut_name}|{device}|{strength}"

        return f"{folder_state}|{lut_name}|{os.path.getmtime(lut_path)}|{device}|{strength}"

    @staticmethod
    def _resolve_device(requested_device, image):
        requested = str(requested_device or "auto").strip().lower()
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("VRGDG_LUTS: CUDA was selected, but CUDA is not available.")
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")

        if image.device.type != "cpu":
            return image.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _get_luts_folder_state():
        if not os.path.isdir(LUTS_DIR):
            return "missing"

        entries = []
        for name in _list_lut_files():
            if name == "No LUT files found":
                continue
            path = os.path.join(LUTS_DIR, name)
            try:
                entries.append(f"{name}:{os.path.getmtime(path)}:{os.path.getsize(path)}")
            except OSError:
                entries.append(f"{name}:missing")
        return "|".join(entries) if entries else "empty"

    @classmethod
    def _load_lut(cls, lut_name):
        if lut_name == "No LUT files found":
            raise ValueError("No LUT files were found in the LUTS folder.")

        lut_path = os.path.join(LUTS_DIR, lut_name)
        if not os.path.isfile(lut_path):
            raise FileNotFoundError(f"LUT file not found: {lut_path}")

        cache_key = (lut_path, os.path.getmtime(lut_path), os.path.getsize(lut_path))
        cached = cls._LUT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        lut_data = cls._parse_cube_file(lut_path)
        cls._LUT_CACHE = {cache_key: lut_data}
        return lut_data

    @staticmethod
    def _parse_cube_file(lut_path):
        size = None
        domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values = []

        with open(lut_path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                upper = line.upper()
                if upper.startswith("TITLE "):
                    continue
                if upper.startswith("LUT_1D_SIZE"):
                    raise ValueError(f"1D LUTs are not supported: {os.path.basename(lut_path)}")
                if upper.startswith("LUT_3D_SIZE"):
                    parts = line.split()
                    if len(parts) != 2:
                        raise ValueError(f"Invalid LUT_3D_SIZE line in {lut_path}")
                    size = int(parts[1])
                    continue
                if upper.startswith("DOMAIN_MIN"):
                    parts = line.split()
                    if len(parts) != 4:
                        raise ValueError(f"Invalid DOMAIN_MIN line in {lut_path}")
                    domain_min = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    continue
                if upper.startswith("DOMAIN_MAX"):
                    parts = line.split()
                    if len(parts) != 4:
                        raise ValueError(f"Invalid DOMAIN_MAX line in {lut_path}")
                    domain_max = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    continue

                parts = line.split()
                if len(parts) != 3:
                    continue
                values.extend(float(part) for part in parts)

        if size is None:
            raise ValueError(f"Missing LUT_3D_SIZE in {lut_path}")

        expected_values = size * size * size * 3
        if len(values) != expected_values:
            raise ValueError(
                f"Invalid LUT data length in {lut_path}. Expected {expected_values} floats, got {len(values)}."
            )

        # .cube 3D LUT data is typically stored with red changing fastest,
        # then green, then blue. In C-order reshape that means [blue, green, red, rgb].
        lut = np.asarray(values, dtype=np.float32).reshape(size, size, size, 3)
        lut = torch.from_numpy(lut)

        return {
            "size": size,
            "lut": lut,
            "domain_min": torch.from_numpy(domain_min),
            "domain_max": torch.from_numpy(domain_max),
        }

    @staticmethod
    def _expand_index(index, channels):
        return index.unsqueeze(-1).expand(*index.shape, channels)

    @classmethod
    def _apply_cube_lut(cls, image, lut_tensor, domain_min, domain_max):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("VRGDG_LUTS expects IMAGE input shaped like [batch, height, width, channels].")

        source = image[..., :3].to(dtype=torch.float32)

        domain_span = torch.clamp(domain_max - domain_min, min=1e-6)
        normalized = (source - domain_min) / domain_span
        normalized = torch.clamp(normalized, 0.0, 1.0)

        max_index = lut_tensor.shape[0] - 1
        coords = normalized * max_index

        r = coords[..., 0]
        g = coords[..., 1]
        b = coords[..., 2]

        r0 = torch.floor(r).long()
        g0 = torch.floor(g).long()
        b0 = torch.floor(b).long()

        r1 = torch.clamp(r0 + 1, max=max_index)
        g1 = torch.clamp(g0 + 1, max=max_index)
        b1 = torch.clamp(b0 + 1, max=max_index)

        fr = (r - r0.float()).unsqueeze(-1)
        fg = (g - g0.float()).unsqueeze(-1)
        fb = (b - b0.float()).unsqueeze(-1)

        c000 = lut_tensor[b0, g0, r0]
        c001 = lut_tensor[b1, g0, r0]
        c010 = lut_tensor[b0, g1, r0]
        c011 = lut_tensor[b1, g1, r0]
        c100 = lut_tensor[b0, g0, r1]
        c101 = lut_tensor[b1, g0, r1]
        c110 = lut_tensor[b0, g1, r1]
        c111 = lut_tensor[b1, g1, r1]

        c00 = c000 * (1.0 - fb) + c001 * fb
        c01 = c010 * (1.0 - fb) + c011 * fb
        c10 = c100 * (1.0 - fb) + c101 * fb
        c11 = c110 * (1.0 - fb) + c111 * fb

        c0 = c00 * (1.0 - fg) + c01 * fg
        c1 = c10 * (1.0 - fg) + c11 * fg

        output_rgb = c0 * (1.0 - fr) + c1 * fr
        output_rgb = torch.clamp(output_rgb, 0.0, 1.0)

        if image.shape[-1] == 3:
            return output_rgb.to(dtype=image.dtype)

        output = image.clone()
        output[..., :3] = output_rgb.to(dtype=image.dtype)
        return output

    def apply_lut(self, image, lut_name, device, strength):
        lut_data = self._load_lut(lut_name)
        target_device = self._resolve_device(device, image)

        working_image = image.to(device=target_device)
        lut_tensor = lut_data["lut"].to(device=target_device)
        domain_min = lut_data["domain_min"].to(device=target_device, dtype=working_image.dtype)
        domain_max = lut_data["domain_max"].to(device=target_device, dtype=working_image.dtype)
        output = self._apply_cube_lut(working_image, lut_tensor, domain_min, domain_max)

        blend = max(0.0, min(10.0, float(strength))) / 10.0
        if blend <= 0.0:
            output = working_image
        elif blend < 1.0:
            output = (working_image * (1.0 - blend)) + (output * blend)

        return (output.to(device=image.device),)


class VRGDG_MakeLUT:
    CATEGORY = "VRGDG/IV Adjustments"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "lut_name", "lut_path")
    FUNCTION = "create_and_apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": (
                    "STRING",
                    {
                        "default": "#0b1d51, #1f6aa5, #f3d27a",
                        "multiline": False,
                    },
                ),
                "name_suffix": ("STRING", {"default": "palette", "multiline": False}),
                "lut_size": ("INT", {"default": 33, "min": 8, "max": 128, "step": 1}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, image, colors, name_suffix, lut_size, device, strength):
        return f"{colors}|{name_suffix}|{lut_size}|{device}|{strength}"

    def create_and_apply(self, image, colors, name_suffix, lut_size, device, strength):
        lut_tensor = _build_palette_lut(colors, lut_size)

        color_slug = "_".join(_sanitize_filename_part(part) for part in str(colors).split(",") if part.strip())
        suffix_slug = _sanitize_filename_part(name_suffix)
        base_name = f"{color_slug}_{suffix_slug}" if suffix_slug else color_slug
        lut_path = _next_available_lut_path(base_name)
        _write_cube_file(lut_tensor, lut_path)

        lut_data = {
            "size": int(lut_tensor.shape[0]),
            "lut": lut_tensor,
            "domain_min": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            "domain_max": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        }

        target_device = VRGDG_LUTS._resolve_device(device, image)
        working_image = image.to(device=target_device)
        lut_on_device = lut_data["lut"].to(device=target_device)
        domain_min = lut_data["domain_min"].to(device=target_device, dtype=working_image.dtype)
        domain_max = lut_data["domain_max"].to(device=target_device, dtype=working_image.dtype)
        output = VRGDG_LUTS._apply_cube_lut(working_image, lut_on_device, domain_min, domain_max)

        blend = max(0.0, min(10.0, float(strength))) / 10.0
        if blend <= 0.0:
            output = working_image
        elif blend < 1.0:
            output = (working_image * (1.0 - blend)) + (output * blend)

        lut_name = os.path.basename(lut_path)
        return (output.to(device=image.device), lut_name, lut_path)


NODE_CLASS_MAPPINGS = {
    "VRGDG_LUTS": VRGDG_LUTS,
    "VRGDG_MakeLUT": VRGDG_MakeLUT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LUTS": "VRGDG_LUTS",
    "VRGDG_MakeLUT": "VRGDG_MakeLUT",
}
