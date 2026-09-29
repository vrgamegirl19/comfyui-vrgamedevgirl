import os

import numpy as np
import torch


LUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LUTS")
SUPPORTED_LUT_EXTENSIONS = (".cube",)


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


NODE_CLASS_MAPPINGS = {

}

NODE_DISPLAY_NAME_MAPPINGS = {

}
