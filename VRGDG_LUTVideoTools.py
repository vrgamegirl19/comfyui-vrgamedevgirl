import os
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from aiohttp import web

from .VRGDG_IV_Adjustments import LUTS_DIR, VRGDG_LUTS

try:
    import folder_paths
except Exception:
    folder_paths = None


_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
_LEGACY_ADJUST_PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets", "adjust")


_VIDEO_CODEC_CANDIDATES = [
    {"fourcc": "avc1", "label": "H.264 avc1", "browser_friendly": True},
    {"fourcc": "H264", "label": "H.264 H264", "browser_friendly": True},
    {"fourcc": "X264", "label": "H.264 X264", "browser_friendly": True},
    {"fourcc": "mp4v", "label": "MPEG-4 mp4v", "browser_friendly": False},
]


def _safe_lut_path(lut_name):
    name = os.path.basename(str(lut_name or "").strip().strip('"'))
    if not name:
        raise ValueError("LUT name is required.")
    path = os.path.abspath(os.path.join(LUTS_DIR, name))
    lut_root = os.path.abspath(LUTS_DIR)
    if os.path.commonpath([lut_root, path]) != lut_root:
        raise ValueError("Invalid LUT path.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LUT file was not found: {name}")
    if os.path.splitext(path)[1].lower() != ".cube":
        raise ValueError("Only .cube LUT files are supported.")
    return path


def _safe_examples_path(example_name):
    name = os.path.basename(str(example_name or "").strip().strip('"'))
    if not name:
        raise ValueError("Example image name is required.")
    examples_dir = os.path.join(LUTS_DIR, "examples")
    path = os.path.abspath(os.path.join(examples_dir, name))
    examples_root = os.path.abspath(examples_dir)
    if os.path.commonpath([examples_root, path]) != examples_root:
        raise ValueError("Invalid LUT example path.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LUT example image was not found: {name}")
    return path


def _safe_adjust_preset_name(name):
    raw = str(name or "").strip().strip('"')
    if not raw:
        raise ValueError("Preset name is required.")
    stem = os.path.splitext(os.path.basename(raw))[0]
    safe = "".join(char if char.isalnum() or char in ("-", "_", " ") else "_" for char in stem).strip(" ._")
    safe = "_".join(safe.split())
    if not safe:
        raise ValueError("Preset name is invalid.")
    return safe[:80]


def _adjust_presets_dir():
    try:
        if folder_paths is not None:
            output_dir = os.path.abspath(folder_paths.get_output_directory())
            return os.path.join(output_dir, "VRGDG_AdjustPresets")
    except Exception:
        pass
    return _LEGACY_ADJUST_PRESETS_DIR


def _safe_adjust_preset_path(name):
    safe = _safe_adjust_preset_name(name)
    presets_dir = _adjust_presets_dir()
    os.makedirs(presets_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(presets_dir, f"{safe}.json"))
    root = os.path.abspath(presets_dir)
    if os.path.commonpath([root, path]) != root:
        raise ValueError("Invalid preset path.")
    return path


def _resolve_media_path(path, label):
    resolved = os.path.abspath(str(path or "").strip().strip('"'))
    if not resolved:
        raise ValueError(f"{label} is required.")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} was not found: {resolved}")
    return resolved


def _default_output_path(input_path, lut_name):
    folder = os.path.dirname(input_path)
    stem, ext = os.path.splitext(os.path.basename(input_path))
    lut_stem = Path(lut_name).stem
    safe_lut = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in lut_stem).strip("_")
    return os.path.join(folder, f"{stem}_lut_{safe_lut}{ext}")


def _default_effect_output_path(input_path, effect_name):
    folder = os.path.dirname(input_path)
    stem, ext = os.path.splitext(os.path.basename(input_path))
    safe_effect = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(effect_name or "effect")).strip("_")
    return os.path.join(folder, f"{stem}_{safe_effect}{ext}")


def _preview_root(project_folder=""):
    project_folder = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if project_folder:
        root = os.path.join(project_folder, "_tmp", "lut_previews")
    else:
        root = os.path.join(tempfile.gettempdir(), "vrgdg_lut_previews")
    os.makedirs(root, exist_ok=True)
    return root


def _safe_preview_path(path, project_folder=""):
    resolved = os.path.abspath(str(path or "").strip().strip('"'))
    root = os.path.abspath(_preview_root(project_folder))
    if not resolved or os.path.commonpath([root, resolved]) != root:
        raise ValueError("Invalid LUT preview path.")
    return resolved


def _example_key(name):
    return "".join(char.lower() for char in str(name or "") if char.isalnum())


def _resolve_device(requested_device):
    import torch

    requested = str(requested_device or "auto").strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _image_to_tensor(image):
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _tensor_to_image(tensor):
    import numpy as np
    from PIL import Image

    array = tensor.squeeze(0).detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _apply_lut_tensor(image_tensor, lut_name, strength, device):
    lut_data = VRGDG_LUTS._load_lut(lut_name)
    source = image_tensor.to(device=device)
    lut_tensor = lut_data["lut"].to(device=device)
    domain_min = lut_data["domain_min"].to(device=device, dtype=source.dtype)
    domain_max = lut_data["domain_max"].to(device=device, dtype=source.dtype)
    output = VRGDG_LUTS._apply_cube_lut(source, lut_tensor, domain_min, domain_max)

    blend = max(0.0, min(10.0, float(strength))) / 10.0
    if blend <= 0.0:
        return source
    if blend < 1.0:
        return (source * (1.0 - blend)) + (output * blend)
    return output


def list_luts():
    examples_dir = os.path.join(LUTS_DIR, "examples")
    items = []
    if not os.path.isdir(LUTS_DIR):
        return {"luts": items, "luts_dir": LUTS_DIR, "examples_dir": examples_dir}

    example_lookup = {}
    if os.path.isdir(examples_dir):
        for name in os.listdir(examples_dir):
            path = os.path.join(examples_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in _SUPPORTED_IMAGE_EXTENSIONS:
                example_lookup[os.path.splitext(name)[0].lower()] = name
                example_lookup[_example_key(os.path.splitext(name)[0])] = name

    for name in sorted(os.listdir(LUTS_DIR), key=lambda value: value.lower()):
        path = os.path.join(LUTS_DIR, name)
        if not os.path.isfile(path) or os.path.splitext(name)[1].lower() != ".cube":
            continue
        stem = os.path.splitext(name)[0]
        example_name = example_lookup.get(stem.lower(), "") or example_lookup.get(_example_key(stem), "")
        items.append(
            {
                "name": name,
                "label": stem.replace("_", " "),
                "path": path,
                "example_name": example_name,
                "example_url": f"/vrgdg/music_builder/luts/example?name={quote(example_name)}" if example_name else "",
                "size": os.path.getsize(path),
                "modified": os.path.getmtime(path),
            }
        )
    return {"luts": items, "luts_dir": LUTS_DIR, "examples_dir": examples_dir}


def apply_lut_to_image(input_path, lut_name, output_path="", strength=10.0, device="auto", replace_source=False):
    from PIL import Image

    input_path = _resolve_media_path(input_path, "Input image")
    if os.path.splitext(input_path)[1].lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Input image type is not supported.")
    lut_path = _safe_lut_path(lut_name)
    lut_name = os.path.basename(lut_path)
    target_device = _resolve_device(device)
    output_path = os.path.abspath(str(output_path or "").strip().strip('"') or _default_output_path(input_path, lut_name))
    if replace_source:
        output_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output = output_path
    if replace_source:
        fd, tmp_output = tempfile.mkstemp(prefix="vrgdg_lut_", suffix=os.path.splitext(input_path)[1], dir=os.path.dirname(input_path))
        os.close(fd)

    started = time.perf_counter()
    source = Image.open(input_path).convert("RGB")
    tensor = _image_to_tensor(source)
    output = _apply_lut_tensor(tensor, lut_name, strength, target_device)
    _tensor_to_image(output).save(tmp_output)

    if replace_source:
        os.replace(tmp_output, output_path)

    elapsed = time.perf_counter() - started
    return {
        "input": input_path,
        "output": output_path,
        "lut": lut_name,
        "device": str(target_device),
        "strength": float(strength),
        "replace_source": bool(replace_source),
        "elapsed_seconds": elapsed,
    }


def _apply_film_grain_tensor(image_tensor, grain_intensity=0.04, saturation_mix=0.5, device="cpu", seed=None):
    import torch

    intensity = max(0.0, min(1.0, float(grain_intensity)))
    saturation = max(0.0, min(1.0, float(saturation_mix)))
    source = image_tensor.to(device=device)
    generator = None
    if seed not in (None, ""):
        generator = torch.Generator(device=source.device)
        generator.manual_seed(int(seed))
    grain = torch.randn(source.shape, dtype=source.dtype, device=source.device, generator=generator)
    grain[:, :, :, 0] *= 2.0
    grain[:, :, :, 2] *= 3.0
    gray = grain[:, :, :, 1].unsqueeze(3).repeat(1, 1, 1, 3)
    grain = saturation * grain + (1.0 - saturation) * gray
    return (source + grain * intensity).clamp(0.0, 1.0)


def _normalize_adjust_settings(settings=None):
    settings = settings if isinstance(settings, dict) else {}
    fields = {
        "temperature": (-100.0, 100.0),
        "tint": (-100.0, 100.0),
        "saturation": (-100.0, 100.0),
        "exposure": (-100.0, 100.0),
        "contrast": (-100.0, 100.0),
        "highlights": (-100.0, 100.0),
        "shadows": (-100.0, 100.0),
        "whites": (-100.0, 100.0),
        "blacks": (-100.0, 100.0),
        "sharpen": (0.0, 100.0),
        "clarity": (-100.0, 100.0),
        "vignette": (0.0, 100.0),
        "fade": (0.0, 100.0),
    }
    normalized = {"enabled": settings.get("enabled", True) is not False}
    for key, (minimum, maximum) in fields.items():
        try:
            value = float(settings.get(key, 0.0))
        except Exception:
            value = 0.0
        normalized[key] = max(minimum, min(maximum, value))
    return normalized


def _apply_adjust_tensor(image_tensor, settings=None, device="cpu"):
    import torch
    import torch.nn.functional as F

    adjust = _normalize_adjust_settings(settings)
    source = image_tensor.to(device=device).clamp(0.0, 1.0)
    if not adjust["enabled"]:
        return source

    out = source
    out = out + torch.tensor(
        [
            adjust["temperature"] / 400.0 - adjust["tint"] / 900.0,
            adjust["tint"] / 450.0,
            -adjust["temperature"] / 400.0 - adjust["tint"] / 900.0,
        ],
        dtype=out.dtype,
        device=out.device,
    ).view(1, 1, 1, 3)

    exposure = 2.0 ** (adjust["exposure"] / 100.0)
    out = out * exposure
    contrast = 1.0 + (adjust["contrast"] / 100.0)
    out = (out - 0.5) * contrast + 0.5

    luma = (out[..., 0:1] * 0.2126) + (out[..., 1:2] * 0.7152) + (out[..., 2:3] * 0.0722)
    gray = luma.repeat(1, 1, 1, 3)
    saturation = 1.0 + (adjust["saturation"] / 100.0)
    out = gray + (out - gray) * saturation

    luma = (out[..., 0:1] * 0.2126) + (out[..., 1:2] * 0.7152) + (out[..., 2:3] * 0.0722)
    highlight_mask = torch.clamp((luma - 0.55) / 0.45, 0.0, 1.0)
    shadow_mask = torch.clamp((0.45 - luma) / 0.45, 0.0, 1.0)
    out = out + highlight_mask * (adjust["highlights"] / 220.0)
    out = out + shadow_mask * (adjust["shadows"] / 220.0)
    out = out + torch.clamp((luma - 0.75) / 0.25, 0.0, 1.0) * (adjust["whites"] / 240.0)
    out = out + torch.clamp((0.25 - luma) / 0.25, 0.0, 1.0) * (adjust["blacks"] / 240.0)

    clarity = adjust["clarity"] / 100.0
    sharpen = adjust["sharpen"] / 100.0
    if abs(clarity) > 0.001 or sharpen > 0.001:
        nchw = out.permute(0, 3, 1, 2)
        height = int(nchw.shape[2])
        width = int(nchw.shape[3])

        def blur_nchw(source, target_kernel):
            kernel = min(int(target_kernel), height if height % 2 else height - 1, width if width % 2 else width - 1)
            if kernel < 3:
                return source
            padding = kernel // 2
            return F.avg_pool2d(F.pad(source, (padding, padding, padding, padding), mode="reflect"), kernel_size=kernel, stride=1)

        if abs(clarity) > 0.001:
            medium_blur = blur_nchw(nchw, 9)
            medium_detail = nchw - medium_blur
            luma_nchw = (
                nchw[:, 0:1, :, :] * 0.2126
                + nchw[:, 1:2, :, :] * 0.7152
                + nchw[:, 2:3, :, :] * 0.0722
            )
            midtone_mask = 1.0 - torch.clamp(torch.abs(luma_nchw - 0.5) / 0.5, 0.0, 1.0)
            nchw = nchw + medium_detail * clarity * 1.55 * (0.35 + midtone_mask * 0.65)

        if sharpen > 0.001:
            fine_blur = F.avg_pool2d(F.pad(nchw, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
            fine_detail = nchw - fine_blur
            nchw = nchw + fine_detail * sharpen * 5.0

        out = nchw.permute(0, 2, 3, 1)

    fade = adjust["fade"] / 100.0
    if fade > 0.0:
        out = out * (1.0 - fade * 0.35) + fade * 0.18

    vignette = adjust["vignette"] / 100.0
    if vignette > 0.0:
        height = out.shape[1]
        width = out.shape[2]
        yy = torch.linspace(-1.0, 1.0, height, dtype=out.dtype, device=out.device).view(1, height, 1, 1)
        xx = torch.linspace(-1.0, 1.0, width, dtype=out.dtype, device=out.device).view(1, 1, width, 1)
        distance = torch.sqrt((xx * xx) + (yy * yy))
        mask = 1.0 - torch.clamp((distance - 0.35) / 1.05, 0.0, 1.0) * vignette * 0.75
        out = out * mask

    return out.clamp(0.0, 1.0)


def apply_adjust_to_image(input_path, output_path="", settings=None, device="auto", replace_source=False):
    from PIL import Image

    input_path = _resolve_media_path(input_path, "Input image")
    if os.path.splitext(input_path)[1].lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Input image type is not supported.")
    target_device = _resolve_device(device)
    output_path = os.path.abspath(str(output_path or "").strip().strip('"') or _default_effect_output_path(input_path, "adjust"))
    if replace_source:
        output_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output = output_path
    if replace_source:
        fd, tmp_output = tempfile.mkstemp(prefix="vrgdg_adjust_", suffix=os.path.splitext(input_path)[1], dir=os.path.dirname(input_path))
        os.close(fd)

    started = time.perf_counter()
    source = Image.open(input_path).convert("RGB")
    tensor = _image_to_tensor(source)
    output = _apply_adjust_tensor(tensor, settings or {}, target_device)
    _tensor_to_image(output).save(tmp_output)

    if replace_source:
        os.replace(tmp_output, output_path)

    elapsed = time.perf_counter() - started
    return {
        "input": input_path,
        "output": output_path,
        "effect": "adjust",
        "device": str(target_device),
        "settings": _normalize_adjust_settings(settings),
        "replace_source": bool(replace_source),
        "elapsed_seconds": elapsed,
    }


def preview_adjust_on_media(input_path, media_type="", settings=None, device="auto", scene_id="", project_folder=""):
    import cv2
    from PIL import Image

    input_path = _resolve_media_path(input_path, "Input media")
    ext = os.path.splitext(input_path)[1].lower()
    requested_type = str(media_type or "").strip().lower()
    if not requested_type:
        requested_type = "video" if ext in _SUPPORTED_VIDEO_EXTENSIONS else "image"
    if requested_type not in {"image", "video"}:
        raise ValueError("Preview media type must be image or video.")
    if requested_type == "image" and ext not in _SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Input image type is not supported.")
    if requested_type == "video" and ext not in _SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("Input video type is not supported.")

    root = _preview_root(project_folder)
    source_stem = Path(input_path).stem
    safe_scene = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(scene_id or "scene")).strip("_") or "scene"
    safe_source = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_stem).strip("_") or "media"
    stamp = int(time.time() * 1000)
    output_path = os.path.join(root, f"{safe_scene}_{safe_source}_adjust_{stamp}.jpg")
    frame_path = ""
    try:
        if requested_type == "video":
            cap = cv2.VideoCapture(input_path)
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open input video: {input_path}")
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                raise RuntimeError("Could not read the first frame from the input video.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_path = os.path.join(root, f"{safe_scene}_{safe_source}_first_frame_{stamp}.jpg")
            Image.fromarray(frame).save(frame_path, quality=92)
            source_path = frame_path
        else:
            source_path = input_path
        result = apply_adjust_to_image(
            input_path=source_path,
            output_path=output_path,
            settings=settings or {},
            device=device,
            replace_source=False,
        )
        return {
            **result,
            "preview_path": result["output"],
            "source_media": input_path,
            "source_type": requested_type,
        }
    finally:
        if frame_path:
            try:
                os.remove(frame_path)
            except OSError:
                pass


def apply_film_grain_to_image(
    input_path,
    output_path="",
    grain_intensity=0.04,
    saturation_mix=0.5,
    device="auto",
    replace_source=False,
    seed=None,
):
    from PIL import Image

    input_path = _resolve_media_path(input_path, "Input image")
    if os.path.splitext(input_path)[1].lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Input image type is not supported.")
    target_device = _resolve_device(device)
    output_path = os.path.abspath(str(output_path or "").strip().strip('"') or _default_effect_output_path(input_path, "film_grain"))
    if replace_source:
        output_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output = output_path
    if replace_source:
        fd, tmp_output = tempfile.mkstemp(prefix="vrgdg_grain_", suffix=os.path.splitext(input_path)[1], dir=os.path.dirname(input_path))
        os.close(fd)

    started = time.perf_counter()
    source = Image.open(input_path).convert("RGB")
    tensor = _image_to_tensor(source)
    output = _apply_film_grain_tensor(tensor, grain_intensity, saturation_mix, target_device, seed=seed)
    _tensor_to_image(output).save(tmp_output)

    if replace_source:
        os.replace(tmp_output, output_path)

    elapsed = time.perf_counter() - started
    return {
        "input": input_path,
        "output": output_path,
        "effect": "film_grain",
        "device": str(target_device),
        "grain_intensity": float(grain_intensity),
        "saturation_mix": float(saturation_mix),
        "replace_source": bool(replace_source),
        "elapsed_seconds": elapsed,
    }


def preview_film_grain_on_media(
    input_path,
    media_type="",
    grain_intensity=0.04,
    saturation_mix=0.5,
    device="auto",
    scene_id="",
    project_folder="",
    seed=None,
):
    import cv2
    from PIL import Image

    input_path = _resolve_media_path(input_path, "Input media")
    ext = os.path.splitext(input_path)[1].lower()
    requested_type = str(media_type or "").strip().lower()
    if not requested_type:
        requested_type = "video" if ext in _SUPPORTED_VIDEO_EXTENSIONS else "image"
    if requested_type not in {"image", "video"}:
        raise ValueError("Preview media type must be image or video.")
    if requested_type == "image" and ext not in _SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Input image type is not supported.")
    if requested_type == "video" and ext not in _SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("Input video type is not supported.")

    root = _preview_root(project_folder)
    source_stem = Path(input_path).stem
    safe_scene = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(scene_id or "scene")).strip("_") or "scene"
    safe_source = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_stem).strip("_") or "media"
    stamp = int(time.time() * 1000)
    output_path = os.path.join(root, f"{safe_scene}_{safe_source}_film_grain_{stamp}.jpg")
    frame_path = ""
    try:
        if requested_type == "video":
            cap = cv2.VideoCapture(input_path)
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open input video: {input_path}")
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                raise RuntimeError("Could not read the first frame from the input video.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_path = os.path.join(root, f"{safe_scene}_{safe_source}_first_frame_{stamp}.jpg")
            Image.fromarray(frame).save(frame_path, quality=92)
            source_path = frame_path
        else:
            source_path = input_path
        result = apply_film_grain_to_image(
            input_path=source_path,
            output_path=output_path,
            grain_intensity=grain_intensity,
            saturation_mix=saturation_mix,
            device=device,
            replace_source=False,
            seed=seed,
        )
        return {
            **result,
            "preview_path": result["output"],
            "source_media": input_path,
            "source_type": requested_type,
        }
    finally:
        if frame_path:
            try:
                os.remove(frame_path)
            except OSError:
                pass


def preview_lut_on_media(input_path, lut_name, media_type="", strength=10.0, device="auto", scene_id="", project_folder=""):
    import cv2
    from PIL import Image

    input_path = _resolve_media_path(input_path, "Input media")
    ext = os.path.splitext(input_path)[1].lower()
    requested_type = str(media_type or "").strip().lower()
    if not requested_type:
        requested_type = "video" if ext in _SUPPORTED_VIDEO_EXTENSIONS else "image"
    if requested_type not in {"image", "video"}:
        raise ValueError("Preview media type must be image or video.")
    if requested_type == "image" and ext not in _SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Input image type is not supported.")
    if requested_type == "video" and ext not in _SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("Input video type is not supported.")

    lut_path = _safe_lut_path(lut_name)
    lut_name = os.path.basename(lut_path)
    root = _preview_root(project_folder)
    source_stem = Path(input_path).stem
    lut_stem = Path(lut_name).stem
    safe_scene = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(scene_id or "scene")).strip("_") or "scene"
    safe_source = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_stem).strip("_") or "media"
    safe_lut = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in lut_stem).strip("_") or "lut"
    stamp = int(time.time() * 1000)
    output_path = os.path.join(root, f"{safe_scene}_{safe_source}_{safe_lut}_{stamp}.jpg")
    frame_path = ""
    try:
        if requested_type == "video":
            cap = cv2.VideoCapture(input_path)
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open input video: {input_path}")
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                raise RuntimeError("Could not read the first frame from the input video.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_path = os.path.join(root, f"{safe_scene}_{safe_source}_first_frame_{stamp}.jpg")
            Image.fromarray(frame).save(frame_path, quality=92)
            source_path = frame_path
        else:
            source_path = input_path
        result = apply_lut_to_image(
            input_path=source_path,
            lut_name=lut_name,
            output_path=output_path,
            strength=strength,
            device=device,
            replace_source=False,
        )
        return {
            **result,
            "preview_path": result["output"],
            "source_media": input_path,
            "source_type": requested_type,
        }
    finally:
        if frame_path:
            try:
                os.remove(frame_path)
            except OSError:
                pass


def list_adjust_presets():
    presets_dir = _adjust_presets_dir()
    os.makedirs(presets_dir, exist_ok=True)
    presets = []
    roots = [presets_dir]
    legacy_dir = os.path.abspath(_LEGACY_ADJUST_PRESETS_DIR)
    if os.path.isdir(legacy_dir) and os.path.abspath(presets_dir) != legacy_dir:
        roots.append(legacy_dir)
    seen = set()
    for root in roots:
        for name in sorted(os.listdir(root), key=lambda value: value.lower()):
            if os.path.splitext(name)[1].lower() != ".json":
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            preset_key = Path(name).stem
            if preset_key.lower() in seen:
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                settings = _normalize_adjust_settings(data.get("adjust") or data.get("settings") or {})
                label = str(data.get("name") or preset_key).strip() or preset_key
                presets.append({
                    "name": preset_key,
                    "label": label,
                    "settings": settings,
                    "path": path,
                    "modified": os.path.getmtime(path),
                    "legacy": os.path.abspath(root) == legacy_dir,
                })
                seen.add(preset_key.lower())
            except Exception:
                continue
    return {"presets": presets, "presets_dir": presets_dir, "legacy_presets_dir": legacy_dir}


def save_adjust_preset(name, settings=None):
    safe = _safe_adjust_preset_name(name)
    path = _safe_adjust_preset_path(safe)
    normalized = _normalize_adjust_settings(settings)
    payload = {
        "type": "vrgdg_adjust_preset",
        "version": 1,
        "name": str(name or safe).strip() or safe,
        "adjust": normalized,
        "saved_at": time.time(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return {
        "name": safe,
        "label": payload["name"],
        "settings": normalized,
        "path": path,
        "modified": os.path.getmtime(path),
    }


def import_adjust_preset(preset_payload=None, fallback_name="Imported Adjust Preset"):
    data = preset_payload if isinstance(preset_payload, dict) else {}
    name = str(data.get("name") or fallback_name or "Imported Adjust Preset").strip()
    settings = data.get("adjust") or data.get("settings") or data
    return save_adjust_preset(name, settings)


def _frames_to_tensor(frames):
    import cv2
    import numpy as np
    import torch

    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    stacked = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


def _tensor_to_frames(tensor):
    import cv2
    import numpy as np

    array = tensor.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in array]


def _open_video_writer(path, fps, size, codec):
    import cv2

    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)


def _validate_video_readable(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        ok, _frame = cap.read()
        return bool(ok)
    finally:
        cap.release()


def _media_has_audio(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return bool(str(result.stdout or "").strip())
    except Exception:
        return None


def _probe_video_fps(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode != 0:
            return None
        for line in str(result.stdout or "").splitlines():
            text = line.strip()
            if not text or text == "0/0":
                continue
            if "/" in text:
                numerator, denominator = text.split("/", 1)
                fps = float(numerator) / float(denominator)
            else:
                fps = float(text)
            if fps > 0:
                return fps
    except Exception:
        return None
    return None


def _run_ffmpeg_browser_encode(input_path, output_path, audio_source_path=""):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"ok": False, "error": "ffmpeg was not found on PATH."}
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="vrgdg_lut_h264_", suffix=".mp4", dir=os.path.dirname(output_path))
    os.close(tmp_fd)
    audio_source_path = os.path.abspath(str(audio_source_path or "").strip().strip('"'))
    use_audio_source = bool(audio_source_path and os.path.isfile(audio_source_path))
    source_had_audio = _media_has_audio(audio_source_path) if use_audio_source else False
    try:
        cmd = [ffmpeg, "-y", "-i", input_path]
        if use_audio_source:
            cmd.extend(["-i", audio_source_path, "-map", "0:v:0", "-map", "1:a?"])
        else:
            cmd.extend(["-map", "0:v:0", "-an"])
        cmd.extend([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            tmp_path,
        ])
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60 * 20)
        if result.returncode != 0:
            return {"ok": False, "error": (result.stderr or result.stdout or "ffmpeg encode failed.").strip()[-1000:]}
        if not _validate_video_readable(tmp_path):
            return {"ok": False, "error": "ffmpeg output could not be read back."}
        output_has_audio = _media_has_audio(tmp_path)
        os.replace(tmp_path, output_path)
        return {
            "ok": True,
            "encoder": "ffmpeg libx264",
            "browser_friendly": True,
            "audio_preserved": bool(source_had_audio and output_has_audio),
            "source_had_audio": source_had_audio,
            "output_has_audio": output_has_audio,
        }
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def apply_lut_to_video(
    input_path,
    lut_name,
    output_path="",
    strength=10.0,
    device="auto",
    batch_size=8,
    replace_source=False,
    thumbnail_path="",
    preserve_audio=True,
):
    import cv2

    input_path = _resolve_media_path(input_path, "Input video")
    if os.path.splitext(input_path)[1].lower() not in _SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("Input video type is not supported.")
    lut_path = _safe_lut_path(lut_name)
    lut_name = os.path.basename(lut_path)
    target_device = _resolve_device(device)
    output_path = os.path.abspath(str(output_path or "").strip().strip('"') or _default_output_path(input_path, lut_name))
    if replace_source:
        output_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output = output_path
    if replace_source:
        fd, tmp_output = tempfile.mkstemp(prefix="vrgdg_lut_", suffix=".mp4", dir=os.path.dirname(input_path))
        os.close(fd)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = _probe_video_fps(input_path) or cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    processed_frames = 0
    selected_encoder = ""
    browser_friendly = False
    fallback_errors = []
    started = time.perf_counter()

    for candidate in _VIDEO_CODEC_CANDIDATES:
        codec = candidate["fourcc"]
        selected_encoder = ""
        processed_frames = 0
        batch = []
        try:
            if os.path.isfile(tmp_output):
                os.remove(tmp_output)
        except OSError:
            pass

        writer = _open_video_writer(tmp_output, fps, (width, height), codec)
        if not writer.isOpened():
            fallback_errors.append(f"{candidate['label']}: writer could not open")
            try:
                writer.release()
            except Exception:
                pass
            continue

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            writer.release()
            raise RuntimeError(f"Could not reopen input video: {input_path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                batch.append(frame)
                if len(batch) < max(1, int(batch_size)):
                    continue
                processed_frames += _process_video_batch(batch, writer, lut_name, strength, target_device)
                batch.clear()
            if batch:
                processed_frames += _process_video_batch(batch, writer, lut_name, strength, target_device)
        finally:
            cap.release()
            writer.release()

        if processed_frames <= 0:
            fallback_errors.append(f"{candidate['label']}: no frames processed")
            continue
        if not _validate_video_readable(tmp_output):
            fallback_errors.append(f"{candidate['label']}: output could not be read back")
            continue
        selected_encoder = candidate["label"]
        browser_friendly = bool(candidate["browser_friendly"])
        break

    if not selected_encoder:
        try:
            if os.path.isfile(tmp_output):
                os.remove(tmp_output)
        except OSError:
            pass
        raise RuntimeError("The LUT video render could not create a readable video with any local codec fallback:\n" + "\n".join(fallback_errors))

    ffmpeg_result = {"ok": False, "error": "not attempted"}
    audio_preserved = False
    source_had_audio = _media_has_audio(input_path)
    if os.path.isfile(tmp_output):
        ffmpeg_result = _run_ffmpeg_browser_encode(tmp_output, tmp_output, input_path if preserve_audio else "")
        if ffmpeg_result.get("ok"):
            selected_encoder = ffmpeg_result.get("encoder") or selected_encoder
            browser_friendly = True
            audio_preserved = bool(ffmpeg_result.get("audio_preserved"))

    if replace_source:
        os.replace(tmp_output, output_path)

    thumbnail_path = _write_video_thumbnail(output_path, thumbnail_path)
    elapsed = time.perf_counter() - started
    return {
        "input": input_path,
        "output": output_path,
        "lut": lut_name,
        "device": str(target_device),
        "strength": float(strength),
        "replace_source": bool(replace_source),
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported_frames,
        "processed_frames": processed_frames,
        "elapsed_seconds": elapsed,
        "processed_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        "audio_preserved": audio_preserved,
        "source_had_audio": source_had_audio,
        "preserve_audio": bool(preserve_audio),
        "thumbnail_path": thumbnail_path,
        "encoder": selected_encoder,
        "browser_friendly": browser_friendly,
        "fallback_errors": fallback_errors,
        "ffmpeg_encode": ffmpeg_result,
    }


def apply_film_grain_to_video(
    input_path,
    output_path="",
    grain_intensity=0.04,
    saturation_mix=0.5,
    device="auto",
    batch_size=8,
    replace_source=False,
    thumbnail_path="",
    seed=None,
    preserve_audio=True,
):
    import cv2

    input_path = _resolve_media_path(input_path, "Input video")
    if os.path.splitext(input_path)[1].lower() not in _SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("Input video type is not supported.")
    target_device = _resolve_device(device)
    output_path = os.path.abspath(str(output_path or "").strip().strip('"') or _default_effect_output_path(input_path, "film_grain"))
    if replace_source:
        output_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output = output_path
    if replace_source:
        fd, tmp_output = tempfile.mkstemp(prefix="vrgdg_grain_", suffix=".mp4", dir=os.path.dirname(input_path))
        os.close(fd)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = _probe_video_fps(input_path) or cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    processed_frames = 0
    selected_encoder = ""
    browser_friendly = False
    fallback_errors = []
    started = time.perf_counter()

    for candidate in _VIDEO_CODEC_CANDIDATES:
        codec = candidate["fourcc"]
        selected_encoder = ""
        processed_frames = 0
        batch = []
        frame_offset = 0
        try:
            if os.path.isfile(tmp_output):
                os.remove(tmp_output)
        except OSError:
            pass

        writer = _open_video_writer(tmp_output, fps, (width, height), codec)
        if not writer.isOpened():
            fallback_errors.append(f"{candidate['label']}: writer could not open")
            try:
                writer.release()
            except Exception:
                pass
            continue

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            writer.release()
            raise RuntimeError(f"Could not reopen input video: {input_path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                batch.append(frame)
                if len(batch) < max(1, int(batch_size)):
                    continue
                batch_seed = None if seed in (None, "") else int(seed) + frame_offset
                processed_frames += _process_film_grain_batch(batch, writer, grain_intensity, saturation_mix, target_device, batch_seed)
                frame_offset += len(batch)
                batch.clear()
            if batch:
                batch_seed = None if seed in (None, "") else int(seed) + frame_offset
                processed_frames += _process_film_grain_batch(batch, writer, grain_intensity, saturation_mix, target_device, batch_seed)
        finally:
            cap.release()
            writer.release()

        if processed_frames <= 0:
            fallback_errors.append(f"{candidate['label']}: no frames processed")
            continue
        if not _validate_video_readable(tmp_output):
            fallback_errors.append(f"{candidate['label']}: output could not be read back")
            continue
        selected_encoder = candidate["label"]
        browser_friendly = bool(candidate["browser_friendly"])
        break

    if not selected_encoder:
        try:
            if os.path.isfile(tmp_output):
                os.remove(tmp_output)
        except OSError:
            pass
        raise RuntimeError("The film grain video render could not create a readable video with any local codec fallback:\n" + "\n".join(fallback_errors))

    ffmpeg_result = {"ok": False, "error": "not attempted"}
    audio_preserved = False
    source_had_audio = _media_has_audio(input_path)
    if os.path.isfile(tmp_output):
        ffmpeg_result = _run_ffmpeg_browser_encode(tmp_output, tmp_output, input_path if preserve_audio else "")
        if ffmpeg_result.get("ok"):
            selected_encoder = ffmpeg_result.get("encoder") or selected_encoder
            browser_friendly = True
            audio_preserved = bool(ffmpeg_result.get("audio_preserved"))

    if replace_source:
        os.replace(tmp_output, output_path)

    thumbnail_path = _write_video_thumbnail(output_path, thumbnail_path)
    elapsed = time.perf_counter() - started
    return {
        "input": input_path,
        "output": output_path,
        "effect": "film_grain",
        "device": str(target_device),
        "grain_intensity": float(grain_intensity),
        "saturation_mix": float(saturation_mix),
        "replace_source": bool(replace_source),
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported_frames,
        "processed_frames": processed_frames,
        "elapsed_seconds": elapsed,
        "processed_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        "audio_preserved": audio_preserved,
        "source_had_audio": source_had_audio,
        "preserve_audio": bool(preserve_audio),
        "thumbnail_path": thumbnail_path,
        "encoder": selected_encoder,
        "browser_friendly": browser_friendly,
        "fallback_errors": fallback_errors,
        "ffmpeg_encode": ffmpeg_result,
    }


def apply_adjust_to_video(
    input_path,
    output_path="",
    settings=None,
    device="auto",
    batch_size=8,
    replace_source=False,
    thumbnail_path="",
    preserve_audio=True,
):
    import cv2

    input_path = _resolve_media_path(input_path, "Input video")
    if os.path.splitext(input_path)[1].lower() not in _SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("Input video type is not supported.")
    target_device = _resolve_device(device)
    settings = _normalize_adjust_settings(settings)
    output_path = os.path.abspath(str(output_path or "").strip().strip('"') or _default_effect_output_path(input_path, "adjust"))
    if replace_source:
        output_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output = output_path
    if replace_source:
        fd, tmp_output = tempfile.mkstemp(prefix="vrgdg_adjust_", suffix=".mp4", dir=os.path.dirname(input_path))
        os.close(fd)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = _probe_video_fps(input_path) or cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    processed_frames = 0
    selected_encoder = ""
    browser_friendly = False
    fallback_errors = []
    started = time.perf_counter()

    for candidate in _VIDEO_CODEC_CANDIDATES:
        codec = candidate["fourcc"]
        selected_encoder = ""
        processed_frames = 0
        batch = []
        try:
            if os.path.isfile(tmp_output):
                os.remove(tmp_output)
        except OSError:
            pass

        writer = _open_video_writer(tmp_output, fps, (width, height), codec)
        if not writer.isOpened():
            fallback_errors.append(f"{candidate['label']}: writer could not open")
            try:
                writer.release()
            except Exception:
                pass
            continue

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            writer.release()
            raise RuntimeError(f"Could not reopen input video: {input_path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                batch.append(frame)
                if len(batch) < max(1, int(batch_size)):
                    continue
                processed_frames += _process_adjust_batch(batch, writer, settings, target_device)
                batch.clear()
            if batch:
                processed_frames += _process_adjust_batch(batch, writer, settings, target_device)
        finally:
            cap.release()
            writer.release()

        if processed_frames <= 0:
            fallback_errors.append(f"{candidate['label']}: no frames processed")
            continue
        if not _validate_video_readable(tmp_output):
            fallback_errors.append(f"{candidate['label']}: output could not be read back")
            continue
        selected_encoder = candidate["label"]
        browser_friendly = bool(candidate["browser_friendly"])
        break

    if not selected_encoder:
        try:
            if os.path.isfile(tmp_output):
                os.remove(tmp_output)
        except OSError:
            pass
        raise RuntimeError("The Adjust video render could not create a readable video with any local codec fallback:\n" + "\n".join(fallback_errors))

    ffmpeg_result = {"ok": False, "error": "not attempted"}
    audio_preserved = False
    source_had_audio = _media_has_audio(input_path)
    if os.path.isfile(tmp_output):
        ffmpeg_result = _run_ffmpeg_browser_encode(tmp_output, tmp_output, input_path if preserve_audio else "")
        if ffmpeg_result.get("ok"):
            selected_encoder = ffmpeg_result.get("encoder") or selected_encoder
            browser_friendly = True
            audio_preserved = bool(ffmpeg_result.get("audio_preserved"))

    if replace_source:
        os.replace(tmp_output, output_path)

    thumbnail_path = _write_video_thumbnail(output_path, thumbnail_path)
    elapsed = time.perf_counter() - started
    return {
        "input": input_path,
        "output": output_path,
        "effect": "adjust",
        "device": str(target_device),
        "settings": settings,
        "replace_source": bool(replace_source),
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported_frames,
        "processed_frames": processed_frames,
        "elapsed_seconds": elapsed,
        "processed_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        "audio_preserved": audio_preserved,
        "source_had_audio": source_had_audio,
        "preserve_audio": bool(preserve_audio),
        "thumbnail_path": thumbnail_path,
        "encoder": selected_encoder,
        "browser_friendly": browser_friendly,
        "fallback_errors": fallback_errors,
        "ffmpeg_encode": ffmpeg_result,
    }


def _process_video_batch(batch, writer, lut_name, strength, target_device):
    source = _frames_to_tensor(batch)
    output = _apply_lut_tensor(source, lut_name, strength, target_device)
    for frame in _tensor_to_frames(output):
        writer.write(frame)
    return len(batch)


def _process_film_grain_batch(batch, writer, grain_intensity, saturation_mix, target_device, seed=None):
    source = _frames_to_tensor(batch)
    output = _apply_film_grain_tensor(source, grain_intensity, saturation_mix, target_device, seed=seed)
    for frame in _tensor_to_frames(output):
        writer.write(frame)
    return len(batch)


def _process_adjust_batch(batch, writer, settings, target_device):
    source = _frames_to_tensor(batch)
    output = _apply_adjust_tensor(source, settings, target_device)
    for frame in _tensor_to_frames(output):
        writer.write(frame)
    return len(batch)


def _write_video_thumbnail(video_path, thumbnail_path=""):
    if not thumbnail_path:
        return ""
    try:
        import cv2
        from PIL import Image

        thumbnail_path = os.path.abspath(str(thumbnail_path or "").strip().strip('"'))
        if not thumbnail_path:
            return ""
        os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        try:
          ok, frame = cap.read()
        finally:
          cap.release()
        if not ok:
            return ""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.thumbnail((960, 540))
        image.save(thumbnail_path, quality=90)
        return thumbnail_path
    except Exception:
        return ""


def delete_lut_preview(path, project_folder=""):
    path = _safe_preview_path(path, project_folder)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def register_lut_routes(server_instance):
    @server_instance.routes.get("/vrgdg/music_builder/luts")
    async def vrgdg_music_builder_luts(request):
        try:
            return web.json_response({"ok": True, **list_luts()})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.get("/vrgdg/music_builder/luts/example")
    async def vrgdg_music_builder_lut_example(request):
        try:
            path = _safe_examples_path(request.query.get("name", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        return web.FileResponse(path)

    @server_instance.routes.post("/vrgdg/music_builder/luts/apply_image")
    async def vrgdg_music_builder_apply_lut_image(request):
        try:
            payload = await request.json()
            result = apply_lut_to_image(
                input_path=payload.get("input_path", ""),
                lut_name=payload.get("lut_name", ""),
                output_path=payload.get("output_path", ""),
                strength=payload.get("strength", 10.0),
                device=payload.get("device", "auto"),
                replace_source=bool(payload.get("replace_source", False)),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/luts/apply_video")
    async def vrgdg_music_builder_apply_lut_video(request):
        try:
            payload = await request.json()
            result = apply_lut_to_video(
                input_path=payload.get("input_path", ""),
                lut_name=payload.get("lut_name", ""),
                output_path=payload.get("output_path", ""),
                strength=payload.get("strength", 10.0),
                device=payload.get("device", "auto"),
                batch_size=payload.get("batch_size", 8),
                replace_source=bool(payload.get("replace_source", False)),
                thumbnail_path=payload.get("thumbnail_path", ""),
                preserve_audio=bool(payload.get("preserve_audio", True)),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/luts/preview")
    async def vrgdg_music_builder_preview_lut(request):
        try:
            payload = await request.json()
            result = preview_lut_on_media(
                input_path=payload.get("input_path", ""),
                lut_name=payload.get("lut_name", ""),
                media_type=payload.get("media_type", ""),
                strength=payload.get("strength", 10.0),
                device=payload.get("device", "auto"),
                scene_id=payload.get("scene_id", ""),
                project_folder=payload.get("project_folder", ""),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/post_process/film_grain/apply_image")
    async def vrgdg_music_builder_apply_film_grain_image(request):
        try:
            payload = await request.json()
            result = apply_film_grain_to_image(
                input_path=payload.get("input_path", ""),
                output_path=payload.get("output_path", ""),
                grain_intensity=payload.get("grain_intensity", 0.04),
                saturation_mix=payload.get("saturation_mix", 0.5),
                device=payload.get("device", "auto"),
                replace_source=bool(payload.get("replace_source", False)),
                seed=payload.get("seed", None),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/post_process/film_grain/apply_video")
    async def vrgdg_music_builder_apply_film_grain_video(request):
        try:
            payload = await request.json()
            result = apply_film_grain_to_video(
                input_path=payload.get("input_path", ""),
                output_path=payload.get("output_path", ""),
                grain_intensity=payload.get("grain_intensity", 0.04),
                saturation_mix=payload.get("saturation_mix", 0.5),
                device=payload.get("device", "auto"),
                batch_size=payload.get("batch_size", 8),
                replace_source=bool(payload.get("replace_source", False)),
                thumbnail_path=payload.get("thumbnail_path", ""),
                seed=payload.get("seed", None),
                preserve_audio=bool(payload.get("preserve_audio", True)),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/post_process/film_grain/preview")
    async def vrgdg_music_builder_preview_film_grain(request):
        try:
            payload = await request.json()
            result = preview_film_grain_on_media(
                input_path=payload.get("input_path", ""),
                media_type=payload.get("media_type", ""),
                grain_intensity=payload.get("grain_intensity", 0.04),
                saturation_mix=payload.get("saturation_mix", 0.5),
                device=payload.get("device", "auto"),
                scene_id=payload.get("scene_id", ""),
                project_folder=payload.get("project_folder", ""),
                seed=payload.get("seed", None),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/post_process/adjust/preview")
    async def vrgdg_music_builder_preview_adjust(request):
        try:
            payload = await request.json()
            result = preview_adjust_on_media(
                input_path=payload.get("input_path", ""),
                media_type=payload.get("media_type", ""),
                settings=payload.get("settings", {}),
                device=payload.get("device", "auto"),
                scene_id=payload.get("scene_id", ""),
                project_folder=payload.get("project_folder", ""),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/post_process/adjust/apply_video")
    async def vrgdg_music_builder_apply_adjust_video(request):
        try:
            payload = await request.json()
            result = apply_adjust_to_video(
                input_path=payload.get("input_path", ""),
                output_path=payload.get("output_path", ""),
                settings=payload.get("settings", {}),
                device=payload.get("device", "auto"),
                batch_size=payload.get("batch_size", 8),
                replace_source=bool(payload.get("replace_source", False)),
                thumbnail_path=payload.get("thumbnail_path", ""),
                preserve_audio=bool(payload.get("preserve_audio", True)),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.get("/vrgdg/music_builder/post_process/adjust/presets")
    async def vrgdg_music_builder_list_adjust_presets(request):
        try:
            return web.json_response({"ok": True, **list_adjust_presets()})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/post_process/adjust/presets/save")
    async def vrgdg_music_builder_save_adjust_preset(request):
        try:
            payload = await request.json()
            result = save_adjust_preset(payload.get("name", ""), payload.get("settings", {}))
            return web.json_response({"ok": True, "preset": result, **list_adjust_presets()})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/post_process/adjust/presets/import")
    async def vrgdg_music_builder_import_adjust_preset(request):
        try:
            payload = await request.json()
            result = import_adjust_preset(payload.get("preset", {}), payload.get("name", "Imported Adjust Preset"))
            return web.json_response({"ok": True, "preset": result, **list_adjust_presets()})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/luts/delete_preview")
    async def vrgdg_music_builder_delete_lut_preview(request):
        try:
            payload = await request.json()
            deleted = delete_lut_preview(payload.get("path", ""), payload.get("project_folder", ""))
            return web.json_response({"ok": True, "deleted": deleted})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
