"""RefMods Studio backend: create a RefMod from chosen images and file it under models/refmods/<type>.

The extraction itself is ComfyUI-MiniMaxH3Mod's *Create H3 RefMod* node, called directly. This module fixes the
settings the Studio does not expose (resolution, pool grid, token cap) and the folder (always the type).

Images are cropped to the boxes the user chose (or kept whole), scaled down to one canvas that holds every image
without cutting any, and padded with each image's own border colour. The canvas rule is mirrored by
``canvasFor`` in ``web/music_video_builder/refmod_trim.mjs``. Keep the two in step.
"""

import asyncio
import json
import math
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from .refmod_library import PREVIEW_SUFFIX, save_preview
from .refmod_picker import IMAGE_EXTENSIONS

# Types the Studio can create. The folder under models/refmods is always the type.
ACTIVE_TYPES = (
    "identity", "clothing_men", "clothing_women", "background", "style", "pose_motion", "generic",
    "object", "prop", "vehicle", "creature",
)

# Settings users cannot change from the Studio.
FIXED_SETTINGS = {
    "ref_resolution": 1024,
    "pool_h": 32,
    "pool_w": 32,
    "latent_frames": 16,
    "multiplier": 1,
    "max_tokens": 5120,
    "merge": False,
    "motion_only": False,
    "budget_policy": "truncate",
    "extraction_preset": "manual",
}
MAX_TOKENS = FIXED_SETTINGS["max_tokens"]
MIN_CROP_SIZE = 32
# Quality presets: the share of each image's detail that is kept. Mirrors QUALITY_PRESETS in refmod_trim.mjs.
QUALITY_SCALES = {"maximum": 1.0, "high": 0.8, "balanced": 0.6, "compact": 0.45, "draft": 0.3}
DEFAULT_QUALITY = "balanced"
CANVAS_LIMIT = 1024
# The video VAE needs at least 320 px on both sides (the RefMod pack scales smaller images up). The canvas is padded to
# this size instead, so an image keeps the chosen scale. Mirrors MIN_CANVAS_SIDE in refmod_trim.mjs.
MIN_CANVAS_SIDE = 320
VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
MODE_MAP = {"Full Reference": "encode", "Compressed Reference": "training"}

_create_lock = threading.Lock()


class RefModExistsError(Exception):
    """The target file already exists and overwrite was not requested."""

    def __init__(self, path: str):
        super().__init__(f"A RefMod already exists at {path}.")
        self.path = path


def _snap32(value: float) -> int:
    """Round half up to a multiple of 32 (same as ``snap32`` in the browser), at least 32."""
    return max(32, int(math.floor(value / 32 + 0.5)) * 32)


def canvas_for(sizes: Sequence[Tuple[int, int]], scale: float = 1.0, limit: int = CANVAS_LIMIT) -> Tuple[int, int, float]:
    """One canvas that holds every (width, height) without cutting any.

    The widest width by the tallest height, each scaled by ``scale`` (never up, never past ``limit`` on the longest
    side), then snapped to multiples of 32 and padded up to ``MIN_CANVAS_SIDE``. Returns ``(width, height, fit)``.
    """
    widest = max(width for width, _ in sizes)
    tallest = max(height for _, height in sizes)
    fit = min(1.0, scale, limit / max(widest, tallest))
    return max(MIN_CANVAS_SIDE, _snap32(widest * fit)), max(MIN_CANVAS_SIDE, _snap32(tallest * fit)), fit


def tokens_for_canvas(frames: int, width: int, height: int) -> int:
    return frames * (width // 32) * (height // 32)


def _border_colour(image: Image.Image) -> Tuple[int, int, int]:
    """Median colour of the outer ring of an image, used to pad it without a visible seam."""
    array = np.asarray(image)
    ring = np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])
    return tuple(int(value) for value in np.median(ring, axis=0))


def _check_crop(crop: Any, width: int, height: int, path: str) -> Optional[Tuple[int, int, int, int]]:
    if crop in (None, [], ()):
        return None
    try:
        x0, y0, x1, y1 = (int(round(float(value))) for value in crop)
    except (TypeError, ValueError):
        raise ValueError(f"Crop for {os.path.basename(path)} must be four numbers: x0, y0, x1, y1.")
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 - x0 < MIN_CROP_SIZE or y1 - y0 < MIN_CROP_SIZE:
        raise ValueError(f"Crop for {os.path.basename(path)} is smaller than {MIN_CROP_SIZE} px.")
    return x0, y0, x1, y1


def prepare_images(paths: Sequence[str], crops: Sequence[Any], quality: str = DEFAULT_QUALITY) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
    """Crop, scale and pad the images onto one shared canvas. Returns ``[1, H, W, 3]`` tensors and ``(width, height)``."""
    images: List[Image.Image] = []
    for path, crop in zip(paths, crops):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        box = _check_crop(crop, image.width, image.height, path)
        images.append(image.crop(box) if box else image)
    canvas_width, canvas_height, fit = canvas_for([image.size for image in images], QUALITY_SCALES[quality])
    tensors: List[torch.Tensor] = []
    for image in images:
        scale = min(fit, canvas_width / image.width, canvas_height / image.height)
        if scale < 1.0:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
        canvas = Image.new("RGB", (canvas_width, canvas_height), _border_colour(image))
        canvas.paste(image, ((canvas_width - image.width) // 2, (canvas_height - image.height) // 2))
        tensors.append(torch.from_numpy(np.asarray(canvas).copy()).float().div(255.0).unsqueeze(0))
    return tensors, (canvas_width, canvas_height)


def _extract_class():
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3RefModExtract")
    if cls is None:
        raise RuntimeError(
            "ComfyUI-MiniMaxH3Mod is not installed or did not load. Install it in custom_nodes and restart ComfyUI.")
    return cls


def _ensure_progress_context() -> None:
    """Give ComfyUI's progress hook an id to report against when no prompt has run yet.

    The hook reads ``PromptServer.last_prompt_id``, which ComfyUI only sets when the first queued prompt starts.
    A RefMod created before any prompt has run would fail with an AttributeError. When a prompt has run, its
    ids are left alone.
    """
    try:
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None or hasattr(instance, "last_prompt_id"):
        return
    instance.last_prompt_id = "vrgdg_refmod_studio"
    if getattr(instance, "last_node_id", None) is None:
        instance.last_node_id = "vrgdg_refmod_studio"


def _load_video_vae():
    import folder_paths
    import nodes

    if VIDEO_VAE_NAME not in folder_paths.get_filename_list("vae"):
        raise FileNotFoundError(f"The MiniMax H3 video VAE '{VIDEO_VAE_NAME}' was not found in models/vae.")
    return nodes.VAELoader().load_vae(VIDEO_VAE_NAME)[0]


def validate_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Check and normalise a create request. Raises ValueError with a message the UI can show.

    Any number of images from one up works for every type. The image order is kept, so for identity the front, left
    and right close-ups come first when they are given.
    """
    concept_type = str(payload.get("type") or "")
    if concept_type not in ACTIVE_TYPES:
        raise ValueError(f"'{concept_type}' cannot be created yet. Choose one of: {', '.join(ACTIVE_TYPES)}.")
    name = str(payload.get("name") or "").strip()
    for char in '\\/:*?"<>|':
        name = name.replace(char, "_")
    name = name.strip(" .")
    if not name:
        raise ValueError("Enter a name for the RefMod.")
    paths = [str(path) for path in (payload.get("paths") or [])]
    if not paths:
        raise ValueError("Add at least one image.")
    for path in paths:
        if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS or not os.path.isfile(path):
            raise ValueError(f"Image not found or not a supported type: {path}")
    raw_crops = payload.get("crops")
    crops = list(raw_crops) if isinstance(raw_crops, (list, tuple)) else []
    if crops and len(crops) != len(paths):
        raise ValueError("Send one crop (or null) for every image.")
    crops = crops or [None] * len(paths)
    quality = str(payload.get("quality") or DEFAULT_QUALITY)
    if quality not in QUALITY_SCALES:
        raise ValueError(f"Quality must be one of: {', '.join(QUALITY_SCALES)}.")
    mode_label = str(payload.get("mode") or "Full Reference")
    if mode_label not in MODE_MAP:
        raise ValueError("Mode must be Full Reference or Compressed Reference.")
    try:
        steps = int(payload.get("steps", 500))
    except (TypeError, ValueError):
        raise ValueError("Refinement steps must be a whole number.")
    if not 0 <= steps <= 2000:
        raise ValueError("Refinement steps must be between 0 and 2000.")
    return {
        "type": concept_type,
        "name": name,
        "paths": paths,
        "crops": crops,
        "quality": quality,
        "mode": MODE_MAP[mode_label],
        "steps": steps,
        "description": str(payload.get("description") or "").strip(),
        "overwrite": bool(payload.get("overwrite")),
    }


def create_refmod(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create the RefMod file. Blocking: run it off the event loop."""
    request = validate_request(payload)
    extract = _extract_class()
    mod_output_path = sys.modules[extract.__module__].mod_output_path
    target = mod_output_path(request["name"], request["type"]) + ".safetensors"
    if os.path.isfile(target) and not request["overwrite"]:
        raise RefModExistsError(target)
    if not _create_lock.acquire(blocking=False):
        raise RuntimeError("A RefMod is already being created. Wait for it to finish.")
    try:
        print(f"[VRGDG RefMod] Creating {request['type']} RefMod '{request['name']}' from {len(request['paths'])} image(s).")
        refs, canvas = prepare_images(request["paths"], request["crops"], request["quality"])
        tokens = tokens_for_canvas(len(refs), canvas[0], canvas[1])
        if request["mode"] == "encode" and tokens > MAX_TOKENS:
            raise ValueError(
                f"These images would use {tokens:,} tokens ({len(refs)} x {canvas[0]}x{canvas[1]}), over the "
                f"{MAX_TOKENS:,} limit. Choose a lower quality, trim closer to the subject, or remove images.")
        print(f"[VRGDG RefMod] Canvas {canvas[0]}x{canvas[1]}, {len(refs)} frame(s), {tokens:,} tokens.")
        vae = _load_video_vae()
        _ensure_progress_context()
        output = extract.execute(
            name=request["name"],
            mode=request["mode"],
            concept_type=request["type"],
            refs_bundle=refs,
            vae=vae,
            identity=request["steps"],
            description=request["description"],
            save=True,
            subfolder=request["type"],
            **FIXED_SETTINGS,
        )
        details = json.loads(output[1])
    finally:
        _create_lock.release()
    saved = (details.get("saved_paths") or [target])[0]
    try:
        # The first image as it was encoded is the preview the pickers show.
        first = (refs[0][0].clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
        from PIL import Image as _Image

        save_preview(_Image.fromarray(first), os.path.splitext(saved)[0] + PREVIEW_SUFFIX)
    except Exception as exc:
        print(f"[VRGDG RefMod] Could not save a preview: {exc}")
    print(f"[VRGDG RefMod] Saved {saved}")
    return {
        "path": saved,
        "folder": request["type"],
        "name": request["name"],
        "kind": details.get("kind"),
        "latent_frames": details.get("latent_frames"),
        "tokens": details.get("tokens"),
        "canvas": [canvas[0], canvas[1]],
        "quality": request["quality"],
        "size": os.path.getsize(saved),
    }


def collect_image_paths(payload: Dict[str, Any], save: Any = None) -> List[str]:
    """The image paths of a Reference Builder save: the card's own image first, then any ``extra_images``.

    Each image is ``{"path": ...}`` or ``{"data": <base64>, "name": ...}``. Data images are written to the temp folder.
    """
    save = save or _save_data_url
    images = [payload.get("image") if isinstance(payload.get("image"), dict) else {}]
    extras = payload.get("extra_images")
    if isinstance(extras, (list, tuple)):
        images.extend(item for item in extras if isinstance(item, dict))
    paths: List[str] = []
    for image in images:
        path = str(image.get("path") or "").strip()
        data = str(image.get("data") or "").strip()
        if not path and data:
            path = save(data, str(image.get("name") or "reference.png"))
        if path:
            paths.append(path)
    return paths


def _save_data_url(data: str, name: str) -> str:
    """Write a base64 image (with or without a data: prefix) to the Studio's temp folder and return its path."""
    import base64
    import io

    from .refmod_picker import _save_upload

    payload = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
    raw = base64.b64decode(payload)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("Image is larger than 64 MB.")
    extension = os.path.splitext(name)[1].lower()
    if extension not in IMAGE_EXTENSIONS:
        extension = ".png"
    return _save_upload(f"reference{extension}", raw)


def _register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return

    @instance.routes.post("/vrgdg/refmod/from_image")
    async def vrgdg_refmod_from_image(request):
        """Save one Reference Builder image as a RefMod in models/refmods/<type>/<name>."""
        try:
            payload = await request.json()
            paths = await asyncio.to_thread(collect_image_paths, payload)
            result = await asyncio.to_thread(create_refmod, {**payload, "paths": paths})
        except RefModExistsError as exc:
            return web.json_response({"ok": False, "exists": True, "path": exc.path, "error": str(exc)}, status=409)
        except Exception as exc:
            print(f"[VRGDG RefMod] Could not save the image as a RefMod: {exc}")
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @instance.routes.post("/vrgdg/refmod/create")
    async def vrgdg_refmod_create(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(create_refmod, payload)
        except RefModExistsError as exc:
            return web.json_response({"ok": False, "exists": True, "path": exc.path, "error": str(exc)}, status=409)
        except Exception as exc:
            print(f"[VRGDG RefMod] Create failed: {exc}")
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})


_register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
