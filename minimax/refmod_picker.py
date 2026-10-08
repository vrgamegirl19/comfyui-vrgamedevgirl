"""RefMod helper nodes: pick reference images with a file dialog, and describe them with the loaded LM Studio model.

``VRGDG_RefModImagePicker`` outputs an ``H3_REF_LIST``, the same type as ComfyUI-MiniMaxH3Mod's folder loader, so it
plugs into the ``refs_bundle`` input of *Create H3 RefMod*. ``VRGDG_RefModDescribe`` turns that list into a text
description for the Create node's ``description`` input.
"""

import asyncio
import json
import os
import struct
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional

import folder_paths
import numpy as np
import torch
from comfy_api.latest import ui
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

_ENDING = " Output only the description as plain text, one paragraph, no preamble."

DESCRIBE_INSTRUCTIONS = {
    "identity": (
        "These images all show the same character or subject. Write one concise description of the subject that "
        "stays true across every image: age and build, face and hair, skin or fur colour, clothing and accessories, "
        "and any distinctive marks. Ignore backgrounds, poses, lighting and camera angle."
        + _ENDING
    ),
    "clothing": (
        "These images all show the same outfit or garment. Write one concise description of the clothing only: "
        "garment types, cut and fit, colours, patterns, fabrics and materials, trims, logos, footwear, jewelry and "
        "other accessories. If a person is wearing it, do not describe the person: no face, hair, skin, body, age "
        "or pose. If an image shows only the clothing, describe it directly. Ignore backgrounds, lighting and "
        "camera angle."
        + _ENDING
    ),
    "background": (
        "These images all show the same place or environment. Write one concise description of the setting: the "
        "type of location, architecture or landscape features, layout, key objects and props, materials and "
        "textures, colours, time of day, weather and atmosphere. Do not describe people or characters. Ignore "
        "camera angle."
        + _ENDING
    ),
    "style": (
        "These images all share one visual style. Write one concise description of the style itself, not of what "
        "the images show: the medium and technique (for example cartoon, 3D render, watercolor, pencil sketch, "
        "photograph, pixel art, black and white), colour palette, line quality, shading and texture, lighting "
        "mood, level of detail, and the era or genre it belongs to. Word it so it could be applied to any subject, "
        "and do not name the specific characters, objects or places in the images."
        + _ENDING
    ),
    "pose_motion": (
        "These images all show the same pose, movement or camera move. Write one concise description of the pose "
        "or motion only: body position, limb placement, direction and flow of movement, gesture, and any camera "
        "movement or framing. Do not describe who the person is, their clothing, or the background."
        + _ENDING
    ),
    "object": (
        "These images all show the same object. Write one concise description of the object only: what it is, its "
        "shape and proportions, materials, colours, surface finish, markings, logos, wear and any distinctive "
        "details. Do not describe who is holding it, the background or the lighting."
        + _ENDING
    ),
    "prop": (
        "These images all show the same prop or set-dressing item. Write one concise description of the prop only: "
        "what it is, its shape, size, materials, colours, condition and distinctive details. Do not describe people, "
        "the background or the lighting."
        + _ENDING
    ),
    "vehicle": (
        "These images all show the same vehicle. Write one concise description of the vehicle only: make or type, "
        "body shape, paint colours and finish, wheels, lights, trim, decals, modifications, damage and wear. Do not "
        "describe the driver, the road or the background."
        + _ENDING
    ),
    "creature": (
        "These images all show the same creature or animal. Write one concise description of the creature only: "
        "species or type, size and build, body shape, skin, fur or scales, colours and patterns, eyes, limbs, "
        "distinctive features and any gear it wears. Ignore the background, pose and lighting."
        + _ENDING
    ),
    "generic": (
        "These images all show the same subject or idea. Write one concise description of what they have in "
        "common: the main subject, its key visual features, colours and materials, and the overall look."
        + _ENDING
    ),
}
# Men's and women's clothing are described the same way as clothing.
DESCRIBE_INSTRUCTIONS["clothing_men"] = DESCRIBE_INSTRUCTIONS["clothing"]
DESCRIBE_INSTRUCTIONS["clothing_women"] = DESCRIBE_INSTRUCTIONS["clothing"]
DEFAULT_DESCRIBE_INSTRUCTION = DESCRIBE_INSTRUCTIONS["identity"]


def _parse_paths(value: str) -> List[str]:
    """Paths from a JSON list, an object with ``image_paths``/``images``, or one path per line."""
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        parsed = parsed.get("image_paths") or parsed.get("images")
    if isinstance(parsed, list):
        items = [str(item) for item in parsed]
    else:
        items = text.splitlines()
    return [item.strip().strip('"') for item in items if item.strip().strip('"')]


def _load_image(path: str, max_edge: int) -> torch.Tensor:
    """One image file as ``[1, H, W, 3]`` float32 in [0, 1], downscaled (never upscaled) to ``max_edge``."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = min(1.0, max_edge / max(img.size))
        if scale < 1.0:
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        return torch.from_numpy(np.asarray(img).copy()).float().div(255.0).unsqueeze(0)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (image[0].clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(array)


def pick_image_files() -> List[str]:
    """Open the native multi-select file dialog and return the chosen image paths (empty when cancelled)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            chosen = filedialog.askopenfilenames(
                title="Choose reference images",
                filetypes=[("Image files", " ".join("*" + ext for ext in IMAGE_EXTENSIONS)), ("All files", "*.*")],
            )
        finally:
            root.destroy()
        return [str(path) for path in chosen]
    except Exception as tk_exc:
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose reference images'
$dialog.Multiselect = $true
$dialog.Filter = 'Image files (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif)|*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write(($dialog.FileNames -join "`n")) }
"""
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", script],
                capture_output=True, text=True, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as ps_exc:
            raise RuntimeError(f"Native file dialog is not available. tkinter error: {tk_exc}; PowerShell error: {ps_exc}")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _downscale(pil: Image.Image, edge: int) -> Image.Image:
    scale = min(1.0, int(edge) / max(pil.size))
    if scale < 1.0:
        pil = pil.resize((max(1, round(pil.width * scale)), max(1, round(pil.height * scale))), Image.LANCZOS)
    return pil


def _pick_images_to_describe(images: List[Image.Image], concept_type: str, max_images: int) -> List[Image.Image]:
    """Up to ``max_images`` images, spread across the list. Identity always keeps its first three close-ups."""
    count = min(max_images, len(images))
    lead = images[:3] if concept_type == "identity" and count >= 3 else []
    rest = images[len(lead):]
    room = count - len(lead)
    step = max(1, len(rest) / room) if room else 1
    return lead + [rest[min(len(rest) - 1, int(i * step))] for i in range(room)] if rest else lead


def describe_images(images: List[Image.Image], concept_type: str = "identity", instruction: str = "",
                    name_hint: str = "", lm_studio_url: str = "http://127.0.0.1:1234/v1",
                    max_images: int = 6, image_edge: int = 768, llm: Optional[Dict[str, Any]] = None) -> str:
    """Describe images with the vision model of the LLM Runner the Video Builder has selected.

    The question asked depends on ``concept_type`` (see ``DESCRIBE_INSTRUCTIONS``). A non-empty ``instruction``
    replaces it. ``llm`` holds the runner settings (``textGemmaRunnerPayload``). LM Studio, LLM API and Custom
    Server are used as selected. Without ``llm``, or with a local runner, LM Studio at ``lm_studio_url`` is used.
    This never loads or switches LM Studio models.
    """
    from ..agent_api.llm_runtime import prepare_llm_payload
    from ..llm.builder_runner import _llm_runner_from_payload, _strip_builder_thinking_text, _try_run_remote_vision

    if not images:
        raise ValueError("[VRGDG RefMod] No reference images to describe.")
    if concept_type not in DESCRIBE_INSTRUCTIONS:
        raise ValueError(f"[VRGDG RefMod] Describing '{concept_type}' is not available.")
    chosen = _pick_images_to_describe(images, concept_type, int(max_images))
    text = str(instruction or "").strip() or DESCRIBE_INSTRUCTIONS[concept_type]
    if str(name_hint or "").strip():
        text += f"\nThe subject is: {str(name_hint).strip()}."
    payload = dict(llm or {})
    if _llm_runner_from_payload(payload) not in ("lm_studio", "llm_api", "own_server"):
        payload = {"text_runner": "lm_studio", "lmstudio_base_url": lm_studio_url}
    payload = prepare_llm_payload(payload)
    runner = _llm_runner_from_payload(payload)
    print(f"[VRGDG RefMod] Describing {len(chosen)} image(s) as {concept_type} with the {runner} runner.")
    result, _info = _try_run_remote_vision(payload, text, [_downscale(pil, image_edge) for pil in chosen], max_new_tokens=400)
    description = _strip_builder_thinking_text(result)
    if not description:
        raise RuntimeError("[VRGDG RefMod] The vision model returned an empty description.")
    print(f"[VRGDG RefMod] Description: {description}")
    return description


def _upload_dir() -> str:
    path = os.path.join(folder_paths.get_temp_directory(), "vrgdg_refmod_studio")
    os.makedirs(path, exist_ok=True)
    return path


def _save_upload(filename: str, data: bytes) -> str:
    stem, extension = os.path.splitext(os.path.basename(filename or "image"))
    extension = extension.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type '{extension or filename}'.")
    safe = "".join(char if char.isalnum() or char in "-_ ." else "_" for char in stem).strip() or "image"
    path = os.path.join(_upload_dir(), f"{uuid.uuid4().hex[:8]}_{safe}{extension}")
    with open(path, "wb") as handle:
        handle.write(data)
    try:
        with Image.open(path) as check:
            check.verify()
    except Exception:
        os.remove(path)
        raise ValueError(f"'{filename}' is not a readable image.")
    return path


MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def _register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return

    @instance.routes.post("/vrgdg/refmod/pick_images")
    async def vrgdg_refmod_pick_images(request):
        try:
            # macOS aborts when a Tk window is created off the main thread, so the picker stays on the event loop there.
            paths = pick_image_files() if sys.platform == "darwin" else await asyncio.to_thread(pick_image_files)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "paths": paths})

    @instance.routes.get("/vrgdg/refmod/describe_prompts")
    async def vrgdg_refmod_describe_prompts(request):
        return web.json_response({"ok": True, "prompts": DESCRIBE_INSTRUCTIONS})

    @instance.routes.post("/vrgdg/refmod/upload_image")
    async def vrgdg_refmod_upload_image(request):
        try:
            reader = await request.multipart()
            field = await reader.next()
            if field is None or field.name != "image":
                raise ValueError("Send the file in a multipart field named 'image'.")
            data = await field.read(decode=False)
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError("Image is larger than 64 MB.")
            path = await asyncio.to_thread(_save_upload, field.filename, bytes(data))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "path": path})

    @instance.routes.post("/vrgdg/refmod/describe")
    async def vrgdg_refmod_describe(request):
        try:
            body = await request.json()
            paths = [str(path) for path in (body.get("paths") or [])]
            if not paths:
                raise ValueError("Add at least one image first.")

            def run() -> str:
                images = []
                for path in paths:
                    with Image.open(path) as img:
                        images.append(img.convert("RGB"))
                return describe_images(
                    images, concept_type=str(body.get("concept_type") or "identity"),
                    name_hint=str(body.get("name_hint") or ""),
                    llm=body.get("llm") if isinstance(body.get("llm"), dict) else None)

            description = await asyncio.to_thread(run)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "description": description})


_register_routes()


class VRGDG_RefModImagePicker:
    """Reference images chosen with a file dialog, as a list for Create H3 RefMod."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON list or one path per line. Use the Browse images button to fill it.",
                }),
                "max_edge": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 64,
                    "tooltip": "Longest edge in px. Larger images are scaled down, smaller ones are kept as they are.",
                }),
            }
        }

    RETURN_TYPES = ("H3_REF_LIST", "INT")
    RETURN_NAMES = ("refs", "count")
    FUNCTION = "load"
    CATEGORY = "VRGDG/RefMod"
    DESCRIPTION = (
        "Pick reference images with a file dialog. Connect refs to the refs_bundle input of Create H3 RefMod "
        "and to VRGDG RefMod Describe."
    )

    @classmethod
    def IS_CHANGED(cls, image_paths, max_edge):
        stamps = []
        for path in _parse_paths(image_paths):
            try:
                stat = os.stat(path)
                stamps.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                stamps.append(f"{path}:missing")
        return "|".join(stamps) + f"|{max_edge}"

    def load(self, image_paths, max_edge):
        paths = _parse_paths(image_paths)
        if not paths:
            raise ValueError("[VRGDG RefMod] No images selected. Click Browse images and choose your files.")
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("[VRGDG RefMod] Image not found: " + ", ".join(missing))
        refs = [_load_image(path, int(max_edge)) for path in paths]
        print(f"[VRGDG RefMod] Loaded {len(refs)} image(s).")
        return (refs, len(refs))


class VRGDG_RefModDescribe:
    """Describe the subject in reference images with the vision model LM Studio has loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refs": ("H3_REF_LIST",),
                "max_images": ("INT", {
                    "default": 6, "min": 1, "max": 16,
                    "tooltip": "How many of the images are sent to the model, evenly spread across the list.",
                }),
                "image_edge": ("INT", {
                    "default": 768, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Longest edge in px of the images sent to the model.",
                }),
                "concept_type": (list(DESCRIBE_INSTRUCTIONS), {
                    "default": "identity",
                    "tooltip": "What the images show. This decides what the model is asked to describe.",
                }),
            },
            "optional": {
                "instruction": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional. Leave empty to use the question for the chosen concept_type.",
                }),
                "name_hint": ("STRING", {
                    "default": "",
                    "tooltip": "Optional name or role of the subject, added to the instruction.",
                }),
                "lm_studio_url": ("STRING", {"default": "http://127.0.0.1:1234/v1"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "describe"
    CATEGORY = "VRGDG/RefMod"
    DESCRIPTION = (
        "Sends the reference images to the vision model already loaded in LM Studio and returns a description. "
        "Connect it to the description input of Create H3 RefMod. It never loads or switches LM Studio models."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def describe(self, refs, max_images, image_edge, concept_type, instruction="", name_hint="",
                 lm_studio_url="http://127.0.0.1:1234/v1"):
        if not refs:
            raise ValueError("[VRGDG RefMod] No reference images to describe.")
        images = [_tensor_to_pil(ref) for ref in refs]
        return (describe_images(images, concept_type, instruction, name_hint, lm_studio_url, max_images, image_edge),)


def _refmod_dirs() -> List[str]:
    """Folders RefMod files are read from: the registered refmods folders, models/refmods and the pack's mods folder."""
    dirs = list(folder_paths.get_folder_paths("refmods")) if "refmods" in folder_paths.folder_names_and_paths else []
    dirs.append(os.path.join(folder_paths.models_dir, "refmods"))
    pack_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dirs.append(os.path.join(pack_root, "ComfyUI-MiniMaxH3Mod", "mods"))
    seen, result = set(), []
    for directory in dirs:
        directory = os.path.normpath(directory)
        if directory not in seen and os.path.isdir(directory):
            seen.add(directory)
            result.append(directory)
    return result


def _list_refmod_names() -> List[str]:
    names = set()
    for directory in _refmod_dirs():
        for root, subdirs, files in os.walk(directory):
            subdirs[:] = [name for name in subdirs if name not in ("graph_presets", ".git", "__pycache__")]
            for name in files:
                if name.endswith(".safetensors"):
                    stem = os.path.join(root, name[:-len(".safetensors")])
                    names.add(os.path.relpath(stem, directory).replace("\\", "/"))
    return sorted(names)


def _read_safetensors_header(path: str) -> Dict[str, Any]:
    """The JSON header of a safetensors file. Reads only the header, never the tensors."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > 64 * 1024 * 1024:
            raise ValueError(f"{os.path.basename(path)} does not look like a safetensors file.")
        return json.loads(handle.read(length))


def _find_refmod_file(name: str) -> str:
    name = str(name or "").replace("\\", "/").strip()
    if not name or name.startswith("/") or ".." in name.split("/") or ":" in name:
        raise ValueError(f"[VRGDG RefMod] Choose a RefMod from the list (got '{name}').")
    for directory in _refmod_dirs():
        path = os.path.join(directory, name + ".safetensors")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"[VRGDG RefMod] RefMod '{name}' was not found in: " + ", ".join(_refmod_dirs()))


def _format_metadata(meta: Dict[str, Any], path: str, header: Dict[str, Any]) -> tuple:
    """(report, trigger text) for one RefMod's stored metadata."""
    description = str(meta.get("description") or "").strip()
    concept = str(meta.get("concept_type") or "").strip()
    tags = [str(tag) for tag in (meta.get("tags") or [])]
    trigger = "; ".join(part for part in ((f"{concept}: {description}" if description else ""), ", ".join(tags)) if part)
    tensors = {key: value.get("shape") for key, value in header.items() if key != "__metadata__"}
    lines = [
        f"name:          {meta.get('name', os.path.basename(path))}",
        f"file:          {path}",
        f"kind:          {meta.get('kind', '-')}",
        f"concept_type:  {concept or '-'}",
        f"description:   {description or '(empty)'}",
        f"tags:          {', '.join(tags) or '-'}",
        f"mode:          {meta.get('mode', '-')}",
        f"source:        {meta.get('source', '-')} ({meta.get('source_shape', '-')})",
        f"pool:          {meta.get('pool', '-')}",
        f"refinement:    {meta.get('optimize_steps', '-')}",
        f"tensors:       {json.dumps(tensors)}",
    ]
    if meta.get("config"):
        lines.append(f"saved_config:  {json.dumps(meta['config'])}")
    if not description and not tags:
        lines.append("")
        lines.append("This file stores no description or tags. RefMods keep no trigger words, "
                     "so put the character's description in your prompt.")
    return "\n".join(lines), trigger


class VRGDG_RefModMetadata:
    """Show the metadata stored in a RefMod: description, concept type, tags and layout."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mod_name": (["(use mods input)"] + _list_refmod_names(), {
                    "tooltip": "A saved RefMod to read from disk. Ignored when the mods input is connected.",
                }),
            },
            "optional": {
                "mods": ("H3_REF_MODS", {"tooltip": "Optional: the bundle from Create H3 RefMod or Load H3 RefMods."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metadata", "trigger_text")
    FUNCTION = "read"
    OUTPUT_NODE = True
    CATEGORY = "VRGDG/RefMod"
    DESCRIPTION = (
        "Reads a RefMod's stored description, concept type, tags and layout without loading its tensors. "
        "trigger_text is 'concept_type: description' plus the tags, ready to add to a prompt."
    )

    @classmethod
    def IS_CHANGED(cls, mod_name, mods=None):
        return float("nan")

    def read(self, mod_name, mods=None):
        paths = []
        if mods:
            for mod, _strength in mods:
                path = str(getattr(mod, "path", "") or "")
                path = path if path.endswith(".safetensors") else path + ".safetensors"
                if path not in paths:
                    paths.append(path)
        elif mod_name and mod_name != "(use mods input)":
            paths.append(_find_refmod_file(mod_name))
        else:
            raise ValueError("[VRGDG RefMod] Choose a mod_name or connect mods.")
        reports, triggers = [], []
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[VRGDG RefMod] RefMod file not found: {path}")
            header = _read_safetensors_header(path)
            raw = header.get("__metadata__") or {}
            meta: Optional[Dict[str, Any]] = None
            for key in ("refmod_meta", "audio_refmod_meta"):
                if key in raw:
                    meta = json.loads(raw[key])
                    break
            if meta is None:
                raise ValueError(f"[VRGDG RefMod] {os.path.basename(path)} has no RefMod metadata.")
            report, trigger = _format_metadata(meta, path, header)
            reports.append(report)
            if trigger:
                triggers.append(trigger)
        text = "\n\n".join(reports)
        print("[VRGDG RefMod] Metadata:\n" + text)
        return {"ui": ui.PreviewText(text).as_dict(), "result": (text, "\n".join(triggers))}


class VRGDG_RefModCombine:
    """Join several RefMod bundles into one, keeping their order."""

    MAX_INPUTS = 6

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {f"mods_{i}": ("H3_REF_MODS",) for i in range(1, cls.MAX_INPUTS + 1)}}

    RETURN_TYPES = ("H3_REF_MODS",)
    RETURN_NAMES = ("mods",)
    FUNCTION = "combine"
    CATEGORY = "VRGDG/RefMod"
    DESCRIPTION = (
        "Joins RefMod bundles (loaders, an audio RefMod) into one list, in input order. Connect the result to "
        "Text Encode with RefMods."
    )

    def combine(self, **bundles):
        combined = []
        for index in range(1, self.MAX_INPUTS + 1):
            value = bundles.get(f"mods_{index}")
            if value:
                combined.extend(value)
        if not combined:
            raise ValueError("[VRGDG RefMod] Connect at least one RefMod bundle.")
        return (combined,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_RefModImagePicker": VRGDG_RefModImagePicker,
    "VRGDG_RefModDescribe": VRGDG_RefModDescribe,
    "VRGDG_RefModMetadata": VRGDG_RefModMetadata,
    "VRGDG_RefModCombine": VRGDG_RefModCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_RefModImagePicker": "VRGDG RefMod Image Picker",
    "VRGDG_RefModDescribe": "VRGDG RefMod Describe",
    "VRGDG_RefModMetadata": "VRGDG RefMod Metadata",
    "VRGDG_RefModCombine": "VRGDG RefMod Combine",
}
