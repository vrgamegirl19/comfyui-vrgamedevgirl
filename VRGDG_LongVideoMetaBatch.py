"""Long-video upload and VHS meta-batch loader.

The browser uploads small binary chunks to avoid ComfyUI's normal upload-size
limit.  Once assembled on disk, the loader uses Video Helper Suite's normal
streaming loader and preserves its VHS_BatchManager contract.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import folder_paths


_NODE_DIR = Path(__file__).resolve().parent
_UPLOAD_DIR = Path(folder_paths.get_input_directory()) / "vrgdg_long_video_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_SAFE_ID = re.compile(r"^[0-9a-fA-F-]{32,40}$")
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}
_CHUNK_LIMIT = 50 * 1024 * 1024


def _safe_filename(value: str) -> str:
    name = Path(str(value or "video.mp4")).name
    stem, suffix = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "video"
    suffix = suffix.lower() if suffix.lower() in _VIDEO_EXTENSIONS else ".mp4"
    return f"{stem[:100]}{suffix}"


def _upload_paths(upload_id: str):
    if not _SAFE_ID.fullmatch(upload_id):
        raise ValueError("Invalid upload id.")
    return _UPLOAD_DIR / f"{upload_id}.part", _UPLOAD_DIR / f"{upload_id}.json"


def _register_upload_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:  # pragma: no cover - only reached outside ComfyUI
        print(f"[VRGDG] Long-video routes unavailable: {exc}")
        return

    route_flag = "_VRGDG_LONG_VIDEO_ROUTES_REGISTERED"
    if getattr(PromptServer.instance, route_flag, False):
        return

    @PromptServer.instance.routes.post("/vrgdg/long_video/upload")
    async def vrgdg_long_video_upload(request):
        try:
            upload_id = str(request.query.get("upload_id", ""))
            index = int(request.query.get("chunk", "-1"))
            total = int(request.query.get("total", "0"))
            filename = _safe_filename(request.query.get("filename", "video.mp4"))
            if not (0 <= index < total <= 10_000_000):
                raise ValueError("Invalid upload chunk numbering.")
            raw = await request.read()
            if not raw or len(raw) > _CHUNK_LIMIT:
                raise ValueError("Upload chunk is empty or larger than 50 MB.")
            part_path, meta_path = _upload_paths(upload_id)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta["total"] != total or meta["filename"] != filename:
                    raise ValueError("Upload metadata does not match the existing upload.")
            else:
                meta = {"total": total, "next": 0, "filename": filename}
            if index != int(meta["next"]):
                return web.json_response(
                    {"ok": False, "error": f"Expected chunk {meta['next']}, received {index}."},
                    status=409,
                )
            with part_path.open("ab") as handle:
                handle.write(raw)
            meta["next"] = index + 1
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            if meta["next"] == total:
                final_path = _UPLOAD_DIR / f"{upload_id}_{filename}"
                os.replace(part_path, final_path)
                meta_path.unlink(missing_ok=True)
                return web.json_response({"ok": True, "complete": True, "path": str(final_path)})
            return web.json_response({"ok": True, "complete": False, "chunk": index})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    setattr(PromptServer.instance, route_flag, True)


_register_upload_routes()


def _load_vhs():
    """Import VHS without requiring a package name containing a hyphen."""
    vhs_root = _NODE_DIR.parent / "comfyui-videohelpersuite"
    if not vhs_root.is_dir():
        raise RuntimeError("ComfyUI-VideoHelperSuite is required for the long-video loader.")
    root_text = str(vhs_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from videohelpersuite.load_video_nodes import load_video
    return load_video


class VRGDGLongVideoMetaBatchLoader:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            from videohelpersuite.load_video_nodes import get_load_formats
            formats = get_load_formats()
        except Exception:
            formats = (["None"], {"default": "None"})
        return {
            "required": {
                "video": ("STRING", {"default": "", "multiline": False}),
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 120, "step": 1}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 10_000_000}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": formats,
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")
    FUNCTION = "load"
    CATEGORY = "VRGDG/Video/Meta Batch"

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        path = Path(str(video or "").strip().strip('"'))
        try:
            stat = path.stat()
            return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return f"missing:{path}"

    def load(self, video, force_rate, custom_width, custom_height, frame_load_cap,
             skip_first_frames, select_every_nth, meta_batch=None, vae=None,
             format="None", prompt=None, unique_id=None):
        raw_path = str(video or "").strip().strip('"').strip("'")
        path = Path(os.path.expandvars(os.path.expanduser(raw_path))).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Long-video source does not exist: {path}")
        load_video = _load_vhs()
        return load_video(
            video=str(path), force_rate=force_rate, custom_width=custom_width,
            custom_height=custom_height, frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames, select_every_nth=select_every_nth,
            meta_batch=meta_batch, vae=vae, format=format,
            unique_id=unique_id,
        )


NODE_CLASS_MAPPINGS = {
    "VRGDGLongVideoMetaBatchLoader": VRGDGLongVideoMetaBatchLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDGLongVideoMetaBatchLoader": "VRGDG Long Video Upload + Meta Batch",
}
