import os
import re
import base64
from io import BytesIO

from aiohttp import web
from PIL import Image
from server import PromptServer


_VRGDG_VIDEO_EDITOR_ROUTES_REGISTERED = False


def _image_from_data_url(data_url):
    text = str(data_url or "").strip()
    match = re.match(r"^data:image/(?:png|jpeg|jpg|webp);base64,(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Expected a base64 image data URL.")
    raw = base64.b64decode(match.group(1))
    return Image.open(BytesIO(raw)).convert("RGB")


def _ensure_video_editor_routes():
    global _VRGDG_VIDEO_EDITOR_ROUTES_REGISTERED
    if _VRGDG_VIDEO_EDITOR_ROUTES_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.get("/vrgdg/video_editor/video")
    async def vrgdg_video_editor_video(request):
        raw_path = str(request.query.get("path", "") or "").strip()
        video_path = os.path.normpath(os.path.abspath(raw_path))
        if not os.path.isfile(video_path):
            return web.json_response({"ok": False, "error": "Video file was not found."}, status=404)
        response = web.FileResponse(video_path)
        # The client versions this URL with the scene's video_cache_bust, which only
        # changes when the scene is re-rendered, so the same URL always means the
        # same bytes. Mark it immutable so the browser reuses a preloaded/played
        # clip from cache on the next cut instead of re-fetching it from disk.
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @server_instance.routes.get("/vrgdg/video_editor/image")
    async def vrgdg_video_editor_image(request):
        raw_path = str(request.query.get("path", "") or "").strip()
        image_path = os.path.normpath(os.path.abspath(raw_path))
        if not os.path.isfile(image_path):
            return web.json_response({"ok": False, "error": "Image file was not found."}, status=404)
        if os.path.splitext(image_path)[1].lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return web.json_response({"ok": False, "error": "Unsupported image file type."}, status=400)
        response = web.FileResponse(image_path)
        if str(request.query.get("thumbv", "") or "").strip():
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    _VRGDG_VIDEO_EDITOR_ROUTES_REGISTERED = True


_ensure_video_editor_routes()


NODE_CLASS_MAPPINGS = {

}

NODE_DISPLAY_NAME_MAPPINGS = {

}
