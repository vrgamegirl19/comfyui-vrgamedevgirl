import asyncio

from aiohttp import web
from server import PromptServer

from ..builder.project import _BUILDER_SAVE_LOCK
from .dialogue_scenes import _build_id_lora_dialogue_scenes, _build_minimax_dialogue_scenes
from .persistence import (
    StoryboardConflictError,
    _export_storyboard_prompts,
    _import_storyboard_reference_image,
    _load_storyboard,
    _save_storyboard,
)
from .scene_prompts import _build_storyboard_image_prompt, _build_storyboard_video_prompt
from .story_book import draft_story_book, rewrite_story_chapter, scan_story_cast, split_pasted_story
from .story_layer import (
    StoryArcFormatError,
    _build_story_layer_arc,
    _build_story_layer_brief,
    _build_story_layer_scene_beat,
)


_VRGDG_STORYBOARD_ROUTES_REGISTERED = False

# Story Mode steps behind /vrgdg/storyboard/story_mode/{action}. Each returns the story layer fields to save.
STORY_MODE_ACTIONS = {
    "book": draft_story_book,
    "chapter": rewrite_story_chapter,
    "split": split_pasted_story,
    "cast": scan_story_cast,
}


def _ensure_storyboard_routes():
    global _VRGDG_STORYBOARD_ROUTES_REGISTERED
    if _VRGDG_STORYBOARD_ROUTES_REGISTERED:
        return
    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.post("/vrgdg/storyboard/load")
    async def vrgdg_storyboard_load(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_load_storyboard, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, "storyboard": result})

    @server_instance.routes.post("/vrgdg/storyboard/save")
    async def vrgdg_storyboard_save(request):
        def save_locked(payload):
            # Agent API scene edits write storyboard.json under the same lock.
            with _BUILDER_SAVE_LOCK:
                return _save_storyboard(payload)

        try:
            payload = await request.json()
            result = await asyncio.to_thread(save_locked, payload)
        except StoryboardConflictError as exc:
            return web.json_response(
                {"ok": False, "conflict": True, "error": str(exc), "current_revision": exc.current_revision},
                status=409,
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, "storyboard": result})

    @server_instance.routes.post("/vrgdg/storyboard/import_reference_image")
    async def vrgdg_storyboard_import_reference_image(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_import_storyboard_reference_image, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/export_prompts")
    async def vrgdg_storyboard_export_prompts(request):
        def export_locked(payload):
            with _BUILDER_SAVE_LOCK:
                return _export_storyboard_prompts(payload)

        try:
            payload = await request.json()
            result = await asyncio.to_thread(export_locked, payload)
        except StoryboardConflictError as exc:
            return web.json_response(
                {"ok": False, "conflict": True, "error": str(exc), "current_revision": exc.current_revision},
                status=409,
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/gemma_video_prompt")
    async def vrgdg_storyboard_gemma_video_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_storyboard_video_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/gemma_image_prompt")
    async def vrgdg_storyboard_gemma_image_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_storyboard_image_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/story_brief")
    async def vrgdg_storyboard_story_brief(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_story_layer_brief, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/story_arc")
    async def vrgdg_storyboard_story_arc(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_story_layer_arc, payload)
        except StoryArcFormatError as exc:
            return web.json_response({
                "ok": False,
                "error": str(exc),
                "diagnostics": {
                    "kind": "story_arc_format",
                    "runner": exc.runner,
                    "expected_sections": exc.expected_sections,
                    "raw_output": exc.raw_output,
                    "cleaned_output": exc.cleaned_output,
                },
            }, status=500)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/story_mode/{action}")
    async def vrgdg_storyboard_story_mode(request):
        action = STORY_MODE_ACTIONS.get(request.match_info.get("action", ""))
        if action is None:
            choices = ", ".join(STORY_MODE_ACTIONS)
            return web.json_response({"ok": False, "error": f"Unknown Story Mode step. Use one of: {choices}."}, status=400)
        try:
            payload = await request.json()
            result = await asyncio.to_thread(action, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/scene_story_beat")
    async def vrgdg_storyboard_scene_story_beat(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_story_layer_scene_beat, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/id_lora_dialogue_scenes")
    async def vrgdg_storyboard_id_lora_dialogue_scenes(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_id_lora_dialogue_scenes, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/storyboard/minimax_dialogue_scenes")
    async def vrgdg_storyboard_minimax_dialogue_scenes(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_build_minimax_dialogue_scenes, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    _VRGDG_STORYBOARD_ROUTES_REGISTERED = True


class VRGDG_StoryboardBuilderUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_folder": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("project_folder",)
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Storyboard planning UI for organizing scene prompts before image/video creation."

    def noop(self, project_folder):
        return (project_folder,)


_ensure_storyboard_routes()


NODE_CLASS_MAPPINGS = {
    "VRGDG_StoryboardBuilderUI": VRGDG_StoryboardBuilderUI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_StoryboardBuilderUI": "VRGDG Storyboard Builder UI",
}
