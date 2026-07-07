try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _json_response(payload):
    if web is None:
        return None
    return web.json_response(payload)


if PromptServer is not None and web is not None:

    @PromptServer.instance.routes.get("/vrgdg/node_canvas/status")
    async def vrgdg_node_canvas_status(request):
        return _json_response(
            {
                "ok": True,
                "name": "VRGDG Node Canvas Prototype",
                "version": 1,
                "builder_connected": False,
            }
        )

