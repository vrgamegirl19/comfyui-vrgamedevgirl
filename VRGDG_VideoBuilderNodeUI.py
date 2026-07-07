try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None


class VRGDG_VideoBuilderNodeCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "notes": (
                    "STRING",
                    {
                        "default": "Standalone node canvas prototype. Use the button on this node to open it.",
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("notes",)
    FUNCTION = "passthrough"
    CATEGORY = "VRGDG/Video Builder"

    def passthrough(self, notes):
        return (notes,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_VideoBuilderNodeCanvas": VRGDG_VideoBuilderNodeCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_VideoBuilderNodeCanvas": "VRGDG Video Builder Node Canvas",
}


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
