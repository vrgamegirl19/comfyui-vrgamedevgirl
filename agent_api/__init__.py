"""VRGDG Agent API package for autonomous and assisted music video production."""

from .router import register_agent_api_routes

# Ensure Agent API routes are registered on module import if ComfyUI PromptServer is running
try:
    register_agent_api_routes()
except Exception as _exc:
    print(f"[VRGDG API] Route registration deferred: {_exc}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    "register_agent_api_routes",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
