"""Keeps the browser from serving stale Video Builder modules after an update.

ComfyUI marks ``.js`` and ``.css`` responses ``no-store`` but leaves ``.mjs`` alone, so Chrome keeps the Builder's
modules for a guessed time (a share of the file's age, which can be hours). After an update it can run a new module
next to an old cached one: the Builder throws on start, or the side panel tabs are blank, stuck or not clickable.

This adds ``Cache-Control: no-cache`` to this pack's ``.mjs`` files, which makes the browser revalidate them each
time through the ETag. An unchanged file costs one 304. Copies a browser cached before this rule existed stay until
they expire or the cache is cleared once (DevTools > Network > Disable cache, then reload).
"""

from pathlib import Path
from typing import Awaitable, Callable

from aiohttp import web
from server import PromptServer

_EXTENSION_PREFIX = f"/extensions/{Path(__file__).resolve().parents[1].name}/"


@web.middleware
async def revalidate_builder_modules(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    """Mark this pack's ES modules as always-revalidate."""
    response = await handler(request)
    if request.path.startswith(_EXTENSION_PREFIX) and request.path.endswith(".mjs"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def register() -> bool:
    """Add the middleware to the running ComfyUI server. Returns False when the server cannot take it."""
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return False
    try:
        if revalidate_builder_modules not in server.app.middlewares:
            server.app.middlewares.append(revalidate_builder_modules)
    except RuntimeError as exc:  # the app was already started and froze its middleware list
        print(f"[VRGDG Cache] Could not add the module cache rule: {exc}")
        return False
    return True


register()
