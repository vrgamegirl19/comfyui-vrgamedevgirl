import asyncio
import os
import subprocess

from aiohttp import web
from server import PromptServer


V10_BRANCH = "dev/music-video-builder-ui-test-v10"
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROUTES_REGISTERED = False


def _run_git(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_NODE_DIR,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=300,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git was not found. Install Git, then try again.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Git timed out while updating. Check your internet connection and try again.") from exc

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        command = "git " + " ".join(args)
        raise RuntimeError(f"{command} failed:\n{output or 'Git returned an unknown error.'}")
    return output


def _update_to_v10():
    if not os.path.isdir(os.path.join(_NODE_DIR, ".git")):
        raise RuntimeError(
            "This installation is not a Git checkout, so the normal Git update commands cannot run."
        )

    logs = []
    for args in (("fetch", "origin"), ("switch", V10_BRANCH), ("pull", "--ff-only")):
        output = _run_git(*args)
        logs.append({"command": "git " + " ".join(args), "output": output})

    branch = _run_git("branch", "--show-current").strip()
    if branch != V10_BRANCH:
        raise RuntimeError(f"Git finished on '{branch or '(detached HEAD)'}' instead of '{V10_BRANCH}'.")
    return {"branch": branch, "directory": _NODE_DIR, "logs": logs}


def _register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    @PromptServer.instance.routes.post("/vrgdg/update/v10")
    async def vrgdg_update_v10(request):
        try:
            result = await asyncio.to_thread(_update_to_v10)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    _ROUTES_REGISTERED = True


_register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
