import asyncio
import os
import subprocess

from aiohttp import web
from server import PromptServer


V9_BRANCH = "dev/music-video-builder-ui-test-v9"
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
        raise RuntimeError("Git timed out. Check your internet connection and try again.") from exc

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        command = "git " + " ".join(args)
        raise RuntimeError(f"{command} failed:\n{output or 'Git returned an unknown error.'}")
    return output


def _switch_to_v9():
    if not os.path.isdir(os.path.join(_NODE_DIR, ".git")):
        raise RuntimeError("This installation is not a Git checkout, so Git cannot switch branches.")

    logs = []
    for args in (("fetch", "origin"), ("switch", V9_BRANCH), ("pull",)):
        logs.append({"command": "git " + " ".join(args), "output": _run_git(*args)})

    branch = _run_git("branch", "--show-current").strip()
    if branch != V9_BRANCH:
        raise RuntimeError(f"Git finished on '{branch or '(detached HEAD)'}' instead of '{V9_BRANCH}'.")
    return {"branch": branch, "logs": logs}


def _register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    @PromptServer.instance.routes.post("/vrgdg/try_v9_dev")
    async def vrgdg_try_v9_dev(request):
        try:
            result = await asyncio.to_thread(_switch_to_v9)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    _ROUTES_REGISTERED = True


class VRGDG_TryV9Dev:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instructions",)
    FUNCTION = "instructions"
    CATEGORY = "VRGDG/Update"

    def instructions(self):
        return ("Use the button on this node to switch this installation from main to the V9 development branch.",)


_register_routes()

NODE_CLASS_MAPPINGS = {"VRGDG_TryV9Dev": VRGDG_TryV9Dev}
NODE_DISPLAY_NAME_MAPPINGS = {"VRGDG_TryV9Dev": "VRGDG - Try V9 Dev"}
