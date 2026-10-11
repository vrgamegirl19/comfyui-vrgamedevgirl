"""Project log HTTP boundary and project-scoped backend failure capture."""

import asyncio
import json
import logging
import os
import traceback
from urllib.parse import unquote

from aiohttp import web

from .error_log import append_project_error, project_error_log, secret_values
from .paths import _model_defaults_path


_LOGGER = logging.getLogger(__name__)
_LOG_ROUTE = "/vrgdg/music_builder/error_log"


def _credentials(payload: dict) -> list[str]:
    values = secret_values(payload)
    path = _model_defaults_path()
    if os.path.isfile(path):
        with open(path, encoding="utf-8-sig") as handle:
            values.extend(secret_values(json.load(handle)))
    return values


def _append(project: str, entry: dict, payload: dict) -> bool:
    try:
        append_project_error(project, entry, _credentials(payload))
        return True
    except (OSError, ValueError, TypeError):
        _LOGGER.warning("[VRGDG Builder] Could not write the project error log.")
        return False


async def _request_payload(request: web.Request) -> dict:
    if request.content_type == "application/json" and request.can_read_body:
        try:
            data = await request.json()
            if isinstance(data, dict):
                return data
        except (ValueError, web.HTTPException):
            pass
    return {}


async def _record_failure(request: web.Request, entry: dict) -> bool:
    payload = await _request_payload(request)
    project = str(payload.get("project_folder") or unquote(request.headers.get("X-VRGDG-Project", "")))
    if not project and request.match_info.get("pid"):
        # Agent API imports Builder routes while initializing its package.
        from ..agent_api.paths import resolve_project_folder
        from ..agent_api.errors import AgentApiError
        try:
            project = await asyncio.to_thread(resolve_project_folder, request.match_info["pid"])
        except (OSError, ValueError, AgentApiError):
            pass
    if not project:
        return False
    try:
        context = json.loads(unquote(request.headers.get("X-VRGDG-Error-Context", "{}")))
        if isinstance(context, dict):
            entry.update({key: context[key] for key in ("scene", "mode") if key in context})
    except ValueError:
        pass
    for key in ("scene", "scene_number", "scene_id"):
        if payload.get(key) is not None:
            entry["scene"] = payload[key]
            break
    entry["mode"] = (payload.get("video_prompt_type") or payload.get("mode")
                     or payload.get("project_video_engine") or entry.get("mode"))
    entry["component"] = request.path
    entry["stage"] = request.path.rsplit("/", 1)[-1]
    return await asyncio.to_thread(_append, project, entry, payload)


@web.middleware
async def project_error_middleware(request: web.Request, handler) -> web.StreamResponse:
    """Log project API failures without changing response bodies or execution behavior."""
    if not request.path.startswith("/vrgdg/") or request.path.startswith(_LOG_ROUTE):
        return await handler(request)
    try:
        response = await handler(request)
    except Exception as error:
        await _record_failure(request, {
            "source": "Backend", "message": str(error), "stack": traceback.format_exc(),
        })
        raise
    if not isinstance(response, web.Response) or response.content_type != "application/json":
        return response
    initialize = request.path.endswith(("/new_project", "/save_session", "/load_session"))
    if response.status < 400 and not initialize and b'"ok": false' not in (response.body or b""):
        return response
    try:
        result = json.loads(response.body)
    except (ValueError, TypeError):
        return response
    if not isinstance(result, dict):
        return response
    if response.status >= 400 or result.get("ok") is False:
        error = result.get("error") or f"HTTP {response.status}"
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        logged = await _record_failure(request, {
            "source": "API" if "/api/v1/" in request.path else "Backend", "message": message,
            "details": f"HTTP {response.status}\n{message}",
        })
        if logged:
            response.headers["X-VRGDG-Error-Logged"] = "1"
    elif initialize and result.get("project_folder"):
        try:
            await asyncio.to_thread(project_error_log, result["project_folder"], False, False)
        except (OSError, ValueError, TypeError):
            _LOGGER.warning("[VRGDG Builder] Could not create the project error log.")
    return response


def register_error_log_routes(server) -> None:
    """Register fixed-path local read, append and clear operations."""
    server.app.middlewares.append(project_error_middleware)

    @server.routes.post(_LOG_ROUTE)
    async def error_log(request: web.Request) -> web.Response:
        """Read/create the saved project's error log."""
        try:
            payload = await _request_payload(request)
            result = await asyncio.to_thread(project_error_log, payload.get("project_folder"))
        except (OSError, ValueError, TypeError) as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400)
        return web.json_response({"ok": True, **result})

    @server.routes.post(_LOG_ROUTE + "/append")
    async def append_error_log(request: web.Request) -> web.Response:
        """Append a redacted browser error without accepting a log filename."""
        try:
            payload = await _request_payload(request)
            result = await asyncio.to_thread(
                append_project_error, payload.get("project_folder"), payload.get("entry", {}),
                await asyncio.to_thread(_credentials, payload),
            )
        except (OSError, ValueError, TypeError) as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400)
        return web.json_response({"ok": True, **result})

    @server.routes.post(_LOG_ROUTE + "/clear")
    async def clear_error_log(request: web.Request) -> web.Response:
        """Explicitly clear the project's error report, retaining its header."""
        try:
            payload = await _request_payload(request)
            result = await asyncio.to_thread(project_error_log, payload.get("project_folder"), True)
        except (OSError, ValueError, TypeError) as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400)
        return web.json_response({"ok": True, **result})
