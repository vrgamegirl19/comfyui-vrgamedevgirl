"""Uniform JSON envelope helpers for VRGDG Agent API (D6)."""

from typing import Any, Dict, Optional
from aiohttp import web

from .errors import AgentApiError, INTERNAL_ERROR


def api_success(data: Any = None, revision: Optional[int] = None, status: int = 200, **kwargs) -> web.Response:
    """Build a standard success response: {"ok": true, "data": {...}, "revision": n, ...}"""
    body: Dict[str, Any] = {"ok": True}
    if data is not None:
        body["data"] = data
    if revision is not None:
        body["revision"] = revision
    body.update(kwargs)
    return web.json_response(body, status=status)


def api_error(
    code: str = INTERNAL_ERROR,
    message: str = "An unexpected error occurred.",
    details: Optional[Dict[str, Any]] = None,
    status: int = 400,
    retryable: bool = False,
) -> web.Response:
    """Build a standard failure response: {"ok": false, "error": {"code": ..., "message": ..., ...}}"""
    return web.json_response(
        {
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "retryable": retryable,
            },
        },
        status=status,
    )


def api_exception(exc: Exception) -> web.Response:
    """Map any exception to a standard API error response."""
    if isinstance(exc, AgentApiError):
        return web.json_response(
            {"ok": False, "error": exc.to_dict()},
            status=exc.status,
        )
    return api_error(
        code=INTERNAL_ERROR,
        message=str(exc) or "Internal server error.",
        status=500,
        retryable=False,
    )
