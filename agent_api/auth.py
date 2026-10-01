"""Authentication and token verification for the VRGDG Agent API (4.1)."""

import json
import os
import secrets
from typing import Any, Dict
from aiohttp import web
import folder_paths

from ..core.atomic_write import atomic_write_json
from .errors import AuthError, AUTH_DISABLED


_AUTH_CONFIG_FILE = "agent_api.json"


def _auth_config_path() -> str:
    folder = os.path.join(folder_paths.get_output_directory(), "VRGDG_Model_Defaults")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, _AUTH_CONFIG_FILE)


def load_auth_config() -> Dict[str, Any]:
    path = _auth_config_path()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    # Initialize default configuration on first access
    token = secrets.token_hex(24)
    config = {
        "enabled": True,
        "require_token_on_loopback": False,
        "token": token,
        "allowed_roots": [],
    }
    try:
        atomic_write_json(path, config)
    except Exception as exc:
        print(f"[VRGDG API] Warning: could not persist default auth config: {exc}")
    return config


def is_loopback(request: web.Request) -> bool:
    """Check if the HTTP request originated from local loopback."""
    remote = request.remote
    if not remote:
        return True
    return remote in ("127.0.0.1", "::1", "localhost", "testclient")


def verify_auth(request: web.Request, config: Dict[str, Any] = None) -> None:
    """Verify authorization header. Raises AuthError on failure."""
    if config is None:
        config = load_auth_config()

    if not config.get("enabled", True):
        raise AuthError("Agent API is currently disabled in configuration.", code=AUTH_DISABLED)

    # Allow unauthenticated loopback requests if not explicitly required
    if is_loopback(request) and not config.get("require_token_on_loopback", False):
        return

    expected_token = str(config.get("token", "") or "").strip()
    if not expected_token:
        # If no token is configured, loopback is required
        if is_loopback(request):
            return
        raise AuthError("No API token is configured on the server.")

    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header:
        raise AuthError("Missing 'Authorization: Bearer <token>' header.")

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Authorization header must use Bearer scheme ('Bearer <token>').")

    provided_token = parts[1].strip()
    if not secrets.compare_digest(provided_token, expected_token):
        raise AuthError("Invalid authentication token.")
