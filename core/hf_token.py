"""The Hugging Face token the Models dialog downloads with.

It lets the installer fetch the files Hugging Face keeps behind a login (the Flux Klein 9B diffusion model), and Hugging Face counts the
downloads against the account instead of the shared limit of anonymous ones. H3 Studio keeps the same token in its App Settings.

It is saved in ``hf_token.json`` next to the LLM keys (``{"hf_token": "hf_..."}``), a file of its own that no project export carries. No route
returns it and nothing logs it. The installer sends it only to huggingface.co (``model_downloads._token_for``).
"""

import json
import os
import urllib.error
import urllib.request

from ..builder.paths import _hf_token_path
from ..llm.http_tls import urlopen_verified
from .atomic_write import atomic_write_json

HF_BASE = "https://huggingface.co"
_MAX_TOKEN_CHARS = 400
_NOT_A_TOKEN = "That does not look like a Hugging Face token. It starts with hf_. Paste the whole token without spaces."


def checked_token(raw) -> str:
    """``raw`` without the spaces around it. An empty string stays empty (it clears the saved token).

    Raises ``ValueError`` for anything that is not shaped like a token. That also keeps a line break out of the Authorization header.
    """
    token = str(raw if raw is not None else "").strip()
    if token and (not token.startswith("hf_") or any(char.isspace() for char in token) or len(token) > _MAX_TOKEN_CHARS):
        raise ValueError(_NOT_A_TOKEN)
    return token


def load_token() -> str:
    """The saved token, or an empty string when none is saved (or the file was edited into something that is not a token)."""
    try:
        with open(_hf_token_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return checked_token(data.get("hf_token")) if isinstance(data, dict) else ""
    except (OSError, ValueError):
        return ""


def is_saved() -> bool:
    return bool(load_token())


def save_token(raw) -> bool:
    """Save the token, or clear it with an empty string. Returns whether a token is saved now."""
    token = checked_token(raw)
    path = _hf_token_path()
    if token:
        atomic_write_json(path, {"hf_token": token})
    else:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    return bool(token)


def whoami(token: str) -> str:
    """The Hugging Face account name ``token`` belongs to. Raises ``ValueError`` with something to tell the person when it does not work."""
    request = urllib.request.Request(
        f"{HF_BASE}/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "comfyui-vrgamedevgirl-model-installer"},
    )
    try:
        with urlopen_verified(request, timeout=20) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError("Hugging Face did not accept that token. Check that you copied the whole token and that it was not revoked.") from None
        raise ValueError(f"Hugging Face answered {exc.code}.") from None
    except (urllib.error.URLError, OSError) as exc:
        raise ValueError(f"Could not reach Hugging Face: {getattr(exc, 'reason', None) or exc}") from None
    try:
        account = json.loads(raw or b"{}")
    except ValueError:
        account = {}
    return str(account.get("name") or "") if isinstance(account, dict) else ""


def verify(sent="") -> str:
    """Ask Hugging Face whose token this is: the one in ``sent`` when there is one, else the saved one. Gives the account name."""
    token = checked_token(sent) or load_token()
    if not token:
        raise ValueError("No Hugging Face token is saved. Paste one and try again.")
    return whoami(token)
