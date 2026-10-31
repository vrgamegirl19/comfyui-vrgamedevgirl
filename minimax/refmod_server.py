"""Share RefMods through the H3 RefMod Server (hosted at https://aidigitalcreations.com, not part of this repository).

The RefMods Viewer lists what other people shared under *Server RefMods*, sends one of your own with *Share to server*, adds
a shared one to your library with *Add to my library* and reports one that should not be there. The browser never talks to the
server: these routes do, so the file, the picture and the sign-in token stay on this computer.

The server needs a sign-in (GitHub or Discord). ``signin_start`` gives the Viewer an address to open in the browser and
``signin_poll`` waits until the person has finished. Nothing secret ships with this code: the server keeps its own secrets and
gives each person a personal token, which is stored with the server address it came from in
``VRGDG_Model_Defaults/refmod_server.json`` in the ComfyUI output folder and is only ever sent to that address. H3 Studio keeps
the same settings in its own config folder and has the same routes, so the Viewer behaves the same in both.
"""

import asyncio
import hashlib
import json
import os
import re
import threading
from typing import Any, Dict, Optional, Tuple

import folder_paths
import requests

from ..core.atomic_write import atomic_write_json
from .refmod_library import PREVIEW_SUFFIX, _entry, find_refmod, list_refmods
from .refmod_picker import _refmod_dirs

# The public RefMod server. Behind HTTPS on port 443, its own port 8787 is not open to the internet.
DEFAULT_SERVER_URL = "https://aidigitalcreations.com"
SETTINGS_FILE = "refmod_server.json"
SETTINGS_KEYS = ("url", "token", "token_url", "account")
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
MAX_PREVIEW_BYTES = 8 * 1024 * 1024
LIST_LIMIT = 2000
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 60
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")
_settings_lock = threading.Lock()
# The sign-in the Viewer started and has not finished. The secret the server gave for it stays here, not in the browser.
_pending: Dict[str, Any] = {}
_pending_lock = threading.Lock()


class ServerError(Exception):
    """The server said no, or could not be reached. ``status`` is the HTTP status the Viewer's route answers with and ``code``
    is the server's machine-readable reason (``signin_required``, ``banned``, ``signin_failed``, ``signin_expired``)."""

    def __init__(self, message: str, status: int = 502, code: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code


def normalize_server_url(value: Any) -> str:
    """``http(s)://host[:port]`` without a trailing slash or ``/api/v1``, or ``""`` when the text is not a web address."""
    text = str(value or "").strip().rstrip("/")
    if text.endswith("/api/v1"):
        text = text[: -len("/api/v1")].rstrip("/")
    return text if re.match(r"^https?://[^\s/]+(/[^\s]*)?$", text) else ""


def _settings_path() -> str:
    folder = os.path.join(folder_paths.get_output_directory(), "VRGDG_Model_Defaults")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, SETTINGS_FILE)


def _read_settings() -> Dict[str, str]:
    try:
        with open(_settings_path(), "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    data = data if isinstance(data, dict) else {}
    return {key: str(data.get(key) or "") for key in SETTINGS_KEYS}


def _write_settings(settings: Dict[str, str]) -> None:
    atomic_write_json(_settings_path(), {key: settings.get(key, "") for key in SETTINGS_KEYS})


def server_url(settings: Optional[Dict[str, str]] = None) -> str:
    """The address in the settings, else ``H3_REFMOD_SERVER``, else the default."""
    settings = settings if settings is not None else _read_settings()
    return normalize_server_url(settings.get("url")) or normalize_server_url(os.environ.get("H3_REFMOD_SERVER")) or DEFAULT_SERVER_URL


def _signed_in(settings: Dict[str, str]) -> bool:
    """A token is only good for the server that gave it."""
    return bool(settings["token"]) and settings["token_url"] == server_url(settings)


def get_settings() -> Dict[str, Any]:
    settings = _read_settings()
    signed_in = _signed_in(settings)
    return {"url": server_url(settings), "default_url": DEFAULT_SERVER_URL, "signed_in": signed_in, "account": settings["account"] if signed_in else ""}


def save_settings(url: Any = None) -> Dict[str, Any]:
    """Change the server address. An empty address goes back to the default. A different server needs a new sign-in."""
    with _settings_lock:
        settings = _read_settings()
        if url is not None:
            text = str(url).strip()
            if text and not normalize_server_url(text):
                raise ValueError("The server address must start with http:// or https://.")
            settings["url"] = normalize_server_url(text)
            if not _signed_in(settings):
                settings["token"] = settings["token_url"] = settings["account"] = ""
        _write_settings(settings)
    return get_settings()


def _forget_token() -> None:
    with _settings_lock:
        settings = _read_settings()
        settings["token"] = settings["token_url"] = settings["account"] = ""
        _write_settings(settings)


def clean_type(text: Any) -> str:
    """A RefMod type that is safe as a folder name: lowercase letters, numbers, ``_`` and ``-``."""
    kind = re.sub(r"[^a-z0-9_-]", "_", str(text or "").strip().lower())[:40]
    return kind if re.search(r"[a-z0-9]", kind) else "generic"


def clean_stem(text: Any) -> str:
    """A RefMod name that is safe as a file name. The server's own cleaning is not relied on."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(text or "")).strip(" .")[:80].strip(" .")
    return name or "shared"


def _check_id(refmod_id: Any) -> str:
    text = str(refmod_id or "")
    if not _ID_PATTERN.match(text):
        raise ValueError("That is not a RefMod id from the server.")
    return text


def _request(method: str, path: str, *, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, str]] = None,
             files: Optional[Dict[str, Tuple[str, bytes, str]]] = None, json_body: Optional[Dict[str, Any]] = None,
             limit: int = 0, auth: bool = True) -> Tuple[requests.Response, bytes]:
    """Ask the server. Returns the response and its body. Raises ``ServerError`` with the server's own message for an error.

    With ``auth`` the sign-in token goes along, and the call stops here when there is none."""
    settings = _read_settings()
    base = server_url(settings)
    headers = {}
    if auth:
        if not _signed_in(settings):
            raise ServerError("Sign in to use the server library.", 401, "signin_required")
        headers["Authorization"] = "Bearer " + settings["token"]
    try:
        response = requests.request(
            method, base + path, params=params, data=data, files=files, json=json_body, headers=headers,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True, allow_redirects=False)
    except requests.RequestException:
        raise ServerError(f"Could not reach the RefMod server at {base}. Is it running, and is the address right?")
    with response:
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=1024 * 256):
            chunks.append(chunk)
            size += len(chunk)
            if limit and size > limit:
                raise ServerError("The server sent a file that is too large.")
        body = b"".join(chunks)
        if response.status_code >= 400:
            message, code = "", ""
            try:
                answer = json.loads(body.decode("utf-8"))
                message, code = str(answer.get("error") or ""), str(answer.get("code") or "")
            except (ValueError, AttributeError):
                pass
            if auth and response.status_code == 401 and code == "signin_required":
                _forget_token()  # the server no longer knows this token
            raise ServerError(message or f"The RefMod server answered {response.status_code}.", response.status_code if response.status_code < 500 else 502, code)
        if 300 <= response.status_code < 400:
            raise ServerError(f"The RefMod server at {base} redirects somewhere else. Use its final address in the server settings.")
        return response, body


def _json(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    _, body = _request(method, path, limit=16 * 1024 * 1024, **kwargs)
    try:
        result = json.loads(body.decode("utf-8"))
    except ValueError:
        raise ServerError("That address did not answer like a RefMod server.")
    if not isinstance(result, dict) or not result.get("ok"):
        raise ServerError(str((result or {}).get("error") or "That address did not answer like a RefMod server."))
    return result


def sign_in_options() -> Dict[str, Any]:
    """The ways the server offers to sign in (``github``, ``discord``), asked without a token."""
    result = _json("GET", "/api/v1/health", auth=False)
    return {"url": server_url(), "providers": [str(name) for name in result.get("sign_in") or []]}


def signin_start(provider: Any) -> Dict[str, Any]:
    """Begin signing in. Returns the address to open in the person's browser."""
    base = server_url()
    result = _json("POST", "/api/v1/auth/start", json_body={"provider": str(provider or "")}, auth=False)
    if not result.get("url") or not result.get("poll"):
        raise ServerError("That address did not answer like a RefMod server.")
    with _pending_lock:
        _pending.clear()
        _pending.update({"poll": str(result["poll"]), "url": base})
    return {"url": str(result["url"]), "interval": int(result.get("interval") or 2), "expires_in": int(result.get("expires_in") or 600)}


def signin_poll() -> Dict[str, Any]:
    """Ask whether the person has finished signing in. Saves the token once they have."""
    with _pending_lock:
        pending = dict(_pending)
    if not pending:
        raise ValueError("Start the sign-in first.")
    if pending["url"] != server_url():
        with _pending_lock:
            _pending.clear()
        raise ValueError("The server address changed. Start the sign-in again.")
    try:
        result = _json("POST", "/api/v1/auth/poll", json_body={"poll": pending["poll"]}, auth=False)
    except ServerError as exc:
        if exc.code in ("signin_failed", "signin_expired"):
            with _pending_lock:
                _pending.clear()
        raise
    if result.get("status") != "signed_in":
        return {"status": "waiting"}
    with _pending_lock:
        _pending.clear()
    with _settings_lock:
        settings = _read_settings()
        settings.update({"token": str(result.get("token") or ""), "token_url": pending["url"], "account": str(result.get("name") or "")})
        _write_settings(settings)
    print(f"[VRGDG RefMod] Signed in to {pending['url']} as {result.get('name')}")
    return {"status": "signed_in", **get_settings()}


def sign_out() -> Dict[str, Any]:
    """Forget the token here and ask the server to end it."""
    try:
        _request("POST", "/api/v1/auth/signout", limit=1024 * 1024)
    except ServerError:
        pass  # the token is forgotten here either way
    _forget_token()
    return get_settings()


def list_server_refmods() -> Dict[str, Any]:
    result = _json("GET", "/api/v1/refmods", params={"limit": LIST_LIMIT})
    return {"url": server_url(), "account": get_settings()["account"], "refmods": result.get("refmods") or [], "types": result.get("types") or []}


def server_preview(refmod_id: Any) -> bytes:
    _, body = _request("GET", f"/api/v1/refmods/{_check_id(refmod_id)}/preview", limit=MAX_PREVIEW_BYTES)
    if not body.startswith(PNG_SIGNATURE):
        raise ServerError("The server sent a preview that is not a PNG picture.")
    return body


def upload_refmod(name: Any) -> Dict[str, Any]:
    """Send one of your RefMods (and its preview) to the server. It is shared under the name of your account."""
    entry = find_refmod(str(name or ""))
    if not entry:
        raise ValueError(f"No RefMod named '{name}'.")
    if entry["size"] > MAX_UPLOAD_BYTES:
        raise ValueError(f"'{entry['name']}' is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB the server takes.")
    with open(entry["path"], "rb") as handle:
        payload = handle.read()
    files = {"file": (os.path.basename(entry["path"]), payload, "application/octet-stream")}
    preview = os.path.splitext(entry["path"])[0] + PREVIEW_SUFFIX
    if os.path.isfile(preview) and os.path.getsize(preview) <= MAX_PREVIEW_BYTES:
        with open(preview, "rb") as handle:
            files["preview"] = (os.path.basename(preview), handle.read(), "image/png")
    result = _json("POST", "/api/v1/refmods", data={"name": entry["name"]}, files=files)
    print(f"[VRGDG RefMod] Shared {entry['name']} with {server_url()}")
    return {"refmod": result.get("refmod"), "duplicate": bool(result.get("duplicate")), "replaced": bool(result.get("replaced")),
            "status": str(result.get("status") or "approved")}


def delete_server_refmod(refmod_id: Any) -> Dict[str, Any]:
    """Take one of your uploads off the server."""
    _json("DELETE", f"/api/v1/refmods/{_check_id(refmod_id)}")
    return {"id": refmod_id}


def report_server_refmod(refmod_id: Any, reason: Any = "") -> Dict[str, Any]:
    """Tell the server a shared RefMod should not be there."""
    _json("POST", f"/api/v1/refmods/{_check_id(refmod_id)}/report", json_body={"reason": " ".join(str(reason or "").split())[:200]})
    return {"id": refmod_id}


def _library_root() -> str:
    """Where a RefMod added from the server is saved: the first RefMod folder that is not the MiniMaxH3Mod pack's own."""
    for directory in _refmod_dirs():
        if os.path.basename(os.path.dirname(directory)) != "ComfyUI-MiniMaxH3Mod":
            return directory
    root = os.path.join(folder_paths.models_dir, "refmods")
    os.makedirs(root, exist_ok=True)
    return root


def _same_file(path: str, digest: str) -> bool:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest() == digest


def _write_beside(path: str, content: bytes) -> None:
    temporary = f"{path}.part"
    with open(temporary, "wb") as handle:
        handle.write(content)
    os.replace(temporary, path)


def download_refmod(refmod_id: Any) -> Dict[str, Any]:
    """Add a shared RefMod to your library. Nothing of yours is overwritten: a taken name gets ``(2)``, ``(3)`` and so on."""
    refmod_id = _check_id(refmod_id)
    meta = _json("GET", f"/api/v1/refmods/{refmod_id}")["refmod"]
    kind, stem = clean_type(meta.get("type")), clean_stem(meta.get("stem"))
    _, payload = _request("GET", f"/api/v1/refmods/{refmod_id}/file", limit=MAX_DOWNLOAD_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    if not digest.startswith(refmod_id):
        raise ServerError("The file from the server is damaged. Try again.")
    preview = None
    if meta.get("has_preview"):
        try:
            preview = server_preview(refmod_id)
        except ServerError:
            preview = None

    root = _library_root()
    folder = os.path.join(root, kind)
    for existing in list_refmods(kind):
        if _same_file(existing["path"], digest):
            return {"name": existing["name"], "already": True, "renamed": False}
    os.makedirs(folder, exist_ok=True)
    chosen, number = stem, 2
    while find_refmod(f"{kind}/{chosen}") or os.path.exists(os.path.join(folder, chosen + ".safetensors")):
        chosen, number = f"{stem} ({number})", number + 1
    target = os.path.join(folder, chosen + ".safetensors")
    _write_beside(target + ".check", payload)
    try:
        if not _entry(f"{kind}/{chosen}", target + ".check", root):
            raise ServerError("The file from the server is not a picture or video RefMod.")
        os.replace(target + ".check", target)
    finally:
        if os.path.exists(target + ".check"):
            os.remove(target + ".check")
    if preview:
        _write_beside(os.path.join(folder, chosen + PREVIEW_SUFFIX), preview)
    print(f"[VRGDG RefMod] Added {kind}/{chosen} from {server_url()}")
    return {"name": f"{kind}/{chosen}", "already": False, "renamed": chosen != stem}


def _register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return

    async def run(work, *args, **kwargs):
        try:
            result = await asyncio.to_thread(work, *args, **kwargs)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except ServerError as exc:
            return web.json_response({"ok": False, "error": str(exc), "code": exc.code}, status=exc.status)
        except Exception as exc:
            print(f"[VRGDG RefMod] Server request failed: {exc}")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    async def body_of(request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except ValueError:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    @instance.routes.get("/vrgdg/refmod/server/settings")
    async def vrgdg_refmod_server_settings(request):
        return await run(get_settings)

    @instance.routes.post("/vrgdg/refmod/server/settings")
    async def vrgdg_refmod_server_settings_save(request):
        return await run(save_settings, (await body_of(request)).get("url"))

    @instance.routes.get("/vrgdg/refmod/server/signin")
    async def vrgdg_refmod_server_signin_options(request):
        return await run(sign_in_options)

    @instance.routes.post("/vrgdg/refmod/server/signin")
    async def vrgdg_refmod_server_signin_start(request):
        return await run(signin_start, (await body_of(request)).get("provider"))

    @instance.routes.post("/vrgdg/refmod/server/signin/poll")
    async def vrgdg_refmod_server_signin_poll(request):
        return await run(signin_poll)

    @instance.routes.post("/vrgdg/refmod/server/signout")
    async def vrgdg_refmod_server_signout(request):
        return await run(sign_out)

    @instance.routes.get("/vrgdg/refmod/server/library")
    async def vrgdg_refmod_server_library(request):
        return await run(list_server_refmods)

    @instance.routes.get("/vrgdg/refmod/server/preview")
    async def vrgdg_refmod_server_preview(request):
        try:
            picture = await asyncio.to_thread(server_preview, request.query.get("id", ""))
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except ServerError as exc:
            return web.json_response({"ok": False, "error": str(exc), "code": exc.code}, status=404 if exc.status == 404 else 401 if exc.status == 401 else 502)
        return web.Response(body=picture, content_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @instance.routes.post("/vrgdg/refmod/server/upload")
    async def vrgdg_refmod_server_upload(request):
        return await run(upload_refmod, (await body_of(request)).get("name"))

    @instance.routes.post("/vrgdg/refmod/server/download")
    async def vrgdg_refmod_server_download(request):
        return await run(download_refmod, (await body_of(request)).get("id"))

    @instance.routes.post("/vrgdg/refmod/server/delete")
    async def vrgdg_refmod_server_delete(request):
        return await run(delete_server_refmod, (await body_of(request)).get("id"))

    @instance.routes.post("/vrgdg/refmod/server/report")
    async def vrgdg_refmod_server_report(request):
        payload = await body_of(request)
        return await run(report_server_refmod, payload.get("id"), payload.get("reason"))


_register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
