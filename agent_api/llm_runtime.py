"""Which LLM the Agent API uses: the project's saved runner settings, and for LM Studio, the model that is loaded.

The API must never change what LM Studio has loaded. A request that names a model that is not loaded
makes LM Studio load it (and unload the current one), so this module:

* reads which models are loaded from LM Studio's ``GET /api/v0/models`` (a read-only call),
* sends requests with the id of a model that is already loaded, whatever the project settings say,
* sizes the request to the loaded context length, and
* refuses to call LM Studio at all when nothing is loaded, instead of triggering a load.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .errors import AgentApiError, LLM_UNAVAILABLE

_LM_STUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
_CHAT_TYPES = ("llm", "vlm")


class LlmUnavailableError(AgentApiError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=LLM_UNAVAILABLE, status=503, details=details, retryable=True)


def llm_payload_from_session(session: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The LLM runner settings a project saved, in the key names the generators read.

    Mirrors ``textGemmaRunnerPayload`` in the Video Builder. ``overrides`` (the caller's request
    parameters) win over the saved settings.
    """
    s = session if isinstance(session, dict) else {}
    runner = str(s.get("text_gemma_runner") or "builtin")
    payload: Dict[str, Any] = {
        "text_runner": runner,
        "qwen_model_file": s.get("qwen_model_file") or "",
        "qwen_mmproj_file": s.get("qwen_mmproj_file") or "",
        "gemma_model_file": s.get("gemma_model_file") or "",
        "n_ctx": s.get("gemma_context_limit") or 8000,
        "gemma_output_token_limit": s.get("gemma_output_token_limit") or 8192,
        "n_gpu_layers": s.get("gemma_gpu_layers", 99),
        "lmstudio_base_url": s.get("lm_studio_base_url") or _LM_STUDIO_DEFAULT_BASE_URL,
        "lmstudio_model": s.get("lm_studio_model") or "",
        "lmstudio_api_key": s.get("lm_studio_api_key") or "",
        "lmstudio_context_limit": s.get("lm_studio_context_limit") or 32768,
        "lmstudio_output_token_limit": s.get("lm_studio_output_token_limit") or 8192,
        "llm_api_provider": s.get("llm_api_provider") or "openai",
        "llm_api_model": s.get("llm_api_model") or "",
        "llm_api_key_project": s.get("llm_api_key_project") or "",
        "own_server_url": s.get("own_server_url") or "http://127.0.0.1:8000/v1",
        "own_server_model": s.get("own_server_model") or "",
        "own_server_api_key": s.get("own_server_api_key") or "",
        "own_server_api_key_project": s.get("own_server_api_key_project") or "",
        "own_server_timeout": s.get("own_server_timeout") or 360,
    }
    if runner == "qwen_local":
        payload["model_file"] = payload["qwen_model_file"]
    for key, value in (overrides or {}).items():
        if value not in (None, ""):
            payload[key] = value
    return payload


def _api_root(payload: Dict[str, Any]) -> str:
    base = str(payload.get("lmstudio_base_url") or _LM_STUDIO_DEFAULT_BASE_URL).strip().rstrip("/")
    return base[:-3].rstrip("/") if base.lower().endswith("/v1") else base


def loaded_lm_studio_models(payload: Dict[str, Any], timeout: float = 10.0) -> List[Dict[str, Any]]:
    """Chat-capable models LM Studio has loaded right now. Read-only: it cannot load or unload anything."""
    root = _api_root(payload)
    headers = {"Accept": "application/json"}
    api_key = str(payload.get("lmstudio_api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{root}/api/v0/models", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        raise LlmUnavailableError(
            f"LM Studio could not report its loaded models ({exc.code}). Update LM Studio, or load a model and try again.",
            {"url": f"{root}/api/v0/models"},
        ) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LlmUnavailableError(
            f"Could not reach LM Studio at {root}. Start LM Studio's local server and load a model.",
            {"url": root},
        ) from exc
    items = data.get("data") if isinstance(data, dict) else None
    return [
        item for item in (items or [])
        if isinstance(item, dict) and item.get("state") == "loaded" and item.get("type") in _CHAT_TYPES
    ]


def prepare_lm_studio_payload(payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """Point an LM Studio payload at the model that is loaded, sized to its context.

    The saved model name is used only when that model happens to be loaded. Otherwise the loaded model
    is used and the project setting is left alone. Raises ``LlmUnavailableError`` when nothing is loaded.
    """
    prepared = dict(payload)
    loaded = loaded_lm_studio_models(prepared, timeout)
    if not loaded:
        raise LlmUnavailableError(
            "LM Studio has no model loaded. Load one in LM Studio; the API never loads or switches models.",
            {"base_url": _api_root(prepared)},
        )
    wanted = str(prepared.get("lmstudio_model") or "").strip()
    chosen = next((item for item in loaded if item.get("id") == wanted), None) or loaded[0]
    prepared["lmstudio_model"] = chosen["id"]
    prepared["model_file"] = chosen["id"]
    loaded_context = int(chosen.get("loaded_context_length") or 0)
    if loaded_context > 0:
        configured = int(prepared.get("lmstudio_context_limit") or loaded_context)
        prepared["lmstudio_context_limit"] = min(configured, loaded_context)
    prepared["_lm_studio_loaded"] = {"model": chosen["id"], "context_length": loaded_context, "saved_model": wanted}
    return prepared


def prepare_llm_payload(payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """Apply the loaded-model rule when the runner is LM Studio; other runners pass through unchanged."""
    runner = str(payload.get("text_runner") or payload.get("text_gemma_runner") or "").strip().lower().replace("-", "_")
    if runner in ("lmstudio", "lm_studio"):
        return prepare_lm_studio_payload(payload, timeout)
    return dict(payload)


def describe_active_llm(payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """What the API would use right now, without calling the model."""
    prepared = prepare_llm_payload(payload, timeout)
    runner = str(prepared.get("text_runner") or "builtin")
    info: Dict[str, Any] = {"runner": runner}
    loaded = prepared.get("_lm_studio_loaded")
    if loaded:
        info.update({
            "model": loaded["model"],
            "context_length": loaded["context_length"],
            "saved_model_setting": loaded["saved_model"],
            "uses_saved_setting": loaded["saved_model"] == loaded["model"],
            "base_url": _api_root(prepared),
        })
    return info
