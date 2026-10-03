"""Runs the Video Builder's LLM calls: runner selection, local GGUF, LM Studio, own server, hosted APIs, and remote vision."""

import json
import os
import re
import gc
import ssl
import urllib.error
import urllib.parse
import urllib.request
from ..core.model_paths import register_custom_model_root
from .text_cleaning import _clean_visual_gemma_text

from .output_checks import _looks_like_gemma_repeat_failure, _looks_like_id_lora_script_prompt, _looks_like_source_lyric_echo, _looks_like_unfilled_prompt_template, _validate_builder_gemma_prompt
from ..builder.media import _pil_image_to_data_url


_LM_STUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


_OWN_SERVER_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


_EXTERNAL_LLM_RUNNERS = frozenset({"lm_studio", "llm_api", "own_server"})
# Runners that can constrain a reply to a JSON schema; the hosted-API runner is not wired for it yet.
_JSON_SCHEMA_RUNNERS = frozenset({"builtin", "qwen_local", "lm_studio", "own_server"})


def _runner_supports_json_schema(payload):
    return _llm_runner_from_payload(payload) in _JSON_SCHEMA_RUNNERS


def _openai_json_schema_format(json_schema):
    return {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": json_schema}}


# Room left in the context window for the chat template around the prompt.
_CONTEXT_TEMPLATE_MARGIN_TOKENS = 64
_MIN_OUTPUT_TOKENS = 256


def _estimate_prompt_tokens(text):
    # Conservative for remote runners without a tokenizer: English prose is ~4 characters per token, JSON-heavy prompts less.
    return len(str(text or "")) // 3 + 1


def _fit_output_to_context(max_new_tokens, prompt_tokens, context_limit, runner_label):
    """Cap the output budget so prompt + output fits the user's context window."""
    available = int(context_limit) - int(prompt_tokens) - _CONTEXT_TEMPLATE_MARGIN_TOKENS
    if available < _MIN_OUTPUT_TOKENS:
        raise ValueError(
            f"The prompt (about {prompt_tokens} tokens) leaves no room for a reply in the {context_limit}-token "
            f"{runner_label} context window. Raise the context limit in LLM Runner."
        )
    if max_new_tokens > available:
        print(f"[VRGDG LLM] output limit lowered from {max_new_tokens} to {available} tokens so the prompt fits the {context_limit}-token {runner_label} context.")
        return available
    return max_new_tokens


def _llm_multi_choices():
    try:
        from .api import VRGDG_LLM_Multi
    except Exception as exc:
        raise RuntimeError(f"Could not load VRGDG LLM Multi choices: {exc}") from exc
    provider_models = getattr(VRGDG_LLM_Multi, "PROVIDER_MODELS", {}) or {}
    default_model = getattr(VRGDG_LLM_Multi, "DEFAULT_MODEL", {}) or {}
    label_map = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Google Gemini",
        "grok": "Grok",
        "deepseek": "DeepSeek",
        "openrouter": "OpenRouter",
        "apifreellm": "APIFreeLLM",
    }
    providers = []
    for provider, models in provider_models.items():
        clean_provider = str(provider or "").strip().lower()
        if not clean_provider or clean_provider == "xai":
            continue
        model_list = [str(model or "").strip() for model in (models or []) if str(model or "").strip()]
        providers.append({
            "id": clean_provider,
            "label": label_map.get(clean_provider, clean_provider),
            "models": model_list,
            "default_model": str(default_model.get(clean_provider) or (model_list[0] if model_list else "")),
        })
    return {"providers": providers}


def _test_llm_api(payload):
    try:
        from .api import VRGDG_LLM_Multi
    except Exception as exc:
        raise RuntimeError(f"Could not load LLM API runner: {exc}") from exc
    provider = str(payload.get("provider") or payload.get("llm_api_provider") or "openai").strip().lower()
    if provider == "xai":
        provider = "grok"
    model = str(payload.get("model") or payload.get("llm_api_model") or "").strip()
    api_key = str(payload.get("api_key") or payload.get("llm_api_key") or "").strip()
    custom_model = str(payload.get("custom_model") or "").strip()
    if not api_key:
        raise ValueError("API key is missing.")
    if not provider:
        raise ValueError("Provider is missing.")
    prompt = str(payload.get("prompt") or "Reply with OK only.").strip() or "Reply with OK only."
    runner = VRGDG_LLM_Multi()
    text, used_provider, used_model, status, _image = runner.generate_text(
        api_key=api_key,
        provider=provider,
        model=model,
        prompt=prompt,
        custom_model=custom_model,
    )
    status = str(status or "").strip()
    if status.lower().startswith("error"):
        raise RuntimeError(status)
    text = str(text or "").strip()
    if not text:
        raise RuntimeError("LLM API returned empty text.")
    return {
        "text": text[:1000],
        "used_provider": used_provider,
        "used_model": used_model,
        "status": status or "ok",
    }


def _llm_runner_from_payload(payload):
    runner = str(payload.get("text_runner") or payload.get("text_gemma_runner") or payload.get("gemma_runner") or "builtin").strip().lower()
    if runner in {"lmstudio", "lm-studio", "lm_studio"}:
        runner = "lm_studio"
    if runner in {"llmapi", "llm-api", "llm_api", "api", "outside_api", "outside_llm_api"}:
        runner = "llm_api"
    if runner in {"ownserver", "own-server", "own_server", "custom_openai", "openai_compatible", "custom_server", "my_server"}:
        runner = "own_server"
    if runner in {"qwen", "qwen_local", "qwen-local", "qwen_gguf", "qwen_gguf_local"}:
        runner = "qwen_local"
    if runner not in {"builtin", "qwen_local", "lm_studio", "llm_api", "own_server"}:
        runner = "builtin"
    return runner


def _llm_runner_display_name(payload, vision=False):
    runner = _llm_runner_from_payload(payload)
    names = {
        "builtin": "Gemma Local",
        "qwen_local": "Qwen Local",
        "lm_studio": "LM Studio",
        "llm_api": "LLM API",
        "own_server": "Custom Server",
    }
    name = names.get(runner, "Gemma Local")
    return f"{name} vision" if vision else name


def _builder_local_llm(payload):
    from .gguf import VRGDG_QwenGGUF, VRGDG_SuperGemmaGGUFChat
    if _llm_runner_from_payload(payload) == "qwen_local":
        llm = VRGDG_QwenGGUF()
    else:
        llm = VRGDG_SuperGemmaGGUFChat()
    # Builder prompt jobs need direct structured output, not a reasoning
    # preamble. Set this explicitly for both local runners because model chat
    # templates can default to thinking mode when the flag is omitted.
    llm._qwen_chat_template_kwargs = {"enable_thinking": False}
    return llm


def _builder_local_model_file(payload, current=""):
    if _llm_runner_from_payload(payload) == "qwen_local":
        return str(payload.get("qwen_model_file") or current or "").strip()
    if _llm_runner_from_payload(payload) == "builtin" and payload.get("gemma_model_file"):
        return str(payload.get("gemma_model_file") or current or "").strip()
    return str(current or "").strip()


def _builder_local_mmproj_file(payload, current=""):
    if _llm_runner_from_payload(payload) == "qwen_local":
        return str(payload.get("qwen_mmproj_file") or current or "").strip()
    return str(current or "").strip()


def _own_server_v1_root(raw_url):
    url = str(raw_url or "").strip()
    if not url:
        raise ValueError("Own server URL is empty. Open LLM Runner and paste your server URL.")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Own server URL must start with http:// or https://.")
    if not parsed.netloc:
        raise ValueError("Own server URL is missing a host.")
    path = (parsed.path or "").rstrip("/")
    lowered = path.lower()
    for suffix in ("/chat/completions", "/completions", "/models"):
        if lowered.endswith(suffix):
            path = path[: -len(suffix)]
            lowered = path.lower()
            break
    if lowered.endswith("/v1"):
        v1_path = path
    elif path:
        v1_path = f"{path}/v1"
    else:
        v1_path = "/v1"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, v1_path, "", "", ""))


def _own_server_api_key(payload):
    payload = payload if isinstance(payload, dict) else {}
    return str(
        payload.get("own_server_api_key")
        or payload.get("own_server_key")
        or payload.get("api_key")
        or ""
    ).strip()


def _own_server_model_name(payload):
    payload = payload if isinstance(payload, dict) else {}
    return str(
        payload.get("own_server_model")
        or payload.get("model")
        or payload.get("model_file")
        or ""
    ).strip()


def _own_server_timeout(payload, default=360):
    payload = payload if isinstance(payload, dict) else {}
    try:
        timeout = float(payload.get("own_server_timeout") or default)
    except (TypeError, ValueError):
        timeout = float(default)
    return max(15.0, min(600.0, timeout))


def _own_server_headers(payload):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "VRGDG-OwnServer/1.0 (+ComfyUI)",
    }
    api_key = _own_server_api_key(payload)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _own_server_ssl_context(url):
    if not str(url or "").lower().startswith("https://"):
        return None
    return ssl.create_default_context()


def _own_server_request_json(url, payload, method="POST", body=None, timeout=180):
    headers = _own_server_headers(payload)
    if str(method or "POST").upper() == "GET":
        headers.pop("Content-Type", None)
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    ssl_context = _own_server_ssl_context(url)

    def _open(context):
        with urllib.request.urlopen(request, timeout=float(timeout), context=context) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            return json.loads(raw)

    try:
        return _open(ssl_context)
    except ssl.SSLError as exc:
        raise RuntimeError(
            f"TLS certificate validation failed for own server at {url}. "
            "Install a trusted certificate or use HTTP only for a server on a trusted local network."
        ) from exc
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"Own server request failed ({exc.code}): {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLError):
            raise RuntimeError(
                f"TLS certificate validation failed for own server at {url}. "
                "Install a trusted certificate or use HTTP only for a server on a trusted local network."
            ) from exc
        raise RuntimeError(
            f"Could not connect to own server at {url}. Check the URL, local bind address, or Cloudflare tunnel."
        ) from exc


def _own_server_message_text(data):
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    item_type = str(item.get("type") or "").lower()
                    if item_type in {"text", "output_text"}:
                        parts.append(str(item.get("text") or ""))
                    elif isinstance(item.get("text"), str):
                        parts.append(item.get("text"))
                    elif isinstance(item.get("content"), str):
                        parts.append(item.get("content"))
            content = "".join(parts)
        text = str(content or "").strip()
        if text:
            return text
        for key in ("reasoning_content", "refusal"):
            extra = message.get(key) if isinstance(message, dict) else None
            if extra:
                return str(extra).strip()
        choice_text = choices[0].get("text") if isinstance(choices[0], dict) else None
        if choice_text:
            return str(choice_text).strip()
    for key in ("output_text", "response", "text", "content"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _own_server_chat_messages(instruction_text, pil_images=None):
    images = [image for image in list(pil_images or []) if image is not None]
    if not images:
        return [{"role": "user", "content": str(instruction_text or "")}]
    content = [{"type": "text", "text": str(instruction_text or "")}]
    for image in images[:4]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": _pil_image_to_data_url(image, max_height=1024, quality=90),
            },
        })
    return [{"role": "user", "content": content}]


def _run_own_server_chat(payload, instruction_text, pil_images=None, temperature=0.6, top_p=0.95, max_new_tokens=1200, json_schema=None):
    v1_root = _own_server_v1_root(
        payload.get("own_server_url") or payload.get("base_url") or _OWN_SERVER_DEFAULT_BASE_URL
    )
    model = _own_server_model_name(payload)
    if not model:
        raise ValueError("Own server model name is missing. Open LLM Runner and enter the served model id.")
    body = {
        "model": model,
        "messages": _own_server_chat_messages(instruction_text, pil_images),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
        "max_tokens": int(_runner_output_token_limit(payload, max_new_tokens)),
    }
    if json_schema:
        body["response_format"] = _openai_json_schema_format(json_schema)
    url = f"{v1_root}/chat/completions"
    try:
        data = _own_server_request_json(url, payload, body=body, timeout=_own_server_timeout(payload))
    except RuntimeError as exc:
        message = str(exc).lower()
        room = re.search(r"max_tokens \(at most (\d+)", message) or re.search(r"at most (\d+) here", message)
        if "exceeds the context" in message and room:
            # The server reports how many tokens are left after the prompt, so retry with exactly that budget.
            available = int(room.group(1)) - _CONTEXT_TEMPLATE_MARGIN_TOKENS
            if available < _MIN_OUTPUT_TOKENS:
                raise
            print(f"[VRGDG LLM] own server output limit lowered from {body['max_tokens']} to {available} tokens to fit its context.")
            retry_body = dict(body)
            retry_body["max_tokens"] = available
            data = _own_server_request_json(url, payload, body=retry_body, timeout=_own_server_timeout(payload))
        elif "max_tokens" in message and ("unknown" in message or "unsupported" in message or "400" in message):
            retry_body = dict(body)
            retry_body["max_completion_tokens"] = retry_body.pop("max_tokens")
            data = _own_server_request_json(url, payload, body=retry_body, timeout=_own_server_timeout(payload))
        else:
            raise
    text = _own_server_message_text(data)
    if not text:
        raise ValueError("Own server returned empty text.")
    return text, {
        "runner": "own_server_vision" if pil_images else "own_server",
        "used_provider": "own_server",
        "used_model": model,
        "unloaded": False,
        "base_url": v1_root,
    }


def _run_own_server_text(payload, instruction_text, temperature=0.6, top_p=0.95, max_new_tokens=1200, json_schema=None):
    return _run_own_server_chat(
        payload,
        instruction_text,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        json_schema=json_schema,
    )


def _run_own_server_vision(payload, instruction_text, pil_images, temperature=0.25, top_p=0.95, max_new_tokens=1200):
    images = list(pil_images or [])
    if not images:
        raise ValueError("Own server vision needs at least one image reference.")
    return _run_own_server_chat(
        payload,
        instruction_text,
        pil_images=images,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )


def _try_run_remote_vision(payload, instruction_text, pil_images, temperature=0.25, top_p=0.95, max_new_tokens=1200):
    runner = _llm_runner_from_payload(payload)
    if runner == "llm_api":
        return _run_llm_api_vision(payload, instruction_text, pil_images)
    if runner == "own_server":
        return _run_own_server_vision(
            payload,
            instruction_text,
            pil_images,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    if runner == "lm_studio":
        text = _run_lm_studio_vision(
            payload,
            instruction_text,
            pil_images,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        return text, {
            "runner": "lm_studio_vision",
            "used_model": str(payload.get("lmstudio_model") or "").strip(),
            "unloaded": False,
        }
    return None


def _list_own_server_models(payload):
    v1_root = _own_server_v1_root(
        payload.get("own_server_url") or payload.get("base_url") or _OWN_SERVER_DEFAULT_BASE_URL
    )
    data = _own_server_request_json(
        f"{v1_root}/models",
        payload,
        method="GET",
        timeout=min(45.0, _own_server_timeout(payload, 45)),
    )
    models = []
    raw_models = data.get("data") if isinstance(data, dict) else None
    if isinstance(raw_models, list):
        for item in raw_models:
            if isinstance(item, str) and item.strip():
                models.append(item.strip())
            elif isinstance(item, dict):
                model_id = str(item.get("id") or item.get("name") or "").strip()
                if model_id:
                    models.append(model_id)
    return {"models": models, "base_url": v1_root}


def _test_own_server(payload):
    prompt = str(payload.get("prompt") or "Reply with OK only.").strip() or "Reply with OK only."
    text, info = _run_own_server_text(payload, prompt, temperature=0.1, top_p=0.9, max_new_tokens=64)
    return {
        "text": text,
        "used_provider": "own_server",
        "used_model": info.get("used_model", ""),
        "base_url": info.get("base_url", ""),
    }


def _normalized_token_limit(value, default, minimum=64, maximum=262144):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), min(int(maximum), parsed))


def _runner_output_token_limit(payload, requested=1200):
    payload = payload if isinstance(payload, dict) else {}
    runner = _llm_runner_from_payload(payload)
    if runner == "lm_studio":
        configured = payload.get("lmstudio_output_token_limit")
        if configured in (None, ""):
            configured = payload.get("lm_studio_output_token_limit")
    elif runner == "own_server":
        configured = payload.get("own_server_output_token_limit")
    elif runner == "builtin":
        configured = payload.get("gemma_output_token_limit")
    else:
        configured = None
    if runner in {"builtin", "lm_studio", "own_server"} and configured in (None, ""):
        configured = payload.get("llm_max_tokens")
    if configured not in (None, ""):
        return _normalized_token_limit(configured, requested)
    return _normalized_token_limit(requested, 1200)


def _lm_studio_context_limit(payload):
    payload = payload if isinstance(payload, dict) else {}
    configured = payload.get("lmstudio_context_limit")
    if configured in (None, ""):
        configured = payload.get("lm_studio_context_limit")
    return _normalized_token_limit(configured, 32768, minimum=512)


def _lm_studio_reasoning_mode(payload, default=""):
    payload = payload if isinstance(payload, dict) else {}
    configured = payload.get("lmstudio_reasoning")
    if configured in (None, ""):
        configured = payload.get("lm_studio_reasoning")
    normalized = str(configured or "").strip().lower()
    allowed = {"off", "low", "medium", "high", "on"}
    if normalized in allowed:
        return normalized
    fallback = str(default or "").strip().lower()
    return fallback if fallback in allowed else ""


def _lm_studio_api_root(payload):
    base_url = str(payload.get("lmstudio_base_url") or _LM_STUDIO_DEFAULT_BASE_URL).strip().rstrip("/")
    if base_url.lower().endswith("/v1"):
        return base_url[:-3].rstrip("/")
    return base_url


def _lm_studio_native_output_text(data):
    output = data.get("output") if isinstance(data, dict) else None
    if not isinstance(output, list):
        return ""
    parts = []
    for item in output:
        if not isinstance(item, dict) or str(item.get("type") or "").lower() != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if text:
                        parts.append(str(text))
    return "\n".join(part.strip() for part in parts if str(part or "").strip()).strip()


def _run_lm_studio_native_chat(payload, input_value, temperature, top_p, max_new_tokens, timeout, label, reasoning_default=""):
    api_root = _lm_studio_api_root(payload)
    model = str(payload.get("lmstudio_model") or payload.get("model_file") or "").strip()
    api_key = str(payload.get("lmstudio_api_key") or "").strip()
    if not api_root:
        raise ValueError("LM Studio base URL is empty.")
    if not model:
        raise ValueError(f"Enter the LM Studio {label}model name shown in LM Studio.")
    context_limit = _lm_studio_context_limit(payload)
    output_limit = _runner_output_token_limit(payload, max_new_tokens)
    if isinstance(input_value, str):
        # Text prompts are sized so prompt + reply fits the context; image inputs cannot be estimated from characters.
        output_limit = _fit_output_to_context(output_limit, _estimate_prompt_tokens(input_value), context_limit, "LM Studio")
    body = {
        "model": model,
        "input": input_value,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "context_length": context_limit,
        "max_output_tokens": output_limit,
        "store": False,
        "stream": False,
    }
    reasoning_mode = _lm_studio_reasoning_mode(payload, reasoning_default)
    if reasoning_mode:
        body["reasoning"] = reasoning_mode
    if payload.get("seed") is not None:
        body["seed"] = int(payload.get("seed"))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _post(request_body):
        request = urllib.request.Request(
            f"{api_root}/api/v1/chat",
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))

    try:
        data = _post(body)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        if exc.code in {404, 405}:
            raise RuntimeError(
                "LM Studio's native /api/v1/chat endpoint is required to apply per-request context and output limits. "
                "Update LM Studio and make sure its Local Server is running."
            ) from exc
        if "seed" in body and "unrecognized_keys" in details.lower() and "seed" in details.lower():
            retry_body = {key: value for key, value in body.items() if key != "seed"}
            try:
                data = _post(retry_body)
            except urllib.error.HTTPError as retry_exc:
                retry_details = retry_exc.read().decode("utf-8", errors="replace") if hasattr(retry_exc, "read") else ""
                raise RuntimeError(f"LM Studio {label}request failed ({retry_exc.code}): {retry_details or retry_exc.reason}") from retry_exc
            except urllib.error.URLError as retry_exc:
                raise RuntimeError(f"Could not connect to LM Studio at {api_root}. Make sure LM Studio's local server is running.") from retry_exc
        else:
            raise RuntimeError(f"LM Studio {label}request failed ({exc.code}): {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to LM Studio at {api_root}. Make sure LM Studio's local server is running.") from exc
    text = _lm_studio_native_output_text(data)
    if not text:
        raise ValueError(f"LM Studio {label}returned empty text.")
    return text


def _resolve_mmproj_dropdown_path(llm, mmproj_file):
    selected = str(mmproj_file or "").strip()
    try:
        if selected:
            return llm._resolve_dropdown_path(selected, llm.MISSING_MMPROJ_OPTION)
    except Exception:
        pass
    try:
        choices = [
            choice for choice in llm._list_local_mmproj_choices()
            if choice and choice != llm.MISSING_MMPROJ_OPTION
        ]
    except Exception:
        choices = []
    if len(choices) == 1:
        return llm._resolve_dropdown_path(choices[0], llm.MISSING_MMPROJ_OPTION)
    if selected:
        return llm._resolve_dropdown_path(selected, llm.MISSING_MMPROJ_OPTION)
    raise ValueError("Choose an mmproj file for the vision model.")


def _run_lm_studio_text(payload, instruction_text, temperature=0.6, top_p=0.95, max_new_tokens=1200):
    return _run_lm_studio_native_chat(
        payload,
        str(instruction_text or ""),
        temperature,
        top_p,
        max_new_tokens,
        payload.get("lmstudio_timeout") or 180,
        "",
    )


def _run_lm_studio_structured(payload, instruction_text, json_schema, temperature=0.6, top_p=0.95, max_new_tokens=1200):
    # LM Studio's native /api/v1/chat rejects response_format, so schema-constrained calls use the
    # OpenAI-compatible endpoint. It cannot set context_length per request; the model's loaded context applies.
    api_root = _lm_studio_api_root(payload)
    model = str(payload.get("lmstudio_model") or payload.get("model_file") or "").strip()
    if not api_root:
        raise ValueError("LM Studio base URL is empty.")
    if not model:
        raise ValueError("Enter the LM Studio model name shown in LM Studio.")
    output_limit = _fit_output_to_context(
        _runner_output_token_limit(payload, max_new_tokens),
        _estimate_prompt_tokens(instruction_text),
        _lm_studio_context_limit(payload),
        "LM Studio",
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": str(instruction_text or "")}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": output_limit,
        "stream": False,
        "response_format": _openai_json_schema_format(json_schema),
    }
    if payload.get("seed") is not None:
        body["seed"] = int(payload.get("seed"))
    headers = {"Content-Type": "application/json"}
    api_key = str(payload.get("lmstudio_api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{api_root}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(payload.get("lmstudio_timeout") or 180)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"LM Studio request failed ({exc.code}): {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to LM Studio at {api_root}. Make sure LM Studio's local server is running.") from exc
    text = _own_server_message_text(data)
    if not text:
        raise ValueError("LM Studio returned empty text.")
    return text


def _run_llm_api_text(payload, instruction_text):
    try:
        from .api import VRGDG_LLM_Multi
    except Exception as exc:
        raise RuntimeError(f"Could not load LLM API runner: {exc}") from exc
    provider = str(payload.get("llm_api_provider") or payload.get("provider") or "openai").strip().lower()
    if provider == "xai":
        provider = "grok"
    model = str(payload.get("llm_api_model") or payload.get("model") or "").strip()
    api_key = str(payload.get("llm_api_key") or payload.get("api_key") or "").strip()
    custom_model = str(payload.get("llm_api_custom_model") or payload.get("custom_model") or "").strip()
    if not api_key:
        raise ValueError("LLM API key is missing. Open LLM Runner and paste your API key.")
    if not provider:
        raise ValueError("LLM API provider is missing.")
    runner = VRGDG_LLM_Multi()
    text, used_provider, used_model, status, _image = runner.generate_text(
        api_key=api_key,
        provider=provider,
        model=model,
        prompt=str(instruction_text or ""),
        custom_model=custom_model,
    )
    status = str(status or "").strip()
    if status.lower().startswith("error"):
        raise RuntimeError(status)
    text = str(text or "").strip()
    if not text:
        raise ValueError("LLM API returned empty text.")
    return text, {
        "runner": "llm_api",
        "used_provider": used_provider or provider,
        "used_model": used_model or model,
        "unloaded": False,
    }


def _run_llm_api_vision(payload, instruction_text, pil_images):
    try:
        from .api import VRGDG_LLM_Multi
    except Exception as exc:
        raise RuntimeError(f"Could not load LLM API runner: {exc}") from exc
    provider = str(payload.get("llm_api_provider") or payload.get("provider") or "openai").strip().lower()
    if provider == "xai":
        provider = "grok"
    model = str(payload.get("llm_api_model") or payload.get("model") or "").strip()
    api_key = str(payload.get("llm_api_key") or payload.get("api_key") or "").strip()
    custom_model = str(payload.get("llm_api_custom_model") or payload.get("custom_model") or "").strip()
    if not api_key:
        raise ValueError("LLM API key is missing. Open LLM Runner and paste your API key.")
    if not provider:
        raise ValueError("LLM API provider is missing.")
    if not (model or custom_model):
        raise ValueError("LLM API vision model is missing. Open LLM Runner and select a vision-capable API model before running Video All with image reference.")
    if provider in {"deepseek", "apifreellm"}:
        raise ValueError(f"{provider} is not configured for image/vision input here. Choose a vision-capable LLM API provider/model, or use LM Studio/Gemma Local vision.")
    images = list(pil_images or [])
    if not images:
        raise ValueError("LLM API vision needs at least one image reference.")
    runner = VRGDG_LLM_Multi()
    image_tensors = [runner._pil_to_tensor(image.convert("RGB")) for image in images[:4]]
    kwargs = {
        "api_key": api_key,
        "provider": provider,
        "model": model,
        "prompt": str(instruction_text or ""),
        "custom_model": custom_model,
    }
    for index, tensor in enumerate(image_tensors, start=1):
        kwargs[f"image{index}"] = tensor
    text, used_provider, used_model, status, _image = runner.generate_text(**kwargs)
    status = str(status or "").strip()
    if status.lower().startswith("error"):
        raise RuntimeError(status)
    text = str(text or "").strip()
    if not text:
        raise ValueError("LLM API vision returned empty text.")
    return text, {
        "runner": "llm_api_vision",
        "used_provider": used_provider or provider,
        "used_model": used_model or model or custom_model,
        "unloaded": False,
    }


def _run_lm_studio_vision(payload, instruction_text, pil_images, temperature=0.25, top_p=0.95, max_new_tokens=1200):
    images = list(pil_images or [])
    if not images:
        raise ValueError("LM Studio vision needs at least one image.")
    # Native LM Studio chat accepts multimodal input blocks directly. Do not
    # wrap the text in an OpenAI-style message object inside `input`.
    content = [{"type": "text", "content": str(instruction_text or "")}]
    for image in images:
        content.append({
            "type": "image",
            "data_url": _pil_image_to_data_url(image),
        })
    return _run_lm_studio_native_chat(
        payload,
        content,
        temperature,
        top_p,
        max_new_tokens,
        payload.get("lmstudio_timeout") or 300,
        "vision ",
        reasoning_default="off",
    )


def _list_lm_studio_models(payload):
    base_url = str(payload.get("lmstudio_base_url") or _LM_STUDIO_DEFAULT_BASE_URL).strip().rstrip("/")
    api_key = str(payload.get("lmstudio_api_key") or "").strip()
    if not base_url:
        raise ValueError("LM Studio base URL is empty.")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url}/models", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=float(payload.get("lmstudio_timeout") or 30)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"LM Studio models request failed ({exc.code}): {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to LM Studio at {base_url}. Make sure LM Studio's local server is running.") from exc
    items = data.get("data") if isinstance(data, dict) else []
    models = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                model_id = str(item.get("id") or "").strip()
                if model_id:
                    models.append(model_id)
    return {"models": models, "raw": data}


def _strip_builder_thinking_text(text):
    """Remove reasoning/control-channel text that models may emit despite disabled thinking."""
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    # Qwen and a few OpenAI-compatible servers can emit an unclosed block. In
    # that case the useful answer starts after the closing marker.
    cleaned = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    closing = re.search(r"</(?:think|thought)>", cleaned, flags=re.IGNORECASE)
    if closing:
        cleaned = cleaned[closing.end():].strip()

    # When a model labels a reasoning preamble and then supplies a final
    # answer, discard the whole preamble rather than only its heading.
    thought_prefix = re.match(
        r"^\s*(?:<\|channel\|>\s*)?(?:<\|(?:thought|thinking|analysis|reasoning)\|>\s*)?"
        r"(?:thought(?: process)?|thinking(?: process)?|analysis|reasoning)\s*:\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    final_marker = re.search(r"\n\s*(?:final(?: answer| response)?|answer|response)\s*:\s*", cleaned, flags=re.IGNORECASE)
    if thought_prefix and final_marker and final_marker.start() >= thought_prefix.end():
        cleaned = cleaned[final_marker.end():].strip()
    else:
        # Strip a leading label/control token while preserving the answer text.
        cleaned = re.sub(
            r"^\s*(?:<\|channel\|>\s*)?(?:<\|(?:thought|thinking|analysis|reasoning)\|>\s*)?"
            r"(?:thought(?: process)?|thinking(?: process)?|analysis|reasoning)\s*:\s*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
    return cleaned


def _clean_lm_studio_plain_text(text):
    cleaned = _strip_builder_thinking_text(text)
    cleaned = re.sub(r"^\s*```(?:text|json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    cleaned = re.sub(r"^(?:Assistant|Answer|Final answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _run_builder_text_llm(payload, instruction_text, temperature=0.6, top_p=0.95, max_new_tokens=1200, label="Gemma", preserve_paragraphs=False, json_schema=None):
    """Run a builder text prompt on the selected runner.

    With json_schema on a runner in _JSON_SCHEMA_RUNNERS, the reply is constrained to that schema and returned as raw
    JSON text; the prose cleaners are skipped because they would cut the JSON apart. Other runners ignore the schema.
    The output budget is capped so prompt + output fits the context window the user set for LM Studio or the GGUF model.
    """
    max_new_tokens = _runner_output_token_limit(payload, max_new_tokens)
    structured = bool(json_schema) and _runner_supports_json_schema(payload)

    def _clean(text):
        if structured:
            return str(text or "").strip()
        return _clean_lm_studio_plain_text(text) if preserve_paragraphs else _clean_visual_gemma_text(text)

    if _llm_runner_from_payload(payload) == "lm_studio":
        if structured:
            text = _run_lm_studio_structured(
                payload,
                instruction_text,
                json_schema,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            text = _run_lm_studio_text(
                payload,
                instruction_text,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        return _clean(text), {
            "runner": "lm_studio",
            "used_model": str(payload.get("lmstudio_model") or "").strip(),
            "unloaded": False,
        }

    if _llm_runner_from_payload(payload) == "llm_api":
        text, info = _run_llm_api_text(payload, instruction_text)
        return _clean(text), info

    if _llm_runner_from_payload(payload) == "own_server":
        text, info = _run_own_server_text(
            payload,
            instruction_text,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            json_schema=json_schema if structured else None,
        )
        return _clean(text), info

    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    if not model_file:
        raise ValueError(f"Choose a {label} model first.")
    llm = _builder_local_llm(payload)
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    print(
        f"[VRGDG LLM] runner={_llm_runner_display_name(payload)} "
        f"model={os.path.basename(str(model_path)) or str(model_path)}"
    )
    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    unload_after = bool(payload.get("unload_after", True))
    seed = payload.get("seed")
    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
        )
        prompt_tokens = len(model.tokenize(str(instruction_text or "").encode("utf-8")))
        max_new_tokens = _fit_output_to_context(int(max_new_tokens), prompt_tokens, model.n_ctx(), "GGUF model")
        text = llm._run_gguf_text_pipeline(
            model=model,
            instruction_text=instruction_text,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            seed=int(seed) if seed is not None else None,
            json_schema=json_schema if structured else None,
        )
        return _clean(text), {
            "runner": _llm_runner_from_payload(payload),
            "used_model": model_path,
            "unloaded": unload_after,
        }
    finally:
        if unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path="",
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _repair_builder_gemma_prompt(payload, text, label):
    original = str(text or "").strip()
    label_key = str(label or "").strip().lower().replace(" ", "_")
    if label_key in {"id-lora", "id_lora", "id-lora_i2v"} and _looks_like_id_lora_script_prompt(original):
        return original
    lyric_echo = _looks_like_source_lyric_echo(original, payload)
    needs_repair = _looks_like_gemma_repeat_failure(original) or _looks_like_unfilled_prompt_template(original) or lyric_echo
    if not original or not needs_repair:
        return original

    repair_payload = dict(payload or {})
    repair_model = str(
        repair_payload.get("repair_model_file")
        or repair_payload.get("text_model_file")
        or repair_payload.get("model_file")
        or ""
    ).strip()
    if repair_model:
        repair_payload["model_file"] = repair_model
    repair_payload["mmproj_file"] = ""
    repair_payload["use_vision"] = False
    broken = original[:5000]
    label_text = str(label or "").strip()
    video_repair = label_text.lower() in {"i2v", "t2v"}
    if video_repair:
        concept_prompt = str(repair_payload.get("t2i_prompt") or "").strip()[:3000]
        motion_notes = str(repair_payload.get("user_notes") or "").strip()[:2000]
        instruction = (
            f"Clean this broken {label_text} video prompt into one usable final video prompt.\n\n"
            "The broken text may contain internal thoughts, repeated tokens, markdown, or unfilled square-bracket placeholders.\n"
            "Use only the concept prompt and motion notes below as context. Do not use project story, lyrics, agent chat, or any other context.\n"
            "Replace placeholders like [Subject], [setting/environment], [time/weather], and [Camera Motion] with concrete details from the concept prompt and motion notes.\n"
            "Do not continue the broken text. Do not explain the repair. Do not mention that it was repaired.\n"
            "Return exactly one normal video prompt paragraph with no square brackets, labels, markdown, or placeholders.\n"
            "Keep it under 120 words.\n\n"
            f"Concept/T2I prompt:\n{concept_prompt or '[none provided]'}\n\n"
            f"I2V/T2V motion notes:\n{motion_notes or '[none provided]'}\n\n"
            f"Broken video prompt:\n{broken}"
        )
    else:
        user_notes = str(repair_payload.get("user_notes") or "").strip()[:3000]
        lyric_text = str(repair_payload.get("lyric_text") or repair_payload.get("lyrics") or "").strip()[:1200]
        instruction = (
            f"Clean this broken {label_text} image prompt into one usable final prompt.\n\n"
            "The broken text may contain internal thoughts, analysis, channel tags, repeated tokens, markdown, unfilled square-bracket placeholders, junk, or the raw scene lyric copied by mistake.\n"
            "Do not continue the broken text. Do not explain the repair. Do not mention that it was repaired.\n"
            "Return exactly one normal image prompt paragraph.\n"
            "Remove all thought, analysis, channel, role, markdown, labels, square brackets, placeholder words, and repeated junk.\n"
            "If the broken text is just the scene lyric, do not quote the lyric; create a visual still-image prompt from the user notes and lyric mood.\n"
            "Replace placeholders like [Subject], [setting/environment], [time/weather], and [Camera Motion] with concrete details inferred from the broken text and user notes.\n"
            "Keep only usable visual image-generation content. If usable details are scarce, create a concise cinematic prompt from the usable fragments.\n"
            "Keep it under 120 words.\n\n"
            f"User notes/context:\n{user_notes or '[none provided]'}\n\n"
            f"Scene lyric, for mood only:\n{lyric_text or '[none provided]'}\n\n"
            f"Broken text:\n{broken}"
        )
    try:
        repaired, _run_info = _run_builder_text_llm(
            repair_payload,
            instruction,
            temperature=0.25,
            top_p=0.85,
            max_new_tokens=350,
            label=f"{label_text} repair Gemma",
        )
        repaired = _clean_visual_gemma_text(repaired)
        if repaired and not _looks_like_gemma_repeat_failure(repaired) and not _looks_like_unfilled_prompt_template(repaired):
            return repaired
    except Exception:
        pass
    return original


def _repair_and_validate_builder_gemma_prompt(payload, text, label):
    repaired = _repair_builder_gemma_prompt(payload, text, label)
    performance_mode = str(
        payload.get("performance_mode")
        or payload.get("performanceMode")
        or payload.get("video_type")
        or payload.get("videoType")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if performance_mode in {"no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"}:
        repaired = _clean_visual_only_positive_prompt(repaired)
    _validate_builder_gemma_prompt(repaired, label, payload)
    return repaired


def _clean_visual_only_positive_prompt(text):
    """Keep visual-only LTX prompts affirmative and free of vocal/mouth concepts."""
    forbidden = re.compile(
        r"\b(?:lip[ -]?sync(?:ing|s)?|sing(?:s|ing)?|sang|sung|rap(?:s|ping)?|"
        r"vocal(?:s|ization)?|lyric(?:s)?|speak(?:s|ing)?|say(?:s|ing)?|said|"
        r"dialogue|mouth(?:s|ed|ing)?|lips?)\b",
        re.IGNORECASE,
    )
    negative = re.compile(
        r"\b(?:no|not|never|without|avoid|omit|exclude|prevent|don['’]t|"
        r"doesn['’]t|isn['’]t|aren['’]t|cannot|can['’]t|do\s+not|does\s+not)\b",
        re.IGNORECASE,
    )
    parts = re.split(r"(?<=[.!?])\s+|\s*;\s*", str(text or ""))
    kept = [part.strip() for part in parts if part.strip() and not forbidden.search(part) and not negative.search(part)]
    return re.sub(r"\s{2,}", " ", " ".join(kept)).strip()


def _clear_comfy_model_memory():
    result = {
        "comfy_loaded_before": None,
        "comfy_loaded_after": None,
        "comfy_cleanup_calls": [],
        "comfy_cleanup_errors": [],
        "torch_cuda_cache_cleared": False,
    }
    try:
        import comfy.model_management as model_management
        import torch

        loaded_models = getattr(model_management, "loaded_models", None)
        if callable(loaded_models):
            try:
                result["comfy_loaded_before"] = len(loaded_models())
            except Exception as exc:
                result["comfy_cleanup_errors"].append(f"loaded_models before: {exc}")

        unload_all_models = getattr(model_management, "unload_all_models", None)
        if callable(unload_all_models):
            unload_all_models()
            result["comfy_cleanup_calls"].append("unload_all_models")

        cleanup_models_gc = getattr(model_management, "cleanup_models_gc", None)
        if callable(cleanup_models_gc):
            cleanup_models_gc()
            result["comfy_cleanup_calls"].append("cleanup_models_gc")

        cleanup_models = getattr(model_management, "cleanup_models", None)
        if callable(cleanup_models):
            cleanup_models()
            result["comfy_cleanup_calls"].append("cleanup_models")

        soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
        if callable(soft_empty_cache):
            try:
                soft_empty_cache(force=True)
            except TypeError:
                soft_empty_cache()
            result["comfy_cleanup_calls"].append("soft_empty_cache")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
            result["torch_cuda_cache_cleared"] = True

        gc.collect()
        if callable(loaded_models):
            try:
                result["comfy_loaded_after"] = len(loaded_models())
            except Exception as exc:
                result["comfy_cleanup_errors"].append(f"loaded_models after: {exc}")
    except Exception as exc:
        result["comfy_cleanup_errors"].append(str(exc))
    return result


def _clear_builder_memory_direct():
    from .cache import _clear_vrgdg_llm_caches

    comfy_result = _clear_comfy_model_memory()
    llm_result = _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
    gc.collect()
    cleanup_calls = ", ".join(comfy_result.get("comfy_cleanup_calls") or []) or "none"
    cleanup_errors = "; ".join(comfy_result.get("comfy_cleanup_errors") or [])
    return {
        "message": (
            "Memory cleanup finished.\n"
            f"Comfy cleanup calls: {cleanup_calls}\n"
            f"Comfy loaded models: {comfy_result.get('comfy_loaded_before')} -> {comfy_result.get('comfy_loaded_after')}\n"
            f"GGUF models unloaded: {llm_result.get('gguf_models_unloaded', 0)}\n"
            f"HF pipelines unloaded: {llm_result.get('hf_pipelines_unloaded', 0)}\n"
            f"CUDA cache cleared: {bool(llm_result.get('cuda_cache_cleared') or comfy_result.get('torch_cuda_cache_cleared'))}"
            + (f"\nCleanup warnings: {cleanup_errors}" if cleanup_errors else "")
        ),
        **comfy_result,
        **llm_result,
    }


def _gemma_choices():
    register_custom_model_root()
    from .gguf import VRGDG_SuperGemmaGGUFChat, VRGDG_QwenGGUF

    return {
        "models": VRGDG_SuperGemmaGGUFChat._list_local_gemma_gguf_choices(),
        "mmproj": VRGDG_SuperGemmaGGUFChat._list_local_mmproj_choices(),
        "qwen_models": VRGDG_QwenGGUF._list_local_qwen_gguf(),
        "qwen_mmproj": VRGDG_QwenGGUF._list_local_qwen_mmproj(),
    }
