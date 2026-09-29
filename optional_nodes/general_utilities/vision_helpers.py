import json

import base64

import ssl

import urllib.error

import urllib.parse

import urllib.request


from PIL import Image


_LM_STUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"

_OWN_SERVER_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


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


def _run_own_server_chat(payload, instruction_text, pil_images=None, temperature=0.6, top_p=0.95, max_new_tokens=1200):
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
    url = f"{v1_root}/chat/completions"
    try:
        data = _own_server_request_json(url, payload, body=body, timeout=_own_server_timeout(payload))
    except RuntimeError as exc:
        message = str(exc).lower()
        if "max_tokens" in message and ("unknown" in message or "unsupported" in message or "400" in message):
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
    body = {
        "model": model,
        "input": input_value,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "context_length": _lm_studio_context_limit(payload),
        "max_output_tokens": _runner_output_token_limit(payload, max_new_tokens),
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


def _run_llm_api_vision(payload, instruction_text, pil_images):
    try:
        from .LLM import VRGDG_LLM_Multi
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


def _pil_image_to_data_url(image, max_height=512, quality=88):
    if image is None:
        raise ValueError("LM Studio vision image is missing.")
    image = image.convert("RGB")
    if image.height > max_height:
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        width = max(1, int(image.width * (float(max_height) / max(1, image.height))))
        image = image.resize((width, int(max_height)), resample)
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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
