import json

import re

import base64

import urllib.error

import urllib.parse

import urllib.request


from PIL import Image


from .VRGDG_ModelPathSettings import register_custom_model_root


_LM_STUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


def _llm_multi_choices():
    try:
        from .LLM import VRGDG_LLM_Multi
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


def _gemma_choices():
    register_custom_model_root()
    from .LLM import VRGDG_SuperGemmaGGUFChat, VRGDG_QwenGGUF

    return {
        "models": VRGDG_SuperGemmaGGUFChat._list_local_gemma_gguf_choices(),
        "mmproj": VRGDG_SuperGemmaGGUFChat._list_local_mmproj_choices(),
        "qwen_models": VRGDG_QwenGGUF._list_local_qwen_gguf(),
        "qwen_mmproj": VRGDG_QwenGGUF._list_local_qwen_mmproj(),
    }
