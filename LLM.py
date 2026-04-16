import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from functools import lru_cache
from typing import Optional, Tuple
import json
import base64
import re
import uuid
import urllib.request
import urllib.error
import folder_paths

_HF_PIPELINE_CACHE: dict[tuple, tuple] = {}


@lru_cache(maxsize=1)
def _load_google_genai_client():
    try:
        from google import genai as genai_new  # type: ignore
        return genai_new
    except Exception:
        return None


def _google_rest_parts_from_contents(contents) -> list[dict]:
    if not isinstance(contents, list):
        contents = [contents]

    parts = []
    for item in contents:
        if isinstance(item, str):
            if item.strip():
                parts.append({"text": item})
            continue
        if isinstance(item, Image.Image):
            buf = BytesIO()
            item.convert("RGB").save(buf, format="PNG")
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
                    }
                }
            )
            continue
        parts.append({"text": str(item)})
    return parts


def _google_generate_content_rest(api_key: str, model: str, contents) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": _google_rest_parts_from_contents(contents),
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "VRGDG-LLM-Multi/1.0 (+ComfyUI)",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise Exception(f"Google REST HTTP {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise Exception(f"Google REST network error: {e}")


def _google_generate_content(api_key: str, model: str, contents):
    genai_new = _load_google_genai_client()
    if genai_new is not None and hasattr(genai_new, "Client"):
        client = genai_new.Client(api_key=api_key)
        return client.models.generate_content(model=model, contents=contents)
    return _google_generate_content_rest(api_key=api_key, model=model, contents=contents)


def _extract_google_inline_image(response) -> Optional[Image.Image]:
    if isinstance(response, dict):
        candidates = response.get("candidates", []) or []
    else:
        candidates = getattr(response, "candidates", []) or []
    for cand in candidates:
        if isinstance(cand, dict):
            content = cand.get("content", {}) or {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
        else:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", []) if content is not None else []
        for part in parts:
            if isinstance(part, dict):
                inline_data = part.get("inlineData", None) or part.get("inline_data", None)
            else:
                inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            if isinstance(inline_data, dict):
                data_bytes = inline_data.get("data", None)
            else:
                data_bytes = getattr(inline_data, "data", None)
            if not data_bytes:
                continue
            try:
                if isinstance(data_bytes, str):
                    data_bytes = base64.b64decode(data_bytes)
                return Image.open(BytesIO(data_bytes)).convert("RGB")
            except Exception:
                pass
    return None


class VRGDG_NanoBananaPro:
    """
    Simple standalone Gemini Nano Banana node
    - API key typed directly into node
    - Prompt box
    - Up to 4 optional image inputs
    - Landscape-only output instruction
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "A cinematic wide landscape", "multiline": True}),
                "model": ([
                    "gemini-3-pro-image-preview",
                    "gemini-3.1-flash-image-preview",
                ], {"default": "gemini-3-pro-image-preview"}),
            },
            "optional": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "VRGDG/NanoBananaPro"

    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> list[Image.Image]:
        if tensor.ndim == 4:
            batch = tensor
        else:
            batch = tensor.unsqueeze(0)
        images = []
        for i in range(batch.shape[0]):
            arr = (batch[i].cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(arr))
        return images

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def generate(
        self,
        api_key: str,
        prompt: str,
        model: str,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:

        if not api_key.strip():
            raise Exception("API key missing")

        contents = []

        # Optional reference images
        for img in [image1, image2, image3, image4]:
            if img is not None:
                contents.extend(self._tensor_to_pil_list(img))

        # Force landscape instruction
        full_prompt = (
            "Generate ONE landscape-only wide image (16:9). "
            "Never return portrait or square.\n\n"
            f"Prompt: {prompt}"
        )

        contents.append(full_prompt)

        response = _google_generate_content(api_key=api_key, model=model, contents=contents)

        pil_img = _extract_google_inline_image(response)
        if pil_img is not None:
            return (self._pil_to_tensor(pil_img),)

        raise Exception("No image returned")


class VRGDG_LLM_Multi:
    """
    Multi-provider text LLM node.
    - API key input
    - Provider dropdown
    - Model dropdown
    - Prompt input
    - Text output
    """

    PROVIDER_MODELS = {
        "openai": [
            "gpt-image-1",
            "gpt-5-nano",
            "o4-mini",
            "gpt-4.1",
            "gpt-4o",
        ],
        "anthropic": [
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
        ],
        "google": [
            "gemini-3-pro-preview",
            "gemini-3-pro-image-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        "xai": [
            "grok-4",
            "grok-4-latest",
            "grok-3",
            "grok-3-latest",
            "grok-3-mini",
            "grok-3-mini-latest",
        ],
        "deepseek": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "openrouter": [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.1-70b-instruct",
        ],
        "apifreellm": [
            "apifreellm",
        ],
    }
    DEFAULT_MODEL = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-2.5-flash",
        "xai": "grok-3-latest",
        "deepseek": "deepseek-chat",
        "openrouter": "openai/gpt-4o",
        "apifreellm": "apifreellm",
    }
    ALL_MODELS = [m for models in PROVIDER_MODELS.values() for m in models]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "provider": (list(cls.PROVIDER_MODELS.keys()), {"default": "openai"}),
                "model": (cls.ALL_MODELS, {"default": "gpt-4o"}),
                "prompt": ("STRING", {"default": "Write a concise answer.", "multiline": True}),
                "custom_model": ("STRING", {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("text", "used_provider", "used_model", "status", "image")
    FUNCTION = "generate_text"
    CATEGORY = "VRGDG/NanoBananaPro"

    def _post_json(self, url: str, headers: dict, payload: dict) -> dict:
        if "Accept" not in headers:
            headers["Accept"] = "application/json"
        if "User-Agent" not in headers:
            headers["User-Agent"] = "VRGDG-LLM-Multi/1.0 (+ComfyUI)"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise Exception(f"HTTP {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {e}")

    def _get_json(self, url: str, headers: Optional[dict] = None, timeout: int = 30) -> dict:
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except Exception:
            return {}

    def _post_multipart(self, url: str, headers: dict, fields: dict, files: list[tuple]) -> dict:
        if "Accept" not in headers:
            headers["Accept"] = "application/json"
        if "User-Agent" not in headers:
            headers["User-Agent"] = "VRGDG-LLM-Multi/1.0 (+ComfyUI)"

        boundary = f"----VRGDG{uuid.uuid4().hex}"
        body = bytearray()

        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        for field_name, filename, mime, file_bytes in files:
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            body.extend(f"Content-Type: {mime}\r\n\r\n".encode("utf-8"))
            body.extend(file_bytes)
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        req = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise Exception(f"HTTP {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {e}")

    def _normalize_error(self, provider: str, err: Exception) -> str:
        msg = str(err)
        low = msg.lower()
        if provider == "deepseek" and "http 402" in low and "insufficient balance" in low:
            return (
                "error: DeepSeek account has insufficient balance (HTTP 402). "
                "Add credits or switch provider/model."
            )
        if provider == "xai" and "http 403" in low and "1010" in low:
            return (
                "error: xAI request blocked (HTTP 403 code 1010). "
                "Usually account/region/firewall edge blocking. Verify xAI API access, "
                "try a different network/IP, or use OpenRouter with a Grok model."
            )
        return f"error: {msg}"

    def _validate_provider_model(self, provider: str, model: str):
        allowed = self.PROVIDER_MODELS.get(provider, [])
        if model not in allowed:
            raise Exception(
                f"Model '{model}' is not valid for provider '{provider}'. "
                f"Choose one of: {', '.join(allowed)}"
            )

    def _resolve_model(self, provider: str, selected_model: str, custom_model: str) -> str:
        custom_model = custom_model.strip()
        if custom_model:
            return custom_model
        allowed = self.PROVIDER_MODELS.get(provider, [])
        if selected_model in allowed:
            return selected_model
        if provider in self.DEFAULT_MODEL:
            return self.DEFAULT_MODEL[provider]
        if allowed:
            return allowed[0]
        raise Exception(f"No models configured for provider '{provider}'")

    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> list[Image.Image]:
        if tensor.ndim == 4:
            batch = tensor
        else:
            batch = tensor.unsqueeze(0)
        images = []
        for i in range(batch.shape[0]):
            arr = (batch[i].cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(arr).convert("RGB"))
        return images

    def _collect_pil_images(self, images: list[Optional[torch.Tensor]]) -> list[Image.Image]:
        out = []
        for img in images:
            if img is not None:
                out.extend(self._tensor_to_pil_list(img))
        return out

    def _pil_to_png_base64(self, img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _pil_to_png_bytes(self, img: Image.Image) -> bytes:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _empty_image(self) -> torch.Tensor:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    def _decode_data_url_image(self, url: str) -> Optional[Image.Image]:
        if not isinstance(url, str):
            return None
        if not url.startswith("data:image"):
            return None
        m = re.match(r"^data:image/[^;]+;base64,(.+)$", url, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        try:
            raw = base64.b64decode(m.group(1))
            return Image.open(BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def _call_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        pil_images: list[Image.Image],
        extra_headers=None,
    ) -> Tuple[str, Optional[Image.Image]]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        if pil_images:
            content = [{"type": "text", "text": prompt}]
            for img in pil_images:
                b64 = self._pil_to_png_base64(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": model,
            "messages": messages,
        }
        data = self._post_json(f"{base_url}/chat/completions", headers, payload)
        choices = data.get("choices", [])
        if not choices:
            raise Exception(f"No choices in response: {data}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        image_out = None
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type in ("text", "output_text"):
                    parts.append(item.get("text", ""))
                elif "content" in item and isinstance(item.get("content"), str):
                    parts.append(item.get("content", ""))
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        image_out = image_out or self._decode_data_url_image(image_url.get("url", ""))
                elif "b64_json" in item:
                    try:
                        raw = base64.b64decode(item.get("b64_json", ""))
                        image_out = image_out or Image.open(BytesIO(raw)).convert("RGB")
                    except Exception:
                        pass
            content = "".join(parts)
        text = str(content).strip()
        if not text and image_out is None:
            raise Exception(f"Empty model text response: {data}")
        if not text and image_out is not None:
            text = "Image generated."
        return text, image_out

    def _call_openai_image_generation(
        self,
        api_key: str,
        model: str,
        prompt: str,
        pil_images: list[Image.Image],
    ) -> Tuple[str, Optional[Image.Image]]:
        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        if pil_images:
            fields = {
                "model": model,
                "prompt": prompt,
                "size": "1024x1024",
            }
            files = [
                ("image", "input.png", "image/png", self._pil_to_png_bytes(pil_images[0])),
            ]
            data = self._post_multipart("https://api.openai.com/v1/images/edits", headers, fields, files)
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "size": "1024x1024",
            }
            headers["Content-Type"] = "application/json"
            data = self._post_json("https://api.openai.com/v1/images/generations", headers, payload)

        arr = data.get("data", [])
        if not arr:
            raise Exception(f"No image data in response: {data}")
        first = arr[0] if isinstance(arr[0], dict) else {}
        b64 = first.get("b64_json", "")
        if not b64:
            raise Exception(f"Missing b64_json in image response: {data}")
        raw = base64.b64decode(b64)
        pil = Image.open(BytesIO(raw)).convert("RGB")
        revised = first.get("revised_prompt", "")
        text = revised.strip() if isinstance(revised, str) and revised.strip() else "Image generated."
        return text, pil

    def _call_anthropic(self, api_key: str, model: str, prompt: str, pil_images: list[Image.Image]) -> Tuple[str, Optional[Image.Image]]:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        content = [{"type": "text", "text": prompt}]
        for img in pil_images:
            b64 = self._pil_to_png_base64(img)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                }
            )

        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": content}],
        }
        data = self._post_json("https://api.anthropic.com/v1/messages", headers, payload)
        content = data.get("content", [])
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        out = "".join(texts).strip()
        if not out:
            raise Exception(f"Empty Anthropic response: {data}")
        return out, None

    def _call_google(self, api_key: str, model: str, prompt: str, pil_images: list[Image.Image]) -> Tuple[str, Optional[Image.Image]]:
        if pil_images:
            contents = []
            contents.extend(pil_images)
            contents.append(prompt)
            response = _google_generate_content(api_key=api_key, model=model, contents=contents)
        else:
            response = _google_generate_content(api_key=api_key, model=model, contents=prompt)
        image_out = None
        txt = None
        if isinstance(response, dict):
            candidates_json = response.get("candidates", [])
            if isinstance(candidates_json, list):
                chunks = []
                for cand in candidates_json:
                    if not isinstance(cand, dict):
                        continue
                    content = cand.get("content", {})
                    parts = content.get("parts", []) if isinstance(content, dict) else []
                    for part in parts:
                        if isinstance(part, dict):
                            t = part.get("text")
                            if isinstance(t, str) and t:
                                chunks.append(t)
                if chunks:
                    txt = "".join(chunks)
        else:
            txt = getattr(response, "text", None)
        if txt and txt.strip():
            return txt.strip(), image_out
        if isinstance(response, dict):
            candidates = response.get("candidates", [])
        else:
            candidates = getattr(response, "candidates", [])
        for cand in candidates:
            if isinstance(cand, dict):
                content = cand.get("content", {})
                parts_iter = content.get("parts", []) if isinstance(content, dict) else []
            else:
                content = getattr(cand, "content", None)
                parts_iter = content.parts if content and hasattr(content, "parts") else []
            if parts_iter:
                parts = []
                for part in parts_iter:
                    if isinstance(part, dict):
                        ptxt = part.get("text", "")
                    else:
                        ptxt = getattr(part, "text", "")
                    if ptxt:
                        parts.append(ptxt)
                    if isinstance(part, dict):
                        inline_data = part.get("inlineData", None) or part.get("inline_data", None)
                    else:
                        inline_data = getattr(part, "inline_data", None)
                    if inline_data is not None:
                        if isinstance(inline_data, dict):
                            mime_type = inline_data.get("mimeType", "") or inline_data.get("mime_type", "")
                            data_bytes = inline_data.get("data", None)
                        else:
                            mime_type = getattr(inline_data, "mime_type", "")
                            data_bytes = getattr(inline_data, "data", None)
                        if mime_type.startswith("image/") and data_bytes:
                            try:
                                if isinstance(data_bytes, str):
                                    data_bytes = base64.b64decode(data_bytes)
                                image_out = image_out or Image.open(BytesIO(data_bytes)).convert("RGB")
                            except Exception:
                                pass
                if parts:
                    return "".join(parts).strip(), image_out
        image_out = image_out or _extract_google_inline_image(response)
        if image_out is not None:
            return "Image generated.", image_out
        raise Exception("Empty Google response")

    def _call_apifreellm(self, api_key: str, model: str, prompt: str) -> Tuple[str, Optional[Image.Image]]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"message": prompt}
        if model.strip():
            payload["model"] = model.strip()

        data = self._post_json("https://apifreellm.com/api/v1/chat", headers, payload)

        if isinstance(data, dict):
            for key in ("response", "message", "text", "output", "content"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip(), None
                if isinstance(val, dict):
                    nested_content = val.get("content")
                    if isinstance(nested_content, str) and nested_content.strip():
                        return nested_content.strip(), None

            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip(), None

        raise Exception(f"Unexpected apifreellm response format: {data}")

    def generate_text(
        self,
        api_key: str,
        provider: str,
        model: str,
        prompt: str,
        custom_model: str,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str, str, str, torch.Tensor]:
        if not api_key.strip():
            raise Exception("API key missing")
        if not prompt.strip():
            raise Exception("Prompt is empty")

        provider = provider.strip().lower()
        if provider not in self.PROVIDER_MODELS:
            raise Exception(f"Unsupported provider: {provider}")
        chosen_model = self._resolve_model(provider, model, custom_model)
        pil_images = self._collect_pil_images([image1, image2, image3, image4])

        try:
            if provider == "openai":
                if chosen_model == "gpt-image-1":
                    text, image_out = self._call_openai_image_generation(api_key, chosen_model, prompt, pil_images)
                else:
                    text, image_out = self._call_openai_compatible(
                        "https://api.openai.com/v1",
                        api_key,
                        chosen_model,
                        prompt,
                        pil_images,
                    )
            elif provider == "xai":
                text, image_out = self._call_openai_compatible(
                    "https://api.x.ai/v1",
                    api_key,
                    chosen_model,
                    prompt,
                    pil_images,
                )
            elif provider == "deepseek":
                text, image_out = self._call_openai_compatible(
                    "https://api.deepseek.com",
                    api_key,
                    chosen_model,
                    prompt,
                    pil_images,
                )
            elif provider == "openrouter":
                text, image_out = self._call_openai_compatible(
                    "https://openrouter.ai/api/v1",
                    api_key,
                    chosen_model,
                    prompt,
                    pil_images,
                    extra_headers={"HTTP-Referer": "https://localhost", "X-Title": "VRGDG_LLM_Multi"},
                )
            elif provider == "anthropic":
                text, image_out = self._call_anthropic(api_key, chosen_model, prompt, pil_images)
            elif provider == "google":
                text, image_out = self._call_google(api_key, chosen_model, prompt, pil_images)
            elif provider == "apifreellm":
                text, image_out = self._call_apifreellm(api_key, chosen_model, prompt)
            else:
                raise Exception(f"Unsupported provider: {provider}")
            status = "ok"
        except Exception as e:
            text = ""
            image_out = None
            status = self._normalize_error(provider, e)
        image_tensor = self._pil_to_tensor(image_out) if image_out is not None else self._empty_image()
        return (text, provider, chosen_model, status, image_tensor)


class VRGDG_LocalLLM:
    """
    Local LLM node for prompt creation/enhancement.
    Supports local backends:
    - Ollama API
    - OpenAI-compatible local servers (LM Studio, vLLM, etc.)
    """

    BACKENDS = ["ollama", "openai_compatible"]
    MODEL_HINTS = [
        "qwen2.5:14b",
        "qwen2.5:32b",
        "qwen2.5-coder:14b",
        "qwen2.5vl:7b",
        "qwen2.5-vl:7b",
        "llava:13b",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (cls.BACKENDS, {"default": "ollama"}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "qwen2.5:14b"}),
                "custom_model": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "Enhance this prompt for image generation.", "multiline": True}),
                "system_prompt": ("STRING", {"default": "You are an expert prompt engineer.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": -1, "max": 262144, "step": 64}),
                "request_timeout_sec": ("INT", {"default": 300, "min": 30, "max": 3600, "step": 10}),
            },
            "optional": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("text", "used_backend", "used_model", "status", "image")
    FUNCTION = "run_local"
    CATEGORY = "VRGDG/NanoBananaPro"

    def _post_json(self, url: str, headers: dict, payload: dict, timeout_sec: int = 180) -> dict:
        if "Accept" not in headers:
            headers["Accept"] = "application/json"
        if "User-Agent" not in headers:
            headers["User-Agent"] = "VRGDG-LocalLLM/1.0 (+ComfyUI)"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise Exception(f"HTTP {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {e}")

    def _get_json(self, url: str, headers: Optional[dict] = None, timeout: int = 30) -> dict:
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except Exception:
            return {}

    def _resolve_model(self, selected_model: str, custom_model: str) -> str:
        m = custom_model.strip()
        return m if m else selected_model

    def _normalize_url(self, base_url: str) -> str:
        return base_url.strip().rstrip("/")

    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> list[Image.Image]:
        if tensor.ndim == 4:
            batch = tensor
        else:
            batch = tensor.unsqueeze(0)
        images = []
        for i in range(batch.shape[0]):
            arr = (batch[i].cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(arr).convert("RGB"))
        return images

    def _collect_pil_images(self, images: list[Optional[torch.Tensor]]) -> list[Image.Image]:
        out = []
        for img in images:
            if img is not None:
                out.extend(self._tensor_to_pil_list(img))
        return out

    def _pil_to_png_base64(self, img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _empty_image(self) -> torch.Tensor:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    def _decode_data_url_image(self, url: str) -> Optional[Image.Image]:
        if not isinstance(url, str) or not url.startswith("data:image"):
            return None
        m = re.match(r"^data:image/[^;]+;base64,(.+)$", url, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        try:
            raw = base64.b64decode(m.group(1))
            return Image.open(BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def _extract_possible_image(self, data: dict) -> Optional[Image.Image]:
        # Common OpenAI-compatible image return shape
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        url_obj = item.get("image_url", {})
                        if isinstance(url_obj, dict):
                            img = self._decode_data_url_image(url_obj.get("url", ""))
                            if img is not None:
                                return img
                    if "b64_json" in item:
                        try:
                            raw = base64.b64decode(item.get("b64_json", ""))
                            return Image.open(BytesIO(raw)).convert("RGB")
                        except Exception:
                            pass

        # Generic direct payload shape
        imgs = data.get("images", [])
        if isinstance(imgs, list) and imgs:
            first = imgs[0]
            if isinstance(first, str):
                img = self._decode_data_url_image(first)
                if img is not None:
                    return img
                try:
                    raw = base64.b64decode(first)
                    return Image.open(BytesIO(raw)).convert("RGB")
                except Exception:
                    pass
            elif isinstance(first, dict):
                if "b64_json" in first:
                    try:
                        raw = base64.b64decode(first.get("b64_json", ""))
                        return Image.open(BytesIO(raw)).convert("RGB")
                    except Exception:
                        pass
                if "url" in first:
                    return self._decode_data_url_image(first.get("url", ""))

        return None

    def _call_ollama(
        self,
        base_url: str,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        pil_images: list[Image.Image],
        request_timeout_sec: int,
    ) -> Tuple[str, Optional[Image.Image]]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "num_predict": int(max_tokens) if int(max_tokens) > 0 else -1,
            },
        }
        if system_prompt.strip():
            payload["system"] = system_prompt
        if pil_images:
            payload["images"] = [self._pil_to_png_base64(img) for img in pil_images]

        data = self._post_json(
            f"{base_url}/api/generate",
            {"Content-Type": "application/json"},
            payload,
            timeout_sec=request_timeout_sec,
        )
        text = str(data.get("response", "")).strip()
        if not text:
            raise Exception(f"Empty Ollama response: {data}")
        img = self._extract_possible_image(data)
        return text, img

    def _list_ollama_models(self, base_url: str) -> list[str]:
        data = self._get_json(f"{base_url}/api/tags")
        models = data.get("models", []) if isinstance(data, dict) else []
        out = []
        for m in models:
            if isinstance(m, dict):
                name = m.get("name")
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
        return out

    def _list_openai_compatible_models(self, base_url: str, api_key: str) -> list[str]:
        headers = {}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        data = self._get_json(f"{base_url}/models", headers=headers)
        arr = data.get("data", []) if isinstance(data, dict) else []
        out = []
        for item in arr:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    out.append(model_id.strip())
        return out

    def _is_model_not_found_error(self, err: Exception) -> bool:
        low = str(err).lower()
        return (
            "http 404" in low
            and "model" in low
            and ("not found" in low or "does not exist" in low)
        )

    def _call_openai_compatible_local(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        pil_images: list[Image.Image],
        request_timeout_sec: int,
    ) -> Tuple[str, Optional[Image.Image]]:
        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        content = [{"type": "text", "text": prompt}]
        for img in pil_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self._pil_to_png_base64(img)}"},
                }
            )

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content if pil_images else prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if int(max_tokens) > 0:
            payload["max_tokens"] = int(max_tokens)
        data = self._post_json(
            f"{base_url}/chat/completions",
            headers,
            payload,
            timeout_sec=request_timeout_sec,
        )

        choices = data.get("choices", [])
        if not choices:
            raise Exception(f"No choices in response: {data}")
        msg = choices[0].get("message", {})
        content_out = msg.get("content", "")
        if isinstance(content_out, list):
            parts = []
            for item in content_out:
                if isinstance(item, dict):
                    if item.get("type") in ("text", "output_text"):
                        parts.append(item.get("text", ""))
                    elif "content" in item and isinstance(item.get("content"), str):
                        parts.append(item.get("content", ""))
            text = "".join(parts).strip()
        else:
            text = str(content_out).strip()
        if not text:
            raise Exception(f"Empty OpenAI-compatible local response: {data}")
        img = self._extract_possible_image(data)
        return text, img

    def _normalize_error(self, backend: str, err: Exception) -> str:
        msg = str(err)
        low = msg.lower()
        if "timed out" in low or "timeout" in low:
            return (
                "error: request timed out. Increase request_timeout_sec in the node "
                "(for large/slow models try 300-900)."
            )
        if backend == "ollama" and "http 404" in low and "model" in low and "not found" in low:
            return (
                "error: Ollama model not found. Pick an installed model in the node, "
                "or install it first with: ollama pull <model_name>."
            )
        if backend == "openai_compatible" and self._is_model_not_found_error(err):
            return "error: model not found on this OpenAI-compatible server. Pick a model from /models."
        if "connection refused" in low or "failed to establish a new connection" in low:
            if backend == "ollama":
                return (
                    "error: cannot reach Ollama at base_url. Start Ollama and verify base_url "
                    "(default http://127.0.0.1:11434)."
                )
            return "error: cannot reach local OpenAI-compatible server at base_url."
        return f"error: {msg}"

    def run_local(
        self,
        backend: str,
        base_url: str,
        api_key: str,
        model: str,
        custom_model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        request_timeout_sec: int,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str, str, str, torch.Tensor]:
        backend = backend.strip().lower()
        if backend not in self.BACKENDS:
            raise Exception(f"Unsupported backend: {backend}")
        if not prompt.strip():
            raise Exception("Prompt is empty")

        url = self._normalize_url(base_url)
        chosen_model = self._resolve_model(model, custom_model)
        pil_images = self._collect_pil_images([image1, image2, image3, image4])
        request_timeout_sec = max(30, int(request_timeout_sec))

        if backend == "ollama":
            installed = self._list_ollama_models(url)
            if installed and chosen_model not in installed:
                chosen_model = installed[0]

        try:
            if backend == "ollama":
                try:
                    text, img = self._call_ollama(
                        url,
                        chosen_model,
                        prompt,
                        system_prompt,
                        temperature,
                        top_p,
                        max_tokens,
                        pil_images,
                        request_timeout_sec,
                    )
                except Exception as e:
                    if self._is_model_not_found_error(e):
                        installed = self._list_ollama_models(url)
                        if installed:
                            chosen_model = installed[0]
                            text, img = self._call_ollama(
                                url,
                                chosen_model,
                                prompt,
                                system_prompt,
                                temperature,
                                top_p,
                                max_tokens,
                                pil_images,
                                request_timeout_sec,
                            )
                        else:
                            raise
                    else:
                        raise
            else:
                try:
                    text, img = self._call_openai_compatible_local(
                        url,
                        api_key,
                        chosen_model,
                        prompt,
                        system_prompt,
                        temperature,
                        top_p,
                        max_tokens,
                        pil_images,
                        request_timeout_sec,
                    )
                except Exception as e:
                    if self._is_model_not_found_error(e):
                        available = self._list_openai_compatible_models(url, api_key)
                        if available:
                            chosen_model = available[0]
                            text, img = self._call_openai_compatible_local(
                                url,
                                api_key,
                                chosen_model,
                                prompt,
                                system_prompt,
                                temperature,
                                top_p,
                                max_tokens,
                                pil_images,
                                request_timeout_sec,
                            )
                        else:
                            raise
                    else:
                        raise
            status = "ok"
        except Exception as e:
            text = ""
            img = None
            status = self._normalize_error(backend, e)

        image_tensor = self._pil_to_tensor(img) if img is not None else self._empty_image()
        return (text, backend, chosen_model, status, image_tensor)


class VRGDG_Qwen35:
    """
    Local Hugging Face Qwen node for prompt creation and image-aware prompt writing.
    - Supports manually downloaded local folders or Hugging Face repo IDs
    - Optional download-on-demand when the model is missing
    - Dynamic image inputs handled in web/VRGDG_Qwen35_dynamic.js
    """

    MAX_IMAGES = 24
    MODEL_PRESETS = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-0.8B-Base",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-2B-Base",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-4B-Base",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-9B-Base",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-27B-FP8",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-35B-A3B-FP8",
        "Qwen/Qwen3.5-35B-A3B-Base",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-122B-A10B-FP8",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "Qwen/Qwen3.5-27B-GPTQ-Int4",
        "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
        "custom",
    ]
    DTYPE_OPTIONS = ["auto", "bfloat16", "float16", "float32"]
    DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
    TASK_PRESETS = [
        "text_to_image",
        "text_to_video",
        "image_to_video",
        "image_edit",
        "captioner_training",
        "custom",
    ]
    REASONING_GUARD_TEXT = (
        "DO NOT SHOW OR DISPLAY ANY REASONING Output only one final prompt paragraph. "
        "No analysis, no steps, no checklist. Output exactly one paragraph only. "
        "No bullets. No field labels. No checklist. No analysis."
    )

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"image{i}": ("IMAGE", {"tooltip": "Optional reference image input."})
            for i in range(1, cls.MAX_IMAGES + 1)
        }
        return {
            "required": {
                "model_preset": (
                    cls.MODEL_PRESETS,
                    {
                        "default": "Qwen/Qwen3.5-4B",
                        "tooltip": "Choose a model preset. Use custom_model_id to override this.",
                    },
                ),
                "custom_model_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional override. Can be a Hugging Face repo id or a local model folder path.",
                    },
                ),
                "task_preset": (
                    cls.TASK_PRESETS,
                    {
                        "default": "text_to_image",
                        "tooltip": "Select a task preset with built-in instructions.",
                    },
                ),
                "custom_instructions": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Used only when task_preset is custom. Enter your own full instruction block.",
                    },
                ),
                "user_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Your task details and creative direction for the selected preset.",
                    },
                ),
                "trigger_word": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional LoRA/training trigger token. Used only by Captioner preset.",
                    },
                ),
                "image_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": cls.MAX_IMAGES,
                        "step": 1,
                        "tooltip": "How many optional image inputs to show on the node.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, missing models can be downloaded to ComfyUI/models/LLM/Qwen.",
                    },
                ),
                "device": (
                    cls.DEVICE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Inference device selection.",
                    },
                ),
                "dtype": (
                    cls.DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Inference precision. Auto is recommended.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Higher = more creative variation, lower = more deterministic output.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling cutoff.",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 800,
                        "min": 32,
                        "max": 32000,
                        "step": 32,
                        "tooltip": "Maximum number of output tokens.",
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "used_model", "status")
    FUNCTION = "generate_prompt"
    CATEGORY = "VRGDG/LLM"

    def _resolve_model_id(self, model_preset: str, custom_model_id: str) -> str:
        custom = str(custom_model_id or "").strip()
        if custom:
            return custom
        preset = str(model_preset or "").strip()
        if not preset or preset == "custom":
            raise Exception("Pick a model preset or provide custom_model_id.")
        return preset

    def _preset_instruction(self, task_preset: str) -> str:
        p = str(task_preset or "").strip().lower()
        if p == "text_to_image":
            return (
                "You are a text-to-image prompt generator that creates one highly detailed prompt for image generation.\n\n"
                "After the user provides details, generate one single prompt based on their input.\n"
                "If the user does not provide enough details, creatively fill in the missing elements while keeping the prompt coherent and visually compelling.\n\n"
                "Your output must be only the final prompt text. Do not output a list, explanation, or table.\n\n"
                "Prompt Creation Guidelines:\n\n"
                "Color Style:\n"
                "Select a color grading style that enhances the mood. Examples: Natural, Matte, HDR, Cinematic, Vintage, Grunge, Black and White, Split Tone, High Contrast.\n\n"
                "Mood:\n"
                "Define the emotional tone of the image. Examples: Bright, Dark, Epic, Dramatic, Cinematic, Peaceful, Mysterious.\n\n"
                "Subject Description:\n"
                "Clearly describe the main person or subject in the image including clothing, hairstyle, hair color, distinct physical features, and pose or action.\n\n"
                "Environment / Setting:\n"
                "Describe a background or environment that complements the color style and mood.\n\n"
                "Camera Angle / Perspective:\n"
                "Choose an angle that enhances the scene. Examples: low angle, eye level, aerial view, over-the-shoulder, close-up, wide shot.\n\n"
                "Weather and Time of Day:\n"
                "Specify weather conditions and time of day to enhance atmosphere.\n\n"
                "Additional Cinematic Details:\n"
                "Add extra visual elements that improve composition such as fog, dramatic lighting, reflections, depth of field, motion blur, particles, or environmental effects.\n\n"
                "Required Prompt Structure:\n"
                "\"[Color Style + Mood] photograph of [Subject description]. [Clothing and appearance details]. "
                "The scene takes place in [environment description]. Camera is [camera angle]. "
                "The weather is [weather] during [time of day]. Additional cinematic details: [extra visual elements].\"\n\n"
                "DO NOT SHOW OR DISPLAY ANY REASONING Output only one final prompt paragraph. No analysis, no steps, no checklist.Output exactly one paragraph only. No bullets. No field labels. No checklist. No analysis"
            )
        if p == "text_to_video":
            return (
                "You are an AI assistant specialized in enhancing creative prompts for cinematic text-to-video generation. "
                "Your goal is to transform simple prompts or ideas into one detailed, vivid cinematic description that includes "
                "camera movement, lighting, atmosphere, mood, and visual composition.\n\n"
                "The prompt should be around 80-100 words and emphasize strong visual storytelling, environment, and cinematic motion. "
                "Include camera techniques such as panning, dolly in/out, tracking shots, tilt, and zoom. Avoid overly fast camera "
                "movements such as whip pans or crash zooms.\n\n"
                "Include lighting descriptions such as soft light, hard light, backlighting, rim lighting, or volumetric lighting. "
                "Use composition techniques such as close-ups, wide shots, low angles, high angles, and environmental framing. "
                "Create mood using terms like somber, euphoric, mysterious, dramatic, or dreamlike.\n\n"
                "Do not mention audio or sound elements. Focus strictly on visual cinematic storytelling.\n\n"
                "Always generate prompts that produce strong visual imagery and align with Wan 2.1 strengths and limitations.\n\n"
                "Output exactly one final prompt paragraph. Do not output multiple prompts, tables, lists, analysis, steps, reasoning, or explanations.\n\n"
                "---------------------------------\n"
                "ENVIRONMENT-FOCUSED WIDE-DISTANCE PATTERN\n"
                "---------------------------------\n"
                "Use for wide landscapes or large spaces with distant framing.\n\n"
                "Structure:\n"
                "Wide Shot Setup →\n"
                "Character + Outfit in Environment →\n"
                "Movement Through Space →\n"
                "Literal Camera Motion →\n"
                "Neutral Environmental Atmosphere →\n"
                "Lighting →\n"
                "Natural Tonal Description (no LUT wording)\n\n"
                "---------------------------------\n"
                "MID–SEMI CLOSE-UP CINEMATIC PATTERN\n"
                "---------------------------------\n"
                "Use for waist-up, chest-up, or profile shots with subtle actions.\n\n"
                "Structure:\n"
                "Mid/Close-Up Setup →\n"
                "Character + Outfit →\n"
                "Small Grounded Action →\n"
                "Minimal Natural Camera Motion →\n"
                "Neutral Environmental Interaction →\n"
                "Natural Lighting →\n"
                "Natural Tonal Description (no LUT wording)\n\n"
                "---------------------------------\n"
                "CLOSE-UP TO MID TRANSITION PATTERN\n"
                "---------------------------------\n"
                "Use for tight face shots that may remain close or pull back.\n\n"
                "Structure:\n"
                "Close-Up Setup →\n"
                "Character + Outfit →\n"
                "Small Natural Action →\n"
                "Slow Pull-Back / Stable Camera →\n"
                "Simple Neutral Background →\n"
                "Even Natural Lighting →\n"
                "Natural Tonal Description (no LUT wording).\n\n"
                "Output only one final cinematic prompt paragraph."
            )
        if p == "image_to_video":
            return (
                "You are an AI assistant that enhances simple ideas into cinematic image-to-video prompts.\n\n"
                "The user will provide a starting image and may include additional details such as subject motion, "
                "camera movement, or environmental movement.\n\n"
                "Your task is to generate one cinematic animation prompt approximately 80-100 words describing how the scene animates.\n\n"
                "The animation must naturally continue from the provided scene, maintaining the same environment, composition, and lighting.\n"
                "Do not reference the input image, starting frame, or source image in the final prompt.\n\n"
                "Focus on describing the following elements:\n"
                "Subject or character movement\n"
                "Camera motion such as slow pan, dolly, tracking, tilt, or subtle zoom\n"
                "Environmental motion such as wind, fog, water movement, or background activity\n"
                "Mood and atmosphere\n"
                "Framing and composition\n\n"
                "You may briefly describe the existing lighting but do not introduce new light sources or lighting changes.\n\n"
                "Avoid audio descriptions and avoid fast chaotic camera movements such as whip pans or crash zooms.\n"
                "The motion should feel smooth, cinematic, and natural.\n\n"
                "Prompt Structure:\n"
                "Scene description → Subject action → Camera motion → Environmental motion → Lighting → Cinematic tone\n\n"
                "Example:\n"
                "A cinematic wide beach scene with a woman standing at the shoreline in a flowing red dress. "
                "She gently shifts her stance as small waves move around her ankles and her braided hair drifts softly "
                "in the ocean breeze. The camera slowly dollies forward, keeping her centered in the frame while the "
                "horizon stretches behind her. Seagulls glide across the sky and waves roll steadily toward the shore. "
                "Lighting is warm late-afternoon sunlight. The scene feels calm, atmospheric, and naturally alive.\n"
                "DO NOT SHOW OR DISPLAY ANY REASONING Output only one final prompt paragraph. No analysis, no steps, no checklist."
            )
        if p == "image_edit":
            return (
                "You generate high-quality image editing prompts.\n\n"
                "Inputs may include:\n"
                "- One or more reference images (character, object, or scene)\n"
                "- Optional user instructions describing changes or a new scene.\n\n"
                "Your task is to produce a single clear cinematic image generation prompt.\n\n"
                "1. The prompt MUST begin with EXACTLY ONE of these two strings (match capitalization and punctuation exactly):\n"
                "   A) If there is ONLY character/object reference (no scene/environment reference):\n"
                "      Using the provided reference images.\n"
                "   B) If there is BOTH (1) at least one character reference AND (2) at least one scene/environment reference:\n"
                "      Using the provided character and scene reference images.\n"
                "   - If the user provided two images and one is a character/subject and the other is a location/background, treat them as character + scene.\n"
                "   - Do not use any other opening text.\n"
                "2. Preserve the identity, appearance, and key details of subjects from the reference images or details of the reference scenes.\n"
                "3. Apply the user's requested changes such as clothing, environment, pose, mood, or composition.\n"
                "4. If multiple references are provided:\n"
                "   - Character references define the subject's appearance.\n"
                "   - Scene references define the environment, lighting, color tone, and atmosphere.\n"
                "5. Describe camera angle, lighting, environment, mood, motion, and realism.\n"
                "6. Use natural descriptive language optimized for image generation models.\n\n"
                "Guidelines:\n"
                "- Be concise but visually descriptive.\n"
                "- Blend characters and objects naturally into the environment.\n"
                "- Emphasize cinematic composition, lighting realism, environmental detail, and photorealistic blending.\n\n"
                "Output only the final image generation prompt.\n\n"
                "DO NOT SHOW OR DISPLAY ANY REASONING Output only one final prompt paragraph. No analysis, no steps, no checklist.\n"
                "user input takes priority and trumps any above rules and instructions."
            )
        if p == "captioner_training":
            return (
                "You generate structured captions for LoRA and image training datasets.\n\n"
                "Follow these rules when generating captions.\n\n"
                "If a trigger word is provided, place it at the very beginning of the caption.\n"
                "If no trigger word is provided, begin with the subject description.\n\n"
                "Caption Structure:\n"
                "Subject description first.\n"
                "Include gender and optionally approximate age.\n"
                "Include hair color and hairstyle.\n"
                "Include facial expression if visible.\n"
                "Describe visible clothing.\n"
                "Describe background color or environment.\n"
                "Describe lighting style such as soft studio lighting or natural light.\n"
                "Describe camera framing such as headshot, portrait, waist-up, or full body.\n"
                "Mention camera angle if notable such as front-facing or profile.\n\n"
                "Include photography style tags useful for training such as:\n"
                "studio photo\n"
                "shallow depth of field\n"
                "professional portrait\n\n"
                "Formatting Rules:\n"
                "Use simple comma-separated phrases.\n"
                "Do not write full sentences.\n"
                "Do not include storytelling or narrative.\n"
                "Do not include identities, names, or character labels.\n"
                "Keep captions short, structured, and consistent.\n"
                "DO NOT SHOW OR DISPLAY ANY REASONING Output only one final prompt paragraph. No analysis, no steps, no checklist."
            )
        return ""

    def _build_instruction_text(
        self,
        task_preset: str,
        user_input: str,
        trigger_word: str,
        custom_instructions: str,
    ) -> str:
        base = self._preset_instruction(task_preset)
        user_text = str(user_input or "").strip()
        custom_text = str(custom_instructions or "").strip()
        guard_text = self.REASONING_GUARD_TEXT
        if str(task_preset or "").strip().lower() == "custom":
            if custom_text and user_text:
                return f"{custom_text}\n\nUser details:\n{user_text}\n\n{guard_text}"
            out = custom_text or user_text
            return f"{out}\n\n{guard_text}" if out else out
        if str(task_preset or "").strip().lower() == "captioner_training":
            t = str(trigger_word or "").strip()
            if t:
                user_text = f"Trigger word: {t}\n{user_text}"
            # Caption preset has its own line-based format rules.
            guard_text = "DO NOT SHOW OR DISPLAY ANY REASONING. Output only the final caption text."
        if not base:
            return f"{user_text}\n\n{guard_text}" if user_text else user_text
        if not user_text:
            return f"{base}\n\n{guard_text}"
        return f"{base}\n\nUser details:\n{user_text}\n\n{guard_text}"

    def _comfy_hf_cache_dir(self) -> Optional[str]:
        base = getattr(folder_paths, "models_dir", None)
        if not base:
            return None
        return os.path.join(base, "LLM", "Qwen")

    def _model_repo_local_dir(self, model_id: str) -> Optional[str]:
        if not isinstance(model_id, str):
            return None
        raw = model_id.strip()
        if not raw or os.path.isdir(raw):
            return raw or None
        if "/" not in raw:
            return None
        cache_root = self._comfy_hf_cache_dir()
        if not cache_root:
            return None
        safe_name = raw.replace("/", "--").replace("\\", "--").replace(":", "_")
        return os.path.join(cache_root, safe_name)

    def _resolve_model_source(self, model_id: str, download_if_missing: bool, hf_token: str = "") -> tuple[str, dict]:
        common_kwargs = {}
        token = str(hf_token or "").strip()
        if token:
            common_kwargs["token"] = token
        local_dir = self._model_repo_local_dir(model_id)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            if os.path.isdir(model_id):
                return model_id, common_kwargs
            try:
                from huggingface_hub import snapshot_download
            except Exception:
                snapshot_download = None

            has_local_files = any(os.scandir(local_dir))
            if has_local_files:
                return local_dir, common_kwargs
            if download_if_missing:
                if snapshot_download is None:
                    raise Exception("Missing dependency: install huggingface_hub to download models.")
                try:
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        token=token if token else None,
                    )
                except Exception as e:
                    raise Exception(
                        self._format_model_load_error(
                            e,
                            model_id=model_id,
                            model_source=local_dir,
                            download_if_missing=download_if_missing,
                            has_token=bool(token),
                        )
                    ) from e
                return local_dir, common_kwargs
            common_kwargs["local_files_only"] = True
            return local_dir, common_kwargs

        cache_dir = self._comfy_hf_cache_dir()
        if cache_dir:
            common_kwargs["cache_dir"] = cache_dir
        common_kwargs["local_files_only"] = not bool(download_if_missing)
        return model_id, common_kwargs

    def _torch_dtype(self, dtype_name: str):
        name = str(dtype_name or "auto").strip().lower()
        if name == "auto":
            return "auto"
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping.get(name, "auto")

    def _extra_model_kwargs(self, model_id: str, task: str) -> dict:
        return {}

    def _format_model_load_error(
        self,
        err: Exception,
        model_id: str,
        model_source: str,
        download_if_missing: bool,
        has_token: bool,
    ) -> str:
        msg = str(err)
        low = msg.lower()
        local_dir = self._model_repo_local_dir(model_id) or str(model_source)

        if "403" in low or "gated repo" in low or "not in the authorized list" in low:
            if not has_token:
                return (
                    f"Hugging Face access denied for '{model_id}'. This appears to be gated.\n"
                    "Set hf_token in the node and ensure your HF account has accepted model access."
                )
            return (
                f"Hugging Face access denied for '{model_id}' even with token.\n"
                "Verify your token account has accepted terms and has explicit model access."
            )
        if "401" in low or "unauthorized" in low or "invalid token" in low:
            return (
                f"Authentication failed for '{model_id}'.\n"
                "Check hf_token value and ensure it has read access."
            )
        if (
            "cannot find the requested files in the local cache" in low
            or "error happened while trying to locate the file on the hub" in low
            or "connection" in low
            or "timed out" in low
            or "name resolution" in low
        ):
            return (
                f"Could not download files for '{model_id}' from Hugging Face.\n"
                "Likely causes: network/proxy/firewall issues, unstable internet, or insufficient model access.\n"
                f"If a partial cache exists, delete and retry: {local_dir}"
            )
        if "unrecognized processing class" in low or "can't instantiate a processor" in low:
            return (
                f"Local files for '{model_id}' look incomplete or incompatible.\n"
                f"Delete this local cache folder and retry download: {local_dir}"
            )
        if not download_if_missing and ("local_files_only" in low or "not found" in low):
            return (
                f"Model '{model_id}' not found locally.\n"
                "Enable download_if_missing or set custom_model_id to a valid local model folder."
            )
        return msg

    def _load_model_with_attention_fallback(self, model_cls, model_source: str, model_kwargs: dict):
        try:
            return model_cls.from_pretrained(model_source, **model_kwargs)
        except Exception as e:
            low = str(e).lower()
            needs_flash_fallback = (
                "flashattention2" in low
                or "flash attention 2" in low
                or "flash_attn" in low
            )
            if not needs_flash_fallback:
                raise
            retry_kwargs = dict(model_kwargs)
            retry_kwargs["attn_implementation"] = "eager"
            retry_kwargs["use_flash_attention_2"] = False
            return model_cls.from_pretrained(model_source, **retry_kwargs)

    def _pipeline_key(
        self,
        task: str,
        model_id: str,
        device: str,
        dtype: str,
        trust_remote_code: bool,
        download_if_missing: bool,
    ) -> tuple:
        return (
            task,
            model_id,
            str(device or "auto").strip().lower(),
            str(dtype or "auto").strip().lower(),
            bool(trust_remote_code),
            bool(download_if_missing),
        )

    def _load_pipeline(
        self,
        task: str,
        model_id: str,
        device: str,
        dtype: str,
        trust_remote_code: bool,
        download_if_missing: bool,
        hf_token: str = "",
    ):
        key = self._pipeline_key(task, model_id, device, dtype, trust_remote_code, download_if_missing)
        cached = _HF_PIPELINE_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
                pipeline,
            )
        except Exception as e:
            raise Exception(
                "Missing dependency: install transformers and accelerate to use the local Qwen node."
            ) from e

        model_source, source_kwargs = self._resolve_model_source(model_id, download_if_missing, hf_token=hf_token)
        common_kwargs = {
            "trust_remote_code": bool(trust_remote_code),
            **source_kwargs,
        }

        torch_dtype = self._torch_dtype(dtype)

        device_value = str(device or "auto").strip().lower()
        model_kwargs = dict(common_kwargs)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs.update(self._extra_model_kwargs(model_id, task))
        if device_value == "auto":
            model_kwargs["device_map"] = "auto"
        elif device_value == "cpu":
            model_kwargs["device_map"] = "cpu"

        try:
            config = AutoConfig.from_pretrained(model_source, **common_kwargs)
        except Exception:
            config = None

        model_type = str(getattr(config, "model_type", "") or "").lower()
        is_vision = (
            task == "image-text-to-text"
            or "vision" in model_type
            or "vl" in model_type
            or "qwen3_5" in model_type
        )

        try:
            if is_vision:
                processor = AutoProcessor.from_pretrained(model_source, **common_kwargs)
                model = self._load_model_with_attention_fallback(
                    AutoModelForImageTextToText, model_source, model_kwargs
                )
                pipe = pipeline(
                    "image-text-to-text",
                    model=model,
                    processor=processor,
                )
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_source, **common_kwargs)
                except Exception:
                    processor = None
                    try:
                        processor = AutoProcessor.from_pretrained(model_source, **common_kwargs)
                    except Exception:
                        processor = None
                    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
                    if tokenizer is None:
                        processor = AutoProcessor.from_pretrained(model_source, **common_kwargs)
                        model = self._load_model_with_attention_fallback(
                            AutoModelForImageTextToText, model_source, model_kwargs
                        )
                        pipe = pipeline(
                            "image-text-to-text",
                            model=model,
                            processor=processor,
                        )
                    else:
                        model = self._load_model_with_attention_fallback(
                            AutoModelForCausalLM, model_source, model_kwargs
                        )
                        pipe_kwargs = {
                            "task": task,
                            "model": model,
                            "tokenizer": tokenizer,
                        }
                        if device_value == "cuda":
                            pipe_kwargs["device"] = 0
                        elif device_value == "cpu":
                            pipe_kwargs["device"] = -1
                        pipe = pipeline(**pipe_kwargs)
                else:
                    model = self._load_model_with_attention_fallback(
                        AutoModelForCausalLM, model_source, model_kwargs
                    )
                    pipe_kwargs = {
                        "task": task,
                        "model": model,
                        "tokenizer": tokenizer,
                    }
                    if device_value == "cuda":
                        pipe_kwargs["device"] = 0
                    elif device_value == "cpu":
                        pipe_kwargs["device"] = -1
                    pipe = pipeline(**pipe_kwargs)
        except Exception as e:
            raise Exception(
                self._format_model_load_error(
                    e,
                    model_id=model_id,
                    model_source=model_source,
                    download_if_missing=download_if_missing,
                    has_token=bool(str(hf_token or "").strip()),
                )
            ) from e

        _HF_PIPELINE_CACHE[key] = pipe
        return pipe

    def _get_context_window(self, pipe) -> str:
        tokenizer = getattr(pipe, "tokenizer", None)
        model = getattr(pipe, "model", None)

        candidates = []
        for obj in (tokenizer, getattr(tokenizer, "model_max_length", None), model, getattr(model, "config", None)):
            if obj is None:
                continue
            if isinstance(obj, int):
                candidates.append(obj)
                continue
            for attr in ("model_max_length", "max_position_embeddings", "max_sequence_length", "seq_length"):
                value = getattr(obj, attr, None)
                if isinstance(value, int):
                    candidates.append(value)

        filtered = [v for v in candidates if isinstance(v, int) and 0 < v < 10**9]
        if filtered:
            return str(max(filtered))
        return "unknown"

    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> list[Image.Image]:
        if tensor.ndim == 4:
            batch = tensor
        else:
            batch = tensor.unsqueeze(0)
        images = []
        for i in range(batch.shape[0]):
            arr = (batch[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr).convert("RGB"))
        return images

    def _collect_pil_images(self, image_count: int, kwargs: dict) -> list[Image.Image]:
        images = []
        count = max(0, min(self.MAX_IMAGES, int(image_count)))
        for idx in range(1, count + 1):
            tensor = kwargs.get(f"image{idx}")
            if tensor is None:
                continue
            images.extend(self._tensor_to_pil_list(tensor))
        return images

    def _extract_generated_text(self, output) -> str:
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                generated = first.get("generated_text", "")
                if isinstance(generated, list) and generated:
                    last = generated[-1]
                    if isinstance(last, dict):
                        content = last.get("content", "")
                        if isinstance(content, list):
                            parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    parts.append(str(item.get("text", "")))
                                elif isinstance(item, str):
                                    parts.append(item)
                            return "\n".join([p for p in parts if p]).strip()
                        return str(content).strip()
                    return str(last).strip()
                return str(generated).strip()
        if isinstance(output, dict):
            for key in ("generated_text", "text"):
                if key in output:
                    return str(output.get(key, "")).strip()
        return str(output).strip()

    def _strip_thinking_text(self, text: str) -> str:
        cleaned = str(text or "")
        # Remove explicit reasoning headers if the model emits them verbatim.
        cleaned = re.sub(r"(?is)^\s*thinking process\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?is)^\s*reasoning\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?is)^\s*analysis\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?is)^\s*\*+\s*thinking process\s*:?\s*\*+\s*", "", cleaned)
        cleaned = re.sub(r"(?is)^\s*\*+\s*reasoning\s*:?\s*\*+\s*", "", cleaned)
        cleaned = re.sub(r"(?is)^\s*\*+\s*analysis\s*:?\s*\*+\s*", "", cleaned)
        if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
            cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"<think>", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        labeled_patterns = [
            r"(?is)(?:\*?\s*final prompt\s*:\*?)(.+)$",
            r"(?is)(?:\*?\s*revised prompt\s*:\*?)(.+)$",
            r"(?is)(?:\*?\s*final plan\s*:\*?)(.+)$",
            r"(?is)(?:\*?\s*draft\s*:\*?)(.+)$",
        ]
        for pattern in labeled_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    return candidate

        analysis_markers = [
            "analyze the image",
            "analyze the image:",
            "drafting the prompt",
            "refining the prompt",
            "final polish",
            "strict rules",
        ]
        lower_cleaned = cleaned.lower()
        if any(marker in lower_cleaned for marker in analysis_markers):
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            non_meta_lines = []
            for line in lines:
                low = line.lower()
                if re.match(r"^\d+\.\s+\*\*", line):
                    continue
                if re.match(r"^\d+\.\s", line):
                    continue
                if line.startswith("*") and line.endswith("*"):
                    continue
                if any(marker in low for marker in analysis_markers):
                    continue
                if low.startswith("must be ") or low.startswith("no ") or low.startswith("return only "):
                    continue
                non_meta_lines.append(line)
            if non_meta_lines:
                return non_meta_lines[-1]

        # If the model emits a valid prompt followed by checklist/meta blocks,
        # keep only the first paragraph-like block.
        meta_tail_markers = [
            "checking constraints",
            "output only the prompt text",
            "exactly one final prompt",
            "constraint",
            "word count check",
            "revision:",
            "draft:",
        ]
        blocks = [b.strip() for b in re.split(r"\n\s*\n", cleaned) if b.strip()]
        if len(blocks) >= 2:
            tail_text = "\n".join(blocks[1:]).lower()
            if any(m in tail_text for m in meta_tail_markers):
                def _is_instruction_like(block_text: str) -> bool:
                    low = block_text.strip().lower()
                    return (
                        low.startswith("you are ")
                        or low.startswith("user details:")
                        or low.startswith("prompt creation guidelines")
                        or low.startswith("required prompt structure")
                        or low.startswith("example")
                    )

                candidate_blocks = []
                for b in blocks:
                    low = b.lower()
                    if any(m in low for m in meta_tail_markers):
                        continue
                    if _is_instruction_like(b):
                        continue
                    candidate_blocks.append(b.strip())

                if candidate_blocks:
                    return candidate_blocks[-1]
                first = blocks[0].strip()
                if first:
                    return first

        # Remove trailing checklist lines appended after a prompt.
        lines = [ln for ln in cleaned.splitlines() if ln.strip()]
        trimmed = []
        for line in lines:
            low = line.strip().lower()
            if re.match(r"^\d+\.\s+\*\*", line.strip()):
                break
            if re.match(r"^\s*\*\s+\*\*", line):
                break
            if re.match(r"^\s*\*\s+", line):
                break
            if low.startswith("*") and "constraint" in low:
                break
            if "word count check" in low or low.startswith("revision:") or low.startswith("draft:"):
                break
            trimmed.append(line.strip())
        if trimmed:
            result = "\n".join(trimmed).strip()
            # Final guard: do not return obvious reasoning headers.
            if re.match(r"(?is)^\s*(thinking process|reasoning|analysis)\s*:?\s*$", result):
                return ""
            return result
        # Final guard on fallback path too.
        if re.match(r"(?is)^\s*(\*+\s*)?(thinking process|reasoning|analysis)\s*:?\s*(\*+\s*)?$", cleaned):
            return ""
        return cleaned

    def _enforce_preset_output(self, task_preset: str, text: str) -> str:
        out = str(text or "").strip()
        if not out:
            return out

        p = str(task_preset or "").strip().lower()

        # Hard stop at common meta-section starts if they leak through.
        stop_markers = [
            r"\n\s*\d+\.\s+\*\*",
            r"\n\s*\d+\.\s+[A-Za-z]",
            r"\n\s*\*\s+\*\*",
            r"\n\s*\*\s+",
            r"\n\s*refining for constraints\s*:",
            r"\n\s*final polish\s*:",
            r"\n\s*thinking process\s*:",
            r"\n\s*reasoning\s*:",
            r"\n\s*analysis\s*:",
        ]
        for pattern in stop_markers:
            m = re.search(pattern, out, flags=re.IGNORECASE)
            if m:
                out = out[: m.start()].strip()
                break

        # Never allow reasoning-header-only outputs through.
        header_only_patterns = [
            r"(?is)^\W*thinking\s*process\W*$",
            r"(?is)^\W*reasoning\W*$",
            r"(?is)^\W*analysis\W*$",
        ]
        for pattern in header_only_patterns:
            if re.match(pattern, out):
                return ""

        if p == "captioner_training":
            # Caption mode should be one concise line.
            first_line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
            for pattern in header_only_patterns:
                if re.match(pattern, first_line):
                    return ""
            return first_line

        # All other presets should return one paragraph only.
        paragraphs = [blk.strip() for blk in re.split(r"\n\s*\n", out) if blk.strip()]
        if paragraphs:
            first = paragraphs[0]
            for pattern in header_only_patterns:
                if re.match(pattern, first):
                    return ""
            return first
        for pattern in header_only_patterns:
            if re.match(pattern, out):
                return ""
        return out

    def _run_text_pipeline(
        self,
        pipe,
        instruction_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        messages = [{"role": "user", "content": instruction_text}]

        try:
            output = pipe(
                messages,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature),
                top_p=float(top_p),
                chat_template_kwargs={"enable_thinking": False},
            )
        except Exception:
            fallback_prompt = instruction_text
            output = pipe(
                fallback_prompt,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature),
                top_p=float(top_p),
            )
        return self._strip_thinking_text(self._extract_generated_text(output))

    def _run_vision_pipeline(
        self,
        pipe,
        pil_images: list[Image.Image],
        instruction_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        user_content = [{"type": "image", "image": img} for img in pil_images]
        user_content.append({"type": "text", "text": instruction_text})

        messages = [{"role": "user", "content": user_content}]

        try:
            output = pipe(
                text=messages,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature),
                top_p=float(top_p),
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            output = pipe(
                messages,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature),
                top_p=float(top_p),
                chat_template_kwargs={"enable_thinking": False},
            )
        return self._strip_thinking_text(self._extract_generated_text(output))

    def generate_prompt(
        self,
        model_preset: str,
        custom_model_id: str,
        task_preset: str,
        user_input: str,
        custom_instructions: str,
        trigger_word: str,
        image_count: int,
        download_if_missing: bool,
        device: str,
        dtype: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        trust_remote_code: bool = False,
        hf_token: str = "",
        **kwargs,
    ) -> Tuple[str, str, str]:
        model_id = self._resolve_model_id(model_preset, custom_model_id)
        instruction_text = self._build_instruction_text(
            task_preset, user_input, trigger_word, custom_instructions
        )
        if not instruction_text:
            return ("", model_id, "error: user_input/custom_instructions is empty")
        pil_images = self._collect_pil_images(image_count, kwargs)
        # Safeguard: if image_count > 0 but no image is connected, fall back to text-generation.
        task = "image-text-to-text" if pil_images else "text-generation"

        try:
            pipe = self._load_pipeline(
                task=task,
                model_id=model_id,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                download_if_missing=download_if_missing,
                hf_token=hf_token,
            )
            if pil_images:
                text = self._run_vision_pipeline(
                    pipe,
                    pil_images,
                    instruction_text,
                    temperature,
                    top_p,
                    max_new_tokens,
                )
            else:
                text = self._run_text_pipeline(
                    pipe,
                    instruction_text,
                    temperature,
                    top_p,
                    max_new_tokens,
                )
            text = str(text or "").strip()
            text = self._enforce_preset_output(task_preset, text)
            if not text:
                raise Exception("Empty model response.")
            return (text, model_id, "ok")
        except Exception as e:
            return ("", model_id, f"error: {e}")


class VRGDG_Qwen25(VRGDG_Qwen35):
    MODEL_PRESETS = [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "custom",
    ]

    def _build_instruction_text(
        self,
        task_preset: str,
        user_input: str,
        trigger_word: str,
        custom_instructions: str,
    ) -> str:
        # Qwen2.5 variant: no appended anti-reasoning guard text.
        base = self._preset_instruction(task_preset)
        user_text = str(user_input or "").strip()
        custom_text = str(custom_instructions or "").strip()
        if str(task_preset or "").strip().lower() == "custom":
            if custom_text and user_text:
                return f"{custom_text}\n\nUser details:\n{user_text}"
            return custom_text or user_text
        if str(task_preset or "").strip().lower() == "captioner_training":
            t = str(trigger_word or "").strip()
            if t:
                user_text = f"Trigger word: {t}\n{user_text}"
        if not base:
            return user_text
        if not user_text:
            return base
        return f"{base}\n\nUser details:\n{user_text}"

    def _run_text_pipeline(
        self,
        pipe,
        instruction_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        messages = [{"role": "user", "content": instruction_text}]
        try:
            output = pipe(
                messages,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature),
                top_p=float(top_p),
            )
        except Exception:
            output = pipe(
                instruction_text,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature),
                top_p=float(top_p),
            )
        return str(self._extract_generated_text(output) or "").strip()

    def _run_vision_pipeline(
        self,
        pipe,
        pil_images: list[Image.Image],
        instruction_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        user_content = [{"type": "image", "image": img} for img in pil_images]
        user_content.append({"type": "text", "text": instruction_text})
        messages = [{"role": "user", "content": user_content}]
        try:
            output = pipe(
                text=messages,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature),
                top_p=float(top_p),
            )
        except TypeError:
            output = pipe(
                messages,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature),
                top_p=float(top_p),
            )
        return str(self._extract_generated_text(output) or "").strip()

    def _enforce_preset_output(self, task_preset: str, text: str) -> str:
        # Qwen2.5 variant: keep output mostly raw, but remove obvious instruction echo blocks.
        out = str(text or "").strip()
        if not out:
            return out

        # Common echo boundary in this node prompt template.
        if "User details:" in out:
            out = out.split("User details:", 1)[-1].strip()

        if "To be completed Prompt:" in out:
            out = out.split("To be completed Prompt:", 1)[-1].strip()

        bad_prefixes = (
            "you are ",
            "after the user provides details",
            "your output must be only",
            "prompt creation guidelines",
            "color style:",
            "mood:",
            "subject description:",
            "environment / setting:",
            "camera angle / perspective:",
            "weather and time of day:",
            "additional cinematic details:",
            "required prompt structure:",
            "do not show or display any reasoning",
            "example",
        )

        # Prefer the last paragraph that looks like actual generated content.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", out) if p.strip()]
        candidates = []
        for p in paragraphs:
            low = p.lower()
            if any(low.startswith(prefix) for prefix in bad_prefixes):
                continue
            if len(p.split()) < 8:
                continue
            candidates.append(p)

        if candidates:
            return candidates[-1]

        # Fallback to last non-empty line that doesn't look like prompt template text.
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        for ln in reversed(lines):
            low = ln.lower()
            if any(low.startswith(prefix) for prefix in bad_prefixes):
                continue
            if len(ln.split()) >= 6:
                return ln
        return out


class VRGDG_GeneralVLM(VRGDG_Qwen25):
    MODEL_PRESETS = [
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "custom",
    ]

    def _resolve_model_id(self, model_preset: str, custom_model_id: str) -> str:
        model_id = super()._resolve_model_id(model_preset, custom_model_id)
        if str(model_id).strip() == "CohereForAI/aya-vision-8b":
            return "CohereLabs/aya-vision-8b"
        return model_id

    def _extra_model_kwargs(self, model_id: str, task: str) -> dict:
        mid = str(model_id or "").lower()
        # Phi-3.5 Vision can attempt FlashAttention2 by default; force a safe fallback.
        if "phi-3.5-vision" in mid:
            return {"attn_implementation": "sdpa"}
        return {}

    @classmethod
    def INPUT_TYPES(cls):
        data = super().INPUT_TYPES()
        required = dict(data.get("required", {}))
        required["hf_token"] = (
            "STRING",
            {
                "default": "",
                "tooltip": "Optional Hugging Face access token for gated/private repos.",
            },
        )
        required["allow_custom_model_code"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Enable only for trusted model repos that require custom code (for example Phi-3.5 Vision).",
            },
        )
        return {
            "required": required,
            "optional": data.get("optional", {}),
        }

    def generate_prompt(
        self,
        allow_custom_model_code: bool = False,
        hf_token: str = "",
        **kwargs,
    ) -> Tuple[str, str, str]:
        return super().generate_prompt(
            trust_remote_code=bool(allow_custom_model_code),
            hf_token=hf_token,
            **kwargs,
        )


NODE_CLASS_MAPPINGS = {
    "VRGDG_NanoBananaPro": VRGDG_NanoBananaPro,
    "VRGDG_LLM_Multi": VRGDG_LLM_Multi,
    "VRGDG_LocalLLM": VRGDG_LocalLLM,
    "VRGDG_Qwen3.5": VRGDG_Qwen35,
    "VRGDG_Qwen2.5": VRGDG_Qwen25,
    "VRGDG_GeneralVLM": VRGDG_GeneralVLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_NanoBananaPro": "🚀 VRGDG NanoBanana Pro 🚀",
    "VRGDG_LLM_Multi": "🤖 VRGDG LLM Multi 🤖",
    "VRGDG_LocalLLM": "💻 VRGDG Local LLM 💻",
    "VRGDG_Qwen3.5": "🧠 VRGDG Qwen 3.5 🧠",
    "VRGDG_Qwen2.5": "🧠 VRGDG Qwen 2.5 🧠",
    "VRGDG_GeneralVLM": "🧠 VRGDG General VLM 🧠",
}
