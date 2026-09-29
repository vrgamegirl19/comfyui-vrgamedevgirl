"""The LLM Multi node: hosted and local chat APIs (OpenAI, Anthropic, Gemini, Grok, LM Studio, Ollama and similar)."""

import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple
import json
import base64
import re
import uuid
import urllib.request
import urllib.error
import urllib.parse

from .google import _extract_google_inline_image, _google_generate_content


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
            "gpt-image-2",
            "gpt-image-1",
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-5.6",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o4-mini",
            "gpt-4.1",
            "gpt-4o",
        ],
        "anthropic": [
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
        ],
        "google": [
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash",
            "gemini-3.1-flash-lite",
            "gemini-3-pro-preview",
            "gemini-3-pro-image-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
        "grok": [
            "grok-4.6",
            "grok-4.5",
            "grok-4.3",
        ],
        "deepseek": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "openrouter": [
            "openai/gpt-6-astra",
            "openai/gpt-6-sol",
            "openai/gpt-6-luna",
            "openai/gpt-5.6",
            "openai/gpt-5.6-sol",
            "openai/gpt-5.6-terra",
            "openai/gpt-5.6-luna",
            "openai/gpt-5.5",
            "openai/gpt-5.4",
            "openai/gpt-5.4-mini",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-opus-4.8",
            "google/gemini-3.5-flash",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.1-70b-instruct",
        ],
        "apifreellm": [
            "apifreellm",
        ],
    }
    DEFAULT_MODEL = {
        "openai": "gpt-6-luna",
        "anthropic": "claude-sonnet-4-6",
        "google": "gemini-3.5-flash",
        "grok": "grok-4.6",
        "deepseek": "deepseek-chat",
        "openrouter": "openai/gpt-6-luna",
        "apifreellm": "apifreellm",
    }
    ALL_MODELS = [m for models in PROVIDER_MODELS.values() for m in models]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "provider": (list(cls.PROVIDER_MODELS.keys()), {"default": "openai"}),
                "model": (cls.ALL_MODELS, {"default": cls.DEFAULT_MODEL["openai"]}),
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
        if provider in ("xai", "grok") and "http 403" in low and "1010" in low:
            return (
                "error: xAI request blocked (HTTP 403 code 1010). "
                "Usually account/region/firewall edge blocking. Verify xAI API access, "
                "try a different network/IP, or use OpenRouter with a Grok model."
            )
        return f"error: {msg}"

    @classmethod
    def _canonical_provider(cls, provider: str) -> str:
        name = str(provider or "").strip().lower()
        if name == "xai":
            return "grok"
        return name

    def _validate_provider_model(self, provider: str, model: str):
        provider = self._canonical_provider(provider)
        allowed = self.PROVIDER_MODELS.get(provider, [])
        if model not in allowed:
            raise Exception(
                f"Model '{model}' is not valid for provider '{provider}'. "
                f"Choose one of: {', '.join(allowed)}"
            )

    def _resolve_model(self, provider: str, selected_model: str, custom_model: str) -> str:
        provider = self._canonical_provider(provider)
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

        provider = self._canonical_provider(provider)
        if provider not in self.PROVIDER_MODELS:
            raise Exception(f"Unsupported provider: {provider}")
        chosen_model = self._resolve_model(provider, model, custom_model)
        pil_images = self._collect_pil_images([image1, image2, image3, image4])

        try:
            if provider == "openai":
                if chosen_model.startswith("gpt-image-"):
                    text, image_out = self._call_openai_image_generation(api_key, chosen_model, prompt, pil_images)
                else:
                    text, image_out = self._call_openai_compatible(
                        "https://api.openai.com/v1",
                        api_key,
                        chosen_model,
                        prompt,
                        pil_images,
                    )
            elif provider == "grok":
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


NODE_CLASS_MAPPINGS = {
    "VRGDG_LLM_Multi": VRGDG_LLM_Multi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LLM_Multi": "🤖 VRGDG LLM Multi 🤖",
}
