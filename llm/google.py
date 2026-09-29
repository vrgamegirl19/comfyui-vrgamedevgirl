"""Google Gemini helpers and the NanoBanana Pro node."""

import torch
import numpy as np
from PIL import Image
from io import BytesIO
from functools import lru_cache
from typing import Optional, Tuple
import json
import base64
import urllib.request
import urllib.error
import urllib.parse


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
    safe_model = urllib.parse.quote(str(model or "").strip(), safe="-_.~")
    safe_key = urllib.parse.quote(str(api_key or "").strip(), safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{safe_model}:generateContent?key={safe_key}"
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


NODE_CLASS_MAPPINGS = {
    "VRGDG_NanoBananaPro": VRGDG_NanoBananaPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_NanoBananaPro": "🚀 VRGDG NanoBanana Pro 🚀",
}
