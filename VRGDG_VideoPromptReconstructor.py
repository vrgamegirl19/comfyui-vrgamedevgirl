"""Reconstruct a concise, visually grounded prompt from a video frame batch."""

import base64
import hashlib
import io
import json
import math
import os
import re
import urllib.error
import urllib.request

import torch
from PIL import Image


VISION_INSTRUCTION = """Analyze these video frames and create a concise visual prompt for recreating
the same video content.

Describe only details clearly visible in the frames:

1. Subject identity, appearance, pose, and visible actions
2. Location and environment
3. Lighting, atmosphere, and color palette
4. Clothing, accessories, props, and important objects

Do not describe camera movement, lens choice, editing, story, emotion, sound,
dialogue, or anything that cannot be confirmed visually.
Do not invent or add people, objects, locations, or actions.
Keep the subject's appearance and clothing consistent.
Return only the final video prompt as plain text."""

_RUNNER_NAMES = ("LM Studio", "LLM API", "Custom Server")
_API_PROVIDER_NAMES = ("openai", "anthropic", "google", "openrouter", "grok")
_MODEL_CHOICES = (
    "gpt-5.6-luna", "gpt-5.6", "gpt-5.5", "gpt-4o", "gpt-4.1", "claude-sonnet-4-6",
    "gemini-2.5-flash", "openai/gpt-4o", "grok-2-vision-1212",
    "qwen2.5-vl-7b-instruct",
)


def _frames_to_pil(batch, label="video"):
    if batch is None:
        raise ValueError(f"No {label} frames were provided.")
    if not isinstance(batch, torch.Tensor) or batch.ndim != 4 or batch.shape[0] < 1:
        raise ValueError(f"{label} must be a non-empty IMAGE batch with shape [frames, height, width, channels].")
    if batch.shape[-1] < 3 or batch.shape[1] < 1 or batch.shape[2] < 1:
        raise ValueError(f"{label} has an unsupported frame shape.")

    images = []
    for frame in batch:
        pixels = frame[..., :3].detach().cpu().float().clamp(0, 1)
        array = (pixels.mul(255).round().byte().numpy())
        images.append(Image.fromarray(array, mode="RGB"))
    return images


def _select_frame_indices(count, requested):
    count = int(count)
    requested = max(1, min(int(requested), count))
    if requested == 1:
        return [0]
    if requested == 2:
        return [0, count - 1]
    return sorted({int(round(index * (count - 1) / (requested - 1))) for index in range(requested)})


def _image_data_url(image, max_side=768, quality=85):
    image = image.convert("RGB")
    longest = max(image.width, image.height)
    if longest > max_side:
        scale = float(max_side) / longest
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), resample)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _endpoint_url(api_url):
    url = str(api_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("API URL is required. Use a local or external vision chat-completions endpoint.")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _extract_response_text(payload):
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Vision LLM response did not contain choices[0].message.content.") from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content or "").strip()
    if not text:
        raise RuntimeError("Vision LLM returned an empty prompt.")
    return text


def _clean_prompt(text, maximum):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:text|plaintext)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(?:final prompt|video prompt|prompt)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise RuntimeError("Vision LLM returned no usable prompt text.")
    return text[: int(maximum)].rstrip()


class VRGDG_VideoPromptReconstructor:
    """Create a batch-specific visual prompt using a configurable vision API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Current video batch as an IMAGE tensor."}),
                "batch_index": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "Current batch number: 0 is the first batch, 1 is the second, and so on. Used to keep cached batch prompts separate."}),
                "frames_to_extract": ("INT", {"default": 3, "min": 1, "max": 12, "step": 1, "tooltip": "Number of representative frames sent to the vision model. Three means first, middle, and final frame. More frames provide more coverage but use more API tokens."}),
                "llm_runner": (list(_RUNNER_NAMES), {"tooltip": "Choose the same kind of LLM runner used by the Video Builder: LM Studio for a local server, LLM API for OpenAI/Anthropic/Google, or Custom Server for another OpenAI-compatible server."}),
                "llm_provider": (list(_API_PROVIDER_NAMES), {"tooltip": "Provider used when llm_runner is LLM API. Choose the service that issued your API key, such as openai for an OpenAI key."}),
                "model_name": (list(_MODEL_CHOICES), {"default": "gpt-4o", "tooltip": "Choose the vision model. The list changes with the selected API provider; LM Studio and Custom Server models are loaded from their server when available."}),
                "api_url": ("STRING", {"default": "http://127.0.0.1:1234/v1/chat/completions", "tooltip": "Vision chat-completions endpoint. The default is for LM Studio. Other servers usually use an OpenAI-compatible /v1 endpoint."}),
                "api_key": ("STRING", {"default": "", "password": True, "tooltip": "External API key. Usually leave blank for local LM Studio or Ollama unless authentication is enabled."}),
                "max_prompt_length": ("INT", {"default": 1200, "min": 1, "max": 10000, "step": 1, "tooltip": "Maximum characters in the generated prompt. Larger values allow more detail; 1200 is a concise default."}),
            },
            "optional": {
                "overlap_frames": ("IMAGE", {"tooltip": "Optional frames from the end of the previous batch. They help maintain the same subject, clothing, props, and environment across batch boundaries."}),
                "overlap_frame_count": ("INT", {"default": 0, "min": 0, "max": 6, "step": 1, "tooltip": "Number of frames to use from overlap_frames. Leave at 0 when none are connected; 2 or 3 is usually enough."}),
                "video_filename": ("STRING", {"default": "", "tooltip": "Optional original filename used for stable caching. Reusing the same filename, batch, model, and frame settings avoids another API call."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Model choices are populated dynamically by the frontend/provider.
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "reconstruct_prompt"
    CATEGORY = "VRGameDevGirl/Video/Prompting"
    DESCRIPTION = "Generate a concise, visually grounded prompt from representative frames of a video batch."

    def _cache_path(self, video_filename, batch_index, model_name, frames_to_extract, overlap_count, frame_bytes):
        identity = str(video_filename or "").strip() or hashlib.sha256(frame_bytes).hexdigest()
        key = "|".join((identity, str(batch_index), str(model_name).strip(), str(frames_to_extract), str(overlap_count)))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vrgdg_prompt_cache")
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"{digest}.json")

    def reconstruct_prompt(self, video, batch_index, frames_to_extract, llm_runner, llm_provider, model_name,
                           api_url, api_key, max_prompt_length, overlap_frames=None,
                           overlap_frame_count=0, video_filename=""):
        images = _frames_to_pil(video, "video")
        requested = max(1, int(frames_to_extract))
        selected = [images[index] for index in _select_frame_indices(len(images), requested)]
        overlap = _frames_to_pil(overlap_frames, "overlap") if overlap_frames is not None else []
        if overlap and int(overlap_frame_count) > 0:
            selected = overlap[-int(overlap_frame_count):] + selected

        frame_bytes = b"".join(_image_data_url(image).encode("ascii") for image in selected)
        cache_path = self._cache_path(video_filename, batch_index, model_name, requested, int(overlap_frame_count), frame_bytes)
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    return (str(json.load(handle)["prompt"]),)
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass

        model = str(model_name or "").strip()
        if not model:
            raise ValueError("Vision model name is required.")
        runner = str(llm_runner or "").strip()
        provider = str(llm_provider or "").strip().lower()
        key = str(api_key or "").strip()
        if runner not in _RUNNER_NAMES:
            raise ValueError("Unsupported LLM runner. Choose LM Studio, LLM API, or Custom Server.")
        try:
            from .VRGDG_MusicVideoBuilderNodes import (
                _run_llm_api_vision,
                _run_lm_studio_vision,
                _run_own_server_vision,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load the Video Builder vision runners: {exc}") from exc

        if runner == "LLM API":
            if provider not in _API_PROVIDER_NAMES:
                raise ValueError("Unsupported LLM API provider. Choose openai, anthropic, google, openrouter, or grok.")
            if not key:
                raise ValueError("LLM API requires an API key. Enter the key issued by the selected provider.")
            text, _info = _run_llm_api_vision({
                "llm_api_provider": provider,
                "llm_api_model": model,
                "llm_api_key": key,
            }, VISION_INSTRUCTION, selected)
        elif runner == "LM Studio":
            base_url = str(api_url or "").strip().rstrip("/")
            if base_url.endswith("/chat/completions"):
                base_url = base_url[:-len("/chat/completions")]
            if not base_url:
                raise ValueError("LM Studio base URL is required.")
            text = _run_lm_studio_vision({
                "lmstudio_base_url": base_url,
                "lmstudio_model": model,
                "lmstudio_api_key": key,
                "lmstudio_timeout": 300,
            }, VISION_INSTRUCTION, selected, max_new_tokens=max(64, min(int(max_prompt_length) + 200, 12000)))
        else:
            base_url = str(api_url or "").strip().rstrip("/")
            if base_url.endswith("/chat/completions"):
                base_url = base_url[:-len("/chat/completions")]
            if not base_url:
                raise ValueError("Custom Server URL is required.")
            text, _info = _run_own_server_vision({
                "own_server_url": base_url,
                "own_server_model": model,
                "own_server_api_key": key,
                "own_server_timeout": 300,
            }, VISION_INSTRUCTION, selected, max_new_tokens=max(64, min(int(max_prompt_length) + 200, 12000)))
        prompt = _clean_prompt(text, max_prompt_length)
        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump({"prompt": prompt}, handle, ensure_ascii=False, indent=2)
        except OSError:
            pass
        return (prompt,)


NODE_CLASS_MAPPINGS = {"VRGDG_VideoPromptReconstructor": VRGDG_VideoPromptReconstructor}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_VideoPromptReconstructor": "Video Prompt Reconstructor — Vision LLM"
}
