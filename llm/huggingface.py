"""Hugging Face transformers runners (Qwen 3.5 and 2.5). Base classes for the GGUF runners; not registered as nodes."""

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import re
import folder_paths

from .cache import _HF_PIPELINE_CACHE


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
