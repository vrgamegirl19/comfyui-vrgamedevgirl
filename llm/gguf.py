"""llama.cpp GGUF runners: the general GGUF base, Qwen GGUF and SuperGemma chat nodes, and the unload node."""

import os
import torch
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple
import base64
import re
import folder_paths
import gc
import time

from .cache import _GGUF_CHAT_HANDLER_CACHE, _GGUF_LOCK, _GGUF_MODEL_CACHE, _clear_vrgdg_llm_caches, _close_gguf_cached_resources
from .huggingface import VRGDG_Qwen25


class VRGDG_GeneralGGUF(VRGDG_Qwen25):
    _GEMMA_STOP_SEQUENCES = (
        "<end_of_turn>",
        "<start_of_turn>",
        "<|end_of_turn|>",
        "<|start_of_turn|>",
        "|end_of_turn|",
        "|start_of_turn|",
        "_end_turn",
        "_start_turn",
        "_end_of_turn",
        "_start_of_turn",
    )

    MODEL_PRESETS = [
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "Jiunsong/supergemma4-26b-uncensored-gguf-v2",
        "custom",
    ]
    MM_PROJ_PRESETS = {
        "unsloth/gemma-4-26B-A4B-it-GGUF": "mmproj-F16.gguf",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_preset": (
                    cls.MODEL_PRESETS,
                    {
                        "default": "unsloth/gemma-4-26B-A4B-it-GGUF",
                        "tooltip": "Choose a GGUF repo preset. Use custom_model_id for a local .gguf file, folder, or another Hugging Face repo id.",
                    },
                ),
                "custom_model_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional override. Can be a local .gguf file path, a local folder containing GGUF files, or a Hugging Face repo id.",
                    },
                ),
                "gguf_filename": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional GGUF filename to use inside the repo/folder. Strongly recommended when a repo has multiple quant files.",
                    },
                ),
                "mmproj_filename": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional multimodal projector filename (.gguf). Required for image-aware GGUF models like Gemma 4 vision GGUFs.",
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
                        "tooltip": "If enabled, a missing GGUF file can be downloaded to ComfyUI/models/LLM/GGUF.",
                    },
                ),
                "advanced": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Show advanced GGUF runtime controls such as context, GPU layers, threads, sampler, and token limits.",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, unload the GGUF model from cache after this run to free RAM/VRAM.",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional Hugging Face access token for private or gated repos.",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 512,
                        "max": 131072,
                        "step": 256,
                        "tooltip": "GGUF context window.",
                    },
                ),
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": 99,
                        "min": -1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "How many layers to offload to GPU. Use -1 to offload all supported layers.",
                    },
                ),
                "n_threads": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "CPU threads used by llama.cpp.",
                    },
                ),
                "chat_format": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional llama.cpp chat format override. Leave blank to use the model default.",
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
            "optional": {
                f"image{i}": ("IMAGE", {"tooltip": "Optional reference image input."})
                for i in range(1, cls.MAX_IMAGES + 1)
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "used_model", "status")
    FUNCTION = "generate_prompt"
    CATEGORY = "VRGDG/LLM"

    def _comfy_gguf_cache_dir(self) -> Optional[str]:
        base = getattr(folder_paths, "models_dir", None)
        if not base:
            return None
        return os.path.join(base, "LLM", "GGUF")

    def _gguf_repo_local_dir(self, model_id: str) -> Optional[str]:
        if not isinstance(model_id, str):
            return None
        raw = model_id.strip()
        if not raw:
            return None
        if os.path.isfile(raw):
            return os.path.dirname(raw)
        if os.path.isdir(raw):
            return raw
        if "/" not in raw:
            return None
        cache_root = self._comfy_gguf_cache_dir()
        if not cache_root:
            return None
        safe_name = raw.replace("/", "--").replace("\\", "--").replace(":", "_")
        return os.path.join(cache_root, safe_name)

    def _list_local_gguf_files(self, local_dir: str) -> list[str]:
        if not local_dir or not os.path.isdir(local_dir):
            return []
        found = []
        try:
            for entry in os.scandir(local_dir):
                if not entry.is_file():
                    continue
                name = str(entry.name or "")
                low = name.lower()
                if not low.endswith(".gguf"):
                    continue
                if "mmproj" in low:
                    continue
                found.append(name)
        except Exception:
            return []
        return sorted(found)

    def _list_local_mmproj_files(self, local_dir: str) -> list[str]:
        if not local_dir or not os.path.isdir(local_dir):
            return []
        found = []
        try:
            for entry in os.scandir(local_dir):
                if not entry.is_file():
                    continue
                name = str(entry.name or "")
                low = name.lower()
                if not low.endswith(".gguf"):
                    continue
                if "mmproj" not in low:
                    continue
                found.append(name)
        except Exception:
            return []
        return sorted(found)

    def _list_repo_gguf_files(self, model_id: str, hf_token: str = "") -> list[str]:
        try:
            from huggingface_hub import list_repo_files
        except Exception as e:
            raise Exception(
                "Missing dependency: install huggingface_hub to inspect GGUF repos."
            ) from e

        token = str(hf_token or "").strip() or None
        files = list_repo_files(repo_id=model_id, token=token)
        candidates = []
        for path in files:
            name = os.path.basename(str(path or ""))
            low = name.lower()
            if low.endswith(".gguf") and "mmproj" not in low:
                candidates.append(name)
        return sorted(set(candidates))

    def _list_repo_mmproj_files(self, model_id: str, hf_token: str = "") -> list[str]:
        try:
            from huggingface_hub import list_repo_files
        except Exception as e:
            raise Exception(
                "Missing dependency: install huggingface_hub to inspect GGUF repos."
            ) from e

        token = str(hf_token or "").strip() or None
        files = list_repo_files(repo_id=model_id, token=token)
        candidates = []
        for path in files:
            name = os.path.basename(str(path or ""))
            low = name.lower()
            if low.endswith(".gguf") and "mmproj" in low:
                candidates.append(name)
        return sorted(set(candidates))

    def _select_gguf_filename(
        self,
        model_id: str,
        gguf_filename: str,
        local_dir: Optional[str],
        download_if_missing: bool,
        hf_token: str = "",
    ) -> str:
        requested = os.path.basename(str(gguf_filename or "").strip())
        if requested:
            return requested

        local_candidates = self._list_local_gguf_files(local_dir or "")
        if len(local_candidates) == 1:
            return local_candidates[0]
        if len(local_candidates) > 1:
            raise Exception(
                "Multiple GGUF files were found locally. Set gguf_filename to choose one.\n"
                f"Found: {', '.join(local_candidates[:12])}"
            )

        if not download_if_missing or "/" not in str(model_id or ""):
            raise Exception(
                "No GGUF filename could be resolved. Set gguf_filename or point custom_model_id to a local .gguf file."
            )

        repo_candidates = self._list_repo_gguf_files(model_id, hf_token=hf_token)
        if len(repo_candidates) == 1:
            return repo_candidates[0]

        raise Exception(
            "This GGUF repo contains multiple quant files. Set gguf_filename to the one you want.\n"
            f"Repo files: {', '.join(repo_candidates[:12])}"
        )

    def _default_mmproj_filename_for_model(self, model_id: str) -> str:
        return str(self.MM_PROJ_PRESETS.get(str(model_id or "").strip(), "")).strip()

    def _select_mmproj_filename(
        self,
        model_id: str,
        mmproj_filename: str,
        local_dir: Optional[str],
        download_if_missing: bool,
        hf_token: str = "",
    ) -> str:
        requested = os.path.basename(str(mmproj_filename or "").strip())
        if requested:
            return requested

        local_candidates = self._list_local_mmproj_files(local_dir or "")
        if len(local_candidates) == 1:
            return local_candidates[0]
        if len(local_candidates) > 1:
            preset_default = self._default_mmproj_filename_for_model(model_id)
            if preset_default and preset_default in local_candidates:
                return preset_default
            raise Exception(
                "Multiple mmproj GGUF files were found locally. Set mmproj_filename to choose one.\n"
                f"Found: {', '.join(local_candidates[:12])}"
            )

        preset_default = self._default_mmproj_filename_for_model(model_id)
        if preset_default:
            return preset_default

        if not download_if_missing or "/" not in str(model_id or ""):
            raise Exception(
                "No mmproj filename could be resolved. Set mmproj_filename or point custom_model_id to a folder containing exactly one mmproj GGUF."
            )

        repo_candidates = self._list_repo_mmproj_files(model_id, hf_token=hf_token)
        if len(repo_candidates) == 1:
            return repo_candidates[0]
        if preset_default and preset_default in repo_candidates:
            return preset_default

        raise Exception(
            "This multimodal GGUF repo contains multiple mmproj files. Set mmproj_filename to the one you want.\n"
            f"Repo files: {', '.join(repo_candidates[:12])}"
        )

    def _resolve_gguf_model_path(
        self,
        model_id: str,
        gguf_filename: str,
        download_if_missing: bool,
        hf_token: str = "",
    ) -> str:
        raw = str(model_id or "").strip()
        if not raw:
            raise Exception("No GGUF model was selected.")

        if os.path.isfile(raw):
            if not raw.lower().endswith(".gguf"):
                raise Exception(f"Expected a .gguf file path, got: {raw}")
            return raw

        if os.path.isdir(raw):
            chosen = self._select_gguf_filename(
                model_id=raw,
                gguf_filename=gguf_filename,
                local_dir=raw,
                download_if_missing=False,
                hf_token=hf_token,
            )
            return os.path.join(raw, chosen)

        local_dir = self._gguf_repo_local_dir(raw)
        chosen = self._select_gguf_filename(
            model_id=raw,
            gguf_filename=gguf_filename,
            local_dir=local_dir,
            download_if_missing=download_if_missing,
            hf_token=hf_token,
        )

        if local_dir:
            local_path = os.path.join(local_dir, chosen)
            if os.path.isfile(local_path):
                return local_path

        if not download_if_missing:
            raise Exception(
                f"GGUF model '{chosen}' for '{raw}' was not found locally.\n"
                "Enable download_if_missing or provide a local .gguf file path."
            )

        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise Exception(
                "Missing dependency: install huggingface_hub to download GGUF models."
            ) from e

        if not local_dir:
            cache_dir = self._comfy_gguf_cache_dir()
            if not cache_dir:
                raise Exception("Could not resolve a GGUF cache directory.")
            local_dir = cache_dir
        os.makedirs(local_dir, exist_ok=True)

        token = str(hf_token or "").strip() or None
        try:
            downloaded = hf_hub_download(
                repo_id=raw,
                filename=chosen,
                local_dir=local_dir,
                token=token,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if "403" in low or "gated repo" in low or "not in the authorized list" in low:
                if token is None:
                    raise Exception(
                        f"Hugging Face access denied for '{raw}'. Set hf_token and make sure your account has accepted model access."
                    ) from e
                raise Exception(
                    f"Hugging Face access denied for '{raw}' even with token. Verify the token account has model access."
                ) from e
            if "401" in low or "unauthorized" in low or "invalid token" in low:
                raise Exception(
                    f"Authentication failed for '{raw}'. Check hf_token and ensure it has read access."
                ) from e
            raise Exception(
                f"Could not download GGUF file '{chosen}' from '{raw}'.\n{msg}"
            ) from e

        return downloaded

    def _resolve_mmproj_path(
        self,
        model_id: str,
        mmproj_filename: str,
        download_if_missing: bool,
        hf_token: str = "",
    ) -> str:
        raw = str(model_id or "").strip()
        mmproj_raw = str(mmproj_filename or "").strip()
        if not raw:
            raise Exception("No GGUF model was selected.")
        if mmproj_raw and os.path.isfile(mmproj_raw):
            if not mmproj_raw.lower().endswith(".gguf"):
                raise Exception(f"Expected a .gguf mmproj file path, got: {mmproj_raw}")
            return mmproj_raw

        if os.path.isfile(raw):
            local_dir = os.path.dirname(raw)
            chosen = self._select_mmproj_filename(
                model_id=raw,
                mmproj_filename=mmproj_filename,
                local_dir=local_dir,
                download_if_missing=False,
                hf_token=hf_token,
            )
            local_path = os.path.join(local_dir, chosen)
            if os.path.isfile(local_path):
                return local_path
            raise Exception(
                f"mmproj file '{chosen}' was not found next to the GGUF file.\n"
                "Set mmproj_filename or place the mmproj GGUF in the same folder."
            )

        if os.path.isdir(raw):
            chosen = self._select_mmproj_filename(
                model_id=raw,
                mmproj_filename=mmproj_filename,
                local_dir=raw,
                download_if_missing=False,
                hf_token=hf_token,
            )
            return os.path.join(raw, chosen)

        local_dir = self._gguf_repo_local_dir(raw)
        chosen = self._select_mmproj_filename(
            model_id=raw,
            mmproj_filename=mmproj_filename,
            local_dir=local_dir,
            download_if_missing=download_if_missing,
            hf_token=hf_token,
        )

        if local_dir:
            local_path = os.path.join(local_dir, chosen)
            if os.path.isfile(local_path):
                return local_path

        if not download_if_missing:
            raise Exception(
                f"mmproj file '{chosen}' for '{raw}' was not found locally.\n"
                "Enable download_if_missing or provide a local mmproj GGUF path via mmproj_filename in the same folder."
            )

        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise Exception(
                "Missing dependency: install huggingface_hub to download GGUF models."
            ) from e

        if not local_dir:
            cache_dir = self._comfy_gguf_cache_dir()
            if not cache_dir:
                raise Exception("Could not resolve a GGUF cache directory.")
            local_dir = cache_dir
        os.makedirs(local_dir, exist_ok=True)

        token = str(hf_token or "").strip() or None
        try:
            downloaded = hf_hub_download(
                repo_id=raw,
                filename=chosen,
                local_dir=local_dir,
                token=token,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise Exception(
                f"Could not download mmproj file '{chosen}' from '{raw}'.\n{e}"
            ) from e

        return downloaded

    def _gguf_model_key(
        self,
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        chat_format: str,
        mmproj_path: str = "",
    ) -> tuple:
        return (
            os.path.normpath(str(model_path or "")),
            int(n_ctx),
            int(n_gpu_layers),
            int(n_threads),
            str(chat_format or "").strip(),
            os.path.normpath(str(mmproj_path or "")),
        )

    def _load_gguf_model(
        self,
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        chat_format: str,
        mmproj_path: str = "",
    ):
        key = self._gguf_model_key(model_path, n_ctx, n_gpu_layers, n_threads, chat_format, mmproj_path)
        cached = _GGUF_MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        normalized_model_path = os.path.normpath(str(model_path or ""))
        if not os.path.isfile(normalized_model_path):
            raise Exception(f"GGUF model file was not found: {normalized_model_path}")

        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except Exception as e:
            raise Exception(
                "Missing dependency: install llama-cpp-python to use the GGUF node."
            ) from e

        kwargs = {
            "model_path": model_path,
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "n_threads": int(n_threads),
            "verbose": False,
        }
        chat_format_value = str(chat_format or "").strip()
        if chat_format_value:
            kwargs["chat_format"] = chat_format_value
        mmproj_value = str(mmproj_path or "").strip()
        chat_handler = None
        if mmproj_value:
            class Gemma4MTMDChatHandler(Llava15ChatHandler):
                DEFAULT_SYSTEM_MESSAGE = None
                CHAT_FORMAT = (
                    "{% for message in messages %}"
                    "{% if message.role == 'user' %}"
                    "<start_of_turn>user\n"
                    "{% if message.content is string %}"
                    "{{ message.content }}"
                    "{% endif %}"
                    "{% if message.content is iterable %}"
                    "{% for content in message.content %}"
                    "{% if content.type == 'image_url' and content.image_url is string %}"
                    "{{ content.image_url }}"
                    "{% endif %}"
                    "{% if content.type == 'image_url' and content.image_url is mapping %}"
                    "{{ content.image_url.url }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% for content in message.content %}"
                    "{% if content.type == 'text' %}"
                    "{{ content.text }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% endif %}"
                    "<end_of_turn>\n"
                    "{% endif %}"
                    "{% if message.role == 'assistant' and message.content is not none %}"
                    "<start_of_turn>model\n{{ message.content }}<end_of_turn>\n"
                    "{% endif %}"
                    "{% endfor %}"
                    "<start_of_turn>model\n"
                )

            chat_handler = Gemma4MTMDChatHandler(
                clip_model_path=mmproj_value,
                verbose=False,
            )
            kwargs["chat_handler"] = chat_handler

        def load_once():
            return Llama(**kwargs)

        try:
            model = load_once()
        except Exception as first_error:
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
            time.sleep(0.25)
            try:
                model = load_once()
            except Exception as second_error:
                size_gb = 0.0
                try:
                    size_gb = os.path.getsize(normalized_model_path) / (1024 ** 3)
                except Exception:
                    pass
                raise Exception(
                    "Failed to load GGUF model after cleanup retry.\n"
                    f"Model file exists: {normalized_model_path}\n"
                    f"Size: {size_gb:.2f} GB\n"
                    f"n_ctx: {int(n_ctx)}, n_gpu_layers: {int(n_gpu_layers)}, n_threads: {int(n_threads)}\n"
                    "This usually means llama-cpp could not mmap/load the file because of memory pressure, a locked/stale model handle, or an incompatible/corrupt GGUF.\n"
                    f"First error: {first_error}\n"
                    f"Retry error: {second_error}"
                ) from second_error
        _GGUF_MODEL_CACHE[key] = model
        if chat_handler is not None:
            _GGUF_CHAT_HANDLER_CACHE[key] = chat_handler
        return model

    def _unload_gguf_model(
        self,
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        chat_format: str,
        mmproj_path: str = "",
    ) -> None:
        key = self._gguf_model_key(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            mmproj_path=mmproj_path,
        )
        with _GGUF_LOCK:
            model = _GGUF_MODEL_CACHE.pop(key, None)
            chat_handler = _GGUF_CHAT_HANDLER_CACHE.pop(key, None)
            if model is not None or chat_handler is not None:
                _close_gguf_cached_resources(model, chat_handler)
            del model, chat_handler
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        except Exception:
            pass

    def _pil_to_data_url(self, img: Image.Image) -> str:
        image = img.convert("RGB")
        longest_side = max(image.size)
        if longest_side > 512:
            scale = 512 / float(longest_side)
            new_size = (
                max(1, int(image.size[0] * scale)),
                max(1, int(image.size[1] * scale)),
            )
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            image = image.resize(new_size, resample)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=86, optimize=True)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    def _extract_gguf_text(self, response) -> str:
        if isinstance(response, dict):
            choices = response.get("choices", []) or []
            if choices:
                message = choices[0].get("message", {}) or {}
                text = message.get("content", "")
                if isinstance(text, list):
                    text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
                return self._clean_gguf_control_text(text)
        return self._clean_gguf_control_text(response)

    def _clean_gguf_control_text(self, text) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        cutoff_match = re.search(
            r"<\|?(?:end|start)_of_turn\|?>|<\|?(?:end|start)_turn\|?>|\|(?:end|start)_of_turn\||_?(?:end|start)_of_turn|_?(?:end|start)_turn",
            cleaned,
            flags=re.IGNORECASE,
        )
        if cutoff_match:
            cleaned = cleaned[:cutoff_match.start()].strip()

        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r"</?thought>", "", cleaned, flags=re.IGNORECASE).strip()

        control_patterns = [
            r"^\s*_?\s*<\|channel>\s*(?:thought|analysis|reasoning)?\s*",
            r"^\s*_?\s*<\|?channel\|?>\s*(?:thought|analysis|reasoning)?\s*",
            r"^\s*_?\s*<channel\|>\s*(?:thought|analysis|reasoning)?\s*",
            r"^\s*_?\s*(?:thought|analysis|reasoning)\s*<channel\|>\s*",
            r"^\s*\d+\s*(?:thought|analysis|reasoning)\s*[:\-]?\s*",
            r"^\s*_?\s*(?:thought|analysis|reasoning)\s*[:\-]?\s*",
            r"^\s*_?\s*(?:end|start)_of_turn\s*",
            r"^\s*_?\s*(?:end|start)_turn\s*",
        ]
        previous = None
        while cleaned and previous != cleaned:
            previous = cleaned
            for pattern in control_patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        return cleaned

    def _build_gguf_messages(
        self,
        instruction_text: str,
        system_prompt: str = "",
    ) -> list[dict]:
        messages = []
        system_text = str(system_prompt or "").strip()
        user_text = str(instruction_text or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        if user_text:
            messages.append({"role": "user", "content": user_text})
        return messages

    def _run_gguf_text_pipeline(
        self,
        model,
        instruction_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        system_prompt: str = "",
        seed: Optional[int] = None,
        json_schema: Optional[dict] = None,
    ) -> str:
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = int(seed)
        call_args = {
            "messages": self._build_gguf_messages(instruction_text, system_prompt),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_new_tokens),
            "stop": list(self._GEMMA_STOP_SEQUENCES),
            **kwargs,
        }
        if json_schema:
            # llama.cpp turns the schema into a grammar, so every sampled token keeps the reply valid JSON.
            call_args["response_format"] = {"type": "json_object", "schema": json_schema}
        template_kwargs = getattr(self, "_qwen_chat_template_kwargs", None)
        if template_kwargs:
            call_args["chat_template_kwargs"] = dict(template_kwargs)
        try:
            response = model.create_chat_completion(**call_args)
        except TypeError:
            call_args.pop("seed", None)
            call_args.pop("chat_template_kwargs", None)
            response = model.create_chat_completion(**call_args)
        return self._extract_gguf_text(response)

    def _run_gguf_vision_pipeline(
        self,
        model,
        pil_images: list[Image.Image],
        instruction_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: Optional[int] = None,
    ) -> str:
        user_content = [
            {"type": "image_url", "image_url": {"url": self._pil_to_data_url(img)}}
            for img in pil_images
        ]
        user_content.append({"type": "text", "text": instruction_text})
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = int(seed)
        call_args = {
            "messages": [{"role": "user", "content": user_content}],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_new_tokens),
            "stop": list(self._GEMMA_STOP_SEQUENCES),
            **kwargs,
        }
        template_kwargs = getattr(self, "_qwen_chat_template_kwargs", None)
        if template_kwargs:
            call_args["chat_template_kwargs"] = dict(template_kwargs)
        try:
            response = model.create_chat_completion(**call_args)
        except TypeError:
            call_args.pop("seed", None)
            call_args.pop("chat_template_kwargs", None)
            response = model.create_chat_completion(**call_args)
        return self._extract_gguf_text(response)

    def generate_prompt(
        self,
        model_preset: str,
        custom_model_id: str,
        gguf_filename: str,
        mmproj_filename: str,
        task_preset: str,
        user_input: str,
        custom_instructions: str,
        trigger_word: str,
        image_count: int,
        download_if_missing: bool,
        unload_after_run: bool,
        hf_token: str,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        chat_format: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        **kwargs,
    ) -> Tuple[str, str, str]:
        model_id = self._resolve_model_id(model_preset, custom_model_id)
        instruction_text = self._build_instruction_text(
            task_preset, user_input, trigger_word, custom_instructions
        )
        if not instruction_text:
            return ("", model_id, "error: user_input/custom_instructions is empty")
        pil_images = self._collect_pil_images(image_count, kwargs)
        model_path = ""
        mmproj_path = ""

        with _GGUF_LOCK:
            try:
                model_path = self._resolve_gguf_model_path(
                    model_id=model_id,
                    gguf_filename=gguf_filename,
                    download_if_missing=bool(download_if_missing),
                    hf_token=hf_token,
                )
                mmproj_path = ""
                if pil_images:
                    mmproj_path = self._resolve_mmproj_path(
                        model_id=model_id,
                        mmproj_filename=mmproj_filename,
                        download_if_missing=bool(download_if_missing),
                        hf_token=hf_token,
                    )
                model = self._load_gguf_model(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    chat_format=chat_format,
                    mmproj_path=mmproj_path,
                )
                if pil_images:
                    text = self._run_gguf_vision_pipeline(
                        model,
                        pil_images,
                        instruction_text,
                        temperature,
                        top_p,
                        max_new_tokens,
                    )
                else:
                    text = self._run_gguf_text_pipeline(
                        model,
                        instruction_text,
                        temperature,
                        top_p,
                        max_new_tokens,
                    )
                text = str(text or "").strip()
                text = self._enforce_preset_output(task_preset, text)
                if not text:
                    raise Exception("Empty model response.")
                used_model = model_path
                if str(model_id or "").strip() != str(model_path or "").strip():
                    used_model = f"{model_id} :: {os.path.basename(model_path)}"
                return (text, used_model, "ok")
            except Exception as e:
                return ("", model_id, f"error: {e}")
            finally:
                if bool(unload_after_run) and model_path:
                    self._unload_gguf_model(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=n_threads,
                        chat_format=chat_format,
                        mmproj_path=mmproj_path,
                    )


class VRGDG_QwenGGUF(VRGDG_GeneralGGUF):
    """Qwen GGUF runner for text and multimodal llama.cpp models.

    The custom option intentionally accepts any local GGUF file/folder or
    Hugging Face repo, so newer Qwen quantizations do not require a code
    update just to be selectable.
    """

    MODELS_SUBDIR = "LLM"
    MISSING_MODEL_OPTION = "[No Qwen GGUF found in models/LLM]"
    MISSING_MMPROJ_OPTION = "[No Qwen mmproj GGUF found in models/LLM]"

    MODEL_PRESETS = [
        "unsloth/Qwen3.8-27B-GGUF",
        "custom",
    ]
    MM_PROJ_PRESETS = {
        "unsloth/Qwen3.8-27B-GGUF": "qwen-mmproj-BF16.gguf",
    }
    _GEMMA_STOP_SEQUENCES = (
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
    )

    @classmethod
    def _qwen_models_roots(cls):
        paths = []
        try:
            paths = folder_paths.get_folder_paths(cls.MODELS_SUBDIR)
        except Exception:
            pass
        if not paths:
            paths = [os.path.join(getattr(folder_paths, "models_dir", ""), cls.MODELS_SUBDIR)]
        return paths

    @classmethod
    def _qwen_models_root(cls):
        """Compatibility helper for callers that need one download root."""
        return cls._qwen_models_roots()[0]

    @classmethod
    def _list_local_qwen_gguf(cls):
        found = []
        for root in cls._qwen_models_roots():
            if not root or not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    low = filename.lower()
                    full_path = os.path.join(dirpath, filename)
                    relative = os.path.relpath(full_path, root)
                    if low.endswith(".gguf") and "mmproj" not in low and "qwen" in relative.lower():
                        found.append(relative)
        return sorted(set(found), key=str.lower) or ["[No Qwen GGUF found in models/LLM]"]

    @classmethod
    def _list_local_qwen_mmproj(cls):
        found = []
        for root in cls._qwen_models_roots():
            if not root or not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    if filename.lower().endswith(".gguf") and "mmproj" in filename.lower():
                        found.append(os.path.relpath(os.path.join(dirpath, filename), root))
        return sorted(set(found), key=str.lower) or ["[No Qwen mmproj GGUF found in models/LLM]"]

    @classmethod
    def _list_local_mmproj_choices(cls):
        return cls._list_local_qwen_mmproj()

    @classmethod
    def _resolve_dropdown_path(cls, selected_value, missing_value):
        selected = str(selected_value or "").strip()
        if not selected or selected == missing_value:
            raise Exception(f"{missing_value}. Choose a GGUF file under models/{cls.MODELS_SUBDIR} or use an absolute path.")
        if os.path.isfile(os.path.normpath(selected)):
            return os.path.normpath(selected)
        for root in cls._qwen_models_roots():
            candidate = os.path.normpath(os.path.join(root, selected))
            if os.path.isfile(candidate):
                return candidate
        raise Exception(f"Selected GGUF file was not found: {selected}")

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        local_models = cls._list_local_qwen_gguf()
        local_mmproj = cls._list_local_qwen_mmproj()
        inputs["required"]["model_file"] = (
            local_models,
            {"default": local_models[0], "tooltip": "Qwen GGUF files found under ComfyUI/models/LLM. Use custom_model_id for another path or Hugging Face repo."},
        )
        inputs["required"]["mmproj_file"] = (
            local_mmproj,
            {"default": local_mmproj[0], "tooltip": "Optional Qwen vision projector found under ComfyUI/models/LLM."},
        )
        inputs["required"]["model_preset"] = (
            cls.MODEL_PRESETS,
            {
                "default": cls.MODEL_PRESETS[0],
                "tooltip": "Choose the Qwen GGUF repo, or select custom and provide a local .gguf file/folder or another Hugging Face repo.",
            },
        )
        inputs["required"]["mmproj_filename"] = (
            "STRING",
            {
                "default": "",
                "tooltip": "Optional multimodal projector (.gguf). Required only for Qwen vision GGUFs when image inputs are used.",
            },
        )
        inputs["required"]["enable_thinking"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Qwen reasoning mode. Disable for a direct answer with lower latency and fewer output tokens.",
            },
        )
        return inputs

    def generate_prompt(self, *args, **kwargs):
        enable_thinking = bool(kwargs.pop("enable_thinking", False))
        model_file = str(kwargs.pop("model_file", "")).strip()
        mmproj_file = str(kwargs.pop("mmproj_file", "")).strip()
        if model_file and not model_file.startswith("["):
            kwargs["custom_model_id"] = self._resolve_dropdown_path(model_file, self.MISSING_MODEL_OPTION)
            kwargs["gguf_filename"] = ""
        if mmproj_file and not mmproj_file.startswith("["):
            kwargs["mmproj_filename"] = self._resolve_dropdown_path(mmproj_file, self.MISSING_MMPROJ_OPTION)
        self._qwen_chat_template_kwargs = {"enable_thinking": enable_thinking}
        try:
            return super().generate_prompt(*args, **kwargs)
        finally:
            self._qwen_chat_template_kwargs = None

    def _load_gguf_model(
        self, model_path, n_ctx, n_gpu_layers, n_threads, chat_format, mmproj_path=""
    ):
        """Use Qwen's native chat template for multimodal llama.cpp calls."""
        if not str(mmproj_path or "").strip():
            return super()._load_gguf_model(
                model_path, n_ctx, n_gpu_layers, n_threads, chat_format, mmproj_path
            )

        # Reuse the base loader's cache and retry behavior, but provide a
        # Qwen-compatible multimodal handler instead of Gemma turn tokens.
        key = self._gguf_model_key(model_path, n_ctx, n_gpu_layers, n_threads, chat_format, mmproj_path)
        cached = _GGUF_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except Exception as e:
            raise Exception("Missing dependency: install llama-cpp-python to use the Qwen GGUF node.") from e

        class QwenMTMDChatHandler(Llava15ChatHandler):
            DEFAULT_SYSTEM_MESSAGE = None
            CHAT_FORMAT = (
                "{% for message in messages %}"
                "{% if message.role == 'system' %}"
                "<|im_start|>system\\n{{ message.content }}<|im_end|>\\n"
                "{% elif message.role == 'user' %}"
                "<|im_start|>user\\n"
                "{% if message.content is string %}{{ message.content }}{% endif %}"
                "{% if message.content is iterable %}"
                "{% for content in message.content %}"
                "{% if content.type == 'image_url' and content.image_url is string %}{{ content.image_url }}{% endif %}"
                "{% if content.type == 'image_url' and content.image_url is mapping %}{{ content.image_url.url }}{% endif %}"
                "{% if content.type == 'text' %}{{ content.text }}{% endif %}"
                "{% endfor %}{% endif %}<|im_end|>\\n"
                "{% elif message.role == 'assistant' %}"
                "<|im_start|>assistant\\n{{ message.content }}<|im_end|>\\n"
                "{% endif %}{% endfor %}<|im_start|>assistant\\n"
                "{% if enable_thinking is defined and enable_thinking %}<think>\\n{% endif %}"
            )

        handler = QwenMTMDChatHandler(clip_model_path=str(mmproj_path), verbose=False)
        kwargs = {
            "model_path": str(model_path), "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers), "n_threads": int(n_threads),
            "verbose": False, "chat_handler": handler,
        }
        if str(chat_format or "").strip():
            kwargs["chat_format"] = str(chat_format).strip()
        model = Llama(**kwargs)
        _GGUF_MODEL_CACHE[key] = model
        _GGUF_CHAT_HANDLER_CACHE[key] = handler
        return model


class VRGDG_SuperGemmaGGUFChat(VRGDG_GeneralGGUF):
    MODELS_SUBDIR = "LLM"
    MISSING_MODEL_OPTION = "[No Gemma GGUF found in models/LLM]"
    MISSING_MMPROJ_OPTION = "[No mmproj GGUF found in models/LLM]"
    DEFAULT_N_CTX = 262144
    LLAMA_CPP_HELP_URLS = (
        "https://pypi.org/project/llama-cpp-python/",
        "https://llama-cpp-python.readthedocs.io/en/latest/",
        "https://github.com/abetlen/llama-cpp-python",
    )

    @classmethod
    def _format_supergemma_failure(cls, status: str, used_model: str = "") -> str:
        status_text = str(status or "").strip() or "Unknown error."
        used_model_text = str(used_model or "").strip()
        lines = [
            "SuperGemma GGUF Chat failed before it could produce text.",
            "",
            "Status:",
            status_text,
            "",
            "Most common cause:",
            "llama-cpp-python is missing, installed into the wrong Python, or installed without the CUDA/CPU wheel your system needs.",
            "",
            "What to try:",
            "1. Install into ComfyUI's embedded Python, not your system Python.",
            "2. Remove conflicting packages first: llama-cpp-python, llama_cpp_python, llama-cpp, llama_cpp, and llama-cpp-py.",
            "3. Install a llama-cpp-python wheel that matches your OS, Python version, CUDA version, and GPU architecture.",
            "4. Restart ComfyUI after installing.",
            "",
            "Useful docs:",
            *cls.LLAMA_CPP_HELP_URLS,
        ]
        if used_model_text:
            lines.extend(["", "Model:", used_model_text])
        return "\n".join(lines)

    @classmethod
    def _supergemma_models_root(cls) -> list[str]:
        paths = []

        try:
            paths = folder_paths.get_folder_paths(cls.MODELS_SUBDIR)
        except Exception:
            pass

        if not paths:
            base = getattr(folder_paths, "models_dir", None) or ""
            fallback = os.path.join(base, cls.MODELS_SUBDIR)
            paths = [fallback]

        return paths

    @classmethod
    def _list_local_gemma_gguf_choices(cls) -> list[str]:
        roots = cls._supergemma_models_root()
        found = []

        for root in roots:
            if not root or not os.path.isdir(root):
                continue

            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    low = str(filename or "").lower()

                    if not low.endswith(".gguf"):
                        continue

                    if "mmproj" in low:
                        continue

                    if "gemma" not in low:
                        continue

                    full_path = os.path.join(dirpath, filename)

                    rel_path = os.path.relpath(full_path, root)

                    found.append(rel_path.replace("/", os.sep))

        found = sorted(set(found), key=str.lower)

        return found or [cls.MISSING_MODEL_OPTION]

    @classmethod
    def _list_local_mmproj_choices(cls) -> list[str]:
        roots = cls._supergemma_models_root()
        found = []

        for root in roots:
            if not root or not os.path.isdir(root):
                continue

            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    low = str(filename or "").lower()

                    if not low.endswith(".gguf"):
                        continue

                    if "mmproj" not in low:
                        continue

                    full_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(full_path, root)
                    found.append(rel_path.replace("/", os.sep))

        found = sorted(set(found), key=str.lower)

        return found or [cls.MISSING_MMPROJ_OPTION]

    @classmethod
    def _is_text_only_supergemma_model(cls, value: str) -> bool:
        low = str(value or "").lower()
        return (
            "26" in low
            and "uncensored" in low
            and "vision" not in low
            and "mmproj" not in low
        )

    @classmethod
    def _resolve_dropdown_path(cls, selected_value: str, missing_value: str) -> str:
        selected = str(selected_value or "").strip()

        if not selected or selected == missing_value:
            raise Exception(
                f"{missing_value}. Place the file under models/{cls.MODELS_SUBDIR} and restart ComfyUI."
            )

        if os.path.isfile(os.path.normpath(selected)):
            return os.path.normpath(selected)

        roots = cls._supergemma_models_root()

        for root in roots:
            candidate = os.path.normpath(os.path.join(root, selected))

            if os.path.isfile(candidate):
                return candidate

        raise Exception(
            f"Selected file was not found in any LLM paths: {selected}"
        )

    @classmethod
    def INPUT_TYPES(cls):
        gemma_choices = cls._list_local_gemma_gguf_choices()
        mmproj_choices = cls._list_local_mmproj_choices()
        required = {
            "model_file": (
                gemma_choices,
                {
                    "default": gemma_choices[0],
                    "tooltip": "Gemma GGUF models found under ComfyUI/models/LLM. Only .gguf files with 'gemma' in the name are shown.",
                },
            ),
            "mmproj_file": (
                mmproj_choices,
                {
                    "default": mmproj_choices[0],
                    "tooltip": "mmproj GGUF files found under ComfyUI/models/LLM. Required only when using image inputs.",
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
            "advanced": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Show advanced GGUF runtime controls such as context, GPU layers, threads, sampler, and token limits.",
                },
            ),
            "unload_after_run": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "If enabled, unload the GGUF model from cache after this run to free RAM/VRAM.",
                },
            ),
            "n_ctx": (
                "INT",
                {
                    "default": cls.DEFAULT_N_CTX,
                    "min": 512,
                    "max": cls.DEFAULT_N_CTX,
                    "step": 256,
                    "tooltip": "GGUF context window. SuperGemma defaults to the full 262144-token training context; lower this if RAM/VRAM is too high.",
                },
            ),
            "n_gpu_layers": (
                "INT",
                {
                    "default": 99,
                    "min": -1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "How many layers to offload to GPU. Use -1 to offload all supported layers.",
                },
            ),
            "n_threads": (
                "INT",
                {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "CPU threads used by llama.cpp.",
                },
            ),
            "chat_format": (
                "STRING",
                {
                    "default": "",
                    "tooltip": "Optional llama.cpp chat format override. Leave blank to use the model default.",
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
        }
        optional = {
            f"image{i}": ("IMAGE", {"tooltip": "Optional reference image input."})
            for i in range(1, cls.MAX_IMAGES + 1)
        }
        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "used_model", "status")
    FUNCTION = "generate_prompt"
    CATEGORY = "VRGDG/LLM"

    def generate_prompt(
        self,
        model_file: str,
        mmproj_file: str,
        task_preset: str,
        user_input: str,
        custom_instructions: str,
        trigger_word: str,
        image_count: int,
        advanced: bool,
        unload_after_run: bool,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        chat_format: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        **kwargs,
    ) -> Tuple[str, str, str]:
        model_path = self._resolve_dropdown_path(model_file, self.MISSING_MODEL_OPTION)
        model_filename = os.path.basename(model_path)
        is_text_only_model = self._is_text_only_supergemma_model(model_file) or self._is_text_only_supergemma_model(model_path)
        if is_text_only_model:
            image_count = 0
            kwargs = {k: v for k, v in kwargs.items() if not str(k).startswith("image")}
        mmproj_path = ""
        try:
            pil_images = [] if is_text_only_model else self._collect_pil_images(image_count, kwargs)
        except Exception:
            pil_images = []
        if pil_images:
            mmproj_path = self._resolve_dropdown_path(mmproj_file, self.MISSING_MMPROJ_OPTION)
        text, used_model, status = super().generate_prompt(
            model_preset="custom",
            custom_model_id=model_path,
            gguf_filename=model_filename,
            mmproj_filename=mmproj_path,
            task_preset=task_preset,
            user_input=user_input,
            custom_instructions=custom_instructions,
            trigger_word=trigger_word,
            image_count=image_count,
            download_if_missing=False,
            unload_after_run=unload_after_run,
            hf_token="",
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        if str(status or "").strip().lower() != "ok":
            raise Exception(self._format_supergemma_failure(status, used_model))

        # --- Remove reasoning / template junk ---

        import re

        # Safety fallback
        if text is None:
            text = ""

        # Remove <think> blocks safely
        try:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r"</?thought>", "", text, flags=re.IGNORECASE)
        except Exception:
            pass

        # Remove leaked chat/control tokens from Gemma/GGUF templates.
        try:
            control_patterns = [
                r"^\s*_?\s*<\|channel>\s*(?:thought|analysis|reasoning)?\s*",
                r"^\s*_?\s*<\|?channel\|?>\s*(?:thought|analysis|reasoning)?\s*",
                r"^\s*_?\s*<channel\|>\s*",
                r"^\s*_?\s*(?:thought|analysis|reasoning)\s*<channel\|>\s*",
                r"^\s*_?\s*(?:thought|analysis|reasoning)\s*[:\-]?\s*",
            ]
            while True:
                previous = text
                for p in control_patterns:
                    text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)
                if text == previous:
                    break

            text = re.sub(r"_?\s*<\|channel>\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"_?\s*<\|?channel\|?>\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"_?\s*<channel\|>\s*", "", text, flags=re.IGNORECASE)
        except Exception:
            pass

        # Remove leading unicode junk safely
        try:
            # Remove known language/template prefixes safely
            prefix_patterns = [
                r"^나의 답변은 다음과 같습니다\.\s*",
                r"^回答[:：]\s*",
                r"^经典[-:：]\s*",
                r"^Assistant:\s*",
                r"^Answer:\s*",
            ]

            for p in prefix_patterns:
                text = re.sub(p, "", text, flags=re.IGNORECASE)
        except Exception:
            pass

        # Stop runaway loop tokens
        loop_markers = [
            "thought_turn",
            "turn_turn",
        ]

        for marker in loop_markers:
            if marker in text:
                text = text.split(marker)[0]

        text = text.strip()

        # --- GUARANTEED RETURN ---
        return (text or "", used_model or "", status or "")


class VRGDG_UnloadGemmaModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear_cuda_cache": ("BOOLEAN", {"default": True}),
                "clear_hf_pipeline_cache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Usually leave this off. Turn it on only if you also want to clear non-GGUF local LLM pipeline caches.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "VRGDG/LLM"
    DESCRIPTION = "Unloads cached VRGDG GGUF/Gemma llama.cpp models and optionally clears CUDA cache."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, clear_cuda_cache: bool = True, clear_hf_pipeline_cache: bool = False):
        result = _clear_vrgdg_llm_caches(
            clear_cuda_cache=bool(clear_cuda_cache),
            clear_hf_pipeline_cache=bool(clear_hf_pipeline_cache),
        )
        status = (
            "VRGDG Gemma/GGUF cleanup complete.\n"
            f"GGUF models unloaded: {result['gguf_models_unloaded']}\n"
            f"HF pipelines unloaded: {result['hf_pipelines_unloaded']}\n"
            f"CUDA cache cleared: {'yes' if result['cuda_cache_cleared'] else 'no'}"
        )
        print("[VRGDG] " + status.replace("\n", " | "))
        return (status,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_QwenGGUF": VRGDG_QwenGGUF,
    "VRGDG_SuperGemmaGGUFChat": VRGDG_SuperGemmaGGUFChat,
    "VRGDG_UnloadGemmaModels": VRGDG_UnloadGemmaModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_QwenGGUF": "🧠 VRGDG Qwen GGUF 🧠",
    "VRGDG_SuperGemmaGGUFChat": "🧠 VRGDG SuperGemma GGUF Chat 🧠",
    "VRGDG_UnloadGemmaModels": "VRGDG Unload Gemma/GGUF Models",
}
