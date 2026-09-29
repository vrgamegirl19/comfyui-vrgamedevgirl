import os
import re
import threading

import folder_paths
import numpy as np
import torch

try:
    import torchaudio
except Exception:
    torchaudio = None


class VRGDG_VoxCPM2Generate:
    RETURN_TYPES = ("AUDIO", "STRING", "INT", "STRING")
    RETURN_NAMES = ("audio", "saved_audio_path", "sample_rate", "status")
    FUNCTION = "generate"
    CATEGORY = "VRGDG/Audio"

    _MODEL_CACHE = {}
    _CACHE_LOCK = threading.Lock()
    TEXT_TOOLTIP = (
        "What you want the model to say.\n"
        "For cloning modes, this is the new target speech, not the transcript of your reference clip."
    )
    MODE_TOOLTIP = (
        "Choose how VoxCPM2 should speak:\n"
        "- text_to_speech: regular TTS with no voice reference.\n"
        "- voice_design: zero-shot styled speech guided by your wording alone.\n"
        "- prompt_continuation: continue from a prompt clip using prompt_audio plus prompt_text.\n"
        "- controllable_clone: easiest voice clone. Provide a clean reference clip.\n"
        "- ultimate_clone: strongest cloning mode. Provide a reference clip and the exact transcript in prompt_text."
    )
    DEVICE_TOOLTIP = (
        "auto picks CUDA when available and falls back to CPU.\n"
        "CUDA is strongly recommended for VoxCPM2."
    )
    CFG_TOOLTIP = (
        "Classifier-free guidance strength.\n"
        "A good starting point is 2.0. Lower can sound looser; higher can sound more forced."
    )
    TIMESTEPS_TOOLTIP = (
        "Number of inference steps.\n"
        "10 is a solid default. Higher may improve quality slightly but is slower."
    )
    DENOISER_TOOLTIP = (
        "Optional cleanup for noisy prompt or reference audio before cloning.\n"
        "Useful if your source clip has hiss, room noise, or light background noise."
    )
    SAVE_TOOLTIP = (
        "Base name for the saved WAV file in ComfyUI/output/VRGDG_AudioFiles."
    )
    REFERENCE_TOOLTIP = (
        "Speaker reference for voice cloning.\n"
        "Use a clean single-speaker clip, ideally 10 to 30 seconds."
    )
    PROMPT_AUDIO_TOOLTIP = (
        "Prompt/continuation audio.\n"
        "Use this for prompt_continuation, or for ultimate_clone when you want the model to match a specific spoken example."
    )
    PROMPT_TEXT_TOOLTIP = (
        "Exact transcript of the prompt/reference clip.\n"
        "Required for prompt_continuation and ultimate_clone. Match the spoken words as closely as possible."
    )
    MIN_LEN_TOOLTIP = (
        "Minimum generated audio token length.\n"
        "Usually safe to leave at the default."
    )
    MAX_LEN_TOOLTIP = (
        "Maximum generated token length.\n"
        "Increase if long text is getting cut off. Higher values can use more VRAM and time."
    )
    NORMALIZE_TOOLTIP = (
        "Normalize text before generation.\n"
        "Can help with number/date expansion and cleaner pronunciation on some inputs."
    )
    RETRY_TOOLTIP = (
        "Retry obviously bad generations automatically.\n"
        "This can improve reliability but may take longer."
    )
    RETRY_TIMES_TOOLTIP = (
        "Maximum retry attempts when retry_badcase is enabled."
    )
    RETRY_RATIO_TOOLTIP = (
        "Bad-case threshold for retry logic.\n"
        "Leave this alone unless you are tuning generation behavior."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "Hello from VoxCPM2.",
                        "multiline": True,
                        "tooltip": cls.TEXT_TOOLTIP,
                    },
                ),
                "mode": (
                    [
                        "text_to_speech",
                        "voice_design",
                        "prompt_continuation",
                        "controllable_clone",
                        "ultimate_clone",
                    ],
                    {"default": "text_to_speech", "tooltip": cls.MODE_TOOLTIP},
                ),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": cls.DEVICE_TOOLTIP}),
                "cfg_value": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": cls.CFG_TOOLTIP},
                ),
                "inference_timesteps": (
                    "INT",
                    {"default": 10, "min": 1, "max": 200, "step": 1, "tooltip": cls.TIMESTEPS_TOOLTIP},
                ),
                "load_denoiser": ("BOOLEAN", {"default": False, "tooltip": cls.DENOISER_TOOLTIP}),
                "normalize_text": ("BOOLEAN", {"default": False, "tooltip": cls.NORMALIZE_TOOLTIP}),
                "retry_badcase": ("BOOLEAN", {"default": True, "tooltip": cls.RETRY_TOOLTIP}),
                "retry_badcase_max_times": (
                    "INT",
                    {"default": 3, "min": 1, "max": 10, "step": 1, "tooltip": cls.RETRY_TIMES_TOOLTIP},
                ),
                "retry_badcase_ratio_threshold": (
                    "FLOAT",
                    {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": cls.RETRY_RATIO_TOOLTIP},
                ),
                "min_len": (
                    "INT",
                    {"default": 2, "min": 1, "max": 128, "step": 1, "tooltip": cls.MIN_LEN_TOOLTIP},
                ),
                "max_len": (
                    "INT",
                    {"default": 4096, "min": 32, "max": 16384, "step": 1, "tooltip": cls.MAX_LEN_TOOLTIP},
                ),
                "save_filename_prefix": (
                    "STRING",
                    {"default": "voxcpm2", "multiline": False, "tooltip": cls.SAVE_TOOLTIP},
                ),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": cls.REFERENCE_TOOLTIP}),
                "reference_audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional path to speaker reference audio",
                        "tooltip": cls.REFERENCE_TOOLTIP,
                    },
                ),
                "prompt_audio": ("AUDIO", {"tooltip": cls.PROMPT_AUDIO_TOOLTIP}),
                "prompt_audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional path to prompt/reference transcript audio",
                        "tooltip": cls.PROMPT_AUDIO_TOOLTIP,
                    },
                ),
                "prompt_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Required for ultimate_clone if prompt audio is used",
                        "tooltip": cls.PROMPT_TEXT_TOOLTIP,
                    },
                ),
            },
        }

    @staticmethod
    def _resolve_device(requested: str) -> str:
        requested = str(requested or "auto").strip().lower()
        if requested == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        value = str(name or "").strip() or "voxcpm2"
        value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", value)
        value = re.sub(r"\s+", "_", value)
        return value[:120] or "voxcpm2"

    @staticmethod
    def _resolve_path(raw_path: str) -> str:
        raw = str(raw_path or "").strip()
        if not raw:
            return ""
        if os.path.isabs(raw) and os.path.isfile(raw):
            return os.path.normpath(raw)

        candidates = [
            raw,
            os.path.join(folder_paths.get_input_directory(), raw),
            os.path.join(folder_paths.get_output_directory(), raw),
        ]
        get_temp = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp):
            candidates.append(os.path.join(get_temp(), raw))

        for candidate in candidates:
            full = os.path.normpath(candidate)
            if os.path.isfile(full):
                return full
        return ""

    @staticmethod
    def _extract_audio_tensor(audio) -> tuple[torch.Tensor, int]:
        if not isinstance(audio, dict):
            raise ValueError("Expected AUDIO input to be a dict.")
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            raise ValueError("AUDIO input is missing waveform or sample_rate.")
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Audio waveform must be [C,T], got {tuple(waveform.shape)}")
        return waveform.detach().cpu().float(), int(sample_rate)

    @classmethod
    def _write_audio_input_to_temp(
        cls,
        audio,
        output_dir: str,
        stem: str,
    ) -> str:
        if torchaudio is None:
            raise ImportError("torchaudio is required when using AUDIO inputs with VoxCPM2.")
        waveform, sample_rate = cls._extract_audio_tensor(audio)
        temp_dir = os.path.join(output_dir, "VRGDG_VoxCPM2_temp")
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, f"{stem}.wav")
        torchaudio.save(path, waveform, sample_rate, format="wav")
        return path

    @classmethod
    def _get_model(cls, device: str, load_denoiser: bool):
        try:
            from voxcpm import VoxCPM
        except Exception as exc:
            raise ImportError(
                "VoxCPM2 requires the `voxcpm` package. Install it with: pip install voxcpm"
            ) from exc

        key = ("openbmb/VoxCPM2", device, bool(load_denoiser), False)
        with cls._CACHE_LOCK:
            cached = cls._MODEL_CACHE.get(key)
            if cached is not None:
                return cached

            model = VoxCPM.from_pretrained(
                "openbmb/VoxCPM2",
                load_denoiser=bool(load_denoiser),
                optimize=False,
            )

            move_to = getattr(model, "to", None)
            if callable(move_to):
                model = move_to(device)

            eval_fn = getattr(model, "eval", None)
            if callable(eval_fn):
                eval_fn()

            cls._MODEL_CACHE[key] = model
            return model

    def _prepare_reference_path(
        self,
        output_dir: str,
        audio_input,
        audio_path: str,
        temp_stem: str,
    ) -> str:
        resolved = self._resolve_path(audio_path)
        if resolved:
            return resolved
        if audio_input is not None:
            return self._write_audio_input_to_temp(audio_input, output_dir, temp_stem)
        return ""

    def generate(
        self,
        text,
        mode,
        device,
        cfg_value,
        inference_timesteps,
        load_denoiser,
        normalize_text,
        retry_badcase,
        retry_badcase_max_times,
        retry_badcase_ratio_threshold,
        min_len,
        max_len,
        save_filename_prefix,
        reference_audio=None,
        reference_audio_path="",
        prompt_audio=None,
        prompt_audio_path="",
        prompt_text="",
    ):
        if torchaudio is None:
            raise ImportError("torchaudio is required for VoxCPM2 output saving.")

        output_dir = os.path.join(folder_paths.get_output_directory(), "VRGDG_AudioFiles")
        os.makedirs(output_dir, exist_ok=True)

        resolved_device = self._resolve_device(device)
        model = self._get_model(resolved_device, bool(load_denoiser))

        safe_prefix = self._sanitize_filename(save_filename_prefix)
        reference_path = self._prepare_reference_path(
            output_dir,
            reference_audio,
            reference_audio_path,
            f"{safe_prefix}_reference",
        )
        prompt_path = self._prepare_reference_path(
            output_dir,
            prompt_audio,
            prompt_audio_path,
            f"{safe_prefix}_prompt",
        )

        generation_kwargs = {
            "text": str(text or ""),
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "min_len": int(min_len),
            "max_len": int(max_len),
            "normalize": bool(normalize_text),
            "denoise": bool(load_denoiser),
            "retry_badcase": bool(retry_badcase),
            "retry_badcase_max_times": int(retry_badcase_max_times),
            "retry_badcase_ratio_threshold": float(retry_badcase_ratio_threshold),
        }

        mode = str(mode or "text_to_speech").strip().lower()
        if mode == "prompt_continuation":
            prompt_text = str(prompt_text or "").strip()
            if not prompt_path:
                raise ValueError(
                    "prompt_continuation needs prompt_audio or prompt_audio_path."
                )
            if not prompt_text:
                raise ValueError(
                    "prompt_continuation needs prompt_text: paste the exact words spoken in the prompt clip."
                )
            generation_kwargs["prompt_wav_path"] = prompt_path
            generation_kwargs["prompt_text"] = prompt_text
        elif mode == "controllable_clone":
            if not reference_path:
                raise ValueError(
                    "controllable_clone needs a speaker reference. Connect reference_audio or fill in reference_audio_path."
                )
            generation_kwargs["reference_wav_path"] = reference_path
        elif mode == "ultimate_clone":
            if not reference_path:
                raise ValueError(
                    "ultimate_clone needs a speaker reference. Connect reference_audio or fill in reference_audio_path."
                )
            if not prompt_path:
                prompt_path = reference_path
            prompt_text = str(prompt_text or "").strip()
            if not prompt_text:
                raise ValueError(
                    "ultimate_clone also needs prompt_text: paste the exact words spoken in your reference clip."
                )
            generation_kwargs["reference_wav_path"] = reference_path
            generation_kwargs["prompt_wav_path"] = prompt_path
            generation_kwargs["prompt_text"] = prompt_text
        elif mode in {"text_to_speech", "voice_design"}:
            pass
        else:
            raise ValueError(f"Unsupported VoxCPM2 mode: {mode}")

        with torch.inference_mode():
            wav = model.generate(**generation_kwargs)

        wav_np = np.asarray(wav, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = np.squeeze(wav_np)
        if wav_np.ndim != 1:
            raise ValueError(f"Unexpected VoxCPM2 output shape: {wav_np.shape}")

        sample_rate = int(
            getattr(getattr(model, "tts_model", None), "sample_rate", 48000)
        )
        waveform = torch.from_numpy(wav_np).unsqueeze(0).contiguous()

        file_path = os.path.join(output_dir, f"{safe_prefix}.wav")
        torchaudio.save(file_path, waveform, sample_rate, format="wav")

        audio = {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": sample_rate,
            "file_path": file_path,
            "filename": os.path.splitext(os.path.basename(file_path))[0],
            "metadata": {
                "model": "openbmb/VoxCPM2",
                "mode": mode,
                "device": resolved_device,
                "optimized": False,
            },
        }
        status = f"Generated with VoxCPM2 on {resolved_device} at {sample_rate} Hz"
        return (audio, file_path, sample_rate, status)


NODE_CLASS_MAPPINGS = {
    "VRGDG_VoxCPM2Generate": VRGDG_VoxCPM2Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_VoxCPM2Generate": "VRGDG VoxCPM2 Voice Clone / TTS",
}
