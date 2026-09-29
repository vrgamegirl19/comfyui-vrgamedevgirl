import os

import torch
import folder_paths

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    from comfy_extras.nodes_audio import load as comfy_load_audio
except Exception:
    comfy_load_audio = None

try:
    from demucs import pretrained
    from demucs.apply import apply_model
except Exception:
    pretrained = None
    apply_model = None


class VRGDG_GetStems:
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "drums", "bass", "other")
    FUNCTION = "run"
    CATEGORY = "VRGDG/Audio"

    _MODEL_CACHE = {}
    MODEL_NAME_TOOLTIP = (
        "Choose the Demucs preset:\n"
        "- htdemucs: Best default balance of quality/speed for most songs.\n"
        "  Does well on general music, but may still leave mild bleed/artifacts.\n"
        "- htdemucs_ft: Fine-tuned htdemucs with often cleaner separation.\n"
        "  Usually slower/heavier, and not always better on every track.\n"
        "- mdx_extra: Alternative tuning that can improve vocal/music split on some songs.\n"
        "  Can be less consistent and may sound worse on certain material.\n"
        "Quick pick: start with htdemucs, then compare htdemucs_ft, then mdx_extra."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["htdemucs", "htdemucs_ft", "mdx_extra"],
                    {"default": "htdemucs", "tooltip": cls.MODEL_NAME_TOOLTIP},
                ),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "audio_file_path": (
                    "STRING",
                    {"default": "", "placeholder": "Optional path. Leave empty to use AUDIO input."},
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    def _resolve_device(self, requested):
        req = str(requested or "auto").strip().lower()
        if req == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_audio_path(self, audio_file_path):
        raw = str(audio_file_path or "").strip()
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

        for path in candidates:
            full = os.path.normpath(path)
            if os.path.isfile(full):
                return full
        return ""

    def _load_from_audio_input(self, audio):
        if not isinstance(audio, dict):
            return None, None
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            return None, None
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(f"Audio waveform must be 3D [B,C,T], got {tuple(waveform.shape)}")
        return waveform.float(), int(sample_rate)

    def _load_waveform(self, audio_file_path, audio):
        waveform, sample_rate = self._load_from_audio_input(audio)
        if waveform is not None:
            return waveform, sample_rate

        resolved = self._resolve_audio_path(audio_file_path)
        if not resolved:
            raise ValueError("Provide a valid AUDIO input or audio_file_path.")

        comfy_error = None
        if comfy_load_audio is not None:
            try:
                wav, sr = comfy_load_audio(resolved)
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                if wav.ndim != 2:
                    raise ValueError(
                        f"ComfyUI audio loader returned unexpected shape {tuple(wav.shape)}"
                    )
                return wav.unsqueeze(0).float(), int(sr)
            except Exception as exc:
                comfy_error = exc

        if torchaudio is None:
            detail = f" ComfyUI audio loader error: {comfy_error}" if comfy_error else ""
            raise ImportError(f"No compatible audio-file loader is available.{detail}")

        try:
            wav, sr = torchaudio.load(resolved)
        except Exception as exc:
            detail = f" ComfyUI audio loader error: {comfy_error}" if comfy_error else ""
            raise RuntimeError(f"Unable to decode audio file: {resolved}.{detail}") from exc
        return wav.unsqueeze(0).float(), int(sr)

    @classmethod
    def _get_model(cls, model_name, device):
        if pretrained is None:
            raise ImportError(
                "demucs is not installed. Install with: pip install demucs torch torchaudio"
            )
        key = (str(model_name), str(device))
        cached = cls._MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        model = pretrained.get_model(model_name)
        model.to(device)
        model.eval()
        cls._MODEL_CACHE[key] = model
        return model

    def _normalize_for_demucs(self, waveform, sample_rate, model):
        # Use first batch item for source separation and force stereo input.
        mix = waveform[0]
        if mix.ndim != 2:
            raise ValueError(f"Expected [C,T] audio after batch select, got {tuple(mix.shape)}")

        if mix.shape[0] == 1:
            mix = mix.repeat(2, 1)
        elif mix.shape[0] > 2:
            mix = mix[:2, :]

        target_sr = int(getattr(model, "samplerate", sample_rate))
        if sample_rate != target_sr:
            if torchaudio is None:
                raise ImportError("torchaudio is required for resampling.")
            mix = torchaudio.functional.resample(mix, int(sample_rate), target_sr)
            sample_rate = target_sr

        return mix.unsqueeze(0).contiguous(), int(sample_rate)

    @staticmethod
    def _stem_audio(stem_tensor, sample_rate):
        return {"waveform": stem_tensor.unsqueeze(0).contiguous().cpu(), "sample_rate": int(sample_rate)}

    def run(self, model_name="htdemucs", device="auto", audio_file_path="", audio=None):
        device_name = self._resolve_device(device)
        model = self._get_model(model_name, device_name)
        waveform, sample_rate = self._load_waveform(audio_file_path, audio)
        mix, sample_rate = self._normalize_for_demucs(waveform, sample_rate, model)

        mix = mix.to(device_name)
        with torch.no_grad():
            try:
                stems = apply_model(model, mix, device=device_name, progress=False)
            except TypeError:
                stems = apply_model(model, mix)

        if not isinstance(stems, torch.Tensor):
            stems = torch.as_tensor(stems)
        stems = stems.detach()

        if stems.ndim == 4:
            stems = stems[0]
        if stems.ndim != 3:
            raise ValueError(f"Unexpected Demucs output shape: {tuple(stems.shape)}")

        source_names = list(getattr(model, "sources", []))
        source_to_tensor = {}
        for idx, name in enumerate(source_names):
            if idx < stems.shape[0]:
                source_to_tensor[str(name).strip().lower()] = stems[idx]

        # Fallback positional mapping if source names are unavailable.
        if not source_to_tensor:
            if stems.shape[0] < 4:
                raise ValueError("Demucs output does not include 4 stems.")
            source_to_tensor = {
                "drums": stems[0],
                "bass": stems[1],
                "other": stems[2],
                "vocals": stems[3],
            }

        missing = [k for k in ("vocals", "drums", "bass", "other") if k not in source_to_tensor]
        if missing:
            raise ValueError(f"Missing expected stems: {', '.join(missing)}")

        vocals = self._stem_audio(source_to_tensor["vocals"], sample_rate)
        drums = self._stem_audio(source_to_tensor["drums"], sample_rate)
        bass = self._stem_audio(source_to_tensor["bass"], sample_rate)
        other = self._stem_audio(source_to_tensor["other"], sample_rate)
        return (vocals, drums, bass, other)


class VRGDG_AudioCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": (
                    "STRING",
                    {
                        "default": "0:00",
                    },
                ),
                "end_time": (
                    "STRING",
                    {
                        "default": "1:00",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"
    DESCRIPTION = "Crop (trim) audio to a specific start and end time."

    def main(
        self,
        audio: "AUDIO",
        start_time: str = "0:00",
        end_time: str = "1:00",
    ):
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        if ":" not in start_time:
            start_time = f"00:{start_time}"
        if ":" not in end_time:
            end_time = f"00:{end_time}"

        # --- UPDATED PARSING (accepts mm:ss or mm:ss.xx) ---
        start_min, start_sec = start_time.split(":")
        start_seconds_time = 60 * int(start_min) + float(start_sec)
        start_frame = int(start_seconds_time * sample_rate)
        if start_frame >= waveform.shape[-1]:
            start_frame = waveform.shape[-1] - 1

        end_min, end_sec = end_time.split(":")
        end_seconds_time = 60 * int(end_min) + float(end_sec)
        end_frame = int(end_seconds_time * sample_rate)
        if end_frame >= waveform.shape[-1]:
            end_frame = waveform.shape[-1] - 1
        # --- END UPDATED SECTION ---

        if start_frame < 0:
            start_frame = 0
        if end_frame < 0:
            end_frame = 0

        if start_frame > end_frame:
            total_duration_sec = waveform.shape[-1] / sample_rate
            raise ValueError(
                f"Invalid crop range:\n"
                f"- Start time: {start_seconds_time} sec\n"
                f"- End time: {end_seconds_time} sec\n"
                f"- Total duration: {total_duration_sec:.2f} sec\n"
                f"Start time must come before end time, and both must be within the audio duration.\n"
                f"If this is your first run, double-check that the index or batch position is set to 0 or not set higher than the total number of sets in the read-me note."
            )

        return (
            {
                "waveform": waveform[..., start_frame:end_frame],
                "sample_rate": sample_rate,
            },
        )


NODE_CLASS_MAPPINGS = {
    "VRGDG_AudioCrop": VRGDG_AudioCrop,
    "VRGDG_GetStems": VRGDG_GetStems,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_AudioCrop": "✂️ VRGDG Audio Crop",
    "VRGDG_GetStems": "VRGDG_GetStems",
}
