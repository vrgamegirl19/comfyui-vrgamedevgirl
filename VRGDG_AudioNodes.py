import os
import re
import hashlib
import shutil

import torch
import folder_paths
from aiohttp import web
from server import PromptServer

try:
    import torchaudio
except Exception:
    torchaudio = None

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

        if torchaudio is None:
            raise ImportError("torchaudio is required to load audio_file_path.")

        resolved = self._resolve_audio_path(audio_file_path)
        if not resolved:
            raise ValueError("Provide a valid AUDIO input or audio_file_path.")

        wav, sr = torchaudio.load(resolved)
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


class VRGDG_LoadAudioWithPath:
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_file_path", "audio_name_clean")
    FUNCTION = "load"
    CATEGORY = "VRGDG/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["audio", "video"])
        files = sorted(files)
        if not files:
            files = [""]
        return {
            "required": {
                "audio": (files,),
            }
        }

    @classmethod
    def IS_CHANGED(cls, audio=""):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        if not str(audio or "").strip():
            return "No audio files found in ComfyUI/input. Add one and refresh."
        if not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        return True

    def _clean_name(self, file_path):
        name = os.path.basename(str(file_path or "")).strip()
        name = re.sub(r"\.[^.]*$", "", name)
        name = re.sub(r"[^A-Za-z0-9]", "", name)
        return name

    def load(self, audio):
        if torchaudio is None:
            raise ImportError("torchaudio is required to load audio from file path.")

        resolved = folder_paths.get_annotated_filepath(audio)

        output_dir = folder_paths.get_output_directory()
        audio_dir = os.path.join(output_dir, "VRGDG_AudioFiles")
        os.makedirs(audio_dir, exist_ok=True)

        for name in os.listdir(audio_dir):
            existing_path = os.path.join(audio_dir, name)
            if not os.path.isfile(existing_path):
                continue
            try:
                os.remove(existing_path)
            except OSError:
                pass

        dest_path = os.path.join(audio_dir, os.path.basename(resolved))
        shutil.copy2(resolved, dest_path)

        waveform, sample_rate = torchaudio.load(resolved)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        audio = {
            "waveform": waveform.unsqueeze(0).contiguous().float(),
            "sample_rate": int(sample_rate),
            "file_path": resolved,
            "filename": os.path.splitext(os.path.basename(resolved))[0],
        }
        clean_name = self._clean_name(resolved)
        return (audio, resolved, clean_name)


class VRGDG_CreateSilentAudio:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "create"
    CATEGORY = "VRGDG/Audio"

    SAMPLE_RATE = 44100
    DURATION_SECONDS = 5
    CHANNELS = 2

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def create(self):
        total_samples = self.SAMPLE_RATE * self.DURATION_SECONDS
        waveform = torch.zeros((1, self.CHANNELS, total_samples), dtype=torch.float32)
        audio = {
            "waveform": waveform,
            "sample_rate": int(self.SAMPLE_RATE),
            "filename": "silence_5s",
        }
        return (audio,)


class VRGDG_SaveAudio:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_audio_path",)
    FUNCTION = "save"
    CATEGORY = "VRGDG/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "source_audio_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def _sanitize_filename(self, filename):
        name = str(filename or "").strip()
        if not name:
            name = "vrgdg_audio"
        invalid = '<>:"/\\|?*'
        for ch in invalid:
            name = name.replace(ch, "_")
        return name

    def _extract_base_name(self, audio):
        if not isinstance(audio, dict):
            return "vrgdg_audio"

        def pick_name(value):
            if value is None:
                return ""
            raw = str(value).strip()
            if not raw:
                return ""
            base = os.path.basename(raw)
            stem = os.path.splitext(base)[0].strip()
            return stem or raw

        for key in ("filename", "file_name", "name", "audio_file_path", "file_path", "path"):
            name = pick_name(audio.get(key))
            if name:
                return name

        metadata = audio.get("metadata")
        if isinstance(metadata, dict):
            for key in ("filename", "file_name", "name", "audio_file_path", "file_path", "path"):
                name = pick_name(metadata.get(key))
                if name:
                    return name

        return "vrgdg_audio"

    def save(self, audio, source_audio_path=""):
        if torchaudio is None:
            raise ImportError("torchaudio is required to save audio as MP3.")

        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        if waveform is None or sample_rate is None:
            raise ValueError("Invalid AUDIO input. Expected keys: waveform, sample_rate.")

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)

        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Audio waveform must be 2D [C,T], got {tuple(waveform.shape)}")

        output_dir = folder_paths.get_output_directory()
        audio_dir = os.path.join(output_dir, "VRGDG_AudioFiles")
        os.makedirs(audio_dir, exist_ok=True)

        source_audio_path = str(source_audio_path or "").strip()
        if source_audio_path:
            base_name = os.path.splitext(os.path.basename(source_audio_path))[0].strip()
        else:
            base_name = self._extract_base_name(audio)
        safe_name = self._sanitize_filename(base_name)
        file_path = os.path.join(audio_dir, f"{safe_name}.mp3")

        for name in os.listdir(audio_dir):
            existing_path = os.path.join(audio_dir, name)
            if not os.path.isfile(existing_path):
                continue
            if not name.lower().endswith(".mp3"):
                continue
            if os.path.normcase(os.path.normpath(existing_path)) == os.path.normcase(os.path.normpath(file_path)):
                continue
            try:
                os.remove(existing_path)
            except OSError:
                pass

        torchaudio.save(file_path, waveform.detach().cpu(), int(sample_rate), format="mp3")
        return (file_path,)


class VRGDG_GetAudioFilePath:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("audio_file_path", "audio_name_clean")
    FUNCTION = "get_path"
    CATEGORY = "VRGDG/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refresh_trigger": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, refresh_trigger=0):
        output_dir = folder_paths.get_output_directory()
        audio_dir = os.path.join(output_dir, "VRGDG_AudioFiles")
        if not os.path.isdir(audio_dir):
            return f"{int(refresh_trigger)}|missing"

        newest = 0.0
        for name in os.listdir(audio_dir):
            full_path = os.path.join(audio_dir, name)
            if not os.path.isfile(full_path):
                continue
            newest = max(newest, os.path.getctime(full_path), os.path.getmtime(full_path))
        return f"{int(refresh_trigger)}|{newest}"

    def _clean_name(self, file_path):
        name = os.path.basename(str(file_path or "")).strip()
        name = re.sub(r"\.[^.]*$", "", name)
        name = re.sub(r"[^A-Za-z0-9]", "", name)
        return name

    def get_path(self, refresh_trigger=0):
        output_dir = folder_paths.get_output_directory()
        audio_dir = os.path.join(output_dir, "VRGDG_AudioFiles")
        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(f"Audio folder not found: {audio_dir}")

        candidates = []
        for name in os.listdir(audio_dir):
            full_path = os.path.join(audio_dir, name)
            if not os.path.isfile(full_path):
                continue
            created_or_modified = max(os.path.getctime(full_path), os.path.getmtime(full_path))
            candidates.append((created_or_modified, full_path))

        if not candidates:
            raise FileNotFoundError(f"No audio files found in: {audio_dir}")

        candidates.sort(key=lambda item: item[0], reverse=True)
        newest_path = os.path.normpath(candidates[0][1])
        clean_name = self._clean_name(newest_path)
        return (newest_path, clean_name)


_VRGDG_AUDIO_ROUTES_REGISTERED = False


def _list_input_audio_files():
    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files = folder_paths.filter_files_content_types(files, ["audio", "video"])
    return sorted(files), input_dir


def _ensure_audio_routes_registered():
    global _VRGDG_AUDIO_ROUTES_REGISTERED
    if _VRGDG_AUDIO_ROUTES_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.get("/vrgdg/audio/list")
    async def vrgdg_audio_list(request):
        files, input_dir = _list_input_audio_files()
        return web.json_response({"files": files, "input_dir": input_dir})

    @server_instance.routes.post("/vrgdg/audio/upload")
    async def vrgdg_audio_upload(request):
        post = await request.post()
        audio = post.get("audio")
        overwrite = str(post.get("overwrite", "")).strip().lower() in {"1", "true", "yes", "on"}
        if audio is None or not getattr(audio, "file", None):
            return web.json_response({"ok": False, "error": "Missing audio file."}, status=400)

        filename = os.path.basename(str(getattr(audio, "filename", "") or "").strip())
        if not filename:
            return web.json_response({"ok": False, "error": "Invalid filename."}, status=400)

        input_dir = folder_paths.get_input_directory()
        base, ext = os.path.splitext(filename)
        if not base:
            base = "audio_upload"

        candidate = os.path.join(input_dir, f"{base}{ext}")
        if not overwrite:
            idx = 1
            while os.path.exists(candidate):
                candidate = os.path.join(input_dir, f"{base} ({idx}){ext}")
                idx += 1

        with open(candidate, "wb") as f:
            f.write(audio.file.read())

        saved_name = os.path.basename(candidate)
        files, _ = _list_input_audio_files()
        return web.json_response({"ok": True, "name": saved_name, "files": files})

    _VRGDG_AUDIO_ROUTES_REGISTERED = True


_ensure_audio_routes_registered()


NODE_CLASS_MAPPINGS = {
    "VRGDG_GetStems": VRGDG_GetStems,
    "VRGDG_LoadAudioWithPath": VRGDG_LoadAudioWithPath,
    "VRGDG_CreateSilentAudio": VRGDG_CreateSilentAudio,
    "VRGDG_SaveAudio": VRGDG_SaveAudio,
    "VRGDG_GetAudioFilePath": VRGDG_GetAudioFilePath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_GetStems": "VRGDG_GetStems",
    "VRGDG_LoadAudioWithPath": "VRGDG_LoadAudioWithPath",
    "VRGDG_CreateSilentAudio": "VRGDG_CreateSilentAudio",
    "VRGDG_SaveAudio": "VRGDG_SaveAudio",
    "VRGDG_GetAudioFilePath": "VRGDG_GetAudioFilePath",
}



