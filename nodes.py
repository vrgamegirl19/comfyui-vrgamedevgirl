import os
import requests
import torch
import torch.nn.functional as F
import comfy
import kornia
import librosa
import torchaudio
import folder_paths
from typing import Union, Tuple
from pathlib import Path
import av
import hashlib




class FastFilmGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "grain_intensity": (
                    "FLOAT", {"default": 0.04, "min": 0.001, "max": 1.0, "step": 0.001}
                ),
                "saturation_mix": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "batch_size": (
                    "INT", {"default": 4, "min": 0, "max": 500, "step": 1}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "video/enhancement"
    DESCRIPTION = "Adds lightweight film grain"

    def apply_grain(self, images, grain_intensity, saturation_mix, batch_size):
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        outputs = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            grain = torch.randn_like(batch)

            grain[:, :, :, 0] *= 2.0  # red
            grain[:, :, :, 2] *= 3.0  # blue

            gray = grain[:, :, :, 1].unsqueeze(3).repeat(1, 1, 1, 3)
            grain = saturation_mix * grain + (1.0 - saturation_mix) * gray

            output = batch + grain * grain_intensity
            output = output.clamp(0.0, 1.0)
            outputs.append(output)

        output = torch.cat(outputs, dim=0)
        output = output.to(comfy.model_management.intermediate_device())
        return (output,)



class ColorMatchToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "match_strength": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "batch_size": (
                    "INT", {"default": 4, "min": 0, "max": 500, "step": 1}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "video/enhancement"
    DESCRIPTION = "Matches the color tone of input image to a reference image using LAB mean/std alignment"

    def match_color(self, images, reference_image, match_strength, batch_size):
        device = comfy.model_management.get_torch_device()
        images = images.to(device)
        reference_image = reference_image.to(device)

        images = images.permute(0, 3, 1, 2)
        reference_image = reference_image.permute(0, 3, 1, 2)

        ref_lab = kornia.color.rgb_to_lab(reference_image)
        ref_mean = ref_lab.mean(dim=[2, 3], keepdim=True)
        ref_std = ref_lab.std(dim=[2, 3], keepdim=True) + 1e-5

        outputs = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            img_lab = kornia.color.rgb_to_lab(batch)
            img_mean = img_lab.mean(dim=[2, 3], keepdim=True)
            img_std = img_lab.std(dim=[2, 3], keepdim=True) + 1e-5

            matched_lab = (img_lab - img_mean) / img_std * ref_std + ref_mean
            blended_lab = match_strength * matched_lab + (1.0 - match_strength) * img_lab

            output = kornia.color.lab_to_rgb(blended_lab)
            outputs.append(output)

        output = torch.cat(outputs, dim=0).clamp(0.0, 1.0)
        output = output.permute(0, 2, 3, 1)
        output = output.to(comfy.model_management.intermediate_device())
        return (output,)



class FastUnsharpSharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": (
                    "FLOAT", {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01
                    }
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_unsharp"
    CATEGORY = "video/enhancement"
    DESCRIPTION = "Sharpens image using a fast unsharp masking technique."

    def apply_unsharp(self, images: torch.Tensor, strength: float) -> Tuple[torch.Tensor]:
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        # Convert to NCHW
        x = images.permute(0, 3, 1, 2)

        # Apply Gaussian blur (3x3 kernel)
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Unsharp mask
        sharpened = x + strength * (x - blur)

        # Clamp and convert back
        sharpened = sharpened.clamp(0.0, 1.0)
        sharpened = sharpened.permute(0, 2, 3, 1)
        sharpened = sharpened.to(comfy.model_management.intermediate_device())

        return (sharpened,)


class FastLaplacianSharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": (
                    "FLOAT", {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01
                    }
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_laplacian"
    CATEGORY = "video/enhancement"
    DESCRIPTION = "Sharpens image using a Laplacian edge enhancement method."

    def apply_laplacian(self, images: torch.Tensor, strength: float) -> Tuple[torch.Tensor]:
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        # Convert to NCHW
        x = images.permute(0, 3, 1, 2)

        # Define Laplacian kernel (3x3)
        kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]], dtype=torch.float32, device=device
        ).expand(3, 1, 3, 3)

        # Apply depthwise convolution
        edges = torch.nn.functional.conv2d(x, kernel, padding=1, groups=3)

        # Enhance with strength
        sharpened = x + strength * edges
        sharpened = sharpened.clamp(0.0, 1.0)

        # Convert back to NHWC
        sharpened = sharpened.permute(0, 2, 3, 1)
        sharpened = sharpened.to(comfy.model_management.intermediate_device())
        return (sharpened,)


class FastSobelSharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": (
                    "FLOAT", {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01
                    }
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sobel"
    CATEGORY = "video/enhancement"
    DESCRIPTION = "Sharpens image using Sobel edge detection to enhance gradients."

    def apply_sobel(self, images: torch.Tensor, strength: float) -> Tuple[torch.Tensor]:
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        # Convert to NCHW
        x = images.permute(0, 3, 1, 2)

        # Sobel kernels (Gx, Gy)
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32, device=device
        ).expand(3, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32, device=device
        ).expand(3, 1, 3, 3)

        # Compute gradients
        grad_x = torch.nn.functional.conv2d(x, sobel_x, padding=1, groups=3)
        grad_y = torch.nn.functional.conv2d(x, sobel_y, padding=1, groups=3)

        # Combine gradients
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # Add to original
        sharpened = x + strength * edges
        sharpened = sharpened.clamp(0.0, 1.0)

        # Convert back to NHWC
        sharpened = sharpened.permute(0, 2, 3, 1)
        sharpened = sharpened.to(comfy.model_management.intermediate_device())
        return (sharpened,)

#################### Added on 9/16/2025

####all the below is new for infinite talk.
# Utility functions

def db_to_scalar(db: float):
    return 10 ** (db / 20)

def load_audio(
    path: Union[str, Path],
    sr: Union[None, int, float] = None,
    offset: float = 0.0,
    duration: Union[float, None] = None,
    make_stereo: bool = True,
):
    mix, sr = librosa.load(path, sr=sr, mono=False, offset=offset, duration=duration)
    mix = torch.from_numpy(mix)

    # Ensure shape is [channels, samples]
    if len(mix.shape) == 1:
        mix = torch.stack([mix], dim=0)

    if make_stereo:
        if mix.shape[0] == 1:
            mix = torch.cat([mix, mix], dim=0)
        elif mix.shape[0] != 2:
            raise ValueError(f"Unsupported channel count: {mix.shape[0]}")

    # Add batch dimension: [1, channels, samples]
    mix = mix.unsqueeze(0)

    return {
        "sample_rate": round(sr),
        "waveform": mix,
    }    





class VRGDG_LoadAudioSplitDynamic:
    RETURN_TYPES = ("DICT", "FLOAT") + tuple(["AUDIO"] * 50)
    RETURN_NAMES = ("meta", "total_duration") + tuple([f"audio_{i}" for i in range(1, 51)])

    FUNCTION = "split_audio"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"duration_{i}": (
                "FLOAT",
                {"default": 3.0, "min": 0.0, "step": 0.01, "round": 0.01}
            )
            for i in range(1, 51)
        }

        return {
            "required": {
                "path": ("STRING", {"default": "./audio.mp3"}),
                "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "scene_count": ("INT", {"default": 1, "min": 1, "max": 50}),
                "using_infinite_talk": (
                    ["false", "true"],
                    {
                        "default": "false",
                        "label": "Using InfiniteTalk?",
                        "tooltip": "If using HUMO, change this to false."
                    }
                ),
            },
            "optional": optional,
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ("DICT", "FLOAT") + tuple(["AUDIO"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ["meta", "total_duration"] + [f"audio_{i+1}" for i in range(count)]

    def split_audio(self, path, offset_seconds, scene_count=1, using_infinite_talk="false", **kwargs):
        # internal processing parameters
        internal_chunk_duration = 8.0
        gain_db = 0.0
        resample_to_hz = 0.0
        make_stereo = True

        # convert toggle to bool
        use_padding = str(using_infinite_talk).lower() == "true"

        # collect durations
        scene_count = max(1, int(scene_count))
        durations = []
        for i in range(scene_count):
            v = kwargs.get(f"duration_{i+1}", 3.0)
            try:
                durations.append(float(v))
            except Exception:
                durations.append(3.0)

        # compute start times
        starts = []
        current_time = float(offset_seconds)
        for d in durations:
            starts.append(current_time)
            current_time += float(d)

        # get audio metadata
        src_sample_rate = 44100
        audio_total_duration = 0.0
        if torchaudio is not None:
            try:
                metadata = torchaudio.info(path)
                src_sample_rate = int(getattr(metadata, "sample_rate", src_sample_rate))
                num_frames = int(getattr(metadata, "num_frames", 0))
                sr_for_duration = max(1, int(getattr(metadata, "sample_rate", src_sample_rate)))
                audio_total_duration = float(num_frames) / float(sr_for_duration)
            except Exception as e:
                print(f"[VRGDG_LoadAudioSplitDynamic] torchaudio.info failed: {e}")

        target_length = int(internal_chunk_duration * src_sample_rate)
        segments = []

        for idx, start_time in enumerate(starts):
            requested_duration = float(durations[idx])
            load_duration = requested_duration if not use_padding else min(
                internal_chunk_duration,
                max(0.0, audio_total_duration - float(start_time))
            )

            try:
                audio = load_audio(
                    path,
                    sr=None if resample_to_hz <= 0 else resample_to_hz,
                    offset=float(start_time),
                    duration=float(load_duration),
                    make_stereo=make_stereo,
                )

                if not audio or "waveform" not in audio:
                    raise RuntimeError("Audio load failed or waveform missing.")

                # apply gain
                if gain_db != 0.0:
                    gain_scalar = db_to_scalar(gain_db)
                    audio["waveform"] *= gain_scalar

                waveform = audio["waveform"]

                if use_padding:
                    # truncate
                    # pad with silence if shorter than 8s
                    current_length = waveform.shape[-1]
                    if current_length < target_length:
                        pad_amount = target_length - current_length
                        silence_shape = list(waveform.shape)
                        silence_shape[-1] = pad_amount
                        silence = torch.zeros(
                            silence_shape, dtype=waveform.dtype, device=getattr(waveform, "device", "cpu")
                        )
                        waveform = torch.cat((waveform, silence), dim=-1)

                audio["waveform"] = waveform

                if "sample_rate" not in audio:
                    audio["sample_rate"] = src_sample_rate

            except Exception as e:
                print(f"[VRGDG_LoadAudioSplitDynamic] Failed to load segment {idx+1}: {e}")
                dummy_waveform = torch.zeros((1, 2, target_length))
                audio = {"waveform": dummy_waveform, "sample_rate": src_sample_rate}

            segments.append(audio)

        meta = {
            "scene_count": scene_count,
            "durations": durations,
            "offset_seconds": float(offset_seconds),
            "starts": starts,
            "sample_rate": int(src_sample_rate),
            "internal_chunk_duration": float(internal_chunk_duration),
            "audio_total_duration": float(audio_total_duration),
            "outputs_count": len(segments),
            "used_padding": use_padding,
        }

        return (meta, float(audio_total_duration), *tuple(segments))




# Utility functions
def db_to_scalar(db: float):
    return 10 ** (db / 20)


def load_audio(
    path: Union[str, Path],
    sr: Union[None, int, float] = None,
    offset: float = 0.0,
    duration: Union[float, None] = None,
    make_stereo: bool = True,
):
    mix, sr = librosa.load(path, sr=sr, mono=False, offset=offset, duration=duration)
    mix = torch.from_numpy(mix)

    # Ensure shape is [channels, samples]
    if len(mix.shape) == 1:
        mix = torch.stack([mix], dim=0)

    if make_stereo:
        if mix.shape[0] == 1:
            mix = torch.cat([mix, mix], dim=0)
        elif mix.shape[0] != 2:
            raise ValueError(f"Unsupported channel count: {mix.shape[0]}")

    # Add batch dimension: [1, channels, samples]
    mix = mix.unsqueeze(0)

    return {
        "sample_rate": round(sr),
        "waveform": mix,
    }



class VRGDG_LoadAudioSplit_HUMO:
    RETURN_TYPES = ("DICT", "FLOAT") + tuple(["AUDIO"] * 50)
    RETURN_NAMES = ("meta", "total_duration") + tuple([f"audio_{i}" for i in range(1, 51)])

    FUNCTION = "split_audio"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),  # <-- changed from path:("STRING") to audio:("AUDIO",)
                "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "scene_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 50,
                    "dynamic": True  # ✅ still allows refresh
                }),
            }
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ("DICT", "FLOAT") + tuple(["AUDIO"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ["meta", "total_duration"] + [f"audio_{i+1}" for i in range(count)]

    def split_audio(self, audio, offset_seconds, scene_count=1):  # <-- input now is audio dict
        # internal processing parameters
        internal_chunk_duration = 8.0
        gain_db = 0.0
        resample_to_hz = 0.0
        make_stereo = True

        # extract waveform + metadata from AUDIO input
        waveform = audio["waveform"]          # (B, C, T) or (C, T)
        src_sample_rate = int(audio.get("sample_rate", 44100))

        # normalize shape
        if waveform.ndim == 2:  # (C, T)
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        audio_total_duration = float(total_samples) / float(src_sample_rate)

        # hardcode duration for each scene
        scene_count = max(1, int(scene_count))
        durations = [3.88] * scene_count

        # compute start times
        starts = []
        current_time = float(offset_seconds)
        for d in durations:
            starts.append(current_time)
            current_time += float(d)

        target_length = int(internal_chunk_duration * src_sample_rate)
        segments = []

        for idx, start_time in enumerate(starts):
            requested_duration = float(durations[idx])
            start_samp = max(0, int(start_time * src_sample_rate))
            end_samp = min(total_samples, int(start_samp + requested_duration * src_sample_rate))

            seg = waveform[..., start_samp:end_samp]

            # apply gain
            if gain_db != 0.0:
                gain_scalar = db_to_scalar(gain_db)
                seg *= gain_scalar

            if make_stereo and seg.shape[1] == 1:  # mono -> stereo
                seg = seg.repeat(1, 2, 1)

            segments.append({"waveform": seg, "sample_rate": src_sample_rate})

        meta = {
            "scene_count": scene_count,
            "durations": durations,
            "offset_seconds": float(offset_seconds),
            "starts": starts,
            "sample_rate": int(src_sample_rate),
            "internal_chunk_duration": float(internal_chunk_duration),
            "audio_total_duration": float(audio_total_duration),
            "outputs_count": len(segments),
            "used_padding": False,
        }

        return (meta, float(audio_total_duration), *tuple(segments))









class VRGDG_Extract_Frame_Number:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_number": ("INT", {"default": 1, "min": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
        }

    RETURN_TYPES = ("LIST", "IMAGE", "MASK")
    RETURN_NAMES = ("index_list", "images", "masks")
    FUNCTION = "extract"

    CATEGORY = "image"

    def extract(self, frame_number, images=None, masks=None):
       

        # make sure index is valid (convert to zero-based)
        idx = max(0, frame_number - 1)

        original_length = 0
        if images is not None:
            original_length = max(original_length, len(images))
        if masks is not None:
            original_length = max(original_length, len(masks))

        if original_length > 0:
            idx = min(idx, original_length - 1)

        ids = [idx]

        new_images = []
        new_masks = []

        for i in ids:
            if images is not None:
                new_images.append(images[min(i, len(images) - 1)].detach().clone())
            else:
                new_images.append(torch.zeros(512, 512, 3))

            if masks is not None:
                new_masks.append(masks[min(i, len(masks) - 1)].detach().clone())
            else:
                new_masks.append(torch.zeros(512, 512))

        return (ids, torch.stack(new_images, dim=0), torch.stack(new_masks, dim=0))




class VRGDG_VideoSplitter:


    MAX_CHUNKS = 50  # Maximum number of outputs (adjust as needed)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "chunk_count": ("INT", {"default": 2, "min": 1, "max": s.MAX_CHUNKS}),
                "frames_per_chunk": ("INT", {"default": 97, "min": 1}),
            }
        }

    # Define outputs for validation
    RETURN_TYPES = ("IMAGE",) * MAX_CHUNKS
    RETURN_NAMES = tuple(f"chunk_{i+1}" for i in range(MAX_CHUNKS))
    FUNCTION = "split"
    CATEGORY = "image/filters/frames"
    DESCRIPTION = "Split an IMAGE batch into fixed-size chunks. Unused outputs return empty IMAGE batches."

    def split(self, images, chunk_count, frames_per_chunk):
        total = len(images)

        if total > 0:
            _, H, W, C = images.shape
            dtype = images.dtype
            device = images.device
        else:
            H, W, C = 512, 512, 3
            dtype = torch.float32
            device = "cpu"

        outputs = []
        for i in range(self.MAX_CHUNKS):
            if i < chunk_count:
                start = i * frames_per_chunk
                end = min(start + frames_per_chunk, total)

                if start < total:
                    chunk = images[start:end].detach().clone()
                else:
                    chunk = torch.zeros((0, H, W, C), dtype=dtype, device=device)
            else:
                # pad unused outputs
                chunk = torch.zeros((0, H, W, C), dtype=dtype, device=device)

            outputs.append(chunk)

        return tuple(outputs)




class VRGDG_LoadAudioSplitUpload:
    """
    Audio Splitter that takes an AUDIO input and splits it into multiple chunks.
    total_duration (2nd output) = sum(duration_1..duration_N)
    """

    RETURN_TYPES = ("DICT", "FLOAT") + tuple(["AUDIO"] * 50)
    RETURN_NAMES = ("meta", "total_duration") + tuple([f"audio_{i}" for i in range(1, 51)])
    FUNCTION = "split_audio"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"duration_{i}": (
                "FLOAT",
                {"default": 3.88, "min": 0.0, "step": 0.01, "round": 0.01}
            )
            for i in range(1, 51)
        }
        return {
            "required": {
                "audio": ("AUDIO",),  # <-- plug LoadAudio or any AUDIO here
                "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "scene_count": ("INT", {"default": 1, "min": 1, "max": 50}),
                "using_infinite_talk": (["false", "true"], {"default": "false"}),
            },
            "optional": optional,
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ("DICT", "FLOAT") + tuple(["AUDIO"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ["meta", "total_duration"] + [f"audio_{i+1}" for i in range(count)]

    def split_audio(self, audio, offset_seconds=0.0, scene_count=1, using_infinite_talk="false", **kwargs):
        waveform = audio["waveform"]          # (B, C, T) or (C, T)
        sample_rate = int(audio["sample_rate"])
        use_padding = str(using_infinite_talk).lower() == "true"
        internal_chunk_duration = 8.0

        # normalize shape
        if waveform.ndim == 2:  # (C, T)
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        source_audio_duration = float(total_samples) / float(sample_rate)

        # always at least 1 scene
        scene_count = max(1, int(scene_count))

        # collect durations safely
        durations = []
        for i in range(scene_count):
            v = kwargs.get(f"duration_{i+1}", 3.0)
            try:
                durations.append(float(v))
            except Exception:
                durations.append(3.0)

        # start times
        starts = []
        t = float(offset_seconds)
        for d in durations:
            starts.append(t)
            t += float(d)

        # split segments
        target_len = int(internal_chunk_duration * sample_rate)
        segments = []

        for idx, start_time in enumerate(starts):
            dur = float(durations[idx])
            start_samp = max(0, int(start_time * sample_rate))
            end_samp = min(total_samples, int(start_samp + dur * sample_rate))

            seg = waveform[..., start_samp:end_samp]

            if use_padding:
                cur_len = seg.shape[-1]
                if cur_len < target_len:
                    pad = target_len - cur_len
                    silence = torch.zeros(
                        (seg.shape[0], seg.shape[1], pad),
                        dtype=seg.dtype,
                        device=seg.device
                    )
                    seg = torch.cat((seg, silence), dim=-1)

            segments.append({"waveform": seg, "sample_rate": sample_rate})

        # ✅ total_duration = sum of requested durations
        requested_total = float(sum(durations)) if durations else 0.0

        meta = {
            "scene_count": scene_count,
            "durations": durations,
            "offset_seconds": float(offset_seconds),
            "starts": starts,
            "sample_rate": sample_rate,
            "internal_chunk_duration": internal_chunk_duration,
            "source_audio_duration": source_audio_duration,  # reference only
            "outputs_count": len(segments),
            "used_padding": use_padding,
        }

        return (meta, requested_total, *tuple(segments))


from transformers import WhisperProcessor, WhisperForConditionalGeneration

class VRGDG_TranscribeLyric:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input for transcription."}),
                "language": (
                    [
                        "auto", "english", "chinese", "german", "spanish", "russian", "korean", "french",
                        "japanese", "portuguese", "turkish", "polish", "catalan", "dutch", "arabic", "swedish",
                        "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew", "ukrainian", "greek",
                        "malay", "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", "thai", "urdu",
                        "croatian", "bulgarian", "lithuanian", "latin", "maori", "malayalam", "welsh", "slovak",
                        "telugu", "persian", "latvian", "bengali", "serbian", "azerbaijani", "slovenian", "kannada",
                        "estonian", "macedonian", "breton", "basque", "icelandic", "armenian", "nepali", "mongolian",
                        "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi", "punjabi", "sinhala",
                        "khmer", "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", "belarusian",
                        "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese", "haitian creole",
                        "pashto", "turkmen", "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan",
                        "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", "hausa", "bashkir",
                        "javanese", "sundanese", "cantonese", "burmese", "valencian", "flemish", "haitian",
                        "letzeburgesch", "pushto", "panjabi", "moldavian", "moldovan", "sinhalese", "castilian", "mandarin"
                    ],
                    {"default": "auto", "tooltip": "Language to transcribe. 'auto' lets Whisper detect it."}
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "transcribe"
    CATEGORY = "WanVideoWrapper"

    def transcribe(self, audio, language):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        waveform = audio["waveform"][0]
        sample_rate = audio["sample_rate"]
        target_sr = 16000

        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.squeeze()

        model_name = "openai/whisper-large-v3"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()

        max_length = target_sr * 30  # 30 seconds = 480000 samples
        total_len = waveform.size(0)
        chunks = [waveform[i:i + max_length] for i in range(0, total_len, max_length)]

        transcriptions = []

        for chunk in chunks:
            # Pad short chunks if using auto language detection
            if language == "auto" and chunk.size(0) < max_length:
                pad_len = max_length - chunk.size(0)
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            inputs = processor(
                chunk,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding="longest",
                truncation=False
            )
            input_features = inputs["input_features"].to(model.device)

            if language == "auto":
                generated_ids = model.generate(input_features)
            else:
                decoder_ids = processor.get_decoder_prompt_ids(language=language)
                generated_ids = model.generate(input_features, forced_decoder_ids=decoder_ids)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            transcriptions.append(text.strip())

        full_transcription = " ".join(transcriptions).strip()
        return (full_transcription,)

#########################################




import random
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class VRGDG_LoadAudioSplit_HUMO_Transcribe:
    RETURN_TYPES = ("DICT", "FLOAT", "STRING") + tuple(["AUDIO"] * 50)
    RETURN_NAMES = ("meta", "total_duration", "lyrics_string") + tuple([f"audio_{i}" for i in range(1, 51)])

    FUNCTION = "split_audio"
    CATEGORY = "VRGDG"

    fallback_words = ["standing", "sitting", "laying", "resting", "waiting","walking","dancing","looking","thinking"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "scene_count": ("INT", {"default": 1, "min": 1, "max": 50, "dynamic": True}),
                "language": (
                    [
                        "auto", "english", "chinese", "german", "spanish", "russian", "korean", "french",
                        "japanese", "portuguese", "turkish", "polish", "catalan", "dutch", "arabic", "swedish",
                        "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew", "ukrainian", "greek",
                        "malay", "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", "thai", "urdu",
                        "croatian", "bulgarian", "lithuanian", "latin", "maori", "malayalam", "welsh", "slovak",
                        "telugu", "persian", "latvian", "bengali", "serbian", "azerbaijani", "slovenian", "kannada",
                        "estonian", "macedonian", "breton", "basque", "icelandic", "armenian", "nepali", "mongolian",
                        "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi", "punjabi", "sinhala",
                        "khmer", "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", "belarusian",
                        "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese", "haitian creole",
                        "pashto", "turkmen", "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan",
                        "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", "hausa", "bashkir",
                        "javanese", "sundanese", "cantonese", "burmese", "valencian", "flemish", "haitian",
                        "letzeburgesch", "pushto", "panjabi", "moldavian", "moldovan", "sinhalese", "castilian", "mandarin"
                    ],
                    {"default": "english", "tooltip": "Language for Whisper transcription."}
                ),
                "enable_lyrics": ("BOOLEAN", {"default": False, "tooltip": "If false, skip transcription."}),
            }
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ("DICT", "FLOAT", "STRING") + tuple(["AUDIO"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ["meta", "total_duration", "lyrics_string"] + [f"audio_{i+1}" for i in range(count)]

    def split_audio(self, audio, offset_seconds, scene_count=1, language="english", enable_lyrics=True):
        internal_chunk_duration = 8.0
        gain_db = 0.0
        make_stereo = True

        waveform = audio["waveform"]
        src_sample_rate = int(audio.get("sample_rate", 44100))

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        audio_total_duration = float(total_samples) / float(src_sample_rate)

        scene_count = max(1, int(scene_count))
        durations = [3.88] * scene_count
        starts = []
        current_time = float(offset_seconds)
        for d in durations:
            starts.append(current_time)
            current_time += float(d)

        segments = []
        transcriptions = []

        if enable_lyrics:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device).eval()
        else:
            processor = None
            model = None
            device = None

        for idx, start_time in enumerate(starts):
            requested_duration = float(durations[idx])
            start_samp = max(0, int(start_time * src_sample_rate))
            end_samp = min(total_samples, int(start_samp + requested_duration * src_sample_rate))

            seg = waveform[..., start_samp:end_samp]

            if seg.numel() == 0:
                print(f"⚠️ Scene {idx} is empty, inserting silence + fallback lyric.")
                seg = torch.zeros((1, 2, int(requested_duration * src_sample_rate)))
                segments.append({"waveform": seg, "sample_rate": src_sample_rate})
                transcriptions.append(random.choice(self.fallback_words))
                continue

            if gain_db != 0.0:
                gain_scalar = 10 ** (gain_db / 20)
                seg *= gain_scalar

            if make_stereo and seg.shape[1] == 1:
                seg = seg.repeat(1, 2, 1)

            expected_len = int(requested_duration * src_sample_rate)
            if seg.shape[-1] < expected_len:
                pad_len = expected_len - seg.shape[-1]
                seg = torch.nn.functional.pad(seg, (0, pad_len))
            elif seg.shape[-1] > expected_len:
                seg = seg[..., :expected_len]

            segments.append({"waveform": seg, "sample_rate": src_sample_rate})

            if not enable_lyrics:
                transcriptions.append("")
                continue

            flat_seg = seg.mean(dim=1).squeeze()

            if src_sample_rate != 16000:
                flat_seg = torchaudio.functional.resample(flat_seg, src_sample_rate, 16000)

            if language == "auto" and flat_seg.size(0) < 480000:
                pad_len = 480000 - flat_seg.size(0)
                flat_seg = torch.nn.functional.pad(flat_seg, (0, pad_len))

            inputs = processor(
                flat_seg,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest",
                truncation=False
            )

            input_features = inputs["input_features"].to(device)

            try:
                if language == "auto":
                    generated_ids = model.generate(input_features)
                else:
                    decoder_ids = processor.get_decoder_prompt_ids(language=language)
                    generated_ids = model.generate(input_features, forced_decoder_ids=decoder_ids)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if not text:
                    text = random.choice(self.fallback_words)
            except Exception:
                text = random.choice(self.fallback_words)

            transcriptions.append(text)

        def count_words(line):
            return len(re.findall(r'\w+', line))

        def collapse_repeats(line):
            tokens = line.split()
            result = []
            last_word = None
            repeat_count = 0
            for word in tokens:
                if word.lower() == last_word:
                    repeat_count += 1
                else:
                    last_word = word.lower()
                    repeat_count = 1
                if repeat_count <= 3:
                    result.append(word)
            cleaned = []
            prev = None
            for word in result:
                if word.lower() == prev:
                    continue
                cleaned.append(word)
                prev = word.lower()
            return " ".join(cleaned)

        lyrics_text = ""
        if enable_lyrics:
            safe_transcriptions = [t if t else random.choice(self.fallback_words) for t in transcriptions]
            enriched_transcriptions = []

            for i in range(len(safe_transcriptions)):
                pieces = []

                # Add previous if exists
                if i > 0:
                    pieces.append(safe_transcriptions[i - 1].strip())

                # Add current
                pieces.append(safe_transcriptions[i].strip())

                combined = " ".join(pieces).strip()
                word_count = count_words(combined)

                # If under 4 words, try appending next ones
                j = i + 1
                while word_count < 4 and j < len(safe_transcriptions):
                    combined += " " + safe_transcriptions[j].strip()
                    word_count = count_words(combined)
                    j += 1

                # If still under 4, prepend fallback
                if word_count < 4:
                    combined = random.choice(self.fallback_words) + " " + combined

                enriched_transcriptions.append(collapse_repeats(combined.strip()))

            lyrics_text = " | ".join(enriched_transcriptions)

        meta = {
            "scene_count": scene_count,
            "durations": durations,
            "offset_seconds": float(offset_seconds),
            "starts": starts,
            "sample_rate": int(src_sample_rate),
            "internal_chunk_duration": float(internal_chunk_duration),
            "audio_total_duration": float(audio_total_duration),
            "outputs_count": len(segments),
            "used_padding": False,
        }

        return (meta, float(audio_total_duration), lyrics_text, *tuple(segments))




##################################################
import imageio
import numpy as np

class VRGDG_LoadVideos:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load_videos"
    CATEGORY = "Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {}),
                "video_folder": ("STRING", {"default": "./videos", "multiline": False}),
                "scene_count": ("INT", {"default": 3, "min": 1, "max": 5}),
            }
        }

    def load_videos(self, trigger, video_folder, scene_count=3):
        # collect up to scene_count videos in order
        videos = sorted(
            [f for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        )
        if not videos:
            raise ValueError(f"No video files found in {video_folder}")

        selected = videos[:scene_count]

        all_frames = []
        for vid in selected:
            path = os.path.join(video_folder, vid)
            reader = imageio.get_reader(path, "ffmpeg")
            frames = []
            for frame in reader:
                # imageio already gives RGB, so just normalize
                frame = frame.astype(np.float32) / 255.0
                tensor = torch.from_numpy(frame).unsqueeze(0)  # (1,H,W,C)
                frames.append(tensor)
            reader.close()

            if not frames:
                continue

            video_tensor = torch.cat(frames, dim=0)  # (F,H,W,C)
            all_frames.append(video_tensor)

        if not all_frames:
            raise ValueError("No frames loaded from any videos.")

        # concatenate all videos along frame dimension
        final_video = torch.cat(all_frames, dim=0)

        return (final_video,)


##################################################
class VRGDG_IndexedPromptChunker:
    RETURN_TYPES = tuple(["STRING"] * 50)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 51)])
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "scene_count": ("INT", {"default": 16, "min": 1, "max": 50}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999}),
                "total_sets": ("INT", {"default": 1, "min": 1, "max": 999}),
            }
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 16))
        return tuple(["STRING"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 16))
        return [f"text_output_{i+1}" for i in range(count)]

    def split_prompt(self, prompt_text, scene_count=16, index=0, total_sets=1, **kwargs):
        scene_count = max(1, min(50, scene_count))

        parts = [p.strip() for p in prompt_text.strip().split("|") if p.strip()]
        chunk_start = index * 16
        chunk_end = chunk_start + scene_count

        if index >= total_sets:
            return tuple([""] * scene_count)

        outputs = [parts[i] if i < len(parts) else "" for i in range(chunk_start, chunk_end)]
        return tuple(outputs)
    


######################################
import re

class VRGDG_IndexedPromptChunkerV2:
    RETURN_TYPES = tuple(["STRING"] * 50)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 51)])
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "scene_count": ("INT", {"default": 16, "min": 1, "max": 50}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999}),
                "total_sets": ("INT", {"default": 1, "min": 1, "max": 999}),
                "any": ("*",),  # ✅ NEW input to force execution early
            }
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 16))
        return tuple(["STRING"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 16))
        return [f"text_output_{i+1}" for i in range(count)]

    def split_prompt(self, prompt_text, scene_count=16, index=0, total_sets=1, **kwargs):
        scene_count = max(1, min(50, scene_count))

        # Extract prompts from format: prompt x: "..." |
        parts = re.findall(r'"(.*?)"', prompt_text, re.DOTALL)
        total_prompts = len(parts)

        chunk_start = index * scene_count
        chunk_end = chunk_start + scene_count

        # Runtime logs
        print(f"🔢 [PromptChunkerV2] index = {index}, scene_count = {scene_count}")
        print(f"🔢 [PromptChunkerV2] chunk_start: {chunk_start}, chunk_end: {chunk_end}")
        print(f"🔢 [PromptChunkerV2] total prompts found: {total_prompts}")

        if total_prompts < chunk_end:
            raise ValueError(
                f"[PromptChunkerV2] ❌ Not enough prompts for index={index} with scene_count={scene_count}. "
                f"Needed prompts up to {chunk_end}, but only {total_prompts} provided."
            )

        outputs = parts[chunk_start:chunk_end]

        for i, out in enumerate(outputs):
            preview = out.strip()[:80] if out.strip() else "[EMPTY]"
            print(f"🔹 Prompt {chunk_start + i}: {preview}...")

        return tuple(outputs)

#########################################
import json
class VRGDG_PostRunIndexStepper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*",),
                "trigger": ("VHS_FILENAMES", {}),
                "reset": ("BOOLEAN", {"default": False}),
                "increment": ("BOOLEAN", {"default": True}),
                "state_file_name": ("STRING", {"default": "vrgdg_index_state.json"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "ANY")
    RETURN_NAMES = ("index", "next_index", "trigger")
    FUNCTION = "run_step"
    CATEGORY = "VRGDG/utils"

    def run_step(self, trigger, reset, increment, state_file_name,any):
        # Step 1: Path setup
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_path = os.getcwd()
        state_path = os.path.join(base_path, state_file_name)
        print(f"[DEBUG-Post] Using path: {state_path}")

        # Step 2: Load existing index
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                index = int(state.get("index", 0))
            print(f"[DEBUG-Post] Loaded index: {index}")
        else:
            index = 0
            print(f"[DEBUG-Post] File not found. Starting at index 0.")

        current_index = index

        # Step 3: Update index
        if reset:
            index = 0
            print(f"[DEBUG-Post] Resetting index to 0.")
        elif increment:
            index += 1
            print(f"[DEBUG-Post] Incrementing index to {index}")

        # Step 4: Save updated index
        try:
            with open(state_path, "w") as f:
                json.dump({"index": index}, f)
            print(f"[DEBUG-Post] Saved index: {index}")
        except Exception as e:
            print(f"[DEBUG-Post] ❌ Failed to write file: {e}")

        return (current_index if not reset else 0, index, trigger)



#########################
import json
class VRGDG_GetRunIndexFromJson:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*",),  # ✅ forces execution each run
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)

    FUNCTION = "get_index"
    CATEGORY = "VRGDG"
    

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True  # 🔁 Always refresh

    def get_index(self, any):
        index = 0  # Default

        try:
            try:
                base_path = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                base_path = os.getcwd()

            json_path = os.path.join(base_path, "vrgdg_index_state.json")
            print(f"[DEBUG] Checking path: {json_path}")

            if not os.path.exists(json_path):
                print("[DEBUG] ❌ JSON file not found.")
                print(f"[DEBUG] 🚨 RECEIVED fallback index = {index}")
                return (index,)

            with open(json_path, "r") as f:
                data = json.load(f)

            print(f"[DEBUG] Raw JSON content: {data}")
            index = int(data.get("index", 0))

        except Exception as e:
            print(f"[DEBUG] ❌ Exception reading JSON: {e}")

        print(f"[DEBUG] 🚨 RECEIVED index = {index}")
        return (index,)

##################################################
class VRGDG_AudioCropTime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "step": 0.01,
                    "precision": 3
                }),
                "end_time": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.01,
                    "step": 0.01,
                    "precision": 3
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "crop_audio"
    CATEGORY = "audio"

    def crop_audio(self, audio, start_time, end_time):
        # Inspect and unpack
        if isinstance(audio, dict):
            sample_rate = float(audio["sample_rate"])
            samples = audio["waveform"]
        else:
            sample_rate, samples = audio
            sample_rate = float(sample_rate)

        print(f"sample_rate: {sample_rate} ({type(sample_rate)})")
        print(f"start_time: {start_time}, end_time: {end_time}")

        def to_float(value):
            if isinstance(value, (tuple, list)):
                value = value[0]
            return float(value)

        start_time = to_float(start_time)
        end_time = to_float(end_time)

        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        cropped_samples = samples[start_sample:end_sample]
        return ({"sample_rate": sample_rate, "waveform": cropped_samples},)




######################old version added back
class VRGDG_LoadAudioSplit_HUMO_Transcribe:
    RETURN_TYPES = ("DICT", "FLOAT", "STRING") + tuple(["AUDIO"] * 50)
    RETURN_NAMES = ("meta", "total_duration", "lyrics_string") + tuple([f"audio_{i}" for i in range(1, 51)])

    FUNCTION = "split_audio"
    CATEGORY = "VRGDG"

    fallback_words = ["standing", "sitting", "laying", "resting", "waiting"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "scene_count": ("INT", {"default": 1, "min": 1, "max": 50, "dynamic": True}),
                "language": (
                    [
                        "auto", "english", "chinese", "german", "spanish", "russian", "korean", "french",
                        "japanese", "portuguese", "turkish", "polish", "catalan", "dutch", "arabic", "swedish",
                        "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew", "ukrainian", "greek",
                        "malay", "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", "thai", "urdu",
                        "croatian", "bulgarian", "lithuanian", "latin", "maori", "malayalam", "welsh", "slovak",
                        "telugu", "persian", "latvian", "bengali", "serbian", "azerbaijani", "slovenian", "kannada",
                        "estonian", "macedonian", "breton", "basque", "icelandic", "armenian", "nepali", "mongolian",
                        "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi", "punjabi", "sinhala",
                        "khmer", "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", "belarusian",
                        "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese", "haitian creole",
                        "pashto", "turkmen", "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan",
                        "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", "hausa", "bashkir",
                        "javanese", "sundanese", "cantonese", "burmese", "valencian", "flemish", "haitian",
                        "letzeburgesch", "pushto", "panjabi", "moldavian", "moldovan", "sinhalese", "castilian", "mandarin"
                    ],
                    {"default": "english", "tooltip": "Language for Whisper transcription."}
                ),
                "enable_lyrics": ("BOOLEAN", {"default": False, "tooltip": "If false, skip transcription."}),
            }
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ("DICT", "FLOAT", "STRING") + tuple(["AUDIO"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ["meta", "total_duration", "lyrics_string"] + [f"audio_{i+1}" for i in range(count)]

    def split_audio(self, audio, offset_seconds, scene_count=1, language="english", enable_lyrics=True):
        internal_chunk_duration = 8.0
        gain_db = 0.0
        make_stereo = True

        waveform = audio["waveform"]
        src_sample_rate = int(audio.get("sample_rate", 44100))

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        audio_total_duration = float(total_samples) / float(src_sample_rate)

        scene_count = max(1, int(scene_count))
        durations = [3.88] * scene_count
        starts = []
        current_time = float(offset_seconds)
        for d in durations:
            starts.append(current_time)
            current_time += float(d)

        segments = []
        transcriptions = []

        if enable_lyrics:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device).eval()
        else:
            processor = None
            model = None
            device = None

        for idx, start_time in enumerate(starts):
            requested_duration = float(durations[idx])
            start_samp = max(0, int(start_time * src_sample_rate))
            end_samp = min(total_samples, int(start_samp + requested_duration * src_sample_rate))

            seg = waveform[..., start_samp:end_samp]

            # If empty, insert silence of correct duration
            if seg.numel() == 0:
                print(f"⚠️ Scene {idx} is empty, inserting silence + fallback lyric.")
                seg = torch.zeros((1, 2, int(requested_duration * src_sample_rate)))  # stereo silence
                segments.append({"waveform": seg, "sample_rate": src_sample_rate})
                transcriptions.append(random.choice(self.fallback_words))
                continue

            if gain_db != 0.0:
                gain_scalar = 10 ** (gain_db / 20)
                seg *= gain_scalar

            if make_stereo and seg.shape[1] == 1:
                seg = seg.repeat(1, 2, 1)

            # Ensure correct length by padding/truncating
            expected_len = int(requested_duration * src_sample_rate)
            if seg.shape[-1] < expected_len:
                pad_len = expected_len - seg.shape[-1]
                seg = torch.nn.functional.pad(seg, (0, pad_len))
            elif seg.shape[-1] > expected_len:
                seg = seg[..., :expected_len]

            segments.append({"waveform": seg, "sample_rate": src_sample_rate})

            if not enable_lyrics:
                transcriptions.append("")
                continue

            flat_seg = seg.mean(dim=1).squeeze()

            if src_sample_rate != 16000:
                flat_seg = torchaudio.functional.resample(flat_seg, src_sample_rate, 16000)

            if language == "auto" and flat_seg.size(0) < 480000:
                pad_len = 480000 - flat_seg.size(0)
                flat_seg = torch.nn.functional.pad(flat_seg, (0, pad_len))

            inputs = processor(
                flat_seg,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest",
                truncation=False
            )

            input_features = inputs["input_features"].to(device)

            try:
                if language == "auto":
                    generated_ids = model.generate(input_features)
                else:
                    decoder_ids = processor.get_decoder_prompt_ids(language=language)
                    generated_ids = model.generate(input_features, forced_decoder_ids=decoder_ids)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if not text:
                    text = random.choice(self.fallback_words)
            except Exception:
                text = random.choice(self.fallback_words)

            transcriptions.append(text)

        def count_words(line):
            return len(re.findall(r'\w+', line))

        def collapse_repeats(line):
            tokens = line.split()
            result = []
            last_word = None
            repeat_count = 0
            for word in tokens:
                if word.lower() == last_word:
                    repeat_count += 1
                else:
                    last_word = word.lower()
                    repeat_count = 1
                if repeat_count <= 3:
                    result.append(word)
            cleaned = []
            prev = None
            for word in result:
                if word.lower() == prev:
                    continue
                cleaned.append(word)
                prev = word.lower()
            return " ".join(cleaned)

        lyrics_text = ""
        if enable_lyrics:
            safe_transcriptions = [t if t else random.choice(self.fallback_words) for t in transcriptions]
            enriched_transcriptions = []
            for i, text in enumerate(safe_transcriptions):
                word_count = count_words(text)
                if word_count >= 3:
                    enriched_transcriptions.append(collapse_repeats(text))
                    continue
                combined = random.choice(self.fallback_words) + " " + text.strip()
                j = i + 1
                while word_count < 3 and j < len(safe_transcriptions):
                    combined += " " + safe_transcriptions[j].strip()
                    word_count = count_words(combined)
                    j += 1
                if word_count < 3 and i > 0:
                    combined = safe_transcriptions[i - 1].strip() + " " + combined
                enriched_transcriptions.append(collapse_repeats(combined.strip()))
            lyrics_text = " | ".join(enriched_transcriptions)

        meta = {
            "scene_count": scene_count,
            "durations": durations,
            "offset_seconds": float(offset_seconds),
            "starts": starts,
            "sample_rate": int(src_sample_rate),
            "internal_chunk_duration": float(internal_chunk_duration),
            "audio_total_duration": float(audio_total_duration),
            "outputs_count": len(segments),
            "used_padding": False,
        }

        return (meta, float(audio_total_duration), lyrics_text, *tuple(segments))


NODE_CLASS_MAPPINGS = {
     "FastFilmGrain": FastFilmGrain,
     "ColorMatchToReference": ColorMatchToReference,
     "FastUnsharpSharpen": FastUnsharpSharpen,
     "FastLaplacianSharpen": FastLaplacianSharpen,
     "FastSobelSharpen": FastSobelSharpen,
     "VRGDG_LoadAudioSplitDynamic": VRGDG_LoadAudioSplitDynamic,
     "VRGDG_LoadAudioSplit_HUMO": VRGDG_LoadAudioSplit_HUMO,
     "VRGDG_Extract_Frame_Number": VRGDG_Extract_Frame_Number,
     "VRGDG_VideoSplitter":VRGDG_VideoSplitter,
     "VRGDG_LoadAudioSplitUpload":VRGDG_LoadAudioSplitUpload,
     "VRGDG_TranscribeText":VRGDG_TranscribeLyric,
     "VRGDG_LoadAudioSplit_HUMO_Transcribe":VRGDG_LoadAudioSplit_HUMO_Transcribe,
     "VRGDG_LoadVideos":VRGDG_LoadVideos,
     "VRGDG_IndexedPromptChunker":VRGDG_IndexedPromptChunker,
     "VRGDG_IndexedPromptChunkerV2":VRGDG_IndexedPromptChunkerV2,
     "VRGDG_PostRunIndexStepper":VRGDG_PostRunIndexStepper,
     "VRGDG_GetRunIndexFromJson":VRGDG_GetRunIndexFromJson,
     "VRGDG_AudioCropTime":VRGDG_AudioCropTime,
     "VRGDG_LoadAudioSplit_HUMO_Transcribe":VRGDG_LoadAudioSplit_HUMO_Transcribe




}

NODE_DISPLAY_NAME_MAPPINGS = {
     "FastFilmGrain": "🎞️ Fast Film Grain",
     "ColorMatchToReference": "🎨 Color Match To Reference",
     "FastUnsharpSharpen": "🎯 Fast Unsharp Sharpen",
     "FastLaplacianSharpen": "🌀 Fast Laplacian Sharpen",
     "FastSobelSharpen": "📏 Fast Sobel Sharpen",
     "VRGDG_LoadAudioSplitDynamic":"VRGDG_LoadAudioSplitDynamic",
     "VRGDG_LoadAudioSplit_HUMO":"VRGDG_LoadAudioSplit_HUMO",
     "VRGDG_Extract_Frame_Number":"🎞️ VRGDG_Extract_Frame_Number",
     "VRGDG_VideoSplitter":"VRGDG_VideoSplitter",
     "VRGDG_LoadAudioSplitUpload":"VRGDG_LoadAudioSplitUpload",
     "VRGDG_PromptSplitter":"VRGDG_PromptSplitter",
     "VRGDG_TranscribeText":"VRGDG_TranscribeLyric",
     "VRGDG_LoadAudioSplit_HUMO_Transcribe":"VRGDG_LoadAudioSplit_HUMO_Transcribe",
     "VRGDG_LoadVideos":"VRGDG_LoadVideos",
     "VRGDG_IndexedPromptChunker":"VRGDG_IndexedPromptChunker",
     "VRGDG_IndexedPromptChunkerV2":"VRGDG_IndexedPromptChunkerV2",
     "VRGDG_PostRunIndexStepper":"VRGDG_PostRunIndexStepper",
     "VRGDG_GetRunIndexFromJson":"VRGDG_GetRunIndexFromJson",
     "VRGDG_AudioCropTime":"VRGDG_AudioCropTime",
     "VRGDG_LoadAudioSplit_HUMO_Transcribe":"VRGDG_LoadAudioSplit_HUMO_Transcribe"

  
   
 

}


print(r"""
__     ______   ____                      ____              ____ _      _ 
\ \   / /  _ \ / ___| __ _ _ __ ___   ___|  _ \  _____   __/ ___(_)_ __| |
 \ \ / /| |_) | |  _ / _` | '_ ` _ \ / _ \ | | |/ _ \ \ / / |  _| | '__| |
  \ V / |  _ <| |_| | (_| | | | | | |  __/ |_| |  __/\ V /| |_| | | |  | |
   \_/  |_| \_\\____|\__,_|_| |_| |_|\___|____/ \___| \_/  \____|_|_|  |_|
                                                                          
             🎮 VRGameDevGirl custom nodes loaded successfully! 🎞️
""")
