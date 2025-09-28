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
                    "FLOAT", {"default": 0.04, "min": 0.01, "max": 1.0, "step": 0.01}
                ),
                "saturation_mix": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "video/enhancement"
    DESCRIPTION = "Adds lightweight film grain"

    def apply_grain(self, images, grain_intensity, saturation_mix):
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        # Generate random noise same size as image
        grain = torch.randn_like(images)

        # Tint channels slightly (like LTX)
        grain[:, :, :, 0] *= 2.0  # red channel
        grain[:, :, :, 2] *= 3.0  # blue channel

        # Blend grayscale noise with color noise
        gray = grain[:, :, :, 1].unsqueeze(3).repeat(1, 1, 1, 3)
        grain = saturation_mix * grain + (1.0 - saturation_mix) * gray

        # Apply grain to image
        output = images + grain * grain_intensity
        output = output.clamp(0.0, 1.0)

        # Return to CPU/mid-device for downstream compatibility
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "video/enhancement"
    DESCRIPTION = "Matches the color tone of input image to a reference image using LAB mean/std alignment"

    def match_color(self, images, reference_image, match_strength):
        device = comfy.model_management.get_torch_device()

        images = images.to(device)
        reference_image = reference_image.to(device)

        # Convert shape: [B, H, W, C] -> [B, C, H, W]
        images = images.permute(0, 3, 1, 2)
        reference_image = reference_image.permute(0, 3, 1, 2)

        # Convert to LAB color space
        img_lab = kornia.color.rgb_to_lab(images)
        ref_lab = kornia.color.rgb_to_lab(reference_image)

        # Compute channel-wise mean and std
        img_mean = img_lab.mean(dim=[2, 3], keepdim=True)
        img_std = img_lab.std(dim=[2, 3], keepdim=True) + 1e-5

        ref_mean = ref_lab.mean(dim=[2, 3], keepdim=True)
        ref_std = ref_lab.std(dim=[2, 3], keepdim=True)

        # Normalize, match stats, blend
        matched_lab = (img_lab - img_mean) / img_std * ref_std + ref_mean
        blended_lab = match_strength * matched_lab + (1.0 - match_strength) * img_lab

        # Convert back to RGB and return
        output = kornia.color.lab_to_rgb(blended_lab)
        output = output.clamp(0.0, 1.0)
        output = output.permute(0, 2, 3, 1)  # Back to [B, H, W, C]
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

# VRGDG_CombinevideosV2 (updated)
# - Optional DICT input: audio_meta (from VRGDG_LoadAudioSplitDynamic)
#   * If provided and contains durations/scene_count, they override local widgets.
# - Keeps existing behavior when meta is not connected.
# - Trims or pads to target per-scene frame counts computed from duration_i * fps.
# - Requires at least two videos overall.

class VRGDG_CombinevideosV2:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_video_frames",)
    FUNCTION = "blend_videos"
    CATEGORY = "Video"

    @classmethod
    def INPUT_TYPES(cls):
        # up to 50 video inputs + duration controls (0.00 => use current frames)
        opt_videos = {f"video_{i}": ("IMAGE",) for i in range(1, 51)}
        opt_durations = {
            f"duration_{i}": (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "step": 0.01, "round": 0.01}
            )
            for i in range(1, 51)
        }
        return {
            "required": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0}),
                "scene_count": ("INT", {"default": 2, "min": 2, "max": 50}),
            },
            "optional": {
                # when True, repeats the last frame to reach target length
              #  "pad_short_videos": ("BOOL", {"default": True}),
                # optional meta from VRGDG_LoadAudioSplitDynamic
                "audio_meta": ("DICT",),
                **opt_videos,
                **opt_durations,
            },
        }

    # --- helpers -------------------------------------------------------------

    def _target_frames_for_index(self, durations, idx_zero_based, fps, current_frames):
        """
        durations[idx] seconds * fps -> target frames.
        If duration <= 0, default to current_frames for that clip.
        """
        try:
            dur_sec = float(durations[idx_zero_based])
        except Exception:
            dur_sec = 0.0
        if dur_sec > 0.0:
            tgt = int(round(dur_sec * float(fps)))
            return max(1, tgt)
        return int(current_frames)

    def _trim_or_pad(self, video, target_frames, pad_short=False):
        if video is None:
            return None
        if video.ndim != 4:
            raise ValueError(f"Expected video tensor with 4 dims (frames,H,W,C), got {tuple(video.shape)}")
        cur = int(video.shape[0])
        if cur > target_frames:
            return video[:target_frames]
        if cur < target_frames and pad_short:
            need = target_frames - cur
            last = video[-1:].clone()
            pad = last.repeat(need, 1, 1, 1)
            return torch.cat([video, pad], dim=0)
        return video

    # --- main op -------------------------------------------------------------

    def blend_videos(self, fps, scene_count=2, **kwargs):
        #pad_short_videos = bool(kwargs.get("pad_short_videos", False))
        pad_short_videos = True

        local_scene_count = max(2, int(scene_count))

        # Prefer durations/scene_count from audio_meta when available
        meta = kwargs.get("audio_meta", None)
        use_meta = isinstance(meta, dict) and (
            ("durations" in meta and isinstance(meta["durations"], (list, tuple)))
            or ("scene_count" in meta)
        )

        if use_meta:
            meta_durations = list(meta.get("durations", []))
            # If meta has explicit scene_count, trust it; else infer from durations
            meta_scene_count = int(meta.get("scene_count", len(meta_durations) or local_scene_count))
            effective_scene_count = max(2, min(50, int(meta_scene_count)))
            # Ensure durations length matches count (pad with 0.0 => use native length)
            if len(meta_durations) < effective_scene_count:
                meta_durations = meta_durations + [0.0] * (effective_scene_count - len(meta_durations))
            else:
                meta_durations = meta_durations[:effective_scene_count]
            durations = meta_durations
        else:
            effective_scene_count = max(2, min(50, local_scene_count))
            # Gather per-scene durations from local widgets
            durations = []
            for i in range(effective_scene_count):
                k = f"duration_{i+1}"
                v = kwargs.get(k, 0.0)
                try:
                    durations.append(float(v))
                except Exception:
                    durations.append(0.0)

        # Collect videos in order video_1..video_N (up to effective_scene_count)
        vids = []
        for i in range(1, effective_scene_count + 1):
            v = kwargs.get(f"video_{i}")
            if v is not None:
                vids.append((i, v))

        if len(vids) < 2:
            raise ValueError("Provide at least two videos (e.g., video_1 and video_2).")

        trimmed = []
        for slot_idx, vid in vids:
            if vid.ndim != 4:
                raise ValueError(f"video_{slot_idx} must have shape (frames,H,W,C), got {tuple(vid.shape)}")
            tgt = self._target_frames_for_index(durations, slot_idx - 1, fps, vid.shape[0])
            trimmed.append(self._trim_or_pad(vid, tgt, pad_short=pad_short_videos))

        # Concatenate along frame dimension
        final = torch.cat([t.to(dtype=torch.float32) for t in trimmed], dim=0).cpu()
        return (final,)



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



class VRGDG_PromptSplitter:
    RETURN_TYPES = tuple(["STRING"] * 50)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 51)])
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "scene_count": ("INT", {"default": 2, "min": 1, "max": 50}),
            }
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 2))
        count = max(1, min(50, count))
        return tuple(["STRING"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 2))
        count = max(1, min(50, count))
        return [f"text_output_{i+1}" for i in range(count)]

    def split_prompt(self, prompt_text, scene_count=2, **kwargs):
        scene_count = max(1, min(50, scene_count))
        parts = [p.strip() for p in prompt_text.strip().split("|") if p.strip()]
        outputs = [parts[i] if i < len(parts) else "" for i in range(scene_count)]
        return tuple(outputs)


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




################################################
class VRGDG_TimecodeFromIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0}),
                "duration_seconds": ("FLOAT", {"default": 62.0, "min": 0.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("start_time",)

    FUNCTION = "format_timecode"

    CATEGORY = "utils"

    def format_timecode(self, index, duration_seconds):
        start_seconds = int(index * duration_seconds)
        start_time_str = f"{start_seconds // 60}:{start_seconds % 60:02d}"
        return (start_time_str,)




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


#############################################
import imageio
import numpy as np
import torch
class VRGDG_ConditionalLoadVideos:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load_videos"
    CATEGORY = "Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ✅ trigger kept as VHS_FILENAMES type
                "trigger": ("VHS_FILENAMES", {}),
                "threshold": ("INT", {"default": 3}),
                "video_folder": ("STRING", {"default": "./videos", "multiline": False}),
            }
        }

    def load_videos(self, trigger, threshold, video_folder):
        # ✅ Clean up accidental newlines/whitespace
        video_folder = video_folder.strip()

        # ✅ Ensure folder exists (create if missing)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder, exist_ok=True)
            print(f"[VRGDG_ConditionalLoadVideos] Created folder: {video_folder}")
            print(f"[VRGDG_ConditionalLoadVideos] No videos yet, skipping.")
            return (None,)

        # ✅ Collect videos
        videos = sorted(
            [f for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        )
        video_count = len(videos)

        print(f"[VRGDG_ConditionalLoadVideos] Found {video_count} videos in {video_folder}")

        # 🚫 Skip if not enough videos yet
        if video_count < threshold:
            print(f"[VRGDG_ConditionalLoadVideos] Threshold not met ({video_count}/{threshold}), skipping.")
            return (None,)

        all_frames = []
        for vid in videos:
            path = os.path.join(video_folder, vid)
            print(f"[VRGDG_ConditionalLoadVideos] Loading {path}")
            reader = imageio.get_reader(path, "ffmpeg")
            frames = []
            for frame in reader:
                # normalize to float32 RGB tensor
                frame = frame.astype(np.float32) / 255.0
                tensor = torch.from_numpy(frame).unsqueeze(0)  # (1,H,W,C)
                frames.append(tensor)
            reader.close()

            if not frames:
                print(f"[VRGDG_ConditionalLoadVideos] Skipped {vid}, no frames read.")
                continue

            video_tensor = torch.cat(frames, dim=0)  # (F,H,W,C)
            all_frames.append(video_tensor)
            print(f"[VRGDG_ConditionalLoadVideos] Loaded {vid} with {len(frames)} frames")

        if not all_frames:
            print("[VRGDG_ConditionalLoadVideos] No valid videos loaded, returning None.")
            return (None,)

        final_video = torch.cat(all_frames, dim=0)
        print(f"[VRGDG_ConditionalLoadVideos] Returning concatenated video tensor with {final_video.shape[0]} frames")
        return (final_video,)



################################################
class VRGDG_CalculateSetsFromAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT",)
    RETURN_NAMES = ("instructions", "end_time", "total_sets",)
    FUNCTION = "calculate"

    CATEGORY = "utils/audio"

    def calculate(self, audio):
        set_duration = 62.0
        group_duration = 3.88
        groups_per_set = 16

        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except Exception:
            return ("❌ Expected audio to be a dict with 'waveform' and 'sample_rate'.", "0:00", 0)

        try:
            num_samples = waveform.shape[-1]
            audio_duration = num_samples / sample_rate
        except Exception:
            return ("❌ Failed to compute audio duration from waveform.", "0:00", 0)

        # Convert duration to MM:SS
        minutes = int(audio_duration // 60)
        seconds = int(audio_duration % 60)
        end_time_str = f"{minutes}:{seconds:02d}"

        # Compute sets
        full_sets = int(audio_duration // set_duration)
        remainder = audio_duration - (full_sets * set_duration)

        if remainder > 0:
            total_sets = full_sets + 1
            groups_in_last_set = int(min(remainder // group_duration, groups_per_set))
        else:
            total_sets = full_sets
            groups_in_last_set = groups_per_set

        # Adjust instructions to account for the fact we've already run once
        if total_sets > 1:
            runs_remaining = max(total_sets - 2, 0)  # subtract 1 extra for the "already ran once"
            if runs_remaining > 0:
                times_word = "time" if runs_remaining == 1 else "times"
                instructions = (
                    f"Click 'RUN' {runs_remaining} more {times_word}. "
                    f"Then update the Fasts Group Muter so you only have {groups_in_last_set} groups set, "
                    f"then hit 'RUN' again."
                )
            else:
                instructions = (
                    f"Update the Fasts Group Muter so you only have {groups_in_last_set} groups set, "
                    f"then hit 'RUN' again."
                )
        else:
            instructions = (
                f"Set Fasts Group Muter to {groups_in_last_set} groups and click 'RUN'."
            )

        return (instructions, end_time_str, total_sets)


##########################################
class VRGDG_GetFilenamePrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Filename_Prefix",)
    FUNCTION = "get_prefix"

    CATEGORY = "utils/files"

    def get_prefix(self, folder_path):
        # Clean up any accidental whitespace or newline characters
        folder_path = folder_path.strip()

        # Ensure the folder exists (create it if missing)
        os.makedirs(folder_path, exist_ok=True)

        # Normalize and extract last folder name
        # This part is already cross-platform
        base_name = os.path.basename(os.path.normpath(folder_path))

        # Use os.path.join to create a cross-platform filename prefix rather than hardcoding "\\" vs "/"
        result = os.path.join(base_name, "video")
        return (result,)
########################################
class VRGDG_TriggerIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {"id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "generate"

    CATEGORY = "utils/control"

    def generate(self, seed, id=None):
        # Just return the seed, ComfyUI handles incrementing via control_after_generate
        return (seed,)





NODE_CLASS_MAPPINGS = {
     "FastFilmGrain": FastFilmGrain,
     "ColorMatchToReference": ColorMatchToReference,
     "FastUnsharpSharpen": FastUnsharpSharpen,
     "FastLaplacianSharpen": FastLaplacianSharpen,
     "FastSobelSharpen": FastSobelSharpen,
     "VRGDG_CombinevideosV2": VRGDG_CombinevideosV2,
     "VRGDG_LoadAudioSplitDynamic": VRGDG_LoadAudioSplitDynamic,
     "VRGDG_LoadAudioSplit_HUMO": VRGDG_LoadAudioSplit_HUMO,
     "VRGDG_Extract_Frame_Number": VRGDG_Extract_Frame_Number,
     "VRGDG_VideoSplitter":VRGDG_VideoSplitter,
     "VRGDG_LoadAudioSplitUpload":VRGDG_LoadAudioSplitUpload,
     "VRGDG_PromptSplitter":VRGDG_PromptSplitter,
     "VRGDG_TranscribeText":VRGDG_TranscribeLyric,
     "VRGDG_LoadAudioSplit_HUMO_Transcribe":VRGDG_LoadAudioSplit_HUMO_Transcribe,
     "VRGDG_TimecodeFromIndex":VRGDG_TimecodeFromIndex,
     "VRGDG_LoadVideos":VRGDG_LoadVideos,
     "VRGDG_ConditionalLoadVideos":VRGDG_ConditionalLoadVideos,
     "VRGDG_CalculateSetsFromAudio":VRGDG_CalculateSetsFromAudio,
     "VRGDG_GetFilenamePrefix":VRGDG_GetFilenamePrefix,
     "VRGDG_TriggerCounter":VRGDG_TriggerIndex,


}

NODE_DISPLAY_NAME_MAPPINGS = {
     "FastFilmGrain": "🎞️ Fast Film Grain",
     "ColorMatchToReference": "🎨 Color Match To Reference",
     "FastUnsharpSharpen": "🎯 Fast Unsharp Sharpen",
     "FastLaplacianSharpen": "🌀 Fast Laplacian Sharpen",
     "FastSobelSharpen": "📏 Fast Sobel Sharpen",
     "VRGDG_CombinevideosV2": "🌀 VRGDG_CombinevideosV2",
     "VRGDG_LoadAudioSplitDynamic":"VRGDG_LoadAudioSplitDynamic",
     "VRGDG_LoadAudioSplit_HUMO":"VRGDG_LoadAudioSplit_HUMO",
     "VRGDG_Extract_Frame_Number":"🎞️ VRGDG_Extract_Frame_Number",
     "VRGDG_VideoSplitter":"VRGDG_VideoSplitter",
     "VRGDG_LoadAudioSplitUpload":"VRGDG_LoadAudioSplitUpload",
     "VRGDG_PromptSplitter":"VRGDG_PromptSplitter",
     "VRGDG_TranscribeText":"VRGDG_TranscribeLyric",
     "VRGDG_LoadAudioSplit_HUMO_Transcribe":"VRGDG_LoadAudioSplit_HUMO_Transcribe",
     "VRGDG_TimecodeFromIndex":"VRGDG_TimecodeFromIndex",
     "VRGDG_LoadVideos":"VRGDG_LoadVideos",
     "VRGDG_ConditionalLoadVideos":"VRGDG_ConditionalLoadVideos",
     "VRGDG_CalculateSetsFromAudio":"VRGDG_CalculateSetsFromAudio",
     "VRGDG_GetFilenamePrefix":"VRGDG_GetFilenamePrefix",
     "VRGDG_TriggerCounter":"VRGDG_TriggerIndex",



}


print(r"""
__     ______   ____                      ____              ____ _      _
\ \   / /  _ \ / ___| __ _ _ __ ___   ___|  _ \  _____   __/ ___(_)_ __| |
 \ \ / /| |_) | |  _ / _` | '_ ` _ \ / _ \ | | |/ _ \ \ / / |  _| | '__| |
  \ V / |  _ <| |_| | (_| | | | | | |  __/ |_| |  __/\ V /| |_| | | |  | |
   \_/  |_| \_\\____|\__,_|_| |_| |_|\___|____/ \___| \_/  \____|_|_|  |_|

             🎮 VRGameDevGirl custom nodes loaded successfully! 🎞️
""")
