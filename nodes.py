import os
import math
from pathlib import Path
from typing import Tuple, Union  # ✅ You already use both

# 🔢 Torch + NumPy
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

# 📦 Third-party tools
import kornia
import librosa  # ✅ Needed for audio loading
from PIL import Image

# 📁 ComfyUI-specific
import comfy
import comfy.sample
import comfy.utils
import comfy.model_management as mm
import folder_paths
import latent_preview

# 🧠 Custom project files
from .utils import load_video, set_initial_camera, build_cameras, traj_map
from .camera import get_camera_embedding
from .lib import image

# 🌐 Optional - OpenAI (used if you really use it elsewhere)
from openai import Client as OpenAIClient

#what you just gave me

from comfy.utils import common_upscale
import gc

import torchaudio


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
    

class VRGDG_Unic3CameraPresets:
    RETURN_TYPES = ("LATENT", "INT", "IMAGE")
    RETURN_NAMES = ("render_latent", "frame_count", "preview_image")
    FUNCTION = "generate"
    CATEGORY = "VRGDG Custom"

    from PIL import Image

    @staticmethod
    def tensor_to_pil_list(video_tensor):
        pil_frames = []
        for frame in video_tensor:
            # Check if it's [H, W, C] and not [C, H, W]
            if frame.dim() == 3 and frame.shape[-1] in [1, 3, 4]:  # likely [H, W, C]
                # Convert to uint8 and PIL
                frame_np = (frame.cpu().numpy() * 255).astype("uint8")
                pil_img = Image.fromarray(frame_np)
                pil_frames.append(pil_img)
            elif frame.dim() == 3 and frame.shape[0] in [1, 3, 4]:  # [C, H, W]
                from torchvision.transforms.functional import to_pil_image
                pil_img = to_pil_image(frame)
                pil_frames.append(pil_img)
            else:
                raise ValueError(f"Unexpected frame shape for PIL conversion: {frame.shape}")
        return pil_frames



    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_motion": (
                    [
                        "crash-zoom-in",
                        "zoom-into-eye",
                        "zoom-way-out",
                        "arc-over-head",
                        "orbit",                
                        "snori-cam",
                        "fast-move-track",
                        "pose",
                        "robo-move",
                        "back-forth",
                        "custom"
                       
                    ],
                    {
                        "default": "zoom-in",
                        "tooltip": "Select one of the built-in camera motion videos or 'custom' to use your own video"
                    }
                ),
                "vae": ("WANVAE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 1}),
                "crop_position": (
                    ["center", "top", "bottom", "left", "right"],
                    {"default": "center", "tooltip": "Cropping strategy for aspect ratio mismatch"}
                ),
                "enable_vae_tiling": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Reduces memory usage but may cause seams"}
                ),
                "frame_count": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 600,
                    "tooltip": "Number of frames to use from the video (will trim or pad if needed)"
                })
            },
            "optional": {
                "video_input": ("IMAGE", {
                    "tooltip": "Optional custom video frames"
                })
            }
        }


    @staticmethod
    def crop_and_resize(frame: Image.Image, target_w, target_h, crop_position):
        #print(f"[DEBUG] Frame type: {type(frame)}, Frame: {frame}")

        src_w, src_h = frame.size
        src_aspect = src_w / src_h
        tgt_aspect = target_w / target_h

        # Step 1: crop to match aspect ratio
        if src_aspect > tgt_aspect:
            # too wide — crop width
            new_w = int(tgt_aspect * src_h)
            if crop_position == "left":
                left = 0
            elif crop_position == "right":
                left = src_w - new_w
            else:  # center
                left = (src_w - new_w) // 2
            box = (left, 0, left + new_w, src_h)
        else:
            # too tall — crop height
            new_h = int(src_w / tgt_aspect)
            if crop_position == "top":
                top = 0
            elif crop_position == "bottom":
                top = src_h - new_h
            else:  # center
                top = (src_h - new_h) // 2
            box = (0, top, src_w, top + new_h)

        cropped = frame.crop(box)
        resized = cropped.resize((target_w, target_h), Image.LANCZOS)
        return TF.to_tensor(resized)  # shape: [C, H, W]



    def generate(self, camera_motion, vae, width, height, crop_position,
                enable_vae_tiling, frame_count, video_input=None):

        # Step 1: Handle mismatched config
        if video_input is not None and camera_motion != "custom":
            raise ValueError("Custom video input is connected, but 'camera_motion' is not set to 'custom'. Please set it to 'custom' to use your input video.")

        if video_input is None and camera_motion == "custom":
            raise ValueError("Camera motion is set to 'custom' but no video input is connected.")

        # Step 2: Load video frames
        if video_input is not None:
            if video_input.dim() == 5:  # Expecting [1, C, T, H, W]
                video_input = video_input.squeeze(0).permute(1, 0, 2, 3)  # → [T, C, H, W]
            frames = self.tensor_to_pil_list(video_input)
            traj_type = "custom1"  # default motion path for input videos
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(current_dir, "Unic3", "videos")
            video_path = os.path.join(base_path, f"{camera_motion}.mp4")
            frames = load_video(video_path)
            traj_type = camera_motion

        # Step 3: Adjust to desired frame count
        if len(frames) > frame_count:
            frames = frames[:frame_count]
        elif len(frames) < frame_count:
            frames += [frames[-1]] * (frame_count - len(frames))  # pad with last frame

        frame_count = len(frames)




        video_tensor = torch.stack([
            self.crop_and_resize(frame, width, height, crop_position)
            for frame in frames
        ])  # shape: [T, C, H, W]

        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # shape: [1, C, T, H, W]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        vae.to(device)
        video_tensor = video_tensor.to(device=device, dtype=vae.dtype)

        # Final encode
        latents = vae.encode(
            video_tensor * 2.0 - 1.0,
            device=device,
            tiled=enable_vae_tiling,
            tile_size=(272 // 8, 272 // 8),
            tile_stride=(144 // 8, 128 // 8)
        )

        vae.to(offload_device)
        mm.soft_empty_cache()

        # Camera logic
        nframe = latents.shape[2]
        focal_length = 1.0
        start_elevation = 5.0
        depth_avg = 0.5
        if camera_motion == "custom":
            if video_input is None:
                raise ValueError("Camera motion is set to 'custom' but no video_input was provided.")
            frames = video_input
            traj_type = "custom1"  # motion logic = no motion
        else:
            video_path = os.path.join(base_path, f"{camera_motion}.mp4")
            frames = load_video(video_path)
            traj_type = camera_motion



        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(traj_type)

        focallength_px = focal_length * width
        K = torch.tensor([[focallength_px, 0, width / 2],
                        [0, focallength_px, height / 2],
                        [0, 0, 1]], dtype=torch.float32)
        intrinsic = K[None].repeat(nframe, 1, 1)

        w2c_0, c2w_0 = set_initial_camera(start_elevation, depth_avg)

        w2cs, c2ws, intrinsic = build_cameras(
            cam_traj, w2c_0, c2w_0, intrinsic,
            nframe=nframe,
            focal_length=focal_length,
            d_theta=d_theta, d_phi=d_phi, d_r=d_r,
            radius=depth_avg,
            x_offset=x_offset, y_offset=y_offset, z_offset=z_offset
        )

        camera_embedding = get_camera_embedding(intrinsic, w2cs, nframe, height, width, normalize=True)

        # Preview image (debug)
        if video_tensor.ndim == 5:
            preview_image = video_tensor.squeeze(0).permute(1, 2, 3, 0).contiguous().to(dtype=torch.float32).cpu()
        else:
            raise ValueError(f"Unexpected video tensor shape: {video_tensor.shape}")

        # Output for render_latent
        render_latent_dict = {"samples": latents}

        return (render_latent_dict, frame_count, preview_image)
    
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
    # provide a fixed 'meta' first, then the maximum possible AUDIO ports (50)  
    RETURN_TYPES = ("DICT", "FLOAT") + tuple(["AUDIO"] * 50)
    RETURN_NAMES = ("meta", "total_duration") + tuple([f"audio_{i}" for i in range(1, 51)])


    FUNCTION = "split_audio"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"duration_{i}": ("FLOAT", {"default": 3.0, "min": 0.0}) for i in range(1, 51)}
        return {
            "required": {
                "path": ("STRING", {"default": "./audio.mp3"}),
                "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "scene_count": ("INT", {"default": 1, "min": 1, "max": 50}),
            },
            "optional": optional,
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        # Always include the fixed 'meta' DICT, then N AUDIO ports
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        ##return ("DICT",) + tuple(["AUDIO"] * count)
        return ("DICT", "FLOAT") + tuple(["AUDIO"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        # Always include the fixed 'meta' name first
        count = int(kwargs.get("scene_count", 1))
        if count < 1:
            count = 1
        return ["meta"] + [f"audio_{i+1}" for i in range(count)]

    def split_audio(self, path, offset_seconds, scene_count=1, **kwargs):
        # import torch
        # try:
        #     import torchaudio
        # except Exception:
        #     torchaudio = None

        # internal processing params (unchanged behavior)
        internal_chunk_duration = 8.0  # seconds per segment window
        gain_db = 0.0                   # optional gain
        resample_to_hz = 0.0            # 0.0 = keep source rate
        make_stereo = True

        # --- collect durations (fallback to 3.0 per scene) ---
        scene_count = max(1, int(scene_count))
        durations = []
        for i in range(scene_count):
            v = kwargs.get(f"duration_{i+1}", 3.0)
            try:
                durations.append(float(v))
            except Exception:
                durations.append(3.0)

        # --- compute cumulative start times per scene ---
        starts = []
        current_time = float(offset_seconds)
        for d in durations:
            starts.append(current_time)
            current_time += float(d)

        # --- probe audio metadata ---
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

        # --- load each segment window ---
        for idx, start_time in enumerate(starts):
            remaining = max(0.0, audio_total_duration - float(start_time))
            load_duration = min(internal_chunk_duration, remaining if remaining > 0 else internal_chunk_duration)

            try:
                # Expecting a helper 'load_audio' provided by the environment.
                # If absent or failing, we'll fall back to silence below.
                audio = load_audio(
                    path,
                    sr=None if resample_to_hz <= 0 else resample_to_hz,
                    offset=float(start_time),
                    duration=float(load_duration),
                    make_stereo=make_stereo,
                )

                if not audio or "waveform" not in audio:
                    raise RuntimeError("Audio load failed or waveform missing.")

                # optional gain
                if gain_db != 0.0:
                    # Expecting a helper 'db_to_scalar' provided by the environment.
                    gain_scalar = db_to_scalar(gain_db)
                    audio["waveform"] *= gain_scalar

                waveform = audio["waveform"]

                # truncate to window
                if waveform.shape[-1] > target_length:
                    waveform = waveform[..., :target_length]

                # pad with silence if shorter
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
                # ensure sample_rate key exists
                if "sample_rate" not in audio:
                    audio["sample_rate"] = src_sample_rate

            except Exception as e:
                print(f"[VRGDG_LoadAudioSplitDynamic] Failed to load segment {idx+1}: {e}")
                # fallback: silence window
                dummy_waveform = torch.zeros((1, 2, target_length))
                audio = {"waveform": dummy_waveform, "sample_rate": src_sample_rate}

            segments.append(audio)

        # --- assemble metadata output (first port) ---
        meta = {
            "scene_count": scene_count,
            "durations": durations,               # per-scene durations you entered
            "offset_seconds": float(offset_seconds),
            "starts": starts,                     # cumulative start time of each scene
            "sample_rate": int(src_sample_rate),
            "internal_chunk_duration": float(internal_chunk_duration),
            "audio_total_duration": float(audio_total_duration),
            "outputs_count": len(segments),
        }

        # Return 'meta' first, then the audio segments as separate outputs
        return (meta, *tuple(segments))










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
        # up to 50 video inputs + duration controls (0.0 => use current frames)
        opt_videos = {f"video_{i}": ("IMAGE",) for i in range(1, 51)}
        opt_durations = {f"duration_{i}": ("FLOAT", {"default": 0.0, "min": 0.0}) for i in range(1, 51)}
        return {
            "required": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0}),
                "scene_count": ("INT", {"default": 2, "min": 2, "max": 50}),
            },
            "optional": {
                # when True, repeats the last frame to reach target length
                "pad_short_videos": ("BOOL", {"default": True}),
                # NEW: optional meta from VRGDG_LoadAudioSplitDynamic
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
        

        pad_short_videos = bool(kwargs.get("pad_short_videos", False))
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



NODE_CLASS_MAPPINGS = {
    "FastFilmGrain": FastFilmGrain,
     "ColorMatchToReference": ColorMatchToReference,
     "FastUnsharpSharpen": FastUnsharpSharpen,
     "FastLaplacianSharpen": FastLaplacianSharpen,
     "FastSobelSharpen": FastSobelSharpen,
     "VRGDG_Unic3CameraPresets":  VRGDG_Unic3CameraPresets,
     "VRGDG_CombinevideosV2": VRGDG_CombinevideosV2,
     "VRGDG_LoadAudioSplitDynamic": VRGDG_LoadAudioSplitDynamic
   
    

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastFilmGrain": "🎞️ Fast Film Grain",
     "ColorMatchToReference": "🎨 Color Match To Reference",
     "FastUnsharpSharpen": "🎯 Fast Unsharp Sharpen",
     "FastLaplacianSharpen": "🌀 Fast Laplacian Sharpen",
     "FastSobelSharpen": "📏 Fast Sobel Sharpen",
     "VRGDG_Unic3CameraPresets": "VRGDG_Unic3CameraPresets",
     "VRGDG_CombinevideosV2": "🌀 VRGDG_CombinevideosV2",
     "VRGDG_LoadAudioSplitDynamic":"VRGDG_LoadAudioSplitDynamic"
   
 

}


print(r"""
__     ______   ____                      ____              ____ _      _ 
\ \   / /  _ \ / ___| __ _ _ __ ___   ___|  _ \  _____   __/ ___(_)_ __| |
 \ \ / /| |_) | |  _ / _` | '_ ` _ \ / _ \ | | |/ _ \ \ / / |  _| | '__| |
  \ V / |  _ <| |_| | (_| | | | | | |  __/ |_| |  __/\ V /| |_| | | |  | |
   \_/  |_| \_\\____|\__,_|_| |_| |_|\___|____/ \___| \_/  \____|_|_|  |_|
                                                                          
             🎮 VRGameDevGirl custom nodes loaded successfully! 🎞️
""")
