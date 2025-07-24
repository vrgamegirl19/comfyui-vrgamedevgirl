import torch
import torch.nn.functional as F
import comfy
import kornia
import os
from PIL import Image
import torchvision.transforms.functional as TF
from .utils import load_video
import numpy as np
import comfy.model_management as mm
from .camera import get_camera_embedding
from .utils import set_initial_camera, build_cameras, traj_map


from typing import Tuple


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
    

NODE_CLASS_MAPPINGS = {
    "FastFilmGrain": FastFilmGrain,
     "ColorMatchToReference": ColorMatchToReference,
     "FastUnsharpSharpen": FastUnsharpSharpen,
     "FastLaplacianSharpen": FastLaplacianSharpen,
     "FastSobelSharpen": FastSobelSharpen,
     "VRGDG_Unic3CameraPresets":  VRGDG_Unic3CameraPresets,
   
    

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastFilmGrain": "🎞️ Fast Film Grain",
     "ColorMatchToReference": "🎨 Color Match To Reference",
     "FastUnsharpSharpen": "🎯 Fast Unsharp Sharpen",
     "FastLaplacianSharpen": "🌀 Fast Laplacian Sharpen",
     "FastSobelSharpen": "📏 Fast Sobel Sharpen",
     "VRGDG_Unic3CameraPresets": "VRGDG_Unic3CameraPresets",
   
 

}


print(r"""
__     ______   ____                      ____              ____ _      _ 
\ \   / /  _ \ / ___| __ _ _ __ ___   ___|  _ \  _____   __/ ___(_)_ __| |
 \ \ / /| |_) | |  _ / _` | '_ ` _ \ / _ \ | | |/ _ \ \ / / |  _| | '__| |
  \ V / |  _ <| |_| | (_| | | | | | |  __/ |_| |  __/\ V /| |_| | | |  | |
   \_/  |_| \_\\____|\__,_|_| |_| |_|\___|____/ \___| \_/  \____|_|_|  |_|
                                                                          
             🎮 VRGameDevGirl custom nodes loaded successfully! 🎞️
""")
