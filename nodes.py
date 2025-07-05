import torch
import torch.nn.functional as F
import comfy
import kornia
import numpy as np
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



NODE_CLASS_MAPPINGS = {
    "FastFilmGrain": FastFilmGrain,
     "ColorMatchToReference": ColorMatchToReference,
     "FastUnsharpSharpen": FastUnsharpSharpen,
     "FastLaplacianSharpen": FastLaplacianSharpen,
     "FastSobelSharpen": FastSobelSharpen
    

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastFilmGrain": "🎞️ Fast Film Grain",
     "ColorMatchToReference": "🎨 Color Match To Reference",
     "FastUnsharpSharpen": "🎯 Fast Unsharp Sharpen",
     "FastLaplacianSharpen": "🌀 Fast Laplacian Sharpen",
     "FastSobelSharpen": "📏 Fast Sobel Sharpen"
 

}


print(r"""
__     ______   ____                      ____              ____ _      _ 
\ \   / /  _ \ / ___| __ _ _ __ ___   ___|  _ \  _____   __/ ___(_)_ __| |
 \ \ / /| |_) | |  _ / _` | '_ ` _ \ / _ \ | | |/ _ \ \ / / |  _| | '__| |
  \ V / |  _ <| |_| | (_| | | | | | |  __/ |_| |  __/\ V /| |_| | | |  | |
   \_/  |_| \_\\____|\__,_|_| |_| |_|\___|____/ \___| \_/  \____|_|_|  |_|
                                                                          
             🎮 VRGameDevGirl custom nodes loaded successfully! 🎞️
""")
