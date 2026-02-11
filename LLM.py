import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple

from google import genai


class VRGDG_NanoBananaPro:
    """
    Simple standalone Gemini Nano Banana node
    - API key typed directly into node
    - Prompt box
    - Up to 4 optional image inputs
    - Landscape-only output instruction
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "A cinematic wide landscape", "multiline": True}),
            },
            "optional": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "VRGDG/NanoBananaPro"

    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> list[Image.Image]:
        if tensor.ndim == 4:
            batch = tensor
        else:
            batch = tensor.unsqueeze(0)
        images = []
        for i in range(batch.shape[0]):
            arr = (batch[i].cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(arr))
        return images

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def generate(
        self,
        api_key: str,
        prompt: str,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:

        if not api_key.strip():
            raise Exception("API key missing")

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-3-pro-image-preview")

        contents = []

        # Optional reference images
        for img in [image1, image2, image3, image4]:
            if img is not None:
                contents.extend(self._tensor_to_pil_list(img))

        # Force landscape instruction
        full_prompt = (
            "Generate ONE landscape-only wide image (16:9). "
            "Never return portrait or square.\n\n"
            f"Prompt: {prompt}"
        )

        contents.append(full_prompt)

        response = model.generate_content(contents)

        # Extract inline returned image
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                img_bytes = part.inline_data.data
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                return (self._pil_to_tensor(pil_img),)

        raise Exception("No image returned")


NODE_CLASS_MAPPINGS = {
    "VRGDG_NanoBananaPro": VRGDG_NanoBananaPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_NanoBananaPro": "VRGDG NanoBanana Pro"
}

