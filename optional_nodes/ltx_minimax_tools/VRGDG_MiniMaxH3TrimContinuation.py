"""MiniMax H3 Latent Continuation Custom Nodes for ComfyUI.

Provides dedicated nodes to save, load, inject, and trim MiniMax H3 latents
for lossless, native temporal chaining across multi-scene video projects.
"""


import torch


class VRGDG_MiniMaxH3TrimContinuation:
    """Slice leading temporal context frames from the decoded video output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "context_frames": ("INT", {
                    "default": 22,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Number of leading context frames to trim from the decoded output",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trimmed_images",)
    FUNCTION = "trim"
    CATEGORY = "VRGDG/MiniMax H3 Latent Continuation"
    DESCRIPTION = (
        "Slices off the leading context frames so only newly generated frames remain for "
        "seamless downstream video assembly and timeline playback."
    )

    def trim(self, images: torch.Tensor, context_frames: int = 22) -> tuple[torch.Tensor]:
        if context_frames <= 0 or images is None:
            return (images,)

        total_frames = int(images.shape[0])
        if total_frames <= context_frames:
            print(
                f"[VRGDG Latent Trim] Warning: Image batch has only {total_frames} frames; "
                f"cannot trim {context_frames} frames. Returning full batch."
            )
            return (images,)

        trimmed = images[context_frames:].clone()
        print(f"[VRGDG Latent Trim] Trimmed {context_frames} context frames: {total_frames} -> {trimmed.shape[0]} frames")
        return (trimmed,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3TrimContinuation": VRGDG_MiniMaxH3TrimContinuation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3TrimContinuation": "VRGDG H3 Trim Continuation Output",
}
