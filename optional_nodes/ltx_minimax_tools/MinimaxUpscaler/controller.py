"""Frontend-controlled settings panel for the MiniMax H3 upscaler workflow."""

from __future__ import annotations


class VRGDGMiniMaxUpscalerControlPanel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "video_mode": (["remix", "replace", "enhance"],),
                "video_denoise": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlap_preset": (["H3 Near 41 (39 / 5)", "H3 Balanced (73 / 17)", "H3 Trained (124 / 22)"],),
                "custom_window": ("INT", {"default": 73, "min": 5, "max": 10000, "step": 17}),
                "custom_overlap": ("INT", {"default": 17, "min": 1, "max": 5000, "step": 1}),
                "blend_mode": (["cosine", "smoothstep", "linear"],),
                "steps": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "sampler_denoise": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler": (["sa_solver", "euler", "dpmpp_2m"],),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "width": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 576, "min": 0, "max": 16384, "step": 8}),
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 120, "step": 1}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 10000000}),
                "tile_batch_size": ("INT", {"default": 8, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "apply"
    CATEGORY = "VRGDG/MiniMax H3 Upscaler"
    OUTPUT_NODE = True

    def apply(self, **kwargs):
        return ("Settings ready. Click Apply & Close in the panel, then queue the workflow.",)


NODE_CLASS_MAPPINGS = {
    "VRGDGMiniMaxUpscalerControlPanel": VRGDGMiniMaxUpscalerControlPanel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDGMiniMaxUpscalerControlPanel": "MiniMax H3 Upscaler Settings Panel",
}
