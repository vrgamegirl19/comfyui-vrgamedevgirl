"""User-friendly sigma presets for LTX 2.5 enhancement and denoising."""

import math

import torch


PRESETS = {
    "Enhance Only — Preserve Original": (0.30, 0.18, 0.06, 0.0),
    "Very Low — Subtle Detail Recovery": (0.50, 0.35, 0.15, 0.0),
    "Low — Light Enhancement": (0.65, 0.50, 0.25, 0.0),
    "Medium — Balanced Enhancement": (0.75, 0.60, 0.35, 0.0),
    "High — Noticeable Refinement": (0.85, 0.725, 0.4219, 0.0),
    "Very High — Strong Reinterpretation": (0.95, 0.80, 0.50, 0.0),
}

PRESET_NAMES = tuple(PRESETS) + ("Custom — Enter Your Own Values",)


def _parse_custom_sigmas(value):
    """Parse and validate a comma-separated sigma schedule."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Custom sigma values are required. Enter comma-separated numbers, "
            "for example: 0.85, 0.725, 0.4219, 0.0"
        )

    parts = [part.strip() for part in value.split(",")]
    if any(not part for part in parts):
        raise ValueError("Custom sigma values must be comma-separated numbers without empty entries.")

    try:
        sigmas = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "Invalid custom sigma values. Enter only comma-separated numbers, "
            "for example: 0.85, 0.725, 0.4219, 0.0"
        ) from exc

    if len(sigmas) < 2:
        raise ValueError("At least two custom sigma values are required for a sampler schedule.")
    if not all(math.isfinite(sigma) for sigma in sigmas):
        raise ValueError("Custom sigma values must all be finite numbers.")

    return sigmas


class LTX25SigmaPreset:
    """Return a native ComfyUI SIGMAS tensor for an LTX 2.5 sigma schedule."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(PRESET_NAMES), {
                    "tooltip": (
                        "Lower presets preserve the original video more closely. "
                        "Higher presets create stronger detail, lighting, texture, "
                        "and motion changes."
                    )
                }),
                "custom_sigmas": ("STRING", {
                    "default": "0.85, 0.725, 0.4219, 0.0",
                    "multiline": False,
                    "tooltip": "Used only with Custom — Enter Your Own Values.",
                }),
                "sigma_values": ("STRING", {
                    "default": "0.30, 0.18, 0.06, 0.0",
                    "multiline": False,
                    "tooltip": "Read-only preview of the selected sigma schedule.",
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "VRGameDevGirl/LTX/Sampling"
    DESCRIPTION = (
        "LTX 2.5 sigma presets for enhancement and denoising. Enhance Only "
        "minimizes changes but cannot guarantee perfect preservation: LTX may "
        "still modify faces, motion, lighting, or textures during denoising."
    )

    def get_sigmas(self, preset, custom_sigmas, sigma_values):
        if preset == "Custom — Enter Your Own Values":
            values = _parse_custom_sigmas(custom_sigmas)
        else:
            try:
                values = PRESETS[preset]
            except KeyError as exc:
                raise ValueError(f"Unknown LTX 2.5 sigma preset: {preset}") from exc

        # ComfyUI's SIGMAS connection convention is a one-dimensional torch tensor.
        return (torch.tensor(values, dtype=torch.float32),)


NODE_CLASS_MAPPINGS = {
    "VRGDG_LTX25SigmaPreset": LTX25SigmaPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LTX25SigmaPreset": "LTX 2.5 Sigma Preset",
}
