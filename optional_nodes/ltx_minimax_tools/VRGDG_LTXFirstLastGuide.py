import torch

import comfy.utils
from comfy_extras.nodes_lt import LTXVAddGuide


class VRGDG_LTXFirstLastGuide:
    """Build a continuous, low-strength temporal guide from two user images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
                "guide_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "transition_start": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.95, "step": 0.01}),
                "transition_end": ("FLOAT", {"default": 0.90, "min": 0.05, "max": 1.0, "step": 0.01}),
                "curve": (["smoothstep", "linear", "ease_in", "ease_out"], {"default": "smoothstep"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "add_first_last_guide"
    CATEGORY = "VRGDG/video/conditioning"

    @staticmethod
    def _curve(value, name):
        if name == "linear":
            return value
        if name == "ease_in":
            return value * value
        if name == "ease_out":
            return 1.0 - (1.0 - value) * (1.0 - value)
        return value * value * (3.0 - 2.0 * value)

    def add_first_last_guide(
        self, positive, negative, vae, latent, first_image, last_image,
        guide_strength=0.35, transition_start=0.05, transition_end=0.90,
        curve="smoothstep",
    ):
        samples = latent["samples"]
        latent_length = int(samples.shape[2])
        time_scale = int(vae.downscale_index_formula[0])
        frame_count = max(1, (latent_length - 1) * time_scale + 1)

        first = first_image[:1]
        last = last_image[:1]
        target_h = int(first.shape[1])
        target_w = int(first.shape[2])
        if int(last.shape[1]) != target_h or int(last.shape[2]) != target_w:
            last = comfy.utils.common_upscale(
                last.movedim(-1, 1), target_w, target_h, "bilinear", "center"
            ).movedim(1, -1)

        start = max(0.0, min(0.95, float(transition_start)))
        end = max(start + 0.01, min(1.0, float(transition_end)))
        frames = []
        for index in range(frame_count):
            position = index / max(1, frame_count - 1)
            amount = max(0.0, min(1.0, (position - start) / (end - start)))
            amount = self._curve(amount, str(curve))
            frames.append(first * (1.0 - amount) + last * amount)
        guide_video = torch.cat(frames, dim=0)

        # Encode the temporal blend directly into the existing latent timeline.
        # Calling LTXVAddGuide.execute here would append every blended frame as a
        # separate attention keyframe, doubling temporal tokens and causing OOM.
        _, guide_latent = LTXVAddGuide.encode(
            vae,
            int(samples.shape[4]),
            int(samples.shape[3]),
            guide_video,
            vae.downscale_index_formula,
        )
        if guide_latent.shape[2] != latent_length:
            raise ValueError(
                f"Temporal guide encoded to {guide_latent.shape[2]} latent frames; "
                f"the destination latent requires {latent_length}."
            )
        if guide_latent.shape[1] != samples.shape[1]:
            raise ValueError(
                f"Temporal guide has {guide_latent.shape[1]} channels; "
                f"the destination latent requires {samples.shape[1]}."
            )

        strength = max(0.0, min(1.0, float(guide_strength)))
        noise_mask = torch.full(
            (samples.shape[0], 1, latent_length, 1, 1),
            1.0 - strength,
            dtype=guide_latent.dtype,
            device=guide_latent.device,
        )
        output_latent = dict(latent)
        output_latent["samples"] = guide_latent
        output_latent["noise_mask"] = noise_mask
        return positive, negative, output_latent


class VRGDG_LTXFirstLastEndpointGuide:
    """Wan-style FLF experiment: pin two endpoints and leave the middle noisy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
                "first_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "add_endpoint_guides"
    CATEGORY = "VRGDG/video/conditioning"

    @staticmethod
    def _encode_single(vae, image, samples):
        _, encoded = LTXVAddGuide.encode(
            vae,
            int(samples.shape[4]),
            int(samples.shape[3]),
            image[:1],
            vae.downscale_index_formula,
        )
        if encoded.shape[2] != 1:
            raise ValueError(f"Expected one encoded endpoint latent frame, received {encoded.shape[2]}.")
        if encoded.shape[1] != samples.shape[1]:
            raise ValueError(
                f"Endpoint guide has {encoded.shape[1]} channels; "
                f"the destination latent requires {samples.shape[1]}."
            )
        return encoded

    def add_endpoint_guides(
        self, positive, negative, vae, latent, first_image, last_image,
        first_strength=1.0, last_strength=1.0,
    ):
        samples = latent["samples"]
        if samples.ndim != 5 or samples.shape[2] < 2:
            raise ValueError("First/Last Endpoint Guide requires a video latent with at least two latent frames.")

        first_latent = self._encode_single(vae, first_image, samples)
        last_latent = self._encode_single(vae, last_image, samples)
        first_latent = first_latent.to(device=samples.device, dtype=samples.dtype)
        last_latent = last_latent.to(device=samples.device, dtype=samples.dtype)

        output_samples = samples.clone()
        output_samples[:, :, 0:1] = first_latent
        output_samples[:, :, -1:] = last_latent

        existing_mask = latent.get("noise_mask")
        if existing_mask is None:
            noise_mask = torch.ones(
                (samples.shape[0], 1, samples.shape[2], 1, 1),
                dtype=samples.dtype,
                device=samples.device,
            )
        else:
            noise_mask = existing_mask.clone().to(device=samples.device)
            if noise_mask.ndim == 3:
                noise_mask = noise_mask.unsqueeze(1).unsqueeze(-1)
            while noise_mask.ndim < 5:
                noise_mask = noise_mask.unsqueeze(-1)

        first_strength = max(0.0, min(1.0, float(first_strength)))
        last_strength = max(0.0, min(1.0, float(last_strength)))
        noise_mask[:, :, 0:1] = 1.0 - first_strength
        noise_mask[:, :, -1:] = 1.0 - last_strength

        output_latent = dict(latent)
        output_latent["samples"] = output_samples
        output_latent["noise_mask"] = noise_mask
        return positive, negative, output_latent


NODE_CLASS_MAPPINGS = {
    "VRGDG_LTXFirstLastGuide": VRGDG_LTXFirstLastGuide,
    "VRGDG_LTXFirstLastEndpointGuide": VRGDG_LTXFirstLastEndpointGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LTXFirstLastGuide": "VRGDG LTX First / Last Temporal Guide",
    "VRGDG_LTXFirstLastEndpointGuide": "VRGDG LTX First / Last Endpoint Guide",
}
