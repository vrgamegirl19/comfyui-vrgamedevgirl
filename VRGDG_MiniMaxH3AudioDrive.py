import torch
import torchaudio

import comfy.nested_tensor


def _nested_av_parts(av_latent):
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("MiniMax H3 Audio Drive requires an AV LATENT input.")

    samples = av_latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "MiniMax H3 Audio Drive expected a joint video+audio latent. "
            "Connect the LATENT output from a MiniMax H3 conditioning node."
        )

    parts = list(samples.unbind())
    if len(parts) < 2:
        raise ValueError("MiniMax H3 Audio Drive could not find the audio half of the AV latent.")
    return parts[0], parts[1]


def _fit_audio_latent(encoded_audio, template_audio):
    if encoded_audio.ndim != 4 or template_audio.ndim != 4:
        raise ValueError(
            "MiniMax H3 audio latents must use [batch, channels, stereo, time] layout."
        )
    if encoded_audio.shape[1:-1] != template_audio.shape[1:-1]:
        raise ValueError(
            "The encoded source audio does not match the MiniMax H3 audio latent layout: "
            f"got {tuple(encoded_audio.shape)}, expected channels {tuple(template_audio.shape[1:-1])}."
        )

    target_batch = template_audio.shape[0]
    if encoded_audio.shape[0] == 1 and target_batch > 1:
        encoded_audio = encoded_audio.repeat(target_batch, 1, 1, 1)
    elif encoded_audio.shape[0] != target_batch:
        encoded_audio = encoded_audio[:target_batch]
        if encoded_audio.shape[0] != target_batch:
            raise ValueError(
                f"Source audio batch {encoded_audio.shape[0]} cannot match latent batch {target_batch}."
            )

    target_t = template_audio.shape[-1]
    current_t = encoded_audio.shape[-1]
    if current_t > target_t:
        encoded_audio = encoded_audio[..., :target_t]
    elif current_t < target_t:
        padding = encoded_audio.new_zeros((*encoded_audio.shape[:-1], target_t - current_t))
        encoded_audio = torch.cat((encoded_audio, padding), dim=-1)

    return encoded_audio.to(device=template_audio.device, dtype=template_audio.dtype)


class VRGDG_MiniMaxH3AudioDrive:
    """Lock source audio into MiniMax H3's main AV latent while generating video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT", {
                    "tooltip": "Joint AV latent from MiniMax H3 Reference/Image to Video."
                }),
                "source_audio": ("AUDIO", {
                    "tooltip": "Audio that should drive the video and remain unchanged in the final mux."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE used to place the source audio in the AV latent."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("audio_driven_av_latent", "original_audio")
    FUNCTION = "apply_audio_drive"
    CATEGORY = "VRGDG/Video/Conditioning"
    DESCRIPTION = (
        "Replaces MiniMax H3's blank generated-audio latent with an encoded source track, "
        "locks that audio with a zero denoise mask, and passes the original AUDIO through "
        "unchanged for the final video mux. Keep the same AUDIO connected to ref_audio_0 "
        "and reference it as <Audio 1> in the prompt."
    )

    def apply_audio_drive(self, av_latent, source_audio, audio_vae):
        if not isinstance(source_audio, dict):
            raise ValueError("MiniMax H3 Audio Drive requires a connected AUDIO input.")
        waveform = source_audio.get("waveform")
        sample_rate = source_audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            raise ValueError("The connected AUDIO is missing waveform or sample_rate data.")
        if waveform.ndim != 3:
            raise ValueError(
                f"Expected source audio waveform [batch, channels, samples], got {tuple(waveform.shape)}."
            )

        video_latent, template_audio = _nested_av_parts(av_latent)
        vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if int(sample_rate) != vae_sample_rate:
            waveform_for_vae = torchaudio.functional.resample(
                waveform, int(sample_rate), vae_sample_rate
            )
        else:
            waveform_for_vae = waveform

        encoded_audio = audio_vae.encode(waveform_for_vae[:1].movedim(1, -1))
        encoded_audio = _fit_audio_latent(encoded_audio, template_audio)

        output = av_latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor((video_latent, encoded_audio))
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((
            torch.ones_like(video_latent),
            torch.zeros_like(encoded_audio),
        ))

        # Deliberately return the original AUDIO object. The VAE round-trip is only
        # for model conditioning; final muxing should use this untouched waveform.
        return output, source_audio


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3AudioDrive": VRGDG_MiniMaxH3AudioDrive,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3AudioDrive": "VRGDG MiniMax H3 Audio Drive",
}
