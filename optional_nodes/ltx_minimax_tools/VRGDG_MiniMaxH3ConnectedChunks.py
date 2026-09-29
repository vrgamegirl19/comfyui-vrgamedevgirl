"""MiniMax H3 connected chunks with source audio or generated-audio continuation."""

import torch

from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo, _resize as resize_h3_frame,
)

from .VRGDG_MiniMaxH3AudioDrive import VRGDG_MiniMaxH3AudioDrive


def chunk_timing(start_frame, chunk_seconds, context_frames, previous_count, continuity):
    body = max(1, round(float(chunk_seconds) * 24))
    start = int(start_frame)
    if start < 0:
        raise ValueError("Chunk start frame must be non-negative.")
    available = min(int(context_frames), int(previous_count), start) if continuity else 0
    prefix = max((n for n in (1, 5, 22, 39) if n <= available), default=0)
    requested = max(5, prefix + body)
    generated = requested + (5 - requested) % 17
    return {"start": start, "body": body, "prefix": prefix, "generated": generated}


def slice_chunk_audio(source_audio, plan):
    if source_audio is None:
        raise ValueError("Connect source_audio, or enable use_built_in_audio.")
    sr = int(source_audio["sample_rate"])
    wave = source_audio["waveform"]
    start = round((plan["start"] - plan["prefix"]) * sr / 24)
    body_end = round((plan["start"] + plan["body"]) * sr / 24)
    if body_end > wave.shape[-1]:
        raise ValueError("Source audio is shorter than the requested chunk. Load more audio or shorten the chunks.")
    size = round(plan["generated"] * sr / 24)
    result = wave[..., start:start + size].clone()
    if result.shape[-1] < size:
        result = torch.nn.functional.pad(result, (0, size - result.shape[-1]))
    return {"sample_rate": sr, "waveform": result}


def generated_audio_latent(latent, audio_vae, previous_audio, prefix, fixed_video=None):
    """Use H3's native per-stream masks to pin only the audible history prefix.

    Unlike a generic audio reference, these encoded samples occupy the actual
    overlapping target timeline. New audio remains denoisable. For a clean-model
    audio pass, fixed_video replaces the video stream and is fully preserved.
    """
    from comfy.nested_tensor import NestedTensor
    streams = tuple(latent["samples"].unbind())
    if len(streams) != 2:
        raise ValueError("Expected a joint video/audio latent.")
    video, template = streams
    audio = torch.zeros_like(template)
    mask = torch.ones_like(template)
    if prefix:
        if previous_audio is None:
            raise ValueError("Built-in audio continuation requires previous_audio for this chunk.")
        sr = int(previous_audio["sample_rate"])
        wave = previous_audio["waveform"]
        if sr <= 0 or wave.ndim != 3:
            raise ValueError("Previous audio must have a positive sample rate and [batch, channels, samples] waveform.")
        wanted = round(prefix * sr / 24)
        if wave.shape[-1] < wanted:
            raise ValueError("Previous audio is shorter than the video context window.")
        wave = wave[:1, ..., -wanted:]
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sr != vae_sr:
            import torchaudio
            wave = torchaudio.functional.resample(wave, sr, vae_sr)
        # ComfyUI's public VAE.encode center-crops to downscale_ratio BEFORE
        # MiniMax's internal encoder can right-pad. Pre-pad here to preserve
        # every history sample and the time origin (22 frames: 29333 -> 29600
        # samples at 32 kHz, yielding 37 rather than 36 audio latent steps).
        hop = getattr(audio_vae, "downscale_ratio", None)
        if not isinstance(hop, int) or isinstance(hop, bool) or hop <= 0:
            hop = max(1, round(vae_sr / 40))
        right_pad = (-wave.shape[-1]) % hop
        if right_pad:
            wave = torch.nn.functional.pad(wave, (0, right_pad))
        encoded = audio_vae.encode(wave.movedim(1, -1))
        steps = round(prefix * 40 / 24)
        if encoded.ndim != 4 or encoded.shape[1:-1] != template.shape[1:-1]:
            raise ValueError("Previous audio VAE latent does not match H3 audio channels.")
        if encoded.shape[-1] < steps or steps >= template.shape[-1]:
            raise ValueError(
                "Previous audio VAE latent cannot cover the requested context prefix: "
                f"need {steps} steps for {prefix} frames, encoded {encoded.shape[-1]}, "
                f"target has {template.shape[-1]} steps (VAE hop {hop} samples).")
        audio[..., :steps] = encoded[..., :steps].to(audio)
        mask[..., :steps] = 0
    out = latent.copy()
    video = video if fixed_video is None else fixed_video
    out["samples"] = NestedTensor((video, audio))
    out["noise_mask"] = NestedTensor((
        torch.ones_like(video) if fixed_video is None else torch.zeros_like(video), mask))
    return out


def assemble_generated_audio(generated_audio, previous_audio, plan):
    if generated_audio is None:
        raise ValueError("Connect the decoded per-chunk generated_audio when built-in audio is enabled.")
    sr = int(generated_audio["sample_rate"])
    wave = generated_audio["waveform"]
    if sr <= 0 or wave.ndim != 3:
        raise ValueError("Generated audio must have a positive sample rate and a 3D waveform.")
    start = round(plan["prefix"] * sr / 24)
    # Absolute output boundaries avoid cumulative sample-rounding drift.
    before = round(plan["start"] * sr / 24)
    total = round((plan["start"] + plan["body"]) * sr / 24)
    wanted = total - before
    body = wave[..., start:start + wanted].to(device="cpu", copy=True)
    missing = wanted - body.shape[-1]
    if missing > round(sr / 40) or body.shape[-1] == 0:
        raise ValueError("Generated audio is too short for the visible chunk; refusing a long silent pad.")
    if missing:
        body = torch.nn.functional.pad(body, (0, missing))
    if before:
        if previous_audio is None:
            raise ValueError("Missing previous assembled audio.")
        if int(previous_audio["sample_rate"]) != sr:
            import torchaudio
            prev = torchaudio.functional.resample(previous_audio["waveform"].cpu(),
                                                  int(previous_audio["sample_rate"]), sr)
        else:
            prev = previous_audio["waveform"].cpu()
        if prev.shape[-1] != before or prev.shape[:-1] != body.shape[:-1]:
            raise ValueError("Previous audio duration/channels do not match this chunk's start.")
        body = torch.cat((prev, body), dim=-1)
    return {"sample_rate": sr, "waveform": body}


class VRGDG_MiniMaxH3ConnectedChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "prompt": ("STRING", {"multiline": True}),
            "chunk_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60, "step": 0.1}),
            "width": ("INT", {"default": 640, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 384, "min": 32, "max": 8192, "step": 32}),
            "context_frames": (["1", "5", "22", "39"], {"default": "22"}),
            "continuity": ("BOOLEAN", {"default": True}),
            "start_frame": ("INT", {"default": 0, "min": 0, "forceInput": True}),
        }, "optional": {
            "previous_tail": ("IMAGE",),
            "reference_image": ("IMAGE", {"tooltip": "Character reference: <Picture 1> when connected."}),
            "location_image": ("IMAGE", {"tooltip": "Location reference: <Picture 2> with a character reference, otherwise <Picture 1>."}),
            "use_built_in_audio": ("BOOLEAN", {"default": False, "tooltip": "Generate audio per chunk and continue previous audio when a motion prefix exists."}),
            "source_audio": ("AUDIO", {"lazy": True}),
            "previous_audio": ("AUDIO", {"lazy": True}),
            "first_frame": ("IMAGE", {"tooltip": "Optional opening frame, stretched like native H3 Image to Video. Overrides previous visual context to start a new shot. Built-in audio may still continue across the cut."}),
            "last_frame": ("IMAGE", {"tooltip": "Optional endpoint, center-cropped like native H3 Image to Video. Anchors the last retained frame, before discarded padding. Works with previous-motion context."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "VRGDG_H3_CHUNK_PLAN", "STRING")
    RETURN_NAMES = ("positive", "av_latent", "trim_plan", "timing_report")
    FUNCTION = "prepare"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = "Connected H3 chunks with optional first/last image anchors and motion history. Built-in audio continues audio history; OFF locks the source soundtrack."

    def check_lazy_status(self, use_built_in_audio=False, source_audio=None,
                          previous_audio=None, continuity=True, start_frame=0, **kwargs):
        if not use_built_in_audio and source_audio is None:
            return ["source_audio"]
        if use_built_in_audio and continuity and start_frame > 0 and previous_audio is None:
            return ["previous_audio"]
        return []

    def prepare(self, clip, video_vae, audio_vae, source_audio=None, prompt="", chunk_seconds=5,
                width=640, height=384, context_frames="22", continuity=True, start_frame=0,
                previous_tail=None, reference_image=None, location_image=None,
                use_built_in_audio=False, previous_audio=None, first_frame=None, last_frame=None):
        history_frames = 0 if previous_tail is None else len(previous_tail)
        if first_frame is not None:
            # A first frame explicitly starts a new visual shot. Only generated
            # audio history can require a discarded prefix here; previous RGB
            # frames neither determine its length nor enter guide conditioning.
            history_frames = 0
            if use_built_in_audio and continuity and previous_audio is not None:
                sr = int(previous_audio["sample_rate"])
                if sr <= 0:
                    raise ValueError("Previous audio sample rate must be positive.")
                history_frames = int(previous_audio["waveform"].shape[-1] * 24 / sr)
        plan = chunk_timing(start_frame, chunk_seconds, context_frames, history_frames, continuity)
        audio = None if use_built_in_audio else slice_chunk_audio(source_audio, plan)
        if use_built_in_audio:
            plan["use_built_in_audio"] = True
        ref_images = {}
        for image in (reference_image, location_image):
            if image is not None:
                ref_images[f"ref_image_{len(ref_images)}"] = image
        result = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt, width=width, height=height, length=plan["generated"],
            vae=video_vae, audio_vae=audio_vae, ref_image_size="match",
            ref_images=ref_images,
            ref_audios={} if use_built_in_audio else {"ref_audio_0": audio},
        )
        positive, latent = result[0], result[1]
        if plan["prefix"] and first_frame is None:
            tail = previous_tail[-plan["prefix"]:].clone()
            positive = MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=0, vae=video_vae, image=tail,
            )[0]
        # Native Image to Video uses stretch for the opening image and cover
        # center-crop for the ending image. AddGuide emits the same native
        # minimax_keyframes metadata while preserving existing references and
        # motion history. Endpoint images are guides, not extra Picture labels.
        if first_frame is not None:
            first = resize_h3_frame(first_frame[:1], width, height, "disabled")
            positive = MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=plan["prefix"], vae=video_vae, image=first,
            )[0]
        if last_frame is not None:
            last = resize_h3_frame(last_frame[:1], width, height, "center")
            positive = MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent,
                frame_idx=plan["prefix"] + plan["body"] - 1,
                vae=video_vae, image=last,
            )[0]
        if use_built_in_audio:
            driven = generated_audio_latent(latent, audio_vae, previous_audio, plan["prefix"])
        else:
            driven, _ = VRGDG_MiniMaxH3AudioDrive().apply_audio_drive(latent, audio, audio_vae)
        report = (f"Output {plan['start'] / 24:.3f}-{(plan['start'] + plan['body']) / 24:.3f}s; "
                  f"generate {plan['generated']} frames ({plan['generated'] / 24:.3f}s); "
                  f"discard prefix {plan['prefix']}, keep {plan['body']}, "
                  f"discard padding {plan['generated'] - plan['prefix'] - plan['body']}.")
        if first_frame is not None:
            report += (f" First-frame anchor: {plan['prefix']} (visible opening); "
                       "previous visual context overridden.")
        if last_frame is not None:
            report += f" Last-frame anchor: {plan['prefix'] + plan['body'] - 1} (visible endpoint)."
        print(f"[VRGDG H3 Connected Chunk] {report}")
        return positive, driven, plan, report


class VRGDG_MiniMaxH3FinishChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "trim_plan": ("VRGDG_H3_CHUNK_PLAN",),
        }, "optional": {"previous_images": ("IMAGE",),
                           "source_audio": ("AUDIO", {"lazy": True}),
                           "generated_audio": ("AUDIO", {"lazy": True}),
                           "previous_audio": ("AUDIO",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "AUDIO", "IMAGE")
    RETURN_NAMES = ("assembled_images", "motion_tail", "next_start_frame", "assembled_audio", "chunk_images")
    FUNCTION = "finish"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = "Trim matching audio/video prefixes and padding, assemble on CPU, and retain motion history. Select generated or source audio automatically from the chunk plan."

    def check_lazy_status(self, trim_plan, source_audio=None, generated_audio=None,
                          previous_audio=None, **kwargs):
        if trim_plan.get("use_built_in_audio", False):
            required = []
            if generated_audio is None:
                required.append("generated_audio")
            if trim_plan["start"] and previous_audio is None:
                required.append("previous_audio")
            return required
        return ["source_audio"] if source_audio is None else []

    def finish(self, images, trim_plan, source_audio=None, previous_images=None,
               generated_audio=None, previous_audio=None):
        plan = trim_plan
        previous_count = 0 if previous_images is None else len(previous_images)
        if previous_count != plan["start"]:
            raise ValueError("Previous assembled frames do not match this chunk's start frame.")
        end = plan["prefix"] + plan["body"]
        if len(images) < end:
            raise ValueError("Decoded chunk is shorter than its trim plan.")
        body = images[plan["prefix"]:end].to(device="cpu", copy=True)
        if previous_images is None:
            assembled = body
        else:
            assembled = torch.cat((previous_images.to("cpu"), body), dim=0)
        tail = assembled[-39:].clone()
        if plan.get("use_built_in_audio", False):
            audio = assemble_generated_audio(generated_audio, previous_audio, plan)
            return assembled, tail, len(assembled), audio, body
        if source_audio is None:
            raise ValueError("Connect source_audio when built-in audio is disabled.")
        sr = int(source_audio["sample_rate"])
        sample_count = round(len(assembled) * sr / 24)
        if source_audio["waveform"].shape[-1] < sample_count:
            raise ValueError("Original audio does not cover the assembled frames.")
        audio = {"sample_rate": sr, "waveform": source_audio["waveform"][..., :sample_count].clone()}
        if previous_audio is not None and plan["start"]:
            start_sample = round(plan["start"] * sr / 24)
            body_audio = {"sample_rate": sr, "waveform": audio["waveform"][..., start_sample:]}
            audio = assemble_generated_audio(body_audio, previous_audio, dict(plan, prefix=0))
        return assembled, tail, len(assembled), audio, body


class VRGDG_MiniMaxH3PrepareCleanAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampled_av": ("LATENT",), "audio_vae": ("VAE",),
                              "trim_plan": ("VRGDG_H3_CHUNK_PLAN",)},
                "optional": {"previous_audio": ("AUDIO",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("video_fixed_fresh_audio",)
    FUNCTION = "prepare"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = "Discard Turbo audio, hold sampled video fixed, and initialize fresh audio plus previous waveform context for a separate clean-model sampler."

    def prepare(self, sampled_av, audio_vae, trim_plan, previous_audio=None):
        video, _ = tuple(sampled_av["samples"].unbind())
        return (generated_audio_latent(sampled_av, audio_vae, previous_audio,
                                       trim_plan["prefix"], fixed_video=video),)


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3ConnectedChunk": VRGDG_MiniMaxH3ConnectedChunk,
    "VRGDG_MiniMaxH3FinishChunk": VRGDG_MiniMaxH3FinishChunk,
    "VRGDG_MiniMaxH3PrepareCleanAudio": VRGDG_MiniMaxH3PrepareCleanAudio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3ConnectedChunk": "VRGDG H3 Connected Chunk",
    "VRGDG_MiniMaxH3FinishChunk": "VRGDG H3 Trim + Assemble Chunk",
    "VRGDG_MiniMaxH3PrepareCleanAudio": "VRGDG H3 Fresh Audio / Fixed Video (Advanced)",
}
