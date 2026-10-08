"""MiniMax H3 nodes used by the Video Builder: batched-tile VAE decode, audio drive, reference media from paths,
and image + reference to video conditioning."""

import copy
import json
import math
import os
import re
import time
from functools import partial

import comfy.model_management
import comfy.model_management as mm
import comfy.nested_tensor
import comfy.utils
import folder_paths
import node_helpers
import nodes
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy_api.latest import io
from PIL import Image, ImageOps

from .keyframes import align_i2v_keyframes
from .vae_decode import tiled_decode_batched


class VRGDG_MiniMaxH3KeyframeTiming:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "first_frame_index": ("INT", {"default": 0, "min": 0}),
            "last_frame_index": ("INT", {"default": 47, "min": 0}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "align"
    CATEGORY = "VRGDG/MiniMax H3"

    def align(self, conditioning: list, first_frame_index: int, last_frame_index: int) -> tuple:
        return (align_i2v_keyframes(conditioning, first_frame_index, last_frame_index),)


class H3FastVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "vae": ("VAE",),
            "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": 16,
                "tooltip": "Spatial tiles decoded together. Uses more VRAM. 1 runs stock decoding."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/VAE"
    DESCRIPTION = "Experimental H3 video decode with batched spatial tiles; preserves stock temporal processing and blending."

    def decode(self, samples, vae, tile_batch_size=4):
        if not isinstance(vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("H3 VAE Decode Fast requires the MiniMax H3 video VAE.")
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        start = time.perf_counter()
        if tile_batch_size == 1:
            images = vae.decode(latent)
        else:
            work_vae = copy.copy(vae)
            work_vae.patcher = vae.patcher.clone()
            work_vae.patcher.add_object_patch("tiled_decode", partial(
                tiled_decode_batched, vae.first_stage_model, tile_batch_size=tile_batch_size))
            memory_estimate = vae.memory_used_decode
            work_vae.memory_used_decode = lambda shape, dtype: memory_estimate(shape, dtype) * tile_batch_size
            try:
                images = work_vae.decode(latent)
            finally:
                mm.unload_model_and_clones(work_vae.patcher)
                work_vae.patcher.unpatch_model(unpatch_weights=False)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        if images.is_cuda:
            torch.cuda.synchronize(images.device)
        elapsed = time.perf_counter() - start
        report = f"H3 decode: {elapsed:.2f}s, {images.shape[0]} frames, tile batch {tile_batch_size} (includes loading and cleanup)."
        return images, report

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

MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
REFERENCE_VIDEO_FPS = 24
REFERENCE_VIDEO_MAX_FRAMES = 15 * REFERENCE_VIDEO_FPS


def _parse_path_values(raw, collection_keys=()):
    text = str(raw or "").strip()
    if not text:
        return []

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        pass

    if isinstance(parsed, list):
        values = parsed
    elif isinstance(parsed, dict):
        values = None
        for key in collection_keys:
            if isinstance(parsed.get(key), list):
                values = parsed[key]
                break
        if values is None:
            values = list(parsed.values())
    else:
        values = re.split(r"[\r\n]+", text)
    return values


def _clean_path(value):
    if isinstance(value, dict):
        value = value.get("path") or value.get("file") or value.get("image") or value.get("video") or ""
    return str(value or "").strip().strip('"').strip("'")


def _parse_image_paths(raw):
    return [
        path
        for path in (_clean_path(item) for item in _parse_path_values(raw, ("image_paths", "images")))
        if path
    ]


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_nonnegative_float(value, default=0.0):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, float(default))


def _parse_video_references(raw):
    references = []
    for item in _parse_path_values(raw, ("video_references", "videos")):
        if isinstance(item, dict):
            path = _clean_path(item)
            start_seconds = _as_nonnegative_float(
                item.get("start_seconds", item.get("start", item.get("seek_seconds", 0)))
            )
            duration = _as_nonnegative_float(
                item.get("duration_seconds", item.get("duration", 0))
            )
            use_audio = _as_bool(
                item.get("use_audio", item.get("include_audio", item.get("reference_audio", False)))
            )
        else:
            path = _clean_path(item)
            start_seconds = 0.0
            duration = 0.0
            use_audio = False
        if path:
            references.append({
                "path": path,
                "start_seconds": start_seconds,
                "duration": duration,
                "use_audio": use_audio,
            })
    return references


def _resolve_media_path(raw_path):
    path_text = _clean_path(raw_path)
    if not path_text:
        raise FileNotFoundError("MiniMax H3 reference media path was empty.")

    candidates = []
    if os.path.isabs(path_text):
        candidates.append(path_text)
    else:
        candidates.extend([
            path_text,
            os.path.abspath(path_text),
            os.path.join(folder_paths.get_input_directory(), path_text),
            os.path.join(folder_paths.get_output_directory(), path_text),
        ])
        get_temp_directory = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_directory):
            candidates.append(os.path.join(get_temp_directory(), path_text))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(os.path.abspath(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isfile(normalized):
            return normalized
    raise FileNotFoundError(f"MiniMax H3 reference media was not found: {path_text}")


def _load_image_tensor(raw_path):
    resolved = _resolve_media_path(raw_path)
    with Image.open(resolved) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _vhs_video_loader():
    import nodes

    loader_class = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoPath")
    if loader_class is None:
        raise RuntimeError(
            "MiniMax H3 video references require Video Helper Suite's VHS_LoadVideoPath node."
        )
    return loader_class()


def _load_video_reference(reference, slot_index):
    resolved = _resolve_media_path(reference["path"])
    fps = REFERENCE_VIDEO_FPS
    start_seconds = _as_nonnegative_float(reference.get("start_seconds", 0))
    duration = _as_nonnegative_float(reference.get("duration", 0))
    skip_first_frames = max(0, round(start_seconds * fps))
    frame_load_cap = (
        min(REFERENCE_VIDEO_MAX_FRAMES, max(1, round(duration * fps)))
        if duration > 0
        else REFERENCE_VIDEO_MAX_FRAMES
    )

    loader = _vhs_video_loader()
    load_function = getattr(loader, getattr(loader, "FUNCTION", "load_video"))
    frames, _frame_count, audio, _video_info = load_function(
        video=resolved,
        force_rate=fps,
        custom_width=0,
        custom_height=0,
        frame_load_cap=frame_load_cap,
        skip_first_frames=skip_first_frames,
        select_every_nth=1,
        unique_id=f"vrgdg_minimax_h3_reference_video_{slot_index}",
    )
    return frames, audio if reference.get("use_audio") else None


def _pad_slots(values, count):
    values = list(values[:count])
    return values + [None] * (count - len(values))


class VRGDG_MiniMaxH3ReferenceMediaFromPaths:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": (
                            "Ordered MiniMax H3 reference images. Use a JSON list, an object with "
                            "image_paths/images, or one path per line. Supports up to 9 images."
                        ),
                    },
                ),
                "video_references": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": (
                            "Ordered MiniMax H3 reference videos. Use a JSON list of paths or objects "
                            "with path, start_seconds, duration, and use_audio. Supports up to 3 videos."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",) * MAX_REFERENCE_IMAGES + ("IMAGE",) * MAX_REFERENCE_VIDEOS + ("AUDIO",) * MAX_REFERENCE_VIDEOS
    RETURN_NAMES = (
        tuple(f"ref_image_{index}" for index in range(MAX_REFERENCE_IMAGES))
        + tuple(f"ref_video_{index}" for index in range(MAX_REFERENCE_VIDEOS))
        + tuple(f"ref_video_audio_{index}" for index in range(MAX_REFERENCE_VIDEOS))
    )
    FUNCTION = "load_references"
    CATEGORY = "VRGDG/Video/Conditioning"
    DESCRIPTION = (
        "Builder-friendly ordered media loader for MiniMax H3. It accepts project/input/output paths, "
        "keeps each image and video in its own H3 reference slot, and returns None for unused slots."
    )

    def load_references(self, image_paths, video_references):
        paths = _parse_image_paths(image_paths)
        videos = _parse_video_references(video_references)
        if len(paths) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"MiniMax H3 supports at most {MAX_REFERENCE_IMAGES} reference images; received {len(paths)}."
            )
        if len(videos) > MAX_REFERENCE_VIDEOS:
            raise ValueError(
                f"MiniMax H3 supports at most {MAX_REFERENCE_VIDEOS} reference videos; received {len(videos)}."
            )

        image_outputs = _pad_slots([_load_image_tensor(path) for path in paths], MAX_REFERENCE_IMAGES)
        loaded_videos = [
            _load_video_reference(reference, index)
            for index, reference in enumerate(videos)
        ]
        video_outputs = _pad_slots([item[0] for item in loaded_videos], MAX_REFERENCE_VIDEOS)
        video_audio_outputs = _pad_slots([item[1] for item in loaded_videos], MAX_REFERENCE_VIDEOS)
        return tuple(image_outputs + video_outputs + video_audio_outputs)

CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length):
    frame_count = max(5, int(length))
    while frame_count % 17 != 5:
        frame_count += 1
    latent_t = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    audio_t = round((frame_count / FPS) * 40)
    video = torch.zeros(
        [1, 24, latent_t, height // 16, width // 16],
        device=comfy.model_management.intermediate_device(),
    )
    audio = torch.zeros(
        [1, 32, 2, audio_t],
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _reference_canvas(width, height, ref_image_size, image):
    h, w = image.shape[1], image.shape[2]
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def _encode_reference(vae, image):
    """Encode a still reference and keep its spatial latent grid patchifiable."""
    latent = vae.encode(image)
    # H3's reference patchifier uses 2x2 latent patches.  The video VAE can
    # ceil an arbitrary still-image size by one latent row/column, so trim the
    # edge row/column and describe the latent we actually pass to the model.
    latent_h = latent.shape[-2] - (latent.shape[-2] % 2)
    latent_w = latent.shape[-1] - (latent.shape[-1] % 2)
    if latent_h < 2 or latent_w < 2:
        raise ValueError("MiniMax H3 reference image produced a latent that is too small.")
    if latent_h != latent.shape[-2] or latent_w != latent.shape[-1]:
        # Keep the CLIP image and VAE latent on the same pixel canvas. Re-encode
        # after removing the VAE's ceil-only edge so reference token counts and
        # packed latent rows cannot disagree.
        image = image[:, :latent_h * 16, :latent_w * 16, :].contiguous()
        latent = vae.encode(image)
        latent_h = latent.shape[-2] - (latent.shape[-2] % 2)
        latent_w = latent.shape[-1] - (latent.shape[-1] % 2)
    if latent_h < 2 or latent_w < 2:
        raise ValueError("MiniMax H3 reference image produced a latent that is too small.")
    if latent_h != latent.shape[-2] or latent_w != latent.shape[-1]:
        latent = latent[..., :latent_h, :latent_w].contiguous()
    return image, latent, latent_h, latent_w


def _encode_keyframe(vae, image, width, height):
    """Encode a frame onto the exact target latent grid used by H3."""
    target_h, target_w = height // 16, width // 16
    latent = vae.encode(image)
    actual_h, actual_w = latent.shape[-2:]
    if (actual_h, actual_w) != (target_h, target_w):
        # The still-image VAE encoder can ceil a canvas such as 1088px to 67
        # latent rows. Add one 16px edge before re-encoding so the keyframe
        # has the same 68x120 grid as the generated target latent.
        pad_h = max(0, target_h - actual_h) * 16
        pad_w = max(0, target_w - actual_w) * 16
        if pad_h or pad_w:
            image = F.pad(image.movedim(-1, 1), (0, pad_w, 0, pad_h)).movedim(1, -1)
            latent = vae.encode(image)
        if latent.shape[-2] < target_h or latent.shape[-1] < target_w:
            raise ValueError(
                "MiniMax H3 could not encode the frame to the target latent grid "
                f"({target_h}x{target_w}); got {latent.shape[-2]}x{latent.shape[-1]}."
            )
        latent = latent[..., :target_h, :target_w].contiguous()
    return latent


class VRGDG_MiniMaxH3ImageReferenceToVideo(io.ComfyNode):
    """Use exact first/last frames and additional identity/reference images."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VRGDG_MiniMaxH3ImageReferenceToVideo",
            display_name="MiniMax H3 Image + Reference to Video",
            category="model/conditioning/minimax",
            description=(
                "Combined MiniMax H3 image-to-video and reference-image conditioning. "
                "First/last frames anchor the motion while reference images preserve identity, "
                "clothing, or other visual details."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip="Reference image sizing: match the generation canvas or preserve a larger identity reference.",
                ),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Additional identity, outfit, or composition reference image.",
                        ),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        prompt,
        width,
        height,
        length,
        ref_image_size="match",
        first_frame=None,
        last_frame=None,
        ref_images=None,
    ):
        latent, frame_count = _empty_av_latent(width, height, length)

        frame_images = []
        keyframes = []
        if first_frame is not None:
            image = _resize(first_frame[:1], width, height, "disabled")
            frame_images.append(image)
            keyframes.append(
                {
                    "resolved_frame_index": 0,
                    "image": image,
                    "latent": _encode_keyframe(vae, image, width, height),
                }
            )
        if last_frame is not None:
            image = _resize(last_frame[:1], width, height, "center")
            frame_images.append(image)
            keyframes.append(
                {
                    "resolved_frame_index": frame_count - 1,
                    "image": image,
                    "latent": _encode_keyframe(vae, image, width, height),
                }
            )

        ref_items = []
        ref_blocks = []
        for image in (ref_images or {}).values():
            if image is None:
                continue
            tw, th = _reference_canvas(width, height, ref_image_size, image)
            resized = _resize(image[:1], tw, th, "disabled")
            resized, reference_latent, latent_h, latent_w = _encode_reference(vae, resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append(
                {
                    "kind": "image",
                    "latent_h": latent_h,
                    "latent_w": latent_w,
                    "latent": reference_latent,
                }
            )

        tokens = clip.tokenize(prompt, images=frame_images, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for keyframe in keyframes:
                keyframe.pop("image", None)
            cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent)

NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3KeyframeTiming": VRGDG_MiniMaxH3KeyframeTiming,
    "H3FastVAEDecode": H3FastVAEDecode,
    "VRGDG_MiniMaxH3AudioDrive": VRGDG_MiniMaxH3AudioDrive,
    "VRGDG_MiniMaxH3ReferenceMediaFromPaths": VRGDG_MiniMaxH3ReferenceMediaFromPaths,
    "VRGDG_MiniMaxH3ImageReferenceToVideo": VRGDG_MiniMaxH3ImageReferenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3KeyframeTiming": "MiniMax H3 Per-scene Keyframe Timing",
    "H3FastVAEDecode": "H3 VAE Decode Fast (Batched Tiles)",
    "VRGDG_MiniMaxH3AudioDrive": "VRGDG MiniMax H3 Audio Drive",
    "VRGDG_MiniMaxH3ReferenceMediaFromPaths": "VRGDG MiniMax H3 Reference Media From Paths",
    "VRGDG_MiniMaxH3ImageReferenceToVideo": "MiniMax H3 Image + Reference to Video",
}
