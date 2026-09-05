"""Combined MiniMax H3 image-to-video and visual-reference conditioning."""

import math

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import io


CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24


def _build_still_prompt(prompt, picture_count, edit_fidelity="balanced"):
    """Turn an ordinary image prompt into H3's short static-shot vocabulary.

    H3 is an audiovisual video model, so the smallest useful still-image job is
    a five-frame burst. Keeping the camera, subjects, lighting, and ambience
    explicitly static makes those five frames behave like candidates for one
    image instead of the beginning of a video.
    """
    request = str(prompt or "").strip()
    if not request:
        raise ValueError("MiniMax H3 still-image generation requires a prompt.")

    count = max(0, int(picture_count))
    if count == 0:
        return (
            "integrated_multimodal_description: [Shot 1] Create one polished still-image "
            f"composition: {request} The camera is completely locked. All subjects, "
            "lighting, fabric, hair, particles, reflections, and background elements remain "
            "motionless across the entire five-frame burst. Treat every frame as the same "
            "finished photograph or illustration, with no transition, animation, camera "
            "movement, or change in exposure.\n"
            "overall_soundscape: Silence.\n"
            "non_diegetic_music: None."
        )

    fidelity = str(edit_fidelity or "balanced").strip().lower()
    relation = {
        "preserve": "fully_preserved",
        "balanced": "partially_preserved",
        "creative": "weak_reference",
    }.get(fidelity, "partially_preserved")
    direction = {
        "preserve": (
            "Preserve <Picture 1>'s composition, crop, perspective, subject identity, pose, "
            "silhouette, and spatial layout exactly except where the requested edit requires a change."
        ),
        "balanced": (
            "Keep <Picture 1>'s recognizable subject identity, composition, perspective, and "
            "spatial layout while making the requested edit cleanly and visibly."
        ),
        "creative": (
            "Use <Picture 1> as recognizable visual inspiration while allowing composition, "
            "styling, lighting, and details to change in service of the request."
        ),
    }.get(
        fidelity,
        "Keep <Picture 1>'s recognizable subject identity, composition, perspective, and "
        "spatial layout while making the requested edit cleanly and visibly.",
    )

    definitions = [
        "<Picture 1> is the primary source image to edit and the composition reference for the final still image."
    ]
    for index in range(2, count + 1):
        definitions.append(
            f"<Picture {index}> is an additional visual reference for identity, appearance, style, or details."
        )
    retention = [
        f"<Picture 1> ([Shot 1] composition and source content): {relation} - {direction}"
    ]
    for index in range(2, count + 1):
        retention.append(
            f"<Picture {index}> ([Shot 1] supporting visual reference): attribute_transfer - "
            "Transfer only the details relevant to the requested edit without replacing the primary composition."
        )
    labels = ", ".join(f"<Picture {index}>" for index in range(2, count + 1))
    supporting = (
        f" Use {labels} only as supporting references for the details requested by the user."
        if labels else ""
    )
    return (
        "subject_definitions:\n"
        + "\n".join(definitions)
        + "\nsummary: [reference generation] Produce one finished still-image edit from "
        "<Picture 1>, using the other supplied pictures only as supporting visual references.\n"
        "retention_analysis:\n"
        + "\n".join(retention)
        + "\ndetailed_description: The target is a single static, polished image represented by "
        "five identical-in-intent frames. [Shot 1] Begin from the composition and visible content "
        f"of <Picture 1>. {direction}{supporting} Apply this edit: {request} The camera is completely "
        "locked. There is no subject motion, lip movement, cloth or hair movement, environmental "
        "animation, transition, reframing, zoom, focus pull, lighting change, or exposure change. "
        "Every frame should read as the same final photograph or illustration.\n"
        "overall_soundscape: Silence.\n"
        "non_diegetic_music: None."
    )


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


def _lock_silent_audio(latent):
    """Keep H3's required audio stream at zero while sampling only video."""
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not getattr(samples, "is_nested", False):
        raise ValueError("MiniMax H3 silent-audio locking requires a joint AV latent.")
    streams = list(samples.unbind())
    if len(streams) < 2:
        raise ValueError("MiniMax H3 silent-audio locking requires video and audio streams.")
    output = latent.copy()
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.ones_like(streams[0]), torch.zeros_like(streams[1]))
    )
    return output


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


class VRGDG_MiniMaxH3StillImage(io.ComfyNode):
    """Generate a five-frame H3 burst for text-to-image or reference editing."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VRGDG_MiniMaxH3StillImage",
            display_name="MiniMax H3 Text / Reference to Image",
            category="model/conditioning/minimax",
            description=(
                "Uses MiniMax H3's minimum five-frame generation as a still-image burst. "
                "Connect source_image to edit an image through the H3 reference model; leave it "
                "empty for text-to-image. Decode the sampled latent with VAE Decode, then use "
                "MiniMax H3 Select Still Frame."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Combo.Input(
                    "prompt_format",
                    options=["still_image", "raw_h3"],
                    default="still_image",
                    tooltip="Wrap a normal image prompt for a static H3 burst, or pass an already-formatted H3 prompt unchanged.",
                ),
                io.Combo.Input(
                    "edit_fidelity",
                    options=["preserve", "balanced", "creative"],
                    default="balanced",
                    tooltip="Prompt-level preservation guidance for the primary source image. H3 does not expose a numeric reference strength.",
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip="Match is faster; max retains more reference detail but can use substantially more VRAM.",
                ),
                io.Image.Input(
                    "source_image",
                    optional=True,
                    tooltip="Primary image to edit. It becomes <Picture 1> but is not locked as a keyframe, allowing visible changes.",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Optional supporting identity, outfit, object, environment, or style reference.",
                        ),
                        prefix="ref_image_",
                        min=0,
                        max=8,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="five_frame_latent"),
                io.String.Output(display_name="formatted_prompt"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        prompt,
        width,
        height,
        prompt_format="still_image",
        edit_fidelity="balanced",
        ref_image_size="match",
        source_image=None,
        ref_images=None,
    ):
        latent, _frame_count = _empty_av_latent(width, height, 5)
        latent = _lock_silent_audio(latent)

        ordered_images = []
        if source_image is not None:
            ordered_images.append(source_image)
        ordered_images.extend(
            image for image in (ref_images or {}).values() if image is not None
        )

        ref_items = []
        ref_blocks = []
        for image in ordered_images:
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

        formatted_prompt = str(prompt or "").strip()
        if prompt_format != "raw_h3":
            formatted_prompt = _build_still_prompt(
                formatted_prompt,
                len(ordered_images),
                edit_fidelity,
            )
        elif not formatted_prompt:
            raise ValueError("MiniMax H3 still-image generation requires a prompt.")

        tokenize_args = {"minimax_ref_items": ref_items} if ref_items else {}
        tokens = clip.tokenize(formatted_prompt, **tokenize_args)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent, formatted_prompt)


class VRGDG_MiniMaxH3SelectStillFrame(io.ComfyNode):
    """Select one image from the five frames decoded from an H3 still burst."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VRGDG_MiniMaxH3SelectStillFrame",
            display_name="MiniMax H3 Select Still Frame",
            category="image/minimax",
            description=(
                "Selects one frame from a decoded MiniMax H3 still-image burst. "
                "The center frame is the recommended default because it has temporal context on both sides."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input(
                    "selection",
                    options=["middle", "first", "last", "index"],
                    default="middle",
                ),
                io.Int.Input("frame_index", default=2, min=0, max=9999),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="all_frames"),
                io.Int.Output(display_name="selected_index"),
            ],
        )

    @classmethod
    def execute(cls, images, selection="middle", frame_index=2):
        if images is None or len(images) < 1:
            raise ValueError("MiniMax H3 Select Still Frame requires at least one decoded image.")
        count = int(images.shape[0])
        if selection == "first":
            index = 0
        elif selection == "last":
            index = count - 1
        elif selection == "index":
            index = max(0, min(count - 1, int(frame_index)))
        else:
            index = count // 2
        return io.NodeOutput(images[index:index + 1], images, index)


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3ImageReferenceToVideo": VRGDG_MiniMaxH3ImageReferenceToVideo,
    "VRGDG_MiniMaxH3StillImage": VRGDG_MiniMaxH3StillImage,
    "VRGDG_MiniMaxH3SelectStillFrame": VRGDG_MiniMaxH3SelectStillFrame,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3ImageReferenceToVideo": "MiniMax H3 Image + Reference to Video",
    "VRGDG_MiniMaxH3StillImage": "MiniMax H3 Text / Reference to Image",
    "VRGDG_MiniMaxH3SelectStillFrame": "MiniMax H3 Select Still Frame",
}
