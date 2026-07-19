import itertools
import os
import shutil
import subprocess
import time
import uuid

import torch
import torch.nn.functional as F


VIDEO_ENHANCE_CONTEXT = "VRGDG_VIDEO_ENHANCE_CONTEXT"


def _log(message):
    print(f"[VRGDG Video Enhance] {message}", flush=True)


def _progress(index, total, label, checkpoints=10):
    if total <= 0:
        return
    step = max(1, total // checkpoints)
    if index == 0 or index + 1 == total or (index + 1) % step == 0:
        _log(f"{label}: {index + 1}/{total}")


def _fps(video_info, fallback):
    if isinstance(video_info, dict):
        for key in ("loaded_fps", "source_fps", "fps"):
            try:
                value = float(video_info.get(key) or 0)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
    return float(fallback)


def _round_dimension(value, multiple):
    value = max(8, int(value))
    multiple = max(1, int(multiple))
    return max(multiple, int(round(value / multiple)) * multiple)


def _interpolation(mode):
    return {
        "Nearest": "nearest",
        "Bilinear": "bilinear",
        "Bicubic (recommended)": "bicubic",
        "Area": "area",
    }.get(str(mode), "bicubic")


def _resize_batch(images, target_width, target_height, fit_mode, resize_method):
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("Video Enhance requires a non-empty IMAGE batch.")
    source_height, source_width = int(images.shape[1]), int(images.shape[2])
    target_width, target_height = int(target_width), int(target_height)
    tensor = images[..., :3].permute(0, 3, 1, 2)
    interpolation = _interpolation(resize_method)
    kwargs = {"mode": interpolation}
    if interpolation in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False

    if fit_mode == "Stretch to dimensions":
        result = F.interpolate(tensor, size=(target_height, target_width), **kwargs)
    else:
        scale = (
            max(target_width / source_width, target_height / source_height)
            if fit_mode == "Crop to fill"
            else min(target_width / source_width, target_height / source_height)
        )
        scaled_width = max(1, int(round(source_width * scale)))
        scaled_height = max(1, int(round(source_height * scale)))
        resized = F.interpolate(tensor, size=(scaled_height, scaled_width), **kwargs)
        if fit_mode == "Crop to fill":
            left = max(0, (scaled_width - target_width) // 2)
            top = max(0, (scaled_height - target_height) // 2)
            result = resized[:, :, top:top + target_height, left:left + target_width]
        else:
            pad_left = max(0, (target_width - scaled_width) // 2)
            pad_right = max(0, target_width - scaled_width - pad_left)
            pad_top = max(0, (target_height - scaled_height) // 2)
            pad_bottom = max(0, target_height - scaled_height - pad_top)
            result = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return result.permute(0, 2, 3, 1).clamp(0, 1)


def _restore_batch(images, source_width, source_height, fit_mode, resize_method):
    """Undo temporary letterboxing before returning frames to source dimensions."""
    if fit_mode != "Fit with letterbox (preserve all)":
        return _resize_batch(
            images, source_width, source_height,
            "Stretch to dimensions", resize_method,
        )
    working_height, working_width = int(images.shape[1]), int(images.shape[2])
    scale = min(working_width / source_width, working_height / source_height)
    content_width = min(working_width, max(1, int(round(source_width * scale))))
    content_height = min(working_height, max(1, int(round(source_height * scale))))
    left = max(0, (working_width - content_width) // 2)
    top = max(0, (working_height - content_height) // 2)
    content = images[:, top:top + content_height, left:left + content_width, :]
    return _resize_batch(
        content, source_width, source_height,
        "Stretch to dimensions", resize_method,
    )


def _save_image_batch(images, folder, prefix):
    from PIL import Image

    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            os.remove(os.path.join(folder, name))
    for index, image in enumerate(images):
        array = (image[..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")
        Image.fromarray(array, mode="RGB").save(os.path.join(folder, f"{prefix}_{index:06d}.png"))
        _progress(index, len(images), f"Saving {prefix} frames")


def _encode_mp4(frames_folder, frame_count, fps, output_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("FFmpeg is required to create the Video Enhance working video.") from exc
    command = [
        ffmpeg, "-y", "-framerate", f"{fps:.12g}",
        "-i", os.path.join(frames_folder, "frame_%06d.png"),
        "-frames:v", str(int(frame_count)), "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "10",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0 or not os.path.isfile(output_path):
        details = (result.stderr or result.stdout or "FFmpeg failed").strip()
        raise RuntimeError(f"Could not create the Video Enhance working MP4: {details[-1600:]}")


def _meta_image_generator(directory, meta_batch=None, unique_id=None):
    import numpy as np
    from PIL import Image, ImageOps

    files = sorted(
        os.path.join(directory, name) for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if not files:
        raise FileNotFoundError(f"No Video Enhance anchor images were found in {directory}")
    with Image.open(files[0]) as first:
        first = ImageOps.exif_transpose(first)
        width, height = first.size
    yield width, height
    if meta_batch is not None:
        yield len(files)
    try:
        for path in files:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                yield np.asarray(image, dtype=np.float32) / 255.0
    finally:
        if meta_batch is not None and unique_id in meta_batch.inputs:
            meta_batch.inputs.pop(unique_id, None)
            meta_batch.has_closed_inputs = True


class VRGDGVideoEnhancePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {"tooltip": "Connect the full IMAGE batch from VHS Load Video. These original frames remain unchanged in the context and are used to enforce the exact final resolution and frame count."}),
                "anchor_interval": (["8 frames", "16 frames (recommended)", "24 frames", "32 frames", "48 frames", "64 frames", "96 frames", "120 frames"], {"default": "16 frames (recommended)", "tooltip": "Spacing between full-frame Z-Image guide anchors. The first and final frames are always included. Smaller spacing gives LTX more enhanced references but requires more Z-Image runs."}),
                "anchor_width": ("INT", {"default": 768, "min": 128, "max": 4096, "step": 8, "tooltip": "Requested width used only for Z-Image anchor enhancement. It does not change the source or final video resolution."}),
                "anchor_height": ("INT", {"default": 432, "min": 128, "max": 4096, "step": 8, "tooltip": "Requested height used only for Z-Image anchor enhancement. Choose dimensions matching the source aspect ratio unless using Fit with letterbox."}),
                "ltx_width": ("INT", {"default": 960, "min": 128, "max": 4096, "step": 8, "tooltip": "Temporary LTX working width. Higher values can improve spatial detail but substantially increase VRAM use and runtime. Final output is restored to the source width."}),
                "ltx_height": ("INT", {"default": 544, "min": 128, "max": 4096, "step": 8, "tooltip": "Temporary LTX working height. Higher values can improve spatial detail but substantially increase VRAM use and runtime. Final output is restored to the source height."}),
                "dimension_multiple": (["8", "16", "32 (recommended)", "64"], {"default": "32 (recommended)", "tooltip": "Rounds requested anchor and LTX dimensions to a model-friendly multiple. The resolved dimensions are exposed as outputs so downstream loaders can match them exactly."}),
                "fit_mode": (["Fit with letterbox (preserve all)", "Crop to fill", "Stretch to dimensions"], {"default": "Fit with letterbox (preserve all)", "tooltip": "Fit preserves the entire frame and adds black bars if aspect ratios differ. Crop fills the dimensions while trimming edges. Stretch changes aspect ratio and is usually not recommended."}),
                "resize_method": (["Bicubic (recommended)", "Bilinear", "Area", "Nearest"], {"default": "Bicubic (recommended)", "tooltip": "Interpolation used for temporary anchor and LTX resizing. Bicubic is a strong general default. Area can be useful for large downscales."}),
                "fallback_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001, "tooltip": "Used only when video_info is not connected or lacks FPS metadata. Connect video_info to preserve exact source timing."}),
            },
            "optional": {
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "Connect VHS Load Video video_info so the temporary LTX video uses the exact source FPS."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING", "STRING", "INT", "INT", "FLOAT", VIDEO_ENHANCE_CONTEXT)
    RETURN_NAMES = ("ltx_working_frames", "anchor_images", "anchor_count", "anchor_indices", "ltx_video_path", "ltx_width", "ltx_height", "fps", "video_enhance_context")
    FUNCTION = "prepare"
    CATEGORY = "VRGameDevGirl/Video Enhance"
    DESCRIPTION = "Creates an independent full-frame Video Enhance job, temporary LTX video, and regularly spaced Z-Image anchors while preserving the exact source frames and metadata for final restoration."

    def prepare(self, video_frames, anchor_interval, anchor_width, anchor_height, ltx_width, ltx_height,
                dimension_multiple, fit_mode, resize_method, fallback_fps, video_info=None):
        import folder_paths

        if video_frames.ndim != 4 or video_frames.shape[0] < 1:
            raise ValueError("Video Enhance Prepare requires a non-empty IMAGE batch from a video loader.")
        frame_count, source_height, source_width = map(int, video_frames.shape[:3])
        multiple = int(str(dimension_multiple).split()[0])
        anchor_width = _round_dimension(anchor_width, multiple)
        anchor_height = _round_dimension(anchor_height, multiple)
        ltx_width = _round_dimension(ltx_width, multiple)
        ltx_height = _round_dimension(ltx_height, multiple)
        fps = _fps(video_info, fallback_fps)
        interval = int(str(anchor_interval).split()[0])
        anchor_indices = list(range(0, frame_count, interval))
        if anchor_indices[-1] != frame_count - 1:
            anchor_indices.append(frame_count - 1)

        job_id = f"video_enhance_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_folder = os.path.join(folder_paths.get_output_directory(), "video_enhance", "jobs", job_id)
        anchor_folder = os.path.join(job_folder, "anchor_sources")
        ltx_frames_folder = os.path.join(job_folder, "ltx_working_frames")
        os.makedirs(job_folder, exist_ok=True)
        _log(
            f"Prepare started. Job={job_id}; source={source_width}x{source_height}x{frame_count} @ {fps:.6g} FPS; "
            f"anchors={anchor_width}x{anchor_height} every {interval}; LTX={ltx_width}x{ltx_height}; mode={fit_mode}."
        )
        ltx_frames = _resize_batch(video_frames, ltx_width, ltx_height, fit_mode, resize_method)
        anchor_source = video_frames[torch.tensor(anchor_indices, device=video_frames.device, dtype=torch.long)]
        anchors = _resize_batch(anchor_source, anchor_width, anchor_height, fit_mode, resize_method)
        _save_image_batch(anchors, anchor_folder, "anchor")
        _save_image_batch(ltx_frames, ltx_frames_folder, "frame")
        ltx_video_path = os.path.join(job_folder, "ltx_working_video.mp4")
        _encode_mp4(ltx_frames_folder, frame_count, fps, ltx_video_path)
        context = {
            "version": 1,
            "job_id": job_id,
            "job_folder": job_folder,
            "original_frames": video_frames,
            "source_width": source_width,
            "source_height": source_height,
            "frame_count": frame_count,
            "fps": fps,
            "anchor_indices": anchor_indices,
            "anchor_sources_folder": anchor_folder,
            "anchor_width": anchor_width,
            "anchor_height": anchor_height,
            "ltx_width": ltx_width,
            "ltx_height": ltx_height,
            "ltx_video_path": ltx_video_path,
            "fit_mode": fit_mode,
            "resize_method": resize_method,
        }
        _log(f"Prepare finished. Job={job_id}; anchors={len(anchor_indices)}; video={ltx_video_path}")
        return (ltx_frames, anchors, len(anchor_indices), ",".join(map(str, anchor_indices)),
                ltx_video_path, ltx_width, ltx_height, fps, context)


class VRGDGVideoEnhanceLoadAnchorsMetaBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_enhance_context": (VIDEO_ENHANCE_CONTEXT, {"tooltip": "Connect Prepare Video and Anchors context. It contains the automatically generated full-frame anchor folder."}),
                "meta_batch": ("VHS_BatchManager", {"tooltip": "Connect VHS Meta Batch Manager and set frames_per_batch from Prepare anchor_count so Z-Image receives the managed anchor sequence in one queued workflow."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", VIDEO_ENHANCE_CONTEXT)
    RETURN_NAMES = ("anchor_images", "mask", "batch_frame_count", "video_enhance_context")
    FUNCTION = "load"
    CATEGORY = "VRGameDevGirl/Video Enhance"
    DESCRIPTION = "Loads full-frame Video Enhance anchors through VHS Meta Batch for sequential Z-Image enhancement."

    def load(self, video_enhance_context, meta_batch, unique_id):
        import numpy as np

        directory = str(video_enhance_context.get("anchor_sources_folder") or "")
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Prepared Video Enhance anchor folder was not found: {directory}")
        key = str(unique_id)
        if key not in meta_batch.inputs:
            generator = _meta_image_generator(directory, meta_batch, key)
            width, height = next(generator)
            meta_batch.inputs[key] = (generator, width, height)
            meta_batch.total_frames = min(meta_batch.total_frames, next(generator))
        else:
            generator, width, height = meta_batch.inputs[key]
        chunk = itertools.islice(generator, int(meta_batch.frames_per_batch))
        images = torch.from_numpy(np.fromiter(chunk, np.dtype((np.float32, (height, width, 3)))))
        if images.shape[0] == 0:
            raise FileNotFoundError("The Video Enhance Meta Batch has no anchor images left to load.")
        masks = torch.zeros((images.shape[0], 64, 64), dtype=torch.float32)
        _log(f"Meta Batch returned {images.shape[0]} anchor(s) at {width}x{height}.")
        return images, masks, int(images.shape[0]), video_enhance_context


class VRGDGVideoEnhanceStoreAnchors:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enhanced_anchors": ("IMAGE", {"tooltip": "Connect the decoded Z-Image IMAGE result. Its count and order must match the anchors loaded through Meta Batch."}),
            "video_enhance_context": (VIDEO_ENHANCE_CONTEXT, {"tooltip": "Connect the context from Load Anchors (Meta Batch) so enhanced results are stored in the correct job."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", VIDEO_ENHANCE_CONTEXT)
    RETURN_NAMES = ("enhanced_anchor_folder", "anchor_indices", "anchor_count", "video_enhance_context")
    FUNCTION = "store"
    CATEGORY = "VRGameDevGirl/Video Enhance"
    OUTPUT_NODE = True
    DESCRIPTION = "Stores enhanced full-frame anchors in deterministic order for LTX conditioning."

    def store(self, enhanced_anchors, video_enhance_context):
        context = dict(video_enhance_context)
        indices = list(context.get("anchor_indices") or [])
        if int(enhanced_anchors.shape[0]) != len(indices):
            raise ValueError(f"Z-Image returned {enhanced_anchors.shape[0]} anchors; expected {len(indices)}.")
        folder = os.path.join(context["job_folder"], "enhanced_anchors")
        _save_image_batch(enhanced_anchors, folder, "anchor")
        context["enhanced_anchor_folder"] = folder
        _log(f"Stored {len(indices)} enhanced anchor(s) in {folder}")
        return folder, ",".join(map(str, indices)), len(indices), context


class VRGDGVideoEnhanceCollectLTXInputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prepared_video_context": (VIDEO_ENHANCE_CONTEXT, {"tooltip": "Connect Prepare Video and Anchors context. This proves the temporary full-frame LTX MP4 exists."}),
            "enhanced_anchor_context": (VIDEO_ENHANCE_CONTEXT, {"tooltip": "Connect Store Enhanced Anchors context. This proves all Z-Image anchor results were saved."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT", VIDEO_ENHANCE_CONTEXT)
    RETURN_NAMES = ("ltx_video_path", "enhanced_anchor_folder", "anchor_indices", "anchor_count", "ltx_width", "ltx_height", "video_enhance_context")
    FUNCTION = "collect"
    CATEGORY = "VRGameDevGirl/Video Enhance"
    DESCRIPTION = "Validates both Video Enhance branches, adjusts LTX-incompatible conditioning positions, and exposes the paths and dimensions required by the LTX workflow."

    @staticmethod
    def _safe_indices(indices, frame_count):
        safe, used = [], set()
        for original in indices:
            candidates = [original]
            for distance in range(1, 9):
                candidates.extend((original - distance, original + distance))
            chosen = next((value for value in candidates
                           if 0 <= value < frame_count and value not in used and value % 8 != 1), None)
            if chosen is None:
                raise ValueError(f"Could not find a safe LTX conditioning position near anchor {original}.")
            safe.append(chosen)
            used.add(chosen)
        return safe

    def collect(self, prepared_video_context, enhanced_anchor_context):
        prepared_job = str(prepared_video_context.get("job_id") or "")
        anchor_job = str(enhanced_anchor_context.get("job_id") or "")
        if not prepared_job or prepared_job != anchor_job:
            raise ValueError("The prepared video and enhanced anchors belong to different Video Enhance jobs.")
        video_path = str(prepared_video_context.get("ltx_video_path") or "")
        folder = str(enhanced_anchor_context.get("enhanced_anchor_folder") or "")
        indices = list(enhanced_anchor_context.get("anchor_indices") or [])
        frame_count = int(prepared_video_context.get("frame_count") or 0)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"The Video Enhance LTX working video is missing: {video_path}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"The enhanced Video Enhance anchor folder is missing: {folder}")
        files = sorted(name for name in os.listdir(folder) if name.lower().endswith(".png"))
        if len(files) != len(indices):
            raise ValueError(f"Enhanced anchor folder contains {len(files)} images; expected {len(indices)}.")
        safe_indices = self._safe_indices(indices, frame_count)
        changes = [f"{old}->{new}" for old, new in zip(indices, safe_indices) if old != new]
        if changes:
            _log(f"Adjusted LTX-incompatible anchor position(s): {', '.join(changes)}")
        context = dict(prepared_video_context)
        context["enhanced_anchor_folder"] = folder
        context["anchor_indices"] = safe_indices
        return (video_path, folder, ",".join(map(str, safe_indices)), len(safe_indices),
                int(context["ltx_width"]), int(context["ltx_height"]), context)


class VRGDGVideoEnhanceRestoreOriginal:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ltx_enhanced_frames": ("IMAGE", {"tooltip": "Connect the final decoded IMAGE batch from LTX. These temporary working-resolution frames are resized back to the exact source dimensions."}),
            "video_enhance_context": (VIDEO_ENHANCE_CONTEXT, {"tooltip": "Connect Collect LTX Inputs context. It contains the untouched source frames, exact source dimensions, and frame count."}),
            "resize_method": (["Bicubic (recommended)", "Bilinear", "Area", "Nearest"], {"default": "Bicubic (recommended)", "tooltip": "Interpolation used to restore LTX frames to the exact source width and height. This changes dimensions only; optional AI upscalers may be inserted before this node if desired."}),
            "enhancement_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Blends the restored LTX result with the untouched original video. 1 uses the complete LTX result; lower values retain more original pixels and can reduce over-processing."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("enhanced_video_frames", "frame_count", "original_width", "original_height", "fps")
    FUNCTION = "restore"
    CATEGORY = "VRGameDevGirl/Video Enhance"
    DESCRIPTION = "Restores decoded LTX output to the exact input resolution and frame count, preserving unmatched source-tail frames and optionally blending with the untouched source video."

    def restore(self, ltx_enhanced_frames, video_enhance_context, resize_method, enhancement_strength):
        originals = video_enhance_context.get("original_frames")
        if not isinstance(originals, torch.Tensor) or originals.ndim != 4:
            raise ValueError("Video Enhance context does not contain valid original frames.")
        source_height = int(video_enhance_context.get("source_height") or originals.shape[1])
        source_width = int(video_enhance_context.get("source_width") or originals.shape[2])
        frame_count = int(video_enhance_context.get("frame_count") or originals.shape[0])
        delta = frame_count - int(ltx_enhanced_frames.shape[0])
        if abs(delta) > 7:
            raise ValueError(f"LTX returned {ltx_enhanced_frames.shape[0]} frames for {frame_count} source frames.")
        restored = _restore_batch(
            ltx_enhanced_frames, source_width, source_height,
            str(video_enhance_context.get("fit_mode") or "Stretch to dimensions"), resize_method,
        ).to(originals.device, originals.dtype)
        output = originals.clone()
        usable = min(frame_count, int(restored.shape[0]))
        strength = float(enhancement_strength)
        output[:usable, ..., :3] = (
            originals[:usable, ..., :3] * (1.0 - strength)
            + restored[:usable, ..., :3] * strength
        )
        _log(
            f"Restore finished: {usable}/{frame_count} LTX frame(s) restored to {source_width}x{source_height}; "
            f"source tail preserved={max(0, delta)}; strength={strength:.2f}."
        )
        return output.clamp(0, 1), frame_count, source_width, source_height, float(video_enhance_context.get("fps") or 0.0)


NODE_CLASS_MAPPINGS = {
    "VRGDGVideoEnhancePrepare": VRGDGVideoEnhancePrepare,
    "VRGDGVideoEnhanceLoadAnchorsMetaBatch": VRGDGVideoEnhanceLoadAnchorsMetaBatch,
    "VRGDGVideoEnhanceStoreAnchors": VRGDGVideoEnhanceStoreAnchors,
    "VRGDGVideoEnhanceCollectLTXInputs": VRGDGVideoEnhanceCollectLTXInputs,
    "VRGDGVideoEnhanceRestoreOriginal": VRGDGVideoEnhanceRestoreOriginal,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDGVideoEnhancePrepare": "Video Enhance - Prepare Video and Anchors",
    "VRGDGVideoEnhanceLoadAnchorsMetaBatch": "Video Enhance - Load Anchors (Meta Batch)",
    "VRGDGVideoEnhanceStoreAnchors": "Video Enhance - Store Enhanced Anchors",
    "VRGDGVideoEnhanceCollectLTXInputs": "Video Enhance - Collect LTX Inputs",
    "VRGDGVideoEnhanceRestoreOriginal": "Video Enhance - Restore Original Resolution",
}
