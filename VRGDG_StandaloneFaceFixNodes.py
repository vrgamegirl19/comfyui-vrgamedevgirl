import os
import shutil
import subprocess
import time
import uuid

import torch
import torch.nn.functional as F


FACE_FIX_CONTEXT = "VRGDG_FACE_FIX_CONTEXT"


def _log(message):
    print(f"[VRGDG Face Fix] {message}", flush=True)


def _progress(index, total, label, checkpoints=10):
    if total <= 0:
        return
    step = max(1, total // checkpoints)
    if index == 0 or index + 1 == total or (index + 1) % step == 0:
        _log(f"{label}: {index + 1}/{total}")


def _meta_image_generator(directory, meta_batch=None, unique_id=None):
    import numpy as np
    from PIL import Image, ImageOps
    files = sorted(
        os.path.join(directory, name) for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if not files:
        raise FileNotFoundError(f"No anchor images were found in {directory}")
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


def _detector():
    import cv2
    assets = os.path.join(os.path.dirname(__file__), "assets")
    config = os.path.join(assets, "opencv_face_deploy.prototxt")
    model = os.path.join(assets, "opencv_face_res10_fp16.caffemodel")
    if not os.path.isfile(config) or not os.path.isfile(model):
        raise RuntimeError("VRGDG OpenCV face-detector assets are missing.")
    return cv2.dnn.readNetFromCaffe(config, model)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def _detect(net, bgr, confidence, minimum_pixels):
    import cv2
    height, width = bgr.shape[:2]
    regions = [(0, 0, width, height)]
    if width >= 600 and height >= 400:
        tile_w, tile_h = int(width * 0.60), int(height * 0.70)
        regions += [(0, 0, tile_w, tile_h), (width - tile_w, 0, width, tile_h),
                    (0, height - tile_h, tile_w, height), (width - tile_w, height - tile_h, width, height)]
    found = []
    for left, top, right, bottom in regions:
        region = bgr[top:bottom, left:right]
        rh, rw = region.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(region, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)
        for item in net.forward()[0, 0]:
            score = float(item[2])
            if score < confidence:
                continue
            x1 = max(left, left + int(float(item[3]) * rw))
            y1 = max(top, top + int(float(item[4]) * rh))
            x2 = min(right, left + int(float(item[5]) * rw))
            y2 = min(bottom, top + int(float(item[6]) * rh))
            w, h = x2 - x1, y2 - y1
            if min(w, h) >= minimum_pixels:
                found.append((float(x1), float(y1), float(w), float(h), score))
    kept = []
    for item in sorted(found, key=lambda value: value[4], reverse=True):
        if not any(_iou(item[:4], other[:4]) > 0.35 for other in kept):
            kept.append(item)
    return kept


def _detect_with_rotation(net, bgr, confidence, minimum_pixels, rotation_assist):
    import cv2
    import numpy as np
    modes = {
        "Off (fastest)": [0],
        "Light: ±15°": [0, -15, 15],
        "Strong: ±15° and ±30°": [0, -15, 15, -30, 30],
    }
    angles = modes.get(str(rotation_assist), [0])
    height, width = bgr.shape[:2]
    center = (width / 2.0, height / 2.0)
    found = []
    for angle in angles:
        if angle == 0:
            rotated = bgr
            inverse = None
        else:
            matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            rotated = cv2.warpAffine(bgr, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            inverse = cv2.invertAffineTransform(matrix)
        for x, y, w, h, score in _detect(net, rotated, confidence, minimum_pixels):
            if inverse is None:
                found.append((x, y, w, h, score))
                continue
            corners = np.array([[x, y, 1.0], [x + w, y, 1.0],
                                [x, y + h, 1.0], [x + w, y + h, 1.0]], dtype=np.float64)
            mapped = corners @ inverse.T
            x1, y1 = mapped[:, 0].min(), mapped[:, 1].min()
            x2, y2 = mapped[:, 0].max(), mapped[:, 1].max()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 > x1 and y2 > y1:
                # Slight penalty makes an equally confident upright detection win.
                found.append((x1, y1, x2 - x1, y2 - y1, score - abs(angle) * 0.0001))
    kept = []
    for item in sorted(found, key=lambda value: value[4], reverse=True):
        if not any(_iou(item[:4], other[:4]) > 0.35 for other in kept):
            kept.append(item)
    return kept


def _choose(candidates, previous, width, height):
    if not candidates:
        return None
    if previous is None:
        return max(candidates, key=lambda item: item[4])
    px, py, pw, ph = previous
    pcx, pcy = px + pw / 2, py + ph / 2
    def score(item):
        x, y, w, h, confidence = item
        distance = ((x + w / 2 - pcx) ** 2 + (y + h / 2 - pcy) ** 2) ** 0.5
        distance /= max(1.0, (width ** 2 + height ** 2) ** 0.5)
        return _iou(previous, item[:4]) * 3.0 + confidence - distance * 4.0
    return max(candidates, key=score)


def _crop_box(face, width, height, padding):
    x, y, fw, fh = face
    side = min(max(fw, fh) * (1.0 + 2.0 * padding), width, height)
    cx, cy = x + fw / 2, y + fh / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = left + int(round(side)), top + int(round(side))
    if left < 0: right -= left; left = 0
    if top < 0: bottom -= top; top = 0
    if right > width: left -= right - width; right = width
    if bottom > height: top -= bottom - height; bottom = height
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def _interval(value):
    return int(str(value).split()[0])


def _distance_repair_strength(face_width_percent, preset, custom_threshold):
    ranges = {
        "Very far faces only": (4.0, 6.0),
        "Far faces (recommended)": (7.0, 9.0),
        "Far and medium faces": (10.0, 12.0),
    }
    if preset == "All detected faces":
        return 1.0
    if preset == "Custom":
        fade_end = max(0.1, float(custom_threshold))
        full_end = max(0.0, fade_end - 2.0)
    else:
        full_end, fade_end = ranges.get(str(preset), (7.0, 9.0))
    value = float(face_width_percent)
    if value <= full_end:
        return 1.0
    if value >= fade_end:
        return 0.0
    return (fade_end - value) / max(0.001, fade_end - full_end)


class VRGDGFaceFixPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        presets = ["8 frames", "16 frames (recommended)", "24 frames", "32 frames",
                   "48 frames", "64 frames", "96 frames", "120 frames"]
        distance_presets = ["All detected faces", "Very far faces only", "Far faces (recommended)",
                            "Far and medium faces", "Custom"]
        return {"required": {
            "video_frames": ("IMAGE", {"tooltip": "Connect the IMAGE output from VHS Load Video. The complete batch is scanned for one primary face and retained for final full-resolution compositing."}),
            "detection_confidence": ("FLOAT", {"default": 0.70, "min": 0.10, "max": 0.99, "step": 0.01, "tooltip": "Minimum confidence accepted from the face detector. Higher values reduce false detections but may miss small, blurry, profile, or motion-blurred faces. Lower cautiously for distant faces. Recommended starting value: 0.70."}),
            "crop_padding": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.5, "step": 0.01, "tooltip": "Extra area around the detected face, measured as a fraction of face size on every side. Lower values crop closer to facial features; higher values include more hair, neck, and background. Recommended starting range: 0.10–0.25."}),
            "minimum_face_pixels": ("INT", {"default": 20, "min": 4, "max": 1024, "tooltip": "Reject detections whose width or height is smaller than this many source pixels. Raising this filters tiny false positives; lowering it permits very distant faces but increases false-detection risk. Recommended starting value: 20."}),
            "rotation_assist": (["Off (fastest)", "Light: ±15°", "Strong: ±15° and ±30°"], {"default": "Light: ±15°", "tooltip": "Also scans rotated copies of each frame, then maps detections back to the original coordinates. Light helps mildly tilted/overhead faces at roughly 3× detection work. Strong adds ±30° for difficult angles at roughly 5× detection work. It affects detection only; output frames are never rotated."}),
            "repair_distance": (distance_presets, {"default": "Far faces (recommended)", "tooltip": "Controls which face sizes are repaired. Very Far fully repairs below 4% of frame width and fades out by 6%. Far fully repairs below 7% and fades out by 9%. Far and Medium fully repairs below 10% and fades out by 12%. All disables distance filtering. Close faces above the fade limit remain unchanged and are not selected as Z-Image anchors."}),
            "custom_distance_threshold": ("FLOAT", {"default": 9.0, "min": 0.1, "max": 50.0, "step": 0.1, "tooltip": "Used only when Repair Distance is Custom. Faces at or above this percentage of frame width remain unchanged. Repair fades in across the 2 percentage points below this value. Example: 9% gives full repair at 7% or smaller, fading to no repair at 9%."}),
            "anchor_interval": (presets, {"default": "16 frames (recommended)", "tooltip": "Approximate spacing between Z-Image identity/detail anchors. Smaller intervals create more anchors and stronger consistency but take longer. Larger intervals are faster but give LTX less identity guidance. The nearest valid detected face is used, and boundary anchors are included automatically."}),
            "short_gap_tracking": ("INT", {"default": 2, "min": 0, "max": 8, "tooltip": "How many consecutive missed detections may reuse the last known face position. Strength fades across the gap. Use 0 to repair only freshly detected frames. The default 2 bridges brief blur without pasting a face into long no-face sections."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING", FACE_FIX_CONTEXT)
    RETURN_NAMES = ("face_video_512", "anchor_images", "anchor_count", "anchor_indices", "face_fix_context")
    RETURN_TOOLTIPS = (
        "Complete 512×512 tracked face sequence used as the LTX video source.",
        "Only the selected 512×512 anchor frames. Use the Meta Batch loader for Z-Image instead of connecting this batch directly when required by your Z-Image setup.",
        "Number of selected anchors. Connect this to VHS Meta Batch Manager frames_per_batch for the supported single-batch workflow.",
        "Comma-separated positions used by LTX optional_cond_image_indices.",
        "Internal Face Fix job data containing original frames, crop positions, detection safety, and anchor mapping.",
    )
    FUNCTION = "prepare"
    CATEGORY = "VRGameDevGirl/Face Fix"
    DESCRIPTION = "Detects and tracks one primary face through a loaded video, makes a 512×512 face sequence, selects safe Z-Image anchors, and preserves the crop mapping needed to paste repaired faces back into the original frames."

    def prepare(self, video_frames, detection_confidence, crop_padding, minimum_face_pixels,
                rotation_assist, repair_distance, custom_distance_threshold,
                anchor_interval, short_gap_tracking):
        import cv2
        if video_frames.ndim != 4 or video_frames.shape[0] < 1:
            raise ValueError("Face Fix Prepare requires a non-empty IMAGE batch from a video loader.")
        net = _detector()
        count, height, width = video_frames.shape[:3]
        _log(
            f"Prepare started: {count} frame(s), {width}x{height}, confidence={detection_confidence:.2f}, "
            f"minimum_face_pixels={minimum_face_pixels}, padding={crop_padding:.2f}, "
            f"rotation={rotation_assist}, repair_distance={repair_distance}, anchor_interval={anchor_interval}."
        )
        entries, crops = [], []
        previous = None
        misses = 0
        fresh_count = tracked_count = missing_count = close_skipped_count = 0
        for index in range(count):
            rgb = (video_frames[index, ..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            candidates = _detect_with_rotation(net, bgr, float(detection_confidence),
                                               int(minimum_face_pixels), rotation_assist)
            chosen = _choose(candidates, previous, width, height)
            fresh = chosen is not None
            if fresh:
                fresh_count += 1
                current = chosen[:4]
                previous = current if previous is None else tuple(previous[i] * 0.35 + current[i] * 0.65 for i in range(4))
                misses = 0
                tracking_strength = 1.0
            else:
                misses += 1
                if previous is None or misses > int(short_gap_tracking):
                    previous = None
                    tracking_strength = 0.0
                    missing_count += 1
                else:
                    tracking_strength = 0.65 if misses == 1 else 0.30
                    tracked_count += 1
            face_width_percent = (float(previous[2]) / float(width) * 100.0) if previous is not None else 0.0
            distance_strength = (
                _distance_repair_strength(face_width_percent, repair_distance, custom_distance_threshold)
                if previous is not None else 0.0
            )
            strength = tracking_strength * distance_strength
            if fresh and distance_strength <= 0.0:
                close_skipped_count += 1
            box = _crop_box(previous, width, height, float(crop_padding)) if previous is not None else None
            crop = None
            if box:
                left, top, right, bottom = box
                item = video_frames[index:index + 1, top:bottom, left:right, :3].permute(0, 3, 1, 2)
                crop = F.interpolate(item, size=(512, 512), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)[0].clamp(0, 1)
            entries.append({
                "index": index, "box": box, "fresh": fresh, "strength": strength,
                "tracking_strength": tracking_strength, "distance_strength": distance_strength,
                "face_width_percent": face_width_percent,
            })
            crops.append(crop)
            _progress(index, count, "Detecting/tracking faces")
        valid = [i for i, crop in enumerate(crops) if crop is not None]
        if not valid:
            raise ValueError("No face was detected in the video. Lower confidence or minimum face pixels.")
        first_valid = crops[valid[0]]
        last = first_valid
        for i in range(count):
            if crops[i] is None:
                crops[i] = last
            else:
                last = crops[i]
        step = _interval(anchor_interval)
        desired = list(range(0, count, step))
        if desired[-1] != count - 1:
            desired.append(count - 1)
        fresh_indices = [entry["index"] for entry in entries if entry["fresh"] and entry["strength"] > 0.0]
        if not fresh_indices:
            raise ValueError(
                "Faces were detected, but none are small enough for the selected Repair Distance preset. "
                "Choose a broader preset or All detected faces."
            )
        anchors = []
        for target in desired:
            nearest = min(fresh_indices, key=lambda value: abs(value - target))
            if nearest not in anchors:
                anchors.append(nearest)
        anchors.sort()
        _log(
            f"Detection complete: {fresh_count} fresh detection(s), {tracked_count} short-gap tracked frame(s), "
            f"{missing_count} confirmed no-face frame(s), {close_skipped_count} close-face frame(s) excluded, "
            f"{len(anchors)} anchor(s) at [{','.join(str(v) for v in anchors)}]."
        )
        context = {
            "version": 1, "job_id": f"standalone_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            "original_frames": video_frames, "entries": entries, "anchor_indices": anchors,
            "frame_count": int(count), "width": int(width), "height": int(height),
        }
        crop_batch = torch.stack(crops)
        anchor_batch = crop_batch[torch.tensor(anchors, device=crop_batch.device, dtype=torch.long)]
        import cv2
        import folder_paths
        source_folder = os.path.join(folder_paths.get_output_directory(), "face_fix_standalone",
                                     context["job_id"], "anchor_sources_512")
        os.makedirs(source_folder, exist_ok=True)
        for order, image in enumerate(anchor_batch):
            rgb = (image[..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")
            cv2.imwrite(os.path.join(source_folder, f"anchor_{order:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            _progress(order, len(anchor_batch), "Saving source anchors")
        context["anchor_sources_folder"] = source_folder
        _log(f"Prepare finished. Job={context['job_id']}; source anchors={source_folder}")
        return crop_batch, anchor_batch, len(anchors), ",".join(str(value) for value in anchors), context


class VRGDGFaceFixLoadAnchorsMetaBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_fix_context": (FACE_FIX_CONTEXT, {"tooltip": "Connect Face Fix - Prepare Video and Anchors: face_fix_context. It contains the automatically generated anchor-source folder."}),
                "meta_batch": ("VHS_BatchManager", {"tooltip": "Connect VHS Meta Batch Manager. Set frames_per_batch from Prepare's anchor_count so all anchors follow the same managed loading behavior used by the tested Z-Image workflow."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", FACE_FIX_CONTEXT)
    RETURN_NAMES = ("anchor_images", "mask", "batch_frame_count", "face_fix_context")
    RETURN_TOOLTIPS = (
        "The current Meta Batch of 512×512 source anchors. Connect this to the Z-Image pixels/VAE Encode input.",
        "Blank compatibility mask supplied for workflows that require a MASK output.",
        "Number of anchors loaded in this Meta Batch execution.",
        "The unchanged Face Fix context. Connect this to Store Enhanced Anchors to preserve execution order.",
    )
    FUNCTION = "load"
    CATEGORY = "VRGameDevGirl/Face Fix"
    DESCRIPTION = "Loads prepared Face Fix anchors through VHS Meta Batch without manually selecting a folder."

    def load(self, face_fix_context, meta_batch, unique_id):
        import itertools
        import numpy as np
        directory = str(face_fix_context.get("anchor_sources_folder") or "")
        _log(
            f"Meta Batch anchor loader started. Job={face_fix_context.get('job_id', 'unknown')}; "
            f"frames_per_batch={getattr(meta_batch, 'frames_per_batch', '?')}; folder={directory}"
        )
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Prepared Face Fix anchor folder was not found: {directory}")
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
            raise FileNotFoundError("The Face Fix Meta Batch has no anchor images left to load.")
        masks = torch.zeros((images.shape[0], 64, 64), dtype=torch.float32)
        _log(
            f"Meta Batch anchor loader returned {images.shape[0]} image(s) at "
            f"{images.shape[2]}x{images.shape[1]}. closed={getattr(meta_batch, 'has_closed_inputs', False)}"
        )
        return images, masks, int(images.shape[0]), face_fix_context


class VRGDGFaceFixStoreAnchors:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enhanced_anchors": ("IMAGE", {"tooltip": "Connect the decoded IMAGE output from Z-Image. The number and order must exactly match the anchors loaded by the Face Fix Meta Batch node."}),
            "face_fix_context": (FACE_FIX_CONTEXT, {"tooltip": "Connect Face Fix - Load Anchors (Meta Batch): face_fix_context. This links the enhanced results to the correct Face Fix job and anchor ordering."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", FACE_FIX_CONTEXT)
    RETURN_NAMES = ("enhanced_anchor_folder", "anchor_indices", "anchor_count", "face_fix_context")
    RETURN_TOOLTIPS = (
        "Automatically managed folder containing the numbered Z-Image enhanced anchors for LTX.",
        "Comma-separated LTX conditioning positions corresponding to the saved anchors.",
        "Number of enhanced anchors validated and saved.",
        "Updated job context containing the enhanced-anchor folder. Connect this to Collect LTX Inputs: enhanced_anchor_context.",
    )
    FUNCTION = "store"
    CATEGORY = "VRGameDevGirl/Face Fix"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves the enhanced anchor batch in deterministic order for LTX conditioning."

    def store(self, enhanced_anchors, face_fix_context):
        import cv2
        import folder_paths
        context = dict(face_fix_context)
        indices = list(context.get("anchor_indices") or [])
        _log(
            f"Store Enhanced Anchors started. Job={context.get('job_id', 'unknown')}; "
            f"received={enhanced_anchors.shape[0]}, expected={len(indices)}."
        )
        if enhanced_anchors.shape[0] != len(indices):
            raise ValueError(f"Z-Image returned {enhanced_anchors.shape[0]} anchors; expected {len(indices)}.")
        folder = os.path.join(folder_paths.get_output_directory(), "face_fix_standalone",
                              context["job_id"], "enhanced_anchors_512")
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):
            if old.lower().endswith(".png"):
                os.remove(os.path.join(folder, old))
        for order, image in enumerate(enhanced_anchors):
            rgb = (image[..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")
            cv2.imwrite(os.path.join(folder, f"anchor_{order:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            _progress(order, len(enhanced_anchors), "Saving enhanced anchors")
        context["enhanced_anchor_folder"] = folder
        _log(f"Store Enhanced Anchors finished: {len(indices)} image(s) saved to {folder}")
        return folder, ",".join(str(value) for value in indices), len(indices), context


class VRGDGFaceFixCreateCropVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_video_512": ("IMAGE", {"tooltip": "Connect Prepare Video and Anchors: face_video_512. These frames are encoded into the silent 512×512 MP4 consumed by LTX Load Video (Path)."}),
                "face_fix_context": (FACE_FIX_CONTEXT, {"tooltip": "Connect a Face Fix context from the current Prepare/Meta Batch branch so the video is saved inside the correct unique job folder."}),
                "fallback_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001, "tooltip": "Frame rate used only when video_info is not connected or has no valid FPS. It must match the source video or repaired motion/audio timing will drift. Connect video_info whenever possible."}),
            },
            "optional": {"video_info": ("VHS_VIDEOINFO", {"tooltip": "Recommended: connect VHS Load Video's video_info output. The node automatically reads loaded_fps/source_fps so the cropped face video matches the original timing."})},
        }

    RETURN_TYPES = ("STRING", FACE_FIX_CONTEXT, "FLOAT", "INT")
    RETURN_NAMES = ("cropped_face_video_path", "face_fix_context", "fps", "frame_count")
    RETURN_TOOLTIPS = (
        "Path to the generated silent 512×512 face MP4. Collect LTX Inputs exposes this after both preparation branches finish.",
        "Updated context containing the cropped video path and FPS. Connect to Collect LTX Inputs: cropped_video_context.",
        "Actual FPS used to encode the cropped face video.",
        "Number of frames written to the cropped face video.",
    )
    FUNCTION = "create"
    CATEGORY = "VRGameDevGirl/Face Fix"
    DESCRIPTION = "Writes the prepared 512px face sequence to a silent MP4 for LTX Load Video (Path)."

    @staticmethod
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

    def create(self, face_video_512, face_fix_context, fallback_fps, video_info=None):
        import cv2
        import folder_paths
        if face_video_512.ndim != 4 or face_video_512.shape[0] < 1:
            raise ValueError("Create Crop Video requires a non-empty face IMAGE batch.")
        context = dict(face_fix_context)
        _log(
            f"Create Cropped Face Video started. Job={context.get('job_id', 'unknown')}; "
            f"frames={face_video_512.shape[0]}."
        )
        job_folder = os.path.join(folder_paths.get_output_directory(), "face_fix_standalone", context["job_id"])
        frames_folder = os.path.join(job_folder, "face_video_frames_512")
        os.makedirs(frames_folder, exist_ok=True)
        for old in os.listdir(frames_folder):
            if old.lower().endswith(".png"):
                os.remove(os.path.join(frames_folder, old))
        for index, image in enumerate(face_video_512):
            rgb = (image[..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")
            if rgb.shape[0] != 512 or rgb.shape[1] != 512:
                rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(os.path.join(frames_folder, f"frame_{index:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            _progress(index, len(face_video_512), "Writing cropped face frames")
        fps = self._fps(video_info, fallback_fps)
        output_path = os.path.join(job_folder, "face_video_512.mp4")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:
                raise RuntimeError("FFmpeg is required to create the cropped Face Fix video.") from exc
        command = [
            ffmpeg, "-y", "-framerate", f"{fps:.12g}",
            "-i", os.path.join(frames_folder, "frame_%06d.png"),
            "-frames:v", str(int(face_video_512.shape[0])), "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", "10",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path,
        ]
        _log(f"Encoding cropped face MP4 at {fps:.6g} FPS with FFmpeg...")
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
        if result.returncode != 0 or not os.path.isfile(output_path):
            details = (result.stderr or result.stdout or "FFmpeg failed").strip()
            raise RuntimeError(f"Could not create the cropped Face Fix MP4: {details[-1600:]}")
        context["crop_video_path"] = output_path
        context["fps"] = fps
        _log(f"Create Cropped Face Video finished: {output_path}")
        return output_path, context, fps, int(face_video_512.shape[0])


class VRGDGFaceFixComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ltx_face_frames": ("IMAGE", {"tooltip": "Connect the final IMAGE batch from the LTX VAE Decode node. Do not connect source crops or anchor images here."}),
            "face_fix_context": (FACE_FIX_CONTEXT, {"tooltip": "Connect Collect LTX Inputs: face_fix_context. It supplies original full-resolution frames, crop rectangles, and no-face safety decisions."}),
            "feather_pixels": ("INT", {"default": 18, "min": 0, "max": 256, "tooltip": "Softens the boundary where each repaired face crop meets the original frame. Higher values create a wider, smoother transition; too high may weaken facial detail. Lower values are sharper but can reveal a visible edge. Recommended starting value: 18."}),
            "color_match": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Shifts the repaired crop's average color toward the original face region before blending. 0 disables matching; 1 applies the full measured correction. Increase for lighting/color seams, decrease if skin tone becomes dull or unstable. Recommended: 0.65."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("repaired_video_frames", "applied_face_mask", "repaired_frame_count")
    RETURN_TOOLTIPS = (
        "Original full-resolution video frame batch with safe repaired faces composited in. Connect to VHS Video Combine images.",
        "Per-frame grayscale mask showing exactly where and how strongly Face Fix was applied. Optional diagnostic output.",
        "Number of frames that received a nonzero repaired-face composite. Optional diagnostic output.",
    )
    FUNCTION = "composite"
    CATEGORY = "VRGameDevGirl/Face Fix"
    DESCRIPTION = "Feathers LTX face frames back into the original video frames; no-face and short LTX tail frames remain unchanged."

    def composite(self, ltx_face_frames, face_fix_context, feather_pixels, color_match):
        originals = face_fix_context["original_frames"]
        entries = face_fix_context["entries"]
        delta = len(entries) - int(ltx_face_frames.shape[0])
        _log(
            f"Composite started. Job={face_fix_context.get('job_id', 'unknown')}; "
            f"source_frames={len(entries)}, LTX_frames={ltx_face_frames.shape[0]}, delta={delta}, "
            f"feather={feather_pixels}, color_match={color_match:.2f}."
        )
        if abs(delta) > 7:
            raise ValueError(f"LTX returned {ltx_face_frames.shape[0]} frames for {len(entries)} source frames.")
        output = originals.clone()
        masks = torch.zeros((originals.shape[0], originals.shape[1], originals.shape[2]), device=originals.device, dtype=originals.dtype)
        repaired = 0
        usable = min(len(entries), int(ltx_face_frames.shape[0]))
        for index in range(usable):
            entry = entries[index]
            box, strength = entry.get("box"), float(entry.get("strength", 0.0))
            if not box or strength <= 0:
                continue
            left, top, right, bottom = box
            h, w = bottom - top, right - left
            face = ltx_face_frames[index:index + 1, ..., :3].to(output.device, output.dtype).permute(0, 3, 1, 2)
            face = F.interpolate(face, size=(h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)[0].clamp(0, 1)
            yy = torch.linspace(-1, 1, h, device=output.device, dtype=output.dtype)[:, None]
            xx = torch.linspace(-1, 1, w, device=output.device, dtype=output.dtype)[None, :]
            radial = 1.0 - torch.sqrt(xx * xx + yy * yy)
            feather_scale = max(1.0, float(feather_pixels) / max(1.0, min(w, h) / 2.0))
            alpha = torch.clamp(radial / feather_scale, 0, 1) * strength
            target = output[index, top:bottom, left:right, :3]
            selected = alpha > 0.35
            if color_match > 0 and int(selected.sum()) >= 16:
                face = torch.clamp(face + (target[selected].mean(0) - face[selected].mean(0)) * float(color_match), 0, 1)
            output[index, top:bottom, left:right, :3] = target * (1 - alpha[..., None]) + face * alpha[..., None]
            masks[index, top:bottom, left:right] = alpha
            repaired += 1
            _progress(index, usable, "Compositing repaired faces")
        _log(
            f"Composite finished: repaired={repaired}, unchanged={len(entries) - repaired}, "
            f"preserved_LTX_tail={max(0, delta)}."
        )
        return output.clamp(0, 1), masks, repaired


class VRGDGFaceFixLTXInputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cropped_video_context": (FACE_FIX_CONTEXT, {"tooltip": "Connect Create Cropped Face Video: face_fix_context. This proves the 512×512 source MP4 was created."}),
            "enhanced_anchor_context": (FACE_FIX_CONTEXT, {"tooltip": "Connect Store Enhanced Anchors: face_fix_context. This proves all required Z-Image anchors were validated and saved."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", FACE_FIX_CONTEXT)
    RETURN_NAMES = ("cropped_face_video_path", "enhanced_anchor_folder", "anchor_indices",
                    "anchor_count", "face_fix_context")
    RETURN_TOOLTIPS = (
        "Validated 512×512 cropped-face MP4 path. Connect to LTX Load Video (Path).",
        "Validated folder of ordered enhanced anchor PNGs. Connect to Load Images From Folder.",
        "Comma-separated anchor frame positions. Connect to LTX optional_cond_image_indices.",
        "Validated number of enhanced anchors. Optional diagnostic output.",
        "Merged context from both completed branches. Connect to Composite Repaired Video: face_fix_context.",
    )
    FUNCTION = "collect"
    CATEGORY = "VRGameDevGirl/Face Fix"
    DESCRIPTION = "Execution barrier that validates and exposes the completed cropped video and enhanced anchors for LTX."

    @classmethod
    def _safe_indices(cls, indices, frame_count):
        safe, used = [], set()
        for original in indices:
            candidates = [original]
            for distance in range(1, 9):
                candidates.extend((original - distance, original + distance))
            chosen = None
            for candidate in candidates:
                if candidate < 0 or candidate >= frame_count or candidate in used:
                    continue
                if candidate % 8 != 1:
                    chosen = candidate
                    break
            if chosen is None:
                raise ValueError(f"Could not find a safe LTX conditioning position near anchor {original}.")
            safe.append(chosen)
            used.add(chosen)
        return safe

    def collect(self, cropped_video_context, enhanced_anchor_context):
        crop_job = str(cropped_video_context.get("job_id") or "")
        anchor_job = str(enhanced_anchor_context.get("job_id") or "")
        if not crop_job or crop_job != anchor_job:
            raise ValueError("The cropped video and enhanced anchors belong to different Face Fix jobs.")
        video_path = str(cropped_video_context.get("crop_video_path") or "")
        folder = str(enhanced_anchor_context.get("enhanced_anchor_folder") or "")
        indices = list(enhanced_anchor_context.get("anchor_indices") or [])
        frame_count = int(cropped_video_context.get("frame_count") or 0)
        _log(
            f"Collect LTX Inputs started. Job={crop_job}; frames={frame_count}; "
            f"requested anchors=[{','.join(str(v) for v in indices)}]."
        )
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"The cropped Face Fix video is missing: {video_path}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"The enhanced Face Fix anchor folder is missing: {folder}")
        files = sorted(name for name in os.listdir(folder) if name.lower().endswith(".png"))
        if len(files) != len(indices):
            raise ValueError(f"Enhanced anchor folder contains {len(files)} images; expected {len(indices)}.")
        original_indices = list(indices)
        indices = self._safe_indices(indices, frame_count)
        changes = [f"{old}->{new}" for old, new in zip(original_indices, indices) if old != new]
        if changes:
            _log(f"Adjusted LTX-incompatible anchor position(s): {', '.join(changes)}")
        else:
            _log("All anchor positions are compatible with LTX guiding latents.")
        context = dict(cropped_video_context)
        context["enhanced_anchor_folder"] = folder
        context["anchor_indices"] = indices
        _log(
            f"Collect LTX Inputs finished: video={video_path}; anchors={folder}; "
            f"indices=[{','.join(str(v) for v in indices)}]."
        )
        return video_path, folder, ",".join(str(value) for value in indices), len(indices), context


NODE_CLASS_MAPPINGS = {
    "VRGDGFaceFixPrepare": VRGDGFaceFixPrepare,
    "VRGDGFaceFixLoadAnchorsMetaBatch": VRGDGFaceFixLoadAnchorsMetaBatch,
    "VRGDGFaceFixStoreAnchors": VRGDGFaceFixStoreAnchors,
    "VRGDGFaceFixCreateCropVideo": VRGDGFaceFixCreateCropVideo,
    "VRGDGFaceFixLTXInputs": VRGDGFaceFixLTXInputs,
    "VRGDGFaceFixComposite": VRGDGFaceFixComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDGFaceFixPrepare": "Face Fix - Prepare Video and Anchors",
    "VRGDGFaceFixLoadAnchorsMetaBatch": "Face Fix - Load Anchors (Meta Batch)",
    "VRGDGFaceFixStoreAnchors": "Face Fix - Store Enhanced Anchors",
    "VRGDGFaceFixCreateCropVideo": "Face Fix - Create Cropped Face Video",
    "VRGDGFaceFixLTXInputs": "Face Fix - Collect LTX Inputs",
    "VRGDGFaceFixComposite": "Face Fix - Composite Repaired Video",
}
