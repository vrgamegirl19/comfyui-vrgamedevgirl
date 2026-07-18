import base64
import json
import math
import os
import shutil
import subprocess
import time
import uuid

from aiohttp import web


_ROUTES_REGISTERED = False


def _absolute_existing_file(value, label):
    path = os.path.abspath(os.path.normpath(str(value or "").strip().strip('"')))
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def _project_folder(value, video_path):
    raw = str(value or "").strip().strip('"')
    folder = os.path.abspath(os.path.normpath(raw)) if raw else os.path.dirname(video_path)
    os.makedirs(folder, exist_ok=True)
    return folder


def _payload_number(payload, key, default):
    value = payload.get(key)
    return float(default if value is None or str(value).strip() == "" else value)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _expanded_region(box, width, height, scale=4.0):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * scale
    left = max(0, int(round(cx - side / 2.0)))
    top = max(0, int(round(cy - side / 2.0)))
    right = min(width, int(round(cx + side / 2.0)))
    bottom = min(height, int(round(cy + side / 2.0)))
    return (left, top, max(left + 1, right), max(top + 1, bottom))


def _initial_regions(width, height):
    regions = [(0, 0, width, height)]
    if width >= 600 and height >= 400:
        tile_w, tile_h = int(round(width * 0.60)), int(round(height * 0.70))
        regions.extend([
            (0, 0, tile_w, tile_h),
            (width - tile_w, 0, width, tile_h),
            (0, height - tile_h, tile_w, height),
            (width - tile_w, height - tile_h, width, height),
        ])
    return regions


def _detect(net, frame, confidence, regions):
    import cv2

    found = []
    for left, top, right, bottom in regions:
        region = frame[top:bottom, left:right]
        region_h, region_w = region.shape[:2]
        if region_w < 8 or region_h < 8:
            continue
        if net.get("kind") == "yunet":
            detector = net["net"]
            detector.setInputSize((region_w, region_h))
            result = detector.detect(region)
            faces = result[1] if isinstance(result, tuple) and len(result) > 1 else result
            for detection in (() if faces is None else faces):
                score = float(detection[-1])
                if score < confidence:
                    continue
                x = max(left, left + int(round(float(detection[0]))))
                y = max(top, top + int(round(float(detection[1]))))
                x2 = min(right, x + int(round(float(detection[2]))))
                y2 = min(bottom, y + int(round(float(detection[3]))))
                if x2 > x and y2 > y:
                    found.append((float(x), float(y), float(x2 - x), float(y2 - y), score))
        else:
            detector = net["net"]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(region, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False,
            )
            detector.setInput(blob)
            output = detector.forward()
            for detection in output[0, 0]:
                score = float(detection[2])
                if score < confidence:
                    continue
                x = max(left, left + int(round(float(detection[3]) * region_w)))
                y = max(top, top + int(round(float(detection[4]) * region_h)))
                x2 = min(right, left + int(round(float(detection[5]) * region_w)))
                y2 = min(bottom, top + int(round(float(detection[6]) * region_h)))
                if x2 > x and y2 > y:
                    found.append((float(x), float(y), float(x2 - x), float(y2 - y), score))
    kept = []
    for item in sorted(found, key=lambda value: value[4], reverse=True):
        if not any(_iou(item[:4], other[:4]) > 0.35 for other in kept):
            kept.append(item)
    return kept


def _detect_with_rotation(net, frame, confidence, regions, rotation_assist):
    import cv2
    import numpy as np

    modes = {
        "off": [0],
        "light": [0, -15, 15],
        "strong": [0, -15, 15, -30, 30],
    }
    angles = modes.get(str(rotation_assist or "light").lower(), [0, -15, 15])
    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    found = []
    for angle in angles:
        if angle == 0:
            rotated = frame
            inverse = None
            scan_regions = regions
        else:
            matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            rotated = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            inverse = cv2.invertAffineTransform(matrix)
            scan_regions = _initial_regions(width, height)
        for x, y, box_width, box_height, score in _detect(net, rotated, confidence, scan_regions):
            if inverse is None:
                found.append((x, y, box_width, box_height, score))
                continue
            corners = np.array([
                [x, y, 1.0], [x + box_width, y, 1.0],
                [x, y + box_height, 1.0], [x + box_width, y + box_height, 1.0],
            ], dtype=np.float64)
            mapped = corners @ inverse.T
            left, top = max(0.0, mapped[:, 0].min()), max(0.0, mapped[:, 1].min())
            right, bottom = min(float(width), mapped[:, 0].max()), min(float(height), mapped[:, 1].max())
            if right > left and bottom > top:
                found.append((left, top, right - left, bottom - top, score - abs(angle) * 0.0001))
    kept = []
    for item in sorted(found, key=lambda value: value[4], reverse=True):
        if not any(_iou(item[:4], other[:4]) > 0.35 for other in kept):
            kept.append(item)
    return kept


def _distance_repair_strength(face_width_percent, preset, custom_threshold):
    ranges = {
        "very_far": (4.0, 6.0),
        "far": (7.0, 9.0),
        "far_medium": (10.0, 12.0),
    }
    preset = str(preset or "far").lower()
    if preset == "all":
        return 1.0
    if preset == "custom":
        fade_end = max(0.1, float(custom_threshold))
        full_end = max(0.0, fade_end - 2.0)
    else:
        full_end, fade_end = ranges.get(preset, (7.0, 9.0))
    value = float(face_width_percent)
    if value <= full_end:
        return 1.0
    if value >= fade_end:
        return 0.0
    return (fade_end - value) / max(0.001, fade_end - full_end)


def _select_tracked(candidates, previous, frame_width, frame_height, minimum_pixels):
    candidates = [item for item in candidates if min(item[2], item[3]) >= minimum_pixels]
    if not candidates:
        return None
    if previous is None:
        return max(candidates, key=lambda item: item[4])
    px, py, pw, ph = previous
    pcx, pcy = px + pw / 2.0, py + ph / 2.0

    def score(item):
        x, y, w, h, confidence = item
        cx, cy = x + w / 2.0, y + h / 2.0
        distance = math.hypot(cx - pcx, cy - pcy) / max(1.0, math.hypot(frame_width, frame_height))
        size_delta = abs(math.log(max(1.0, w * h) / max(1.0, pw * ph)))
        return _iou(previous, item[:4]) * 3.0 + confidence - distance * 4.0 - size_delta * 0.35

    return max(candidates, key=score)


def _smooth_box(previous, current, alpha=0.65):
    if previous is None:
        return tuple(float(value) for value in current[:4])
    return tuple(previous[index] * (1.0 - alpha) + float(current[index]) * alpha for index in range(4))


def _square_crop_box(face_box, width, height, padding):
    x, y, face_w, face_h = face_box
    cx, cy = x + face_w / 2.0, y + face_h / 2.0
    side = max(face_w, face_h) * (1.0 + 2.0 * max(0.0, padding))
    side = min(side, width, height)
    left, top = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    right, bottom = left + int(round(side)), top + int(round(side))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return (max(0, left), max(0, top), min(width, right), min(height, bottom))


def _is_forbidden_ltx_conditioning_index(index):
    return int(index) % 8 == 1


def _safe_ltx_conditioning_indices(indices, frame_count):
    """Move LTX guide-image indices away from forbidden 8n+1 positions."""
    count = max(0, int(frame_count or 0))
    if count <= 0:
        return []
    safe = []
    used = set()
    for raw in indices or []:
        original = max(0, min(count - 1, int(raw)))
        candidates = sorted(
            (index for index in range(count) if not _is_forbidden_ltx_conditioning_index(index) and index not in used),
            key=lambda index: (abs(index - original), index),
        )
        if not candidates:
            continue
        selected = candidates[0]
        safe.append(selected)
        used.add(selected)
    return safe


def _anchor_indices(frame_count, interval):
    count = max(0, int(frame_count or 0))
    if count <= 0:
        return []
    step = max(1, min(240, int(interval or 16)))
    indices = list(range(0, count, step))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return _safe_ltx_conditioning_indices(indices, count)


def _encode_crop_video(crops_folder, output_path, fps, start_frame, frame_count):
    command = [
        _find_ffmpeg(), "-y",
        "-framerate", f"{float(fps):.12g}",
        "-start_number", str(int(start_frame)),
        "-i", os.path.join(crops_folder, "frame_%06d.png"),
        "-frames:v", str(int(frame_count)),
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "10",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not os.path.isfile(output_path):
        details = (completed.stderr or completed.stdout or "Unknown FFmpeg error").strip()
        raise RuntimeError(f"Could not create the 512x512 Face Fix crop video: {details[-1600:]}")
    return output_path


def estimate_face_fix_anchors(payload):
    import cv2

    video_path = _absolute_existing_file(payload.get("video_path"), "Scene video")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open scene video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("The scene video has invalid frame metadata.")
    if bool(payload.get("whole_scene", False)):
        start_frame, end_frame = 0, total_frames - 1
    else:
        start_time = max(0.0, _payload_number(payload, "in_time", 0.0))
        end_time = max(start_time, _payload_number(payload, "out_time", start_time))
        start_frame = min(max(0, int(math.floor(start_time * fps))), total_frames - 1)
        end_frame = min(max(start_frame, int(math.ceil(end_time * fps))), total_frames - 1)
    frame_count = end_frame - start_frame + 1
    interval = max(1, min(240, int(_payload_number(payload, "anchor_interval", 16))))
    indices = _anchor_indices(frame_count, interval)
    return {
        "fps": fps,
        "total_video_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": frame_count,
        "anchor_interval": interval,
        "anchor_count": len(indices),
        "anchor_indices": indices,
        "anchor_indices_text": ",".join(str(index) for index in indices),
    }


def prepare_face_fix(payload):
    import cv2

    video_path = _absolute_existing_file(payload.get("video_path"), "Scene video")
    project_folder = _project_folder(payload.get("project_folder"), video_path)
    start_time = max(0.0, float(payload.get("in_time") or 0.0))
    end_time = max(start_time, float(payload.get("out_time") if payload.get("out_time") is not None else start_time))
    whole_scene = bool(payload.get("whole_scene", False))
    preview_only = str(payload.get("mode") or "range") == "frame"
    confidence = max(0.1, min(0.99, float(payload.get("confidence") or 0.70)))
    padding = max(0.0, min(2.0, float(payload.get("crop_padding_factor") or 0.10)))
    minimum_pixels = max(4, int(payload.get("minimum_face_pixels") or 20))
    rotation_assist = str(payload.get("rotation_assist") or "light").lower()
    repair_distance = str(payload.get("repair_distance") or "far").lower()
    custom_distance_threshold = max(0.1, min(50.0, float(payload.get("custom_distance_threshold") or 9.0)))
    enhance_size = 512
    ltx_settings = {
        "guiding_strength": max(0.0, min(2.0, _payload_number(payload, "ltx_guiding_strength", 0.20))),
        "temporal_overlap_cond_strength": max(0.0, min(2.0, _payload_number(payload, "ltx_temporal_overlap_cond_strength", 0.50))),
        "cond_image_strength": max(0.0, min(2.0, _payload_number(payload, "ltx_cond_image_strength", 0.50))),
        "seed": max(0, int(payload.get("seed") or 42)),
        "sampler": str(payload.get("ltx_sampler") or "euler_ancestral").strip(),
        "sigmas": str(payload.get("ltx_sigmas") or "0.909375, 0.725, 0.421875, 0.0").strip(),
    }

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open scene video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The scene video has invalid frame metadata.")
    if whole_scene and not preview_only:
        start_time = 0.0
        end_time = max(0.0, (total_frames - 1) / fps)
        start_frame = 0
        end_frame = max(0, total_frames - 1)
    else:
        start_frame = min(max(0, int(math.floor(start_time * fps))), max(0, total_frames - 1))
        end_frame = start_frame if preview_only else min(max(start_frame, int(math.ceil(end_time * fps))), max(0, total_frames - 1))
    if end_frame - start_frame + 1 > 1800:
        capture.release()
        raise ValueError("Face Fix currently supports at most 1,800 frames per range.")

    job_id = f"face_fix_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_folder = os.path.join(project_folder, "face_fix", "jobs", job_id)
    originals_folder = os.path.join(job_folder, "original_frames")
    crops_folder = os.path.join(job_folder, "crops_512")
    enhanced_folder = os.path.join(job_folder, "enhanced_512")
    anchor_sources_folder = os.path.join(job_folder, "anchor_sources_512")
    enhanced_anchors_folder = os.path.join(job_folder, "enhanced_anchors_512")
    os.makedirs(originals_folder, exist_ok=True)
    os.makedirs(crops_folder, exist_ok=True)
    os.makedirs(enhanced_folder, exist_ok=True)
    os.makedirs(anchor_sources_folder, exist_ok=True)
    os.makedirs(enhanced_anchors_folder, exist_ok=True)

    assets = os.path.join(os.path.dirname(__file__), "assets")
    config_path = os.path.join(assets, "opencv_face_deploy.prototxt")
    model_path = os.path.join(assets, "opencv_face_res10_fp16.caffemodel")
    yunet_path = os.path.join(assets, "face_detection_yunet_2023mar.onnx")
    detector = None
    caffe_error = None
    if os.path.isfile(config_path) and os.path.isfile(model_path):
        try:
            caffe_loader = getattr(cv2.dnn, "readNetFromCaffe", None)
            if callable(caffe_loader):
                detector = {"kind": "caffe", "net": caffe_loader(config_path, model_path)}
            else:
                generic_loader = getattr(cv2.dnn, "readNet", None)
                if callable(generic_loader):
                    detector = {"kind": "caffe", "net": generic_loader(model_path, config_path)}
        except Exception as exc:
            caffe_error = exc
    if detector is None and os.path.isfile(yunet_path):
        face_detector_yn = getattr(cv2, "FaceDetectorYN", None)
        yunet_loader = getattr(face_detector_yn, "create", None) if face_detector_yn is not None else None
        if not callable(yunet_loader):
            yunet_loader = getattr(cv2, "FaceDetectorYN_create", None)
        if callable(yunet_loader):
            try:
                detector = {"kind": "yunet", "net": yunet_loader(yunet_path, "", (320, 320), 0.1, 0.3, 5000)}
            except Exception as exc:
                capture.release()
                raise RuntimeError(f"OpenCV could not load the YuNet ONNX face detector: {exc}") from exc
    if detector is None:
        capture.release()
        detail = f" Caffe loader error: {caffe_error}" if caffe_error else ""
        raise RuntimeError(f"Face Fix could not load a compatible OpenCV face detector.{detail}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    entries = []
    previous_face = None
    missed_count = 0
    active_run = None
    next_run_index = 0
    carried_frames = 0
    skipped_frames = 0
    close_skipped_frames = 0
    for frame_number in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if previous_face is None:
            regions = _initial_regions(width, height)
        else:
            regions = [_expanded_region(previous_face, width, height, 4.5)]
        candidates = _detect_with_rotation(detector, frame, confidence, regions, rotation_assist)
        chosen = _select_tracked(candidates, previous_face, width, height, minimum_pixels)
        detected = chosen is not None
        if detected:
            missed_count = 0
            if active_run is None:
                active_run = next_run_index
                next_run_index += 1
        elif previous_face is not None and missed_count < 2:
            missed_count += 1
            chosen = (*previous_face, 0.0)
            carried_frames += 1
        else:
            chosen = None
            previous_face = None
            missed_count = 0
            active_run = None
            skipped_frames += 1
        base_name = f"frame_{frame_number:06d}.png"
        original_path = os.path.join(originals_folder, base_name)
        cv2.imwrite(original_path, frame)
        tracking_strength = 1.0 if detected else (0.65 if missed_count == 1 else (0.30 if missed_count == 2 else 0.0))
        entry = {
            "index": len(entries),
            "frame_number": frame_number,
            "time": frame_number / fps,
            "original_path": original_path,
            "detected": detected,
            "carried": bool(chosen is not None and not detected),
            "missed_count": missed_count if chosen is not None and not detected else 0,
            "run_index": active_run,
            "confidence": float(chosen[4]) if chosen is not None else 0.0,
        }
        if chosen is not None:
            previous_face = _smooth_box(previous_face, chosen)
            face_width_percent = float(previous_face[2]) / float(width) * 100.0
            distance_strength = _distance_repair_strength(
                face_width_percent, repair_distance, custom_distance_threshold
            )
            entry["tracking_strength"] = tracking_strength
            entry["distance_strength"] = distance_strength
            entry["face_width_percent"] = face_width_percent
            entry["composite_strength"] = tracking_strength * distance_strength
            if detected and distance_strength <= 0.0:
                close_skipped_frames += 1
            crop_box = _square_crop_box(previous_face, width, height, padding)
            left, top, right, bottom = crop_box
            crop = frame[top:bottom, left:right]
            resized = cv2.resize(crop, (enhance_size, enhance_size), interpolation=cv2.INTER_LANCZOS4)
            crop_path = os.path.join(crops_folder, base_name)
            cv2.imwrite(crop_path, resized)
            entry.update({
                "crop_path": crop_path,
                "enhanced_path": os.path.join(enhanced_folder, base_name),
                "crop_box": list(crop_box),
                "face_box": [round(value, 3) for value in previous_face],
            })
        else:
            entry.update({
                "tracking_strength": 0.0, "distance_strength": 0.0,
                "face_width_percent": 0.0, "composite_strength": 0.0,
            })
        entries.append(entry)
    capture.release()
    if not entries:
        raise RuntimeError("No frames were extracted from the selected Face Fix range.")

    anchor_interval = max(1, min(240, int(payload.get("anchor_interval") or 16)))
    runs = []
    anchors = []
    for run_index in range(next_run_index):
        run_entries = [entry for entry in entries if entry.get("run_index") == run_index]
        if not run_entries:
            continue
        run_folder = os.path.join(job_folder, "runs", f"run_{run_index:03d}")
        run_crops_folder = os.path.join(run_folder, "crop_frames_512")
        run_anchor_sources = os.path.join(run_folder, "anchor_sources_512")
        run_enhanced_anchors = os.path.join(run_folder, "enhanced_anchors_512")
        run_ltx_frames = os.path.join(run_folder, "ltx_frames_512")
        for folder in (run_crops_folder, run_anchor_sources, run_enhanced_anchors, run_ltx_frames):
            os.makedirs(folder, exist_ok=True)
        for local_index, entry in enumerate(run_entries):
            entry["run_local_index"] = local_index
            shutil.copy2(entry["crop_path"], os.path.join(run_crops_folder, f"frame_{local_index:06d}.png"))
        desired_indices = _anchor_indices(len(run_entries), anchor_interval)
        detected_indices = [
            index for index, entry in enumerate(run_entries)
            if entry.get("detected") and float(entry.get("composite_strength") or 0.0) > 0.0
        ]
        safe_detected_indices = [index for index in detected_indices if not _is_forbidden_ltx_conditioning_index(index)]
        if safe_detected_indices:
            detected_indices = safe_detected_indices
        selected_indices = []
        for desired in desired_indices:
            if not detected_indices:
                break
            selected = min(detected_indices, key=lambda index: (abs(index - desired), index))
            if selected not in selected_indices:
                selected_indices.append(selected)
        if not selected_indices:
            continue
        run_anchors = []
        for order, local_index in enumerate(selected_indices):
            entry = run_entries[local_index]
            anchor_name = f"anchor_{order:04d}_index_{local_index:06d}.png"
            source_path = os.path.join(run_anchor_sources, anchor_name)
            enhanced_path = os.path.join(run_enhanced_anchors, anchor_name)
            shutil.copy2(entry["crop_path"], source_path)
            anchor = {
                "run_index": run_index, "order": order, "index": local_index,
                "entry_index": entry["index"], "frame_number": entry["frame_number"],
                "source_path": source_path, "enhanced_path": enhanced_path,
            }
            run_anchors.append(anchor)
            anchors.append(anchor)
        crop_video_path = os.path.join(run_folder, "face_crops_512.mp4")
        _encode_crop_video(run_crops_folder, crop_video_path, fps, 0, len(run_entries))
        runs.append({
            "run_index": run_index,
            "start_entry_index": run_entries[0]["index"],
            "end_entry_index": run_entries[-1]["index"],
            "start_frame": run_entries[0]["frame_number"],
            "end_frame": run_entries[-1]["frame_number"],
            "frame_count": len(run_entries),
            "crop_video_path": crop_video_path,
            "anchor_indices": selected_indices,
            "anchor_indices_text": ",".join(str(index) for index in selected_indices),
            "anchor_sources_folder": run_anchor_sources,
            "enhanced_anchors_folder": run_enhanced_anchors,
            "ltx_frames_folder": run_ltx_frames,
            "anchors": run_anchors,
        })
    if not runs:
        if close_skipped_frames > 0:
            raise ValueError(
                "Faces were detected, but none are distant enough for the selected Repair Distance preset. "
                "Choose a broader preset or All detected faces."
            )
        raise ValueError("No face was detected in the selected Face Fix range.")

    manifest = {
        "version": 1,
        "job_id": job_id,
        "video_path": video_path,
        "project_folder": project_folder,
        "job_folder": job_folder,
        "fps": fps,
        "width": width,
        "height": height,
        "total_video_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": entries[-1]["frame_number"],
        "start_time": start_time,
        "end_time": end_time,
        "whole_scene": whole_scene and not preview_only,
        "enhance_size": enhance_size,
        "anchor_interval": anchor_interval,
        "face_run_count": len(runs),
        "runs": runs,
        "anchors": anchors,
        "ltx_settings": ltx_settings,
        "carried_frames": carried_frames,
        "skipped_frames": skipped_frames,
        "close_skipped_frames": close_skipped_frames,
        "settings": {
            "confidence": confidence,
            "crop_padding_factor": padding,
            "minimum_face_pixels": minimum_pixels,
            "rotation_assist": rotation_assist,
            "repair_distance": repair_distance,
            "custom_distance_threshold": custom_distance_threshold,
            "enhance_amount": max(1, min(20, int(_payload_number(payload, "enhance_amount", 8)))),
        },
        "entries": entries,
    }
    manifest_path = os.path.join(job_folder, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    first_face_entry = next(entry for entry in entries if entry.get("crop_path"))
    preview_ok, preview_buffer = cv2.imencode(".jpg", cv2.imread(first_face_entry["crop_path"]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    preview_data = ""
    if preview_ok:
        preview_data = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.tobytes()).decode("ascii")
    return {
        "job_id": job_id,
        "job_folder": job_folder,
        "manifest_path": manifest_path,
        "frame_count": len(entries),
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": entries[-1]["frame_number"],
        "carried_frames": carried_frames,
        "skipped_frames": skipped_frames,
        "close_skipped_frames": close_skipped_frames,
        "face_run_count": len(runs),
        "runs": runs,
        "anchor_interval": anchor_interval,
        "anchor_count": len(anchors),
        "anchors": anchors,
        "ltx_settings": ltx_settings,
        "first_crop_path": first_face_entry["crop_path"],
        "crop_preview_data": preview_data,
        "crops": [
            {
                "index": entry["index"],
                "frame_number": entry["frame_number"],
                "crop_path": entry["crop_path"],
            }
            for entry in entries if entry.get("crop_path")
        ],
    }


def accept_enhanced_crop(payload):
    from .VRGDG_WorkflowRunnerNodes import _resolve_comfy_image_path

    manifest_path = _absolute_existing_file(payload.get("manifest_path"), "Face Fix manifest")
    if os.path.basename(manifest_path).lower() != "manifest.json":
        raise ValueError("Invalid Face Fix manifest path.")
    normalized_parts = [part.lower() for part in os.path.normpath(manifest_path).split(os.sep)]
    if "face_fix" not in normalized_parts or "jobs" not in normalized_parts:
        raise ValueError("The manifest is not inside a Face Fix job folder.")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    index = int(payload.get("index", -1))
    entries = manifest.get("entries") or []
    if index < 0 or index >= len(entries):
        raise IndexError(f"Face Fix crop index is out of range: {index}")
    image_info = payload.get("image")
    if not isinstance(image_info, dict):
        raise ValueError("Generated image metadata is missing.")
    source_path = _resolve_comfy_image_path(image_info)
    target_path = os.path.abspath(str(entries[index].get("enhanced_path") or ""))
    enhanced_root = os.path.abspath(os.path.join(manifest["job_folder"], "enhanced_512"))
    if os.path.commonpath([enhanced_root, target_path]) != enhanced_root:
        raise ValueError("Enhanced crop path escapes the Face Fix job folder.")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(source_path, target_path)
    entries[index]["enhanced_source"] = dict(image_info)
    entries[index]["enhanced_complete"] = True
    manifest["enhanced_count"] = sum(1 for entry in entries if entry.get("enhanced_complete"))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    preview_data = ""
    try:
        import cv2
        preview_image = cv2.imread(target_path)
        preview_ok, preview_buffer = cv2.imencode(".jpg", preview_image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if preview_ok:
            preview_data = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.tobytes()).decode("ascii")
    except Exception:
        preview_data = ""
    return {
        "index": index,
        "frame_number": entries[index].get("frame_number"),
        "enhanced_path": target_path,
        "enhanced_count": manifest["enhanced_count"],
        "frame_count": len(entries),
        "enhanced_preview_data": preview_data,
    }


def accept_enhanced_anchor(payload):
    from .VRGDG_WorkflowRunnerNodes import _resolve_comfy_image_path

    manifest_path = _absolute_existing_file(payload.get("manifest_path"), "Face Fix manifest")
    if os.path.basename(manifest_path).lower() != "manifest.json":
        raise ValueError("Invalid Face Fix manifest path.")
    normalized_parts = [part.lower() for part in os.path.normpath(manifest_path).split(os.sep)]
    if "face_fix" not in normalized_parts or "jobs" not in normalized_parts:
        raise ValueError("The manifest is not inside a Face Fix job folder.")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    run_index = int(payload.get("run_index", -1))
    runs = manifest.get("runs") or []
    if run_index < 0 or run_index >= len(runs):
        raise IndexError(f"Face Fix run index is out of range: {run_index}")
    order = int(payload.get("order", -1))
    anchors = runs[run_index].get("anchors") or []
    if order < 0 or order >= len(anchors):
        raise IndexError(f"Face Fix anchor order is out of range: {order}")
    image_info = payload.get("image")
    if not isinstance(image_info, dict):
        raise ValueError("Generated anchor image metadata is missing.")
    source_path = _resolve_comfy_image_path(image_info)
    target_path = os.path.abspath(str(anchors[order].get("enhanced_path") or ""))
    enhanced_root = os.path.abspath(str(runs[run_index].get("enhanced_anchors_folder") or ""))
    if not enhanced_root or os.path.commonpath([enhanced_root, target_path]) != enhanced_root:
        raise ValueError("Enhanced anchor path escapes the Face Fix job folder.")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(source_path, target_path)
    anchors[order]["enhanced_source"] = dict(image_info)
    anchors[order]["enhanced_complete"] = True
    manifest["enhanced_anchor_count"] = sum(
        1 for run in runs for anchor in (run.get("anchors") or []) if anchor.get("enhanced_complete")
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    preview_data = ""
    try:
        import cv2
        preview_image = cv2.imread(target_path)
        preview_ok, preview_buffer = cv2.imencode(".jpg", preview_image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if preview_ok:
            preview_data = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.tobytes()).decode("ascii")
    except Exception:
        preview_data = ""
    return {
        "run_index": run_index, "order": order,
        "index": anchors[order].get("index"),
        "frame_number": anchors[order].get("frame_number"),
        "enhanced_path": target_path,
        "enhanced_anchor_count": manifest["enhanced_anchor_count"],
        "anchor_count": sum(len(run.get("anchors") or []) for run in runs),
        "enhanced_preview_data": preview_data,
    }


def build_ltx_face_fix_prompt(payload):
    manifest_path = _absolute_existing_file(payload.get("manifest_path"), "Face Fix manifest")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    run_index = int(payload.get("run_index", -1))
    runs = manifest.get("runs") or []
    if run_index < 0 or run_index >= len(runs):
        raise IndexError(f"Face Fix run index is out of range: {run_index}")
    run = runs[run_index]
    anchors = run.get("anchors") or []
    if not anchors or any(not anchor.get("enhanced_complete") or not os.path.isfile(str(anchor.get("enhanced_path") or "")) for anchor in anchors):
        raise ValueError("All Face Fix anchors must be enhanced before LTX can run.")
    crop_video_path = _absolute_existing_file(run.get("crop_video_path"), "512x512 face crop video")
    enhanced_anchors_folder = os.path.abspath(str(run.get("enhanced_anchors_folder") or ""))
    if not os.path.isdir(enhanced_anchors_folder):
        raise FileNotFoundError("The enhanced anchor folder was not found.")
    workflow_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Workflows", "UsedForUIDoNotTouch", "LTX2.3_FaceFixV1_API.json"
    )
    with open(workflow_path, "r", encoding="utf-8") as handle:
        prompt = json.load(handle)
    required = {"5298", "5301", "4880", "4638", "4637", "4896", "5303"}
    missing = sorted(required.difference(prompt.keys()))
    if missing:
        raise KeyError(f"LTX Face Fix workflow is missing required node(s): {', '.join(missing)}")
    settings = manifest.get("ltx_settings") or {}
    prompt["5298"]["inputs"]["video"] = crop_video_path
    prompt["5301"]["inputs"]["folder"] = enhanced_anchors_folder
    prompt["4880"]["inputs"]["guiding_strength"] = float(settings.get("guiding_strength", 0.20))
    prompt["4880"]["inputs"]["temporal_overlap_cond_strength"] = float(settings.get("temporal_overlap_cond_strength", 0.50))
    prompt["4880"]["inputs"]["cond_image_strength"] = float(settings.get("cond_image_strength", 0.50))
    original_indices = [int(anchor.get("index", 0)) for anchor in anchors]
    safe_indices = _safe_ltx_conditioning_indices(original_indices, int(run.get("frame_count") or 0))
    if len(safe_indices) != len(anchors):
        raise ValueError("Face Fix could not assign a valid LTX conditioning index to every enhanced anchor.")
    safe_indices_text = ",".join(str(index) for index in safe_indices)
    prompt["4880"]["inputs"]["optional_cond_image_indices"] = safe_indices_text
    prompt["4638"]["inputs"]["noise_seed"] = int(settings.get("seed", 42))
    prompt["4637"]["inputs"]["sampler_name"] = str(settings.get("sampler") or "euler_ancestral")
    prompt["4896"]["inputs"]["sigmas"] = str(settings.get("sigmas") or "0.909375, 0.725, 0.421875, 0.0")
    return {
        "workflow_path": workflow_path,
        "prompt": prompt,
        "run_index": run_index,
        "frame_count": int(run.get("frame_count") or 0),
        "anchor_count": len(anchors),
        "anchor_indices_text": safe_indices_text,
    }


def accept_ltx_frame_batch(payload):
    from .VRGDG_WorkflowRunnerNodes import _resolve_comfy_image_path
    import cv2

    manifest_path = _absolute_existing_file(payload.get("manifest_path"), "Face Fix manifest")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    run_index = int(payload.get("run_index", -1))
    runs = manifest.get("runs") or []
    if run_index < 0 or run_index >= len(runs):
        raise IndexError(f"Face Fix run index is out of range: {run_index}")
    run = runs[run_index]
    all_entries = manifest.get("entries") or []
    entries = [entry for entry in all_entries if entry.get("run_index") == run_index]
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("LTX Preview Image batch metadata is missing.")
    frame_delta = len(entries) - len(images)
    if abs(frame_delta) > 7:
        raise ValueError(
            f"LTX returned {len(images)} frames, but Face Fix prepared {len(entries)}; "
            "the difference is larger than one normal LTX temporal-length adjustment."
        )
    images = images[:len(entries)]
    output_folder = os.path.abspath(str(run.get("ltx_frames_folder") or ""))
    if not output_folder:
        raise ValueError("The LTX run output folder is missing.")
    os.makedirs(output_folder, exist_ok=True)
    saved = []
    for index, image_info in enumerate(images):
        if not isinstance(image_info, dict):
            raise ValueError(f"LTX frame {index} metadata is invalid.")
        source_path = _resolve_comfy_image_path(image_info)
        frame = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read LTX frame {index}: {source_path}")
        height, width = frame.shape[:2]
        if width != 512 or height != 512:
            raise ValueError(f"LTX frame {index} is {width}x{height}; expected exactly 512x512.")
        target_path = os.path.join(output_folder, f"frame_{index:06d}.png")
        if not cv2.imwrite(target_path, frame):
            raise RuntimeError(f"Could not save LTX frame {index}.")
        entries[index]["ltx_frame_path"] = target_path
        entries[index]["ltx_source"] = dict(image_info)
        saved.append(target_path)
    # LTX emits temporal batches in valid 8n+1 lengths. If it rounds a run down
    # by a few frames, preserve those unmatched source frames instead of
    # rejecting the entire otherwise-valid batch.
    for entry in entries[len(saved):]:
        entry["composite_strength"] = 0.0
        entry["ltx_skipped_reason"] = "LTX temporal-length tail; original frame preserved"
    run["ltx_frames_folder"] = output_folder
    run["ltx_frame_count"] = len(saved)
    run["ltx_complete"] = True
    manifest["ltx_frame_count"] = sum(int(item.get("ltx_frame_count") or 0) for item in runs)
    manifest["ltx_complete"] = all(bool(item.get("ltx_complete")) for item in runs)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    preview_data = ""
    if saved:
        preview_image = cv2.imread(saved[0])
        preview_ok, preview_buffer = cv2.imencode(".jpg", preview_image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if preview_ok:
            preview_data = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.tobytes()).decode("ascii")
    return {
        "run_index": run_index, "ltx_frames_folder": output_folder,
        "ltx_frame_count": len(saved),
        "frame_count": len(entries),
        "preserved_tail_frames": max(0, len(entries) - len(saved)),
        "ltx_preview_data": preview_data,
    }


def _find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg is required to rebuild the repaired video.") from exc


def _soft_ellipse_mask(width, height, feather):
    import cv2
    import numpy as np

    mask = np.zeros((height, width), dtype=np.float32)
    feather = max(0, int(feather))
    inset = max(2, int(round(min(width, height) * 0.035)))
    axes = (max(1, width // 2 - inset), max(1, height // 2 - inset))
    cv2.ellipse(mask, (width // 2, height // 2), axes, 0, 0, 360, 1.0, -1)
    if feather > 0:
        kernel = max(3, feather * 4 + 1)
        if kernel % 2 == 0:
            kernel += 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), max(0.1, feather))
    return mask.clip(0.0, 1.0)


def _color_match(enhanced, original, alpha, strength):
    import numpy as np

    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return enhanced
    selected = alpha > 0.35
    if int(selected.sum()) < 16:
        return enhanced
    source = enhanced.astype(np.float32)
    target = original.astype(np.float32)
    source_mean = source[selected].mean(axis=0)
    target_mean = target[selected].mean(axis=0)
    return np.clip(source + (target_mean - source_mean) * strength, 0, 255).astype(np.uint8)


def finalize_face_fix(payload):
    import cv2
    import numpy as np

    manifest_path = _absolute_existing_file(payload.get("manifest_path"), "Face Fix manifest")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("entries") or []
    if not entries:
        raise ValueError("The Face Fix job has no prepared frames.")
    repair_entries = [entry for entry in entries if float(entry.get("composite_strength") or 0.0) > 0.0]
    incomplete = [entry for entry in repair_entries if not os.path.isfile(str(entry.get("ltx_frame_path") or ""))]
    if incomplete:
        raise ValueError(f"Face Fix still has {len(incomplete)} frame(s) without validated LTX output.")
    if not repair_entries:
        raise ValueError("Face Fix has no safe face-visible frames to composite.")

    feather = max(0, min(256, int(payload.get("feather") or 18)))
    color_match = max(0.0, min(1.0, float(payload.get("color_match") or 0.65)))
    job_folder = os.path.abspath(manifest["job_folder"])
    composited_folder = os.path.join(job_folder, "composited_frames")
    os.makedirs(composited_folder, exist_ok=True)
    composited_by_frame = {}
    faded_frames = 0
    for entry in repair_entries:
        composite_strength = max(0.0, min(1.0, float(entry.get("composite_strength") or 0.0)))
        if composite_strength < 1.0:
            faded_frames += 1
        original = cv2.imread(_absolute_existing_file(entry.get("original_path"), "Original Face Fix frame"))
        enhanced = cv2.imread(_absolute_existing_file(entry.get("ltx_frame_path"), "LTX Face Fix frame"))
        if original is None or enhanced is None:
            raise RuntimeError(f"Could not decode Face Fix frame {entry.get('frame_number')}.")
        left, top, right, bottom = (int(value) for value in entry["crop_box"])
        target_h, target_w = bottom - top, right - left
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Invalid crop box for frame {entry.get('frame_number')}.")
        resized = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        target = original[top:bottom, left:right]
        base_alpha = _soft_ellipse_mask(target_w, target_h, feather)
        resized = _color_match(resized, target, base_alpha, color_match)
        alpha = base_alpha * composite_strength
        alpha3 = alpha[:, :, None]
        blended = target.astype(np.float32) * (1.0 - alpha3) + resized.astype(np.float32) * alpha3
        output = original.copy()
        output[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)
        output_path = os.path.join(composited_folder, f"frame_{int(entry['frame_number']):06d}.png")
        cv2.imwrite(output_path, output)
        entry["composited_path"] = output_path
        composited_by_frame[int(entry["frame_number"])] = output_path

    source_video = _absolute_existing_file(manifest.get("video_path"), "Source scene video")
    capture = cv2.VideoCapture(source_video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen source scene video: {source_video}")
    fps = float(manifest.get("fps") or capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(manifest.get("width") or capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(manifest.get("height") or capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    silent_path = os.path.join(job_folder, "face_fix_silent.avi")
    writer = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the temporary lossless Face Fix video.")
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        repaired_path = composited_by_frame.get(frame_number)
        if repaired_path:
            repaired = cv2.imread(repaired_path)
            if repaired is not None:
                frame = repaired
        writer.write(frame)
        frame_number += 1
    capture.release()
    writer.release()

    source_dir = os.path.dirname(source_video)
    source_stem = os.path.splitext(os.path.basename(source_video))[0]
    output_path = os.path.join(source_dir, f"{source_stem}_facefix_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    command = [
        _find_ffmpeg(), "-y", "-i", silent_path, "-i", source_video,
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
        "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0 or not os.path.isfile(output_path):
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg failed to rebuild the Face Fix video.").strip())
    try:
        os.remove(silent_path)
    except OSError:
        pass
    manifest["composite_complete"] = True
    manifest["output_video_path"] = output_path
    manifest["feather"] = feather
    manifest["color_match"] = color_match
    manifest["frames_repaired"] = len(repair_entries)
    manifest["frames_faded"] = faded_frames
    manifest["frames_skipped"] = len(entries) - len(repair_entries)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return {
        "output_video_path": output_path,
        "source_video_path": source_video,
        "frames_repaired": len(repair_entries),
        "frames_faded": faded_frames,
        "frames_skipped": len(entries) - len(repair_entries),
        "close_skipped_frames": int(manifest.get("close_skipped_frames") or 0),
        "start_frame": manifest.get("start_frame"),
        "end_frame": manifest.get("end_frame"),
        "fps": fps,
        "width": width,
        "height": height,
    }


def register_face_fix_routes(server_instance):
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED or server_instance is None:
        return

    @server_instance.routes.post("/vrgdg/face_fix/prepare")
    async def vrgdg_face_fix_prepare(request):
        import asyncio
        try:
            payload = await request.json()
            result = await asyncio.to_thread(prepare_face_fix, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/face_fix/estimate_anchors")
    async def vrgdg_face_fix_estimate_anchors(request):
        import asyncio
        try:
            payload = await request.json()
            result = await asyncio.to_thread(estimate_face_fix_anchors, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/face_fix/accept_enhanced")
    async def vrgdg_face_fix_accept_enhanced(request):
        import asyncio
        try:
            payload = await request.json()
            result = await asyncio.to_thread(accept_enhanced_crop, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/face_fix/accept_enhanced_anchor")
    async def vrgdg_face_fix_accept_enhanced_anchor(request):
        import asyncio
        try:
            payload = await request.json()
            result = await asyncio.to_thread(accept_enhanced_anchor, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/face_fix/build_ltx_prompt")
    async def vrgdg_face_fix_build_ltx_prompt(request):
        import asyncio
        try:
            payload = await request.json()
            result = await asyncio.to_thread(build_ltx_face_fix_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/face_fix/accept_ltx_frames")
    async def vrgdg_face_fix_accept_ltx_frames(request):
        import asyncio
        try:
            payload = await request.json()
            result = await asyncio.to_thread(accept_ltx_frame_batch, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/face_fix/finalize")
    async def vrgdg_face_fix_finalize(request):
        import asyncio
        try:
            payload = await request.json()
            result = await asyncio.to_thread(finalize_face_fix, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    _ROUTES_REGISTERED = True


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
