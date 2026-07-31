import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid

import folder_paths
import torch
import torch.nn.functional as F
from aiohttp import web
from server import PromptServer


_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_JOBS = {}
_JOB_CANCEL = {}
_JOB_LOCK = threading.RLock()
_ROUTES_REGISTERED = False


def _safe_name(value, fallback="video"):
    name = os.path.basename(str(value or "").strip()) or fallback
    stem, extension = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or fallback
    extension = re.sub(r"[^A-Za-z0-9.]+", "", extension)
    return stem[:100] + extension[:12]


def _root_folder():
    path = os.path.join(folder_paths.get_output_directory(), "VRGDG_VideoEnhancer")
    os.makedirs(path, exist_ok=True)
    return path


def _upload_folder():
    path = os.path.join(_root_folder(), "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def _preview_folder():
    path = os.path.join(_root_folder(), "previews")
    os.makedirs(path, exist_ok=True)
    return path


def _jobs_folder():
    path = os.path.join(_root_folder(), "jobs")
    os.makedirs(path, exist_ok=True)
    return path


def _find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg is required for final video rendering.") from exc


def _media_has_audio(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
        return bool(result.returncode == 0 and str(result.stdout or "").strip())
    except Exception:
        return None


def _normalize_video_path(value):
    path = os.path.normpath(os.path.abspath(str(value or "").strip().strip('"')))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file was not found: {path}")
    if os.path.splitext(path)[1].lower() not in _VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video type. Use MP4, MOV, MKV, WEBM, AVI, or M4V.")
    return path


def _probe_video(path):
    import cv2

    path = _normalize_video_path(path)
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError("The video could not be opened.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width < 1 or height < 1 or fps <= 0:
            raise ValueError("The video does not contain readable dimensions or frame-rate metadata.")
        duration = frame_count / fps if frame_count > 0 else 0.0
        codec_value = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((codec_value >> (8 * index)) & 0xFF) for index in range(4)).strip()
    finally:
        capture.release()
    stat = os.stat(path)
    return {
        "path": path,
        "name": os.path.basename(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "codec": codec,
        "has_audio": _media_has_audio(path),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def _normalize_settings(payload):
    payload = payload if isinstance(payload, dict) else {}

    def number(name, default, minimum, maximum):
        try:
            value = float(payload.get(name, default))
        except (TypeError, ValueError):
            value = float(default)
        return max(minimum, min(maximum, value))

    def integer(name, default, minimum, maximum):
        try:
            value = int(round(float(payload.get(name, default))))
        except (TypeError, ValueError):
            value = int(default)
        return max(minimum, min(maximum, value))

    preset = str(payload.get("encode_preset") or "medium").strip().lower()
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
        preset = "medium"
    upscale_resolution = str(payload.get("upscale_resolution") or "original").strip().lower()
    if upscale_resolution not in {"original", "2k", "3k", "4k"}:
        upscale_resolution = "original"
    return {
        "upscale_resolution": upscale_resolution,
        "sharpen_enabled": bool(payload.get("sharpen_enabled", True)),
        "sharpen_strength": number("sharpen_strength", 0.5, 0.0, 10.0),
        "grain_enabled": bool(payload.get("grain_enabled", False)),
        "grain_intensity": number("grain_intensity", 0.04, 0.0, 1.0),
        "saturation_mix": number("saturation_mix", 0.5, 0.0, 1.0),
        "seed": integer("seed", 42, 0, 2**31 - 1),
        "use_gpu": bool(payload.get("use_gpu", True)),
        "batch_size": integer("batch_size", 0, 0, 128),
        "segment_seconds": integer("segment_seconds", 30, 5, 300),
        "encode_crf": integer("encode_crf", 18, 12, 35),
        "encode_preset": preset,
        "preserve_audio": bool(payload.get("preserve_audio", True)),
        "output_name": _safe_name(payload.get("output_name") or "enhanced_video.mp4", "enhanced_video"),
    }


def _output_dimensions(width, height, upscale_resolution):
    width = max(1, int(width))
    height = max(1, int(height))
    target_long_edge = {
        "2k": 2560,
        "3k": 3072,
        "4k": 3840,
    }.get(str(upscale_resolution or "original").strip().lower(), 0)
    source_long_edge = max(width, height)
    if target_long_edge <= 0 or source_long_edge >= target_long_edge:
        return width, height
    scale = target_long_edge / source_long_edge
    output_width = max(2, int(round((width * scale) / 2.0)) * 2)
    output_height = max(2, int(round((height * scale) / 2.0)) * 2)
    return output_width, output_height


def _auto_batch_size(width, height):
    pixels = max(1, int(width) * int(height))
    if pixels <= 1280 * 720:
        return 16
    if pixels <= 1920 * 1080:
        return 8
    if pixels <= 2560 * 1440:
        return 4
    if pixels <= 3200 * 1800:
        return 2
    return 1


def _resize_frames(frames, output_width, output_height):
    import cv2

    output_width = max(1, int(output_width))
    output_height = max(1, int(output_height))
    resized = []
    for frame in frames:
        if frame.shape[1] == output_width and frame.shape[0] == output_height:
            resized.append(frame)
        else:
            resized.append(
                cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            )
    return resized


def _apply_unsharp(images, strength, use_gpu):
    if strength <= 0:
        return images
    if use_gpu:
        nchw = images.permute(0, 3, 1, 2)
        blurred = F.avg_pool2d(nchw, kernel_size=3, stride=1, padding=1)
        return (nchw + strength * (nchw - blurred)).clamp(0.0, 1.0).permute(0, 2, 3, 1)

    import numpy as np

    array = images.detach().cpu().contiguous().numpy()
    padded = np.pad(array, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = (
        padded[:, 0:-2, 0:-2]
        + padded[:, 0:-2, 1:-1]
        + padded[:, 0:-2, 2:]
        + padded[:, 1:-1, 0:-2]
        + padded[:, 1:-1, 1:-1]
        + padded[:, 1:-1, 2:]
        + padded[:, 2:, 0:-2]
        + padded[:, 2:, 1:-1]
        + padded[:, 2:, 2:]
    ) / 9.0
    output = array + strength * (array - blurred)
    np.clip(output, 0.0, 1.0, out=output)
    return torch.from_numpy(output)


def _apply_seeded_grain(images, intensity, saturation_mix, seed, frame_start):
    if intensity <= 0:
        return images
    grain_frames = []
    device = images.device
    for offset, frame in enumerate(images):
        generator = torch.Generator(device=device)
        generator.manual_seed((int(seed) + int(frame_start) + offset) & 0x7FFFFFFF)
        grain = torch.randn(frame.shape, generator=generator, device=device, dtype=frame.dtype)
        grain[..., 0] *= 2.0
        grain[..., 2] *= 3.0
        gray = grain[..., 1:2].repeat(1, 1, 3)
        grain_frames.append(saturation_mix * grain + (1.0 - saturation_mix) * gray)
    grain_batch = torch.stack(grain_frames, dim=0)
    return (images + grain_batch * intensity).clamp(0.0, 1.0)


def _apply_effects_batch(images, settings, frame_start=0):
    requested_gpu = bool(settings.get("use_gpu", True))
    use_gpu = bool(requested_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_gpu else "cpu")
    batch = images.to(device)
    with torch.inference_mode():
        if settings.get("sharpen_enabled", True):
            batch = _apply_unsharp(batch, float(settings.get("sharpen_strength", 0.5)), use_gpu)
        if settings.get("grain_enabled", False):
            batch = _apply_seeded_grain(
                batch,
                float(settings.get("grain_intensity", 0.04)),
                float(settings.get("saturation_mix", 0.5)),
                int(settings.get("seed", 42)),
                int(frame_start),
            )
    return batch.detach().cpu()


def _process_with_retry(images, settings, frame_start):
    try:
        return _apply_effects_batch(images, settings, frame_start), len(images)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(images) <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = max(1, len(images) // 2)
        left, left_size = _process_with_retry(images[:midpoint], settings, frame_start)
        right, right_size = _process_with_retry(images[midpoint:], settings, frame_start + midpoint)
        return torch.cat((left, right), dim=0), min(left_size, right_size)


def _frames_to_tensor(frames):
    import cv2
    import numpy as np

    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return torch.from_numpy(np.stack(rgb, axis=0).astype(np.float32) / 255.0)


def _tensor_to_frames(tensor):
    import cv2
    import numpy as np

    array = np.clip(tensor.numpy() * 255.0, 0, 255).astype(np.uint8)
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in array]


def _job_update(job_id, **values):
    with _JOB_LOCK:
        job = _JOBS.setdefault(job_id, {"job_id": job_id})
        job.update(values)
        job["updated_at"] = time.time()


def _job_snapshot(job_id):
    with _JOB_LOCK:
        job = dict(_JOBS.get(job_id) or {})
    job.pop("thread", None)
    job.pop("process", None)
    return job


def _manifest_path(job_folder):
    return os.path.join(job_folder, "manifest.json")


def _write_manifest(job_folder, document):
    path = _manifest_path(job_folder)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    os.replace(temp_path, path)


def _read_manifest(job_folder):
    path = _manifest_path(job_folder)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _settings_fingerprint(source_path, settings, metadata):
    stat = os.stat(source_path)
    document = {
        "source_path": source_path,
        "source_size": int(stat.st_size),
        "source_mtime": float(stat.st_mtime),
        "frame_count": int(metadata["frame_count"]),
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode("utf-8")).hexdigest()


def _render_segment(source_path, segment_path, start_frame, end_frame, metadata, settings, job_id, cancel_event):
    import cv2

    capture = cv2.VideoCapture(source_path)
    writer = None
    frames_done = 0
    output_width, output_height = _output_dimensions(
        metadata["width"],
        metadata["height"],
        settings.get("upscale_resolution"),
    )
    smallest_batch = int(settings["batch_size"] or _auto_batch_size(output_width, output_height))
    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open the source video for rendering.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        writer = cv2.VideoWriter(
            segment_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(metadata["fps"]),
            (output_width, output_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create a temporary video segment.")
        frame_index = int(start_frame)
        while frame_index < end_frame:
            if cancel_event.is_set():
                raise InterruptedError("Render canceled.")
            frames = []
            requested = min(smallest_batch, end_frame - frame_index)
            for _ in range(requested):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            frames = _resize_frames(frames, output_width, output_height)
            tensor = _frames_to_tensor(frames)
            enhanced, successful_batch = _process_with_retry(tensor, settings, frame_index)
            smallest_batch = max(1, min(smallest_batch, successful_batch))
            for frame in _tensor_to_frames(enhanced):
                writer.write(frame)
            count = len(frames)
            frame_index += count
            frames_done += count
            current = int(_job_snapshot(job_id).get("frames_processed") or 0) + count
            total = max(1, int(metadata["frame_count"]))
            _job_update(
                job_id,
                frames_processed=current,
                progress=min(0.94, current / total * 0.94),
                batch_size=smallest_batch,
                message=f"Upscaling and enhancing frames {current:,}/{total:,}",
            )
        if frames_done <= 0:
            raise RuntimeError("The source video ended before this segment could be rendered.")
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return frames_done, smallest_batch


def _concat_and_mux(segment_paths, source_path, output_path, settings, job_id, cancel_event):
    job_folder = os.path.dirname(segment_paths[0])
    concat_path = os.path.join(job_folder, "segments.txt")
    with open(concat_path, "w", encoding="utf-8") as handle:
        for path in segment_paths:
            escaped = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    command = [
        _find_ffmpeg(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_path,
        "-i",
        source_path,
        "-map",
        "0:v:0",
    ]
    if settings.get("preserve_audio", True):
        command.extend(["-map", "1:a?"])
    else:
        command.append("-an")
    command.extend([
        "-c:v",
        "libx264",
        "-preset",
        str(settings["encode_preset"]),
        "-crf",
        str(settings["encode_crf"]),
        "-pix_fmt",
        "yuv420p",
    ])
    if settings.get("preserve_audio", True):
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(["-movflags", "+faststart", "-shortest", output_path])
    _job_update(job_id, stage="encoding", progress=0.95, message="Joining segments and restoring audio…")
    log_path = os.path.join(job_folder, "ffmpeg.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=log_handle,
            text=True,
            errors="replace",
        )
        with _JOB_LOCK:
            _JOBS[job_id]["process"] = process
        while process.poll() is None:
            if cancel_event.wait(0.25):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise InterruptedError("Render canceled.")
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_handle:
            stderr = log_handle.read()
    except OSError:
        stderr = ""
    with _JOB_LOCK:
        _JOBS[job_id].pop("process", None)
    if process.returncode != 0 or not os.path.isfile(output_path):
        raise RuntimeError(f"FFmpeg could not create the final video: {stderr[-1800:]}")


def _render_job(job_id, payload, resume=False):
    cancel_event = _JOB_CANCEL[job_id]
    job_folder = os.path.join(_jobs_folder(), job_id)
    segments_folder = os.path.join(job_folder, "segments")
    os.makedirs(segments_folder, exist_ok=True)
    try:
        source_path = _normalize_video_path(payload.get("source_path"))
        metadata = _probe_video(source_path)
        settings = _normalize_settings(payload.get("settings"))
        output_width, output_height = _output_dimensions(
            metadata["width"],
            metadata["height"],
            settings["upscale_resolution"],
        )
        fingerprint = _settings_fingerprint(source_path, settings, metadata)
        manifest = _read_manifest(job_folder) if resume else {}
        if manifest and manifest.get("fingerprint") != fingerprint:
            raise ValueError("The source video or enhancement settings changed, so this job cannot resume.")
        completed = {
            int(value)
            for value in (manifest.get("completed_segments") or [])
            if str(value).isdigit()
        }
        frames_per_segment = max(1, int(round(float(metadata["fps"]) * settings["segment_seconds"])))
        total_segments = max(1, int(math.ceil(metadata["frame_count"] / frames_per_segment)))
        completed = {
            index
            for index in completed
            if 0 <= index < total_segments
            and os.path.isfile(os.path.join(segments_folder, f"segment_{index:05d}.mp4"))
        }
        completed_frames = sum(
            max(0, min(metadata["frame_count"], (index + 1) * frames_per_segment) - index * frames_per_segment)
            for index in completed
        )
        manifest = {
            "version": 1,
            "job_id": job_id,
            "fingerprint": fingerprint,
            "source_path": source_path,
            "settings": settings,
            "metadata": metadata,
            "completed_segments": sorted(completed),
        }
        _write_manifest(job_folder, manifest)
        _job_update(
            job_id,
            status="running",
            stage="enhancing",
            source_path=source_path,
            metadata=metadata,
            settings=settings,
            output_width=output_width,
            output_height=output_height,
            frames_processed=completed_frames,
            total_frames=metadata["frame_count"],
            segment_index=len(completed),
            total_segments=total_segments,
            progress=(completed_frames / max(1, metadata["frame_count"])) * 0.94,
            can_resume=False,
            error="",
            message=f"Starting {output_width}×{output_height} batched enhancement…",
        )
        for segment_index in range(total_segments):
            if segment_index in completed:
                continue
            if cancel_event.is_set():
                raise InterruptedError("Render canceled.")
            start_frame = segment_index * frames_per_segment
            end_frame = min(metadata["frame_count"], start_frame + frames_per_segment)
            segment_path = os.path.join(segments_folder, f"segment_{segment_index:05d}.mp4")
            partial_path = segment_path + ".partial.mp4"
            if os.path.isfile(partial_path):
                os.remove(partial_path)
            _job_update(
                job_id,
                segment_index=segment_index + 1,
                message=f"Enhancing checkpoint {segment_index + 1}/{total_segments}",
            )
            frames_done, _batch = _render_segment(
                source_path,
                partial_path,
                start_frame,
                end_frame,
                metadata,
                settings,
                job_id,
                cancel_event,
            )
            os.replace(partial_path, segment_path)
            completed.add(segment_index)
            manifest["completed_segments"] = sorted(completed)
            _write_manifest(job_folder, manifest)
            _job_update(
                job_id,
                frames_processed=min(metadata["frame_count"], start_frame + frames_done),
                segment_index=segment_index + 1,
            )
        segment_paths = [
            os.path.join(segments_folder, f"segment_{index:05d}.mp4")
            for index in range(total_segments)
        ]
        output_stem = os.path.splitext(settings["output_name"])[0] or "enhanced_video"
        output_name = f"{output_stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(_root_folder(), output_name)
        _concat_and_mux(segment_paths, source_path, output_path, settings, job_id, cancel_event)
        output_metadata = _probe_video(output_path)
        manifest["output_path"] = output_path
        manifest["status"] = "complete"
        manifest["completed_segments"] = []
        manifest["checkpoints_cleaned"] = True
        _write_manifest(job_folder, manifest)
        shutil.rmtree(segments_folder, ignore_errors=True)
        _job_update(
            job_id,
            status="complete",
            stage="complete",
            progress=1.0,
            frames_processed=metadata["frame_count"],
            output_path=output_path,
            output_metadata=output_metadata,
            checkpoints_cleaned=True,
            can_resume=False,
            message="Enhancement complete.",
        )
    except InterruptedError as exc:
        _job_update(
            job_id,
            status="canceled",
            stage="canceled",
            can_resume=True,
            error="",
            message=str(exc),
        )
    except Exception as exc:
        _job_update(
            job_id,
            status="failed",
            stage="failed",
            can_resume=True,
            error=str(exc),
            message=f"Render failed: {exc}",
        )


def _start_render(payload, resume_job_id=""):
    resume_job_id = str(resume_job_id or "").strip()
    with _JOB_LOCK:
        active_job = next(
            (
                item
                for item in _JOBS.values()
                if item.get("job_id") != resume_job_id
                and item.get("status") in {"queued", "running", "encoding"}
            ),
            None,
        )
    if active_job:
        raise ValueError(
            f"Enhancement job {active_job.get('job_id')} is already running. "
            "Wait for it to finish or cancel it first."
        )
    if resume_job_id:
        job_id = resume_job_id
        existing = _job_snapshot(job_id)
        if not existing:
            job_folder = os.path.join(_jobs_folder(), job_id)
            manifest = _read_manifest(job_folder)
            if not manifest:
                raise ValueError("The requested render checkpoint was not found.")
            payload = {
                "source_path": manifest.get("source_path"),
                "settings": manifest.get("settings"),
            }
        elif existing.get("status") in {"running", "encoding"}:
            raise ValueError("That enhancement job is already running.")
    else:
        job_id = f"enhancer_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    cancel_event = threading.Event()
    _JOB_CANCEL[job_id] = cancel_event
    _job_update(
        job_id,
        status="queued",
        stage="queued",
        progress=0.0,
        created_at=time.time(),
        can_resume=False,
        message="Queued…",
    )
    thread = threading.Thread(
        target=_render_job,
        args=(job_id, payload, bool(resume_job_id)),
        daemon=True,
        name=f"VRGDGVideoEnhancer-{job_id}",
    )
    with _JOB_LOCK:
        _JOBS[job_id]["thread"] = thread
    thread.start()
    return _job_snapshot(job_id)


def _preview_frame(source_path, timestamp, settings):
    import cv2

    metadata = _probe_video(source_path)
    capture = cv2.VideoCapture(source_path)
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Could not decode the selected preview frame.")
    finally:
        capture.release()
    frame_index = max(0, min(metadata["frame_count"] - 1, int(round(float(timestamp) * metadata["fps"]))))
    original = frame.copy()
    output_width, output_height = _output_dimensions(
        metadata["width"],
        metadata["height"],
        settings.get("upscale_resolution"),
    )
    resized = _resize_frames([frame], output_width, output_height)[0]
    enhanced = _tensor_to_frames(
        _apply_effects_batch(_frames_to_tensor([resized]), settings, frame_index)
    )[0]
    token = f"preview_{uuid.uuid4().hex}"
    before_path = os.path.join(_preview_folder(), f"{token}_before.png")
    after_path = os.path.join(_preview_folder(), f"{token}_after.png")
    if not cv2.imwrite(before_path, original) or not cv2.imwrite(after_path, enhanced):
        raise RuntimeError("Could not save the preview images.")
    return {
        "before_path": before_path,
        "after_path": after_path,
        "timestamp": max(0.0, float(timestamp)),
        "frame_index": frame_index,
        "metadata": metadata,
        "output_width": output_width,
        "output_height": output_height,
    }


def _ensure_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.post("/vrgdg/video_enhancer/upload")
    async def vrgdg_video_enhancer_upload(request):
        try:
            reader = await request.multipart()
            saved_path = ""
            async for part in reader:
                if part.name != "video" or not part.filename:
                    continue
                safe = _safe_name(part.filename, "uploaded_video")
                extension = os.path.splitext(safe)[1].lower()
                if extension not in _VIDEO_EXTENSIONS:
                    raise ValueError("Unsupported video type.")
                saved_path = os.path.join(
                    _upload_folder(),
                    f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe}",
                )
                with open(saved_path, "wb") as handle:
                    while True:
                        chunk = await part.read_chunk(size=1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                break
            if not saved_path:
                raise ValueError("No video was uploaded.")
            return web.json_response({"ok": True, "video": _probe_video(saved_path)})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/video_enhancer/load")
    async def vrgdg_video_enhancer_load(request):
        try:
            payload = await request.json()
            return web.json_response({"ok": True, "video": _probe_video(payload.get("path"))})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/video_enhancer/preview")
    async def vrgdg_video_enhancer_preview(request):
        try:
            payload = await request.json()
            settings = _normalize_settings(payload.get("settings"))
            result = _preview_frame(
                _normalize_video_path(payload.get("source_path")),
                float(payload.get("timestamp") or 0),
                settings,
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/video_enhancer/render/start")
    async def vrgdg_video_enhancer_render_start(request):
        try:
            payload = await request.json()
            result = _start_render(payload, payload.get("resume_job_id"))
            return web.json_response({"ok": True, "job": result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.get("/vrgdg/video_enhancer/render/status")
    async def vrgdg_video_enhancer_render_status(request):
        job_id = str(request.query.get("job_id") or "").strip()
        job = _job_snapshot(job_id)
        if not job:
            return web.json_response({"ok": False, "error": "Enhancement job was not found."}, status=404)
        return web.json_response({"ok": True, "job": job})

    @server_instance.routes.post("/vrgdg/video_enhancer/render/cancel")
    async def vrgdg_video_enhancer_render_cancel(request):
        try:
            payload = await request.json()
            job_id = str(payload.get("job_id") or "").strip()
            event = _JOB_CANCEL.get(job_id)
            if event is None:
                raise ValueError("Enhancement job was not found.")
            event.set()
            with _JOB_LOCK:
                process = (_JOBS.get(job_id) or {}).get("process")
            if process is not None and process.poll() is None:
                process.terminate()
            return web.json_response({"ok": True, "job": _job_snapshot(job_id)})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.get("/vrgdg/video_enhancer/media")
    async def vrgdg_video_enhancer_media(request):
        try:
            path = os.path.normpath(os.path.abspath(str(request.query.get("path") or "").strip()))
            if not os.path.isfile(path):
                raise FileNotFoundError("Media file was not found.")
            extension = os.path.splitext(path)[1].lower()
            if extension not in _VIDEO_EXTENSIONS | {".png", ".jpg", ".jpeg", ".webp"}:
                raise ValueError("Unsupported media type.")
            return web.FileResponse(path)
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    _ROUTES_REGISTERED = True


_ensure_routes()


class VRGDGStandaloneVideoEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Updated by the standalone UI after a successful render.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_video_path",)
    FUNCTION = "return_output"
    OUTPUT_NODE = True
    CATEGORY = "VRGDG/Video"
    DESCRIPTION = "Standalone batched 2K–4K resize, video sharpening, film grain, and before/after comparison UI."

    def return_output(self, output_path):
        return (str(output_path or ""),)


NODE_CLASS_MAPPINGS = {
    "VRGDGStandaloneVideoEnhancer": VRGDGStandaloneVideoEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDGStandaloneVideoEnhancer": "VRGDG Standalone Video Enhancer",
}
