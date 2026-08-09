import os
import shutil
import subprocess
import uuid

import folder_paths


def _find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def _video_path_candidates(value):
    candidates = []
    if isinstance(value, str):
        candidates.append(value)
    elif isinstance(value, dict):
        for key in ("fullpath", "path", "video_path", "filename"):
            item = value.get(key)
            if isinstance(item, str):
                candidates.append(item)
        for key in ("files", "filenames", "videos", "gifs"):
            candidates.extend(_video_path_candidates(value.get(key)))
    elif isinstance(value, (list, tuple)):
        for item in value:
            candidates.extend(_video_path_candidates(item))
    return candidates


def _resolve_video_path(value, label):
    roots = [
        "",
        folder_paths.get_output_directory(),
        folder_paths.get_temp_directory(),
        folder_paths.get_input_directory(),
    ]
    candidates = _video_path_candidates(value)
    for raw_path in reversed(candidates):
        text = str(raw_path or "").strip().strip('"')
        if not text or os.path.splitext(text)[1].lower() not in _VIDEO_EXTENSIONS:
            continue
        for root in roots:
            path = text if not root or os.path.isabs(text) else os.path.join(root, text)
            path = os.path.normpath(os.path.abspath(path))
            if os.path.isfile(path):
                return path
    raise ValueError(
        f"{label} video was not found. Connect the Filenames output from a "
        "Video Combine node that has already created a video."
    )


def _write_image_batch_video(images, label, fps):
    """Materialize a Comfy IMAGE batch so it can be previewed as a video."""
    import numpy as np

    if images is None:
        return None
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None, ...]
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] < 3:
        raise ValueError(f"{label} image input must be a non-empty IMAGE batch.")

    height, width = int(images.shape[1]), int(images.shape[2])
    output_dir = folder_paths.get_temp_directory()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"vrgdg_compare_{label.lower()}_{uuid.uuid4().hex}.mp4")
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("The compare slider needs FFmpeg to convert IMAGE batches into a browser-playable video.")

    command = [
        ffmpeg_path, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(max(1.0, float(fps))), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output_path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in images:
            frame = np.clip(frame[..., :3] * 255.0, 0, 255).astype(np.uint8)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg could not create the compare video: {stderr[-1000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return output_path


def _ensure_video_compare_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None or getattr(server_instance, "_vrgdg_compare_routes", False):
        return

    @server_instance.routes.post("/vrgdg/video_compare/save_recording")
    async def vrgdg_video_compare_save_recording(request):
        temporary_path = ""
        try:
            reader = await request.multipart()
            field = await reader.next()
            if field is None or field.name != "file":
                return web.json_response({"ok": False, "error": "Recording upload is missing its file."}, status=400)
            temporary_path = os.path.join(
                folder_paths.get_temp_directory(),
                f"vrgdg_compare_upload_{uuid.uuid4().hex}.webm",
            )
            with open(temporary_path, "wb") as handle:
                while True:
                    chunk = await field.read_chunk(size=1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

            ffmpeg_path = _find_ffmpeg()
            if not ffmpeg_path:
                raise RuntimeError("FFmpeg is required to save the recording as MP4.")
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"vrgdg_video_compare_{uuid.uuid4().hex[:10]}.mp4")
            command = [
                ffmpeg_path, "-y", "-i", temporary_path,
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", output_path,
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0 or not os.path.isfile(output_path):
                raise RuntimeError(f"FFmpeg could not save the MP4: {completed.stderr[-1000:]}")
            return web.json_response({"ok": True, "path": output_path, "name": os.path.basename(output_path)})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        finally:
            if temporary_path:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    server_instance._vrgdg_compare_routes = True


_ensure_video_compare_routes()


class VRGDG_VideoCompareSlider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slider_position": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Starting position of the before/after wipe.",
                    },
                ),
                "before_label": (
                    "STRING",
                    {"default": "Before", "tooltip": "Label shown over the original video."},
                ),
                "after_label": (
                    "STRING",
                    {"default": "After", "tooltip": "Label shown over the processed video."},
                ),
                "show_labels": ("BOOLEAN", {"default": True}),
                "loop": ("BOOLEAN", {"default": True}),
                "muted": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Mute playback. When unmuted, audio comes from the before video.",
                    },
                ),
                "record_duration": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.5,
                        "max": 300.0,
                        "step": 0.5,
                        "tooltip": "How long the browser Record button captures the labeled slider preview.",
                    },
                ),
            },
            "optional": {
                "before_images": (
                    "IMAGE",
                    {"tooltip": "Connect the IMAGE output from a video loader for the original video."},
                ),
                "after_images": (
                    "IMAGE",
                    {"tooltip": "Connect the IMAGE output from a video loader for the processed video."},
                ),
                "before_video": (
                    "VHS_FILENAMES",
                    {"tooltip": "Legacy: connect a Filenames output for the original video."},
                ),
                "after_video": (
                    "VHS_FILENAMES",
                    {"tooltip": "Legacy: connect a Filenames output for the processed video."},
                ),
            }
        }

    RETURN_TYPES = ("VHS_FILENAMES", "VHS_FILENAMES")
    RETURN_NAMES = ("before_video", "after_video")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "VRGDG/Image"

    def compare(
        self,
        slider_position,
        before_label,
        after_label,
        show_labels,
        loop,
        muted,
        record_duration,
        before_images=None,
        after_images=None,
        before_video=None,
        after_video=None,
    ):
        before_path = _write_image_batch_video(before_images, "before", 24.0) if before_images is not None else _resolve_video_path(before_video, "Before")
        after_path = _write_image_batch_video(after_images, "after", 24.0) if after_images is not None else _resolve_video_path(after_video, "After")
        return {
            "ui": {
                "compare_videos": [
                    {
                        "compare_role": "before",
                        "path": before_path,
                        "name": os.path.basename(before_path),
                    },
                    {
                        "compare_role": "after",
                        "path": after_path,
                        "name": os.path.basename(after_path),
                    },
                ],
                "compare_video_settings": {
                    "slider_position": float(slider_position),
                    "before_label": str(before_label or "Before"),
                    "after_label": str(after_label or "After"),
                    "show_labels": bool(show_labels),
                    "loop": bool(loop),
                    "muted": bool(muted),
                    "record_duration": max(0.5, float(record_duration)),
                },
            },
            "result": ((True, [before_path]), (True, [after_path])),
        }


NODE_CLASS_MAPPINGS = {
    "VRGDG_VideoCompareSlider": VRGDG_VideoCompareSlider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_VideoCompareSlider": "VRGDG Video Compare Slider",
}
