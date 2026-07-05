import os
import wave

from aiohttp import web
from server import PromptServer

try:
    from .VRGDG_MusicVideoBuilderNodes import _read_audio_peaks
except Exception:
    try:
        from VRGDG_MusicVideoBuilderNodes import _read_audio_peaks
    except Exception:
        _read_audio_peaks = None


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _clean_duration(value):
    try:
        duration = float(value)
    except Exception:
        duration = 0.0
    if duration <= 0:
        raise ValueError("Silence duration must be greater than 0 seconds.")
    return max(0.1, min(duration, 24 * 60 * 60))


def _safe_scene_number(value):
    try:
        return max(1, int(value or 1))
    except Exception:
        return 1


def _duration_label(duration):
    text = f"{duration:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def _write_silent_wav(path, duration, sample_rate=44100, channels=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total_frames = int(round(duration * sample_rate))
    chunk_frames = sample_rate
    frame = b"\x00\x00" * channels
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        remaining = total_frames
        while remaining > 0:
            count = min(chunk_frames, remaining)
            handle.writeframes(frame * count)
            remaining -= count
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise ValueError("Silent WAV file was not created.")


def _create_silent_audio(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    os.makedirs(project_folder, exist_ok=True)

    duration = _clean_duration(payload.get("duration"))
    scope = str(payload.get("scope") or "project").strip().lower()
    duration_tag = _duration_label(duration)

    if scope == "scene":
        scene_number = _safe_scene_number(payload.get("scene_number"))
        folder = os.path.join(project_folder, "scene_audio")
        path = os.path.join(folder, f"audio_{scene_number:04d}.wav")
        display_name = f"Silence {duration:.2f}s"
        target_peaks = 600
    else:
        scope = "project"
        scene_number = 0
        folder = os.path.join(project_folder, "project_audio")
        path = os.path.join(folder, f"project_silence_{duration_tag}s.wav")
        display_name = f"Silent timeline {duration:.2f}s"
        target_peaks = 1600

    _write_silent_wav(path, duration)
    info = _read_audio_peaks(path, target_peaks) if callable(_read_audio_peaks) else {"duration": duration, "peaks": [], "beats": []}
    return {
        "ok": True,
        "audio_path": path,
        "saved_path": path,
        "audio_folder": folder,
        "audio_name": display_name,
        "scope": scope,
        "scene_number": scene_number,
        **info,
    }


@PromptServer.instance.routes.post("/vrgdg/music_builder/create_silent_audio")
async def vrgdg_music_builder_create_silent_audio(request):
    try:
        payload = await request.json()
        return web.json_response(_create_silent_audio(payload or {}))
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
