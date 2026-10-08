"""Per-scene audio stems and the masked mix the video model hears (the Audio Mask feature).

A scene's audio is split into stems with Demucs: vocals, drums, bass and other, or with ``htdemucs_6s`` also guitar and
piano. For every stem the user sets whether it is masked, the regions that stay audible, a level and mute. The masked mix
is the sum of the stems. It is what the video model is rendered with, so the character only lip-syncs the wanted vocal
parts. The finished clip gets the original audio back.

Files live in ``<project>/audio_masks/<scene_id>/``: ``original.wav``, one wav per stem, ``masked_mix.wav`` and
``mask.json``. Every wav is 44.1 kHz stereo 16 bit and exactly the scene length.
"""

import math
import os
import re
import subprocess
import threading
import time
import wave
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.atomic_write import atomic_write_json
from .audio import _find_ffmpeg_path
from .paths import _load_json_file, _resolve_existing_file

SAMPLE_RATE = 44100
FOUR_STEMS = ("vocals", "drums", "bass", "other")
SIX_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
ALL_STEM_NAMES = SIX_STEMS
STEMS_BY_MODEL = {"htdemucs": FOUR_STEMS, "htdemucs_ft": FOUR_STEMS, "mdx_extra": FOUR_STEMS, "htdemucs_6s": SIX_STEMS}
MODEL_NAMES = tuple(STEMS_BY_MODEL)
DEVICES = ("auto", "cuda", "cpu")
META_NAME = "mask.json"
PEAKS_NAME = "peaks.json"
PEAK_COUNT = 900
DEFAULT_FADE_MS = 30.0
MIN_REGION_SECONDS = 0.005
# Names the scene folder may hold: everything this module writes.
FILE_NAMES = ("original", *ALL_STEM_NAMES, "masked_mix")

_SEPARATE_LOCK = threading.Lock()


def _number(payload: Dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


def scene_folder(project_folder: Any, scene_id: Any, create: bool = False) -> str:
    """The folder that holds one scene's stems, inside the project folder."""
    text = str(project_folder or "").strip().strip('"')
    project = os.path.abspath(text) if text else ""
    if not project or not os.path.isdir(project):
        raise ValueError("Project folder is empty or does not exist.")
    name = str(scene_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name) or name in {".", ".."}:
        raise ValueError("Scene id is missing or not valid.")
    folder = os.path.abspath(os.path.join(project, "audio_masks", name))
    if os.path.commonpath([project, folder]) != project:
        raise ValueError("Scene folder escapes the project folder.")
    if create:
        os.makedirs(folder, exist_ok=True)
    return folder


def file_path(project_folder: Any, scene_id: Any, name: str) -> str:
    """Path of one named wav (see ``FILE_NAMES``) for a scene. Raises when the name is not allowed."""
    if name not in FILE_NAMES:
        raise ValueError(f"Unknown audio mask file: {name}")
    return os.path.join(scene_folder(project_folder, scene_id), f"{name}.wav")


def _read_wav(path: str) -> np.ndarray:
    """A 16 bit wav as float32 ``[2, samples]``."""
    with wave.open(path, "rb") as handle:
        channels, width, frames = handle.getnchannels(), handle.getsampwidth(), handle.getnframes()
        if width != 2:
            raise ValueError(f"Expected 16 bit audio in {path}.")
        data = np.frombuffer(handle.readframes(frames), dtype="<i2").astype(np.float32) / 32768.0
    data = data.reshape(-1, channels).T if channels > 1 else data.reshape(1, -1)
    return data if data.shape[0] >= 2 else np.repeat(data, 2, axis=0)


def _write_wav(path: str, audio: np.ndarray) -> None:
    """Write float ``[2, samples]`` as a 44.1 kHz 16 bit wav. Replaces the file in one step."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped.T * 32767.0).round().astype("<i2")
    temp = f"{path}.tmp"
    with wave.open(temp, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    os.replace(temp, path)


def _fit(audio: np.ndarray, samples: int) -> np.ndarray:
    """Cut or zero-pad ``[2, T]`` to exactly ``samples``."""
    if audio.shape[1] >= samples:
        return audio[:, :samples]
    return np.pad(audio, ((0, 0), (0, samples - audio.shape[1])))


def peaks(audio: np.ndarray, count: int = PEAK_COUNT) -> List[float]:
    """Largest absolute level in each of ``count`` equal slices, for drawing a waveform."""
    level = np.abs(audio).max(axis=0) if audio.size else np.zeros(1, dtype=np.float32)
    bucket = max(1, int(math.ceil(level.shape[0] / float(count))))
    padded = np.pad(level, (0, (-level.shape[0]) % bucket))
    return [round(float(value), 4) for value in padded.reshape(-1, bucket).max(axis=1)]


def _ffmpeg(command: List[str], message: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or message).strip()[-600:])


def _run_demucs(mix: np.ndarray, model_name: str, device: str) -> Dict[str, np.ndarray]:
    """Split a ``[2, T]`` mix into the stems of ``model_name``, with the same Demucs the VRGDG Get Stems node uses."""
    import torch

    from ..general import audio as demucs_audio

    node = demucs_audio.VRGDG_GetStems()
    device_name = node._resolve_device(device)
    model = demucs_audio.VRGDG_GetStems._get_model(model_name, device_name)
    model.to(device_name)
    # On the CPU, leave half of it to the server that streams the video, so a split in the background does not stall playback.
    previous_threads = torch.get_num_threads()
    if device_name == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    try:
        waveform = torch.from_numpy(np.ascontiguousarray(mix)).unsqueeze(0)
        prepared, rate = node._normalize_for_demucs(waveform, SAMPLE_RATE, model)
        with torch.no_grad(), demucs_audio._real_demucs_active():
            try:
                separated = demucs_audio.apply_model(model, prepared.to(device_name), device=device_name, progress=False)
            except TypeError:
                separated = demucs_audio.apply_model(model, prepared.to(device_name))
        separated = separated.detach()
        if separated.ndim == 4:
            separated = separated[0]
        sources = [str(name).strip().lower() for name in getattr(model, "sources", [])]
    finally:
        torch.set_num_threads(previous_threads)
        # The video render needs the memory next, so the model leaves the GPU until the next separation.
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    wanted = STEMS_BY_MODEL[model_name]
    missing = [name for name in wanted if name not in sources]
    if missing:
        raise ValueError(f"Demucs model {model_name} did not return: {', '.join(missing)}")
    stems: Dict[str, np.ndarray] = {}
    for name in wanted:
        stem = separated[sources.index(name)].float().cpu()
        if rate != SAMPLE_RATE:
            import torchaudio

            stem = torchaudio.functional.resample(stem, rate, SAMPLE_RATE)
        stems[name] = stem.numpy()
    return stems


def normalize_regions(raw: Any, duration: float) -> List[Dict[str, float]]:
    """Keep regions as sorted ``{start, end, fade_ms}`` in scene seconds, clamped to the scene."""
    regions: List[Dict[str, float]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            start, end = float(item.get("start")), float(item.get("end"))
            fade = float(item.get("fade_ms", DEFAULT_FADE_MS))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(fade)):
            continue
        start, end = max(0.0, min(duration, start)), max(0.0, min(duration, end))
        if end - start < MIN_REGION_SECONDS:
            continue
        regions.append({"start": round(start, 4), "end": round(end, 4), "fade_ms": max(0.0, min(1000.0, fade))})
    return sorted(regions, key=lambda region: region["start"])


def normalize_stem_settings(raw: Any, names: Any, duration: float) -> Dict[str, Dict[str, Any]]:
    """Per stem: ``mask`` (only the regions are audible), ``regions``, ``db`` (level) and ``mute``. Missing stems play as they are."""
    source = raw if isinstance(raw, dict) else {}
    settings: Dict[str, Dict[str, Any]] = {}
    for name in names:
        item = source.get(name) if isinstance(source.get(name), dict) else {}
        settings[name] = {
            "mask": bool(item.get("mask", False)),
            "regions": normalize_regions(item.get("regions"), duration),
            "db": _number(item, "db", 0.0, -100.0, 24.0),
            "mute": bool(item.get("mute", False)),
        }
    return settings


def keep_envelope(regions: List[Dict[str, float]], samples: int) -> np.ndarray:
    """1 where a stem is kept, 0 where it is silenced, with a smooth fade inside each region's edges."""
    envelope = np.zeros(samples, dtype=np.float32)
    for region in regions:
        first = max(0, int(round(region["start"] * SAMPLE_RATE)))
        last = min(samples, int(round(region["end"] * SAMPLE_RATE)))
        if last <= first:
            continue
        piece = np.ones(last - first, dtype=np.float32)
        fade = min(int(region["fade_ms"] / 1000.0 * SAMPLE_RATE), (last - first) // 2)
        if fade > 0:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(fade, dtype=np.float32) / fade)).astype(np.float32)
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
        envelope[first:last] = np.maximum(envelope[first:last], piece)
    return envelope


def mix_stems(stems: Dict[str, np.ndarray], settings: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """The masked mix, ``[2, samples]``: every stem with its mask, level and mute applied, added together."""
    samples = next(iter(stems.values())).shape[1]
    mix = np.zeros((2, samples), dtype=np.float32)
    for name, audio in stems.items():
        item = settings.get(name) or {"mask": False, "regions": [], "db": 0.0, "mute": False}
        if item["mute"]:
            continue
        piece = audio * (10.0 ** (item["db"] / 20.0))
        if item["mask"]:
            piece = piece * keep_envelope(item["regions"], samples)[None, :]
        mix += piece
    top = float(np.abs(mix).max()) if mix.size else 0.0
    if top > 0.98:
        mix = mix * (0.98 / top)
    return mix


def _meta_path(folder: str) -> str:
    return os.path.join(folder, META_NAME)


def _load_meta(folder: str) -> Dict[str, Any]:
    path = _meta_path(folder)
    if not os.path.isfile(path):
        return {}
    data = _load_json_file(path)
    return data if isinstance(data, dict) else {}


def _stem_names(meta: Dict[str, Any]) -> List[str]:
    """The stems a scene was split into. Scenes split before the 6 stem model existed have the four basic ones."""
    saved = (meta.get("separation") or {}).get("stem_names")
    names = [name for name in saved if name in ALL_STEM_NAMES] if isinstance(saved, list) else []
    return names or list(FOUR_STEMS)


def _wav_samples(path: str) -> int:
    """Length of a wav in samples, from its header."""
    with wave.open(path, "rb") as handle:
        return handle.getnframes()


def _lane_peaks(folder: str, files: Dict[str, str]) -> Dict[str, List[float]]:
    """Waveform peaks for each wav. They are kept in peaks.json, so reading a scene's state does not read and analyze
    every stem again. An entry is reused while its wav has the same size and modified time, so no invalidation is needed."""
    cache_path = os.path.join(folder, PEAKS_NAME)
    saved: Any = {}
    if os.path.isfile(cache_path):
        try:
            saved = _load_json_file(cache_path)
        except ValueError:
            saved = {}
    saved = saved if isinstance(saved, dict) else {}
    lanes: Dict[str, List[float]] = {}
    changed = False
    for name, wav_path in files.items():
        stamp = f"{os.path.getmtime(wav_path):.3f}:{os.path.getsize(wav_path)}"
        entry = saved.get(name)
        if isinstance(entry, dict) and entry.get("stamp") == stamp and isinstance(entry.get("peaks"), list):
            lanes[name] = entry["peaks"]
            continue
        lanes[name] = peaks(_read_wav(wav_path))
        saved[name] = {"stamp": stamp, "peaks": lanes[name]}
        changed = True
    for name in [key for key in saved if key not in files]:
        del saved[name]
        changed = True
    if changed:
        atomic_write_json(cache_path, saved)
    return lanes


def scene_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    """What exists for a scene: settings used, file paths and waveform peaks. ``exists`` is false before the first separation."""
    folder = scene_folder(payload.get("project_folder"), payload.get("scene_id"))
    meta = _load_meta(folder)
    names = _stem_names(meta)
    stem_files = {name: os.path.join(folder, f"{name}.wav") for name in names}
    if not meta or not all(os.path.isfile(path) for path in stem_files.values()):
        return {"ok": True, "exists": False, "scene_id": str(payload.get("scene_id") or "")}
    samples = _wav_samples(next(iter(stem_files.values())))
    files = {name: os.path.join(folder, f"{name}.wav") for name in ("original", *names, "masked_mix")}
    files = {name: path for name, path in files.items() if os.path.isfile(path)}
    mix = meta.get("mix") if isinstance(meta.get("mix"), dict) else None
    if not (mix and "masked_mix" in files):
        mix = None
    wanted = {name: path for name, path in files.items() if name != "masked_mix" or mix}
    lanes = _lane_peaks(folder, wanted)
    return {
        "ok": True,
        "exists": True,
        "scene_id": str(payload.get("scene_id") or ""),
        "duration": round(samples / float(SAMPLE_RATE), 4),
        "stem_names": names,
        "separation": meta.get("separation", {}),
        "mix": mix,
        "files": files,
        "peaks": lanes,
    }


MAX_BATCH_SCENES = 100


def scene_states(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The state of many scenes at once (what ``scene_state`` returns for each), so a project opens with one request instead of
    one per scene. A scene whose state cannot be read is reported as having no stems."""
    ids = payload.get("scene_ids")
    if not isinstance(ids, list):
        raise ValueError("scene_ids must be a list.")
    states: Dict[str, Any] = {}
    for scene_id in [str(item) for item in ids[:MAX_BATCH_SCENES]]:
        try:
            states[scene_id] = scene_state({"project_folder": payload.get("project_folder"), "scene_id": scene_id})
        except (ValueError, OSError):
            states[scene_id] = {"ok": True, "exists": False, "scene_id": scene_id}
    return {"ok": True, "states": states}


def separate_scene_stems(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Cut the scene's audio (with some context on both sides), apply the input gain, split it into stems and save them.

    The context makes Demucs more accurate at the scene edges. Only the scene itself is kept. Splitting again with a
    higher ``input_gain_db`` helps when the vocals are quiet. The old masked mix is removed because it no longer matches.
    """
    folder = scene_folder(payload.get("project_folder"), payload.get("scene_id"), create=True)
    source = _resolve_existing_file(payload.get("audio_path"), "Scene audio")
    start = _number(payload, "start_seconds", 0.0, 0.0, 86400.0)
    duration = _number(payload, "duration_seconds", 0.0, 0.0, 300.0)
    if duration < 0.05:
        raise ValueError("The scene needs a length before its audio can be split.")
    model_name = str(payload.get("model_name") or "htdemucs")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown Demucs model: {model_name}")
    device = str(payload.get("device") or "auto")
    if device not in DEVICES:
        raise ValueError(f"Unknown device: {device}")
    gain_db = _number(payload, "input_gain_db", 0.0, -24.0, 24.0)
    context = _number(payload, "context_seconds", 3.0, 0.0, 10.0)
    lead = min(context, start)
    samples = int(round(duration * SAMPLE_RATE))
    ffmpeg = _find_ffmpeg_path()
    context_path = os.path.join(folder, "_context.wav")
    original_path = os.path.join(folder, "original.wav")
    filters = []
    if gain_db:
        filters.append(f"volume={gain_db:.2f}dB")
        if gain_db > 0:
            filters.append("alimiter=limit=0.97")
    common_out = ["-vn", "-ac", "2", "-ar", str(SAMPLE_RATE)]
    started = time.time()
    with _SEPARATE_LOCK:
        _ffmpeg([ffmpeg, "-y", "-ss", f"{start - lead:.6f}", "-i", source, "-t", f"{lead + duration + context:.6f}", *common_out,
                 *(["-af", ",".join(filters)] if filters else []), "-c:a", "pcm_s16le", context_path], "ffmpeg could not cut the scene audio.")
        _ffmpeg([ffmpeg, "-y", "-ss", f"{start:.6f}", "-i", source, "-t", f"{duration:.6f}", *common_out,
                 "-c:a", "pcm_s16le", original_path], "ffmpeg could not cut the scene audio.")
        try:
            stems = _run_demucs(_read_wav(context_path), model_name, device)
        finally:
            if os.path.isfile(context_path):
                os.remove(context_path)
    first = int(round(lead * SAMPLE_RATE))
    for name in ALL_STEM_NAMES:
        stale_path = os.path.join(folder, f"{name}.wav")
        if name in stems:
            _write_wav(stale_path, _fit(stems[name][:, first:first + samples], samples))
        elif os.path.isfile(stale_path):
            os.remove(stale_path)  # a stem the new model does not have, left by an earlier split
    _write_wav(original_path, _fit(_read_wav(original_path), samples))
    if os.path.isfile(os.path.join(folder, "masked_mix.wav")):
        os.remove(os.path.join(folder, "masked_mix.wav"))
    atomic_write_json(_meta_path(folder), {
        "separation": {
            "model_name": model_name, "stem_names": list(stems), "input_gain_db": gain_db, "context_seconds": context,
            "device": device, "source_path": source, "start_seconds": start, "duration_seconds": duration,
            "created_at": time.time(), "seconds_taken": round(time.time() - started, 2),
        },
    })
    return scene_state(payload)


def render_masked_mix(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build ``masked_mix.wav`` from the saved stems and the per stem settings in ``payload["stems"]``."""
    folder = scene_folder(payload.get("project_folder"), payload.get("scene_id"))
    meta = _load_meta(folder)
    names = _stem_names(meta)
    stem_files = {name: os.path.join(folder, f"{name}.wav") for name in names}
    if not meta or not all(os.path.isfile(path) for path in stem_files.values()):
        raise ValueError("Split this scene's audio into stems first.")
    stems = {name: _read_wav(path) for name, path in stem_files.items()}
    samples = next(iter(stems.values())).shape[1]
    duration = samples / float(SAMPLE_RATE)
    settings = normalize_stem_settings(payload.get("stems"), names, duration)
    _write_wav(os.path.join(folder, "masked_mix.wav"), _fit(mix_stems(stems, settings), samples))
    meta["mix"] = {"stems": settings, "built_at": time.time(), "path": os.path.join(folder, "masked_mix.wav"),
                   "duration_seconds": round(duration, 4)}
    atomic_write_json(_meta_path(folder), meta)
    return scene_state(payload)


def list_scene_stems(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The ids of the scenes that have stems saved in the project, so stems of scenes that no longer exist can be removed."""
    text = str(payload.get("project_folder") or "").strip().strip('"')
    project = os.path.abspath(text) if text else ""
    if not project or not os.path.isdir(project):
        raise ValueError("Project folder is empty or does not exist.")
    root = os.path.join(project, "audio_masks")
    scene_ids = []
    if os.path.isdir(root):
        scene_ids = sorted(name for name in os.listdir(root) if os.path.isfile(os.path.join(root, name, META_NAME)))
    return {"ok": True, "scene_ids": scene_ids}


def delete_scene_stems(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove a scene's stems and masked mix."""
    folder = scene_folder(payload.get("project_folder"), payload.get("scene_id"))
    removed = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name in (META_NAME, PEAKS_NAME) or name.endswith((".wav", ".tmp")):
                os.remove(os.path.join(folder, name))
                removed += 1
        try:
            os.rmdir(folder)
        except OSError:
            pass
    return {"ok": True, "removed": removed}


def audio_override_for_scene(segment: Dict[str, Any], project_folder: str, duration: float) -> Optional[Dict[str, Any]]:
    """The masked mix a render should use for this scene, or None when the scene has no enabled, current mask.

    ``duration`` is the scene length now. A mix built for a different length is out of date and raises, so a render
    never uses audio that no longer matches the scene.
    """
    mask = segment.get("audio_mask") if isinstance(segment, dict) else None
    if not isinstance(mask, dict) or not mask.get("enabled"):
        return None
    folder = scene_folder(project_folder, segment.get("id"))
    meta = _load_meta(folder)
    mix = meta.get("mix") if isinstance(meta.get("mix"), dict) else None
    path = os.path.join(folder, "masked_mix.wav")
    if not mix or not os.path.isfile(path):
        raise ValueError("This scene's Audio Mask is on but its masked mix has not been built. Open Audio Mask and build it, or turn the mask off.")
    if abs(float(mix.get("duration_seconds", 0.0)) - float(duration)) > 0.02:
        raise ValueError("This scene's length changed after its Audio Mask was built. Open Audio Mask, split again and rebuild.")
    return {"path": path, "start_seconds": 0.0, "duration_seconds": float(duration)}
