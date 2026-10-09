"""Video Builder audio: SRT files, waveform peaks and beats, saving and trimming audio, and scene audio mixes."""

import math
import os
import re
import subprocess
import shutil
import sys
import wave
import base64
import array
import uuid
import folder_paths

from ..core.atomic_write import atomic_write_text
from .paths import _copy_file_into_folder, _load_json_file, _newest_file, _resolve_existing_file, _safe_project_name


def _default_audio_srt_paths():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    srt_folder = os.path.join(repo_dir, "srt_files")
    legacy_srt_folder = os.path.join(repo_dir, "SRT_Files")
    audio_folder = os.path.join(folder_paths.get_output_directory(), "VRGDG_AudioFiles")
    srt_path = _newest_file(srt_folder, (".srt",)) or _newest_file(legacy_srt_folder, (".srt",))
    return {
        "audio_path": _newest_file(audio_folder, (".wav", ".mp3", ".flac", ".m4a", ".ogg")),
        "srt_path": srt_path,
        "audio_folder": audio_folder,
        "srt_folder": srt_folder,
    }


def _srt_path(project_folder):
    return os.path.join(project_folder, "builder_segments.srt")


def _scene_audio_folder(project_folder):
    return os.path.join(project_folder, "scene_audio")


def _scene_audio_path(project_folder, scene_number, extension=".wav"):
    scene = max(1, int(scene_number or 1))
    ext = str(extension or ".wav").lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        ext = ".wav"
    return os.path.join(_scene_audio_folder(project_folder), f"audio_{scene:04d}{ext}")


def _find_ffmpeg_path():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except Exception:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or imageio-ffmpeg to mix scene audio.") from exc


def _convert_audio_to_wav(source_path, target_path):
    source = _resolve_existing_file(source_path, "Audio file")
    target = os.path.abspath(str(target_path or "").strip().strip('"'))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    command = [
        _find_ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        target,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        error_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Could not convert audio to WAV: {error_text or f'ffmpeg exited with code {result.returncode}'}")
    if not os.path.isfile(target) or os.path.getsize(target) <= 0:
        raise ValueError("Audio conversion finished, but the WAV file was not created.")
    return target


def _copy_or_convert_project_audio(source_path, target_folder, target_name=None):
    source = _resolve_existing_file(source_path, "Audio file")
    name = target_name or os.path.basename(source)
    if os.path.splitext(source)[1].lower() == ".m4a":
        safe_stem = _safe_project_name(os.path.splitext(name)[0])
        return _convert_audio_to_wav(source, os.path.join(target_folder, f"{safe_stem}.wav"))
    return _copy_file_into_folder(source, target_folder, name)


def _format_srt_time(seconds):
    total_ms = max(0, int(round(float(seconds or 0) * 1000)))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_srt_time(text):
    match = re.match(r"^\s*(\d+):(\d+):(\d+)[,.](\d+)\s*$", str(text or ""))
    if not match:
        raise ValueError(f"Invalid SRT time: {text}")
    hours, minutes, seconds, millis = [int(part) for part in match.groups()]
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _parse_srt_segments(srt_text):
    blocks = re.split(r"\n\s*\n", str(srt_text or "").strip(), flags=re.MULTILINE)
    segments = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_index = next((idx for idx, line in enumerate(lines) if "-->" in line), -1)
        if timing_index < 0:
            continue
        left, right = [part.strip() for part in lines[timing_index].split("-->", 1)]
        start = _parse_srt_time(left)
        end = max(start + 0.1, _parse_srt_time(right))
        label = " ".join(lines[timing_index + 1:]).strip() or f"Scene {len(segments) + 1}"
        segments.append(
            {
                "id": f"srt_{len(segments) + 1}_{int(start * 1000)}",
                "start": round(start, 3),
                "end": round(end, 3),
                "label": label[:80] or f"Scene {len(segments) + 1}",
                "notes": label,
                "t2i_prompt": "",
                "i2v_prompt": "",
                "ref_image_path": "",
                "use_vision_reference": False,
                "image": None,
                "source": "srt",
            }
        )
    return segments


def _load_srt_segments(path):
    srt_path = _resolve_existing_file(path, "SRT file")
    with open(srt_path, "r", encoding="utf-8-sig") as handle:
        segments = _parse_srt_segments(handle.read())
    if not segments:
        raise ValueError("No SRT timing blocks were found.")
    return {"srt_path": srt_path, "segments": segments}


def _segments_to_srt(segments, text_field="label"):
    lines = []
    ordered = sorted(segments, key=lambda item: float(item.get("start", 0) or 0))
    for index, segment in enumerate(ordered, start=1):
        start = float(segment.get("start", 0) or 0)
        end = max(start + 0.1, float(segment.get("end", start + 4) or start + 4))
        text = str(segment.get(text_field) or segment.get("label") or segment.get("t2i_prompt") or f"Scene {index}").strip()
        lines.extend([str(index), f"{_format_srt_time(start)} --> {_format_srt_time(end)}", text, ""])
    return "\n".join(lines).strip() + "\n"


def _read_audio_peaks_with_torchaudio(audio_path, target_peaks=1600):
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        return {"duration": 0, "sample_rate": sample_rate, "channels": 0, "peaks": []}
    channels = int(waveform.shape[0])
    mono = waveform.mean(dim=0).abs()
    total_samples = int(mono.numel())
    duration = total_samples / float(sample_rate or 1)
    peak_count = max(1, min(int(target_peaks or 1600), total_samples))
    samples_per_peak = max(1, math.ceil(total_samples / peak_count))
    padded = peak_count * samples_per_peak - total_samples
    if padded > 0:
        mono = torch.nn.functional.pad(mono, (0, padded))
    chunks = mono.reshape(peak_count, samples_per_peak)
    peaks_tensor = torch.sqrt(torch.mean(chunks * chunks, dim=1)).clamp(0, 1)
    return {
        "duration": duration,
        "sample_rate": int(sample_rate or 0),
        "channels": channels,
        "peaks": [float(value) for value in peaks_tensor.cpu().tolist()],
    }


def _read_audio_peaks_with_wave(audio_path, target_peaks=1600):
    with wave.open(audio_path, "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        duration = frames / float(sample_rate or 1)
        if sample_width not in (1, 2, 4):
            raise ValueError("Only 8-bit, 16-bit, and 32-bit WAV files are supported by the fallback reader.")
        raw = handle.readframes(frames)

    if not raw:
        return {"duration": duration, "sample_rate": sample_rate, "channels": channels, "peaks": []}

    import audioop

    mono = raw
    if channels > 1:
        mono = audioop.tomono(raw, sample_width, 0.5, 0.5)
    total_samples = len(mono) // sample_width
    peak_count = max(1, min(int(target_peaks or 1600), total_samples))
    samples_per_peak = max(1, math.ceil(total_samples / peak_count))
    peaks = []
    for offset in range(0, len(mono), samples_per_peak * sample_width):
        chunk = mono[offset: offset + samples_per_peak * sample_width]
        if not chunk:
            continue
        rms = audioop.rms(chunk, sample_width)
        maximum = float((1 << ((sample_width * 8) - 1)) - 1)
        peaks.append(min(1.0, rms / maximum if maximum else 0.0))

    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "peaks": peaks,
    }


def _read_audio_peaks_with_ffmpeg(audio_path, target_peaks=1600):
    sample_rate = 16000
    command = [
        _find_ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        error_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(error_text or f"ffmpeg exited with code {result.returncode}")
    samples = array.array("f")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    total_samples = len(samples)
    duration = total_samples / float(sample_rate)
    if total_samples <= 0:
        return {"duration": 0, "sample_rate": sample_rate, "channels": 1, "peaks": []}
    peak_count = max(1, min(int(target_peaks or 1600), total_samples))
    samples_per_peak = max(1, math.ceil(total_samples / peak_count))
    peaks = []
    for start in range(0, total_samples, samples_per_peak):
        chunk = samples[start: start + samples_per_peak]
        if not chunk:
            continue
        total = 0.0
        for value in chunk:
            total += float(value) * float(value)
        peaks.append(min(1.0, math.sqrt(total / len(chunk))))
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": 1,
        "peaks": peaks,
    }


def _read_audio_peaks(audio_path, target_peaks=1600):
    try:
        return _read_audio_peaks_with_torchaudio(audio_path, target_peaks)
    except Exception as torch_exc:
        try:
            return _read_audio_peaks_with_wave(audio_path, target_peaks)
        except Exception as wave_exc:
            try:
                return _read_audio_peaks_with_ffmpeg(audio_path, target_peaks)
            except Exception as ffmpeg_exc:
                raise ValueError(
                    "Could not read audio for waveform. Try a standard WAV, MP3, FLAC, or M4A file. "
                    f"torchaudio error: {torch_exc}; wav fallback error: {wave_exc}; ffmpeg fallback error: {ffmpeg_exc}"
                )


def _estimate_beats_from_peaks(peaks, duration):
    values = [float(value or 0) for value in peaks or []]
    total_duration = float(duration or 0)
    if len(values) < 8 or total_duration <= 0:
        return []
    step = total_duration / len(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(max(0.0, variance))
    threshold = mean + (std * 0.65)
    min_gap = max(0.22, min(0.55, total_duration / 500))
    beats = []
    beat_values = []
    last_time = -999.0
    for index in range(1, len(values) - 1):
        value = values[index]
        if value < threshold:
            continue
        if value < values[index - 1] or value < values[index + 1]:
            continue
        beat_time = index * step
        if beat_time - last_time < min_gap:
            # Keep the strongest peak in the minimum-gap window. Do not turn a
            # rounded timestamp back into an array index: floating-point
            # truncation can select the preceding bin and incorrectly replace
            # a stronger, correctly timed peak with a later weaker one.
            if beats and value > beat_values[-1]:
                beats[-1] = round(beat_time, 3)
                beat_values[-1] = value
                last_time = beat_time
            continue
        beats.append(round(beat_time, 3))
        beat_values.append(value)
        last_time = beat_time
    return beats


def _coerce_tempo_bpm(value):
    try:
        if hasattr(value, "reshape"):
            value = value.reshape(-1)[0]
        elif isinstance(value, (list, tuple)):
            value = value[0] if value else 0
        bpm = float(value or 0)
    except (TypeError, ValueError, IndexError):
        return 0.0
    return round(bpm, 6) if math.isfinite(bpm) and bpm > 0 else 0.0


def _tempo_from_beat_times(beats):
    values = sorted(float(value) for value in beats or [] if math.isfinite(float(value)))
    intervals = [
        values[index] - values[index - 1]
        for index in range(1, len(values))
        if values[index] - values[index - 1] > 0.05
    ]
    if not intervals:
        return 0.0
    intervals.sort()
    middle = len(intervals) // 2
    median = intervals[middle] if len(intervals) % 2 else (intervals[middle - 1] + intervals[middle]) / 2.0
    return round(60.0 / median, 6) if median > 0 else 0.0


def _estimate_beats_from_audio(audio_path, peaks, duration, include_tempo=False):
    """Return musical beat positions, falling back to RMS peaks when needed."""
    try:
        import librosa

        waveform, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
        if waveform is None or len(waveform) < 2:
            raise ValueError("Audio contains no samples.")
        onset_envelope = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
        if onset_envelope is None or len(onset_envelope) < 2:
            raise ValueError("Audio contains no detectable onset envelope.")
        tempo_bpm, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            trim=False,
        )
        # Put each grid beat on the transient's leading edge instead of the
        # later energy maximum, matching what the waveform shows visually.
        beat_frames = librosa.onset.onset_backtrack(beat_frames, onset_envelope)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        maximum = max(0.0, float(duration or (len(waveform) / float(sample_rate or 1))))
        result = []
        for value in beat_times:
            beat_time = round(float(value), 3)
            if beat_time < 0 or (maximum > 0 and beat_time > maximum + 0.001):
                continue
            if not result or beat_time > result[-1]:
                result.append(beat_time)
        if result:
            bpm = _coerce_tempo_bpm(tempo_bpm) or _tempo_from_beat_times(result)
            return (result, bpm) if include_tempo else result
    except Exception:
        # librosa is optional in some ComfyUI installations. The corrected RMS
        # detector remains available so audio loading never depends on it.
        pass
    result = _estimate_beats_from_peaks(peaks, duration)
    bpm = _tempo_from_beat_times(result)
    return (result, bpm) if include_tempo else result


def _extract_capcut_project_beats(draft, draft_path=""):
    if not isinstance(draft, dict):
        return None
    materials = draft.get("materials") if isinstance(draft.get("materials"), dict) else {}
    audio_materials = {
        str(item.get("id") or ""): item
        for item in materials.get("audios", []) or []
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    audio_segments = []
    for track in draft.get("tracks", []) or []:
        if not isinstance(track, dict) or str(track.get("type") or "").lower() != "audio":
            continue
        audio_segments.extend(item for item in track.get("segments", []) or [] if isinstance(item, dict))
    audio_segment = audio_segments[0] if audio_segments else {}
    audio_material = audio_materials.get(str(audio_segment.get("material_id") or ""), {})
    referenced_ids = {str(value) for value in audio_segment.get("extra_material_refs", []) or [] if str(value)}

    time_marks = [item for item in materials.get("time_marks", []) or [] if isinstance(item, dict)]
    linked_time_marks = [item for item in time_marks if str(item.get("id") or "") in referenced_ids]
    marker_times = []
    for collection in linked_time_marks or time_marks:
        for marker in collection.get("mark_items", []) or []:
            if not isinstance(marker, dict):
                continue
            time_range = marker.get("time_range") if isinstance(marker.get("time_range"), dict) else {}
            try:
                marker_time = float(time_range.get("start") or 0) / 1_000_000.0
            except (TypeError, ValueError):
                continue
            if marker_time >= 0:
                marker_times.append(round(marker_time, 6))
    marker_times = sorted(set(marker_times))

    beat_materials = [item for item in materials.get("beats", []) or [] if isinstance(item, dict)]
    linked_beats = [item for item in beat_materials if str(item.get("id") or "") in referenced_ids]
    beat_material = (linked_beats or beat_materials or [{}])[0]
    ai_beats = beat_material.get("ai_beats") if isinstance(beat_material.get("ai_beats"), dict) else {}
    beat_cache_path = os.path.normpath(str(ai_beats.get("beats_path") or "").strip())
    cache_times = []
    beat_values = []
    if beat_cache_path and os.path.isfile(beat_cache_path):
        try:
            cache_data = _load_json_file(beat_cache_path)
            if isinstance(cache_data, dict):
                for value in cache_data.get("time", []) or []:
                    try:
                        cache_time = float(value) / 1000.0
                    except (TypeError, ValueError):
                        continue
                    if cache_time >= 0:
                        cache_times.append(round(cache_time, 6))
                beat_values = list(cache_data.get("value", []) or [])
        except Exception:
            cache_times = []
            beat_values = []

    # CapCut's visible project markers are frame-aligned. Prefer them when they
    # correspond one-for-one with the AI cache; otherwise use the raw AI times.
    if marker_times and (not cache_times or abs(len(marker_times) - len(cache_times)) <= 1):
        beats = marker_times
        beat_source = "timeline_markers"
    else:
        beats = sorted(set(cache_times))
        beat_source = "ai_beat_cache"
    if len(beats) < 2:
        return None
    duration = float(draft.get("duration") or 0) / 1_000_000.0
    return {
        "project_name": str(draft.get("name") or "").strip() or os.path.basename(os.path.dirname(draft_path)),
        "draft_path": os.path.abspath(draft_path) if draft_path else "",
        "project_fps": float(draft.get("fps") or 0),
        "project_duration": duration,
        "audio_name": str(audio_material.get("name") or "").strip(),
        "audio_path": str(audio_material.get("path") or "").strip(),
        "beat_cache_path": beat_cache_path,
        "beat_source": beat_source,
        "beats": beats,
        "raw_ai_beats": cache_times,
        "beat_values": beat_values,
    }


def _find_latest_capcut_beats(audio_duration=0):
    local_app_data = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    index_path = os.path.join(
        local_app_data,
        "CapCut",
        "User Data",
        "Projects",
        "com.lveditor.draft",
        "root_meta_info.json",
    )
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"CapCut project index was not found: {index_path}")
    index_data = _load_json_file(index_path)
    entries = index_data.get("all_draft_store", []) if isinstance(index_data, dict) else []
    entries = sorted(
        (item for item in entries if isinstance(item, dict) and not item.get("tm_draft_removed")),
        key=lambda item: float(item.get("tm_draft_modified") or 0),
        reverse=True,
    )
    requested_duration = max(0.0, float(audio_duration or 0))
    latest_with_beats = None
    for entry in entries[:150]:
        draft_path = os.path.normpath(str(entry.get("draft_json_file") or "").strip())
        if not draft_path or not os.path.isfile(draft_path):
            continue
        try:
            result = _extract_capcut_project_beats(_load_json_file(draft_path), draft_path)
        except Exception:
            continue
        if not result:
            continue
        result["project_name"] = str(entry.get("draft_name") or result.get("project_name") or "").strip()
        result["project_modified"] = float(entry.get("tm_draft_modified") or 0)
        latest_with_beats = latest_with_beats or result
        if requested_duration <= 0 or abs(float(result.get("project_duration") or 0) - requested_duration) <= 0.75:
            return result
    if latest_with_beats and requested_duration <= 0:
        return latest_with_beats
    if latest_with_beats:
        raise ValueError(
            "CapCut projects with beat data were found, but none matched the loaded audio duration within 0.75 seconds."
        )
    raise ValueError("No CapCut project containing beat data was found.")


def _audio_bytes_from_data_url(audio_data):
    raw = str(audio_data or "").strip()
    if not raw:
        raise ValueError("Audio data is empty.")
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def _save_scene_audio(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    scene_number = int(payload.get("scene_number") or 1)
    os.makedirs(_scene_audio_folder(project_folder), exist_ok=True)

    source_name = str(payload.get("audio_name", "") or "").strip()
    source_ext = os.path.splitext(source_name)[1].lower()
    audio_data = str(payload.get("audio_data", "") or "").strip()
    if audio_data:
        target_path = _scene_audio_path(project_folder, scene_number, source_ext or ".wav")
        if payload.get("preserve_source"):
            folder = os.path.join(_scene_audio_folder(project_folder), "sources")
            os.makedirs(folder, exist_ok=True)
            target_path = os.path.join(folder, f"source_{uuid.uuid4().hex}{source_ext or '.wav'}")
        with open(target_path, "wb") as handle:
            handle.write(_audio_bytes_from_data_url(audio_data))
    else:
        source_path = _resolve_existing_file(payload.get("source_path", ""), "Audio file")
        target_path = _scene_audio_path(project_folder, scene_number, os.path.splitext(source_path)[1] or ".wav")
        if payload.get("preserve_source"):
            folder = os.path.join(_scene_audio_folder(project_folder), "sources")
            os.makedirs(folder, exist_ok=True)
            extension = os.path.splitext(source_path)[1] or ".wav"
            target_path = os.path.join(folder, f"source_{uuid.uuid4().hex}{extension}")
        shutil.copy2(source_path, target_path)

    audio_info = _read_audio_peaks(target_path, 600)
    return {
        "saved_path": target_path,
        "audio_folder": _scene_audio_folder(project_folder),
        "scene_number": scene_number,
        **audio_info,
    }


def _save_project_audio(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    os.makedirs(project_folder, exist_ok=True)
    folder = os.path.join(project_folder, "project_audio")
    os.makedirs(folder, exist_ok=True)
    source_name = str(payload.get("audio_name", "") or "").strip() or "project_audio.wav"
    ext = os.path.splitext(source_name)[1].lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        ext = ".wav"
    target_path = os.path.join(folder, f"project_audio{'.wav' if ext == '.m4a' else ext}")
    raw_target_path = os.path.join(folder, f"project_audio_source{ext}") if ext == ".m4a" else target_path
    audio_data = str(payload.get("audio_data", "") or "").strip()
    if audio_data:
        with open(raw_target_path, "wb") as handle:
            handle.write(_audio_bytes_from_data_url(audio_data))
    else:
        source_path = _resolve_existing_file(payload.get("source_path", ""), "Audio file")
        if ext == ".m4a":
            shutil.copy2(source_path, raw_target_path)
        else:
            shutil.copy2(source_path, target_path)
    if ext == ".m4a":
        target_path = _convert_audio_to_wav(raw_target_path, target_path)
        try:
            if os.path.abspath(raw_target_path) != os.path.abspath(target_path):
                os.remove(raw_target_path)
        except Exception:
            pass
    audio_info = _read_audio_peaks(target_path, 1600)
    beats, tempo_bpm = _estimate_beats_from_audio(
        target_path,
        audio_info.get("peaks", []),
        audio_info.get("duration", 0),
        include_tempo=True,
    )
    return {"saved_path": target_path, "audio_folder": folder, **audio_info, "beats": beats, "tempo_bpm": tempo_bpm}


def _save_project_srt(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    os.makedirs(project_folder, exist_ok=True)
    srt_text = str(payload.get("srt_text", "") or "")
    if not srt_text.strip():
        raise ValueError("SRT text is empty.")
    path = _srt_path(project_folder)
    atomic_write_text(path, srt_text)
    segments = _parse_srt_segments(srt_text)
    return {"srt_path": path, "segments": segments}


def _save_single_scene_srt(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    scene_number = int(payload.get("scene_number") or 1)
    duration = max(0.1, float(payload.get("duration") or 4))
    start_time = max(0.0, float(payload.get("start_time") or 0))
    end_time = start_time + duration
    label = str(payload.get("label") or f"Scene {scene_number}").strip()
    folder = os.path.join(project_folder, "scene_srt")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"scene_{scene_number:04d}.srt")
    text = "\n".join([
        "1",
        f"{_format_srt_time(start_time)} --> {_format_srt_time(end_time)}",
        label,
        "",
    ])
    atomic_write_text(path, text)
    return {"srt_path": path, "scene_number": scene_number, "start_time": start_time, "duration": duration}


def _trim_scene_audio(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    source_path = _resolve_existing_file(payload.get("source_path", ""), "Audio file")
    scene_number = int(payload.get("scene_number") or 1)
    start = max(0.0, float(payload.get("start") or 0))
    duration = max(0.05, float(payload.get("duration") or 0))
    source_info = _read_audio_peaks(source_path, 16)
    source_duration = float(source_info.get("duration") or 0)
    if source_duration > 0:
        remaining = source_duration - start
        if remaining <= 0.01:
            raise ValueError(
                f"Scene {scene_number} audio trim starts after the source audio ends. "
                f"Trim start: {start:.3f}s; audio length: {source_duration:.3f}s. "
                "Shorten or move the scene, load longer audio, or add silence before rendering."
            )
        duration = min(duration, max(0.05, remaining))
    folder = os.path.join(project_folder, "scene_audio_trimmed")
    os.makedirs(folder, exist_ok=True)
    target_path = os.path.join(folder, f"scene_audio_{scene_number:04d}.wav")
    cmd = [
        _find_ffmpeg_path(),
        "-y",
        "-ss",
        str(start),
        "-i",
        source_path,
        "-t",
        str(duration),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        target_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed to trim scene audio.").strip())
    trimmed_info = _read_audio_peaks(target_path, 16)
    trimmed_duration = float(trimmed_info.get("duration") or 0)
    if trimmed_duration <= 0.01:
        raise ValueError(
            f"Scene {scene_number} audio trim was empty. "
            f"Trim start: {start:.3f}s; requested duration: {duration:.3f}s. "
            "Shorten or move the scene, load longer audio, or add silence before rendering."
        )
    return {
        "audio_path": target_path,
        "scene_number": scene_number,
        "start": start,
        "duration": trimmed_duration,
        "requested_duration": float(payload.get("duration") or 0),
        "format": "pcm_s16le_wav",
    }




def _concat_file_path(path):
    return os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")


def _scene_audio_mix_folder(project_folder):
    return os.path.join(project_folder, "project_audio")


def _prepare_scene_audio_mix(payload):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        raise ValueError("Project folder is empty.")
    segments = payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError("No scenes were provided for scene audio mix.")
    allow_missing_scene_audio = bool(payload.get("allow_missing_scene_audio", False))
    global_audio_path = os.path.abspath(
        str(payload.get("global_audio_path", "") or "").strip().strip('"')
    )
    if not os.path.isfile(global_audio_path):
        global_audio_path = ""

    ffmpeg_path = _find_ffmpeg_path()
    folder = _scene_audio_mix_folder(project_folder)
    os.makedirs(folder, exist_ok=True)
    parts_folder = os.path.join(folder, "_scene_audio_mix_parts")
    if os.path.isdir(parts_folder):
        shutil.rmtree(parts_folder, ignore_errors=True)
    os.makedirs(parts_folder, exist_ok=True)

    timeline_items = []
    missing = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            missing.append(f"Scene {index}: invalid scene data.")
            continue
        path = str(segment.get("custom_audio_path", "") or "").strip().strip('"')
        if not path:
            start = max(0.0, float(segment.get("start", 0) or 0))
            end = max(start + 0.05, float(segment.get("end", start + 4) or start + 4))
            duration = max(0.05, end - start)
            if global_audio_path:
                timeline_items.append({
                    "index": index,
                    "path": global_audio_path,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "source_start": start,
                    "silent": False,
                })
                continue
            if allow_missing_scene_audio:
                timeline_items.append({
                    "index": index,
                    "path": "",
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "source_start": 0.0,
                    "silent": True,
                })
                continue
            missing.append(f"Scene {index}: custom audio is missing.")
            continue
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            missing.append(f"Scene {index}: custom audio file was not found: {path}")
            continue
        segment_start = max(0.0, float(segment.get("start", 0) or 0))
        segment_end = max(segment_start + 0.05, float(segment.get("end", segment_start + 4) or segment_start + 4))
        start = max(0.0, float(segment.get("custom_audio_timeline_start", segment_start) or segment_start))
        duration = float(segment.get("custom_audio_duration", 0) or 0)
        if duration <= 0:
            duration = segment_end - segment_start
        duration = max(0.05, duration)
        source_start = max(0.0, float(segment.get("custom_audio_source_start", 0) or 0))
        timeline_items.append({
            "index": index,
            "path": path,
            "start": start,
            "end": start + duration,
            "duration": duration,
            "source_start": source_start,
            "silent": False,
        })
    if missing:
        raise ValueError("\n".join(missing))

    timeline_items.sort(key=lambda item: (item["start"], item["index"]))
    concat_file = os.path.join(parts_folder, "scene_audio_mix_list.txt")
    part_paths = []
    cursor = 0.0
    part_index = 1
    for item in timeline_items:
        gap = max(0.0, item["start"] - cursor)
        if gap > 0.01:
            silence_path = os.path.join(parts_folder, f"part_{part_index:04d}_silence.wav")
            part_index += 1
            silence_cmd = [
                ffmpeg_path,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{gap:.6f}",
                "-c:a",
                "pcm_s16le",
                silence_path,
            ]
            result = subprocess.run(silence_cmd, capture_output=True, text=True, errors="replace", check=False)
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed to create silence.").strip())
            part_paths.append(silence_path)

        clip_path = os.path.join(parts_folder, f"part_{part_index:04d}_scene_{item['index']:04d}.wav")
        part_index += 1
        if item.get("silent"):
            clip_cmd = [
                ffmpeg_path,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{item['duration']:.6f}",
                "-c:a",
                "pcm_s16le",
                clip_path,
            ]
        else:
            clip_cmd = [
                ffmpeg_path,
                "-y",
                "-ss",
                f"{item['source_start']:.6f}",
                "-i",
                item["path"],
                "-t",
                f"{item['duration']:.6f}",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                clip_path,
            ]
        result = subprocess.run(clip_cmd, capture_output=True, text=True, errors="replace", check=False)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or f"ffmpeg failed to prepare scene {item['index']} audio.").strip())
        part_paths.append(clip_path)
        cursor = max(cursor, item["start"] + item["duration"])

    if not part_paths:
        raise ValueError("No scene audio parts were created.")

    mix_path = os.path.join(folder, "scene_audio_mix.wav")
    atomic_write_text(concat_file, "".join(f"file '{_concat_file_path(path)}'\n" for path in part_paths))
    mix_cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c:a",
        "pcm_s16le",
        mix_path,
    ]
    result = subprocess.run(mix_cmd, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed to create scene audio mix.").strip())

    srt_path = _srt_path(project_folder)
    text_field = "lyric_text" if str(payload.get("srt_text_field", "") or "").strip() == "lyric_text" else "label"
    atomic_write_text(srt_path, _segments_to_srt(segments, text_field=text_field))

    shutil.rmtree(parts_folder, ignore_errors=True)
    audio_info = _read_audio_peaks(mix_path, 1600)
    beats, tempo_bpm = _estimate_beats_from_audio(
        mix_path,
        audio_info.get("peaks", []),
        audio_info.get("duration", cursor),
        include_tempo=True,
    )
    return {
        "audio_path": mix_path,
        "srt_path": srt_path,
        "duration": audio_info.get("duration", cursor),
        "peaks": audio_info.get("peaks", []),
        "beats": beats,
        "tempo_bpm": tempo_bpm,
        "scene_count": len(timeline_items),
        "used_scene_audio": True,
    }


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
    return {
        "ok": True,
        "audio_path": path,
        "saved_path": path,
        "audio_folder": folder,
        "audio_name": display_name,
        "scope": scope,
        "scene_number": scene_number,
        **_read_audio_peaks(path, target_peaks),
    }
