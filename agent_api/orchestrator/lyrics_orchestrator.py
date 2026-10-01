"""Lyric timing and scenes-from-lines jobs (Apireport Phase 1; the Video Builder's Line Mapping).

* ``lyrics.align`` runs the same ComfyUI timestamp workflow the Builder's Line Mapping dialog queues
  (Stable-ts on the song, matched to the pasted lyrics) and saves the timed lines.
* ``timeline.from_lines`` turns those timed lines into timeline scenes, then applies the minimum and
  maximum scene length, like creating scenes in Line Mapping and then merging and cutting them.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from ...builder.lyric_scenes import (
    DEFAULT_INSTRUMENTAL_TEXT,
    REFERENCE_UNIT_MODES,
    apply_lyric_sections,
    clean_timestamped_lyric_text,
    enforce_scene_lengths,
    normalize_scene_durations,
    segments_from_timestamped_payload,
)
from ...core.atomic_write import atomic_write_json
from ...runner import utility_workflows
from ..errors import JobCancelledError, ValidationError
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session
from ..paths import session_audio_path
from .comfy_client import extract_text_from_history, get_comfy_client

_PAYLOAD_FILE = "timestamped_lyrics.json"
_SEGMENT_MODES = ("whisper_chunks", "reference_lines", "exact_reference_lines", "reference_stanzas")


def parse_timestamped_lyrics_output(text: str) -> Dict[str, Any]:
    """Find the timestamped-lyrics JSON in the workflow's text output (mirrors ``parseTimestampedLyricsOutput``)."""
    raw = str(text or "").strip()
    try:
        direct = json.loads(raw)
        if isinstance(direct, dict) and isinstance(direct.get("segments"), list):
            return direct
    except ValueError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, dict) and isinstance(parsed.get("segments"), list):
                return parsed
        except ValueError:
            pass
    raise ValidationError("Timestamped transcription finished, but no timestamped lyrics JSON was found.")


def assert_no_bundled_reference_lyrics(lyric_values: List[str], reference_lyrics: str) -> None:
    """Mirror ``assertNoBundledReferenceLyrics``: one scene must not receive several pasted lyric lines."""
    def normalize(value: Any) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[\W_]+", " ", clean_timestamped_lyric_text(value).lower())).strip()

    reference_lines = list(dict.fromkeys(
        line for line in (normalize(raw) for raw in str(reference_lyrics or "").replace("\r\n", "\n").split("\n")) if len(line) >= 4
    ))
    if len(reference_lines) < 2:
        return
    for index, value in enumerate(lyric_values, start=1):
        text = normalize(value)
        if not text or text in reference_lines:
            continue
        contained = [line for line in reference_lines if line in text]
        if len(contained) >= 2:
            raise ValidationError(
                f"Transcription safety check stopped the update: scene {index} received {len(contained)} pasted lyric lines in one result.",
                details={"scene": index},
            )


def project_reference_lyrics(session: Dict[str, Any], folder: str) -> str:
    """The pasted reference lyrics: the Line Mapping text, else the project's saved lyrics file."""
    mapper = session.get("lyric_mapper") if isinstance(session.get("lyric_mapper"), dict) else {}
    text = str(mapper.get("source_text") or session.get("canonical_lyrics") or "").strip()
    if text:
        return text
    for candidate in (
        os.path.join(folder, "project_context", "full_lyrics.txt"),
        os.path.join(folder, "full_lyrics.txt"),
    ):
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
            if text:
                return text
    return ""


def _alignment_request(session: Dict[str, Any], folder: str, params: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = str(params.get("audio_path") or "").strip().strip('"') or session_audio_path(session)
    if not audio_path or not os.path.isfile(audio_path):
        raise ValidationError("The project has no audio file to time the lyrics against. Attach audio first.")
    lyrics = str(params.get("reference_lyrics") or "").strip() or project_reference_lyrics(session, folder)
    mode = str(params.get("segment_mode") or "reference_lines").strip().lower()
    if mode not in _SEGMENT_MODES:
        raise ValidationError(f"segment_mode must be one of: {', '.join(_SEGMENT_MODES)}.")
    if mode != "whisper_chunks" and not lyrics:
        raise ValidationError("Reference lyrics are needed for this segment mode. Set the project lyrics first.")
    return {
        "audio_path": audio_path,
        "reference_lyrics": lyrics,
        "language": str(params.get("language") or "english"),
        "segment_mode": mode,
        "include_instrumental_gaps": bool(params.get("include_instrumental_gaps", True)),
        "instrumental_text": str(params.get("instrumental_text") or DEFAULT_INSTRUMENTAL_TEXT),
        "min_gap_seconds": float(params.get("min_gap_seconds", 2.0)),
        "min_scene_seconds": float(params.get("min_scene_seconds", 1.0)),
        "max_scene_seconds": float(params.get("max_scene_seconds", 8.0)),
        "vocal_tail_padding_seconds": float(params.get("vocal_tail_padding_seconds", 0.6)),
        "model_name": str(params.get("model_name") or "large-v3"),
    }


def _payload_path(folder: str) -> str:
    return os.path.join(folder, "project_context", _PAYLOAD_FILE)


async def align_lyrics(
    project_id: str,
    params: Optional[Dict[str, Any]] = None,
    job: Optional[Job] = None,
    manager: Optional[JobManager] = None,
) -> Dict[str, Any]:
    """Time the lyrics against the song with the ComfyUI timestamp workflow and save the result."""
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    request = _alignment_request(session, folder, params)

    if manager and job:
        manager.update_progress(job.id, 5.0, "building_workflow", message="Building the timestamp workflow...")
    graph = await asyncio.to_thread(utility_workflows._build_timestamped_transcribe_api_prompt, request)
    client = get_comfy_client()
    queued = await asyncio.to_thread(client.queue_prompt, graph["prompt"])
    prompt_id = queued["prompt_id"]
    if manager and job:
        manager.set_current_comfy_prompt(job.id, prompt_id, project_folder=folder)
        manager.update_progress(job.id, 15.0, "aligning_lyrics", message="Timing the lyrics against the song...")

    check_cancel = (lambda: job.cancel_requested) if job else None
    history = await client.wait_for_prompt(prompt_id, timeout_seconds=45 * 60, check_cancel=check_cancel)
    texts = extract_text_from_history(history, prompt_id)
    payload = parse_timestamped_lyrics_output("\n".join(texts))
    payload.setdefault("segment_mode", request["segment_mode"])

    path = _payload_path(folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    await asyncio.to_thread(atomic_write_json, path, payload)
    return {
        "payload_path": path,
        "lines": len(payload.get("segments") or []),
        "duration": payload.get("duration"),
        "segment_mode": payload.get("segment_mode"),
        "request": {key: value for key, value in request.items() if key not in ("reference_lyrics", "audio_path")},
    }


def _has_scene_media(segment: Dict[str, Any]) -> bool:
    keys = ("approved_image_path", "custom_image_path", "video_path", "rendered_video_path")
    return any(str(segment.get(key) or "").strip() for key in keys) or bool(segment.get("image_history")) or bool(segment.get("video_history"))


def build_scenes_from_payload(payload: Dict[str, Any], params: Dict[str, Any], reference_lyrics: str) -> List[Dict[str, Any]]:
    """Timed lines -> scenes with section labels and the min/max length rule. Pure, no files."""
    mode = str(params.get("segment_mode") or payload.get("segment_mode") or "reference_lines").strip().lower()
    instrumental = str(params.get("instrumental_text") or DEFAULT_INSTRUMENTAL_TEXT)
    min_s = float(params.get("min_scene_seconds", 1.0))
    max_s = float(params.get("max_scene_seconds", 8.0))
    scenes = segments_from_timestamped_payload(
        payload,
        segment_mode=mode,
        include_instrumental_gaps=bool(params.get("include_instrumental_gaps", True)),
        instrumental_text=instrumental,
        min_gap_seconds=float(params.get("min_gap_seconds", 2.0)),
        max_scene_seconds=max_s,
    )
    scenes = normalize_scene_durations(scenes, min_scene_seconds=min_s, max_scene_seconds=max_s, segment_mode=mode, instrumental_text=instrumental)
    if not scenes:
        raise ValidationError("Timestamped lines did not produce any usable scene segments.")
    if mode in ("reference_lines", "exact_reference_lines"):
        assert_no_bundled_reference_lyrics([s.get("lyric_text", "") for s in scenes], reference_lyrics)
    if params.get("enforce_lengths", True):
        # Reference-line modes keep one lyric line per scene whatever its length. The agent asked for
        # a length range, so merge short lines together and cut long ones, like editing in the Builder.
        enforce_scene_lengths(scenes, min_s, max_s, instrumental)
    apply_lyric_sections(scenes, reference_lyrics)
    return scenes


async def create_timeline_from_lines(
    project_id: str,
    params: Optional[Dict[str, Any]] = None,
    job: Optional[Job] = None,
    manager: Optional[JobManager] = None,
) -> Dict[str, Any]:
    """Create the project's scenes from timed lyric lines (aligning first unless a saved result is reused)."""
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    existing = [s for s in session.get("segments") or [] if isinstance(s, dict)]
    if existing and not params.get("replace_existing"):
        raise ValidationError(
            f"The project already has {len(existing)} scenes. Pass replace_existing=true to rebuild them from the lyrics."
        )
    if any(_has_scene_media(s) for s in existing):
        raise ValidationError("Some scenes already have images or videos. Delete those first; rebuilding would orphan them.")

    request = _alignment_request(session, folder, params)
    saved_path = _payload_path(folder)
    if params.get("use_saved_alignment") and os.path.isfile(saved_path):
        with open(saved_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if manager and job:
            manager.update_progress(job.id, 50.0, "using_saved_alignment", message="Using the saved lyric timing...")
    else:
        aligned = await align_lyrics(project_id, params, job=job, manager=manager)
        with open(aligned["payload_path"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)

    if job and job.cancel_requested:
        raise JobCancelledError(job.id)
    if manager and job:
        manager.update_progress(job.id, 80.0, "creating_scenes", message="Creating timeline scenes...")
    scenes = await asyncio.to_thread(build_scenes_from_payload, payload, {**params, **request}, request["reference_lyrics"])

    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        session["segments"] = scenes
        session["overlay_segments"] = []
        session["lyric_mapper"] = {"source_text": request["reference_lyrics"], "lines": []}
        # The Video Builder turns the lyric lane on once lyrics are placed on the timeline.
        session["show_timeline_lyric_notes"] = True
        duration = float(payload.get("duration") or 0.0)
        if duration > 0:
            session["audio_duration"] = max(float(session.get("audio_duration") or 0.0), duration)
        saved = _persist_session(folder, session)

    lengths = [round(float(s["end"]) - float(s["start"]), 2) for s in scenes]
    return {
        "scenes": len(scenes),
        "shortest_seconds": min(lengths),
        "longest_seconds": max(lengths),
        "segment_mode": request["segment_mode"],
        "preserved_reference_units": request["segment_mode"] in REFERENCE_UNIT_MODES and not params.get("enforce_lengths", True),
        "revision": saved.get("revision"),
    }


async def run_lyrics_align_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    return await align_lyrics(job.project_id, job.params, job=job, manager=manager)


async def run_timeline_from_lines_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    result = await create_timeline_from_lines(job.project_id, job.params, job=job, manager=manager)
    manager.update_progress(job.id, 100.0, "completed", message=f"Created {result['scenes']} scenes.")
    return result


def register_lyrics_orchestrator_handlers(manager: Optional[JobManager] = None) -> None:
    """Register the lyric timing and scene creation jobs."""
    manager = manager or get_job_manager()
    manager.register_handler("lyrics.align", run_lyrics_align_job)
    manager.register_handler("timeline.from_lines", run_timeline_from_lines_job)
