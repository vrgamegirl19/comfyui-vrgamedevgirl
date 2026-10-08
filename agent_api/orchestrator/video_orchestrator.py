"""Server-side video generation and lifecycle orchestrator (Section 6.9, Section 24.2, C4)."""

import asyncio
import copy
import logging
import math
import os
import re
import shutil
import time
from typing import Any, Callable, Dict, List, Optional
import folder_paths

from ...builder.audio import _find_ffmpeg_path
from ...builder import media as builder_media
from ...runner import ltx_workflows, minimax_inputs, minimax_workflows, video_files
from ..errors import (
    ComfyExecutionError,
    JobCancelledError,
    LatentStaleError,
    PredecessorMissingError,
    ProjectNotFoundError,
    SceneNotFoundError,
    ValidationError,
)
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..scene_video import apply_scene_video
from ..mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session
from ..paths import resolve_project_folder, session_audio_path, session_video_mode
from ...minimax.scene_inputs import canonical_continuity_mode, resolve_scene_inputs
from ...minimax.settings_payload import (
    build_minimax_render_payload,
    random_seed_value,
    randomize_minimax_seeds,
    minimax_h3_settings_for_scene,
    minimax_workflow_key,
)
from ..schemas import extract_effective_settings
from .comfy_client import (
    extract_videos_from_history,
    get_comfy_client,
)

logger = logging.getLogger("vrgdg.agent_api.video_orchestrator")


def resolve_comfy_video_path(video_info: Dict[str, Any]) -> str:
    """Resolve full filesystem path for a ComfyUI video output item."""
    filename = os.path.basename(str(video_info.get("filename", "") or ""))
    subfolder = str(video_info.get("subfolder", "") or "")
    vtype = str(video_info.get("type", "output") or "output").lower()
    if vtype == "temp":
        base_dir = folder_paths.get_temp_directory()
    elif vtype == "input":
        base_dir = folder_paths.get_input_directory()
    else:
        base_dir = folder_paths.get_output_directory()
    target = os.path.join(base_dir, subfolder, filename)
    return os.path.abspath(target)


def _extract_final_frame_for_continuity(project_folder: str, video_path: str, scene_number: int) -> str:
    """Save the last frame of ``video_path`` for scene ``scene_number`` and return its path."""
    extracted = builder_media._extract_video_final_frame_as_scene_image({
        "project_folder": project_folder,
        "source_path": video_path,
        "scene_number": scene_number,
    })
    return str(extracted.get("saved_path") or "").strip()


def build_video_graph_for_mode(mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compile ComfyUI prompt graph for the requested video mode (Section 6.9)."""
    m = str(mode or "i2v").strip().lower()
    if m in ("t2v", "text_to_video"):
        return ltx_workflows._build_t2v_api_prompt(payload)
    if m in ("rtv", "reference_to_video"):
        return ltx_workflows._build_rtv_api_prompt(payload)
    if m in ("ingredients", "ingredients_to_video"):
        return ltx_workflows._build_ingredients_api_prompt(payload)
    if m in ("flf", "first_last_frame"):
        return ltx_workflows._build_flf_api_prompt(payload)
    if m in ("id_lora", "id_lora_i2v"):
        return ltx_workflows._build_id_lora_api_prompt(payload)
    if m in ("minimax_h3_2pass",):
        return minimax_workflows._build_minimax_h3_2pass_api_prompt(payload)
    if m in ("minimax_h3_advanced_2pass",):
        return minimax_workflows._build_minimax_h3_advanced_2pass_api_prompt(payload)
    if m in ("minimax_h3_3pass",):
        return minimax_workflows._build_minimax_h3_3pass_api_prompt(payload)
    if m in ("minimax_h3", "minimax"):
        return minimax_workflows._build_minimax_h3_api_prompt(payload)
    # Default to standard I2V
    return ltx_workflows._build_i2v_api_prompt(payload)


async def render_scene_video_async(
    project_id: str,
    scene_id: str,
    params: Optional[Dict[str, Any]] = None,
    job: Optional[Job] = None,
    manager: Optional[JobManager] = None,
) -> Dict[str, Any]:
    """Render a single scene video with full pipeline steps (Section 6.9).

    Steps:
    1. Prepare scene audio clip and timing.
    2. Compile ComfyUI prompt graph for the requested mode.
    3. Queue prompt, wait for generation.
    4. Collect output video into rendered_scene_videos/ (with backup).
    5. Optional color match against predecessor scene.
    6. Optional trim.
    7. Update scene in session, bump revision, and persist.
    """
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    seg = segments[idx]
    scene_number = idx + 1
    p = dict(params or {})

    # Timing
    start_sec = float(seg.get("start", 0.0) or 0.0)
    end_sec = float(seg.get("end", start_sec + 4.0) or (start_sec + 4.0))
    duration_sec = max(0.25, end_sec - start_sec)

    # Mode
    mode = str(p.get("mode") or session_video_mode(session)).strip().lower()

    if manager and job:
        manager.update_progress(job.id, 5.0, "preparing", scene_id=scene_id, message=f"Preparing {mode} video render...")

    # Merge effective settings with parameter overrides
    effective_settings = extract_effective_settings(session)
    mode_group = "minimax_h3" if "minimax" in mode else "ltx_video"
    if mode_group == "minimax_h3":
        # Use every MiniMax option the Video Builder saved (scene overrides included)
        # so agent and UI renders match. A plain "minimax_h3" mode follows the saved
        # render pass (single, 2 Pass, 2 Pass Advanced); an explicit pass mode wins.
        minimax_settings = minimax_h3_settings_for_scene(session, seg)
        if p.get("randomize_seed"):
            # Same as the Render All "new seeds" option: fresh seeds for every pass, saved with the
            # project (or the scene, when it has its own MiniMax settings) so the render is reproducible.
            fresh_seeds = randomize_minimax_seeds(minimax_settings)
            with _BUILDER_SAVE_LOCK:
                _, live_session = _get_active_session_and_folder(project_id)
                live_segment = live_session["segments"][idx]
                if live_segment.get("use_scene_minimax_h3_settings") and isinstance(live_segment.get("minimax_h3_settings"), dict):
                    live_segment["minimax_h3_settings"].update(fresh_seeds)
                else:
                    live_session.setdefault("minimax_h3_settings", {}).update(fresh_seeds)
                _persist_session(folder, live_session)
        if mode in ("minimax_h3", "minimax"):
            mode = minimax_workflow_key(minimax_settings)
        try:
            payload = build_minimax_render_payload(minimax_settings)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        pass2_prompt = str(seg.get("minimax_h3_pass2_prompt") or "").strip()
        if pass2_prompt:
            payload["pass2_prompt"] = pass2_prompt
        if minimax_settings.get("pipeline") == "refmod":
            # The RefMod pipeline renders from saved RefMods in scene order (minimax/refmod_scene.py), with no images.
            from ...minimax.refmod_scene import assign_labels, reference_payload, refmod_items_for_scene

            items = assign_labels([i for i in refmod_items_for_scene(session, seg, idx) if i["strength"] > 0])
            if not items:
                raise ValidationError(
                    f"Scene {scene_number} needs at least one RefMod. Pick a RefMod on a Reference Builder card and map it to the scene.")
            continuity = str(p.get("continuity_mode") or minimax_settings["continuity_mode"] or "off")
            payload["pipeline"] = "refmod"
            payload["refmod_references"] = reference_payload(items)
            payload["continuity_mode"] = "latent_continuation_masked" if canonical_continuity_mode(continuity) == "latent_continuation_masked" else "off"
            payload["image_paths"] = []
            payload["video_references"] = []
        else:
            # Scene inputs the UI resolves per render: reference images, continuity frames,
            # reference videos and the explicit last frame (minimax/scene_inputs.py).
            try:
                scene_inputs = await asyncio.to_thread(
                    resolve_scene_inputs,
                    session,
                    seg,
                    minimax_settings["video_mode"],
                    idx,
                    continuity_mode=str(p.get("continuity_mode") or minimax_settings["continuity_mode"] or "off"),
                    previous_segment=segments[idx - 1] if idx > 0 else None,
                    project_folder=folder,
                    scene_number=scene_number,
                    extract_final_frame=_extract_final_frame_for_continuity,
                    configured_image_paths=p.get("image_paths"),
                    configured_video_references=p.get("video_references"),
                    render_pass=str(minimax_settings.get("render_pass") or "single"),
                )
            except ValueError as exc:
                raise ValidationError(f"Scene {scene_number}: {exc}") from exc
            if scene_inputs["missing_image_paths"]:
                raise ValidationError(
                    f"Scene {scene_number} references image files that do not exist: "
                    + ", ".join(scene_inputs["missing_image_paths"])
                )
            payload["continuity_mode"] = scene_inputs["continuity_mode"]
            payload["image_paths"] = scene_inputs["image_paths"]
            payload["video_references"] = scene_inputs["video_references"]
            if scene_inputs.get("last_frame_path"):
                payload["last_frame_path"] = scene_inputs["last_frame_path"]
    else:
        payload = dict(effective_settings.get(mode_group, {}))
        if p.get("randomize_seed"):
            # Same as setVideoSeedRandom in the UI: new seed, saved where the scene's LTX settings live.
            payload["seed"] = random_seed_value()
            with _BUILDER_SAVE_LOCK:
                _, live_session = _get_active_session_and_folder(project_id)
                live_segment = live_session["segments"][idx]
                if live_segment.get("use_scene_i2v_video_settings") and isinstance(live_segment.get("i2v_video_settings"), dict):
                    live_segment["i2v_video_settings"]["seed"] = payload["seed"]
                else:
                    live_session.setdefault("i2v_video_settings", {})["seed"] = payload["seed"]
                _persist_session(folder, live_session)
    payload.update(p)
    payload["project_folder"] = folder
    payload["scene_number"] = scene_number
    payload["start"] = start_sec
    payload["end"] = end_sec
    payload["duration"] = duration_sec

    # MiniMax Latent Continuity Validation (Section 6.10, Invariant 4)
    if "minimax" in mode:
        continuity_mode = str(payload.get("continuity_mode") or "").strip().lower()
        use_latent = payload.get("use_latent_continuation")
        is_latent_cont = canonical_continuity_mode(continuity_mode) == "latent_continuation_masked" or bool(use_latent)
        if is_latent_cont and scene_number > 1:
            from ...minimax.latent_manager import SceneLatentManager
            pred_scene = scene_number - 1
            if not SceneLatentManager.latent_exists(folder, pred_scene):
                raise PredecessorMissingError(scene_number, pred_scene)
            if SceneLatentManager.is_dirty(folder, pred_scene):
                raise LatentStaleError(scene_number, pred_scene)

        # Normalize common aspect ratio aliases
        raw_ar = str(payload.get("aspect_ratio") or "").strip()
        if raw_ar:
            ar_map = {
                "16:9": "16:9 (Widescreen)",
                "9:16": "9:16 (Portrait Widescreen)",
                "1:1": "1:1 (Square)",
                "4:3": "4:3 (Standard)",
                "3:4": "3:4 (Portrait Standard)",
                "3:2": "3:2 (Photo)",
                "2:3": "2:3 (Portrait Photo)",
                "21:9": "21:9 (Ultrawide)",
            }
            payload["aspect_ratio"] = ar_map.get(raw_ar, raw_ar)

    # Prompt text
    if "minimax" in mode:
        # The Video Builder's "create each next scene prompt from the previous rendered final frame": a scene continued with
        # latent_continuation_masked is written from the real last frame of the scene before it, right before it renders.
        if (
            mode_group == "minimax_h3" and scene_number > 1 and not p.get("prompt")
            and str(payload.get("continuity_mode") or "") == "latent_continuation_masked"
            and minimax_settings.get("continuity_prompt_from_last_frame")
        ):
            previous = segments[idx - 1]
            previous_video = str(previous.get("video_path") or previous.get("rendered_video_path") or "").strip()
            if not previous_video or not os.path.isfile(previous_video):
                raise ValidationError(
                    f"Scene {scene_number} is set to write its prompt from scene {scene_number - 1}'s final frame, "
                    f"but scene {scene_number - 1} has no rendered video. Render it first."
                )
            if manager and job:
                manager.update_progress(job.id, 6.0, "preparing", scene_id=scene_id, message=f"Writing the scene {scene_number} prompt from scene {scene_number - 1}'s final frame...")
            frame_path = await asyncio.to_thread(_extract_final_frame_for_continuity, folder, previous_video, scene_number)
            from .minimax_prompt_orchestrator import write_continued_scene_prompt

            written = await asyncio.to_thread(
                write_continued_scene_prompt, project_id, str(seg.get("id") or scene_id), frame_path, str(previous.get("id") or ""), p,
            )
            seg["minimax_h3_prompt"] = written["prompt"]
            for warning in written.get("warnings") or []:
                logger.warning(f"Scene {scene_number}: {warning}")
        # The Video Builder renders the saved MiniMax prompt, falling back to the video prompt.
        video_prompt = str(p.get("prompt") or seg.get("minimax_h3_prompt") or seg.get("i2v_prompt") or "").strip()
        if not video_prompt:
            raise ValidationError(f"Scene {scene_number} needs a MiniMax H3 prompt before it can render.")
    else:
        video_prompt = str(p.get("prompt") or seg.get("i2v_prompt") or seg.get("t2v_prompt") or seg.get("lyric_text") or "").strip()
    payload["i2v_prompt"] = video_prompt
    payload["t2v_prompt"] = video_prompt
    payload["prompt"] = video_prompt

    # Audio preparation
    audio_path = str(p.get("audio_path") or "").strip()
    project_audio = session_audio_path(session)
    if not audio_path and project_audio and os.path.isfile(project_audio):
        if mode_group == "minimax_h3":
            audio_path = project_audio
        else:
            try:
                prep_audio_res = await asyncio.to_thread(
                    minimax_inputs._prepare_scene_audio_clip,
                    {
                        "audio_path": project_audio,
                        "project_folder": folder,
                        "scene_number": scene_number,
                        "start_seconds": start_sec,
                        "duration_seconds": duration_sec,
                    },
                )
                audio_path = prep_audio_res.get("audio_path", "")
            except Exception as exc:
                logger.warning(f"Could not prepare scene audio clip for scene {scene_id}: {exc}")
    payload["audio_path"] = audio_path

    # SRT preparation
    srt_dir = os.path.join(folder, "scene_srt")
    os.makedirs(srt_dir, exist_ok=True)
    srt_file = os.path.join(srt_dir, f"scene_{scene_number:04d}.srt")
    if not os.path.isfile(srt_file):
        try:
            with open(srt_file, "w", encoding="utf-8") as f:
                f.write(f"1\n00:00:00,000 --> 00:00:{int(duration_sec):02d},000\n{seg.get('lyric_text') or video_prompt or 'Scene'}\n")
        except Exception:
            pass
    payload["srt_path"] = srt_file

    # Image inputs for I2V / MiniMax
    images_dir = os.path.join(folder, "images")
    os.makedirs(images_dir, exist_ok=True)
    payload["image_folder"] = images_dir

    scene_img = seg.get("approved_image_path") or seg.get("custom_image_path") or (seg.get("image_history") or [""])[-1]
    if scene_img and os.path.isfile(scene_img):
        scene_img_target = os.path.join(images_dir, f"image_{scene_number:04d}.png")
        try:
            if not os.path.isfile(scene_img_target) or os.path.getmtime(scene_img) > os.path.getmtime(scene_img_target):
                shutil.copy2(scene_img, scene_img_target)
        except Exception:
            pass
        payload["image_path"] = scene_img_target
        payload["first_frame"] = {"path": scene_img_target}

    if manager and job:
        manager.update_progress(job.id, 15.0, "compiling_graph", scene_id=scene_id, message="Compiling ComfyUI video graph...")

    # Compile prompt graph
    graph_res = await asyncio.to_thread(build_video_graph_for_mode, mode, payload)
    prompt_graph = graph_res.get("prompt")
    if not prompt_graph:
        raise ValidationError(f"Video mode '{mode}' compilation did not return a valid prompt graph.")

    # Queue with ComfyUI
    client = get_comfy_client()
    queue_res = await asyncio.to_thread(client.queue_prompt, prompt_graph)
    prompt_id = queue_res["prompt_id"]

    if manager and job:
        manager.set_current_comfy_prompt(job.id, prompt_id, project_folder=folder)
        manager.update_progress(job.id, 30.0, "rendering", scene_id=scene_id, message=f"Rendering {mode} video with ComfyUI...")

    # Wait for completion
    timeout_hours = float(p.get("timeout_hours", 1.0))
    timeout_sec = max(60.0, timeout_hours * 3600.0)
    check_cancel = (lambda: job.cancel_requested) if job else None
    history = await client.wait_for_prompt(prompt_id, timeout_seconds=timeout_sec, check_cancel=check_cancel)

    # Locate generated video output
    source_video_path = ""
    # MiniMax graphs save the final clip on node 142 and, for 2 Pass Advanced, a Pass 1 backup on
    # another node. Take the final clip, never the backup.
    final_node_id = graph_res.get("final_video_node_id") or ("142" if "minimax" in mode else None)
    videos = extract_videos_from_history(history, prompt_id, node_id=final_node_id)
    if videos:
        resolved = resolve_comfy_video_path(videos[-1])
        if os.path.isfile(resolved):
            source_video_path = resolved

    if not source_video_path:
        find_res = await asyncio.to_thread(
            video_files._find_scene_video_output,
            {
                "project_folder": folder,
                "video_mode": mode,
                "scene_number": scene_number,
                "prompt_number_one_based": scene_number,
                "output_folder": graph_res.get("output_folder", ""),
            },
        )
        source_video_path = find_res.get("video_path", "")

    if not source_video_path or not os.path.isfile(source_video_path):
        raise ComfyExecutionError(
            f"Video generation completed but output video was not found for prompt '{prompt_id}'.",
            prompt_id=prompt_id,
        )

    raw_render_path = source_video_path
    if manager and job:
        manager.update_progress(job.id, 75.0, "collecting", scene_id=scene_id, message="Collecting scene video...")

    # Like the Video Builder: cut the raw render to the exact timeline length first (label
    # "minimax_exact", marked as the audio video), then collect that clip as video_NNNN-audio.mp4.
    # Collecting the raw render would keep its extra frames and drift the stitched video off the song.
    trim_info = p.get("trim_params") or graph_res.get("post_render_trim") or {}
    if p.get("trim") or trim_info:
        if manager and job:
            manager.update_progress(job.id, 75.0, "trimming", scene_id=scene_id, message="Trimming scene video to the exact timeline length...")
        try:
            trim_res = await asyncio.to_thread(
                video_files._trim_scene_video,
                {
                    "project_folder": folder,
                    "scene_number": scene_number,
                    "source_path": source_video_path,
                    "start": float(trim_info.get("start", p.get("trim_start", 0.0))),
                    "duration": float(trim_info.get("duration", p.get("trim_duration", duration_sec))),
                    "frames": int(trim_info.get("frames", 0)),
                    "label": "minimax_exact" if "minimax" in mode else "trim",
                    "mark_as_audio_video": "minimax" in mode,
                },
            )
            source_video_path = trim_res.get("video_path") or source_video_path
        except Exception as t_err:
            logger.warning(f"Trimming failed for scene {scene_id}: {t_err}")

    if manager and job:
        manager.update_progress(job.id, 85.0, "collecting", scene_id=scene_id, message="Collecting scene video...")

    # Collect into rendered_scene_videos/ with backup per Section 24.2
    collect_payload = {
        "project_folder": folder,
        "scene_number": scene_number,
        "source_path": source_video_path,
        "existing_action": "backup",
    }
    collect_res = await asyncio.to_thread(video_files._collect_scene_video, collect_payload)
    final_video_path = collect_res["video_path"]
    final_thumbnail_path = collect_res.get("thumbnail_path", "")

    # Optional start color matching
    if p.get("match_start_color") and idx > 0:
        prev_seg = segments[idx - 1]
        prev_video = prev_seg.get("video_path") or prev_seg.get("rendered_video_path")
        if prev_video and os.path.isfile(prev_video):
            if manager and job:
                manager.update_progress(job.id, 90.0, "color_matching", scene_id=scene_id, message="Matching opening color to previous scene...")
            try:
                color_res = await asyncio.to_thread(
                    video_files._apply_scene_start_color_match,
                    {
                        "project_folder": folder,
                        "video_path": final_video_path,
                        "reference_video_path": prev_video,
                        "fade_seconds": float(p.get("color_match_fade_seconds", 1.0)),
                        "strength": float(p.get("color_match_strength", 0.85)),
                    },
                )
                final_video_path = color_res.get("video_path", final_video_path)
                final_thumbnail_path = color_res.get("thumbnail_path", final_thumbnail_path)
            except Exception as c_err:
                logger.warning(f"Color match failed for scene {scene_id}: {c_err}")

    # Update scene in session following Section 24.2 / Section 6.9 lifecycle
    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        target_seg = session["segments"][idx]
        apply_scene_video(target_seg, final_video_path, final_thumbnail_path if final_thumbnail_path else "")
        # The untrimmed render, so a later re-trim (POST .../video/trim with take) can start from the whole take.
        target_seg["raw_video_path"] = raw_render_path
        raw_history = [str(x) for x in target_seg.get("raw_video_history") or [] if str(x).strip()]
        if raw_render_path not in raw_history:
            raw_history.append(raw_render_path)
        target_seg["raw_video_history"] = raw_history
        # A replaced render is kept as a backup the timeline can switch back to (the UI records it the same way).
        if collect_res.get("backup_path"):
            backups = target_seg.setdefault("video_backup_paths", [])
            if collect_res["backup_path"] not in backups:
                backups.append(collect_res["backup_path"])
            if collect_res.get("backup_thumbnail_path"):
                thumbs = target_seg.setdefault("video_backup_thumbnail_paths", [])
                if collect_res["backup_thumbnail_path"] not in thumbs:
                    thumbs.append(collect_res["backup_thumbnail_path"])
        save_res = _persist_session(folder, session)

    if "minimax" in mode:
        from ...minimax.latent_manager import SceneLatentManager
        SceneLatentManager.clear_dirty(folder, scene_number)
        successor_scene = scene_number + 1
        if SceneLatentManager.latent_exists(folder, successor_scene):
            SceneLatentManager.mark_dirty(folder, successor_scene, reason=f"Scene {scene_number:03d} re-rendered")

    if manager and job:
        manager.update_progress(job.id, 100.0, "completed", scene_id=scene_id, message="Scene video rendered successfully.")

    return {
        "scene_id": scene_id,
        "video_path": final_video_path,
        "thumbnail_path": final_thumbnail_path,
        "backup_path": collect_res.get("backup_path", ""),
        "history_index": target_seg["video_history_index"],
        "history_count": len(target_seg["video_history"]),
        "revision": save_res.get("revision"),
    }


# ==============================================================================
# Job Handlers (Section 5.1, Section 6.9)
# ==============================================================================

async def run_scene_video_render_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute single-scene video render background job."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    scene_id = job.params.get("scene_id")
    if not scene_id:
        raise ValidationError("scene_id is required in params.")

    return await render_scene_video_async(job.project_id, scene_id, job.params, job=job, manager=manager)


# ==============================================================================
# Raw takes (the untrimmed renders a scene can be re-trimmed from)
# ==============================================================================

_TAKE_FPS = 24


def _same_file(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _inside(folder: str, path: str) -> bool:
    try:
        return os.path.normcase(os.path.commonpath([os.path.realpath(folder), os.path.realpath(path)])) == os.path.normcase(os.path.realpath(folder))
    except ValueError:
        return False


def _take_entry(path: str, exists: bool, recorded: bool, active_raw: str) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": path, "name": os.path.basename(path), "exists": exists, "recorded": recorded,
                             "is_latest_render": bool(active_raw) and _same_file(path, active_raw)}
    if exists:
        stat = os.stat(path)
        entry.update(size_bytes=stat.st_size, modified=stat.st_mtime)
        try:
            duration = float(minimax_inputs._probe_media_duration_seconds(path))
            entry.update(duration_seconds=round(duration, 3), frame_count=int(round(duration * _TAKE_FPS)))
        except Exception:
            pass
    return entry


def list_scene_takes(project_id: str, scene_id: str) -> Dict[str, Any]:
    """The scene's raw (untrimmed) renders, newest first: the ones the API recorded plus MiniMax renders found in the scene's output folder.

    A take whose file is gone (the Video Builder deletes a scene's scratch folder after each of its own renders) is
    listed with ``exists: false`` so the caller can see why it cannot be used. A take in a sibling scratch folder of
    the same project name is listed with ``other_folder: true``.
    """
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)
    seg = segments[idx]
    scene_number = idx + 1
    scratch, _ = minimax_inputs._minimax_h3_output_location(folder, scene_number, create=False)
    recorded = [str(x) for x in seg.get("raw_video_history") or [] if str(x).strip()]
    active_raw = str(seg.get("raw_video_path") or "")
    found: Dict[str, Dict[str, Any]] = {}
    for path in recorded:
        found[os.path.normcase(os.path.realpath(path))] = _take_entry(path, os.path.isfile(path), True, active_raw)
    # The scratch folder name ends with a hash of the project folder text, so a project rendered under a different
    # spelling of its path (the Video Builder's, a renamed or copied project) has its takes in a sibling folder with
    # the same project name. Those are listed too, marked ``other_folder`` and never picked by ``latest``.
    project_dir = os.path.dirname(scratch)
    base, prefix = os.path.dirname(project_dir), os.path.basename(project_dir).rsplit("_", 1)[0]
    candidates = [(scratch, False)]
    if os.path.isdir(base):
        for sibling in sorted(os.listdir(base)):
            sibling_dir = os.path.join(base, sibling)
            if sibling != os.path.basename(project_dir) and sibling.rsplit("_", 1)[0] == prefix and os.path.isdir(sibling_dir):
                candidates.append((os.path.join(sibling_dir, os.path.basename(scratch)), True))
    for folder_path, other in candidates:
        if not os.path.isdir(folder_path):
            continue
        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            lowered = name.lower()
            if lowered.endswith("-audio.mp4") and "stage1" not in lowered and os.path.isfile(path):
                key = os.path.normcase(os.path.realpath(path))
                if key not in found:
                    entry = _take_entry(path, True, False, active_raw)
                    entry["other_folder"] = other
                    found[key] = entry
    takes = sorted(found.values(), key=lambda t: t.get("modified") or 0.0, reverse=True)
    for index, take in enumerate(takes):
        take["index"] = index
    return {"scene_id": seg.get("id"), "scene_number": scene_number, "scratch_folder": scratch, "takes": takes,
            "usable": sum(1 for t in takes if t["exists"]),
            "note": "" if any(t["exists"] for t in takes) else
            "No raw take is on disk. Render the scene through the API again; the Video Builder deletes its scratch renders after each render."}


def resolve_scene_take(project_id: str, scene_id: str, take: Any) -> str:
    """The file path of a raw take named by ``latest`` or its index in ``list_scene_takes`` (newest first)."""
    listing = list_scene_takes(project_id, scene_id)
    usable = [t for t in listing["takes"] if t["exists"]]
    if not usable:
        raise ValidationError(f"Scene '{scene_id}' has no raw take on disk to trim. {listing['note']}")
    if str(take).strip().lower() in ("latest", "newest", ""):
        own = [t for t in usable if not t.get("other_folder")]
        if not own:
            raise ValidationError(
                f"Scene '{scene_id}' has raw takes only in another scratch folder with the same project name (rendered under a "
                "different path spelling). Name one by its index from GET .../video/takes if it is yours.")
        return own[0]["path"]
    try:
        wanted = int(take)
    except (TypeError, ValueError):
        raise ValidationError("take must be 'latest' or the index of a take from GET .../video/takes.")
    match = next((t for t in listing["takes"] if t["index"] == wanted), None)
    if match is None:
        raise ValidationError(f"Scene '{scene_id}' has no take {wanted}; it has {len(listing['takes'])}.")
    if not match["exists"]:
        raise ValidationError(f"Take {wanted} of scene '{scene_id}' was recorded but its file is gone: {match['path']}")
    return match["path"]


async def run_video_trim_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute video trimming background job (Section 6.9)."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    scene_id = job.params.get("scene_id")
    if not scene_id:
        raise ValidationError("scene_id is required in params.")

    folder, session = _get_active_session_and_folder(job.project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, job.project_id)

    seg = segments[idx]
    scene_number = idx + 1
    take = job.params.get("take")
    from_take = take is not None and not job.params.get("source_path")
    if from_take:
        source_path = await asyncio.to_thread(resolve_scene_take, job.project_id, scene_id, take)
    else:
        source_path = job.params.get("source_path") or seg.get("video_path") or seg.get("rendered_video_path")
    if not source_path or not os.path.isfile(source_path):
        raise ValidationError(f"No source video found to trim for scene '{scene_id}'.")

    start = float(job.params.get("start", 0.0))
    duration = float(job.params.get("duration", 0.0) or 0.0)
    frames = int(job.params.get("frames", 0))
    if job.params.get("to_end") and not duration:
        # End on the source's last frame: the rest of the take after ``start``.
        total = float(await asyncio.to_thread(minimax_inputs._probe_media_duration_seconds, source_path))
        duration = max(0.05, total - start - 0.01)
        frames = frames or int(duration * _TAKE_FPS)
        duration = frames / _TAKE_FPS
    duration = duration or 4.0

    manager.update_progress(job.id, 20.0, "trimming", scene_id=scene_id, message="Executing video trim...")

    res = await asyncio.to_thread(
        video_files._trim_scene_video,
        {
            "project_folder": folder,
            "scene_number": scene_number,
            "source_path": source_path,
            "start": start,
            "duration": duration,
            "frames": frames,
            "label": str(job.params.get("label") or ("retrim" if from_take else "trim")),
            "mark_as_audio_video": bool(job.params.get("mark_as_audio_video", from_take)),
        },
    )

    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(job.project_id)
        target_seg = session["segments"][idx]
        apply_scene_video(target_seg, res["video_path"], res["thumbnail_path"] if res.get("thumbnail_path") else "")
        save_res = _persist_session(folder, session)

    manager.update_progress(job.id, 100.0, "completed", scene_id=scene_id, message="Video trimmed.")
    return {
        "scene_id": scene_id,
        "video_path": res["video_path"],
        "thumbnail_path": res.get("thumbnail_path", ""),
        "source_path": source_path,
        "start": start,
        "duration": duration,
        "frames": frames,
        "revision": save_res.get("revision"),
    }


async def run_video_match_color_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute opening color match background job (Section 6.9)."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    scene_id = job.params.get("scene_id")
    if not scene_id:
        raise ValidationError("scene_id is required in params.")

    folder, session = _get_active_session_and_folder(job.project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, job.project_id)

    seg = segments[idx]
    video_path = job.params.get("video_path") or seg.get("video_path") or seg.get("rendered_video_path")
    ref_path = job.params.get("reference_video_path")
    if not ref_path and idx > 0:
        prev_seg = segments[idx - 1]
        ref_path = prev_seg.get("video_path") or prev_seg.get("rendered_video_path")

    if not video_path or not os.path.isfile(video_path):
        raise ValidationError(f"Scene video not found for scene '{scene_id}'.")
    if not ref_path or not os.path.isfile(ref_path):
        raise ValidationError(f"Reference video not found to match color for scene '{scene_id}'.")

    manager.update_progress(job.id, 20.0, "matching_color", scene_id=scene_id, message="Matching opening color...")

    res = await asyncio.to_thread(
        video_files._apply_scene_start_color_match,
        {
            "project_folder": folder,
            "video_path": video_path,
            "reference_video_path": ref_path,
            "fade_seconds": float(job.params.get("fade_seconds", 1.0)),
            "strength": float(job.params.get("strength", 0.85)),
        },
    )

    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(job.project_id)
        target_seg = session["segments"][idx]
        apply_scene_video(target_seg, res["video_path"], res["thumbnail_path"] if res.get("thumbnail_path") else "")
        save_res = _persist_session(folder, session)

    manager.update_progress(job.id, 100.0, "completed", scene_id=scene_id, message="Color match applied.")
    return {
        "scene_id": scene_id,
        "video_path": res["video_path"],
        "thumbnail_path": res.get("thumbnail_path", ""),
        "revision": save_res.get("revision"),
    }


# ==============================================================================
# Synchronous Video Lifecycle Operations (Section 6.9, Section 24.2)
# ==============================================================================

def recover_scene_video(
    project_id: str,
    scene_id: str,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Find and link a finished but uncollected scene video (Section 6.9)."""
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    scene_number = idx + 1
    target_source = source_path
    if not target_source:
        mode = session.get("video_mode", "i2v")
        found = video_files._find_scene_video_output({
            "project_folder": folder,
            "video_mode": mode,
            "scene_number": scene_number,
            "prompt_number_one_based": scene_number,
        })
        target_source = found.get("video_path")

    if not target_source or not os.path.isfile(target_source):
        raise ValidationError(f"No recoverable scene video found for scene '{scene_id}'.")

    res = builder_media._restore_scene_video({
        "project_folder": folder,
        "scene_number": scene_number,
        "source_path": target_source,
    })

    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        target_seg = session["segments"][idx]
        apply_scene_video(target_seg, res["video_path"], res["thumbnail_path"] if res.get("thumbnail_path") else "")
        save_res = _persist_session(folder, session)

    return {
        "scene_id": scene_id,
        "video_path": res["video_path"],
        "thumbnail_path": res.get("thumbnail_path", ""),
        "revision": save_res.get("revision"),
    }


def select_scene_video(
    project_id: str,
    scene_id: str,
    source_path: str,
) -> Dict[str, Any]:
    """Choose active take from video_history or backups (Section 6.9)."""
    if not source_path or not os.path.isfile(source_path):
        raise ValidationError(f"Invalid video source path: {source_path}")

    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    scene_number = idx + 1
    res = builder_media._restore_scene_video({
        "project_folder": folder,
        "scene_number": scene_number,
        "source_path": source_path,
    })

    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        target_seg = session["segments"][idx]
        apply_scene_video(target_seg, res["video_path"], res["thumbnail_path"] if res.get("thumbnail_path") else "")
        save_res = _persist_session(folder, session)

    return {
        "scene_id": scene_id,
        "video_path": res["video_path"],
        "thumbnail_path": res.get("thumbnail_path", ""),
        "revision": save_res.get("revision"),
    }


def delete_scene_video(project_id: str, scene_id: str) -> Dict[str, Any]:
    """Clear active scene video render while keeping filesystem archive (Section 6.9)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.get("segments", [])
        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        seg = segments[idx]
        seg["video_path"] = ""
        seg["rendered_video_path"] = ""
        seg["thumbnail_path"] = ""
        seg["video_thumbnail_path"] = ""
        seg["video_status"] = ""
        if seg.get("preview_mode") == "video":
            seg["preview_mode"] = "image"
        save_res = _persist_session(folder, session)
        return {
            "scene_id": scene_id,
            "cleared": True,
            "revision": save_res.get("revision"),
        }


def scan_project_scene_videos(project_id: str) -> Dict[str, Any]:
    """Scan and index scene video versions and backups across project (Section 6.9)."""
    folder = resolve_project_folder(project_id)
    return builder_media._scan_builder_scene_videos(folder)


async def run_batch_video_render_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute batch scene video rendering (Section 6.9, renderAllScenes)."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    folder, session = _get_active_session_and_folder(job.project_id)
    segments = session.get("segments", [])

    scope = str(job.params.get("scope") or "all").strip().lower()
    force = bool(job.params.get("force", False))
    skip_final_stitch = bool(job.params.get("skip_final_stitch", False))
    specified_ids = set(job.params.get("scene_ids") or [])
    range_bounds = job.params.get("range")

    target_segments = []
    for idx, seg in enumerate(segments):
        sid = seg.get("id") or str(idx + 1)
        scene_num = idx + 1

        if scope == "selected" and specified_ids and sid not in specified_ids:
            continue
        if scope == "range" and range_bounds and len(range_bounds) == 2:
            if not (range_bounds[0] <= scene_num <= range_bounds[1]):
                continue

        has_video = bool(seg.get("video_path") or seg.get("rendered_video_path"))
        if scope == "missing" and has_video:
            continue
        if not force and job.params.get("run_mode") == "missing" and has_video:
            continue

        target_segments.append(seg)

    total = len(target_segments)
    if total == 0:
        return {"processed": 0, "message": "No scenes matched batch video render criteria."}

    manager.update_progress(job.id, 0.0, "batch_videos", stage_index=0, stage_count=total, message=f"Starting batch render for {total} scene(s)...")

    results = []
    for idx, seg in enumerate(target_segments):
        if job.cancel_requested:
            raise JobCancelledError(job.id)

        sid = seg.get("id") or str(idx + 1)
        pct = round((idx / total) * 90.0, 1)
        manager.update_progress(
            job.id,
            pct,
            "batch_videos",
            stage_index=idx + 1,
            stage_count=total,
            scene_index=idx + 1,
            scene_count=total,
            scene_id=sid,
            message=f"Rendering video for scene {idx + 1} of {total} ({sid})...",
        )

        try:
            res = await render_scene_video_async(job.project_id, sid, job.params, job=job, manager=manager)
            results.append({"scene_id": sid, "status": "success", "video_path": res.get("video_path")})
        except Exception as e:
            logger.warning(f"Video render failed for scene {sid}: {e}")
            results.append({"scene_id": sid, "status": "error", "error": str(e)})

    # Final stitch if not skipped and not single selected
    final_video_path = ""
    if not skip_final_stitch and scope != "selected":
        manager.update_progress(job.id, 92.0, "stitching", message="Stitching final project video...")
        try:
            _, fresh_session = _get_active_session_and_folder(job.project_id)
            # Same payload as the stitch route and the Builder's Render All stitch (frame sync, scene audio).
            stitch_payload, _summary = build_stitch_payload(folder, fresh_session, {}, strict=False)
            stitch_res = await asyncio.to_thread(video_files._stitch_scene_videos, stitch_payload)
            final_video_path = stitch_res.get("final_video_path", "")
        except Exception as s_err:
            logger.warning(f"Final video stitch after batch render failed: {s_err}")

    manager.update_progress(job.id, 100.0, "completed", stage_index=total, stage_count=total, message="Batch video render finished.")
    return {
        "processed": len(results),
        "total_targets": total,
        "results": results,
        "final_video_path": final_video_path,
    }


# The Video Builder's MiniMax H3 stitch syncs every clip to the timeline at this rate
# (web/music_video_builder/video_render.mjs stitchRenderedScenes: timeline_fps 24).
STITCH_TIMELINE_FPS = 24
STITCH_AUDIO_MODES = ("auto", "embedded", "project")
STITCH_BODY_KEYS = ("scene_ids", "output_prefix", "audio", "audio_path", "overlays")
_LTX_INGREDIENTS_CANVAS = (768, 448)  # DEFAULT_LTX_INGREDIENTS_WIDTH / _HEIGHT in web/music_video_builder/constants.mjs


def _round_half_up(value: float) -> int:
    """``Math.round`` for the non-negative times used here (Python's ``round`` rounds half to even)."""
    return int(math.floor(float(value) + 0.5))


def _segment_time(segment: Dict[str, Any], key: str) -> float:
    try:
        return float(segment.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _select_stitch_segments(segments: List[Dict[str, Any]], scene_ids: Any) -> List[int]:
    """Indexes into ``segments`` for ``scene_ids`` (scene numbers, 1 = first, or scene ids), in timeline order.

    No ``scene_ids`` selects every scene. A name that matches no scene is an error that lists it.
    """
    if scene_ids is None or scene_ids == []:
        indexes = list(range(len(segments)))
    else:
        if not isinstance(scene_ids, list):
            raise ValidationError("scene_ids must be a list of scene numbers (1 = first scene) or scene ids.")
        by_key: Dict[str, int] = {}
        for index, segment in enumerate(segments):
            by_key.setdefault(str(index + 1), index)
        for index, segment in enumerate(segments):
            sid = str(segment.get("id") or "").strip()
            if sid:
                by_key[sid] = index
        chosen, unknown = set(), []
        for raw in scene_ids:
            if isinstance(raw, bool) or not isinstance(raw, (str, int)):
                raise ValidationError("scene_ids must be a list of scene numbers (1 = first scene) or scene ids.")
            key = str(raw).strip()
            if key in by_key:
                chosen.add(by_key[key])
            else:
                unknown.append(key)
        if unknown:
            raise ValidationError(
                f"Unknown scene(s) for stitch: {', '.join(unknown)}. Use a scene number (1 = first scene, "
                f"{len(segments)} scenes) or a scene id.",
                details={"unknown_scene_ids": unknown},
            )
        indexes = list(chosen)
    # Like the Builder's preview stitch, play the scenes in timeline order whatever order they were named in.
    return sorted(indexes, key=lambda i: (_segment_time(segments[i], "start"), i))


def _stitch_canvas(session: Dict[str, Any], minimax_project: bool) -> tuple:
    """Output size the Builder passes: 0x0 (keep the clips' size) for MiniMax, the LTX render size otherwise."""
    if minimax_project:
        return 0, 0
    settings = session.get("i2v_video_settings") if isinstance(session.get("i2v_video_settings"), dict) else {}
    try:
        if session_video_mode(session) == "ingredients":
            return (int(settings.get("ingredients_width") or _LTX_INGREDIENTS_CANVAS[0]),
                    int(settings.get("ingredients_height") or _LTX_INGREDIENTS_CANVAS[1]))
        return int(settings.get("width") or 1920), int(settings.get("height") or 1080)
    except (TypeError, ValueError):
        return 0, 0


def build_stitch_payload(
    folder: str,
    session: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> tuple:
    """Build the ``_stitch_scene_videos`` payload the Video Builder would send, plus a summary for the job result.

    Mirrors ``stitchRenderedScenes`` in web/music_video_builder/video_render.mjs:

    * MiniMax H3 project: one timing item per scene and ``timeline_fps`` 24. The stitcher then cuts or pads
      every clip to ``round(end * 24) - round(start * 24)`` frames and muxes without ``-shortest``, so the
      output has exactly the timeline's frames. Without timing items it muxes with ``-shortest`` and a
      scene audio track a few ms shorter than the video cuts the last frame (the 870-of-871 bug).
    * A preview of some scenes (``scene_ids``) shifts the timing by a whole number of frames, and cuts the
      project song to the selected scenes (``audio_start`` / ``audio_duration``).
    * ``audio``: ``auto`` (default) uses the scenes' own audio when every selected MiniMax scene renders
      with built-in audio (or the LTX mode is ID-LoRA), the project song otherwise. ``embedded`` and
      ``project`` force one or the other.

    ``strict`` (the stitch route) makes a selected scene without a video file an error. The stitch at the end
    of a batch render or pipeline passes ``strict=False`` and leaves such scenes out, as it did before.
    """
    p = dict(params or {})
    segments = [s for s in (session.get("segments") or []) if isinstance(s, dict)]

    audio = str(p.get("audio") or "auto").strip().lower()
    if audio not in STITCH_AUDIO_MODES:
        raise ValidationError(
            f"audio must be one of {', '.join(STITCH_AUDIO_MODES)} (got {p.get('audio')!r}).",
            details={"allowed": list(STITCH_AUDIO_MODES)},
        )
    overlays = p.get("overlays") or []
    if not isinstance(overlays, list):
        raise ValidationError("overlays must be a list of {path, start, end, source_start} items.")
    output_prefix = p.get("output_prefix") or "FINAL_VIDEO"
    if not isinstance(output_prefix, str):
        raise ValidationError("output_prefix must be a string.")

    explicit_selection = bool(p.get("scene_ids"))
    indexes = _select_stitch_segments(segments, p.get("scene_ids"))

    selected: List[tuple] = []
    missing: List[str] = []
    for index in indexes:
        segment = segments[index]
        path = str(segment.get("video_path") or segment.get("rendered_video_path") or "").strip()
        if not path or not os.path.isfile(path):
            missing.append(str(segment.get("id") or index + 1))
            continue
        selected.append((index, segment, path))
    if missing and strict:
        raise ValidationError(
            f"Scene(s) without a rendered video: {', '.join(missing)}. Render them or leave them out of scene_ids.",
            details={"scenes_without_video": missing},
        )
    if not selected:
        raise ValidationError("No scene video files available to stitch.")

    minimax_project = str(session.get("video_engine") or "").strip().lower() == "minimax_h3"
    if audio == "auto":
        built_in = minimax_project and all(
            str(minimax_h3_settings_for_scene(session, segment).get("audio_mode") or "") == "built_in_audio"
            for _index, segment, _path in selected
        )
        embedded = built_in or (not minimax_project and session_video_mode(session) == "id_lora")
    else:
        embedded = audio == "embedded"

    audio_path = ""
    if not embedded:
        audio_path = str(p.get("audio_path") or session_audio_path(session) or "").strip()
        if not audio_path or not os.path.isfile(audio_path):
            raise ValidationError(
                "The project has no audio file to stitch with. Attach the song, pass audio_path, or use audio: embedded.",
                details={"audio_path": audio_path},
            )

    start_time = min(_segment_time(segment, "start") for _i, segment, _p in selected)
    end_time = max(_segment_time(segment, "end") for _i, segment, _p in selected)
    audio_start = 0.0
    audio_duration = 0.0
    timeline_offset = 0.0
    if explicit_selection:
        positions = [index for index, _segment, _path in selected]
        contiguous = all(b == a + 1 for a, b in zip(positions, positions[1:]))
        if not embedded and not contiguous:
            raise ValidationError(
                "A stitch with the project song needs contiguous scenes, so one window of the song fits them. "
                "Pick one continuous scene range, or use audio: embedded.",
            )
        timeline_offset = start_time
        audio_start = start_time
        audio_duration = max(0.1, end_time - start_time)

    timing_items: List[Dict[str, float]] = []
    expected_frames = None
    if minimax_project:
        # A whole number of frames, so each clip rounds to the same frame count as on the full timeline.
        aligned_offset = _round_half_up(timeline_offset * STITCH_TIMELINE_FPS) / STITCH_TIMELINE_FPS
        expected_frames = 0
        for _index, segment, _path in selected:
            start = max(0.0, _segment_time(segment, "start") - aligned_offset)
            end = max(0.0, _segment_time(segment, "end") - aligned_offset)
            timing_items.append({"start": start, "end": end})
            # Same rounding as the stitcher (runner/video_files.py _stitch_scene_videos).
            expected_frames += max(1, int(max(start, end) * STITCH_TIMELINE_FPS + 0.5) - int(start * STITCH_TIMELINE_FPS + 0.5))

    width, height = _stitch_canvas(session, minimax_project)
    payload = {
        "project_folder": folder,
        "scene_paths": [path for _index, _segment, path in selected],
        "audio_path": audio_path,
        "scene_audio_paths": [],
        "scene_audio_items": [],
        "scene_timing_items": timing_items,
        "timeline_fps": STITCH_TIMELINE_FPS if minimax_project else 0,
        "use_embedded_scene_audio": embedded,
        "overlay_items": overlays,
        "width": width,
        "height": height,
        "audio_start": audio_start,
        "audio_duration": audio_duration,
        "output_prefix": output_prefix,
    }
    summary = {
        "scene_ids": [str(segment.get("id") or index + 1) for index, segment, _path in selected],
        "scene_numbers": [index + 1 for index, _segment, _path in selected],
        "skipped_scene_ids": missing,
        "audio": "embedded" if embedded else "project",
        "audio_start": audio_start,
        "audio_duration": audio_duration,
        "expected_frame_count": expected_frames,
    }
    return payload, summary


def validate_stitch_request(project_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Check a stitch request against the project before its job starts. Raises ``ValidationError``."""
    folder, session = _get_active_session_and_folder(project_id)
    _payload, summary = build_stitch_payload(folder, session, params, strict=True)
    return summary


async def run_video_stitch_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute final video stitch job (Section 6.12, stitch_scene_videos), frame-accurate like the Builder."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    folder, session = _get_active_session_and_folder(job.project_id)
    stitch_payload, summary = build_stitch_payload(folder, session, job.params, strict=True)

    manager.update_progress(
        job.id, 20.0, "stitching", message=f"Stitching {len(stitch_payload['scene_paths'])} scene videos..."
    )
    res = await asyncio.to_thread(video_files._stitch_scene_videos, stitch_payload)

    manager.update_progress(job.id, 100.0, "completed", message="Stitch completed.")
    result = dict(res or {})
    result.update(summary)
    return result


async def run_image_slideshow_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute image slideshow render job (Section 6.12, render_image_slideshow)."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    folder, session = _get_active_session_and_folder(job.project_id)

    manager.update_progress(job.id, 20.0, "rendering_slideshow", message="Rendering image slideshow...")

    p = dict(job.params or {})
    p["project_folder"] = folder
    if not p.get("audio_path") and session_audio_path(session):
        p["audio_path"] = session_audio_path(session)

    res = await asyncio.to_thread(video_files._render_image_slideshow, p)

    manager.update_progress(job.id, 100.0, "completed", message="Slideshow rendered.")
    return res


def list_project_final_videos(project_id: str) -> List[Dict[str, Any]]:
    """List final assembled video files in project (Section 6.12)."""
    folder = resolve_project_folder(project_id)
    finals = []

    pattern = re.compile(r"^FINAL_VIDEO.*\.mp4$", re.IGNORECASE)
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and (pattern.match(name) or name.lower().endswith(".mp4")):
            if "temp" in name.lower() or name.startswith("_"):
                continue
            stat = os.stat(path)
            finals.append({
                "filename": name,
                "path": path,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            })

    finals_dir = os.path.join(folder, "final_videos")
    if os.path.isdir(finals_dir):
        for name in os.listdir(finals_dir):
            path = os.path.join(finals_dir, name)
            if os.path.isfile(path) and name.lower().endswith(".mp4"):
                stat = os.stat(path)
                finals.append({
                    "filename": name,
                    "path": path,
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                })

    finals.sort(key=lambda item: item["modified_at"], reverse=True)
    return finals


def register_video_orchestrator_handlers(manager: Optional[JobManager] = None) -> None:
    """Register video generator job handlers with JobManager."""
    if manager is None:
        manager = get_job_manager()

    manager.register_handler("video.render", run_scene_video_render_job)
    manager.register_handler("videos.render_batch", run_batch_video_render_job)
    manager.register_handler("video.render_batch", run_batch_video_render_job)
    manager.register_handler("video.trim", run_video_trim_job)
    manager.register_handler("video.match_start_color", run_video_match_color_job)
    manager.register_handler("video.stitch", run_video_stitch_job)
    manager.register_handler("video.slideshow", run_image_slideshow_job)
