"""Server-side image generation and lifecycle orchestrator (Section 6.8, Section 24.1, C4)."""

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from ...builder import media as builder_media
from ...runner.image_workflows import (
    _build_ernie_image_api_prompt,
    _build_flux_klein_api_prompt,
    _build_krea2_2pass_api_prompt,
    _build_krea2_api_prompt,
    _build_nb_image_api_prompt,
    _build_z_upscale_enhance_prompt,
    _build_zimage_api_prompt,
)
from ..errors import (
    ComfyExecutionError,
    JobCancelledError,
    ProjectNotFoundError,
    SceneNotFoundError,
    ValidationError,
)
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session
from ..paths import resolve_project_folder
from ..schemas import extract_effective_settings
from .comfy_client import (
    extract_images_from_history,
    get_comfy_client,
)

logger = logging.getLogger("vrgdg.agent_api.image_orchestrator")


def build_image_graph_for_mode(mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compile ComfyUI prompt graph for the requested image mode (Section 6.8)."""
    m = str(mode or "zimage").strip().lower()
    if m in ("flux_klein", "flux"):
        return _build_flux_klein_api_prompt(payload)
    if m in ("krea2_2pass",):
        return _build_krea2_2pass_api_prompt(payload)
    if m in ("krea2",):
        return _build_krea2_api_prompt(payload)
    if m in ("ernie_image", "ernie"):
        return _build_ernie_image_api_prompt(payload)
    if m in ("nano_banana", "nb"):
        return _build_nb_image_api_prompt(payload)
    if m in ("z_upscale_enhance", "enhance"):
        return _build_z_upscale_enhance_prompt(payload)
    # Default to Z-Image
    return _build_zimage_api_prompt(payload)


async def generate_scene_image_async(
    project_id: str,
    scene_id: str,
    params: Optional[Dict[str, Any]] = None,
    job: Optional[Job] = None,
    manager: Optional[JobManager] = None,
) -> Dict[str, Any]:
    """Generate image for a scene, archive preview to image_history, and update session (Section 24.1)."""
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    seg = segments[idx]
    scene_number = idx + 1
    p = dict(params or {})

    # Determine prompt text
    prompt = str(p.get("prompt") or seg.get("t2i_prompt") or "").strip()
    if not prompt:
        raise ValidationError(f"Scene '{scene_id}' has no prompt specified and t2i_prompt is empty.")

    mode = str(p.get("mode") or "zimage").strip().lower()

    if manager and job:
        manager.update_progress(job.id, 10.0, "compiling_graph", scene_id=scene_id, message=f"Compiling {mode} graph...")

    # Merge effective settings with parameter overrides
    effective_settings = extract_effective_settings(session)
    payload = dict(effective_settings.get(mode, {}))
    payload.update(p)
    payload["prompt"] = prompt
    payload["project_folder"] = folder

    # Img2Img continuity from previous scene final frame
    if p.get("from_previous_final_frame") and idx > 0:
        prev_seg = segments[idx - 1]
        source_img = prev_seg.get("approved_image_path") or (prev_seg.get("image_history") or [""])[-1]
        if source_img and os.path.isfile(source_img):
            payload["use_image_to_image"] = True
            payload["image_to_image_path"] = source_img

    graph_res = await asyncio.to_thread(build_image_graph_for_mode, mode, payload)
    prompt_graph = graph_res["prompt"]

    # Queue with ComfyUI
    client = get_comfy_client()
    queue_res = await asyncio.to_thread(client.queue_prompt, prompt_graph)
    prompt_id = queue_res["prompt_id"]

    if manager and job:
        manager.set_current_comfy_prompt(job.id, prompt_id, project_folder=folder)
        manager.update_progress(job.id, 30.0, "generating", scene_id=scene_id, message="Waiting for ComfyUI generation...")

    # Wait for completion
    timeout_sec = float(p.get("timeout_seconds", 600.0))
    check_cancel = (lambda: job.cancel_requested) if job else None
    history = await client.wait_for_prompt(prompt_id, timeout_seconds=timeout_sec, check_cancel=check_cancel)

    images = extract_images_from_history(history, prompt_id)
    if not images:
        raise ComfyExecutionError(f"Image generation succeeded but produced no image output for prompt '{prompt_id}'.", prompt_id=prompt_id)

    image_info = images[-1]  # {"filename": ..., "subfolder": ..., "type": ...}

    if manager and job:
        manager.update_progress(job.id, 80.0, "archiving", scene_id=scene_id, message="Archiving scene preview...")

    # Archive into scene_image_previews/scene_NNNN/preview_<timestamp>.<ext>
    archive_payload = {
        "project_folder": folder,
        "scene_number": scene_number,
        "image": image_info,
    }
    archive_res = await asyncio.to_thread(builder_media._archive_scene_image, archive_payload)
    saved_path = archive_res["saved_path"]

    # Update scene in session following Section 24.1 lifecycle
    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        target_seg = session["segments"][idx]
        target_seg["image"] = image_info
        history_list = target_seg.setdefault("image_history", [])
        if saved_path not in history_list:
            history_list.append(saved_path)
        target_seg["image_history_index"] = len(history_list) - 1
        # Clear custom and approved per Section 24.1
        target_seg["custom_image_path"] = ""
        target_seg["custom_image_data"] = ""
        target_seg["custom_image_name"] = ""
        target_seg["approved_image_path"] = ""
        target_seg["preview_mode"] = "image"
        save_res = _persist_session(folder, session)

    if manager and job:
        manager.update_progress(job.id, 100.0, "completed", scene_id=scene_id, message="Image generated and archived.")

    return {
        "scene_id": scene_id,
        "saved_path": saved_path,
        "image": image_info,
        "history_index": target_seg["image_history_index"],
        "history_count": len(history_list),
        "revision": save_res.get("revision"),
    }


# ==============================================================================
# Job Handlers
# ==============================================================================

async def run_scene_image_generation_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute single-scene image generation as a background job."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    scene_id = job.params.get("scene_id")
    if not scene_id:
        raise ValidationError("scene_id is required in params.")

    return await generate_scene_image_async(job.project_id, scene_id, job.params, job=job, manager=manager)


async def run_batch_image_generation_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute batch image generation across scenes (replaces zImageAllScenes)."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    folder, session = _get_active_session_and_folder(job.project_id)
    segments = session.get("segments", [])

    scope = str(job.params.get("scope") or "all").strip().lower()
    run_mode = str(job.params.get("run_mode") or "all").strip().lower()
    mode = str(job.params.get("mode") or "zimage").strip().lower()
    specified_ids = set(job.params.get("scene_ids") or [])

    target_segments = []
    for idx, seg in enumerate(segments):
        sid = seg.get("id") or str(idx + 1)
        if specified_ids and sid not in specified_ids:
            continue

        has_image = bool(seg.get("image_history") or seg.get("approved_image_path") or seg.get("custom_image_path"))
        if run_mode == "resume_missing" and has_image:
            continue

        target_segments.append(seg)

    total = len(target_segments)
    if total == 0:
        return {"processed": 0, "message": "No scenes matched batch image criteria."}

    manager.update_progress(job.id, 0.0, "batch_images", stage_index=0, stage_count=total, message=f"Starting image batch for {total} scene(s)...")

    results = []
    for idx, seg in enumerate(target_segments):
        if job.cancel_requested:
            raise JobCancelledError(job.id)

        sid = seg.get("id") or str(idx + 1)
        pct = round((idx / total) * 100.0, 1)
        manager.update_progress(
            job.id,
            pct,
            "batch_images",
            stage_index=idx + 1,
            stage_count=total,
            scene_index=idx + 1,
            scene_count=total,
            scene_id=sid,
            message=f"Generating {mode} image for scene {idx + 1} of {total} ({sid})...",
        )

        try:
            res = await generate_scene_image_async(job.project_id, sid, job.params, job=job, manager=manager)
            results.append({"scene_id": sid, "status": "success", "saved_path": res.get("saved_path")})
        except Exception as e:
            logger.warning(f"Image generation failed for scene {sid}: {e}")
            results.append({"scene_id": sid, "status": "error", "error": str(e)})

    manager.update_progress(job.id, 100.0, "completed", stage_index=total, stage_count=total, message="Batch image generation finished.")
    return {
        "processed": len(results),
        "total_targets": total,
        "results": results,
    }


# ==============================================================================
# Synchronous Scene Image Lifecycle Operations (Section 6.8, Section 24.1)
# ==============================================================================

def approve_scene_image(project_id: str, scene_id: str, image_path: Optional[str] = None) -> Dict[str, Any]:
    """Approve a preview image from history as the definitive scene image."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.get("segments", [])
        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        seg = segments[idx]
        history = seg.get("image_history") or []
        hist_idx = seg.get("image_history_index", -1)

        target_path = image_path
        if not target_path:
            if 0 <= hist_idx < len(history):
                target_path = history[hist_idx]
            elif history:
                target_path = history[-1]
            elif seg.get("custom_image_path"):
                target_path = seg["custom_image_path"]

        if not target_path or not os.path.isfile(target_path):
            raise ValidationError(f"No valid image path found to approve for scene '{scene_id}'.")

        seg["approved_image_path"] = os.path.abspath(target_path)
        seg["preview_mode"] = "image"
        save_res = _persist_session(folder, session)
        return {
            "scene_id": scene_id,
            "approved_image_path": seg["approved_image_path"],
            "revision": save_res.get("revision"),
        }


def revert_scene_image(project_id: str, scene_id: str, delta: int = -1, index: Optional[int] = None) -> Dict[str, Any]:
    """Move image_history_index to navigate earlier takes."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.get("segments", [])
        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        seg = segments[idx]
        history = seg.get("image_history") or []
        if not history:
            raise ValidationError(f"Scene '{scene_id}' has no image history to revert.")

        curr_idx = int(seg.get("image_history_index", len(history) - 1))
        if index is not None:
            new_idx = max(0, min(len(history) - 1, int(index)))
        else:
            new_idx = max(0, min(len(history) - 1, curr_idx + delta))

        seg["image_history_index"] = new_idx
        seg["preview_mode"] = "image"
        save_res = _persist_session(folder, session)
        return {
            "scene_id": scene_id,
            "history_index": new_idx,
            "current_image": history[new_idx],
            "total_takes": len(history),
            "revision": save_res.get("revision"),
        }


def delete_scene_image(project_id: str, scene_id: str) -> Dict[str, Any]:
    """Archive and clear active scene image without destroying filesystem history."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.get("segments", [])
        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        seg = segments[idx]
        seg["approved_image_path"] = ""
        seg["custom_image_path"] = ""
        seg["image_assignment_cleared"] = True
        save_res = _persist_session(folder, session)
        return {
            "scene_id": scene_id,
            "cleared": True,
            "revision": save_res.get("revision"),
        }


def save_scene_image_custom(
    project_id: str,
    scene_id: str,
    image_data: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload or link a custom image file to a scene (save_scene_image)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.get("segments", [])
        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        scene_number = idx + 1
        res = builder_media._save_scene_image({
            "project_folder": folder,
            "scene_number": scene_number,
            "image_data": image_data,
            "source_path": source_path,
        })
        saved_path = res["saved_path"]

        seg = segments[idx]
        seg["custom_image_path"] = saved_path
        seg["preview_mode"] = "image"
        save_res = _persist_session(folder, session)
        return {
            "scene_id": scene_id,
            "custom_image_path": saved_path,
            "revision": save_res.get("revision"),
        }


def extract_frame_from_video_to_image(
    project_id: str,
    scene_id: str,
    source_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract final frame from preceding scene video and assign as scene start image."""
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    video_path = source_video_path
    if not video_path and idx > 0:
        prev_seg = segments[idx - 1]
        video_path = prev_seg.get("video_path") or prev_seg.get("rendered_video_path")
        if not video_path:
            # Check rendered_scene_videos/
            candidate = os.path.join(folder, "rendered_scene_videos", f"scene_{idx:04d}.mp4")
            if os.path.isfile(candidate):
                video_path = candidate

    if not video_path or not os.path.isfile(video_path):
        raise ValidationError(f"No source video found to extract frame for scene '{scene_id}'.")

    scene_number = idx + 1
    res = builder_media._extract_video_final_frame_as_scene_image({
        "project_folder": folder,
        "scene_number": scene_number,
        "source_path": video_path,
    })
    saved_path = res["saved_path"]

    with _BUILDER_SAVE_LOCK:
        _, session = _get_active_session_and_folder(project_id)
        seg = session["segments"][idx]
        seg["custom_image_path"] = saved_path
        history = seg.setdefault("image_history", [])
        if saved_path not in history:
            history.append(saved_path)
        seg["image_history_index"] = len(history) - 1
        seg["preview_mode"] = "image"
        save_res = _persist_session(folder, session)

    return {
        "scene_id": scene_id,
        "saved_path": saved_path,
        "source_video_path": video_path,
        "revision": save_res.get("revision"),
    }


def register_image_orchestrator_handlers(manager: Optional[JobManager] = None) -> None:
    """Register image generator job handlers with JobManager."""
    if manager is None:
        manager = get_job_manager()

    manager.register_handler("image.generate", run_scene_image_generation_job)
    manager.register_handler("images.generate_batch", run_batch_image_generation_job)
