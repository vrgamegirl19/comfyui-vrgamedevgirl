"""MiniMax H3 latent continuity orchestrator (Section 6.10, Invariant 4)."""

import asyncio
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from ...builder.project import _write_minimax_project_index
from ...minimax.latent_manager import SceneLatentManager
from ...runner import minimax_inputs, video_files
from ..errors import (
    JobCancelledError,
    LatentStaleError,
    PredecessorMissingError,
    ProjectNotFoundError,
    SceneNotFoundError,
    ValidationError,
)
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..mutations import _get_active_session_and_folder
from .video_orchestrator import render_scene_video_async

logger = logging.getLogger("vrgdg.agent_api.latent_orchestrator")


def get_project_latents_status(project_id: str) -> Dict[str, Any]:
    """Retrieve latent status for all scenes in a project (Section 6.10)."""
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    dirty_scenes = SceneLatentManager.list_dirty(folder)
    dirty_set = set(dirty_scenes)

    scenes_status: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        scene_num = idx + 1
        sid = seg.get("id") or str(scene_num)
        info = SceneLatentManager.get_latent_info(folder, scene_num)
        is_dirty = scene_num in dirty_set or bool(info.get("dirty"))

        scenes_status.append({
            "scene_id": sid,
            "scene_number": scene_num,
            "exists": bool(info.get("exists", False)),
            "dirty": is_dirty,
            "path": info.get("path", ""),
            "frame_count": int(info.get("frame_count", 0)),
            "token_count": int(info.get("token_count", 0)),
            "fps": float(info.get("fps", 24.0)),
            "size_bytes": int(info.get("size_bytes", 0)),
            "tail_padding_frames": info.get("tail_padding_frames"),
            "timestamp": info.get("timestamp"),
        })

    return {
        "project_id": project_id,
        "scenes": scenes_status,
        "dirty_scenes": dirty_scenes,
        "count": len(scenes_status),
        "dirty_count": len(dirty_scenes),
    }


def get_dirty_latents(project_id: str) -> Dict[str, Any]:
    """List all scene numbers with active dirty latent flags (Section 6.10)."""
    folder, _session = _get_active_session_and_folder(project_id)
    dirty_scenes = SceneLatentManager.list_dirty(folder)
    return {
        "project_id": project_id,
        "dirty_scenes": dirty_scenes,
        "count": len(dirty_scenes),
    }


def get_scene_latent_status(project_id: str, scene_id: str) -> Dict[str, Any]:
    """Query latent info and predecessor dependency for a single scene (Section 6.10)."""
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    scene_number = idx + 1
    info = SceneLatentManager.get_latent_info(folder, scene_number)
    is_dirty = SceneLatentManager.is_dirty(folder, scene_number)

    if scene_number <= 1:
        pred_needed = False
        pred_exists = True
        pred_scene = 0
        pred_path = ""
        pred_dirty = False
    else:
        pred_scene = scene_number - 1
        pred_info = SceneLatentManager.get_latent_info(folder, pred_scene)
        pred_needed = True
        pred_exists = bool(pred_info.get("exists", False))
        pred_path = str(pred_info.get("path", "") or "")
        pred_dirty = bool(pred_info.get("dirty", False)) or SceneLatentManager.is_dirty(folder, pred_scene)

    return {
        "project_id": project_id,
        "scene_id": scene_id,
        "scene_number": scene_number,
        "exists": bool(info.get("exists", False)),
        "path": info.get("path", ""),
        "frame_count": int(info.get("frame_count", 0)),
        "token_count": int(info.get("token_count", 0)),
        "fps": float(info.get("fps", 24.0)),
        "size_bytes": int(info.get("size_bytes", 0)),
        "tail_padding_frames": info.get("tail_padding_frames"),
        "dirty": is_dirty,
        "predecessor_needed": pred_needed,
        "predecessor_scene": pred_scene,
        "predecessor_exists": pred_exists,
        "predecessor_path": pred_path,
        "predecessor_dirty": pred_dirty,
    }


def delete_scene_latent(
    project_id: str,
    scene_id: str,
    all_latents: bool = False,
    reindex: bool = False,
) -> Dict[str, Any]:
    """Delete a scene's latent (or all latents) and mark successors dirty (Section 6.10)."""
    folder, session = _get_active_session_and_folder(project_id)

    if all_latents:
        removed = SceneLatentManager.delete_all_latents(folder)
        return {
            "project_id": project_id,
            "deleted": removed > 0,
            "removed_files": removed,
            "all": True,
        }

    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    scene_number = idx + 1
    deleted = SceneLatentManager.delete_latent(folder, scene_number, reindex=reindex)

    # Invariant 4: Mark downstream successor dirty if it exists
    successor = scene_number + 1
    if SceneLatentManager.latent_exists(folder, successor):
        SceneLatentManager.mark_dirty(
            folder,
            successor,
            reason=f"Predecessor scene {scene_number:03d} latent was deleted",
        )

    return {
        "project_id": project_id,
        "scene_id": scene_id,
        "scene_number": scene_number,
        "deleted": deleted,
    }


def validate_latent_continuity(folder: str, scene_number: int) -> None:
    """Validate latent continuity rule (Invariant 4) before rendering.

    Raises:
        PredecessorMissingError: If predecessor scene latent does not exist.
        LatentStaleError: If predecessor scene latent exists but is marked dirty.
    """
    if scene_number <= 1:
        return
    pred_scene = scene_number - 1
    if not SceneLatentManager.latent_exists(folder, pred_scene):
        raise PredecessorMissingError(scene_number, pred_scene)
    if SceneLatentManager.is_dirty(folder, pred_scene):
        raise LatentStaleError(scene_number, pred_scene)


async def run_rebuild_dirty_latents_job(
    job_id: str,
    payload: Dict[str, Any],
    event_bus: Any,
) -> Dict[str, Any]:
    """Rebuild dirty latent chain in strict sequential order (Section 6.10, Invariant 4)."""
    manager = get_job_manager()
    project_id = str(payload.get("project_id", "") or "").strip()
    if not project_id:
        raise ValidationError("project_id is required in job payload.")

    folder, session = _get_active_session_and_folder(project_id)
    dirty_scenes = SceneLatentManager.list_dirty(folder)
    if not dirty_scenes:
        manager.update_progress(job_id, 100.0, "completed", message="No dirty latents found.")
        return {
            "project_id": project_id,
            "rebuilt_scenes": [],
            "count": 0,
            "message": "No dirty latents found",
        }

    # Strict ascending sequential order per Invariant 4
    dirty_scenes.sort()
    total = len(dirty_scenes)
    rebuilt: List[Dict[str, Any]] = []

    segments = session.get("segments", [])

    for i, scene_num in enumerate(dirty_scenes):
        job = manager.get_job(job_id)
        if job.cancel_requested:
            raise JobCancelledError(job_id)

        idx = scene_num - 1
        if idx < 0 or idx >= len(segments):
            logger.warning(f"Scene number {scene_num} is outside session segments bounds ({len(segments)}). Skipping.")
            continue

        seg = segments[idx]
        sid = seg.get("id") or str(scene_num)

        pct = (i / total) * 100.0
        manager.update_progress(
            job_id,
            pct,
            "rendering",
            scene_id=sid,
            message=f"Rebuilding scene {scene_num:03d} latent ({i + 1}/{total})...",
        )

        render_params = dict(payload.get("render_params") or {})
        if "mode" not in render_params:
            render_params["mode"] = seg.get("video_mode") or session.get("video_mode") or "minimax_h3"

        render_res = await render_scene_video_async(
            project_id=project_id,
            scene_id=sid,
            params=render_params,
            job=job,
            manager=manager,
        )

        SceneLatentManager.clear_dirty(folder, scene_num)
        rebuilt.append({
            "scene_number": scene_num,
            "scene_id": sid,
            "video_path": render_res.get("video_path", ""),
        })

    manager.update_progress(job_id, 100.0, "completed", message=f"Rebuilt {len(rebuilt)} scene latents.")
    return {
        "project_id": project_id,
        "rebuilt_scenes": rebuilt,
        "count": len(rebuilt),
    }


def minimax_stage_recover(
    project_id: str,
    scene_id: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find and backup intermediate MiniMax H3 stage outputs (Section 6.10)."""
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    scene_number = idx + 1
    p = payload or {}
    output_folder = str(p.get("output_folder") or "").strip()
    if not output_folder:
        output_folder, _ = minimax_inputs._minimax_h3_output_location(folder, scene_number, create=False)

    min_mtime = float(p.get("min_mtime") or 0.0)
    found = video_files._find_minimax_h3_stage_outputs({
        "output_folder": output_folder,
        "min_mtime": min_mtime,
    })

    backups: Dict[str, Any] = {}
    for stage in ("stage1", "stage2"):
        stage_key = f"{stage}_path"
        stage_file = found.get(stage_key, "")
        if stage_file and os.path.isfile(stage_file):
            try:
                res = video_files._collect_minimax_h3_stage_backup({
                    "source_path": stage_file,
                    "project_folder": folder,
                    "stage": stage,
                    "scene_number": scene_number,
                })
                backups[stage] = res
            except Exception as exc:
                logger.warning(f"Could not backup {stage} for scene {scene_number}: {exc}")

    return {
        "project_id": project_id,
        "scene_id": scene_id,
        "scene_number": scene_number,
        "output_folder": output_folder,
        "stage_outputs": found,
        "backups": backups,
    }


def cleanup_minimax_output(
    project_id: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Safely clean up MiniMax H3 scratch directory (Section 6.10)."""
    folder, session = _get_active_session_and_folder(project_id)
    p = payload or {}
    scene_number = int(p.get("scene_number") or 0)
    if scene_number <= 0 and p.get("scene_id"):
        sid = p.get("scene_id")
        idx = next((i for i, s in enumerate(session.get("segments", [])) if s.get("id") == sid or str(i + 1) == str(sid)), -1)
        if idx >= 0:
            scene_number = idx + 1
    if scene_number <= 0:
        scene_number = 1

    output_folder = str(p.get("output_folder") or "").strip()
    if not output_folder:
        output_folder, _ = minimax_inputs._minimax_h3_output_location(folder, scene_number, create=False)

    clean_res = minimax_inputs._cleanup_minimax_h3_output_folder({
        "project_folder": folder,
        "output_folder": output_folder,
        "scene_number": scene_number,
    })

    return {
        "project_id": project_id,
        "scene_number": scene_number,
        **clean_res,
    }


def get_minimax_project_index(project_id: str) -> Dict[str, Any]:
    """Generate and return MiniMax H3 project index documentation (Section 6.10)."""
    folder, session = _get_active_session_and_folder(project_id)
    session_for_index = dict(session)
    session_for_index["video_engine"] = "minimax_h3"
    path = _write_minimax_project_index(folder, session_for_index)

    content = ""
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            logger.warning(f"Could not read MiniMax project index: {exc}")

    return {
        "project_id": project_id,
        "path": path,
        "content": content,
    }


def register_latent_orchestrator_handlers(manager: Optional[JobManager] = None) -> None:
    """Register MiniMax latent rebuilding job handler with JobManager."""
    if manager is None:
        manager = get_job_manager()
    manager.register_handler("latents.rebuild", run_rebuild_dirty_latents_job)
