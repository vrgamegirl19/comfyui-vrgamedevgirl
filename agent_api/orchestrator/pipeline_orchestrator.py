"""Full pipelines and dry-run planning orchestrator (Section 6.13, Section 5, C4)."""

import asyncio
import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ...runner import video_files
from ..errors import (
    ComfyExecutionError,
    JobCancelledError,
    ProjectNotFoundError,
    SceneNotFoundError,
    ValidationError,
)
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..mutations import (
    _BUILDER_SAVE_LOCK,
    _get_active_session_and_folder,
    _persist_session,
    set_scene_prompt_field_endpoint,
)
from .image_orchestrator import approve_scene_image, generate_scene_image_async
from .video_orchestrator import render_scene_video_async

logger = logging.getLogger("vrgdg.agent_api.pipeline_orchestrator")


def _filter_target_scenes(
    segments: List[Dict[str, Any]],
    scope: str = "all",
    scene_ids: Optional[List[str]] = None,
) -> List[Tuple[int, str, Dict[str, Any]]]:
    """Resolve target scenes as (index, scene_id, segment) list based on scope or explicit scene_ids."""
    if not segments:
        return []

    target_items: List[Tuple[int, str, Dict[str, Any]]] = []
    for idx, seg in enumerate(segments):
        sid = str(seg.get("id") or (idx + 1))
        target_items.append((idx, sid, seg))

    if scene_ids:
        id_set = {str(x).strip() for x in scene_ids if str(x).strip()}
        return [
            (idx, sid, seg)
            for idx, sid, seg in target_items
            if sid in id_set or str(idx + 1) in id_set
        ]

    norm_scope = str(scope or "all").strip().lower()
    if norm_scope == "selected":
        selected = [(idx, sid, seg) for idx, sid, seg in target_items if seg.get("selected")]
        return selected if selected else [target_items[0]]

    if norm_scope == "from_selected":
        first_sel = next((i for i, (_, _, seg) in enumerate(target_items) if seg.get("selected")), 0)
        return target_items[first_sel:]

    # Default "all"
    return target_items


# ==============================================================================
# Dry-Run Planning Service (Mode S, Section 6.13)
# ==============================================================================

def get_pipeline_plan(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Calculate what actions a pipeline will perform, estimating GPU time and checking prerequisites."""
    p = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])

    build_mode = str(p.get("build_mode") or "resume_missing").strip().lower()
    scope = str(p.get("scope") or "all").strip().lower()
    raw_scene_ids = p.get("scene_ids")
    if isinstance(raw_scene_ids, str):
        scene_ids = [s.strip() for s in raw_scene_ids.split(",") if s.strip()]
    elif isinstance(raw_scene_ids, list):
        scene_ids = [str(s).strip() for s in raw_scene_ids if str(s).strip()]
    else:
        scene_ids = None

    pipeline_kind = str(p.get("pipeline") or "full_video").strip().lower()

    # Determine video engine & mode
    video_engine = session.get("video_engine") or ("minimax_h3" if session.get("video_mode") == "minimax_h3" else "ltx")
    active_video_mode = str(
        p.get("video_mode")
        or session.get("video_mode")
        or ("minimax_h3" if video_engine == "minimax_h3" else "i2v")
    ).strip().lower()
    if pipeline_kind == "flf":
        active_video_mode = "flf"

    active_image_mode = str(
        p.get("image_mode")
        or session.get("image_model_mode")
        or (session.get("settings") or {}).get("image_model_mode")
        or "zimage"
    ).strip().lower()

    target_items = _filter_target_scenes(segments, scope=scope, scene_ids=scene_ids)

    scenes_plan: List[Dict[str, Any]] = []
    images_to_generate = 0
    videos_to_render = 0
    prompts_to_generate = 0

    is_image_applicable = active_video_mode not in ("t2v", "rtv") and build_mode != "redo_videos"

    for idx, sid, seg in target_items:
        scene_number = idx + 1
        duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))

        has_image = bool(
            seg.get("approved_image_path")
            or seg.get("custom_image_path")
            or (seg.get("image_history") and len(seg["image_history"]) > 0)
        )
        has_video = bool(
            seg.get("video_path")
            or seg.get("rendered_video_path")
            or (seg.get("video_history") and len(seg["video_history"]) > 0)
        )

        has_image_prompt = bool(str(seg.get("t2i_prompt") or "").strip())
        video_prompt_field = "minimax_h3_prompt" if active_video_mode.startswith("minimax") else "i2v_prompt"
        has_video_prompt = bool(str(seg.get(video_prompt_field) or "").strip())

        # Determine actions
        need_img_prompt = is_image_applicable and (not has_image_prompt or build_mode == "fresh_rebuild")
        need_img = is_image_applicable and (not has_image or build_mode == "fresh_rebuild")

        need_vid_prompt = (
            build_mode != "redo_videos"
            and (not has_video_prompt or build_mode in ("fresh_rebuild", "redo_i2v_prompts_videos"))
        )
        need_vid = (not has_video) or build_mode in ("fresh_rebuild", "redo_videos", "redo_i2v_prompts_videos")

        if need_img_prompt or need_vid_prompt:
            prompts_to_generate += int(need_img_prompt) + int(need_vid_prompt)
        if need_img:
            images_to_generate += 1
        if need_vid:
            videos_to_render += 1

        scenes_plan.append({
            "scene_id": sid,
            "scene_number": scene_number,
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "duration": duration,
            "has_image": has_image,
            "has_video": has_video,
            "has_image_prompt": has_image_prompt,
            "has_video_prompt": has_video_prompt,
            "actions": {
                "generate_image_prompt": need_img_prompt,
                "generate_image": need_img,
                "generate_video_prompt": need_vid_prompt,
                "render_video": need_vid,
            },
        })

    # GPU time estimation
    # Image: ~15s (0.25 min)
    # Video: LTX ~45s (0.75 min), MiniMax H3 ~180s (3.0 min)
    # Prompts: ~10s (0.16 min)
    # Stitch: ~15s (0.25 min)
    video_sec_per_scene = 180.0 if active_video_mode.startswith("minimax") else 45.0
    image_sec_per_scene = 25.0 if "flux" in active_image_mode else 15.0
    prompt_sec_each = 10.0
    stitch_sec = 15.0 if p.get("stitch", True) and len(target_items) > 0 else 0.0

    total_gpu_sec = (
        (images_to_generate * image_sec_per_scene)
        + (videos_to_render * video_sec_per_scene)
        + (prompts_to_generate * prompt_sec_each)
        + stitch_sec
    )
    total_gpu_min = round(total_gpu_sec / 60.0, 2)

    # Missing prerequisites validation
    missing_prerequisites: List[str] = []
    warnings: List[str] = []

    audio_file = session.get("audio_file") or ""
    if not audio_file:
        missing_prerequisites.append("Project has no audio file assigned.")
    elif not os.path.isfile(audio_file):
        missing_prerequisites.append(f"Project audio file was not found on disk: {audio_file}")

    if not target_items:
        missing_prerequisites.append("No scenes matched the requested scope or scene_ids.")

    for item in scenes_plan:
        if item["duration"] <= 0.0:
            warnings.append(f"Scene {item['scene_id']} has zero or negative duration ({item['duration']}s).")

    return {
        "project_id": project_id,
        "pipeline": pipeline_kind,
        "build_mode": build_mode,
        "scope": scope,
        "video_engine": video_engine,
        "video_mode": active_video_mode,
        "image_mode": active_image_mode,
        "target_scenes": scenes_plan,
        "summary": {
            "total_target_scenes": len(target_items),
            "images_to_generate": images_to_generate,
            "videos_to_render": videos_to_render,
            "prompts_to_generate": prompts_to_generate,
            "will_stitch": bool(p.get("stitch", True) and len(target_items) > 0),
            "estimated_gpu_seconds": round(total_gpu_sec, 1),
            "estimated_gpu_minutes": total_gpu_min,
        },
        "missing_prerequisites": missing_prerequisites,
        "warnings": warnings,
        "can_run": (len(missing_prerequisites) == 0),
    }


# ==============================================================================
# Full Video Pipeline Job Handler (Mode J, Section 6.13)
# ==============================================================================

async def run_build_full_video_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute end-to-end full video generation and stitch pipeline (GPU job)."""
    project_id = job.project_id
    if not project_id:
        raise ValidationError("project_id is required.")

    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])

    build_mode = str(job.params.get("build_mode") or "resume_missing").strip().lower()
    scope = str(job.params.get("scope") or "all").strip().lower()
    scene_ids = job.params.get("scene_ids")
    max_auto_retries = max(0, min(5, int(job.params.get("max_auto_retries", 3))))
    video_seed_mode = str(job.params.get("video_seed_mode") or "randomize").strip().lower()
    stitch = bool(job.params.get("stitch", True))

    video_engine = session.get("video_engine") or ("minimax_h3" if session.get("video_mode") == "minimax_h3" else "ltx")
    active_video_mode = str(
        job.params.get("video_mode")
        or session.get("video_mode")
        or ("minimax_h3" if video_engine == "minimax_h3" else "i2v")
    ).strip().lower()

    active_image_mode = str(
        job.params.get("image_mode")
        or session.get("image_model_mode")
        or (session.get("settings") or {}).get("image_model_mode")
        or "zimage"
    ).strip().lower()

    target_items = _filter_target_scenes(segments, scope=scope, scene_ids=scene_ids)
    if not target_items:
        manager.update_progress(job.id, 100.0, "completed", message="No scenes to process.")
        return {"project_id": project_id, "scenes_processed": 0, "images_generated": 0, "videos_rendered": 0}

    total_scenes = len(target_items)
    is_image_applicable = active_video_mode not in ("t2v", "rtv") and build_mode != "redo_videos"

    images_generated_count = 0
    videos_rendered_count = 0
    scene_results: List[Dict[str, Any]] = []

    # ==========================================================================
    # Stage 1: Image Generation Pass (5% -> 35%)
    # ==========================================================================
    if is_image_applicable:
        manager.update_progress(job.id, 5.0, "images_pass", message="Starting image generation pass...")
        for i, (idx, sid, seg) in enumerate(target_items):
            if job.cancel_requested:
                raise JobCancelledError(job.id)

            has_image = bool(
                seg.get("approved_image_path")
                or seg.get("custom_image_path")
                or (seg.get("image_history") and len(seg["image_history"]) > 0)
            )
            need_img = (build_mode == "fresh_rebuild") or (not has_image)

            if need_img:
                pct = 5.0 + (i / total_scenes) * 30.0
                manager.update_progress(
                    job.id,
                    pct,
                    "generating_image",
                    scene_id=sid,
                    message=f"Generating image for scene {sid} ({i+1}/{total_scenes})...",
                )

                # Ensure scene has a prompt
                if not seg.get("t2i_prompt"):
                    t2i_p = str(seg.get("lyric_text") or f"Scene {idx+1} visual depiction").strip()
                    set_scene_prompt_field_endpoint(project_id, sid, "t2i_prompt", t2i_p, origin="pipeline")
                    seg["t2i_prompt"] = t2i_p

                # Generate image
                img_res = await generate_scene_image_async(
                    project_id=project_id,
                    scene_id=sid,
                    params={"mode": active_image_mode},
                    job=job,
                    manager=manager,
                )
                # Auto-approve image
                approve_scene_image(project_id, sid, image_path=img_res.get("image_path"))
                images_generated_count += 1

                # Update local segment copy
                seg["approved_image_path"] = img_res.get("image_path")
                seg["preview_mode"] = "image"

    # ==========================================================================
    # Stage 2: Video Prompt Pass (35% -> 45%)
    # ==========================================================================
    if build_mode != "redo_videos":
        manager.update_progress(job.id, 35.0, "prompts_pass", message="Checking video prompts...")
        video_prompt_field = "minimax_h3_prompt" if active_video_mode.startswith("minimax") else "i2v_prompt"

        for i, (idx, sid, seg) in enumerate(target_items):
            if job.cancel_requested:
                raise JobCancelledError(job.id)

            cur_p = str(seg.get(video_prompt_field) or "").strip()
            if not cur_p or build_mode in ("fresh_rebuild", "redo_i2v_prompts_videos"):
                fallback_p = str(
                    seg.get("lyric_text")
                    or seg.get("t2i_prompt")
                    or f"Cinematic motion sequence for scene {idx+1}"
                ).strip()
                set_scene_prompt_field_endpoint(project_id, sid, video_prompt_field, fallback_p, origin="pipeline")
                seg[video_prompt_field] = fallback_p

    # ==========================================================================
    # Stage 3: Video Render Pass (45% -> 85%)
    # ==========================================================================
    manager.update_progress(job.id, 45.0, "videos_pass", message="Starting video rendering pass...")
    for i, (idx, sid, seg) in enumerate(target_items):
        if job.cancel_requested:
            raise JobCancelledError(job.id)

        has_video = bool(
            seg.get("video_path")
            or seg.get("rendered_video_path")
            or (seg.get("video_history") and len(seg["video_history"]) > 0)
        )
        need_vid = (not has_video) or build_mode in ("fresh_rebuild", "redo_videos", "redo_i2v_prompts_videos")

        if not need_vid:
            scene_results.append({
                "scene_id": sid,
                "status": "skipped_existing",
                "video_path": seg.get("video_path") or seg.get("rendered_video_path"),
            })
            continue

        pct = 45.0 + (i / total_scenes) * 40.0
        manager.update_progress(
            job.id,
            pct,
            "rendering_video",
            scene_id=sid,
            message=f"Rendering video for scene {sid} ({i+1}/{total_scenes})...",
        )

        last_error: Optional[Exception] = None
        for attempt in range(max_auto_retries + 1):
            if job.cancel_requested:
                raise JobCancelledError(job.id)
            try:
                render_res = await render_scene_video_async(
                    project_id=project_id,
                    scene_id=sid,
                    params={
                        "mode": active_video_mode,
                        "randomize_seed": (video_seed_mode == "randomize"),
                    },
                    job=job,
                    manager=manager,
                )
                videos_rendered_count += 1
                scene_results.append({
                    "scene_id": sid,
                    "status": "success",
                    "video_path": render_res.get("video_path"),
                })
                seg["video_path"] = render_res.get("video_path")
                break
            except Exception as exc:
                last_error = exc
                if job.cancel_requested:
                    raise JobCancelledError(job.id)
                if attempt >= max_auto_retries:
                    logger.error(f"Scene {sid} video render failed after {attempt+1} attempts: {exc}")
                    scene_results.append({"scene_id": sid, "status": "failed", "error": str(exc)})
                    raise exc
                logger.warning(f"Scene {sid} render attempt {attempt+1} failed ({exc}); retrying...")

    # ==========================================================================
    # Stage 4: Final Stitch Pass (85% -> 100%)
    # ==========================================================================
    final_video_path = ""
    if stitch and total_scenes > 0:
        manager.update_progress(job.id, 88.0, "stitching", message="Stitching scene videos into final video...")
        try:
            _, fresh_session = _get_active_session_and_folder(project_id)
            fresh_segments = fresh_session.get("segments", [])

            # Collect all scene video paths in sequential order
            stitch_paths = []
            for s in fresh_segments:
                v = s.get("video_path") or s.get("rendered_video_path")
                if v and os.path.isfile(v):
                    stitch_paths.append(v)

            if stitch_paths:
                stitch_res = await asyncio.to_thread(
                    video_files._stitch_scene_videos,
                    {
                        "project_folder": folder,
                        "scene_paths": stitch_paths,
                        "audio_path": fresh_session.get("audio_file", ""),
                    },
                )
                final_video_path = stitch_res.get("final_video_path", "")
        except Exception as stitch_err:
            logger.warning(f"Final video stitch after full pipeline failed: {stitch_err}")

    manager.update_progress(job.id, 100.0, "completed", message="Full video pipeline completed.")
    return {
        "project_id": project_id,
        "pipeline": "full_video",
        "build_mode": build_mode,
        "scenes_processed": total_scenes,
        "images_generated": images_generated_count,
        "videos_rendered": videos_rendered_count,
        "final_video_path": final_video_path,
        "results": scene_results,
    }


# ==============================================================================
# Full FLF Pipeline Job Handler (Mode J, Section 6.13)
# ==============================================================================

async def run_build_flf_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute end-to-end First/Last Frame (FLF) video pipeline (GPU job)."""
    project_id = job.project_id
    if not project_id:
        raise ValidationError("project_id is required.")

    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])

    scope = str(job.params.get("scope") or "all").strip().lower()
    scene_ids = job.params.get("scene_ids")
    redo_images = bool(job.params.get("redo_images", False))
    redo_videos = bool(job.params.get("redo_videos", False))
    max_auto_retries = max(0, min(5, int(job.params.get("max_auto_retries", 3))))
    stitch = bool(job.params.get("stitch", True))

    active_image_mode = str(
        job.params.get("image_mode")
        or session.get("image_model_mode")
        or (session.get("settings") or {}).get("image_model_mode")
        or "zimage"
    ).strip().lower()

    target_items = _filter_target_scenes(segments, scope=scope, scene_ids=scene_ids)
    if not target_items:
        manager.update_progress(job.id, 100.0, "completed", message="No scenes to process for FLF.")
        return {"project_id": project_id, "scenes_processed": 0, "videos_rendered": 0}

    total_scenes = len(target_items)
    manager.update_progress(job.id, 10.0, "flf_image_chain", message="Building FLF image chain...")

    # Stage 1: Build image chain (Start images)
    for i, (idx, sid, seg) in enumerate(target_items):
        if job.cancel_requested:
            raise JobCancelledError(job.id)

        has_image = bool(
            seg.get("approved_image_path")
            or seg.get("custom_image_path")
            or (seg.get("image_history") and len(seg["image_history"]) > 0)
        )
        if redo_images or not has_image:
            pct = 10.0 + (i / total_scenes) * 30.0
            manager.update_progress(
                job.id,
                pct,
                "flf_image",
                scene_id=sid,
                message=f"Generating start image for FLF scene {sid}...",
            )
            if not seg.get("t2i_prompt"):
                t2i_p = str(seg.get("lyric_text") or f"FLF scene {idx+1} initial anchor").strip()
                set_scene_prompt_field_endpoint(project_id, sid, "t2i_prompt", t2i_p, origin="pipeline_flf")
                seg["t2i_prompt"] = t2i_p

            img_res = await generate_scene_image_async(
                project_id=project_id,
                scene_id=sid,
                params={"mode": active_image_mode},
                job=job,
                manager=manager,
            )
            approve_scene_image(project_id, sid, image_path=img_res.get("image_path"))
            seg["approved_image_path"] = img_res.get("image_path")

    # Stage 2: Render FLF videos
    manager.update_progress(job.id, 45.0, "flf_videos", message="Rendering FLF scene videos...")
    videos_rendered_count = 0
    scene_results: List[Dict[str, Any]] = []

    for i, (idx, sid, seg) in enumerate(target_items):
        if job.cancel_requested:
            raise JobCancelledError(job.id)

        has_video = bool(
            seg.get("video_path")
            or seg.get("rendered_video_path")
            or (seg.get("video_history") and len(seg["video_history"]) > 0)
        )
        if not redo_videos and has_video:
            scene_results.append({
                "scene_id": sid,
                "status": "skipped_existing",
                "video_path": seg.get("video_path") or seg.get("rendered_video_path"),
            })
            continue

        pct = 45.0 + (i / total_scenes) * 40.0
        manager.update_progress(
            job.id,
            pct,
            "flf_render",
            scene_id=sid,
            message=f"Rendering FLF video for scene {sid} ({i+1}/{total_scenes})...",
        )

        for attempt in range(max_auto_retries + 1):
            if job.cancel_requested:
                raise JobCancelledError(job.id)
            try:
                render_res = await render_scene_video_async(
                    project_id=project_id,
                    scene_id=sid,
                    params={"mode": "flf"},
                    job=job,
                    manager=manager,
                )
                videos_rendered_count += 1
                scene_results.append({
                    "scene_id": sid,
                    "status": "success",
                    "video_path": render_res.get("video_path"),
                })
                seg["video_path"] = render_res.get("video_path")
                break
            except Exception as exc:
                if job.cancel_requested:
                    raise JobCancelledError(job.id)
                if attempt >= max_auto_retries:
                    scene_results.append({"scene_id": sid, "status": "failed", "error": str(exc)})
                    raise exc

    # Stage 3: Final Stitch
    final_video_path = ""
    if stitch and total_scenes > 0:
        manager.update_progress(job.id, 88.0, "flf_stitching", message="Stitching FLF scene videos...")
        try:
            _, fresh_session = _get_active_session_and_folder(project_id)
            stitch_paths = [
                s.get("video_path")
                for s in fresh_session.get("segments", [])
                if s.get("video_path") and os.path.isfile(s.get("video_path"))
            ]
            if stitch_paths:
                stitch_res = await asyncio.to_thread(
                    video_files._stitch_scene_videos,
                    {
                        "project_folder": folder,
                        "scene_paths": stitch_paths,
                        "audio_path": fresh_session.get("audio_file", ""),
                    },
                )
                final_video_path = stitch_res.get("final_video_path", "")
        except Exception as stitch_err:
            logger.warning(f"Final video stitch after FLF pipeline failed: {stitch_err}")

    manager.update_progress(job.id, 100.0, "completed", message="FLF pipeline completed.")
    return {
        "project_id": project_id,
        "pipeline": "flf",
        "scenes_processed": total_scenes,
        "videos_rendered": videos_rendered_count,
        "final_video_path": final_video_path,
        "results": scene_results,
    }


# ==============================================================================
# Handler Registration
# ==============================================================================

def register_pipeline_orchestrator_handlers(manager: Optional[JobManager] = None) -> None:
    """Register all pipeline job handlers with JobManager."""
    if manager is None:
        manager = get_job_manager()

    manager.register_handler("pipeline.build_full_video", run_build_full_video_job)
    manager.register_handler("pipeline.build_flf", run_build_flf_job)
