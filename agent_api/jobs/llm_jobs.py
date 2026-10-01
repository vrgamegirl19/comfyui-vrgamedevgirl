"""LLM prompt generation jobs and background handlers (Section 6.7)."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from ...llm.builder_instructions import (
    _BUILDER_INSTRUCTION_DEFAULTS,
    _get_builder_instruction,
    _list_builder_instruction_presets,
    _reset_builder_instruction,
    _safe_builder_instruction_key,
    _safe_preset_name,
    _save_builder_instruction,
)
from ...llm.builder_runner import (
    _EXTERNAL_LLM_RUNNERS,
    _list_lm_studio_models,
    _list_own_server_models,
    _test_llm_api,
    _test_own_server,
)
from ...llm.cache import _clear_vrgdg_llm_caches
from ...llm import image_prompt_generation as img_gen
from ...llm import video_prompt_generation as vid_gen
from ..errors import (
    AgentApiError,
    JobCancelledError,
    ProjectNotFoundError,
    SceneNotFoundError,
    ValidationError,
)
from ..mutations import (
    _get_active_session_and_folder,
    set_scene_prompt_field_endpoint,
)
from ..llm_runtime import llm_payload_from_session, prepare_llm_payload
from ..paths import resolve_project_folder
from .manager import JobManager, get_job_manager
from .models import Job

logger = logging.getLogger("vrgdg.agent_api.llm_jobs")


def is_llm_runner_gpu(params: Dict[str, Any], project_folder: Optional[str] = None) -> bool:
    """Determine whether the runner consumes local GPU VRAM."""
    runner = str(params.get("text_runner") or params.get("runner") or "").strip().lower()
    if not runner and project_folder:
        try:
            _, session = _get_active_session_and_folder(os.path.basename(project_folder))
            runner = str(session.get("text_gemma_runner") or "").strip().lower()
        except Exception:
            pass
    runner = runner.replace("-", "_")
    if runner == "lmstudio":
        runner = "lm_studio"
    if runner in _EXTERNAL_LLM_RUNNERS:
        return False
    return True


def _prepare_llm_payload(job: Job, project_folder: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the payload the generator functions read.

    The project's saved LLM runner settings come first and the request's own parameters win. For LM
    Studio the payload is pointed at the model that is currently loaded (see ``llm_runtime``), so a
    job never makes LM Studio load or switch a model.
    """
    session: Dict[str, Any] = {}
    if job.project_id:
        try:
            _, session = _get_active_session_and_folder(job.project_id)
        except Exception:
            session = {}
    payload = {**llm_payload_from_session(session), **(job.params or {})}
    payload["project_folder"] = project_folder
    if extra:
        payload.update(extra)
    return prepare_llm_payload(payload)


# ==============================================================================
# 1. Concept Prompts & Motion Notes
# ==============================================================================

def run_concept_prompts_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute concept prompts generation job."""
    if not job.project_id:
        raise ValidationError("project_id is required for concept prompts.")
    folder = resolve_project_folder(job.project_id)

    manager.update_progress(job.id, 10.0, "preparing", message="Loading context...")
    payload = _prepare_llm_payload(job, folder)

    manager.update_progress(job.id, 40.0, "generating", message="Generating concept prompts via LLM...")
    res = img_gen._generate_builder_concept_prompts(payload)

    manager.update_progress(job.id, 90.0, "saving", message="Finalizing concept prompts...")
    return res if isinstance(res, dict) else {"result": res}


def run_motion_notes_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute motion notes generation job."""
    if not job.project_id:
        raise ValidationError("project_id is required for motion notes.")
    folder = resolve_project_folder(job.project_id)

    manager.update_progress(job.id, 10.0, "preparing", message="Loading context...")
    payload = _prepare_llm_payload(job, folder)

    manager.update_progress(job.id, 40.0, "generating", message="Generating motion notes via LLM...")
    res = vid_gen._generate_builder_motion_notes(payload)

    manager.update_progress(job.id, 90.0, "saving", message="Finalizing motion notes...")
    return res if isinstance(res, dict) else {"result": res}


# ==============================================================================
# 2. Scene Image & Video Prompt Generation
# ==============================================================================

def run_scene_image_prompt_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Generate image prompt for a specific scene and save to session."""
    if not job.project_id:
        raise ValidationError("project_id is required for scene image prompt.")
    folder = resolve_project_folder(job.project_id)
    scene_id = job.params.get("scene_id")
    if not scene_id:
        raise ValidationError("scene_id is required.")

    manager.update_progress(job.id, 10.0, "preparing", scene_id=scene_id, message="Loading scene data...")
    payload = _prepare_llm_payload(job, folder, {"scene_id": scene_id})

    mode = str(job.params.get("mode") or "zimage").strip().lower()
    manager.update_progress(job.id, 40.0, "generating", scene_id=scene_id, message=f"Generating {mode} image prompt...")

    if mode in ("flux_klein", "flux"):
        res = img_gen._generate_flux_klein_prompt(payload)
    elif mode in ("nano_banana", "nb"):
        res = img_gen._generate_nb_image_prompt(payload)
    else:
        res = img_gen._generate_builder_t2i_prompt(payload)

    prompt_text = ""
    if isinstance(res, dict):
        prompt_text = str(res.get("prompt") or res.get("t2i_prompt") or res.get("text") or "").strip()
    elif isinstance(res, str):
        prompt_text = res.strip()

    if prompt_text:
        set_scene_prompt_field_endpoint(job.project_id, scene_id, "t2i_prompt", prompt_text, origin="llm")

    manager.update_progress(job.id, 90.0, "saving", scene_id=scene_id, message="Saved image prompt.")
    return {"scene_id": scene_id, "prompt": prompt_text, "raw": res}


def run_scene_video_prompt_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Generate video prompt for a specific scene and save to session."""
    if not job.project_id:
        raise ValidationError("project_id is required for scene video prompt.")
    folder = resolve_project_folder(job.project_id)
    scene_id = job.params.get("scene_id")
    if not scene_id:
        raise ValidationError("scene_id is required.")

    mode = str(job.params.get("mode") or "i2v").strip().lower()
    manager.update_progress(job.id, 10.0, "preparing", scene_id=scene_id, message="Loading scene context...")
    payload = _prepare_llm_payload(job, folder, {"scene_id": scene_id, "mode": mode})

    manager.update_progress(job.id, 40.0, "generating", scene_id=scene_id, message=f"Generating {mode} video prompt...")

    if mode == "t2v":
        res = vid_gen._generate_builder_t2v_prompt(payload)
    else:
        res = vid_gen._generate_builder_i2v_prompt(payload)

    prompt_text = ""
    if isinstance(res, dict):
        prompt_text = str(res.get("prompt") or res.get("i2v_prompt") or res.get("text") or "").strip()
    elif isinstance(res, str):
        prompt_text = res.strip()

    if prompt_text:
        target_field = "minimax_h3_prompt" if mode.startswith("minimax") else "i2v_prompt"
        set_scene_prompt_field_endpoint(job.project_id, scene_id, target_field, prompt_text, origin="llm")

    manager.update_progress(job.id, 90.0, "saving", scene_id=scene_id, message="Saved video prompt.")
    return {"scene_id": scene_id, "prompt": prompt_text, "mode": mode, "raw": res}


def run_scene_chained_video_prompt_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Generate chained video prompt for scene continuity."""
    if not job.project_id:
        raise ValidationError("project_id is required for chained video prompt.")
    folder = resolve_project_folder(job.project_id)
    scene_id = job.params.get("scene_id")
    if not scene_id:
        raise ValidationError("scene_id is required.")

    manager.update_progress(job.id, 10.0, "preparing", scene_id=scene_id, message="Preparing chained prompt context...")
    payload = _prepare_llm_payload(job, folder, {"scene_id": scene_id})

    manager.update_progress(job.id, 40.0, "generating", scene_id=scene_id, message="Generating chained video prompt...")
    res = vid_gen._generate_builder_chained_i2v_prompt(payload)

    prompt_text = ""
    if isinstance(res, dict):
        prompt_text = str(res.get("prompt") or res.get("i2v_prompt") or res.get("text") or "").strip()
    elif isinstance(res, str):
        prompt_text = res.strip()

    if prompt_text:
        set_scene_prompt_field_endpoint(job.project_id, scene_id, "i2v_prompt", prompt_text, origin="llm")

    return {"scene_id": scene_id, "prompt": prompt_text, "raw": res}


def run_enhance_prompt_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Enhance an existing video prompt."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    folder = resolve_project_folder(job.project_id)
    scene_id = job.params.get("scene_id")

    manager.update_progress(job.id, 20.0, "enhancing", scene_id=scene_id, message="Enhancing prompt...")
    payload = _prepare_llm_payload(job, folder)
    res = vid_gen._enhance_builder_video_prompt(payload)

    prompt_text = ""
    if isinstance(res, dict):
        prompt_text = str(res.get("prompt") or res.get("enhanced_prompt") or res.get("text") or "").strip()
    elif isinstance(res, str):
        prompt_text = res.strip()

    if prompt_text and scene_id:
        set_scene_prompt_field_endpoint(job.project_id, scene_id, "i2v_prompt", prompt_text, origin="llm")

    return {"scene_id": scene_id, "prompt": prompt_text, "raw": res}


def run_edit_prompt_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Edit an image or video prompt using natural language instructions."""
    if not job.project_id:
        raise ValidationError("project_id is required.")
    folder = resolve_project_folder(job.project_id)
    scene_id = job.params.get("scene_id")
    target = str(job.params.get("target") or "video").strip().lower()

    manager.update_progress(job.id, 20.0, "editing", scene_id=scene_id, message=f"Editing {target} prompt...")
    payload = _prepare_llm_payload(job, folder)

    if target == "image":
        res = img_gen._edit_builder_image_prompt(payload)
        field = "t2i_prompt"
    else:
        res = vid_gen._edit_builder_video_prompt(payload)
        field = "i2v_prompt"

    prompt_text = ""
    if isinstance(res, dict):
        prompt_text = str(res.get("prompt") or res.get("text") or "").strip()
    elif isinstance(res, str):
        prompt_text = res.strip()

    if prompt_text and scene_id:
        set_scene_prompt_field_endpoint(job.project_id, scene_id, field, prompt_text, origin="llm")

    return {"scene_id": scene_id, "prompt": prompt_text, "target": target, "raw": res}


# ==============================================================================
# 3. Batch Prompts Generation Across Scenes (Section 6.7)
# ==============================================================================

def run_batch_prompts_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    """Execute batch prompt generation across multiple scenes (Section 6.7)."""
    if not job.project_id:
        raise ValidationError("project_id is required for batch prompts.")
    folder = resolve_project_folder(job.project_id)
    _, session = _get_active_session_and_folder(job.project_id)
    segments = session.get("segments", [])

    kind = str(job.params.get("kind") or "video").strip().lower()
    scope = str(job.params.get("scope") or "all").strip().lower()
    run_mode = str(job.params.get("run_mode") or "all").strip().lower()
    specified_ids = set(job.params.get("scene_ids") or [])

    # Filter target scenes
    target_segments = []
    for idx, seg in enumerate(segments):
        sid = seg.get("id") or str(idx + 1)
        if specified_ids and sid not in specified_ids:
            continue

        prompt_field = "t2i_prompt" if kind == "image" else "i2v_prompt"
        has_prompt = bool(str(seg.get(prompt_field) or "").strip())

        if run_mode == "resume_missing" and has_prompt:
            continue

        target_segments.append(seg)

    total = len(target_segments)
    if total == 0:
        return {"processed": 0, "message": "No scenes matched batch prompt criteria."}

    manager.update_progress(job.id, 0.0, "batch_prompts", stage_index=0, stage_count=total, message=f"Starting batch for {total} scene(s)...")

    results = []
    for idx, seg in enumerate(target_segments):
        if job.cancel_requested:
            raise JobCancelledError(job.id)

        sid = seg.get("id") or str(idx + 1)
        pct = round((idx / total) * 100.0, 1)
        manager.update_progress(
            job.id,
            pct,
            "batch_prompts",
            stage_index=idx + 1,
            stage_count=total,
            scene_id=sid,
            message=f"Generating {kind} prompt for scene {idx + 1} of {total} ({sid})...",
        )

        try:
            payload = _prepare_llm_payload(job, folder, {"scene_id": sid})
            if kind == "image":
                mode = str(job.params.get("mode") or "zimage").strip().lower()
                if mode in ("flux_klein", "flux"):
                    res = img_gen._generate_flux_klein_prompt(payload)
                elif mode in ("nano_banana", "nb"):
                    res = img_gen._generate_nb_image_prompt(payload)
                else:
                    res = img_gen._generate_builder_t2i_prompt(payload)
                field = "t2i_prompt"
            else:
                mode = str(job.params.get("mode") or "i2v").strip().lower()
                if mode == "t2v":
                    res = vid_gen._generate_builder_t2v_prompt(payload)
                else:
                    res = vid_gen._generate_builder_i2v_prompt(payload)
                field = "minimax_h3_prompt" if mode.startswith("minimax") else "i2v_prompt"

            prompt_text = ""
            if isinstance(res, dict):
                prompt_text = str(res.get("prompt") or res.get(field) or res.get("text") or "").strip()
            elif isinstance(res, str):
                prompt_text = res.strip()

            if prompt_text:
                set_scene_prompt_field_endpoint(job.project_id, sid, field, prompt_text, origin="llm_batch")
                results.append({"scene_id": sid, "status": "success", "field": field})
            else:
                results.append({"scene_id": sid, "status": "empty_prompt"})
        except Exception as e:
            logger.warning(f"Failed prompt for scene {sid}: {e}")
            results.append({"scene_id": sid, "status": "error", "error": str(e)})

    manager.update_progress(job.id, 100.0, "completed", stage_index=total, stage_count=total, message="Batch prompt generation finished.")
    return {
        "processed": len(results),
        "total_targets": total,
        "results": results,
    }


# ==============================================================================
# 4. Handler Registration
# ==============================================================================

def register_llm_job_handlers(manager: Optional[JobManager] = None) -> None:
    """Register all LLM generator handlers with JobManager."""
    if manager is None:
        manager = get_job_manager()

    manager.register_handler("llm.concepts", run_concept_prompts_job)
    manager.register_handler("llm.motion_notes", run_motion_notes_job)
    manager.register_handler("llm.scene_image_prompt", run_scene_image_prompt_job)
    manager.register_handler("llm.scene_video_prompt", run_scene_video_prompt_job)
    manager.register_handler("llm.scene_chained_video_prompt", run_scene_chained_video_prompt_job)
    manager.register_handler("llm.enhance_prompt", run_enhance_prompt_job)
    manager.register_handler("llm.edit_prompt", run_edit_prompt_job)
    manager.register_handler("llm.batch_prompts", run_batch_prompts_job)
