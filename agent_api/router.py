"""HTTP Route definitions for VRGDG Agent API v1 (Section 6)."""

import asyncio
import functools
import json
import os
from aiohttp import web
from server import PromptServer

from ..builder.audio import _find_ffmpeg_path
from ..llm.builder_instructions import (
    _get_builder_instruction,
    _list_builder_instruction_presets,
    _load_builder_instruction_preset,
    _reset_builder_instruction,
    _save_builder_instruction,
    _save_builder_instruction_preset,
)
from ..llm.cache import _clear_vrgdg_llm_caches
from ..llm.builder_runner import (
    _gemma_choices,
    _list_lm_studio_models,
    _list_own_server_models,
    _llm_multi_choices,
    _test_llm_api,
    _test_own_server,
)
from ..runner.models import _folder_choices, _lora_choices, _ltx_video_model_choices

from .auth import verify_auth
from .envelope import api_error, api_exception, api_success
from .llm_runtime import describe_active_llm, llm_payload_from_session
from .errors import ValidationError
from .jobs import (
    get_event_broadcaster,
    get_job_manager,
    is_llm_runner_gpu,
    register_llm_job_handlers,
)
from .orchestrator import (
    approve_scene_image,
    build_video_graph_for_mode,
    cleanup_minimax_output,
    delete_preview_service,
    delete_scene_image,
    delete_scene_latent,
    delete_scene_video,
    estimate_scene_face_fix_anchors,
    extract_frame_from_video_to_image,
    get_adjust_presets,
    get_dirty_latents,
    get_minimax_project_index,
    get_pipeline_plan,
    list_scene_takes,
    get_project_latents_status,
    get_scene_latent_status,
    list_luts_service,
    list_project_final_videos,
    minimax_stage_recover,
    preview_scene_adjust,
    preview_scene_grain,
    preview_scene_lut,
    put_adjust_preset,
    recover_scene_video,
    register_image_orchestrator_handlers,
    register_latent_orchestrator_handlers,
    assign_scenes,
    register_lyrics_orchestrator_handlers,
    register_reference_orchestrator_handlers,
    register_minimax_prompt_orchestrator_handlers,
    register_storyboard_orchestrator_handlers,
    set_story_settings,
    register_pipeline_orchestrator_handlers,
    register_post_orchestrator_handlers,
    register_video_orchestrator_handlers,
    revert_scene_image,
    save_scene_image_custom,
    scan_project_scene_videos,
    select_scene_video,
    validate_stitch_request,
    upload_lut_service,
)
from .paths import resolve_project_folder
from .modes import get_modes_catalog
from .mutations import (
    get_scene_minimax_references,
    set_scene_minimax_references,
    assemble_minimax_prompt_endpoint,
    attach_project_audio,
    bulk_scene_operations,
    calibrate_beats,
    create_project,
    create_project_silent_audio,
    create_scene,
    delete_project_by_id,
    delete_reference,
    delete_scene,
    duplicate_project,
    export_project,
    get_audio_beats,
    get_audio_waveform,
    get_project_lyrics,
    get_project_references,
    get_project_story,
    get_prompt_context,
    merge_scenes,
    move_scene,
    patch_project_settings,
    patch_scene,
    preflight_project_settings,
    put_project_story,
    resize_scene,
    set_audio_beats,
    set_project_lyrics,
    set_scene_prompt_field_endpoint,
    split_scene,
    _get_active_session_and_folder,
    enforce_scene_lengths_on_project,
    timeline_bulk,
    timeline_close_gaps,
    timeline_snap,
    update_scene_reference_mapping,
    upsert_reference_location,
    upsert_reference_subject,
    validate_minimax_prompt_endpoint,
    validate_project,
)
from .projects import (
    get_project_assets,
    get_project_detail,
    get_project_scenes,
    get_project_summary,
    get_scene_detail,
    list_projects,
)
from ..minimax.settings_payload import minimax_h3_settings_schema
from .schemas import extract_effective_settings


_API_V1_PREFIX = "/vrgdg/api/v1"
_VRGDG_AGENT_API_ROUTES_REGISTERED = False


def _api_endpoint(handler):
    """Decorator to enforce auth and map exceptions to standard JSON error envelopes."""
    @functools.wraps(handler)
    async def wrapper(request: web.Request, *args, **kwargs):
        try:
            verify_auth(request)
            return await handler(request, *args, **kwargs)
        except Exception as exc:
            return api_exception(exc)
    return wrapper


def register_agent_api_routes(server_instance=None):
    """Register all Agent API v1 endpoints with PromptServer."""
    global _VRGDG_AGENT_API_ROUTES_REGISTERED
    if _VRGDG_AGENT_API_ROUTES_REGISTERED:
        return

    if server_instance is None:
        server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    # Crash recovery for unfinished jobs left running across projects (Section 5.3)
    try:
        recovered_count = get_job_manager().recover_on_startup()
        if recovered_count:
            print(f"[VRGDG API] Recovered {recovered_count} interrupted job(s) across projects.")
    except Exception as _re:
        print(f"[VRGDG API] Job startup recovery warning: {_re}")

    # Register background job handlers
    try:
        register_llm_job_handlers(get_job_manager())
        register_image_orchestrator_handlers(get_job_manager())
        register_video_orchestrator_handlers(get_job_manager())
        register_latent_orchestrator_handlers(get_job_manager())
        register_post_orchestrator_handlers(get_job_manager())
        register_pipeline_orchestrator_handlers(get_job_manager())
        register_lyrics_orchestrator_handlers(get_job_manager())
        register_reference_orchestrator_handlers(get_job_manager())
        register_storyboard_orchestrator_handlers(get_job_manager())
        register_minimax_prompt_orchestrator_handlers(get_job_manager())
    except Exception as _je:
        print(f"[VRGDG API] Job handlers registration warning: {_je}")

    # 1. System, capabilities, and modes
    @server_instance.routes.get(f"{_API_V1_PREFIX}/meta")
    @_api_endpoint
    async def api_meta(request: web.Request):
        from .. import __updated__, __version__
        data = {
            "api_version": "v1",
            "pack_version": __version__,
            "pack_updated": __updated__,
            "capabilities": [
                "projects",
                "scenes",
                "assets",
                "modes",
                "models",
                "settings",
                "summary",
                "mutations",
                "references",
                "lyrics",
                "audio",
                "prompts",
                "images",
                "videos",
                "latents",
                "post",
                "face_fix",
                "pipelines",
                "jobs",
                "events",
                "queue",
            ],
            "schema_version": 1,
        }
        return api_success(data)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/health")
    @_api_endpoint
    async def api_health(request: web.Request):
        def _get_health():
            queue_depth = 0
            comfy_up = True
            try:
                queue = PromptServer.instance.prompt_queue
                queue_depth = queue.get_tasks_remaining()
            except Exception:
                comfy_up = False

            ffmpeg_ok = False
            try:
                ffmpeg_ok = bool(_find_ffmpeg_path())
            except Exception:
                pass

            gpu_available = False
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except Exception:
                pass

            return {
                "status": "healthy" if comfy_up else "degraded",
                "comfy_up": comfy_up,
                "queue_depth": queue_depth,
                "ffmpeg_available": ffmpeg_ok,
                "gpu_available": gpu_available,
            }

        health = await asyncio.to_thread(_get_health)
        return api_success(health)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/modes")
    @_api_endpoint
    async def api_modes(request: web.Request):
        modes = await asyncio.to_thread(get_modes_catalog)
        return api_success(modes)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/refmods")
    @_api_endpoint
    async def api_refmods(request: web.Request):
        from ..minimax.refmod_library import list_refmods

        entries = await asyncio.to_thread(list_refmods, str(request.query.get("folder", "") or ""))
        return api_success({"refmods": [{k: v for k, v in entry.items() if k not in ("path", "directory")} for entry in entries]})

    @server_instance.routes.get(f"{_API_V1_PREFIX}/models")
    @_api_endpoint
    async def api_models(request: web.Request):
        def _get_models():
            video_ggufs, video_diff = _ltx_video_model_choices()
            return {
                "unets": _folder_choices(("unet", "diffusion_models")),
                "video_gguf_unets": video_ggufs,
                "video_diffusion_models": video_diff,
                "vae": _folder_choices("vae"),
                "clip": _folder_choices(("clip", "text_encoders")),
                "latent_upscalers": _folder_choices("latent_upscale_models"),
                "loras": _lora_choices(),
                "gemma_models": _gemma_choices().get("models", []),
                "llm_choices": _llm_multi_choices(),
            }

        models = await asyncio.to_thread(_get_models)
        return api_success(models)

    # 2. Projects read endpoints
    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects")
    @_api_endpoint
    async def api_list_projects(request: web.Request):
        root = request.query.get("root")
        projects = await asyncio.to_thread(list_projects, root)
        return api_success({"items": projects, "count": len(projects)})

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}")
    @_api_endpoint
    async def api_get_project(request: web.Request):
        pid = request.match_info["pid"]
        inc_param = request.query.get("include", "")
        includes = [s.strip() for s in inc_param.split(",") if s.strip()] if inc_param else None
        detail = await asyncio.to_thread(get_project_detail, pid, includes)
        return api_success(detail, revision=detail.get("revision"))

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/summary")
    @_api_endpoint
    async def api_get_project_summary(request: web.Request):
        pid = request.match_info["pid"]
        summary = await asyncio.to_thread(get_project_summary, pid)
        return api_success(summary, revision=summary.get("revision"))

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/settings")
    @_api_endpoint
    async def api_get_project_settings(request: web.Request):
        pid = request.match_info["pid"]
        detail = await asyncio.to_thread(get_project_detail, pid, ["settings"])
        return api_success(detail.get("settings", {}), revision=detail.get("revision"))

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/assets")
    @_api_endpoint
    async def api_get_project_assets(request: web.Request):
        pid = request.match_info["pid"]
        assets = await asyncio.to_thread(get_project_assets, pid)
        return api_success({"items": assets, "count": len(assets)})

    # 3. Scenes read endpoints
    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes")
    @_api_endpoint
    async def api_get_project_scenes(request: web.Request):
        pid = request.match_info["pid"]
        has_img = request.query.get("has_image")
        has_vid = request.query.get("has_video")
        has_prm = request.query.get("has_prompt")
        status_filter = request.query.get("status")

        scenes = await asyncio.to_thread(
            get_project_scenes,
            pid,
            has_image=bool(has_img.lower() in ("true", "1")) if has_img is not None else None,
            has_video=bool(has_vid.lower() in ("true", "1")) if has_vid is not None else None,
            has_prompt=bool(has_prm.lower() in ("true", "1")) if has_prm is not None else None,
            status=status_filter,
        )
        return api_success({"items": scenes, "count": len(scenes)})

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}")
    @_api_endpoint
    async def api_get_scene_detail(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        scene = await asyncio.to_thread(get_scene_detail, pid, sid)
        return api_success(scene)

    # 4. Project and Settings Mutations
    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects")
    @_api_endpoint
    async def api_create_project(request: web.Request):
        payload = await request.json() if request.can_read_body else {}
        res = await asyncio.to_thread(create_project, payload.get("name", ""), payload.get("template_from"))
        return api_success(res, revision=res.get("revision"), status=201)

    @server_instance.routes.delete(f"{_API_V1_PREFIX}/projects/{{pid}}")
    @_api_endpoint
    async def api_delete_project(request: web.Request):
        pid = request.match_info["pid"]
        confirm = request.query.get("confirm", "")
        res = await asyncio.to_thread(delete_project_by_id, pid, confirm)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/duplicate")
    @_api_endpoint
    async def api_duplicate_project(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        res = await asyncio.to_thread(duplicate_project, pid, payload.get("new_name", ""), payload.get("options"))
        return api_success(res, revision=res.get("revision"), status=201)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/validate")
    @_api_endpoint
    async def api_validate_project(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(validate_project, pid)
        return api_success(res)

    @server_instance.routes.patch(f"{_API_V1_PREFIX}/projects/{{pid}}/settings")
    @_api_endpoint
    async def api_patch_project_settings(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(patch_project_settings, pid, payload, if_match_revision=if_match)
        return api_success(res.get("settings"), revision=res.get("revision"))

    @server_instance.routes.get(f"{_API_V1_PREFIX}/settings/minimax-h3/schema")
    @_api_endpoint
    async def api_minimax_h3_settings_schema(request: web.Request):
        """Every MiniMax H3 setting an agent can patch under the `minimax_h3` group."""
        return api_success(minimax_h3_settings_schema())

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/settings/preflight")
    @_api_endpoint
    async def api_preflight_settings(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(preflight_project_settings, pid)
        return api_success(res)

    # 5. Scene Mutations
    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes")
    @_api_endpoint
    async def api_create_scene(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            create_scene,
            pid,
            position=payload.get("position", "append"),
            ref_scene_id=payload.get("ref_scene_id"),
            duration=float(payload.get("duration", 4.0)),
            label=payload.get("label", ""),
            t2i_prompt=payload.get("t2i_prompt", ""),
            i2v_prompt=payload.get("i2v_prompt", ""),
            notes=payload.get("notes", ""),
            if_match_revision=if_match,
        )
        return api_success(res.get("scene"), revision=res.get("revision"), renamed=res.get("renamed"), status=201)

    @server_instance.routes.delete(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}")
    @_api_endpoint
    async def api_delete_scene(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        ripple = request.query.get("ripple", "true").lower() in ("true", "1")
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(delete_scene, pid, sid, ripple=ripple, if_match_revision=if_match)
        return api_success(res, revision=res.get("revision"), renamed=res.get("renamed"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/split")
    @_api_endpoint
    async def api_split_scene(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            split_scene,
            pid,
            sid,
            at_time=payload.get("at_time"),
            clear_right_media=bool(payload.get("clear_right_media", True)),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"), renamed=res.get("renamed"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/merge")
    @_api_endpoint
    async def api_merge_scenes(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            merge_scenes,
            pid,
            sid,
            with_direction=payload.get("with_direction", "next"),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"), renamed=res.get("renamed"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/move")
    @_api_endpoint
    async def api_move_scene(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            move_scene,
            pid,
            sid,
            start_time=float(payload.get("start_time", 0.0)),
            ripple=bool(payload.get("ripple", False)),
            if_match_revision=if_match,
        )
        return api_success(res.get("scene"), revision=res.get("revision"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/resize")
    @_api_endpoint
    async def api_resize_scene(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            resize_scene,
            pid,
            sid,
            duration=float(payload["duration"]) if "duration" in payload else None,
            end_time=float(payload["end_time"]) if "end_time" in payload else None,
            ripple=bool(payload.get("ripple", False)),
            if_match_revision=if_match,
        )
        return api_success(res.get("scene"), revision=res.get("revision"))

    @server_instance.routes.patch(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}")
    @_api_endpoint
    async def api_patch_scene(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(patch_scene, pid, sid, payload, if_match_revision=if_match)
        return api_success(res.get("scene"), revision=res.get("revision"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/bulk")
    @_api_endpoint
    async def api_bulk_scene_operations(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            bulk_scene_operations,
            pid,
            payload.get("operations", []),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"))

    # 6. Timeline Batch Operations
    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/references/{{kind}}/{{rid}}/describe")
    @_api_endpoint
    async def api_reference_describe(request: web.Request):
        """Describe a subject or location image with the project's LLM (Gemma Describe). Runs as a job."""
        pid, kind, rid = request.match_info["pid"], request.match_info["kind"], request.match_info["rid"]
        payload = await request.json() if request.can_read_body else {}
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        params = {**payload, "kind": kind, "ref_id": rid}
        job = get_job_manager().submit_job("reference.describe", project_id=pid, params=params, is_gpu=is_llm_runner_gpu(payload, folder))
        return api_success({"job_id": job.id, "status": job.status, "job": job.to_dict()}, status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/references/locations/extract")
    @_api_endpoint
    async def api_reference_extract_locations(request: web.Request):
        """Ask the project's LLM for filming locations (LM Extract) and add them to the Reference Builder. Runs as a job."""
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        job = get_job_manager().submit_job("reference.extract_locations", project_id=pid, params=dict(payload), is_gpu=is_llm_runner_gpu(payload, folder))
        return api_success({"job_id": job.id, "status": job.status, "job": job.to_dict()}, status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/references/assign-scenes")
    @_api_endpoint
    async def api_reference_assign_scenes(request: web.Request):
        """Assign saved characters and locations to scenes with a pattern (Assign Scenes). Use dry_run to preview."""
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        res = await asyncio.to_thread(assign_scenes, pid, payload)
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/story/settings")
    @_api_endpoint
    async def api_story_settings(request: web.Request):
        """Save Storyboard scene defaults and the story idea. Body: {"defaults": {...}, "story": {...}}."""
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(set_story_settings, pid, payload, if_match_revision=if_match)
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/story/{{step}}")
    @_api_endpoint
    async def api_story_step(request: web.Request):
        """Create the story arc, story brief or scene beats with the project's LLM. Runs as a job."""
        pid, step = request.match_info["pid"], request.match_info["step"]
        job_types = {"arc": "storyboard.story_arc", "brief": "storyboard.story_brief", "beats": "storyboard.scene_beats"}
        if step not in job_types:
            raise ValidationError("step must be one of: arc, brief, beats.")
        payload = await request.json() if request.can_read_body else {}
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        job = get_job_manager().submit_job(job_types[step], project_id=pid, params=dict(payload), is_gpu=is_llm_runner_gpu(payload, folder))
        return api_success({"job_id": job.id, "status": job.status, "job": job.to_dict()}, status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/minimax-prompts")
    @_api_endpoint
    async def api_minimax_prompts(request: web.Request):
        """Write MiniMax H3 reference-to-video prompts for the scenes that have none, with the project's LLM. Runs as a job."""
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        job = get_job_manager().submit_job("minimax.prompts", project_id=pid, params=dict(payload), is_gpu=is_llm_runner_gpu(payload, folder))
        return api_success({"job_id": job.id, "status": job.status, "job": job.to_dict()}, status=202)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/llm/active")
    @_api_endpoint
    async def api_llm_active(request: web.Request):
        """The LLM the API would use for this project right now. For LM Studio, the loaded model; it is never changed."""
        pid = request.query.get("project_id", "")
        if pid:
            _folder, session = await asyncio.to_thread(_get_active_session_and_folder, pid)
        else:
            # No project yet: report what a new project would use (the saved model defaults).
            from ..builder.project import _load_model_defaults

            loaded = await asyncio.to_thread(_load_model_defaults)
            session = dict(loaded.get("defaults") or {})
        info = await asyncio.to_thread(describe_active_llm, llm_payload_from_session(session))
        return api_success(info)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/timeline/enforce-length")
    @_api_endpoint
    async def api_timeline_enforce_length(request: web.Request):
        """Merge scenes shorter than min_scene_seconds and cut scenes longer than max_scene_seconds."""
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        try:
            min_seconds = float(payload.get("min_scene_seconds"))
            max_seconds = float(payload.get("max_scene_seconds"))
        except (TypeError, ValueError):
            raise ValidationError("min_scene_seconds and max_scene_seconds are required numbers.")
        if min_seconds <= 0 or max_seconds < min_seconds:
            raise ValidationError("min_scene_seconds must be positive and not larger than max_scene_seconds.")
        res = await asyncio.to_thread(
            enforce_scene_lengths_on_project, pid, min_seconds, max_seconds, bool(payload.get("dry_run", False))
        )
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/lyrics/align")
    @_api_endpoint
    async def api_lyrics_align(request: web.Request):
        """Time the project lyrics against the song (ComfyUI timestamp workflow). Runs as a job."""
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        job = get_job_manager().submit_job(job_type="lyrics.align", project_id=pid, params=dict(payload), is_gpu=True)
        return api_success({"job_id": job.id, "status": job.status, "job": job.to_dict()}, status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/timeline/from-lines")
    @_api_endpoint
    async def api_timeline_from_lines(request: web.Request):
        """Create the timeline scenes from the lyrics, like Line Mapping, with a min/max scene length. Runs as a job."""
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        job = get_job_manager().submit_job(job_type="timeline.from_lines", project_id=pid, params=dict(payload), is_gpu=True)
        return api_success({"job_id": job.id, "status": job.status, "job": job.to_dict()}, status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/timeline/close-gaps")
    @_api_endpoint
    async def api_timeline_close_gaps(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(timeline_close_gaps, pid)
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/timeline/snap")
    @_api_endpoint
    async def api_timeline_snap(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        res = await asyncio.to_thread(
            timeline_snap,
            pid,
            scope=payload.get("scope", "edge"),
            scene_id=payload.get("scene_id"),
            edge=payload.get("edge", "start"),
        )
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/timeline/bulk")
    @_api_endpoint
    async def api_timeline_bulk(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        res = await asyncio.to_thread(
            timeline_bulk,
            pid,
            text=payload.get("text", ""),
            mode=payload.get("mode", "durations"),
            action=payload.get("action", "replace"),
            append_start=float(payload.get("append_start", 0.0)),
            clear_media=bool(payload.get("clear_media", False)),
        )
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/timeline/calibrate")
    @_api_endpoint
    async def api_calibrate_beats(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            calibrate_beats,
            pid,
            offset_seconds=float(payload.get("offset_seconds", 0.0)),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"))

    # 7. References CRUD (Section 6.5)
    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/references")
    @_api_endpoint
    async def api_get_references(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(get_project_references, pid)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/references/subjects/{{rid}}")
    @_api_endpoint
    async def api_upsert_subject(request: web.Request):
        pid = request.match_info["pid"]
        rid = request.match_info["rid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(upsert_reference_subject, pid, rid, payload, if_match_revision=if_match)
        return api_success(res.get("subject"), revision=res.get("revision"))

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/references/locations/{{rid}}")
    @_api_endpoint
    async def api_upsert_location(request: web.Request):
        pid = request.match_info["pid"]
        rid = request.match_info["rid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(upsert_reference_location, pid, rid, payload, if_match_revision=if_match)
        return api_success(res.get("location"), revision=res.get("revision"))

    @server_instance.routes.delete(f"{_API_V1_PREFIX}/projects/{{pid}}/references/{{kind}}/{{rid}}")
    @_api_endpoint
    async def api_delete_reference(request: web.Request):
        pid = request.match_info["pid"]
        kind = request.match_info["kind"]
        rid = request.match_info["rid"]
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(delete_reference, pid, kind, rid, if_match_revision=if_match)
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/references/scene-mapping")
    @_api_endpoint
    async def api_get_reference_mapping(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(get_project_references, pid)
        return api_success(res.get("scene_mapping"))

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/references/scene-mapping")
    @_api_endpoint
    async def api_update_reference_mapping(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(update_scene_reference_mapping, pid, payload, if_match_revision=if_match)
        return api_success(res.get("scene_mapping"), revision=res.get("revision"))

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/minimax-references")
    @_api_endpoint
    async def api_get_scene_minimax_references(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        res = await asyncio.to_thread(get_scene_minimax_references, pid, sid)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/minimax-references")
    @_api_endpoint
    async def api_set_scene_minimax_references(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            set_scene_minimax_references, pid, sid, payload.get("keys"), bool(payload.get("automatic")),
            if_match_revision=if_match,
        )
        revision = res.pop("revision", None)
        return api_success(res, revision=revision)

    # 8. Lyrics, Audio, and Beats (Section 6.3)
    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/lyrics")
    @_api_endpoint
    async def api_get_lyrics(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(get_project_lyrics, pid)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/lyrics")
    @_api_endpoint
    async def api_set_lyrics(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            set_project_lyrics,
            pid,
            lyrics_text=payload.get("lyrics_text"),
            srt_text=payload.get("srt_text"),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/audio")
    @_api_endpoint
    async def api_attach_audio(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            attach_project_audio,
            pid,
            audio_path=payload.get("audio_path"),
            audio_data=payload.get("audio_data"),
            audio_name=payload.get("audio_name"),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/audio/waveform")
    @_api_endpoint
    async def api_get_waveform(request: web.Request):
        pid = request.match_info["pid"]
        peaks_count = int(request.query.get("peaks", 1600))
        res = await asyncio.to_thread(get_audio_waveform, pid, target_peaks=peaks_count)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/audio/silent")
    @_api_endpoint
    async def api_silent_audio(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            create_project_silent_audio,
            pid,
            duration=float(payload.get("duration", 10.0)),
            scope=payload.get("scope", "project"),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/audio/beats")
    @_api_endpoint
    async def api_get_beats(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(get_audio_beats, pid)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/audio/beats")
    @_api_endpoint
    async def api_set_beats(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            set_audio_beats,
            pid,
            beats=payload.get("beats", []),
            tempo_bpm=float(payload["tempo_bpm"]) if "tempo_bpm" in payload else None,
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"))

    # 9. Prompts: Context, Assembly, and Validation (Section 18.7)
    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/context")
    @_api_endpoint
    async def api_get_prompt_context(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        kind = request.query.get("kind", "minimax")
        res = await asyncio.to_thread(get_prompt_context, pid, sid, kind=kind)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/minimax/assemble")
    @_api_endpoint
    async def api_assemble_minimax_prompt(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            assemble_minimax_prompt_endpoint,
            pid,
            sid,
            shots=payload.get("shots", []),
            mode=payload.get("mode"),
            save=bool(payload.get("save", False)),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/minimax/validate")
    @_api_endpoint
    async def api_validate_minimax_prompt(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        res = await asyncio.to_thread(
            validate_minimax_prompt_endpoint,
            pid,
            sid,
            prompt=payload.get("prompt", ""),
            mode=payload.get("mode"),
        )
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/{{field}}")
    @_api_endpoint
    async def api_set_prompt_field(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        field = request.match_info["field"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(
            set_scene_prompt_field_endpoint,
            pid,
            sid,
            field,
            prompt=payload.get("prompt", ""),
            origin=payload.get("origin", "agent"),
            if_match_revision=if_match,
        )
        return api_success(res, revision=res.get("revision"))

    # 10. Project Story and Export (Section 6.2, 6.5)
    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{pid}}/story")
    @_api_endpoint
    async def api_get_story(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(get_project_story, pid)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{pid}}/story")
    @_api_endpoint
    async def api_put_story(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        if_match = int(request.headers.get("If-Match")) if request.headers.get("If-Match", "").isdigit() else None
        res = await asyncio.to_thread(put_project_story, pid, payload, if_match_revision=if_match)
        return api_success(res.get("story"), revision=res.get("revision"))

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/export")
    @_api_endpoint
    async def api_export_project(request: web.Request):
        pid = request.match_info["pid"]
        res = await asyncio.to_thread(export_project, pid)
        return api_success(res)

    # 11. Jobs and Event Streaming (Section 5.2)
    @server_instance.routes.get(f"{_API_V1_PREFIX}/jobs")
    @_api_endpoint
    async def api_list_jobs(request: web.Request):
        project_id = request.query.get("project_id")
        status = request.query.get("status")
        job_type = request.query.get("type")
        manager = get_job_manager()
        jobs = await asyncio.to_thread(manager.list_jobs, project_id=project_id, status=status, job_type=job_type)
        return api_success([j.to_dict() for j in jobs])

    @server_instance.routes.get(f"{_API_V1_PREFIX}/jobs/{{id}}")
    @_api_endpoint
    async def api_get_job(request: web.Request):
        job_id = request.match_info["id"]
        manager = get_job_manager()
        job = await asyncio.to_thread(manager.get_job, job_id)
        return api_success(job.to_dict())

    @server_instance.routes.get(f"{_API_V1_PREFIX}/jobs/{{id}}/log")
    @_api_endpoint
    async def api_get_job_log(request: web.Request):
        job_id = request.match_info["id"]
        since_str = request.query.get("since", "0")
        since = int(since_str) if since_str.isdigit() else 0
        manager = get_job_manager()
        logs = await asyncio.to_thread(manager.get_logs, job_id, since=since)
        return api_success({"job_id": job_id, "logs": logs})

    @server_instance.routes.post(f"{_API_V1_PREFIX}/jobs/{{id}}/cancel")
    @_api_endpoint
    async def api_cancel_job(request: web.Request):
        job_id = request.match_info["id"]
        manager = get_job_manager()
        job = await asyncio.to_thread(manager.cancel_job, job_id)
        return api_success(job.to_dict())

    @server_instance.routes.post(f"{_API_V1_PREFIX}/jobs/{{id}}/retry")
    @_api_endpoint
    async def api_retry_job(request: web.Request):
        job_id = request.match_info["id"]
        payload = await request.json() if request.can_read_body else {}
        resume = bool(payload.get("resume", False))
        manager = get_job_manager()
        new_job = await asyncio.to_thread(manager.retry_job, job_id, resume=resume)
        return api_success(new_job.to_dict(), status=202)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/queue")
    @_api_endpoint
    async def api_get_queue(request: web.Request):
        manager = get_job_manager()
        summary = await asyncio.to_thread(manager.get_queue_summary)
        return api_success(summary)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/events")
    @_api_endpoint
    async def api_events(request: web.Request):
        project_id = request.query.get("project_id")
        broadcaster = get_event_broadcaster()
        queue = broadcaster.subscribe(project_id)

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        await response.write(b": connected\n\n")

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    evt_name = event.get("event", "message")
                    data_str = json.dumps(event.get("data", {}))
                    chunk = f"event: {evt_name}\ndata: {data_str}\n\n".encode("utf-8")
                    await response.write(chunk)
                except asyncio.TimeoutError:
                    await response.write(b": ping\n\n")
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            broadcaster.unsubscribe(queue)

        return response

    # 12. LLM Prompt Jobs and Instruction Presets (Section 6.7)
    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/prompts/concepts")
    @_api_endpoint
    async def api_prompt_concepts(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        is_gpu = is_llm_runner_gpu(payload, folder)
        manager = get_job_manager()
        job = manager.submit_job("llm.concepts", project_id=pid, params=payload, is_gpu=is_gpu)
        return api_success(job.to_dict(), status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/prompts/motion-notes")
    @_api_endpoint
    async def api_prompt_motion_notes(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        is_gpu = is_llm_runner_gpu(payload, folder)
        manager = get_job_manager()
        job = manager.submit_job("llm.motion_notes", project_id=pid, params=payload, is_gpu=is_gpu)
        return api_success(job.to_dict(), status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/image")
    @_api_endpoint
    async def api_prompt_scene_image(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        payload["scene_id"] = sid
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        is_gpu = is_llm_runner_gpu(payload, folder)
        manager = get_job_manager()
        job = manager.submit_job("llm.scene_image_prompt", project_id=pid, params=payload, is_gpu=is_gpu)
        return api_success(job.to_dict(), status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/video")
    @_api_endpoint
    async def api_prompt_scene_video(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        payload["scene_id"] = sid
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        is_gpu = is_llm_runner_gpu(payload, folder)
        manager = get_job_manager()
        job = manager.submit_job("llm.scene_video_prompt", project_id=pid, params=payload, is_gpu=is_gpu)
        return api_success(job.to_dict(), status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/video-chained")
    @_api_endpoint
    async def api_prompt_scene_video_chained(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        payload["scene_id"] = sid
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        is_gpu = is_llm_runner_gpu(payload, folder)
        manager = get_job_manager()
        job = manager.submit_job("llm.scene_chained_video_prompt", project_id=pid, params=payload, is_gpu=is_gpu)
        return api_success(job.to_dict(), status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/enhance")
    @_api_endpoint
    async def api_prompt_scene_enhance(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        payload["scene_id"] = sid
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        is_gpu = is_llm_runner_gpu(payload, folder)
        manager = get_job_manager()
        job = manager.submit_job("llm.enhance_prompt", project_id=pid, params=payload, is_gpu=is_gpu)
        return api_success(job.to_dict(), status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/scenes/{{sid}}/prompts/edit")
    @_api_endpoint
    async def api_prompt_scene_edit(request: web.Request):
        pid = request.match_info["pid"]
        sid = request.match_info["sid"]
        payload = await request.json() if request.can_read_body else {}
        payload["scene_id"] = sid
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        is_gpu = is_llm_runner_gpu(payload, folder)
        manager = get_job_manager()
        job = manager.submit_job("llm.edit_prompt", project_id=pid, params=payload, is_gpu=is_gpu)
        return api_success(job.to_dict(), status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{pid}}/prompts/batch")
    @_api_endpoint
    async def api_prompt_batch(request: web.Request):
        pid = request.match_info["pid"]
        payload = await request.json() if request.can_read_body else {}
        folder = await asyncio.to_thread(resolve_project_folder, pid)
        is_gpu = is_llm_runner_gpu(payload, folder)
        manager = get_job_manager()
        job = manager.submit_job("llm.batch_prompts", project_id=pid, params=payload, is_gpu=is_gpu)
        return api_success(job.to_dict(), status=202)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/llm/test")
    @_api_endpoint
    async def api_llm_test(request: web.Request):
        payload = await request.json() if request.can_read_body else {}
        provider = str(payload.get("provider") or payload.get("text_runner") or "").strip().lower()
        if provider in ("own_server", "own"):
            res = await asyncio.to_thread(_test_own_server, payload)
        else:
            res = await asyncio.to_thread(_test_llm_api, payload)
        return api_success(res)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/llm/models")
    @_api_endpoint
    async def api_llm_models(request: web.Request):
        provider = str(request.query.get("provider", "lm_studio")).strip().lower()
        query_dict = dict(request.query)
        if provider in ("own_server", "own"):
            models = await asyncio.to_thread(_list_own_server_models, query_dict)
        else:
            models = await asyncio.to_thread(_list_lm_studio_models, query_dict)
        return api_success(models)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/llm/unload")
    @_api_endpoint
    async def api_llm_unload(request: web.Request):
        res = await asyncio.to_thread(_clear_vrgdg_llm_caches, clear_cuda_cache=True, clear_hf_pipeline_cache=True)
        return api_success(res or {"unloaded": True})

    # Instruction presets (Section 6.7)
    @server_instance.routes.get(f"{_API_V1_PREFIX}/instructions")
    @_api_endpoint
    async def api_list_instructions(request: web.Request):
        query_dict = dict(request.query)
        if "key" not in query_dict:
            query_dict["key"] = "zimage_t2i"
        res = await asyncio.to_thread(_list_builder_instruction_presets, query_dict)
        return api_success(res)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/instructions/{{key}}")
    @_api_endpoint
    async def api_get_instruction(request: web.Request):
        key = request.match_info["key"]
        payload = {"key": key, **dict(request.query)}
        if "project_id" in payload:
            folder = await asyncio.to_thread(resolve_project_folder, payload["project_id"])
            payload["project_folder"] = folder
        res = await asyncio.to_thread(_get_builder_instruction, payload)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/instructions/{{key}}")
    @_api_endpoint
    async def api_save_instruction(request: web.Request):
        key = request.match_info["key"]
        payload = await request.json() if request.can_read_body else {}
        payload["key"] = key
        if "project_id" in payload:
            folder = await asyncio.to_thread(resolve_project_folder, payload["project_id"])
            payload["project_folder"] = folder
        res = await asyncio.to_thread(_save_builder_instruction, payload)
        return api_success(res)

    @server_instance.routes.delete(f"{_API_V1_PREFIX}/instructions/{{key}}/override")
    @_api_endpoint
    async def api_reset_instruction(request: web.Request):
        key = request.match_info["key"]
        payload = await request.json() if request.can_read_body else {}
        payload["key"] = key
        if "project_id" in payload:
            folder = await asyncio.to_thread(resolve_project_folder, payload["project_id"])
            payload["project_folder"] = folder
        res = await asyncio.to_thread(_reset_builder_instruction, payload)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/instructions/presets/{{name}}")
    @_api_endpoint
    async def api_save_instruction_preset(request: web.Request):
        name = request.match_info["name"]
        payload = await request.json() if request.can_read_body else {}
        payload["name"] = name
        if "key" not in payload:
            payload["key"] = "zimage_t2i"
        res = await asyncio.to_thread(_save_builder_instruction_preset, payload)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/instructions/presets/{{name}}/load")
    @_api_endpoint
    async def api_load_instruction_preset(request: web.Request):
        name = request.match_info["name"]
        payload = await request.json() if request.can_read_body else {}
        payload["name"] = name
        if "key" not in payload:
            payload["key"] = "zimage_t2i"
        res = await asyncio.to_thread(_load_builder_instruction_preset, payload)
        return api_success(res)

    # 8. Image Generation and Lifecycle (Section 6.8, Section 24.1)
    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/image/generate")
    @_api_endpoint
    async def api_generate_scene_image(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        params = {"scene_id": scene_id, **payload}
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="image.generate",
            project_id=project_id,
            params=params,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/images/generate")
    @_api_endpoint
    async def api_generate_batch_images(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="images.generate_batch",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/image/approve")
    @_api_endpoint
    async def api_approve_scene_image(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        image_path = payload.get("image_path")
        res = await asyncio.to_thread(approve_scene_image, project_id, scene_id, image_path=image_path)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/image/revert")
    @_api_endpoint
    async def api_revert_scene_image(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        delta = int(payload.get("delta", -1))
        index = payload.get("index")
        if index is not None:
            index = int(index)
        res = await asyncio.to_thread(revert_scene_image, project_id, scene_id, delta=delta, index=index)
        return api_success(res)

    @server_instance.routes.delete(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/image")
    @_api_endpoint
    async def api_delete_scene_image(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        res = await asyncio.to_thread(delete_scene_image, project_id, scene_id)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/image")
    @_api_endpoint
    async def api_put_scene_image(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        image_data = payload.get("image_data")
        source_path = payload.get("source_path")
        res = await asyncio.to_thread(
            save_scene_image_custom,
            project_id,
            scene_id,
            image_data=image_data,
            source_path=source_path,
        )
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/image/from-video-frame")
    @_api_endpoint
    async def api_extract_scene_image_from_video_frame(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        source_video_path = payload.get("source_video_path")
        res = await asyncio.to_thread(
            extract_frame_from_video_to_image,
            project_id,
            scene_id,
            source_video_path=source_video_path,
        )
        return api_success(res)

    # 9. Video Generation and Lifecycle (Section 6.9, Section 24.2)
    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/video/render")
    @_api_endpoint
    async def api_render_scene_video(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        params = {"scene_id": scene_id, **payload}
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="video.render",
            project_id=project_id,
            params=params,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/video/recover")
    @_api_endpoint
    async def api_recover_scene_video(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        source_path = payload.get("source_path")
        res = await asyncio.to_thread(recover_scene_video, project_id, scene_id, source_path=source_path)
        return api_success(res)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/video/takes")
    @_api_endpoint
    async def api_scene_video_takes(request: web.Request):
        res = await asyncio.to_thread(list_scene_takes, request.match_info["project_id"], request.match_info["scene_id"])
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/video/trim")
    @_api_endpoint
    async def api_trim_scene_video(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        params = {"scene_id": scene_id, **payload}
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="video.trim",
            project_id=project_id,
            params=params,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/video/match-start-color")
    @_api_endpoint
    async def api_match_scene_video_start_color(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        params = {"scene_id": scene_id, **payload}
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="video.match_start_color",
            project_id=project_id,
            params=params,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/video/select")
    @_api_endpoint
    async def api_select_scene_video(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        source_path = payload.get("source_path")
        if not source_path:
            raise ValidationError("source_path is required.")
        res = await asyncio.to_thread(select_scene_video, project_id, scene_id, source_path=source_path)
        return api_success(res)

    @server_instance.routes.delete(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/video")
    @_api_endpoint
    async def api_delete_scene_video(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        res = await asyncio.to_thread(delete_scene_video, project_id, scene_id)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/video/scan")
    @_api_endpoint
    async def api_scan_scene_videos(request: web.Request):
        project_id = request.match_info["project_id"]
        res = await asyncio.to_thread(scan_project_scene_videos, project_id)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/video/graph")
    @_api_endpoint
    async def api_video_graph_dry_run(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        folder = await asyncio.to_thread(resolve_project_folder, project_id)
        p = dict(payload)
        p["project_folder"] = folder
        mode = str(p.get("mode") or "i2v").strip().lower()
        res = await asyncio.to_thread(build_video_graph_for_mode, mode, p)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/video/render")
    @_api_endpoint
    async def api_render_batch_videos(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="videos.render_batch",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    # 10. Assembly, Stitching, and Export (Section 6.12)
    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/stitch")
    @_api_endpoint
    async def api_stitch_project_video(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            raise ValidationError("The stitch body must be a JSON object.")
        params = {
            "scene_ids": payload.get("scene_ids"),
            "output_prefix": payload.get("output_prefix"),
            "audio": payload.get("audio"),
            "audio_path": payload.get("audio_path"),
            "overlays": payload.get("overlays"),
        }
        unknown = sorted(str(key) for key in payload if key not in params)
        if unknown:
            raise ValidationError(
                f"Unknown stitch field(s): {', '.join(unknown)}. Supported: {', '.join(params)}.",
                details={"unknown": unknown, "supported": list(params)},
            )
        params = {key: value for key, value in params.items() if value is not None}
        # Check the scenes, audio mode and audio file now, so a bad request is a 400 instead of a failed job.
        validate_stitch_request(project_id, params)
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="video.stitch",
            project_id=project_id,
            params=params,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/slideshow")
    @_api_endpoint
    async def api_render_project_slideshow(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="video.slideshow",
            project_id=project_id,
            params=payload,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{project_id}}/finals")
    @_api_endpoint
    async def api_list_project_finals(request: web.Request):
        project_id = request.match_info["project_id"]
        finals = await asyncio.to_thread(list_project_final_videos, project_id)
        return api_success(finals)

    # ==============================================================================
    # 13. MiniMax H3 Latents & Continuity (Section 6.10, Invariant 4)
    # ==============================================================================

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{project_id}}/latents")
    @_api_endpoint
    async def api_get_project_latents(request: web.Request):
        project_id = request.match_info["project_id"]
        result = await asyncio.to_thread(get_project_latents_status, project_id)
        return api_success(result)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{project_id}}/latents/dirty")
    @_api_endpoint
    async def api_get_dirty_latents(request: web.Request):
        project_id = request.match_info["project_id"]
        result = await asyncio.to_thread(get_dirty_latents, project_id)
        return api_success(result)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/latent")
    @_api_endpoint
    async def api_get_scene_latent(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        result = await asyncio.to_thread(get_scene_latent_status, project_id, scene_id)
        return api_success(result)

    @server_instance.routes.delete(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/latent")
    @_api_endpoint
    async def api_delete_scene_latent(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        all_latents = request.query.get("all", "false").lower() in ("true", "1")
        reindex = request.query.get("reindex", "false").lower() in ("true", "1")
        if request.can_read_body:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    if "all" in body:
                        all_latents = bool(body["all"])
                    if "reindex" in body:
                        reindex = bool(body["reindex"])
            except Exception:
                pass
        result = await asyncio.to_thread(
            delete_scene_latent,
            project_id,
            scene_id,
            all_latents=all_latents,
            reindex=reindex,
        )
        return api_success(result)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/latents/rebuild")
    @_api_endpoint
    async def api_rebuild_latents_job(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="latents.rebuild",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/video/minimax-stage-recover")
    @_api_endpoint
    async def api_minimax_stage_recover(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        result = await asyncio.to_thread(minimax_stage_recover, project_id, scene_id, payload)
        return api_success(result)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/minimax/cleanup")
    @_api_endpoint
    async def api_minimax_cleanup(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        result = await asyncio.to_thread(cleanup_minimax_output, project_id, payload)
        return api_success(result)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{project_id}}/minimax/index")
    @_api_endpoint
    async def api_minimax_index(request: web.Request):
        project_id = request.match_info["project_id"]
        result = await asyncio.to_thread(get_minimax_project_index, project_id)
        return api_success(result)

    # ==============================================================================
    # 14. Post-Processing & Face Fix (Section 6.11, 23.1, 23.2)
    # ==============================================================================

    @server_instance.routes.get(f"{_API_V1_PREFIX}/post/luts")
    @_api_endpoint
    async def api_list_luts(request: web.Request):
        res = await asyncio.to_thread(list_luts_service)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/post/luts/upload")
    @_api_endpoint
    async def api_upload_lut(request: web.Request):
        filename = request.query.get("filename", "")
        content = b""
        if request.content_type.startswith("multipart/"):
            reader = await request.multipart()
            field = await reader.next()
            if field is not None:
                filename = filename or field.filename or "custom.cube"
                content = await field.read()
        else:
            if request.can_read_body:
                try:
                    body = await request.json()
                    if isinstance(body, dict):
                        filename = filename or str(body.get("filename") or "custom.cube")
                        data_raw = body.get("data") or body.get("content") or ""
                        import base64
                        content = base64.b64decode(data_raw) if data_raw else b""
                except Exception:
                    content = await request.read()
            else:
                content = await request.read()
        if not filename:
            filename = "custom.cube"
        res = await asyncio.to_thread(upload_lut_service, filename, content)
        return api_success(res)

    @server_instance.routes.delete(f"{_API_V1_PREFIX}/post/luts/previews/{{id}}")
    @_api_endpoint
    async def api_delete_preview(request: web.Request):
        preview_id = request.match_info["id"]
        res = await asyncio.to_thread(delete_preview_service, preview_id)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/post/lut")
    @_api_endpoint
    async def api_apply_scene_lut(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        payload["scene_id"] = scene_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="post.lut",
            project_id=project_id,
            params=payload,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/post/lut/preview")
    @_api_endpoint
    async def api_preview_scene_lut(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        res = await asyncio.to_thread(preview_scene_lut, project_id, scene_id, payload)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/post/film-grain")
    @_api_endpoint
    async def api_apply_scene_grain(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        payload["scene_id"] = scene_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="post.film_grain",
            project_id=project_id,
            params=payload,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/post/film-grain/preview")
    @_api_endpoint
    async def api_preview_scene_grain(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        res = await asyncio.to_thread(preview_scene_grain, project_id, scene_id, payload)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/post/adjust")
    @_api_endpoint
    async def api_apply_scene_adjust(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        payload["scene_id"] = scene_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="post.adjust",
            project_id=project_id,
            params=payload,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/post/adjust/preview")
    @_api_endpoint
    async def api_preview_scene_adjust(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        res = await asyncio.to_thread(preview_scene_adjust, project_id, scene_id, payload)
        return api_success(res)

    @server_instance.routes.get(f"{_API_V1_PREFIX}/post/adjust/presets")
    @_api_endpoint
    async def api_list_adjust_presets(request: web.Request):
        res = await asyncio.to_thread(get_adjust_presets)
        return api_success(res)

    @server_instance.routes.put(f"{_API_V1_PREFIX}/post/adjust/presets/{{name}}")
    @_api_endpoint
    async def api_put_adjust_preset(request: web.Request):
        name = request.match_info["name"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        settings = payload.get("settings", payload)
        res = await asyncio.to_thread(put_adjust_preset, name, settings)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/post/apply-all")
    @_api_endpoint
    async def api_post_apply_all(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="post.apply_all",
            project_id=project_id,
            params=payload,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/face-fix/estimate")
    @_api_endpoint
    async def api_estimate_face_fix(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        res = await asyncio.to_thread(estimate_scene_face_fix_anchors, project_id, scene_id, payload)
        return api_success(res)

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/face-fix/prepare")
    @_api_endpoint
    async def api_prepare_face_fix(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        payload["scene_id"] = scene_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="face_fix.prepare",
            project_id=project_id,
            params=payload,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/face-fix/anchors/{{n}}/enhance")
    @_api_endpoint
    async def api_enhance_face_fix_anchor(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        n = request.match_info["n"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        payload["scene_id"] = scene_id
        payload["order"] = int(n)
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="face_fix.enhance_anchor",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/face-fix/runs/{{n}}/ltx")
    @_api_endpoint
    async def api_run_face_fix_ltx(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        n = request.match_info["n"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        payload["scene_id"] = scene_id
        payload["run_index"] = int(n)
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="face_fix.ltx_run",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/face-fix/finalize")
    @_api_endpoint
    async def api_finalize_face_fix(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        payload["scene_id"] = scene_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="face_fix.finalize",
            project_id=project_id,
            params=payload,
            is_gpu=False,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/scenes/{{scene_id}}/face-fix/auto")
    @_api_endpoint
    async def api_auto_face_fix(request: web.Request):
        project_id = request.match_info["project_id"]
        scene_id = request.match_info["scene_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        payload["scene_id"] = scene_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="face_fix.auto",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    # ==============================================================================
    # 15. Pipelines & Dry-Run Planning (Section 6.13)
    # ==============================================================================

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/pipelines/build-full-video")
    @_api_endpoint
    async def api_pipeline_build_full_video(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="pipeline.build_full_video",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/projects/{{project_id}}/pipelines/build-flf")
    @_api_endpoint
    async def api_pipeline_build_flf(request: web.Request):
        project_id = request.match_info["project_id"]
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["project_id"] = project_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="pipeline.build_flf",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.post(f"{_API_V1_PREFIX}/pipelines/from-song")
    @_api_endpoint
    async def api_pipeline_from_song(request: web.Request):
        """Song to final video: create the project if needed, prepare scenes, then build the full video."""
        payload = await request.json() if request.can_read_body else {}
        if not isinstance(payload, dict):
            payload = {}
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id:
            project_name = str(payload.get("project_name") or "").strip()
            if not project_name:
                raise ValidationError("Provide project_id for an existing project, or project_name to create one.")
            created = await asyncio.to_thread(create_project, project_name)
            project_id = created["project_id"]
        payload["project_id"] = project_id
        manager = get_job_manager()
        job = manager.submit_job(
            job_type="pipeline.from_song",
            project_id=project_id,
            params=payload,
            is_gpu=True,
        )
        return api_success(
            {"project_id": project_id, "job_id": job.id, "status": job.status, "job": job.to_dict()},
            status=202,
        )

    @server_instance.routes.get(f"{_API_V1_PREFIX}/projects/{{project_id}}/pipelines/plan")
    @_api_endpoint
    async def api_pipeline_plan(request: web.Request):
        project_id = request.match_info["project_id"]
        params = dict(request.query)
        res = await asyncio.to_thread(get_pipeline_plan, project_id, params)
        return api_success(res)

    _VRGDG_AGENT_API_ROUTES_REGISTERED = True
