"""HTTP routes for the workflow runner."""

import subprocess
from aiohttp import web
from server import PromptServer
from ..core.model_paths import load_custom_model_root, register_custom_model_root, save_custom_model_root

from .models import _folder_choices, _lora_choices, _ltx_video_model_choices
from .api_graph import _ensure_placeholder_load_image
from .image_workflows import _build_ernie_image_api_prompt, _build_flux_klein_api_prompt, _build_krea2_2pass_api_prompt, _build_krea2_api_prompt, _build_nb_image_api_prompt, _build_z_upscale_enhance_prompt, _build_zimage_api_prompt
from .ltx_workflows import _build_flf_api_prompt, _build_i2v_api_prompt, _build_id_lora_api_prompt, _build_ingredients_api_prompt, _build_rtv_api_prompt, _build_t2v_api_prompt
from .minimax_inputs import _cleanup_minimax_h3_output_folder, _prepare_scene_audio_clip
from .minimax_workflows import _build_minimax_h3_2pass_api_prompt, _build_minimax_h3_3pass_api_prompt, _build_minimax_h3_advanced_2pass_api_prompt, _build_minimax_h3_api_prompt, _save_minimax_h3_advanced_2pass_debug_workflow
from .utility_workflows import _build_clear_memory_prompt, _build_timestamped_transcribe_api_prompt, _build_transcribe_api_prompt
from .video_files import _apply_scene_start_color_match, _collect_minimax_h3_stage_backup, _collect_scene_video, _find_minimax_h3_stage_outputs, _find_scene_video_output, _render_image_slideshow, _save_generated_image, _stitch_scene_videos, _trim_scene_video


_VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED = False


def _ensure_workflow_runner_routes():
    global _VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED
    if _VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    try:
        _ensure_placeholder_load_image()
    except Exception as exc:
        print(f"[VRGDG] Could not prepare placeholder image for LoadImage validation: {exc}")

    @server_instance.routes.get("/vrgdg/workflow_runner/lora_list")
    async def vrgdg_workflow_runner_lora_list(request):
        return web.json_response({"ok": True, "loras": _lora_choices()})

    @server_instance.routes.get("/vrgdg/workflow_runner/i2v_choices")
    async def vrgdg_workflow_runner_i2v_choices(request):
        video_gguf_unets, video_diffusion_models = _ltx_video_model_choices()
        return web.json_response({
            "ok": True,
            "unets": _folder_choices(("unet", "diffusion_models")),
            "video_gguf_unets": video_gguf_unets,
            "video_diffusion_models": video_diffusion_models,
            "vae": _folder_choices("vae"),
            "clip": _folder_choices(("clip", "text_encoders")),
            # LatentUpscaleModelLoader validates against latent_upscale_models,
            # not the ESRGAN/image upscale_models category.
            "upscale_models": _folder_choices("latent_upscale_models"),
        })

    @server_instance.routes.get("/vrgdg/workflow_runner/model_root")
    async def vrgdg_workflow_runner_model_root(request):
        result = load_custom_model_root()
        result["registered"] = register_custom_model_root(result.get("models_root", ""))
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/model_root")
    async def vrgdg_workflow_runner_save_model_root(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = save_custom_model_root(payload.get("models_root", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_zimage_prompt")
    async def vrgdg_workflow_runner_build_zimage_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_zimage_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_krea2_prompt")
    async def vrgdg_workflow_runner_build_krea2_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_krea2_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_krea2_2pass_prompt")
    async def vrgdg_workflow_runner_build_krea2_2pass_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_krea2_2pass_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_ernie_image_prompt")
    async def vrgdg_workflow_runner_build_ernie_image_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_ernie_image_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_i2v_prompt")
    async def vrgdg_workflow_runner_build_i2v_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_i2v_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_t2v_prompt")
    async def vrgdg_workflow_runner_build_t2v_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_t2v_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_prompt")
    async def vrgdg_workflow_runner_build_minimax_h3_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_minimax_h3_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_2pass_prompt")
    async def vrgdg_workflow_runner_build_minimax_h3_2pass_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_minimax_h3_2pass_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_advanced_2pass_prompt")
    async def vrgdg_workflow_runner_build_minimax_h3_advanced_2pass_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_minimax_h3_advanced_2pass_api_prompt(payload)
            result["debug_workflow_path"] = _save_minimax_h3_advanced_2pass_debug_workflow(result, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_3pass_prompt")
    async def vrgdg_workflow_runner_build_minimax_h3_3pass_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_minimax_h3_3pass_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_rtv_prompt")
    async def vrgdg_workflow_runner_build_rtv_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_rtv_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_ingredients_prompt")
    async def vrgdg_workflow_runner_build_ingredients_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_ingredients_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_flf_prompt")
    async def vrgdg_workflow_runner_build_flf_prompt(request):
        try:
            payload = await request.json()
            result = _build_flf_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_id_lora_prompt")
    async def vrgdg_workflow_runner_build_id_lora_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_id_lora_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_flux_klein_prompt")
    async def vrgdg_workflow_runner_build_flux_klein_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_flux_klein_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_nb_image_prompt")
    async def vrgdg_workflow_runner_build_nb_image_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_nb_image_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_z_upscale_enhance_prompt")
    async def vrgdg_workflow_runner_build_z_upscale_enhance_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_z_upscale_enhance_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_clear_memory_prompt")
    async def vrgdg_workflow_runner_build_clear_memory_prompt(request):
        try:
            result = _build_clear_memory_prompt()
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_transcribe_prompt")
    async def vrgdg_workflow_runner_build_transcribe_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_transcribe_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/build_timestamped_transcribe_prompt")
    async def vrgdg_workflow_runner_build_timestamped_transcribe_prompt(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _build_timestamped_transcribe_api_prompt(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/prepare_scene_audio_clip")
    async def vrgdg_workflow_runner_prepare_scene_audio_clip(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _prepare_scene_audio_clip(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/save_image")
    async def vrgdg_workflow_runner_save_image(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _save_generated_image(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/collect_scene_video")
    async def vrgdg_workflow_runner_collect_scene_video(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _collect_scene_video(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/match_scene_video_start_color")
    async def vrgdg_workflow_runner_match_scene_video_start_color(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _apply_scene_start_color_match(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/trim_scene_video")
    async def vrgdg_workflow_runner_trim_scene_video(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _trim_scene_video(payload)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr or exc.stdout or str(exc)
            return web.json_response({"ok": False, "error": f"FFmpeg failed:\n{error}"}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/find_scene_video_output")
    async def vrgdg_workflow_runner_find_scene_video_output(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _find_scene_video_output(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/collect_minimax_h3_stage_backup")
    async def vrgdg_workflow_runner_collect_minimax_h3_stage_backup(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _collect_minimax_h3_stage_backup(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/find_minimax_h3_stage_outputs")
    async def vrgdg_workflow_runner_find_minimax_h3_stage_outputs(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _find_minimax_h3_stage_outputs(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/cleanup_minimax_h3_output")
    async def vrgdg_workflow_runner_cleanup_minimax_h3_output(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _cleanup_minimax_h3_output_folder(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/stitch_scene_videos")
    async def vrgdg_workflow_runner_stitch_scene_videos(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _stitch_scene_videos(payload)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr or exc.stdout or str(exc)
            return web.json_response({"ok": False, "error": f"FFmpeg failed:\n{error}"}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/workflow_runner/render_image_slideshow")
    async def vrgdg_workflow_runner_render_image_slideshow(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _render_image_slideshow(payload)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr or exc.stdout or str(exc)
            return web.json_response({"ok": False, "error": f"FFmpeg failed:\n{error}"}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    _VRGDG_WORKFLOW_RUNNER_ROUTES_REGISTERED = True
