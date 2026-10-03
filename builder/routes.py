"""HTTP routes for the Video Builder."""

import os
import asyncio
import sys
import tempfile
import zipfile
from aiohttp import web
from server import PromptServer
from ..minimax.latent_manager import SceneLatentManager
from ..post_process.lut_video_tools import register_lut_routes
from ..post_process.face_fix import register_face_fix_routes

from .video_profiles import ProfileExistsError, delete_video_profile, list_video_profiles, load_video_profile, save_video_profile
from .paths import _open_local_file, _open_native_picker, _resolve_existing_file
from .audio import _convert_audio_to_wav, _create_silent_audio, _default_audio_srt_paths, _estimate_beats_from_audio, _find_latest_capcut_beats, _load_srt_segments, _prepare_scene_audio_mix, _read_audio_peaks, _save_project_audio, _save_project_srt, _save_scene_audio, _save_single_scene_srt, _trim_scene_audio
from .media import _archive_scene_image, _delete_project_media, _extract_video_final_frame_as_scene_image, _import_reference_locations_from_project, _import_reference_subjects_from_project, _restore_scene_video, _save_flux_reference_image, _save_scene_image, _scan_builder_scene_videos
from .project import _copy_latest_prompt_creator_outputs, _copy_prompt_creator_outputs_from_source, _default_context_paths, _delete_builder_project, _import_builder_project_zip, _list_builder_projects, _load_builder_session, _load_editable_text_file, _load_model_defaults, _load_prompt_json, _load_wizard_draft, _new_builder_project, _prepare_builder_project_export, _project_prompt_creator_paths, _renumber_scene_assets_after_insert, _renumber_scene_assets_after_removal, _save_builder_project_as, _save_builder_render_log, _save_builder_session, _save_editable_text_file, _save_wizard_draft
from ..llm.builder_instructions import _get_builder_instruction, _list_builder_instruction_presets, _load_builder_instruction_preset, _reset_builder_instruction, _save_builder_instruction, _save_builder_instruction_preset
from ..llm.builder_runner import _clear_builder_memory_direct, _gemma_choices, _list_lm_studio_models, _list_own_server_models, _llm_multi_choices, _test_llm_api, _test_own_server
from ..llm.video_prompt_generation import _edit_builder_video_prompt, _enhance_builder_video_prompt, _generate_builder_chained_i2v_prompt, _generate_builder_i2v_prompt, _generate_builder_motion_notes, _generate_builder_t2v_prompt
from ..llm.image_prompt_generation import _analyze_builder_story_references, _edit_builder_image_prompt, _generate_builder_concept_prompts, _generate_builder_reference_description, _generate_builder_t2i_prompt, _generate_flux_klein_prompt, _generate_flux_reference_location_map, _generate_flux_reference_locations, _generate_flux_reference_subjects, _generate_flux_reference_zimage_prompt, _generate_lm_scout_locations, _generate_nb_image_prompt, _generate_wizard_locations_from_lyrics
from ..llm.builder_agent import _generate_builder_agent_reply
from ..llm.gemma4 import _run_gemma4_prompt


_VRGDG_MUSIC_BUILDER_ROUTES_REGISTERED = False


def _ensure_music_builder_routes():
    global _VRGDG_MUSIC_BUILDER_ROUTES_REGISTERED
    if _VRGDG_MUSIC_BUILDER_ROUTES_REGISTERED:
        return
    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return
    register_lut_routes(server_instance)
    register_face_fix_routes(server_instance)

    @server_instance.routes.post("/vrgdg/music_builder/analyze_audio")
    async def vrgdg_music_builder_analyze_audio(request):
        try:
            payload = await request.json()
            def _analyze():
                audio_path = _resolve_existing_file(payload.get("audio_path", ""), "Audio file")
                project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
                if os.path.splitext(audio_path)[1].lower() == ".m4a" and project_folder:
                    audio_path = _convert_audio_to_wav(
                        audio_path,
                        os.path.join(project_folder, "project_audio", "project_audio.wav"),
                    )
                res = _read_audio_peaks(audio_path, payload.get("target_peaks", 1600))
                res["beats"], res["tempo_bpm"] = _estimate_beats_from_audio(
                    audio_path,
                    res.get("peaks", []),
                    res.get("duration", 0),
                    include_tempo=True,
                )
                return {"audio_path": audio_path, **res}
            data = await asyncio.to_thread(_analyze)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **data})

    @server_instance.routes.post("/vrgdg/music_builder/import_capcut_beats")
    async def vrgdg_music_builder_import_capcut_beats(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_find_latest_capcut_beats, payload.get("audio_duration", 0))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_session")
    async def vrgdg_music_builder_save_session(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_builder_session, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_render_log")
    async def vrgdg_music_builder_save_render_log(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_builder_render_log, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_wizard_draft")
    async def vrgdg_music_builder_save_wizard_draft(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_wizard_draft, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_wizard_draft")
    async def vrgdg_music_builder_load_wizard_draft(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_load_wizard_draft, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/model_defaults")
    async def vrgdg_music_builder_model_defaults(request):
        try:
            result = await asyncio.to_thread(_load_model_defaults)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/new_project")
    async def vrgdg_music_builder_new_project(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_new_builder_project, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_project_as")
    async def vrgdg_music_builder_save_project_as(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_builder_project_as, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/export_project")
    async def vrgdg_music_builder_export_project(request):
        zip_path = ""
        response = None
        try:
            zip_path, download_name = await asyncio.to_thread(
                _prepare_builder_project_export,
                request.query.get("project_folder", ""),
            )
            response = web.StreamResponse(status=200, headers={
                "Content-Type": "application/zip",
                "Content-Disposition": f'attachment; filename="{download_name}"',
                "Content-Length": str(os.path.getsize(zip_path)),
                "Cache-Control": "no-store",
            })
            await response.prepare(request)
            with open(zip_path, "rb") as handle:
                while True:
                    chunk = await asyncio.to_thread(handle.read, 1024 * 1024)
                    if not chunk:
                        break
                    await response.write(chunk)
            await response.write_eof()
            return response
        except Exception as exc:
            if response is not None and response.prepared:
                raise
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        finally:
            if zip_path:
                try:
                    os.remove(zip_path)
                except OSError:
                    pass

    @server_instance.routes.post("/vrgdg/music_builder/import_project")
    async def vrgdg_music_builder_import_project(request):
        temp_path = ""
        try:
            reader = await request.multipart()
            requested_name = ""
            upload = None
            async for part in reader:
                if part.name == "project_name":
                    requested_name = (await part.text()).strip()
                elif part.name == "project_zip":
                    suffix = os.path.splitext(part.filename or "project.zip")[1] or ".zip"
                    temp_handle = tempfile.NamedTemporaryFile(prefix="vrgdg_builder_import_", suffix=suffix, delete=False)
                    temp_path = temp_handle.name
                    upload = temp_handle
                    try:
                        while True:
                            chunk = await part.read_chunk(size=1024 * 1024)
                            if not chunk:
                                break
                            upload.write(chunk)
                    finally:
                        upload.close()
            if not temp_path or not os.path.isfile(temp_path):
                raise ValueError("Choose a .vrgdg.zip project package to import.")
            if not zipfile.is_zipfile(temp_path):
                raise ValueError("The selected file is not a valid ZIP project package.")
            result = await asyncio.to_thread(_import_builder_project_zip, temp_path, requested_name)
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @server_instance.routes.post("/vrgdg/music_builder/save_scene_image")
    async def vrgdg_music_builder_save_scene_image(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_scene_image, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/delete_project_media")
    async def vrgdg_music_builder_delete_project_media(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_delete_project_media, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/archive_scene_image")
    async def vrgdg_music_builder_archive_scene_image(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_archive_scene_image, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/extract_video_final_frame")
    async def vrgdg_music_builder_extract_video_final_frame(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_extract_video_final_frame_as_scene_image, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_flux_reference_image")
    async def vrgdg_music_builder_save_flux_reference_image(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_flux_reference_image, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/import_reference_subjects")
    async def vrgdg_music_builder_import_reference_subjects(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_import_reference_subjects_from_project, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/import_reference_locations")
    async def vrgdg_music_builder_import_reference_locations(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_import_reference_locations_from_project, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_scene_audio")
    async def vrgdg_music_builder_save_scene_audio(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_scene_audio, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_project_audio")
    async def vrgdg_music_builder_save_project_audio(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_project_audio, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_project_srt")
    async def vrgdg_music_builder_save_project_srt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_project_srt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_single_scene_srt")
    async def vrgdg_music_builder_save_single_scene_srt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_single_scene_srt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/trim_scene_audio")
    async def vrgdg_music_builder_trim_scene_audio(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_trim_scene_audio, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/create_silent_audio")
    async def vrgdg_music_builder_create_silent_audio(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_create_silent_audio, payload or {})
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/prepare_scene_audio_mix")
    async def vrgdg_music_builder_prepare_scene_audio_mix(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_prepare_scene_audio_mix, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_session")
    async def vrgdg_music_builder_load_session(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_load_builder_session, payload.get("project_folder", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/list_projects")
    async def vrgdg_music_builder_list_projects(request):
        try:
            result = await asyncio.to_thread(_list_builder_projects, request.query.get("project_root", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/delete_project")
    async def vrgdg_music_builder_delete_project(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_delete_builder_project, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/latent_status")
    async def vrgdg_music_builder_latent_status(request):
        try:
            payload = await request.json()
            project_folder = str(payload.get("project_folder", "") or "").strip().strip('"')
            scene_number = int(payload.get("scene_number", 1))
            result = await asyncio.to_thread(SceneLatentManager.get_latent_info, project_folder, scene_number)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response(result)

    @server_instance.routes.post("/vrgdg/music_builder/check_latent_predecessor")
    async def vrgdg_music_builder_check_latent_predecessor(request):
        try:
            payload = await request.json()
            project_folder = str(payload.get("project_folder", "") or "").strip().strip('"')
            scene_number = int(payload.get("scene_number", 1))
            if scene_number <= 1:
                return web.json_response({
                    "ok": True,
                    "scene_number": scene_number,
                    "predecessor_needed": False,
                    "predecessor_exists": True,
                    "predecessor_scene": 0,
                    "predecessor_path": "",
                    "frame_count": 0,
                    "token_count": 0,
                    "dirty": False,
                })
            pred_scene = scene_number - 1
            info = await asyncio.to_thread(SceneLatentManager.get_latent_info, project_folder, pred_scene)
            return web.json_response({
                "ok": True,
                "scene_number": scene_number,
                "predecessor_scene": pred_scene,
                "predecessor_needed": True,
                "predecessor_exists": info.get("exists", False),
                "predecessor_path": info.get("path", ""),
                "frame_count": info.get("frame_count", 0),
                "token_count": info.get("token_count", 0),
                "tail_padding_known": info.get("tail_padding_frames") is not None,
                "dirty": info.get("dirty", False),
            })
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server_instance.routes.post("/vrgdg/music_builder/delete_scene_latent")
    async def vrgdg_music_builder_delete_scene_latent(request):
        try:
            payload = await request.json()
            project_folder = str(payload.get("project_folder", "") or "").strip().strip('"')
            if bool(payload.get("all", False)):
                removed = await asyncio.to_thread(SceneLatentManager.delete_all_latents, project_folder)
                return web.json_response({"ok": True, "deleted": removed > 0, "removed_files": removed, "all": True})
            scene_number = int(payload.get("scene_number", 1))
            reindex = bool(payload.get("reindex", True))
            def _delete_and_reindex(p_folder, s_num, should_reindex):
                del_result = SceneLatentManager.delete_latent(p_folder, s_num)
                if should_reindex:
                    SceneLatentManager.reindex_latents(p_folder, s_num)
                return del_result
            deleted = await asyncio.to_thread(_delete_and_reindex, project_folder, scene_number, reindex)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "deleted": deleted, "scene_number": scene_number})

    @server_instance.routes.post("/vrgdg/music_builder/renumber_scenes_after_removal")
    async def vrgdg_music_builder_renumber_scenes_after_removal(request):
        try:
            payload = await request.json()
            project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
            if not os.path.isdir(project_folder):
                raise ValueError(f"Project folder was not found: {project_folder}")
            removed_scene_number = int(payload.get("removed_scene_number"))
            renamed = await asyncio.to_thread(_renumber_scene_assets_after_removal, project_folder, removed_scene_number)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "renamed": renamed})

    @server_instance.routes.post("/vrgdg/music_builder/renumber_scenes_after_insert")
    async def vrgdg_music_builder_renumber_scenes_after_insert(request):
        try:
            payload = await request.json()
            project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
            if not os.path.isdir(project_folder):
                raise ValueError(f"Project folder was not found: {project_folder}")
            inserted_scene_number = int(payload.get("inserted_scene_number"))
            renamed = await asyncio.to_thread(_renumber_scene_assets_after_insert, project_folder, inserted_scene_number)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "renamed": renamed})

    @server_instance.routes.post("/vrgdg/music_builder/list_dirty_latents")
    async def vrgdg_music_builder_list_dirty_latents(request):
        try:
            payload = await request.json()
            project_folder = str(payload.get("project_folder", "") or "").strip().strip('"')
            dirty_scenes = await asyncio.to_thread(SceneLatentManager.list_dirty, project_folder)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "dirty_scenes": dirty_scenes})

    @server_instance.routes.post("/vrgdg/music_builder/scan_scene_videos")
    async def vrgdg_music_builder_scan_scene_videos(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_scan_builder_scene_videos, payload.get("project_folder", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/restore_scene_video")
    async def vrgdg_music_builder_restore_scene_video(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_restore_scene_video, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_srt")
    async def vrgdg_music_builder_load_srt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_load_srt_segments, payload.get("srt_path", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_prompt_json")
    async def vrgdg_music_builder_load_prompt_json(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_load_prompt_json, payload.get("prompt_json_path", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_text_file")
    async def vrgdg_music_builder_load_text_file(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_load_editable_text_file, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_text_file")
    async def vrgdg_music_builder_save_text_file(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_editable_text_file, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/open_local_file")
    async def vrgdg_music_builder_open_local_file(request):
        try:
            payload = await request.json()
            path = await asyncio.to_thread(_open_local_file, payload.get("path", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "path": path})

    @server_instance.routes.get("/vrgdg/music_builder/default_context_paths")
    async def vrgdg_music_builder_default_context_paths(request):
        try:
            result = await asyncio.to_thread(_default_context_paths)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/project_prompt_creator_paths")
    async def vrgdg_music_builder_project_prompt_creator_paths(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_project_prompt_creator_paths, payload.get("project_folder", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/import_latest_prompt_creator_outputs")
    async def vrgdg_music_builder_import_latest_prompt_creator_outputs(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_copy_latest_prompt_creator_outputs, payload.get("project_folder", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/copy_prompt_creator_outputs")
    async def vrgdg_music_builder_copy_prompt_creator_outputs(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(
                _copy_prompt_creator_outputs_from_source,
                payload.get("project_folder", ""),
                payload.get("source_project_folder", ""),
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/default_audio_srt_paths")
    async def vrgdg_music_builder_default_audio_srt_paths(request):
        try:
            result = await asyncio.to_thread(_default_audio_srt_paths)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/pick_path")
    async def vrgdg_music_builder_pick_path(request):
        try:
            payload = await request.json()
            kind = str(payload.get("kind", "") or "")
            # macOS aborts when a Tk window is created off the main thread, so the picker stays on the event loop there.
            path = _open_native_picker(kind) if sys.platform == "darwin" else await asyncio.to_thread(_open_native_picker, kind)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "path": path})

    @server_instance.routes.get("/vrgdg/music_builder/audio")
    async def vrgdg_music_builder_audio(request):
        raw_path = str(request.query.get("path", "") or "").strip()
        audio_path = os.path.normpath(os.path.abspath(raw_path))
        if not await asyncio.to_thread(os.path.isfile, audio_path):
            return web.json_response({"ok": False, "error": "Audio file was not found."}, status=404)
        return web.FileResponse(audio_path)

    @server_instance.routes.get("/vrgdg/music_builder/gemma_choices")
    async def vrgdg_music_builder_gemma_choices(request):
        try:
            result = await asyncio.to_thread(_gemma_choices)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/music_builder/llm_api_choices")
    async def vrgdg_music_builder_llm_api_choices(request):
        try:
            result = await asyncio.to_thread(_llm_multi_choices)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/test_llm_api")
    async def vrgdg_music_builder_test_llm_api(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_test_llm_api, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/test_own_server")
    async def vrgdg_music_builder_test_own_server(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_test_own_server, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/own_server_models")
    async def vrgdg_music_builder_own_server_models(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_list_own_server_models, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/lm_studio_models")
    async def vrgdg_music_builder_lm_studio_models(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_list_lm_studio_models, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/get_instruction")
    async def vrgdg_music_builder_get_instruction(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_get_builder_instruction, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_instruction")
    async def vrgdg_music_builder_save_instruction(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_builder_instruction, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/reset_instruction")
    async def vrgdg_music_builder_reset_instruction(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_reset_builder_instruction, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/list_instruction_presets")
    async def vrgdg_music_builder_list_instruction_presets(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_list_builder_instruction_presets, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/save_instruction_preset")
    async def vrgdg_music_builder_save_instruction_preset(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_save_builder_instruction_preset, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/load_instruction_preset")
    async def vrgdg_music_builder_load_instruction_preset(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_load_builder_instruction_preset, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/list_video_profiles")
    async def vrgdg_music_builder_list_video_profiles(request):
        try:
            profiles = await asyncio.to_thread(list_video_profiles)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "profiles": profiles})

    @server_instance.routes.post("/vrgdg/music_builder/load_video_profile")
    async def vrgdg_music_builder_load_video_profile(request):
        try:
            payload = await request.json()
            profile = await asyncio.to_thread(load_video_profile, payload.get("name"))
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "profile": profile})

    @server_instance.routes.post("/vrgdg/music_builder/save_video_profile")
    async def vrgdg_music_builder_save_video_profile(request):
        try:
            payload = await request.json()
            profile = await asyncio.to_thread(
                save_video_profile, payload.get("name"), payload.get("settings"), bool(payload.get("overwrite")),
            )
        except ProfileExistsError as exc:
            return web.json_response({"ok": False, "error": str(exc), "exists": True, "name": exc.name}, status=409)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "profile": profile})

    @server_instance.routes.post("/vrgdg/music_builder/delete_video_profile")
    async def vrgdg_music_builder_delete_video_profile(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(delete_video_profile, payload.get("name"))
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/clear_memory_direct")
    async def vrgdg_music_builder_clear_memory_direct(request):
        try:
            result = await asyncio.to_thread(_clear_builder_memory_direct)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_t2i")
    async def vrgdg_music_builder_generate_t2i(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_t2i_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/agent_chat")
    async def vrgdg_music_builder_agent_chat(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_agent_reply, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_concept_prompts")
    async def vrgdg_music_builder_generate_concept_prompts(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_concept_prompts, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_motion_notes")
    async def vrgdg_music_builder_generate_motion_notes(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_motion_notes, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_i2v")
    async def vrgdg_music_builder_generate_i2v(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_i2v_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_chained_i2v")
    async def vrgdg_music_builder_generate_chained_i2v(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_chained_i2v_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_t2v")
    async def vrgdg_music_builder_generate_t2v(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_t2v_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/enhance_video_prompt")
    async def vrgdg_music_builder_enhance_video_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_enhance_builder_video_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/edit_video_prompt")
    async def vrgdg_music_builder_edit_video_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_edit_builder_video_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/edit_image_prompt")
    async def vrgdg_music_builder_edit_image_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_edit_builder_image_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_flux_klein_prompt")
    async def vrgdg_music_builder_generate_flux_klein_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_klein_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/describe_reference_image")
    async def vrgdg_music_builder_describe_reference_image(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_builder_reference_description, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/analyze_story_references")
    async def vrgdg_music_builder_analyze_story_references(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_analyze_builder_story_references, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/generate_nb_image_prompt")
    async def vrgdg_music_builder_generate_nb_image_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_nb_image_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/flux_reference_location_map")
    async def vrgdg_music_builder_flux_reference_location_map(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_reference_location_map, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/flux_reference_extract_locations")
    async def vrgdg_music_builder_flux_reference_extract_locations(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_reference_locations, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/lm_scout_locations")
    async def vrgdg_music_builder_lm_scout_locations(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_lm_scout_locations, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/wizard_locations_from_lyrics")
    async def vrgdg_music_builder_wizard_locations_from_lyrics(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_wizard_locations_from_lyrics, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/flux_reference_extract_subjects")
    async def vrgdg_music_builder_flux_reference_extract_subjects(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_reference_subjects, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/music_builder/flux_reference_zimage_prompt")
    async def vrgdg_music_builder_flux_reference_zimage_prompt(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_generate_flux_reference_zimage_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/gemma4/generate")
    async def vrgdg_gemma4_generate(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        if not isinstance(payload, dict):
            return web.json_response({"ok": False, "error": "JSON body must be an object."}, status=400)

        try:
            result = await asyncio.to_thread(_run_gemma4_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        return web.json_response({"ok": True, **result})

    _VRGDG_MUSIC_BUILDER_ROUTES_REGISTERED = True
