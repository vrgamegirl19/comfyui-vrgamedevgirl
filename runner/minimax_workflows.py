"""MiniMax H3 graph builders: single pass, 2-pass, advanced 2-pass and 3-pass."""

import copy
import json
import math
import os
import random
from ..minimax.latent_manager import calculate_minimax_h3_timing
from ..minimax.resolution import frame_size
from ..minimax.tile_plan import HIDDEN_ADVANCED_SETTINGS, PLAN_OUTPUT_NAMES, normalize_vram_preset, plan_spatial_tiles

from .paths import _bool_payload, _first_payload_value, _float_payload, _int_payload
from .models import _NONE_LORA, _clean_lora_name, _model_choice_exists, _require_model_choice
from .api_graph import _api_node_id_by_class, _compat_node_inputs, _get_comfy_node_mappings, _load_api_template, _set_api_input
from .minimax_inputs import _MINIMAX_H3_ASPECT_RATIOS, _minimax_h3_2pass_api_template_path, _minimax_h3_3pass_api_template_path, _minimax_h3_api_template_path, _minimax_h3_built_in_audio_api_template_path, _minimax_h3_image_paths, _minimax_h3_output_location, _minimax_h3_video_references, _patch_minimax_h3_image_to_video_node, _probe_media_duration_seconds, _trim_minimax_h3_audio_context
from .minimax_patches import _minimax_h3_effective_warmup_frames, _minimax_h3_is_latent_mode, _minimax_h3_latent_continuation_mode, _patch_minimax_h3_advanced_settings, _patch_minimax_h3_fast_decode, _patch_minimax_h3_latent_continuation, _patch_minimax_h3_loras, _patch_minimax_h3_memory_efficient_sage_attention, _patch_minimax_h3_optional_model_paths, _patch_minimax_h3_save_latent, _patch_minimax_h3_te_speed, _patch_minimax_h3_turbo, _require_minimax_h3_memory_efficient_sage_attention


def _build_minimax_h3_api_prompt(payload):
    if _bool_payload(payload, "use_memory_efficient_sage_attention", False):
        _require_minimax_h3_memory_efficient_sage_attention()
    raw_audio_mode = str(payload.get("audio_mode") or payload.get("audioMode") or "input_audio").strip().lower().replace("-", "_").replace(" ", "_")
    audio_mode = "built_in_audio" if raw_audio_mode in {"built_in_audio", "native_audio", "generated_audio"} else "input_audio"
    workflow_template = _minimax_h3_built_in_audio_api_template_path() if audio_mode == "built_in_audio" else _minimax_h3_api_template_path()
    workflow_path, prompt = _load_api_template(workflow_template)
    prompt = copy.deepcopy(prompt)

    video_prompt = str(_first_payload_value(
        payload, "prompt", "video_prompt", "i2v_prompt", "t2v_prompt", default=""
    ) or "").strip()
    if not video_prompt:
        raise ValueError("MiniMax H3 video prompt is empty.")

    audio_path = ""
    if audio_mode == "input_audio":
        audio_text = str(_first_payload_value(
            payload, "audio_path", "source_audio_path", default=""
        ) or "").strip().strip('"')
        if not audio_text:
            raise ValueError("MiniMax H3 source audio path is empty.")
        audio_path = os.path.abspath(audio_text)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"MiniMax H3 source audio was not found: {audio_path}")

    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_text:
        raise ValueError("Project folder is empty.")
    project_folder = os.path.abspath(project_text)
    if not os.path.isdir(project_folder):
        raise FileNotFoundError(f"Project folder was not found: {project_folder}")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)

    timeline_start = _first_payload_value(
        payload, "timeline_start_seconds", "scene_start_seconds", "start", default=0
    )
    timeline_end = _first_payload_value(
        payload, "timeline_end_seconds", "scene_end_seconds", "end", default=None
    )
    if timeline_end is None:
        scene_duration = _first_payload_value(
            payload, "scene_duration_seconds", "scene_duration", "duration", default=None
        )
        if scene_duration is None:
            raise ValueError("MiniMax H3 needs timeline_end_seconds or scene_duration_seconds.")
        try:
            timeline_end = float(timeline_start) + float(scene_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("MiniMax H3 timeline timing must be numeric.") from exc

    source_duration = _first_payload_value(
        payload, "source_duration_seconds", "audio_duration_seconds", default=None
    )
    if source_duration is None and audio_mode == "input_audio":
        source_duration = _probe_media_duration_seconds(audio_path)
    source_start = _first_payload_value(
        payload, "source_start_seconds", "audio_start_seconds", default=None
    )
    warmup_frames = _minimax_h3_effective_warmup_frames(payload)
    cooldown_frames = _first_payload_value(
        payload, "cooldown_frames", "tail_loss_frames", default=0
    )
    timing = calculate_minimax_h3_timing(
        timeline_start,
        timeline_end,
        warmup_frames,
        cooldown_frames,
        source_start_seconds=source_start,
        source_duration_seconds=source_duration,
        pad_warmup=(
            _minimax_h3_is_latent_mode(_minimax_h3_latent_continuation_mode(payload))
            and scene_number > 1
        ),
    )
    prepared_audio = None
    if audio_mode == "input_audio":
        prepared_audio = _trim_minimax_h3_audio_context(
            audio_path,
            project_folder,
            scene_number,
            timing,
        )

    image_paths = _minimax_h3_image_paths(payload)
    video_references = _minimax_h3_video_references(payload)
    video_mode = str(payload.get("video_mode") or payload.get("mode") or "text_to_video").strip().lower().replace("-", "_").replace(" ", "_")
    if video_mode in {"image_to_video", "image_reference_to_video"}:
        if not image_paths:
            raise ValueError("MiniMax H3 image-to-video requires a scene image as the first frame.")
        last_frame_path = str(payload.get("last_frame_path") or "").strip().strip('"')
        if last_frame_path:
            last_frame_path = os.path.abspath(last_frame_path)
            if not os.path.isfile(last_frame_path):
                raise FileNotFoundError(f"MiniMax H3 last-frame image was not found: {last_frame_path}")
        if video_mode == "image_reference_to_video":
            start_frame_path = image_paths[0]
            image_paths = image_paths[1:]
            if last_frame_path and os.path.abspath(start_frame_path) == last_frame_path:
                last_frame_path = ""
            combined_images = [start_frame_path] + ([last_frame_path] if last_frame_path else []) + image_paths
            image_paths = combined_images
            _patch_minimax_h3_image_to_video_node(
                prompt,
                image_paths,
                include_references=True,
                has_last_frame=bool(last_frame_path),
            )
        else:
            if last_frame_path and os.path.abspath(image_paths[0]) != last_frame_path:
                image_paths = [image_paths[0], last_frame_path]
            _patch_minimax_h3_image_to_video_node(prompt, image_paths)
        video_references = []
    aspect_ratio = str(payload.get("aspect_ratio") or "16:9 (Widescreen)").strip()
    if aspect_ratio not in _MINIMAX_H3_ASPECT_RATIOS:
        raise ValueError(f"Unsupported MiniMax H3 aspect ratio: {aspect_ratio}")
    megapixels = _float_payload(payload, "megapixels", 0.9, 0.1, 16.0)
    diffusion_model_name = str(
        payload.get("diffusion_model_name") or "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    ).strip()
    clip_name = str(
        payload.get("clip_name") or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    ).strip()
    video_vae_name = str(
        payload.get("video_vae_name") or "minimax_h3_video_vae_fp16.safetensors"
    ).strip()
    audio_vae_name = str(
        payload.get("audio_vae_name") or "minimax_h3_audio_vae_fp32.safetensors"
    ).strip()
    if diffusion_model_name.lower().endswith(".gguf"):
        raise ValueError("MiniMax H3 GGUF loading is not enabled yet. Choose a non-GGUF diffusion model.")
    _require_model_choice(("diffusion_models", "unet"), diffusion_model_name, "MiniMax H3 diffusion model")
    _require_model_choice(("text_encoders", "clip"), clip_name, "MiniMax H3 text encoder")
    _require_model_choice("vae", video_vae_name, "MiniMax H3 video VAE")
    _require_model_choice("vae", audio_vae_name, "MiniMax H3 audio VAE")

    seed_value = payload.get("seed", 69)
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        seed = 69
    if seed < 0:
        seed = random.randrange(0, 0xFFFFFFFFFFFFFFFF + 1)
    seed = min(seed, 0xFFFFFFFFFFFFFFFF)

    output_folder, filename_prefix = _minimax_h3_output_location(project_folder, scene_number)
    _set_api_input(prompt, "132", "value", timing.workflow_duration_input_seconds)
    _set_api_input(prompt, "138", "value", video_prompt)
    _set_api_input(prompt, "129", "noise_seed", seed)
    _set_api_input(prompt, "115", "aspect_ratio", aspect_ratio)
    _set_api_input(prompt, "115", "megapixels", megapixels)
    _set_api_input(prompt, "115", "multiple", 32)
    _set_api_input(prompt, "141", "model_name", diffusion_model_name)
    _set_api_input(prompt, "128", "clip_name", clip_name)
    _set_api_input(prompt, "119", "vae_name", video_vae_name)
    _set_api_input(prompt, "120", "vae_name", audio_vae_name)
    if audio_mode == "input_audio":
        _set_api_input(prompt, "171", "audio_file", prepared_audio["audio_path"])
        _set_api_input(prompt, "171", "seek_seconds", 0)
        _set_api_input(prompt, "171", "duration", 0)
    _set_api_input(prompt, "180", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    _set_api_input(prompt, "180", "video_references", json.dumps(video_references, ensure_ascii=False))
    _set_api_input(prompt, "142", "frame_rate", 24)
    _set_api_input(prompt, "142", "filename_prefix", filename_prefix)
    # Keep every aligned H3 frame. VHS trim_to_audio muxes with -shortest while
    # stream-copying H.264, which can discard final video packets before our
    # exact scene trimmer receives them.
    _set_api_input(prompt, "142", "trim_to_audio", False)
    if str(payload.get("video_mode") or "reference_to_video") == "reference_to_video":
        payload = {**payload, "easy_cache_bypass": True, "use_turbo_lora": False}
    advanced_settings = _patch_minimax_h3_advanced_settings(prompt, payload)
    if _bool_payload(payload, "use_memory_efficient_sage_attention", False):
        _set_api_input(prompt, "141", "sage_attention", "disabled")
        advanced_settings["sage_attention"] = "disabled"
    lora_settings = _patch_minimax_h3_loras(prompt, payload)
    turbo_settings = _patch_minimax_h3_turbo(prompt, payload)
    memory_efficient_sage_attention = _patch_minimax_h3_memory_efficient_sage_attention(prompt, payload)
    optional_patch_nodes = []
    if _bool_payload(payload, "use_feedforward", False) or _bool_payload(payload, "use_block_sparse_attention", False):
        scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
        guider_id = _api_node_id_by_class(prompt, "BasicGuider", fallback="126")
        model_ref = prompt.get(scheduler_id, {}).get("inputs", {}).get("model")
        if not isinstance(model_ref, list) or len(model_ref) != 2:
            raise ValueError("MiniMax H3 optional model patches could not find the current model connection.")
        model_refs = {"single_pass": model_ref}
        optional_patch_nodes = _patch_minimax_h3_optional_model_paths(
            prompt, model_refs,
            _bool_payload(payload, "use_feedforward", False),
            _bool_payload(payload, "use_block_sparse_attention", False),
        )
        for node_id in (scheduler_id, guider_id):
            _set_api_input(prompt, node_id, "model", list(model_refs["single_pass"]))
    if _bool_payload(payload, "use_te_speed", False):
        scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
        guider_id = _api_node_id_by_class(prompt, "BasicGuider", fallback="126")
        model_ref = _patch_minimax_h3_te_speed(prompt, prompt[scheduler_id]["inputs"]["model"], payload, "9210")
        for node_id in (scheduler_id, guider_id):
            _set_api_input(prompt, node_id, "model", list(model_ref))
    if _bool_payload(payload, "use_fast_vae_decode", False):
        _patch_minimax_h3_fast_decode(prompt)
    if turbo_settings["enabled"]:
        advanced_settings = {
            **advanced_settings,
            "effective_sampler_name": "MiniMaxH3TurboSampler",
            "effective_scheduler": "simple",
            "effective_steps": turbo_settings["steps"],
        }

    latent_continuation_settings = _patch_minimax_h3_latent_continuation(prompt, payload)
    save_latent_settings = _patch_minimax_h3_save_latent(prompt, payload, timing)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": prompt,
        "latent_continuation_settings": latent_continuation_settings,
        "save_latent_settings": save_latent_settings,
        "used_seed": seed,
        "audio_mode": audio_mode,
        "timing": timing.to_dict(),
        "prepared_audio": prepared_audio,
        "post_render_trim": {
            "start": timing.final_trim_start_seconds,
            "duration": timing.final_trim_duration_seconds,
            "frames": timing.final_frame_count,
        },
        "reference_inputs": {
            "image_count": len(image_paths),
            "video_count": len(video_references),
            "video_audio_count": sum(1 for item in video_references if item.get("use_audio")),
        },
        "model_settings": {
            "diffusion_model_name": diffusion_model_name,
            "clip_name": clip_name,
            "video_vae_name": video_vae_name,
            "audio_vae_name": audio_vae_name,
        },
        "advanced_settings": advanced_settings,
        "lora_settings": lora_settings,
        "turbo_settings": turbo_settings,
        "memory_efficient_sage_attention": memory_efficient_sage_attention,
        "optional_patch_nodes": optional_patch_nodes,
    }


def _build_minimax_h3_2pass_api_prompt(payload):
    """Build the cleaned external-audio MiniMax H3 two-pass API prompt.

    This intentionally has its own adapter instead of reusing the one-pass
    node IDs or mutating the existing MiniMax template path.
    """
    workflow_path, prompt = _load_api_template(_minimax_h3_2pass_api_template_path())
    prompt = copy.deepcopy(prompt)
    video_prompt = str(_first_payload_value(payload, "prompt", "video_prompt", default="") or "").strip()
    if not video_prompt:
        raise ValueError("MiniMax H3 two-pass video prompt is empty.")

    audio_path = str(_first_payload_value(payload, "audio_path", "source_audio_path", default="") or "").strip().strip('"')
    if not audio_path:
        raise ValueError("MiniMax H3 two-pass source audio path is empty.")
    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"MiniMax H3 two-pass source audio was not found: {audio_path}")

    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_text:
        raise ValueError("Project folder is empty.")
    project_folder = os.path.abspath(project_text)
    if not os.path.isdir(project_folder):
        raise FileNotFoundError(f"Project folder was not found: {project_folder}")

    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    timeline_start = _first_payload_value(payload, "timeline_start_seconds", "scene_start_seconds", "start", default=0)
    timeline_end = _first_payload_value(payload, "timeline_end_seconds", "scene_end_seconds", "end", default=None)
    if timeline_end is None:
        duration = _first_payload_value(payload, "scene_duration_seconds", "scene_duration", "duration", default=None)
        if duration is None:
            raise ValueError("MiniMax H3 two-pass needs timeline_end_seconds or scene_duration_seconds.")
        timeline_end = float(timeline_start) + float(duration)

    source_duration = _first_payload_value(payload, "source_duration_seconds", "audio_duration_seconds", default=None)
    if source_duration is None:
        source_duration = _probe_media_duration_seconds(audio_path)
    source_start = _first_payload_value(payload, "source_start_seconds", "audio_start_seconds", default=None)
    timing = calculate_minimax_h3_timing(
        timeline_start,
        timeline_end,
        _minimax_h3_effective_warmup_frames(payload),
        _first_payload_value(payload, "cooldown_frames", "tail_loss_frames", default=0),
        source_start_seconds=source_start,
        source_duration_seconds=source_duration,
        pad_warmup=(
            _minimax_h3_is_latent_mode(_minimax_h3_latent_continuation_mode(payload))
            and scene_number > 1
        ),
    )
    prepared_audio = _trim_minimax_h3_audio_context(audio_path, project_folder, scene_number, timing)

    image_paths = _minimax_h3_image_paths(payload)
    video_references = _minimax_h3_video_references(payload)
    video_mode = str(payload.get("video_mode") or payload.get("mode") or "reference_to_video").strip().lower().replace("-", "_").replace(" ", "_")
    if video_mode == "image_reference_to_video":
        if not image_paths:
            raise ValueError("MiniMax H3 image-to-video requires a scene image as the first frame.")
        start_frame_path = image_paths[0]
        image_paths = image_paths[1:]
        last_frame_path = str(payload.get("last_frame_path") or "").strip().strip('"')
        if last_frame_path:
            last_frame_path = os.path.abspath(last_frame_path)
            if not os.path.isfile(last_frame_path):
                raise FileNotFoundError(f"MiniMax H3 last-frame image was not found: {last_frame_path}")
            if os.path.abspath(start_frame_path) == last_frame_path:
                last_frame_path = ""
        combined_images = [start_frame_path] + ([last_frame_path] if last_frame_path else []) + image_paths
        image_paths = combined_images
        _patch_minimax_h3_image_to_video_node(
            prompt,
            image_paths,
            include_references=True,
            has_last_frame=bool(last_frame_path),
        )
        video_references = []
    diffusion_model_name = str(payload.get("diffusion_model_name") or "minimax_h3_ref2va_pruned_int8_convrot.safetensors").strip()
    clip_name = str(payload.get("clip_name") or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors").strip()
    video_vae_name = str(payload.get("video_vae_name") or "minimax_h3_video_vae_fp16.safetensors").strip()
    audio_vae_name = str(payload.get("audio_vae_name") or "minimax_h3_audio_vae_fp32.safetensors").strip()
    latent_upscaler_name = str(payload.get("latent_upscaler_name") or "minimax_h3_latent_upscaler_3d_bf16.safetensors").strip()
    _require_model_choice(("diffusion_models", "unet"), diffusion_model_name, "MiniMax H3 two-pass diffusion model")
    _require_model_choice(("text_encoders", "clip"), clip_name, "MiniMax H3 two-pass text encoder")
    _require_model_choice("vae", video_vae_name, "MiniMax H3 two-pass video VAE")
    _require_model_choice("vae", audio_vae_name, "MiniMax H3 two-pass audio VAE")
    _require_model_choice("latent_upscale_models", latent_upscaler_name, "MiniMax H3 learned latent upscaler")

    def _seed_payload(key, default):
        try:
            value = int(payload.get(key, default))
        except Exception:
            value = int(default)
        if value < 0:
            value = random.randrange(0, 0xFFFFFFFFFFFFFFFF + 1)
        return min(max(0, value), 0xFFFFFFFFFFFFFFFF)

    seed = _seed_payload("seed", 69)
    pass1_seed = _seed_payload("pass1_seed", -1)
    pass2_seed = _seed_payload("pass2_seed", -1)
    final_width = _int_payload(payload, "final_width", 1920, 64, 16384)
    final_height = _int_payload(payload, "final_height", 1080, 64, 16384)
    latent_scale = _float_payload(payload, "latent_upscale_scale", 2.0, 1.0, 8.0)

    _set_api_input(prompt, "138", "value", video_prompt)
    _set_api_input(prompt, "132", "value", float(timing.workflow_duration_input_seconds))
    _set_api_input(prompt, "171", "audio_file", prepared_audio["audio_path"])
    _set_api_input(prompt, "171", "seek_seconds", 0)
    _set_api_input(prompt, "171", "duration", 0)
    _set_api_input(prompt, "180", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    _set_api_input(prompt, "180", "video_references", json.dumps(video_references, ensure_ascii=False))
    ref_image_size = str(payload.get("ref_image_size") or "max").strip().lower()
    _set_api_input(prompt, "136", "ref_image_size", ref_image_size if ref_image_size in {"match", "max"} else "max")

    _set_api_input(prompt, "115", "value", final_width)
    _set_api_input(prompt, "184", "value", final_height)
    _set_api_input(prompt, "185", "value", latent_scale)
    _set_api_input(prompt, "188", "model_name", latent_upscaler_name)
    _set_api_input(prompt, "188", "device", str(payload.get("latent_upscaler_device") or "cuda"))
    _set_api_input(prompt, "188", "precision", str(payload.get("latent_upscaler_precision") or "bf16"))

    _set_api_input(prompt, "129", "noise_seed", pass1_seed)
    _set_api_input(prompt, "211", "noise_seed", pass2_seed)
    _set_api_input(prompt, "123", "sampler_name", str(payload.get("pass1_sampler_name") or "res_multistep").strip() or "res_multistep")
    _set_api_input(prompt, "210", "sampler_name", str(payload.get("pass2_sampler_name") or "res_multistep").strip() or "res_multistep")
    _set_api_input(prompt, "124", "scheduler", str(payload.get("pass1_scheduler") or "simple").strip() or "simple")
    _set_api_input(prompt, "124", "steps", _int_payload(payload, "pass1_steps", 20, 1, 1000))
    _set_api_input(prompt, "124", "denoise", _float_payload(payload, "pass1_denoise", 1.0, 0.0, 1.0))
    _set_api_input(prompt, "192", "scheduler", str(payload.get("pass2_scheduler") or "simple").strip() or "simple")
    _set_api_input(prompt, "190", "value", _int_payload(payload, "pass2_steps", 2, 1, 1000))
    _set_api_input(prompt, "191", "value", _float_payload(payload, "pass2_denoise", 0.2, 0.0, 1.0))

    _set_api_input(prompt, "141", "model_name", diffusion_model_name)
    _set_api_input(prompt, "141", "sage_attention", str(payload.get("sage_attention") or "auto"))
    _set_api_input(prompt, "141", "enable_fp16_accumulation", _bool_payload(payload, "enable_fp16_accumulation", True))
    _set_api_input(prompt, "128", "clip_name", clip_name)
    _set_api_input(prompt, "119", "vae_name", video_vae_name)
    _set_api_input(prompt, "120", "vae_name", audio_vae_name)

    turbo_lora_name = _clean_lora_name(payload.get("two_pass_lora_name", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"))
    if turbo_lora_name == _NONE_LORA:
        raise ValueError("Select the MiniMax H3 two-pass Turbo LoRA; it is required by this fast workflow.")
    _require_model_choice("loras", turbo_lora_name, "MiniMax H3 two-pass Turbo LoRA")
    _set_api_input(prompt, "207", "lora_name", turbo_lora_name)
    _set_api_input(prompt, "207", "strength_model", _float_payload(payload, "two_pass_lora_strength", 1.0, -10.0, 10.0))

    use_te_speed = _bool_payload(payload, "two_pass_use_te_speed", False)
    _set_api_input(prompt, "207", "model", ["141", 0])
    prompt.pop("208", None)
    pass1_model = ["141", 0]
    pass2_model = ["207", 0]

    extra_loras = []
    if _bool_payload(payload, "use_loras", False) or _bool_payload(payload, "use_custom_loras", False):
        raw_loras = payload.get("loras") if isinstance(payload.get("loras"), list) else []
        lora_count = _int_payload(payload, "lora_count", len(raw_loras), 0, 4)
        for item in raw_loras[:lora_count]:
            if not isinstance(item, dict):
                continue
            name = _clean_lora_name(item.get("name") or item.get("lora_name") or item.get("loraName") or _NONE_LORA)
            if not name or name == _NONE_LORA:
                continue
            if not _model_choice_exists("loras", name):
                raise ValueError(
                    f"MiniMax extra LoRA '{name}' was not found in ComfyUI/models/loras. "
                    "Download it, refresh/restart ComfyUI, and select it in MiniMax Video Settings."
                )
            apply_to = str(item.get("apply_to") or item.get("applyTo") or "both").strip().lower()
            if apply_to not in {"both", "pass1", "pass2"}:
                apply_to = "both"
            extra_loras.append({
                "name": name,
                "strength": _float_payload(item, "strength", 1.0, -10.0, 10.0),
                "apply_to": apply_to,
            })

    next_lora_node_id = 9201
    applied_extra_loras = []

    def add_extra_lora(model_ref, item, target, index):
        nonlocal next_lora_node_id
        while str(next_lora_node_id) in prompt:
            next_lora_node_id += 1
        node_id = str(next_lora_node_id)
        next_lora_node_id += 1
        prompt[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(model_ref),
                "lora_name": item["name"],
                "strength_model": item["strength"],
            },
            "_meta": {"title": f"Extra MiniMax LoRA {index} - {target}"},
        }
        applied_extra_loras.append({**item, "target": target, "node": node_id})
        return [node_id, 0]

    for index, item in enumerate(extra_loras, start=1):
        if item["apply_to"] in {"both", "pass1"}:
            pass1_model = add_extra_lora(pass1_model, item, "pass1", index)
        if item["apply_to"] in {"both", "pass2"}:
            pass2_model = add_extra_lora(pass2_model, item, "pass2", index)

    use_feedforward = _bool_payload(payload, "two_pass_use_feedforward", False)
    use_block_sparse_attention = _bool_payload(payload, "two_pass_use_block_sparse_attention", False)
    model_refs = {"pass1": pass1_model, "pass2": pass2_model}
    pass_acceleration = {
        target: {
            "te_speed": _bool_payload(payload, f"{target}_use_te_speed", use_te_speed),
            "feedforward": _bool_payload(payload, f"{target}_use_feedforward", use_feedforward),
            "block_sparse_attention": _bool_payload(payload, f"{target}_use_block_sparse_attention", use_block_sparse_attention),
        }
        for target in ("pass1", "pass2")
    }
    optional_patch_nodes = []
    for target, acceleration in pass_acceleration.items():
        if acceleration["te_speed"]:
            model_refs[target] = _patch_minimax_h3_te_speed(
                prompt, model_refs[target], payload, "208" if target == "pass1" else "9211",
            )
        branch = {target: model_refs[target]}
        optional_patch_nodes.extend(_patch_minimax_h3_optional_model_paths(
            prompt, branch, acceleration["feedforward"], acceleration["block_sparse_attention"],
        ))
        model_refs[target] = branch[target]
    pass1_model, pass2_model = model_refs["pass1"], model_refs["pass2"]

    for node_id in ("124", "126"):
        _set_api_input(prompt, node_id, "model", list(pass1_model))
    for node_id in ("192", "193"):
        _set_api_input(prompt, node_id, "model", list(pass2_model))

    use_fast_vae_decode = _bool_payload(payload, "two_pass_use_fast_vae_decode", False)
    if use_fast_vae_decode:
        _patch_minimax_h3_fast_decode(prompt)

    _set_api_input(prompt, "183", "upscale_method", str(payload.get("final_resize_method") or "nvidia_rtx_vsr"))
    _set_api_input(prompt, "142", "crf", _int_payload(payload, "output_crf", 19, 0, 100))
    output_folder, filename_prefix = _minimax_h3_output_location(project_folder, scene_number)
    _set_api_input(prompt, "142", "filename_prefix", f"{filename_prefix}_stage2")
    latent_continuation_settings = None
    save_latent_settings = None
    if not payload.get("_skip_latent_patches"):
        latent_continuation_settings = _patch_minimax_h3_latent_continuation(prompt, payload)
        save_latent_settings = _patch_minimax_h3_save_latent(prompt, payload, timing)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": prompt,
        "latent_continuation_settings": latent_continuation_settings,
        "save_latent_settings": save_latent_settings,
        "used_seed": seed,
        "audio_mode": "input_audio",
        "timing": timing.to_dict(),
        "prepared_audio": prepared_audio,
        "post_render_trim": {"start": timing.final_trim_start_seconds, "duration": timing.final_trim_duration_seconds, "frames": timing.final_frame_count},
        "reference_inputs": {"image_count": len(image_paths), "video_count": len(video_references)},
        "two_pass": {
            "pass1_steps": prompt["124"]["inputs"]["steps"],
            "pass2_steps": prompt["190"]["inputs"]["value"],
            "final_width": final_width,
            "final_height": final_height,
            "latent_upscale_scale": latent_scale,
            "te_speed_enabled": any(item["te_speed"] for item in pass_acceleration.values()),
            "pass_acceleration": pass_acceleration,
            "extra_loras": applied_extra_loras,
            "feedforward_enabled": any(item["feedforward"] for item in pass_acceleration.values()),
            "block_sparse_attention_enabled": any(item["block_sparse_attention"] for item in pass_acceleration.values()),
            "optional_patch_nodes": optional_patch_nodes,
            "fast_vae_decode_enabled": use_fast_vae_decode,
            "fast_vae_decode_tile_batch_size": 8 if use_fast_vae_decode else None,
        },
    }


def _build_minimax_h3_advanced_2pass_api_prompt(payload):
    """Build the MMH3 tiled/chunked advanced two-pass API prompt.

    The established two-pass adapter supplies the model, reference, audio,
    timing, LoRA, and sampler branches.  This adapter replaces its whole-frame
    learned-upscale/refinement tail with Comfyui-MMH3-UltimateUpscale and gives
    each pass an independent ResolutionSelector.
    """
    try:
        mappings = _get_comfy_node_mappings()
    except Exception as exc:
        raise ValueError(
            "Could not inspect ComfyUI custom-node registrations. Restart ComfyUI "
            "after installing Comfyui-MMH3-UltimateUpscale."
        ) from exc
    required_nodes = (
        "MMH3UltimateUpscale",
        "VRGDG_MiniMaxH3UltimateUpscaleParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "VRGDG_MiniMaxH3SpatialTilePlan",
    )
    missing_nodes = [name for name in required_nodes if name not in mappings]
    if missing_nodes:
        raise ValueError(
            "MiniMax H3 2 Pass Advanced requires the latest "
            "Comfyui-MMH3-UltimateUpscale custom nodes. Missing: "
            + ", ".join(missing_nodes)
            + ". Install or update the repository, then restart ComfyUI."
        )

    base_payload = dict(payload)
    # The shared adapter validates and configures this same learned-upscaler
    # checkpoint, Turbo LoRA, references, source audio, and exact timing.
    base_payload.setdefault("final_width", 1920)
    base_payload.setdefault("final_height", 1080)
    base_payload.setdefault("latent_upscale_scale", 2.0)
    base_payload.setdefault("pass2_steps", 1)
    base_payload.setdefault("pass2_denoise", 0.2)
    base_payload.setdefault("pass2_sampler_name", "sa_solver")
    base_payload.setdefault("pass2_scheduler", "simple")
    base_payload["_skip_latent_patches"] = True
    result = _build_minimax_h3_2pass_api_prompt(base_payload)
    prompt = result["prompt"]

    aspect_ratio = str(payload.get("aspect_ratio") or "16:9 (Widescreen)").strip()
    if aspect_ratio not in _MINIMAX_H3_ASPECT_RATIOS:
        raise ValueError(f"Unsupported MiniMax H3 advanced two-pass aspect ratio: {aspect_ratio}")
    pass1_megapixels = _float_payload(payload, "advanced_pass1_megapixels", 0.4, 0.1, 16.0)
    pass2_megapixels = _float_payload(payload, "advanced_pass2_megapixels", 2.0, 0.1, 16.0)
    if pass2_megapixels < pass1_megapixels:
        raise ValueError("2 Pass Advanced Pass 2 resolution must be at least Pass 1 resolution.")

    # Every tile, chunk and fade setting is derived from the Pass 2 size at run time by
    # VRGDG_MiniMaxH3SpatialTilePlan (see minimax/tile_plan.py). The only user choice is
    # the VRAM preset; retired presets (32gb, custom) map to 24gb.
    vram_preset = normalize_vram_preset(_first_payload_value(payload, "advanced_vram_preset", "vram_preset", default=""))

    def resolved_dimensions(target_megapixels):
        return frame_size(target_megapixels, aspect_ratio)

    pass1_width, pass1_height = resolved_dimensions(pass1_megapixels)
    pass2_width, pass2_height = resolved_dimensions(pass2_megapixels)

    # Independent target resolutions for the generation and tiled refinement.
    prompt["9300"] = {
        "class_type": "ResolutionSelector",
        "inputs": {"aspect_ratio": aspect_ratio, "megapixels": pass1_megapixels, "multiple": 32},
        "_meta": {"title": "2 Pass Advanced - Pass 1 Resolution"},
    }
    prompt["9301"] = {
        "class_type": "ResolutionSelector",
        "inputs": {"aspect_ratio": aspect_ratio, "megapixels": pass2_megapixels, "multiple": 32},
        "_meta": {"title": "2 Pass Advanced - Pass 2 Resolution"},
    }
    _set_api_input(prompt, "136", "width", ["9300", 0])
    _set_api_input(prompt, "136", "height", ["9300", 1])

    pass2_conditioning = copy.deepcopy(prompt["136"])
    pass2_conditioning["inputs"]["width"] = ["9301", 0]
    pass2_conditioning["inputs"]["height"] = ["9301", 1]
    pass2_conditioning["inputs"]["prompt"] = str(_first_payload_value(payload, "pass2_prompt", "minimax_h3_pass2_prompt", default="") or "")
    pass2_conditioning["_meta"] = {"title": "2 Pass Advanced - Final Resolution Conditioning"}
    prompt["9302"] = pass2_conditioning

    prompt["9303"] = {
        "class_type": "VRGDG_MiniMaxH3UltimateUpscaleParams",
        "inputs": {
            "model_name": str(payload.get("latent_upscaler_name") or "minimax_h3_latent_upscaler_3d_bf16.safetensors"),
            "width": ["9301", 0],
            "height": ["9301", 1],
            "device": HIDDEN_ADVANCED_SETTINGS["upscaler_device"],
            "precision": HIDDEN_ADVANCED_SETTINGS["upscaler_precision"],
        },
        "_meta": {"title": "2 Pass Advanced - H3 Learned Latent Upscale"},
    }

    # Resolution-driven tiling: the plan node reads the real Pass 2 size and outputs the grid,
    # overlaps, fades, minimum tile and temporal chunk settings for the VRAM preset.
    prompt["9309"] = {
        "class_type": "VRGDG_MiniMaxH3SpatialTilePlan",
        "inputs": {"width": ["9301", 0], "height": ["9301", 1], "vram_preset": vram_preset},
        "_meta": {"title": "2 Pass Advanced - Spatial Tile Plan"},
    }

    def plan_ref(name):
        return ["9309", PLAN_OUTPUT_NAMES.index(name)]

    prompt["9304"] = {
        "class_type": "MMH3TemporalSplitParams",
        "inputs": {
            "chunk_length": plan_ref("chunk_length"),
            "temporal_overlap": plan_ref("temporal_overlap"),
            "anchor_strength": HIDDEN_ADVANCED_SETTINGS["anchor_strength"],
        },
        "_meta": {"title": "2 Pass Advanced - Temporal Chunks"},
    }
    spatial_inputs = _compat_node_inputs(
        "MMH3SpatialSplitParams",
        mappings,
        {
            "upscale_width": ["9301", 0],
            "upscale_height": ["9301", 1],
            "tile_size_mode": HIDDEN_ADVANCED_SETTINGS["tile_size_mode"],
            # tile_width/tile_height are ignored in rows_cols mode; the plan node supplies the solved sizes.
            "tile_width": plan_ref("tile_width"),
            "tile_height": plan_ref("tile_height"),
            "grid_rows": plan_ref("grid_rows"),
            "grid_cols": plan_ref("grid_cols"),
            "spatial_w_overlap": plan_ref("spatial_w_overlap"),
            "spatial_h_overlap": plan_ref("spatial_h_overlap"),
            "fade_width": plan_ref("fade_width"),
            "fade_height": plan_ref("fade_height"),
            "min_tile_size": plan_ref("min_tile_size"),
            "overlap_mode": HIDDEN_ADVANCED_SETTINGS["overlap_mode"],
            "overlap_blend": HIDDEN_ADVANCED_SETTINGS["overlap_blend"],
        },
        extra_defaults={
            "masked_area_noise": HIDDEN_ADVANCED_SETTINGS["masked_area_noise"],
            "brightness_match": HIDDEN_ADVANCED_SETTINGS["brightness_match"],
            "dynamic_fade": HIDDEN_ADVANCED_SETTINGS["dynamic_fade"],
            "dynamic_fade_min": HIDDEN_ADVANCED_SETTINGS["dynamic_fade_min"],
        },
    )
    prompt["9305"] = {
        "class_type": "MMH3SpatialSplitParams",
        "inputs": spatial_inputs,
        "_meta": {"title": "2 Pass Advanced - Spatial Tiles"},
    }
    pass2_model_ref = copy.deepcopy(prompt["192"]["inputs"]["model"])
    prompt["9306"] = {
        "class_type": "MMH3UltimateUpscale",
        "inputs": {
            "model": pass2_model_ref,
            "conditioning": ["9302", 0],
            "latent": ["125", 0],
            "noise": ["211", 0],
            "sampler": ["210", 0],
            "sigmas": ["192", 0],
            "cfg": 1.0,
            "latent_upscale_param": ["9303", 0],
            "temporal_split_param": ["9304", 0],
            "spatial_split_param": ["9305", 0],
        },
        "_meta": {"title": "2 Pass Advanced - MMH3 Ultimate Upscale"},
    }
    _set_api_input(prompt, "122", "samples", ["9306", 0])
    _set_api_input(prompt, "142", "images", ["122", 0])

    # Expose Pass 1 as a reviewable backup while Pass 2 remains the final clip.
    prompt["9307"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["125", 0], "vae": ["119", 0]},
        "_meta": {"title": "2 Pass Advanced - Decode Pass 1 Preview"},
    }
    pass1_output = copy.deepcopy(prompt["142"])
    pass1_output["inputs"]["images"] = ["9307", 0]
    pass1_output["inputs"]["filename_prefix"] = str(prompt["142"]["inputs"]["filename_prefix"]).replace("_stage2", "_stage1")
    pass1_output["_meta"] = {"title": "2 Pass Advanced - Pass 1 Backup"}
    prompt["9308"] = pass1_output
    prompt["142"]["inputs"]["filename_prefix"] = str(prompt["142"]["inputs"]["filename_prefix"]).replace("_stage2", "_advanced_stage2")

    # Remove the original whole-frame upscale/refinement tail now replaced by MMH3.
    for node_id in ("115", "181", "182", "183", "184", "185", "186", "187", "188", "189", "193", "194"):
        prompt.pop(node_id, None)

    latent_continuation_settings = _patch_minimax_h3_latent_continuation(prompt, payload)
    save_latent_settings = _patch_minimax_h3_save_latent(prompt, payload, result.get("timing"))

    result["prompt"] = prompt
    result["latent_continuation_settings"] = latent_continuation_settings
    result["save_latent_settings"] = save_latent_settings
    # Same plan the plan node computes at run time; kept for progress text and the debug JSON.
    plan = plan_spatial_tiles(pass2_width, pass2_height, vram_preset)
    result["advanced_two_pass"] = {
        "pass1_megapixels": pass1_megapixels,
        "pass2_megapixels": pass2_megapixels,
        "pass1_width": pass1_width,
        "pass1_height": pass1_height,
        "pass2_width": pass2_width,
        "pass2_height": pass2_height,
        "vram_preset": vram_preset,
        "tile_size_mode": HIDDEN_ADVANCED_SETTINGS["tile_size_mode"],
        "tile_width": plan["tile_width"],
        "tile_height": plan["tile_height"],
        "grid_rows": plan["grid_rows"],
        "grid_cols": plan["grid_cols"],
        "chunk_length": plan["chunk_length"],
        "temporal_overlap": plan["temporal_overlap"],
    }
    result.pop("two_pass", None)
    return result


def _save_minimax_h3_advanced_2pass_debug_workflow(result, payload):
    """Persist the exact generated API graph for inspection before execution."""
    output_folder = os.path.abspath(str(result.get("output_folder") or "").strip())
    if not output_folder:
        return ""
    os.makedirs(output_folder, exist_ok=True)
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    debug_path = os.path.join(
        output_folder,
        f"scene_{scene_number:04d}_minimax_h3_advanced_2pass_api.json",
    )
    snapshot = {
        "workflow_type": "minimax_h3_advanced_2pass",
        "source_template": result.get("workflow_path", ""),
        "advanced_two_pass": result.get("advanced_two_pass", {}),
        "prompt": result.get("prompt", {}),
    }
    with open(debug_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return debug_path


def _build_minimax_h3_3pass_api_prompt(payload):
    """Build the experimental external-audio MiniMax H3 three-pass prompt."""
    workflow_path, prompt = _load_api_template(_minimax_h3_3pass_api_template_path())
    prompt = copy.deepcopy(prompt)
    video_prompt = str(_first_payload_value(payload, "prompt", "video_prompt", default="") or "").strip()
    if not video_prompt:
        raise ValueError("MiniMax H3 three-pass video prompt is empty.")
    audio_path = str(_first_payload_value(payload, "audio_path", "source_audio_path", default="") or "").strip().strip('"')
    if not audio_path:
        raise ValueError("MiniMax H3 three-pass source audio path is empty.")
    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"MiniMax H3 three-pass source audio was not found: {audio_path}")
    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_text:
        raise ValueError("Project folder is empty.")
    project_folder = os.path.abspath(project_text)
    if not os.path.isdir(project_folder):
        raise FileNotFoundError(f"Project folder was not found: {project_folder}")
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    timeline_start = _first_payload_value(payload, "timeline_start_seconds", "scene_start_seconds", "start", default=0)
    timeline_end = _first_payload_value(payload, "timeline_end_seconds", "scene_end_seconds", "end", default=None)
    if timeline_end is None:
        duration = _first_payload_value(payload, "scene_duration_seconds", "scene_duration", "duration", default=None)
        if duration is None:
            raise ValueError("MiniMax H3 three-pass needs timeline_end_seconds or scene_duration_seconds.")
        timeline_end = float(timeline_start) + float(duration)
    source_duration = _first_payload_value(payload, "source_duration_seconds", "audio_duration_seconds", default=None)
    if source_duration is None:
        source_duration = _probe_media_duration_seconds(audio_path)
    source_start = _first_payload_value(payload, "source_start_seconds", "audio_start_seconds", default=None)
    timing = calculate_minimax_h3_timing(
        timeline_start,
        timeline_end,
        _minimax_h3_effective_warmup_frames(payload),
        _first_payload_value(payload, "cooldown_frames", "tail_loss_frames", default=0),
        source_start_seconds=source_start,
        source_duration_seconds=source_duration,
        pad_warmup=(
            _minimax_h3_is_latent_mode(_minimax_h3_latent_continuation_mode(payload))
            and scene_number > 1
        ),
    )
    prepared_audio = _trim_minimax_h3_audio_context(audio_path, project_folder, scene_number, timing)
    image_paths = _minimax_h3_image_paths(payload)
    video_references = _minimax_h3_video_references(payload)
    diffusion_model_name = str(payload.get("diffusion_model_name") or "minimax_h3_ref2va_pruned_int8_convrot.safetensors").strip()
    clip_name = str(payload.get("clip_name") or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors").strip()
    video_vae_name = str(payload.get("video_vae_name") or "minimax_h3_video_vae_fp16.safetensors").strip()
    audio_vae_name = str(payload.get("audio_vae_name") or "minimax_h3_audio_vae_fp32.safetensors").strip()
    _require_model_choice(("diffusion_models", "unet"), diffusion_model_name, "MiniMax H3 three-pass diffusion model")
    _require_model_choice(("text_encoders", "clip"), clip_name, "MiniMax H3 three-pass text encoder")
    _require_model_choice("vae", video_vae_name, "MiniMax H3 three-pass video VAE")
    _require_model_choice("vae", audio_vae_name, "MiniMax H3 three-pass audio VAE")

    seed = _int_payload(payload, "seed", 69, 0, 0xFFFFFFFFFFFFFFFF)
    aspect_ratio = str(payload.get("aspect_ratio") or "16:9 (Widescreen)").strip()
    if aspect_ratio not in _MINIMAX_H3_ASPECT_RATIOS:
        raise ValueError(f"Unsupported MiniMax H3 three-pass aspect ratio: {aspect_ratio}")
    _set_api_input(prompt, "329", "text", video_prompt)
    _set_api_input(prompt, "84", "value", float(timing.workflow_duration_input_seconds))
    _set_api_input(prompt, "9001", "audio_file", prepared_audio["audio_path"])
    _set_api_input(prompt, "9001", "seek_seconds", 0)
    _set_api_input(prompt, "9001", "duration", 0)
    _set_api_input(prompt, "9000", "image_paths", json.dumps(image_paths, ensure_ascii=False))
    _set_api_input(prompt, "9000", "video_references", json.dumps(video_references, ensure_ascii=False))
    ref_image_size = str(payload.get("ref_image_size") or "max").strip().lower()
    if ref_image_size not in {"match", "max"}:
        ref_image_size = "max"
    _set_api_input(prompt, "108", "ref_image_size", ref_image_size)
    _set_api_input(prompt, "330", "model_name", diffusion_model_name)
    _set_api_input(prompt, "4", "clip_name", clip_name)
    _set_api_input(prompt, "5", "vae_name", video_vae_name)
    _set_api_input(prompt, "6", "vae_name", audio_vae_name)
    pass_specs = [
        (1, "105", "248", "249", "243", "328", "246", 0.4, 20, 1.0, True),
        (2, "297", "290", "289", "300", "187", "294", 1.0, 5, 0.2, False),
        (3, "334", "344", "341", "345", "187", "340", 2.0, 5, 0.2, False),
    ]
    for pass_number, resolution_id, scheduler_id, sampler_id, noise_id, default_model_id, guider_id, default_mp, default_steps, default_denoise, default_speed in pass_specs:
        prefix = f"three_pass_pass{pass_number}_"
        _set_api_input(prompt, resolution_id, "aspect_ratio", str(payload.get(f"{prefix}aspect_ratio") or aspect_ratio))
        _set_api_input(prompt, resolution_id, "megapixels", _float_payload(payload, f"{prefix}megapixels", default_mp, 0.1, 16.0))
        _set_api_input(prompt, scheduler_id, "steps", _int_payload(payload, f"{prefix}steps", default_steps, 1, 1000))
        _set_api_input(prompt, scheduler_id, "denoise", _float_payload(payload, f"{prefix}denoise", default_denoise, 0.0, 1.0))
        _set_api_input(prompt, scheduler_id, "scheduler", str(payload.get(f"{prefix}scheduler") or "beta").strip() or "beta")
        _set_api_input(prompt, sampler_id, "sampler_name", str(payload.get(f"{prefix}sampler") or "euler").strip() or "euler")
        _set_api_input(prompt, noise_id, "noise_seed", _int_payload(payload, f"{prefix}seed", seed, 0, 0xFFFFFFFFFFFFFFFF))

    pass1_speed = _bool_payload(payload, "three_pass_pass1_te_speed", True)
    pass2_speed = _bool_payload(payload, "three_pass_pass2_te_speed", False)
    pass3_speed = _bool_payload(payload, "three_pass_pass3_te_speed", False)
    lora_name = _clean_lora_name(payload.get("three_pass_lightx_lora_name", "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"))
    if lora_name == _NONE_LORA:
        raise ValueError("A valid Multi-Pass LightX2V LoRA must be selected.")
    _require_model_choice("loras", lora_name, "MiniMax H3 three-pass LightX2V LoRA")
    lora_strength = _float_payload(payload, "three_pass_lightx_lora_strength", 0.5, -10.0, 10.0)
    _set_api_input(prompt, "187", "lora_name", lora_name)
    _set_api_input(prompt, "187", "strength_model", lora_strength)
    pass1_model_ref = ["328", 0] if pass1_speed else ["330", 0]
    _set_api_input(prompt, "328", "model", ["330", 0])
    _set_api_input(prompt, "248", "model", list(pass1_model_ref))
    _set_api_input(prompt, "246", "model", list(pass1_model_ref))

    pass_model_refs = []
    for pass_number, enabled, scheduler_id, guider_id, speed_node_id, lora_node_id in (
        (2, pass2_speed, "290", "294", "9202", "9204"),
        (3, pass3_speed, "344", "340", "9203", "9205"),
    ):
        pass_base_ref = [speed_node_id, 0] if enabled else ["330", 0]
        if enabled:
            prompt[speed_node_id] = {
                "class_type": "TESpeedMiniMaxH3",
                "inputs": {
                    "processing_control_value": 0.07,
                    "processing_percent_1": 0.1,
                    "processing_percent_2": 0.9,
                    "mcs": 2,
                    "device": "auto",
                    "cache_depth": 0.75,
                    "model": ["330", 0],
                },
                "_meta": {"title": f"TE-Speed-MiniMaxH3 Pass {pass_number}"},
            }
        prompt[lora_node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(pass_base_ref),
                "lora_name": lora_name,
                "strength_model": lora_strength,
            },
            "_meta": {"title": f"MiniMax H3 Pass {pass_number} LoRA"},
        }
        pass_model_refs.append((scheduler_id, guider_id, [lora_node_id, 0]))
    for scheduler_id, guider_id, model_ref in pass_model_refs:
        _set_api_input(prompt, scheduler_id, "model", list(model_ref))
        _set_api_input(prompt, guider_id, "model", list(model_ref))
    output_folder, filename_prefix = _minimax_h3_output_location(project_folder, scene_number)
    _set_api_input(prompt, "91", "filename_prefix", f"{filename_prefix}_stage1")
    _set_api_input(prompt, "299", "filename_prefix", f"{filename_prefix}_stage2")
    _set_api_input(prompt, "353", "filename_prefix", f"{filename_prefix}_stage3")
    latent_continuation_settings = _patch_minimax_h3_latent_continuation(prompt, payload)
    save_latent_settings = _patch_minimax_h3_save_latent(prompt, payload, timing)
    return {
        "workflow_path": workflow_path,
        "output_folder": output_folder,
        "prompt": prompt,
        "latent_continuation_settings": latent_continuation_settings,
        "save_latent_settings": save_latent_settings,
        "used_seed": seed,
        "audio_mode": "input_audio",
        "timing": timing.to_dict(),
        "prepared_audio": prepared_audio,
        "post_render_trim": {"start": timing.final_trim_start_seconds, "duration": timing.final_trim_duration_seconds, "frames": timing.final_frame_count},
        "reference_inputs": {"image_count": len(image_paths), "video_count": len(video_references)},
        "three_pass": {"pass1_steps": prompt["248"]["inputs"]["steps"], "pass2_steps": prompt["290"]["inputs"]["steps"], "pass3_steps": prompt["344"]["inputs"]["steps"]},
        "te_speed": {"pass1": pass1_speed, "pass2": pass2_speed, "pass3": pass3_speed},
    }
