"""MiniMax H3 graph patches: Sage attention, latent save and continuation, advanced settings, turbo, LoRAs and optional models."""

import copy
import importlib
import os
import sys
from ..minimax.latent_manager import (
    SceneLatentManager,
    normalize_masked_context_frames,
    plan_masked_context,
)

from .paths import _bool_payload, _first_payload_value, _float_payload, _int_payload
from .models import _NONE_LORA, _clean_lora_name, _model_choice_exists
from .api_graph import _api_node_id_by_class, _find_final_vae_decode_id, _get_comfy_node_mappings, _optional_api_node_id_by_class, _replace_api_input_refs, _set_api_input


_MINIMAX_H3_SAGE_ATTENTION_MODES = {
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
}


def _require_minimax_h3_memory_efficient_sage_attention():
    try:
        from importlib import metadata

        try:
            installed_version = metadata.version("sageattention")
        except metadata.PackageNotFoundError:
            installed_version = "not installed"
        core = importlib.import_module("sageattention.core")
        get_arch_versions = getattr(core, "get_cuda_arch_versions", None)
        if not callable(get_arch_versions):
            raise RuntimeError("sageattention.core.get_cuda_arch_versions is missing")
        arch_versions = get_arch_versions()
        if not arch_versions:
            raise RuntimeError("CUDA architecture detection returned no architectures")
        supported_arches = {"sm75", "sm80", "sm86", "sm89", "sm90", "sm120", "sm121"}
        detected_arch = str(arch_versions[0])
        if detected_arch not in supported_arches:
            raise RuntimeError(f"detected GPU architecture {detected_arch} is not supported by this KJNodes patch")
    except Exception as exc:
        python_executable = sys.executable
        raise RuntimeError(
            "MiniMax H3 memory-efficient Sage Attention is enabled, but SageAttention is missing or incompatible.\n"
            f"Detected SageAttention version: {installed_version if 'installed_version' in locals() else 'unknown'}.\n"
            "KJNodes requires SageAttention 2.2.0 or newer with CUDA architecture detection.\n\n"
            "Install it with ComfyUI's embedded Python, then restart ComfyUI:\n"
            f'\"{python_executable}\" -m pip install sageattention==2.2.0 --no-build-isolation\n\n'
            "To continue without this optimization, turn off the MiniMax H3 memory-efficient Sage Attention checkbox.\n"
            f"Technical detail: {exc}"
        ) from exc


def _patch_minimax_h3_save_latent(prompt, payload, timing=None):
    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_text or not os.path.isdir(project_text):
        return {"enabled": False, "reason": "No valid project folder"}
    project_folder = os.path.abspath(project_text)
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)

    decode_id = _find_final_vae_decode_id(prompt)
    if not decode_id or decode_id not in prompt:
        return {"enabled": False, "reason": "VAEDecode node not found in prompt"}

    samples_input = prompt[decode_id].get("inputs", {}).get("samples")
    if not samples_input or not isinstance(samples_input, list) or len(samples_input) < 2:
        return {"enabled": False, "reason": "VAEDecode samples input not found or invalid"}

    upstream_node_id = str(samples_input[0])
    if upstream_node_id not in prompt:
        return {"enabled": False, "reason": f"Upstream latent source node '{upstream_node_id}' not found in prompt"}

    tail_padding = _minimax_h3_tail_padding_frames(timing) if timing is not None else None

    save_id = "9850"
    while save_id in prompt:
        save_id = str(int(save_id) + 1)

    prompt[save_id] = {
        "class_type": "VRGDG_MiniMaxH3SaveLatent",
        "inputs": {
            "latent": list(samples_input),
            "project_folder": project_folder,
            "scene_number": scene_number,
            "frame_count": 0,
            "fps": 24.0,
            "tail_padding_frames": tail_padding if tail_padding is not None else -1,
        },
        "_meta": {
            "title": f"Universal Latent Auto-Save (Scene {scene_number:03d})",
        },
    }
    prompt[decode_id]["inputs"]["samples"] = [save_id, 0]

    return {
        "enabled": True,
        "save_node_id": save_id,
        "project_folder": project_folder,
        "scene_number": scene_number,
    }


_MMH3_LATENT_MASKED_MODE = "latent_continuation_masked"


_MMH3_RETIRED_LATENT_MODES = (
    "latent", "latent_continuation", "continuation",
    "latent_exact", "latent_exact_frame", "latent_continuation_exact", "latent_continuation_exact_frame",
)


# The conditioning nodes masked continuation supports. Image + Reference is excluded on purpose.
_MMH3_CONDITIONING_CLASSES = (
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3ImageToVideo",
)


def _minimax_h3_latent_continuation_mode(payload):
    mode = str(
        payload.get("minimax_h3_continuity_mode")
        or payload.get("continuity_mode")
        or payload.get("continuityMode")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    # The standard and exact-last-frame modes were retired. Saved projects that still name them continue masked.
    if mode in _MMH3_RETIRED_LATENT_MODES or mode in ("latent_masked", "latent_masked_av"):
        return _MMH3_LATENT_MASKED_MODE
    return mode


def _minimax_h3_is_latent_mode(mode):
    return mode == _MMH3_LATENT_MASKED_MODE


def _minimax_h3_masked_latent_plan(payload):
    """The masked-continuation window of the predecessor's saved latent, or None when it cannot be planned."""
    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    if scene_number <= 1 or not project_text or not os.path.isdir(project_text):
        return None
    info = SceneLatentManager.get_latent_info(os.path.abspath(project_text), scene_number - 1)
    if not info.get("exists") or int(info.get("token_count") or 0) <= 0:
        return None
    try:
        return plan_masked_context(
            int(info["token_count"]), info.get("tail_padding_frames"), _minimax_h3_latent_context_frames_setting(payload)
        )
    except ValueError:
        return None


def _minimax_h3_tail_padding_frames(timing):
    """Frames at the end of this render that lie after the scene's visible last frame."""
    data = timing.to_dict() if hasattr(timing, "to_dict") else dict(timing or {})
    try:
        visible = round(float(data["actual_warmup_seconds"]) * 24) + int(data["final_frame_count"])
        return max(0, int(data["h3_frame_count"]) - int(visible))
    except (KeyError, TypeError, ValueError):
        return None


def _minimax_h3_latent_context_frames_setting(payload):
    raw_cf = payload.get("minimax_h3_latent_context_frames") or payload.get("latent_context_frames") or payload.get("context_frames") or 22
    try:
        context_frames = int(raw_cf)
    except (ValueError, TypeError):
        context_frames = 22
    return normalize_masked_context_frames(context_frames)


def _minimax_h3_cooldown_frames(payload):
    """Cool-down frames for the timing plan. Masked continuation plans its own tail, so the setting does not apply."""
    if _minimax_h3_is_latent_mode(_minimax_h3_latent_continuation_mode(payload)):
        return 0
    return _first_payload_value(payload, "cooldown_frames", "tail_loss_frames", default=0)


def _minimax_h3_effective_warmup_frames(payload):
    """Warm-up frames for the timing plan.

    Latent Continuation anchors the predecessor's trailing frames at frame 0 of the
    render, so those frames (and the audio that played over them) must be a warm-up
    in front of the scene. The timing plan then starts the audio that much earlier
    and trims the same span off the finished video AND audio together. Without this
    the context frames sit on top of the scene's own first lyrics and the video
    drifts out of sync with the audio.
    """
    requested = _first_payload_value(payload, "warmup_frames", "pre_frames", default=0)
    try:
        requested = max(0, int(float(requested or 0)))
    except (TypeError, ValueError):
        requested = 0
    mode = _minimax_h3_latent_continuation_mode(payload)
    if not _minimax_h3_is_latent_mode(mode):
        return requested
    # Masked continuation sets its own warm-up, so the Render settings warmup frames do not apply.
    requested = 0
    if _int_payload(payload, "scene_number", 1, 1, 999999) <= 1:
        return requested
    # the copied window plus the predecessor frames after it, so audio and picture stay on the same timeline
    masked_plan = _minimax_h3_masked_latent_plan(payload)
    return max(requested, int(masked_plan["warmup_frames"]) if masked_plan else _minimax_h3_latent_context_frames_setting(payload))


def _patch_minimax_h3_latent_continuation_masked(prompt, payload):
    """Latent Continuation Masked: the predecessor's latent becomes the protected head of the sampled latent.

    A Load Latent node slices a phase-aligned window of the predecessor, and an Apply Masked Continuation node
    copies it into the first tokens of the sampler's input latent with a zero denoise mask. Nothing is added to
    the conditioning. The head is trimmed off through the timing plan's warm-up, like the other latent modes.

    Single pass has one sampler. 2 Pass has a second sampler fed by the learned upscale, which resets the video
    mask on purpose, so the same window is applied again to that sampler's latent. The predecessor is saved at the
    pass 2 size, so the second head is the exact predecessor, not the resized copy pass 1 had to use. 2 Pass
    Advanced is not supported: it samples its second pass inside the MMH3 Ultimate Upscale node, which builds its
    own masks and would resample the head.
    """
    if any(node.get("class_type") == "MMH3UltimateUpscale" for node in prompt.values()):
        raise ValueError(
            "Latent Continuation Masked is not available in 2 Pass Advanced. Use Single pass or 2 Pass, or set continuity to off."
        )
    project_text = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_text or not os.path.isdir(project_text):
        return {"enabled": False, "reason": f"Project folder not found: {project_text}"}
    project_folder = os.path.abspath(project_text)

    scene_number = _int_payload(payload, "scene_number", 1, 1, 999999)
    if scene_number <= 1:
        return {
            "enabled": False,
            "scene_number": scene_number,
            "reason": "Scene 1 is the opening scene; no predecessor latent needed",
        }

    pred_scene = scene_number - 1
    if not SceneLatentManager.latent_exists(project_folder, pred_scene):
        raise FileNotFoundError(
            f"Latent Continuation Masked for Scene {scene_number:03d} requires Scene {pred_scene:03d} latent, "
            f"but '{SceneLatentManager.get_path(project_folder, pred_scene)}' was not found. "
            f"Render Scene {pred_scene:03d} first."
        )

    samplers = {
        str(node_id): node
        for node_id, node in prompt.items()
        if node.get("class_type") == "SamplerCustomAdvanced"
    }
    replace_ids = {
        str(node_id) for node_id, node in prompt.items()
        if node.get("class_type") == "VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent"
    }

    def reads_upscaled_latent(node):
        source = node.get("inputs", {}).get("latent_image")
        return isinstance(source, list) and len(source) == 2 and str(source[0]) in replace_ids

    second_pass = {sid: node for sid, node in samplers.items() if reads_upscaled_latent(node)}
    first_pass = {sid: node for sid, node in samplers.items() if sid not in second_pass}
    if len(first_pass) != 1 or len(second_pass) > 1:
        raise ValueError(
            "Latent Continuation Masked supports Single pass and 2 Pass renders "
            f"(found {len(samplers)} samplers it cannot map to those). Use Latent Continuation for this render."
        )
    sampler_id, sampler = next(iter(first_pass.items()))
    latent_source = sampler.get("inputs", {}).get("latent_image")
    if not isinstance(latent_source, list) or len(latent_source) != 2:
        return {"enabled": False, "reason": "Sampler latent input not found"}

    context_frames = _minimax_h3_latent_context_frames_setting(payload)
    info = SceneLatentManager.get_latent_info(project_folder, pred_scene)
    plan = plan_masked_context(int(info.get("token_count") or 0), info.get("tail_padding_frames"), context_frames)

    # With Audio Drive the song audio is already locked into the latent. Built-in audio keeps the predecessor's.
    source_node = prompt.get(str(latent_source[0]), {})
    include_audio = source_node.get("class_type") != "VRGDG_MiniMaxH3AudioDrive"

    # The head is resized through pictures when the predecessor was saved at another size, which needs the video VAE.
    conditioning_nodes = [
        node for node in prompt.values()
        if node.get("class_type") in _MMH3_CONDITIONING_CLASSES
    ]
    video_vae = next(
        (
            node["inputs"]["vae"] for node in conditioning_nodes
            if isinstance(node.get("inputs", {}).get("vae"), list)
        ),
        None,
    )
    # Image to Video pins the scene image as a keyframe at frame 0. The protected head already
    # holds the opening frames, so that keyframe would fight it. Later scenes open on the predecessor instead.
    dropped_first_frame = False
    for node in conditioning_nodes:
        if node.get("class_type") != "MiniMaxH3ReferenceToVideo" and node.get("inputs", {}).pop("first_frame", None) is not None:
            dropped_first_frame = True

    load_id = "9210"
    while load_id in prompt:
        load_id = str(int(load_id) + 1)
    apply_id = str(int(load_id) + 1)
    while apply_id in prompt:
        apply_id = str(int(apply_id) + 1)

    prompt[load_id] = {
        "class_type": "VRGDG_MiniMaxH3LoadLatent",
        "inputs": {
            "project_folder": project_folder,
            "scene_number": pred_scene,
            "context_frames": context_frames,
            "exact_frame_mode": False,
            "masked_av": True,
        },
        "_meta": {"title": f"Predecessor Latent Context (Scene {pred_scene:03d} · {context_frames} frames · masked)"},
    }
    prompt[apply_id] = {
        "class_type": "VRGDG_MiniMaxH3ApplyMaskedContinuation",
        "inputs": {
            "latent": latent_source,
            "context_latent": [load_id, 0],
            "include_audio": include_audio,
            **({"vae": list(video_vae)} if video_vae else {}),
        },
        "_meta": {"title": f"MiniMax H3 Latent Continuation Masked ({plan['context_frames']} frames)"},
    }
    sampler["inputs"]["latent_image"] = [apply_id, 0]

    second_apply_id = None
    if second_pass:
        second_sampler_id, second_sampler = next(iter(second_pass.items()))
        second_apply_id = str(int(apply_id) + 1)
        while second_apply_id in prompt:
            second_apply_id = str(int(second_apply_id) + 1)
        prompt[second_apply_id] = {
            "class_type": "VRGDG_MiniMaxH3ApplyMaskedContinuation",
            "inputs": {
                "latent": second_sampler["inputs"]["latent_image"],
                "context_latent": [load_id, 0],
                # the audio is the locked source track in every 2 pass graph, its mask comes through the upscale
                "include_audio": False,
                **({"vae": list(video_vae)} if video_vae else {}),
            },
            "_meta": {"title": f"MiniMax H3 Latent Continuation Masked · Pass 2 ({plan['context_frames']} frames)"},
        }
        second_sampler["inputs"]["latent_image"] = [second_apply_id, 0]

    return {
        "enabled": True,
        "mode": _MMH3_LATENT_MASKED_MODE,
        "predecessor_scene": pred_scene,
        "context_frames": plan["context_frames"],
        "warmup_frames": _minimax_h3_effective_warmup_frames(payload),
        "load_node_id": load_id,
        "apply_node_id": apply_id,
        "second_pass_apply_node_id": second_apply_id,
        "sampler_node_id": sampler_id,
        "include_audio": include_audio,
        "lost_tail_frames": plan["lost_tail_frames"],
        "tail_padding_known": info.get("tail_padding_frames") is not None,
        "dropped_first_frame": dropped_first_frame,
        "trim_node_id": None,
    }


def _patch_minimax_h3_latent_continuation(prompt, payload):
    if _minimax_h3_latent_continuation_mode(payload) != _MMH3_LATENT_MASKED_MODE:
        return {"enabled": False, "reason": "Continuity mode is not latent_continuation_masked"}
    return _patch_minimax_h3_latent_continuation_masked(prompt, payload)


def _patch_minimax_h3_advanced_settings(prompt, payload):
    sampler_id = _api_node_id_by_class(prompt, "KSamplerSelect", fallback="123")
    scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
    loader_id = _api_node_id_by_class(prompt, "DiffusionModelLoaderKJ", fallback="141")
    easy_cache_id = _optional_api_node_id_by_class(prompt, "EasyCache", fallback_ids=("174",))

    sampler_name = str(payload.get("sampler_name") or "res_multistep").strip() or "res_multistep"
    scheduler = str(payload.get("scheduler") or "simple").strip() or "simple"
    steps = _int_payload(payload, "steps", 20, 1, 1000)
    denoise = _float_payload(payload, "denoise", 1.0, 0.0, 1.0)
    easy_cache_bypass = _bool_payload(payload, "easy_cache_bypass", False)
    easy_cache_reuse_threshold = _float_payload(payload, "easy_cache_reuse_threshold", 0.3, 0.0, 1.0)
    easy_cache_start_percent = _float_payload(payload, "easy_cache_start_percent", 0.2, 0.0, 1.0)
    easy_cache_end_percent = _float_payload(payload, "easy_cache_end_percent", 0.9, 0.0, 1.0)
    easy_cache_verbose = _bool_payload(payload, "easy_cache_verbose", False)
    sage_attention = str(payload.get("sage_attention") or "auto").strip()
    if sage_attention not in _MINIMAX_H3_SAGE_ATTENTION_MODES:
        sage_attention = "auto"
    enable_fp16_accumulation = _bool_payload(payload, "enable_fp16_accumulation", True)

    _set_api_input(prompt, sampler_id, "sampler_name", sampler_name)
    _set_api_input(prompt, scheduler_id, "scheduler", scheduler)
    _set_api_input(prompt, scheduler_id, "steps", steps)
    _set_api_input(prompt, scheduler_id, "denoise", denoise)
    _set_api_input(prompt, loader_id, "sage_attention", sage_attention)
    _set_api_input(prompt, loader_id, "enable_fp16_accumulation", enable_fp16_accumulation)

    if easy_cache_id:
        _set_api_input(prompt, easy_cache_id, "reuse_threshold", easy_cache_reuse_threshold)
        _set_api_input(prompt, easy_cache_id, "start_percent", easy_cache_start_percent)
        _set_api_input(prompt, easy_cache_id, "end_percent", easy_cache_end_percent)
        _set_api_input(prompt, easy_cache_id, "verbose", easy_cache_verbose)
        if easy_cache_bypass:
            _replace_api_input_refs(prompt, (easy_cache_id, 0), (loader_id, 0))
            prompt.pop(easy_cache_id, None)

    return {
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "steps": steps,
        "denoise": denoise,
        "easy_cache_bypass": easy_cache_bypass,
        "easy_cache_reuse_threshold": easy_cache_reuse_threshold,
        "easy_cache_start_percent": easy_cache_start_percent,
        "easy_cache_end_percent": easy_cache_end_percent,
        "easy_cache_verbose": easy_cache_verbose,
        "sage_attention": sage_attention,
        "enable_fp16_accumulation": enable_fp16_accumulation,
    }


def _patch_minimax_h3_turbo(prompt, payload):
    """Apply the legacy MiniMax Turbo LoRA through ComfyUI's native loader.

    This setting predates the separate MiniMax-H3-Turbo custom-node project.
    It is intentionally only a LoRA toggle; it must not require or inject an
    external Turbo sampler/solver.
    """
    enabled = _bool_payload(payload, "use_turbo_lora", False)
    if not enabled:
        return {
            "enabled": False,
            "lora_name": "",
            "strength": 0.0,
            "scheduler": "",
            "steps": 0,
        }

    lora_name = str(
        payload.get("turbo_lora_name") or "minimax_h3_turbo_4step_ema_ckpt850.safetensors"
    ).strip()
    if not lora_name:
        raise ValueError("MiniMax-H3 Turbo is enabled, but no Turbo LoRA file is selected.")
    if not _model_choice_exists("loras", lora_name):
        raise ValueError(
            f"MiniMax-H3 Turbo LoRA '{lora_name}' was not found in ComfyUI/models/loras. "
            "Download the LoRA, refresh/restart ComfyUI, and select it in MiniMax Video Settings."
        )
    strength = _float_payload(payload, "turbo_lora_strength", 1.0, -10.0, 10.0)
    scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
    guider_id = _api_node_id_by_class(prompt, "BasicGuider", fallback="126")
    scheduler_inputs = prompt.get(scheduler_id, {}).get("inputs", {})
    model_ref = scheduler_inputs.get("model")
    if not isinstance(model_ref, list) or len(model_ref) != 2:
        raise ValueError("MiniMax-H3 Turbo could not find the current model connection feeding BasicScheduler.")

    turbo_lora_id = "9001"
    while turbo_lora_id in prompt:
        turbo_lora_id = str(int(turbo_lora_id) + 1)
    prompt[turbo_lora_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": list(model_ref),
            "lora_name": lora_name,
            "strength_model": strength,
        },
    }
    _set_api_input(prompt, scheduler_id, "model", [turbo_lora_id, 0])
    _set_api_input(prompt, guider_id, "model", [turbo_lora_id, 0])

    return {
        "enabled": True,
        "lora_name": lora_name,
        "strength": strength,
        "scheduler": "",
        "steps": 0,
        "lora_node": "LoraLoaderModelOnly",
        "sampler_node": "",
    }


def _patch_minimax_h3_memory_efficient_sage_attention(prompt, payload):
    enabled = _bool_payload(payload, "use_memory_efficient_sage_attention", False)
    if not enabled:
        return {"enabled": False, "node": ""}

    scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
    guider_id = _api_node_id_by_class(prompt, "BasicGuider", fallback="126")
    model_ref = prompt.get(scheduler_id, {}).get("inputs", {}).get("model")
    if not isinstance(model_ref, list) or len(model_ref) != 2:
        raise ValueError("MiniMax H3 memory-efficient Sage Attention patch could not find the current model connection.")

    node_id = "9201"
    while node_id in prompt:
        node_id = str(int(node_id) + 1)
    prompt[node_id] = {
        "class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "inputs": {"model": list(model_ref)},
        "_meta": {"title": "MiniMax H3 Mem Eff Sage Attention Patch"},
    }
    patched_ref = [node_id, 0]
    _set_api_input(prompt, scheduler_id, "model", patched_ref)
    _set_api_input(prompt, guider_id, "model", patched_ref)
    return {"enabled": True, "node": node_id}


def _patch_minimax_h3_loras(prompt, payload):
    enabled = _bool_payload(payload, "use_loras", False) or _bool_payload(payload, "use_custom_loras", False)
    if not enabled:
        return {
            "enabled": False,
            "count": 0,
            "loras": [],
        }
    if _bool_payload(payload, "use_turbo_lora", False):
        raise ValueError("MiniMax normal LoRAs and MiniMax-H3 Turbo LoRA cannot be enabled at the same time.")

    raw_loras = payload.get("loras")
    configured = []
    if isinstance(raw_loras, list):
        for item in raw_loras:
            if not isinstance(item, dict):
                continue
            configured.append({
                "name": _clean_lora_name(item.get("name") or item.get("lora_name") or item.get("loraName") or _NONE_LORA),
                "strength": _float_payload(item, "strength", 1.0, -10.0, 10.0),
            })
    lora_count = _int_payload(payload, "lora_count", len(configured), 0, 4)
    if not configured:
        for slot in range(1, lora_count + 1):
            configured.append({
                "name": _clean_lora_name(payload.get(f"lora_{slot}", _NONE_LORA)),
                "strength": _float_payload(payload, f"lora_{slot}_strength", 1.0, -10.0, 10.0),
            })
    configured = [
        item for item in configured[:lora_count]
        if item["name"] and item["name"] != _NONE_LORA
    ]
    if not configured:
        return {
            "enabled": False,
            "count": 0,
            "loras": [],
        }
    for item in configured:
        if not _model_choice_exists("loras", item["name"]):
            raise ValueError(
                f"MiniMax LoRA '{item['name']}' was not found in ComfyUI/models/loras. "
                "Download the LoRA, refresh/restart ComfyUI, and select it in MiniMax Video Settings."
            )

    scheduler_id = _api_node_id_by_class(prompt, "BasicScheduler", fallback="124")
    guider_id = _api_node_id_by_class(prompt, "BasicGuider", fallback="126")
    scheduler_inputs = prompt.get(scheduler_id, {}).get("inputs", {})
    model_ref = scheduler_inputs.get("model")
    if not isinstance(model_ref, list) or len(model_ref) != 2:
        raise ValueError("MiniMax LoRA patch could not find the current model connection feeding BasicScheduler.")

    next_id = 9101
    current_ref = list(model_ref)
    applied = []
    for index, item in enumerate(configured, start=1):
        while str(next_id) in prompt:
            next_id += 1
        node_id = str(next_id)
        next_id += 1
        prompt[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(current_ref),
                "lora_name": item["name"],
                "strength_model": item["strength"],
            },
            "_meta": {
                "title": f"MiniMax LoRA {index}",
            },
        }
        current_ref = [node_id, 0]
        applied.append({
            "name": item["name"],
            "strength": item["strength"],
            "node": node_id,
        })

    _set_api_input(prompt, scheduler_id, "model", list(current_ref))
    _set_api_input(prompt, guider_id, "model", list(current_ref))
    return {
        "enabled": True,
        "count": len(applied),
        "loras": applied,
    }


def _patch_minimax_h3_optional_model_paths(prompt, model_refs, use_feedforward, use_block_sparse_attention):
    if use_feedforward or use_block_sparse_attention:
        try:
            mappings = _get_comfy_node_mappings()
        except Exception as exc:
            raise ValueError(
                "Could not inspect ComfyUI node registrations for the optional MiniMax H3 model patches. "
                "Restart ComfyUI after installing or updating their nodes."
            ) from exc
        required_nodes = []
        if use_feedforward:
            required_nodes.append("MiniMaxChunkFeedForward")
        if use_block_sparse_attention:
            required_nodes.append("BlockSparseAttention")
        missing_nodes = [name for name in required_nodes if name not in mappings]
        if missing_nodes:
            raise ValueError(
                "MiniMax H3 optional model patches are enabled, but these nodes are missing: "
                + ", ".join(missing_nodes)
                + ". Install/update ComfyUI-KJNodes and ComfyUI as needed, then restart ComfyUI."
            )

    if use_block_sparse_attention:
        sparse_input_types = mappings["BlockSparseAttention"].INPUT_TYPES()
        sparse_inputs = {
            **sparse_input_types.get("required", {}),
            **sparse_input_types.get("optional", {}),
        }
        selection_spec = sparse_inputs.get("selection", ())
        selection_options = selection_spec[1].get("options", []) if len(selection_spec) > 1 else []
        selection_keys = {option["key"] for option in selection_options}
        # DynamicCombo keys changed when Block Sparse Attention became Model Sparse Attention.
        sparse_selection = next(
            (key for key in ("sol-attn", "Sol-Attn (adaptive tau)") if key in selection_keys),
            None,
        )
        if sparse_selection is None:
            raise ValueError(
                "The installed BlockSparseAttention node does not advertise a supported Sol-Attn method. "
                "Disable Use Block Sparse Attention or update the VRGDG nodes for this ComfyUI version."
            )

    optional_patch_nodes = []
    next_lora_node_id = 9202

    def add_optional_model_patch(model_ref, class_type, inputs, target, title):
        nonlocal next_lora_node_id
        while str(next_lora_node_id) in prompt:
            next_lora_node_id += 1
        node_id = str(next_lora_node_id)
        next_lora_node_id += 1
        prompt[node_id] = {
            "class_type": class_type,
            "inputs": {"model": list(model_ref), **inputs},
            "_meta": {"title": f"{title} - {target}"},
        }
        optional_patch_nodes.append({"node": node_id, "class_type": class_type, "target": target})
        return [node_id, 0]

    for target, model_ref in model_refs.items():
        if use_feedforward:
            model_ref = add_optional_model_patch(
                model_ref,
                "MiniMaxChunkFeedForward",
                {"chunks": 8, "seq_threshold": 4096},
                target,
                "MiniMax H3 Chunk FeedForward",
            )
        if use_block_sparse_attention:
            model_ref = add_optional_model_patch(
                model_ref,
                "BlockSparseAttention",
                {
                    "selection": sparse_selection,
                    "selection.tau": 1.3,
                    "start_percent": 0.2,
                    "end_percent": 1.0,
                    "dense_blocks": "",
                    "min_tokens": 12288,
                    "extra_tokens": 256,
                    "sink_conditioning": "exact_kv_and_rows",
                    "verbose": False,
                },
                target,
                "Block Sparse Attention",
            )
        model_refs[target] = model_ref
    return optional_patch_nodes


def _patch_minimax_h3_te_speed(prompt, model_ref, payload, node_id):
    prompt[node_id] = {
        "class_type": "TESpeedMiniMaxH3",
        "inputs": {
            "model": list(model_ref),
            "processing_control_value": _float_payload(payload, "te_speed_processing_control", 0.07, 0.0, 1.0),
            "processing_percent_1": _float_payload(payload, "te_speed_start_percent", 0.1, 0.0, 1.0),
            "processing_percent_2": _float_payload(payload, "te_speed_end_percent", 0.9, 0.0, 1.0),
            "mcs": _int_payload(payload, "te_speed_mcs", 2, 1, 64),
            "cache_depth": _float_payload(payload, "te_speed_cache_depth", 0.75, 0.0, 1.0),
            "device": str(payload.get("te_speed_device") or "auto"),
        },
        "_meta": {"title": "TE-Speed MiniMax H3"},
    }
    return [node_id, 0]


def _patch_minimax_h3_fast_decode(prompt):
    if "H3FastVAEDecode" not in _get_comfy_node_mappings():
        raise ValueError("Fast VAE decode requires H3FastVAEDecode. Update VRGameDevGirl nodes and restart ComfyUI.")
    decoder_inputs = prompt["122"]["inputs"]
    prompt["122"] = {
        "class_type": "H3FastVAEDecode",
        "inputs": {
            "samples": copy.deepcopy(decoder_inputs["samples"]),
            "vae": copy.deepcopy(decoder_inputs["vae"]),
            "tile_batch_size": 8,
        },
        "_meta": {"title": "H3 VAE Decode Fast - Final Output"},
    }
