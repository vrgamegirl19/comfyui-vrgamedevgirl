import importlib
import sys
import folder_paths

from .models import _MAX_LORA_SLOTS, _NONE_LORA, _lora_choices
from .image_workflows import _zimage_api_template_path
from .routes import _ensure_workflow_runner_routes


class VRGDG_MiniMaxH3TurboLoRACompat:
    """Apply the upstream Turbo LoRA with its missing pruned Ref2VA audio-row fix.

    The upstream v1.2.1 adapter derives only video/audio and visual-reference
    time rows for pruned MiniMax checkpoints. Reference audio adds another row
    in ComfyUI core, causing AdaLN's base projection to have three rows while
    the LoRA delta has two. Delegate whenever upstream includes audio-row
    support; otherwise reproduce its pruned path with the complete core layout.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "VRGDG/Compatibility"
    DESCRIPTION = (
        "MiniMax-H3 Turbo LoRA adapter with pruned-model reference-audio "
        "conditioning compatibility."
    )

    @staticmethod
    def _condition_times(upstream, timestep, payload, shift_v, shift_a):
        sigma_v = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
        t_video = 1.0 - sigma_v
        t_audio = 1.0 - upstream._time_shift_sigma(sigma_v, shift_v, shift_a)
        layout = payload.get("layout")
        if layout is not None:
            segments = getattr(layout, "segments", ()) or ()
            has_visual_condition = any(
                kind in ("cond", "ref_img") for _, _, kind in segments
            )
            has_audio_condition = any(
                kind == "ref_audio" for _, _, kind in segments
            )
        else:
            refs = payload.get("refs") or ()
            ref_kinds = {
                str(item.get("kind") or "")
                for item in refs
                if isinstance(item, dict)
            }
            has_visual_condition = bool(payload.get("keyframes")) or bool(
                ref_kinds.intersection({"image", "video", "video_audio"})
            )
            has_audio_condition = bool(
                ref_kinds.intersection({"audio", "video_audio"})
            )
        visual_aug = float(payload.get("visual_cond_noise_aug", 0.999))
        audio_aug = float(payload.get("audio_cond_noise_aug", 1.0))
        times = {t_video, t_audio}
        if has_visual_condition:
            times.add(max(t_video, visual_aug))
        if has_audio_condition:
            times.add(max(t_audio, audio_aug))
        return sorted(times)

    def apply_lora(self, model, lora_name, strength):
        import inspect
        import nodes as comfy_nodes

        upstream_class = (getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}).get(
            "MiniMaxH3TurboLoRA"
        )
        if upstream_class is None:
            raise RuntimeError(
                "MiniMaxH3TurboLoRA is not registered. Install or update "
                "ComfyUI-MiniMax-H3-Turbo, then restart ComfyUI."
            )
        upstream = sys.modules.get(upstream_class.__module__)
        if upstream is None:
            upstream = importlib.import_module(upstream_class.__module__)
        upstream_node = upstream_class()
        unique_t = getattr(upstream, "_unique_t", None)
        upstream_supports_audio = False
        if callable(unique_t):
            try:
                upstream_supports_audio = "has_aud_cond" in inspect.signature(unique_t).parameters
            except (TypeError, ValueError):
                upstream_supports_audio = False

        diffusion_model = model.model.diffusion_model
        pruned = bool(getattr(diffusion_model, "use_adaln_curves", False))
        if not pruned or upstream_supports_audio:
            return upstream_node.apply_lora(model, lora_name, strength)

        required_helpers = (
            "_apply_bypass_lora",
            "_egrid",
            "_interp_egrid",
            "_add_dbg_wrapper",
            "_time_shift_sigma",
        )
        missing_helpers = [name for name in required_helpers if not hasattr(upstream, name)]
        make_adaln_forward = getattr(upstream, "_make_adaln_forward", None)
        legacy_adaln_delta = getattr(upstream, "_AdalnDelta", None)
        if not callable(make_adaln_forward) and legacy_adaln_delta is None:
            missing_helpers.append("_make_adaln_forward (or legacy _AdalnDelta)")
        if missing_helpers:
            raise RuntimeError(
                "The installed ComfyUI-MiniMax-H3-Turbo version is incompatible "
                "with Builder reference-audio support. Missing helpers: "
                + ", ".join(missing_helpers)
                + ". Update the Turbo extension and restart ComfyUI."
            )

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise RuntimeError(f"MiniMax-H3 Turbo LoRA was not found: {lora_name}")
        lora = upstream.comfy.utils.load_torch_file(lora_path, safe_load=True)
        modules = sorted({key.rsplit(".lora_", 1)[0] for key in lora})
        new_model = model.clone()
        backbone = [name for name in modules if "adaln_proj" not in name]
        adaln = [name for name in modules if "adaln_proj" in name]
        bound = upstream._apply_bypass_lora(new_model, lora, backbone, strength)

        embedding_grid = upstream._egrid()
        shared = {"silu_temb": None}
        shift_v = float(getattr(diffusion_model, "sigma_shift_video", upstream.SHIFT_V))
        shift_a = float(getattr(diffusion_model, "sigma_shift_audio", upstream.SHIFT_A))

        def wrap(executor, *args, **kwargs):
            timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
            context = args[2] if len(args) > 2 else kwargs.get("context")
            payload = kwargs.get("minimax_payload") or {}
            times = self._condition_times(upstream, timestep, payload, shift_v, shift_a)
            shared["silu_temb"] = upstream._interp_egrid(
                times,
                embedding_grid,
                context.device,
                context.dtype,
            )
            return executor(*args, **kwargs)

        new_model.add_wrapper_with_key(
            upstream.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "vrgdg_h3turbo_ref_audio",
            wrap,
        )
        for name in adaln:
            lora_a = lora[name + ".lora_A.weight"]
            lora_b = lora[name + ".lora_B.weight"] * strength
            model_key = "diffusion_model." + name.rsplit(".linear", 1)[0]
            base_adaln = new_model.get_model_object(model_key)
            if callable(make_adaln_forward):
                # Turbo v1.2.2+ patches only the forward attribute so ComfyUI's
                # dynamic-VRAM unload keeps the original module tree intact.
                new_model.add_object_patch(
                    model_key + ".forward",
                    make_adaln_forward(base_adaln, lora_a, lora_b, shared),
                )
            else:
                # Retain compatibility with the earlier upstream API that
                # exposed a whole-module AdaLN wrapper.
                new_model.add_object_patch(
                    model_key,
                    legacy_adaln_delta(base_adaln, lora_a, lora_b, shared),
                )
        print(
            "[VRGDG MiniMaxH3TurboLoRACompat] pruned base: "
            f"{bound} backbone adapters + {len(adaln)} AdaLN adapters; "
            "reference-audio time rows enabled",
            flush=True,
        )
        dbg_wrapper = upstream._add_dbg_wrapper
        try:
            dbg_has_mode = "mode" in inspect.signature(dbg_wrapper).parameters
        except (TypeError, ValueError):
            dbg_has_mode = False
        if dbg_has_mode:
            dbg_wrapper(
                new_model,
                diffusion_model,
                "pruned-ref-audio-compat",
                "bypass",
            )
        else:
            dbg_wrapper(new_model, diffusion_model, "pruned-ref-audio-compat")
        return (new_model,)


class VRGDG_ZImageWorkflowRunnerUI:
    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = _lora_choices()
        required = {
            "workflow_path": ("STRING", {"default": _zimage_api_template_path()}),
            "save_folder": ("STRING", {"default": "VRGDG_WorkflowRunner_Saved"}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "first_pass_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
            "first_pass_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
            "second_pass_width": ("INT", {"default": 1920, "min": 64, "max": 4096, "step": 8}),
            "second_pass_height": ("INT", {"default": 1080, "min": 64, "max": 4096, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            "use_custom_loras": ("BOOLEAN", {"default": False}),
            "lora_count": ("INT", {"default": 0, "min": 0, "max": _MAX_LORA_SLOTS, "step": 1}),
            "ltx_two_pass_mode": ("BOOLEAN", {"default": False}),
        }
        for slot in range(1, _MAX_LORA_SLOTS + 1):
            required[f"lora_{slot}"] = (lora_choices, {"default": _NONE_LORA})
            required[f"first_pass_strength_{slot}"] = (
                "FLOAT",
                {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01},
            )
            required[f"second_pass_strength_{slot}"] = (
                "FLOAT",
                {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
            )
        return {"required": required}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Canvas UI for running the bundled Z-Image text-to-image workflow template without opening it."

    def noop(self, **kwargs):
        return ("Open the Z-Image workflow runner UI and press Run Image Workflow.",)


class VRGDG_ClearMemoryButtonUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Small canvas button that queues the bundled ClearMemory_API workflow."

    def noop(self):
        return ("Press Clear Memory to run the bundled ClearMemory_API workflow.",)


_ensure_workflow_runner_routes()


NODE_CLASS_MAPPINGS = {
    "VRGDG_MiniMaxH3TurboLoRACompat": VRGDG_MiniMaxH3TurboLoRACompat,
    "VRGDG_ZImageWorkflowRunnerUI": VRGDG_ZImageWorkflowRunnerUI,
    "VRGDG_ClearMemoryButtonUI": VRGDG_ClearMemoryButtonUI,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MiniMaxH3TurboLoRACompat": "VRGDG MiniMax-H3 Turbo LoRA Compatibility",
    "VRGDG_ZImageWorkflowRunnerUI": "VRGDG Z-Image Workflow Runner UI",
    "VRGDG_ClearMemoryButtonUI": "VRGDG Clear Memory Button",
}
