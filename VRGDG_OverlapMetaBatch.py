"""Overlap-aware processing built around VideoHelperSuite's native meta batch.

``VRGDGOverlapPreset`` supplies the stride to VHS's real ``VHS_BatchManager``.
``VRGDGOverlapWindow`` decorates that manager with overlap state and prepends
the carried source tail. ``VRGDGOverlapBlend`` crossfades the duplicated
processed frames and emits each source frame exactly once.
"""

from __future__ import annotations

import math

import torch


BIGMAX = 2**31 - 1

PRESETS = {
    "H3 Near 41 (39 / 5)": (39, 5),
    "H3 Balanced (73 / 17)": (73, 17),
    "H3 Trained (124 / 22)": (124, 22),
}
CUSTOM_PRESET = "Custom H3 (17n+5)"
BLEND_MODES = ("cosine", "smoothstep", "linear")


def resolve_h3_settings(preset: str, custom_window: int, custom_overlap: int):
    if preset in PRESETS:
        window, overlap = PRESETS[preset]
    elif preset == CUSTOM_PRESET:
        window, overlap = int(custom_window), int(custom_overlap)
    else:
        raise ValueError(f"Unknown overlap preset: {preset!r}")

    if window < 5 or (window - 5) % 17:
        raise ValueError(
            f"MiniMax H3 window must follow 17n+5 (5, 22, 39, 56, 73, ...); got {window}."
        )
    if overlap < 1:
        raise ValueError("Overlap must be at least one frame.")
    if overlap * 2 >= window:
        raise ValueError(
            f"Overlap must be less than half of the window so each batch emits frames; "
            f"got window={window}, overlap={overlap}."
        )
    return window, overlap, window - overlap


class VRGDGOverlapBatchState:
    """Duck-compatible replacement for VHS ``BatchManager``."""

    def __init__(self):
        self.frames_per_batch = -1
        self.inputs = {}
        self.outputs = {}
        self.unique_id = None
        self.has_closed_inputs = False
        self.total_frames = float("inf")
        self.window_frames = 73
        self.overlap_frames = 17
        self.stride_frames = 56
        self.blend_mode = "cosine"
        self.batch_index = 0
        self._overlap_input_carry = None
        self._overlap_output_tail = None
        self._overlap_input_index = -1
        self._overlap_output_index = -1

    def has_open_inputs(self):
        return bool(self.inputs)

    def close_inputs(self):
        for value in tuple(self.inputs.values()):
            generator = value[-1]
            if getattr(generator, "gi_suspended", False):
                try:
                    generator.send(1)
                except StopIteration:
                    pass
        self.inputs = {}

    def reset(self):
        self.close_inputs()
        for value in tuple(self.outputs.values()):
            generator = value[-1]
            if getattr(generator, "gi_suspended", False):
                try:
                    generator.send(None)
                except StopIteration:
                    pass
        self.__init__()


class VRGDGOverlapPreset:
    """Resolve a friendly H3 preset without replacing VHS's real manager."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (tuple(PRESETS) + (CUSTOM_PRESET,),),
                "custom_window": (
                    "INT",
                    {"default": 73, "min": 5, "max": BIGMAX, "step": 17},
                ),
                "custom_overlap": (
                    "INT",
                    {"default": 17, "min": 1, "max": BIGMAX, "step": 1},
                ),
                "blend_mode": (BLEND_MODES,),
            }
        }

    RETURN_TYPES = ("VRGDG_OVERLAP_CONFIG", "INT", "INT", "INT")
    RETURN_NAMES = ("overlap_config", "window", "overlap", "stride")
    FUNCTION = "configure"
    CATEGORY = "VRGDG/Video/Meta Batch"
    DESCRIPTION = (
        "Creates validated H3 overlap settings. Connect stride to the standard VHS Batch "
        "Manager's frames_per_batch input and overlap_config to Build Overlap Window."
    )

    def configure(self, preset, custom_window, custom_overlap, blend_mode):
        window, overlap, stride = resolve_h3_settings(
            preset, custom_window, custom_overlap
        )
        if blend_mode not in BLEND_MODES:
            raise ValueError(f"Unknown overlap blend mode: {blend_mode!r}")
        config = {
            "preset": str(preset),
            "window": window,
            "overlap": overlap,
            "stride": stride,
            "blend_mode": str(blend_mode),
        }
        return config, window, overlap, stride

class VRGDGOverlapMetaBatchManager(VRGDGOverlapBatchState):
    """Legacy prototype retained only so old canvases can display a migration warning."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (tuple(PRESETS) + (CUSTOM_PRESET,),),
                "custom_window": (
                    "INT",
                    {"default": 73, "min": 5, "max": BIGMAX, "step": 17},
                ),
                "custom_overlap": (
                    "INT",
                    {"default": 17, "min": 1, "max": BIGMAX, "step": 1},
                ),
                "blend_mode": (BLEND_MODES,),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VHS_BatchManager", "INT", "INT", "INT")
    RETURN_NAMES = ("meta_batch", "window", "overlap", "stride")
    FUNCTION = "update_batch"
    CATEGORY = "VRGDG/Video/Meta Batch"
    DESCRIPTION = (
        "Deprecated: VideoHelperSuite requeues only its native VHS_BatchManager class. "
        "Use VRGDG Overlap Preset plus the standard VHS Batch Manager instead."
    )

    def update_batch(
        self,
        preset,
        custom_window,
        custom_overlap,
        blend_mode,
        prompt=None,
        unique_id=None,
    ):
        window, overlap, stride = resolve_h3_settings(
            preset, custom_window, custom_overlap
        )
        if blend_mode not in BLEND_MODES:
            raise ValueError(f"Unknown overlap blend mode: {blend_mode!r}")

        requeue = 0
        if unique_id is not None and prompt is not None:
            node_prompt = prompt.get(str(unique_id))
            if node_prompt is None:
                try:
                    node_prompt = prompt.get(unique_id, {})
                except TypeError:
                    node_prompt = {}
            requeue = int((node_prompt or {}).get("inputs", {}).get("requeue", 0))

        if requeue == 0:
            self.reset()
            self.unique_id = unique_id
        self.window_frames = window
        self.overlap_frames = overlap
        self.stride_frames = stride
        self.frames_per_batch = stride
        self.blend_mode = blend_mode
        self.batch_index = requeue

        if requeue:
            total = self.total_frames
            batches = "?"
            if math.isfinite(total):
                batches = math.ceil(float(total) / stride)
            print(
                f"VRGDG Overlap Meta-Batch {requeue + 1}/{batches}: "
                f"window={window}, overlap={overlap}, stride={stride}"
            )
        return self, window, overlap, stride


def _require_manager(meta_batch):
    required = (
        "window_frames",
        "overlap_frames",
        "stride_frames",
        "blend_mode",
        "batch_index",
    )
    missing = [name for name in required if not hasattr(meta_batch, name)]
    if missing:
        raise ValueError(
            "Connect the native VHS Batch Manager through VRGDG Build Overlap Window; "
            "the supplied manager is missing "
            + ", ".join(missing)
        )


def _is_final_batch(meta_batch, raw_count: int) -> bool:
    if bool(getattr(meta_batch, "has_closed_inputs", False)):
        return True
    stride = int(meta_batch.stride_frames)
    if raw_count < stride:
        return True
    total = getattr(meta_batch, "total_frames", float("inf"))
    if math.isfinite(total):
        consumed = int(meta_batch.batch_index) * stride + raw_count
        return consumed >= max(1, int(round(float(total))))
    return False


def _prompt_node(prompt, node_id):
    if prompt is None or node_id is None:
        return None
    if hasattr(prompt, "get"):
        node = prompt.get(str(node_id))
        if node is not None:
            return node
        try:
            return prompt.get(node_id)
        except TypeError:
            return None
    return None


def _vhs_requeue_index(prompt, unique_id, meta_batch):
    """Read the native VHS manager's injected requeue counter from the prompt."""
    window_node = _prompt_node(prompt, unique_id)
    if isinstance(window_node, dict):
        manager_ref = (window_node.get("inputs") or {}).get("meta_batch")
        if isinstance(manager_ref, (list, tuple)) and manager_ref:
            manager_node = _prompt_node(prompt, manager_ref[0])
            if isinstance(manager_node, dict):
                return int((manager_node.get("inputs") or {}).get("requeue", 0))
    return int(getattr(meta_batch, "_vrgdg_next_batch_index", 0))


class VRGDGOverlapWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "meta_batch": ("VHS_BatchManager",),
                "overlap_config": ("VRGDG_OVERLAP_CONFIG",),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "VRGDG_OVERLAP_INFO", "INT", "BOOLEAN")
    RETURN_NAMES = ("window_images", "overlap_info", "new_frame_count", "is_final")
    FUNCTION = "build_window"
    CATEGORY = "VRGDG/Video/Meta Batch"
    DESCRIPTION = (
        "Prepends the prior source tail and pads the final window so the model always sees "
        "the preset H3 frame count."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build_window(
        self,
        images,
        meta_batch,
        overlap_config,
        prompt=None,
        unique_id=None,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1:
            raise ValueError("Overlap Window requires a non-empty IMAGE batch [N,H,W,C].")

        if not isinstance(overlap_config, dict):
            raise ValueError("Connect overlap_config from VRGDG Overlap Preset.")
        window = int(overlap_config.get("window", 0))
        overlap = int(overlap_config.get("overlap", 0))
        stride = int(overlap_config.get("stride", 0))
        blend_mode = str(overlap_config.get("blend_mode", ""))
        validated_window, validated_overlap, validated_stride = resolve_h3_settings(
            CUSTOM_PRESET, window, overlap
        )
        if (window, overlap, stride) != (
            validated_window,
            validated_overlap,
            validated_stride,
        ):
            raise ValueError("The overlap preset contains inconsistent window/overlap/stride values.")
        if blend_mode not in BLEND_MODES:
            raise ValueError(f"Unknown overlap blend mode: {blend_mode!r}")
        if int(getattr(meta_batch, "frames_per_batch", -1)) != stride:
            raise ValueError(
                f"The standard VHS Batch Manager must receive stride={stride} as "
                f"frames_per_batch; got {getattr(meta_batch, 'frames_per_batch', None)}."
            )

        meta_batch.window_frames = window
        meta_batch.overlap_frames = overlap
        meta_batch.stride_frames = stride
        meta_batch.blend_mode = blend_mode
        batch_index = _vhs_requeue_index(prompt, unique_id, meta_batch)
        meta_batch.batch_index = batch_index
        if batch_index == 0:
            meta_batch._overlap_input_carry = None
            meta_batch._overlap_output_tail = None
            meta_batch._overlap_input_index = -1
            meta_batch._overlap_output_index = -1

        raw_count = int(images.shape[0])
        if raw_count > stride:
            raise ValueError(
                f"VHS supplied {raw_count} frames, but the overlap stride is {stride}. "
                "Connect this manager directly to the VHS loader."
            )

        expected_index = int(meta_batch._overlap_input_index) + 1
        if batch_index != expected_index:
            if batch_index == 0:
                meta_batch._overlap_input_carry = None
            else:
                raise RuntimeError(
                    f"Overlap input state skipped from batch {expected_index} to {batch_index}. "
                    "Queue the workflow again to reset the meta batch."
                )

        if batch_index == 0:
            prefix = images[:1].repeat(overlap, 1, 1, 1)
            synthetic_prefix = overlap
        else:
            prefix = meta_batch._overlap_input_carry
            if prefix is None or prefix.shape[0] != overlap:
                raise RuntimeError("The previous overlap carry is missing; queue the workflow again.")
            prefix = prefix.to(device=images.device, dtype=images.dtype)
            synthetic_prefix = 0

        assembled = torch.cat((prefix, images), dim=0)
        if assembled.shape[0] < window:
            assembled = torch.cat(
                (assembled, assembled[-1:].repeat(window - assembled.shape[0], 1, 1, 1)),
                dim=0,
            )
        elif assembled.shape[0] > window:
            raise RuntimeError(
                f"Overlap window assembled {assembled.shape[0]} frames; expected {window}."
            )

        final = _is_final_batch(meta_batch, raw_count)
        if not final:
            if raw_count < overlap:
                raise RuntimeError(
                    f"A non-final stride must contain at least {overlap} frames; got {raw_count}."
                )
            meta_batch._overlap_input_carry = images[-overlap:].detach().clone()
        else:
            # This also fixes VHS's exact-multiple end condition: Video Combine can close
            # after the final real stride rather than requesting an empty extra batch.
            meta_batch.has_closed_inputs = True
            meta_batch._overlap_input_carry = None

        meta_batch._overlap_input_index = batch_index
        meta_batch._vrgdg_next_batch_index = batch_index + 1
        info = {
            "batch_index": batch_index,
            "window": window,
            "overlap": overlap,
            "stride": stride,
            "new_frame_count": raw_count,
            "synthetic_prefix": synthetic_prefix,
            "is_final": final,
            "blend_mode": str(meta_batch.blend_mode),
        }
        return assembled, info, raw_count, final


def _blend_weights(count: int, mode: str, reference: torch.Tensor):
    t = torch.arange(1, count + 1, device=reference.device, dtype=torch.float32)
    t = t / float(count + 1)
    if mode == "cosine":
        weights = 0.5 - 0.5 * torch.cos(math.pi * t)
    elif mode == "smoothstep":
        weights = t * t * (3.0 - 2.0 * t)
    elif mode == "linear":
        weights = t
    else:
        raise ValueError(f"Unknown overlap blend mode: {mode!r}")
    return weights.to(dtype=reference.dtype).reshape(count, 1, 1, 1)


class VRGDGOverlapBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "overlap_info": ("VRGDG_OVERLAP_INFO",),
                "meta_batch": ("VHS_BatchManager",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("assembled_images", "emitted_frame_count")
    FUNCTION = "blend_window"
    CATEGORY = "VRGDG/Video/Meta Batch"
    DESCRIPTION = (
        "Cosine/smoothstep/linear blends matching processed overlap frames, holds the next "
        "tail, and emits every original source frame exactly once."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def blend_window(self, images, overlap_info, meta_batch):
        _require_manager(meta_batch)
        if not isinstance(overlap_info, dict):
            raise ValueError("Overlap Blend requires overlap_info from VRGDG Overlap Window.")
        window = int(overlap_info["window"])
        overlap = int(overlap_info["overlap"])
        stride = int(overlap_info["stride"])
        raw_count = int(overlap_info["new_frame_count"])
        batch_index = int(overlap_info["batch_index"])
        final = bool(overlap_info["is_final"])
        mode = str(overlap_info["blend_mode"])

        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Overlap Blend requires processed IMAGE frames [N,H,W,C].")
        if images.shape[0] != window:
            raise ValueError(
                f"The processed model output must preserve the {window}-frame window; "
                f"received {images.shape[0]} frames."
            )

        expected_index = int(meta_batch._overlap_output_index) + 1
        if batch_index != expected_index:
            if batch_index == 0:
                meta_batch._overlap_output_tail = None
            else:
                raise RuntimeError(
                    f"Overlap output state skipped from batch {expected_index} to {batch_index}. "
                    "Queue the workflow again to reset the meta batch."
                )

        if batch_index == 0:
            real = images[overlap : overlap + raw_count]
            if final:
                output = real
                meta_batch._overlap_output_tail = None
            else:
                output = real[:-overlap]
                meta_batch._overlap_output_tail = real[-overlap:].detach().clone()
        else:
            previous = meta_batch._overlap_output_tail
            if previous is None or previous.shape[0] != overlap:
                raise RuntimeError("The previous processed overlap tail is missing.")
            previous = previous.to(device=images.device, dtype=images.dtype)
            current_head = images[:overlap]
            weights = _blend_weights(overlap, mode, images)
            blended = previous * (1.0 - weights) + current_head * weights
            new_frames = images[overlap : overlap + raw_count]
            combined = torch.cat((blended, new_frames), dim=0)
            if final:
                output = combined
                meta_batch._overlap_output_tail = None
            else:
                output = combined[:-overlap]
                meta_batch._overlap_output_tail = combined[-overlap:].detach().clone()

        if output.shape[0] < 1:
            raise RuntimeError(
                "Overlap settings emitted an empty batch. Use a smaller overlap or larger window."
            )
        meta_batch._overlap_output_index = batch_index
        if final:
            meta_batch._overlap_input_index = -1
            meta_batch._overlap_output_index = -1
        return output, int(output.shape[0])


NODE_CLASS_MAPPINGS = {
    "VRGDGOverlapPreset": VRGDGOverlapPreset,
    "VRGDGOverlapMetaBatchManager": VRGDGOverlapMetaBatchManager,
    "VRGDGOverlapWindow": VRGDGOverlapWindow,
    "VRGDGOverlapBlend": VRGDGOverlapBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDGOverlapPreset": "VRGDG Overlap Preset",
    "VRGDGOverlapMetaBatchManager": "DEPRECATED - VRGDG Overlap Meta Batch Manager",
    "VRGDGOverlapWindow": "VRGDG Build Overlap Window",
    "VRGDGOverlapBlend": "VRGDG Blend Overlap Output",
}
