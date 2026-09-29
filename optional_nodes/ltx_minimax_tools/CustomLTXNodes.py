"""Sigma-aware guidance and guide-control nodes for LTX sampling.

The stock ComfyUI CFG Guider accepts a scalar FLOAT.  These nodes use a
dedicated schedule type so a list of CFG values cannot accidentally be passed
to the stock guider and fail during sampling.
"""

from __future__ import annotations

import math
import importlib
from typing import Any

import torch

import comfy.samplers


LTX_CFG_SCHEDULE_TYPE = "VRGDG_LTX_CFG_SCHEDULE"


def _sigma_tensor(sigmas) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigmas).detach().flatten().to(
        device="cpu", dtype=torch.float64
    )
    if sigma_tensor.numel() < 2:
        raise ValueError("sigmas must contain at least two values")
    if not torch.isfinite(sigma_tensor).all():
        raise ValueError("every sigma value must be finite")
    return sigma_tensor


def _interpolation_factor(interpolation: str, amount: float) -> float:
    if interpolation == "linear":
        return amount
    if interpolation == "ease_in":
        return amount * amount
    if interpolation == "ease_out":
        return amount * (2.0 - amount)
    raise ValueError(f"Unsupported interpolation: {interpolation}")


def _build_transition_values(
    sigmas,
    value_start: float,
    value_end: float,
    interpolation: str,
    start_percent: float,
    end_percent: float,
    *,
    outside_value: float | None,
) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Build one value per sigma transition.

    ``outside_value=None`` holds the start/end values before/after the ramp.
    Otherwise the supplied neutral value is used outside the active window.
    """
    sigma_tensor = _sigma_tensor(sigmas)
    if start_percent > end_percent:
        raise ValueError("start_percent must be less than or equal to end_percent")

    transition_count = int(sigma_tensor.numel()) - 1
    start_index = min(int(transition_count * start_percent), transition_count - 1)
    end_index = min(int(transition_count * end_percent), transition_count - 1)

    if outside_value is None:
        values = [float(value_start)] * transition_count
        for index in range(end_index + 1, transition_count):
            values[index] = float(value_end)
    else:
        values = [float(outside_value)] * transition_count

    for index in range(start_index, end_index + 1):
        amount = 0.0 if end_index == start_index else (
            (index - start_index) / (end_index - start_index)
        )
        factor = _interpolation_factor(interpolation, amount)
        values[index] = round(
            float(value_start + factor * (value_end - value_start)), 4
        )

    return sigma_tensor, tuple(values)


def _runtime_schedule_offset(expected_sigmas, runtime_sigmas) -> int:
    expected = _sigma_tensor(expected_sigmas)
    runtime = _sigma_tensor(runtime_sigmas)
    if runtime.numel() <= expected.numel():
        for offset in range(int(expected.numel() - runtime.numel()) + 1):
            candidate = expected[offset : offset + runtime.numel()]
            if torch.allclose(runtime, candidate, rtol=1e-5, atol=1e-7):
                return offset
    raise ValueError(
        "The sampler's sigma range is not part of the connected ManualSigmas. "
        "Connect the same SIGMAS output to this node and the sampler."
    )


def _current_transition_index(sample_sigmas, timestep) -> int:
    sigmas = _sigma_tensor(sample_sigmas)
    # ManualSigmas is normalized to CPU above, while ComfyUI supplies the live
    # sampler timestep on the model device (normally CUDA). Keep the comparison
    # on CPU so schedule lookup never mixes devices.
    current = (
        torch.as_tensor(timestep)
        .detach()
        .flatten()[0]
        .to(device="cpu", dtype=torch.float64)
    )
    transition_sigmas = sigmas[:-1]

    exact = torch.isclose(transition_sigmas, current, rtol=1e-5, atol=1e-7).nonzero()
    if exact.numel() > 0:
        return int(exact.flatten()[0].item())

    for index in range(sigmas.numel() - 1):
        left = sigmas[index]
        right = sigmas[index + 1]
        if torch.minimum(left, right) <= current <= torch.maximum(left, right):
            return index
    return int(torch.argmin(torch.abs(transition_sigmas - current)).item())


def _schedule_index(expected_sigmas, runtime_sigmas, timestep) -> int:
    return _runtime_schedule_offset(expected_sigmas, runtime_sigmas) + (
        _current_transition_index(runtime_sigmas, timestep)
    )


def _ltx_stg_module():
    """Load the installed ComfyUI-LTXVideo STG wrappers only when needed."""
    package = __package__ or ""
    custom_nodes_package = package.rsplit(".", 1)[0] if "." in package else ""
    candidates = []
    if custom_nodes_package:
        candidates.append(f"{custom_nodes_package}.ComfyUI-LTXVideo.stg")
    candidates.extend(("custom_nodes.ComfyUI-LTXVideo.stg", "ComfyUI-LTXVideo.stg"))

    errors = []
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except (ImportError, ModuleNotFoundError) as error:
            errors.append(str(error))
    raise ImportError(
        "VRGDG LTX Sigma Advanced Guider requires ComfyUI-LTXVideo. "
        + " | ".join(errors)
    )


def _build_cfg_schedule(
    sigmas,
    cfg_scale_start: float,
    cfg_scale_end: float,
    interpolation: str,
    start_percent: float,
    end_percent: float,
) -> dict[str, Any]:
    """Build one scalar CFG value for each denoising transition."""
    sigma_tensor, values = _build_transition_values(
        sigmas,
        cfg_scale_start,
        cfg_scale_end,
        interpolation,
        start_percent,
        end_percent,
        outside_value=1.0,
    )
    transition_count = len(values)

    return {
        "kind": LTX_CFG_SCHEDULE_TYPE,
        "transitions": transition_count,
        "sigmas": sigma_tensor.tolist(),
        "values": list(values),
    }


class VRGDGLTXCFGSchedule:
    """Create a per-transition CFG schedule for the matching LTX CFG Guider."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Connect the same ManualSigmas used by the sampler."},
                ),
                "cfg_scale_start": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "cfg_scale_end": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "interpolation": (["linear", "ease_in", "ease_out"],),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = (LTX_CFG_SCHEDULE_TYPE,)
    RETURN_NAMES = ("cfg",)
    FUNCTION = "create_schedule"
    CATEGORY = "VRGameDevGirl/LTX/Sampling"
    DESCRIPTION = (
        "Creates one CFG value per transition in a connected LTX sigma schedule. "
        "Values outside the selected percentage range are 1.0. Connect this only "
        "to VRGDG LTX Scheduled CFG Guider."
    )

    def create_schedule(
        self,
        sigmas,
        cfg_scale_start,
        cfg_scale_end,
        interpolation,
        start_percent,
        end_percent,
    ):
        schedule = _build_cfg_schedule(
            sigmas,
            cfg_scale_start,
            cfg_scale_end,
            interpolation,
            start_percent,
            end_percent,
        )
        return (schedule,)


class _LTXScheduledCFGGuider(comfy.samplers.CFGGuider):
    """CFG Guider that resolves a schedule to a scalar at every model call."""

    def set_cfg_schedule(self, schedule):
        if not isinstance(schedule, dict):
            raise TypeError("cfg must come from VRGDG LTX CFG Schedule")
        if schedule.get("kind") != LTX_CFG_SCHEDULE_TYPE:
            raise ValueError("Unrecognized LTX CFG schedule data")

        values = schedule.get("values")
        sigmas = schedule.get("sigmas")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("The LTX CFG schedule contains no values")
        if not isinstance(sigmas, (list, tuple)) or len(sigmas) != len(values) + 1:
            raise ValueError("The LTX CFG schedule contains invalid sigma data")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("Every LTX CFG schedule value must be a finite number")
        if not all(isinstance(sigma, (int, float)) and math.isfinite(sigma) for sigma in sigmas):
            raise ValueError("Every LTX CFG schedule sigma must be a finite number")

        self.cfg_schedule = tuple(float(value) for value in values)
        self.sigma_schedule = tuple(float(sigma) for sigma in sigmas)

    @staticmethod
    def _current_transition_index(sample_sigmas, timestep):
        sigmas = torch.as_tensor(sample_sigmas).detach().flatten()
        if sigmas.numel() < 2:
            raise ValueError("Sampling requires at least two sigma values")

        current = torch.as_tensor(timestep, device=sigmas.device, dtype=sigmas.dtype)
        current = current.detach().flatten()[0]
        transition_sigmas = sigmas[:-1]

        exact = torch.isclose(transition_sigmas, current, rtol=1e-5, atol=1e-7).nonzero()
        if exact.numel() > 0:
            return int(exact.flatten()[0].item())

        # Some samplers evaluate the model between two scheduled sigmas. Map
        # those evaluations to the transition whose interval contains them.
        for index in range(sigmas.numel() - 1):
            left = sigmas[index]
            right = sigmas[index + 1]
            if torch.minimum(left, right) <= current <= torch.maximum(left, right):
                return index

        # Extrapolated evaluations use the nearest scheduled transition.
        return int(torch.argmin(torch.abs(transition_sigmas - current)).item())

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}

        transformer_options = model_options.get("transformer_options", {})
        sample_sigmas = transformer_options.get("sample_sigmas")
        if sample_sigmas is None:
            raise ValueError("LTX Scheduled CFG Guider could not find the sampler sigma schedule")

        runtime_sigmas = torch.as_tensor(sample_sigmas).detach().flatten().to(
            device="cpu", dtype=torch.float64
        )
        expected_sigmas = torch.tensor(self.sigma_schedule, dtype=torch.float64)
        if runtime_sigmas.numel() < 2:
            raise ValueError("LTX sampling requires at least one sigma transition")

        # LTX's easy samplers may split ManualSigmas into several contiguous
        # ranges. Locate the active range within the original schedule so CFG
        # values keep their original alignment.
        sigma_offset = None
        if runtime_sigmas.numel() <= expected_sigmas.numel():
            possible_offsets = range(
                int(expected_sigmas.numel() - runtime_sigmas.numel()) + 1
            )
            for offset in possible_offsets:
                candidate = expected_sigmas[offset : offset + runtime_sigmas.numel()]
                if torch.allclose(runtime_sigmas, candidate, rtol=1e-5, atol=1e-7):
                    sigma_offset = offset
                    break

        if sigma_offset is None:
            raise ValueError(
                "The sampler's sigma range is not part of the LTX CFG Schedule. "
                "Connect the same ManualSigmas output to the schedule and sampler."
            )

        transition_index = sigma_offset + self._current_transition_index(
            runtime_sigmas, timestep
        )
        cfg = self.cfg_schedule[transition_index]

        return comfy.samplers.sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative"),
            self.conds.get("positive"),
            cfg,
            model_options=model_options,
            seed=seed,
        )


class VRGDGLTXScheduledCFGGuider:
    """Create an LTX-compatible guider that consumes an LTX CFG schedule."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": (
                    LTX_CFG_SCHEDULE_TYPE,
                    {
                        "forceInput": True,
                        "tooltip": "Connect the cfg output from VRGDG LTX CFG Schedule.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "VRGameDevGirl/LTX/Sampling"
    DESCRIPTION = (
        "A schedule-aware replacement for ComfyUI's scalar CFG Guider. "
        "It selects one CFG value for each LTX sigma transition."
    )

    def get_guider(self, model, positive, negative, cfg):
        guider = _LTXScheduledCFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg_schedule(cfg)
        guider.raw_conds = (positive, negative)
        return (guider,)


class _LTXSigmaAdvancedGuider(comfy.samplers.CFGGuider):
    """ManualSigmas-driven CFG/APG + STG guider for current LTX models."""

    def __init__(
        self,
        model,
        sigmas,
        cfg_values,
        stg_values,
        rescale_values,
        stg_blocks,
        guidance_mode,
        cfg_star,
        apg_eta,
        apg_norm_threshold,
        apg_momentum,
    ):
        stg_module = _ltx_stg_module()
        model = model.clone()
        super().__init__(model)

        self.sigma_schedule = tuple(float(value) for value in _sigma_tensor(sigmas))
        self.cfg_values = tuple(float(value) for value in cfg_values)
        self.stg_values = tuple(float(value) for value in stg_values)
        self.rescale_values = tuple(float(value) for value in rescale_values)
        self.guidance_mode = guidance_mode
        self.cfg_star = bool(cfg_star)
        self.apg_eta = float(apg_eta)
        self.apg_norm_threshold = float(apg_norm_threshold)
        self.apg_momentum = float(apg_momentum)
        self.apg_running_average = None
        self.previous_sigma = None

        self.stg_flag = stg_module.STGFlag(
            do_skip=False,
            skip_layers=list(stg_blocks),
        )
        transformer_blocks = stg_module.STGGuider.get_transformer_blocks(model)
        invalid_blocks = [
            index for index in stg_blocks
            if index < 0 or index >= len(transformer_blocks)
        ]
        if invalid_blocks:
            raise ValueError(
                f"stg_blocks contains invalid indices {invalid_blocks}; this LTX model "
                f"has {len(transformer_blocks)} transformer blocks"
            )
        for index, block in enumerate(transformer_blocks):
            model.set_model_patch_replace(
                stg_module.STGBlockWrapper(block, self.stg_flag, index),
                "dit",
                "double_block",
                index,
            )

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    @staticmethod
    def _cfg_star_negative(positive, negative):
        batch_size = positive.shape[0]
        positive_flat = positive.reshape(batch_size, -1)
        negative_flat = negative.reshape(batch_size, -1)
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat.square(), dim=1, keepdim=True)
        alpha = dot_product / (squared_norm + 1e-8)
        return negative * alpha.reshape(
            batch_size, *([1] * (negative.ndim - 1))
        )

    @staticmethod
    def _project(guidance, positive):
        dtype = guidance.dtype
        guidance_64 = guidance.double()
        positive_64 = torch.nn.functional.normalize(
            positive.double(), dim=[-1, -2, -3]
        )
        parallel = (
            (guidance_64 * positive_64)
            .sum(dim=[-1, -2, -3], keepdim=True)
            * positive_64
        )
        return parallel.to(dtype), (guidance_64 - parallel).to(dtype)

    def _apg(self, positive, negative, cfg, sigma):
        sigma_value = float(torch.as_tensor(sigma).detach().flatten()[0])
        if self.previous_sigma is not None and sigma_value > self.previous_sigma + 1e-7:
            self.apg_running_average = None
        self.previous_sigma = sigma_value

        guidance = positive - negative
        if not math.isclose(self.apg_momentum, 0.0):
            if self.apg_running_average is None:
                self.apg_running_average = guidance
            else:
                self.apg_running_average = (
                    self.apg_momentum * self.apg_running_average + guidance
                )
            guidance = self.apg_running_average

        if self.apg_norm_threshold > 0:
            guidance_norm = guidance.norm(
                p=2, dim=[-1, -2, -3], keepdim=True
            ).clamp_min(1e-12)
            scale = torch.minimum(
                torch.ones_like(guidance_norm),
                self.apg_norm_threshold / guidance_norm,
            )
            guidance = guidance * scale

        parallel, orthogonal = self._project(guidance, positive)
        modified_guidance = orthogonal + self.apg_eta * parallel
        return positive + (cfg - 1.0) * modified_guidance

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        transformer_options = model_options.setdefault("transformer_options", {})
        runtime_sigmas = transformer_options.get("sample_sigmas")
        if runtime_sigmas is None:
            raise ValueError(
                "LTX Sigma Advanced Guider could not find the sampler sigma schedule"
            )

        index = _schedule_index(self.sigma_schedule, runtime_sigmas, timestep)
        cfg = self.cfg_values[index]
        stg_scale = self.stg_values[index]
        rescale = self.rescale_values[index]
        positive_cond = self.conds.get("positive")
        negative_cond = self.conds.get("negative")

        positive = comfy.samplers.calc_cond_batch(
            self.inner_model, [positive_cond], x, timestep, model_options
        )[0]
        negative = None
        if not math.isclose(cfg, 1.0):
            negative = comfy.samplers.calc_cond_batch(
                self.inner_model, [negative_cond], x, timestep, model_options
            )[0]
            if self.cfg_star:
                negative = self._cfg_star_negative(positive, negative)

        if negative is None:
            guided = positive
        elif self.guidance_mode == "APG":
            guided = self._apg(positive, negative, cfg, timestep)
        else:
            guided = positive + (cfg - 1.0) * (positive - negative)

        perturbed = None
        if not math.isclose(stg_scale, 0.0):
            try:
                transformer_options["ptb_index"] = 0
                self.stg_flag.do_skip = True
                perturbed = comfy.samplers.calc_cond_batch(
                    self.inner_model, [positive_cond], x, timestep, model_options
                )[0]
            finally:
                self.stg_flag.do_skip = False
                transformer_options.pop("ptb_index", None)
            guided = guided + stg_scale * (positive - perturbed)

        if not math.isclose(rescale, 0.0):
            guided_std = guided.std().clamp_min(1e-12)
            factor = positive.std() / guided_std
            guided = guided * (rescale * factor + (1.0 - rescale))

        uncond_for_hook = negative if negative is not None else positive
        perturbed_for_hook = perturbed if perturbed is not None else positive
        for function in model_options.get("sampler_post_cfg_function", []):
            guided = function(
                {
                    "denoised": guided,
                    "cond": positive_cond,
                    "uncond": negative_cond,
                    "model": self.inner_model,
                    "uncond_denoised": uncond_for_hook,
                    "cond_denoised": positive,
                    "sigma": timestep,
                    "model_options": model_options,
                    "input": x,
                    "perturbed_cond": positive_cond,
                    "perturbed_cond_denoised": perturbed_for_hook,
                }
            )
        return guided


class VRGDGLTXSigmaAdvancedGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Connect the same ManualSigmas used by the sampler."},
                ),
                "cfg_start": (
                    "FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.01}
                ),
                "cfg_end": (
                    "FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.01}
                ),
                "stg_start": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}
                ),
                "stg_end": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}
                ),
                "rescale_start": (
                    "FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "rescale_end": (
                    "FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "interpolation": (["linear", "ease_in", "ease_out"],),
                "start_percent": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "end_percent": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "stg_blocks": (
                    "STRING",
                    {
                        "default": "14, 19",
                        "tooltip": "Comma-separated LTX transformer blocks to perturb.",
                    },
                ),
                "guidance_mode": (["CFG", "APG"],),
                "cfg_star": ("BOOLEAN", {"default": False}),
                "apg_eta": (
                    "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}
                ),
                "apg_norm_threshold": (
                    "FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}
                ),
                "apg_momentum": (
                    "FLOAT", {"default": 0.0, "min": -5.0, "max": 1.0, "step": 0.01}
                ),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "VRGameDevGirl/LTX/Sampling"
    DESCRIPTION = (
        "One ManualSigmas-driven LTX guider with scheduled CFG/APG, STG, and "
        "variance rescale. APG replaces the CFG vector when selected; it is not "
        "stacked as a second guidance pass. Outside the active percentage range "
        "CFG is 1 and STG/rescale are 0."
    )

    def get_guider(
        self,
        model,
        positive,
        negative,
        sigmas,
        cfg_start,
        cfg_end,
        stg_start,
        stg_end,
        rescale_start,
        rescale_end,
        interpolation,
        start_percent,
        end_percent,
        stg_blocks,
        guidance_mode,
        cfg_star,
        apg_eta,
        apg_norm_threshold,
        apg_momentum,
    ):
        sigma_tensor, cfg_values = _build_transition_values(
            sigmas, cfg_start, cfg_end, interpolation, start_percent, end_percent,
            outside_value=1.0,
        )
        _, stg_values = _build_transition_values(
            sigmas, stg_start, stg_end, interpolation, start_percent, end_percent,
            outside_value=0.0,
        )
        _, rescale_values = _build_transition_values(
            sigmas, rescale_start, rescale_end, interpolation, start_percent, end_percent,
            outside_value=0.0,
        )
        try:
            block_indices = [
                int(value.strip()) for value in stg_blocks.split(",") if value.strip()
            ]
        except ValueError as error:
            raise ValueError("stg_blocks must be comma-separated integers") from error
        if not block_indices and any(not math.isclose(value, 0.0) for value in stg_values):
            raise ValueError("At least one stg_blocks index is required when STG is active")

        guider = _LTXSigmaAdvancedGuider(
            model,
            sigma_tensor,
            cfg_values,
            stg_values,
            rescale_values,
            block_indices,
            guidance_mode,
            cfg_star,
            apg_eta,
            apg_norm_threshold,
            apg_momentum,
        )
        guider.set_conds(positive, negative)
        guider.raw_conds = (positive, negative)
        return (guider,)


def _conditioning_guide_entries(model):
    """Yield converted LTX guide-entry lists from positive/negative conds."""
    for key, conditions in getattr(model, "conds", {}).items():
        if "positive" not in key and "negative" not in key:
            continue
        for condition in conditions:
            model_conds = condition.get("model_conds", {})
            constant = model_conds.get("guide_attention_entries")
            entries = getattr(constant, "cond", None)
            if isinstance(entries, list):
                yield entries


def _set_converted_denoise_mask(model, denoise_mask):
    for key, conditions in getattr(model, "conds", {}).items():
        if "positive" not in key and "negative" not in key:
            continue
        for condition in conditions:
            model_conds = condition.get("model_conds", {})
            converted_mask = model_conds.get("denoise_mask")
            if converted_mask is not None and hasattr(converted_mask, "cond"):
                converted_mask.cond = denoise_mask


class VRGDGLTXSigmaGuideRelease:
    """Schedule LTX guide locking and per-guide attention by ManualSigmas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Connect the same ManualSigmas used by the sampler."},
                ),
                "influence_start": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "1 preserves the guide's original strength; 0 releases it.",
                    },
                ),
                "influence_end": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "interpolation": (["linear", "ease_in", "ease_out"],),
                "start_percent": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "end_percent": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "affect_latent_lock": ("BOOLEAN", {"default": True}),
                "affect_attention": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "VRGameDevGirl/LTX/Guides"
    DESCRIPTION = (
        "Ramps LTX image/video-guide influence over the connected ManualSigmas. "
        "It scales both the guide's latent denoise lock and its per-guide attention. "
        "The start value is held before start_percent and the end value after "
        "end_percent. Value 1 preserves the original guide strength; 0 releases it."
    )

    def apply(
        self,
        model,
        sigmas,
        influence_start,
        influence_end,
        interpolation,
        start_percent,
        end_percent,
        affect_latent_lock,
        affect_attention,
    ):
        sigma_tensor, influence_values = _build_transition_values(
            sigmas,
            influence_start,
            influence_end,
            interpolation,
            start_percent,
            end_percent,
            outside_value=None,
        )
        model = model.clone()
        existing_function = model.model_options.get("denoise_mask_function")
        expected_sigmas = tuple(float(value) for value in sigma_tensor)

        def scheduled_guide_mask(sigma, denoise_mask, extra_options):
            if existing_function is not None:
                denoise_mask = existing_function(sigma, denoise_mask, extra_options)

            runtime_sigmas = extra_options.get("sigmas")
            if runtime_sigmas is None:
                raise ValueError(
                    "LTX Sigma Guide Release could not find the sampler sigma schedule"
                )
            index = _schedule_index(expected_sigmas, runtime_sigmas, sigma)
            influence = influence_values[index]
            inner_model = extra_options["model"]
            entry_lists = list(_conditioning_guide_entries(inner_model))

            guide_frames = 0
            for entries in entry_lists:
                current_frames = 0
                for entry in entries:
                    latent_shape = entry.get("latent_shape", ())
                    if latent_shape:
                        current_frames += max(0, int(latent_shape[0]))
                    if affect_attention:
                        base_key = "_vrgdg_original_strength"
                        if base_key not in entry:
                            entry[base_key] = float(entry.get("strength", 1.0))
                        entry["strength"] = entry[base_key] * influence
                guide_frames = max(guide_frames, current_frames)

            if affect_latent_lock and guide_frames > 0 and denoise_mask.ndim >= 3:
                guide_frames = min(guide_frames, int(denoise_mask.shape[2]))
                denoise_mask = denoise_mask.clone()
                guide_slice = denoise_mask[:, :, -guide_frames:]
                valid = guide_slice >= 0
                scheduled = (1.0 - influence * (1.0 - guide_slice)).clamp(0.0, 1.0)
                guide_slice.copy_(torch.where(valid, scheduled, guide_slice))

            _set_converted_denoise_mask(inner_model, denoise_mask)
            return denoise_mask

        model.set_model_denoise_mask_function(scheduled_guide_mask)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_LTXCFGSchedule": VRGDGLTXCFGSchedule,
    "VRGDG_LTXScheduledCFGGuider": VRGDGLTXScheduledCFGGuider,
    "VRGDG_LTXSigmaAdvancedGuider": VRGDGLTXSigmaAdvancedGuider,
    "VRGDG_LTXSigmaGuideRelease": VRGDGLTXSigmaGuideRelease,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LTXCFGSchedule": "VRGDG LTX CFG Schedule",
    "VRGDG_LTXScheduledCFGGuider": "VRGDG LTX Scheduled CFG Guider",
    "VRGDG_LTXSigmaAdvancedGuider": "VRGDG LTX Sigma Advanced Guider",
    "VRGDG_LTXSigmaGuideRelease": "VRGDG LTX Sigma Guide Release",
}
