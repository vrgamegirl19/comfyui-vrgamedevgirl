"""VRGDG extensions for the official ComfyUI-LTXVideo looping sampler."""

import copy
import importlib
import os
import sys

import torch


def _find_ltx_package_name():
    """Return ComfyUI's runtime package name for the installed LTX extension."""
    for module_name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file or not hasattr(module, "__path__"):
            continue
        package_folder = os.path.basename(os.path.dirname(os.path.abspath(module_file)))
        if package_folder.casefold() == "comfyui-ltxvideo":
            return module_name

    # Some ComfyUI loaders do not retain an easily identifiable package alias. Add
    # custom_nodes itself to the import search path so Python can resolve the folder.
    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if custom_nodes_dir not in sys.path:
        sys.path.insert(0, custom_nodes_dir)
    importlib.import_module("ComfyUI-LTXVideo")
    return "ComfyUI-LTXVideo"


_LTX_PACKAGE = _find_ltx_package_name()
_ltx_looping = importlib.import_module(f"{_LTX_PACKAGE}.looping_sampler")
_ltx_easy = importlib.import_module(f"{_LTX_PACKAGE}.easy_samplers")
_ltx_latents = importlib.import_module(f"{_LTX_PACKAGE}.latents")

LTXVLoopingSampler = _ltx_looping.LTXVLoopingSampler
LTXVBaseSampler = _ltx_easy.LTXVBaseSampler
LTXVExtendSampler = _ltx_easy.LTXVExtendSampler
LTXVInContextSampler = _ltx_easy.LTXVInContextSampler
LTXVSelectLatents = _ltx_latents.LTXVSelectLatents


class VRGDGLTXVLoopingSamplerAdvanced(LTXVLoopingSampler):
    """Official LTXV Looping Sampler with a per-temporal-tile image-strength schedule."""

    @classmethod
    def INPUT_TYPES(cls):
        input_types = copy.deepcopy(super().INPUT_TYPES())
        input_types["optional"]["use_scheduled_strengths"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Enable the three comma-separated strength schedules. When disabled, "
                    "the original scalar strength widgets behave exactly like the official node."
                ),
            },
        )
        input_types["optional"]["cond_image_strength_schedule"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "tooltip": (
                    "Comma-separated conditioning-image strengths, one per temporal tile. "
                    "Example: 1.0,0.6,0.3. The last value repeats when there are more tiles. "
                    "A tile's value applies to every keyframe routed into that tile."
                ),
            },
        )
        input_types["optional"]["guiding_strength_schedule"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "tooltip": (
                    "Comma-separated optional-guiding-latent strengths, one per temporal tile. "
                    "Example: 0.8,0.5,0.3. The last value repeats."
                ),
            },
        )
        input_types["optional"]["temporal_overlap_cond_strength_schedule"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "tooltip": (
                    "Comma-separated overlap-conditioning strengths. The first value belongs "
                    "to the first extension boundary (tile 1); later values follow later boundaries. "
                    "Example: 0.8,0.65,0.5. The last value repeats."
                ),
            },
        )
        return input_types

    CATEGORY = "VRGameDevGirl/LTX"
    DESCRIPTION = (
        "LTXV Looping Sampler with optional per-temporal-tile schedules for image guidance, "
        "latent guidance, and overlap conditioning. The official LTX node is not modified."
    )

    def sample(
        self,
        *args,
        use_scheduled_strengths=False,
        cond_image_strength_schedule="",
        guiding_strength_schedule="",
        temporal_overlap_cond_strength_schedule="",
        **kwargs,
    ):
        enabled = bool(use_scheduled_strengths)
        self._vrgdg_strength_schedules = {
            "cond": self._parse_strength_schedule(
                cond_image_strength_schedule,
                kwargs.get("cond_image_strength", 1.0),
                "cond_image_strength_schedule",
            ) if enabled else None,
            "guiding": self._parse_strength_schedule(
                guiding_strength_schedule,
                kwargs.get("guiding_strength", 1.0),
                "guiding_strength_schedule",
            ) if enabled else None,
            "overlap": self._parse_strength_schedule(
                temporal_overlap_cond_strength_schedule,
                kwargs.get("temporal_overlap_cond_strength", 0.5),
                "temporal_overlap_cond_strength_schedule",
            ) if enabled else None,
        }
        try:
            return super().sample(*args, **kwargs)
        finally:
            self._vrgdg_strength_schedules = None

    @staticmethod
    def _parse_strength_schedule(value, fallback, field_name):
        text = str(value or "").strip()
        if not text:
            return [float(fallback)]
        strengths = []
        for position, item in enumerate(text.split(","), start=1):
            item = item.strip()
            if not item:
                raise ValueError(
                    f"{field_name} item {position} is empty."
                )
            try:
                strength = float(item)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} item {position} is not a number: {item!r}"
                ) from exc
            if not 0.0 <= strength <= 1.0:
                raise ValueError(
                    f"{field_name} values must be between 0.0 and 1.0; "
                    f"item {position} was {strength}."
                )
            strengths.append(strength)
        return strengths

    def _scheduled_strength(self, schedule_name, index, fallback):
        schedules = getattr(self, "_vrgdg_strength_schedules", None) or {}
        schedule = schedules.get(schedule_name)
        if not schedule:
            return float(fallback)
        return schedule[min(index, len(schedule) - 1)]

    def _process_temporal_chunks(self, tile_config, sampling_config, model_config):
        chunk_index = 0
        tile_out_latents = None
        first_tile_out_latents = None
        tile_step = sampling_config.temporal_tile_size - sampling_config.temporal_overlap

        if tile_step <= 0:
            raise ValueError(
                "temporal_tile_size must be greater than temporal_overlap after VAE scaling."
            )

        starts = range(
            0,
            tile_config.tile_latents["samples"].shape[2] + tile_step,
            tile_step,
        )
        ends = range(
            sampling_config.temporal_tile_size,
            tile_config.tile_latents["samples"].shape[2]
            + sampling_config.temporal_tile_size
            - sampling_config.temporal_overlap,
            tile_step,
        )

        for i_temporal_tile, (start_index, end_index) in enumerate(zip(starts, ends)):
            actual_end = min(
                end_index - 1, tile_config.tile_latents["samples"].shape[2] - 1
            )
            latent_chunk = LTXVSelectLatents().select_latents(
                tile_config.tile_latents, start_index, actual_end
            )[0]

            if tile_config.tile_guiding_latents is not None:
                guiding_end = min(
                    end_index - 1,
                    tile_config.tile_guiding_latents["samples"].shape[2] - 1,
                )
                guiding_latent_chunk = LTXVSelectLatents().select_latents(
                    tile_config.tile_guiding_latents, start_index, guiding_end
                )[0]
            else:
                guiding_latent_chunk = None

            if tile_config.tile_normalizing_latents is not None:
                normalizing_end = min(
                    end_index - 1,
                    tile_config.tile_normalizing_latents["samples"].shape[2] - 1,
                )
                normalizing_latent_chunk = LTXVSelectLatents().select_latents(
                    tile_config.tile_normalizing_latents,
                    start_index,
                    normalizing_end,
                )[0]
                normalize_per_frame = True
            else:
                normalizing_latent_chunk = first_tile_out_latents
                normalize_per_frame = False

            seed_offset = self._get_per_tile_value(
                sampling_config.per_tile_seed_offsets, i_temporal_tile
            )
            model_config.noise.seed = self._calculate_tile_seed(
                tile_config.first_seed,
                start_index,
                tile_config.vertical_tiles,
                tile_config.horizontal_tiles,
                tile_config.v,
                tile_config.h,
                seed_offset,
            )
            new_guider = self._prepare_guider_for_chunk(
                model_config.guider,
                sampling_config.optional_positive_conditionings,
                chunk_index,
            )

            this_chunk_indices = [
                in_tile_index
                for tile_index, in_tile_index in tile_config.keyframe_per_tile_indices
                if tile_index == i_temporal_tile
            ]
            if this_chunk_indices and tile_config.tile_keyframes is not None:
                this_chunk_keyframes = torch.cat(
                    [
                        tile_config.tile_keyframes[keyframe_index].unsqueeze(0)
                        for keyframe_index, (tile_index, _in_tile_index) in enumerate(
                            tile_config.keyframe_per_tile_indices
                        )
                        if tile_index == i_temporal_tile
                    ]
                )
            else:
                this_chunk_keyframes = None
            this_chunk_indices_text = ",".join(map(str, this_chunk_indices))
            cond_strength = self._scheduled_strength(
                "cond", i_temporal_tile, sampling_config.cond_image_strength
            )
            guiding_strength = self._scheduled_strength(
                "guiding", i_temporal_tile, sampling_config.guiding_strength
            )
            overlap_strength = self._scheduled_strength(
                "overlap", max(0, i_temporal_tile - 1),
                sampling_config.temporal_overlap_cond_strength,
            )
            print(
                f"[VRGDG LTX] Temporal tile {i_temporal_tile}: "
                f"latent {start_index}..{actual_end}, cond_image_strength={cond_strength}, "
                f"guiding_strength={guiding_strength}, overlap_strength={overlap_strength}"
            )

            common_optional = {
                "optional_cond_images": this_chunk_keyframes,
                "optional_cond_indices": this_chunk_indices_text,
                "optional_negative_index_latents": tile_config.tile_negative_index_latents,
                "optional_negative_index": sampling_config.optional_negative_index,
                "optional_negative_index_strength": sampling_config.optional_negative_index_strength,
                "optional_initialization_latents": latent_chunk,
                "cond_image_strength": cond_strength,
                "guiding_start_step": sampling_config.guiding_start_step,
                "guiding_end_step": sampling_config.guiding_end_step,
            }

            if start_index == 0:
                if tile_config.tile_guiding_latents is not None:
                    tile_out_latents = LTXVInContextSampler().sample(
                        vae=model_config.vae,
                        guider=new_guider,
                        sampler=model_config.sampler,
                        sigmas=model_config.sigmas,
                        noise=model_config.noise,
                        guiding_latents=guiding_latent_chunk,
                        num_frames=-1,
                        guiding_strength=guiding_strength,
                        **common_optional,
                    )[0]
                else:
                    base_optional = dict(common_optional)
                    base_optional["strength"] = base_optional.pop("cond_image_strength")
                    tile_out_latents = LTXVBaseSampler().sample(
                        model=model_config.model,
                        vae=model_config.vae,
                        noise=model_config.noise,
                        sampler=model_config.sampler,
                        sigmas=model_config.sigmas,
                        guider=new_guider,
                        num_frames=(
                            min(
                                sampling_config.temporal_tile_size,
                                tile_config.tile_latents["samples"].shape[2],
                            )
                            - 1
                        )
                        * sampling_config.time_scale_factor
                        + 1,
                        width=tile_config.tile_width * sampling_config.width_scale_factor,
                        height=tile_config.tile_height
                        * sampling_config.height_scale_factor,
                        crop="center",
                        crf=30,
                        **base_optional,
                    )[0]
                first_tile_out_latents = copy.deepcopy(tile_out_latents)
            else:
                tile_out_latents = LTXVExtendSampler().sample(
                    model=model_config.model,
                    vae=model_config.vae,
                    sampler=model_config.sampler,
                    sigmas=model_config.sigmas,
                    noise=model_config.noise,
                    latents=tile_out_latents,
                    num_new_frames=(
                        latent_chunk["samples"].shape[2]
                        - sampling_config.temporal_overlap
                    )
                    * sampling_config.time_scale_factor,
                    frame_overlap=sampling_config.temporal_overlap
                    * sampling_config.time_scale_factor,
                    guider=new_guider,
                    strength=overlap_strength,
                    guiding_strength=guiding_strength,
                    optional_guiding_latents=guiding_latent_chunk,
                    optional_reference_latents=normalizing_latent_chunk,
                    normalize_per_frame=normalize_per_frame,
                    adain_factor=sampling_config.adain_factor,
                    **common_optional,
                )[0]
            chunk_index += 1

        return tile_out_latents


NODE_CLASS_MAPPINGS = {
    "VRGDG_LTXVLoopingSamplerAdvanced": VRGDGLTXVLoopingSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LTXVLoopingSamplerAdvanced": "VRGDG_LTXV Looping Sampler Advanced",
}
