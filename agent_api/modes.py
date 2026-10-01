"""Dynamic discovery and enumeration of image & video generation modes (C9, Section 16.9)."""

from typing import Any, Dict

from ..minimax.settings_payload import minimax_h3_defaults


def get_modes_catalog() -> Dict[str, Any]:
    """Generate the structured /modes catalog describing supported engines, modes, and features."""
    return {
        "video_engines": {
            "minimax_h3": {
                "name": "MiniMax Hailuo H3",
                "modes": {
                    "image_to_video": {
                        "name": "Image to Video",
                        "scene_inputs": ["approved_image", "audio"],
                        "prompt_fields": ["i2v_prompt"],
                        "settings_group": "minimax_h3",
                        # Full list: GET /settings/minimax-h3/schema (types, defaults, limits).
                        "settings_keys": sorted(minimax_h3_defaults()),
                        "supports": {
                            "scene_override": True,
                            "post_process": True,
                            "latent_continuation": True,
                            "audio_drive": True,
                            "continuity": ["off", "i2v_chain"],
                        },
                    },
                    "reference_to_video": {
                        "name": "Reference to Video",
                        "scene_inputs": ["reference_images", "audio"],
                        "limits": {"reference_images": 9, "video_references": 0},
                        "render_pass": ["single", "two_pass", "three_pass"],
                        "audio_mode": ["input_audio", "built_in_audio"],
                        "supports": {
                            "reference_images": True,
                            "audio_drive": True,
                        },
                    },
                },
            },
            "ltx_video": {
                "name": "LTX-Video 2.x",
                "modes": {
                    "image_to_video": {
                        "name": "Image to Video",
                        "scene_inputs": ["approved_image", "audio"],
                        "prompt_fields": ["i2v_prompt"],
                        "settings_group": "ltx_video",
                        "settings_keys": ["fps", "width", "height", "steps", "motion_bucket", "guidance_scale"],
                        "supports": {
                            "scene_override": True,
                            "post_process": True,
                            "latent_continuation": False,
                            "audio_drive": False,
                        },
                    },
                    "flf": {
                        "name": "First and Last Frame (FLF)",
                        "scene_inputs": ["start_image", "end_image", "audio"],
                        "settings_group": "ltx_video",
                        "settings_keys": ["fps", "width", "height", "flf_pre_frames", "flf_first_guide_strength"],
                        "supports": {
                            "end_image": True,
                            "scene_override": True,
                        },
                    },
                },
            },
        },
        "image_modes": {
            "zimage": {
                "name": "Z-Image Turbo",
                "prompt_fields": ["t2i_prompt"],
                "settings_group": "zimage",
                "settings_keys": ["steps", "guidance_scale", "sampler_name", "scheduler", "width", "height", "seed_mode"],
                "supports": {
                    "reference_conditioning": True,
                    "scene_override": True,
                },
            },
            "flux_klein": {
                "name": "FLUX Klein",
                "prompt_fields": ["t2i_prompt", "flux_prompt"],
                "settings_group": "flux_klein",
                "settings_keys": ["steps", "guidance_scale", "width", "height"],
                "supports": {
                    "prompt_enhancer": True,
                },
            },
            "ernie_image": {
                "name": "Ernie / Image Gen",
                "prompt_fields": ["t2i_prompt"],
                "settings_group": "ernie_image",
                "settings_keys": ["width", "height", "steps"],
            },
            "krea2_2pass": {
                "name": "Krea2 Two-Pass",
                "prompt_fields": ["t2i_prompt"],
                "settings_group": "krea2",
                "settings_keys": ["pass1_steps", "pass2_steps", "upscale_ratio"],
            },
        },
        "enums": {
            "continuity_mode": ["off", "i2v_chain", "img2img"],
            "video_engines": ["minimax_h3", "ltx_video"],
            "render_passes": ["single", "two_pass", "three_pass"],
            "audio_modes": ["input_audio", "built_in_audio"],
            "seed_modes": ["randomize", "fixed"],
        },
    }
