"""Typed settings schemas and extraction for the VRGDG Agent API (C5, D5, Section 16)."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..minimax.settings_payload import (
    minimax_h3_defaults,
    normalize_minimax_h3_settings,
    validate_minimax_h3_patch,
)
from .errors import SettingsInvalidError


SETTINGS_VERSION = 1


@dataclass
class ProjectSettings:
    video_engine: str = "minimax_h3"
    video_model_mode: str = "image_to_video"
    image_model_mode: str = "zimage"
    continuity_mode: str = "off"
    timing_frozen: bool = False
    omit_lyrics_from_video_prompts: bool = False
    use_structured_outputs: bool = False
    auto_save_enabled: bool = True
    automatic_memory_cleanup: bool = True
    scene_render_wait_hours: float = 2.0


@dataclass
class LtxVideoSettings:
    fps: int = 24
    width: int = 1280
    height: int = 720
    steps: int = 30
    motion_bucket: int = 127
    guidance_scale: float = 3.0
    flf_pre_frames: int = 8
    flf_first_guide_strength: float = 0.5


@dataclass
class ZImageSettings:
    steps: int = 20
    guidance_scale: float = 4.0
    sampler_name: str = "euler"
    scheduler: str = "normal"
    width: int = 1280
    height: int = 720
    seed_mode: str = "randomize"


@dataclass
class FluxKleinSettings:
    steps: int = 4
    guidance_scale: float = 1.0
    width: int = 1280
    height: int = 720


@dataclass
class LlmSettings:
    text_gemma_runner: str = "auto"
    gemma_context_limit: int = 8000
    gemma_output_token_limit: int = 8192
    gemma_gpu_layers: int = 99
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str = ""
    llm_api_provider: str = "openai"
    llm_api_model: str = ""
    own_server_url: str = "http://127.0.0.1:8000/v1"
    own_server_model: str = ""
    own_server_timeout: int = 360


@dataclass
class PostProcessSettings:
    lut_enabled: bool = False
    lut_path: str = ""
    lut_strength: float = 1.0
    grain_enabled: bool = False
    grain_strength: float = 0.2
    adjust_enabled: bool = False
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0


@dataclass
class EffectiveSettings:
    settings_version: int = SETTINGS_VERSION
    project: ProjectSettings = field(default_factory=ProjectSettings)
    # Every MiniMax H3 option the Video Builder saves (see minimax/h3_settings_defaults.json).
    minimax_h3: Dict[str, Any] = field(default_factory=minimax_h3_defaults)
    ltx_video: LtxVideoSettings = field(default_factory=LtxVideoSettings)
    zimage: ZImageSettings = field(default_factory=ZImageSettings)
    flux_klein: FluxKleinSettings = field(default_factory=FluxKleinSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    post_process: PostProcessSettings = field(default_factory=PostProcessSettings)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_effective_settings(session: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize all settings from a project session dict into structured groups."""
    if not isinstance(session, dict):
        return EffectiveSettings().to_dict()

    project = ProjectSettings(
        video_engine=str(session.get("video_engine") or "minimax_h3"),
        video_model_mode=str(session.get("video_model_mode") or "image_to_video"),
        image_model_mode=str(session.get("image_model_mode") or "zimage"),
        continuity_mode=str(session.get("continuity_mode") or "off"),
        timing_frozen=bool(session.get("timing_frozen", False)),
        auto_save_enabled=bool(session.get("auto_save_enabled", True)),
        omit_lyrics_from_video_prompts=bool(session.get("omit_lyrics_from_video_prompts", False)),
        use_structured_outputs=bool(session.get("use_structured_outputs", False)),
        automatic_memory_cleanup=bool(session.get("automatic_memory_cleanup", True)),
        scene_render_wait_hours=float(session.get("scene_render_wait_hours", 2.0)),
    )

    mm_raw = session.get("minimax_h3_settings") if isinstance(session.get("minimax_h3_settings"), dict) else {}
    minimax = normalize_minimax_h3_settings(mm_raw)

    ltx_raw = session.get("i2v_video_settings") if isinstance(session.get("i2v_video_settings"), dict) else {}
    ltx = LtxVideoSettings(
        fps=int(ltx_raw.get("fps", 24)),
        width=int(ltx_raw.get("width", 1280)),
        height=int(ltx_raw.get("height", 720)),
        steps=int(ltx_raw.get("steps", 30)),
        motion_bucket=int(ltx_raw.get("motion_bucket", 127)),
        guidance_scale=float(ltx_raw.get("guidance_scale", 3.0)),
    )

    zimg_raw = session.get("zimage_settings") if isinstance(session.get("zimage_settings"), dict) else {}
    zimage = ZImageSettings(
        steps=int(zimg_raw.get("steps", 20)),
        guidance_scale=float(zimg_raw.get("guidance_scale", 4.0)),
        sampler_name=str(zimg_raw.get("sampler_name") or "euler"),
        scheduler=str(zimg_raw.get("scheduler") or "normal"),
        width=int(zimg_raw.get("width", 1280)),
        height=int(zimg_raw.get("height", 720)),
        seed_mode=str(zimg_raw.get("seed_mode") or "randomize"),
    )

    flux_raw = session.get("flux_klein_settings") if isinstance(session.get("flux_klein_settings"), dict) else {}
    flux = FluxKleinSettings(
        steps=int(flux_raw.get("steps", 4)),
        guidance_scale=float(flux_raw.get("guidance_scale", 1.0)),
        width=int(flux_raw.get("width", 1280)),
        height=int(flux_raw.get("height", 720)),
    )

    llm = LlmSettings(
        text_gemma_runner=str(session.get("text_gemma_runner") or "auto"),
        gemma_context_limit=int(session.get("gemma_context_limit", 8000)),
        gemma_output_token_limit=int(session.get("gemma_output_token_limit", 8192)),
        gemma_gpu_layers=int(session.get("gemma_gpu_layers", 99)),
        lm_studio_base_url=str(session.get("lm_studio_base_url") or "http://127.0.0.1:1234/v1"),
        lm_studio_model=str(session.get("lm_studio_model") or ""),
        llm_api_provider=str(session.get("llm_api_provider") or "openai"),
        llm_api_model=str(session.get("llm_api_model") or ""),
        own_server_url=str(session.get("own_server_url") or "http://127.0.0.1:8000/v1"),
        own_server_model=str(session.get("own_server_model") or ""),
        own_server_timeout=int(session.get("own_server_timeout", 360)),
    )

    post = PostProcessSettings(
        lut_enabled=bool(session.get("lut_enabled", False)),
        lut_path=str(session.get("lut_path") or ""),
        lut_strength=float(session.get("lut_strength", 1.0)),
        grain_enabled=bool(session.get("grain_enabled", False)),
        grain_strength=float(session.get("grain_strength", 0.2)),
        adjust_enabled=bool(session.get("adjust_enabled", False)),
        brightness=float(session.get("brightness", 0.0)),
        contrast=float(session.get("contrast", 1.0)),
        saturation=float(session.get("saturation", 1.0)),
    )

    effective = EffectiveSettings(
        settings_version=SETTINGS_VERSION,
        project=project,
        minimax_h3=minimax,
        ltx_video=ltx,
        zimage=zimage,
        flux_klein=flux,
        llm=llm,
        post_process=post,
    )
    return effective.to_dict()


def validate_settings_patch(patch: Dict[str, Any]) -> None:
    """Validate a patch dictionary against known settings groups and types."""
    if not isinstance(patch, dict):
        raise SettingsInvalidError("Settings patch must be a JSON object.")

    errors = {}
    known_groups = {"project", "minimax_h3", "ltx_video", "zimage", "flux_klein", "llm", "post_process"}
    for key, value in patch.items():
        if key not in known_groups:
            # Top-level direct keys or groups are both allowed
            continue
        if not isinstance(value, dict):
            errors[key] = f"Group '{key}' must be a dictionary of settings."
        elif key == "minimax_h3":
            for setting, problem in validate_minimax_h3_patch(value).items():
                errors[f"minimax_h3.{setting}"] = problem

    if errors:
        raise SettingsInvalidError("Settings patch contained invalid groups.", invalid_fields=errors)
