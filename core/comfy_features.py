"""What the running ComfyUI has built in, so the pack uses it and asks for a custom node pack only when it is missing.

Also whether the optional ``nvidia-vfx`` library (RTX Video Super Resolution) is installed.
"""

import importlib
import re
from typing import Optional, Tuple

Version = Tuple[int, int, int]

# The MiniMax H3 latent upscaler: LatentUpscaleModelLoader reads its checkpoints and MiniMaxH3LatentUpscaleWithModel applies
# it (ComfyUI commit 35060f015, first in v0.44.0). Before that the Comfyui_Minimax_h3_latent_Upscaler pack is needed.
MINIMAX_H3_LATENT_UPSCALER_SINCE: Version = (0, 44, 0)


def parse_version(text) -> Optional[Version]:
    """``"0.45.0"``, ``"v0.44"`` or ``"0.44.0rc1"`` as ``(major, minor, patch)``, or None when it is not a version."""
    match = re.match(r"\s*v?(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def version_text(version: Optional[Version]) -> str:
    """The version as ``"0.45.0"``, or an empty string when it is unknown."""
    return ".".join(str(part) for part in version) if version else ""


def comfyui_version() -> Optional[Version]:
    """The version of the running ComfyUI (its ``comfyui_version.py``), or None when it cannot be read."""
    try:
        from comfyui_version import __version__
    except Exception:
        return None
    return parse_version(__version__)


def has_built_in(since: Version, version: Optional[Version] = None) -> bool:
    """True when ComfyUI ``version`` (default: the running one) is ``since`` or newer. An unknown version is not."""
    version = comfyui_version() if version is None else version
    return version is not None and version >= since


def nvidia_vsr_available() -> bool:
    """True when ``nvvfx`` (pip package ``nvidia-vfx``) imports, which KJNodes' ``nvidia_rtx_vsr`` resize method needs.

    Import caches are cleared first, so a package installed while ComfyUI runs is found without a restart.
    """
    importlib.invalidate_caches()
    try:
        import nvvfx  # noqa: F401
    except Exception:
        return False
    return True


def native_minimax_h3_upscaler() -> bool:
    """True when this ComfyUI has the MiniMax H3 latent upscaler built in (0.44.0 and newer) and its nodes import."""
    if not has_built_in(MINIMAX_H3_LATENT_UPSCALER_SINCE):
        return False
    try:
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader  # noqa: F401
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3LatentUpscaleWithModel  # noqa: F401
    except Exception:
        return False
    return True
