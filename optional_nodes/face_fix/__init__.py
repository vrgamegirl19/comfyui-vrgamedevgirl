"""VRGDG Face Fix and Video Enhance nodes: shot-aware face repair, LTX face-crop round trips, video enhancement and image paste-back."""
import importlib

_MODULES = (
    ".VRGDG_StandaloneFaceFixNodes",
    ".VRGDG_VideoEnhanceNodes",
    ".VRGDG_StandaloneVideoEnhancerNodes",
    ".VRGDG_ImagePasteBack",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _name in _MODULES:
    _module = importlib.import_module(_name, package=__name__)
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
