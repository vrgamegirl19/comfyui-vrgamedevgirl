"""VRGDG general utility nodes: text load/save and cycling pickers, prompt splitters and chunkers, run-index helpers, JSON/string tools, image switches, group/mute state toggles, note boxes, film grain, sharpening and color matching, LUTs, image/video compare, audio split loaders and final-video helpers."""
import importlib

_MODULES = (
    ".VRGDG_GeneralNodes",
    ".VRGDGswtichNodes",
    ".VRGDG_ImageCompareNode",
    ".VRGDG_VideoCompareNode",
    ".VRGDG_EnsureVideoAudio",
    ".VRGDG_VideoPromptReconstructor",
    ".VRGDG_UtilityNodes",
    ".VRGDG_GeneralVideoNodes",
    ".VRGDG_GeneralVideoNodes2",
    ".VRGDG_GeneralNodes2",
    ".VRGDG_LUTNodes",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _name in _MODULES:
    _module = importlib.import_module(_name, package=__name__)
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
