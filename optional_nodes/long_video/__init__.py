"""VRGDG long-video tools: overlap meta-batch windows and blending, the long-video meta-batch loader, and LongShot director contexts and extractors."""
import importlib

_MODULES = (
    ".VRGDG_OverlapMetaBatch",
    ".VRGDG_LongVideoMetaBatch",
    ".VRGDG_LongShotLLMContext",
    ".VRGDG_LongShotAutoDirector",
    ".VRGDG_LongShotKeyframeDirector",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _name in _MODULES:
    _module = importlib.import_module(_name, package=__name__)
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
