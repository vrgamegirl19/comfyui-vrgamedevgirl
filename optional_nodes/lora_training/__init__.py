"""VRGDG LoRA training nodes: LTX 2.3, Krea 2 and Z-Image LoRA trainers, Musubi/AI Toolkit installers, XYZ preview plots and the LoRA Dataset Creator."""
import importlib

_MODULES = (
    ".LTXLoraTrain",
    ".VRGDG_LoraDatasetCreatorNodes",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _name in _MODULES:
    _module = importlib.import_module(_name, package=__name__)
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
