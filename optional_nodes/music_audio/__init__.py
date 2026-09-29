"""VRGDG music and audio nodes: YuE2 and SheetSage2 song generation, MiniMax Music 3 helpers, VoxCPM2 speech, and audio load/save helpers."""
import importlib

_MODULES = (
    ".Yue2",
    ".VRGDG_MiniMaxMusic3Helpers",
    ".VRGDG_VoxCPM2Node",
    ".VRGDG_AudioFileNodes",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _name in _MODULES:
    _module = importlib.import_module(_name, package=__name__)
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
