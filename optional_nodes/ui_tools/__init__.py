"""VRGDG UI tool nodes: Video Editor, Prompt Creator V1/V2 with the Part 2/3 workflow UIs and T2V-from-concepts, Start Image Storyboard and the Node Canvas prototype."""
import importlib

_MODULES = (
    ".VRGDG_StartImageStoryboard",
    ".VRGDG_VideoBuilderNodeUI",
    ".VRGDG_VideoEditorNodes",
    ".VRGDG_PromptCreatorNodes",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _name in _MODULES:
    _module = importlib.import_module(_name, package=__name__)
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
