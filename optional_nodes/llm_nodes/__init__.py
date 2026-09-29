"""VRGDG standalone LLM nodes: Qwen 3.5/2.5 and General VLM (Hugging Face transformers), General GGUF and Local LLM runners, and the llama.cpp doctor."""
import importlib

_MODULES = (
    ".VRGDG_LLMNodes",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _name in _MODULES:
    _module = importlib.import_module(_name, package=__name__)
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
