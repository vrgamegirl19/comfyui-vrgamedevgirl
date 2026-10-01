import importlib

__version__ = "v9.1.1"
__updated__ = "2026-08-06"

_VRGDG_SUBMODULES = (
    ".general.lyrics",
    ".general.video",
    ".llm.api",
    ".llm.google",
    ".llm.gguf",
    ".general.audio",
    ".general.utility",
    ".post_process.luts",
    ".builder.video_editor",
    ".runner.nodes",
    ".builder.nodes",
    ".storyboard.nodes",
    ".prompt_creator.nodes",
    ".general.ltx_msr_reference",
    ".minimax.nodes",
    ".minimax.latent_upscaler",
    ".browser.nodes",
    ".core.system_routes",
    ".minimax.latent_manager",
    ".minimax.latent_continuation",
    ".agent_api",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_VRGDG_FAILED = []
_VRGDG_NODE_SOURCES = {}

for _modname in _VRGDG_SUBMODULES:
    try:
        _mod = importlib.import_module(_modname, package=__name__)
    except Exception as exc:
        _VRGDG_FAILED.append((_modname, f"{type(exc).__name__}: {exc}"))
        continue
    _mappings = getattr(_mod, "NODE_CLASS_MAPPINGS", {})
    for _name in _mappings:
        if _name in _VRGDG_NODE_SOURCES:
            print(f"[VRGDG] Warning: node {_name} from {_modname} replaces the one from {_VRGDG_NODE_SOURCES[_name]}; rename one of them.")
        _VRGDG_NODE_SOURCES[_name] = _modname
    NODE_CLASS_MAPPINGS.update(_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

print(
    f"[VRGDG] comfyui-vrgamedevgirl {__version__} loaded "
    f"(updated {__updated__}; "
    f"{len(NODE_CLASS_MAPPINGS)} nodes, {len(_VRGDG_FAILED)} failed submodule(s))."
)

if _VRGDG_FAILED:
    print("[VRGDG] Some submodules failed to import; their nodes will be unavailable:")
    for _name, _err in _VRGDG_FAILED:
        print(f"  - {_name}: {_err}")
    print(
        "[VRGDG] The rest of the pack still loaded. "
        "Install the missing dep(s) above to enable the rest."
    )

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

