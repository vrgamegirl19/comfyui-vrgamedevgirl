import importlib
import importlib.util
import os
import sys

import folder_paths

__version__ = "v9-dev-2026-07-04"
__updated__ = "2026-07-04"

_VRGDG_SUBMODULES = (
    ".nodes",
    ".HumoAutomation",
    ".HumoAutomationExtra1",
    ".HumoAutomationExtra2",
    ".GeneralVideoNodes",
    ".GeneralVideoNodes2",
    ".LLM",
    ".VRGDGswtichNodes",
    ".VRGDG_GeneralNodes",
    ".VRGDG_AudioNodes",
    ".VRGDG_GeneralNodes2",
    ".VRGDG_IV_Adjustments",
    ".LTXLoraTrain",
    ".VRGDG_VoxCPM2Node",
    ".VRGDG_VideoEditorNodes",
    ".VRGDG_WorkflowRunnerNodes",
    ".VRGDG_MusicVideoBuilderNodes",
    ".VRGDG_StoryboardBuilderNodes",
    ".VRGDG_MusicVideoPromptCreatorNodes",
    ".VRGDG_ImageCompareNode",
    ".VRGDG_StandaloneFaceFixNodes",
    ".vrgdg_ltx_msr_reference_builder",
    ".VRGDG_LTXICIngredientsGrid",
    ".VRGDG_LTXFirstLastGuide",
    ".VRGDG_FlowBrowserNodes",
    ".VRGDG_BrowserImageRoutes",
    ".VRGDG_SilentAudioRoutes",
    ".VRGDG_UpdateRoutes",
    ".VRGDG_VideoBuilderNodeUI",
)

_VRGDG_OPTIONAL_SUBMODULES = (
    ".VRGDG_FileDeleteNode",
)

_VRGDG_DIR = os.path.dirname(os.path.abspath(__file__))


def _vrgdg_module_file(modname):
    module_name = str(modname or "").lstrip(".")
    if not module_name:
        return ""
    return os.path.join(_VRGDG_DIR, f"{module_name}.py")


def _vrgdg_has_submodule(modname):
    try:
        if importlib.util.find_spec(modname, package=__name__) is not None:
            return True
    except Exception:
        pass
    return os.path.isfile(_vrgdg_module_file(modname))


def _import_vrgdg_submodule(modname):
    try:
        return importlib.import_module(modname, package=__name__)
    except ModuleNotFoundError as exc:
        module_path = _vrgdg_module_file(modname)
        if not os.path.isfile(module_path):
            raise
        module_name = f"_vrgdg_custom_{os.path.splitext(os.path.basename(module_path))[0]}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise exc
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


for _modname in _VRGDG_OPTIONAL_SUBMODULES:
    if _vrgdg_has_submodule(_modname):
        _VRGDG_SUBMODULES += (_modname,)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_VRGDG_FAILED = []

for _modname in _VRGDG_SUBMODULES:
    try:
        _mod = _import_vrgdg_submodule(_modname)
    except Exception as exc:
        _VRGDG_FAILED.append((_modname, f"{type(exc).__name__}: {exc}"))
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
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

_VRGDG_TEXTFILE_TEMPLATES = (
    ("fulllyrics", "full_lyrics.txt"),
    ("themestyle", "themestyle.txt"),
    ("storyconcept", "storyconcept.txt"),
    ("storygroups", "storygroups.txt"),
    ("subjectandscenes", "subjectsandscenes.txt"),
    ("t2iNotes", "t2iNotes.txt"),
    ("i2vNotes", "i2vNotes.txt"),
    ("t2i_Prompts", "t2i_Prompts.txt"),
    ("t2v_Prompts", "t2v_Prompts.txt"),
    ("lyricsegements", "lyricsegements.txt"),
    ("themestyle2", "themestyle2.txt"),
    ("fullstory", "fullstory.txt"),
)


def _ensure_vrgdg_textfile_structure():
    base_dir = os.path.join(
        folder_paths.get_output_directory(),
        "VRGDG_TEMP",
        "TextFiles",
    )
    created_paths = []

    for folder_name, file_name in _VRGDG_TEXTFILE_TEMPLATES:
        folder_path = os.path.join(base_dir, folder_name)
        file_path = os.path.join(folder_path, file_name)
        os.makedirs(folder_path, exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write("")
            created_paths.append(file_path)

    if created_paths:
        print(f"[VRGDG] Created {len(created_paths)} placeholder text file(s) in {base_dir}")


try:
    _ensure_vrgdg_textfile_structure()
except Exception as exc:
    print(f"[VRGDG] Failed to prepare TextFiles placeholders: {exc}")


WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

