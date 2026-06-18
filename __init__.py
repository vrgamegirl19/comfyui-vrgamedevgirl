import importlib
import importlib.util
import os

import folder_paths

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
    ".VRGDG_LTXICIngredientsGrid",
)

_VRGDG_OPTIONAL_SUBMODULES = (
    ".VRGDG_FileDeleteNode",
)

for _modname in _VRGDG_OPTIONAL_SUBMODULES:
    if importlib.util.find_spec(_modname, package=__name__) is not None:
        _VRGDG_SUBMODULES += (_modname,)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_VRGDG_FAILED = []

for _modname in _VRGDG_SUBMODULES:
    try:
        _mod = importlib.import_module(_modname, package=__name__)
    except Exception as exc:
        _VRGDG_FAILED.append((_modname, f"{type(exc).__name__}: {exc}"))
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

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

