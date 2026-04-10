import os

import folder_paths

from .nodes import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_2,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_2,
)

from .HumoAutomation import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_3,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_3,
)

from .HumoAutomationExtra1 import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_4,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_4,
)

from .HumoAutomationExtra2 import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_5,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_5,
)


from .GeneralVideoNodes import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_6,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_6,
)

from .GeneralVideoNodes2 import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_7,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_7,
)

from .LLM import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_8,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_8,
)

from .VRGDGswtichNodes import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_9,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_9,
)

from .VRGDG_GeneralNodes import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_10,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_10,
)

from .VRGDG_AudioNodes import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_11,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_11,
)

from .VRGDG_GeneralNodes2 import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_12,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_12,
)

from .VRGDG_IV_Adjustments import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_13,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_13,
)

from .LTXLoraTrain import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_14,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_14,
)

from .VRGDG_VoxCPM2Node import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_15,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_15,
)

_VRGDG_TEXTFILE_TEMPLATES = (
    ("fulllyrics", "full_lyrics.txt"),
    ("themestyle", "themestyle.txt"),
    ("storyconcept", "storyconcept.txt"),
    ("storygroups", "storygroups.txt"),
    ("subjectandscenes", "subjectsandscenes.txt"),
    ("t2iNotes", "t2iNotes.txt"),
    ("i2vNotes", "i2vNotes.txt"),
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


NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS_2,
    **NODE_CLASS_MAPPINGS_3,
    **NODE_CLASS_MAPPINGS_4,
    **NODE_CLASS_MAPPINGS_5,
    **NODE_CLASS_MAPPINGS_6,  
    **NODE_CLASS_MAPPINGS_7,
    **NODE_CLASS_MAPPINGS_8,   
    **NODE_CLASS_MAPPINGS_9,
    **NODE_CLASS_MAPPINGS_10,
    **NODE_CLASS_MAPPINGS_11, 
    **NODE_CLASS_MAPPINGS_12,
    **NODE_CLASS_MAPPINGS_13,    
    **NODE_CLASS_MAPPINGS_14,
    **NODE_CLASS_MAPPINGS_15,    
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS_2,
    **NODE_DISPLAY_NAME_MAPPINGS_3,
    **NODE_DISPLAY_NAME_MAPPINGS_4,
    **NODE_DISPLAY_NAME_MAPPINGS_5,
    **NODE_DISPLAY_NAME_MAPPINGS_6,  
    **NODE_DISPLAY_NAME_MAPPINGS_7,
    **NODE_DISPLAY_NAME_MAPPINGS_8,      
    **NODE_DISPLAY_NAME_MAPPINGS_9,
    **NODE_DISPLAY_NAME_MAPPINGS_10, 
    **NODE_DISPLAY_NAME_MAPPINGS_11,
    **NODE_DISPLAY_NAME_MAPPINGS_12,   
    **NODE_DISPLAY_NAME_MAPPINGS_13,   
    **NODE_DISPLAY_NAME_MAPPINGS_14,
    **NODE_DISPLAY_NAME_MAPPINGS_15,    
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

