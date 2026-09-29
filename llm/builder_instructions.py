"""Video Builder LLM instructions: defaults, labels, presets, and per-project and per-scene overrides."""

import os
import re
import folder_paths
from .prompts.image import _FLOW_GPT_T2I_INSTRUCTIONS, _FLUX_KLEIN_T2I_INSTRUCTIONS, _NANO_B_T2I_INSTRUCTIONS, _STANDARD_IMAGE_T2I_INSTRUCTIONS
from .prompts.video import _I2V_INSTRUCTIONS, _ID_LORA_INSTRUCTIONS, _T2V_INSTRUCTIONS
from .prompts.minimax import MINIMAX_H3_IMAGE_REFERENCE_TO_VIDEO_INSTRUCTIONS, MINIMAX_H3_IMAGE_TO_VIDEO_INSTRUCTIONS, MINIMAX_H3_FRAME_CONTINUITY_INSTRUCTIONS, MINIMAX_H3_REFERENCE_TO_VIDEO_INSTRUCTIONS, MINIMAX_H3_TEXT_TO_VIDEO_INSTRUCTIONS, MINIMAX_H3_VIDEO_TO_VIDEO_INSTRUCTIONS, MINIMAX_H3_SHORT_FILM_GUIDED_INSTRUCTIONS_BY_MODE, MINIMAX_H3_SHORT_FILM_CUSTOM_INSTRUCTIONS_BY_MODE

from ..builder.paths import _context_folder, _project_folder_from_builder_payload, _safe_builder_scene_id


_BUILDER_INSTRUCTION_DEFAULTS = {
    "flux_klein_t2i": _FLUX_KLEIN_T2I_INSTRUCTIONS,
    "flow_gpt_t2i": _FLOW_GPT_T2I_INSTRUCTIONS,
    "ernie_t2i": _STANDARD_IMAGE_T2I_INSTRUCTIONS,
    "id_lora": _ID_LORA_INSTRUCTIONS,
    "ingredients": _T2V_INSTRUCTIONS,
    "i2v": _I2V_INSTRUCTIONS,
    "krea2_t2i": _STANDARD_IMAGE_T2I_INSTRUCTIONS,
    "minimax_h3_image_to_video": MINIMAX_H3_IMAGE_TO_VIDEO_INSTRUCTIONS,
    "minimax_h3_frame_continuity": MINIMAX_H3_FRAME_CONTINUITY_INSTRUCTIONS,
    "minimax_h3_image_reference_to_video": MINIMAX_H3_IMAGE_REFERENCE_TO_VIDEO_INSTRUCTIONS,
    "minimax_h3_reference_to_video": MINIMAX_H3_REFERENCE_TO_VIDEO_INSTRUCTIONS,
    "minimax_h3_text_to_video": MINIMAX_H3_TEXT_TO_VIDEO_INSTRUCTIONS,
    "minimax_h3_video_to_video": MINIMAX_H3_VIDEO_TO_VIDEO_INSTRUCTIONS,
    "nano_b_t2i": _NANO_B_T2I_INSTRUCTIONS,
    "rtv": _T2V_INSTRUCTIONS,
    "t2v": _T2V_INSTRUCTIONS,
    "zimage_t2i": _STANDARD_IMAGE_T2I_INSTRUCTIONS,
}


for _minimax_mode, _instructions in MINIMAX_H3_SHORT_FILM_GUIDED_INSTRUCTIONS_BY_MODE.items():
    _BUILDER_INSTRUCTION_DEFAULTS[f"minimax_h3_short_film_guided_{_minimax_mode}"] = _instructions


for _minimax_mode, _instructions in MINIMAX_H3_SHORT_FILM_CUSTOM_INSTRUCTIONS_BY_MODE.items():
    _BUILDER_INSTRUCTION_DEFAULTS[f"minimax_h3_short_film_custom_{_minimax_mode}"] = _instructions


_BUILDER_INSTRUCTION_LABELS = {
    "flux_klein_t2i": "Flux/Klein Text to Image",
    "flow_gpt_t2i": "Flow/GPT Text to Image",
    "ernie_t2i": "Ernie Text to Image",
    "id_lora": "ID-LoRA I2V",
    "ingredients": "Ingredients to Video",
    "i2v": "Image to Video",
    "krea2_t2i": "Krea 2 Text to Image",
    "minimax_h3_image_to_video": "MiniMax H3 Image to Video",
    "minimax_h3_frame_continuity": "MiniMax H3 Frame-to-Frame Continuity",
    "minimax_h3_image_reference_to_video": "MiniMax H3 Image + Reference to Video",
    "minimax_h3_reference_to_video": "MiniMax H3 Reference to Video",
    "minimax_h3_text_to_video": "MiniMax H3 Text to Video",
    "minimax_h3_video_to_video": "MiniMax H3 Video to Video",
    "nano_b_t2i": "Nano B Text to Image",
    "rtv": "Reference to Video",
    "t2v": "Text to Video",
    "zimage_t2i": "ZImage Text to Image",
}


for _minimax_mode in MINIMAX_H3_SHORT_FILM_GUIDED_INSTRUCTIONS_BY_MODE:
    _mode_label = _minimax_mode.replace("_", " ").title()
    _BUILDER_INSTRUCTION_LABELS[f"minimax_h3_short_film_guided_{_minimax_mode}"] = f"MiniMax H3 Guided Short Film - {_mode_label}"
    _BUILDER_INSTRUCTION_LABELS[f"minimax_h3_short_film_custom_{_minimax_mode}"] = f"MiniMax H3 Fully Custom Short Film - {_mode_label}"


_BUILDER_INSTRUCTION_PRESET_GROUPS = {
    "ernie_t2i": "standard_image_t2i",
    "krea2_t2i": "standard_image_t2i",
    "zimage_t2i": "standard_image_t2i",
    "flow_gpt_t2i": "reference_image_t2i",
    "flux_klein_t2i": "reference_image_t2i",
    "nano_b_t2i": "reference_image_t2i",
}


_BUILDER_INSTRUCTION_PRESET_GROUP_LABELS = {
    "standard_image_t2i": "Standard Image T2I",
    "reference_image_t2i": "Reference/Image Edit T2I",
}


def _safe_builder_instruction_key(value):
    key = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if key not in _BUILDER_INSTRUCTION_DEFAULTS:
        raise ValueError(f"Unknown Builder instruction key: {value}")
    return key


def _safe_preset_name(value):
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_. -]+", "_", text).strip(" ._")
    if not text:
        raise ValueError("Preset name is empty.")
    return text[:80]


def _builder_instruction_folder(project_folder):
    return os.path.join(_context_folder(project_folder), "custom_builder_instructions")


def _builder_instruction_all_scenes_path(project_folder, key):
    return os.path.join(_builder_instruction_folder(project_folder), f"{_safe_builder_instruction_key(key)}.txt")


def _builder_instruction_scene_path(project_folder, key, scene_id):
    safe_scene = _safe_builder_scene_id(scene_id)
    if not safe_scene:
        raise ValueError("Scene id is missing.")
    return os.path.join(_builder_instruction_folder(project_folder), "scenes", safe_scene, f"{_safe_builder_instruction_key(key)}.txt")


def _builder_instruction_preset_root():
    return os.path.join(folder_paths.get_output_directory(), "VRGDG_LLM_Instruction_Presets", "builder")


def _builder_instruction_preset_group(key):
    safe_key = _safe_builder_instruction_key(key)
    return _BUILDER_INSTRUCTION_PRESET_GROUPS.get(safe_key, safe_key)


def _builder_instruction_preset_group_label(key):
    group = _builder_instruction_preset_group(key)
    return _BUILDER_INSTRUCTION_PRESET_GROUP_LABELS.get(group, _BUILDER_INSTRUCTION_LABELS.get(group, group))


def _builder_instruction_preset_path(key, name):
    return os.path.join(_builder_instruction_preset_root(), _builder_instruction_preset_group(key), f"{_safe_preset_name(name)}.txt")


def _legacy_builder_instruction_preset_path(key, name):
    return os.path.join(_builder_instruction_preset_root(), _safe_builder_instruction_key(key), f"{_safe_preset_name(name)}.txt")


def _read_optional_text_file(path):
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        return handle.read().strip()


def _builder_instruction_state(project_folder, key, scene_id=""):
    key = _safe_builder_instruction_key(key)
    default_text = _BUILDER_INSTRUCTION_DEFAULTS[key]
    scene_path = ""
    scene_text = ""
    if scene_id:
        scene_path = _builder_instruction_scene_path(project_folder, key, scene_id)
        scene_text = _read_optional_text_file(scene_path)
    all_path = _builder_instruction_all_scenes_path(project_folder, key)
    all_text = _read_optional_text_file(all_path)
    if scene_text:
        source = "scene"
        text = scene_text
        path = scene_path
    elif all_text:
        source = "all_scenes"
        text = all_text
        path = all_path
    else:
        source = "default"
        text = default_text
        path = ""
    return {
        "key": key,
        "label": _BUILDER_INSTRUCTION_LABELS.get(key, key),
        "scene_id": str(scene_id or ""),
        "default_text": default_text,
        "scene_text": scene_text,
        "all_scenes_text": all_text,
        "text": text,
        "source": source,
        "path": path,
        "scene_path": scene_path,
        "all_scenes_path": all_path,
        "has_scene_custom": bool(scene_text),
        "has_all_scenes_custom": bool(all_text),
    }


def _effective_builder_instruction(payload, key, default_text):
    project_folder = str(payload.get("project_folder", "") or "").strip().strip('"')
    if not project_folder:
        return str(default_text or "")
    try:
        state = _builder_instruction_state(os.path.abspath(project_folder), key, payload.get("scene_id", ""))
        return str(state.get("text") or default_text or "")
    except Exception:
        return str(default_text or "")


def _get_builder_instruction(payload):
    project_folder = _project_folder_from_builder_payload(payload)
    key = _safe_builder_instruction_key(payload.get("key"))
    return {
        "project_folder": project_folder,
        **_builder_instruction_state(project_folder, key, payload.get("scene_id", "")),
    }


def _save_builder_instruction(payload):
    project_folder = _project_folder_from_builder_payload(payload)
    key = _safe_builder_instruction_key(payload.get("key"))
    scope = str(payload.get("scope", "scene") or "scene").strip().lower()
    text = str(payload.get("text", "") or "").strip()
    if not text:
        raise ValueError("Instruction text is empty.")
    if scope in {"all", "all_scenes", "global"}:
        path = _builder_instruction_all_scenes_path(project_folder, key)
    else:
        path = _builder_instruction_scene_path(project_folder, key, payload.get("scene_id", ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    return _get_builder_instruction({"project_folder": project_folder, "key": key, "scene_id": payload.get("scene_id", "")})


def _reset_builder_instruction(payload):
    project_folder = _project_folder_from_builder_payload(payload)
    key = _safe_builder_instruction_key(payload.get("key"))
    scope = str(payload.get("scope", "scene") or "scene").strip().lower()
    if scope in {"all", "all_scenes", "global"}:
        path = _builder_instruction_all_scenes_path(project_folder, key)
    else:
        path = _builder_instruction_scene_path(project_folder, key, payload.get("scene_id", ""))
    if os.path.isfile(path):
        os.remove(path)
    return _get_builder_instruction({"project_folder": project_folder, "key": key, "scene_id": payload.get("scene_id", "")})


def _list_builder_instruction_presets(payload):
    key = _safe_builder_instruction_key(payload.get("key"))
    group = _builder_instruction_preset_group(key)
    folder = os.path.join(_builder_instruction_preset_root(), group)
    presets = []
    seen = set()
    folders = [(folder, False)]
    legacy_folder = os.path.join(_builder_instruction_preset_root(), key)
    if os.path.normcase(os.path.abspath(legacy_folder)) != os.path.normcase(os.path.abspath(folder)):
        folders.append((legacy_folder, True))
    for scan_folder, legacy in folders:
        if not os.path.isdir(scan_folder):
            continue
        for filename in os.listdir(scan_folder):
            if not filename.lower().endswith(".txt"):
                continue
            preset_name = os.path.splitext(filename)[0]
            dedupe_key = preset_name.lower()
            if dedupe_key in seen:
                continue
            path = os.path.join(scan_folder, filename)
            if os.path.isfile(path):
                seen.add(dedupe_key)
                presets.append({
                    "name": preset_name,
                    "path": os.path.abspath(path),
                    "updated": os.path.getmtime(path),
                    "legacy": legacy,
                })
    presets.sort(key=lambda item: item.get("updated", 0), reverse=True)
    return {
        "key": key,
        "label": _BUILDER_INSTRUCTION_LABELS.get(key, key),
        "preset_group": group,
        "preset_group_label": _builder_instruction_preset_group_label(key),
        "presets": presets,
        "preset_folder": folder,
    }


def _save_builder_instruction_preset(payload):
    key = _safe_builder_instruction_key(payload.get("key"))
    name = _safe_preset_name(payload.get("name"))
    text = str(payload.get("text", "") or "").strip()
    if not text:
        raise ValueError("Preset instruction text is empty.")
    path = _builder_instruction_preset_path(key, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    return {
        "key": key,
        "name": name,
        "path": path,
        "preset_folder": os.path.dirname(path),
        "preset_group": _builder_instruction_preset_group(key),
        "preset_group_label": _builder_instruction_preset_group_label(key),
    }


def _load_builder_instruction_preset(payload):
    key = _safe_builder_instruction_key(payload.get("key"))
    name = _safe_preset_name(payload.get("name"))
    path = _builder_instruction_preset_path(key, name)
    text = _read_optional_text_file(path)
    if not text:
        legacy_path = _legacy_builder_instruction_preset_path(key, name)
        if os.path.normcase(os.path.abspath(legacy_path)) != os.path.normcase(os.path.abspath(path)):
            legacy_text = _read_optional_text_file(legacy_path)
            if legacy_text:
                path = legacy_path
                text = legacy_text
    if not text:
        raise FileNotFoundError(f"Instruction preset was not found or is empty: {path}")
    return {
        "key": key,
        "name": name,
        "path": path,
        "preset_folder": os.path.dirname(path),
        "preset_group": _builder_instruction_preset_group(key),
        "preset_group_label": _builder_instruction_preset_group_label(key),
        "text": text,
    }
