import os


import folder_paths


def _safe_subfolder_path(base_dir, subfolder):
    base_abs = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.join(base_abs, str(subfolder or "")))
    if os.path.commonpath([base_abs, candidate]) != base_abs:
        raise ValueError("Image subfolder escapes the allowed ComfyUI folder.")
    return candidate


def _resolve_comfy_image_path(image_info):
    filename = os.path.basename(str(image_info.get("filename", "") or ""))
    if not filename:
        raise ValueError("Image filename is empty.")
    image_type = str(image_info.get("type", "output") or "output").lower()
    if image_type == "temp":
        base_dir = folder_paths.get_temp_directory()
    elif image_type == "input":
        base_dir = folder_paths.get_input_directory()
    else:
        base_dir = folder_paths.get_output_directory()
    folder = _safe_subfolder_path(base_dir, image_info.get("subfolder", ""))
    image_path = os.path.abspath(os.path.join(folder, filename))
    if os.path.commonpath([os.path.abspath(base_dir), image_path]) != os.path.abspath(base_dir):
        raise ValueError("Image path escapes the allowed ComfyUI folder.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Generated image was not found: {image_path}")
    return image_path
