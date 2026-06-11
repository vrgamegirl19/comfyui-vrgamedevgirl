import os

import cv2
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps


NONE_IMAGE = "(none)"
NEUTRAL_GRAY = 127


def _input_images():
    input_dir = folder_paths.get_input_directory()
    files = []
    if os.path.isdir(input_dir):
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
    return [NONE_IMAGE] + sorted(files)


def _load_image_file(image_name):
    if not image_name or image_name == NONE_IMAGE:
        return None

    image_path = folder_paths.get_annotated_filepath(image_name)
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _pil_to_rgb_array(pil_image, target_size):
    pil_image = pil_image.convert("RGB")
    image_array = np.array(pil_image)
    if image_array.shape[1] == target_size[0] and image_array.shape[0] == target_size[1]:
        return np.ascontiguousarray(image_array)
    return cv2.resize(image_array, target_size, interpolation=cv2.INTER_LANCZOS4)


def _expand_frames(images, frame_count):
    base_count = frame_count // len(images)
    remainder = frame_count % len(images)
    frames = []
    for index, image in enumerate(images):
        repeats = base_count + (1 if index < remainder else 0)
        frames.extend([image] * repeats)
    return frames


class VRGDG_LTXMSRReferenceBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        images = _input_images()
        return {
            "required": {
                "subject_1": (images, {"image_upload": True}),
                "subject_2": (images, {"image_upload": True}),
                "subject_3": (images, {"image_upload": True}),
                "subject_4": (images, {"image_upload": True}),
                "background_image": (images, {"image_upload": True}),
                "background_mode": (
                    ["use_uploaded_background", "neutral_placeholder_wip"],
                    {"default": "use_uploaded_background"},
                ),
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "reference_strength": (
                    [
                        "auto - based on subject count",
                        "17 - light",
                        "25 - balanced",
                        "33 - strong",
                        "41 - strongest",
                    ],
                    {"default": "auto - based on subject count"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "build_reference"
    CATEGORY = "VRGDG/LTX MSR"

    def build_reference(
        self,
        subject_1,
        subject_2,
        subject_3,
        subject_4,
        background_image,
        background_mode,
        width,
        height,
        reference_strength,
    ):
        target_size = (width, height)
        subject_files = [subject_1, subject_2, subject_3, subject_4]
        subject_images = []

        for image_name in subject_files:
            image = _load_image_file(image_name)
            if image is not None:
                subject_images.append(_pil_to_rgb_array(image, target_size))

        if not subject_images:
            raise ValueError("At least subject_1 must be set to an uploaded image.")

        if background_mode == "neutral_placeholder_wip":
            background = np.full((height, width, 3), NEUTRAL_GRAY, dtype=np.uint8)
        else:
            background = _load_image_file(background_image)
            if background is None:
                raise ValueError(
                    "background_image is required unless background_mode is neutral_placeholder_wip."
                )
            background = _pil_to_rgb_array(background, target_size)

        frame_count = self._resolve_frame_count(reference_strength, len(subject_images))
        images = subject_images + [background]
        frames = _expand_frames(images, frame_count)
        output = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        return (output,)

    @staticmethod
    def _resolve_frame_count(reference_strength, subject_count):
        if str(reference_strength).startswith("17"):
            return 17
        if str(reference_strength).startswith("25"):
            return 25
        if str(reference_strength).startswith("33"):
            return 33
        if str(reference_strength).startswith("41"):
            return 41

        if subject_count <= 1:
            return 17
        if subject_count == 2:
            return 25
        if subject_count == 3:
            return 33
        return 41


NODE_CLASS_MAPPINGS = {
    "VRGDG_LTXMSRReferenceBuilder": VRGDG_LTXMSRReferenceBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LTXMSRReferenceBuilder": "VRGDG LTX MSR Reference Builder",
}
