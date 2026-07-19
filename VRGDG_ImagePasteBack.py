import os

import torch
import torch.nn.functional as F


def _batch_item(tensor, index):
    return tensor[min(index, tensor.shape[0] - 1)]


def _soft_blend_mask(height, width, inset, feather, shape, device, dtype):
    yy = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, width)
    inset = max(0.0, min(float(inset), (min(width, height) - 1) / 2.0))

    if shape == "ellipse":
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        rx, ry = max(0.5, cx - inset), max(0.5, cy - inset)
        distance = 1.0 - torch.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        # Convert the normalized ellipse distance to an approximate pixel distance.
        distance = distance * min(rx, ry)
    else:
        distance = torch.minimum(
            torch.minimum(xx - inset, (width - 1 - inset) - xx),
            torch.minimum(yy - inset, (height - 1 - inset) - yy),
        )

    if feather <= 0:
        return (distance >= 0).to(dtype)
    return torch.clamp(distance / float(feather), 0.0, 1.0)


def _match_color(source, target, alpha, strength):
    if strength <= 0:
        return source
    selected = alpha[..., 0] > 0.25
    if int(selected.sum()) < 16:
        return source
    src_mean = source[selected].mean(dim=0)
    dst_mean = target[selected].mean(dim=0)
    return torch.clamp(source + (dst_mean - src_mean) * float(strength), 0.0, 1.0)


class VRGDG_ModernFaceCrop:
    """Confidence-based MediaPipe face crop with WAS-compatible crop data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_range": (["full_range", "short_range"], {"default": "full_range"}),
                "confidence": ("FLOAT", {"default": 0.70, "min": 0.1, "max": 0.99, "step": 0.01}),
                "crop_padding_factor": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 2.0, "step": 0.01}),
                "minimum_face_pixels": ("INT", {"default": 24, "min": 4, "max": 2048, "step": 1}),
                "face_selection": (["highest_confidence", "largest", "closest_to_center"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA", "FLOAT")
    RETURN_NAMES = ("cropped_face", "crop_data", "detection_confidence")
    FUNCTION = "crop_face"
    CATEGORY = "VRGameDevGirl/Image"
    DESCRIPTION = "OpenCV DNN face detection with confidence filtering, tiled long-range scanning, and WAS-compatible CROP_DATA."

    def crop_face(self, image, model_range, confidence, crop_padding_factor,
                  minimum_face_pixels, face_selection):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for Modern Face Crop.") from exc

        frame = image[0]
        height, width = frame.shape[:2]
        rgb = (torch.clamp(frame[..., :3], 0.0, 1.0).detach().cpu().numpy() * 255.0).round().astype("uint8")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        config_path = os.path.join(assets_dir, "opencv_face_deploy.prototxt")
        model_path = os.path.join(assets_dir, "opencv_face_res10_fp16.caffemodel")
        if not os.path.isfile(config_path) or not os.path.isfile(model_path):
            raise RuntimeError("OpenCV DNN face detector assets are missing from the custom node's assets folder.")
        detector = cv2.dnn.readNetFromCaffe(config_path, model_path)

        # Full range also scans four overlapping tiles. This keeps distant faces large
        # enough for the 300x300 detector instead of shrinking the whole wide shot once.
        regions = [(0, 0, width, height)]
        if model_range == "full_range" and width >= 600 and height >= 600:
            tile_w, tile_h = int(round(width * 0.60)), int(round(height * 0.60))
            regions.extend([
                (0, 0, tile_w, tile_h),
                (width - tile_w, 0, width, tile_h),
                (0, height - tile_h, tile_w, height),
                (width - tile_w, height - tile_h, width, height),
            ])

        candidates = []
        for region_left, region_top, region_right, region_bottom in regions:
            region = bgr[region_top:region_bottom, region_left:region_right]
            region_h, region_w = region.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(region, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False,
            )
            detector.setInput(blob)
            detections = detector.forward()
            for detection in detections[0, 0]:
                score = float(detection[2])
                if score < float(confidence):
                    continue
                x = region_left + int(round(float(detection[3]) * region_w))
                y = region_top + int(round(float(detection[4]) * region_h))
                right = region_left + int(round(float(detection[5]) * region_w))
                bottom = region_top + int(round(float(detection[6]) * region_h))
                x, y = max(region_left, x), max(region_top, y)
                right, bottom = min(region_right, right), min(region_bottom, bottom)
                w, h = right - x, bottom - y
                if min(w, h) < int(minimum_face_pixels):
                    continue
                cx, cy = x + w / 2.0, y + h / 2.0
                center_distance = ((cx - width / 2.0) / width) ** 2 + ((cy - height / 2.0) / height) ** 2
                candidates.append((x, y, w, h, score, center_distance))

        # Candidate fields are x, y, width, height, confidence, center distance.
        filtered_candidates = []
        for candidate in sorted(candidates, key=lambda item: item[4], reverse=True):
            x, y, w, h = candidate[:4]
            duplicate = False
            for kept in filtered_candidates:
                kx, ky, kw, kh = kept[:4]
                intersection = max(0, min(x + w, kx + kw) - max(x, kx)) * max(0, min(y + h, ky + kh) - max(y, ky))
                union = w * h + kw * kh - intersection
                if union > 0 and intersection / union > 0.35:
                    duplicate = True
                    break
            if not duplicate:
                filtered_candidates.append(candidate)
        candidates = filtered_candidates

        # Keep this separate from detection so minimum size also protects against
        # tiny high-confidence artifacts from tiled scans.
        candidates = [item for item in candidates if min(item[2], item[3]) >= int(minimum_face_pixels)]

        if not candidates:
            raise ValueError(
                "No face passed the detection settings. Try full_range, lower confidence slightly, "
                "or reduce minimum_face_pixels."
            )
        if face_selection == "largest":
            chosen = max(candidates, key=lambda item: item[2] * item[3])
        elif face_selection == "closest_to_center":
            chosen = min(candidates, key=lambda item: item[5])
        else:
            chosen = max(candidates, key=lambda item: item[4])

        x, y, face_w, face_h, score, _ = chosen
        side = max(face_w, face_h) * (1.0 + 2.0 * float(crop_padding_factor))
        side = max(float(minimum_face_pixels), side)
        cx, cy = x + face_w / 2.0, y + face_h / 2.0
        left, top = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
        right, bottom = int(round(cx + side / 2.0)), int(round(cy + side / 2.0))

        # Shift the square inside the image instead of distorting its size at an edge.
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > width:
            left -= right - width
            right = width
        if bottom > height:
            top -= bottom - height
            bottom = height
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        crop = image[:, top:bottom, left:right, :].clone()
        crop_data = ((right - left, bottom - top), (left, top, right, bottom))
        return (crop, crop_data, score)


class VRGDG_ImagePasteBack:
    """Resize and softly composite an enhanced crop into its original rectangle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "enhanced_crop": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "inset_padding": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "feather_strength": ("INT", {"default": 24, "min": 0, "max": 1024, "step": 1}),
                "blend_shape": (["ellipse", "rectangle"], {"default": "ellipse"}),
                "color_match": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "blend_mask")
    FUNCTION = "paste_back"
    CATEGORY = "VRGameDevGirl/Image"
    DESCRIPTION = (
        "Pastes an enhanced crop back using WAS Image Crop Face CROP_DATA, then "
        "blends the edge with padding and feathering."
    )

    def paste_back(self, original_image, enhanced_crop, crop_data,
                   inset_padding, feather_strength, blend_shape, color_match, mask=None):
        if crop_data is False or not crop_data:
            raise ValueError("No valid CROP_DATA. Connect Image Crop Face's CROP_DATA output.")
        try:
            _original_size, box = crop_data
            x, y, right_edge, bottom_edge = (int(value) for value in box)
            crop_width, crop_height = right_edge - x, bottom_edge - y
        except (TypeError, ValueError) as exc:
            raise ValueError("Unsupported CROP_DATA format; connect WAS Image Crop Face directly.") from exc
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError(f"Invalid crop rectangle in CROP_DATA: {box!r}")
        batch = max(original_image.shape[0], enhanced_crop.shape[0], mask.shape[0] if mask is not None else 1)
        outputs, masks = [], []

        for index in range(batch):
            original = _batch_item(original_image, index).clone()
            height, width = original.shape[:2]
            left, top = min(int(x), width), min(int(y), height)
            right = min(left + int(crop_width), width)
            bottom = min(top + int(crop_height), height)
            paste_w, paste_h = right - left, bottom - top
            full_mask = torch.zeros((height, width), device=original.device, dtype=original.dtype)

            if paste_w <= 0 or paste_h <= 0:
                outputs.append(original)
                masks.append(full_mask)
                continue

            crop = _batch_item(enhanced_crop, index).to(device=original.device, dtype=original.dtype)
            crop = F.interpolate(crop.permute(2, 0, 1).unsqueeze(0), size=(int(crop_height), int(crop_width)),
                                 mode="bicubic", align_corners=False)[0].permute(1, 2, 0)
            crop = crop[:paste_h, :paste_w, :original.shape[2]]

            alpha = _soft_blend_mask(int(crop_height), int(crop_width), inset_padding,
                                     feather_strength, blend_shape, original.device, original.dtype)
            alpha = alpha[:paste_h, :paste_w]
            if mask is not None:
                user_mask = _batch_item(mask, index).to(device=original.device, dtype=original.dtype)
                if user_mask.ndim == 3:
                    user_mask = user_mask[..., 0]
                user_mask = F.interpolate(user_mask[None, None], size=(int(crop_height), int(crop_width)),
                                          mode="bilinear", align_corners=False)[0, 0]
                alpha = alpha * torch.clamp(user_mask[:paste_h, :paste_w], 0.0, 1.0)

            alpha3 = alpha.unsqueeze(-1)
            target = original[top:bottom, left:right, :crop.shape[2]]
            crop = _match_color(crop, target, alpha3, color_match)
            original[top:bottom, left:right, :crop.shape[2]] = target * (1.0 - alpha3) + crop * alpha3
            full_mask[top:bottom, left:right] = alpha
            outputs.append(torch.clamp(original, 0.0, 1.0))
            masks.append(full_mask)

        return (torch.stack(outputs), torch.stack(masks))


NODE_CLASS_MAPPINGS = {
    "VRGDG_ModernFaceCrop": VRGDG_ModernFaceCrop,
    "VRGDG_ImagePasteBack": VRGDG_ImagePasteBack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ModernFaceCrop": "VRGDG Modern Face Crop (DNN)",
    "VRGDG_ImagePasteBack": "VRGDG Image Paste Back (Feathered)",
}
