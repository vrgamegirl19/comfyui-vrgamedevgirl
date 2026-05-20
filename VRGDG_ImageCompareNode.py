from nodes import PreviewImage


class VRGDG_ImageCompare(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "First image to compare."}),
                "image_b": ("IMAGE", {"tooltip": "Second image to compare."}),
                "mode": (
                    ["side_by_side", "slider", "overlay", "difference", "blink"],
                    {"default": "slider"},
                ),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "slider_position": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "overlay_opacity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "blink_speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1},
                ),
                "show_labels": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_a", "image_b")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "VRGDG/Image"

    def _select_image(self, image, batch_index):
        if image is None:
            return image

        if not hasattr(image, "shape") or len(image.shape) < 4:
            return image

        batch_count = int(image.shape[0])
        if batch_count <= 0:
            return image

        selected_index = max(0, min(int(batch_index), batch_count - 1))
        return image[selected_index:selected_index + 1]

    def _preview_one(self, image, prefix, prompt, extra_pnginfo):
        if image is None:
            return []

        payload = self.save_images(
            image,
            filename_prefix=prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return payload.get("ui", {}).get("images", [])

    def compare(
        self,
        image_a,
        image_b,
        mode,
        batch_index,
        slider_position,
        overlay_opacity,
        blink_speed,
        show_labels,
        prompt=None,
        extra_pnginfo=None,
    ):
        preview_a = self._select_image(image_a, batch_index)
        preview_b = self._select_image(image_b, batch_index)

        images = []
        for info in self._preview_one(preview_a, "VRGDG_ImageCompare_A", prompt, extra_pnginfo):
            info["compare_role"] = "a"
            images.append(info)
        for info in self._preview_one(preview_b, "VRGDG_ImageCompare_B", prompt, extra_pnginfo):
            info["compare_role"] = "b"
            images.append(info)

        return {
            "ui": {
                "compare_images": images,
                "compare": {
                    "mode": mode,
                    "batch_index": int(batch_index),
                    "slider_position": float(slider_position),
                    "overlay_opacity": float(overlay_opacity),
                    "blink_speed": float(blink_speed),
                    "show_labels": bool(show_labels),
                },
            },
            "result": (image_a, image_b),
        }


NODE_CLASS_MAPPINGS = {
    "VRGDG_ImageCompare": VRGDG_ImageCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ImageCompare": "VRGDG Image Compare",
}
