"""The image workflows the Thumbnail dialog sends a frame through: model names that match what ComfyUI lists, Flux/Klein, and MOSS-Image image to image."""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
models = importlib.import_module(f"{pkg_name}.runner.models")
workflows = importlib.import_module(f"{pkg_name}.runner.image_workflows")

# What a ComfyUI that keeps its models in the models folders themselves (no `flux` sub folder) lists.
INSTALLED = {
    "diffusion_models": ["flux-2-klein-4b-fp8.safetensors", "moss_image_sft_bf16.safetensors", "Other\\Model.safetensors"],
    "text_encoders": ["qwen_3_4b.safetensors", "qwen3.5_9b_int8_convrot.safetensors"],
    "vae": ["flux2-vae.safetensors"],
    "loras": ["moss_image_sft_8_steps_dmd_lora.safetensors"],
}


def fake_folder_choices(category):
    key = category[0] if isinstance(category, (list, tuple)) else category
    if key == "unet":
        key = "diffusion_models"
    if key == "clip":
        key = "text_encoders"
    return list(INSTALLED.get(key, []))


class ResolveModelChoiceTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(models, "_folder_choices", fake_folder_choices)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_listed_spelling_is_kept(self):
        self.assertEqual(models._resolve_model_choice("vae", "flux2-vae.safetensors"), "flux2-vae.safetensors")

    def test_a_factory_default_with_a_folder_finds_the_installed_file(self):
        self.assertEqual(models._resolve_model_choice("vae", "flux\\flux2-vae.safetensors"), "flux2-vae.safetensors")
        self.assertEqual(models._resolve_model_choice(("diffusion_models", "unet"), "flux\\flux-2-klein-4b-fp8.safetensors"),
                         "flux-2-klein-4b-fp8.safetensors")

    def test_slash_direction_and_case_do_not_matter_and_the_installed_spelling_comes_back(self):
        self.assertEqual(models._resolve_model_choice("diffusion_models", "other/model.SAFETENSORS"), "Other\\Model.safetensors")
        self.assertEqual(models._resolve_model_choice("vae", "FLUX2-VAE.safetensors"), "flux2-vae.safetensors")

    def test_a_model_that_is_not_installed_or_not_named_gives_none(self):
        self.assertIsNone(models._resolve_model_choice("vae", "ae.safetensors"))
        self.assertIsNone(models._resolve_model_choice("vae", ""))
        self.assertIsNone(models._resolve_model_choice("vae", None))


class FluxKleinNamesTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(models, "_folder_choices", fake_folder_choices)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.picture = os.path.join(self.folder.name, "frame.png")
        Image.new("RGB", (64, 36), (10, 10, 10)).save(self.picture)

    def payload(self, **changes):
        base = {
            "prompt": "Recrop the attached image.", "image_ingredients": [{"path": self.picture}],
            "unet_name": "flux\\flux-2-klein-4b-fp8.safetensors", "clip_name": "qwen_3_4b.safetensors",
            "vae_name": "flux\\flux2-vae.safetensors", "width": 1280, "height": 720, "seed": 5,
        }
        return {**base, **changes}

    def test_the_default_names_become_what_comfyui_lists(self):
        graph = workflows._build_flux_klein_api_prompt(self.payload())["prompt"]
        self.assertEqual(graph["1068"]["inputs"]["unet_name"], "flux-2-klein-4b-fp8.safetensors")
        self.assertEqual(graph["1066"]["inputs"]["clip_name"], "qwen_3_4b.safetensors")
        self.assertEqual(graph["1064"]["inputs"]["vae_name"], "flux2-vae.safetensors")

    def test_a_model_that_is_not_installed_says_which_one(self):
        with self.assertRaises(ValueError) as caught:
            workflows._build_flux_klein_api_prompt(self.payload(vae_name="flux\\missing-vae.safetensors"))
        self.assertIn("Flux/Klein VAE 'flux\\missing-vae.safetensors' was not found in ComfyUI/models/vae", str(caught.exception))


class MossImageToImageTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(models, "_folder_choices", fake_folder_choices)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **changes):
        payload = {"prompt": "A neon portrait.", "width": 1280, "height": 720, "seed": 3, "steps": 8, "cfg": 1, **changes}
        return workflows._build_moss_image_api_prompt(payload)["prompt"]

    def test_text_to_image_is_unchanged(self):
        graph = self.build()
        self.assertEqual(graph["9"]["class_type"], "EmptyFlux2LatentImage")
        self.assertEqual(graph["10"]["inputs"]["latent_image"], ["9", 0])
        self.assertEqual(graph["10"]["inputs"]["denoise"], 1.0)
        self.assertFalse({"13", "14", "15"} & set(graph))

    def test_image_to_image_encodes_the_picture_and_samples_from_it_at_the_chosen_denoise(self):
        with patch.object(workflows, "_prepare_load_image_name", return_value="vrgdg_i2i_1.png") as prepared:
            graph = self.build(use_image_to_image=True, image_to_image_path="C:/frame.png", denoise=0.4)
        prepared.assert_called_once_with("C:/frame.png", "", "image.png")
        self.assertEqual(graph["13"], {"class_type": "LoadImage", "inputs": {"image": "vrgdg_i2i_1.png"}})
        self.assertEqual(graph["14"]["class_type"], "ImageScale")
        self.assertEqual((graph["14"]["inputs"]["width"], graph["14"]["inputs"]["height"], graph["14"]["inputs"]["crop"]), (1280, 720, "center"))
        self.assertEqual(graph["15"], {"class_type": "VAEEncode", "inputs": {"pixels": ["14", 0], "vae": ["3", 0]}})
        self.assertEqual(graph["10"]["inputs"]["latent_image"], ["15", 0])
        self.assertEqual(graph["10"]["inputs"]["denoise"], 0.4)
        self.assertNotIn("9", graph, "there is no empty latent when the picture is the start")

    def test_denoise_defaults_to_065_and_is_kept_in_range(self):
        with patch.object(workflows, "_prepare_load_image_name", return_value="x.png"):
            self.assertEqual(self.build(use_image_to_image=True, image_to_image_path="a")["10"]["inputs"]["denoise"], 0.65)
            self.assertEqual(self.build(use_image_to_image=True, image_to_image_path="a", denoise=0)["10"]["inputs"]["denoise"], 0.05)
            self.assertEqual(self.build(use_image_to_image=True, image_to_image_path="a", denoise=7)["10"]["inputs"]["denoise"], 1.0)

    def test_image_to_image_without_a_picture_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.build(use_image_to_image=True)
        self.assertIn("no source image was provided", str(caught.exception))

    def test_model_names_become_what_comfyui_lists(self):
        graph = self.build(vae_name="flux\\flux2-vae.safetensors", lora_name="moss_image_sft_8_steps_dmd_lora.safetensors")
        self.assertEqual(graph["3"]["inputs"]["vae_name"], "flux2-vae.safetensors")
        self.assertEqual(graph["4"]["inputs"]["lora_name"], "moss_image_sft_8_steps_dmd_lora.safetensors")


if __name__ == "__main__":
    unittest.main()
