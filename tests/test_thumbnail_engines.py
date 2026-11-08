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
    "loras": ["moss_image_sft_8_steps_dmd_lora.safetensors", "moss\\moss_image_edit_lora.safetensors"],
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


QWEN_INSTALLED = {
    "diffusion_models": [
        "qwen_image_2512_fp8_e4m3fn.safetensors", "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "qwen_image_edit_2511_fp8mixed.safetensors",
        "moss_image_sft_bf16.safetensors",
    ],
    "text_encoders": ["qwen3vl_4b_fp8_scaled.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors"],
    "vae": ["qwen_image_2.1_vae_bf16.safetensors", "qwen_image_vae.safetensors"],
    "loras": [
        "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors", "Qwen-Image-Lightning-4steps-V1.0.safetensors",
        "style\\film_look.safetensors",
    ],
}


class QwenImageEditTests(unittest.TestCase):
    """Qwen Image Edit: the graph of ComfyUI's own image_qwen_image_edit_2511 template, with the newest installed model picked."""

    def setUp(self):
        self.installed = {key: list(values) for key, values in QWEN_INSTALLED.items()}

        def choices(category):
            key = category[0] if isinstance(category, (list, tuple)) else category
            key = {"unet": "diffusion_models", "clip": "text_encoders"}.get(key, key)
            return list(self.installed.get(key, []))

        for module in (models, workflows):
            patcher = patch.object(module, "_folder_choices", choices)
            patcher.start()
            self.addCleanup(patcher.stop)
        copies = patch.object(workflows, "_prepare_load_image_name", side_effect=lambda path, *rest: f"copy_{os.path.basename(path)}")
        copies.start()
        self.addCleanup(copies.stop)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.pictures = []
        for index in range(4):
            path = os.path.join(self.folder.name, f"frame{index}.png")
            Image.new("RGB", (64, 36), (10, 10, 10)).save(path)
            self.pictures.append(path)

    def build(self, count=1, **changes):
        payload = {
            "prompt": 'Add the title text "HEAT". Keep the man unchanged.', "seed": 3,
            "image_ingredients": [{"path": path} for path in self.pictures[:count]], **changes,
        }
        return workflows._build_qwen_image_edit_api_prompt(payload)

    def test_the_newest_installed_model_and_its_lightning_lora_are_used_with_ComfyUIs_own_graph(self):
        built = self.build()
        graph = built["prompt"]
        self.assertEqual(built["used_seed"], 3)
        self.assertEqual({key: node["class_type"] for key, node in graph.items()}, {
            "1": "UNETLoader", "2": "CLIPLoader", "3": "VAELoader", "4": "ModelSamplingAuraFlow", "5": "CFGNorm", "6": "LoraLoaderModelOnly",
            "20": "LoadImage", "30": "FluxKontextImageScale", "7": "TextEncodeQwenImageEditPlus", "8": "TextEncodeQwenImageEditPlus",
            "9": "FluxKontextMultiReferenceLatentMethod", "10": "FluxKontextMultiReferenceLatentMethod", "11": "VAEEncode",
            "12": "KSampler", "13": "VAEDecode", "14": "PreviewImage",
        })
        self.assertEqual(graph["1"]["inputs"], {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}, "2511, not 2509 or 2512")
        self.assertEqual(graph["2"]["inputs"], {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"})
        self.assertEqual(graph["3"]["inputs"], {"vae_name": "qwen_image_vae.safetensors"}, "not the 2.1 VAE")
        self.assertEqual(graph["4"]["inputs"], {"model": ["1", 0], "shift": 3.1})
        self.assertEqual(graph["5"]["inputs"], {"model": ["4", 0], "strength": 1.0})
        self.assertEqual(graph["6"]["inputs"], {"model": ["5", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0})
        sampler = graph["12"]["inputs"]
        self.assertEqual(sampler["model"], ["6", 0])
        self.assertEqual((sampler["positive"], sampler["negative"], sampler["latent_image"]), (["9", 0], ["10", 0], ["11", 0]))
        self.assertEqual((sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]), (4, 1.0, "euler", "simple", 1.0))

    def test_the_picture_is_scaled_and_both_encoders_and_the_start_of_the_sampling_use_it(self):
        graph = self.build()["prompt"]
        self.assertEqual(graph["30"]["inputs"], {"image": ["20", 0]})
        self.assertEqual(graph["11"]["inputs"], {"pixels": ["30", 0], "vae": ["3", 0]})
        for key in ("7", "8"):
            self.assertEqual(graph[key]["inputs"]["clip"], ["2", 0])
            self.assertEqual(graph[key]["inputs"]["vae"], ["3", 0])
            self.assertEqual(graph[key]["inputs"]["image1"], ["30", 0])
        self.assertEqual(graph["7"]["inputs"]["prompt"], 'Add the title text "HEAT". Keep the man unchanged.')
        self.assertEqual(graph["8"]["inputs"]["prompt"], "", "the negative is empty, as in the template")
        self.assertEqual(self.build(negative_prompt="blurry")["prompt"]["8"]["inputs"]["prompt"], "blurry")
        for key, source in (("9", "7"), ("10", "8")):
            self.assertEqual(graph[key]["inputs"], {"conditioning": [source, 0], "reference_latents_method": "index_timestep_zero"})

    def test_without_the_lightning_lora_it_samples_the_whole_way(self):
        for graph in (self.build(turbo=False)["prompt"], self.build(lora_name="[none]")["prompt"]):
            self.assertNotIn("6", graph)
            self.assertEqual(graph["12"]["inputs"]["model"], ["5", 0])
            self.assertEqual((graph["12"]["inputs"]["steps"], graph["12"]["inputs"]["cfg"]), (40, 4.0))
        self.installed["loras"] = ["style\\film_look.safetensors"]
        graph = self.build()["prompt"]
        self.assertNotIn("6", graph, "no Lightning LoRA installed: no turbo")
        self.assertEqual((graph["12"]["inputs"]["steps"], graph["12"]["inputs"]["cfg"]), (40, 4.0), "and then not 4 steps either")

    def test_an_install_with_only_2509_uses_2509_and_the_lightning_lora_of_its_own_version(self):
        self.installed["diffusion_models"] = ["qwen_image_edit_2509_fp8_e4m3fn.safetensors"]
        graph = self.build()["prompt"]
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "qwen_image_edit_2509_fp8_e4m3fn.safetensors")
        self.assertEqual(graph["6"]["inputs"]["lora_name"], "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
        self.installed["loras"] = ["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"]
        self.assertNotIn("6", self.build()["prompt"], "the 2511 LoRA does not fit the 2509 model")

    def test_a_smaller_file_of_the_same_version_comes_first_and_so_does_a_bf16_lora(self):
        self.installed["diffusion_models"] = ["qwen_image_edit_2511_bf16.safetensors", "qwen_image_edit_2511_fp8mixed.safetensors"]
        self.assertEqual(self.build()["prompt"]["1"]["inputs"]["unet_name"], "qwen_image_edit_2511_fp8mixed.safetensors")
        self.installed["loras"] = [
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"]
        self.assertEqual(self.build()["prompt"]["6"]["inputs"]["lora_name"], "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")

    def test_the_request_can_name_its_own_models_and_sampling(self):
        graph = self.build(unet_name="qwen_image_edit_2509_fp8_e4m3fn.safetensors", lora_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                           steps=6, cfg=2)["prompt"]
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "qwen_image_edit_2509_fp8_e4m3fn.safetensors")
        self.assertEqual(graph["6"]["inputs"]["lora_name"], "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
        self.assertEqual((graph["12"]["inputs"]["steps"], graph["12"]["inputs"]["cfg"]), (6, 2.0))
        self.assertEqual(self.build(seed_mode="random")["used_seed"] >= 0, True)

    def test_up_to_three_pictures_the_first_scaled_and_the_others_as_they_are(self):
        graph = self.build(count=4)["prompt"]
        self.assertEqual(graph["20"]["inputs"], {"image": "copy_frame0.png"})
        self.assertEqual(graph["21"]["inputs"], {"image": "copy_frame1.png"})
        self.assertEqual(graph["22"]["inputs"], {"image": "copy_frame2.png"})
        self.assertNotIn("23", graph, "a fourth picture is left out")
        for key in ("7", "8"):
            inputs = graph[key]["inputs"]
            self.assertEqual((inputs["image1"], inputs["image2"], inputs["image3"]), (["30", 0], ["21", 0], ["22", 0]))
        self.assertNotIn("image2", self.build()["prompt"]["7"]["inputs"])

    def test_a_missing_picture_prompt_or_model_is_refused_with_what_to_do(self):
        with self.assertRaisesRegex(ValueError, "at least one reference image"):
            self.build(count=0)
        with self.assertRaisesRegex(ValueError, "Prompt text is empty"):
            self.build(prompt=" ")
        self.installed["diffusion_models"] = ["qwen_image_2512_fp8_e4m3fn.safetensors"]
        with self.assertRaisesRegex(ValueError, "No Qwen Image Edit diffusion model was found.*Comfy-Org/Qwen-Image-Edit_ComfyUI"):
            self.build()
        self.installed["diffusion_models"] = QWEN_INSTALLED["diffusion_models"]
        for key, label in (("clip_name", "Qwen Image Edit text encoder"), ("vae_name", "Qwen Image Edit VAE"),
                           ("unet_name", "Qwen Image Edit diffusion model"), ("lora_name", "Qwen Image Edit LoRA")):
            with self.subTest(missing=key), self.assertRaises(ValueError) as caught:
                self.build(**{key: "not_there.safetensors"})
            self.assertIn(label, str(caught.exception))


class MossImageEditTests(unittest.TestCase):
    """The instruction edit: the edit LoRA and ComfyUI's Text Encode MOSS-Image Edit, as in its own image_moss_image_edit template."""

    def setUp(self):
        patcher = patch.object(models, "_folder_choices", fake_folder_choices)
        patcher.start()
        self.addCleanup(patcher.stop)
        copies = patch.object(workflows, "_prepare_load_image_name", side_effect=lambda path, *rest: f"copy_{os.path.basename(path)}")
        copies.start()
        self.addCleanup(copies.stop)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.pictures = []
        for index in range(5):
            path = os.path.join(self.folder.name, f"frame{index}.png")
            Image.new("RGB", (64, 36), (10, 10, 10)).save(path)
            self.pictures.append(path)

    def build(self, count=1, **changes):
        payload = {
            "use_edit": True, "prompt": 'Add the title text "HEAT". Keep the man unchanged.', "seed": 3,
            "image_ingredients": [{"path": path} for path in self.pictures[:count]], **changes,
        }
        return workflows._build_moss_image_api_prompt(payload)

    def test_the_graph_is_the_one_comfyui_ships_for_an_edit(self):
        built = self.build()
        graph = built["prompt"]
        self.assertEqual(built["used_seed"], 3)
        self.assertEqual({key: node["class_type"] for key, node in graph.items()}, {
            "1": "UNETLoader", "2": "CLIPLoader", "3": "VAELoader", "4": "LoraLoaderModelOnly", "5": "SkipLayerGuidanceDiTSimple",
            "20": "LoadImage", "30": "ImageScaleToTotalPixels", "40": "GetImageSize", "9": "EmptyFlux2LatentImage",
            "7": "TextEncodeMOSSImageEdit", "8": "TextEncodeMOSSImageEdit", "10": "KSampler", "11": "VAEDecode", "12": "PreviewImage",
        })
        self.assertEqual(graph["2"]["inputs"]["type"], "moss_image")
        # The edit LoRA runs on the model, then skip layer guidance, then the sampler.
        self.assertEqual(graph["4"]["inputs"], {"model": ["1", 0], "lora_name": "moss\\moss_image_edit_lora.safetensors", "strength_model": 1.0})
        self.assertEqual(graph["5"]["inputs"], {"model": ["4", 0], "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 1.0})
        sampler = graph["10"]["inputs"]
        self.assertEqual(sampler["model"], ["5", 0])
        self.assertEqual(
            (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]), (30, 4.0, "euler", "normal", 1.0))
        self.assertEqual((sampler["positive"], sampler["negative"], sampler["latent_image"]), (["7", 0], ["8", 0], ["9", 0]))

    def test_the_picture_is_scaled_once_and_the_result_takes_its_size(self):
        graph = self.build()["prompt"]
        self.assertEqual(graph["20"]["inputs"], {"image": "copy_frame0.png"})
        self.assertEqual(
            graph["30"]["inputs"], {"image": ["20", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16})
        self.assertEqual(graph["40"]["inputs"], {"image": ["30", 0]})
        self.assertEqual(graph["9"]["inputs"], {"width": ["40", 0], "height": ["40", 1], "batch_size": 1})
        self.assertEqual(self.build(megapixels=9)["prompt"]["30"]["inputs"]["megapixels"], 4.0, "kept in range")

    def test_both_encoders_see_the_picture_and_the_vae_and_the_negative_is_the_recommended_one(self):
        graph = self.build()["prompt"]
        for key in ("7", "8"):
            inputs = graph[key]["inputs"]
            self.assertEqual(inputs["clip"], ["2", 0])
            self.assertEqual(inputs["edit_format"], "Moss-Image-Edit")
            self.assertEqual(inputs["images.image_0"], ["30", 0])
            self.assertEqual(inputs["vae"], ["3", 0], "without the VAE the picture only reaches the text encoder")
            self.assertEqual(inputs["image_strength"], 1.0)
        self.assertEqual(graph["7"]["inputs"]["prompt"], 'Add the title text "HEAT". Keep the man unchanged.')
        self.assertEqual(graph["8"]["inputs"]["prompt"], "nsfw, worst quality, bad anatomy, watermark, jpeg artifact")
        self.assertEqual(self.build(negative_prompt="blurry")["prompt"]["8"]["inputs"]["prompt"], "blurry")

    def test_up_to_four_pictures_each_scaled_and_named_image_n(self):
        graph = self.build(count=5)["prompt"]
        for index in range(4):
            self.assertEqual(graph[str(20 + index)]["inputs"], {"image": f"copy_frame{index}.png"})
            self.assertEqual(graph[str(30 + index)]["inputs"]["image"], [str(20 + index), 0])
            self.assertEqual(graph["7"]["inputs"][f"images.image_{index}"], [str(30 + index), 0])
        self.assertNotIn("24", graph)
        self.assertNotIn("images.image_4", graph["7"]["inputs"])
        self.assertEqual(graph["40"]["inputs"]["image"], ["30", 0], "the size is the first picture's")

    def test_the_turbo_settings_of_text_to_image_are_not_used(self):
        graph = self.build(lora_name="moss_image_sft_8_steps_dmd_lora.safetensors", denoise=0.3, use_image_to_image=True)["prompt"]
        self.assertEqual([node["inputs"].get("lora_name") for node in graph.values() if node["class_type"] == "LoraLoaderModelOnly"],
                         ["moss\\moss_image_edit_lora.safetensors"])
        self.assertEqual(graph["10"]["inputs"]["denoise"], 1.0)
        self.assertEqual(graph["10"]["inputs"]["steps"], 30)
        self.assertNotIn("ImageScale", {node["class_type"] for node in graph.values()})
        self.assertNotIn("VAEEncode", {node["class_type"] for node in graph.values()})
        self.assertEqual(self.build(steps=12, cfg=3)["prompt"]["10"]["inputs"]["steps"], 12)

    def test_skip_layer_guidance_and_the_format_can_be_changed(self):
        graph = self.build(skip_layer_guidance=False, edit_format="ai-toolkit")["prompt"]
        self.assertNotIn("5", graph)
        self.assertEqual(graph["10"]["inputs"]["model"], ["4", 0])
        self.assertEqual(graph["7"]["inputs"]["edit_format"], "ai-toolkit")
        self.assertEqual(self.build(edit_format="nonsense")["prompt"]["7"]["inputs"]["edit_format"], "Moss-Image-Edit")

    def test_a_missing_picture_prompt_or_lora_is_refused_with_what_to_do(self):
        with self.assertRaisesRegex(ValueError, "at least one reference image"):
            self.build(count=0)
        with self.assertRaisesRegex(ValueError, "Prompt text is empty"):
            self.build(prompt="  ")
        with self.assertRaises(ValueError) as caught:
            self.build(edit_lora_name="not_there.safetensors")
        self.assertIn("MOSS-Image Edit LoRA", str(caught.exception))

    def test_a_random_seed_is_used_when_asked_for(self):
        seeds = {self.build(seed_mode="random")["used_seed"] for _ in range(3)}
        self.assertGreater(len(seeds), 1)


if __name__ == "__main__":
    unittest.main()
