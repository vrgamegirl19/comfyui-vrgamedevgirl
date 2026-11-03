"""A MiniMax H3 scene can be made again from its saved latent, at a chosen quality, without sampling.

The compiled scene graph is cut down to: load the whole saved latent, decode it, resize, encode at the chosen CRF.
Nothing that samples, saves a latent or encodes text may stay in it: a leftover save node is an output node, and
ComfyUI would run it and overwrite the latent that is being decoded.
"""

import copy
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
redecode = importlib.import_module(f"{pkg_name}.runner.minimax_redecode")
patches = importlib.import_module(f"{pkg_name}.runner.minimax_patches")

TEMPLATE = ROOT / "Workflows/UsedForUIDoNotTouch/minimax_audio_driven_builder_latent_upscale_2pass_api.json"
FRAMES = 150


class RedecodeCrfTests(unittest.TestCase):
    def test_default_and_range(self):
        self.assertEqual(redecode.redecode_crf(None), redecode.DEFAULT_REDECODE_CRF)
        self.assertEqual(redecode.redecode_crf(""), redecode.DEFAULT_REDECODE_CRF)
        self.assertEqual(redecode.redecode_crf("10"), 10)
        self.assertEqual(redecode.redecode_crf(0), 0)
        self.assertEqual(redecode.redecode_crf(redecode.MAX_REDECODE_CRF), redecode.MAX_REDECODE_CRF)

    def test_bad_values_are_rejected(self):
        for bad in (-1, redecode.MAX_REDECODE_CRF + 1, "fine", [12]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                redecode.redecode_crf(bad)

    def test_the_default_is_finer_than_the_render(self):
        self.assertLess(redecode.DEFAULT_REDECODE_CRF, 19)


class RedecodeGraphTests(unittest.TestCase):
    def setUp(self):
        self.project = tempfile.mkdtemp(prefix="vrgdg_redecode_")
        self.addCleanup(shutil.rmtree, self.project, ignore_errors=True)
        os.makedirs(os.path.join(self.project, "latents"))
        Path(self.project, "latents", "scene_001.latent").write_bytes(b"latent")
        Path(self.project, "latents", "scene_001.latent.json").write_text(
            json.dumps({"frame_count": str(FRAMES), "token_count": "45"}), encoding="utf-8")
        self.prompt = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        # A normal render saves the latent right before the decode: the cut has to get rid of that.
        patches._patch_minimax_h3_save_latent(self.prompt, {"project_folder": self.project, "scene_number": 1})
        self.graph_res = {"prompt": self.prompt, "timing": {"h3_frame_count": FRAMES}, "audio_mode": "input_audio"}

    def build(self, crf=12, **changes):
        return redecode.build_redecode_graph({**self.graph_res, **changes}, self.project, 1, crf)

    def classes(self, result):
        return {node["class_type"] for node in result["prompt"].values()}

    def test_the_graph_is_load_decode_resize_encode(self):
        result = self.build()
        prompt = result["prompt"]
        self.assertEqual(self.classes(result), {
            redecode.REDECODE_LOADER_CLASS, "VAEDecode", "VAELoader", "ImageResizeKJv2", "VHS_VideoCombine",
            "VHS_LoadAudio", "PrimitiveInt",
        })
        decode_id = next(key for key, node in prompt.items() if node["class_type"] == "VAEDecode")
        loader_id = prompt[decode_id]["inputs"]["samples"][0]
        self.assertEqual(prompt[loader_id]["class_type"], redecode.REDECODE_LOADER_CLASS)
        self.assertEqual(prompt[loader_id]["inputs"], {"project_folder": self.project, "scene_number": 1})
        self.assertEqual(result["final_video_node_id"], "142")

    def test_nothing_that_samples_saves_or_encodes_text_stays(self):
        classes = self.classes(self.build())
        for gone in ("SamplerCustomAdvanced", "VRGDG_MiniMaxH3SaveLatent", "CLIPLoader", "DiffusionModelLoaderKJ",
                     "MiniMaxH3ReferenceToVideo", "VRGDG_MiniMaxH3AudioDrive", "VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent"):
            self.assertNotIn(gone, classes)

    def test_the_encoder_gets_the_chosen_crf_and_the_source_audio(self):
        for crf in (0, 12, 30):
            result = self.build(crf)
            video = result["prompt"]["142"]["inputs"]
            self.assertEqual(video["crf"], crf)
            self.assertEqual(result["redecode"]["crf"], crf)
            self.assertEqual(result["prompt"][video["audio"][0]]["class_type"], "VHS_LoadAudio")

    def test_every_link_points_at_a_node_that_is_still_there(self):
        prompt = self.build()["prompt"]
        for key, node in prompt.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], prompt, f"{key}.{name} -> {value[0]}")

    def test_the_compiled_graph_is_not_changed(self):
        before = copy.deepcopy(self.graph_res["prompt"])
        self.build()
        self.assertEqual(self.graph_res["prompt"], before)

    def test_the_result_keeps_what_the_render_steps_after_it_need(self):
        result = self.build(post_render_trim={"start": 0.0, "duration": 6.0, "frames": 144}, output_folder="x")
        self.assertEqual(result["post_render_trim"]["frames"], 144)
        self.assertEqual(result["output_folder"], "x")
        self.assertEqual(result["timing"]["h3_frame_count"], FRAMES)

    def test_a_scene_without_a_saved_latent_is_refused(self):
        os.remove(os.path.join(self.project, "latents", "scene_001.latent"))
        with self.assertRaisesRegex(ValueError, "no saved latent"):
            self.build()

    def test_a_latent_that_no_longer_matches_the_scene_is_refused(self):
        with self.assertRaisesRegex(ValueError, "150 frames.*call for 175"):
            self.build(timing={"h3_frame_count": 175})

    def test_built_in_audio_is_refused(self):
        with self.assertRaisesRegex(ValueError, "built-in audio"):
            self.build(audio_mode="built_in_audio")

    def test_a_graph_without_a_decode_or_output_is_refused(self):
        no_output = {key: node for key, node in self.prompt.items() if node["class_type"] != "VHS_VideoCombine"}
        with self.assertRaisesRegex(ValueError, "video output"):
            redecode.build_redecode_graph({**self.graph_res, "prompt": no_output}, self.project, 1, 12)


if __name__ == "__main__":
    unittest.main()
