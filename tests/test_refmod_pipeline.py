import copy
import importlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
TEMPLATES = ROOT / "Workflows" / "UsedForUIDoNotTouch"


def _load():
    for name, path in (("vrgdg_pipe_test", ROOT), ("vrgdg_pipe_test.runner", ROOT / "runner"), ("vrgdg_pipe_test.minimax", ROOT / "minimax")):
        package = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None, is_package=True))
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return importlib.import_module("vrgdg_pipe_test.runner.minimax_refmod")


refmod = _load()

ENTRIES = {
    "identity/brad": {"name": "identity/brad", "kind": "video", "tokens": 5632},
    "identity/darrel": {"name": "identity/darrel", "kind": "video", "tokens": 2394},
    "background/meadow": {"name": "background/meadow", "kind": "image", "tokens": 600},
}


def fake_find(name):
    return ENTRIES.get(name)


def template(name):
    return copy.deepcopy(json.loads((TEMPLATES / name).read_text(encoding="utf-8")))


def payload(**extra):
    base = {
        "pipeline": "refmod", "prompt": "Brad <Video 1> and Darrel <Video 2> in <Picture 1> <Audio 1>",
        "refmod_references": [
            {"name": "identity/brad", "strength": 1.0},
            {"name": "identity/darrel", "strength": 0.8},
            {"name": "background/meadow", "strength": 1.0},
        ],
    }
    base.update(extra)
    return base


def classes(prompt):
    return {node["class_type"] for node in prompt.values()}


class RefModPipelineTests(unittest.TestCase):
    def apply(self, template_name, **extra):
        prompt = template(template_name)
        with mock.patch.object(refmod, "find_refmod", fake_find):
            summary = refmod.apply_refmod_pipeline(prompt, payload(**extra))
        return prompt, summary

    def test_requested_only_for_the_refmod_pipeline(self):
        self.assertTrue(refmod.refmod_requested({"pipeline": "refmod"}))
        self.assertFalse(refmod.refmod_requested({"pipeline": "standard"}))
        self.assertFalse(refmod.refmod_requested({}))

    def test_single_pass_graph_is_rewired(self):
        prompt, summary = self.apply("minimax_audio_driven_builder_api.json")
        self.assertNotIn("VRGDG_MiniMaxH3ReferenceMediaFromPaths", classes(prompt))
        self.assertIn("MiniMaxH3RefModTextEncode", classes(prompt))
        self.assertIn("MiniMaxH3RefModsLoader", classes(prompt))
        self.assertIn("MiniMaxH3RefModAudioExtract", classes(prompt))
        self.assertIn("VRGDG_RefModCombine", classes(prompt))
        encode = next(k for k, n in prompt.items() if n["class_type"] == "MiniMaxH3RefModTextEncode")
        self.assertEqual(prompt["126"]["inputs"]["conditioning"], [encode, 0])
        self.assertEqual(summary["guiders_rewired"], 1)
        self.assertEqual(summary["labels"], ["<Video 1>", "<Video 2>", "<Picture 1>", "<Audio 1>"])
        self.assertEqual(summary["labels_missing_from_prompt"], [])
        self.assertEqual(summary["tokens"], 5632 + 2394 + 600)

    def test_reference_node_keeps_only_the_latent_job(self):
        prompt, _ = self.apply("minimax_audio_driven_builder_api.json")
        inputs = prompt["136"]["inputs"]
        self.assertFalse([name for name in inputs if name.startswith(("ref_images.", "ref_videos.", "ref_audios."))])
        self.assertEqual(inputs["prompt"], " ")

    def test_two_pass_graph_rewires_both_guiders(self):
        prompt, summary = self.apply("minimax_audio_driven_builder_latent_upscale_2pass_api.json")
        encode = next(k for k, n in prompt.items() if n["class_type"] == "MiniMaxH3RefModTextEncode")
        self.assertEqual(prompt["126"]["inputs"]["conditioning"], [encode, 0])
        self.assertEqual(prompt["193"]["inputs"]["conditioning"], [encode, 0])
        self.assertEqual(summary["guiders_rewired"], 2)

    def test_built_in_audio_has_no_scene_audio_mod(self):
        prompt, summary = self.apply("minimax_built_in_audio_builder_api.json", audio_mode="built_in_audio",
                                     prompt="Brad <Video 1> Darrel <Video 2> <Picture 1>")
        self.assertNotIn("MiniMaxH3RefModAudioExtract", classes(prompt))
        self.assertFalse(summary["scene_audio_mod"])
        self.assertEqual(summary["labels"], ["<Video 1>", "<Video 2>", "<Picture 1>"])

    def test_no_links_are_left_dangling(self):
        for name in ("minimax_audio_driven_builder_api.json", "minimax_audio_driven_builder_latent_upscale_2pass_api.json"):
            prompt, _ = self.apply(name)
            for key, node in prompt.items():
                for value in node["inputs"].values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                        self.assertIn(value[0], prompt, f"{name}: node {key} links to missing {value[0]}")

    def test_more_than_eight_mods_use_extra_loaders(self):
        many = [{"name": "identity/brad", "strength": 1.0}] * 10
        prompt = template("minimax_audio_driven_builder_api.json")
        with mock.patch.object(refmod, "find_refmod", fake_find):
            refmod.apply_refmod_pipeline(prompt, payload(refmod_references=many))
        loaders = [n for n in prompt.values() if n["class_type"] == "MiniMaxH3RefModsLoader"]
        self.assertEqual(len(loaders), 2)
        self.assertEqual(loaders[1]["inputs"]["mod_3"], "(none)")

    def test_strengths_and_names_land_in_the_loader(self):
        prompt, _ = self.apply("minimax_audio_driven_builder_api.json")
        loader = next(n for n in prompt.values() if n["class_type"] == "MiniMaxH3RefModsLoader")
        self.assertEqual(loader["inputs"]["mod_2"], "identity/darrel")
        self.assertEqual(loader["inputs"]["strength_2"], 0.8)
        self.assertEqual(loader["inputs"]["mod_4"], "(none)")

    def test_missing_labels_are_reported(self):
        _, summary = self.apply("minimax_audio_driven_builder_api.json", prompt="Brad <Video 1> only")
        self.assertIn("<Picture 1>", summary["labels_missing_from_prompt"])

    def test_errors_are_clear(self):
        with mock.patch.object(refmod, "find_refmod", fake_find):
            with self.assertRaises(ValueError):
                refmod.apply_refmod_pipeline(template("minimax_audio_driven_builder_api.json"), payload(refmod_references=[]))
            with self.assertRaises(FileNotFoundError):
                refmod.apply_refmod_pipeline(template("minimax_audio_driven_builder_api.json"),
                                             payload(refmod_references=[{"name": "identity/nobody", "strength": 1}]))
            with self.assertRaises(ValueError):
                refmod.apply_refmod_pipeline(template("minimax_audio_driven_builder_api.json"),
                                             payload(refmod_references=[{"name": "identity/brad", "strength": 2}]))
            with self.assertRaises(ValueError):
                refmod.apply_refmod_pipeline(template("minimax_audio_driven_builder_api.json"),
                                             payload(refmod_references=[{"name": "identity/brad"}] * 25))


class RefModSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = importlib.import_module("vrgdg_pipe_test.minimax.settings_payload")

    def test_refmod_forces_one_mode_and_no_advanced_pass(self):
        result = self.settings.normalize_minimax_h3_settings(
            {"pipeline": "refmod", "video_mode": "text_to_video", "render_pass": "three_pass",
             "continuity_mode": "spatial_reference"})
        self.assertEqual(result["pipeline"], "refmod")
        self.assertEqual(result["video_mode"], "reference_to_video")
        self.assertEqual(result["render_pass"], "two_pass")
        self.assertEqual(result["continuity_mode"], "off")

    def test_standard_is_untouched(self):
        result = self.settings.normalize_minimax_h3_settings({"video_mode": "text_to_video", "render_pass": "single"})
        self.assertEqual((result["pipeline"], result["video_mode"]), ("standard", "text_to_video"))

    def test_latent_continuation_is_kept(self):
        result = self.settings.normalize_minimax_h3_settings({"pipeline": "refmod", "continuity_mode": "latent_continuation_masked"})
        self.assertEqual(result["continuity_mode"], "latent_continuation_masked")
        # the retired standard mode continues masked
        result = self.settings.normalize_minimax_h3_settings({"pipeline": "refmod", "continuity_mode": "latent_continuation"})
        self.assertEqual(result["continuity_mode"], "latent_continuation_masked")

    def test_scene_settings_follow_the_project_pipeline(self):
        session = {"minimax_h3_settings": {"pipeline": "refmod"}}
        segment = {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": {"pipeline": "standard", "steps": 12}}
        result = self.settings.minimax_h3_settings_for_scene(session, segment)
        self.assertEqual(result["pipeline"], "refmod")
        self.assertEqual(result["steps"], 12)

    def test_payload_carries_the_pipeline(self):
        result = self.settings.build_minimax_render_payload(
            self.settings.normalize_minimax_h3_settings({"pipeline": "refmod"}))
        self.assertEqual(result["pipeline"], "refmod")
        self.assertEqual(result["video_mode"], "reference_to_video")


if __name__ == "__main__":
    unittest.main()
