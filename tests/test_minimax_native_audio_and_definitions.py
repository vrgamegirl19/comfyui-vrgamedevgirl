"""2 Pass with H3 built-in audio, and the reference definitions saved with API-written MiniMax prompts."""

import ast
import copy
import importlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
payload_mod = importlib.import_module(f"{pkg_name}.minimax.settings_payload")
sp = importlib.import_module(f"{pkg_name}.minimax.shot_prompt")
assembly = importlib.import_module(f"{pkg_name}.minimax.prompt_assembly")

RUNNER = ROOT / "runner" / "minimax_workflows.py"
TEMPLATE = ROOT / "Workflows" / "UsedForUIDoNotTouch" / "minimax_audio_driven_builder_latent_upscale_2pass_api.json"


def _rewire():
    """The rewire function on its own: the runner module needs ComfyUI to import."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_use_minimax_h3_native_audio")
    namespace = {}
    exec(compile(ast.Module([fn], []), "native_audio", "exec"), namespace)
    return namespace["_use_minimax_h3_native_audio"]


class NativeAudioTwoPassTests(unittest.TestCase):
    def setUp(self):
        self.template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        self.prompt = _rewire()(copy.deepcopy(self.template))

    def by_class(self, class_type):
        return [k for k, v in self.prompt.items() if v["class_type"] == class_type]

    def test_the_audio_file_and_the_audio_lock_are_gone(self):
        self.assertEqual(self.by_class("VHS_LoadAudio"), [])
        self.assertEqual(self.by_class("VRGDG_MiniMaxH3AudioDrive"), [])
        (r2v,) = self.by_class("MiniMaxH3ReferenceToVideo")
        self.assertEqual([n for n in self.prompt[r2v]["inputs"] if n.startswith("ref_audios.")], [])

    def test_pass_one_samples_the_h3_joint_latent_and_its_audio_is_decoded(self):
        (r2v,) = self.by_class("MiniMaxH3ReferenceToVideo")
        (sampler,) = [k for k in self.by_class("SamplerCustomAdvanced") if self.prompt[k]["inputs"]["latent_image"] == [r2v, 1]]
        (decode,) = self.by_class("LTXVAudioVAEDecode")
        self.assertEqual(self.prompt[decode]["inputs"]["samples"], [sampler, 0])
        self.assertEqual(self.prompt[decode]["inputs"]["audio_vae"], self.prompt[r2v]["inputs"]["audio_vae"])
        (combine,) = self.by_class("VHS_VideoCombine")
        self.assertEqual(self.prompt[combine]["inputs"]["audio"], [decode, 0])

    def test_no_link_points_at_a_removed_node(self):
        for key, node in self.prompt.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], self.prompt, f"{key}.{name}")

    def test_a_changed_template_is_refused_with_a_clear_message(self):
        broken = copy.deepcopy(self.template)
        for key in [k for k, v in broken.items() if v["class_type"] == "VRGDG_MiniMaxH3AudioDrive"]:
            broken.pop(key)
        with self.assertRaises(ValueError):
            _rewire()(broken)

    def test_the_runner_takes_audio_mode_and_skips_the_audio_file_for_built_in_audio(self):
        source = RUNNER.read_text(encoding="utf-8")
        start = source.index("def _build_minimax_h3_2pass_api_prompt(payload):")
        body = source[start:source.index("\ndef ", start + 10)]
        self.assertIn('audio_mode == "input_audio"', body)
        self.assertIn("_use_minimax_h3_native_audio(prompt)", body)
        self.assertIn('"audio_mode": audio_mode,', body)

    def test_two_pass_settings_with_built_in_audio_build_a_payload_and_advanced_still_refuses(self):
        base = {"video_mode": "reference_to_video", "audio_mode": "built_in_audio"}
        two = payload_mod.build_minimax_render_payload(payload_mod.normalize_minimax_h3_settings({**base, "render_pass": "two_pass"}))
        self.assertEqual(two["audio_mode"], "built_in_audio")
        with self.assertRaises(ValueError):
            payload_mod.build_minimax_render_payload(payload_mod.normalize_minimax_h3_settings({**base, "render_pass": "three_pass"}))


ITEMS = [
    {"kind": "subject", "label": "Darrel", "description": "A tall man in a green jacket. He wears boots."},
    {"kind": "subject", "label": "The Car", "description": "A sleek silver sports car with a rear wing."},
    {"kind": "location", "label": "Rooftop", "description": "A neon rooftop lounge at night."},
]
PLAN = {"shot_count": 2, "cut_count": 1, "cut_times_seconds": [2.0], "exact_duration_seconds": 4.0, "continuous_shot": False}


class ReferenceFrameTests(unittest.TestCase):
    def test_input_audio_frame_ties_subjects_pictures_and_the_song_together(self):
        frame = sp.reference_frame(ITEMS, PLAN, "cinematic_realism", "input_audio")
        head = frame["head"]
        self.assertIn("<Subject 1> is the Darrel in <Picture 1>; <Picture 1> is the visual authority for character identity", head)
        self.assertIn("<Subject 2> is the Car in <Picture 2>, used as character identity, face, hair, clothing, and body-proportion reference: A sleek silver sports car", head)
        self.assertIn("<Subject 3> is the environment in <Picture 3>, used as environment, location, architecture, layout, and atmosphere reference: A neon rooftop lounge at night.", head)
        self.assertIn(sp.AUDIO_DEFINITION, head)
        self.assertIn("[reference generation + audio reuse] The target video is a cinematic_realism scene featuring <Subject 1> (Darrel) and <Subject 2> (the Car) and <Subject 3> (environment).", head)
        self.assertIn("<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved", head)
        self.assertIn("<Audio 1>: fully_copy", head)
        self.assertIn("<Audio 1> remains the sole complete audience-facing soundtrack", frame["tail"])

    def test_built_in_audio_frame_has_no_audio_reference(self):
        frame = sp.reference_frame(ITEMS, PLAN, "", "built_in_audio", "Room tone and a kettle.")
        self.assertNotIn("<Audio 1>", frame["head"] + frame["tail"])
        self.assertIn("[reference generation] The target video is a photorealistic cinematic scene", frame["head"])
        self.assertIn("MiniMax generates the native audio requested by the scene.", frame["head"])
        self.assertIn("overall_soundscape:\nRoom tone and a kettle.", frame["tail"])

    def test_start_frames_are_pictures_but_not_subjects(self):
        items = [{"kind": "start_frame", "label": "Scene start frame"}, *ITEMS[:1]]
        head = sp.reference_frame(items, PLAN, "style", "input_audio")["head"]
        self.assertIn("<Picture 1> is the first frame of [Shot 1]", head)
        self.assertIn("<Subject 1> is the Darrel in <Picture 2>", head)
        self.assertIn("keyframe completion + reference generation + audio reuse", head)

    def test_wrapping_is_idempotent_and_keeps_the_creative_section_intact(self):
        core = sp.assemble_prompt(["A wide shot of <Subject 1> (Darrel).", "A close shot."], PLAN, "cinematic_realism")
        frame = sp.reference_frame(ITEMS, PLAN, "cinematic_realism", "input_audio")
        wrapped = sp.wrap_reference_prompt(core, frame)
        self.assertTrue(wrapped.startswith("subject_definitions:"))
        self.assertIn(core, wrapped)
        self.assertEqual(sp.wrap_reference_prompt(wrapped, frame), wrapped)
        self.assertEqual(sp.validate_prompt(core, PLAN), core)

    def test_long_descriptions_are_cut_so_the_prompt_stays_inside_the_limit(self):
        items = [{"kind": "subject", "label": "A", "description": "word " * 600}, {"kind": "subject", "label": "B", "description": "word " * 600}]
        frame = sp.reference_frame(items, PLAN, "style", "input_audio")
        self.assertLess(len(frame["head"]) + len(frame["tail"]), 2200)

    def test_the_assemble_route_wraps_reference_to_video_prompts(self):
        session = {
            "minimax_h3_settings": {"audio_mode": "input_audio"},
            "builder_storyboard_defaults": {"video_style": "cinematic_realism", "minimax_h3_cut_frequency": 0},
            "flux_reference_builder": {
                "subjects": [{"id": "d", "name": "Darrel", "description": "A man.", "image": {"path": "d.png"}}],
                "locations": [], "subject_scene_map": {"s1": ["d"]}, "scene_map": {},
            },
            "segments": [{"id": "s1", "start": 0.0, "end": 4.0}],
        }
        result = assembly.assemble_minimax_h3_prompt(session["segments"][0], session, ["<Subject 1> (Darrel) paces the rooftop at night."], "reference_to_video")
        prompt = result["prompt"]
        self.assertTrue(prompt.startswith("subject_definitions:\n<Subject 1> is the Darrel in <Picture 1>"), prompt[:100])
        self.assertIn("detailed_description:\nThe target video is in a cinematic_realism music-video style.", prompt)
        self.assertEqual(result["characters"], len(prompt))


if __name__ == "__main__":
    unittest.main()
