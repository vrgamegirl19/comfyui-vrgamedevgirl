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

    def test_every_mode_and_supported_pass_accepts_built_in_audio(self):
        for mode in payload_mod.SETTINGS_ENUMS["video_mode"]:
            for render_pass in ("single", "two_pass", "three_pass"):
                with self.subTest(mode=mode, render_pass=render_pass):
                    settings = payload_mod.normalize_minimax_h3_settings({
                        "video_mode": mode, "audio_mode": "built_in_audio", "render_pass": render_pass,
                        "i2v_pass_settings_version": 1, "t2v_pass_settings_version": 1,
                    })
                    self.assertEqual(payload_mod.build_minimax_render_payload(settings)["audio_mode"], "built_in_audio")

    def test_legacy_three_pass_reuses_generated_audio_in_all_outputs(self):
        template = TEMPLATE.with_name("minimax_ref2video_3pass_audio_driven_api.json")
        graph = _rewire()(json.loads(template.read_text(encoding="utf-8")))
        decode = next(key for key, node in graph.items() if node["class_type"] == "LTXVAudioVAEDecode")
        self.assertEqual(graph[decode]["inputs"]["samples"], ["247", 0])
        self.assertEqual(graph["347"]["inputs"]["samples"], ["350", 0])
        for node in graph.values():
            if node["class_type"] == "VHS_VideoCombine":
                self.assertEqual(node["inputs"]["audio"], [decode, 0])

    def test_i2v_and_image_reference_branches_generate_audio_without_input(self):
        tree = ast.parse((ROOT / "runner/minimax_inputs.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_patch_minimax_h3_image_to_video_node")
        namespace = {}
        exec(compile(ast.Module([fn], []), "image_conditioning", "exec"), namespace)
        for references in (False, True):
            with self.subTest(references=references):
                graph = copy.deepcopy(self.template)
                namespace[fn.name](graph, ["first.png", "last.png", "ref.png"] if references else ["first.png", "last.png"],
                                   include_references=references, has_last_frame=True)
                graph["214"] = copy.deepcopy(graph["136"])
                graph["193"]["inputs"]["conditioning"] = ["214", 0]
                graph = _rewire()(graph)
                self.assertEqual(graph["125"]["inputs"]["latent_image"], ["136", 1])
                self.assertIn("last_frame", graph["214"]["inputs"])
                for key, node in graph.items():
                    for name, value in node["inputs"].items():
                        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                            self.assertIn(value[0], graph, f"{key}.{name}")

    def test_advanced_native_audio_survives_draft_and_finalize(self):
        import importlib.util
        import tempfile
        from test_builder_minimax_advanced_two_pass import BuilderMiniMaxAdvancedTwoPassTests

        fixture = BuilderMiniMaxAdvancedTwoPassTests()
        namespace = fixture._advanced_prompt_namespace(object())
        base = namespace["_build_minimax_h3_2pass_api_prompt"]
        def native_base(payload):
            result = base(payload)
            result["prompt"] = _rewire()(result["prompt"])
            return result
        namespace["_build_minimax_h3_2pass_api_prompt"] = native_base
        result = namespace["_build_minimax_h3_advanced_2pass_api_prompt"]({"audio_mode": "built_in_audio"})
        graph = result["prompt"]
        decode = next(key for key, node in graph.items() if node["class_type"] == "LTXVAudioVAEDecode")
        self.assertEqual(graph[decode]["inputs"]["samples"], ["125", 0])
        for output in ("142", "9308"):
            self.assertEqual(graph[output]["inputs"]["audio"], [decode, 0])
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], graph)

        spec = importlib.util.spec_from_file_location("native_drafts", ROOT / "runner/minimax_drafts.py")
        drafts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drafts)
        with tempfile.TemporaryDirectory() as folder:
            payload = {"project_folder": folder, "scene_number": 1, "timeline_start_seconds": 0, "timeline_end_seconds": 5}
            draft = drafts._build_minimax_draft(result, payload)
            self.assertIn(decode, draft["prompt"])
            Path(draft["prompt"]["draft_save"]["inputs"]["checkpoint_path"]).touch()
            final = drafts._build_minimax_finalize({**payload, "draft_id": draft["draft_id"]})
            self.assertEqual(final["prompt"]["125"]["class_type"], "VRGDG_MiniMaxH3LoadDraft")
            self.assertEqual(final["prompt"][decode]["inputs"]["samples"], ["125", 0])

    def test_native_compilers_need_no_audio_file_or_audio_preparation(self):
        """Compile real templates without importing ComfyUI, torch, or model weights."""
        import os
        import random
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import Mock

        folder = TEMPLATE.parent
        namespace = {"copy": copy, "json": json, "os": os, "random": random,
                     "_NONE_LORA": "[none]", "_MINIMAX_H3_ASPECT_RATIOS": {"16:9 (Widescreen)"}}
        for path, names in ((RUNNER, {"_build_minimax_h3_2pass_api_prompt", "_build_minimax_h3_3pass_api_prompt", "_use_minimax_h3_native_audio"}),
                            (ROOT / "runner/paths.py", {"_first_payload_value", "_bool_payload", "_int_payload", "_float_payload"}),
                            (ROOT / "runner/minimax_inputs.py", {"_patch_minimax_h3_image_to_video_node"})):
            nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name in names]
            exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)
        def load(path):
            return str(path), json.loads(Path(path).read_text(encoding="utf-8"))
        namespace.update({
            "_load_api_template": load,
            "_minimax_h3_2pass_api_template_path": lambda mode: folder / ("minimax_i2v_audio_driven_builder_latent_upscale_2pass_api.json" if mode == "image_to_video" else TEMPLATE.name),
            "_minimax_h3_3pass_api_template_path": lambda: folder / "minimax_ref2video_3pass_audio_driven_api.json",
            "_set_api_input": lambda p, n, k, v: p[n]["inputs"].update({k: v}),
            "_require_model_choice": lambda *args: None,
            "_clean_lora_name": lambda name: name,
            "_minimax_h3_image_paths": lambda p: p.get("image_paths", []),
            "_minimax_h3_video_references": lambda p: [],
            "_minimax_h3_output_location": lambda *args: ("output", "scene"),
            "_minimax_h3_effective_warmup_frames": lambda p: 5,
            "_minimax_h3_cooldown_frames": lambda p: 0,
            "_minimax_h3_latent_continuation_mode": lambda p: "off",
            "_minimax_h3_is_latent_mode": lambda mode: False,
            "_patch_minimax_h3_latent_continuation": lambda *args: None,
            "_patch_minimax_h3_save_latent": lambda *args: None,
            "_patch_minimax_h3_optional_model_paths": lambda *args: [],
            "patch_i2v_keyframe_timing": lambda *args: None,
            "calculate_minimax_h3_timing": Mock(return_value=SimpleNamespace(
                workflow_duration_input_seconds=5, final_trim_start_seconds=0,
                final_trim_duration_seconds=5, final_frame_count=120, to_dict=lambda: {})),
            "_probe_media_duration_seconds": Mock(side_effect=AssertionError("must not probe input audio")),
            "_trim_minimax_h3_audio_context": Mock(side_effect=AssertionError("must not prepare input audio")),
        })
        with tempfile.TemporaryDirectory() as project:
            for mode in payload_mod.SETTINGS_ENUMS["video_mode"]:
                for compiler in ("_build_minimax_h3_2pass_api_prompt", "_build_minimax_h3_3pass_api_prompt"):
                    with self.subTest(mode=mode, compiler=compiler):
                        result = namespace[compiler]({"project_folder": project, "prompt": "A cinematic scene with speech.",
                            "audio_mode": "built_in_audio", "video_mode": mode, "scene_duration_seconds": 5,
                            "image_paths": ["first.png", "ref.png"] if mode.startswith("image_") else []})
                        self.assertEqual(result["audio_mode"], "built_in_audio")
                        self.assertIsNone(result["prepared_audio"])
                        self.assertTrue(namespace["calculate_minimax_h3_timing"].call_args.kwargs["pad_warmup"])
                        self.assertFalse(any(n["class_type"] == "VHS_LoadAudio" for n in result["prompt"].values()))
                        for node in result["prompt"].values():
                            for value in node["inputs"].values():
                                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                                    self.assertIn(value[0], result["prompt"])
            # Supplied audio continues to use the external-audio templates.
            source = Path(project) / "source.wav"
            source.touch()
            namespace["_probe_media_duration_seconds"].side_effect = None
            namespace["_probe_media_duration_seconds"].return_value = 5
            namespace["_trim_minimax_h3_audio_context"].side_effect = None
            namespace["_trim_minimax_h3_audio_context"].return_value = {"audio_path": str(source)}
            for compiler in ("_build_minimax_h3_2pass_api_prompt", "_build_minimax_h3_3pass_api_prompt"):
                result = namespace[compiler]({"project_folder": project, "prompt": "A cinematic scene.",
                    "audio_mode": "input_audio", "audio_path": str(source), "scene_duration_seconds": 5})
                self.assertEqual(result["audio_mode"], "input_audio")
                self.assertTrue(any(n["class_type"] == "VRGDG_MiniMaxH3AudioDrive" for n in result["prompt"].values()))


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

    def test_image_reference_compact_character_assignments_follow_connected_order(self):
        segment = {"id": "s1", "start": 0.0, "end": 4.0, "approved_image_path": "start.png", "minimax_h3_use_scene_image_as_start_frame": True}
        subjects = [{"id": str(i), "name": f"Character {i}", "description": "A character.", "image": {"path": f"char{i}.png"}} for i in range(1, 5)]
        session = {"segments": [segment], "builder_storyboard_defaults": {"video_style": "1990s_grunge"},
                   "minimax_h3_settings": {"audio_mode": "input_audio"},
                   "flux_reference_builder": {"subjects": subjects, "locations": [], "subject_scene_map": {"s1": ["1", "2", "3", "4"]}, "scene_map": {}}}
        descriptions = ["<Subject 1> turns toward <Subject 3> beside the stage."]
        compact = assembly.assemble_minimax_h3_prompt(segment, session, descriptions, "image_reference_to_video")["prompt"]
        self.assertIn("Use <Picture 2> for <Subject 1>'s character identity", compact)
        self.assertIn("Use <Picture 4> for <Subject 3>'s character identity", compact)
        self.assertIn("[Shot 1] The shot begins exactly from <Picture 1>.", compact)
        self.assertNotIn("<Subject 2>", compact)
        self.assertNotIn("<Subject 4>", compact)
        self.assertNotIn("subject_definitions:", compact)
        segment["first_last_frame_end_image_path"] = "end.png"
        segment["minimax_h3_start_frame_character_influence"] = "face_hair_only"
        compact = assembly.assemble_minimax_h3_prompt(segment, session, descriptions, "image_reference_to_video")["prompt"]
        self.assertIn("Use <Picture 3> for <Subject 1>'s face identity and hair.", compact)
        self.assertIn("Use <Picture 5> for <Subject 3>'s face identity and hair.", compact)
        self.assertIn("<Picture 2> as the last frame", compact)
        self.assertNotIn("clothing", compact)
        session["use_structured_outputs"] = True
        structured = assembly.assemble_minimax_h3_prompt(segment, session, descriptions, "image_reference_to_video")["prompt"]
        self.assertIn("subject_definitions:", structured)
        self.assertIn("<Subject 1> is the Character 1 in <Picture 3>", structured)

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
            "use_structured_outputs": True,
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
        session.pop("use_structured_outputs")
        compact = assembly.assemble_minimax_h3_prompt(session["segments"][0], session, ["<Subject 1> (Darrel) paces the rooftop at night."], "reference_to_video")["prompt"]
        self.assertTrue(compact.startswith("detailed_description:"))
        self.assertIn("[Shot 1] <Subject 1> paces", compact)
        self.assertNotIn("<Subject 1> (", compact)
        self.assertNotIn("<Subject 1> is", compact)
        for header in ("subject_definitions:", "summary:", "retention_analysis:", "overall_soundscape:", "non_diegetic_music:"):
            self.assertNotIn(header, compact)
        self.assertNotIn("first frame", compact)


if __name__ == "__main__":
    unittest.main()
