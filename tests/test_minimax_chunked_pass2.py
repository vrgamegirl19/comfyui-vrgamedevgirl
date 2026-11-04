"""Chunked second pass: the 8 s rule, the window plan, the prompt rewrite, the settings and the compiled graph."""

import copy
import importlib
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

plan_mod = importlib.import_module(f"{ROOT.name}.minimax.chunk_plan")
payload_mod = importlib.import_module(f"{ROOT.name}.minimax.settings_payload")
chunked = importlib.import_module(f"{ROOT.name}.runner.minimax_chunked_pass2")
workflows = importlib.import_module(f"{ROOT.name}.runner.minimax_workflows")
patches = importlib.import_module(f"{ROOT.name}.runner.minimax_patches")
models = importlib.import_module(f"{ROOT.name}.runner.models")

SHOT_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is Gabriel from the <Picture 1> character reference.\n\n"
    "<Audio 1> is the complete synchronized song and vocal track.\n\n"
    "detailed_description:\n"
    "The target video is a gritty, photorealistic cinematic music video.\n\n"
    "[Shot 1] A continuous medium close-up of <Subject 1>'s face: <Subject 1> sings in sync with the vocals in "
    "<Audio 1> from 1.0 s to 4.44 s, <d>[English, somber, soft singing] My <i>brother's</i> gone quiet <breath> the "
    "<i>throne</i> stands bare</d>, delivered softly with small natural lip and jaw movements.\n\n"
    "[Shot 2] At 00:04.600, continuing directly from the previous shot's final moment, <Subject 1> sings in sync "
    "with the vocals in <Audio 1> from 4.76 s to 5.76 s, <d>[English, quiet] I breathe the <i>air</i>.</d>, "
    "delivered softly.\n\n"
    "overall_soundscape:\n"
    "<Audio 1> remains the sole complete soundtrack.\n"
)


def _settings(**overrides):
    return payload_mod.normalize_minimax_h3_settings(overrides)


class WindowPlanTests(unittest.TestCase):
    def test_windows_start_on_groups_of_five_and_hold_5k_plus_2_tokens(self) -> None:
        total = plan_mod.tokens_for_frames(362)
        plan = plan_mod.window_plan(total, 3.0, 1.6, 0.7)
        covered = 0
        for start, end, keep_end in plan:
            self.assertEqual(start % 5, 0)
            self.assertEqual((end - start) % 5, 2)
            self.assertLessEqual(start, covered)
            self.assertLess(covered, keep_end)
            covered = keep_end
        self.assertEqual(covered, total)

    def test_the_heat_test_plan_is_nine_windows_of_22_tokens(self) -> None:
        plan = plan_mod.window_plan(plan_mod.tokens_for_frames(345), 3.0, 1.6, 0.7)
        self.assertEqual(len(plan), 9)
        self.assertEqual(plan[0], (0, 22, 17))

    def test_a_clip_that_is_not_17k_plus_5_frames_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            plan_mod.tokens_for_frames(100)

    def test_a_small_token_budget_uses_the_smallest_window_and_says_so(self) -> None:
        self.assertEqual(plan_mod.budget_window_tokens(0, 2040), (0, ""))
        frames, note = plan_mod.budget_window_tokens(1000, 2040)
        self.assertEqual(frames, plan_mod.MIN_WINDOW_TOKENS)
        self.assertIn("below the smallest supported window", note)

    def test_odd_spatial_sizes_round_up_to_the_patch_grid(self) -> None:
        self.assertEqual(plan_mod.tokens_per_latent_frame(68, 120), 2040)
        self.assertEqual(plan_mod.tokens_per_latent_frame(69, 121), 2135)


class ChunkingDecisionTests(unittest.TestCase):
    def test_longer_than_the_limit_is_chunked_and_at_the_limit_is_not(self) -> None:
        self.assertTrue(plan_mod.chunking_decision("auto", 8, 294)["enabled"])  # 12.25 s
        self.assertFalse(plan_mod.chunking_decision("auto", 8, 107)["enabled"])  # 4.46 s
        self.assertFalse(plan_mod.chunking_decision("auto", 12.25, 294)["enabled"])  # exactly the limit
        self.assertTrue(plan_mod.chunking_decision("auto", 12.2, 294)["enabled"])

    def test_off_and_unknown_modes_never_chunk(self) -> None:
        for mode in ("off", "", None, "sometimes"):
            self.assertFalse(plan_mod.chunking_decision(mode, 8, 294)["enabled"], mode)

    def test_the_limit_is_clamped_to_one_to_fifteen_seconds(self) -> None:
        self.assertEqual(plan_mod.normalize_min_seconds(0), 1.0)
        self.assertEqual(plan_mod.normalize_min_seconds(99), 15.0)
        self.assertEqual(plan_mod.normalize_min_seconds("junk"), plan_mod.DEFAULT_MIN_SECONDS)
        self.assertEqual(plan_mod.normalize_min_seconds(float("nan")), plan_mod.DEFAULT_MIN_SECONDS)

    def test_the_decision_reports_the_window_count_and_the_reason(self) -> None:
        decision = plan_mod.chunking_decision("auto", 8, 294)
        self.assertEqual(decision["windows"], len(plan_mod.window_plan(87, 3.0, 1.6, 0.7)))
        self.assertIn("longer than 8 s", decision["reason"])
        self.assertIn("not longer than 8 s", plan_mod.chunking_decision("auto", 8, 107)["reason"])

    def test_an_unknown_length_does_not_chunk(self) -> None:
        self.assertFalse(plan_mod.chunking_decision("auto", 8, None)["enabled"])


class StripPromptTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = chunked.strip_prompt_timing(SHOT_PROMPT)

    def test_lyric_cues_dialogue_and_times_are_gone(self) -> None:
        for fragment in ("<d>", "</d>", " from 1.0 s", " s to ", "At 00:04.600", "throne", "brother"):
            self.assertNotIn(fragment, self.text, fragment)

    def test_the_sentence_around_a_cue_still_reads(self) -> None:
        self.assertIn("sings in sync with the vocals in <Audio 1>, delivered softly with small natural lip", self.text)
        self.assertIn("[Shot 2] Continuing directly from the previous shot's final moment", self.text)

    def test_looks_references_and_sections_are_kept(self) -> None:
        for fragment in ("subject_definitions:", "<Subject 1> is Gabriel", "<Audio 1> is the complete synchronized",
                         "overall_soundscape:", "[Shot 1] A continuous medium close-up", "photorealistic"):
            self.assertIn(fragment, self.text, fragment)

    def test_a_prompt_with_no_timing_is_unchanged(self) -> None:
        plain = "detailed_description:\n[Shot 1] A wide shot of the street at night.\n"
        self.assertEqual(chunked.strip_prompt_timing(plain), plain)

    def test_shots_that_only_differed_by_lyric_and_time_are_merged_and_renumbered(self) -> None:
        sing = ("<Subject 1> sings in sync with the vocals in <Audio 1> from {a}.0 s to {b}.0 s, "
                "<d>[tag] line {n}</d>, delivered softly.")
        shots = [f"[Shot {n}] At 00:0{n}.000, continuing directly, " + sing.format(n=n, a=n, b=n + 1) for n in (1, 2, 3, 4)]
        prompt = "detailed_description:\nLook.\n\n" + "\n\n".join(shots) + "\n\noverall_soundscape:\nSong.\n"
        text = chunked.strip_prompt_timing(prompt)
        self.assertEqual(text.count("[Shot "), 1)
        self.assertIn("[Shot 1] Continuing directly, <Subject 1> sings", text)
        self.assertTrue(text.endswith("\n\noverall_soundscape:\nSong.\n"))

    def test_a_shot_that_returns_after_a_different_one_stays(self) -> None:
        prompt = "detailed_description:\n[Shot 1] A wide shot.\n\n[Shot 2] A close-up.\n\n[Shot 3] A wide shot.\n\n[Shot 4] A wide shot.\n"
        self.assertEqual(
            chunked.strip_prompt_timing(prompt),
            "detailed_description:\n[Shot 1] A wide shot.\n\n[Shot 2] A close-up.\n\n[Shot 3] A wide shot.\n",
        )

    def test_the_heat_prompt_collapses_to_the_shot_and_one_continuation(self) -> None:
        lead = "A continuous medium close-up, one unbroken take:"
        cont = "continuing directly from the previous shot's final moment in the same unbroken take, with no cut and the same framing,"
        sing = "<Subject 1> sings in sync with the vocals in <Audio 1> from {a} s to {b} s, <d>[tag] words</d>, delivered softly."
        shots = [f"[Shot 1] {lead} {sing.format(a=1.0, b=4.44)}"] + [
            f"[Shot {n}] At 00:{4 + n:02d}.000, {cont} {sing.format(a=n * 2, b=n * 2 + 1)}" for n in range(2, 7)
        ]
        text = chunked.strip_prompt_timing("detailed_description:\nLook.\n\n" + "\n\n".join(shots) + "\n")
        self.assertEqual(text.count("[Shot "), 2)
        self.assertIn("[Shot 2] Continuing directly from the previous shot's final moment", text)

    def test_no_stray_double_spaces_or_space_before_a_comma(self) -> None:
        self.assertNotIn("  ", self.text)
        self.assertNotIn(" ,", self.text)


class SettingsTests(unittest.TestCase):
    def test_defaults_are_auto_at_eight_seconds(self) -> None:
        settings = _settings()
        self.assertEqual((settings["pass2_chunking"], settings["pass2_chunking_min_seconds"]), ("auto", 8))

    def test_the_two_pass_payload_carries_both_settings_and_other_passes_do_not(self) -> None:
        two = payload_mod.build_minimax_render_payload(_settings(render_pass="two_pass", pass2_chunking="off", pass2_chunking_min_seconds=6.5))
        self.assertEqual((two["pass2_chunking"], two["pass2_chunking_min_seconds"]), ("off", 6.5))
        for passes in ("single", "three_pass"):
            self.assertNotIn("pass2_chunking", payload_mod.build_minimax_render_payload(_settings(render_pass=passes)), passes)

    def test_patches_are_checked(self) -> None:
        self.assertEqual(payload_mod.validate_minimax_h3_patch({"pass2_chunking": "off", "pass2_chunking_min_seconds": 7.5}), {})
        problems = payload_mod.validate_minimax_h3_patch({"pass2_chunking": "sometimes", "pass2_chunking_min_seconds": 40})
        self.assertEqual(set(problems), {"pass2_chunking", "pass2_chunking_min_seconds"})

    def test_saved_text_is_read_back_leniently(self) -> None:
        settings = _settings(pass2_chunking_min_seconds="9.5")
        self.assertEqual(settings["pass2_chunking_min_seconds"], 9.5)


class CompiledGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vrgdg_chunk_")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.image = self.project / "first.png"
        Image.new("RGB", (64, 64), "red").save(self.image)
        self.audio = self.project / "source.wav"
        with wave.open(str(self.audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\0\0" * 16000 * 20)
        choices = [
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors",
            "minimax_h3_latent_upscaler_3d_bf16.safetensors",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        ]
        for name in ("_folder_choices", "_lora_choices"):
            started = patch.object(models, name, return_value=choices)
            started.start()
            self.addCleanup(started.stop)

    def payload(self, seconds: float, prompt: str = SHOT_PROMPT, **setting_overrides):
        # fixed seeds: a negative seed is drawn at random, which would make two builds of one scene differ
        settings = _settings(
            video_mode="reference_to_video", render_pass="two_pass", two_pass_pass1_seed=11, two_pass_pass2_seed=22,
            **setting_overrides,
        )
        scene = {
            "prompt": prompt,
            "project_folder": str(self.project),
            "audio_path": str(self.audio),
            "source_duration_seconds": 20,
            "scene_duration_seconds": seconds,
            "scene_number": 1,
            "timeline_start_seconds": 0,
            "timeline_end_seconds": seconds,
            "image_paths": [str(self.image)],
        }
        return {**payload_mod.build_minimax_render_payload(settings), **scene}

    def build(self, seconds: float, **setting_overrides):
        return workflows._build_minimax_h3_2pass_api_prompt(self.payload(seconds, **setting_overrides))

    @staticmethod
    def classes(graph, name):
        return {key: node for key, node in graph.items() if node.get("class_type") == name}

    def test_a_clip_over_the_limit_swaps_the_second_pass_for_the_chunk_node(self) -> None:
        built = self.build(12)
        graph = built["prompt"]
        summary = built["chunked_pass2"]
        self.assertTrue(summary["enabled"], summary)
        chunk = self.classes(graph, chunked.CHUNK_CLASS)
        self.assertEqual(list(chunk), [summary["chunk_node_id"]])
        inputs = chunk[summary["chunk_node_id"]]["inputs"]
        self.assertEqual(
            {name: inputs[name] for name in ("chunk_seconds", "overlap_seconds", "overlap_mode", "tail_trim_seconds", "max_window_tokens", "upscale")},
            {"chunk_seconds": 3.0, "overlap_seconds": 1.6, "overlap_mode": "locked", "tail_trim_seconds": 0.7,
             "max_window_tokens": 0, "upscale": "per_window"},
        )
        self.assertNotIn("head_latent", inputs)
        for node_id in ("181", "182", "189", "193", "194", "211"):
            self.assertNotIn(node_id, graph, node_id)
        self.assertEqual(len(self.classes(graph, "SamplerCustomAdvanced")), 1, "only the first-pass sampler is left")
        self.assertEqual(len(self.classes(graph, "VRGDG_MiniMaxH3LearnedLatentUpscale")), 0)

    def test_the_chunk_node_reads_the_saved_first_pass_and_feeds_the_final_save(self) -> None:
        built = self.build(12)
        graph = built["prompt"]
        chunk_id = built["chunked_pass2"]["chunk_node_id"]
        pass1 = built["save_latent_settings"]["pass1_save_node_id"]
        final = built["save_latent_settings"]["save_node_id"]
        self.assertEqual(graph[chunk_id]["inputs"]["latent"], [pass1, 0])
        self.assertEqual(graph[final]["inputs"]["latent"], [chunk_id, 0])
        self.assertEqual(graph["142"]["class_type"], "VHS_VideoCombine")

    def test_the_chunk_node_keeps_the_passes_models_sampler_sigmas_scale_and_seed(self) -> None:
        reference = self.build(12, pass2_chunking="off")["prompt"]
        built = self.build(12)
        inputs = built["prompt"][built["chunked_pass2"]["chunk_node_id"]]["inputs"]
        self.assertEqual(inputs["model"], reference["193"]["inputs"]["model"])
        self.assertEqual(inputs["sampler"], reference["194"]["inputs"]["sampler"])
        self.assertEqual(inputs["sigmas"], reference["194"]["inputs"]["sigmas"])
        self.assertEqual(inputs["scale"], reference["182"]["inputs"]["scale"])
        self.assertEqual(inputs["upscale_model"], reference["182"]["inputs"]["upscale_model"])
        self.assertEqual(inputs["seed"], reference["211"]["inputs"]["noise_seed"])

    def test_the_windows_read_a_prompt_with_no_lyric_text_or_times(self) -> None:
        built = self.build(12)
        graph = built["prompt"]
        summary = built["chunked_pass2"]
        conditioning = graph[graph[summary["chunk_node_id"]]["inputs"]["conditioning"][0]]
        self.assertEqual(conditioning["class_type"], "MiniMaxH3ReferenceToVideo")
        text = graph[conditioning["inputs"]["prompt"][0]]["inputs"]["value"]
        self.assertNotIn("<d>", text)
        self.assertNotIn(" from 1.0 s", text)
        self.assertIn("[Shot 1] A continuous medium close-up", text)
        # the first pass keeps the full prompt, cues and all
        first_pass = graph[graph["126"]["inputs"]["conditioning"][0]]
        self.assertIn("<d>", graph[first_pass["inputs"]["prompt"][0]]["inputs"]["value"])
        self.assertNotEqual(graph[summary["chunk_node_id"]]["inputs"]["conditioning"], graph["126"]["inputs"]["conditioning"])

    def test_no_link_is_left_dangling(self) -> None:
        graph = self.build(12)["prompt"]
        for key, node in graph.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], graph, f"{key}.{name} -> {value}")

    def test_a_short_clip_keeps_the_normal_graph(self) -> None:
        built = self.build(4)
        self.assertFalse(built["chunked_pass2"]["enabled"])
        self.assertEqual(self.classes(built["prompt"], chunked.CHUNK_CLASS), {})
        self.assertEqual(built["prompt"], self.build(4, pass2_chunking="off")["prompt"])

    def test_off_and_a_missing_setting_keep_the_normal_graph_for_a_long_clip(self) -> None:
        off = self.build(12, pass2_chunking="off")
        self.assertIsNone(off["chunked_pass2"])
        self.assertEqual(self.classes(off["prompt"], chunked.CHUNK_CLASS), {})
        payload = self.payload(12)
        del payload["pass2_chunking"]
        direct = workflows._build_minimax_h3_2pass_api_prompt(payload)
        self.assertIsNone(direct["chunked_pass2"])
        self.assertEqual(direct["prompt"], off["prompt"])

    def test_the_limit_decides(self) -> None:
        self.assertTrue(self.build(12, pass2_chunking_min_seconds=12)["chunked_pass2"]["enabled"])
        self.assertFalse(self.build(12, pass2_chunking_min_seconds=13)["chunked_pass2"]["enabled"])

    def test_a_graph_it_does_not_recognise_stays_whole(self) -> None:
        payload = self.payload(12)
        built = workflows._build_minimax_h3_2pass_api_prompt({**payload, "pass2_chunking": "off"})
        graph = built["prompt"]
        graph["136"]["class_type"] = "MiniMaxH3ImageToVideo"
        before = copy.deepcopy(graph)
        summary = chunked.apply_chunked_pass2(graph, payload, built["timing"])
        self.assertFalse(summary["enabled"])
        self.assertIn("only Reference to Video and RefMod are chunked", summary["reason"])
        self.assertEqual(graph, before)

    def test_the_refmod_encoder_is_cloned_too(self) -> None:
        payload = self.payload(12)
        built = workflows._build_minimax_h3_2pass_api_prompt({**payload, "pass2_chunking": "off"})
        graph = built["prompt"]
        graph["136"]["class_type"] = "MiniMaxH3RefModTextEncode"
        summary = chunked.apply_chunked_pass2(graph, payload, built["timing"])
        self.assertTrue(summary["enabled"], summary)
        self.assertEqual(graph[summary["encode_node_id"]]["class_type"], "MiniMaxH3RefModTextEncode")

    def test_a_masked_continuation_head_is_handed_to_the_chunk_node(self) -> None:
        payload = self.payload(12)
        built = workflows._build_minimax_h3_2pass_api_prompt({**payload, "pass2_chunking": "off"})
        graph = built["prompt"]
        # what _patch_minimax_h3_latent_continuation_masked adds in front of the second sampler
        graph["9210"] = {"class_type": "VRGDG_MiniMaxH3LoadLatent", "inputs": {"scene_number": 1, "masked_av": True}}
        graph["9213"] = {
            "class_type": "VRGDG_MiniMaxH3ApplyMaskedContinuation",
            "inputs": {"latent": graph["194"]["inputs"]["latent_image"], "context_latent": ["9210", 0], "include_audio": False,
                       "vae": ["119", 0]},
        }
        graph["194"]["inputs"]["latent_image"] = ["9213", 0]
        summary = chunked.apply_chunked_pass2(graph, payload, built["timing"])
        self.assertTrue(summary["enabled"], summary)
        self.assertTrue(summary["head_locked"])
        inputs = graph[summary["chunk_node_id"]]["inputs"]
        self.assertEqual((inputs["head_latent"], inputs["head_vae"]), (["9210", 0], ["119", 0]))
        self.assertNotIn("9213", graph)
        self.assertIn("9210", graph)


if __name__ == "__main__":
    unittest.main()
