"""Custom-audio lyric omission and UI/API parity regressions."""

import importlib
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
lfp = importlib.import_module(f"{ROOT.name}.minimax.lyric_free_performance")
pa = importlib.import_module(f"{ROOT.name}.minimax.prompt_assembly")


class LyricFreePerformanceTests(unittest.TestCase):
    def test_owned_acting_keeps_one_performance_instruction(self):
        scene = {"lyric_text": "Secret song words", "facial_performance": "custom", "facial_performance_custom": "angry"}
        draft = ("<Subject 1> sings in sync with <Audio 1> [angry, singing] while she grips the zipper pull with her right hand. "
                 "<Subject 1> looks down at the zipper, then toward the camera. "
                 "<Subject 1>'s mouth follows only the audible vocal phrasing.")
        result = lfp.apply_shots([draft], scene, {"exact_duration_seconds": 4}, "<Subject 1>")[0]
        self.assertEqual(result.count("sings "), 1)
        self.assertEqual(result.count("audible vocal phrasing"), 1)
        self.assertIn("she grips the zipper pull with her right hand", result)

    def test_deduplication_requires_correct_actor_and_timing(self):
        contract = lfp.shot_direction("<Subject 2>", False, "2.5s–4s", False).replace("sings with passion", "sings")
        self.assertEqual(lfp.remaining_contract("<Subject 2> sings in sync with <Audio 1> during 2.5s–4s.", contract), "")
        self.assertEqual(lfp.remaining_contract("<Subject 1> sings in sync with <Audio 1> during 2.5s–4s.", contract), contract)
        self.assertEqual(lfp.remaining_contract("<Subject 2> sings in sync with <Audio 1>.", contract), contract)

    def setUp(self):
        self.plan = {"exact_duration_seconds": 4, "cut_times_seconds": [2], "shot_count": 2}
        self.scene = {
            "start": 0, "end": 4, "lyric_text": "Secret song words",
            "lyric_performance_mode": "cue_map",
            "lyric_cue_map": [
                {"type": "instrumental", "start": 0, "end": 2},
                {"type": "vocal", "start": 2, "end": 4, "text": "Secret song words", "singer_id": "b"},
            ],
        }

    def test_mixed_shots_do_not_leak_articulation_or_lyrics(self):
        shots = lfp.apply_shots(
            ["A camera tracks left. His jaw moves.", 'He sings "Secret song words". A camera pushes closer.'],
            self.scene, self.plan, labels={"b": "<Subject 2>"})
        self.assertNotIn("sings", shots[0])
        self.assertIn("<Subject 2> sings with passion", shots[1])
        self.assertIn("during 2s–4s", shots[1])
        for text in shots:
            self.assertNotRegex(text, r"Secret song words|mouth|lip|jaw|<d>")

    def test_continuous_shot_has_timed_singing(self):
        plan = {**self.plan, "cut_times_seconds": []}
        result = lfp.apply_shots(["A camera tracks left."], self.scene, plan)[0]
        self.assertIn("during 2s–4s", result)
        self.assertIn("mouth and jaw movement follows only the audible vocal", result)

    def test_instrumental_and_visual_only(self):
        for scene in [{"lyric_text": "[instrumental]"}, {**self.scene, "lyric_no_lip_sync": True},
                      {**self.scene, "no_character_present": True}]:
            self.assertNotIn("sings", lfp.apply_shots(["A tracking shot."], scene, self.plan)[0])

    def test_option_is_limited_to_supplied_audio_singing(self):
        self.assertTrue(lfp.enabled(True, "singing", "input_audio"))
        for values in [(False, "singing", "input_audio"), (True, "speaking", "input_audio"),
                       (True, "singing", "built_in_audio")]:
            self.assertFalse(lfp.enabled(*values))

    def test_api_assembly_uses_project_setting(self):
        session = {"omit_lyrics_from_video_prompts": True, "video_type": "singing",
                   "builder_storyboard_defaults": {"minimax_h3_cut_frequency": 5}}
        result = pa.assemble_minimax_h3_prompt(self.scene, session,
                                              ["A camera tracks left.", "A dolly pushes closer."], mode="text_to_video")
        self.assertNotRegex(result["prompt"], r"Secret song words|mouth|jaw")
        self.assertIn("sings with passion", result["prompt"])

    def test_context_replaces_exact_vocal_contract(self):
        text = 'MiniMax H3 shot-description task.\n\nMANDATORY VOCAL PERFORMANCE: sing "Secret song words".\n\nMotion: walk forward.'
        result = lfp.prompt_context(text, self.scene, self.plan)
        self.assertNotIn("Secret song words", result)
        self.assertIn("Motion: walk forward", result)
        self.assertIn("Do not mention mouth, lip, or jaw movement", result)

    def test_facial_presets_match_browser_and_keep_expression(self):
        source = (ROOT / "web/storyboard_builder/performance_presets.mjs").read_text(encoding="utf-8")
        bank = source.split("export const FACIAL_PERFORMANCE_PRESETS = [", 1)[1].split("];", 1)[0]
        presets = dict(re.findall(r'value: "([^"]*)",[\s\S]*?direction: "([^"]*)"', bank))
        self.assertEqual(lfp._FACIAL_PRESETS, presets)
        result = lfp.facial_text({"facial_performance": "sad_wounded"})
        self.assertIn("watery eyes", result)
        self.assertNotRegex(result, r"mouth|lip|jaw")

    def test_selected_singer_ids_use_the_correct_renderer_label(self):
        scene = {"id": "scene", "start": 0, "lyric_singers": ["Bob"]}
        session = {"segments": [scene], "flux_reference_builder": {
            "subjects": [
                {"id": "a", "name": "Alice", "image": {"path": "alice.png"}},
                {"id": "b", "name": "Bob", "image": {"path": "bob.png"}},
            ], "subject_scene_map": {"scene": ["a", "b"]},
        }}
        performer, labels = lfp.scene_performers(session, scene, "reference_to_video")
        self.assertEqual(performer, "<Subject 2>")
        self.assertEqual(labels["b"], "<Subject 2>")

    def test_javascript_assembly(self):
        subprocess.run(["node", "--test", str(ROOT / "tests/lyric_free_performance.cjs")], check=True)


if __name__ == "__main__":
    unittest.main()
