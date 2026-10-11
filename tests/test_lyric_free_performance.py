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
    def test_tagged_pronoun_cue_is_kept_once_with_its_action_and_emotion(self):
        contract = "<Subject 1> sings in sync with the vocals in <Audio 1>."
        draft = ("<Subject 1> keeps her stance planted and carries a restrained pulse through her shoulders "
                 "as she [intense, singing] sings in sync with the vocals in <Audio 1>. "
                 "<Subject 1> holds an engaged gaze toward the camera. Warm light falls across the doorway.")
        for text in (draft, draft + " " + contract):
            merged, extra = lfp.merge_contract(text, contract)
            self.assertEqual(extra, "")
            self.assertEqual(len(re.findall(r"sings\b", merged)), 1)
            self.assertEqual(merged.count("<Audio 1>"), 1)
            self.assertIn("she [intense, singing] sings", merged)
            self.assertIn("restrained pulse through her shoulders", merged)
            self.assertEqual(lfp.merge_contract(merged, contract), (merged, ""))
        scene = {"lyric_text": "Secret song words", "facial_performance_custom": "intense"}
        out = lfp.apply_shots([draft], scene, {"exact_duration_seconds": 4}, "<Subject 1>")[0]
        self.assertEqual(len(re.findall(r"sings\b", out)), 1)
        self.assertIn("[intense, singing]", out)

    def test_pronoun_ownership_requires_one_named_actor_and_preserves_other_times(self):
        contract = "<Subject 1> sings in sync with the vocals in <Audio 1> during 2.5s–4s."
        draft = "<Subject 1> stands beside the doorway. She [intense, singing] sings in sync with <Audio 1> during 2.5s–4s."
        merged, extra = lfp.merge_contract(draft, contract)
        self.assertEqual(extra, "")
        self.assertEqual(merged.count("<Audio 1>"), 1)
        self.assertIn("during 2.5s–4s.", merged)
        for text in ('<Subject 1> watches <Subject 2> as she [sad, singing] sings.',
                     '<Subject 1> holds a sign saying "she sings".',
                     '<Subject 1> stands nearby. She sings during 1s–2s.'):
            self.assertEqual(lfp.merge_contract(text, contract), (text, contract))
        text = "<Subject 1> [intense, singing] sings in sync with <Audio 1>. <Subject 2> sings in sync with <Audio 1> during 1s–2s."
        merged, _ = lfp.merge_contract(text, contract)
        self.assertIn("<Subject 2> sings in sync with <Audio 1> during 1s–2s.", merged)

    def test_inline_singing_merges_audio_once_and_removes_articulation(self):
        draft = ("<Subject 1> walks slowly alongside the table; <Subject 1> sings passionately in sync with the audible vocals, "
                 "with natural mouth and jaw articulation, narrowed eyes, and brows drawn in a defiant expression. Dust drifts.")
        scene = {"lyric_text": "Secret song words", "facial_performance_custom": "Rock/Punk"}
        result = lfp.apply_shots([draft], scene, {"exact_duration_seconds": 4}, "<Subject 1>")[0]
        self.assertEqual(result.count("sings "), 1)
        self.assertEqual(result.count("<Audio 1>"), 1)
        self.assertNotRegex(result, r"mouth|lips?|jaw")
        for text in ("walks slowly alongside the table", "narrowed eyes", "brows drawn", "Dust drifts"):
            self.assertIn(text, result)

    def test_articulation_removal_preserves_quoted_words_and_sentence_boundaries(self):
        result = lfp.strip_singing_articulation('She walks, her mouth and jaw moving to the vocals. She sings "My lips are sealed."')
        self.assertIn("She walks.", result)
        self.assertIn('"My lips are sealed."', result)
        self.assertNotIn("jaw moving", result)

    def test_distinct_vocal_intervals_for_one_singer_remain_distinct(self):
        first = "<Subject 1> sings in sync with the vocals in <Audio 1> during 1s–2s."
        second = "<Subject 1> sings in sync with the vocals in <Audio 1> during 3s–4s."
        self.assertEqual(lfp.merge_contract(first, second), (first, second))

    def test_lyrics_included_still_removes_articulation_but_not_exact_words(self):
        scene = {"start": 0, "end": 4, "lyric_text": "My lips are sealed"}
        session = {"video_type": "singing", "omit_lyrics_from_video_prompts": False}
        result = pa.assemble_minimax_h3_prompt(scene, session,
            ['She sings "My lips are sealed". Her mouth and jaw move to the vocals. The camera tracks left.'], "text_to_video")
        self.assertIn('"My lips are sealed"', result["prompt"])
        self.assertNotIn("jaw", result["prompt"])
        self.assertIn("camera tracks left", result["prompt"])

    def test_owned_acting_keeps_one_performance_instruction(self):
        scene = {"lyric_text": "Secret song words", "facial_performance": "custom", "facial_performance_custom": "angry"}
        draft = ("<Subject 1> sings in sync with <Audio 1> [angry, singing] while she grips the zipper pull with her right hand. "
                 "<Subject 1> looks down at the zipper, then toward the camera. "
                 "<Subject 1>'s mouth follows only the audible vocal phrasing.")
        result = lfp.apply_shots([draft], scene, {"exact_duration_seconds": 4}, "<Subject 1>")[0]
        self.assertEqual(result.count("sings "), 1)
        self.assertNotRegex(result, r"mouth|lips?|jaw|audible vocal phrasing")
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
        self.assertNotRegex(result, r"mouth|lips?|jaw")

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
