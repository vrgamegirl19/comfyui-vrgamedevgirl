"""MiniMax H3 reference-to-video prompt format (Python twin of the Video Builder's assembly)."""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

sp = importlib.import_module(f"{ROOT.name}.minimax.shot_prompt")
pa = importlib.import_module(f"{ROOT.name}.minimax.prompt_assembly")


class ShotPromptTests(unittest.TestCase):
    def plan(self, duration, frequency=0):
        return pa.storyboard_cut_plan_for_duration(duration, frequency)

    def test_single_shot_matches_the_saved_ui_format(self):
        plan = self.plan(4.0)
        prompt = sp.assemble_prompt(["The camera opens wide on the porch. A slow dolly moves left."], plan, "cinematic_realism")
        self.assertEqual(prompt, "detailed_description:\nThe target video is in a cinematic_realism music-video style.\n\n"
                                 "[Shot 1] The camera opens wide on the porch. A slow dolly moves left.")
        self.assertEqual(sp.validate_prompt(prompt, plan), prompt)

    def test_cuts_get_timecodes_and_post_cut_text(self):
        plan = self.plan(8.0, 5)
        count = len(sp.shot_plan(plan))
        self.assertGreater(count, 1)
        prompt = sp.assemble_prompt([f"Shot description number {i} keeps moving." for i in range(count)], plan, "x")
        self.assertIn("[Shot 2] At ", prompt)
        self.assertIn(", the camera cuts. Shot description number 1", prompt)
        sp.validate_prompt(prompt, plan)

    def test_json_is_parsed_and_missing_shots_filled(self):
        raw = 'noise {"shots":[{"description":"One."},{"description":"Two."}]} trailing'
        self.assertEqual(sp.parse_shot_descriptions(raw, 2), ["One.", "Two."])
        self.assertEqual(sp.parse_shot_descriptions('{"shots":[{"description":"One."}]}', 2)[1], sp.FALLBACK_SHOT)

    def test_bad_output_is_rejected(self):
        with self.assertRaises(sp.ShotPromptError):
            sp.parse_shot_descriptions("not json", 1)
        with self.assertRaises(sp.ShotPromptError):
            sp.parse_shot_descriptions('{"shots":["a","b"]}', 1)
        with self.assertRaises(sp.ShotPromptError):
            sp.parse_shot_descriptions('{"shots":[{"description":"[Shot 1] hi"}]}', 1)

    def test_negative_sentences_are_dropped_and_labels_normalised(self):
        raw = '{"shots":[{"description":"Subject 1 walks forward. Do not show a crowd. The light glows."}]}'
        self.assertEqual(sp.parse_shot_descriptions(raw, 1), ["<Subject 1> walks forward. The light glows."])

    def test_an_unpaired_quote_is_removed(self):
        raw = '{"shots":[{"description":"He stands near the window. \\" A slight tension enters his jaw."}]}'
        self.assertEqual(sp.parse_shot_descriptions(raw, 1), ["He stands near the window. A slight tension enters his jaw."])

    def test_lyrics_are_quoted_in_the_shot(self):
        lyric = "If you open up your eyes and feel the dark,\nLike the heavy world is falling all apart"
        said = "A man paces. He sings the lyric line, If you open up your eyes and feel the dark, Like the heavy world is falling all apart."
        out = sp.ensure_quoted_lyrics([said], lyric, "<Subject 1> (dave)")
        self.assertIn('"If you open up your eyes and feel the dark, Like the heavy world is falling all apart"', out[0])
        out = sp.ensure_quoted_lyrics(["A man paces slowly."], lyric, "<Subject 1> (dave)")
        self.assertEqual(out[0], 'A man paces slowly. <Subject 1> (dave) sings the lyric line, '
                                 '"If you open up your eyes and feel the dark, Like the heavy world is falling all apart".')
        already = 'He sings the lyric line, "If you open up your eyes and feel the dark, Like the heavy world is falling all apart".'
        self.assertEqual(sp.ensure_quoted_lyrics([already], lyric), [already])

    def test_lyric_lines_are_shared_out_across_cuts_in_order(self):
        lines = "one\ntwo\nthree\n[Chorus]\nfour"
        self.assertEqual(sp.lyric_chunks(lines, 1), ["one two three four"])
        self.assertEqual(sp.lyric_chunks(lines, 2), ["one two", "three four"])
        self.assertEqual(sp.lyric_chunks("", 2), ["", ""])

    def test_quoted_lyric_words_are_never_dropped_as_negative_wording(self):
        raw = '{"shots":[{"description":"He steps forward. He sings \\"I don\'t ever look back\\". Do not show a crowd."}]}'
        self.assertEqual(sp.parse_shot_descriptions(raw, 1), ['He steps forward. He sings "I don\'t ever look back".'])

    def test_the_task_asks_for_the_lyric_in_quotes(self):
        plan = self.plan(4.0)
        labels = sp.reference_labels([{"kind": "subject", "label": "dave"}])
        task = sp.build_shot_task(mode_label="Reference to Video", duration=4, aspect_ratio="16:9", audio_mode="input_audio", cut_plan=plan,
                                  style="s", labels=labels, camera_speed=3, character_speed=3, lyric_text="line one\nline two")
        self.assertIn("LYRIC IN THE SHOT", task)
        self.assertIn('<Subject 1> (dave) sings the lyric line, "the exact words"', task)
        self.assertIn("Exact lyric line:\nline one line two", task)

    def test_too_long_and_incomplete_prompts_fail_validation(self):
        plan = self.plan(4.0)
        with self.assertRaises(sp.ShotPromptError) as ctx:
            sp.validate_prompt(sp.assemble_prompt(["x" * 7100 + "."], plan, "s"), plan)
        self.assertEqual(ctx.exception.code, "MINIMAX_H3_PROMPT_TOO_LONG")
        with self.assertRaises(sp.ShotPromptError):
            sp.validate_prompt("detailed_description:\n\n[Shot 1] short", plan)

    def test_budget_and_labels(self):
        plan = self.plan(4.0)
        budget = sp.character_budget(plan, "cinematic_realism", 7000)
        self.assertEqual(budget["shot_chars"], 7000 - budget["fixed_chars"])
        labels = sp.reference_labels([{"kind": "start_frame"}, {"kind": "subject", "label": "Darrel"}, {"kind": "location", "label": "Rooftop"}])
        self.assertEqual([(l["label"], l["picture"], l["kind"]) for l in labels],
                         [("<Subject 1>", "<Picture 2>", "subject"), ("<Subject 2>", "<Picture 3>", "location")])

    def test_task_carries_cast_lyric_and_cut_plan(self):
        plan = self.plan(4.0)
        labels = sp.reference_labels([{"kind": "subject", "label": "Darrel"}])
        task = sp.build_shot_task(mode_label="Reference to Video", duration=4, aspect_ratio="16:9", audio_mode="input_audio", cut_plan=plan,
                                  style="s", labels=labels, camera_speed=7, character_speed=5, lyric_text="line one",
                                  story_beat="Darrel walks.", location_text="Rooftop")
        self.assertTrue(task.startswith("MiniMax H3 shot-description task."))
        self.assertIn("<Subject 1> (Darrel)", task)
        self.assertIn("Exact lyric line:\nline one", task)
        self.assertIn("one smooth, continuous", task)
        self.assertIn("<Subject 1> walks.", task, "names in the scene text become labels")
        self.assertIn("Camera rule", task)


if __name__ == "__main__":
    unittest.main()
