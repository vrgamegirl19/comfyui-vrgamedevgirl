"""Performance craft, timed cue delivery and cut snapping in the MiniMax H3 prompt writers (Builder and Agent API)."""

import importlib
import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

sp = importlib.import_module(f"{ROOT.name}.minimax.shot_prompt")
pa = importlib.import_module(f"{ROOT.name}.minimax.prompt_assembly")
lfp = importlib.import_module(f"{ROOT.name}.minimax.lyric_free_performance")
CASES = json.loads((ROOT / "tests" / "minimax_craft_cases.json").read_text(encoding="utf-8"))


def craft(item):
    return sp.performance_craft_instruction(item["style"], item["hasCast"], item["hasVocals"], item["continuation"])


class PerformanceCraftTests(unittest.TestCase):
    def test_craft_follows_the_scene(self):
        full, silent, continued, empty = (craft(item) for item in CASES["craft"])
        self.assertIn("Scale every expression to the Anime video style", full)
        self.assertIn(sp.MINIMAX_H3_CRAFT_FRAMING, full)
        self.assertIn(sp.MINIMAX_H3_CRAFT_OPENING, full)
        self.assertIn("the chosen video style", silent)
        self.assertNotIn(sp.MINIMAX_H3_CRAFT_FRAMING, silent, "no vocals, no singing framing")
        self.assertNotIn(sp.MINIMAX_H3_CRAFT_FRAMING, continued, "a continued scene keeps its inherited framing")
        self.assertNotIn(sp.MINIMAX_H3_CRAFT_OPENING, continued)
        self.assertIn(sp.MINIMAX_H3_CRAFT_CLEAR_FACE, continued)
        self.assertEqual(empty, sp.MINIMAX_H3_CRAFT_OPENING + "\n" + sp.MINIMAX_H3_CRAFT_ACTION)

    def test_no_genre_words_in_the_craft(self):
        text = craft(CASES["craft"][0]).lower()
        for word in ("punk", "party", "scornful", "defiant", "guitar"):
            self.assertNotIn(word, text)

    def test_cue_delivery_keeps_the_cue_words(self):
        self.assertEqual(sp.cue_delivery(*CASES["delivery"][1]), "Say the quiet lasted <i>too long</i>. <breath> If the verdict")
        self.assertEqual(sp.cue_delivery(*CASES["delivery"][2]), "Hold me close.", "changed words drop the marks")
        self.assertEqual(sp.cue_header("sad, longing, singing"), "English, sad, longing, singing")

    def test_enforced_cue_shot_has_header_window_and_survives_the_singing_stripper(self):
        cue = {"type": "vocal", "text": "fuck the fear that calls itself fate", "label": "<Subject 1> (Ava)",
               "vocal_start": 2.5, "vocal_end": 4.7}
        llm = ('A medium shot of <Subject 1> swinging her hips, "Fuck the fear that calls itself fate", '
               "<d>[English, fierce, drunk, singing] Fuck the <i>fear</i> that calls itself fate</d>, her brows knotting.")
        out = lfp.strip_singing_articulation(sp.enforce_cue_shot(llm, cue))
        self.assertIn("in sync with <Audio 1> from 2.5 s to 4.7 s: <d>[English, fierce, drunk, singing] Fuck the <i>fear</i> that calls itself fate.</d>", out)
        self.assertEqual(out.count("<d>"), 1)
        self.assertIn("every other visible performer remains silent", out)

    def test_instrumental_cue_shot_loses_vocal_sentences(self):
        out = sp.enforce_cue_shot("She spins on the spot. She sings the chorus.", {"type": "instrumental", "text": ""})
        self.assertEqual(out, "She spins on the spot.")

    def test_timed_cue_rows_and_cut_plan(self):
        segment = {"lyric_performance_mode": "cue_map", "lyric_cue_map": [
            {"type": "vocal", "text": "one two", "start": 0, "vocal_start": 0.2, "vocal_end": 1.5},
            {"type": "instrumental", "text": "", "start": 2.1},
            {"type": "vocal", "text": "three", "start": 4.0, "vocal_start": 4.1, "vocal_end": 5.0}]}
        rows = sp.timed_cue_rows(segment)
        self.assertEqual([r["type"] for r in rows], ["vocal", "instrumental", "vocal"])
        plan = sp.cue_cut_plan({"exact_duration_seconds": 6.0, "cut_times_seconds": [3.0]}, rows)
        self.assertEqual(plan["cut_times_seconds"], [2.1, 4.0])
        self.assertEqual(sp.timed_cue_rows({**segment, "lyric_performance_mode": "together"}), [])
        untimed = {"lyric_performance_mode": "cue_map", "lyric_cue_map": [{"type": "vocal", "text": "x"}]}
        self.assertEqual(sp.timed_cue_rows(untimed), [])

    def test_task_has_craft_and_no_mouth_direction_for_supplied_audio(self):
        plan = pa.storyboard_cut_plan_for_duration(5.0, 0)
        labels = [{"kind": "subject", "label": "<Subject 1>", "name": "Ava", "picture": "<Picture 1>"}]
        task = sp.build_shot_task(mode_label="Reference to Video", duration=5.0, aspect_ratio="16:9", audio_mode="input_audio",
                                  cut_plan=plan, style="Watercolor", labels=labels, camera_speed=4, character_speed=4,
                                  lyric_text="Hold me close tonight")
        self.assertIn("PERFORMANCE CRAFT — MANDATORY", task)
        self.assertIn("Watercolor video style", task)
        self.assertNotIn("visible mouth, lip, and jaw movement", task)
        self.assertNotIn("Show clear, natural mouth", task)
        built_in = sp.build_shot_task(mode_label="Reference to Video", duration=5.0, aspect_ratio="16:9", audio_mode="built_in_audio",
                                      cut_plan=plan, style="Watercolor", labels=labels, camera_speed=4, character_speed=4,
                                      lyric_text="Hold me close tonight")
        self.assertIn("visible mouth, lip, and jaw movement", built_in)

    def test_closed_lips_survive_the_singing_stripper(self):
        out = lfp.strip_singing_articulation("<Subject 2> watches, his lips pressed together. She sways, her mouth moving to the words.")
        self.assertIn("his lips pressed together", out)
        self.assertNotIn("mouth moving", out)

    def test_negative_words_inside_a_cue_keep_the_sentence(self):
        text = "She turns. <Subject 1> sings <d>[English, tender, singing] Don't let go.</d>"
        self.assertEqual(sp.strip_negative_sentences(text), text)


@unittest.skipUnless(shutil.which("node"), "Node.js is required for UI tests")
class BuilderParityTests(unittest.TestCase):
    def test_inline_cue_sentences_preserve_acting_and_match_the_builder(self):
        cases = json.loads((ROOT / "tests/minimax_cue_sentence_cases.json").read_text(encoding="utf-8"))
        result = subprocess.run(["node", str(ROOT / "tests/minimax_cue_sentences.cjs")],
                                capture_output=True, text=True, encoding="utf-8", cwd=ROOT, check=True)
        outputs = []
        for case in cases:
            with self.subTest(case=case["name"]):
                merged = sp.enforce_cue_shot(case["description"], case["cue"])
                clean = lfp.strip_singing_articulation(merged)
                for text in case["contains"]:
                    self.assertIn(text, clean)
                self.assertNotIn("assigned vocal cue", clean)
                self.assertEqual(clean.count("<d>"), 1)
                visual_prose = re.sub(r"<d>[\s\S]*?</d>", "", clean, flags=re.I)
                self.assertEqual(len(re.findall(r"\b(?:sings|singing)\b", visual_prose)), 1)
                self.assertEqual(sp.enforce_cue_shot(merged, case["cue"]), merged)
                outputs.append(clean)
        self.assertEqual(json.loads(result.stdout), outputs)

    def test_python_twins_match_the_video_builder(self):
        result = subprocess.run(["node", str(ROOT / "tests" / "minimax_craft_parity.cjs")], capture_output=True, text=True,
                                encoding="utf-8", cwd=ROOT, check=True)
        js = json.loads(result.stdout)
        self.assertEqual(js["craft"], [craft(item) for item in CASES["craft"]])
        self.assertEqual(js["header"], [sp.cue_header(item) for item in CASES["header"]])
        self.assertEqual(js["delivery"], [sp.cue_delivery(text, body) for text, body in CASES["delivery"]])
        self.assertEqual(js["snap"], [[2.45, 4.5], [2.5], [1.2], []])

    def test_builder_cue_sentence_has_no_lip_words(self):
        source = (ROOT / "web" / "music_video_builder" / "minimax_prompt.mjs").read_text(encoding="utf-8")
        self.assertIn("${performer} sings in sync with <Audio 1>${vocalWindow}: ${cueTag}", source)
        self.assertIn("miniMaxH3PerformanceCraftInstruction({", source)
        cues = (ROOT / "web" / "music_video_builder" / "lyric_cues.mjs").read_text(encoding="utf-8")
        self.assertIn("snapCutsToVocalGaps(", cues)


if __name__ == "__main__":
    unittest.main()
