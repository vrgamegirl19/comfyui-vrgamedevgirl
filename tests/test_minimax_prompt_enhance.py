"""MiniMax H3 Enhance Prompts: only the rewritten [Shot N] blocks replace the original's, and only when safe."""

import ast
import importlib
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
# Load the real modules without booting the ComfyUI node registry.
PACKAGE = "_vrgdg_prompt_enhance_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
with patch.dict(sys.modules, {PACKAGE: package, "folder_paths": types.ModuleType("folder_paths")}):
    pe = importlib.import_module(f"{PACKAGE}.minimax.prompt_enhance")
    instructions = importlib.import_module(f"{PACKAGE}.llm.builder_instructions")
    minimax_instructions = importlib.import_module(f"{PACKAGE}.llm.prompts.minimax")

HEAD = ("subject_definitions:\n<Subject 1> is the singer from the first character reference.\n\n"
        "<Audio 1> is the complete song.\n\ndetailed_description:\n"
        "The target video is in a photorealistic cinematic music-video style.\n\n")
TAIL = "\n\noverall_soundscape:\n<Audio 1> remains the sole soundtrack."
SHOTS = ("[Shot 1] A medium shot of <Subject 1> at the table as she sings in sync with the vocals in <Audio 1> "
         "from 0.3 s to 3.3 s, <d>[English, soft, singing] A silver <i>fork</i></d>, smiling.\n\n"
         "[Shot 2] At 00:03.450, continuing directly from the previous shot's final moment, <Subject 1> still at "
         "the table sings in sync with the vocals in <Audio 1> from 3.6 s to 4.8 s, <d>[English, singing] A hundred "
         "little</d>, looking down.")
ORIGINAL = HEAD + SHOTS + TAIL

GOOD_REPLY = ("[Shot 1] A wide-angle close-up of <Subject 1> at the table, her face large in the frame, wagging a "
              "fork as she sings in sync with the vocals in <Audio 1> from 0.3 s to 3.3 s, <d>[English, cheeky, "
              "playful, singing] A silver <i>fork</i></d>, one brow lifting.\n\n"
              "[Shot 2] At 00:03.450, continuing directly from the previous shot's final moment, <Subject 1> still "
              "at the table, the camera cuts to a low side angle as she sings in sync with the vocals in <Audio 1> "
              "from 3.6 s to 4.8 s, <d>[English, mocking, sly, singing] A hundred little</d>, then holds a sly grin.")

REFERENCES = [
    {"label": "<Subject 1>", "picture_label": "<Picture 1>", "name": "the singer", "kind": "subject",
     "purpose": "character identity and wardrobe reference", "describe": True, "path": "singer.png"},
    {"label": "<Subject 2>", "picture_label": "<Picture 2>", "name": "the environment", "kind": "location",
     "purpose": "environment reference", "describe": True, "path": "room.png"},
]
VISION_REPLY = {"prompt": GOOD_REPLY, "reference_descriptions": [
    {"label": "<Subject 1>", "description": "A woman with a red bob, an ivory ruffled top and a black pleated skirt."},
    {"label": "<Subject 2>", "description": "A wood-paneled dining room with a long table and warm overhead lights."},
]}


class PromptEnhanceTests(unittest.TestCase):
    def test_vision_adds_mapped_appearance_and_location_but_preserves_fixed_sections(self):
        merged = pe.merge_reference_enhanced_prompt(ORIGINAL, VISION_REPLY, REFERENCES)
        self.assertIn("<Subject 1> is the singer in <Picture 1>: A woman with a red bob", merged)
        self.assertIn("<Subject 2> is the environment in <Picture 2>: A wood-paneled", merged)
        self.assertIn("<Audio 1> is the complete song.", merged)
        self.assertIn("The target video is in a photorealistic cinematic music-video style.", merged)
        self.assertTrue(merged.endswith(TAIL.strip()))
        self.assertIn("[Shot 2] At 00:03.450, " + pe.CONTINUATION, merged)

    def test_vision_can_enrich_compact_prompt_and_reenhance_without_duplicate_definitions(self):
        merged = pe.merge_reference_enhanced_prompt(SHOTS, VISION_REPLY, REFERENCES)
        self.assertTrue(merged.startswith("subject_definitions:\n"))
        twice = pe.merge_reference_enhanced_prompt(merged, VISION_REPLY, REFERENCES)
        self.assertEqual(merged, twice)
        self.assertEqual(merged.count("<Subject 1> is"), 1)

    def test_vision_definitions_cannot_bypass_lyric_or_timing_checks(self):
        for bad_shots in (GOOD_REPLY.replace("A hundred little", "A thousand little"),
                          GOOD_REPLY.replace("from 3.6 s", "from 3.5 s")):
            with self.subTest(bad_shots=bad_shots), self.assertRaises(pe.PromptEnhanceError):
                pe.merge_reference_enhanced_prompt(ORIGINAL, {**VISION_REPLY, "prompt": bad_shots}, REFERENCES)

    def test_vision_rejects_missing_duplicate_unknown_or_injected_descriptions(self):
        for change in ("missing", "duplicate", "unknown", "injection", "too long"):
            reply = copy.deepcopy(VISION_REPLY)
            if change == "missing": reply["reference_descriptions"].pop()
            elif change == "duplicate": reply["reference_descriptions"][1]["label"] = "<Subject 1>"
            elif change == "unknown": reply["reference_descriptions"][1]["label"] = "<Subject 9>"
            elif change == "injection": reply["reference_descriptions"][0]["description"] = "Red hair.\n[Shot 9] Someone else."
            else: reply["reference_descriptions"][0]["description"] = "x" * 451
            with self.subTest(change=change), self.assertRaises(pe.PromptEnhanceError):
                pe.merge_reference_enhanced_prompt(ORIGINAL, reply, REFERENCES)

    def test_vision_descriptions_count_toward_the_final_character_limit(self):
        with patch.object(pe, "HARD_LIMIT", len(ORIGINAL)):
            with self.assertRaises(pe.PromptEnhanceError):
                pe.merge_reference_enhanced_prompt(ORIGINAL, VISION_REPLY, REFERENCES)

    def test_compact_image_reference_prompt_keeps_start_and_end_picture_assignments(self):
        refs = copy.deepcopy(REFERENCES)
        refs[0]["picture_label"] = "<Picture 2>"
        refs[1]["picture_label"] = "<Picture 3>"
        refs += [
            {"label": "<Picture 1>", "picture_label": "<Picture 1>", "kind": "start_frame", "describe": False},
            {"label": "<Picture 4>", "picture_label": "<Picture 4>", "kind": "end_frame", "describe": False},
        ]
        merged = pe.merge_reference_enhanced_prompt(SHOTS, VISION_REPLY, refs)
        self.assertIn("<Picture 1> is the first frame of [Shot 1]", merged)
        self.assertIn("<Picture 4> is the last frame of [Shot 2]", merged)
        self.assertIn("<Subject 1> is the singer in <Picture 2>", merged)
        twice = pe.merge_reference_enhanced_prompt(merged, VISION_REPLY, refs)
        self.assertEqual(merged, twice)

    def test_image_and_text_enhancement_have_separate_editable_instructions(self):
        defaults = instructions._BUILDER_INSTRUCTION_DEFAULTS
        self.assertIn("reference_descriptions", defaults["minimax_h3_prompt_enhance_with_references"])
        self.assertNotIn("reference_descriptions", defaults["minimax_h3_prompt_enhance"])

    def test_only_the_shot_blocks_change(self):
        merged = pe.merge_enhanced_prompt(ORIGINAL, "Here you go:\n" + HEAD.replace("singer", "STAR") + GOOD_REPLY
                                          + "\n\noverall_soundscape:\nchanged")
        self.assertTrue(merged.startswith(HEAD.rstrip()))
        self.assertTrue(merged.endswith(TAIL.strip()))
        self.assertIn("wide-angle close-up", merged)
        self.assertNotIn("STAR", merged)
        self.assertNotIn("changed", merged)

    def test_inline_shot_mentions_in_reference_sections_are_not_shot_blocks(self):
        prefix = HEAD.replace("<Audio 1>", "<Picture 1> is the first frame of [Shot 1].\n<Audio 1>", 1)
        prefix = prefix.replace("detailed_description:", "retention_analysis:\n<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved.\n\ndetailed_description:")
        merged = pe.merge_enhanced_prompt(prefix + SHOTS + TAIL, GOOD_REPLY)
        self.assertTrue(merged.startswith(prefix))
        self.assertEqual(len(pe.split_prompt(merged)[1]), 2)

    def test_a_lost_cue_time_is_put_back(self):
        reply = GOOD_REPLY.replace("[Shot 2] At 00:03.450, ", "[Shot 2] ")
        self.assertIn("[Shot 2] At 00:03.450, continuing", pe.merge_enhanced_prompt(ORIGINAL, reply))

    def test_bad_rewrites_are_refused(self):
        cases = {
            "changed sung words": GOOD_REPLY.replace("A hundred little", "A thousand little"),
            "changed timing": GOOD_REPLY.replace("from 3.6 s to 4.8 s", "from 3.0 s to 4.8 s"),
            "lost label": GOOD_REPLY.replace("vocals in <Audio 1> from 0.3", "vocals from 0.3"),
            "new label": GOOD_REPLY.replace("fork as", "fork at <Subject 2> as"),
            "missing shot": GOOD_REPLY.split("\n\n")[0],
            "no shots": "I improved the camera and the faces.",
            "cut off": GOOD_REPLY.rstrip(".") + " and then she",
        }
        for name, reply in cases.items():
            with self.subTest(name), self.assertRaises(pe.PromptEnhanceError):
                pe.merge_enhanced_prompt(ORIGINAL, reply)

    def test_compact_prompts_without_sections_work(self):
        merged = pe.merge_enhanced_prompt(SHOTS, GOOD_REPLY)
        self.assertEqual(merged, pe.CONTINUOUS_PERFORMANCE + "\n\n" + GOOD_REPLY)

    def test_missing_continuity_opening_is_restored_after_original_timestamp(self):
        reply = GOOD_REPLY.replace(pe.CONTINUATION + " ", "")
        merged = pe.merge_enhanced_prompt(ORIGINAL, reply)
        self.assertIn("[Shot 2] At 00:03.450, " + pe.CONTINUATION + " <Subject 1> still", merged)
        self.assertIn(pe.CONTINUOUS_PERFORMANCE + "\n\n[Shot 1] A wide-angle", merged)

    def test_old_shot_one_rule_moves_to_detailed_description_without_duplicates(self):
        old = ORIGINAL.replace("[Shot 1] ", "[Shot 1] " + pe.CONTINUOUS_PERFORMANCE + " ", 1)
        reply = GOOD_REPLY.replace("[Shot 1] ", "[Shot 1] " + pe.CONTINUOUS_PERFORMANCE + " ", 1)
        merged = pe.merge_enhanced_prompt(old, reply)
        prefix, shots, _ = pe.split_prompt(merged)
        self.assertTrue(prefix.startswith(HEAD))
        self.assertTrue(prefix.endswith(pe.CONTINUOUS_PERFORMANCE + "\n\n"))
        self.assertNotIn(pe.CONTINUOUS_PERFORMANCE, shots[0])
        self.assertEqual(merged.count(pe.CONTINUOUS_PERFORMANCE), 1)
        self.assertEqual(pe.merge_enhanced_prompt(merged, merged), merged)

    def test_repeated_music_video_wording_is_cleaned_in_preserved_style_line(self):
        old = ORIGINAL.replace("photorealistic cinematic", "1990's grunge music video")
        merged = pe.merge_enhanced_prompt(old, GOOD_REPLY)
        self.assertIn("The target video is in a 1990's grunge music-video style.", merged)
        self.assertNotIn("music video music-video", merged)

    def test_repeated_enhancement_does_not_duplicate_continuity_text(self):
        once = pe.merge_enhanced_prompt(ORIGINAL, GOOD_REPLY)
        twice = pe.merge_enhanced_prompt(once, once)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count(pe.CONTINUOUS_PERFORMANCE), 1)
        self.assertEqual(twice.count(pe.CONTINUATION), 1)

    def test_all_later_shots_get_the_opening_even_without_timestamps(self):
        original = SHOTS + "\n\n[Shot 3] <Subject 1> keeps singing beside the same table."
        reply = GOOD_REPLY + "\n\n[Shot 3] Continuing directly from the previous shot’s final moment, <Subject 1> keeps singing beside the same table."
        merged = pe.merge_enhanced_prompt(original, reply)
        self.assertIn("[Shot 3] " + pe.CONTINUATION + " <Subject 1>", merged)
        self.assertEqual(merged.count(pe.CONTINUATION), 2)

    def test_single_shot_does_not_get_multi_shot_boilerplate(self):
        old = SHOTS.split("\n\n")[0]
        new = GOOD_REPLY.split("\n\n")[0]
        self.assertEqual(pe.merge_enhanced_prompt(old, new), new)

    def test_empty_shot_cannot_be_hidden_by_continuity_boilerplate(self):
        with self.assertRaises(pe.PromptEnhanceError):
            pe.merge_enhanced_prompt(ORIGINAL, "[Shot 1] \n\n[Shot 2] tiny.")

    def test_continuity_additions_still_respect_character_limit(self):
        with patch.object(pe, "HARD_LIMIT", len(ORIGINAL)):
            with self.assertRaises(pe.PromptEnhanceError):
                pe.merge_enhanced_prompt(ORIGINAL, GOOD_REPLY)

    def test_saved_instructions_receive_continuity_contract_in_shared_backend(self):
        # Exercise the endpoint function without loading ComfyUI or an actual LLM.
        source = ast.parse((ROOT / "llm" / "video_prompt_generation.py").read_text(encoding="utf-8"))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_enhance_minimax_h3_prompt")
        defaults = instructions.MINIMAX_H3_PROMPT_ENHANCE_INSTRUCTIONS
        contract = minimax_instructions.MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT
        for configured in ("Open every shot with a new action and end in a held pose.", defaults):
            with self.subTest(custom=configured != defaults):
                calls = []
                def runner(payload, prompt, **kwargs):
                    calls.append(prompt)
                    return GOOD_REPLY, {}
                namespace = {
                    "MINIMAX_H3_PROMPT_ENHANCE_INSTRUCTIONS": defaults,
                    "MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT": contract,
                    "_effective_builder_instruction": lambda *args: configured,
                    "_run_builder_text_llm": runner,
                    "merge_enhanced_prompt": pe.merge_enhanced_prompt,
                }
                exec(compile(ast.Module(body=[function], type_ignores=[]), "enhance_endpoint", "exec"), namespace)
                result = namespace[function.name]({"current_prompt": ORIGINAL})
                self.assertEqual(calls[0].count(contract), 1)
                self.assertIn(pe.CONTINUOUS_PERFORMANCE + "\n\n[Shot 1]", result["prompt"])

    def test_instructions_are_an_editable_builder_instruction(self):
        self.assertIn("minimax_h3_prompt_enhance", instructions._BUILDER_INSTRUCTION_DEFAULTS)
        self.assertIn("minimax_h3_prompt_enhance", instructions._BUILDER_INSTRUCTION_LABELS)


if __name__ == "__main__":
    unittest.main()
