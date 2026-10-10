"""Lyric-aware emotion input, prompt preservation, and Python/UI parity."""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
emotion = importlib.import_module(f"{ROOT.name}.llm.prompts.emotion_expression")
lfp = importlib.import_module(f"{ROOT.name}.minimax.lyric_free_performance")
shots = importlib.import_module(f"{ROOT.name}.minimax.shot_prompt")
assembly = importlib.import_module(f"{ROOT.name}.minimax.prompt_assembly")
generation = importlib.import_module(f"{ROOT.name}.llm.video_prompt_generation")


class EmotionExpressionTests(unittest.TestCase):
    def test_native_speech_uses_inline_delivery_tags_and_emotion_headers(self):
        instruction = emotion.emotion_expression_instruction({"performance_mode": "speaking", "audio_mode": "built_in_audio",
                                                              "facial_performance": "custom", "facial_performance_custom": "Curious"})
        self.assertIn("<d>[English, curious]", instruction)
        self.assertIn("<pause>", instruction)
        self.assertIn("<breath>", instruction)
        self.assertIn("INLINE DELIVERY TAGS ARE REQUIRED", instruction)
        self.assertIn("An emotion header alone is incomplete", instruction)
        self.assertIn("<i>...</i>", instruction)
        self.assertIn("Preserve the supplied spoken words and their order", instruction)
        self.assertIn("BUILT-IN H3 SPEECH DELIVERY", emotion.emotion_expression_instruction(
            {"performance_mode": "speaking", "audio_mode": "built_in_audio"}))
        self.assertNotIn("BUILT-IN H3 SPEECH DELIVERY", emotion.emotion_expression_instruction(
            {"performance_mode": "speaking", "audio_mode": "input_audio", "emotion_expression_tags": "Curious"}))

    def test_inline_delivery_tags_do_not_duplicate_supplied_dialogue(self):
        description = "<Subject 1> says <d>[English, curious] <breath> I was <i>not</i> expecting that. <pause></d>"
        self.assertEqual(shots.ensure_quoted_lyrics([description], "I was not expecting that."), [description])

    def test_defaults_custom_and_explicit_override(self):
        defaults = {"default_facial_performance": "custom", "default_facial_performance_custom": "Angry"}
        value = emotion.emotion_expression_input({}, defaults)
        self.assertEqual(value["facial_performance_custom"], "Angry")
        self.assertTrue(emotion.has_emotion_expression_input(value))
        self.assertFalse(emotion.has_emotion_expression_input({**value, "facial_performance": "off"}))
        self.assertTrue(emotion.has_emotion_expression_input({**value, "facial_performance": "off",
                                                            "emotion_expression_tags": "Sad"}))

    def test_instruction_interprets_lyrics_and_progression_without_changing_audio(self):
        instruction = emotion.emotion_expression_instruction({"emotion_expression_tags": "Start happy, then end sad",
                                                              "lyric_text": "We lost the home we built"})
        self.assertIn("We lost the home we built", instruction)
        self.assertIn("never flatten", instruction)
        self.assertIn("Explicit scene emotion input takes priority", instruction)
        self.assertIn("never singing tags", instruction)
        self.assertIn("do not request changes to the vocal delivery", instruction)
        self.assertEqual(emotion.emotion_expression_instruction({}), "")

    def test_assembly_preserves_llm_acting_but_removes_instrumental_singing(self):
        scene = {"emotion_expression_tags": "Angry", "lyric_text": "I gave you everything",
                 "lyric_performance_mode": "cue_map", "lyric_cue_map": [
                     {"type": "instrumental", "start": 0, "end": 2},
                     {"type": "vocal", "start": 2, "end": 4, "text": "I gave you everything"}]}
        plan = {"exact_duration_seconds": 4, "cut_times_seconds": [2]}
        result = lfp.apply_shots(["The camera moves left. He sings [angry, singing].",
                                 'He sings [angry, singing] "I gave you everything" with narrowed eyes, his jaw moves. He grips the railing.'],
                                scene, plan)
        self.assertNotIn("sing", result[0])
        self.assertIn("[angry, singing] with narrowed eyes", result[1])
        self.assertIn("grips the railing", result[1])
        self.assertIn("during 2s–4s", result[1])
        self.assertNotRegex(" ".join(result), r"jaw|mouth|lips|I gave you everything|with passion")

    def test_tagged_lyrics_are_not_quoted_or_duplicated(self):
        description = "He sings with narrowed eyes. <d>[English, angry, singing] I gave you everything!</d>"
        self.assertEqual(shots.ensure_quoted_lyrics([description], "I gave you everything"), [description])

    def test_headless_context_keeps_lyrics_for_interpretation(self):
        scene = {"start": 0, "end": 4, "lyric_text": "I gave you everything", "emotion_expression_tags": "Angry"}
        context = assembly.build_minimax_prompt_context(scene, {"omit_lyrics_from_video_prompts": True,
                                                               "video_type": "singing"})
        self.assertEqual(context["lyric_text"], scene["lyric_text"])
        self.assertIn("Scene emotion/expression input: Angry", context["instruction_text"])

    def test_actual_llm_request_keeps_lyrics_when_omitted_from_output(self):
        captured = []

        def fake(payload, instruction, **kwargs):
            captured.append(instruction)
            return '{"shots":[{"description":"He sings [angry, singing] with narrowed eyes."}]}', {}

        payload = {"text_gemma_runner": "lm_studio", "builder_instruction_key": "minimax_h3_text_to_video",
                   "t2i_prompt": "MiniMax H3 shot-description task. Return shot JSON.",
                   "lyric_text": "I gave you everything", "emotion_expression_tags": "Angry",
                   "omit_lyrics_from_video_prompts": True, "performance_mode": "singing", "audio_mode": "input_audio"}
        with patch.object(generation, "_run_builder_text_llm", fake), \
                patch.object(generation, "_repair_and_validate_builder_gemma_prompt", lambda p, t, l: t):
            generation._generate_builder_t2v_prompt(payload)
        self.assertIn("I gave you everything", captured[0])
        self.assertIn("Scene emotion/expression input: Angry", captured[0])
        self.assertIn("omit these words from the final prompt", captured[0])

    def test_actual_native_speech_request_includes_inline_tag_policy(self):
        captured = []

        def fake(payload, instruction, **kwargs):
            captured.append(instruction)
            return '{"shots":[{"description":"<Subject 1> says <d>[English, curious] <breath> I was <i>not</i> expecting that.</d>"}]}', {}

        payload = {"text_gemma_runner": "lm_studio", "builder_instruction_key": "minimax_h3_text_to_video",
                   "t2i_prompt": "MiniMax H3 shot-description task. Return shot JSON.",
                   "lyric_text": "I was not expecting that.", "facial_performance": "custom",
                   "facial_performance_custom": "Curious / inquisitive", "performance_mode": "speaking",
                   "audio_mode": "built_in_audio", "omit_lyrics_from_video_prompts": True}
        with patch.object(generation, "_run_builder_text_llm", fake), \
                patch.object(generation, "_repair_and_validate_builder_gemma_prompt", lambda p, t, l: t):
            generation._generate_builder_t2v_prompt(payload)
        self.assertIn("BUILT-IN H3 SPEECH DELIVERY", captured[0])
        self.assertIn("INLINE DELIVERY TAGS ARE REQUIRED", captured[0])
        self.assertIn("Curious / inquisitive", captured[0])
        self.assertIn("<i>...</i>", captured[0])
        self.assertNotIn("LYRIC-FREE CUSTOM AUDIO", captured[0])


if __name__ == "__main__":
    unittest.main()
