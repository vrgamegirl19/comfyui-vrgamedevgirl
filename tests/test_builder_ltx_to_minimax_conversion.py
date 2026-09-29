import unittest
from pathlib import Path

from builder_source import function_source, read_builder_source


BUILDER_SOURCE = read_builder_source()
CONVERTER = function_source(BUILDER_SOURCE, "convertAllLtxVideoPromptsToMiniMaxH3")


class BuilderLtxToMiniMaxConversionTests(unittest.TestCase):
    def test_converter_is_a_single_tools_panel_action(self):
        self.assertIn(
            'makeButton("Convert LTX Video Prompts to MiniMax H3", "primary")',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "convertLtxPromptsToMiniMaxButton.onclick = convertAllLtxVideoPromptsToMiniMaxH3",
            BUILDER_SOURCE,
        )

    def test_converter_reads_ltx_prompts_and_writes_separate_minimax_prompts(self):
        self.assertIn(
            'ltxPrompt: String(segment?.i2v_prompt || "").trim()',
            BUILDER_SOURCE,
        )
        self.assertIn("segment.minimax_h3_prompt = prompt", BUILDER_SOURCE)
        self.assertNotIn("segment.i2v_prompt = prompt;\n        converted += 1", BUILDER_SOURCE)

    def test_converter_keeps_global_audio_as_input_audio(self):
        self.assertIn('audioMode: "input_audio",', CONVERTER)

    def test_converter_is_limited_to_music_video_projects(self):
        self.assertIn(
            'if (normalizeVideoType(state.videoType) === "speaking")',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "This converter is for music-video projects, not speaking / short-film projects.",
            BUILDER_SOURCE,
        )

    def test_converter_creates_one_undo_checkpoint_before_first_write(self):
        self.assertEqual(CONVERTER.count("pushHistory();"), 1)
        self.assertLess(
            CONVERTER.index("pushHistory();"),
            CONVERTER.index("segment.minimax_h3_prompt = prompt"),
        )


if __name__ == "__main__":
    unittest.main()
