import unittest
from pathlib import Path


BUILDER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "VRGDG_MusicVideoBuilderUI.js"
).read_text(encoding="utf-8")


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
        self.assertIn('audio_mode: "input_audio"', BUILDER_SOURCE)
        self.assertIn(
            "The existing global Audio 1 file remains the only audio source and must stay completely unchanged.",
            BUILDER_SOURCE,
        )

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
        converter_start = BUILDER_SOURCE.index(
            "async function convertAllLtxVideoPromptsToMiniMaxH3()"
        )
        converter_end = BUILDER_SOURCE.index(
            "async function createI2VPromptWithGemma()", converter_start
        )
        converter = BUILDER_SOURCE[converter_start:converter_end]
        self.assertEqual(converter.count("pushHistory();"), 1)
        self.assertLess(
            converter.index("pushHistory();"),
            converter.index("segment.minimax_h3_prompt = prompt"),
        )


if __name__ == "__main__":
    unittest.main()
