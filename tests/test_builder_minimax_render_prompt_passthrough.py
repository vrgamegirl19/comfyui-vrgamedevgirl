import unittest
from pathlib import Path

from builder_source import python_function_source, read_builder_source, read_runner_source


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = read_builder_source()
RUNNER_SOURCE = read_runner_source()


RENDER_START = BUILDER_SOURCE.index(
    "async function renderMiniMaxSceneVideoWithProgress"
)
RENDER_END = BUILDER_SOURCE.index(
    "async function createMiniMaxSceneVideo", RENDER_START
)
RENDER_SOURCE = BUILDER_SOURCE[RENDER_START:RENDER_END]


class BuilderMiniMaxRenderPromptPassthroughTests(unittest.TestCase):
    def test_render_uses_saved_prompt_unless_frame_continuity_generates_one(self):
        self.assertIn(
            "const generatedContinuityPrompt = miniMaxH3FrameContinuityPromptEnabled(segment)",
            RENDER_SOURCE,
        )
        self.assertIn(
            "generatedContinuityPrompt || (options.prompt ?? (segment?.minimax_h3_prompt || segment?.i2v_prompt || \"\"))",
            RENDER_SOURCE,
        )
        self.assertNotIn("applyMiniMaxH3NativeVoiceBlock", RENDER_SOURCE)
        self.assertNotIn("applyMiniMaxH3ContinuityPromptBlock", RENDER_SOURCE)

    def test_same_prompt_is_shown_and_sent(self):
        self.assertIn("progress?.setSceneDetails?.({", RENDER_SOURCE)
        self.assertIn("prompt,", RENDER_SOURCE)
        payload_start = RENDER_SOURCE.index("const payload = {")
        payload_end = RENDER_SOURCE.index("};", payload_start)
        self.assertIn("prompt,", RENDER_SOURCE[payload_start:payload_end])
        self.assertIn("pass2_prompt:", RENDER_SOURCE[payload_start:payload_end])

    def test_workflow_runner_writes_the_complete_payload_string_to_h3(self):
        source = python_function_source(
            RUNNER_SOURCE,
            "_build_minimax_h3_api_prompt",
            "_build_minimax_h3_2pass_api_prompt",
            "_build_minimax_h3_advanced_2pass_api_prompt",
            "_save_minimax_h3_advanced_2pass_debug_workflow",
            "_remap_api_prompt_references",
            "_prune_api_prompt_to_roots",
            "_build_minimax_h3_3pass_api_prompt",
        )
        self.assertIn(
            '_set_api_input(prompt, "138", "value", video_prompt)',
            source,
        )
        self.assertNotIn("video_prompt[:", source)


if __name__ == "__main__":
    unittest.main()
