import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)
RUNNER_SOURCE = (ROOT / "VRGDG_WorkflowRunnerNodes.py").read_text(encoding="utf-8")


RENDER_START = BUILDER_SOURCE.index(
    "async function renderMiniMaxSceneVideoWithProgress"
)
RENDER_END = BUILDER_SOURCE.index(
    "async function createMiniMaxSceneVideo", RENDER_START
)
RENDER_SOURCE = BUILDER_SOURCE[RENDER_START:RENDER_END]


class BuilderMiniMaxRenderPromptPassthroughTests(unittest.TestCase):
    def test_render_uses_the_saved_prompt_without_rewriting_it(self):
        self.assertIn(
            "const prompt = String(\n      options.prompt\n      ?? (segment?.minimax_h3_prompt || segment?.i2v_prompt || \"\")\n    ).trim();",
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

    def test_workflow_runner_writes_the_complete_payload_string_to_h3(self):
        start = RUNNER_SOURCE.index("def _build_minimax_h3_api_prompt")
        end = RUNNER_SOURCE.index("def _build_i2v_api_prompt", start)
        source = RUNNER_SOURCE[start:end]
        self.assertIn(
            '_set_api_input(prompt, "138", "value", video_prompt)',
            source,
        )
        self.assertNotIn("video_prompt[:", source)


if __name__ == "__main__":
    unittest.main()
