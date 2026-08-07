import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
STORYBOARD_SOURCE = (ROOT / "web" / "VRGDG_StoryboardBuilderUI.js").read_text(encoding="utf-8")


class StoryboardGemmaNetworkTimeoutTests(unittest.TestCase):
    def test_video_builder_storyboard_batch_uses_long_gemma_timeout(self):
        batch = re.search(
            r"const runWizardStoryboardGemmaAll = async \(\) => \{(?P<body>.*?)\n    \};",
            BUILDER_SOURCE,
            re.DOTALL,
        )
        self.assertIsNotNone(batch)
        self.assertIn(
            'postJson("/vrgdg/storyboard/gemma_video_prompt"',
            batch.group("body"),
        )
        self.assertIn("GEMMA_VIDEO_PROMPT_TIMEOUT_MS", batch.group("body"))
        self.assertNotIn("}, 240000);", batch.group("body"))

    def test_storyboard_prompt_requests_use_ten_minute_timeout(self):
        self.assertIn("const STORYBOARD_GEMMA_TIMEOUT_MS = 600000;", STORYBOARD_SOURCE)
        for endpoint in ("gemma_image_prompt", "gemma_video_prompt"):
            request = re.search(
                rf'postJson\("/vrgdg/storyboard/{endpoint}".*?STORYBOARD_GEMMA_TIMEOUT_MS\)',
                STORYBOARD_SOURCE,
                re.DOTALL,
            )
            self.assertIsNotNone(request, endpoint)

    def test_fetch_wrappers_translate_timeout_and_network_failures(self):
        for source in (BUILDER_SOURCE, STORYBOARD_SOURCE):
            self.assertIn("timedOut || controller.signal.aborted", source)
            self.assertIn("NetworkError|Failed to fetch|fetch resource|Load failed", source)
            self.assertIn("Connection to the ComfyUI backend was lost", source)


if __name__ == "__main__":
    unittest.main()
