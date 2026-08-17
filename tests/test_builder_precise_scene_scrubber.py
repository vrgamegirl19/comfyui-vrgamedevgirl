import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
EDITOR_SOURCE = (ROOT / "web" / "VRGDG_VideoEditorUI.js").read_text(encoding="utf-8")


class BuilderPreciseSceneScrubberTests(unittest.TestCase):
    def test_music_video_builder_has_scene_and_global_scrub_details(self):
        self.assertIn("globalScrubSceneDetail", BUILDER_SOURCE)
        self.assertIn("globalScrubGlobalDetail", BUILDER_SOURCE)
        self.assertIn("🎬 ${sceneLabel}: ${localSecs.toFixed(2)}s / ${segDur.toFixed(2)}s (${localSecs.toFixed(2)}s in scene)", BUILDER_SOURCE)
        self.assertIn("Global: ${formatTime(current)} / ${formatTime(maxTime)}", BUILDER_SOURCE)

    def test_video_editor_has_scene_and_global_scrub_details(self):
        self.assertIn("globalScrubSceneDetail", EDITOR_SOURCE)
        self.assertIn("globalScrubGlobalDetail", EDITOR_SOURCE)
        self.assertIn("🎬 ${clipName}: ${localSecs.toFixed(2)}s / ${clipDur.toFixed(2)}s (${localSecs.toFixed(2)}s in scene)", EDITOR_SOURCE)
        self.assertIn("Global: ${formatTime(absoluteTime)} / ${formatTime(state.totalDuration)}", EDITOR_SOURCE)


if __name__ == "__main__":
    unittest.main()
