import unittest
from pathlib import Path


BUILDER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "VRGDG_MusicVideoBuilderUI.js"
).read_text(encoding="utf-8")


def function_source(name, next_name):
    start = BUILDER_SOURCE.index(f"function {name}")
    end = BUILDER_SOURCE.index(f"function {next_name}", start)
    return BUILDER_SOURCE[start:end]


class BuilderMiniMaxBuiltInAudioTrimTests(unittest.TestCase):
    def test_rendered_minimax_built_in_audio_is_a_timeline_playback_source(self):
        source = function_source(
            "segmentUsesRenderedTimelineAudio", "usingRenderedSceneAudioMode"
        )
        self.assertIn('normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"', source)
        self.assertIn('audio_mode === "built_in_audio"', source)
        self.assertIn("selectedSegmentVideoPath(segment)", source)

    def test_rendered_scene_media_starts_at_local_zero(self):
        source = function_source(
            "timelineAudioSourceStartForSegment", "timelineAudioDurationForSegment"
        )
        self.assertIn("segmentUsesRenderedTimelineAudio(segment)", source)
        self.assertIn("return 0;", source)

    def test_scrubbing_without_global_audio_uses_scene_timeline_time(self):
        source = function_source("setGlobalPlaybackTime", "playSceneAudioFrom")
        self.assertIn("if (usingSceneAudioPlaybackMode())", source)
        self.assertIn("state.sceneSelectionUsesGlobalAudio = false", source)
        self.assertIn("state.sceneAudioGlobalTime = time", source)

    def test_right_clicking_scissors_offers_before_or_after_trim(self):
        self.assertIn("splitSceneButton.oncontextmenu", BUILDER_SOURCE)
        source = function_source(
            "chooseRenderedSceneTrimAtPlayhead", "trimBaseSceneVideoAtPlayhead"
        )
        self.assertIn('value: "before"', source)
        self.assertIn('label: "Remove before playhead"', source)
        self.assertIn('value: "after"', source)
        self.assertIn('label: "Remove after playhead"', source)
        self.assertIn(
            'choice === "before" ? "left" : "right"',
            source,
        )

    def test_left_clicking_scissors_routes_rendered_minimax_to_trim(self):
        start = BUILDER_SOURCE.index("async function splitActiveSceneAtPlayhead()")
        end = BUILDER_SOURCE.index("function setTimelineRangePoint", start)
        source = BUILDER_SOURCE[start:end]
        self.assertIn("if (baseSceneVideoTrimKind(segment))", source)
        self.assertIn("await chooseRenderedSceneTrimAtPlayhead()", source)
        self.assertLess(
            source.index("await chooseRenderedSceneTrimAtPlayhead()"),
            source.index("cannot be split"),
        )


if __name__ == "__main__":
    unittest.main()
