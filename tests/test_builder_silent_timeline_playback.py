import unittest
from pathlib import Path

from builder_source import function_source, read_builder_source


BUILDER_SOURCE = read_builder_source()


def between(start_text, end_text):
    start = BUILDER_SOURCE.index(start_text)
    end = BUILDER_SOURCE.index(end_text, start)
    return BUILDER_SOURCE[start:end]


class BuilderSilentTimelinePlaybackTests(unittest.TestCase):
    def test_silent_clock_advances_with_animation_frames(self):
        source = function_source(BUILDER_SOURCE, "startSilentTimelinePlayback")
        self.assertIn("performance.now()", source)
        self.assertIn("window.requestAnimationFrame(tick)", source)
        self.assertIn("state.sceneAudioGlobalTime = current", source)
        self.assertIn("updateAudioScrubbers()", source)

    def test_silent_clock_counts_as_timeline_playback(self):
        source = function_source(BUILDER_SOURCE, "isTimelinePlaying")
        self.assertIn("silentTimeline.playing", source)

    def test_no_audio_play_falls_back_to_silent_clock(self):
        source = between(
            "playButton.onclick = async () => {",
            "multiSelectButton.onclick",
        )
        self.assertIn("if (!ensureGlobalTimelineAudioSource(startTime))", source)
        self.assertIn("startSilentTimelinePlayback(startTime)", source)
        self.assertNotIn("Load audio first, or add custom audio to scenes.", source)

    def test_unplayable_audio_also_falls_back_to_silent_clock(self):
        self.assertIn(
            "audio.play().then(updatePlayPauseButton).catch(() => startSilentTimelinePlayback(startTime))",
            BUILDER_SOURCE,
        )
        scene_source = function_source(BUILDER_SOURCE, "playSceneAudioFrom")
        self.assertIn("sceneAudio.play().catch(() =>", scene_source)
        self.assertIn("startSilentTimelinePlayback(state.sceneAudioGlobalTime)", scene_source)

    def test_no_audio_scrubbing_uses_virtual_timeline_time(self):
        source = function_source(BUILDER_SOURCE, "currentGlobalTime")
        self.assertIn("if (!currentProjectAudioPath())", source)
        self.assertIn("state.sceneAudioGlobalTime", source)

    def test_pause_stops_the_silent_clock(self):
        source = function_source(BUILDER_SOURCE, "pauseAllAudio")
        self.assertIn("stopSilentTimelinePlayback()", source)


if __name__ == "__main__":
    unittest.main()
