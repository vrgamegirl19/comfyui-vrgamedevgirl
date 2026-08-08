import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)


class BuilderSceneRenderTimeoutTests(unittest.TestCase):
    def test_settings_menu_exposes_persistent_one_to_twenty_four_hour_limit(self):
        for expected in (
            "const DEFAULT_SCENE_RENDER_WAIT_HOURS = 2;",
            "const MAX_SCENE_RENDER_WAIT_HOURS = 24;",
            'makeField("Scene render wait limit (hours)", sceneRenderWaitHoursInput)',
            'sceneRenderWaitHoursInput.min = "1";',
            'sceneRenderWaitHoursInput.max = String(MAX_SCENE_RENDER_WAIT_HOURS);',
            "scene_render_wait_hours: normalizeSceneRenderWaitHours(state.sceneRenderWaitHours)",
            "data.session.scene_render_wait_hours ?? state.sceneRenderWaitHours",
            "session.scene_render_wait_hours ?? state.sceneRenderWaitHours",
        ):
            self.assertIn(expected, BUILDER_SOURCE)

    def test_video_waiter_uses_configured_limit_for_ltx_and_minimax(self):
        self.assertIn(
            "const timeoutMs = timeoutHours * 60 * 60 * 1000;",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "while (Date.now() - started < timeoutMs)",
            BUILDER_SOURCE,
        )
        self.assertGreaterEqual(
            BUILDER_SOURCE.count("timeoutHours: state.sceneRenderWaitHours"),
            2,
        )
        self.assertGreaterEqual(
            BUILDER_SOURCE.count("waitHours: state.sceneRenderWaitHours"),
            2,
        )
        self.assertNotIn(
            "while (Date.now() - started < 2 * 60 * 60 * 1000)",
            BUILDER_SOURCE,
        )

    def test_timeout_message_reports_the_selected_limit(self):
        self.assertIn(
            "did not finish before the ${sceneRenderWaitLabel(waitHours)} wait limit",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "This does not cancel ComfyUI's render.",
            BUILDER_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
