import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)
UPDATE_NOTES = json.loads((ROOT / "update_notes.json").read_text(encoding="utf-8"))


class BuilderUpdateBannerNotesTests(unittest.TestCase):
    def test_latest_release_documents_the_workspace_updates(self):
        release = UPDATE_NOTES["releases"][0]
        self.assertEqual(
            release["id"],
            "2026-08-06-minimax-turbo-reference-llm-controls",
        )
        self.assertEqual(
            UPDATE_NOTES["releases"][1]["id"],
            "2026-08-06-timeline-editing-minimax-prompt-reliability",
        )
        items = "\n".join(
            item
            for section in release["sections"]
            for item in section.get("items", [])
        )
        for expected in (
            "6 editable steps",
            "automatically bypasses EasyCache",
            "environment inspiration",
            "Face + hair only",
            "Cut frequency",
            "maximum-output controls",
            "AI Video Builder product name",
            "3-versus-2 AdaLN tensor mismatch",
            "Video Wizard button",
        ):
            self.assertIn(expected, items)

    def test_banner_uses_the_ai_video_builder_name(self):
        self.assertNotIn("LTX 2.3 Video Builder", BUILDER_SOURCE)
        self.assertIn(
            'updateStatusText.textContent = "AI Video Builder — Checking for updates…"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'heading.textContent = "What\'s New in AI Video Builder"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            '"Dismiss AI Video Builder version status"',
            BUILDER_SOURCE,
        )

    def test_topbar_has_live_project_video_engine_badge(self):
        self.assertIn(
            'projectVideoEngineBadge.textContent = miniMaxProject ? "◈ MiniMax" : "◈ LTX";',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'projectVideoEngineBadge.dataset.engine = miniMaxProject ? "minimax_h3" : "ltx";',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "utilityActions.append(projectVideoEngineBadge, stopWorkflowButton",
            BUILDER_SOURCE,
        )

    def test_uncommitted_local_release_is_visible_in_current_view(self):
        self.assertIn(
            '!String(release.commit || "").trim()',
            BUILDER_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
