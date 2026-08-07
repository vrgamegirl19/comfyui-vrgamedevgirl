import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_METADATA = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
PACKAGE_INIT = (ROOT / "__init__.py").read_text(encoding="utf-8")
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)
UPDATE_NOTES = json.loads((ROOT / "update_notes.json").read_text(encoding="utf-8"))
BUILDER_GUIDE = (
    ROOT / "Workflows" / "LTX-2_Workflows" / "Video_Builder" / "readme.md"
).read_text(encoding="utf-8")


class BuilderUpdateBannerNotesTests(unittest.TestCase):
    def test_latest_release_documents_storyboard_gemma_reliability(self):
        release = UPDATE_NOTES["releases"][0]
        self.assertEqual(
            release["id"],
            "2026-08-06-storyboard-gemma-network-reliability",
        )
        self.assertEqual(
            release["commit"],
            "84fa509b85c88b7d6a8af9b9ca5eafcadf6b8c11",
        )
        items = "\n".join(
            item
            for section in release["sections"]
            for item in section.get("items", [])
        )
        for expected in (
            "ten minutes per scene",
            "lost ComfyUI backend connection",
            "four-minute request cutoff",
            "NetworkError when attempting to fetch resource",
            "Expanded the AI Video Builder guide",
            "ComfyUI Registry version 9.1.0",
            "Manager continuing to install the July 3 version 9.0.0",
            "may not contain Git metadata",
        ):
            self.assertIn(expected, items)

    def test_registry_and_runtime_versions_match_current_release(self):
        self.assertIn('version = "9.1.0"', PACKAGE_METADATA)
        self.assertIn('__version__ = "v9.1.0"', PACKAGE_INIT)
        self.assertIn('__updated__ = "2026-08-06"', PACKAGE_INIT)

    def test_previous_release_documents_minimax_builder_controls(self):
        release = UPDATE_NOTES["releases"][1]
        self.assertEqual(
            release["id"],
            "2026-08-06-minimax-turbo-reference-llm-controls",
        )
        self.assertEqual(
            UPDATE_NOTES["releases"][2]["id"],
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

    def test_guide_covers_the_latest_builder_controls_and_recovery_tools(self):
        for expected in (
            "MiniMax-H3 Turbo Acceleration",
            "Environment inspiration only (LLM only — ignore framing)",
            "Face + hair only (keep the rest of the start frame)",
            "Cut frequency",
            "Maximum output tokens",
            "Convert LTX Video Prompts to MiniMax H3",
            "Delete ALL Videos",
            "silent timeline clock",
            "allow up to ten minutes per scene",
            "Connection to the ComfyUI backend was lost",
            "current ComfyUI Registry release is `9.1.0`",
            "Manager-installed folder",
        ):
            self.assertIn(expected, BUILDER_GUIDE)


if __name__ == "__main__":
    unittest.main()
