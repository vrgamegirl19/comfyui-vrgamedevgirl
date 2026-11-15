import json
import re
import unittest
from pathlib import Path

from builder_source import read_builder_source


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_METADATA = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
PACKAGE_INIT = (ROOT / "__init__.py").read_text(encoding="utf-8")
BUILDER_SOURCE = read_builder_source()
UPDATE_NOTES = json.loads((ROOT / "update_notes.json").read_text(encoding="utf-8"))
BUILDER_GUIDE = (
    ROOT / "Workflows" / "LTX-2_Workflows" / "Video_Builder" / "readme.md"
).read_text(encoding="utf-8")


class BuilderUpdateBannerNotesTests(unittest.TestCase):
    # update_notes.json holds the last five builds, newest first. When a build ships, add its notes at the top and drop the oldest past five.
    def test_notes_keep_the_last_five_builds_newest_first(self):
        releases = UPDATE_NOTES["releases"]
        self.assertEqual(UPDATE_NOTES["schema_version"], 1)
        self.assertEqual(UPDATE_NOTES["product"], "AI Video Builder")
        self.assertTrue(1 <= len(releases) <= 5, f"{len(releases)} releases, the file keeps at most five")
        dates = [release["date"] for release in releases]
        self.assertEqual(dates, sorted(dates, reverse=True), "newest first")
        ids = [release["id"] for release in releases]
        self.assertEqual(len(ids), len(set(ids)), "ids are unique")

    def test_every_release_has_what_the_banner_shows(self):
        for release in UPDATE_NOTES["releases"]:
            with self.subTest(release=release.get("id")):
                for key in ("id", "version", "date", "title", "commit", "guide_url"):
                    self.assertTrue(str(release.get(key) or "").strip(), key)
                self.assertRegex(release["commit"], r"^[0-9a-f]{7,40}$")
                self.assertIsInstance(release["restart_required"], bool)
                self.assertTrue(release["sections"])
                for section in release["sections"]:
                    self.assertTrue(section["title"].strip())
                    self.assertTrue(section["items"])
                    for item in section["items"]:
                        self.assertTrue(item.strip())

    def test_latest_release_is_the_version_the_pack_reports(self):
        latest = UPDATE_NOTES["releases"][0]
        version = re.search(r'__version__ = "v([^"]+)"', PACKAGE_INIT).group(1)
        updated = re.search(r'__updated__ = "([^"]+)"', PACKAGE_INIT).group(1)
        self.assertIn(f'version = "{version}"', PACKAGE_METADATA)
        self.assertEqual(latest["version"], version)
        self.assertEqual(latest["date"], updated)

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
            "current ComfyUI Registry release is `9.1.1`",
            "Manager-installed folder",
            "Scene render wait limit (hours)",
        ):
            self.assertIn(expected, BUILDER_GUIDE)


if __name__ == "__main__":
    unittest.main()
