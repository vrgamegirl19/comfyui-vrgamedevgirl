import unittest
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "VRGDG_MusicVideoBuilderUI.js"
).read_text(encoding="utf-8")
VIDEO_EDITOR_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "VRGDG_VideoEditorNodes.py"
).read_text(encoding="utf-8")


class BuilderHistoryMemoryTests(unittest.TestCase):
    def test_history_snapshots_deduplicate_embedded_media(self):
        self.assertIn("HISTORY_BLOB_TOKEN_PREFIX", SOURCE)
        self.assertIn("historySnapshotReplacer", SOURCE)
        self.assertIn("JSON.parse(snapshot, historySnapshotReviver)", SOURCE)
        self.assertIn("pruneHistoryBlobCache();", SOURCE)

    def test_inspector_text_edits_use_one_history_checkpoint_per_focus(self):
        self.assertIn('control.addEventListener("focus", pushHistory);', SOURCE)
        self.assertIn(
            'control.addEventListener("input", () => updateActiveFromInputs({ skipHistory: true }));',
            SOURCE,
        )
        self.assertIn(
            'control.addEventListener("change", () => updateActiveFromInputs({ skipHistory: true }));',
            SOURCE,
        )

    def test_scene_thumbnails_keep_stable_urls_across_renders(self):
        self.assertIn("function makeEditorThumbnailUrl(path)", SOURCE)
        self.assertIn("EDITOR_THUMBNAIL_SESSION_VERSION", SOURCE)
        self.assertIn("refreshEditorThumbnailUrl(imagePath);", SOURCE)
        self.assertIn("makeEditorThumbnailUrl(previewThumbPath)", SOURCE)
        self.assertIn("makeEditorThumbnailUrl(imagePath)", SOURCE)
        self.assertIn('request.query.get("thumbv"', VIDEO_EDITOR_SOURCE)
        self.assertIn('"Cache-Control"] = "public, max-age=31536000, immutable"', VIDEO_EDITOR_SOURCE)


if __name__ == "__main__":
    unittest.main()
