import unittest
from pathlib import Path


BUILDER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "VRGDG_MusicVideoBuilderUI.js"
).read_text(encoding="utf-8")


DELETE_ALL_START = BUILDER_SOURCE.index("async function deleteAllTimelineVideos()")
DELETE_ALL_END = BUILDER_SOURCE.index(
    "async function deleteAllTimelineImages()", DELETE_ALL_START
)
DELETE_ALL_SOURCE = BUILDER_SOURCE[DELETE_ALL_START:DELETE_ALL_END]


class BuilderDeleteAllVideosTests(unittest.TestCase):
    def test_delete_all_videos_button_is_wired(self):
        self.assertIn('makeButton("Delete ALL Videos")', BUILDER_SOURCE)
        self.assertIn(
            "deleteAllTimelineVideosButton.onclick = deleteAllTimelineVideos",
            BUILDER_SOURCE,
        )

    def test_action_clears_timeline_video_assignments_and_history(self):
        self.assertIn("const segments = allEditableSegments()", DELETE_ALL_SOURCE)
        for line in (
            'segment.video_path = ""',
            "segment.video_history = []",
            "segment.video_thumbnail_history = []",
            "segment.video_backup_paths = []",
            "segment.video_history_index = -1",
            "segment.video_output = null",
            'segment.video_status = "none"',
        ):
            self.assertIn(line, DELETE_ALL_SOURCE)

    def test_action_deletes_every_video_and_thumbnail_file(self):
        self.assertIn("/vrgdg/music_builder/delete_project_media", DELETE_ALL_SOURCE)
        self.assertIn("for (const path of [...videoPaths, ...thumbnailPaths])", DELETE_ALL_SOURCE)
        self.assertIn("Permanently delete ALL", DELETE_ALL_SOURCE)
        self.assertIn("cannot be undone", DELETE_ALL_SOURCE.lower())

    def test_action_is_undoable_and_saved(self):
        self.assertIn("pushHistory()", DELETE_ALL_SOURCE)
        self.assertIn(
            'autoSaveSessionQuiet("all timeline videos removed")',
            DELETE_ALL_SOURCE,
        )

    def test_action_does_not_clear_images_or_prompts(self):
        self.assertNotIn("segment.image_history = []", DELETE_ALL_SOURCE)
        self.assertNotIn('segment.i2v_prompt = ""', DELETE_ALL_SOURCE)
        self.assertNotIn('segment.minimax_h3_prompt = ""', DELETE_ALL_SOURCE)


if __name__ == "__main__":
    unittest.main()
