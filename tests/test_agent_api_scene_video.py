"""A rendered scene video is recorded like the Video Builder does, so the timeline shows its picture."""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

sv = importlib.import_module(f"{ROOT.name}.agent_api.scene_video")


class SceneVideoTests(unittest.TestCase):
    def test_video_thumbnail_and_history_are_set(self):
        seg = {}
        sv.apply_scene_video(seg, r"C:\p\rendered_scene_videos\video_0001-audio.mp4", r"C:\p\scene_video_thumbnails\video_0001-audio.jpg")
        self.assertEqual(seg["video_thumbnail_path"], r"C:\p\scene_video_thumbnails\video_0001-audio.jpg")
        self.assertEqual(seg["video_history"], [r"C:\p\rendered_scene_videos\video_0001-audio.mp4"])
        self.assertEqual(seg["video_thumbnail_history"], [seg["video_thumbnail_path"]])
        self.assertEqual((seg["video_history_index"], seg["video_status"], seg["preview_mode"]), (0, "done", "video"))
        self.assertTrue(seg["video_cache_bust"])

    def test_a_new_render_is_added_and_older_thumbnails_are_kept(self):
        seg = {}
        sv.apply_scene_video(seg, r"C:\p\v\a.mp4", r"C:\p\t\a.jpg")
        sv.apply_scene_video(seg, r"C:\p\v\b.mp4", r"C:\p\t\b.jpg")
        sv.apply_scene_video(seg, r"C:\p\v\b.mp4", r"C:\p\t\b.jpg")
        self.assertEqual(seg["video_history"], [r"C:\p\v\a.mp4", r"C:\p\v\b.mp4"])
        self.assertEqual(seg["video_thumbnail_history"], [r"C:\p\t\a.jpg", r"C:\p\t\b.jpg"])
        self.assertEqual(seg["video_history_index"], 1)

    def test_a_missing_thumbnail_is_found_by_the_builder_naming(self):
        with tempfile.TemporaryDirectory() as root:
            videos = os.path.join(root, "rendered_scene_videos")
            thumbs = os.path.join(root, "scene_video_thumbnails")
            os.makedirs(videos)
            os.makedirs(thumbs)
            Path(thumbs, "video_0001-audio.jpg").write_bytes(b"jpg")
            seg = {}
            sv.apply_scene_video(seg, os.path.join(videos, "video_0001-audio.mp4"))
            self.assertEqual(os.path.basename(seg["video_thumbnail_path"]), "video_0001-audio.jpg")

    def test_no_path_changes_nothing(self):
        seg = {"video_path": "keep"}
        sv.apply_scene_video(seg, "")
        self.assertEqual(seg, {"video_path": "keep"})


if __name__ == "__main__":
    unittest.main()
