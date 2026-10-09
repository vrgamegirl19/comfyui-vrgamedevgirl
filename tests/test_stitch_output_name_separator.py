"""A repeated stitch with the same output prefix gets a separated collision suffix.

Live report: re-stitching the S07-S08 preview on Osiris wrote PREVIEW_SCENES_007-0082.mp4 - the collision
suffix "2" was glued to the scene range, so it reads as scenes 007-0082. The repo's convention for a free
name (storyboard references, runner/paths.py _unique_copy_path) is ``<stem>_<n>`` from 2, so the second
stitch must be PREVIEW_SCENES_007-008_2.mp4.
"""

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
video_files = importlib.import_module(f"{pkg_name}.runner.video_files")
video_orch = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.video_orchestrator")
project_copy = importlib.import_module(f"{pkg_name}.builder.project_copy")


def have_ffmpeg():
    try:
        for tool in ("ffmpeg", "ffprobe"):
            subprocess.run([tool, "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


class UniqueFinalVideoPathTests(unittest.TestCase):
    def setUp(self):
        self.project = tempfile.mkdtemp(prefix="vrgdg_names_")

    def tearDown(self):
        shutil.rmtree(self.project, ignore_errors=True)

    def touch(self, name):
        with open(os.path.join(self.project, name), "wb") as handle:
            handle.write(b"x")

    def name(self, prefix):
        return os.path.basename(video_files._unique_final_video_path(self.project, prefix))

    def test_first_stitch_keeps_the_plain_name(self):
        self.assertEqual(self.name("PREVIEW_SCENES_007-008"), "PREVIEW_SCENES_007-008.mp4")

    def test_collision_suffix_is_separated(self):
        self.touch("PREVIEW_SCENES_007-008.mp4")
        self.assertEqual(self.name("PREVIEW_SCENES_007-008"), "PREVIEW_SCENES_007-008_2.mp4")
        self.touch("PREVIEW_SCENES_007-008_2.mp4")
        self.assertEqual(self.name("PREVIEW_SCENES_007-008"), "PREVIEW_SCENES_007-008_3.mp4")

    def test_final_video_and_slideshow_prefixes(self):
        self.touch("FINAL_VIDEO.mp4")
        self.assertEqual(self.name("FINAL_VIDEO"), "FINAL_VIDEO_2.mp4")
        self.touch("IMAGE_SLIDESHOW_PREVIEW.mp4")
        self.assertEqual(self.name("IMAGE_SLIDESHOW_PREVIEW"), "IMAGE_SLIDESHOW_PREVIEW_2.mp4")

    def test_a_name_that_already_ends_in_a_suffix_is_not_reused(self):
        # A prefix that itself ends in _2 must not collide with the first prefix's second stitch.
        self.touch("PREVIEW_SCENES_all.mp4")
        self.touch("PREVIEW_SCENES_all_2.mp4")
        self.assertEqual(self.name("PREVIEW_SCENES_all"), "PREVIEW_SCENES_all_3.mp4")
        self.assertEqual(self.name("PREVIEW_SCENES_all_2"), "PREVIEW_SCENES_all_2_2.mp4")

    def test_suffixed_outputs_are_still_listed_and_recognised(self):
        # Everything that finds exports by name: the API's final-video listing and the branch-copy filter.
        for name in ("FINAL_VIDEO.mp4", "FINAL_VIDEO_2.mp4", "PREVIEW_SCENES_007-008.mp4", "PREVIEW_SCENES_007-008_2.mp4"):
            self.touch(name)
            self.assertTrue(project_copy._is_project_root_export_video(name), name)
        with patch.object(video_orch, "resolve_project_folder", return_value=self.project):
            listed = {item["filename"] for item in video_orch.list_project_final_videos("P")}
        self.assertTrue({"FINAL_VIDEO_2.mp4", "PREVIEW_SCENES_007-008_2.mp4"} <= listed, listed)


@unittest.skipUnless(have_ffmpeg(), "ffmpeg/ffprobe not installed")
class RestitchNameTests(unittest.TestCase):
    """The real stitcher, twice with the Builder's preview prefix for S07-S08."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vrgdg_restitch_")
        self.project = os.path.join(self.tmp, "Project")
        clip_dir = os.path.join(self.project, "rendered_scene_videos")
        os.makedirs(clip_dir)
        self.clips = []
        for index, frames in ((7, 161), (8, 168)):
            clip = os.path.join(clip_dir, f"video_{index:04d}-audio.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=24",
                            "-f", "lavfi", "-i", "sine=f=440:r=48000", "-frames:v", str(frames), "-t", f"{frames / 24:.6f}",
                            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", clip],
                           capture_output=True, check=True)
            self.clips.append(clip)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stitch(self):
        return video_files._stitch_scene_videos({
            "project_folder": self.project,
            "scene_paths": self.clips,
            "audio_path": "",
            "use_embedded_scene_audio": True,
            "scene_timing_items": [{"start": 0.0, "end": 6.68}, {"start": 6.68, "end": 13.72}],
            "timeline_fps": 24,
            "output_prefix": "PREVIEW_SCENES_007-008",
        })["final_video_path"]

    def test_restitch_names(self):
        names = [os.path.basename(self.stitch()) for _ in range(3)]
        self.assertEqual(names, ["PREVIEW_SCENES_007-008.mp4", "PREVIEW_SCENES_007-008_2.mp4", "PREVIEW_SCENES_007-008_3.mp4"])


if __name__ == "__main__":
    unittest.main()
