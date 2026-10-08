"""Tests for per-take scene latents: each render keeps its own latent and the selected take's latent is the active one."""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

takes = importlib.import_module(f"{ROOT.name}.minimax.latent_takes")


class LatentTakesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = self._tmp.name
        self.latents = os.path.join(self.project, "latents")
        self.videos = os.path.join(self.project, "rendered_scene_videos")
        self.backups = os.path.join(self.project, "rendered_scene_videos_backup", "scene_0021")
        for folder in (self.latents, self.videos, self.backups):
            os.makedirs(folder)

    def _render(self, scene, latent_bytes, video_bytes, backup_previous=True):
        """What one render does: save the latent, then collect the video (the old one moves to the backups)."""
        active = os.path.join(self.latents, f"scene_{scene:03d}.latent")
        with open(active, "wb") as handle:
            handle.write(latent_bytes)
        with open(active + ".json", "w", encoding="utf-8") as handle:
            handle.write("{}")
        takes.archive_latent(self.latents, scene, active)
        video = os.path.join(self.videos, f"video_{scene:04d}-audio.mp4")
        if os.path.exists(video) and backup_previous:
            shutil.move(video, os.path.join(self.backups, f"video_{scene:04d}-audio_{len(os.listdir(self.backups))}.mp4"))
        with open(video, "wb") as handle:
            handle.write(video_bytes)
        takes.attach_video(self.project, scene, video)
        return video

    def _active(self, scene=21):
        with open(os.path.join(self.latents, f"scene_{scene:03d}.latent"), "rb") as handle:
            return handle.read()

    def test_selecting_an_older_take_brings_its_latent_back(self):
        self._render(21, b"latent-one" * 100, b"video-one" * 1000)
        self._render(21, b"latent-two" * 100, b"video-two" * 1000)
        first_backup = os.path.join(self.backups, os.listdir(self.backups)[0])
        self.assertEqual(self._active(), b"latent-two" * 100)
        result = takes.activate_for_video(self.project, 21, first_backup)
        self.assertEqual(result["status"], "activated")
        self.assertEqual(self._active(), b"latent-one" * 100)
        newest = os.path.join(self.videos, "video_0021-audio.mp4")
        self.assertEqual(takes.activate_for_video(self.project, 21, newest)["status"], "activated")
        self.assertEqual(self._active(), b"latent-two" * 100)
        self.assertEqual(takes.activate_for_video(self.project, 21, newest)["status"], "already_active")

    def test_a_video_without_a_take_is_reported_when_the_scene_has_takes(self):
        self._render(21, b"latent" * 100, b"video" * 1000)
        stranger = os.path.join(self.project, "other.mp4")
        with open(stranger, "wb") as handle:
            handle.write(b"unrelated" * 1000)
        self.assertEqual(takes.activate_for_video(self.project, 21, stranger)["status"], "no_archive")

    def test_scenes_rendered_before_takes_were_kept_are_left_alone(self):
        active = os.path.join(self.latents, "scene_021.latent")
        with open(active, "wb") as handle:
            handle.write(b"legacy")
        video = os.path.join(self.videos, "video_0021-audio.mp4")
        with open(video, "wb") as handle:
            handle.write(b"video" * 1000)
        self.assertEqual(takes.activate_for_video(self.project, 21, video)["status"], "no_takes")
        self.assertEqual(self._active(), b"legacy")

    def test_a_take_whose_video_was_deleted_is_removed(self):
        self._render(21, b"latent-one" * 100, b"video-one" * 1000)
        self._render(21, b"latent-two" * 100, b"video-two" * 1000)
        for name in os.listdir(self.backups):
            os.remove(os.path.join(self.backups, name))
        self._render(21, b"latent-three" * 100, b"video-three" * 1000, backup_previous=False)
        self.assertEqual(len(takes.list_takes(self.project, 21)), 1)
        archived = [name for name in os.listdir(os.path.join(self.latents, "scene_021.takes")) if name.endswith(".latent")]
        self.assertEqual(len(archived), 1)

    def test_a_rewritten_clip_keeps_its_take(self):
        video = self._render(21, b"latent-one" * 100, b"video-one" * 1000)
        before = takes.video_fingerprint(video)
        with open(video, "wb") as handle:
            handle.write(b"color matched" * 1000)
        takes.rename_video(self.project, 21, before, takes.video_fingerprint(video))
        self.assertEqual(takes.activate_for_video(self.project, 21, video)["status"], "already_active")

    def test_an_uncollected_render_does_not_pile_up(self):
        active = os.path.join(self.latents, "scene_021.latent")
        for payload in (b"a" * 100, b"b" * 100):
            with open(active, "wb") as handle:
                handle.write(payload)
            takes.archive_latent(self.latents, 21, active)
        folder = os.path.join(self.latents, "scene_021.takes")
        self.assertEqual(len([name for name in os.listdir(folder) if name.endswith(".latent")]), 1)

    def test_take_folders_follow_scenes_that_are_inserted_or_removed(self):
        self._render(21, b"latent" * 100, b"video" * 1000)
        takes.shift_takes(self.latents, 21, 1)
        self.assertFalse(os.path.isdir(os.path.join(self.latents, "scene_021.takes")))
        self.assertTrue(os.path.isdir(os.path.join(self.latents, "scene_022.takes")))
        takes.shift_takes(self.latents, 22, -1)
        self.assertTrue(os.path.isdir(os.path.join(self.latents, "scene_021.takes")))
        takes.delete_scene_takes(self.latents, 21)
        self.assertFalse(os.path.isdir(os.path.join(self.latents, "scene_021.takes")))


if __name__ == "__main__":
    unittest.main()
