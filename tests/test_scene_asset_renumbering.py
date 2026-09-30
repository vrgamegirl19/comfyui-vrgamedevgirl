import ast
import os
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from builder_source import read_builder_backend_source


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "builder/nodes.py"


class _Latents:
    calls = []

    @classmethod
    def delete_latent(cls, project_folder, scene_number):
        cls.calls.append(("delete", scene_number))

    @classmethod
    def reindex_latents(cls, project_folder, scene_number):
        cls.calls.append(("reindex", scene_number))

    @classmethod
    def make_room_for_scene(cls, project_folder, scene_number):
        cls.calls.append(("make_room", scene_number))


def load_renumber():
    tree = ast.parse(read_builder_backend_source(), filename=str(NODE_SOURCE))
    names = {
        "_SCENE_ASSET_FOLDERS", "_SCENE_ASSET_NAME", "_scene_asset_number", "_renumbered_scene_asset_name",
        "_shift_scene_assets", "_renumber_scene_assets_after_removal", "_renumber_scene_assets_after_insert",
        "_MAX_REMOVED_SCENE_ASSETS", "_prune_removed_scene_assets",
    }
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(target, "id", "") in names for target in node.targets))
    ]
    namespace = {"os": os, "re": re, "shutil": shutil, "time": time, "SceneLatentManager": _Latents}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace["_renumber_scene_assets_after_removal"], namespace["_renumber_scene_assets_after_insert"]


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class SceneAssetRenumberingTests(unittest.TestCase):
    def setUp(self):
        self.renumber, self.renumber_insert = load_renumber()
        self.project = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.project)
        self.scenes = [f"scene{i}" for i in range(1, 31)]
        for number, scene in enumerate(self.scenes, start=1):
            write(self.project / "rendered_scene_videos" / f"video_{number:04d}-audio.mp4", scene)
            write(self.project / "rendered_scene_videos_backup" / f"scene_{number:04d}" / f"video_{number:04d}-audio_20260101.mp4", scene)
            write(self.project / "scene_video_thumbnails" / f"video_{number:04d}-audio.jpg", scene)
            write(self.project / "zimage_approved" / f"image_{number:04d}.png", scene)
            write(self.project / "project_context" / f"scene_{number:04d}" / "notes.txt", scene)
            write(self.project / "image_to_video_clips" / f"scene_{number:04d}" / "clip.mp4", scene)
        write(self.project / "rendered_scene_videos" / "video_10001-audio.mp4", "insert1")

    def read(self, *parts):
        path = self.project.joinpath(*parts)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def merge(self, left_position):
        """Delete both scenes' videos, then merge the scene at left_position with the one after it."""
        for position in (left_position, left_position + 1):
            (self.project / "rendered_scene_videos" / f"video_{position:04d}-audio.mp4").unlink()
        self.scenes[left_position - 1] += "+merged"
        del self.scenes[left_position]
        return self.renumber(str(self.project), left_position + 1)

    def split(self, position):
        """Delete the scene's video, then split it; the right-hand half becomes a new scene with no files."""
        (self.project / "rendered_scene_videos" / f"video_{position:04d}-audio.mp4").unlink()
        self.scenes.insert(position, None)
        return self.renumber_insert(str(self.project), position + 1)

    def assert_files_match_positions(self):
        for number, scene in enumerate(self.scenes, start=1):
            if scene is None:
                self.assertIsNone(self.read("rendered_scene_videos", f"video_{number:04d}-audio.mp4"))
                self.assertIsNone(self.read("zimage_approved", f"image_{number:04d}.png"))
                self.assertIsNone(self.read("project_context", f"scene_{number:04d}", "notes.txt"))
                continue
            base = scene.split("+")[0]
            video = self.read("rendered_scene_videos", f"video_{number:04d}-audio.mp4")
            self.assertIn(video, (base, None), f"video_{number:04d} belongs to {video}, expected {scene}")
            self.assertEqual(self.read("zimage_approved", f"image_{number:04d}.png"), base)
            self.assertEqual(self.read("rendered_scene_videos_backup", f"scene_{number:04d}", f"video_{number:04d}-audio_20260101.mp4"), base)
            self.assertEqual(self.read("project_context", f"scene_{number:04d}", "notes.txt"), base)
            self.assertEqual(self.read("image_to_video_clips", f"scene_{number:04d}", "clip.mp4"), base)
        self.assertIsNone(self.read("zimage_approved", f"image_{len(self.scenes) + 1:04d}.png"))
        self.assertEqual(self.read("rendered_scene_videos", "video_10001-audio.mp4"), "insert1")

    def test_several_merges_keep_every_file_on_its_scene_position(self):
        for left_position in (15, 20, 22):
            self.merge(left_position)
            self.assert_files_match_positions()
        self.assertEqual(len(self.scenes), 27)
        self.assertEqual(self.read("rendered_scene_videos", "video_0016-audio.mp4"), "scene17")
        self.assertEqual(_Latents.calls[-2:], [("delete", 23), ("reindex", 23)])

    def test_removed_scene_files_are_archived_not_deleted(self):
        self.merge(15)
        archived = list((self.project / "removed_scene_assets").glob("scene_0016_*/zimage_approved/image_0016.png"))
        self.assertEqual([path.read_text(encoding="utf-8") for path in archived], ["scene16"])

    def test_renamed_pairs_cover_files_inside_renamed_folders(self):
        renamed = self.merge(15)
        pairs = {os.path.relpath(old, self.project): os.path.relpath(new, self.project) for old, new in renamed}
        self.assertEqual(pairs[os.path.join("rendered_scene_videos", "video_0017-audio.mp4")], os.path.join("rendered_scene_videos", "video_0016-audio.mp4"))
        self.assertEqual(
            pairs[os.path.join("rendered_scene_videos_backup", "scene_0017", "video_0017-audio_20260101.mp4")],
            os.path.join("rendered_scene_videos_backup", "scene_0016", "video_0016-audio_20260101.mp4"),
        )
        self.assertNotIn(os.path.join("rendered_scene_videos", "video_10001-audio.mp4"), pairs)

    def test_split_shifts_later_scene_files_up_and_leaves_the_new_scene_empty(self):
        renamed = self.split(15)
        self.assert_files_match_positions()
        self.assertEqual(len(self.scenes), 31)
        self.assertEqual(self.read("rendered_scene_videos", "video_0017-audio.mp4"), "scene16")
        self.assertEqual(self.read("rendered_scene_videos", "video_0031-audio.mp4"), "scene30")
        self.assertEqual(self.read("zimage_approved", "image_0015.png"), "scene15")
        self.assertEqual(_Latents.calls[-1], ("make_room", 16))
        pairs = {os.path.relpath(old, self.project): os.path.relpath(new, self.project) for old, new in renamed}
        self.assertEqual(
            pairs[os.path.join("rendered_scene_videos_backup", "scene_0016", "video_0016-audio_20260101.mp4")],
            os.path.join("rendered_scene_videos_backup", "scene_0017", "video_0017-audio_20260101.mp4"),
        )

    def test_splits_and_merges_in_any_order_keep_files_on_their_positions(self):
        self.split(15)
        self.merge(20)
        self.split(3)
        self.merge(29)
        self.assert_files_match_positions()
        self.assertEqual(len(self.scenes), 30)


if __name__ == "__main__":
    unittest.main()
