import ast
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODES = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(encoding="utf-8")
UI = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")

HELPER_NAMES = {
    "_is_inside_folder",
    "_normalize_branch_keep",
    "_looks_like_filesystem_path",
    "_rewrite_project_path_string",
    "_rewrite_project_paths",
    "_clear_remaining_source_paths",
    "_rewrite_project_json_file",
    "_rewrite_project_sidecar_json",
    "_branch_copy_ignore",
    "_copy_project_tree",
    "_is_project_root_export_video",
    "_remove_project_root_export_videos",
    "_empty_keep_value",
    "_strip_segment_keep_fields",
    "_apply_branch_keep_to_session",
    "_branch_scene_number",
    "_copy_external_media_into_branch",
    "_relocate_external_minimax_stage_media",
    "_strip_storyboard_images",
    "_prune_unkept_project_folders",
    "_rebase_project_owned_paths",
    "_atomic_write_text",
    "_atomic_write_json",
}


def load_helpers():
    tree = ast.parse(NODES)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES:
            kept.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and (
                    target.id.startswith("_BRANCH_") or target.id.startswith("_MINIMAX_H3_STAGE_")
                ):
                    kept.append(node)
    module = ast.Module(body=kept, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "json": json,
        "os": os,
        "re": __import__("re"),
        "shutil": shutil,
        "tempfile": tempfile,
    }
    exec(compile(module, "VRGDG_MusicVideoBuilderNodes.py", "exec"), namespace)
    return namespace


def write_file(path, content="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


class BuilderBranchProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_helpers()

    def test_save_as_copies_tree_and_reads_keep(self):
        self.assertIn("_copy_project_tree(source, target, keep)", NODES)
        self.assertIn("keep = _normalize_branch_keep(payload)", NODES)
        self.assertIn("keep: branchKeepFlags(choice.preset, choice.keep)", UI)
        self.assertIn("Storyboard Builder data", UI)
        self.assertIn("minimax_h3_continuity_frame_path", UI)
        self.assertIn("minimax_h3_stage1_source_path", UI)
        self.assertIn("minimax_h3_stage1_source_path", NODES)
        self.assertIn("id_lora_reference_builder", UI)
        self.assertIn("story_beat", UI)

    def test_zip_import_rebases_before_load(self):
        self.assertIn("imported_session = _rebase_project_owned_paths(target, old_folder, imported_session)", NODES)

    def test_full_copy_rewrites_session_and_storyboard_paths(self):
        ns = self.ns
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "src")
            target = os.path.join(root, "dst")
            image = os.path.join(source, "zimage_approved", "scene.png")
            video = os.path.join(source, "rendered_scene_videos", "video_0001-audio.mp4")
            continuity = os.path.join(source, "scene_image_previews", "scene_0001", "frame.png")
            ref = os.path.join(source, "project_context", "flux_references", "subjects", "hero.png")
            board_image = os.path.join(source, "storyboard", "scene1.png")
            final_video = os.path.join(source, "FINAL_VIDEO.mp4")
            preview_video = os.path.join(source, "PREVIEW_SCENES_all.mp4")
            session_backup = os.path.join(source, "session_backups", "vrgdg_builder_session.json")
            for path in (image, video, continuity, ref, board_image, final_video, preview_video, session_backup):
                write_file(path, "data")
            write_file(os.path.join(source, "storyboard", "storyboard.json"), json.dumps({
                "scenes": [{"id": "s1", "image_path": board_image}],
            }))
            keep = ns["_normalize_branch_keep"]({"keep": {key: True for key in ns["_BRANCH_KEEP_KEYS"]}})
            ns["_copy_project_tree"](source, target, keep)
            session = {
                "project_folder": source,
                "approved_image_path": image,
                "segments": [{
                    "approved_image_path": image,
                    "video_history": [video],
                    "minimax_h3_continuity_frame_path": continuity,
                    "lyric_text": "hello",
                }],
                "flux_reference_builder": {"subjects": [{"image": {"path": ref}}]},
            }
            session = ns["_rebase_project_owned_paths"](target, source, session)
            blob = json.dumps(session)
            self.assertNotIn(os.path.normcase(source), os.path.normcase(blob))
            self.assertTrue(os.path.isfile(session["segments"][0]["approved_image_path"]))
            self.assertTrue(os.path.isfile(session["segments"][0]["video_history"][0]))
            self.assertTrue(os.path.isfile(session["segments"][0]["minimax_h3_continuity_frame_path"]))
            self.assertTrue(os.path.isfile(session["flux_reference_builder"]["subjects"][0]["image"]["path"]))
            board = json.loads((Path(target) / "storyboard" / "storyboard.json").read_text(encoding="utf-8"))
            self.assertTrue(os.path.isfile(board["scenes"][0]["image_path"]))
            self.assertNotIn(os.path.normcase(source), os.path.normcase(board["scenes"][0]["image_path"]))
            self.assertTrue(os.path.isfile(image))
            self.assertTrue(os.path.isfile(final_video))
            self.assertTrue(os.path.isfile(preview_video))
            self.assertFalse(os.path.isfile(os.path.join(target, "FINAL_VIDEO.mp4")))
            self.assertFalse(os.path.isfile(os.path.join(target, "PREVIEW_SCENES_all.mp4")))
            self.assertFalse(os.path.isdir(os.path.join(target, "session_backups")))
            self.assertTrue(os.path.isfile(session_backup))

    def test_fresh_media_drops_media_and_keeps_lyrics(self):
        ns = self.ns
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "src")
            target = os.path.join(root, "dst")
            video = os.path.join(source, "rendered_scene_videos", "video_0001-audio.mp4")
            image = os.path.join(source, "zimage_approved", "scene.png")
            write_file(video, "v")
            write_file(image, "i")
            write_file(os.path.join(source, "storyboard", "storyboard.json"), json.dumps({"scenes": [{"id": "s1"}]}))
            keep = ns["_normalize_branch_keep"]({"keep": {"lyrics": True}})
            ns["_copy_project_tree"](source, target, keep)
            ns["_prune_unkept_project_folders"](target, keep)
            session = ns["_apply_branch_keep_to_session"]({
                "segments": [{
                    "lyric_text": "keep me",
                    "approved_image_path": image,
                    "video_path": video,
                    "t2i_prompt": "gone",
                    "story_beat": "gone beat",
                }],
                "flux_reference_builder": {"subjects": [{"name": "A"}]},
            }, keep)
            self.assertEqual(session["segments"][0]["lyric_text"], "keep me")
            self.assertEqual(session["segments"][0]["approved_image_path"], "")
            self.assertEqual(session["segments"][0]["video_path"], "")
            self.assertEqual(session["segments"][0]["t2i_prompt"], "")
            self.assertEqual(session["flux_reference_builder"], {})
            self.assertFalse(os.path.isdir(os.path.join(target, "rendered_scene_videos")))
            self.assertFalse(os.path.isdir(os.path.join(target, "storyboard")))

    def test_minimax_stage_paths_move_into_branch_or_clear(self):
        ns = self.ns
        with tempfile.TemporaryDirectory() as root:
            outside = os.path.join(root, "VRGDG_MiniMaxH3", "oldproj_abc123", "scene_0001", "clip_stage1-audio.mp4")
            write_file(outside, "stage")
            target = os.path.join(root, "dst")
            os.makedirs(target, exist_ok=True)
            relocated = ns["_relocate_external_minimax_stage_media"]({
                "segments": [{
                    "minimax_h3_stage1_source_path": outside,
                    "minimax_h3_stage1_path": outside,
                    "minimax_h3_stage2_path": outside,
                }],
            }, target)
            new_path = relocated["segments"][0]["minimax_h3_stage1_path"]
            self.assertTrue(os.path.isfile(new_path))
            self.assertTrue(ns["_is_inside_folder"](new_path, target))
            self.assertNotEqual(os.path.normcase(new_path), os.path.normcase(outside))
            self.assertEqual(relocated["segments"][0]["minimax_h3_stage1_source_path"], new_path)
            self.assertTrue(os.path.isfile(outside))
            stripped = ns["_apply_branch_keep_to_session"]({
                "segments": [{"minimax_h3_stage1_path": outside, "minimax_h3_stage1_source_path": outside, "lyric_text": "keep"}],
            }, ns["_normalize_branch_keep"]({"keep": {"lyrics": True}}))
            self.assertEqual(stripped["segments"][0]["minimax_h3_stage1_path"], "")
            self.assertEqual(stripped["segments"][0]["minimax_h3_stage1_source_path"], "")
            self.assertEqual(stripped["segments"][0]["lyric_text"], "keep")

    def test_missing_copy_clears_old_path(self):
        ns = self.ns
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "src")
            target = os.path.join(root, "dst")
            os.makedirs(source, exist_ok=True)
            os.makedirs(target, exist_ok=True)
            missing = os.path.join(source, "gone.png")
            write_file(missing, "x")
            session = ns["_rebase_project_owned_paths"](target, source, {"path": missing})
            self.assertEqual(session["path"], "")


if __name__ == "__main__":
    unittest.main()
