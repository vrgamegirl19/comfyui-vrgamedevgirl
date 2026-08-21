import ast
import os
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "VRGDG_MusicVideoBuilderNodes.py"


class BuilderThumbnailRecoveryTests(unittest.TestCase):
    def test_recovery_creates_thumbnail_parent_before_running_ffmpeg(self):
        tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_ensure_builder_scene_video_thumbnail"
        )

        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        mkdir_call = next(
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "os"
            and call.func.attr == "makedirs"
        )
        subprocess_call = next(
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "subprocess"
            and call.func.attr == "run"
        )

        self.assertLess(mkdir_call.lineno, subprocess_call.lineno)
        keywords = {item.arg: item.value for item in mkdir_call.keywords}
        self.assertIn("exist_ok", keywords)
        self.assertIsInstance(keywords["exist_ok"], ast.Constant)
        self.assertTrue(keywords["exist_ok"].value)

    def test_recovery_writes_into_a_previously_missing_thumbnail_folder(self):
        tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_ensure_builder_scene_video_thumbnail"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            video = project / "rendered_scene_videos" / "video_0001-audio.mp4"
            thumbnail = project / "scene_video_thumbnails" / "video_0001-audio.jpg"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")

            def fake_run(command, **_kwargs):
                self.assertTrue(thumbnail.parent.is_dir())
                Path(command[-1]).write_bytes(b"thumbnail")
                return types.SimpleNamespace(returncode=0, stderr="", stdout="")

            namespace = {
                "os": os,
                "subprocess": types.SimpleNamespace(run=fake_run),
                "_builder_scene_video_thumbnail_path": lambda _path: str(thumbnail),
                "_find_ffmpeg_path": lambda: "ffmpeg",
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

            result = namespace["_ensure_builder_scene_video_thumbnail"](str(video))

            self.assertEqual(result, str(thumbnail))
            self.assertTrue(thumbnail.is_file())


if __name__ == "__main__":
    unittest.main()
