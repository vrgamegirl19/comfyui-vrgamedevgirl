import ast
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "VRGDG_VideoCompareNode.py"
UI_SOURCE = ROOT / "web" / "VRGDG_VideoCompare.js"
INIT_SOURCE = ROOT / "__init__.py"


class _FolderPaths:
    def __init__(self, root):
        self.root = root

    def get_output_directory(self):
        return self.root

    def get_temp_directory(self):
        return self.root

    def get_input_directory(self):
        return self.root


def load_video_path_resolver(root):
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    names = {"_video_path_candidates", "_resolve_video_path"}
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "os": os,
        "folder_paths": _FolderPaths(root),
        "_VIDEO_EXTENSIONS": {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"},
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace["_resolve_video_path"]


class VideoCompareSliderTests(unittest.TestCase):
    def test_vhs_output_resolves_the_final_video_file(self):
        with tempfile.TemporaryDirectory() as folder:
            metadata = Path(folder) / "preview.png"
            silent_video = Path(folder) / "preview.mp4"
            audio_video = Path(folder) / "preview-audio.mp4"
            for path in (metadata, silent_video, audio_video):
                path.touch()
            resolve = load_video_path_resolver(folder)
            result = resolve((True, [str(metadata), str(silent_video), str(audio_video)]), "Before")
            self.assertEqual(Path(result), audio_video)

    def test_node_and_frontend_are_registered(self):
        node_source = NODE_SOURCE.read_text(encoding="utf-8")
        ui_source = UI_SOURCE.read_text(encoding="utf-8")
        init_source = INIT_SOURCE.read_text(encoding="utf-8")

        self.assertIn('"VRGDG_VideoCompareSlider": VRGDG_VideoCompareSlider', node_source)
        self.assertIn('".VRGDG_VideoCompareNode"', init_source)
        self.assertIn('const NODE_NAME = "VRGDG_VideoCompareSlider"', ui_source)
        self.assertIn('this.addDOMWidget("video_compare"', ui_source)
        self.assertIn("afterClip.style.clipPath", ui_source)
        self.assertIn("Promise.all([beforeVideo.play(), afterVideo.play()])", ui_source)
        self.assertIn("requestAnimationFrame(animationSync)", ui_source)


if __name__ == "__main__":
    unittest.main()
