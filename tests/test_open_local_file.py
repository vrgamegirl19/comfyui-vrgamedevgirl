import ast
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve().parents[1] / "builder" / "paths.py"


def load_open_local_file(opened):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_open_local_file")
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_OPENABLE_EXTENSIONS" for t in node.targets))
    ]
    # Record what would be opened instead of launching anything.
    fake_os = types.SimpleNamespace(path=os.path, name="nt", startfile=opened.append)
    namespace = {"os": fake_os, "sys": sys, "subprocess": subprocess}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["_open_local_file"]


class OpenLocalFileTests(unittest.TestCase):
    def setUp(self):
        self.opened = []
        self.open_local_file = load_open_local_file(self.opened)
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_media_text_and_folders_open(self):
        for name in ("fox_girl_final.mp4", "render_report.txt", "Poster.PNG"):
            (self.root / name).write_bytes(b"x")
            self.open_local_file(str(self.root / name))
        self.open_local_file(f'"{self.root}"')
        self.assertEqual([Path(p).name for p in self.opened], ["fox_girl_final.mp4", "render_report.txt", "Poster.PNG", self.root.name])

    def test_programs_and_scripts_are_refused(self):
        for name in ("tool.exe", "run.bat", "setup.ps1", "shortcut.lnk", "no_extension"):
            (self.root / name).write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "Only video, image, audio and text files"):
                self.open_local_file(str(self.root / name))
        self.assertEqual(self.opened, [])

    def test_missing_file_is_reported(self):
        with self.assertRaisesRegex(ValueError, "File was not found"):
            self.open_local_file(str(self.root / "missing.mp4"))


if __name__ == "__main__":
    unittest.main()
