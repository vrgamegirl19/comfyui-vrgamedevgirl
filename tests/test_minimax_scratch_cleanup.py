import ast
import hashlib
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class ScratchCleanupTests(unittest.TestCase):
    def test_cleanup_only_accepts_expected_scene_and_reports_errors(self):
        source = (Path(__file__).resolve().parents[1] / "VRGDG_WorkflowRunnerNodes.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {"_minimax_h3_output_location", "_cleanup_minimax_h3_output_folder"}
        code = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        with tempfile.TemporaryDirectory() as tmp:
            remove = Mock()
            ns = dict(os=os, re=re, hashlib=hashlib, shutil=SimpleNamespace(rmtree=remove), folder_paths=SimpleNamespace(get_output_directory=lambda: tmp))
            exec(compile(code, "<cleanup>", "exec"), ns)
            project = os.path.join(tmp, "project")
            scene, _ = ns["_minimax_h3_output_location"](project, 1)
            cleanup = ns["_cleanup_minimax_h3_output_folder"]
            payload = dict(project_folder=project, scene_number=1, output_folder=scene)
            for invalid in ["", tmp, os.path.join(tmp, "VRGDG_MiniMaxH3"), os.path.dirname(scene), os.path.join(os.path.dirname(scene), "scene_0002")]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    cleanup({**payload, "output_folder": invalid})
            with self.assertRaises(ValueError):
                cleanup({**payload, "project_folder": ""})
            with self.assertRaises(ValueError):
                cleanup({**payload, "scene_number": 0})
            remove.assert_not_called()
            self.assertTrue(cleanup(payload)["removed"])
            remove.assert_called_once_with(scene)
            remove.side_effect = PermissionError("locked")
            with self.assertRaises(PermissionError):
                cleanup(payload)


if __name__ == "__main__":
    unittest.main()
