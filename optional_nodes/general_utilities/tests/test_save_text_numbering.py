import ast
import json
import os
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "VRGDG_GeneralNodes.py"


def load_save_nodes(folder):
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    names = {"_sanitize_text_segment", "_coerce_text_payload", "_next_incremental_prefixed_file_name", "VRGDG_SaveTextAdvanced", "VRGDG_SaveTextAdvancedConcat"}
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {"os": os, "re": re, "json": json, "_get_text_files_manual_folder": lambda _name: (folder, "")}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace


class SaveTextNumberingTests(unittest.TestCase):
    def test_save_without_overwrite_numbers_each_file(self):
        with tempfile.TemporaryDirectory() as folder:
            nodes = load_save_nodes(folder)
            saver = nodes["VRGDG_SaveTextAdvanced"]()
            paths = [saver.run("story", "story", False, f"take {i}", 0)[1] for i in (1, 2)]
            self.assertEqual([os.path.basename(p) for p in paths], ["story_001.txt", "story_002.txt"])
            self.assertEqual(Path(paths[1]).read_text(encoding="utf-8"), "take 2")

    def test_concat_node_without_overwrite_or_concat_numbers_each_file(self):
        with tempfile.TemporaryDirectory() as folder:
            nodes = load_save_nodes(folder)
            saver = nodes["VRGDG_SaveTextAdvancedConcat"]()
            first = saver.run("story", "story", False, False, "one", 0)[1]
            second = saver.run("story", "story", False, False, "two", 0)[1]
            self.assertEqual((os.path.basename(first), os.path.basename(second)), ("story_001.txt", "story_002.txt"))


if __name__ == "__main__":
    unittest.main()
