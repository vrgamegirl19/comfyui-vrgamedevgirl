import ast
import os
import unittest


class InitSubmodulesTests(unittest.TestCase):
    def test_all_submodules_exist(self):
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        init_py = os.path.join(repo_dir, "__init__.py")
        self.assertTrue(os.path.isfile(init_py))

        with open(init_py, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse _VRGDG_SUBMODULES tuple
        tree = ast.parse(content)
        submodules = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_VRGDG_SUBMODULES":
                        submodules = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
        self.assertIsNotNone(submodules)
        for modname in submodules:
            py_file = os.path.join(repo_dir, f"{modname.lstrip('.')}.py")
            self.assertTrue(
                os.path.isfile(py_file),
                f"Submodule '{modname}' referenced in _VRGDG_SUBMODULES does not exist at {py_file}"
            )


if __name__ == "__main__":
    unittest.main()
