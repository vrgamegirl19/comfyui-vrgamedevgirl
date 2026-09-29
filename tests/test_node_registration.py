import ast
import re
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def submodule_files():
    source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    names = re.findall(r'^\s+"\.([\w.]+)",', source, re.M)
    for name in names:
        path = ROOT / (name.replace(".", "/") + ".py")
        if not path.exists():
            path = ROOT / name.replace(".", "/") / "__init__.py"
        if path.exists():
            yield name, path


def literal_mapping_keys(path, mapping_name):
    """Node names in each literal `mapping_name = {...}` in a file, duplicates included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(getattr(target, "id", "") == mapping_name for target in node.targets):
            yield [key.value for key in node.value.keys if isinstance(key, ast.Constant)]


class NodeRegistrationTests(unittest.TestCase):
    def test_node_names_are_unique(self):
        owners = defaultdict(list)
        for module, path in submodule_files():
            for mapping_name in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
                for keys in literal_mapping_keys(path, mapping_name):
                    repeated = sorted({key for key in keys if keys.count(key) > 1})
                    self.assertEqual(repeated, [], f"{path.name} repeats {mapping_name} keys; only the last one is kept")
                    if mapping_name == "NODE_CLASS_MAPPINGS":
                        for key in keys:
                            owners[key].append(module)
        collisions = {name: modules for name, modules in owners.items() if len(modules) > 1}
        self.assertEqual(collisions, {}, "the same node name is registered by several submodules")


if __name__ == "__main__":
    unittest.main()
