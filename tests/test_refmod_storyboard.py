import ast
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_scene_helpers():
    """Load only the reference-card helpers from storyboard/scene_helpers.py (the module needs package imports)."""
    path = ROOT / "storyboard" / "scene_helpers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {"_clean_scene_text", "_normalize_reference_image", "_refmod_card_fields", "_normalize_reference_item"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class StoryboardRefModCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.helpers = _load_scene_helpers()
        except NameError as exc:  # pragma: no cover - helper dependencies changed
            raise unittest.SkipTest(f"scene_helpers dependencies changed: {exc}")

    def test_refmod_fields_survive_normalization(self):
        card = {
            "id": "a", "name": "Brad", "reference_type": "character", "source": "refmod",
            "refmod": {"name": "identity/brad", "kind": "video", "tokens": "1440", "frames": 4, "strength": 2, "type": "identity"},
            "wears": "b", "follow": False,
        }
        fields = self.helpers["_refmod_card_fields"](card)
        self.assertEqual(fields["source"], "refmod")
        self.assertEqual(fields["refmod"]["tokens"], 1440)
        self.assertEqual(fields["refmod"]["strength"], 1.0)
        self.assertEqual(fields["wears"], "b")
        self.assertIs(fields["follow"], False)

    def test_other_cards_are_left_alone(self):
        helper = self.helpers["_refmod_card_fields"]
        self.assertEqual(helper({"id": "a", "name": "Brad"}), {})
        self.assertEqual(helper({"source": "refmod", "refmod": {"name": ""}}), {})


class StoryboardRefModJsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the storyboard tests")
    def test_storyboard_catalog_labels_and_gpt_payload(self):
        result = subprocess.run(["node", "--test", str(Path(__file__).with_name("storyboard_refmod.mjs"))],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
