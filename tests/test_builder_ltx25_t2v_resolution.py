import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
RUNNER_SOURCE = (ROOT / "VRGDG_WorkflowRunnerNodes.py").read_text(encoding="utf-8")


class Ltx25T2VResolutionTests(unittest.TestCase):
    def test_ltx25_panel_exposes_explicit_dimensions(self):
        self.assertIn('i2vSettingsGrid.style.display = "grid";', BUILDER_SOURCE)
        self.assertIn('ltx25ResolutionGrid.style.display = "none";', BUILDER_SOURCE)
        self.assertIn('makeField("FPS", i2vFpsInput)', BUILDER_SOURCE)
        self.assertIn('makeField("Seed", i2vSeedInput)', BUILDER_SOURCE)
        self.assertIn('makeField("Width", i2vWidthInput)', BUILDER_SOURCE)
        self.assertIn('makeField("Height", i2vHeightInput)', BUILDER_SOURCE)

    def test_ltx25_generates_aligned_and_outputs_exact_requested_size(self):
        tree = ast.parse(RUNNER_SOURCE)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_patch_t2v_api_prompt"
        )
        source = ast.get_source_segment(RUNNER_SOURCE, function)
        self.assertIn("math.ceil(width / 64.0) * 64", source)
        self.assertIn("math.ceil(height / 64.0) * 64", source)
        self.assertIn('"width": width', source)
        self.assertIn('"height": height', source)
        self.assertIn('"crop": "center"', source)
        self.assertIn('("936", 0), (final_resize_id, 0)', source)


if __name__ == "__main__":
    unittest.main()
