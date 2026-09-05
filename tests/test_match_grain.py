import ast
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nodes.py"


def load_helper():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_match_grain_to_reference"
    )
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_match_grain_to_reference"]


class MatchGrainTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "Torch is provided by ComfyUI's Python environment.")
    def test_matches_reference_statistics_without_copying_reference_content(self):
        match_grain = load_helper()
        target = torch.full((2, 24, 32, 3), 0.5, dtype=torch.float32)
        reference = torch.full((1, 24, 32, 3), 0.5, dtype=torch.float32)
        reference[:, ::2, ::2, 0] = 0.8
        output_a = match_grain(target, reference, 1.0, 2, 0.35, 123, 2)
        output_b = match_grain(target, reference, 1.0, 2, 0.35, 123)
        self.assertEqual(tuple(output_a.shape), tuple(target.shape))
        self.assertTrue(torch.equal(output_a, output_b))
        self.assertTrue(torch.isfinite(output_a).all())
        self.assertGreater(float((output_a - target).abs().mean()), 0.0)
        self.assertLessEqual(float(output_a.min()), 1.0)
        self.assertGreaterEqual(float(output_a.max()), 0.0)

    def test_node_is_registered_as_a_standalone_image_effect(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("class MatchGrainToReference:", source)
        self.assertIn('"MatchGrainToReference": MatchGrainToReference', source)
        self.assertIn('"MatchGrainToReference": "🎞️ Match Grain To Reference"', source)


if __name__ == "__main__":
    unittest.main()
