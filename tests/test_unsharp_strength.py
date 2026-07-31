import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "nodes.py").read_text(encoding="utf-8")


class FastUnsharpStrengthTests(unittest.TestCase):
    def test_strength_widget_supports_values_up_to_ten(self):
        start = SOURCE.index("class FastUnsharpSharpen:")
        end = SOURCE.index("class FastLaplacianSharpen:", start)
        unsharp_source = SOURCE[start:end]
        self.assertIn('"max": 10.0', unsharp_source)
        self.assertNotIn('"max": 2.0', unsharp_source)


if __name__ == "__main__":
    unittest.main()
