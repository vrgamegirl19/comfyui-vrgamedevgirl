"""Model selection validation without ComfyUI imports or GPU work."""

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "minimax/pass2_enhancer.py"
SPEC = importlib.util.spec_from_file_location("pass2_enhancer", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PassTwoEnhancerTests(unittest.TestCase):
    def test_resolves_registered_subfolders_and_custom_selected_names(self):
        available = ["[none]", "H3\\dmad.safetensors", "H3/refiner.safetensors"]
        self.assertEqual(MODULE.resolve_enhancer_loras(available, "dmad.safetensors", "refiner.safetensors"),
                         (available[1], available[2]))

    def test_missing_models_are_reported_together(self):
        with self.assertRaisesRegex(ValueError, "DMAD.*dmad.safetensors.*Open-H3 Refiner.*refiner.safetensors"):
            MODULE.resolve_enhancer_loras([], "dmad.safetensors", "refiner.safetensors")

    def test_duplicate_filenames_require_an_explicit_path(self):
        available = ["a/dmad.safetensors", "b/dmad.safetensors", "refiner.safetensors"]
        with self.assertRaisesRegex(ValueError, "multiple files"):
            MODULE.resolve_enhancer_loras(available, "dmad.safetensors", "refiner.safetensors")
        self.assertEqual(MODULE.resolve_enhancer_loras(available, "b/dmad.safetensors", "refiner.safetensors"),
                         ("b/dmad.safetensors", "refiner.safetensors"))


if __name__ == "__main__":
    unittest.main()
