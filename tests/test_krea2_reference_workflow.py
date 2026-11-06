"""Reference Builder uses the Krea-only two-pass graph with explicit sizes."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
image_workflows = importlib.import_module(f"{ROOT.name}.runner.image_workflows")

class Krea2ReferenceWorkflowTests(unittest.TestCase):
    def build(self, **settings):
        with patch.object(image_workflows, "_require_model_choice"):
            return image_workflows._build_krea2_2pass_api_prompt({
                "prompt": "A singer on a rooftop.", "seed": 42, **settings,
            })

    def test_reference_graph_uses_krea_for_both_passes_and_explicit_sizes(self):
        built = self.build(first_pass_width=768, first_pass_height=1024, width=1536, height=2048)
        graph = built["prompt"]
        self.assertEqual(graph["240"]["inputs"]["width"], 768)
        self.assertEqual(graph["240"]["inputs"]["height"], 1024)
        self.assertEqual(graph["226"]["inputs"]["width"], 1536)
        self.assertEqual(graph["226"]["inputs"]["height"], 2048)
        self.assertEqual(graph["231"]["inputs"]["model"], ["248", 0])
        self.assertEqual(graph["230"]["inputs"]["model"], ["248", 1])
        self.assertEqual(graph["248"]["inputs"]["model"], ["236", 0])
        self.assertEqual(graph["236"]["inputs"]["unet_name"], "krea2_turbo_fp8_scaled.safetensors")
        self.assertFalse(any("z_image" in str(node) for node in graph.values()))
        self.assertFalse(graph["242"]["inputs"]["switch"])
        self.assertFalse(graph["243"]["inputs"]["switch"])
        self.assertEqual(built["used_seed"], 42)

    def test_scene_workflow_keeps_its_resolution_selectors(self):
        graph = self.build()["prompt"]
        self.assertEqual(graph["240"]["inputs"]["width"], ["238", 0])
        self.assertEqual(graph["226"]["inputs"]["width"], ["49", 0])

    def test_explicit_sizes_are_clamped(self):
        graph = self.build(first_pass_width=1, first_pass_height=99999, width=1, height=99999)["prompt"]
        self.assertEqual(graph["240"]["inputs"]["width"], 64)
        self.assertEqual(graph["240"]["inputs"]["height"], 4096)
        self.assertEqual(graph["226"]["inputs"]["width"], 64)
        self.assertEqual(graph["226"]["inputs"]["height"], 4096)

if __name__ == "__main__":
    unittest.main()
