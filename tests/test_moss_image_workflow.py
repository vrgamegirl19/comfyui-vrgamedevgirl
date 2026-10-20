"""MOSS-Image reference graph: native loaders, optional turbo LoRA and no sampling-shift node."""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

image_workflows = importlib.import_module(f"{ROOT.name}.runner.image_workflows")


def _classes(prompt: dict) -> list[str]:
    return [node["class_type"] for node in prompt.values()]


class MossImageWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(image_workflows, "_require_model_choice", lambda *_args: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_turbo_graph_uses_native_loaders_and_lora(self) -> None:
        built = image_workflows._build_moss_image_api_prompt({
            "prompt": "A singer on a rooftop.",
            "lora_name": "moss_image_sft_8_steps_dmd_lora.safetensors",
            "steps": 8, "cfg": 1, "width": 1920, "height": 1088, "seed": 5,
        })
        prompt = built["prompt"]
        self.assertEqual(prompt["2"]["inputs"]["type"], "moss_image")
        self.assertEqual(prompt["4"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(prompt["10"]["inputs"]["model"], ["4", 0])
        self.assertEqual(prompt["10"]["inputs"]["steps"], 8)
        self.assertEqual(prompt["9"]["inputs"]["width"], 1920)
        self.assertEqual(built["used_seed"], 5)
        # Native MOSS-Image samples with multiplier 1; ModelSamplingSD3 would force 1000 and give noise.
        self.assertNotIn("ModelSamplingSD3", _classes(prompt))

    def test_quality_graph_skips_lora(self) -> None:
        prompt = image_workflows._build_moss_image_api_prompt({
            "prompt": "A singer on a rooftop.", "lora_name": "[none]", "steps": 50, "cfg": 3,
        })["prompt"]
        self.assertNotIn("LoraLoaderModelOnly", _classes(prompt))
        self.assertEqual(prompt["10"]["inputs"]["model"], ["1", 0])
        self.assertEqual(prompt["10"]["inputs"]["cfg"], 3.0)

    def test_empty_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            image_workflows._build_moss_image_api_prompt({"prompt": "  "})


if __name__ == "__main__":
    unittest.main()
