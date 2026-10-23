"""The agent API's temporal / world effect block matches the Video Builder's, and lands last in the prompt."""

import importlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

temporal = importlib.import_module(f"{ROOT.name}.agent_api.temporal_effect")
CASES = json.loads((Path(__file__).parent / "temporal_effect_cases.json").read_text(encoding="utf-8"))["cases"]


class TemporalEffectTests(unittest.TestCase):
    def test_matches_the_video_builder_cases(self):
        for case in CASES:
            with self.subTest(case["name"]):
                self.assertEqual(temporal.temporal_world_effect_verbiage(case["card"], case["defaults"]), case["expected"])

    def test_the_contract_goes_last_once(self):
        block = temporal.temporal_world_effect_verbiage({}, {"temporal_world_effect": "frozen_world"})
        prompt = f"detailed_description:\n[Shot 1] She walks.\n\n{block}\n\n[Shot 2] She turns."
        out = temporal.append_temporal_world_effect(prompt, block)
        self.assertTrue(out.endswith(block))
        self.assertEqual(out.count(block), 1)
        self.assertEqual(temporal.append_temporal_world_effect("A prompt.", ""), "A prompt.")


if __name__ == "__main__":
    unittest.main()
