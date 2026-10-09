"""Sage Attention and fp16 accumulation are Model Loader settings that every pass layout must honor and show."""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

WORKFLOWS_SOURCE = (ROOT / "runner" / "minimax_workflows.py").read_text(encoding="utf-8")
PANEL_SOURCE = (ROOT / "web" / "music_video_builder" / "minimax_panel.mjs").read_text(encoding="utf-8")


class LoaderSettingsTests(unittest.TestCase):
    def test_two_pass_and_advanced_two_pass_builders_set_their_loader(self):
        self.assertIn('_set_api_input(prompt, "141", "sage_attention", str(payload.get("sage_attention") or "auto"))', WORKFLOWS_SOURCE)
        self.assertIn('_set_api_input(prompt, "330", "sage_attention", str(payload.get("sage_attention") or "auto"))', WORKFLOWS_SOURCE)
        self.assertIn('_set_api_input(prompt, "330", "enable_fp16_accumulation", _bool_payload(payload, "enable_fp16_accumulation", True))', WORKFLOWS_SOURCE)

    def test_panel_keeps_the_model_loader_section_visible_in_multi_pass(self):
        self.assertIn('miniMaxAdvancedSettings.style.display = "";', PANEL_SOURCE)
        self.assertNotIn("miniMaxAdvancedSettings.style.display = multiPassMode", PANEL_SOURCE)
        self.assertIn('miniMaxModelLoaderSettings.style.display = "";', PANEL_SOURCE)
        self.assertNotIn("miniMaxModelLoaderSettings.style.display = hideMultiPassIgnoredSettings", PANEL_SOURCE)


if __name__ == "__main__":
    unittest.main()
