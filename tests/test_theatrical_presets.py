"""Theatrical UI/API preset parity and scene override behavior."""

import importlib.util
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("theatrical_presets", ROOT / "storyboard/theatrical_presets.py")
PRESETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRESETS)


class TheatricalPresetTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI/Python snapshot parity")
    def test_generated_snapshot_matches_all_three_ui_presets(self):
        subprocess.run(
            ["node", "scripts/export_theatrical_presets.mjs", "--check"], cwd=ROOT, check=True,
        )

    def test_each_preset_can_be_selected_independently(self):
        data = PRESETS.theatrical_presets()
        for field, key, direction in [
            ("camera_flow", data["camera_key"], data["camera"]["guidance"]),
            ("performance_style", "theatrical", data["performance"]["direction"]),
            ("facial_performance", "theatrical", data["facial"]["direction"]),
        ]:
            with self.subTest(field=field):
                lines = PRESETS.theatrical_direction_lines({field: key})
                self.assertEqual(len(lines), 1)
                self.assertIn(direction, lines[0])
        self.assertEqual(PRESETS.theatrical_direction_lines({}), [])

    def test_scene_overrides_replace_project_defaults(self):
        defaults = {"performance_style": "theatrical", "facial_performance": "theatrical"}
        card = {"scene_defaults": {"performance_style": "off", "facial_performance": "off"}}
        self.assertEqual(PRESETS.theatrical_direction_lines(card, defaults), [])
        self.assertEqual(len(PRESETS.theatrical_direction_lines({}, defaults)), 2)

    def test_authored_motion_and_facial_direction_keep_priority(self):
        card = {"camera_flow": "pretty_little_malfunction", "motion_summary": "Track the dancer's hands.",
                "facial_performance": "theatrical", "facial_performance_custom": "Angry"}
        self.assertEqual(PRESETS.theatrical_direction_lines(card), [])


if __name__ == "__main__":
    unittest.main()
