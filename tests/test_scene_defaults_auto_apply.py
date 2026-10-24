"""Scoped Scene Defaults survive storage and drive headless prompt planning."""
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_scoped_defaults_test"
for name, folder in [(PACKAGE, ROOT), (PACKAGE + ".storyboard", ROOT / "storyboard"),
                     (PACKAGE + ".minimax", ROOT / "minimax")]:
    module = types.ModuleType(name)
    module.__path__ = [str(folder)]
    sys.modules.setdefault(name, module)
scoped = importlib.import_module(PACKAGE + ".storyboard.scene_defaults")
fields = importlib.import_module(PACKAGE + ".storyboard.scene_card_fields")
sync = importlib.import_module(PACKAGE + ".storyboard.session_sync")
assembly = importlib.import_module(PACKAGE + ".minimax.prompt_assembly")


class SceneDefaultsAutoApplyTests(unittest.TestCase):
    def test_scene_defaults_validate_zero_off_and_reject_project_only_fields(self):
        values = {"camera_motion_speed": 0, "minimax_h3_cut_frequency": 0,
                  "facial_performance": "", "temporal_allow_background_extras": False}
        self.assertEqual(scoped.normalize_scene_defaults(values, strict=True), values)
        for invalid in ({"story_arc_detail": "rich"}, {"camera_motion_speed": 11},
                        {"camera_motion_speed": True}, {"temporal_allow_background_extras": "yes"}):
            with self.assertRaises(ValueError):
                scoped.normalize_scene_defaults(invalid, strict=True)

    def test_api_card_and_segment_preserve_and_edit_the_same_scoped_fields(self):
        record = {"performance_style": "metal_screaming", "camera_motion_speed": 8}
        parsed, _ = fields.parse_scene_card_patch({"scene_defaults": record})
        segment, card = {}, {}
        fields.apply_to_segment(segment, parsed, video_engine="minimax_h3")
        fields.apply_to_card(card, parsed)
        self.assertEqual(segment["scene_defaults"], card["scene_defaults"])
        self.assertEqual(segment["performance_style"], "metal_screaming")
        parsed, _ = fields.parse_scene_card_patch({"performance_style": "rock_punk"})
        fields.apply_to_segment(segment, parsed, video_engine="minimax_h3")
        fields.apply_to_card(card, parsed)
        self.assertEqual(segment["scene_defaults"]["performance_style"], "rock_punk")
        self.assertEqual(card["scene_defaults"]["performance_style"], "rock_punk")
        parsed, _ = fields.parse_scene_card_patch({"scene_defaults": {"camera_motion_speed": 0}})
        fields.apply_to_segment(segment, parsed, video_engine="minimax_h3")
        fields.apply_to_card(card, parsed)
        self.assertEqual(segment["scene_defaults"]["performance_style"], "rock_punk")
        self.assertEqual(card["scene_defaults"]["camera_motion_speed"], 0)

    def test_timeline_cards_keep_verse_and_chorus_independent(self):
        session = {"builder_storyboard_defaults": {"performance_style": "pop_performance"},
                   "segments": [
                       {"id": "verse", "start": 0, "end": 8,
                        "scene_defaults": {"performance_style": "rock_punk", "camera_motion_speed": 2}},
                       {"id": "chorus", "start": 8, "end": 16,
                        "scene_defaults": {"performance_style": "metal_screaming", "camera_motion_speed": 8}},
                   ]}
        cards = sync.scene_cards(session)
        self.assertEqual(cards[0]["performance_style"], "rock_punk")
        self.assertEqual(cards[1]["performance_style"], "metal_screaming")
        self.assertEqual(cards[0]["scene_defaults"]["camera_motion_speed"], 2)
        self.assertEqual(cards[1]["scene_defaults"]["camera_motion_speed"], 8)
        self.assertEqual(session["builder_storyboard_defaults"]["performance_style"], "pop_performance")

    def test_headless_cut_plan_uses_each_scene_and_preserves_zero(self):
        session = {"builder_storyboard_defaults": {"minimax_h3_cut_frequency": 9}}
        verse = {"start": 0, "end": 8, "scene_defaults": {"minimax_h3_cut_frequency": 0}}
        chorus = {"start": 8, "end": 16, "scene_defaults": {"minimax_h3_cut_frequency": 8}}
        self.assertEqual(assembly.scene_cut_plan(verse, session)["frequency"], 0)
        self.assertEqual(assembly.scene_cut_plan(chorus, session)["frequency"], 8)
        self.assertEqual(session["builder_storyboard_defaults"]["minimax_h3_cut_frequency"], 9)

    def test_storyboard_disk_save_and_reload_preserves_per_scene_defaults(self):
        from test_storyboard_persistence import load_persistence
        store = load_persistence()
        record = {"camera_motion_speed": 0, "character_motion_speed": 8,
                  "minimax_h3_cut_frequency": 0, "performance_style": "metal_screaming",
                  "facial_performance_custom": "Brows tense, eyes narrow"}
        with tempfile.TemporaryDirectory() as folder:
            store["_save_storyboard"]({"project_folder": folder,
                "storyboard": {"scenes": [{"id": "a", "scene_defaults": record}]}})
            loaded = store["_load_storyboard"]({"project_folder": folder})
            self.assertEqual(loaded["scenes"][0]["scene_defaults"], record)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_scoped_ui(self):
        result = subprocess.run(["node", "--test", str(ROOT / "tests/scene_defaults_auto_apply.cjs")],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
