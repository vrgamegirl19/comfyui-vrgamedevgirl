import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
NODES = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(encoding="utf-8")


class BuilderSaveOrderTests(unittest.TestCase):
    def test_all_session_saves_use_ordered_save_queue(self):
        self.assertIn("let builderSessionSaveQueue = Promise.resolve();", UI)
        self.assertIn("function saveBuilderSessionJson(payload", UI)
        self.assertEqual(UI.count('postJson("/vrgdg/music_builder/save_session"'), 1)
        self.assertIn('saveBuilderSessionJson({', UI)

    def test_manual_performer_assignment_autosaves(self):
        self.assertIn('autoSaveSessionQuiet("performer assignment changed")', UI)
        self.assertIn("performer_scene_map", UI)

    def test_bulk_lyric_restore_does_not_override_user_edits(self):
        self.assertNotIn("Prevented accidental bulk lyric clearing", NODES)
        self.assertNotIn("if not bool(session.get(\"allow_bulk_lyric_clear\"))", NODES)


if __name__ == "__main__":
    unittest.main()
