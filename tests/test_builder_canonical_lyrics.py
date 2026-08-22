import ast
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "VRGDG_MusicVideoBuilderNodes.py"
STORY_SOURCE = ROOT / "VRGDG_StoryboardBuilderNodes.py"
UI_SOURCE = ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js"


def load_save_helper():
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_context_folder", "_save_canonical_full_lyrics"}
    ]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace["_save_canonical_full_lyrics"]


class BuilderCanonicalLyricsTests(unittest.TestCase):
    def test_canonical_lyrics_are_saved_verbatim_except_outer_whitespace(self):
        save_lyrics = load_save_helper()
        lyrics = "[Intro]\nFirst line\n\n[Outro]\nLast line"
        with tempfile.TemporaryDirectory() as folder:
            path = save_lyrics(folder, f"\n{lyrics}\n")
            self.assertEqual(Path(path).read_text(encoding="utf-8"), lyrics + "\n")

    def test_blank_session_value_does_not_erase_existing_canonical_lyrics(self):
        save_lyrics = load_save_helper()
        with tempfile.TemporaryDirectory() as folder:
            path = save_lyrics(folder, "[Verse]\nKeep me")
            self.assertEqual(save_lyrics(folder, ""), "")
            self.assertEqual(Path(path).read_text(encoding="utf-8"), "[Verse]\nKeep me\n")

    def test_story_arc_prefers_project_full_lyrics_over_timeline_sections(self):
        source = STORY_SOURCE.read_text(encoding="utf-8")
        self.assertIn("lyrics = prompt_creator_lyrics or line_mapping_lyrics or timeline_lyrics", source)
        arc_start = source.index("def _build_story_layer_arc")
        compact_start = source.index("compact_scenes.append({", arc_start)
        compact_block = source[compact_start:source.index("reference_builder =", compact_start)]
        self.assertNotIn('"lyrics": normalized.get("lyrics"', compact_block)

    def test_auto_build_and_transcription_capture_reference_source(self):
        source = UI_SOURCE.read_text(encoding="utf-8")
        self.assertIn("source_text: referenceLyrics", source)
        self.assertIn("source_text: lyrics", source)


if __name__ == "__main__":
    unittest.main()
