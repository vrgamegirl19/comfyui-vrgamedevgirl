import ast
import json
import os
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "VRGDG_MusicVideoBuilderNodes.py"
UI_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


def load_save_helpers():
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    names = {
        "_safe_project_name",
        "_session_path",
        "_srt_path",
        "_context_folder",
        "_atomic_write_text",
        "_atomic_write_json",
        "_fallback_project_context_text",
        "_save_project_context_files",
        "_validate_saved_project",
        "_save_reference_descriptions",
    }
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "json": json,
        "os": os,
        "re": re,
        "tempfile": tempfile,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace


class BuilderReferenceSaveAndCueTests(unittest.TestCase):
    def test_reference_manifest_context_and_mappings_round_trip(self):
        helpers = load_save_helpers()
        refs = {
            "subjects": [{"id": "singer", "name": "Singer", "description": "Dark hair", "image": {"path": "singer.png"}}],
            "locations": [{"id": "entry", "name": "Entry", "description": "Deadbolted entry", "image": {"path": "entry.png"}}],
            "subject_scene_map": {"scene-1": ["singer"]},
            "performer_scene_map": {"scene-1": ["singer"]},
            "scene_map": {"scene-1": "entry"},
            "scene_trigger_map": {"scene-1": "deadbolt"},
        }
        session = {
            "segments": [{"id": "scene-1", "scene_summary": "Singer waits at the locked door."}],
            "flux_reference_builder": refs,
            "builder_story_layer": {"image_world_style": "cinematic realism"},
            "project_context_files": {},
        }
        with tempfile.TemporaryDirectory() as folder:
            helpers["_atomic_write_json"](helpers["_session_path"](folder), session)
            helpers["_atomic_write_text"](helpers["_srt_path"](folder), "1\n00:00:00,000 --> 00:00:08,000\nScene\n")
            context_paths = helpers["_save_project_context_files"](folder, session)
            manifest_path = helpers["_save_reference_descriptions"](folder, session)
            helpers["_validate_saved_project"](folder, session, context_paths)

            self.assertTrue(all(Path(path).read_text(encoding="utf-8").strip() for path in context_paths))
            self.assertEqual(json.loads(Path(manifest_path).read_text(encoding="utf-8")), refs)
            self.assertEqual(json.loads(Path(manifest_path).read_text(encoding="utf-8"))["performer_scene_map"], {"scene-1": ["singer"]})

    def test_final_prompt_assembly_enforces_cue_map_and_request_is_audited(self):
        self.assertIn("enforceMiniMaxH3CueOnShotDescription(segment, descriptions[index], index)", UI_SOURCE)
        self.assertIn('cue.type === "instrumental" && dialogueTags.length', UI_SOURCE)
        self.assertIn("segment.minimax_h3_llm_request_audit_path", UI_SOURCE)
        self.assertIn("Array.isArray(segment.lyric_cue_map) && segment.lyric_cue_map.length", UI_SOURCE)
        self.assertIn("the explicit singer cue map was dropped before LLM prompting", UI_SOURCE)
        self.assertIn("canonicalCutPlan?.cue_driven ? canonicalCutPlan.instruction", UI_SOURCE)
        self.assertIn("miniMaxH3CapitalizeCueText(miniMaxH3PunctuatedCueText(cue.text))", UI_SOURCE)
        node_source = NODE_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"actual_llm_instruction": prompt', node_source)
        self.assertIn('"raw_llm_response"', node_source)

    def test_complete_manual_timing_returns_before_transcription(self):
        start = UI_SOURCE.index("async function ensureAutoTimedSingerCuesBeforePrompt")
        end = UI_SOURCE.index("async function runMiniMaxH3PromptGeneration", start)
        source = UI_SOURCE[start:end]
        self.assertLess(source.index("if (existingTimingComplete) return true;"), source.index("autoTimeMiniMaxSingerCuesForSegment(segment)"))


if __name__ == "__main__":
    unittest.main()
