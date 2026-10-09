"""Timeline markers reach Story Arc planning with their original timing."""

import ast
import importlib.util
import json
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_storyboard_scene_card_context import load_prompt_functions

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str) -> types.ModuleType:
    """Load a pure helper without initializing ComfyUI."""
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NOTES = load_module("storyboard/timeline_notes.py")


def arc_functions() -> dict:
    """Load actual prompt assembly and story services, omitting GPU imports."""
    namespace = load_prompt_functions()
    namespace.update(vars(load_module("storyboard/cast_guard.py")))
    namespace.update({
        "__package__": "storyboard_test.storyboard", "__spec__": None,
        "json": json, "re": re,
        "story_arc_timeline_notes": NOTES.story_arc_timeline_notes,
    })
    source = ROOT / "storyboard/story_layer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class StoryboardTimelineNoteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenes = [
            {"scene_number": 1, "timeline_start": 0, "timeline_end": 5,
             "lyric_section": "Verse", "lyrics": "A lamp"},
            {"scene_number": 2, "timeline_start": 5, "timeline_end": 10,
             "lyric_section": "Chorus", "lyrics": "A light"},
            {"scene_number": 3, "timeline_start": 10, "timeline_end": 15},
        ]

    def test_ranges_points_boundaries_and_full_note_text(self) -> None:
        long_note = "Long direction " * 400 + "END OF NOTE"
        payload = {"scenes": self.scenes, "timeline_markers": [
            {"start": 10, "end": None, "note": "Reveal at ten"},
            {"start": 4, "end": 10, "note": long_note},
            {"start": 5, "end": "", "note": "Chorus event"},
            {"start": "bad", "note": "Invalid"},
            {"start": float("inf"), "note": "Invalid"},
            {"start": 0, "note": "  "},
        ]}
        notes = NOTES.story_arc_timeline_notes(payload)
        self.assertEqual([note["start"] for note in notes], [4, 5, 10])
        self.assertEqual([note["scene_numbers"] for note in notes], [[1, 2], [2], [3]])
        self.assertEqual(notes[0]["note"], long_note)
        self.assertIsNone(notes[1]["end"])

    def test_explicit_empty_notes_do_not_restore_removed_markers(self) -> None:
        storyboard = {"timeline_markers": [{"start": 0, "note": "Old"}]}
        self.assertEqual(len(NOTES.story_arc_timeline_notes({"storyboard": storyboard})), 1)
        self.assertEqual(NOTES.story_arc_timeline_notes({
            "timeline_markers": [], "storyboard": storyboard,
        }), [])

    def test_actual_arc_schema_retry_and_scene_entries_receive_notes(self) -> None:
        functions = arc_functions()
        for structured, retry, detail in ((True, False, "standard"),
                                          (False, True, "standard"),
                                          (False, False, "detailed")):
            with self.subTest(structured=structured, retry=retry, detail=detail):
                captured = []
                runner = types.ModuleType("storyboard_test.llm.builder_runner")

                def run_text(payload: dict, instruction: str, **kwargs) -> tuple:
                    captured.append(instruction)
                    if "Story Arc Scene" in kwargs.get("label", ""):
                        return "The lamp flickers beside the window.", {}
                    if retry and len(captured) == 1:
                        return "Verse:\nThe lamp turns off.", {}
                    sections = {"Verse": "The lamp turns off.", "Chorus": "The lamp lights up."}
                    text = json.dumps(sections) if kwargs.get("json_schema") else "\n\n".join(
                        f"{key}:\n{value}" for key, value in sections.items()
                    )
                    return text, {"runner": "test"}

                runner._run_builder_text_llm = run_text
                runner._runner_supports_json_schema = lambda payload: structured
                runner._llm_runner_display_name = lambda payload: "Test LLM"
                payload = {"scenes": self.scenes[:2], "lyrics": "[Verse]\nA lamp\n[Chorus]\nA light",
                           "story_arc_detail": detail, "timeline_markers": [
                               {"start": 5, "end": 10, "note": "REVEAL SENTINEL"},
                           ]}
                with patch.dict(sys.modules, {runner.__name__: runner}):
                    result = functions["_build_story_layer_arc"](payload)
                self.assertIn("Chorus:", result["story_arc"])
                self.assertIn("REVEAL SENTINEL", captured[0])
                matching = re.search(r'"scene_numbers":\s*(\[[^\]]*\])', captured[0])
                self.assertIsNotNone(matching)
                self.assertEqual(json.loads(matching.group(1)), [2])
                if retry:
                    self.assertIn("REVEAL SENTINEL", captured[1])
                if detail == "detailed":
                    self.assertNotIn("REVEAL SENTINEL", captured[1])
                    self.assertIn("REVEAL SENTINEL", captured[2])

    def test_script_premise_receives_timed_notes_without_changing_dialogue(self) -> None:
        functions = arc_functions()
        captured = []
        runner = types.ModuleType("storyboard_test.llm.builder_runner")

        def run_text(payload: dict, instruction: str, **kwargs) -> tuple:
            captured.append(instruction)
            return "The lamp lights up.", {}

        runner._run_builder_text_llm = run_text
        payload = {"scenes": self.scenes, "timeline_markers": [
            {"start": 5, "note": "SCRIPT EVENT SENTINEL"},
        ]}
        script = {"raw_text": "Alex: Keep these exact words.", "scene_plan": {"scenes": []}}
        with patch.dict(sys.modules, {runner.__name__: runner}):
            functions["_build_short_film_script_story_text"](payload, script, "premise")
        self.assertIn("SCRIPT EVENT SENTINEL", captured[0])
        self.assertIn("Alex: Keep these exact words.", captured[0])


if __name__ == "__main__":
    unittest.main()
