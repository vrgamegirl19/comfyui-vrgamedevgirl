"""All scene-card directions reach each storyboard LLM prompt path."""

import ast
import json
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_storyboard_persistence import load_persistence

ROOT = Path(__file__).resolve().parents[1]


def load_prompt_functions() -> dict:
    """Load prompt assembly without importing ComfyUI or starting an LLM."""
    namespace = load_persistence()
    namespace.update({
        "__package__": "storyboard_test.storyboard", "__spec__": None,
        "json": json, "re": re,
        "extract_prompt_text_from_gemma_output": lambda text, *args: text,
    })
    constants = {
        "_STORYBOARD_T2I_GEMMA_INSTRUCTIONS",
        "_STORYBOARD_T2V_GEMMA_INSTRUCTIONS",
        "_STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTIONS",
        "_STORYBOARD_IMAGE_WORLD_STYLE_PRESETS",
    }
    for relative in ("llm/prompts/emotion_expression.py", "llm/prompts/storyboard.py", "storyboard/scene_prompts.py"):
        source = ROOT / relative
        tree = ast.parse(source.read_text(encoding="utf-8"))
        body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                or isinstance(node, ast.ImportFrom) and node.module == "typing"
                or (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in constants
                    for target in node.targets))]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"),
             namespace)
    return namespace


class StoryboardSceneCardContextTests(unittest.TestCase):
    def test_complete_card_reaches_image_text_video_and_vision_video(self) -> None:
        """Capture the actual runner inputs, including the I2V adapter input."""
        functions = load_prompt_functions()
        captured = []
        runner = types.ModuleType("storyboard_test.llm.builder_runner")
        video = types.ModuleType("storyboard_test.llm.video_prompt_generation")

        def run_text(payload: dict, instruction: str, **kwargs) -> tuple:
            captured.append(instruction)
            return "A quiet landscape.", {"runner": "test"}

        def run_vision(payload: dict) -> dict:
            captured.append(payload["user_notes"])
            return {"prompt": "A quiet landscape."}

        runner._run_builder_text_llm = run_text
        video._generate_builder_i2v_prompt = run_vision
        card = {
            "timeline_note": "Director sentinel",
            "notes": "Planning sentinel",
            "motion_summary": "Motion sentinel",
            "audio_direction": "Sound sentinel",
            "continuity": "Continuity sentinel",
            "speaker_assignments": [{"text": "Dialogue sentinel"}],
            "include_microphone": True,
        }
        modules = {runner.__name__: runner, video.__name__: video}
        with patch.dict(sys.modules, modules):
            for path in ("image", "text_video", "vision_video"):
                scene = {"scene_number": 1, "scene_card": card,
                         "project_video_engine": "ltx"}
                if path == "vision_video":
                    scene["image_path"] = "test-frame.png"
                payload = {"storyboard_payload": {"scenes": [scene]}}
                name = "_build_storyboard_image_prompt" if path == "image" \
                    else "_build_storyboard_video_prompt"
                functions[name](payload)
                for value in ("Director sentinel", "Planning sentinel",
                              "Motion sentinel", "Sound sentinel",
                              "Continuity sentinel", "Dialogue sentinel"):
                    self.assertIn(value, captured[-1], path)
                self.assertIn('"include_microphone": true', captured[-1])
                self.assertIn("Read every populated field", captured[-1])


if __name__ == "__main__":
    unittest.main()
