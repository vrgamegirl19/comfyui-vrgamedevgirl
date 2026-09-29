import ast
import json
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "llm" / "prompts" / "storyboard.py"


def load_functions():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    wanted = {"_storyboard_dialogue_planner_instruction", "_storyboard_dialogue_plan_schema"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


FUNCTIONS = load_functions()


def prompt_shape(is_minimax):
    """The example JSON the planner prompt tells the model to return, parsed."""
    instruction = FUNCTIONS["_storyboard_dialogue_planner_instruction"](
        is_minimax=is_minimax,
        has_authoritative_script=False,
        camera_flow="balanced",
        camera_motion_speed=4,
        character_motion_speed=4,
        scene_count=4,
        story_source="A fox girl with fennec ears loses her voice before her first show.",
        script_mapper_plan_json="[none]",
        story_layer_json="{}",
        project_motion_settings_json="{}",
        subjects_json="[none provided]",
        locations_json="[none provided]",
        compact_existing_json="[none]",
    )
    shape = instruction.split("Return only valid JSON with this exact shape:\n", 1)[1].split("\n\nRequested scene count", 1)[0]
    return json.loads(shape)


class StoryboardDialoguePlanSchemaTests(unittest.TestCase):
    def test_schema_fields_match_the_prompt_shape_for_both_planner_profiles(self):
        for is_minimax in (False, True):
            with self.subTest(is_minimax=is_minimax):
                shape = prompt_shape(is_minimax)
                schema = FUNCTIONS["_storyboard_dialogue_plan_schema"](is_minimax)
                self.assertEqual(set(shape), set(schema["properties"]))
                self.assertEqual(set(schema["required"]), set(schema["properties"]))
                scene_schema = schema["properties"]["scenes"]["items"]
                self.assertEqual(set(shape["scenes"][0]), set(scene_schema["properties"]))
                self.assertEqual(set(scene_schema["required"]), set(scene_schema["properties"]))
                if is_minimax:
                    cue_schema = scene_schema["properties"]["dialogue_cues"]["items"]
                    self.assertEqual(set(shape["scenes"][0]["dialogue_cues"][0]), set(cue_schema["properties"]))

    def test_schema_rejects_extra_keys_at_every_level(self):
        schema = FUNCTIONS["_storyboard_dialogue_plan_schema"](True)
        scene_schema = schema["properties"]["scenes"]["items"]
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(scene_schema["additionalProperties"])
        self.assertFalse(scene_schema["properties"]["dialogue_cues"]["items"]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
