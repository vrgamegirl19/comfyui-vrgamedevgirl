import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, wanted, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = dict(namespace or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


PROMPTS = load_functions(
    ROOT / "llm" / "prompts" / "storyboard.py",
    {"_storyboard_scene_beat_output_rules", "_storyboard_flf_endpoint_schema", "_storyboard_story_arc_schema"},
)
STORY_LAYER = load_functions(
    ROOT / "storyboard/story_layer.py",
    {"_cap_story_arc_words", "_normalize_story_arc_output", "_parse_flf_endpoint_json"},
    {"re": re, "json": json},
)


class StoryboardRetrySchemaTests(unittest.TestCase):
    def test_flf_schema_matches_the_keys_the_output_rules_ask_for(self):
        rules = PROMPTS["_storyboard_scene_beat_output_rules"](True, 60)
        asked = re.search(r"exactly these string keys: ([^.]+)\.", rules).group(1).split(", ")
        schema = PROMPTS["_storyboard_flf_endpoint_schema"]()
        self.assertEqual(schema["required"], asked)
        self.assertEqual(list(schema["properties"]), asked)
        self.assertFalse(schema["additionalProperties"])

    def test_flf_schema_reply_parses(self):
        reply = json.dumps({key: f"{key} text" for key in PROMPTS["_storyboard_flf_endpoint_schema"]()["required"]})
        self.assertEqual(STORY_LAYER["_parse_flf_endpoint_json"](reply)["flf_end_state"], "flf_end_state text")

    def test_story_arc_schema_reply_rebuilds_into_the_required_headings(self):
        labels = ["Intro", "Verse 1", "Chorus", "Verse 2", "Chorus 2"]
        schema = PROMPTS["_storyboard_story_arc_schema"](labels)
        self.assertEqual(schema["required"], labels)
        self.assertEqual(list(schema["properties"]), labels)
        sections = {label: f"The fox girl with fennec ears crosses the neon rooftop in the {label.lower()}." for label in labels}
        # Same rebuild as the schema retry in _build_story_layer_arc.
        text = "\n\n".join(f"{label}:\n{str(sections.get(label) or '').strip()}" for label in labels)
        normalized = STORY_LAYER["_normalize_story_arc_output"](text, labels, 100, "LM Studio")
        self.assertEqual(re.findall(r"(?m)^([^\n:]+):$", normalized), labels)


if __name__ == "__main__":
    unittest.main()
