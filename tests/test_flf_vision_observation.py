import ast
import json
import re
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "VRGDG_MusicVideoBuilderNodes.py"


def load_observation_normalizer():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_flf_vision_observation"
    )
    namespace = {"json": json, "re": re}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["_normalize_flf_vision_observation"]


normalize_observation = load_observation_normalizer()


class FLFVisionObservationTests(unittest.TestCase):
    def test_preserves_blank_line_separated_start_and_end(self):
        normalized, missing = normalize_observation(
            "START: A woman stands beside a red door.\n\n"
            "END: The woman sits beyond the same open red door."
        )
        self.assertEqual(missing, [])
        self.assertIn("START: A woman stands", normalized)
        self.assertIn("END: The woman sits", normalized)

    def test_accepts_markdown_and_numbered_labels(self):
        normalized, missing = normalize_observation(
            "1. **START:** A profile close-up under blue light.\n\n"
            "2. **END DESCRIPTION:** A wide rear view under orange light."
        )
        self.assertEqual(missing, [])
        self.assertIn("START: A profile close-up", normalized)
        self.assertIn("END: A wide rear view", normalized)

    def test_accepts_json_endpoint_descriptions(self):
        normalized, missing = normalize_observation(
            '{"start_description":"Opening portrait", "end_description":"Ending full-body shot"}'
        )
        self.assertEqual(missing, [])
        self.assertEqual(normalized, "START: Opening portrait\nEND: Ending full-body shot")

    def test_accepts_fenced_multiline_output(self):
        normalized, missing = normalize_observation(
            "```text\n### START\nOpening image details.\n\n### END\nEnding image details.\n```"
        )
        self.assertEqual(missing, [])
        self.assertEqual(normalized, "START: Opening image details.\nEND: Ending image details.")

    def test_reports_only_the_endpoint_that_is_actually_missing(self):
        normalized, missing = normalize_observation("START: Only the opening description was returned.")
        self.assertEqual(missing, ["END"])
        self.assertTrue(normalized.startswith("START:"))


if __name__ == "__main__":
    unittest.main()
