import ast
import copy
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
RUNNER_SOURCE = (ROOT / "VRGDG_WorkflowRunnerNodes.py").read_text(encoding="utf-8")


class BuilderMiniMaxAdvancedTwoPassTests(unittest.TestCase):
    def test_former_three_pass_button_is_advanced_two_pass(self):
        self.assertIn('makeButton("Ref to Video\\n2 Pass Advanced")', BUILDER_SOURCE)
        self.assertIn(
            '"/vrgdg/workflow_runner/build_minimax_h3_advanced_2pass_prompt"',
            BUILDER_SOURCE,
        )

    def test_vram_presets_match_mmh3_starting_points(self):
        for text in (
            '"8gb": { tile: 352, chunk: 51 }',
            '"12gb": { tile: 512, chunk: 85 }',
            '"16gb": { tile: 576, chunk: 119 }',
            '"24gb": { tile: 672, chunk: 153 }',
        ):
            self.assertIn(text, BUILDER_SOURCE)
        self.assertIn('advanced_two_pass_pass2_steps: 1', BUILDER_SOURCE)
        self.assertIn('advanced_two_pass_pass2_sampler: "sa_solver"', BUILDER_SOURCE)
        self.assertIn('advanced_two_pass_pass2_scheduler: "simple"', BUILDER_SOURCE)

    def test_resolutions_are_visible_and_expert_controls_are_collapsed(self):
        self.assertIn('"Pass 1 resolution"', BUILDER_SOURCE)
        self.assertIn('"Pass 2 resolution"', BUILDER_SOURCE)
        self.assertIn('makeSettingsSection("Pass Sampling (Advanced)"', BUILDER_SOURCE)
        self.assertIn('makeSettingsSection("Hidden MMH3 Advanced Settings"', BUILDER_SOURCE)

    def test_hidden_prompt_uses_independent_resolutions_and_mmh3_nodes(self):
        start = RUNNER_SOURCE.index("def _build_minimax_h3_advanced_2pass_api_prompt")
        end = RUNNER_SOURCE.index("def _remap_api_prompt_references", start)
        source = RUNNER_SOURCE[start:end]
        self.assertEqual(source.count('"class_type": "ResolutionSelector"'), 2)
        for node_type in (
            "VRGDG_MiniMaxH3UltimateUpscaleParams",
            "MMH3TemporalSplitParams",
            "MMH3SpatialSplitParams",
            "MMH3UltimateUpscale",
        ):
            self.assertIn(f'"class_type": "{node_type}"', source)
        self.assertIn('_set_api_input(prompt, "122", "samples", ["9306", 0])', source)

    def test_existing_two_pass_route_is_preserved(self):
        self.assertIn(
            '@server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_2pass_prompt")',
            RUNNER_SOURCE,
        )
        self.assertIn(
            '@server_instance.routes.post("/vrgdg/workflow_runner/build_minimax_h3_advanced_2pass_prompt")',
            RUNNER_SOURCE,
        )

    def test_generated_advanced_graph_has_no_dangling_node_links(self):
        module = ast.parse(RUNNER_SOURCE)
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_minimax_h3_advanced_2pass_api_prompt"
        )
        namespace = {
            "copy": copy,
            "math": math,
            "_MINIMAX_H3_ASPECT_RATIOS": {"16:9 (Widescreen)": (16, 9)},
            "_get_comfy_node_mappings": lambda: {
                name: object() for name in (
                    "MMH3UltimateUpscale",
                    "VRGDG_MiniMaxH3UltimateUpscaleParams",
                    "MMH3TemporalSplitParams",
                    "MMH3SpatialSplitParams",
                )
            },
            "_int_payload": lambda payload, key, default, low, high: max(low, min(high, int(payload.get(key, default)))),
            "_float_payload": lambda payload, key, default, low, high: max(low, min(high, float(payload.get(key, default)))),
        }

        def set_api_input(prompt, node_id, input_name, value):
            prompt[str(node_id)]["inputs"][input_name] = value

        template_path = ROOT / "Workflows" / "UsedForUIDoNotTouch" / "minimax_audio_driven_builder_latent_upscale_2pass_api.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        namespace["_set_api_input"] = set_api_input
        namespace["_build_minimax_h3_2pass_api_prompt"] = lambda payload: {
            "prompt": copy.deepcopy(template),
            "two_pass": {},
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "advanced_two_pass", "exec"), namespace)
        result = namespace["_build_minimax_h3_advanced_2pass_api_prompt"]({})
        prompt = result["prompt"]
        self.assertEqual(prompt["136"]["inputs"]["width"], ["9300", 0])
        self.assertEqual(prompt["9302"]["inputs"]["width"], ["9301", 0])
        self.assertEqual(prompt["9306"]["inputs"]["model"], prompt["192"]["inputs"]["model"])
        self.assertEqual(prompt["142"]["inputs"]["images"], ["122", 0])
        self.assertEqual(prompt["9308"]["inputs"]["images"], ["9307", 0])
        self.assertEqual(prompt["9304"]["inputs"]["chunk_length"], 85)
        self.assertEqual(prompt["9305"]["inputs"]["tile_size_mode"], "rows_cols")
        self.assertEqual(prompt["9305"]["inputs"]["grid_rows"], 2)
        self.assertEqual(prompt["9305"]["inputs"]["grid_cols"], 2)
        self.assertEqual(prompt["9305"]["inputs"]["fade_width"], 64)
        self.assertEqual(prompt["9305"]["inputs"]["overlap_mode"], "later")
        self.assertEqual(result["advanced_two_pass"]["pass2_width"], 1920)
        self.assertEqual(result["advanced_two_pass"]["pass2_height"], 1088)

        result = namespace["_build_minimax_h3_advanced_2pass_api_prompt"]({
            "advanced_pass1_megapixels": 0.4,
            "advanced_pass2_megapixels": 2.1,
        })
        self.assertEqual(result["advanced_two_pass"]["pass2_width"], 1984)
        self.assertEqual(result["advanced_two_pass"]["pass2_height"], 1120)

        dangling = []
        for node_id, node in prompt.items():
            for input_name, value in (node.get("inputs") or {}).items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    if str(value[0]) not in prompt:
                        dangling.append((node_id, input_name, value[0]))
        self.assertEqual(dangling, [])


if __name__ == "__main__":
    unittest.main()
