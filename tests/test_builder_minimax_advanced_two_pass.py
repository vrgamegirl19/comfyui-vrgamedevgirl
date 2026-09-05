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

    _OLD_SPATIAL_INPUTS = (
        "upscale_width", "upscale_height", "tile_size_mode", "tile_width", "tile_height",
        "grid_rows", "grid_cols", "spatial_w_overlap", "spatial_h_overlap",
        "fade_width", "fade_height", "min_tile_size", "overlap_mode", "overlap_blend",
    )
    _NEW_SPATIAL_INPUTS = (
        "masked_area_noise", "brightness_match", "dynamic_fade", "dynamic_fade_min",
    )

    class _OldSpatialSplit:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {name: ("INT",) for name in BuilderMiniMaxAdvancedTwoPassTests._OLD_SPATIAL_INPUTS}}

    class _NewSpatialSplit:
        @classmethod
        def INPUT_TYPES(cls):
            required = {name: ("INT",) for name in BuilderMiniMaxAdvancedTwoPassTests._OLD_SPATIAL_INPUTS}
            required.update({name: ("INT",) for name in BuilderMiniMaxAdvancedTwoPassTests._NEW_SPATIAL_INPUTS})
            return {"required": required}

    def _advanced_prompt_namespace(self, spatial_node_class):
        module = ast.parse(RUNNER_SOURCE)
        wanted = {
            "_build_minimax_h3_advanced_2pass_api_prompt",
            "_compat_node_inputs",
            "_node_input_names",
            "_input_names_for_node",
        }
        functions = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        self.assertEqual({node.name for node in functions}, wanted)
        namespace = {
            "copy": copy,
            "inspect": __import__("inspect"),
            "math": math,
            "_MINIMAX_H3_ASPECT_RATIOS": {"16:9 (Widescreen)": (16, 9)},
            "_MMH3_DYNAMIC_FADE_MODES": {"off", "narrowing", "widening"},
            "_MMH3_SPATIAL_SPLIT_NEW_DEFAULTS": {
                "masked_area_noise": 0.0,
                "brightness_match": False,
                "dynamic_fade": "off",
                "dynamic_fade_min": 32,
            },
            "_get_comfy_node_mappings": lambda: {
                "MMH3UltimateUpscale": object(),
                "VRGDG_MiniMaxH3UltimateUpscaleParams": object(),
                "MMH3TemporalSplitParams": object(),
                "MMH3SpatialSplitParams": spatial_node_class,
            },
            "_int_payload": lambda payload, key, default, low, high: max(low, min(high, int(payload.get(key, default)))),
            "_float_payload": lambda payload, key, default, low, high: max(low, min(high, float(payload.get(key, default)))),
            "_bool_payload": lambda payload, key, default=False: bool(payload.get(key, default)),
            "_first_payload_value": lambda payload, *keys, default=None: next(
                (payload.get(key) for key in keys if key in payload and payload.get(key) is not None),
                default,
            ),
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
        exec(compile(ast.Module(body=functions, type_ignores=[]), "advanced_two_pass", "exec"), namespace)
        return namespace

    def test_generated_advanced_graph_has_no_dangling_node_links(self):
        namespace = self._advanced_prompt_namespace(object())
        result = namespace["_build_minimax_h3_advanced_2pass_api_prompt"]({})
        prompt = result["prompt"]
        self.assertEqual(prompt["136"]["inputs"]["width"], ["9300", 0])
        self.assertEqual(prompt["136"]["inputs"]["prompt"], ["138", 0])
        self.assertEqual(prompt["9302"]["inputs"]["width"], ["9301", 0])
        self.assertEqual(prompt["9302"]["inputs"]["prompt"], "")
        self.assertEqual(prompt["9306"]["inputs"]["conditioning"], ["9302", 0])
        self.assertEqual(prompt["9306"]["inputs"]["model"], prompt["192"]["inputs"]["model"])
        self.assertEqual(prompt["142"]["inputs"]["images"], ["122", 0])
        self.assertEqual(prompt["9308"]["inputs"]["images"], ["9307", 0])
        self.assertEqual(prompt["9304"]["inputs"]["chunk_length"], 85)
        self.assertEqual(prompt["9305"]["inputs"]["tile_size_mode"], "rows_cols")
        self.assertEqual(prompt["9305"]["inputs"]["grid_rows"], 2)
        self.assertEqual(prompt["9305"]["inputs"]["grid_cols"], 2)
        self.assertEqual(prompt["9305"]["inputs"]["fade_width"], 64)
        self.assertEqual(prompt["9305"]["inputs"]["overlap_mode"], "later")
        for name in self._NEW_SPATIAL_INPUTS:
            self.assertNotIn(name, prompt["9305"]["inputs"])
        self.assertEqual(result["advanced_two_pass"]["pass2_width"], 1920)
        self.assertEqual(result["advanced_two_pass"]["pass2_height"], 1088)

        result = namespace["_build_minimax_h3_advanced_2pass_api_prompt"]({
            "advanced_pass1_megapixels": 0.4,
            "advanced_pass2_megapixels": 2.1,
            "pass2_prompt": "sharp focus, clear details",
        })
        self.assertEqual(result["advanced_two_pass"]["pass2_width"], 1984)
        self.assertEqual(result["advanced_two_pass"]["pass2_height"], 1120)
        self.assertEqual(result["prompt"]["9302"]["inputs"]["prompt"], "sharp focus, clear details")
        self.assertEqual(result["prompt"]["136"]["inputs"]["prompt"], ["138", 0])

        dangling = []
        for node_id, node in prompt.items():
            for input_name, value in (node.get("inputs") or {}).items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    if str(value[0]) not in prompt:
                        dangling.append((node_id, input_name, value[0]))
        self.assertEqual(dangling, [])

    def test_spatial_split_omits_new_inputs_on_old_mmh3_node(self):
        namespace = self._advanced_prompt_namespace(self._OldSpatialSplit)
        prompt = namespace["_build_minimax_h3_advanced_2pass_api_prompt"]({})["prompt"]
        for name in self._OLD_SPATIAL_INPUTS:
            self.assertIn(name, prompt["9305"]["inputs"])
        for name in self._NEW_SPATIAL_INPUTS:
            self.assertNotIn(name, prompt["9305"]["inputs"])

    def test_spatial_split_uses_new_mmh3_defaults_when_node_declares_them(self):
        namespace = self._advanced_prompt_namespace(self._NewSpatialSplit)
        prompt = namespace["_build_minimax_h3_advanced_2pass_api_prompt"]({})["prompt"]
        inputs = prompt["9305"]["inputs"]
        for name in self._OLD_SPATIAL_INPUTS:
            self.assertIn(name, inputs)
        self.assertEqual(inputs["masked_area_noise"], 0.0)
        self.assertIs(inputs["brightness_match"], False)
        self.assertEqual(inputs["dynamic_fade"], "off")
        self.assertEqual(inputs["dynamic_fade_min"], 32)

    def test_builder_ui_exposes_and_persists_pass2_prompt(self):
        self.assertIn('makeField("2nd Pass Prompt", miniMaxPass2Prompt)', BUILDER_SOURCE)
        self.assertIn("miniMaxPass2PromptField.style.display = threePass ? \"flex\" : \"none\"", BUILDER_SOURCE)
        self.assertIn("minimax_h3_pass2_prompt: \"\"", BUILDER_SOURCE)
        self.assertIn("if (segment.minimax_h3_pass2_prompt == null) segment.minimax_h3_pass2_prompt = \"\"", BUILDER_SOURCE)
        self.assertIn("pass2_prompt: String(segment?.minimax_h3_pass2_prompt || \"\")", BUILDER_SOURCE)
        self.assertIn('pass2_prompt: String(segment?.minimax_h3_pass2_prompt || ""),', BUILDER_SOURCE)
        self.assertIn("Object.prototype.hasOwnProperty.call(scene, \"minimax_h3_pass2_prompt\")", BUILDER_SOURCE)
        self.assertIn('"minimax_h3_prompt", "minimax_h3_pass2_prompt"', BUILDER_SOURCE)

if __name__ == "__main__":
    unittest.main()
