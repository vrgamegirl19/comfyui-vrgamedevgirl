import ast
import copy
import importlib.util
import json
import math
import unittest
from pathlib import Path

from builder_source import python_function_source, read_builder_source, read_runner_source


ROOT = Path(__file__).resolve().parents[1]


def _load_pure_module(name):
    spec = importlib.util.spec_from_file_location(f"vrgdg_{name}", ROOT / "minimax" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TILE_PLAN = _load_pure_module("tile_plan")
RESOLUTION = _load_pure_module("resolution")
BUILDER_SOURCE = read_builder_source()
RUNNER_SOURCE = read_runner_source()


class BuilderMiniMaxAdvancedTwoPassTests(unittest.TestCase):
    def test_minimax_two_pass_redo_rerolls_all_seed_paths(self):
        for assignment in (
            "settings.two_pass_pass1_seed = randomSeedValue();",
            "settings.two_pass_pass2_seed = randomSeedValue();",
            "settings.advanced_two_pass_pass1_seed = randomSeedValue();",
            "settings.advanced_two_pass_pass2_seed = randomSeedValue();",
        ):
            self.assertIn(assignment, BUILDER_SOURCE)

        self.assertIn("def _seed_payload(key, default):", RUNNER_SOURCE)
        self.assertIn("if value < 0:", RUNNER_SOURCE)
        self.assertIn("random.randrange(0, 0xFFFFFFFFFFFFFFFF + 1)", RUNNER_SOURCE)
        self.assertIn('_seed_payload("pass1_seed", -1)', RUNNER_SOURCE)
        self.assertIn('_seed_payload("pass2_seed", -1)', RUNNER_SOURCE)

    def test_former_three_pass_button_is_advanced_two_pass(self):
        self.assertIn('makeButton("2 pass advanced")', BUILDER_SOURCE)
        self.assertIn(
            '"/vrgdg/workflow_runner/build_minimax_h3_advanced_2pass_prompt"',
            BUILDER_SOURCE,
        )

    def test_vram_presets_cover_8_to_24_gb_only(self):
        # Presets are tile-area targets (megapixels) plus chunk length; the plan node derives the grid from
        # the output resolution. 2 Pass Advanced is for cards up to 24 GB.
        for text in (
            '"8gb": { tileMegapixels: 0.2, chunk: 51, overlap: 128 }',
            '"12gb": { tileMegapixels: 0.3, chunk: 85, overlap: 128 }',
            '"16gb": { tileMegapixels: 0.43, chunk: 119, overlap: 128 }',
            '"24gb": { tileMegapixels: 0.65, chunk: 153, overlap: 160 }',
        ):
            self.assertIn(text, BUILDER_SOURCE)
        presets_block = BUILDER_SOURCE[BUILDER_SOURCE.index("const MINIMAX_H3_VRAM_PRESETS = {"):]
        presets_block = presets_block[:presets_block.index("};")]
        self.assertNotIn("32gb", presets_block)
        # The JS preview and the Python plan must use the same table.
        for key, preset in TILE_PLAN.VRAM_PRESETS.items():
            self.assertIn(
                f'"{key}": {{ tileMegapixels: {preset["tile_megapixels"]}, chunk: {preset["chunk"]}, overlap: {preset["overlap"]} }}',
                BUILDER_SOURCE,
            )
        self.assertIn('advanced_two_pass_pass2_steps: 1', BUILDER_SOURCE)
        self.assertIn('advanced_two_pass_pass2_sampler: "sa_solver"', BUILDER_SOURCE)
        self.assertIn('advanced_two_pass_pass2_scheduler: "simple"', BUILDER_SOURCE)

    def test_tiling_settings_are_derived_so_no_tile_defaults_remain(self):
        # Tiles, chunks, fades and the upscaler device come from the output resolution at run time.
        defaults = BUILDER_SOURCE[BUILDER_SOURCE.index("const DEFAULT_MINIMAX_H3_SETTINGS = {"):]
        defaults = defaults[:defaults.index("};")]
        for name in (
            "tile_size_mode", "tile_width", "grid_rows", "chunk_length", "spatial_w_overlap", "fade_width",
            "min_tile_size", "overlap_mode", "overlap_blend", "brightness_match", "dynamic_fade",
            "masked_area_noise", "upscaler_device",
        ):
            self.assertNotIn(f"advanced_two_pass_{name}", defaults, name)
        self.assertNotIn("two_pass_final_width", defaults)
        self.assertNotIn("advanced_two_pass_pass2_megapixels", defaults)
        # The panel and the render payload no longer read or send them either.
        for name in ("advanced_two_pass_tile_width", "advanced_two_pass_grid_rows", "advanced_tile_width", "advanced_grid_rows"):
            self.assertNotIn(name, BUILDER_SOURCE, name)
        # The fade must stay below the overlap so a frozen seam anchor survives (fade == overlap smudges the grid).
        hidden = TILE_PLAN.HIDDEN_ADVANCED_SETTINGS
        for preset in TILE_PLAN.VRAM_PRESETS:
            plan = TILE_PLAN.plan_spatial_tiles(1920, 1088, preset)
            if plan["grid_cols"] > 1:
                self.assertLess(plan["fade_width"], plan["solved_overlap_w"], preset)
        self.assertEqual(hidden["tile_size_mode"], "rows_cols")

    def test_one_resolution_picker_and_a_vram_preset_are_all_that_is_exposed(self):
        self.assertIn('"Pass 1 resolution"', BUILDER_SOURCE)
        self.assertNotIn('"Pass 2 resolution"', BUILDER_SOURCE)
        self.assertNotIn('makeSettingsSection("Hidden MMH3 Advanced Settings"', BUILDER_SOURCE)
        self.assertIn('makeField("Output resolution", miniMaxResolutionPreset', BUILDER_SOURCE)
        self.assertIn('makeField("VRAM preset", miniMaxAdvancedVramPreset', BUILDER_SOURCE)
        self.assertIn('makeSettingsSection("Pass Sampling (Advanced)"', BUILDER_SOURCE)
        # Every pass type shows the shared resolution: nothing hides the megapixel field per mode any more.
        self.assertNotIn("miniMaxMegapixelsField.style.display", BUILDER_SOURCE)
        self.assertIn("outputFrame", BUILDER_SOURCE)

    def test_hidden_prompt_uses_independent_resolutions_and_mmh3_nodes(self):
        source = python_function_source(
            RUNNER_SOURCE,
            "_build_minimax_h3_advanced_2pass_api_prompt",
            "_save_minimax_h3_advanced_2pass_debug_workflow",
        )
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
            "_MINIMAX_H3_ASPECT_RATIOS": {"16:9 (Widescreen)": (16, 9), "9:16 (Portrait Widescreen)": (9, 16)},
            "HIDDEN_ADVANCED_SETTINGS": TILE_PLAN.HIDDEN_ADVANCED_SETTINGS,
            "PLAN_OUTPUT_NAMES": TILE_PLAN.PLAN_OUTPUT_NAMES,
            "normalize_vram_preset": TILE_PLAN.normalize_vram_preset,
            "plan_spatial_tiles": TILE_PLAN.plan_spatial_tiles,
            "frame_size": RESOLUTION.frame_size,
            "_get_comfy_node_mappings": lambda: {
                "MMH3UltimateUpscale": object(),
                "VRGDG_MiniMaxH3UltimateUpscaleParams": object(),
                "MMH3TemporalSplitParams": object(),
                "MMH3SpatialSplitParams": spatial_node_class,
                "VRGDG_MiniMaxH3SpatialTilePlan": object(),
            },
            "_patch_minimax_h3_latent_continuation": lambda _prompt, _payload: {"enabled": False},
            "_patch_minimax_h3_save_latent": lambda _prompt, _payload, _timing=None: {"enabled": False},
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
        plan_outputs = TILE_PLAN.PLAN_OUTPUT_NAMES
        self.assertEqual(prompt["9309"]["class_type"], "VRGDG_MiniMaxH3SpatialTilePlan")
        self.assertEqual(prompt["9309"]["inputs"], {"width": ["9301", 0], "height": ["9301", 1], "vram_preset": "16gb"})
        self.assertEqual(prompt["9304"]["inputs"]["chunk_length"], ["9309", plan_outputs.index("chunk_length")])
        self.assertEqual(prompt["9304"]["inputs"]["temporal_overlap"], ["9309", plan_outputs.index("temporal_overlap")])
        self.assertEqual(prompt["9304"]["inputs"]["anchor_strength"], 0.999)
        spatial = prompt["9305"]["inputs"]
        self.assertEqual(spatial["tile_size_mode"], "rows_cols")
        self.assertEqual(spatial["upscale_width"], ["9301", 0])
        for name in ("grid_rows", "grid_cols", "spatial_w_overlap", "spatial_h_overlap", "fade_width", "fade_height",
                     "min_tile_size", "tile_width", "tile_height"):
            self.assertEqual(spatial[name], ["9309", plan_outputs.index(name)], name)
        self.assertEqual((spatial["overlap_mode"], spatial["overlap_blend"]), ("later", "linear"))
        self.assertEqual(prompt["9303"]["inputs"]["device"], "cuda")
        self.assertEqual(prompt["9303"]["inputs"]["precision"], "bf16")
        for name in self._NEW_SPATIAL_INPUTS:
            self.assertNotIn(name, spatial)
        self.assertEqual(result["advanced_two_pass"]["pass2_width"], 1920)
        self.assertEqual(result["advanced_two_pass"]["pass2_height"], 1088)
        self.assertEqual(result["advanced_two_pass"]["vram_preset"], "16gb")
        self.assertEqual(result["advanced_two_pass"]["grid_rows"], 2)

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
