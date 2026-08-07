import ast
import importlib
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)
RUNNER_SOURCE = (ROOT / "VRGDG_WorkflowRunnerNodes.py").read_text(
    encoding="utf-8"
)


def _load_compat_class(folder_paths):
    tree = ast.parse(RUNNER_SOURCE)
    compat_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VRGDG_MiniMaxH3TurboLoRACompat"
    )
    namespace = {
        "folder_paths": folder_paths,
        "importlib": importlib,
        "sys": sys,
    }
    exec(compile(ast.Module(body=[compat_class], type_ignores=[]), str(ROOT), "exec"), namespace)
    return namespace["VRGDG_MiniMaxH3TurboLoRACompat"]


class BuilderMiniMaxTurboTests(unittest.TestCase):
    def test_video_settings_exposes_turbo_checkbox_picker_and_strength(self):
        self.assertIn(
            'makeCheckbox("Use MiniMax-H3 Turbo LoRA (4-step)"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'turbo_lora_name: "minimax_h3_turbo_4step_ema_ckpt850.safetensors"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'const miniMaxTurboSection = makeSettingsSection("Turbo acceleration"',
            BUILDER_SOURCE,
        )

    def test_turbo_settings_follow_existing_global_or_locked_scene_settings(self):
        self.assertIn(
            "use_turbo_lora: turboEnabled",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "turbo_lora_name: miniMaxTurboLoraPicker.input.value",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "turbo_lora_strength: miniMaxTurboLoraStrength.value",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "if (segment?.use_scene_minimax_h3_settings)",
            BUILDER_SOURCE,
        )

    def test_render_payload_sends_turbo_settings_to_hidden_workflow_builder(self):
        self.assertIn(
            "use_turbo_lora: miniMaxSettings.use_turbo_lora",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "turbo_lora_name: miniMaxSettings.turbo_lora_name",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "turbo_lora_strength: miniMaxSettings.turbo_lora_strength",
            BUILDER_SOURCE,
        )

    def test_hidden_api_graph_injects_both_required_custom_nodes(self):
        self.assertIn(
            '"class_type": "VRGDG_MiniMaxH3TurboLoRACompat"',
            RUNNER_SOURCE,
        )
        self.assertIn(
            '"class_type": "MiniMaxH3TurboSampler"',
            RUNNER_SOURCE,
        )
        self.assertIn(
            '_set_api_input(prompt, guider_id, "model", [turbo_lora_id, 0])',
            RUNNER_SOURCE,
        )
        self.assertIn(
            '_set_api_input(prompt, scheduler_id, "model", [turbo_lora_id, 0])',
            RUNNER_SOURCE,
        )
        self.assertIn(
            '_set_api_input(prompt, sampler_advanced_id, "sampler", [turbo_sampler_id, 0])',
            RUNNER_SOURCE,
        )

    def test_pruned_reference_audio_uses_layout_aware_compatibility_adapter(self):
        self.assertIn(
            'class VRGDG_MiniMaxH3TurboLoRACompat:',
            RUNNER_SOURCE,
        )
        self.assertIn(
            'kind == "ref_audio" for _, _, kind in segments',
            RUNNER_SOURCE,
        )
        self.assertIn(
            'times.add(max(t_audio, audio_aug))',
            RUNNER_SOURCE,
        )
        self.assertIn(
            'upstream_supports_audio = "has_aud_cond" in inspect.signature(unique_t).parameters',
            RUNNER_SOURCE,
        )

    def test_current_upstream_adaln_forward_api_is_supported(self):
        calls = {"patches": [], "debug": []}

        class Weight:
            def __mul__(self, _strength):
                return self

        class DiffusionModel:
            use_adaln_curves = True
            sigma_shift_video = 1.0
            sigma_shift_audio = 1.0

        class NewModel:
            def __init__(self):
                self.model = types.SimpleNamespace(diffusion_model=DiffusionModel())

            def add_wrapper_with_key(self, *_args):
                pass

            def get_model_object(self, key):
                return ("base", key)

            def add_object_patch(self, key, value):
                calls["patches"].append((key, value))

        new_model = NewModel()

        class Model:
            model = types.SimpleNamespace(diffusion_model=DiffusionModel())

            def clone(self):
                return new_model

        upstream_name = "_vrgdg_test_minimax_turbo"
        upstream = types.ModuleType(upstream_name)
        upstream.SHIFT_V = 1.0
        upstream.SHIFT_A = 1.0
        upstream._unique_t = lambda timestep, shift_v, shift_a, has_vis_cond: []
        upstream._time_shift_sigma = lambda sigma, shift_v, shift_a: sigma
        upstream._egrid = lambda: object()
        upstream._interp_egrid = lambda *args: object()
        upstream._apply_bypass_lora = lambda *args: 1
        upstream._make_adaln_forward = (
            lambda base, a, b, shared: ("forward", base, a, b, shared)
        )

        def add_debug(_model, _diffusion_model, tag, mode):
            calls["debug"].append((tag, mode))

        upstream._add_dbg_wrapper = add_debug
        upstream.comfy = types.SimpleNamespace(
            utils=types.SimpleNamespace(
                load_torch_file=lambda *_args, **_kwargs: {
                    "blocks.0.attn.qkv_proj.lora_A.weight": Weight(),
                    "blocks.0.attn.qkv_proj.lora_B.weight": Weight(),
                    "blocks.0.adaln_proj.linear.lora_A.weight": Weight(),
                    "blocks.0.adaln_proj.linear.lora_B.weight": Weight(),
                }
            ),
            patcher_extension=types.SimpleNamespace(
                WrappersMP=types.SimpleNamespace(DIFFUSION_MODEL="diffusion")
            ),
        )

        class UpstreamNode:
            __module__ = upstream_name

            def apply_lora(self, *_args):
                raise AssertionError("the stale upstream audio path must not be delegated")

        fake_nodes = types.ModuleType("nodes")
        fake_nodes.NODE_CLASS_MAPPINGS = {"MiniMaxH3TurboLoRA": UpstreamNode}
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda _kind: ["turbo.safetensors"],
            get_full_path=lambda _kind, _name: "turbo.safetensors",
        )
        compat_class = _load_compat_class(folder_paths)

        old_nodes = sys.modules.get("nodes")
        old_upstream = sys.modules.get(upstream_name)
        sys.modules["nodes"] = fake_nodes
        sys.modules[upstream_name] = upstream
        try:
            result = compat_class().apply_lora(Model(), "turbo.safetensors", 1.0)
        finally:
            if old_nodes is None:
                sys.modules.pop("nodes", None)
            else:
                sys.modules["nodes"] = old_nodes
            if old_upstream is None:
                sys.modules.pop(upstream_name, None)
            else:
                sys.modules[upstream_name] = old_upstream

        self.assertEqual(result, (new_model,))
        self.assertEqual(calls["patches"][0][0], "diffusion_model.blocks.0.adaln_proj.forward")
        self.assertEqual(calls["patches"][0][1][0], "forward")
        self.assertEqual(calls["debug"], [("pruned-ref-audio-compat", "bypass")])

    def test_turbo_forces_required_sampler_but_allows_experimental_low_steps(self):
        self.assertIn(
            '_set_api_input(prompt, scheduler_id, "scheduler", "simple")',
            RUNNER_SOURCE,
        )
        self.assertIn(
            'turbo_steps = _int_payload(payload, "steps", 4, 1, 1000)',
            RUNNER_SOURCE,
        )
        self.assertIn(
            '_set_api_input(prompt, scheduler_id, "steps", turbo_steps)',
            RUNNER_SOURCE,
        )
        self.assertIn(
            '"effective_sampler_name": "MiniMaxH3TurboSampler"',
            RUNNER_SOURCE,
        )

    def test_turbo_ui_keeps_steps_and_easy_cache_available(self):
        self.assertIn(
            'miniMaxSteps.disabled = false;',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'miniMaxSteps.min = "1";',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Steps defaults to 4 when Turbo is switched on and remains editable down to 1 for experiments",
            BUILDER_SOURCE,
        )
        self.assertIn(
            'miniMaxSteps.value = "4";',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "currentSettings.steps_before_turbo",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "migrateOldTurboDefault ? 4 : rawSteps",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "miniMaxEasyCacheBypass.input.checked = true;",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "currentSettings.easy_cache_bypass_before_turbo",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "settings.use_turbo_lora && !sceneHasPreTurboEasyCache",
            BUILDER_SOURCE,
        )
        self.assertIn(
            'model_ref = scheduler_inputs.get("model")',
            RUNNER_SOURCE,
        )

    def test_missing_extension_or_lora_produces_actionable_error(self):
        self.assertIn(
            "Install or update ComfyUI-MiniMax-H3-Turbo, then restart ComfyUI.",
            RUNNER_SOURCE,
        )
        self.assertIn(
            "was not found in ComfyUI/models/loras",
            RUNNER_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
