import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
RUNNER_SOURCE = (ROOT / "VRGDG_WorkflowRunnerNodes.py").read_text(encoding="utf-8")


def _load_sol_patch():
    tree = ast.parse(RUNNER_SOURCE)
    wanted = {
        "_api_node_id_by_class",
        "_int_payload",
        "_float_payload",
        "_bool_payload",
        "_set_api_input",
        "_patch_minimax_h3_sol_attention",
    }
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ROOT), "exec"), namespace)
    namespace["_require_minimax_h3_sol_attention"] = lambda: None
    return namespace["_patch_minimax_h3_sol_attention"]


class BuilderMiniMaxSolAttentionTests(unittest.TestCase):
    def test_video_settings_exposes_a_separate_sol_attention_block_after_fp16(self):
        sage_block = BUILDER_SOURCE.index('makeSettingsSection("Sage Attention"')
        fp16 = BUILDER_SOURCE.index("miniMaxFp16Accumulation.wrapper", sage_block)
        sol_block = BUILDER_SOURCE.index('makeSettingsSection("Sol Attention"', fp16)
        self.assertLess(sage_block, fp16)
        self.assertLess(fp16, sol_block)
        for option in (
            "use_sol_attention",
            "sol_attention_tau",
            "sol_attention_start_percentage",
            "sol_attention_end_percentage",
            "sol_attention_min_tokens",
        ):
            self.assertIn(option, BUILDER_SOURCE)

    def test_render_payload_persists_all_sol_attention_options(self):
        payload_start = BUILDER_SOURCE.index("use_sol_attention: miniMaxSolAttention.input.checked")
        payload = BUILDER_SOURCE[payload_start : payload_start + 500]
        self.assertIn("sol_attention_tau: miniMaxSolAttentionTau.value", payload)
        self.assertIn("sol_attention_start_percentage: miniMaxSolAttentionStartPercentage.value", payload)
        self.assertIn("sol_attention_end_percentage: miniMaxSolAttentionEndPercentage.value", payload)
        self.assertIn("sol_attention_min_tokens: miniMaxSolAttentionMinTokens.value", payload)

    def test_disabled_sol_attention_does_not_change_the_prompt(self):
        patch = _load_sol_patch()
        prompt = {
            "124": {"class_type": "BasicScheduler", "inputs": {"model": ["141", 0]}},
            "126": {"class_type": "BasicGuider", "inputs": {"model": ["141", 0]}},
        }
        result = patch(prompt, {"use_sol_attention": False})
        self.assertEqual(result, {"enabled": False, "node": ""})
        self.assertNotIn("9301", prompt)

    def test_enabled_sol_attention_injects_and_connects_the_patch_node(self):
        patch = _load_sol_patch()
        prompt = {
            "124": {"class_type": "BasicScheduler", "inputs": {"model": ["141", 0]}},
            "126": {"class_type": "BasicGuider", "inputs": {"model": ["141", 0]}},
        }
        result = patch(prompt, {
            "use_sol_attention": True,
            "sol_attention_tau": 1.2,
            "sol_attention_start_percentage": 0.2,
            "sol_attention_end_percentage": 0.9,
            "sol_attention_min_tokens": 4096,
        })
        self.assertTrue(result["enabled"])
        self.assertEqual(prompt["9301"]["class_type"], "SolAttnPatch")
        self.assertEqual(prompt["9301"]["inputs"], {
            "model": ["141", 0],
            "tau": 1.2,
            "start_percent": 0.2,
            "end_percent": 0.9,
            "min_tokens": 4096,
        })
        self.assertEqual(prompt["124"]["inputs"]["model"], ["9301", 0])
        self.assertEqual(prompt["126"]["inputs"]["model"], ["9301", 0])


if __name__ == "__main__":
    unittest.main()
