import ast
import unittest
from pathlib import Path

from builder_source import read_builder_source, read_builder_backend_source


ROOT = Path(__file__).resolve().parents[1]
BUILDER_UI = read_builder_source()
PROMPT_CREATOR_UI = (ROOT / "web" / "music_video_builder" / "prompt_creator.mjs").read_text(encoding="utf-8")
BUILDER_BACKEND = read_builder_backend_source()
PROMPT_CREATOR_BACKEND = (ROOT / "prompt_creator/nodes.py").read_text(encoding="utf-8")


def load_token_helpers():
    wanted = {
        "_llm_runner_from_payload",
        "_normalized_token_limit",
        "_runner_output_token_limit",
        "_lm_studio_context_limit",
        "_lm_studio_api_root",
        "_lm_studio_native_output_text",
        "_estimate_prompt_tokens",
        "_fit_output_to_context",
        "_runner_supports_json_schema",
    }
    constants = {"_LM_STUDIO_DEFAULT_BASE_URL", "_JSON_SCHEMA_RUNNERS", "_CONTEXT_TEMPLATE_MARGIN_TOKENS", "_MIN_OUTPUT_TOKENS"}
    tree = ast.parse(BUILDER_BACKEND)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        ):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "token_helpers", "exec"), namespace)
    return namespace


TOKEN_HELPERS = load_token_helpers()


class BuilderLlmRunnerTokenLimitTests(unittest.TestCase):
    def test_runner_modal_exposes_requested_limits(self):
        self.assertIn('makeField("Context limit / n_ctx", gemmaContextLimit)', BUILDER_UI)
        self.assertIn('makeField("Maximum output tokens", gemmaOutputTokenLimit)', BUILDER_UI)
        self.assertIn('makeField("Input context limit", lmStudioContextLimit)', BUILDER_UI)
        self.assertIn('makeField("Maximum output tokens", lmStudioOutputTokenLimit)', BUILDER_UI)

    def test_limits_reach_runner_payload_and_persist(self):
        for key in (
            "gemma_output_token_limit",
            "lmstudio_context_limit",
            "lmstudio_output_token_limit",
            "lm_studio_context_limit",
            "lm_studio_output_token_limit",
        ):
            self.assertIn(key, BUILDER_UI)
        self.assertIn('"gemma_output_token_limit"', BUILDER_BACKEND)
        self.assertIn('"lm_studio_context_limit"', BUILDER_BACKEND)
        self.assertIn('"lm_studio_output_token_limit"', BUILDER_BACKEND)

    def test_backend_replaces_direct_per_task_caps(self):
        self.assertIn(
            "max_new_tokens = _runner_output_token_limit(payload, max_new_tokens)",
            BUILDER_BACKEND,
        )
        self.assertGreaterEqual(
            BUILDER_BACKEND.count("max_new_tokens = _runner_output_token_limit"),
            13,
        )

    def test_lm_studio_receives_context_and_output_limits(self):
        self.assertIn('f"{api_root}/api/v1/chat"', BUILDER_BACKEND)
        self.assertIn("context_limit = _lm_studio_context_limit(payload)", BUILDER_BACKEND)
        self.assertIn("output_limit = _runner_output_token_limit(payload, max_new_tokens)", BUILDER_BACKEND)
        self.assertIn('"context_length": context_limit', BUILDER_BACKEND)
        self.assertIn('"max_output_tokens": output_limit', BUILDER_BACKEND)
        self.assertIn('"type": "image",', BUILDER_BACKEND)

    def test_output_budget_is_capped_to_fit_the_context_window(self):
        fit = TOKEN_HELPERS["_fit_output_to_context"]
        margin = TOKEN_HELPERS["_CONTEXT_TEMPLATE_MARGIN_TOKENS"]
        self.assertEqual(4000, fit(4000, 3000, 32768, "LM Studio"))
        self.assertEqual(8000 - 6000 - margin, fit(32768, 6000, 8000, "GGUF model"))
        with self.assertRaisesRegex(ValueError, "Raise the context limit in LLM Runner"):
            fit(2000, 7900, 8000, "GGUF model")

    def test_prompt_estimate_errs_high_for_remote_runners(self):
        estimate = TOKEN_HELPERS["_estimate_prompt_tokens"]
        text = "A fox girl with fennec ears sings on a neon rooftop. " * 40
        self.assertGreaterEqual(estimate(text), len(text) // 4)
        self.assertEqual(1, estimate(""))

    def test_json_schema_runners_exclude_the_hosted_api_runner(self):
        supports = TOKEN_HELPERS["_runner_supports_json_schema"]
        for runner in ("builtin", "qwen_local", "lm_studio", "own_server"):
            self.assertTrue(supports({"text_runner": runner}), runner)
        self.assertFalse(supports({"text_runner": "llm_api"}))

    def test_gguf_calls_are_sized_with_the_loaded_model_tokenizer(self):
        self.assertIn('prompt_tokens = len(model.tokenize(str(instruction_text or "").encode("utf-8")))', BUILDER_BACKEND)
        self.assertIn('_fit_output_to_context(int(max_new_tokens), prompt_tokens, model.n_ctx(), "GGUF model")', BUILDER_BACKEND)

    def test_configured_limits_override_old_task_defaults(self):
        output_limit = TOKEN_HELPERS["_runner_output_token_limit"]
        context_limit = TOKEN_HELPERS["_lm_studio_context_limit"]
        self.assertEqual(
            64000,
            output_limit(
                {"text_runner": "builtin", "gemma_output_token_limit": 64000},
                180,
            ),
        )
        self.assertEqual(
            48000,
            output_limit(
                {"text_runner": "lm_studio", "lmstudio_output_token_limit": 48000},
                500,
            ),
        )
        self.assertEqual(
            65536,
            context_limit({"lmstudio_context_limit": 65536}),
        )

    def test_pr_114_legacy_limit_migrates_but_api_runner_ignores_it(self):
        output_limit = TOKEN_HELPERS["_runner_output_token_limit"]
        self.assertEqual(
            12000,
            output_limit({"text_runner": "builtin", "llm_max_tokens": 12000}, 180),
        )
        self.assertEqual(
            12000,
            output_limit({"text_runner": "lm_studio", "llm_max_tokens": 12000}, 500),
        )
        self.assertEqual(
            500,
            output_limit(
                {
                    "text_runner": "llm_api",
                    "llm_max_tokens": 12000,
                    "gemma_output_token_limit": 16000,
                    "lmstudio_output_token_limit": 24000,
                },
                500,
            ),
        )
        self.assertIn("legacyLlmMaxTokens", BUILDER_UI)
        self.assertIn("data.session.llm_max_tokens", BUILDER_UI)

    def test_runner_labels_only_force_builtin_for_the_actual_local_only_draft(self):
        self.assertEqual(1, BUILDER_UI.count("forceBuiltin: true"))
        self.assertIn(
            'progress.set(`Creating draft from your notes...\\n${gemmaRunnerLine({ forceBuiltin: true })}`',
            BUILDER_UI,
        )
        self.assertIn(
            'gemma_output_token_limit: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit)',
            BUILDER_UI,
        )

    def test_prompt_creator_preserves_and_uses_runner_limits(self):
        self.assertIn("gemma_output_token_limit", PROMPT_CREATOR_UI)
        self.assertIn("lmstudio_context_limit", PROMPT_CREATOR_UI)
        self.assertIn("lmstudio_output_token_limit", PROMPT_CREATOR_UI)
        self.assertIn("_runner_output_token_limit", PROMPT_CREATOR_BACKEND)
        self.assertIn('runner_payload.get("gemma_context_limit")', PROMPT_CREATOR_BACKEND)


if __name__ == "__main__":
    unittest.main()
