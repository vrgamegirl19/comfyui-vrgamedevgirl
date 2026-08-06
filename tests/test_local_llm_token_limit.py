import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
BACKEND_SOURCE = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(encoding="utf-8")


class LocalLlmTokenLimitTests(unittest.TestCase):
    def test_local_token_limit_is_hidden_and_omitted_for_api_runner(self):
        self.assertIn('localTokenPanel.style.display = runner.value === "llm_api" ? "none" : "flex";', UI_SOURCE)
        self.assertIn('if (runner === "llm_api") return undefined;', UI_SOURCE)

    def test_forced_builtin_gemma_still_receives_local_token_limit(self):
        self.assertIn('getLlmRunnerMaxTokens({ runner: "builtin" })', UI_SOURCE)

    def test_local_token_limit_drives_gemma_context_and_is_persisted(self):
        self.assertIn("const DEFAULT_LLM_MAX_TOKENS = 8192;", UI_SOURCE)
        self.assertIn("n_ctx: getLlmRunnerMaxTokens(),", UI_SOURCE)
        self.assertNotIn("normalizeGemmaContextLimit", UI_SOURCE)
        self.assertNotIn("state.gemmaContextLimit", UI_SOURCE)
        self.assertIn('llm_max_tokens: normalizeLlmMaxTokens(state.llmMaxTokens),', UI_SOURCE)
        self.assertIn('"llm_max_tokens",', BACKEND_SOURCE)
        self.assertIn('"llm_max_tokens": 8192,', BACKEND_SOURCE)


if __name__ == "__main__":
    unittest.main()
