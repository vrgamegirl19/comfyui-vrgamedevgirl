import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLM_BACKEND = (ROOT / "LLM.py").read_text(encoding="utf-8")
BUILDER_BACKEND = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(encoding="utf-8")
LLM_MULTI_UI = (ROOT / "web" / "VRGDG_LLM_Multi_dynamic.js").read_text(encoding="utf-8")


def load_llm_multi_tables():
    tree = ast.parse(LLM_BACKEND)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VRGDG_LLM_Multi"
    )
    namespace = {}
    assignments = [
        node for node in class_node.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in {"PROVIDER_MODELS", "DEFAULT_MODEL"} for target in node.targets)
    ]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "llm_tables", "exec"), namespace)
    return namespace


TABLES = load_llm_multi_tables()


class BuilderLlmRunnerGrokModelTests(unittest.TestCase):
    def test_grok_is_the_only_xai_provider(self):
        providers = TABLES["PROVIDER_MODELS"]
        self.assertIn("grok", providers)
        self.assertNotIn("xai", providers)

    def test_grok_lists_only_current_chat_models(self):
        self.assertEqual(
            TABLES["PROVIDER_MODELS"]["grok"],
            ["grok-4.6", "grok-4.5", "grok-4.3"],
        )
        self.assertEqual(TABLES["DEFAULT_MODEL"]["grok"], "grok-4.6")
        for stale in (
            "grok-imagine-image",
            "grok-imagine-image-quality",
            "grok-imagine-video",
            "grok-imagine-video-1.5",
            "grok-4.3-latest",
            "grok-build-0.1",
        ):
            self.assertNotIn(stale, TABLES["PROVIDER_MODELS"]["grok"])

    def test_builder_skips_legacy_xai_provider_in_choices(self):
        self.assertIn('clean_provider == "xai"', BUILDER_BACKEND)
        self.assertIn('provider = "grok"', BUILDER_BACKEND)

    def test_dynamic_ui_matches_backend_grok_list(self):
        self.assertIn('grok: [\n    "grok-4.6",\n    "grok-4.5",\n    "grok-4.3",\n  ]', LLM_MULTI_UI)
        self.assertNotIn("xai:", LLM_MULTI_UI)
        self.assertNotIn("grok-imagine", LLM_MULTI_UI)


if __name__ == "__main__":
    unittest.main()
