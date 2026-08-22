import ast
import os
import tempfile
import types
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "LLM.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def class_node(name):
    return next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == name)


class QwenGgufNodeTests(unittest.TestCase):
    def test_qwen_node_is_registered_and_has_generic_presets(self):
        node = class_node("VRGDG_QwenGGUF")
        source = ast.get_source_segment(SOURCE, node)
        self.assertIn('"unsloth/Qwen3.8-27B-GGUF"', source)
        self.assertIn('"custom"', source)
        self.assertIn('"qwen-mmproj-BF16.gguf"', source)
        self.assertIn("class VRGDG_QwenGGUF(VRGDG_GeneralGGUF)", source)

    def test_qwen_template_and_stop_tokens_are_not_gemma_tokens(self):
        source = ast.get_source_segment(SOURCE, class_node("VRGDG_QwenGGUF"))
        self.assertIn("<|im_start|>", source)
        self.assertIn("<|im_end|>", source)
        self.assertIn('"enable_thinking": enable_thinking', source)
        self.assertIn("enable_thinking is defined and enable_thinking", source)
        self.assertNotIn("<start_of_turn>", source)

    def test_mapping_and_display_name_exist(self):
        self.assertIn('"VRGDG_QwenGGUF": VRGDG_QwenGGUF', SOURCE)
        self.assertIn('"VRGDG_QwenGGUF": "🧠 VRGDG Qwen GGUF 🧠"', SOURCE)

    def test_qwen_discovery_scans_every_registered_llm_root_and_qwen_folder(self):
        node = class_node("VRGDG_QwenGGUF")
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            first.mkdir()
            nested = second / "Qwen"
            nested.mkdir(parents=True)
            (first / "Qwen3-small.gguf").write_bytes(b"model")
            (nested / "model-Q4_K_M.gguf").write_bytes(b"model")

            namespace = {
                "os": os,
                "VRGDG_GeneralGGUF": object,
                "folder_paths": types.SimpleNamespace(
                    get_folder_paths=lambda _category: [str(first), str(second)],
                    models_dir=str(first.parent),
                ),
            }
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "LLM.py"), "exec"), namespace)
            qwen = namespace["VRGDG_QwenGGUF"]

            choices = qwen._list_local_qwen_gguf()

            self.assertIn("Qwen3-small.gguf", choices)
            self.assertIn(os.path.join("Qwen", "model-Q4_K_M.gguf"), choices)
            self.assertEqual(
                qwen._resolve_dropdown_path(os.path.join("Qwen", "model-Q4_K_M.gguf"), qwen.MISSING_MODEL_OPTION),
                str(nested / "model-Q4_K_M.gguf"),
            )


if __name__ == "__main__":
    unittest.main()
