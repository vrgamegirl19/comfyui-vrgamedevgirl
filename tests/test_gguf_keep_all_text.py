"""The Gemma node answers with one paragraph unless keep_all_text is on (H3 Studio asks for a whole story that way)."""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

gguf = importlib.import_module(f"{ROOT.name}.llm.gguf")

STORY = (
    "Title: The Last Train\n\n"
    "Act 1: Leaving\n\n"
    "Mara waits on the empty platform at dawn with her guitar case at her feet.\n\n"
    "Act 2: The Town\n\n"
    "She walks main street and finds every shop shuttered until one window glows at the end."
)


def run_node(reply, **extra):
    node = gguf.VRGDG_SuperGemmaGGUFChat()
    with patch.object(gguf.VRGDG_SuperGemmaGGUFChat, "_resolve_dropdown_path", return_value=str(Path(__file__))), \
            patch.object(gguf.VRGDG_GeneralGGUF, "_resolve_model_id", return_value="m"), \
            patch.object(gguf.VRGDG_GeneralGGUF, "_resolve_gguf_model_path", return_value="m.gguf"), \
            patch.object(gguf.VRGDG_GeneralGGUF, "_load_gguf_model", return_value=object()), \
            patch.object(gguf.VRGDG_GeneralGGUF, "_run_gguf_text_pipeline", return_value=reply):
        return node.generate_prompt(
            model_file="m.gguf", mmproj_file="", task_preset="custom", user_input="", custom_instructions="Write a story.",
            trigger_word="", image_count=0, advanced=False, unload_after_run=False, n_ctx=2048, n_gpu_layers=99,
            n_threads=8, chat_format="", temperature=0.6, top_p=0.95, max_new_tokens=800, **extra,
        )


class KeepAllTextTests(unittest.TestCase):
    def test_the_node_has_the_optional_input_last_and_off_by_default(self):
        inputs = gguf.VRGDG_SuperGemmaGGUFChat.INPUT_TYPES()
        self.assertNotIn("keep_all_text", inputs["required"])
        self.assertEqual(list(inputs["optional"])[-1], "keep_all_text")
        self.assertEqual(inputs["optional"]["keep_all_text"][0], "BOOLEAN")
        self.assertIs(inputs["optional"]["keep_all_text"][1]["default"], False)

    def test_a_node_in_a_workflow_still_returns_one_paragraph(self):
        text, _model, status = run_node(STORY)
        self.assertEqual(status, "ok")
        self.assertEqual(text, "She walks main street and finds every shop shuttered until one window glows at the end.")

    def test_keep_all_text_returns_the_whole_story(self):
        text, _model, status = run_node(STORY, keep_all_text=True)
        self.assertEqual(status, "ok")
        self.assertEqual(text, STORY)
        self.assertEqual(text.count("Act "), 2)

    def test_keep_all_text_off_is_the_same_as_leaving_it_out(self):
        self.assertEqual(run_node(STORY, keep_all_text=False), run_node(STORY))


if __name__ == "__main__":
    unittest.main()
