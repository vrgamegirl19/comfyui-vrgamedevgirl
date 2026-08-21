from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(encoding="utf-8")
UI = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
STORYBOARD_UI = (ROOT / "web" / "VRGDG_StoryboardBuilderUI.js").read_text(encoding="utf-8")


class BuilderQwenRunnerTests(unittest.TestCase):
    def test_backend_accepts_qwen_local_runner_and_model(self):
        self.assertIn('runner in {"qwen", "qwen_local", "qwen-local", "qwen_gguf", "qwen_gguf_local"}', BACKEND)
        self.assertIn('"qwen_model_file"', BACKEND)
        self.assertIn('"qwen_mmproj_file"', BACKEND)
        self.assertIn('"gemma_model_file"', BACKEND)
        self.assertIn("VRGDG_QwenGGUF", BACKEND)

    def test_builder_local_thinking_is_explicitly_disabled(self):
        self.assertIn('llm._qwen_chat_template_kwargs = {"enable_thinking": False}', BACKEND)
        helper_start = BACKEND.index("def _builder_local_llm(payload):")
        helper_end = BACKEND.index("\ndef _builder_local_model_file", helper_start)
        helper = BACKEND[helper_start:helper_end]
        self.assertLess(helper.index("else:"), helper.index('llm._qwen_chat_template_kwargs = {"enable_thinking": False}'))

    def test_builder_output_strips_model_thinking_preambles(self):
        self.assertIn("def _strip_builder_thinking_text(text):", BACKEND)
        self.assertIn(r"<(?:think|thought)>.*?</(?:think|thought)>", BACKEND)
        self.assertIn("thought(?: process)?|thinking(?: process)?|analysis|reasoning", BACKEND)
        self.assertIn("cleaned = _strip_builder_thinking_text(text)", BACKEND)

    def test_choices_include_qwen_and_external_picker_support(self):
        self.assertIn('"qwen_models": VRGDG_QwenGGUF._list_local_qwen_gguf()', BACKEND)
        self.assertIn('kind == "gguf"', BACKEND)
        self.assertIn('"qwen_local"', UI)
        self.assertIn('Choose GGUF file', UI)

    def test_qwen_selection_hides_gemma_fields_and_uses_qwen_confirmation(self):
        self.assertIn('gemmaLocalPanel.style.display = runner.value === "builtin"', UI)
        self.assertIn('qwenLocalPanel.style.display = runner.value === "qwen_local"', UI)
        self.assertIn('Text LLM runner set to Qwen Local.', UI)
        self.assertIn('Lower GPU layers if Qwen Local runs out of VRAM.', UI)

    def test_runner_model_selection_syncs_builder_text_and_vision_fields(self):
        self.assertIn("function syncBuilderLlmModelSelectsFromRunner()", UI)
        self.assertIn("...builderTextLlmModelSelects, ...builderVisionLlmModelSelects", UI)
        self.assertIn("selectBuilderLlmValue(select, state.qwenMmprojFile)", UI)
        self.assertIn('makeField("Non-Vision text LLM model"', UI)
        self.assertIn('makeField("Vision LLM model"', UI)
        self.assertNotIn('makeField("Non-Vision text Gemma model"', UI)
        self.assertNotIn('makeField("Vision Gemma model"', UI)

    def test_runner_names_are_dynamic_in_builder_and_storyboard_windows(self):
        for expected in ("Gemma Local", "Qwen Local", "LM Studio", "LLM API", "Custom Server"):
            self.assertIn(expected, UI)
            self.assertIn(expected, STORYBOARD_UI)
        self.assertIn('"Short Film Premise" : "Story Arc"', STORYBOARD_UI)
        self.assertIn('${promptRunnerName()}', STORYBOARD_UI)
        self.assertIn('document.createTextNode(`Use in ${promptRunnerName()} prompts`)', STORYBOARD_UI)
        self.assertIn('createProgressWindow(`${runnerName} → Create Scene Video`)', UI)
        self.assertIn("function runnerAwareLlmText(value)", UI)
        self.assertIn("progress.set(runnerAwareLlmText(message), percent)", UI)


if __name__ == "__main__":
    unittest.main()
