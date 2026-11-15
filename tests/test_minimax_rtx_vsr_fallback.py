"""Final resize: nvidia_rtx_vsr falls back to lanczos when the nvidia-vfx library is not installed."""

import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

features = importlib.import_module(f"{ROOT.name}.core.comfy_features")
api_graph = importlib.import_module(f"{ROOT.name}.runner.api_graph")
workflows = importlib.import_module(f"{ROOT.name}.runner.minimax_workflows")
chunk_tests = importlib.import_module("test_minimax_chunked_pass2")


def _graph():
    return {
        "183": {"class_type": "ImageResizeKJv2", "inputs": {"upscale_method": "nvidia_rtx_vsr"}},
        "291": {"class_type": "ImageResizeKJv2", "inputs": {"upscale_method": "nvidia_rtx_vsr"}},
        "300": {"class_type": "ImageResizeKJv2", "inputs": {"upscale_method": "bicubic"}},
        "301": {"class_type": "SaveImage", "inputs": {}},
    }


class RtxVsrFallbackTests(unittest.TestCase):
    def _run(self, available):
        graph = _graph()
        output = io.StringIO()
        with patch.object(api_graph, "nvidia_vsr_available", return_value=available), contextlib.redirect_stdout(output):
            changed = api_graph._fall_back_from_unavailable_rtx_vsr(graph)
        return graph, changed, output.getvalue()

    def test_every_rtx_vsr_resize_becomes_lanczos_when_the_library_is_missing(self) -> None:
        graph, changed, output = self._run(False)
        self.assertEqual(sorted(changed), ["183", "291"])
        self.assertEqual(graph["183"]["inputs"]["upscale_method"], "lanczos")
        self.assertEqual(graph["291"]["inputs"]["upscale_method"], "lanczos")
        self.assertEqual(graph["300"]["inputs"]["upscale_method"], "bicubic")
        self.assertIn("[VRGDG MiniMax]", output)
        self.assertIn("--extra-index-url https://pypi.nvidia.com", output)

    def test_the_graph_is_left_alone_when_the_library_is_installed(self) -> None:
        graph, changed, output = self._run(True)
        self.assertEqual(changed, [])
        self.assertEqual(graph, _graph())
        self.assertEqual(output, "")

    def test_a_graph_without_rtx_vsr_does_not_check_the_library(self) -> None:
        graph = {"1": {"class_type": "ImageResizeKJv2", "inputs": {"upscale_method": "lanczos"}}}
        with patch.object(api_graph, "nvidia_vsr_available", side_effect=AssertionError("checked")):
            self.assertEqual(api_graph._fall_back_from_unavailable_rtx_vsr(graph), [])

    def test_availability_follows_whether_nvvfx_imports(self) -> None:
        with patch.dict(sys.modules, {"nvvfx": object()}):
            self.assertTrue(features.nvidia_vsr_available())
        with patch.dict(sys.modules, {"nvvfx": None}):
            self.assertFalse(features.nvidia_vsr_available())

    def test_both_h3_builders_apply_the_fallback_last(self) -> None:
        # Source check: the 3-pass builder needs a real model setup to compile.
        source = (ROOT / "runner" / "minimax_workflows.py").read_text(encoding="utf-8")
        for name in ("_build_minimax_h3_2pass_api_prompt", "_build_minimax_h3_3pass_api_prompt"):
            start = source.index(f"def {name}(payload):")
            end = source.find("\ndef ", start + 10)
            body = source[start:] if end == -1 else source[start:end]
            self.assertIn("_fall_back_from_unavailable_rtx_vsr(prompt)\n    return {", body, name)


class CompiledTwoPassTests(unittest.TestCase):
    """A real compiled 2-pass graph, set up the way the chunked second pass tests do."""

    setUp = chunk_tests.CompiledGraphTests.setUp
    payload = chunk_tests.CompiledGraphTests.payload
    build = chunk_tests.CompiledGraphTests.build

    def test_the_default_final_resize_is_lanczos_without_the_library(self) -> None:
        with patch.object(api_graph, "nvidia_vsr_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
            graph = self.build(5)["prompt"]
        self.assertEqual(graph["183"]["inputs"]["upscale_method"], "lanczos")

    def test_the_default_final_resize_stays_rtx_vsr_with_the_library(self) -> None:
        with patch.object(api_graph, "nvidia_vsr_available", return_value=True):
            graph = self.build(5)["prompt"]
        self.assertEqual(graph["183"]["inputs"]["upscale_method"], "nvidia_rtx_vsr")


if __name__ == "__main__":
    unittest.main()
