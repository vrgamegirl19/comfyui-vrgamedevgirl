"""Exercise both supported external MiniMax H3 latent-upscaler loader signatures."""

import ast
import inspect
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "VRGDG_MiniMaxH3LatentUpscaler.py"


class FakeModel:
    def load_state_dict(self, state, strict=True):
        self.state = state

    def to(self, *, device, dtype):
        self.device = device
        self.dtype = dtype
        return self

    def eval(self):
        return self


def load_with_backend(backend):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_model")
    torch = type("Torch", (), {
        "device": staticmethod(lambda value: value),
        "cuda": type("Cuda", (), {"is_available": staticmethod(lambda: True)}),
        "float32": object(), "float16": object(), "bfloat16": object(),
    })
    namespace = {
        "_resolve_model_path": lambda name: name,
        "_load_backend": lambda: backend,
        "_MODEL_CACHE": {},
        "inspect": inspect,
        "torch": torch,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_load_model"]("model.safetensors", "cuda", "bf16"), torch


class LoaderCompatibilityTests(unittest.TestCase):
    def test_original_one_argument_backend(self):
        calls = []

        class Backend:
            def _load_raw_sd(self, path):
                calls.append((path,))
                return {"weight": 1}

            def _extract_upscaler_sd(self, state):
                return state

            def _detect_arch(self, state):
                return dict.fromkeys(("in_channels", "in_blocks", "out_blocks", "channels", "dropout", "attn", "temporal_every", "temporal_kernel"), 1)

            def LatentResizer3D(self, **kwargs):
                return FakeModel()

        result, torch = load_with_backend(Backend())
        self.assertEqual(calls, [("model.safetensors",)])
        self.assertIs(result["dtype"], torch.bfloat16)
        self.assertEqual(result["model"].state, {"weight": 1})

    def test_new_three_argument_backend_receives_dtype_object(self):
        calls = []

        class Backend:
            def _load_raw_sd(self, path, device, dtype):
                calls.append((path, device, dtype))
                return {"weight": 2}

            def _extract_upscaler_sd(self, state):
                return state

            def _detect_arch(self, state):
                return dict.fromkeys(("in_channels", "in_blocks", "out_blocks", "channels", "dropout", "attn", "temporal_every", "temporal_kernel"), 1)

            def LatentResizer3D(self, **kwargs):
                return FakeModel()

        result, torch = load_with_backend(Backend())
        self.assertEqual(calls, [("model.safetensors", "cuda", torch.bfloat16)])
        self.assertIs(result["model"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
