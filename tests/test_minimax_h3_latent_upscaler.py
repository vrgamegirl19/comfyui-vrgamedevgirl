import ast
import importlib
import sys
import types
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "VRGDG_MiniMaxH3LatentUpscaler.py"
INIT_SOURCE = ROOT / "__init__.py"


def load_node_helpers():
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    names = {"_load_raw_state_dict"}
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    import inspect
    namespace = {
        "inspect": inspect,
        "AttributeError": AttributeError,
        "TypeError": TypeError,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace


class MiniMaxH3LatentUpscalerLoaderTests(unittest.TestCase):
    def test_node_is_registered_in_init(self):
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        node_source = NODE_SOURCE.read_text(encoding="utf-8")
        self.assertIn('".VRGDG_MiniMaxH3LatentUpscaler"', init_source)
        self.assertIn('"VRGDG_MiniMaxH3LatentUpscaleModelLoader": VRGDG_MiniMaxH3LatentUpscaleModelLoader', node_source)
        self.assertIn('"VRGDG_MiniMaxH3LearnedLatentUpscale": VRGDG_MiniMaxH3LearnedLatentUpscale', node_source)

    def test_load_raw_state_dict_supports_upstream_single_arg_signature(self):
        helpers = load_node_helpers()
        load_raw_state_dict = helpers["_load_raw_state_dict"]

        class UpstreamBackend:
            def __init__(self):
                self.calls = []

            def _load_raw_sd(self, path):
                self.calls.append(path)
                return {"weight": "loaded_on_cpu", "path": path}

        backend = UpstreamBackend()
        result = load_raw_state_dict(backend, "models/upscaler.safetensors", "cuda", "bf16")
        self.assertEqual(result, {"weight": "loaded_on_cpu", "path": "models/upscaler.safetensors"})
        self.assertEqual(backend.calls, ["models/upscaler.safetensors"])

    def test_load_raw_state_dict_supports_multi_arg_backend_signature(self):
        helpers = load_node_helpers()
        load_raw_state_dict = helpers["_load_raw_state_dict"]

        class MultiArgBackend:
            def __init__(self):
                self.calls = []

            def _load_raw_sd(self, path, device, dtype):
                self.calls.append((path, device, dtype))
                return {"weight": "loaded_with_dtype", "device": device, "dtype": dtype}

        backend = MultiArgBackend()
        dtype_obj = torch.bfloat16 if torch is not None else "bf16"
        result = load_raw_state_dict(backend, "models/upscaler.safetensors", "cuda", dtype_obj)
        self.assertEqual(result["weight"], "loaded_with_dtype")
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["dtype"], dtype_obj)
        self.assertEqual(backend.calls, [("models/upscaler.safetensors", "cuda", dtype_obj)])

    def test_load_raw_state_dict_supports_kwarg_backend_signature(self):
        helpers = load_node_helpers()
        load_raw_state_dict = helpers["_load_raw_state_dict"]

        class KwargBackend:
            def __init__(self):
                self.calls = []

            def _load_raw_sd(self, path, **kwargs):
                self.calls.append((path, kwargs))
                return {"weight": "loaded_with_kwargs", "path": path, "kwargs": kwargs}

        backend = KwargBackend()
        dtype_obj = torch.float16 if torch is not None else "fp16"
        result = load_raw_state_dict(backend, "models/upscaler.safetensors", "cpu", dtype_obj)
        self.assertEqual(result["weight"], "loaded_with_kwargs")
        self.assertEqual(backend.calls[0][0], "models/upscaler.safetensors")
        self.assertIn("device", backend.calls[0][1])
        self.assertIn("dtype", backend.calls[0][1])

    def test_load_raw_state_dict_raises_when_backend_missing_load_fn(self):
        helpers = load_node_helpers()
        load_raw_state_dict = helpers["_load_raw_state_dict"]

        class BrokenBackend:
            pass

        with self.assertRaises(AttributeError):
            load_raw_state_dict(BrokenBackend(), "model.safetensors", "cuda", "bf16")

    @unittest.skipUnless(torch is not None, "Torch is provided by ComfyUI's Python environment.")
    def test_load_model_executes_with_mocked_backend_and_converts_precision(self):
        comfyui_root = ROOT.parents[1]
        for path in (str(comfyui_root), str(ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)

        module = importlib.import_module("VRGDG_MiniMaxH3LatentUpscaler")

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(24, 24)

            def load_state_dict(self, state_dict, strict=True):
                self.loaded_state = state_dict
                return []

        class FakeBackend:
            def __init__(self):
                self.load_calls = []

            def _load_raw_sd(self, path):
                self.load_calls.append(path)
                return {"fake.weight": torch.zeros((24, 24))}

            def _extract_upscaler_sd(self, raw_state):
                return raw_state

            def _detect_arch(self, state):
                return {
                    "in_channels": 24,
                    "in_blocks": 2,
                    "out_blocks": 2,
                    "channels": 64,
                    "dropout": 0.0,
                    "attn": True,
                    "temporal_every": 2,
                    "temporal_kernel": 3,
                }

            def LatentResizer3D(self, **kwargs):
                return FakeModel()

        original_resolve = module._resolve_model_path
        original_backend = module._load_backend
        fake_backend = FakeBackend()
        try:
            module._resolve_model_path = lambda name: "C:/fake/path/model.safetensors"
            module._load_backend = lambda: fake_backend
            module._MODEL_CACHE.clear()

            loaded = module._load_model("test_model", "cpu", "bf16")
            self.assertIn("model", loaded)
            self.assertEqual(loaded["dtype"], torch.bfloat16)
            self.assertEqual(loaded["precision"], "bf16")
            self.assertEqual(fake_backend.load_calls, ["C:/fake/path/model.safetensors"])

            # Verify caching: second call returns cached model without reloading
            cached = module._load_model("test_model", "cpu", "bf16")
            self.assertIs(cached, loaded)
            self.assertEqual(len(fake_backend.load_calls), 1)

            # Different precision should resolve to separate cache entry
            loaded_fp32 = module._load_model("test_model", "cpu", "fp32")
            self.assertEqual(loaded_fp32["dtype"], torch.float32)
            self.assertEqual(len(fake_backend.load_calls), 2)
        finally:
            module._resolve_model_path = original_resolve
            module._load_backend = original_backend
            module._MODEL_CACHE.clear()


if __name__ == "__main__":
    unittest.main()
