"""The MiniMax H3 latent upscaler: ComfyUI's built-in one from 0.44.0 on, the Comfyui_Minimax_h3_latent_Upscaler pack before."""

import ast
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

features = importlib.import_module(f"{ROOT.name}.core.comfy_features")
upscaler = importlib.import_module(f"{ROOT.name}.minimax.latent_upscaler")


def fake_comfy_extras(calls):
    """ComfyUI's two built-in modules with a loader and an apply node that only record what they were asked."""

    class Loader:
        @staticmethod
        def execute(model_name):
            calls.append(("load", model_name))
            return ("patcher:" + model_name,)

    class Apply:
        @staticmethod
        def execute(patcher, samples, width, height):
            calls.append(("apply", patcher, tuple(samples["samples"].shape), width, height))
            video = samples["samples"]
            resized = torch.nn.functional.interpolate(video, size=(video.shape[2], height // 16, width // 16))
            out = dict(samples)
            out["samples"] = resized
            out.pop("noise_mask", None)
            return (out,)

    extras = types.ModuleType("comfy_extras")
    hunyuan = types.ModuleType("comfy_extras.nodes_hunyuan")
    hunyuan.LatentUpscaleModelLoader = Loader
    minimax = types.ModuleType("comfy_extras.nodes_minimax_h3")
    minimax.MiniMaxH3LatentUpscaleWithModel = Apply
    return {"comfy_extras": extras, "comfy_extras.nodes_hunyuan": hunyuan, "comfy_extras.nodes_minimax_h3": minimax}


class VersionTests(unittest.TestCase):
    def test_versions_are_read_from_the_usual_spellings(self) -> None:
        self.assertEqual(features.parse_version("0.45.0"), (0, 45, 0))
        self.assertEqual(features.parse_version("v0.44"), (0, 44, 0))
        self.assertEqual(features.parse_version("0.44.0rc1"), (0, 44, 0))
        self.assertIsNone(features.parse_version("nightly"))
        self.assertIsNone(features.parse_version(None))
        self.assertEqual(features.version_text((0, 45, 0)), "0.45.0")
        self.assertEqual(features.version_text(None), "")

    def test_built_in_starts_at_the_version_and_an_unknown_version_has_nothing(self) -> None:
        since = features.MINIMAX_H3_LATENT_UPSCALER_SINCE
        self.assertEqual(since, (0, 44, 0))
        self.assertFalse(features.has_built_in(since, (0, 43, 9)))
        self.assertTrue(features.has_built_in(since, (0, 44, 0)))
        self.assertTrue(features.has_built_in(since, (0, 45, 0)))
        self.assertTrue(features.has_built_in(since, (1, 0, 0)))
        with patch.object(features, "comfyui_version", return_value=None):
            self.assertFalse(features.has_built_in(since))

    def test_the_built_in_upscaler_needs_the_version_and_the_modules(self) -> None:
        with patch.dict(sys.modules, fake_comfy_extras([])):
            with patch.object(features, "comfyui_version", return_value=(0, 43, 0)):
                self.assertFalse(features.native_minimax_h3_upscaler())
            with patch.object(features, "comfyui_version", return_value=(0, 44, 0)):
                self.assertTrue(features.native_minimax_h3_upscaler())
        broken = fake_comfy_extras([])
        del broken["comfy_extras.nodes_minimax_h3"].MiniMaxH3LatentUpscaleWithModel
        with patch.dict(sys.modules, broken), patch.object(features, "comfyui_version", return_value=(0, 45, 0)):
            self.assertFalse(features.native_minimax_h3_upscaler())


class LoaderChoiceTests(unittest.TestCase):
    def load(self, native):
        loader = upscaler.VRGDG_MiniMaxH3LatentUpscaleModelLoader()
        with patch.object(upscaler, "native_minimax_h3_upscaler", return_value=native), \
                patch.object(upscaler, "_load_native_model", return_value={"native": True}) as built_in, \
                patch.object(upscaler, "_load_model", return_value={"native": False}) as pack:
            result = loader.load("minimax_h3_latent_upscaler_3d_bf16.safetensors", "cuda", "bf16")[0]
        return result, built_in, pack

    def test_new_comfyui_uses_its_own_loader_and_never_the_pack(self) -> None:
        result, built_in, pack = self.load(True)
        self.assertEqual(result, {"native": True})
        built_in.assert_called_once_with("minimax_h3_latent_upscaler_3d_bf16.safetensors")
        pack.assert_not_called()

    def test_old_comfyui_uses_the_pack_with_the_device_and_precision(self) -> None:
        result, built_in, pack = self.load(False)
        self.assertEqual(result, {"native": False})
        pack.assert_called_once_with("minimax_h3_latent_upscaler_3d_bf16.safetensors", "cuda", "bf16")
        built_in.assert_not_called()

    def test_the_native_loader_goes_through_comfyui_and_checks_the_file(self) -> None:
        calls = []
        with patch.dict(sys.modules, fake_comfy_extras(calls)), \
                patch.object(upscaler, "_resolve_registered_model_path", return_value="C:/models/up.safetensors"):
            loaded = upscaler._load_native_model("up.safetensors")
        self.assertEqual(calls, [("load", "up.safetensors")])
        self.assertEqual(loaded, {"native": True, "patcher": "patcher:up.safetensors", "model_name": "up.safetensors", "path": "C:/models/up.safetensors"})

    def test_the_older_comfyui_error_says_what_to_do(self) -> None:
        with patch.dict(sys.modules), patch.object(upscaler, "_backend_path", return_value="C:/nope/minimax_h3_latent_upscaler_3d.py"):
            sys.modules.pop(upscaler._BACKEND_MODULE_NAME, None)
            with self.assertRaises(RuntimeError) as raised:
                upscaler._load_backend()
        message = str(raised.exception)
        self.assertIn("older than 0.44.0", message)
        self.assertIn("Comfyui_Minimax_h3_latent_Upscaler", message)


class NativeApplyTests(unittest.TestCase):
    def apply(self, samples, scale=2.0, **extra):
        calls = []
        node = upscaler.VRGDG_MiniMaxH3LearnedLatentUpscale()
        with patch.dict(sys.modules, fake_comfy_extras(calls)):
            result = node.upscale({"samples": samples, **extra}, {"native": True, "patcher": "P"}, scale)[0]
        return result, calls

    def test_the_size_is_handed_over_in_pixels_on_the_16_grid(self) -> None:
        result, calls = self.apply(torch.zeros(1, 24, 3, 4, 6), 2.0)
        self.assertEqual(calls, [("apply", "P", (1, 24, 3, 4, 6), 6 * 2 * 16, 4 * 2 * 16)])
        self.assertEqual(tuple(result["samples"].shape), (1, 24, 3, 8, 12))

    def test_a_scale_with_a_fraction_rounds_the_latent_size_like_the_pack_did(self) -> None:
        result, calls = self.apply(torch.zeros(1, 24, 2, 10, 10), 1.5)
        self.assertEqual(calls[0][3:], (15 * 16, 15 * 16))
        self.assertEqual(tuple(result["samples"].shape), (1, 24, 2, 15, 15))

    def test_a_4d_latent_comes_back_4d(self) -> None:
        result, calls = self.apply(torch.zeros(1, 24, 4, 4), 2.0)
        self.assertEqual(calls[0][2], (1, 24, 1, 4, 4))
        self.assertEqual(tuple(result["samples"].shape), (1, 24, 8, 8))

    def test_other_latent_keys_stay_and_the_stale_noise_mask_goes(self) -> None:
        result, _ = self.apply(torch.zeros(1, 24, 3, 4, 4), 2.0, noise_mask=torch.ones(1, 1, 3, 4, 4), batch_index=[0])
        self.assertEqual(result["batch_index"], [0])
        self.assertNotIn("noise_mask", result)

    def test_a_scale_of_one_returns_the_latent_untouched_and_never_asks_comfyui(self) -> None:
        samples = torch.zeros(1, 24, 3, 4, 4)
        result, calls = self.apply(samples, 1.0)
        self.assertIs(result["samples"], samples)
        self.assertEqual(calls, [])

    def test_the_checks_before_the_upscale_still_apply(self) -> None:
        with self.assertRaisesRegex(ValueError, "24-channel"):
            self.apply(torch.zeros(1, 16, 3, 4, 4), 2.0)
        with self.assertRaisesRegex(ValueError, "scale >= 1.0"):
            self.apply(torch.zeros(1, 24, 3, 4, 4), 0.5)


def status_functions(version):
    """``_built_in`` and ``_node_status`` of core/system_routes.py (which needs a running ComfyUI server to import)."""
    source = ROOT / "core" / "system_routes.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    wanted = {"_built_in", "_node_status"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {
        "_CUSTOM_NODES_DIR": Path("C:/ComfyUI/custom_nodes"),
        "_installed_node_path": lambda item: None,
        "has_built_in": lambda since: features.has_built_in(since, version),
        "version_text": features.version_text,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


ITEM = {"install": "https://github.com/x/y.git", "folders": ("y",), "built_in_since": features.MINIMAX_H3_LATENT_UPSCALER_SINCE}


class CustomNodeStatusTests(unittest.TestCase):
    def test_the_upscaler_pack_is_built_in_from_0_44_0(self) -> None:
        old = status_functions((0, 43, 9))["_node_status"]("minimax_h3_latent_upscaler", ITEM)
        self.assertEqual((old["built_in"], old["built_in_since"], old["installed"]), (False, "0.44.0", False))
        new = status_functions((0, 45, 0))["_node_status"]("minimax_h3_latent_upscaler", ITEM)
        self.assertEqual((new["built_in"], new["built_in_since"], new["installed"]), (True, "0.44.0", False))

    def test_a_pack_with_nothing_built_in_has_no_built_in_fields(self) -> None:
        plain = {"install": "comfyui-kjnodes", "folders": ("ComfyUI-KJNodes",)}
        status = status_functions((0, 45, 0))["_node_status"]("kjnodes", plain)
        self.assertNotIn("built_in", status)
        self.assertEqual(status["source"], "registry")

    def test_the_allowlist_marks_only_the_upscaler_pack(self) -> None:
        source = ROOT / "core" / "system_routes.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        table = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                     and any(getattr(target, "id", "") == "VIDEO_BUILDER_CUSTOM_NODES" for target in node.targets))
        marked = [key.value for key, entry in zip(table.keys, table.values)
                  if any(getattr(name, "value", None) == "built_in_since" for name in entry.keys)]
        self.assertEqual(marked, ["minimax_h3_latent_upscaler"])


if __name__ == "__main__":
    unittest.main()
