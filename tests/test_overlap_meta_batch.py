import importlib.util
import pathlib
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vrgdg_overlap_meta_batch", ROOT / "VRGDG_OverlapMetaBatch.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def frames(start, count):
    values = torch.arange(start, start + count, dtype=torch.float32)
    return values.reshape(count, 1, 1, 1)


class OverlapMetaBatchTests(unittest.TestCase):
    def manager(self, window=39, overlap=5, total=0):
        class NativeVHSManager:
            pass

        manager = NativeVHSManager()
        config, _, _, stride = MODULE.VRGDGOverlapPreset().configure(
            MODULE.CUSTOM_PRESET,
            window,
            overlap,
            "linear",
        )
        manager.frames_per_batch = stride
        manager.total_frames = total or float("inf")
        manager.has_closed_inputs = False
        return manager, config

    @staticmethod
    def prompt(index):
        manager_inputs = {"frames_per_batch": 34}
        if index:
            manager_inputs["requeue"] = index
        return {
            "2": {"class_type": "VHS_BatchManager", "inputs": manager_inputs},
            "11": {
                "class_type": "VRGDGOverlapWindow",
                "inputs": {"meta_batch": ["2", 0], "overlap_config": ["1", 0]},
            },
        }

    def process(self, source_count, window=39, overlap=5):
        manager, config = self.manager(window, overlap, source_count)
        build = MODULE.VRGDGOverlapWindow()
        blend = MODULE.VRGDGOverlapBlend()
        stride = window - overlap
        outputs = []
        for index, start in enumerate(range(0, source_count, stride)):
            raw = frames(start, min(stride, source_count - start))
            prompt = self.prompt(index)
            prompt["2"]["inputs"]["frames_per_batch"] = stride
            model_input, info, _, _ = build.build_window(
                raw, manager, config, prompt=prompt, unique_id="11"
            )
            result, _ = blend.blend_window(model_input, info, manager)
            outputs.append(result)
        return torch.cat(outputs).flatten()

    def test_h3_presets_and_validation(self):
        self.assertEqual(MODULE.resolve_h3_settings("H3 Near 41 (39 / 5)", 73, 17), (39, 5, 34))
        self.assertEqual(MODULE.resolve_h3_settings("H3 Balanced (73 / 17)", 39, 5), (73, 17, 56))
        self.assertEqual(MODULE.resolve_h3_settings("H3 Trained (124 / 22)", 39, 5), (124, 22, 102))
        with self.assertRaisesRegex(ValueError, r"17n\+5"):
            MODULE.resolve_h3_settings(MODULE.CUSTOM_PRESET, 41, 5)
        with self.assertRaisesRegex(ValueError, "less than half"):
            MODULE.resolve_h3_settings(MODULE.CUSTOM_PRESET, 39, 20)

    def test_short_single_batch_preserves_every_frame(self):
        result = self.process(12)
        torch.testing.assert_close(result, torch.arange(12, dtype=torch.float32))

    def test_multiple_batches_preserve_count_and_order_for_identity_model(self):
        result = self.process(100)
        torch.testing.assert_close(result, torch.arange(100, dtype=torch.float32))

    def test_exact_stride_multiple_flushes_tail(self):
        result = self.process(68)
        torch.testing.assert_close(result, torch.arange(68, dtype=torch.float32))

    def test_blend_uses_both_processed_overlap_versions(self):
        manager, config = self.manager(39, 5, 50)
        build = MODULE.VRGDGOverlapWindow()
        blend = MODULE.VRGDGOverlapBlend()

        first, info, _, _ = build.build_window(
            frames(0, 34), manager, config, prompt=self.prompt(0), unique_id="11"
        )
        first_processed = first.clone()
        first_processed[-5:] = 0
        first_out, _ = blend.blend_window(first_processed, info, manager)

        second, info, _, _ = build.build_window(
            frames(34, 16), manager, config, prompt=self.prompt(1), unique_id="11"
        )
        second_processed = second.clone()
        second_processed[:5] = 10
        second_out, _ = blend.blend_window(second_processed, info, manager)

        overlap_values = second_out[:5].flatten()
        self.assertTrue(torch.all(overlap_values > 0))
        self.assertTrue(torch.all(overlap_values < 10))
        self.assertEqual(first_out.shape[0] + second_out.shape[0], 50)

    def test_native_vhs_requeue_counter_is_followed(self):
        manager, config = self.manager(39, 5, 50)
        build = MODULE.VRGDGOverlapWindow()
        build.build_window(
            frames(0, 34), manager, config, prompt=self.prompt(0), unique_id="11"
        )
        _, info, _, _ = build.build_window(
            frames(34, 16), manager, config, prompt=self.prompt(1), unique_id="11"
        )
        self.assertEqual(info["batch_index"], 1)


if __name__ == "__main__":
    unittest.main()
