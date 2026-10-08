"""Per-scene endpoint metadata must survive exact trimming without copying GPU tensors."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("keyframes", ROOT / "minimax/keyframes.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class KeyframeTimingTests(unittest.TestCase):
    def test_endpoint_indices_preserve_tensors_and_other_metadata(self):
        embedding, first, last = object(), object(), object()
        original = [[embedding, {"minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": first},
            {"resolved_frame_index": 55, "latent": last},
        ], "other": 123}]]
        result = module.align_i2v_keyframes(original, 5, 52)
        self.assertIs(result[0][0], embedding)
        frames = result[0][1]["minimax_keyframes"]
        self.assertEqual([f["resolved_frame_index"] for f in frames], [5, 52])
        self.assertIs(frames[0]["latent"], first)
        self.assertIs(frames[1]["latent"], last)
        self.assertEqual(result[0][1]["other"], 123)
        self.assertEqual(original[0][1]["minimax_keyframes"][1]["resolved_frame_index"], 55)

    def test_invalid_range_is_rejected(self):
        for first, last in [(-1, 10), (10, 5)]:
            with self.assertRaises(ValueError):
                module.align_i2v_keyframes([], first, last)
