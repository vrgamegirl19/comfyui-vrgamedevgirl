import importlib.util
import json
import pathlib
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "VRGDG_EnsureVideoAudio.py"
WORKFLOW_DIR = ROOT / "Workflows" / "VideoUpscaleExperimental"
ORIGINAL_WORKFLOW = WORKFLOW_DIR / "VRGDG_MiniMaxH3_Upscaler.json"
V2_WORKFLOW = WORKFLOW_DIR / "VRGDG_MiniMaxH3_Upscaler_V2.json"

SPEC = importlib.util.spec_from_file_location("vrgdg_ensure_video_audio", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BrokenLazyAudio:
    def __getitem__(self, key):
        raise RuntimeError("ffmpeg found no audio stream")


class EnsureVideoAudioTests(unittest.TestCase):
    def test_valid_audio_passes_through(self):
        waveform = torch.ones((1, 2, 16000), dtype=torch.float32)
        source = {"waveform": waveform, "sample_rate": 32000}

        audio, has_source_audio, status = MODULE.ensure_video_audio(
            source, {"source_duration": 10.0}
        )

        self.assertIs(audio["waveform"], waveform)
        self.assertEqual(audio["sample_rate"], 32000)
        self.assertTrue(has_source_audio)
        self.assertIn("Using source audio", status)

    def test_missing_audio_becomes_video_length_silence(self):
        audio, has_source_audio, status = MODULE.ensure_video_audio(
            BrokenLazyAudio(),
            {"source_duration": 1.25},
            fallback_sample_rate=32000,
            fallback_channels=2,
        )

        self.assertEqual(audio["waveform"].shape, (1, 2, 40000))
        self.assertEqual(audio["sample_rate"], 32000)
        self.assertEqual(torch.count_nonzero(audio["waveform"]).item(), 0)
        self.assertFalse(has_source_audio)
        self.assertIn("generated 1.250s", status)

    def test_duration_can_fall_back_to_frame_count_and_fps(self):
        audio, has_source_audio, _ = MODULE.ensure_video_audio(
            BrokenLazyAudio(),
            {"source_frame_count": 60, "source_fps": 24},
            fallback_sample_rate=8000,
            fallback_channels=1,
        )

        self.assertEqual(audio["waveform"].shape, (1, 1, 20000))
        self.assertFalse(has_source_audio)

    def test_v2_workflow_routes_all_audio_through_fallback_node(self):
        workflow = json.loads(V2_WORKFLOW.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        fallback = next(node for node in nodes.values() if node["type"] == "VRGDGEnsureVideoAudio")
        loader = next(node for node in nodes.values() if node["type"] == "VHS_LoadVideo")

        self.assertEqual(loader["outputs"][2]["links"], [55])
        self.assertEqual(set(fallback["outputs"][0]["links"]), {31, 39})
        self.assertEqual(workflow["last_node_id"], max(nodes))
        self.assertEqual(workflow["last_link_id"], max(link[0] for link in workflow["links"]))

        seen = set()
        for link_id, source_id, source_slot, target_id, target_slot, type_name in workflow["links"]:
            self.assertNotIn(link_id, seen)
            seen.add(link_id)
            source = nodes[source_id]["outputs"][source_slot]
            target = nodes[target_id]["inputs"][target_slot]
            self.assertEqual(source["type"], type_name)
            self.assertEqual(target["type"], type_name)
            self.assertIn(link_id, source["links"])
            self.assertEqual(target["link"], link_id)

    def test_original_workflow_remains_without_fallback_node(self):
        original = json.loads(ORIGINAL_WORKFLOW.read_text(encoding="utf-8"))
        self.assertNotIn("VRGDGEnsureVideoAudio", {node["type"] for node in original["nodes"]})


if __name__ == "__main__":
    unittest.main()
