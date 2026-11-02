import importlib.util
import json
import shutil
import struct
import wave
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ApiVideoWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.input = self.folder / "input"
        self.input.mkdir()
        self.folder_paths = types.SimpleNamespace(get_input_directory=lambda: str(self.input), get_output_directory=lambda: str(self.folder / "output"))
        self.nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={name: object for name in ["ByteDance2ReferenceNodeV2", "SaveVideo", "LoadImage", "LoadVideo", "LoadAudio"]})
        self.modules = patch.dict(sys.modules, {"folder_paths": self.folder_paths, "nodes": self.nodes})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location("api_video_workflows", ROOT / "runner/api_video_workflows.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def reference(self, name):
        path = self.folder / name
        path.write_bytes(name.encode())
        return {"path": str(path)}

    def payload(self, **changes):
        return {"prompt": "Image 1 walks toward Image 2.", "duration": 5.2,
                "reference_images": [self.reference("one.png"), self.reference("two.png")], **changes}

    def test_multimodal_graph_uses_official_node_and_unique_local_inputs(self):
        result = self.module._build_seedance25_reference_prompt(self.payload(
            reference_videos=[self.reference("clip.mp4")], reference_audios=[self.reference("voice.wav")], generate_audio=False))
        graph = result["prompt"]
        inputs = graph["seedance"]["inputs"]
        self.assertEqual(inputs["model"], "Seedance 2.5")
        self.assertEqual(inputs["model.task_type"], "reference")
        self.assertEqual(inputs["model.duration"], 6)
        self.assertFalse(inputs["model.generate_audio"])
        self.assertEqual(inputs["model.reference_images.image_2"], ["image_2", 0])
        self.assertEqual(inputs["model.reference_videos.video_1"], ["video_1", 0])
        self.assertEqual(inputs["model.reference_audios.audio_1"], ["audio_1", 0])
        self.assertEqual(graph["save_video"]["inputs"]["video"], ["seedance", 0])
        names = [graph[node]["inputs"][field] for node, field in [("image_1", "image"), ("image_2", "image"), ("video_1", "file"), ("audio_1", "audio")]]
        self.assertEqual(len(set(names)), 4)
        self.assertEqual((self.input / names[0]).read_bytes(), b"one.png")
        self.assertEqual((self.input / names[1]).read_bytes(), b"two.png")
        self.assertNotIn("api_key", json.dumps(graph))

    def test_short_scene_generates_four_seconds_for_later_trim(self):
        result = self.module._build_seedance25_reference_prompt(self.payload(duration=1.25))
        self.assertEqual(result["generation_duration"], 4)
        self.assertEqual(result["timeline_duration"], 1.25)

    def test_audio_only_reference_supported(self):
        result = self.module._build_seedance25_reference_prompt(self.payload(reference_images=[], reference_audios=[self.reference("voice.wav")]))
        self.assertIn("audio_1", result["prompt"])

    def test_image_data_urls_get_unique_names_and_preserve_bytes(self):
        result = self.module._build_seedance25_reference_prompt(self.payload(reference_images=[
            {"data": "data:image/png;base64,b25l"}, {"data": "data:image/png;base64,dHdv"}]))
        graph = result["prompt"]
        first = graph["image_1"]["inputs"]["image"]
        second = graph["image_2"]["inputs"]["image"]
        self.assertNotEqual(first, second)
        self.assertEqual((self.input / first).read_bytes(), b"one")
        self.assertEqual((self.input / second).read_bytes(), b"two")

    def test_bad_duration_and_reference_limits_fail_before_staging(self):
        for changes in [dict(duration=31), dict(duration=0), dict(duration=float("nan")),
                        dict(reference_images=[]), dict(reference_images=[{}] * 31),
                        dict(resolution="4k"), dict(ratio="invalid"), dict(prompt="")]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.module._build_seedance25_reference_prompt(self.payload(**changes))
        self.assertEqual(list(self.input.iterdir()), [])

    def test_missing_api_node_has_actionable_error(self):
        del self.nodes.NODE_CLASS_MAPPINGS["ByteDance2ReferenceNodeV2"]
        with self.assertRaisesRegex(ValueError, "Update ComfyUI and enable API nodes"):
            self.module._build_seedance25_reference_prompt(self.payload())

    def test_scene_audio_guidance_binds_the_selected_performer_and_actual_audio(self):
        result = self.module._build_seedance25_reference_prompt(self.payload(
            audio_mode="scene_audio", lip_sync_image=2, reference_audios=[self.reference("scene.wav")]))
        inputs = result["prompt"]["seedance"]["inputs"]
        self.assertEqual(inputs["model.reference_audios.audio_1"], ["audio_1", 0])
        self.assertIn("performer from @Image 2 lip-syncs", inputs["model.prompt"])
        self.assertIn("original timestamps", inputs["model.prompt"])
        self.assertIn("Other people keep their lips closed", inputs["model.prompt"])

    def test_instrumental_scene_keeps_mouths_closed_and_missing_performer_is_rejected(self):
        payload = self.payload(audio_mode="scene_audio", lip_sync_image=1, no_lip_sync=True,
                               reference_audios=[self.reference("scene.wav")])
        result = self.module._build_seedance25_reference_prompt(payload)
        self.assertIn("Keep every person's lips closed", result["prompt"]["seedance"]["inputs"]["model.prompt"])
        with self.assertRaisesRegex(ValueError, "performer image"):
            self.module._build_seedance25_reference_prompt({**payload, "lip_sync_image": 3})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.module._build_seedance25_reference_prompt({**payload, "reference_audios": []})

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for the actual audio timing check")
    def test_scene_audio_uses_exact_source_offset_and_pads_only_after_the_scene(self):
        source = self.folder / "song.wav"
        with wave.open(str(source), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            for second in range(1, 7):
                handle.writeframes(struct.pack("<hh", second * 1000, second * 1000) * 44100)
        with patch.object(self.module, "_seedance_ffmpeg_path", return_value=shutil.which("ffmpeg")), patch.object(self.module, "_seedance_audio_source_duration", return_value=6):
            result = self.module._prepare_seedance_scene_audio({
                "source_path": str(source), "project_folder": str(self.folder), "start": 1.25, "duration": 1.5})
        with wave.open(result["audio_path"], "rb") as handle:
            self.assertEqual(handle.getnframes(), 4 * 44100)
            self.assertEqual(struct.unpack("<hh", handle.readframes(1)), (2000, 2000))
            handle.setpos(44100)
            self.assertEqual(struct.unpack("<hh", handle.readframes(1)), (3000, 3000))
            handle.setpos(2 * 44100)
            self.assertEqual(struct.unpack("<hh", handle.readframes(1)), (0, 0))
        self.assertEqual(result["duration"], 1.5)
        self.assertEqual(result["generation_duration"], 4)

    def test_audio_after_source_end_fails_before_any_render_or_ffmpeg_preparation(self):
        source = self.reference("song.wav")["path"]
        with patch.object(self.module, "_seedance_audio_source_duration", return_value=6), patch.object(self.module.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "after its source audio ends"):
                self.module._prepare_seedance_scene_audio({"source_path": source, "project_folder": str(self.folder), "start": 8, "duration": 5})
            run.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for the audio placement check")
    def test_custom_audio_leading_silence_and_selected_clip_end_are_preserved(self):
        source = self.folder / "voice.wav"
        with wave.open(str(source), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(struct.pack("<hh", 1500, 1500) * 44100 * 3)
        with patch.object(self.module, "_seedance_ffmpeg_path", return_value=shutil.which("ffmpeg")), patch.object(self.module, "_seedance_audio_source_duration", return_value=3):
            result = self.module._prepare_seedance_scene_audio({"source_path": str(source), "project_folder": str(self.folder),
                "start": 0, "duration": 3, "source_duration": 0.5, "leading_silence": 0.75})
        with wave.open(result["audio_path"], "rb") as handle:
            self.assertEqual(struct.unpack("<hh", handle.readframes(1)), (0, 0))
            handle.setpos(44100)
            self.assertEqual(struct.unpack("<hh", handle.readframes(1)), (1500, 1500))
            handle.setpos(2 * 44100)
            self.assertEqual(struct.unpack("<hh", handle.readframes(1)), (0, 0))


if __name__ == "__main__":
    unittest.main()
