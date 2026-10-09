"""Edited audio keeps source offsets and renders real silence between pieces."""

import asyncio
import importlib
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
audio_clips = importlib.import_module(f"{ROOT.name}.builder.audio_clips")


class AudioClipEditsTests(unittest.TestCase):
    def test_reimport_keeps_previous_clip_sources(self) -> None:
        """Importing another file for the same scene cannot overwrite its old pieces."""
        audio = importlib.import_module(f"{ROOT.name}.builder.audio")
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "speech.wav"
            payload = {"project_folder": folder, "scene_number": 1, "source_path": str(source),
                       "preserve_source": True}
            with patch.object(audio, "_read_audio_peaks", return_value={"duration": 1, "peaks": []}):
                source.write_bytes(b"first original")
                first = audio._save_scene_audio(payload)
                source.write_bytes(b"second original")
                second = audio._save_scene_audio(payload)
            self.assertNotEqual(first["saved_path"], second["saved_path"])
            self.assertEqual(Path(first["saved_path"]).read_bytes(), b"first original")
            self.assertEqual(Path(second["saved_path"]).read_bytes(), b"second original")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_frontend_audio_edits(self) -> None:
        result = subprocess.run(["node", "--test", str(ROOT / "tests/audio_clip_edits.cjs")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_mix_has_exact_silent_gaps_and_preserves_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "speech.wav"
            rate = 44100
            values = [int((8000 if i < rate else 16000) * math.sin(2 * math.pi * 440 * i / rate))
                      for i in range(rate * 2)]
            with wave.open(str(source), "wb") as output:
                output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                output.writeframes(struct.pack(f"<{len(values)}h", *values))
            original = source.read_bytes()
            clips = [{"path": str(source), "start": 0.5, "source_start": 0, "duration": 0.5},
                     {"path": str(source), "start": 2, "source_start": 1, "duration": 0.5}]
            result = audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": clips, "duration": 4})
            with wave.open(result["audio_path"], "rb") as rendered:
                self.assertEqual(rendered.getnframes(), rate * 4)
                self.assertEqual(rendered.getnchannels(), 2)
                samples = struct.unpack(f"<{rate * 4 * 2}h", rendered.readframes(rate * 4))
            def rms(start: float, end: float) -> float:
                """Measure the selected region of the rendered stereo track."""
                region = samples[int(start * rate) * 2:int(end * rate) * 2]
                return math.sqrt(sum(value * value for value in region) / len(region))
            self.assertEqual(rms(0, 0.4), 0)
            self.assertEqual(rms(1.1, 1.9), 0)
            self.assertEqual(rms(2.6, 4), 0)
            self.assertGreater(rms(0.6, 0.9), 3000)
            self.assertGreater(rms(2.1, 2.4), rms(0.6, 0.9) * 1.8)
            self.assertEqual(source.read_bytes(), original)
            cached = audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": clips, "duration": 4})
            self.assertEqual(cached["audio_path"], result["audio_path"])
            silence = audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": [], "duration": 4})
            with wave.open(silence["audio_path"], "rb") as rendered:
                self.assertFalse(any(rendered.readframes(rate * 4)))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_overlapping_score_volume_mute_and_generation_filter(self) -> None:
        """The score mixes underneath dialogue and can stay out of speaking generation."""
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "constant.wav"
            with wave.open(str(source), "wb") as output:
                output.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                output.writeframes(struct.pack("<44100h", *([1000] * 44100)))
            voice = {"path": str(source), "start": 0, "duration": 1, "role": "dialogue"}
            score = {**voice, "role": "music", "volume": 0.25, "include_in_generation": False}

            def level(clips: list[dict], generation: bool = False) -> int:
                """Read one stereo sample from the actual rendered mix."""
                result = audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": clips,
                                                            "duration": 1, "generation_only": generation})
                with wave.open(result["audio_path"], "rb") as output:
                    return struct.unpack("<hh", output.readframes(1))[0]

            original = level([voice])
            self.assertAlmostEqual(level([voice, score]), original * 1.25, delta=2)
            self.assertEqual(level([voice, score], True), original)
            self.assertEqual(level([voice, {**score, "muted": True}]), original)
            self.assertEqual(level([{**voice, "volume": 0}]), 0)
            self.assertGreater(level([voice, {**score, "include_in_generation": True}], True), original)

    def test_invalid_timing_is_rejected_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    audio_clips.prepare_audio_clip_mix({"project_folder": folder, "clips": [], "duration": value})

    def test_api_stitch_uses_score_with_embedded_voice_only_in_speaking(self) -> None:
        """Capture headless mix inputs and prevent speaking tracks affecting other video types."""
        orchestrator = importlib.import_module(f"{ROOT.name}.agent_api.orchestrator.video_orchestrator")
        for video_type in ("speaking", "singing", "no_lip_sync"):
            with self.subTest(video_type=video_type):
                score = {"path": "score.wav", "role": "music", "volume": 0.25}
                session = {"video_type": video_type, "audio_clips": [score],
                           "segments": [{"id": "s", "start": 0, "end": 4}]}
                payload = {"scene_paths": ["voice.mp4"], "use_embedded_scene_audio": True}
                job = SimpleNamespace(project_id="project", id="job", params={})
                with (
                    patch.object(orchestrator, "_get_active_session_and_folder", return_value=("folder", session)),
                    patch.object(orchestrator, "build_stitch_payload", return_value=(payload, {"scene_ids": ["s"]})),
                    patch.object(audio_clips, "prepare_audio_clip_mix", return_value={"audio_path": "mix.wav"}) as mix,
                    patch.object(orchestrator.video_files, "_stitch_scene_videos", return_value={}) as stitch,
                ):
                    asyncio.run(orchestrator.run_video_stitch_job(job, MagicMock()))
                if video_type == "speaking":
                    self.assertEqual(mix.call_args.args[0]["clips"][0]["path"], "score.wav")
                    self.assertEqual(mix.call_args.args[0]["clips"][1]["path"], "voice.mp4")
                    self.assertEqual(stitch.call_args.args[0]["audio_path"], "mix.wav")
                    self.assertFalse(stitch.call_args.args[0]["use_embedded_scene_audio"])
                else:
                    mix.assert_not_called()
                    self.assertTrue(stitch.call_args.args[0]["use_embedded_scene_audio"])


if __name__ == "__main__":
    unittest.main()
