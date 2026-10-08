"""Embedded scene audio stays in lip sync at every join of a stitch (runner/video_files.py _stitch_scene_videos).

Each scene's own audio is cut or padded to its clip's frame length (frames / fps) and joined as PCM, then
encoded once. Before, every part was encoded to AAC on its own and joined as-is, so a part shorter or
longer than its clip (Cut and Gun: 5-17 ms short; AAC frame rounding: up to 21 ms long) moved every
later scene's audio, and the error added up join by join.
"""

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
video_files = importlib.import_module(f"{pkg_name}.runner.video_files")

FPS = 24
SR = 48000
# Cut and Gun scenes 1-6 (ffprobe -count_packets on video_0001..0006-audio.mp4).
SCENE_FRAMES = [162, 117, 141, 183, 148, 120]
# Audio length minus frames/24 per scene, in seconds.
OSIRIS_SHORT_AUDIO = [-0.016, -0.015, -0.005, -0.009, -0.0167, -0.008]
LONG_AUDIO = [0.012, 0.010, 0.013, 0.012, 0.020, 0.013]
ONE_VIDEO_FRAME = 1.0 / FPS
BURST = 0.040  # each scene's audio opens with a 40 ms tone, so its start can be found in the output


def have_ffmpeg():
    try:
        for tool in ("ffmpeg", "ffprobe"):
            subprocess.run([tool, "-version"], capture_output=True, check=True)
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def make_clip(path, frames, audio_delta=None, sample_rate=SR, tone_hz=1000):
    """24 fps clip of ``frames`` frames. Its audio lasts frames/24 + ``audio_delta`` s (None = no audio stream).

    Video and audio are encoded separately and muxed with stream copy, so neither is cut to the other.
    """
    video_only = path + ".v.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=160x90:rate={FPS}",
        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", video_only,
    ], capture_output=True, check=True)
    if audio_delta is None:
        os.replace(video_only, path)
        return path
    audio_only = path + ".a.m4a"
    duration = frames / FPS + audio_delta
    expr = f"if(lt(t\\,{BURST})\\,0.8*sin(2*PI*{tone_hz}*t)\\,0)"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"aevalsrc={expr}:s={sample_rate}:d={duration:.6f}",
        "-c:a", "aac", audio_only,
    ], capture_output=True, check=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", video_only, "-i", audio_only,
        "-map", "0:v", "-map", "1:a", "-c", "copy", path,
    ], capture_output=True, check=True)
    for scratch in (video_only, audio_only):
        os.remove(scratch)
    return path


def audio_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def frame_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def decoded_audio(path):
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def join_offsets(path, frames_list, fps=FPS):
    """Seconds from each scene's video cut to where its audio's opening tone lands in ``path``."""
    import numpy as np
    audio = decoded_audio(path)
    offsets, cut = [], 0
    for frames in frames_list:
        t_cut = cut / fps
        lo = max(0, int((t_cut - 0.2) * SR))
        hi = min(len(audio), int((t_cut + 0.2) * SR))
        hits = np.nonzero(np.abs(audio[lo:hi]) > 0.2)[0]
        offsets.append((lo + hits[0]) / SR - t_cut if len(hits) else None)
        cut += frames
    return offsets


def timing_items(frames_list, fps=FPS, jitter=0.004):
    """Builder-style scene times (a little off the frame grid) whose rounding gives ``frames_list``."""
    items, total = [], 0
    for frames in frames_list:
        start = total / fps + (jitter if total else 0.0)
        total += frames
        items.append({"start": start, "end": total / fps + jitter})
    return items


@unittest.skipUnless(have_ffmpeg(), "ffmpeg/ffprobe (and numpy) not installed")
class EmbeddedAudioDriftTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_audio_drift_")
        self.project = os.path.join(self.test_dir, "Project")
        self.clip_dir = os.path.join(self.project, "rendered_scene_videos")
        os.makedirs(self.clip_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def clips(self, deltas, frames_list=SCENE_FRAMES, sample_rate=SR):
        return [
            make_clip(os.path.join(self.clip_dir, f"video_{i:04d}-audio.mp4"), frames, delta, sample_rate)
            for i, (frames, delta) in enumerate(zip(frames_list, deltas), start=1)
        ]

    def stitch(self, paths, timing=True, frames_list=SCENE_FRAMES):
        payload = {
            "project_folder": self.project,
            "scene_paths": paths,
            "audio_path": "",
            "use_embedded_scene_audio": True,
            "output_prefix": "DRIFT",
        }
        if timing:
            payload["scene_timing_items"] = timing_items(frames_list)
            payload["timeline_fps"] = FPS
        return video_files._stitch_scene_videos(payload)["final_video_path"]

    def assert_in_sync(self, out, frames_list=SCENE_FRAMES):
        offsets = join_offsets(out, frames_list)
        self.assertNotIn(None, offsets, offsets)
        worst = max(abs(o) for o in offsets)
        self.assertLess(worst, ONE_VIDEO_FRAME, f"per-scene offsets (ms): {[round(o * 1000, 1) for o in offsets]}")
        # PCM join + one encode: every scene start lands within a couple of ms, not just within a frame.
        self.assertLess(worst, 0.003, f"per-scene offsets (ms): {[round(o * 1000, 1) for o in offsets]}")
        total = sum(frames_list) / FPS
        audio_len = len(decoded_audio(out)) / SR
        # The encoder may round the tail up to one AAC frame (1024 samples); never short, never drifting.
        self.assertGreater(audio_len, total - 0.001)
        self.assertLess(audio_len, total + 1024 / SR + 0.001)
        self.assertEqual(frame_count(out), sum(frames_list))
        return offsets

    def test_fixture_clips_have_the_intended_audio_lengths(self):
        for deltas in (OSIRIS_SHORT_AUDIO, LONG_AUDIO):
            paths = self.clips(deltas)
            for path, frames, delta in zip(paths, SCENE_FRAMES, deltas):
                self.assertAlmostEqual(audio_duration(path) - frames / FPS, delta, delta=0.0015)
                self.assertEqual(frame_count(path), frames)

    def test_short_audio_like_osiris_timing_path(self):
        self.assert_in_sync(self.stitch(self.clips(OSIRIS_SHORT_AUDIO), timing=True))

    def test_short_audio_like_osiris_plain_path(self):
        self.assert_in_sync(self.stitch(self.clips(OSIRIS_SHORT_AUDIO), timing=False))

    def test_long_audio_timing_path(self):
        self.assert_in_sync(self.stitch(self.clips(LONG_AUDIO), timing=True))

    def test_long_audio_plain_path(self):
        self.assert_in_sync(self.stitch(self.clips(LONG_AUDIO), timing=False))

    def test_exact_audio_stays_exact(self):
        self.assert_in_sync(self.stitch(self.clips([0.0] * 6), timing=True))

    def test_44100_hz_scene_audio(self):
        # 1837.5 samples per frame at 44.1 kHz: rounding must not build up either.
        self.assert_in_sync(self.stitch(self.clips(OSIRIS_SHORT_AUDIO, sample_rate=44100), timing=True))

    def test_timing_path_uses_the_timeline_frame_count(self):
        # Scene 2's clip has 6 frames more than its timeline slot; the stitcher keeps 117, so its audio
        # must be cut to 117 frames too or scene 3 would start late.
        frames = list(SCENE_FRAMES)
        rendered = list(frames)
        rendered[1] += 6
        paths = self.clips([0.0] * 6, frames_list=rendered)
        self.assert_in_sync(self.stitch(paths, timing=True, frames_list=frames), frames_list=frames)

    def test_scene_without_audio_becomes_silence(self):
        deltas = list(OSIRIS_SHORT_AUDIO)
        deltas[2] = None  # scene 3 has no audio stream
        out = self.stitch(self.clips(deltas), timing=True)
        offsets = join_offsets(out, SCENE_FRAMES)
        self.assertIsNone(offsets[2])  # silent
        for index in (0, 1, 3, 4, 5):
            self.assertLess(abs(offsets[index]), 0.003, offsets)
        self.assertEqual(frame_count(out), sum(SCENE_FRAMES))

    def test_scene_without_audio_plain_path(self):
        deltas = [0.0] * 6
        deltas[0] = None
        out = self.stitch(self.clips(deltas), timing=False)
        offsets = join_offsets(out, SCENE_FRAMES)
        self.assertIsNone(offsets[0])
        for index in range(1, 6):
            self.assertLess(abs(offsets[index]), 0.003, offsets)

    def test_scratch_files_are_removed(self):
        self.stitch(self.clips(OSIRIS_SHORT_AUDIO), timing=True)
        leftovers = [n for n in os.listdir(self.clip_dir) if n.startswith("_temp") or n.endswith(".txt")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
