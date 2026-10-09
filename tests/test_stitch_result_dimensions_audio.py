"""The stitch result describes the file it wrote: its real size and its real audio length.

Live report (Osiris, PREVIEW_API_001-012): the job result said output_width = output_height = 0 (a MiniMax
stitch sends no canvas size, and the runner echoed the request), and in embedded mode audio_duration was the
selection's song window, not the length of the audio in the file.
"""

import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from test_agent_api_stitch_frame_accurate import FPS, SCENE_FRAMES, StitchBase, _have_ffmpeg, _stream_duration  # noqa: E402

pkg_name = ROOT.name
video_files = importlib.import_module(f"{pkg_name}.runner.video_files")

WIDTH, HEIGHT = 160, 90
# Scene boundaries just inside half a frame off the grid, alternating early and late, so each clip still
# rounds to its live frame count but a selection's song window differs from its frames by up to 40 ms.
JITTER = 0.02


def jittered_timeline():
    bounds, total = [0.0], 0
    for index, frames in enumerate(SCENE_FRAMES, start=1):
        total += frames
        bounds.append(total / FPS + (JITTER if index % 2 else -JITTER))
    return [(bounds[i], bounds[i + 1]) for i in range(len(SCENE_FRAMES))]


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg/ffprobe not installed")
class StitchResultTests(StitchBase):
    def setUp(self):
        super().setUp()
        for index, (clip, frames) in enumerate(zip(self.clips, SCENE_FRAMES), start=1):
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}",
                "-f", "lavfi", "-i", f"sine=frequency={300 + 60 * index}:sample_rate=48000:duration={frames / FPS - 0.02:.6f}",
                "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", clip,
            ], capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:r=48000:d=40",
                        "-c:a", "pcm_s16le", self.song], capture_output=True, check=True)
        self.timeline = jittered_timeline()
        self.write_session()

    def assert_describes_file(self, result):
        out = result["final_video_path"]
        self.assertEqual((result["output_width"], result["output_height"]), (WIDTH, HEIGHT))
        self.assertAlmostEqual(result["audio_duration"], _stream_duration(out, "a:0"), places=3)
        self.assertAlmostEqual(result["output_audio_duration"], _stream_duration(out, "a:0"), places=3)
        self.assertAlmostEqual(result["output_duration"], _stream_duration(out, "v:0"), places=3)

    def test_embedded_preview_reports_the_file_not_the_window(self):
        result = self.run_stitch({"scene_ids": ["2", "3", "4"], "audio": "embedded", "output_prefix": "API_PREVIEW_SCENES_002-004"})
        self.assert_describes_file(result)
        start, end = self.timeline[1][0], self.timeline[3][1]
        frames = sum(SCENE_FRAMES[1:4])
        # The window is 40 ms shorter than the 441 frames; the file's audio is the frames' length.
        self.assertAlmostEqual(end - start, frames / FPS - 2 * JITTER, places=6)
        self.assertGreater(result["audio_duration"], frames / FPS - 0.001)
        self.assertLess(result["audio_duration"], frames / FPS + 1024 / 48000 + 0.001)
        # The requested window is kept under its own name.
        self.assertAlmostEqual(result["requested_audio_start"], start, places=6)
        self.assertAlmostEqual(result["requested_audio_duration"], end - start, places=6)

    def test_whole_project_reports_the_file(self):
        result = self.run_stitch({"audio": "embedded"})
        self.assert_describes_file(result)
        self.assertEqual(result["requested_audio_duration"], 0.0)  # whole song, no window requested
        self.assertGreater(result["audio_duration"], sum(SCENE_FRAMES) / FPS - 0.001)

    def test_project_song_preview_reports_the_file(self):
        result = self.run_stitch({"scene_ids": ["2", "3", "4"], "audio": "project"})
        self.assert_describes_file(result)

    def test_runner_reports_the_normalized_canvas(self):
        # LTX-style stitch with a canvas size: the result is the size of the written file.
        result = video_files._stitch_scene_videos({
            "project_folder": self.project_dir,
            "scene_paths": self.clips[:2],
            "audio_path": self.song,
            "width": 320,
            "height": 180,
            "output_prefix": "CANVAS",
        })
        out = result["final_video_path"]
        self.assertEqual((result["output_width"], result["output_height"]), (320, 180))
        self.assertAlmostEqual(result["output_audio_duration"], _stream_duration(out, "a:0"), places=3)


if __name__ == "__main__":
    unittest.main()
