"""Scene copies and stitch re-encodes keep the render's quality instead of x264's CRF 23 / veryfast default.

Report: a MiniMax H3 render is CRF 19, but the trimmed scene copy (and with it the stitched final video) came out
as ``rc=crf crf=23.0`` with AAC at its default bitrate, because the ffmpeg commands passed no -crf and no -b:a.
"""

import importlib
import json
import os
import re
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
WIDTH, HEIGHT = 320, 180


def _have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _x264_setting(path: str, key: str) -> str:
    """Return one value from the x264 settings string embedded in an H.264 file."""
    match = re.search(rb"x264 - core .*?(?=\x00)", Path(path).read_bytes(), re.S)
    if not match:
        raise AssertionError(f"{path} has no x264 settings string")
    found = re.search(rf"\b{re.escape(key)}=(\S+)", match.group(0).decode("latin1"))
    if not found:
        raise AssertionError(f"{path} has no x264 setting {key!r}")
    return found.group(1)


def _audio_bitrate(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=bit_rate", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(json.loads(out)["streams"][0]["bit_rate"])


class SceneEncodeArgsTests(unittest.TestCase):
    def test_video_args_set_crf_and_preset(self):
        args = video_files._scene_video_encode_args()
        self.assertEqual(args[args.index("-crf") + 1], str(video_files.SCENE_VIDEO_CRF))
        self.assertEqual(args[args.index("-preset") + 1], video_files.SCENE_VIDEO_PRESET)
        self.assertLess(video_files.SCENE_VIDEO_CRF, 19, "scene copies must not be coarser than the CRF 19 render")

    def test_audio_args_set_bitrate(self):
        args = video_files._scene_audio_encode_args()
        self.assertEqual(args[args.index("-b:a") + 1], video_files.SCENE_AUDIO_BITRATE)

    def test_a_copy_is_never_coarser_than_its_source(self):
        crf = video_files._scene_video_crf
        self.assertEqual(crf(None), video_files.SCENE_VIDEO_CRF)
        self.assertEqual(crf(19), video_files.SCENE_VIDEO_CRF, "a CRF 19 render gets the default, which is finer")
        self.assertEqual(crf(12), 12, "a finer source keeps its own CRF")
        self.assertEqual(crf(12.9), 12, "a fractional CRF rounds toward the finer side")
        self.assertEqual(crf("nonsense"), video_files.SCENE_VIDEO_CRF)
        self.assertEqual(crf(-4), 0)

    def test_an_explicit_crf_wins_over_the_files(self):
        self.assertEqual(video_files._source_crf(["no-such-file.mp4"], 12), 12)
        self.assertIsNone(video_files._source_crf(["no-such-file.mp4"]))


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg/ffprobe not installed")
class SceneVideoQualityTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="vrgdg_quality_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        # A CRF 19 render, like the MiniMax H3 output the trim reads.
        self.raw = self.make_render(os.path.join(self.dir, "raw.mp4"), 19)

    def make_render(self, path, crf):
        """A short clip with audio written at the given CRF. Stereo noise, not a sine: AAC spends only the bits the
        signal needs, so a pure tone would not reach the requested bitrate."""
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}",
            "-f", "lavfi", "-i", "anoisesrc=color=pink:sample_rate=48000:amplitude=0.5",
            "-t", "2", "-ac", "2", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", path,
        ], capture_output=True, check=True)
        return path

    def trim(self, source, **extra):
        return video_files._trim_scene_video({
            "source_path": source, "project_folder": self.dir, "scene_number": 1, "start": 0, "duration": 1.5,
            "frames": 36, "label": "minimax_exact", "mark_as_audio_video": True, **extra,
        })["video_path"]

    def test_the_crf_of_a_file_is_read_from_the_file(self):
        self.assertEqual(video_files._probe_x264_crf(self.raw), 19.0)
        self.assertEqual(video_files._probe_x264_crf(self.make_render(os.path.join(self.dir, "fine.mp4"), 10)), 10.0)
        audio_only = os.path.join(self.dir, "audio.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=d=1", audio_only], capture_output=True, check=True)
        self.assertIsNone(video_files._probe_x264_crf(audio_only), "a file with no x264 stream states no CRF")
        self.assertIsNone(video_files._probe_x264_crf(os.path.join(self.dir, "missing.mp4")))

    def test_a_finer_render_keeps_its_crf_in_the_copy(self):
        fine = self.make_render(os.path.join(self.dir, "fine.mp4"), 10)
        self.assertEqual(float(_x264_setting(self.trim(fine), "crf")), 10.0)

    def test_an_explicit_crf_sets_the_copy(self):
        self.assertEqual(float(_x264_setting(self.trim(self.raw, crf=12), "crf")), 12.0)
        self.assertEqual(float(_x264_setting(self.trim(self.raw, crf=20), "crf")), float(video_files.SCENE_VIDEO_CRF))

    def test_a_stitch_of_finer_copies_keeps_their_crf(self):
        fine = self.make_render(os.path.join(self.dir, "fine.mp4"), 12)
        result = video_files._stitch_scene_videos({
            "project_folder": self.dir, "scene_paths": [fine, fine], "audio_path": self.raw,
            "timeline_fps": FPS, "use_embedded_scene_audio": True,
            "scene_timing_items": [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}],
        })
        self.assertEqual(float(_x264_setting(result["final_video_path"], "crf")), 12.0)

    def test_trim_keeps_quality(self):
        result = video_files._trim_scene_video({
            "source_path": self.raw, "project_folder": self.dir, "scene_number": 1,
            "start": 0, "duration": 1.5, "frames": 36, "label": "minimax_exact", "mark_as_audio_video": True,
        })
        out = result["video_path"]
        self.assertEqual(float(_x264_setting(out, "crf")), float(video_files.SCENE_VIDEO_CRF))
        self.assertGreaterEqual(_audio_bitrate(out), 200_000)

    def test_timeline_sync_reencode_keeps_quality(self):
        result = video_files._stitch_scene_videos({
            "project_folder": self.dir, "scene_paths": [self.raw, self.raw], "audio_path": self.raw,
            "timeline_fps": FPS, "use_embedded_scene_audio": True,
            "scene_timing_items": [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}],
        })
        self.assertEqual(float(_x264_setting(result["final_video_path"], "crf")), float(video_files.SCENE_VIDEO_CRF))

    def test_canvas_fit_reencode_keeps_quality(self):
        result = video_files._stitch_scene_videos({
            "project_folder": self.dir, "scene_paths": [self.raw], "audio_path": self.raw,
            "width": 640, "height": 360,
        })
        self.assertTrue(result["normalized_canvas"])
        self.assertEqual(float(_x264_setting(result["final_video_path"], "crf")), float(video_files.SCENE_VIDEO_CRF))

    def test_final_mux_keeps_audio_bitrate(self):
        result = video_files._stitch_scene_videos({
            "project_folder": self.dir, "scene_paths": [self.raw], "audio_path": self.raw,
        })
        self.assertGreaterEqual(_audio_bitrate(result["final_video_path"]), 200_000)


if __name__ == "__main__":
    unittest.main()
