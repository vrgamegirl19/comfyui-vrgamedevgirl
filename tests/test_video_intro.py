"""Tests for the video intro: the person's own clip joined in front of the stitched video."""

import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
video_intro = importlib.import_module(f"{pkg_name}.builder.video_intro")
thumbnail = importlib.import_module(f"{pkg_name}.builder.thumbnail")

FFMPEG = shutil.which("ffmpeg")


def _make_video(path, seconds=3, audio=True, size="320x180", rate=24, source="testsrc2"):
    """A video of a lavfi ``source``: the ``testsrc2`` test pattern, or ``color=c=red`` for a solid color."""
    spec = f"{source}:s={size}:r={rate}:d={seconds}" if "=" in source else f"{source}=s={size}:r={rate}:d={seconds}"
    command = [FFMPEG, "-y", "-f", "lavfi", "-i", spec]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    command += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
    subprocess.run([*command, path], check=True, capture_output=True)
    return path


def _red_video(path, seconds=2, audio=True, size="160x90", rate=24):
    return _make_video(path, seconds=seconds, audio=audio, size=size, rate=rate, source="color=c=red")


def _probe(path, entries, select=None):
    command = ["ffprobe", "-v", "error"]
    if select:
        command += ["-select_streams", select]
    command += ["-show_entries", entries, "-of", "default=noprint_wrappers=1:nokey=1", path]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.split()


def _frame_color(path, number):
    frame = f"{path}.{number}.png"
    subprocess.run([FFMPEG, "-y", "-i", path, "-vf", f"select=eq(n\\,{number})", "-frames:v", "1", "-fps_mode", "passthrough", frame],
                   check=True, capture_output=True)
    with Image.open(frame) as image:
        return image.convert("RGB").resize((1, 1)).getpixel((0, 0))


def _frame_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "default=noprint_wrappers=1:nokey=1", path],
                         check=True, capture_output=True, text=True).stdout.split()
    return int(out[0])


def _is_red(color):
    return color[0] > 150 and color[1] < 90 and color[2] < 90


def _prepend(folder, source, intro, **extra):
    return video_intro._prepend_video_intro({"input_path": source, "intro_path": intro, **extra})


@unittest.skipUnless(FFMPEG, "ffmpeg is not installed")
class PrependVideoIntroTests(unittest.TestCase):
    def test_the_intro_plays_first_and_the_video_follows_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            intro = _red_video(os.path.join(folder, "intro.mp4"), seconds=2)
            before = _frame_count(video)
            self.assertEqual(before, 72)
            result = _prepend(folder, video, intro)
            self.assertEqual(result["final_video_path"], video)
            self.assertEqual(result["intro_frames"], 48)
            self.assertEqual(result["method"], "copy", "the clip was made to match, so no second encode of the video is needed")
            self.assertAlmostEqual(result["intro_seconds"], 2.0, places=3)
            self.assertEqual(_frame_count(video), before + 48)
            self.assertTrue(_is_red(_frame_color(video, 0)), "the first frame is the intro's")
            self.assertTrue(_is_red(_frame_color(video, 47)), "the last frame of the intro is still the intro")
            self.assertFalse(_is_red(_frame_color(video, 48)), "the video's own first frame follows the intro")
            self.assertEqual(_probe(video, "stream=width,height", "v:0"), ["320", "180"])
            self.assertEqual(_probe(video, "stream=codec_name", "a:0"), ["aac"])
            self.assertAlmostEqual(float(_probe(video, "format=duration")[0]), 5.0, delta=0.08)
            self.assertAlmostEqual(result["duration"], 5.0, delta=0.08)
            self.assertFalse([name for name in os.listdir(folder) if name.startswith(".video_intro")])

    def test_an_intro_of_another_shape_and_rate_is_fitted_without_cropping(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"), size="640x360", rate=24)
            # A tall 30 fps intro: it keeps its whole picture inside the wide frame, with bars on the sides.
            intro = _red_video(os.path.join(folder, "tall.mp4"), seconds=2, size="180x320", rate=30)
            result = _prepend(folder, video, intro)
            self.assertEqual(_probe(video, "stream=width,height", "v:0"), ["640", "360"])
            self.assertEqual(_probe(video, "stream=r_frame_rate", "v:0"), ["24/1"])
            self.assertEqual(result["intro_frames"], 48, "2 s at the video's 24 fps, not the intro's 30")
            self.assertEqual(_frame_count(video), 72 + 48)
            frame = f"{video}.bars.png"
            subprocess.run([FFMPEG, "-y", "-i", video, "-vf", "select=eq(n\\,10)", "-frames:v", "1", "-fps_mode", "passthrough", frame],
                           check=True, capture_output=True)
            with Image.open(frame) as image:
                rgb = image.convert("RGB")
                left, centre = rgb.getpixel((5, 180)), rgb.getpixel((320, 180))
            self.assertLess(sum(left), 60, "black bar at the side")
            self.assertTrue(_is_red(centre), "the picture in the middle")

    def test_an_intro_without_sound_gets_silence_so_the_sound_stays_with_the_picture(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            intro = _red_video(os.path.join(folder, "silent.mp4"), seconds=2, audio=False)
            _prepend(folder, video, intro)
            self.assertEqual(_probe(video, "stream=codec_type", "a:0"), ["audio"])
            self.assertAlmostEqual(float(_probe(video, "stream=duration", "a:0")[0]), 5.0, delta=0.1)
            self.assertEqual(_frame_count(video), 72 + 48)

    def test_the_intro_sound_is_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            video = os.path.join(folder, "FINAL_VIDEO.mp4")
            subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=3", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                            "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", video], check=True, capture_output=True)
            intro = _red_video(os.path.join(folder, "intro.mp4"), seconds=2)
            _prepend(folder, video, intro)
            loud = subprocess.run([FFMPEG, "-i", video, "-t", "1.5", "-af", "volumedetect", "-vn", "-f", "null", "-"],
                                  capture_output=True, text=True).stderr
            mean = float(re.search(r"mean_volume: (-?[\d.]+) dB", loud).group(1))
            self.assertGreater(mean, -40, "the intro's sine tone plays in the first seconds")

    def test_a_video_without_sound_gets_no_sound(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "silent.mp4"), audio=False)
            intro = _red_video(os.path.join(folder, "intro.mp4"), seconds=2)
            _prepend(folder, video, intro)
            self.assertEqual(_probe(video, "stream=codec_type", "a:0"), [])
            self.assertEqual(_frame_count(video), 72 + 48)

    def test_replace_source_false_leaves_the_original(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            intro = _red_video(os.path.join(folder, "intro.mp4"))
            result = _prepend(folder, video, intro, replace_source=False)
            self.assertTrue(result["final_video_path"].endswith("FINAL_VIDEO_with_intro.mp4"))
            self.assertEqual(_frame_count(video), 72)
            self.assertEqual(_frame_count(result["final_video_path"]), 72 + 48)

    def test_missing_files_and_the_video_itself_are_refused_and_nothing_is_left_behind(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"), seconds=2)
            intro = _red_video(os.path.join(folder, "intro.mp4"))
            before = sorted(os.listdir(folder))
            with self.assertRaisesRegex(FileNotFoundError, "intro video was not found"):
                _prepend(folder, video, os.path.join(folder, "missing.mp4"))
            with self.assertRaisesRegex(FileNotFoundError, "video to put the intro in front of"):
                _prepend(folder, os.path.join(folder, "x.mp4"), intro)
            with self.assertRaisesRegex(ValueError, "the final video itself"):
                _prepend(folder, video, video)
            text = os.path.join(folder, "notes.txt")
            Path(text).write_text("not a video", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                _prepend(folder, video, text)
            fake = os.path.join(folder, "fake.mp4")
            Path(fake).write_text("not a video", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _prepend(folder, video, fake)
            self.assertEqual(_frame_count(video), 48, "the video is untouched after a refusal")
            self.assertEqual(sorted(os.listdir(folder)), sorted([*before, "notes.txt", "fake.mp4"]))


class CapabilityTests(unittest.TestCase):
    def test_the_capabilities_offer_the_video_intro(self):
        self.assertIs(thumbnail._thumbnail_capabilities()["video_intro"], True)


if __name__ == "__main__":
    unittest.main()
