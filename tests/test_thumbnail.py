"""Tests for the Thumbnail intro: saving the picture, putting it in front of a video, and the LLM prompt text."""

import importlib
import os
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
thumbnail = importlib.import_module(f"{pkg_name}.builder.thumbnail")
prompts = importlib.import_module(f"{pkg_name}.llm.prompts.thumbnail")
thumbnail_prompt = importlib.import_module(f"{pkg_name}.llm.thumbnail_prompt")

FFMPEG = shutil.which("ffmpeg")


def _make_video(path, seconds=3, audio=True, size="320x180", rate=24):
    command = [FFMPEG, "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={seconds}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    command += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
    subprocess.run([*command, path], check=True, capture_output=True)
    return path


def _make_image(path, color=(220, 30, 30), size=(640, 360)):
    Image.new("RGB", size, color).save(path)
    return path


def _probe(path, entries, select=None):
    command = ["ffprobe", "-v", "error"]
    if select:
        command += ["-select_streams", select]
    command += ["-show_entries", entries, "-of", "default=noprint_wrappers=1:nokey=1", path]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.split()


def _first_frame_color(path, at):
    """Average color of the frame at ``at`` seconds as an (r, g, b) tuple."""
    frame = path + ".png"
    subprocess.run([FFMPEG, "-y", "-ss", str(at), "-i", path, "-frames:v", "1", frame], check=True, capture_output=True)
    with Image.open(frame) as image:
        return image.convert("RGB").resize((1, 1)).getpixel((0, 0))


class SecondsTests(unittest.TestCase):
    def test_length_is_kept_in_range(self):
        self.assertEqual(thumbnail._intro_seconds(0), thumbnail.MIN_INTRO_SECONDS)
        self.assertEqual(thumbnail._intro_seconds(99), thumbnail.MAX_INTRO_SECONDS)
        self.assertEqual(thumbnail._intro_seconds("3.5"), 3.5)
        for bad in (None, "", "abc", float("nan")):
            self.assertEqual(thumbnail._intro_seconds(bad), thumbnail.DEFAULT_INTRO_SECONDS)


class SaveImageTests(unittest.TestCase):
    def test_each_save_gets_its_own_numbered_png(self):
        with tempfile.TemporaryDirectory() as project:
            source = _make_image(os.path.join(project, "source.png"))
            first = thumbnail._save_thumbnail_image({"project_folder": project, "source_path": source})
            second = thumbnail._save_thumbnail_image({"project_folder": project, "source_path": source})
            self.assertEqual(os.path.basename(first["image_path"]), "thumbnail_0001.png")
            self.assertEqual(os.path.basename(second["image_path"]), "thumbnail_0002.png")
            self.assertEqual((first["width"], first["height"]), (640, 360))
            self.assertEqual(os.path.dirname(first["image_path"]), os.path.join(project, "thumbnail"))

    def test_a_missing_or_non_image_file_is_refused(self):
        with tempfile.TemporaryDirectory() as project:
            with self.assertRaises(FileNotFoundError):
                thumbnail._save_thumbnail_image({"project_folder": project, "source_path": os.path.join(project, "nope.png")})
            text = os.path.join(project, "notes.txt")
            Path(text).write_text("not an image", encoding="utf-8")
            with self.assertRaises(Exception):
                thumbnail._save_thumbnail_image({"project_folder": project, "source_path": text})

    def test_an_empty_project_folder_is_refused(self):
        with self.assertRaises(ValueError):
            thumbnail._save_thumbnail_image({"project_folder": "", "source_path": "x.png"})


@unittest.skipUnless(FFMPEG, "ffmpeg is not installed")
class PrependTests(unittest.TestCase):
    def test_the_picture_shows_first_and_the_video_follows_with_its_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            image = _make_image(os.path.join(folder, "thumb.png"))
            result = thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": image, "seconds": 2})
            self.assertEqual(result["final_video_path"], video)
            self.assertAlmostEqual(result["duration"], 5.0, delta=0.4)
            self.assertAlmostEqual(float(_probe(video, "format=duration")[0]), 5.0, delta=0.4)
            self.assertEqual(_probe(video, "stream=width,height", "v:0"), ["320", "180"])
            self.assertEqual(_probe(video, "stream=codec_name", "a:0"), ["aac"])
            red, green, blue = _first_frame_color(video, 1.0)
            self.assertGreater(red, 150)
            self.assertLess(green, 90)
            self.assertLess(blue, 90)
            # The scene's own picture follows after the intro: testsrc2 is not red.
            self.assertFalse(_first_frame_color(video, 3.0)[0] > 150 and _first_frame_color(video, 3.0)[1] < 90)
            self.assertEqual(os.listdir(folder).count("FINAL_VIDEO.mp4"), 1)
            self.assertFalse([name for name in os.listdir(folder) if name.startswith(".thumbnail_intro")])

    def test_a_video_without_audio_gets_a_silent_free_intro(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "silent.mp4"), audio=False)
            image = _make_image(os.path.join(folder, "thumb.png"))
            thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": image, "seconds": 1})
            self.assertEqual(_probe(video, "stream=codec_type", "a:0"), [])
            self.assertAlmostEqual(float(_probe(video, "format=duration")[0]), 4.0, delta=0.4)

    def test_a_picture_of_another_shape_fills_the_frame_without_changing_the_video_size(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "wide.mp4"), size="640x360", rate=30)
            image = _make_image(os.path.join(folder, "tall.png"), size=(400, 800))
            thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": image, "seconds": 1})
            self.assertEqual(_probe(video, "stream=width,height", "v:0"), ["640", "360"])
            self.assertEqual(_probe(video, "stream=r_frame_rate", "v:0"), ["30/1"])

    def test_replace_source_false_leaves_the_original(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            image = _make_image(os.path.join(folder, "thumb.png"))
            result = thumbnail._prepend_thumbnail_intro(
                {"input_path": video, "image_path": image, "seconds": 1, "replace_source": False})
            self.assertEqual(os.path.basename(result["final_video_path"]), "FINAL_VIDEO_with_thumbnail.mp4")
            self.assertAlmostEqual(float(_probe(video, "format=duration")[0]), 3.0, delta=0.3)
            self.assertAlmostEqual(float(_probe(result["final_video_path"], "format=duration")[0]), 4.0, delta=0.4)

    def test_missing_files_are_refused_and_nothing_is_left_behind(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            before = sorted(os.listdir(folder))
            with self.assertRaises(FileNotFoundError):
                thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": os.path.join(folder, "missing.png")})
            with self.assertRaises(FileNotFoundError):
                thumbnail._prepend_thumbnail_intro({"input_path": os.path.join(folder, "x.mp4"), "image_path": video})
            self.assertEqual(sorted(os.listdir(folder)), before)


def _make_two_color_video(path, rate=24):
    """One second of red, then one second of blue: frames 0-23 are red and 24-47 are blue at 24 fps."""
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", f"color=c=red:s=160x90:r={rate}:d=1",
         "-f", "lavfi", "-i", f"color=c=blue:s=160x90:r={rate}:d=1",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        check=True, capture_output=True)
    return path


def _dominant(path):
    """``"red"``, ``"blue"`` or ``"other"`` for the average color of a picture."""
    with Image.open(path) as image:
        red, green, blue = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
    if red > 150 and blue < 100:
        return "red"
    if blue > 150 and red < 100:
        return "blue"
    return "other"


@unittest.skipUnless(FFMPEG, "ffmpeg is not installed")
class FrameTests(unittest.TestCase):
    def _take(self, project, video, seconds):
        return thumbnail._extract_thumbnail_frame({"project_folder": project, "video_path": video, "seconds": seconds})

    def test_the_frame_at_a_moment_is_the_frame_the_player_shows_there(self):
        with tempfile.TemporaryDirectory() as project:
            video = _make_two_color_video(os.path.join(project, "scene.mp4"))
            # Frame 23 is the last red one, frame 24 the first blue one: a cut that lands exactly on a frame edge.
            before = self._take(project, video, 23 / 24)
            after = self._take(project, video, 24 / 24)
            self.assertEqual((before["frame"], _dominant(before["image_path"])), (23, "red"))
            self.assertEqual((after["frame"], _dominant(after["image_path"])), (24, "blue"))
            self.assertEqual(_dominant(self._take(project, video, 0)["image_path"]), "red")
            self.assertEqual(_dominant(self._take(project, video, 0.5)["image_path"]), "red")
            self.assertEqual(_dominant(self._take(project, video, 1.5)["image_path"]), "blue")

    def test_a_moment_between_frames_snaps_to_the_nearest_frame(self):
        with tempfile.TemporaryDirectory() as project:
            video = _make_two_color_video(os.path.join(project, "scene.mp4"))
            snapped = self._take(project, video, 0.9999)
            self.assertEqual(snapped["frame"], 24)
            self.assertAlmostEqual(snapped["seconds"], 1.0, places=6)
            self.assertEqual(snapped["fps"], 24.0)
            self.assertEqual((snapped["width"], snapped["height"]), (160, 90))

    def test_a_moment_past_the_end_gives_the_last_frame(self):
        with tempfile.TemporaryDirectory() as project:
            video = _make_two_color_video(os.path.join(project, "scene.mp4"))
            last = self._take(project, video, 99)
            self.assertEqual(last["frame"], 47)
            self.assertEqual(_dominant(last["image_path"]), "blue")

    def test_each_frame_gets_its_own_numbered_file_in_the_thumbnail_folder(self):
        with tempfile.TemporaryDirectory() as project:
            video = _make_two_color_video(os.path.join(project, "scene.mp4"))
            names = [os.path.basename(self._take(project, video, 0.1)["image_path"]) for _ in range(2)]
            self.assertEqual(names, ["source_0001.png", "source_0002.png"])
            self.assertEqual(sorted(os.listdir(os.path.join(project, "thumbnail"))), names)

    def test_bad_requests_are_refused_and_leave_nothing_behind(self):
        with tempfile.TemporaryDirectory() as project:
            video = _make_two_color_video(os.path.join(project, "scene.mp4"))
            with self.assertRaises(ValueError):
                thumbnail._extract_thumbnail_frame({"project_folder": "", "video_path": video, "seconds": 0})
            with self.assertRaises(FileNotFoundError):
                self._take(project, os.path.join(project, "missing.mp4"), 0)
            for bad in (-1, "abc", None, float("nan")):
                with self.assertRaises(ValueError):
                    self._take(project, video, bad)
            self.assertFalse(os.path.isdir(os.path.join(project, "thumbnail")) and os.listdir(os.path.join(project, "thumbnail")))


class PromptTextTests(unittest.TestCase):
    def test_a_title_is_asked_for_exactly_and_no_title_means_no_text(self):
        with_title = prompts.thumbnail_instruction('NEVER "LOOK" BACK')
        self.assertIn("NEVER 'LOOK' BACK", with_title)
        self.assertNotIn("{text_rule}", with_title)
        without = prompts.thumbnail_instruction("   ")
        self.assertIn("Do not add any text", without)
        self.assertNotIn("{text_rule}", without)

    def test_the_reply_is_cleaned(self):
        clean = thumbnail_prompt.clean_thumbnail_prompt
        self.assertEqual(clean('```text\n"Recrop the attached image.\nMake it pop."\n```'), "Recrop the attached image. Make it pop.")
        self.assertEqual(clean("Prompt: Recrop the image."), "Recrop the image.")
        self.assertEqual(clean("Here is the prompt: Recrop the image."), "Recrop the image.")
        self.assertEqual(clean(None), "")

    def test_an_image_is_required(self):
        with self.assertRaises(ValueError):
            thumbnail_prompt._generate_thumbnail_prompt({"title": "x"})


if __name__ == "__main__":
    unittest.main()
