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


def _frame_color(path, number):
    """Average color of frame ``number`` (0 is the first frame) as an (r, g, b) tuple."""
    frame = f"{path}.{number}.png"
    subprocess.run([FFMPEG, "-y", "-i", path, "-vf", f"select=eq(n\\,{number})", "-frames:v", "1", "-fps_mode", "passthrough", frame],
                   check=True, capture_output=True)
    with Image.open(frame) as image:
        return image.convert("RGB").resize((1, 1)).getpixel((0, 0))


def _frame_count(path):
    """How many video frames a file holds, counted from its packets."""
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "default=noprint_wrappers=1:nokey=1", path],
                         check=True, capture_output=True, text=True).stdout.split()
    return int(out[0])


def _is_red(color):
    return color[0] > 150 and color[1] < 90 and color[2] < 90


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
    def test_the_picture_is_exactly_the_first_frame_and_the_video_follows_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            before = _frame_count(video)
            self.assertEqual(before, 72)
            second_frame_before = _frame_color(video, 1)
            image = _make_image(os.path.join(folder, "thumb.png"))
            result = thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": image})
            self.assertEqual(result["final_video_path"], video)
            self.assertEqual(result["intro_frames"], 1)
            self.assertAlmostEqual(result["intro_seconds"], 1 / 24, places=6)
            # One frame more, no more: the picture is frame 0, and the old first frame is now frame 1.
            self.assertEqual(_frame_count(video), before + 1)
            self.assertTrue(_is_red(_frame_color(video, 0)), "frame 0 is the picture")
            self.assertFalse(_is_red(_frame_color(video, 1)), "frame 1 is the video's own first frame")
            self.assertFalse(_is_red(_frame_color(video, 2)))
            self.assertEqual(_probe(video, "stream=width,height", "v:0"), ["320", "180"])
            self.assertEqual(_probe(video, "stream=codec_name", "a:0"), ["aac"])
            # The audio is not pushed by more than the one frame: the file is 1/24 s longer, within an audio frame.
            self.assertAlmostEqual(float(_probe(video, "format=duration")[0]), 3.0 + 1 / 24, delta=0.06)
            self.assertAlmostEqual(result["duration"], 3.0 + 1 / 24, delta=0.06)
            self.assertEqual(os.listdir(folder).count("FINAL_VIDEO.mp4"), 1)
            self.assertFalse([name for name in os.listdir(folder) if name.startswith(".thumbnail_intro")])
            del second_frame_before

    def test_a_length_in_the_request_does_not_make_the_picture_stay_longer(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            image = _make_image(os.path.join(folder, "thumb.png"))
            result = thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": image, "seconds": 5})
            self.assertEqual(_frame_count(video), 73)
            self.assertEqual(result["intro_frames"], 1)

    def test_a_video_without_audio_gets_no_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "silent.mp4"), audio=False)
            image = _make_image(os.path.join(folder, "thumb.png"))
            thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": image})
            self.assertEqual(_probe(video, "stream=codec_type", "a:0"), [])
            self.assertEqual(_frame_count(video), 73)

    def test_a_picture_of_another_shape_fills_the_frame_without_changing_the_video_size_or_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "wide.mp4"), size="640x360", rate=30)
            image = _make_image(os.path.join(folder, "tall.png"), size=(400, 800))
            result = thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": image})
            self.assertEqual(_probe(video, "stream=width,height", "v:0"), ["640", "360"])
            self.assertEqual(_probe(video, "stream=r_frame_rate", "v:0"), ["30/1"])
            self.assertEqual(_frame_count(video), 91, "3 s at 30 fps plus one frame")
            self.assertAlmostEqual(result["intro_seconds"], 1 / 30, places=6)

    def test_replace_source_false_leaves_the_original(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _make_video(os.path.join(folder, "FINAL_VIDEO.mp4"))
            image = _make_image(os.path.join(folder, "thumb.png"))
            result = thumbnail._prepend_thumbnail_intro({"input_path": video, "image_path": image, "replace_source": False})
            self.assertEqual(os.path.basename(result["final_video_path"]), "FINAL_VIDEO_with_thumbnail.mp4")
            self.assertEqual(_frame_count(video), 72)
            self.assertEqual(_frame_count(result["final_video_path"]), 73)

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

    def test_a_restyle_workflow_gets_a_description_of_the_finished_picture_not_an_edit_instruction(self):
        edit = prompts.thumbnail_instruction("RUN", "edit")
        restyle = prompts.thumbnail_instruction("RUN", "restyle")
        self.assertIn('Recrop the attached image', edit)
        self.assertNotIn('Recrop the attached image', restyle)
        self.assertIn("describes the finished thumbnail", restyle)
        self.assertIn("do not mention the attached image", restyle.lower())
        self.assertIn('"RUN"', restyle)
        self.assertNotIn("{text_rule}", restyle)
        self.assertIn("Do not add any text", prompts.thumbnail_instruction("", "restyle"))
        for other in (None, "", "edit", "anything else"):
            self.assertEqual(prompts.thumbnail_instruction("RUN", other), edit)

    def test_moss_image_edit_gets_the_short_add_and_keep_instruction_it_was_trained_on(self):
        moss = prompts.thumbnail_instruction("HEAT", "moss_edit")
        self.assertIn("in the form MOSS-Image Edit was trained on", moss)
        self.assertIn('one sentence that starts with "Keep"', moss)
        self.assertIn("always name the headwear, hair and clothes", moss)
        self.assertIn("leave out any cropping or reframing", moss)
        self.assertNotIn("Recrop the attached image", moss)
        self.assertNotIn("{text_rule}", moss)
        self.assertNotEqual(moss, prompts.thumbnail_instruction("HEAT", "edit"))
        self.assertIn("Do not add any text", prompts.thumbnail_instruction("", "moss_edit"))
        self.assertEqual(prompts.thumbnail_instruction("HEAT", " MOSS_EDIT "), moss, "the kind is matched without caring about case or spaces")

    def test_every_kind_starts_with_the_project_owners_brief_unchanged_and_then_asks_for_one_prompt(self):
        brief = prompts.THUMBNAIL_ROLE_INSTRUCTIONS
        self.assertTrue(brief.startswith("# Role and Objective\nYou are an expert AI image editor and creative director"))
        self.assertTrue(brief.endswith("optimized for small mobile screens."))
        for sentence in (
            "# Core Enhancement Guidelines", "# Output Constraints",
            "1. **Subject Isolation & Depth:** Separate the main artist/subject sharply from the background.",
            "Introduce strong **rim lighting** or colored edge-glows (e.g., contrasting neon cyan and magenta, electric blue, or warm sunset gold)",
            "Deepen shadows and boost mid-tone clarity on the artist’s face and clothing to create a high-end editorial look.",
            "Crop or frame tightly on the subject (chest-up or waist-up) with intense, engaging eye contact toward the camera.",
            "Maintain intentional **negative space** on one side of the frame (left or right) so a title text overlay can be added later.",
            "Apply a subtle, premium digital film grain or grade to unify the subject and background.",
            "- Keep the face, likeness, and key identity of the subject from the input image recognizable.",
            "- Deliver a clean, crisp, 16:9 aspect ratio final image optimized for small mobile screens.",
        ):
            self.assertIn(sentence, brief)
        for kind in ("edit", "restyle", "moss_edit"):
            with self.subTest(kind=kind):
                instruction = prompts.thumbnail_instruction("HEAT", kind)
                self.assertTrue(instruction.startswith(brief), "the brief comes first, untouched")
                after = instruction[len(brief):]
                self.assertTrue(after.startswith("\n\n# Your Answer\n"))
                self.assertIn("write the one prompt it will receive", after)
                self.assertIn("Output only the prompt: no title, no quotes around it, no explanation, no markdown.", after)
                self.assertNotIn("{text_rule}", instruction)
        # The three kinds differ only in the answer section.
        self.assertEqual(
            {prompts.thumbnail_instruction("HEAT", kind)[: len(brief)] for kind in ("edit", "restyle", "moss_edit")}, {brief})

    def test_qwen_image_edit_gets_a_prompt_about_the_change_not_the_picture(self):
        qwen = prompts.thumbnail_instruction("HEAT", "qwen_edit")
        self.assertNotIn("# Role and Objective", qwen, "Qwen does not use the brief: it already sees the frame")
        self.assertNotIn(prompts.THUMBNAIL_ROLE_INSTRUCTIONS, qwen)
        self.assertIn("The model already sees the frame, so the prompt describes only what to ADD or CHANGE, never the picture itself", qwen)
        self.assertIn("most of your words go there", qwen, "the lettering is where the words go")
        self.assertIn("Add the text \"<the title>\" as", qwen)
        for feature in ("lettering style", "fill colors or gradient", "outline and glow", "drips, splatter, sparks or shine", "decorative lines, swirls"):
            self.assertIn(feature, qwen)
        for placement in ("fully inside the frame", "how big it is", "never covers the person's face"):
            self.assertIn(placement, qwen)
        self.assertIn('starts with "Keep"', qwen)
        self.assertIn("no recropping or reframing, no blur or depth of field, no film grain", qwen)
        self.assertIn('The title text is mandatory: the prompt must contain the text "HEAT" in double quotes, spelled exactly as given', qwen)
        self.assertIn("wildstyle graffiti lettering with an orange-to-red gradient fill", qwen, "the example shows how much detail the lettering needs")
        self.assertNotIn("{text_rule}", qwen)
        self.assertNotEqual(qwen, prompts.thumbnail_instruction("HEAT", "edit"))
        self.assertEqual(prompts.thumbnail_instruction("HEAT", " QWEN_EDIT "), qwen, "the kind is matched without caring about case or spaces")
        self.assertIn('the text "NEVER \'LOOK\' BACK" in double quotes', prompts.thumbnail_instruction('NEVER "LOOK" BACK', "qwen_edit"))

    def test_qwen_image_edit_without_a_title_adds_no_text_and_ignores_the_example(self):
        qwen = prompts.thumbnail_instruction("  ", "qwen_edit")
        self.assertIn("There is no title text: skip step 1, do not add any text, letters or numbers", qwen)
        self.assertIn("Ignore the example below: it is about a title.", qwen)
        self.assertNotIn("is mandatory", qwen)
        self.assertNotIn("{text_rule}", qwen)

    def test_the_title_is_mandatory_and_quoted_in_every_kind(self):
        for kind in ("edit", "restyle", "moss_edit"):
            with self.subTest(kind=kind):
                instruction = prompts.thumbnail_instruction('NEVER "LOOK" BACK', kind)
                self.assertIn("is mandatory", instruction)
                self.assertIn("in double quotes", instruction)
                self.assertIn("\"NEVER 'LOOK' BACK\" in double quotes", instruction)
                self.assertIn("Follow any lettering style in the style notes", instruction)
                self.assertNotIn("is mandatory", prompts.thumbnail_instruction("", kind))

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
