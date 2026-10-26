"""Tests for the Video Builder's Export: the save dialog's start name and the copy to the chosen place."""

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
export_video = importlib.import_module(f"{pkg_name}.builder.export_video")


def _write(path, data=b"video"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


class ExportFileNameTests(unittest.TestCase):
    def test_names_are_safe_and_end_in_mp4(self):
        name = export_video._export_file_name
        self.assertEqual(name("My Song"), "My Song.mp4")
        self.assertEqual(name("My Song.mp4"), "My Song.mp4")
        self.assertEqual(name("My Song.MP4"), "My Song.mp4")
        self.assertEqual(name('a<b>:c"d/e\\f|g?h*i'), "a_b__c_d_e_f_g_h_i.mp4")
        self.assertEqual(name("night. "), "night.mp4")
        for empty in ("", None, "   ", "...", ".mp4"):
            self.assertEqual(name(empty), "video.mp4", repr(empty))


class ExportFinalVideoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = os.path.join(self._tmp.name, "project")
        self.out = os.path.join(self._tmp.name, "out")
        os.makedirs(self.out)
        self.stitch = _write(os.path.join(self.project, "EXPORT_VIDEO.mp4"), b"stitched video")

    def export(self, **overrides):
        payload = {
            "source_path": self.stitch,
            "destination_path": os.path.join(self.out, "song.mp4"),
            "project_folder": self.project,
            "remove_source": True,
        }
        payload.update(overrides)
        return export_video._export_final_video(payload)

    def read(self, path):
        with open(path, "rb") as handle:
            return handle.read()

    def test_copies_the_video_and_removes_the_stitch(self):
        result = self.export()
        self.assertEqual(result["destination_path"], os.path.join(self.out, "song.mp4"))
        self.assertEqual(result["size_bytes"], len(b"stitched video"))
        self.assertTrue(result["removed_source"])
        self.assertEqual(self.read(result["destination_path"]), b"stitched video")
        self.assertFalse(os.path.exists(self.stitch))
        self.assertEqual(os.listdir(self.out), ["song.mp4"], "no partial file is left behind")

    def test_keeps_the_stitch_unless_asked_to_remove_it(self):
        result = self.export(remove_source=False)
        self.assertFalse(result["removed_source"])
        self.assertTrue(os.path.isfile(self.stitch))

    def test_only_the_stitch_of_the_export_button_is_ever_removed(self):
        final = _write(os.path.join(self.project, "FINAL_VIDEO.mp4"))
        self.assertFalse(self.export(source_path=final)["removed_source"])
        self.assertTrue(os.path.isfile(final))
        nested = _write(os.path.join(self.project, "rendered_scene_videos", "EXPORT_VIDEO.mp4"))
        self.assertFalse(self.export(source_path=nested, destination_path=os.path.join(self.out, "b.mp4"))["removed_source"])
        self.assertTrue(os.path.isfile(nested))
        elsewhere = _write(os.path.join(self._tmp.name, "other", "EXPORT_VIDEO.mp4"))
        self.assertFalse(self.export(source_path=elsewhere, destination_path=os.path.join(self.out, "c.mp4"))["removed_source"])
        self.assertTrue(os.path.isfile(elsewhere))
        self.assertFalse(self.export(project_folder="", destination_path=os.path.join(self.out, "d.mp4"))["removed_source"])
        self.assertTrue(os.path.isfile(self.stitch))

    def test_a_name_without_an_extension_gets_mp4(self):
        result = self.export(destination_path=os.path.join(self.out, "no extension"))
        self.assertEqual(os.path.basename(result["destination_path"]), "no extension.mp4")
        self.assertTrue(os.path.isfile(result["destination_path"]))

    def test_only_an_mp4_name_is_accepted(self):
        for name in ("run.exe", "clip.mov", "notes.txt", "x.py"):
            with self.assertRaisesRegex(ValueError, "ends in .mp4"):
                self.export(destination_path=os.path.join(self.out, name))
        self.assertEqual(os.listdir(self.out), [])
        self.assertTrue(os.path.isfile(self.stitch))

    def test_refuses_a_missing_folder_source_or_empty_paths(self):
        with self.assertRaisesRegex(FileNotFoundError, "folder to save in does not exist"):
            self.export(destination_path=os.path.join(self.out, "missing", "song.mp4"))
        with self.assertRaisesRegex(FileNotFoundError, "stitched video was not found"):
            self.export(source_path=os.path.join(self.project, "none.mp4"))
        with self.assertRaisesRegex(ValueError, "not a supported video"):
            self.export(source_path=_write(os.path.join(self.project, "vrgdg_builder_session.json")))
        with self.assertRaisesRegex(ValueError, "path is empty"):
            self.export(source_path="  ")
        with self.assertRaisesRegex(ValueError, "Choose where to save"):
            self.export(destination_path="")
        with self.assertRaisesRegex(ValueError, "other than the stitched video"):
            self.export(destination_path=self.stitch)
        self.assertTrue(os.path.isfile(self.stitch))

    def test_a_video_already_there_is_replaced_and_survives_a_failed_copy(self):
        destination = _write(os.path.join(self.out, "song.mp4"), b"old video")
        with patch.object(export_video.shutil, "copyfile", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.export()
        self.assertEqual(self.read(destination), b"old video")
        self.assertEqual(os.listdir(self.out), ["song.mp4"], "the half written file is removed")
        self.assertTrue(os.path.isfile(self.stitch), "the stitch is kept when the copy failed")
        self.export()
        self.assertEqual(self.read(destination), b"stitched video")


class SaveDialogScriptTests(unittest.TestCase):
    """The PowerShell fallback of the picker builds a Save dialog that starts with a safe, quoted name."""

    def setUp(self):
        previous = sys.modules.get("folder_paths")
        sys.modules["folder_paths"] = types.SimpleNamespace(get_output_directory=lambda: "")

        def restore():
            if previous is None:
                sys.modules.pop("folder_paths", None)
            else:
                sys.modules["folder_paths"] = previous

        self.addCleanup(restore)
        self.paths = importlib.import_module(f"{pkg_name}.builder.paths")

    def script(self, file_name):
        done = types.SimpleNamespace(returncode=0, stdout="C:\\Out\\song.mp4\n", stderr="")
        with patch.object(self.paths.subprocess, "run", return_value=done) as run:
            path = self.paths._open_powershell_picker("save_video", file_name)
        self.assertEqual(path, "C:\\Out\\song.mp4")
        return run.call_args.args[0][-1]

    def test_the_dialog_is_a_save_dialog_for_mp4_that_asks_before_overwriting(self):
        script = self.script("My Song")
        self.assertIn("System.Windows.Forms.SaveFileDialog", script)
        self.assertIn("$dialog.Filter = 'MP4 video (*.mp4)|*.mp4'", script)
        self.assertIn("$dialog.DefaultExt = 'mp4'", script)
        self.assertIn("$dialog.OverwritePrompt = $true", script)
        self.assertIn("$dialog.FileName = 'My Song.mp4'", script)

    def test_a_quote_in_the_name_cannot_end_the_powershell_string(self):
        self.assertIn("$dialog.FileName = 'Don''t stop.mp4'", self.script("Don't stop"))
        injected = self.script("a'; Remove-Item x; '")
        self.assertIn("$dialog.FileName = 'a''; Remove-Item x; ''.mp4'", injected)


if __name__ == "__main__":
    unittest.main()
