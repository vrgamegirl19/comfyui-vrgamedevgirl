"""An agent makes a RefMod from pictures and splits the song into stems: the checks, the jobs and the files."""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT.parent), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from test_audio_stems import stems_module  # noqa: E402  (imports builder/audio_stems.py with ComfyUI stubbed out)

pkg_name = ROOT.name
refmod_stems = importlib.import_module(f"{pkg_name}.agent_api.refmod_stems")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
refmod_studio = importlib.import_module(f"{pkg_name}.minimax.refmod_studio")

RATE = stems_module.SAMPLE_RATE


class FakeManager:
    def __init__(self):
        self.progress = []

    def update_progress(self, job_id, percent, stage, message=""):
        self.progress.append((percent, stage, message))


class FakeJob:
    def __init__(self, params, project_id=None):
        self.id = "job_1"
        self.params = params
        self.project_id = project_id


class SplitTheSongTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = os.path.join(self.temp.name, "Song")
        os.makedirs(self.project)
        self.song = os.path.join(self.project, "song.wav")
        with open(self.song, "wb") as handle:
            handle.write(b"RIFF")
        self.asked = []

        def fake_ffmpeg(command, message):
            # The song, read as 44.1 kHz stereo: two seconds of tone, at the file the command names last.
            self.asked.append(command)
            stems_module._write_wav(command[-1], np.full((2, 2 * RATE), 0.25, dtype=np.float32))

        def fake_demucs(mix, model_name, device):
            self.asked.append((mix.shape, model_name, device))
            length = mix.shape[1]
            stems = {"vocals": np.full((2, length + 7), 0.5, dtype=np.float32), "drums": np.full((2, length), 0.1, dtype=np.float32),
                     "bass": np.full((2, length), 0.1, dtype=np.float32), "other": np.full((2, length), 0.05, dtype=np.float32)}
            if model_name == "htdemucs_6s":
                stems.update({"guitar": np.full((2, length), 0.05, dtype=np.float32), "piano": np.full((2, length), 0.05, dtype=np.float32)})
            return stems

        for target in (patch.object(stems_module, "_ffmpeg", fake_ffmpeg), patch.object(stems_module, "_run_demucs", fake_demucs),
                       patch.object(stems_module, "_find_ffmpeg_path", return_value="ffmpeg")):
            target.start()
            self.addCleanup(target.stop)

    def split(self, **extra):
        return stems_module.separate_project_audio({"project_folder": self.project, "audio_path": self.song, **extra})

    def test_the_song_is_split_and_a_vocal_free_mix_is_written_beside_the_stems(self):
        result = self.split()
        folder = os.path.join(self.project, "project_audio", "stems")
        self.assertEqual(sorted(result["stems"]), ["bass", "drums", "other", "vocals"])
        self.assertEqual(result["vocals_path"], os.path.join(folder, "vocals.wav"))
        self.assertEqual(result["no_vocals_path"], os.path.join(folder, "no_vocals.wav"))
        self.assertEqual((result["model_name"], result["duration_seconds"]), ("htdemucs", 2.0))
        for name in ("vocals", "drums", "bass", "other", "no_vocals"):
            self.assertEqual(stems_module._read_wav(os.path.join(folder, f"{name}.wav")).shape, (2, 2 * RATE), name)
        vocals, rest = stems_module._read_wav(result["vocals_path"]), stems_module._read_wav(result["no_vocals_path"])
        self.assertAlmostEqual(float(vocals.mean()), 0.5, places=2)
        self.assertAlmostEqual(float(rest.mean()), 0.25, places=2, msg="the vocal-free mix is every stem but the vocals")
        self.assertTrue(os.path.isfile(os.path.join(folder, "stems.json")))
        self.assertFalse(os.path.exists(os.path.join(folder, "_full.wav")), "the temporary copy of the song is removed")

    def test_the_settings_reach_the_split_and_a_six_stem_model_keeps_its_stems(self):
        result = self.split(model_name="htdemucs_6s", device="cpu", input_gain_db=6)
        self.assertEqual(sorted(result["stems"]), ["bass", "drums", "guitar", "other", "piano", "vocals"])
        self.assertEqual(self.asked[1][1:], ("htdemucs_6s", "cpu"))
        self.assertIn("volume=6.00dB", " ".join(self.asked[0]))
        self.assertAlmostEqual(float(stems_module._read_wav(result["no_vocals_path"]).mean()), 0.35, places=2)
        # A split with a four stem model afterwards removes the stems it does not make.
        self.split()
        self.assertFalse(os.path.exists(os.path.join(self.project, "project_audio", "stems", "guitar.wav")))

    def test_bad_settings_and_a_missing_song_are_refused(self):
        for extra, message in (({"model_name": "demucs9"}, "Unknown Demucs model"), ({"device": "tpu"}, "Unknown device")):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self.split(**extra)
        with self.assertRaises(Exception):
            stems_module.separate_project_audio({"project_folder": self.project, "audio_path": os.path.join(self.project, "nope.wav")})
        with self.assertRaisesRegex(ValueError, "Project folder"):
            stems_module.separate_project_audio({"project_folder": os.path.join(self.temp.name, "missing"), "audio_path": self.song})


class AgentJobTests(unittest.TestCase):
    def test_a_bad_refmod_request_is_a_400_before_any_job_exists(self):
        good = {"name": "Darrel_noclothes", "type": "identity", "paths": []}
        for payload, message in (
            ({}, "cannot be created yet"),
            ({"name": "x", "type": "identity", "paths": []}, "Add at least one image"),
            ({"name": "", "type": "identity", "paths": ["a.png"]}, "Enter a name"),
            ({**good, "paths": ["C:/nope/missing.png"]}, "Image not found"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(errors.ValidationError, message):
                refmod_stems.check_refmod_request(payload)

    def test_a_refmod_that_exists_says_how_to_replace_it(self):
        manager = FakeManager()
        with patch.object(refmod_studio, "create_refmod", side_effect=refmod_studio.RefModExistsError("C:/refmods/identity/x.safetensors")):
            with self.assertRaises(errors.ValidationError) as caught:
                refmod_stems.run_refmod_create_job(FakeJob({"name": "x"}), manager)
        self.assertIn("overwrite: true", str(caught.exception))
        self.assertEqual(caught.exception.details["exists"], True)
        with patch.object(refmod_studio, "create_refmod", return_value={"name": "x", "folder": "identity"}) as made:
            self.assertEqual(refmod_stems.run_refmod_create_job(FakeJob({"name": "x"}), manager)["folder"], "identity")
        made.assert_called_once_with({"name": "x"})

    def test_the_split_uses_the_projects_song_unless_another_is_sent(self):
        with tempfile.TemporaryDirectory() as temp:
            song, other = os.path.join(temp, "song.wav"), os.path.join(temp, "other.wav")
            for path in (song, other):
                Path(path).write_bytes(b"RIFF")
            with patch.object(refmod_stems, "_get_active_session_and_folder", return_value=(temp, {"audio_path": song})):
                request = refmod_stems.stems_request("Song", {"model_name": "htdemucs_ft", "device": None})
                self.assertEqual(request, {"project_folder": temp, "audio_path": song, "model_name": "htdemucs_ft"})
                self.assertEqual(refmod_stems.stems_request("Song", {"audio_path": other})["audio_path"], other)
            with patch.object(refmod_stems, "_get_active_session_and_folder", return_value=(temp, {})):
                with self.assertRaisesRegex(errors.ValidationError, "no song to split"):
                    refmod_stems.stems_request("Song", {})
                with patch.object(refmod_stems, "separate_project_audio", return_value={"no_vocals_path": "x"}) as split:
                    manager = FakeManager()
                    with self.assertRaises(errors.ValidationError):
                        refmod_stems.run_audio_stems_job(FakeJob({}, "Song"), manager)
                    split.assert_not_called()


if __name__ == "__main__":
    unittest.main()
