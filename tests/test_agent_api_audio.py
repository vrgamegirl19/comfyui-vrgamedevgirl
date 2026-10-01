"""Attaching project audio must save the keys the Video Builder reads (audio_path, audio_peaks, beat_markers)."""

import importlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
builder_project = importlib.import_module(f"{pkg_name}.builder.project")


def write_click_track(path, seconds=6, rate=22050, bpm=120):
    """A WAV with a click every beat so beat detection has something to find."""
    frames = []
    beat_every = int(rate * 60 / bpm)
    for index in range(seconds * rate):
        click = 0.9 * math.sin(2 * math.pi * 880 * index / rate) if index % beat_every < 200 else 0.0
        frames.append(struct.pack("<h", int(click * 32767)))
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(frames))


class AttachAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        os.makedirs(self.root)
        self.song = os.path.join(self.temp, "song.wav")
        write_click_track(self.song)
        for target, name, value in (
            (builder_project, "_model_defaults_path", lambda: os.path.join(self.temp, "model_defaults.json")),
            (builder_project, "_project_target_from_payload", lambda payload, key: os.path.join(self.root, payload["project_name"])),
            (paths, "get_allowed_project_roots", lambda: [self.root]),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        mutations.create_project("Song")

    def session(self):
        with open(os.path.join(self.root, "Song", "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_attached_audio_is_saved_under_the_keys_the_ui_reads(self):
        result = mutations.attach_project_audio("Song", audio_path=self.song)
        session = self.session()
        self.assertTrue(os.path.isfile(session["audio_path"]))
        self.assertTrue(session["audio_path"].startswith(os.path.join(self.root, "Song")), "audio is copied into the project")
        self.assertAlmostEqual(session["audio_duration"], 6.0, delta=0.2)
        self.assertIsInstance(session["audio_peaks"], list)
        self.assertGreater(len(session["audio_peaks"]), 10)
        self.assertIsInstance(session["beat_markers"], list)
        self.assertEqual(result["audio_path"], session["audio_path"])
        self.assertEqual(result["beat_count"], len(session["beat_markers"]))

    def test_beats_set_through_the_api_use_the_ui_key_and_snapping_reads_them(self):
        mutations.attach_project_audio("Song", audio_path=self.song)
        mutations.set_audio_beats("Song", [1.0, 2.0, 3.0, 4.0], tempo_bpm=120.0)
        session = self.session()
        self.assertEqual(session["beat_markers"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(session["detected_tempo_bpm"], 120.0)
        self.assertNotIn("beats", session)
        reported = mutations.get_audio_beats("Song")
        self.assertEqual(reported["beats"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(reported["tempo_bpm"], 120.0)
        mutations.calibrate_beats("Song", offset_seconds=0.5)
        self.assertEqual(self.session()["beat_markers"], [1.5, 2.5, 3.5, 4.5])

    def test_older_sessions_with_a_beats_key_still_work(self):
        mutations.attach_project_audio("Song", audio_path=self.song)
        session = self.session()
        session.pop("beat_markers", None)
        session["beats"] = [2.0, 4.0]
        with open(os.path.join(self.root, "Song", "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(session, handle)
        self.assertEqual(mutations.get_audio_beats("Song")["beats"], [2.0, 4.0])

    def test_the_saved_file_keeps_the_source_extension(self):
        captured = {}

        def fake_save(payload):
            captured.update(payload)
            saved = os.path.join(self.root, "Song", "project_audio", "project_audio.mp3")
            os.makedirs(os.path.dirname(saved), exist_ok=True)
            shutil.copy2(self.song, saved)
            return {"saved_path": saved, "duration": 3.0, "peaks": [], "beats": [], "tempo_bpm": 0}

        with patch.object(mutations, "_save_project_audio", side_effect=fake_save):
            mutations.attach_project_audio("Song", audio_path=os.path.join(self.temp, "busting a nut.mp3"))
        self.assertEqual(captured["audio_name"], "busting a nut.mp3")

    def test_silent_audio_keeps_the_peak_list(self):
        mutations.create_project_silent_audio("Song", duration=5.0)
        session = self.session()
        self.assertTrue(os.path.isfile(session["audio_path"]))
        self.assertIsInstance(session["audio_peaks"], list)


if __name__ == "__main__":
    unittest.main()
