"""Speaking audio settings preserve source timing, ownership and ripple offsets."""

import copy
import importlib
import json
import math
import struct
import sys
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
audio = importlib.import_module(f"{ROOT.name}.builder.scene_audio_settings")
api = importlib.import_module(f"{ROOT.name}.agent_api.scene_audio")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")
paths = importlib.import_module(f"{ROOT.name}.agent_api.paths")
mixer = importlib.import_module(f"{ROOT.name}.builder.audio_clips")
atomic = importlib.import_module(f"{ROOT.name}.core.atomic_write")


def project():
    return {
        "video_type": "speaking", "revision": 7,
        "segments": [
            {"id": "a", "start": 0, "end": 4, "video_path": "video_0001.mp4"},
            {"id": "b", "start": 4, "end": 8, "video_path": "video_0002.mp4"},
            {"id": "c", "start": 10, "end": 14},
        ],
        "audio_clips": [
            {"id": "da", "scene_id": "a", "role": "dialogue", "start": 0, "duration": 6,
             "source_start": 2, "full_duration": 20, "path": "original.wav"},
            {"id": "db", "scene_id": "b", "role": "dialogue", "start": 4.2, "duration": 3,
             "source_start": 1, "path": "b.wav"},
            {"id": "score", "scene_id": "", "role": "music", "start": 0, "duration": 30, "path": "score.wav"},
        ],
    }


class SceneAudioSettingsTests(unittest.TestCase):
    def test_browser_state_and_shortcuts(self):
        completed = subprocess.run(["node", str(ROOT / "tests/scene_audio_settings.cjs")],
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_fit_silence_and_ripple_preserve_sources_and_other_settings(self):
        original = project()
        saved = copy.deepcopy(original)
        result = audio.update_audio_settings(original, {
            "use_project_defaults": False, "silence_before": .5, "silence_after": 1,
        }, "a")
        self.assertEqual(original, saved)
        self.assertEqual(result["segments"][0]["end"], 7.5)
        self.assertEqual((result["segments"][1]["start"], result["segments"][1]["end"]), (7.5, 11.5))
        self.assertEqual(result["segments"][2]["start"], 13.5)  # existing gap preserved
        self.assertEqual(result["audio_clips"][1]["start"], 7.7)
        self.assertEqual(result["audio_clips"][0]["start"], .5)
        self.assertEqual(result["audio_clips"][0]["source_start"], 2)
        self.assertEqual(result["audio_clips"][2], original["audio_clips"][2])
        self.assertTrue(result["segments"][0]["scene_audio_render_dirty"])
        self.assertNotIn("scene_audio_settings", result["segments"][1])
        self.assertNotIn("scene_audio_render_dirty", result["segments"][1])
        self.assertEqual(result["segments"][1]["video_path"], "video_0002.mp4")

    def test_repeated_save_is_idempotent_and_shorter_audio_ripples_left(self):
        result = audio.update_audio_settings(project(), {}, "a")
        again = audio.update_audio_settings(result, {}, "a")
        self.assertEqual(result, again)
        result["audio_clips"][0]["duration"] = 2
        shorter = audio.update_audio_settings(result, {}, "a")
        self.assertEqual(shorter["segments"][1]["start"], 2)
        self.assertEqual(shorter["audio_clips"][1]["start"], 2.2)

    def test_multiple_dialogue_pieces_keep_internal_gaps(self):
        session = project()
        session["audio_clips"].append({"id": "da2", "scene_id": "a", "role": "dialogue",
                                      "start": 7, "duration": 2, "source_start": 0, "path": "a2.wav"})
        result = audio.update_audio_settings(session, {"use_project_defaults": False, "silence_before": 1}, "a")
        self.assertEqual(result["segments"][0]["end"], 10)
        self.assertEqual(result["audio_clips"][-1]["start"], 8)
        self.assertEqual(audio.audio_settings_view(result, result["segments"][0])["audio_duration"], 9)

    def test_legacy_empty_role_is_dialogue_like_the_browser(self):
        session = project()
        session["audio_clips"][0]["role"] = ""
        result = audio.update_audio_settings(session, {}, "a")
        self.assertEqual(result["segments"][0]["end"], 6)

    def test_global_defaults_only_change_inherited_scenes(self):
        session = project()
        session["segments"][1]["scene_audio_settings"] = {"use_project_defaults": False, "silence_before": .2}
        result = audio.update_audio_settings(session, {"silence_before": .5, "silence_after": 1})
        self.assertEqual(result["segments"][0]["end"], 7.5)
        self.assertEqual(result["segments"][1]["end"] - result["segments"][1]["start"], 4)
        self.assertEqual(result["audio_clips"][1]["start"], 7.7)

    def test_fixed_duration_keeps_boundaries_and_view_reports_overflow(self):
        result = audio.update_audio_settings(project(), {"use_project_defaults": False,
            "silence_before": 1, "silence_after": 2, "fit_duration": False}, "a")
        view = audio.audio_settings_view(result, result["segments"][0])
        self.assertEqual(view["total_duration"], 9)
        self.assertEqual(view["scene_duration"], 4)
        self.assertEqual(result["segments"][1]["start"], 4)

    def test_replace_and_remove_dialogue_keep_independent_tracks(self):
        result = audio.update_audio_settings(project(), {}, "a", {
            "saved_path": "new.wav", "audio_name": "New", "duration": 2.5, "peaks": [1],
        })
        self.assertEqual(result["segments"][0]["end"], 2.5)
        self.assertEqual(audio.scene_clips(result, "a")[0]["path"], "new.wav")
        cleared = audio.update_audio_settings(result, {}, "a", {})
        self.assertEqual(audio.scene_clips(cleared, "a"), [])
        self.assertEqual(cleared["segments"][0]["custom_audio_path"], "")
        self.assertEqual(cleared["segments"][0]["end"], 2.5)
        self.assertEqual(cleared["audio_clips"][-1]["role"], "music")

    def test_legacy_scene_sources_migrate_without_consuming_global_audio(self):
        session = project()
        session["audio_clips"] = None
        session["audio_path"] = "global.wav"
        session["audio_duration"] = 30
        session["segments"][0].update(custom_audio_path="a.wav", custom_audio_duration=6)
        result = audio.update_audio_settings(session, {}, "a")
        self.assertEqual(len(result["audio_clips"]), 1)
        self.assertEqual(result["segments"][0]["end"], 6)
        session["segments"][0].pop("custom_audio_path")
        result = audio.update_audio_settings(session, {}, "a")
        self.assertEqual(result["audio_clips"][0]["scene_id"], "")
        self.assertEqual(result["segments"][0]["end"], 4)

    def test_invalid_settings_modes_and_busy_timelines_are_rejected(self):
        for settings in [{"silence_before": -1}, {"silence_after": float("nan")},
                         {"silence_before": "1"}, {"fit_duration": 1}, {"unknown": 0}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                audio.update_audio_settings(project(), settings, "a")
        for key, value in [("video_type", "singing"), ("timing_frozen", True)]:
            session = project(); session[key] = value
            with self.assertRaises(ValueError):
                audio.update_audio_settings(session, {}, "a")
        session = project(); session["segments"][1]["video_status"] = "running"
        with self.assertRaises(ValueError):
            audio.update_audio_settings(session, {}, "a")

    def test_api_revision_conflict_precedes_import_and_save(self):
        with patch.object(api, "_get_active_session_and_folder", return_value=("project", project())), \
             patch.object(api, "_persist_session") as save, patch.object(api, "_save_scene_audio") as load:
            with self.assertRaises(errors.RevisionConflictError):
                api.patch_audio_settings("p", {}, "a", if_match_revision=6, audio_data="data")
            save.assert_not_called(); load.assert_not_called()

    def test_api_and_browser_use_identical_timing_service(self):
        session = project()
        with patch.object(api, "_get_active_session_and_folder", return_value=("project", session)), \
             patch.object(api, "_persist_session", return_value={"revision": 8}) as save:
            response = api.patch_audio_settings("p", {"use_project_defaults": False, "silence_after": 1}, "a", 7)
            expected = audio.update_audio_settings(session, {"use_project_defaults": False, "silence_after": 1}, "a")
            self.assertEqual(save.call_args.args[1], expected)
            self.assertEqual(response["settings"]["total_duration"], 7)
            self.assertEqual(response["revision"], 8)

    def test_real_mixer_adds_silence_without_modifying_original(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "dialogue.wav")
            with wave.open(str(source), "wb") as output:
                output.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                output.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * n / 44100)))
                                          for n in range(44100)))
            original = source.read_bytes()
            result = mixer.prepare_audio_clip_mix({"project_folder": folder, "duration": 1.5,
                "clips": [{"path": str(source), "start": .2, "source_start": 0, "duration": 1, "volume": 1}]})
            self.assertEqual(source.read_bytes(), original)
            with wave.open(result["audio_path"], "rb") as rendered:
                self.assertAlmostEqual(rendered.getnframes() / rendered.getframerate(), 1.5, places=3)
                data = rendered.readframes(rendered.getnframes())
                samples = struct.unpack(f"<{len(data) // 2}h", data)
                rate = rendered.getframerate() * rendered.getnchannels()
                self.assertEqual(max(abs(n) for n in samples[:int(.15 * rate)]), 0)
                self.assertGreater(max(abs(n) for n in samples[int(.3 * rate):int(.8 * rate)]), 1000)
                self.assertEqual(max(abs(n) for n in samples[int(1.25 * rate):]), 0)

    def test_api_persists_timing_and_defaults_in_builder_session(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root, "SpeakingTest"); folder.mkdir()
            session = project(); session["project_folder"] = str(folder)
            session["project_name"] = "SpeakingTest"
            atomic.atomic_write_json(str(folder / "vrgdg_builder_session.json"), session)
            with patch.object(paths, "get_allowed_project_roots", return_value=[root]):
                pid = paths.get_project_id(str(folder))
                response = api.patch_audio_settings(
                    pid, {"silence_before": .5, "silence_after": 1}, if_match_revision=7,
                )
                persisted = json.loads((folder / "vrgdg_builder_session.json").read_text(encoding="utf-8"))
                self.assertEqual(persisted["speaking_audio_defaults"]["silence_after"], 1)
                self.assertEqual(persisted["segments"][0]["end"], 7.5)
                self.assertEqual(persisted["audio_clips"][0]["source_start"], 2)
                self.assertEqual(persisted["revision"], response["revision"])


if __name__ == "__main__":
    unittest.main()
