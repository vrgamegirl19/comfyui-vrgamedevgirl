"""An Agent API render records the continuity it really used on the scene, like the Video Builder.

The Builder (``prepareMiniMaxH3ContinuityReference`` in ``web/music_video_builder/video_render.mjs``) saves
``minimax_h3_continuity_mode_used`` and ``minimax_h3_continuity_source_scene_id`` on the scene it renders.
Cut and Gun S10 was rendered through the API with a scene-locked ``latent_continuation_masked`` and really
continued from S09's latent, but the session still said "off" with no source scene.
"""

import asyncio
import importlib
import json
import os
import shutil
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
orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator")
video_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.video_orchestrator")
video_files_mod = importlib.import_module(f"{pkg_name}.runner.video_files")
latents = importlib.import_module(f"{pkg_name}.minimax.latent_manager")

LOCKED = "latent_continuation_masked"


class ApiRenderRecordsContinuityTests(unittest.TestCase):
    """Three scenes; scene 3 (``seg_10``) has its own locked MiniMax settings and the project default is off."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_render_continuity_")
        self.project_dir = os.path.join(self.test_dir, "ContinuityProject")
        os.makedirs(self.project_dir)
        self.audio = os.path.join(self.test_dir, "song.wav")
        with wave.open(self.audio, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(b"\x00\x00\x00\x00" * 44100 * 9)
        self.location_image = os.path.join(self.test_dir, "safehouse.png")
        Path(self.location_image).write_bytes(b"png")
        self.video = os.path.join(self.test_dir, "out.mp4")
        Path(self.video).write_bytes(b"\x00\x00\x00\x20ftypisom")

        scene_settings = {"video_mode": "reference_to_video", "render_pass": "single",
                          "continuity_mode": LOCKED, "latent_context_frames": 39}
        self.session = {
            "project_name": "ContinuityProject",
            "audio_path": self.audio,
            "revision": 1,
            "video_engine": "minimax_h3",
            "video_mode": "minimax_h3",
            # Project default is OFF: only the scene's locked settings ask for masked continuation.
            "minimax_h3_settings": {"video_mode": "reference_to_video", "render_pass": "single", "continuity_mode": "off"},
            "flux_reference_builder": {
                "locations": [{"id": "l1", "name": "Safehouse", "image": {"path": self.location_image}}],
                "scene_map": {"seg_08": "l1", "seg_09": "l1", "seg_10": "l1"},
            },
            "segments": [
                {"id": "seg_08", "start": 0.0, "end": 3.0, "minimax_h3_prompt": "scene 8"},
                {"id": "seg_09", "start": 3.0, "end": 6.0, "minimax_h3_prompt": "scene 9", "video_path": self.video},
                {"id": "seg_10", "start": 6.0, "end": 9.0, "minimax_h3_prompt": "scene 10",
                 "use_scene_minimax_h3_settings": True, "minimax_h3_settings": scene_settings,
                 # What the session held after the API render on Osiris.
                 "minimax_h3_continuity_mode_used": "off", "minimax_h3_continuity_source_scene_id": ""},
            ],
        }
        self._save_session()
        import torch

        latents.SceneLatentManager.save_latent(
            self.project_dir, 2, {"video": torch.zeros(1, 24, 12, 4, 4), "audio": torch.zeros(1, 32, 2, 65)},
            metadata={"tail_padding_frames": 0},
        )
        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir
        orch_mod.set_comfy_client(orch_mod.FakeComfyClient())

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _save_session(self):
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)

    def _saved_segment(self, scene_id):
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        return next(s for s in saved["segments"] if s["id"] == scene_id)

    def _render(self, scene_id, graph_latent_settings="omit"):
        """Render with a mocked graph builder and ComfyUI. ``graph_latent_settings`` is what the graph reports."""
        captured = {}

        def fake_build(mode, payload):
            captured["payload"] = dict(payload)
            result = {"prompt": {"1": {"class_type": "Noop", "inputs": {}}}, "output_folder": self.project_dir}
            if graph_latent_settings != "omit":
                result["latent_continuation_settings"] = graph_latent_settings
            return result

        number = [s["id"] for s in self.session["segments"]].index(scene_id) + 1
        target = os.path.join(self.project_dir, "rendered_scene_videos", f"video_{number:04d}-audio.mp4")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(self.video, target)
        with patch.object(video_orch_mod, "build_video_graph_for_mode", side_effect=fake_build), \
                patch.object(video_orch_mod, "resolve_comfy_video_path", return_value=self.video), \
                patch.object(video_files_mod, "_collect_scene_video", return_value={"video_path": target, "thumbnail_path": ""}):
            asyncio.run(video_orch_mod.render_scene_video_async("ContinuityProject", scene_id, {"mode": "minimax_h3"}))
        return captured

    def test_scene_locked_masked_continuation_is_recorded_with_its_source(self):
        # The graph loaded scene 2's latent with 39 context frames (what runner/minimax_patches.py reports).
        captured = self._render("seg_10", {"enabled": True, "mode": LOCKED, "predecessor_scene": 2, "context_frames": 39})
        self.assertEqual(captured["payload"]["continuity_mode"], LOCKED)
        seg = self._saved_segment("seg_10")
        self.assertEqual(seg["minimax_h3_continuity_mode_used"], LOCKED)
        self.assertEqual(seg["minimax_h3_continuity_source_scene_id"], "seg_09")

    def test_without_a_graph_report_the_resolved_scene_settings_decide(self):
        self._render("seg_10")
        seg = self._saved_segment("seg_10")
        self.assertEqual(seg["minimax_h3_continuity_mode_used"], LOCKED)
        self.assertEqual(seg["minimax_h3_continuity_source_scene_id"], "seg_09")

    def test_a_scene_locked_off_records_off_even_when_the_project_default_is_masked(self):
        self.session["minimax_h3_settings"]["continuity_mode"] = LOCKED
        seg = self.session["segments"][2]
        seg["minimax_h3_settings"]["continuity_mode"] = "off"
        seg["minimax_h3_continuity_mode_used"] = LOCKED
        seg["minimax_h3_continuity_source_scene_id"] = "seg_09"
        self._save_session()
        captured = self._render("seg_10", {"enabled": False, "reason": "Continuity mode is not latent_continuation_masked"})
        self.assertEqual(captured["payload"]["continuity_mode"], "off")
        seg = self._saved_segment("seg_10")
        self.assertEqual(seg["minimax_h3_continuity_mode_used"], "off")
        self.assertEqual(seg["minimax_h3_continuity_source_scene_id"], "")

    def test_a_graph_that_did_not_load_a_latent_records_off(self):
        # Scene 1 asks for masked continuation, but the graph has no predecessor to load.
        self.session["segments"][0]["use_scene_minimax_h3_settings"] = True
        self.session["segments"][0]["minimax_h3_settings"] = dict(self.session["segments"][2]["minimax_h3_settings"])
        self._save_session()
        self._render("seg_08", {"enabled": False, "scene_number": 1, "reason": "Scene 1 is the opening scene; no predecessor latent needed"})
        seg = self._saved_segment("seg_08")
        self.assertEqual(seg["minimax_h3_continuity_mode_used"], "off")
        self.assertEqual(seg["minimax_h3_continuity_source_scene_id"], "")


if __name__ == "__main__":
    unittest.main()
