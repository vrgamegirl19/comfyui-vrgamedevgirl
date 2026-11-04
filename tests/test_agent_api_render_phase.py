"""The Agent API's scene render takes ``render_phase`` like the Video Builder's draft buttons.

``render_phase: "draft"`` is "Render Draft — Pass 1 Only": the two-pass graph is cut after pass 1 and the pass-1
latent is saved under ``minimax_drafts/<id>/``. ``render_phase: "finalize"`` is "Finalize Draft — Pass 2 Only": the
saved graph runs pass 2 from that latent. The scene keeps the same fields the Builder saves
(``finishMiniMaxVideoRender`` in ``web/music_video_builder/video_render.mjs``).
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
errors_mod = importlib.import_module(f"{pkg_name}.agent_api.errors")

TEMPLATE = ROOT / "Workflows/UsedForUIDoNotTouch/minimax_audio_driven_builder_latent_upscale_2pass_api.json"


class ApiRenderPhaseTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_render_phase_")
        self.project_dir = os.path.join(self.test_dir, "PhaseProject")
        os.makedirs(self.project_dir)
        self.audio = os.path.join(self.test_dir, "song.wav")
        with wave.open(self.audio, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(b"\x00\x00\x00\x00" * 44100 * 6)
        self.location_image = os.path.join(self.test_dir, "foyer.png")
        Path(self.location_image).write_bytes(b"png")
        self.video = os.path.join(self.test_dir, "out.mp4")
        Path(self.video).write_bytes(b"\x00\x00\x00\x20ftypisom")
        self.session = {
            "project_name": "PhaseProject",
            "audio_path": self.audio,
            "revision": 1,
            "video_engine": "minimax_h3",
            "video_mode": "minimax_h3",
            "minimax_h3_settings": {"video_mode": "reference_to_video", "render_pass": "two_pass",
                                    "continuity_mode": "off"},
            "flux_reference_builder": {
                "locations": [{"id": "l1", "name": "Foyer", "image": {"path": self.location_image}}],
                "scene_map": {"seg_01": "l1"},
            },
            "segments": [{"id": "seg_01", "start": 0.0, "end": 6.0, "minimax_h3_prompt": "scene 1"}],
        }
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)
        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir
        self.client = orch_mod.FakeComfyClient()
        orch_mod.set_comfy_client(self.client)

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _saved_segment(self):
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)["segments"][0]

    def _render(self, params, graph_extra=None):
        """Render scene 1 with a real two-pass template, a mocked ComfyUI and mocked file collection."""
        captured = {}

        def fake_build(mode, payload):
            captured["mode"] = mode
            captured["payload"] = dict(payload)
            prompt = json.loads(TEMPLATE.read_text(encoding="utf-8"))
            return {"prompt": prompt, "two_pass": {}, "timing": {"h3_frame_count": 150},
                    "post_render_trim": {"duration": 6.0}, "output_folder": self.project_dir, **(graph_extra or {})}

        target = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-audio.mp4")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(self.video, target)
        with patch.object(video_orch_mod, "build_video_graph_for_mode", side_effect=fake_build), \
                patch.object(video_orch_mod, "resolve_comfy_video_path", return_value=self.video), \
                patch.object(video_files_mod, "_trim_scene_video", return_value={"video_path": self.video}), \
                patch.object(video_files_mod, "_collect_scene_video",
                             side_effect=lambda payload: captured.setdefault("collect", payload) and
                             {"video_path": target, "thumbnail_path": ""}):
            captured["result"] = asyncio.run(
                video_orch_mod.render_scene_video_async("PhaseProject", "seg_01", {"mode": "minimax_h3", **params}))
        captured["queued"] = self.client.submitted_prompts[-1] if self.client.submitted_prompts else None
        return captured

    def test_draft_saves_the_pass1_latent_and_finalize_runs_pass2_from_it(self):
        draft = self._render({"render_phase": "draft"})
        self.assertEqual(draft["mode"], "minimax_h3_2pass")
        self.assertNotIn("render_phase", draft["payload"])
        draft_id = draft["result"]["draft_id"]
        self.assertTrue(os.path.isfile(os.path.join(self.project_dir, "minimax_drafts", draft_id, "workflow.json")))
        seg = self._saved_segment()
        self.assertEqual(seg["minimax_h3_draft"]["id"], draft_id)
        self.assertTrue(seg["minimax_h3_stage1_path"])
        self.assertEqual(seg["minimax_h3_stage2_path"], "")
        self.assertEqual(draft["collect"]["existing_action"], "backup")

        # The real render writes pass1.pt from the draft_save node.
        Path(self.project_dir, "minimax_drafts", draft_id, "pass1.pt").touch()
        final = self._render({"render_phase": "finalize"})
        self.assertEqual(final["result"]["render_phase"], "finalize")
        self.assertEqual(final["collect"]["existing_action"], "overwrite")
        seg = self._saved_segment()
        self.assertIsNone(seg["minimax_h3_draft"])
        self.assertEqual(seg["minimax_h3_stage1_path"], "")
        self.assertTrue(seg["minimax_h3_stage2_path"])

    def test_the_result_says_whether_pass_2_ran_in_windows(self):
        report = {"mode": "auto", "min_seconds": 5.0, "rendered_seconds": 6.0, "enabled": True, "windows": 4,
                  "reason": "6.00 s is longer than 5 s, so pass 2 runs in windows."}
        self.assertEqual(self._render({}, graph_extra={"chunked_pass2": report})["result"]["chunked_pass2"], report)
        self.assertIsNone(self._render({})["result"]["chunked_pass2"])

    def test_full_render_clears_an_old_draft_like_the_builder(self):
        self._render({})
        seg = self._saved_segment()
        self.assertIsNone(seg["minimax_h3_draft"])
        self.assertTrue(seg["minimax_h3_stage2_path"])

    def test_finalize_without_a_draft_and_unknown_phases_are_rejected(self):
        with self.assertRaises(errors_mod.ValidationError):
            self._render({"render_phase": "finalize"})
        with self.assertRaises(errors_mod.ValidationError):
            self._render({"render_phase": "preview"})

    def _save_latent(self, frames=150):
        latents = os.path.join(self.project_dir, "latents")
        os.makedirs(latents, exist_ok=True)
        Path(latents, "scene_001.latent").write_bytes(b"latent")
        Path(latents, "scene_001.latent.json").write_text(json.dumps({"frame_count": str(frames)}), encoding="utf-8")

    def test_redecode_decodes_the_saved_latent_and_leaves_the_rest_of_the_project_alone(self):
        self._save_latent()
        manager = importlib.import_module(f"{pkg_name}.minimax.latent_manager").SceneLatentManager
        with patch.object(manager, "mark_successor_dirty") as mark, patch.object(manager, "clear_dirty") as clear:
            done = self._render({"render_phase": "redecode", "crf": 10, "randomize_seed": True})
        # No sampling and no latent save: the compiled graph is cut down to the decode.
        self.assertTrue(done["payload"]["_skip_latent_patches"])
        queued = done["queued"]["prompt"]
        classes = {node["class_type"] for node in queued.values()}
        self.assertIn("VRGDG_MiniMaxH3LoadLatentFull", classes)
        self.assertNotIn("SamplerCustomAdvanced", classes)
        self.assertNotIn("VRGDG_MiniMaxH3SaveLatent", classes)
        self.assertEqual(queued["142"]["inputs"]["crf"], 10)
        # The old copy is backed up, not overwritten, and the result says what was done.
        self.assertEqual(done["collect"]["existing_action"], "backup")
        self.assertEqual(done["result"]["render_phase"], "redecode")
        self.assertEqual(done["result"]["redecode"]["crf"], 10)
        # The latent is the same one, so the next scene's continuation from it is still valid.
        mark.assert_not_called()
        clear.assert_not_called()
        # Seeds are not part of a decode, so asking for new ones changes nothing in the project.
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["minimax_h3_settings"], self.session["minimax_h3_settings"])

    def test_redecode_defaults_to_a_crf_finer_than_the_render(self):
        self._save_latent()
        done = self._render({"render_phase": "redecode"})
        self.assertEqual(done["queued"]["prompt"]["142"]["inputs"]["crf"], 12)

    def test_redecode_is_refused_without_a_latent_or_for_a_changed_scene_or_a_bad_crf_or_other_engines(self):
        with self.assertRaises(errors_mod.ValidationError):
            self._render({"render_phase": "redecode"})  # nothing saved yet
        self._save_latent(frames=175)
        with self.assertRaises(errors_mod.ValidationError):
            self._render({"render_phase": "redecode"})  # the scene now calls for 150 frames
        self._save_latent()
        with self.assertRaises(errors_mod.ValidationError):
            self._render({"render_phase": "redecode", "crf": 99})
        with self.assertRaises(errors_mod.ValidationError):
            self._render({"render_phase": "redecode", "mode": "t2v"})

    def test_draft_needs_a_two_pass_workflow(self):
        self.session["minimax_h3_settings"]["render_pass"] = "single"
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)
        with self.assertRaises(errors_mod.ValidationError):
            self._render({"render_phase": "draft"})


if __name__ == "__main__":
    unittest.main()
