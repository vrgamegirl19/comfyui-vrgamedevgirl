"""Server-side MiniMax H3 render uses the same settings and scene inputs as the Video Builder UI."""

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
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator")
video_orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.video_orchestrator")
video_files_mod = importlib.import_module(f"{pkg_name}.runner.video_files")


class MiniMaxRenderPayloadTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_minimax_render_")
        self.project_dir = os.path.join(self.test_dir, "MiniMaxProject")
        os.makedirs(self.project_dir)

        self.audio = os.path.join(self.test_dir, "song.wav")
        with wave.open(self.audio, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(b"\x00\x00\x00\x00" * 44100 * 3)

        self.location_image = os.path.join(self.test_dir, "roof.png")
        self.subject_image = os.path.join(self.test_dir, "ava.png")
        for path in (self.location_image, self.subject_image):
            Path(path).write_bytes(b"png")

        self.video = os.path.join(self.test_dir, "out.mp4")
        Path(self.video).write_bytes(b"\x00\x00\x00\x20ftypisom")

        self.session = {
            "project_name": "MiniMaxProject",
            "audio_path": self.audio,
            "revision": 1,
            "video_engine": "minimax_h3",
            "video_mode": "minimax_h3",
            "minimax_h3_settings": {
                "video_mode": "reference_to_video",
                "render_pass": "three_pass",
                "advanced_two_pass_grid_rows": 2,
                "advanced_two_pass_grid_cols": 3,
                "advanced_two_pass_spatial_w_overlap": 192,
                "advanced_two_pass_spatial_h_overlap": 192,
                "advanced_two_pass_dynamic_fade": "widening",
                "advanced_two_pass_pass2_megapixels": 7.97,
                "advanced_two_pass_pass2_resolution_preset": "4k",
            },
            "flux_reference_builder": {
                "use_subject_reference": True,
                "subject_count": 1,
                "subjects": [{"id": "s1", "name": "Ava", "image": {"path": self.subject_image}}],
                "locations": [{"id": "l1", "name": "Rooftop", "image": {"path": self.location_image}}],
                "scene_map": {"scene_001": "l1"},
            },
            "segments": [
                {"id": "scene_001", "start": 0.0, "end": 3.0, "i2v_prompt": "Ava sings on a rooftop", "minimax_h3_pass2_prompt": "sharp detail"},
            ],
        }
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)

        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir
        orch_mod.set_comfy_client(orch_mod.FakeComfyClient())

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _render(self, params=None):
        captured = {}

        def fake_build(mode, payload):
            captured["mode"] = mode
            captured["payload"] = dict(payload)
            return {"prompt": {"1": {"class_type": "Noop", "inputs": {}}}, "output_folder": self.project_dir}

        target = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-audio.mp4")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(self.video, target)
        with patch.object(video_orch_mod, "build_video_graph_for_mode", side_effect=fake_build), \
                patch.object(video_orch_mod, "resolve_comfy_video_path", return_value=self.video), \
                patch.object(video_files_mod, "_collect_scene_video", return_value={"video_path": target, "thumbnail_path": ""}):
            asyncio.run(video_orch_mod.render_scene_video_async("MiniMaxProject", "scene_001", params or {}))
        return captured

    def test_advanced_project_builds_the_advanced_graph_with_saved_ui_settings(self):
        captured = self._render({"mode": "minimax_h3"})
        payload = captured["payload"]
        self.assertEqual(captured["mode"], "minimax_h3_advanced_2pass")
        # The fixture is a project saved before the shared resolution: its Pass 2 preset (4k) becomes the
        # output resolution, and the old tile settings are ignored (the graph plans tiles from the resolution).
        self.assertEqual(payload["advanced_pass2_megapixels"], 7.9688)
        self.assertEqual(payload["megapixels"], 7.9688)
        self.assertEqual(payload["advanced_vram_preset"], "16gb")
        for key in ("advanced_grid_rows", "advanced_grid_cols", "advanced_spatial_w_overlap", "advanced_dynamic_fade"):
            self.assertNotIn(key, payload)
        self.assertEqual(payload["pass2_prompt"], "sharp detail")

    def test_reference_images_come_from_the_saved_reference_builder(self):
        payload = self._render({"mode": "minimax_h3"})["payload"]
        self.assertEqual(payload["image_paths"], [self.subject_image, self.location_image])
        self.assertEqual(payload["video_references"], [])
        self.assertEqual(payload["continuity_mode"], "off")

    def test_explicit_pass_mode_and_overrides_win(self):
        captured = self._render({"mode": "minimax_h3_2pass", "seed": 7})
        self.assertEqual(captured["mode"], "minimax_h3_2pass")
        self.assertEqual(captured["payload"]["seed"], 7)

    def test_randomize_seed_gives_new_seeds_and_saves_them(self):
        before = self.session["minimax_h3_settings"].get("seed")
        payload = self._render({"mode": "minimax_h3", "randomize_seed": True})["payload"]
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            saved = json.load(handle)["minimax_h3_settings"]
        for field in ("seed", "advanced_two_pass_pass1_seed", "advanced_two_pass_pass2_seed"):
            self.assertGreaterEqual(saved[field], 1)
        self.assertNotEqual(saved["seed"], before)
        self.assertEqual(payload["seed"], saved["seed"])
        self.assertEqual(payload["pass1_seed"], saved["advanced_two_pass_pass1_seed"])
        self.assertEqual(payload["pass2_seed"], saved["advanced_two_pass_pass2_seed"])

    def test_saved_minimax_prompt_is_used_not_the_lyric_line(self):
        self.session["segments"][0]["minimax_h3_prompt"] = "[Shot 1] Ava sings on the rooftop."
        self.session["segments"][0]["lyric_text"] = "la la la"
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)
        payload = self._render({"mode": "minimax_h3"})["payload"]
        self.assertEqual(payload["prompt"], "[Shot 1] Ava sings on the rooftop.")

    def test_a_scene_without_any_prompt_fails_instead_of_rendering_lyrics(self):
        self.session["segments"][0].pop("i2v_prompt", None)
        self.session["segments"][0]["lyric_text"] = "la la la"
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)
        with self.assertRaises(errors.ValidationError):
            self._render({"mode": "minimax_h3"})

    def test_the_final_clip_is_collected_not_the_pass_one_backup(self):
        history = {"p1": {"outputs": {
            "142": {"videos": [{"filename": "scene_0003_advanced_stage2_00002.mp4", "subfolder": "", "type": "output"}]},
            "9308": {"videos": [{"filename": "scene_0003_stage1_00002.mp4", "subfolder": "", "type": "output"}]},
        }}}
        videos = orch_mod.extract_videos_from_history(history, "p1", node_id="142")
        self.assertEqual([v["filename"] for v in videos], ["scene_0003_advanced_stage2_00002.mp4"])
        # Without a node id every video is returned, in node order (the old behavior picked the backup last).
        self.assertEqual(len(orch_mod.extract_videos_from_history(history, "p1")), 2)
        # A node that produced no video falls back to all videos instead of finding nothing.
        self.assertEqual(len(orch_mod.extract_videos_from_history(history, "p1", node_id="999")), 2)

    def test_minimax_render_asks_for_the_final_output_node(self):
        requested = []
        original = video_orch_mod.extract_videos_from_history

        def spy(history, prompt_id, node_id=None):
            requested.append(node_id)
            return original(history, prompt_id, node_id=node_id)

        with patch.object(video_orch_mod, "extract_videos_from_history", side_effect=spy):
            self._render({"mode": "minimax_h3"})
        self.assertEqual(requested, ["142"])

    def test_missing_reference_file_fails_before_queueing(self):
        os.remove(self.location_image)
        with self.assertRaises(errors.ValidationError) as ctx:
            self._render({"mode": "minimax_h3"})
        self.assertIn("roof.png", str(ctx.exception))

    def test_built_in_audio_with_multipass_is_rejected(self):
        self.session["minimax_h3_settings"]["audio_mode"] = "built_in_audio"
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)
        with self.assertRaises(errors.ValidationError):
            self._render({"mode": "minimax_h3"})



class EventLoopSafetyTests(unittest.TestCase):
    setUp = MiniMaxRenderPayloadTests.setUp
    tearDown = MiniMaxRenderPayloadTests.tearDown
    _render = MiniMaxRenderPayloadTests._render

    """ComfyUI answers /prompt on the same event loop that runs jobs, so blocking calls to it deadlock."""

    def test_queue_prompt_never_runs_on_the_event_loop_thread(self):
        client = orch_mod.FakeComfyClient()
        seen = []
        original = client.queue_prompt

        def spy(prompt, client_id=None):
            try:
                asyncio.get_running_loop()
                seen.append("on_event_loop")
            except RuntimeError:
                seen.append("worker_thread")
            return original(prompt, client_id)

        client.queue_prompt = spy
        orch_mod.set_comfy_client(client)
        self._render({"mode": "minimax_h3"})
        self.assertEqual(seen, ["worker_thread"])

    def test_job_routes_do_not_read_a_value_off_the_plain_string_status(self):
        router_source = (ROOT / "agent_api" / "router.py").read_text(encoding="utf-8")
        self.assertNotIn("job.status.value", router_source)
        self.assertNotIn("new_job.status.value", router_source)


if __name__ == "__main__":
    unittest.main()
