"""Agent/MCP parity for Krea reference cards and DMAD preflight."""
import asyncio
import copy
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

refs = importlib.import_module(f"{ROOT.name}.agent_api.orchestrator.reference_orchestrator")
images = importlib.import_module(f"{ROOT.name}.agent_api.orchestrator.image_orchestrator")
videos = importlib.import_module(f"{ROOT.name}.agent_api.orchestrator.video_orchestrator")
settings_mod = importlib.import_module(f"{ROOT.name}.minimax.settings_payload")
models = importlib.import_module(f"{ROOT.name}.runner.models")
workflows = importlib.import_module(f"{ROOT.name}.runner.image_workflows")
client_mod = importlib.import_module(f"{ROOT.name}.agent_api.orchestrator.comfy_client")
from mcp_server.tools import ALL_TOOLS


class ReferenceImageAgentTests(unittest.TestCase):
    def setUp(self):
        self.session = {
            "reference_krea2_settings": {"first_pass_width": 768, "first_pass_height": 1024, "width": 1536, "height": 2048, "seed": 12,
                                         "use_zimage_enhance": True, "z_unet_name": "missing-z-image.safetensors"},
            "flux_reference_builder": {
                "subjects": [{"id": "alice", "name": "Alice", "description": "Portrait of Alice", "image": {"path": "old.png"}}],
                "locations": [{"id": "roof", "name": "Roof", "description": "A rooftop at sunset", "image": {"path": ""}}],
            },
        }
        self.client = client_mod.FakeComfyClient()
        self.client.set_mock_output("fake_prompt_1", images=[{"filename": "new.png", "subfolder": "", "type": "output"}])
        for target, name, value in (
            (refs, "_get_active_session_and_folder", lambda _pid: ("project", self.session)),
            (refs, "get_comfy_client", lambda: self.client),
            (workflows, "_require_model_choice", lambda *_args: None),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_card_job_generates_two_krea_passes_and_saves_the_correct_reference(self):
        for kind, rid in (("subjects", "alice"), ("locations", "roof")):
            with self.subTest(kind=kind):
                self.client._prompt_counter = 0
                job = SimpleNamespace(id="job_ref", project_id="p", params={"kind": kind, "ref_id": rid, "settings": {"seed": 0}}, cancel_requested=False)
                with patch.object(refs, "_save_flux_reference_image", return_value={"saved_path": "new-reference.png"}) as save_image, patch.object(refs, "_persist_session", return_value={"revision": 2}):
                    result = asyncio.run(refs.run_reference_image_job(job, MagicMock()))
                graph = self.client.submitted_prompts[-1]["prompt"]
                self.assertEqual(graph["240"]["inputs"]["width"], 768)
                self.assertEqual(graph["226"]["inputs"]["width"], 1536)
                self.assertEqual(graph["231"]["inputs"]["model"], ["248", 0])
                self.assertEqual(graph["230"]["inputs"]["model"], ["248", 1])
                self.assertFalse(any("z_image" in str(node) for node in graph.values()))
                self.assertEqual(result["used_seed"], 0)
                self.assertEqual(result["revision"], 2)
                self.assertEqual(self.session["flux_reference_builder"][kind][0]["image"]["path"], "new-reference.png")
                self.assertEqual(save_image.call_args.args[0]["reference_type"], "subject" if kind == "subjects" else "location")

    def test_bad_card_mode_and_settings_fail_without_queueing(self):
        before = copy.deepcopy(self.session)
        cases = [("wrong", "alice", {}), ("subjects", "absent", {}), ("subjects", "alice", {"mode": "zimage"}), ("subjects", "alice", {"settings": []})]
        cases.extend(("subjects", "alice", {"timeout_seconds": value}) for value in (None, "bad", 0, -1, float("inf")))
        for kind, rid, params in cases:
            with self.subTest(params=params):
                with self.assertRaises(refs.ValidationError):
                    refs.prepare_reference_image_request("p", kind, rid, params)
        self.assertEqual(self.session, before)
        self.assertEqual(self.client.submitted_prompts, [])

    def test_cancelled_job_never_queues_or_changes_reference(self):
        job = SimpleNamespace(id="job_ref", project_id="p", params={"ref_id": "alice"}, cancel_requested=True)
        with self.assertRaises(refs.JobCancelledError):
            asyncio.run(refs.run_reference_image_job(job, MagicMock()))
        self.assertEqual(self.client.submitted_prompts, [])
        self.assertEqual(self.session["flux_reference_builder"]["subjects"][0]["image"]["path"], "old.png")

    def test_legacy_krea_alias_also_builds_krea_only_two_passes(self):
        graph = images.build_image_graph_for_mode("krea2", {"prompt": "A singer", **self.session["reference_krea2_settings"]})["prompt"]
        self.assertIn("230", graph)
        self.assertIn("231", graph)
        self.assertNotIn("193:87", graph)

    def test_mcp_tool_sends_settings_and_returns_job(self):
        client = MagicMock()
        client.post.return_value = {"ok": True, "data": {"job_id": "j1", "status": "queued"}}
        ALL_TOOLS["reference_image_generate"].handler(client, {"project_id": "p", "kind": "locations", "ref_id": "roof", "prompt": "A roof", "settings": {"seed": 0}})
        client.post.assert_called_once_with("/projects/p/references/locations/roof/image/generate", json_data={"prompt": "A roof", "settings": {"seed": 0}})

    def test_agent_dmad_render_refuses_missing_lora_before_compiling_or_queueing(self):
        dmad = settings_mod.set_minimax_h3_dmad({"video_mode": "reference_to_video", "render_pass": "two_pass"}, True)
        session = {"video_engine": "minimax_h3", "minimax_h3_settings": dmad,
                   "segments": [{"id": "s1", "start": 0, "end": 2}]}
        with patch.object(videos, "_get_active_session_and_folder", return_value=("project", session)), patch.object(models, "_folder_choices", return_value=[]), patch.object(videos, "build_video_graph_for_mode") as compile_graph, patch.object(videos, "get_comfy_client") as get_client:
            with self.assertRaisesRegex(videos.ValidationError, "DMAD rendering requires"):
                asyncio.run(videos.render_scene_video_async("p", "s1"))
        compile_graph.assert_not_called()
        get_client.assert_not_called()

    def test_batch_preflights_a_locked_dmad_scene_before_any_render(self):
        dmad = settings_mod.set_minimax_h3_dmad({"video_mode": "reference_to_video", "render_pass": "two_pass"}, True)
        session = {"video_engine": "minimax_h3", "minimax_h3_settings": {"video_mode": "reference_to_video", "render_pass": "two_pass"},
                   "segments": [{"id": "s1", "start": 0, "end": 2},
                                {"id": "s2", "start": 2, "end": 4, "use_scene_minimax_h3_settings": True, "minimax_h3_settings": dmad}]}
        job = SimpleNamespace(id="job_batch", project_id="p", params={}, cancel_requested=False)
        with patch.object(videos, "_get_active_session_and_folder", return_value=("project", session)), patch.object(models, "_folder_choices", return_value=[]), patch.object(videos, "render_scene_video_async") as render:
            with self.assertRaisesRegex(videos.ValidationError, "DMAD rendering requires"):
                asyncio.run(videos.run_batch_video_render_job(job, MagicMock()))
        render.assert_not_called()


if __name__ == "__main__":
    unittest.main()
