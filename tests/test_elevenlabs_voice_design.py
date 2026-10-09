"""Voice Design provider contracts, selected runner and durable project drafts."""

import base64
import importlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
service = importlib.import_module(f"{ROOT.name}.builder.elevenlabs_voice_design")
llm = importlib.import_module(f"{ROOT.name}.llm.voice_design")
api = importlib.import_module(f"{ROOT.name}.agent_api.elevenlabs")
mutations = importlib.import_module(f"{ROOT.name}.agent_api.mutations")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")

DESCRIPTION = "A warm, mature voice with a gentle rasp, clear articulation and a measured cadence."


class VoiceDesignTests(unittest.TestCase):
    def test_preview_contract_and_explicit_creation_are_separate(self):
        response = {"previews": [{"generated_voice_id": "temporary1", "media_type": "audio/mpeg",
                                  "audio_base_64": base64.b64encode(b"audio").decode()}], "text": "Generated sample"}
        with patch.object(service, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as fetch:
            result = service.design_voice("secret", {"voice_description": DESCRIPTION})
        request = fetch.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.elevenlabs.io/v1/text-to-voice/design")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Xi-api-key"), "secret")
        body = json.loads(request.data)
        self.assertTrue(body["auto_generate_text"])
        self.assertFalse(body["should_enhance"])
        self.assertFalse(body["stream_previews"])
        self.assertEqual(body["model_id"], "eleven_ttv_v3")
        self.assertEqual(result["previews"][0]["generated_voice_id"], "temporary1")
        with patch.object(service, "urlopen", return_value=io.BytesIO(b'{"voice_id":"permanent1"}')) as fetch:
            saved = service.create_designed_voice("secret", {"voice_name": "Alice", "voice_description": DESCRIPTION, "generated_voice_id": "temporary1"})
        self.assertEqual(fetch.call_args.args[0].full_url, "https://api.elevenlabs.io/v1/text-to-voice")
        self.assertEqual(saved["voice_id"], "permanent1")
        self.assertEqual(saved["name"], "Alice")

    def test_custom_preview_text_and_validation_before_network(self):
        with patch.object(service, "_post_voice_design", return_value={"previews": []}) as fetch:
            with self.assertRaises(ValueError):
                service.design_voice("secret", {"voice_description": DESCRIPTION, "text": "x" * 100})
            self.assertFalse(fetch.call_args.args[2]["auto_generate_text"])
            self.assertEqual(fetch.call_args.args[2]["text"], "x" * 100)
            fetch.reset_mock()
            for body in ({"voice_description": "short"}, {"voice_description": DESCRIPTION, "model_id": "unknown"},
                         {"voice_description": DESCRIPTION, "text": "too short"}):
                with self.assertRaises(ValueError):
                    service.design_voice("secret", body)
            fetch.assert_not_called()

    def test_invalid_audio_and_ambiguous_save_never_retry_or_leak(self):
        response = {"previews": [{"generated_voice_id": "temporary1", "media_type": "audio/mpeg", "audio_base_64": "bad!"}]}
        with patch.object(service, "_post_voice_design", return_value=response):
            with self.assertRaisesRegex(ValueError, "invalid preview audio"):
                service.design_voice("secret", {"voice_description": DESCRIPTION})
        with patch.object(service, "urlopen", side_effect=URLError("secret")) as fetch:
            with self.assertRaisesRegex(ValueError, "Refresh account voices") as raised:
                service.create_designed_voice("secret", {"voice_name": "Alice", "voice_description": DESCRIPTION, "generated_voice_id": "temporary1"})
            self.assertNotIn("secret", str(raised.exception))
            self.assertEqual(fetch.call_count, 1)

    def test_llm_dispatches_selected_runner_and_only_returns_reviewable_text(self):
        for runner in ("llm_api", "own_server", "lm_studio", "builtin", "qwen_local"):
            payload = {"text_runner": runner, "user_input": "A warm storyteller", "model_file": "selected.safetensors"}
            with patch.object(llm, "_run_builder_text_llm", return_value=(DESCRIPTION, {"runner": runner})) as run:
                result = llm.generate_voice_description(payload)
            self.assertEqual(run.call_args.args[0]["text_runner"], runner)
            self.assertIn("A warm storyteller", run.call_args.args[1])
            self.assertEqual(result["voice_description"], DESCRIPTION)
        with patch.object(llm, "_run_builder_text_llm", return_value=("short", {})):
            with self.assertRaises(ValueError):
                llm.generate_voice_description({"user_input": "test"})

    def test_project_mode_key_and_saved_runner(self):
        session = {"video_type": "speaking", "elevenlabs_api_key_project": "saved-secret",
                   "text_gemma_runner": "llm_api", "flux_reference_builder": {"subjects": [{"id": "alice", "name": "Alice"}]}}
        with patch.object(api, "_get_active_session_and_folder", return_value=("folder", session)), \
             patch.object(api, "design_voice", return_value={"previews": []}) as design:
            self.assertEqual(api.project_voice_design("p", {"voice_description": DESCRIPTION}), {"previews": []})
            design.assert_called_with("saved-secret", {"voice_description": DESCRIPTION})
            request = api.voice_description_request("p", "alice", "warm storyteller")
            self.assertEqual(request["text_runner"], "llm_api")
            self.assertEqual(request["character_name"], "Alice")
            self.assertFalse(request["unload_after"])
            with self.assertRaises(errors.ValidationError):
                api.voice_description_request("p", "alice", "")
            session["video_type"] = "singing"
            design.reset_mock()
            with self.assertRaises(errors.ValidationError):
                api.project_voice_design("p", {})
            design.assert_not_called()

    def test_draft_upsert_mirrors_preserves_and_discards_temporary_audio(self):
        session = {"video_type": "speaking", "revision": 7, "flux_reference_builder": {"subjects": [{"id": "alice"}]}}
        draft = {"user_input": "warm", "voice_description": DESCRIPTION, "audio_base_64": "private", "generated_voice_id": "temp"}
        with patch.object(mutations, "_get_active_session_and_folder", return_value=("folder", session)), \
             patch.object(mutations, "_persist_session", return_value={"revision": 8}):
            first = mutations.upsert_reference_subject("p", "alice", {"elevenlabs_voice_design": draft}, 7)
            saved = first["subject"]["elevenlabs_voice_design"]
            self.assertNotIn("audio_base_64", saved)
            self.assertNotIn("generated_voice_id", saved)
            self.assertEqual(saved, session["flux_reference_builder"]["subject"]["elevenlabs_voice_design"])
            second = mutations.upsert_reference_subject("p", "alice", {"name": "Alice"}, 7)
            self.assertEqual(second["subject"]["elevenlabs_voice_design"], saved)
            with self.assertRaises(errors.ValidationError):
                mutations.upsert_reference_subject("p", "alice", {"elevenlabs_voice_design": {"voice_description": 42}}, 7)

    def test_description_job_prepares_loaded_model_and_redacts_runner_errors(self):
        request = {"text_runner": "lm_studio", "lmstudio_model": "saved-unloaded", "lmstudio_api_key": "secret"}
        prepared = {**request, "lmstudio_model": "already-loaded"}
        job = SimpleNamespace(id="job1", project_id="p", params={"subject_id": "alice", "user_input": "warm"})
        manager = SimpleNamespace(update_progress=lambda *args, **kwargs: None)
        with patch.object(api, "voice_description_request", return_value=request), \
             patch.object(api, "prepare_llm_payload", return_value=prepared) as prepare, \
             patch.object(api, "generate_voice_description", return_value={"voice_description": DESCRIPTION}) as generate:
            self.assertEqual(api.run_voice_description_job(job, manager)["voice_description"], DESCRIPTION)
            prepare.assert_called_once_with(request)
            generate.assert_called_once_with(prepared)
            generate.side_effect = ValueError("remote response secret")
            with self.assertRaises(errors.ValidationError) as raised:
                api.run_voice_description_job(job, manager)
            self.assertNotIn("secret", str(raised.exception))

    def test_browser_voice_design_flow(self):
        result = subprocess.run(["node", str(ROOT / "tests/elevenlabs_voice_design.cjs")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
