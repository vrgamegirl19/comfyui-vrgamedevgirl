"""ElevenLabs discovery, credential persistence and subject/UI parity."""

import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
service = importlib.import_module(f"{ROOT.name}.builder.elevenlabs")
api = importlib.import_module(f"{ROOT.name}.agent_api.elevenlabs")
mutations = importlib.import_module(f"{ROOT.name}.agent_api.mutations")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")
paths = importlib.import_module(f"{ROOT.name}.agent_api.paths")
atomic = importlib.import_module(f"{ROOT.name}.core.atomic_write")


class ElevenLabsTests(unittest.TestCase):
    def test_discovery_uses_auth_header_pagination_and_safe_public_previews(self):
        data = {"voices": [
            {"voice_id": "voice1", "name": "Test voice", "category": "generated",
             "preview_url": "https://example.com/sample.mp3", "samples": ["private metadata"]},
            {"voice_id": "voice2", "preview_url": "javascript:bad"},
        ], "has_more": True, "next_page_token": "page 2"}
        with patch.object(service, "urlopen", return_value=io.BytesIO(json.dumps(data).encode())) as fetch:
            result = service.list_voices("test-secret", "page 1")
        request = fetch.call_args.args[0]
        self.assertEqual(request.get_header("Xi-api-key"), "test-secret")
        self.assertIn("next_page_token=page+1", request.full_url)
        self.assertNotIn("test-secret", request.full_url)
        self.assertEqual(fetch.call_args.kwargs["timeout"], 25)
        self.assertEqual(result["next_page_token"], "page 2")
        self.assertNotIn("samples", result["voices"][0])
        self.assertEqual(result["voices"][1]["preview_url"], "")

    def test_provider_errors_never_echo_secret_or_upstream_body(self):
        for code in (401, 403, 429, 500):
            exc = HTTPError("https://api.elevenlabs.io", code, "test-secret", {}, io.BytesIO(b"test-secret"))
            with patch.object(service, "urlopen", side_effect=exc):
                with self.assertRaises(ValueError) as raised:
                    service.list_voices("test-secret")
                self.assertNotIn("test-secret", str(raised.exception))
        with patch.object(service, "urlopen", side_effect=URLError("test-secret")):
            with self.assertRaises(ValueError) as raised:
                service.list_voices("test-secret")
            self.assertNotIn("test-secret", str(raised.exception))

    def test_key_and_response_validation(self):
        with patch.object(service, "urlopen") as fetch:
            for key in ("", None, "abc\r\nheader", "x" * 513):
                with self.assertRaises(ValueError):
                    service.list_voices(key)
            fetch.assert_not_called()
        for data in ({"voices": [] , "has_more": True}, {"voices": [], "has_more": True, "next_page_token": "same"}, []):
            with patch.object(service, "urlopen", return_value=io.BytesIO(json.dumps(data).encode())):
                with self.assertRaises(ValueError):
                    service.list_voices("test-secret", "same")

    def test_permission_error_on_401_is_not_reported_as_an_invalid_key(self):
        data = {"detail": {"status": "missing_permissions", "message": "test-secret lacks voices_read"}}
        body = io.BytesIO(json.dumps(data).encode())
        error = HTTPError("https://api.elevenlabs.io", 401, "Unauthorized", {}, body)
        with patch.object(service, "urlopen", side_effect=error):
            with self.assertRaises(ValueError) as raised:
                service.list_voices("test-secret")
        message = str(raised.exception)
        self.assertIn("Voices → Read", message)
        self.assertNotIn("invalid", message)
        self.assertNotIn("test-secret", message)
        self.assertTrue(body.closed)

    def test_invalid_key_code_and_unknown_auth_body_are_distinguished(self):
        cases = [
            (json.dumps({"detail": {"status": "invalid_api_key", "message": "test-secret"}}).encode(), "invalid_api_key"),
            (b"not json test-secret", "HTTP 401"),
            (b"x" * 8193, "HTTP 401"),
            (json.dumps({"detail": {"status": ["invalid_api_key"], "message": "test-secret"}}).encode(), "HTTP 401"),
        ]
        for body, expected in cases:
            error = HTTPError("https://api.elevenlabs.io", 401, "Unauthorized", {}, io.BytesIO(body))
            with patch.object(service, "urlopen", side_effect=error):
                with self.assertRaises(ValueError) as raised:
                    service.list_voices("test-secret")
            self.assertIn(expected, str(raised.exception))
            self.assertNotIn("test-secret", str(raised.exception))

    def test_project_writes_are_revision_checked_and_status_is_redacted(self):
        session = {"video_type": "speaking", "revision": 7}
        with patch.object(api, "_get_active_session_and_folder", return_value=("project", session)), \
             patch.object(api, "_persist_session", return_value={"revision": 8}) as save:
            with self.assertRaises(errors.RevisionConflictError):
                api.project_credentials("p", "test-secret", 6)
            save.assert_not_called()
            result = api.project_credentials("p", "test-secret", 7)
            self.assertEqual(result, {"configured": True, "revision": 8})
            self.assertEqual(save.call_args.args[1]["elevenlabs_api_key_project"], "test-secret")
            self.assertNotIn("test-secret", json.dumps(result))
            api.project_credentials("p", "", 7)
            self.assertEqual(session["elevenlabs_api_key_project"], "")

    def test_mode_guards_precede_writes_or_network(self):
        session = {"video_type": "singing", "revision": 7}
        with patch.object(api, "_get_active_session_and_folder", return_value=("project", session)), \
             patch.object(api, "_persist_session") as save, patch.object(api, "list_voices") as fetch:
            for action in (lambda: api.project_credentials("p"),
                           lambda: api.project_credentials("p", "test-secret"), lambda: api.project_voices("p")):
                with self.assertRaises(errors.ValidationError):
                    action()
            save.assert_not_called()
            fetch.assert_not_called()

    def test_saved_key_used_for_discovery_and_test(self):
        session = {"video_type": "speaking", "elevenlabs_api_key_project": "saved-secret"}
        with patch.object(api, "_get_active_session_and_folder", return_value=("project", session)), \
             patch.object(api, "list_voices", return_value={"voices": []}) as fetch:
            self.assertEqual(api.project_voices("p", "page"), {"voices": []})
            fetch.assert_called_with("saved-secret", "page")
            self.assertEqual(api.project_voices("p", test=True), {"connected": True})

    def test_subject_assignment_validation_and_preservation(self):
        voice = {"enabled": True, "voice_id": "voice1", "name": "Test"}
        session = {"video_type": "speaking", "revision": 7,
                   "flux_reference_builder": {"subjects": [{"id": "s", "elevenlabs_voice": voice}]}}
        with patch.object(mutations, "_get_active_session_and_folder", return_value=("project", session)), \
             patch.object(mutations, "_persist_session", return_value={"revision": 8}) as save:
            result = mutations.upsert_reference_subject("p", "s", {"name": "Alice"}, 7)
            self.assertEqual(result["subject"]["elevenlabs_voice"], voice)
            for payload in ({"enabled": True}, {"enabled": "yes"}, {"voice_id": "../bad"}):
                with self.assertRaises(errors.ValidationError):
                    mutations.upsert_reference_subject("p", "s", {"elevenlabs_voice": payload}, 7)
            with self.assertRaises(errors.ValidationError):
                mutations.upsert_reference_subject("p", "s", {"reference_type": "prop", "elevenlabs_voice": voice}, 7)
            session["video_type"] = "singing"
            save.reset_mock()
            with self.assertRaises(errors.ValidationError):
                mutations.upsert_reference_subject("p", "s", {"elevenlabs_voice": voice}, 7)
            save.assert_not_called()

    def test_real_session_persistence_matches_ui_and_voice_survives_key_clear(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root, "VoiceTest")
            folder.mkdir()
            session = {"video_type": "speaking", "revision": 7, "segments": [],
                       "project_folder": str(folder), "project_name": "VoiceTest"}
            atomic.atomic_write_json(str(folder / "vrgdg_builder_session.json"), session)
            with patch.object(paths, "get_allowed_project_roots", return_value=[root]):
                pid = paths.get_project_id(str(folder))
                response = api.project_credentials(pid, "test-secret", 7)
                saved = json.loads((folder / "vrgdg_builder_session.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["elevenlabs_api_key_project"], "test-secret")
                assignment = {"enabled": True, "voice_id": "voice1", "name": "Test"}
                updated = mutations.upsert_reference_subject(pid, "alice", {"elevenlabs_voice": assignment}, response["revision"])
                api.project_credentials(pid, "", updated["revision"])
                saved = json.loads((folder / "vrgdg_builder_session.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["elevenlabs_api_key_project"], "")
                self.assertEqual(saved["flux_reference_builder"]["subjects"][0]["elevenlabs_voice"], assignment)
                self.assertEqual(saved["flux_reference_builder"]["subject"]["elevenlabs_voice"], assignment)

    def test_browser_settings_picker_and_normalization(self):
        result = subprocess.run(["node", str(ROOT / "tests/elevenlabs.cjs")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
