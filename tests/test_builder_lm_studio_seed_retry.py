import ast
import io
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BUILDER_BACKEND = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(encoding="utf-8")


def load_lm_studio_helpers():
    wanted = {
        "_llm_runner_from_payload",
        "_normalized_token_limit",
        "_runner_output_token_limit",
        "_lm_studio_context_limit",
        "_lm_studio_api_root",
        "_lm_studio_native_output_text",
        "_run_lm_studio_native_chat",
    }
    tree = ast.parse(BUILDER_BACKEND)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_LM_STUDIO_DEFAULT_BASE_URL"
            for target in node.targets
        ):
            nodes.append(node)
    namespace = {"json": json, "urllib": urllib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "lm_studio_helpers", "exec"), namespace)
    return namespace


LM_STUDIO_HELPERS = load_lm_studio_helpers()


class FakeLMStudioResponse:
    def __init__(self, payload_bytes):
        self._payload_bytes = payload_bytes

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._payload_bytes


def make_http_error(code, error_payload):
    body = json.dumps(error_payload).encode("utf-8")
    return urllib.error.HTTPError("http://127.0.0.1:1234/api/v1/chat", code, "Bad Request", {}, io.BytesIO(body))


SUCCESS_PAYLOAD = json.dumps({
    "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "Hello from LM Studio"}]},
    ]
}).encode("utf-8")


class LmStudioSeedRetryTests(unittest.TestCase):
    def _run(self, payload, urlopen_side_effect):
        run_chat = LM_STUDIO_HELPERS["_run_lm_studio_native_chat"]
        with mock.patch("urllib.request.urlopen", side_effect=urlopen_side_effect) as urlopen_mock:
            text = run_chat(payload, "hello", 0.7, 0.9, 500, 30, "")
        return text, urlopen_mock

    def test_retries_without_seed_when_lm_studio_rejects_unrecognized_seed_key(self):
        calls = []

        def urlopen_side_effect(request, timeout=None):
            body = json.loads(request.data.decode("utf-8"))
            calls.append(body)
            if "seed" in body:
                raise make_http_error(400, {
                    "error": {
                        "message": "Unrecognized key(s) in object: 'seed'",
                        "type": "invalid_request",
                        "code": "unrecognized_keys",
                    }
                })
            return FakeLMStudioResponse(SUCCESS_PAYLOAD)

        payload = {"lmstudio_base_url": "http://127.0.0.1:1234/v1", "lmstudio_model": "test-model", "seed": 42}
        text, urlopen_mock = self._run(payload, urlopen_side_effect)

        self.assertEqual("Hello from LM Studio", text)
        self.assertEqual(2, urlopen_mock.call_count)
        self.assertIn("seed", calls[0])
        self.assertNotIn("seed", calls[1])
        self.assertEqual("test-model", calls[1]["model"])

    def test_no_retry_needed_when_no_seed_is_configured(self):
        calls = []

        def urlopen_side_effect(request, timeout=None):
            calls.append(json.loads(request.data.decode("utf-8")))
            return FakeLMStudioResponse(SUCCESS_PAYLOAD)

        payload = {"lmstudio_base_url": "http://127.0.0.1:1234/v1", "lmstudio_model": "test-model"}
        text, urlopen_mock = self._run(payload, urlopen_side_effect)

        self.assertEqual("Hello from LM Studio", text)
        self.assertEqual(1, urlopen_mock.call_count)
        self.assertNotIn("seed", calls[0])

    def test_unrelated_bad_request_is_not_treated_as_seed_rejection(self):
        def urlopen_side_effect(request, timeout=None):
            raise make_http_error(400, {
                "error": {
                    "message": "Unrecognized key(s) in object: 'temperature'",
                    "type": "invalid_request",
                    "code": "unrecognized_keys",
                }
            })

        payload = {"lmstudio_base_url": "http://127.0.0.1:1234/v1", "lmstudio_model": "test-model", "seed": 42}
        with mock.patch("urllib.request.urlopen", side_effect=urlopen_side_effect) as urlopen_mock:
            run_chat = LM_STUDIO_HELPERS["_run_lm_studio_native_chat"]
            with self.assertRaises(RuntimeError) as ctx:
                run_chat(payload, "hello", 0.7, 0.9, 500, 30, "")
        self.assertEqual(1, urlopen_mock.call_count)
        self.assertIn("request failed (400)", str(ctx.exception))

    def test_missing_native_endpoint_still_raises_upgrade_message(self):
        def urlopen_side_effect(request, timeout=None):
            raise make_http_error(404, {"error": {"message": "not found"}})

        payload = {"lmstudio_base_url": "http://127.0.0.1:1234/v1", "lmstudio_model": "test-model", "seed": 42}
        with mock.patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            run_chat = LM_STUDIO_HELPERS["_run_lm_studio_native_chat"]
            with self.assertRaises(RuntimeError) as ctx:
                run_chat(payload, "hello", 0.7, 0.9, 500, 30, "")
        self.assertIn("native /api/v1/chat endpoint is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
