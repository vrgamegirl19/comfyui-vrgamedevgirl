import ast
import ssl
import urllib
import urllib.error
import urllib.request
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BUILDER_UI = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
BUILDER_BACKEND = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(encoding="utf-8")


def load_own_server_helpers():
    wanted = {
        "_llm_runner_from_payload",
        "_own_server_v1_root",
        "_own_server_api_key",
        "_own_server_model_name",
        "_own_server_chat_messages",
        "_own_server_message_text",
        "_own_server_timeout",
        "_own_server_headers",
        "_own_server_ssl_context",
        "_own_server_request_json",
        "_normalized_token_limit",
        "_runner_output_token_limit",
    }
    tree = ast.parse(BUILDER_BACKEND)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {
                "_OWN_SERVER_DEFAULT_BASE_URL",
                "_EXTERNAL_LLM_RUNNERS",
                "_LM_STUDIO_DEFAULT_BASE_URL",
            }
            for target in node.targets
        ):
            nodes.append(node)
    import urllib.parse
    namespace = {
        "ssl": ssl,
        "urllib": urllib,
        "json": __import__("json"),
    }
    namespace["urllib.parse"] = urllib.parse
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "own_server_helpers", "exec"), namespace)
    return namespace


HELPERS = load_own_server_helpers()


class BuilderOwnServerLlmRunnerTests(unittest.TestCase):
    def test_payload_aliases_own_server_runner(self):
        parse = HELPERS["_llm_runner_from_payload"]
        self.assertEqual(parse({"text_runner": "own_server"}), "own_server")
        self.assertEqual(parse({"text_gemma_runner": "Use my own server"}), "builtin")
        self.assertEqual(parse({"text_runner": "openai_compatible"}), "own_server")
        self.assertEqual(parse({"text_runner": "custom_server"}), "own_server")
        self.assertEqual(parse({"text_runner": "llm_api"}), "llm_api")

    def test_external_runner_set_includes_own_server(self):
        self.assertEqual(
            HELPERS["_EXTERNAL_LLM_RUNNERS"],
            frozenset({"lm_studio", "llm_api", "own_server"}),
        )

    def test_url_accepts_local_and_cloudflare_and_full_chat_path(self):
        root = HELPERS["_own_server_v1_root"]
        self.assertEqual(root("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1")
        self.assertEqual(root("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1")
        self.assertEqual(
            root("https://random-words.trycloudflare.com"),
            "https://random-words.trycloudflare.com/v1",
        )
        self.assertEqual(
            root("https://random-words.trycloudflare.com/v1/chat/completions"),
            "https://random-words.trycloudflare.com/v1",
        )
        with self.assertRaises(ValueError):
            root("ftp://example.com")
        with self.assertRaises(ValueError):
            root("")

    def test_timeout_defaults_to_six_minutes_and_clamps(self):
        timeout = HELPERS["_own_server_timeout"]
        self.assertEqual(timeout({}), 360.0)
        self.assertEqual(timeout({"own_server_timeout": 360}), 360.0)
        self.assertEqual(timeout({"own_server_timeout": 5}), 15.0)
        self.assertEqual(timeout({"own_server_timeout": 900}), 600.0)

    def test_api_key_is_optional(self):
        self.assertEqual(HELPERS["_own_server_api_key"]({}), "")
        self.assertEqual(HELPERS["_own_server_api_key"]({"own_server_api_key": "  sk-test  "}), "sk-test")

    def test_tls_certificate_failure_does_not_retry_without_verification(self):
        request_json = HELPERS["_own_server_request_json"]
        certificate_error = ssl.SSLCertVerificationError(1, "certificate verify failed")
        for failure in (certificate_error, urllib.error.URLError(certificate_error)):
            with self.subTest(failure=type(failure).__name__):
                with mock.patch("urllib.request.urlopen", side_effect=failure) as urlopen:
                    with self.assertRaisesRegex(RuntimeError, "TLS certificate validation failed"):
                        request_json(
                            "https://example.invalid/v1/models",
                            {"own_server_api_key": "secret"},
                            method="GET",
                        )
                self.assertEqual(urlopen.call_count, 1)

    def test_text_messages_are_openai_chat_completions(self):
        messages = HELPERS["_own_server_chat_messages"]("Hello there")
        self.assertEqual(messages, [{"role": "user", "content": "Hello there"}])

    def test_response_parser_reads_openai_choice_content(self):
        parse = HELPERS["_own_server_message_text"]
        self.assertEqual(
            parse({"choices": [{"message": {"content": "OK"}}]}),
            "OK",
        )
        self.assertEqual(
            parse({"choices": [{"message": {"content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]}}]}),
            "Hello world",
        )

    def test_ui_exposes_own_server_runner_and_required_fields(self):
        self.assertIn('makeSelect(["builtin", "qwen_local", "lm_studio", "llm_api", "own_server"]', BUILDER_UI)
        self.assertIn('runner.options[4].textContent = "Custom Server"', BUILDER_UI)
        self.assertIn('makeField("Server URL", ownUrl)', BUILDER_UI)
        self.assertIn('makeField("API key (optional)", ownApiKey)', BUILDER_UI)
        self.assertIn('makeField("Test response", ownTestOutput)', BUILDER_UI)
        self.assertIn('makeField("Request timeout (minutes)", ownTimeoutMinutes)', BUILDER_UI)
        self.assertIn("own_server_timeout", BUILDER_UI)
        self.assertIn("ownServerPayloadTimeoutMs", BUILDER_UI)
        self.assertIn("/vrgdg/music_builder/test_own_server", BUILDER_UI)
        self.assertIn("OpenAI Chat Completions standard", BUILDER_UI)
        self.assertIn("image_url", BUILDER_UI)
        self.assertIn("Cloudflare", BUILDER_UI)

    def test_loading_project_clears_unsaved_key_instead_of_reusing_it(self):
        self.assertIn(
            "state.ownServerApiKey = state.ownServerApiKeyProject;",
            BUILDER_UI,
        )

    def test_backend_registers_test_and_models_routes(self):
        self.assertIn('/vrgdg/music_builder/test_own_server', BUILDER_BACKEND)
        self.assertIn('/vrgdg/music_builder/own_server_models', BUILDER_BACKEND)
        self.assertIn("/chat/completions", BUILDER_BACKEND)
        self.assertIn('_try_run_remote_vision', BUILDER_BACKEND)


if __name__ == "__main__":
    unittest.main()
