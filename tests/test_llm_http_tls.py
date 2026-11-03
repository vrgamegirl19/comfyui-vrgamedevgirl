"""Verify certificate-store fallback without making any provider requests."""

import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import ssl
import unittest
import urllib.error
import urllib.request
import uuid
from typing import Optional
from unittest.mock import patch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("llm_http_tls", SOURCE_ROOT / "llm/http_tls.py")
TLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TLS)


def expired_certificate():
    return urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate has expired"))


class LlmTlsTests(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, {}, clear=True)
        self.environ.start()
        self.addCleanup(self.environ.stop)
        self.request = urllib.request.Request("https://api.example.test/v1/chat?key=never-log-this",
                                              data=b'{"prompt":"hello"}', headers={"Authorization": "Bearer secret"})

    def test_default_context_verifies_hostname_and_certificates(self):
        response = object()
        with patch.object(TLS.urllib.request, "urlopen", return_value=response) as open_url:
            self.assertIs(TLS.urlopen_verified(self.request, 90), response)
        context = open_url.call_args.kwargs["context"]
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(open_url.call_count, 1)

    def test_expired_system_chain_uses_a_separate_verified_store(self):
        response = object()
        fallback = ssl.create_default_context()
        with patch.object(TLS, "_bundled_ca_context", return_value=fallback), \
                patch.object(TLS.urllib.request, "urlopen", side_effect=[expired_certificate(), response]) as open_url:
            self.assertIs(TLS.urlopen_verified(self.request, 90), response)
        self.assertEqual(open_url.call_count, 2)
        for call in open_url.call_args_list:
            self.assertIs(call.args[0], self.request)
            self.assertTrue(call.kwargs["context"].check_hostname)
            self.assertEqual(call.kwargs["context"].verify_mode, ssl.CERT_REQUIRED)
        self.assertIs(open_url.call_args_list[1].kwargs["context"], fallback)
        self.assertEqual(self.request.data, b'{"prompt":"hello"}')

    def test_http_errors_timeouts_and_network_failures_are_not_retried(self):
        errors = [urllib.error.HTTPError(self.request.full_url, 429, "limit", {}, None),
                  urllib.error.HTTPError(self.request.full_url, 401, "unauthorized", {}, None),
                  urllib.error.URLError("connection refused"), TimeoutError("timeout"),
                  ssl.SSLError("connection closed")]
        for error in errors:
            with self.subTest(error=error), patch.object(TLS, "_bundled_ca_context") as fallback, \
                    patch.object(TLS.urllib.request, "urlopen", side_effect=error) as open_url:
                with self.assertRaises(type(error)):
                    TLS.urlopen_verified(self.request, 90)
                self.assertEqual(open_url.call_count, 1)
                fallback.assert_not_called()

    def test_both_stores_failing_reports_host_and_remediation_without_credentials(self):
        with patch.object(TLS, "_bundled_ca_context", return_value=ssl.create_default_context()), \
                patch.object(TLS.urllib.request, "urlopen", side_effect=expired_certificate()) as open_url:
            with self.assertRaises(urllib.error.URLError) as caught:
                TLS.urlopen_verified(self.request, 90)
        message = str(caught.exception)
        self.assertEqual(open_url.call_count, 2)
        for text in ("api.example.test", "certificate has expired", "certifi", "date/time", "renewed"):
            self.assertIn(text, message)
        for text in ("never-log-this", "Bearer secret", "hello"):
            self.assertNotIn(text, message)

    def test_explicit_ca_configuration_is_not_overridden(self):
        for key in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            with self.subTest(key=key), patch.dict(os.environ, {key: "custom-ca"}), \
                    patch.object(TLS.ssl, "create_default_context", return_value=object()), \
                    patch.object(TLS, "_bundled_ca_context") as fallback, \
                    patch.object(TLS.urllib.request, "urlopen", side_effect=expired_certificate()) as open_url:
                with self.assertRaisesRegex(urllib.error.URLError, "configured SSL_CERT_FILE"):
                    TLS.urlopen_verified(self.request, 90)
                self.assertEqual(open_url.call_count, 1)
                fallback.assert_not_called()

    def test_missing_bundle_explains_how_to_repair_it(self):
        with patch.object(TLS, "_bundled_ca_context", side_effect=ImportError("No module named certifi")), \
                patch.object(TLS.urllib.request, "urlopen", side_effect=expired_certificate()) as open_url:
            with self.assertRaisesRegex(urllib.error.URLError, "Update certifi"):
                TLS.urlopen_verified(self.request, 90)
        self.assertEqual(open_url.call_count, 1)

    def test_http_local_servers_keep_the_existing_transport(self):
        with patch.object(TLS.urllib.request, "urlopen", return_value="local") as open_url, \
                patch.object(TLS.ssl, "create_default_context") as context:
            self.assertEqual(TLS.urlopen_verified("http://127.0.0.1:1234/v1/models", 30), "local")
        open_url.assert_called_once_with("http://127.0.0.1:1234/v1/models", timeout=30)
        context.assert_not_called()

    def test_hosted_api_json_and_multipart_calls_use_the_verified_transport(self):
        source = (SOURCE_ROOT / "llm/api.py").read_text(encoding="utf-8")
        cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef))
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("_post_json", "_get_json", "_post_multipart")]
        namespace = {"json": json, "uuid": uuid, "urllib": urllib, "Optional": Optional,
                     "urlopen_verified": TLS.urlopen_verified}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "api_transport", "exec"), namespace)
        for method, arguments in (
            ("_post_json", ({"Authorization": "Bearer secret"}, {"prompt": "hello"})),
            ("_post_multipart", ({"Authorization": "Bearer secret"}, {"prompt": "hello"}, [])),
            ("_get_json", ({"Authorization": "Bearer secret"},)),
        ):
            with self.subTest(method=method), \
                    patch.object(TLS, "_bundled_ca_context", return_value=ssl.create_default_context()), \
                    patch.object(TLS.urllib.request, "urlopen", side_effect=[
                        expired_certificate(), io.BytesIO(b'{"result":"ok"}')]) as open_url:
                result = namespace[method](None, "https://api.example.test/v1/test", *arguments)
                self.assertEqual(result, {"result": "ok"})
                self.assertEqual(open_url.call_count, 2)
                self.assertEqual(open_url.call_args.args[0].get_header("Authorization"), "Bearer secret")


if __name__ == "__main__":
    unittest.main()
