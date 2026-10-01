"""The API must use the model LM Studio already has loaded and never ask it to load or switch models."""

import importlib
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
rt = importlib.import_module(f"{pkg_name}.agent_api.llm_runtime")
llm_jobs = importlib.import_module(f"{pkg_name}.agent_api.jobs.llm_jobs")
jobs_models = importlib.import_module(f"{pkg_name}.agent_api.jobs.models")


class FakeLmStudio:
    """Records every request so a test can prove nothing but the read-only model list was called."""

    def __init__(self, models):
        self.models = models
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests.append(("GET", self.path))
                if self.path == "/api/v0/models":
                    body = json.dumps({"data": outer.models}).encode()
                    self.send_response(200)
                else:
                    body = b"{}"
                    self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                outer.requests.append(("POST", self.path))
                self.send_response(500)
                self.end_headers()

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def model(model_id, state="loaded", kind="vlm", context=32768):
    return {"id": model_id, "state": state, "type": kind, "loaded_context_length": context if state == "loaded" else None}


class LoadedModelTests(unittest.TestCase):
    def serve(self, models):
        fake = FakeLmStudio(models)
        self.addCleanup(fake.close)
        return fake

    def payload(self, fake, **extra):
        return {"text_runner": "lm_studio", "lmstudio_base_url": fake.url, **extra}

    def test_the_loaded_model_is_used_even_when_the_project_names_another(self):
        fake = self.serve([model("gemma-4-e4b-it"), model("bonsai-27b", "not-loaded")])
        prepared = rt.prepare_llm_payload(self.payload(fake, lmstudio_model="bonsai-27b", lmstudio_context_limit=131072))
        self.assertEqual(prepared["lmstudio_model"], "gemma-4-e4b-it")
        self.assertEqual(prepared["lmstudio_context_limit"], 32768, "never ask for more context than is loaded")

    def test_the_saved_model_is_kept_when_it_is_the_loaded_one(self):
        fake = self.serve([model("a-model"), model("gemma-4-e4b-it")])
        prepared = rt.prepare_llm_payload(self.payload(fake, lmstudio_model="gemma-4-e4b-it"))
        self.assertEqual(prepared["lmstudio_model"], "gemma-4-e4b-it")

    def test_a_smaller_saved_context_is_respected(self):
        fake = self.serve([model("gemma-4-e4b-it", context=32768)])
        prepared = rt.prepare_llm_payload(self.payload(fake, lmstudio_context_limit=8000))
        self.assertEqual(prepared["lmstudio_context_limit"], 8000)

    def test_nothing_loaded_is_an_error_and_no_chat_request_is_made(self):
        fake = self.serve([model("gemma-4-e4b-it", "not-loaded"), model("embed", kind="embeddings")])
        with self.assertRaises(rt.LlmUnavailableError) as ctx:
            rt.prepare_llm_payload(self.payload(fake, lmstudio_model="gemma-4-e4b-it"))
        self.assertEqual(ctx.exception.code, "LLM_UNAVAILABLE")
        self.assertIn("never loads", str(ctx.exception))
        self.assertEqual([m for m, _ in fake.requests if m != "GET"], [])

    def test_embedding_models_are_never_chosen(self):
        fake = self.serve([model("text-embedding-x", kind="embeddings"), model("gemma-4-e4b-it")])
        self.assertEqual(rt.prepare_llm_payload(self.payload(fake))["lmstudio_model"], "gemma-4-e4b-it")

    def test_only_the_read_only_model_list_is_ever_requested(self):
        fake = self.serve([model("gemma-4-e4b-it")])
        rt.prepare_llm_payload(self.payload(fake))
        rt.describe_active_llm(self.payload(fake))
        self.assertTrue(fake.requests)
        self.assertEqual(set(fake.requests), {("GET", "/api/v0/models")})

    def test_an_unreachable_server_is_a_clear_error(self):
        with self.assertRaises(rt.LlmUnavailableError):
            rt.prepare_llm_payload({"text_runner": "lm_studio", "lmstudio_base_url": "http://127.0.0.1:1/v1"}, timeout=1)

    def test_other_runners_are_not_touched(self):
        payload = {"text_runner": "own_server", "own_server_url": "http://x/v1", "lmstudio_model": "bonsai-27b"}
        self.assertEqual(rt.prepare_llm_payload(payload), payload)

    def test_describe_reports_what_would_be_used(self):
        fake = self.serve([model("gemma-4-e4b-it")])
        info = rt.describe_active_llm(self.payload(fake, lmstudio_model="bonsai-27b"))
        self.assertEqual(info["model"], "gemma-4-e4b-it")
        self.assertEqual(info["saved_model_setting"], "bonsai-27b")
        self.assertFalse(info["uses_saved_setting"])
        self.assertEqual(info["context_length"], 32768)


class SessionSettingsTests(unittest.TestCase):
    def test_saved_runner_settings_become_generator_keys(self):
        session = {"text_gemma_runner": "lm_studio", "lm_studio_base_url": "http://h:1234/v1", "lm_studio_model": "m",
                   "lm_studio_context_limit": 131072, "gemma_output_token_limit": 4096}
        payload = rt.llm_payload_from_session(session, {"scene_id": "x", "lmstudio_model": "override"})
        self.assertEqual(payload["text_runner"], "lm_studio")
        self.assertEqual(payload["lmstudio_base_url"], "http://h:1234/v1")
        self.assertEqual(payload["lmstudio_model"], "override")
        self.assertEqual(payload["lmstudio_context_limit"], 131072)
        self.assertEqual(payload["gemma_output_token_limit"], 4096)

    def test_lm_studio_jobs_are_not_scheduled_as_gpu_jobs_but_local_runners_are(self):
        self.assertFalse(llm_jobs.is_llm_runner_gpu({"text_runner": "lm_studio"}))
        self.assertFalse(llm_jobs.is_llm_runner_gpu({"text_runner": "lmstudio"}))
        self.assertTrue(llm_jobs.is_llm_runner_gpu({"text_runner": "builtin"}))

    def test_a_job_payload_uses_the_loaded_model(self):
        fake = FakeLmStudio([model("gemma-4-e4b-it")])
        self.addCleanup(fake.close)
        job = jobs_models.Job(id="j", type="llm.scene_video_prompt", project_id=None, params={
            "text_runner": "lm_studio", "lmstudio_base_url": fake.url, "lmstudio_model": "bonsai-27b", "scene_id": "s1",
        })
        payload = llm_jobs._prepare_llm_payload(job, "C:/proj")
        self.assertEqual(payload["lmstudio_model"], "gemma-4-e4b-it")
        self.assertEqual(payload["scene_id"], "s1")
        self.assertEqual(payload["project_folder"], "C:/proj")


if __name__ == "__main__":
    unittest.main()
