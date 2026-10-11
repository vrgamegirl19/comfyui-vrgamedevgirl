"""Project journals append safely, redact credentials, and preserve failed API responses."""

import asyncio
import ast
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import time
import traceback
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "vrgdg_log_test"
for name in (PACKAGE, PACKAGE + ".core", PACKAGE + ".builder"):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def load(relative: str):
    name = PACKAGE + "." + relative.replace("/", ".").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("core/atomic_write.py")
LOG = load("builder/error_log.py")
paths = types.ModuleType(PACKAGE + ".builder.paths")
paths._model_defaults_path = lambda: str(ROOT / "tests/nonexistent-log-defaults.json")
sys.modules[paths.__name__] = paths
ROUTES = load("builder/error_log_routes.py")


class ProjectErrorLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        (self.folder / "vrgdg_builder_session.json").write_text(json.dumps({
            "project_name": "Crown of Ash", "llm_api_key_project": "private-project-value",
        }), encoding="utf-8")

    def test_append_format_redaction_and_clear(self):
        LOG.project_error_log(str(self.folder))
        LOG.append_project_error(str(self.folder), {
            "scene": 44, "source": "ComfyUI", "stage": "Video Generation", "mode": "MiniMax H3 - Ref2V",
            "component": "Prompt Execution", "message": "Failed private-project-value",
            "details": "Authorization: Bearer private-unlisted-token",
            "stack": "Traceback\napi_key=private-typed-value\noriginal function failed",
        }, ["private-typed-value"])
        LOG.append_project_error(str(self.folder), {"message": "Second failure"})
        report = LOG.project_error_log(str(self.folder))["text"]
        self.assertIn("Project: Crown of Ash", report)
        self.assertRegex(report, r"\[\d{4}-\d\d-\d\d .*?[+-]\d\d:\d\d\] \[ERROR\]")
        self.assertIn("Scene: 44", report)
        self.assertIn("original function failed", report)
        self.assertLess(report.index("Failed"), report.index("Second failure"))
        for secret in ("private-project-value", "private-unlisted-token", "private-typed-value"):
            self.assertNotIn(secret, report)
        cleared = LOG.project_error_log(str(self.folder), True)["text"]
        self.assertIn("Project: Crown of Ash", cleared)
        self.assertNotIn("Second failure", cleared)

    def test_concurrent_appends_and_separate_projects(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda n: LOG.append_project_error(str(self.folder), {"message": f"Failure {n:03d}"}), range(40)))
        report = LOG.project_error_log(str(self.folder))["text"]
        self.assertEqual(report.count("[ERROR]"), 40)
        for n in range(40):
            self.assertEqual(report.count(f"Message: Failure {n:03d}\n"), 1)
        other = self.folder / "other"
        other.mkdir()
        (other / "vrgdg_builder_session.json").write_text("{}")
        self.assertNotIn("Failure", LOG.project_error_log(str(other))["text"])

    def test_reject_unsaved_project_and_escaping_log_symlink(self):
        with self.assertRaises(FileNotFoundError):
            LOG.project_error_log(str(self.folder / "unsaved"))
        with patch.object(LOG.os.path, "realpath", side_effect=[str(self.folder), str(self.folder.parent / "escape.txt")]):
            with self.assertRaisesRegex(ValueError, "inside"):
                LOG.project_error_log(str(self.folder))

    def test_failed_background_jobs_keep_their_status_and_original_traceback(self):
        tree = ast.parse((ROOT / "agent_api/jobs/manager.py").read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_job")
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        statuses = types.SimpleNamespace(RUNNING="running", FAILED="failed", CANCELLED="cancelled", SUCCEEDED="succeeded")
        namespace = {
            "asyncio": asyncio, "inspect": inspect, "time": time, "traceback": traceback,
            "JobStatus": statuses, "AgentApiError": type("AgentError", (Exception,), {}),
            "JobCancelledError": type("Cancelled", (Exception,), {}),
            "append_project_error": LOG.append_project_error, "logger": LOG._LOGGER,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "jobs", "exec"), namespace)

        async def fail(job, manager):
            raise RuntimeError("Background original private-project-value")

        job = types.SimpleNamespace(id="job1", type="video.render", params={"scene_id": "scene44"},
                                    is_gpu=False, cancel_requested=False, progress=types.SimpleNamespace())
        manager = types.SimpleNamespace(
            _cpu_semaphore=asyncio.Semaphore(1), _handlers={"video.render": fail},
            append_log=lambda *args: None, _persist_and_emit=lambda *args: None, _cleanup_active=lambda *args: None,
        )
        asyncio.run(namespace["_run_job"](manager, job, str(self.folder)))
        self.assertEqual(job.status, "failed")
        report = LOG.project_error_log(str(self.folder))["text"]
        self.assertIn("Background original [redacted]", report)
        self.assertIn("Stack Trace:", report)
        self.assertIn("Scene: scene44", report)


    def test_backend_errors_keep_response_and_do_not_recurse(self):
        async def exercise():
            app = web.Application(middlewares=[ROUTES.project_error_middleware])
            server = types.SimpleNamespace(app=app, routes=web.RouteTableDef())
            # Register endpoints without adding the middleware twice.
            app.middlewares.clear()
            ROUTES.register_error_log_routes(server)

            @server.routes.post("/vrgdg/test/fail")
            async def fail(request):
                return web.json_response({"ok": False, "error": "Failure private-project-value"}, status=500)

            @server.routes.post("/vrgdg/test/crash")
            async def crash(request):
                raise RuntimeError("Uncaught original failure")

            @server.routes.post("/vrgdg/music_builder/save_session")
            async def save(request):
                return web.json_response({"ok": True, "project_folder": str(self.folder)})

            app.add_routes(server.routes)
            async with TestClient(TestServer(app)) as client:
                self.assertFalse((self.folder / "logs/error_log.txt").exists())
                response = await client.post("/vrgdg/music_builder/save_session", json={"project_folder": str(self.folder)})
                self.assertEqual(response.status, 200)
                self.assertTrue((self.folder / "logs/error_log.txt").is_file())
                response = await client.post("/vrgdg/test/fail", json={"project_folder": str(self.folder), "scene_number": 12})
                self.assertEqual(response.status, 500)
                self.assertEqual((await response.json())["error"], "Failure private-project-value")
                self.assertEqual(response.headers["X-VRGDG-Error-Logged"], "1")
                await client.post("/vrgdg/test/crash", json={"project_folder": str(self.folder)})
                report = LOG.project_error_log(str(self.folder))["text"]
                self.assertIn("Scene: 12", report)
                self.assertIn("Uncaught original failure", report)
                self.assertIn("Stack Trace:", report)
                self.assertNotIn("private-project-value", report)
                with patch.object(ROUTES, "append_project_error", side_effect=OSError("disk full")):
                    response = await client.post("/vrgdg/test/fail", json={"project_folder": str(self.folder)})
                    self.assertEqual(response.status, 500)
                    self.assertNotIn("X-VRGDG-Error-Logged", response.headers)
                response = await client.post("/vrgdg/music_builder/error_log/append", json={"project_folder": "", "entry": {"message": "Bad log"}})
                self.assertEqual(response.status, 400)
                self.assertEqual(LOG.project_error_log(str(self.folder))["text"], report)
                response = await client.post("/vrgdg/music_builder/error_log/clear", json={"project_folder": str(self.folder)})
                self.assertNotIn("[ERROR]", (await response.json())["text"])
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
