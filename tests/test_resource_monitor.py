import asyncio
import importlib.util
import json
import pathlib
import subprocess
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch


server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(
    instance=types.SimpleNamespace(routes=types.SimpleNamespace(
        get=lambda path: lambda fn: fn, post=lambda path: lambda fn: fn))))
spec = importlib.util.spec_from_file_location(
    "resource_monitor", pathlib.Path(__file__).resolve().parents[1] / "VRGDG_ResourceMonitor.py")
monitor = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"server": server}):
    spec.loader.exec_module(monitor)


class ResourceMonitorTests(unittest.TestCase):
    def test_memory_logs_match_gpu_indices_and_keep_negative_changes(self):
        gib = 2 ** 30
        before = {"ram": {"used": 8*gib}, "process_ram": 4*gib, "gpus": [
            {"index": "0", "name": "First", "used": 6*gib},
            {"index": "1", "name": "Second", "used": 2*gib}]}
        after = {"ram": {"used": 7*gib}, "process_ram": 3*gib, "gpus": [
            {"index": "1", "name": "Second", "used": 3*gib},
            {"index": "0", "name": "First", "used": 2*gib}]}
        with self.assertLogs(monitor._log, level="INFO") as logs:
            monitor._log_memory(12, "After request", after, before)
        output = "\n".join(logs.output)
        self.assertIn("System RAM: 7.00 GiB used (decrease: +1.00 GiB)", output)
        self.assertIn("GPU 1 VRAM (Second): 3.00 GiB used (decrease: -1.00 GiB)", output)
        self.assertIn("GPU 0 VRAM (First): 2.00 GiB used (decrease: +4.00 GiB)", output)

    def test_missing_gpu_readings_are_logged_as_unavailable(self):
        snapshot = {"ram": {"used": 0}, "process_ram": 0, "gpus": []}
        with self.assertLogs(monitor._log, level="INFO") as logs:
            monitor._log_memory(1, "Before", snapshot)
        self.assertIn("GPU VRAM: unavailable", "\n".join(logs.output))

    def test_clear_memory_requires_idle_queue(self):
        queue = types.SimpleNamespace(mutex=threading.RLock(), get_tasks_remaining=lambda: 1, set_flag=Mock())
        llm = types.SimpleNamespace(_clear_vrgdg_llm_caches=Mock())
        with patch.object(monitor.PromptServer.instance, "prompt_queue", queue, create=True), patch.dict(sys.modules, {"_vrgdg_custom_LLM": llm}):
            response = asyncio.run(monitor.clear_memory(None))
        self.assertEqual(response.status, 409)
        queue.set_flag.assert_not_called()
        llm._clear_vrgdg_llm_caches.assert_not_called()

    def test_clear_memory_uses_native_worker_and_optional_llm(self):
        for available in (False, True):
            queue = types.SimpleNamespace(mutex=threading.RLock(), get_tasks_remaining=lambda: 0, set_flag=Mock())
            llm = types.SimpleNamespace(_clear_vrgdg_llm_caches=Mock(return_value={"gguf_models_unloaded": 2}))
            with self.subTest(llm_available=available), patch.object(monitor.PromptServer.instance, "prompt_queue", queue, create=True), patch.dict(sys.modules, {"_vrgdg_custom_LLM": llm if available else None}):
                response = asyncio.run(monitor.clear_memory(None))
            self.assertEqual(response.status, 202)
            result = json.loads(response.text)
            self.assertEqual(result["status"], "requested")
            self.assertEqual(result["gguf_models_unloaded"], 2 if available else 0)
            self.assertEqual(queue.set_flag.call_args_list, [unittest.mock.call("unload_models", True), unittest.mock.call("free_memory", True)])
            if available:
                llm._clear_vrgdg_llm_caches.assert_called_once_with(clear_cuda_cache=False, clear_hf_pipeline_cache=False)

    def test_multiple_gpus_and_unsupported_sensors(self):
        output = '0, "GPU, first", 0, 1024, 8192, 34, 0, 200, 20.5, 300\n1, Second GPU, 95, 2048, 4096, [N/A], [Not Supported], 1500, NaN, 250\n'
        with patch.object(monitor.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout=output)):
            result = monitor._read_resources()
        first, second = result["gpus"]
        self.assertEqual(first["name"], "GPU, first")
        self.assertEqual(first["used"], 1024 ** 3)
        self.assertEqual(first["fan"], 0)
        self.assertEqual(second["index"], "1")
        self.assertIsNone(second["temperature"])
        self.assertIsNone(second["power"])
        json.dumps(result, allow_nan=False)

    def test_gpu_failure_preserves_ram(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 2)):
            with self.subTest(error=type(error).__name__), patch.object(monitor.subprocess, "run", side_effect=error):
                result = monitor._read_resources()
            self.assertGreater(result["ram"]["total"], 0)
            self.assertEqual(result["gpus"], [])
            self.assertEqual(result["gpu_status"], "unavailable")

    def test_concurrent_requests_share_one_sample(self):
        async def exercise():
            monitor._lock = asyncio.Lock()
            monitor._cached = None
            with patch.object(monitor, "_read_resources", return_value={"ram": {}, "gpus": []}) as read:
                responses = await asyncio.gather(*(monitor.resource_monitor(None) for _ in range(5)))
                self.assertEqual(read.call_count, 1)
                self.assertTrue(all(response.status == 200 for response in responses))
                self.assertEqual(responses[0].headers["Cache-Control"], "no-store")
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
