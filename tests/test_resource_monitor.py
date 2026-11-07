import asyncio
import importlib
import json
import pathlib
import subprocess
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(
    instance=types.SimpleNamespace(routes=types.SimpleNamespace(
        get=lambda path: lambda fn: fn, post=lambda path: lambda fn: fn))))
# The stub stays in ``sys.modules`` for the whole file: everything this test needs has been imported through it by now, and a
# ``patch.dict`` that later removes the modules the package pulled in breaks things like ``psutil`` that look themselves up there.
sys.modules["server"] = server
monitor = importlib.import_module(f"{ROOT.name}.core.system_routes")
LLM_MODULE = f"{monitor.__package__}.llm.cache"


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
        llm = types.SimpleNamespace(_clear_vrgdg_llm_caches=Mock(), _GGUF_LOCK=threading.RLock())
        with patch.object(monitor.PromptServer.instance, "prompt_queue", queue, create=True), patch.dict(sys.modules, {LLM_MODULE: llm}):
            response = asyncio.run(monitor.clear_memory(None))
        self.assertEqual(response.status, 409)
        queue.set_flag.assert_not_called()
        llm._clear_vrgdg_llm_caches.assert_not_called()

    def test_clear_memory_uses_native_worker_and_optional_llm(self):
        for available in (False, True):
            queue = types.SimpleNamespace(mutex=threading.RLock(), get_tasks_remaining=lambda: 0, set_flag=Mock())
            llm = types.SimpleNamespace(_clear_vrgdg_llm_caches=Mock(return_value={"gguf_models_unloaded": 2}), _GGUF_LOCK=threading.RLock())
            with self.subTest(llm_available=available), patch.object(monitor.PromptServer.instance, "prompt_queue", queue, create=True), patch.dict(sys.modules, {LLM_MODULE: llm if available else None}):
                response = asyncio.run(monitor.clear_memory(None))
            self.assertEqual(response.status, 202)
            result = json.loads(response.text)
            self.assertEqual(result["status"], "requested")
            self.assertEqual(result["gguf_models_unloaded"], 2 if available else 0)
            self.assertEqual(queue.set_flag.call_args_list, [unittest.mock.call("unload_models", True), unittest.mock.call("free_memory", True)])
            if available:
                llm._clear_vrgdg_llm_caches.assert_called_once_with(clear_cuda_cache=False, clear_hf_pipeline_cache=False)
                self.assertTrue(llm._GGUF_LOCK.acquire(blocking=False))
                llm._GGUF_LOCK.release()

    def test_clear_memory_skips_while_llm_request_runs(self):
        queue = types.SimpleNamespace(mutex=threading.RLock(), get_tasks_remaining=lambda: 0, set_flag=Mock())
        llm = types.SimpleNamespace(_clear_vrgdg_llm_caches=Mock(), _GGUF_LOCK=threading.RLock())
        with llm._GGUF_LOCK, patch.object(monitor.PromptServer.instance, "prompt_queue", queue, create=True), patch.dict(sys.modules, {LLM_MODULE: llm}):
            response = asyncio.run(monitor.clear_memory(None))
        self.assertEqual(response.status, 409)
        self.assertEqual(json.loads(response.text)["error"], "Wait for the running LLM request to finish.")
        queue.set_flag.assert_not_called()
        llm._clear_vrgdg_llm_caches.assert_not_called()

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
            # Without ``nvidia-smi`` the reader asks ``amd-smi`` too, so the test says that tool has nothing either; that is what makes the whole
            # GPU side unavailable instead of only the NVIDIA half of it.
            with self.subTest(error=type(error).__name__), patch.object(monitor.subprocess, "run", side_effect=error), \
                    patch.object(monitor.amd_smi, "gpus_amd", return_value=[]):
                result = monitor._read_resources()
            self.assertGreater(result["ram"]["total"], 0)
            self.assertEqual(result["gpus"], [])
            self.assertEqual(result["gpu_status"], "unavailable")

    def test_nvidia_failure_falls_back_to_amd_sensors(self):
        gib = 2 ** 30
        records = [{"gpu": 0, "name": "AMD Radeon AI PRO R9700", "load": 5.0, "used": 6 * gib, "total": 30 * gib, "free": 24 * gib,
                    "temperature": 56.0, "fan": None, "clock": 84.0, "power": 52.0, "power_limit": 300.0, "uuid": "", "pci_bus": "0000:03:00.0"}]
        with patch.object(monitor.subprocess, "run", side_effect=FileNotFoundError()), \
                patch.object(monitor.amd_smi, "gpus_amd", return_value=records):
            result = monitor._read_resources()
        self.assertEqual(result["gpu_status"], "available")
        gpu = result["gpus"][0]
        self.assertEqual(gpu["index"], "0")
        self.assertEqual(gpu["name"], "AMD Radeon AI PRO R9700")
        self.assertEqual(gpu["used"], 6 * gib)
        self.assertIsNone(gpu["fan"])
        self.assertEqual(gpu["power_limit"], 300.0)
        json.dumps(result, allow_nan=False)

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
