"""Unit tests for Agent API Phase A4: Jobs, Events, Concurrency, and Harness (Section 5)."""

import asyncio
import json
import os
import shutil
import tempfile
import time
import unittest

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
orch_mod = importlib.import_module(f"{pkg_name}.agent_api.orchestrator")

ComfyExecutionError = errors.ComfyExecutionError
JobCancelledError = errors.JobCancelledError
JobNotFoundError = errors.JobNotFoundError
ValidationError = errors.ValidationError

EventBroadcaster = jobs_mod.EventBroadcaster
Job = jobs_mod.Job
JobLogEntry = jobs_mod.JobLogEntry
JobManager = jobs_mod.JobManager
JobProgress = jobs_mod.JobProgress
JobStatus = jobs_mod.JobStatus
generate_job_id = jobs_mod.generate_job_id
load_project_jobs = jobs_mod.load_project_jobs
recover_interrupted_jobs = jobs_mod.recover_interrupted_jobs
save_project_jobs = jobs_mod.save_project_jobs
save_single_project_job = jobs_mod.save_single_project_job

ComfyClient = orch_mod.ComfyClient
FakeComfyClient = orch_mod.FakeComfyClient
extract_images_from_history = orch_mod.extract_images_from_history
extract_prompt_error_from_history = orch_mod.extract_prompt_error_from_history
extract_text_from_history = orch_mod.extract_text_from_history
extract_videos_from_history = orch_mod.extract_videos_from_history
prompt_history_finished = orch_mod.prompt_history_finished
set_comfy_client = orch_mod.set_comfy_client


class TestAgentApiJobs(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_jobs_test_")
        self.project_dir = os.path.join(self.test_dir, "TestProject")
        os.makedirs(self.project_dir, exist_ok=True)
        # Setup minimal session.json
        with open(os.path.join(self.project_dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump({"project_name": "TestProject", "segments": []}, f)

        # Point VRGDG_PROJECT_ROOTS to test_dir
        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir

        self.fake_comfy = FakeComfyClient()
        set_comfy_client(self.fake_comfy)

    def tearDown(self):
        set_comfy_client(None)
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_job_model_and_serialization(self):
        job_id = generate_job_id()
        self.assertTrue(job_id.startswith("job_"))

        prog = JobProgress(percent=45.2, stage="rendering", stage_index=2, stage_count=5, message="Scene 2/5")
        job = Job(
            id=job_id,
            type="pipeline.build_full_video",
            project_id="TestProject",
            status=JobStatus.RUNNING,
            progress=prog,
            params={"build_mode": "zimage_i2v"},
            warnings=["Low disk space"],
        )

        d = job.to_dict()
        self.assertEqual(d["id"], job_id)
        self.assertEqual(d["type"], "pipeline.build_full_video")
        self.assertEqual(d["status"], JobStatus.RUNNING)
        self.assertEqual(d["progress"]["percent"], 45.2)
        self.assertEqual(d["progress"]["stage"], "rendering")
        self.assertEqual(d["warnings"], ["Low disk space"])

        restored = Job.from_dict(d)
        self.assertEqual(restored.id, job.id)
        self.assertEqual(restored.type, job.type)
        self.assertEqual(restored.progress.percent, 45.2)
        self.assertEqual(restored.progress.message, "Scene 2/5")
        self.assertFalse(restored.is_terminal())

    def test_02_jobs_persistence_and_recovery(self):
        jobs_file = os.path.join(self.project_dir, "jobs", "jobs.json")
        self.assertFalse(os.path.exists(jobs_file))

        job1 = Job(id="job_1", type="render", project_id="TestProject", status=JobStatus.RUNNING)
        job2 = Job(id="job_2", type="llm", project_id="TestProject", status=JobStatus.SUCCEEDED)

        save_project_jobs(self.project_dir, {"job_1": job1, "job_2": job2})
        self.assertTrue(os.path.exists(jobs_file))

        loaded = load_project_jobs(self.project_dir)
        self.assertEqual(len(loaded), 2)
        self.assertIn("job_1", loaded)
        self.assertIn("job_2", loaded)
        self.assertEqual(loaded["job_1"].status, JobStatus.RUNNING)

        # Simulate crash recovery: job_1 was running and should become interrupted
        recovered = recover_interrupted_jobs(self.project_dir)
        self.assertEqual(recovered, ["job_1"])

        loaded_after = load_project_jobs(self.project_dir)
        self.assertEqual(loaded_after["job_1"].status, JobStatus.INTERRUPTED)
        self.assertTrue(any("Server restarted" in w for w in loaded_after["job_1"].warnings))
        self.assertIsNotNone(loaded_after["job_1"].finished_at)
        self.assertEqual(loaded_after["job_2"].status, JobStatus.SUCCEEDED)

    def test_03_event_broadcaster_and_filtering(self):
        broadcaster = EventBroadcaster()
        q_all = broadcaster.subscribe(project_id=None)
        q_p1 = broadcaster.subscribe(project_id="ProjectA")
        q_p2 = broadcaster.subscribe(project_id="ProjectB")

        broadcaster.emit("job.updated", {"id": "j1"}, project_id="ProjectA")
        broadcaster.emit("job.updated", {"id": "j2"}, project_id="ProjectB")
        broadcaster.emit("queue.updated", {"gpu_running": 1}, project_id=None)

        # q_all should receive all 3
        self.assertEqual(q_all.qsize(), 3)
        # q_p1 should receive j1 and queue.updated
        self.assertEqual(q_p1.qsize(), 2)
        # q_p2 should receive j2 and queue.updated
        self.assertEqual(q_p2.qsize(), 2)

        evt = q_p1.get_nowait()
        self.assertEqual(evt["event"], "job.updated")
        self.assertEqual(evt["data"]["id"], "j1")

        broadcaster.unsubscribe(q_all)
        broadcaster.unsubscribe(q_p1)
        broadcaster.unsubscribe(q_p2)
        self.assertEqual(broadcaster.subscriber_count, 0)

    def test_04_job_manager_execution_lifecycle(self):
        manager = JobManager()

        executed = []

        def sample_sync_handler(job: Job, mgr: JobManager):
            executed.append(job.id)
            mgr.update_progress(job.id, 50.0, "halfway", message="Almost done")
            mgr.append_log(job.id, "info", "Working on sync task")
            return {"generated_scenes": 3}

        manager.register_handler("test.sync", sample_sync_handler)

        async def _test():
            job = manager.submit_job("test.sync", project_id="TestProject", params={"count": 3}, is_gpu=False)
            self.assertEqual(job.status, JobStatus.QUEUED)

            # Wait for execution task to complete
            for _ in range(50):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job.status, JobStatus.SUCCEEDED)
            self.assertEqual(job.progress.percent, 100.0)
            self.assertEqual(job.result, {"generated_scenes": 3})
            self.assertIn(job.id, executed)

            # Check logs
            logs = manager.get_logs(job.id, since=0)
            self.assertGreaterEqual(len(logs), 2)
            messages = [l["message"] for l in logs]
            self.assertTrue(any("Working on sync task" in m for m in messages))

            # Check persistence in jobs.json
            disk_jobs = load_project_jobs(self.project_dir)
            self.assertIn(job.id, disk_jobs)
            self.assertEqual(disk_jobs[job.id].status, JobStatus.SUCCEEDED)

        asyncio.run(_test())

    def test_05_job_manager_gpu_concurrency(self):
        manager = JobManager(max_gpu_concurrency=1)

        active_jobs = []
        max_concurrent = 0

        async def long_gpu_handler(job: Job, mgr: JobManager):
            nonlocal max_concurrent
            active_jobs.append(job.id)
            if len(active_jobs) > max_concurrent:
                max_concurrent = len(active_jobs)
            await asyncio.sleep(0.1)
            active_jobs.remove(job.id)
            return {"done": True}

        manager.register_handler("test.gpu", long_gpu_handler)

        async def _test():
            job1 = manager.submit_job("test.gpu", project_id="TestProject", is_gpu=True)
            job2 = manager.submit_job("test.gpu", project_id="TestProject", is_gpu=True)

            summary = manager.get_queue_summary()
            self.assertGreaterEqual(summary["gpu_queued"] + summary["gpu_running"], 1)

            for _ in range(100):
                if job1.is_terminal() and job2.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job1.status, JobStatus.SUCCEEDED)
            self.assertEqual(job2.status, JobStatus.SUCCEEDED)
            self.assertEqual(max_concurrent, 1)  # Strict FIFO 1-job GPU serialization

        asyncio.run(_test())

    def test_06_job_cancellation_queued_and_running(self):
        manager = JobManager()

        # 1. Cancel a queued job before it starts
        job_q = Job(id="job_queued", type="test.cancel", project_id="TestProject", status=JobStatus.QUEUED)
        manager._jobs["job_queued"] = job_q
        manager.cancel_job("job_queued")
        self.assertEqual(job_q.status, JobStatus.CANCELLED)
        self.assertTrue(job_q.cancel_requested)

        # 2. Cancel a running job with active comfy prompt
        async def cancelable_handler(job: Job, mgr: JobManager):
            mgr.set_current_comfy_prompt(job.id, "fake_prompt_99")
            while not job.cancel_requested:
                await asyncio.sleep(0.01)
            raise JobCancelledError(job.id)

        manager.register_handler("test.cancel", cancelable_handler)

        async def _test():
            job_run = manager.submit_job("test.cancel", project_id="TestProject", is_gpu=False)

            # Wait until it starts
            for _ in range(50):
                if job_run.status == JobStatus.RUNNING and job_run.comfy.get("current_prompt_id"):
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job_run.status, JobStatus.RUNNING)
            self.assertEqual(job_run.comfy["current_prompt_id"], "fake_prompt_99")

            manager.cancel_job(job_run.id)

            for _ in range(50):
                if job_run.is_terminal():
                    break
                await asyncio.sleep(0.02)

            self.assertEqual(job_run.status, JobStatus.CANCELLED)
            self.assertGreater(self.fake_comfy.interrupted_count, 0)
            self.assertIn("fake_prompt_99", self.fake_comfy.deleted_prompt_ids)

        asyncio.run(_test())

    def test_07_job_retry_lifecycle(self):
        manager = JobManager()
        failed_job = Job(
            id="job_fail_1",
            type="render.scene",
            project_id="TestProject",
            status=JobStatus.FAILED,
            params={"scene_id": "seg_001", "seed": 42},
            attempt=1,
            max_attempts=3,
        )
        manager._jobs[failed_job.id] = failed_job

        # Test retry without resume
        new_job = manager.retry_job("job_fail_1", resume=False)
        self.assertEqual(new_job.type, "render.scene")
        self.assertEqual(new_job.project_id, "TestProject")
        self.assertEqual(new_job.attempt, 2)
        self.assertEqual(new_job.status, JobStatus.QUEUED)
        self.assertEqual(new_job.params["seed"], 42)

        # Test retry with resume
        new_resumed_job = manager.retry_job("job_fail_1", resume=True)
        self.assertTrue(new_resumed_job.params.get("resume"))
        self.assertEqual(new_resumed_job.params.get("run_mode"), "resume_missing")

        # Test retry on non-terminal job raises error
        running_job = Job(id="job_running_1", type="render", project_id="TestProject", status=JobStatus.RUNNING)
        manager._jobs[running_job.id] = running_job
        with self.assertRaises(ValidationError):
            manager.retry_job("job_running_1")

    def test_08_fake_comfy_client_harness(self):
        client = FakeComfyClient()

        # Submit prompt
        res = client.queue_prompt({"1": {"class_type": "FakeNode"}})
        prompt_id = res["prompt_id"]
        self.assertTrue(prompt_id.startswith("fake_prompt_"))

        # Check default outputs
        history = client.get_history(prompt_id)
        self.assertTrue(prompt_history_finished(history, prompt_id))
        images = extract_images_from_history(history, prompt_id)
        videos = extract_videos_from_history(history, prompt_id)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(videos), 1)

        # Test error simulation
        client.set_mock_output("err_prompt", error="Out of memory on node 4")
        err_history = client.get_history("err_prompt")
        err_msg = extract_prompt_error_from_history(err_history, "err_prompt")
        self.assertIn("Out of memory", err_msg)

        async def _test_wait():
            # Wait for successful prompt
            out = await client.wait_for_prompt(prompt_id, timeout_seconds=2.0)
            self.assertIn(prompt_id, out)

            # Wait for error prompt raises ComfyExecutionError
            with self.assertRaises(ComfyExecutionError):
                await client.wait_for_prompt("err_prompt", timeout_seconds=2.0)

            # Wait with check_cancel returning True raises JobCancelledError
            with self.assertRaises(JobCancelledError):
                await client.wait_for_prompt(prompt_id, timeout_seconds=2.0, check_cancel=lambda: True)

        asyncio.run(_test_wait())


if __name__ == "__main__":
    unittest.main()
