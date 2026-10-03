"""Background Job Manager with GPU queue limits and persistence (Section 5)."""

import asyncio
from collections import deque
import inspect
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from ..errors import (
    AgentApiError,
    JobCancelledError,
    JobNotFoundError,
    ProjectNotFoundError,
    ValidationError,
)
from ..paths import get_allowed_project_roots, resolve_project_folder
from .events import get_event_broadcaster
from .models import (
    Job,
    JobLogEntry,
    JobProgress,
    JobStatus,
    generate_job_id,
)
from .persistence import (
    load_project_jobs,
    recover_interrupted_jobs,
    save_project_jobs,
    save_single_project_job,
)

logger = logging.getLogger("vrgdg.agent_api.jobs")


class JobManager:
    """Orchestrates job execution, FIFO concurrency, progress events, and persistence."""

    def __init__(self, max_gpu_concurrency: int = 1, max_cpu_concurrency: int = 4):
        self._jobs: Dict[str, Job] = {}
        self._logs: Dict[str, deque] = {}  # job_id -> deque of JobLogEntry
        self._log_counters: Dict[str, int] = {}
        self._handlers: Dict[str, Callable] = {}
        self._max_logs_per_job = 1000

        # Concurrency semaphores: 1 global GPU job (Section 5), pool for CPU jobs
        self._gpu_semaphore = asyncio.Semaphore(max_gpu_concurrency)
        self._cpu_semaphore = asyncio.Semaphore(max_cpu_concurrency)
        self._broadcaster = get_event_broadcaster()

        # Track active executing job id per worker type
        self._active_gpu_job_id: Optional[str] = None
        self._queued_gpu_jobs: List[str] = []

    def register_handler(self, job_type: str, handler: Callable) -> None:
        """Register an execution handler for a specific job type."""
        self._handlers[job_type] = handler

    def submit_job(
        self,
        job_type: str,
        project_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        is_gpu: bool = True,
        max_attempts: int = 1,
    ) -> Job:
        """Create and queue a new job (Section 5.1)."""
        project_folder: Optional[str] = None
        if project_id:
            project_folder = resolve_project_folder(project_id)

        job_id = generate_job_id()
        job = Job(
            id=job_id,
            type=job_type,
            project_id=project_id,
            status=JobStatus.QUEUED,
            created_at=time.time(),
            params=dict(params or {}),
            is_gpu=is_gpu,
            max_attempts=max_attempts,
            progress=JobProgress(stage="queued", message="Waiting in queue..."),
        )

        self._jobs[job_id] = job
        self._logs[job_id] = deque(maxlen=self._max_logs_per_job)
        self._log_counters[job_id] = 0

        self.append_log(job_id, "info", f"Job '{job_id}' ({job_type}) submitted and queued.")

        if project_folder:
            save_single_project_job(project_folder, job)

        if is_gpu:
            self._queued_gpu_jobs.append(job_id)

        self._broadcaster.emit("job.updated", job.to_dict(), project_id=project_id)
        self._broadcaster.emit("queue.updated", self.get_queue_summary(), project_id=project_id)

        # Dispatch async worker
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._run_job(job, project_folder))
        except RuntimeError:
            # No running event loop yet (e.g. synchronous unit test context)
            pass

        return job

    def get_job(self, job_id: str) -> Job:
        """Fetch a job by ID, looking in-memory and then on disk."""
        if job_id in self._jobs:
            return self._jobs[job_id]

        # Scan allowed roots if not in active memory
        for root in get_allowed_project_roots():
            if not os.path.isdir(root):
                continue
            for item in os.listdir(root):
                pfolder = os.path.join(root, item)
                if os.path.isdir(pfolder):
                    disk_jobs = load_project_jobs(pfolder)
                    if job_id in disk_jobs:
                        self._jobs[job_id] = disk_jobs[job_id]
                        return disk_jobs[job_id]

        raise JobNotFoundError(job_id)

    def list_jobs(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
    ) -> List[Job]:
        """List jobs matching optional filters (Section 5.2)."""
        candidates: Dict[str, Job] = dict(self._jobs)

        if project_id:
            try:
                pfolder = resolve_project_folder(project_id)
                disk_jobs = load_project_jobs(pfolder)
                candidates.update(disk_jobs)
            except Exception:
                pass

        results: List[Job] = []
        for job in candidates.values():
            if project_id and job.project_id != project_id:
                continue
            if status and job.status != status:
                continue
            if job_type and job.type != job_type:
                continue
            results.append(job)

        results.sort(key=lambda j: j.created_at, reverse=True)
        return results

    def append_log(
        self,
        job_id: str,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> JobLogEntry:
        """Append a log entry with monotonic sequence numbering."""
        if job_id not in self._logs:
            self._logs[job_id] = deque(maxlen=self._max_logs_per_job)
            self._log_counters[job_id] = 0

        self._log_counters[job_id] += 1
        seq = self._log_counters[job_id]
        entry = JobLogEntry(
            seq=seq,
            timestamp=time.time(),
            level=level,
            message=message,
            details=details,
        )
        self._logs[job_id].append(entry)
        return entry

    def get_logs(self, job_id: str, since: int = 0) -> List[Dict[str, Any]]:
        """Get log entries with sequence > since."""
        self.get_job(job_id)  # raises JobNotFoundError if invalid
        log_queue = self._logs.get(job_id, deque())
        return [entry.to_dict() for entry in log_queue if entry.seq > since]

    def update_progress(
        self,
        job_id: str,
        percent: float,
        stage: str,
        stage_index: int = 0,
        stage_count: int = 1,
        message: str = "",
        scene_id: Optional[str] = None,
        eta_seconds: Optional[float] = None,
        project_folder: Optional[str] = None,
        scene_index: Optional[int] = None,
        scene_count: Optional[int] = None,
    ) -> None:
        """Update job progress and emit SSE event.

        ``scene_index`` / ``scene_count`` describe a batch ("scene 3 of 8"). Per-scene render steps
        do not pass them, so the last values a batch set are kept instead of being overwritten.
        """
        job = self.get_job(job_id)
        previous = job.progress if isinstance(job.progress, JobProgress) else JobProgress()
        job.progress = JobProgress(
            scene_index=previous.scene_index if scene_index is None else scene_index,
            scene_count=previous.scene_count if scene_count is None else scene_count,
            percent=percent,
            stage=stage,
            stage_index=stage_index,
            stage_count=stage_count,
            message=message,
            scene_id=scene_id,
            eta_seconds=eta_seconds,
        )

        pfolder = project_folder
        if not pfolder and job.project_id:
            try:
                pfolder = resolve_project_folder(job.project_id)
            except Exception:
                pass

        if pfolder:
            save_single_project_job(pfolder, job)

        self._broadcaster.emit("job.updated", job.to_dict(), project_id=job.project_id)

    def set_current_comfy_prompt(
        self,
        job_id: str,
        prompt_id: str,
        project_folder: Optional[str] = None,
    ) -> None:
        """Associate an active ComfyUI prompt_id with this job (Section 5.1)."""
        job = self.get_job(job_id)
        if prompt_id not in job.comfy.get("prompt_ids", []):
            job.comfy.setdefault("prompt_ids", []).append(prompt_id)
        job.comfy["current_prompt_id"] = prompt_id

        pfolder = project_folder
        if not pfolder and job.project_id:
            try:
                pfolder = resolve_project_folder(job.project_id)
            except Exception:
                pass

        if pfolder:
            save_single_project_job(pfolder, job)

    def cancel_job(self, job_id: str) -> Job:
        """Request job cancellation and interrupt active ComfyUI executions (Section 5.4)."""
        job = self.get_job(job_id)
        job.cancel_requested = True
        self.append_log(job_id, "warning", f"Cancellation requested for job '{job_id}'.")

        # If queued and not started yet, transition immediately to cancelled
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.finished_at = time.time()
            if job_id in self._queued_gpu_jobs:
                self._queued_gpu_jobs.remove(job_id)

        # Clear active ComfyUI execution and job's pending prompts if available
        curr_prompt = job.comfy.get("current_prompt_id")
        prompt_ids = job.comfy.get("prompt_ids", [])

        try:
            from ..orchestrator.comfy_client import get_comfy_client
            client = get_comfy_client()
            if curr_prompt:
                client.interrupt()
            if prompt_ids:
                client.delete_queue_items(prompt_ids)
        except Exception as exc:
            logger.debug(f"Could not signal ComfyUI client during cancel: {exc}")

        pfolder = None
        if job.project_id:
            try:
                pfolder = resolve_project_folder(job.project_id)
                save_single_project_job(pfolder, job)
            except Exception:
                pass

        self._broadcaster.emit("job.updated", job.to_dict(), project_id=job.project_id)
        self._broadcaster.emit("queue.updated", self.get_queue_summary(), project_id=job.project_id)
        return job

    def retry_job(self, job_id: str, resume: bool = False) -> Job:
        """Create and queue a new job retrying a failed/interrupted job (Section 5.2, 5.3)."""
        old_job = self.get_job(job_id)
        if not old_job.is_terminal():
            raise ValidationError(
                f"Cannot retry job '{job_id}' with non-terminal status '{old_job.status}'."
            )

        new_params = dict(old_job.params)
        if resume:
            new_params["resume"] = True
            new_params["run_mode"] = "resume_missing"

        new_job = self.submit_job(
            job_type=old_job.type,
            project_id=old_job.project_id,
            params=new_params,
            is_gpu=old_job.is_gpu,
            max_attempts=max(old_job.max_attempts, old_job.attempt + 1),
        )
        new_job.attempt = old_job.attempt + 1
        return new_job

    def get_queue_summary(self) -> Dict[str, Any]:
        """Summary of GPU job queue and execution state (Section 5.2)."""
        comfy_depth = 0
        try:
            from ..orchestrator.comfy_client import get_comfy_client
            client = get_comfy_client()
            q_info = client.get_queue()
            comfy_depth = len(q_info.get("queue_running", [])) + len(q_info.get("queue_pending", []))
        except Exception:
            pass

        return {
            "gpu_running": 1 if self._active_gpu_job_id else 0,
            "gpu_queued": len(self._queued_gpu_jobs),
            "active_job_id": self._active_gpu_job_id,
            "queue": list(self._queued_gpu_jobs),
            "comfy_queue_depth": comfy_depth,
        }

    def recover_on_startup(self) -> int:
        """Scan all project roots and mark unfinished jobs as interrupted (Section 5.3)."""
        total_recovered = 0
        for root in get_allowed_project_roots():
            if not os.path.isdir(root):
                continue
            for item in os.listdir(root):
                pfolder = os.path.join(root, item)
                if os.path.isdir(pfolder):
                    try:
                        recovered = recover_interrupted_jobs(pfolder)
                        total_recovered += len(recovered)
                    except Exception:
                        pass
        return total_recovered

    async def _run_job(self, job: Job, project_folder: Optional[str]) -> None:
        """Internal worker executing a job within appropriate concurrency locks."""
        sem = self._gpu_semaphore if job.is_gpu else self._cpu_semaphore

        async with sem:
            if job.cancel_requested:
                job.status = JobStatus.CANCELLED
                job.finished_at = time.time()
                self._persist_and_emit(job, project_folder)
                return

            if job.is_gpu:
                self._active_gpu_job_id = job.id
                if job.id in self._queued_gpu_jobs:
                    self._queued_gpu_jobs.remove(job.id)

            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            job.progress.stage = "started"
            job.progress.message = "Running..."
            self.append_log(job.id, "info", f"Job '{job.id}' started execution.")
            self._persist_and_emit(job, project_folder)

            handler = self._handlers.get(job.type)
            if not handler:
                job.status = JobStatus.FAILED
                job.error = {
                    "code": "HANDLER_NOT_FOUND",
                    "message": f"No handler registered for job type '{job.type}'.",
                }
                job.finished_at = time.time()
                self.append_log(job.id, "error", f"No handler for job type '{job.type}'.")
                self._cleanup_active(job)
                self._persist_and_emit(job, project_folder)
                return

            try:
                # Check for cancellation before calling handler
                if job.cancel_requested:
                    raise JobCancelledError(job.id)

                if inspect.iscoroutinefunction(handler):
                    result = await handler(job, self)
                else:
                    result = await asyncio.to_thread(handler, job, self)

                if job.cancel_requested:
                    raise JobCancelledError(job.id)

                job.status = JobStatus.SUCCEEDED
                job.progress.percent = 100.0
                job.progress.stage = "completed"
                job.progress.message = "Job finished successfully."
                job.result = result
                job.finished_at = time.time()
                self.append_log(job.id, "info", f"Job '{job.id}' completed successfully.")

            except JobCancelledError:
                job.status = JobStatus.CANCELLED
                job.finished_at = time.time()
                self.append_log(job.id, "warning", f"Job '{job.id}' was cancelled.")

            except Exception as exc:
                job.status = JobStatus.FAILED
                job.finished_at = time.time()
                if isinstance(exc, AgentApiError):
                    job.error = exc.to_dict()
                else:
                    job.error = {
                        "code": "INTERNAL_ERROR",
                        "message": str(exc) or "Internal job execution error.",
                        "retryable": True,
                    }
                self.append_log(job.id, "error", f"Job '{job.id}' failed: {exc}")

            finally:
                self._cleanup_active(job)
                self._persist_and_emit(job, project_folder)

    def _cleanup_active(self, job: Job) -> None:
        if job.is_gpu and self._active_gpu_job_id == job.id:
            self._active_gpu_job_id = None
        if job.id in self._queued_gpu_jobs:
            self._queued_gpu_jobs.remove(job.id)

    def _persist_and_emit(self, job: Job, project_folder: Optional[str]) -> None:
        pfolder = project_folder
        if not pfolder and job.project_id:
            try:
                pfolder = resolve_project_folder(job.project_id)
            except Exception:
                pass

        if pfolder:
            try:
                save_single_project_job(pfolder, job)
            except Exception as e:
                logger.error(f"Failed to persist job '{job.id}': {e}")

        self._broadcaster.emit("job.updated", job.to_dict(), project_id=job.project_id)
        self._broadcaster.emit("queue.updated", self.get_queue_summary(), project_id=job.project_id)


_JOB_MANAGER: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global JobManager singleton."""
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager()
    return _JOB_MANAGER
