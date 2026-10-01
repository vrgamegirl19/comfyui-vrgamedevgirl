"""Background Job system and event streaming package (Section 5)."""

from .events import EventBroadcaster, get_event_broadcaster
from .manager import JobManager, get_job_manager
from .models import (
    Job,
    JobLogEntry,
    JobProgress,
    JobStatus,
    generate_job_id,
)
from .llm_jobs import is_llm_runner_gpu, register_llm_job_handlers
from .persistence import (
    get_project_jobs_path,
    load_project_jobs,
    recover_interrupted_jobs,
    save_project_jobs,
    save_single_project_job,
)

__all__ = [
    "Job",
    "JobStatus",
    "JobProgress",
    "JobLogEntry",
    "generate_job_id",
    "JobManager",
    "get_job_manager",
    "EventBroadcaster",
    "get_event_broadcaster",
    "get_project_jobs_path",
    "load_project_jobs",
    "save_project_jobs",
    "save_single_project_job",
    "recover_interrupted_jobs",
    "register_llm_job_handlers",
    "is_llm_runner_gpu",
]
