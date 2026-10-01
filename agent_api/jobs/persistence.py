"""Persistence of jobs to <project>/jobs/jobs.json with crash recovery (Section 5.3)."""

import json
import os
import time
from typing import Dict, List, Optional

from ...core.atomic_write import atomic_write_json
from .models import Job, JobStatus


def get_project_jobs_path(project_folder: str) -> str:
    """Return path to <project>/jobs/jobs.json."""
    return os.path.join(project_folder, "jobs", "jobs.json")


def load_project_jobs(project_folder: str) -> Dict[str, Job]:
    """Load all jobs for a project from <project>/jobs/jobs.json."""
    path = get_project_jobs_path(project_folder)
    if not os.path.isfile(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return {}

    jobs: Dict[str, Job] = {}
    if isinstance(raw, dict):
        # Can be {"jobs": {id: job_dict}} or direct {id: job_dict}
        items = raw.get("jobs", raw) if "jobs" in raw and isinstance(raw["jobs"], dict) else raw
        for job_id, data in items.items():
            if isinstance(data, dict) and "id" in data:
                try:
                    jobs[str(job_id)] = Job.from_dict(data)
                except Exception:
                    pass
    elif isinstance(raw, list):
        for data in raw:
            if isinstance(data, dict) and "id" in data:
                try:
                    jobs[str(data["id"])] = Job.from_dict(data)
                except Exception:
                    pass

    return jobs


def save_project_jobs(project_folder: str, jobs: Dict[str, Job]) -> None:
    """Save all jobs for a project to <project>/jobs/jobs.json atomically."""
    path = get_project_jobs_path(project_folder)
    serialized = {job_id: job.to_dict() for job_id, job in jobs.items()}
    atomic_write_json(path, serialized)


def save_single_project_job(project_folder: str, job: Job) -> None:
    """Update or insert a single job into <project>/jobs/jobs.json atomically."""
    current = load_project_jobs(project_folder)
    current[job.id] = job
    save_project_jobs(project_folder, current)


def recover_interrupted_jobs(project_folder: str) -> List[str]:
    """Recover dangling jobs left running after an ungraceful server shutdown (Section 5.3).
    
    Any job left with status 'running' (or dangling queued) becomes 'interrupted'.
    Returns list of recovered job IDs.
    """
    jobs = load_project_jobs(project_folder)
    changed = False
    recovered_ids: List[str] = []

    now = time.time()
    for job_id, job in jobs.items():
        if job.status == JobStatus.RUNNING:
            job.status = JobStatus.INTERRUPTED
            job.finished_at = now
            msg = "Server restarted while job was running."
            if msg not in job.warnings:
                job.warnings.append(msg)
            recovered_ids.append(job_id)
            changed = True

    if changed:
        save_project_jobs(project_folder, jobs)

    return recovered_ids
