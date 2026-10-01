"""Data models for background jobs and progress tracking (Section 5.1)."""

import datetime
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import uuid


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    ALL = {QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED, INTERRUPTED}
    TERMINAL = {SUCCEEDED, FAILED, CANCELLED, INTERRUPTED}


@dataclass
class JobProgress:
    percent: float = 0.0
    stage: str = "queued"
    stage_index: int = 0
    stage_count: int = 1
    message: str = ""
    scene_id: Optional[str] = None
    eta_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "percent": round(float(self.percent), 1),
            "stage": str(self.stage or ""),
            "stage_index": int(self.stage_index),
            "stage_count": int(self.stage_count),
            "message": str(self.message or ""),
            "scene_id": self.scene_id,
            "eta_seconds": self.eta_seconds,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "JobProgress":
        if not isinstance(data, dict):
            return cls()
        return cls(
            percent=float(data.get("percent", 0.0)),
            stage=str(data.get("stage", "queued")),
            stage_index=int(data.get("stage_index", 0)),
            stage_count=int(data.get("stage_count", 1)),
            message=str(data.get("message", "")),
            scene_id=data.get("scene_id"),
            eta_seconds=data.get("eta_seconds"),
        )


@dataclass
class JobLogEntry:
    seq: int
    timestamp: float
    level: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
        }
        if self.details is not None:
            res["details"] = self.details
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobLogEntry":
        return cls(
            seq=int(data.get("seq", 0)),
            timestamp=float(data.get("timestamp", time.time())),
            level=str(data.get("level", "info")),
            message=str(data.get("message", "")),
            details=data.get("details"),
        )


def generate_job_id() -> str:
    """Generate job id format: job_YYYYMMDD_HHMMSS_<4 hex chars> (Section 5.1)."""
    now = datetime.datetime.now()
    tag = uuid.uuid4().hex[:4]
    return f"job_{now.strftime('%Y%m%d_%H%M%S')}_{tag}"


@dataclass
class Job:
    id: str
    type: str
    project_id: Optional[str] = None
    status: str = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: JobProgress = field(default_factory=JobProgress)
    comfy: Dict[str, Any] = field(default_factory=lambda: {"prompt_ids": [], "current_prompt_id": None})
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    cancel_requested: bool = False
    attempt: int = 1
    max_attempts: int = 1
    is_gpu: bool = True

    def is_terminal(self) -> bool:
        return self.status in JobStatus.TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "project_id": self.project_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress.to_dict() if hasattr(self.progress, "to_dict") else self.progress,
            "comfy": dict(self.comfy or {}),
            "params": dict(self.params or {}),
            "result": self.result,
            "error": self.error,
            "warnings": list(self.warnings or []),
            "cancel_requested": bool(self.cancel_requested),
            "attempt": int(self.attempt),
            "max_attempts": int(self.max_attempts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        prog = JobProgress.from_dict(data.get("progress"))
        return cls(
            id=str(data["id"]),
            type=str(data.get("type", "unknown")),
            project_id=data.get("project_id"),
            status=str(data.get("status", JobStatus.QUEUED)),
            created_at=float(data.get("created_at", time.time())),
            started_at=float(data["started_at"]) if data.get("started_at") is not None else None,
            finished_at=float(data["finished_at"]) if data.get("finished_at") is not None else None,
            progress=prog,
            comfy=dict(data.get("comfy") or {"prompt_ids": [], "current_prompt_id": None}),
            params=dict(data.get("params") or {}),
            result=data.get("result"),
            error=data.get("error"),
            warnings=list(data.get("warnings") or []),
            cancel_requested=bool(data.get("cancel_requested", False)),
            attempt=int(data.get("attempt", 1)),
            max_attempts=int(data.get("max_attempts", 1)),
            is_gpu=bool(data.get("is_gpu", True)),
        )
