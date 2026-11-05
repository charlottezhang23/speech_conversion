"""Single-worker background job manager (keeps heavy models serialised)."""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from app.util import now_iso

logger = logging.getLogger("interviewai.jobs")


@dataclass
class Job:
    kind: str
    project_id: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    logs: deque = field(default_factory=lambda: deque(maxlen=600))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _log_sink: Callable[[str], None] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ api
    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def log(self, message: str) -> None:
        line = f"{now_iso()}  {message}"
        with self._lock:
            self.logs.append(line)
        if self._log_sink:
            try:
                self._log_sink(message)
            except Exception:
                pass

    def set_progress(self, progress: float, stage: str | None = None, message: str | None = None) -> None:
        self.progress = max(0.0, min(100.0, float(progress)))
        if stage:
            self.stage = stage
        if message is not None:
            self.message = message
            self.log(message)

    def cancel(self) -> None:
        self.cancel_event.set()
        self.log("收到取消请求，正在安全停止…")

    def to_dict(self, log_lines: int = 40) -> dict[str, Any]:
        with self._lock:
            tail = list(self.logs)[-log_lines:]
        return {
            "id": self.id,
            "kind": self.kind,
            "project_id": self.project_id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 1),
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "logs": tail,
        }


class JobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="interviewai")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._active: str | None = None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_job(self) -> Job | None:
        with self._lock:
            job = self._jobs.get(self._active) if self._active else None
            return job if job and job.status in ("queued", "running") else None

    def list_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict(log_lines=5) for j in jobs if project_id in (None, j.project_id)]

    def submit(
        self,
        kind: str,
        project_id: str,
        fn: Callable[[Job], dict[str, Any] | None],
        *,
        log_sink: Callable[[str], None] | None = None,
    ) -> Job:
        active = self.active_job()
        if active:
            raise RuntimeError(
                f"已有任务正在运行（{active.kind} / {active.id}），请等待完成或先取消。"
            )
        job = Job(kind=kind, project_id=project_id, _log_sink=log_sink)
        with self._lock:
            self._jobs[job.id] = job
            self._active = job.id
            # keep memory bounded
            if len(self._jobs) > 40:
                for jid in sorted(self._jobs, key=lambda k: self._jobs[k].created_at)[:10]:
                    if self._jobs[jid].status not in ("running", "queued"):
                        self._jobs.pop(jid, None)
        job.log(f"任务已排队：{kind} / 项目 {project_id}")
        self._executor.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], dict[str, Any] | None]) -> None:
        job.status = "running"
        job.started_at = now_iso()
        t0 = time.perf_counter()
        try:
            result = fn(job)
            if job.cancelled:
                job.status = "cancelled"
                job.message = "已取消（已完成的中间结果已保存）"
            else:
                job.status = "done"
                job.progress = 100.0
                job.message = job.message or "完成"
            job.result = result or {}
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.log("失败：" + job.error)
            job.log(traceback.format_exc(limit=6))
            logger.exception("job %s failed", job.id)
        finally:
            job.elapsed_sec = round(time.perf_counter() - t0, 1)  # type: ignore[attr-defined]
            job.finished_at = now_iso()
            with self._lock:
                if self._active == job.id:
                    self._active = None


MANAGER = JobManager()
