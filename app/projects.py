"""Interview project storage: one folder per project under projects/."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from app.config import PROJECTS_DIR
from app.util import (
    new_project_id,
    now_iso,
    read_json,
    slugify,
    write_json,
)

SCHEMA_VERSION = 1
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".m4v", ".mov", ".webm", ".amr", ".aiff", ".aif", ".caf"}

STAGES = ("transcribe", "diarize", "merge", "summary")


class ProjectError(Exception):
    pass


def _validate_id(project_id: str) -> str:
    if not ID_RE.match(project_id or ""):
        raise ProjectError("无效的项目 ID")
    return project_id


class Project:
    """A single interview project folder."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    # ---------------------------------------------------------------- paths
    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def dir(self) -> Path:
        return PROJECTS_DIR / self.id

    @property
    def meta_path(self) -> Path:
        return self.dir / "project.json"

    @property
    def audio_dir(self) -> Path:
        return self.dir / "audio"

    @property
    def original_audio(self) -> Path:
        return self.dir / self.data.get("audio", {}).get("relative_path", "audio/source.bin")

    @property
    def normalized_wav(self) -> Path:
        return self.audio_dir / "normalized_16k.wav"

    @property
    def asr_dir(self) -> Path:
        return self.dir / "asr"

    @property
    def asr_manifest(self) -> Path:
        return self.asr_dir / "manifest.json"

    @property
    def diarization_dir(self) -> Path:
        return self.dir / "diarization"

    @property
    def diarization_raw(self) -> Path:
        return self.dir / "diarization" / "raw.json"

    @property
    def draft_transcript(self) -> Path:
        return self.dir / "transcript" / "draft.json"

    @property
    def edited_transcript(self) -> Path:
        return self.dir / "transcript" / "edited.json"

    @property
    def summary_dir(self) -> Path:
        return self.dir / "summary"

    @property
    def summary_final(self) -> Path:
        return self.summary_dir / "final.json"

    @property
    def summary_state(self) -> Path:
        return self.summary_dir / "state.json"

    @property
    def log_path(self) -> Path:
        return self.dir / "logs" / "job.log"

    @property
    def timings_path(self) -> Path:
        return self.dir / "logs" / "timings.json"

    # -------------------------------------------------------------- settings
    @property
    def settings(self) -> dict[str, Any]:
        return self.data.setdefault("settings", {})

    @property
    def speaker_names(self) -> dict[str, str]:
        return self.data.setdefault("speaker_names", {})

    @property
    def stages(self) -> dict[str, Any]:
        return self.data.setdefault("stages", {})

    # ------------------------------------------------------------------ save
    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        write_json(self.meta_path, self.data)

    def touch_stage(self, name: str, status: str, **extra: Any) -> None:
        stage = self.stages.setdefault(name, {})
        stage.update({"status": status, "updated_at": now_iso()})
        if status == "running":
            stage["started_at"] = now_iso()
            stage.pop("elapsed_sec", None)
        if status in ("done", "failed", "cancelled"):
            stage["finished_at"] = now_iso()
        if status == "done":
            stage.pop("error", None)
        stage.update({k: v for k, v in extra.items() if v is not None})
        self.save()

    def log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = now_iso()
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {message}\n")

    def record_timing(self, stage: str, elapsed_sec: float, **extra: Any) -> None:
        timings = read_json(self.timings_path, {}) or {}
        timings[stage] = {"elapsed_sec": round(elapsed_sec, 1), **extra}
        write_json(self.timings_path, timings)
        self.touch_stage(stage, "done", elapsed_sec=round(elapsed_sec, 1), **extra)

    # ------------------------------------------------------------- statuses
    def stage_status(self, name: str) -> str:
        return self.stages.get(name, {}).get("status", "pending")

    def has_draft(self) -> bool:
        return self.draft_transcript.exists()

    def has_edited(self) -> bool:
        return self.edited_transcript.exists()

    def has_summary(self) -> bool:
        return self.summary_final.exists()

    def summary(self) -> dict[str, Any]:
        audio = self.data.get("audio", {})
        return {
            "id": self.id,
            "name": self.data.get("name", self.id),
            "created_at": self.data.get("created_at"),
            "updated_at": self.data.get("updated_at"),
            "duration_sec": audio.get("duration_sec"),
            "size_bytes": audio.get("size_bytes"),
            "original_name": audio.get("original_name"),
            "format": audio.get("format"),
            "audio": audio,
            "audio_ready": self.original_audio.exists(),
            "stages": {k: self.stage_status(k) for k in STAGES},
            "stage_details": {k: dict(self.stages.get(k) or {}) for k in STAGES},
            "has_draft": self.has_draft(),
            "has_edited": self.has_edited(),
            "has_summary": self.has_summary(),
            "language": self.settings.get("language", "auto"),
            "model": self.settings.get("model"),
        }


# ------------------------------------------------------------------ factory
def create_project(
    name: str,
    source_audio: Path,
    original_filename: str,
    *,
    duration_sec: float | None = None,
    size_bytes: int | None = None,
    fmt: str | None = None,
    extra_audio: dict[str, Any] | None = None,
) -> Project:
    source_audio = Path(source_audio)
    if not source_audio.exists():
        raise ProjectError("音频文件不存在")
    ext = source_audio.suffix.lower()
    if ext not in AUDIO_EXTS:
        raise ProjectError(f"不支持的音频格式: {ext}")

    pid = new_project_id(name)
    pdir = PROJECTS_DIR / pid
    (pdir / "audio").mkdir(parents=True, exist_ok=True)

    rel = f"audio/source{ext}"
    shutil.copyfile(source_audio, pdir / rel)

    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": pid,
        "name": (name or Path(original_filename).stem or "未命名访谈").strip()[:120],
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "audio": {
            "original_name": original_filename,
            "relative_path": rel,
            "format": (fmt or ext.lstrip(".")).lower(),
            "duration_sec": duration_sec,
            "size_bytes": size_bytes if size_bytes is not None else source_audio.stat().st_size,
            **(extra_audio or {}),
        },
        "settings": {},
        "speaker_names": {},
        "stages": {k: {"status": "pending"} for k in STAGES},
        "notes": [],
    }
    project = Project(data)
    project.save()
    project.log(f"创建项目 {pid} ← {original_filename}")
    return project


def get_project(project_id: str) -> Project:
    pid = _validate_id(project_id)
    meta = PROJECTS_DIR / pid / "project.json"
    data = read_json(meta)
    if not isinstance(data, dict):
        raise ProjectError("项目不存在")
    return Project(data)


def list_projects() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not PROJECTS_DIR.exists():
        return items
    for child in PROJECTS_DIR.iterdir():
        if not child.is_dir() or not ID_RE.match(child.name):
            continue
        data = read_json(child / "project.json")
        if not isinstance(data, dict):
            continue
        items.append(Project(data).summary())
    items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return items


def rename_project(project_id: str, name: str) -> Project:
    project = get_project(project_id)
    project.data["name"] = (name or "").strip()[:120] or project.data["name"]
    project.save()
    return project


def delete_project(project_id: str) -> None:
    pid = _validate_id(project_id)
    pdir = PROJECTS_DIR / pid
    if not pdir.exists():
        raise ProjectError("项目不存在")
    shutil.rmtree(pdir)


def export_dir(project: Project) -> Path:
    d = project.dir / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_note(project: Project, text: str) -> None:
    project.log(text)


def read_log(project: Project, tail_bytes: int = 12000) -> str:
    if not project.log_path.exists():
        return ""
    with project.log_path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - tail_bytes))
        return fh.read().decode("utf-8", errors="replace")


__all__ = [
    "Project",
    "ProjectError",
    "create_project",
    "get_project",
    "list_projects",
    "rename_project",
    "delete_project",
    "export_dir",
    "read_log",
    "write_note",
    "AUDIO_EXTS",
    "slugify",
]
