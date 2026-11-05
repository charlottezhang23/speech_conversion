"""Local FastAPI server (binds 127.0.0.1 only)."""
from __future__ import annotations

import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import audio as A
from app import config
from app import hf
from app import transcript as T
from app.config import STATIC_DIR, TMP_DIR, load_settings, save_settings, set_hf_token
from app.jobs import MANAGER
from app.pipeline import (
    PipelineError,
    run_align_only,
    run_diarization_only,
    run_rebuild_draft,
    run_refine_speakers,
    run_transcription,
)
from app.projects import (
    AUDIO_EXTS,
    Project,
    ProjectError,
    create_project,
    delete_project,
    export_dir,
    get_project,
    list_projects,
    read_log,
    rename_project,
)
from app.util import hhmmss, now_iso, read_json, slugify, write_json
from exporters import format as F
from exporters.docx_writer import build_docx
from summarization import ollama_client
from summarization.prompts import TEMPLATES
from summarization.summarize import SummaryError, summarize

logger = logging.getLogger("interviewai.server")

app = FastAPI(title="访谈转录与会议纪要", docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _project(project_id: str) -> Project:
    try:
        return get_project(project_id)
    except ProjectError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "time": now_iso()}


@app.get("/api/system")
def system_info() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    def v(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    settings = load_settings()
    return {
        "env": config.env_report(),
        "packages": {
            "whisperx": v("whisperx"),
            "faster-whisper": v("faster-whisper"),
            "ctranslate2": v("ctranslate2"),
            "torch": v("torch"),
            "pyannote.audio": v("pyannote.audio"),
        },
        "ollama": {"url": settings["ollama_url"], "online": ollama_client.ping(settings["ollama_url"])},
        "settings": settings,
        "templates": {
            k: {"label": val["label"], "description": val["description"], "sections": val["sections"]}
            for k, val in TEMPLATES.items()
        },
    }


# ----------------------------------------------------------------- settings
@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    settings = load_settings()
    settings["hf_token_set"] = config.has_hf_token()
    return settings


@app.post("/api/settings")
def post_settings(patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    saved = save_settings(patch)
    saved["hf_token_set"] = config.has_hf_token()
    return saved


@app.post("/api/settings/hf-token")
def post_hf_token(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    token = str(payload.get("token") or "").strip()
    set_hf_token(token or None)
    return {"hf_token_set": config.has_hf_token()}


@app.get("/api/hf/verify")
def api_hf_verify() -> dict[str, Any]:
    """Check the stored token and whether it may access the gated diarization model."""
    return hf.check(config.get_hf_token())


# ----------------------------------------------------------------- projects
@app.get("/api/projects")
def api_list_projects() -> list[dict[str, Any]]:
    items = list_projects()
    active = MANAGER.active_job()
    for item in items:
        item["busy"] = bool(active and active.project_id == item["id"])
    return items


@app.post("/api/projects")
async def api_create_project(
    file: UploadFile = File(...),
    name: str = Form(""),
) -> dict[str, Any]:
    filename = file.filename or "audio"
    suffix = Path(filename).suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式：{suffix or '未知'}")
    tmp_path = TMP_DIR / f"upload-{now_iso().replace(':', '')}-{slugify(filename)}{suffix}"
    try:
        with tmp_path.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    finally:
        await file.close()

    try:
        info = await run_in_threadpool(A.probe, tmp_path)
        duration = info.duration_sec
        project = await run_in_threadpool(
            create_project,
            name or Path(filename).stem,
            tmp_path,
            filename,
            duration_sec=duration,
            size_bytes=tmp_path.stat().st_size,
            fmt=info.format,
            extra_audio={
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "codec": info.codec,
                "bit_rate": info.bit_rate,
            },
        )
    except (A.FfmpegError, ProjectError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"project": project.summary()}


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str) -> dict[str, Any]:
    project = _project(project_id)
    data = project.summary()
    data["settings"] = project.settings
    data["speaker_names"] = project.speaker_names
    data["timings"] = read_json(project.timings_path, {}) or {}
    data["log_tail"] = read_log(project)
    data["has_normalized"] = project.normalized_wav.exists()
    data["ollama_summary"] = (read_json(project.summary_final, {}) or {}).get("generated_at")
    return data


@app.patch("/api/projects/{project_id}")
def api_rename(project_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project = rename_project(project_id, str(payload.get("name") or ""))
    return project.summary()


@app.delete("/api/projects/{project_id}")
def api_delete(project_id: str) -> dict[str, Any]:
    active = MANAGER.active_job()
    if active and active.project_id == project_id:
        raise HTTPException(status_code=409, detail="该项目有任务正在运行，请先取消")
    try:
        delete_project(project_id)
    except ProjectError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": project_id}


# -------------------------------------------------------------------- audio
@app.get("/api/projects/{project_id}/audio")
def api_audio(project_id: str, request: Request):
    project = _project(project_id)
    path = project.original_audio
    if not path.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")
    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type=media_type, filename=path.name)

    try:
        units, rng = range_header.split("=", 1)
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except (ValueError, AttributeError):
        start, end = 0, file_size - 1
    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        raise HTTPException(status_code=416, detail="Invalid range")

    def iter_file():
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = fh.read(min(256 * 1024, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    }
    return StreamingResponse(iter_file(), status_code=206, headers=headers, media_type=media_type)


# --------------------------------------------------------------------- jobs
@app.get("/api/jobs/active")
def api_active_job() -> dict[str, Any] | None:
    job = MANAGER.active_job()
    return job.to_dict() if job else None


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict[str, Any]:
    job = MANAGER.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    payload = job.to_dict()
    payload["elapsed_sec"] = getattr(job, "elapsed_sec", None)
    return payload


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str) -> dict[str, Any]:
    job = MANAGER.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    job.cancel()
    return job.to_dict()


def _start_job(kind: str, project: Project, fn) -> dict[str, Any]:
    if not config.find_ffmpeg():
        raise HTTPException(status_code=400, detail="未找到 FFmpeg，请先运行 install.ps1")
    try:
        job = MANAGER.submit(kind, project.id, fn, log_sink=project.log)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.to_dict()


@app.post("/api/projects/{project_id}/transcribe")
def api_transcribe(project_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    project = _project(project_id)
    settings_patch = payload.get("settings") or {}
    allowed = {
        "model",
        "compute_type",
        "language",
        "beam_size",
        "batch_size",
        "threads",
        "num_speakers",
        "min_speakers",
        "max_speakers",
        "align",
        "diarize_enabled",
        "chunk_minutes",
        "diarization_model",
    }
    for key, value in settings_patch.items():
        if key in allowed:
            project.settings[key] = value
    project.save()
    return _start_job("transcribe", project, run_transcription)


@app.post("/api/projects/{project_id}/diarize")
def api_diarize(project_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    project = _project(project_id)
    for key in ("num_speakers", "min_speakers", "max_speakers", "diarization_model"):
        if key in (payload or {}):
            project.settings[key] = payload[key]
    project.save()
    return _start_job("diarize", project, run_diarization_only)


@app.post("/api/projects/{project_id}/rebuild")
def api_rebuild(project_id: str) -> dict[str, Any]:
    project = _project(project_id)
    return _start_job("rebuild", project, run_rebuild_draft)


@app.post("/api/projects/{project_id}/align")
def api_align(project_id: str) -> dict[str, Any]:
    """Add word-level timestamps to existing ASR output (no re-transcription)."""
    project = _project(project_id)
    return _start_job("align", project, run_align_only)


@app.post("/api/projects/{project_id}/refine")
def api_refine(project_id: str) -> dict[str, Any]:
    """Re-verify speaker labels with per-turn embeddings (no re-diarization)."""
    project = _project(project_id)
    return _start_job("refine", project, run_refine_speakers)


# --------------------------------------------------------------- transcript
@app.get("/api/projects/{project_id}/transcript")
def api_transcript(project_id: str, version: str = "effective") -> dict[str, Any]:
    project = _project(project_id)
    try:
        return T.transcript_payload(project, version)
    except T.TranscriptError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/projects/{project_id}/transcript")
def api_save_transcript(project_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project = _project(project_id)
    try:
        edited = T.save_edits(project, payload.get("utterances") or [], payload.get("speaker_names") or {})
    except (T.TranscriptError, ProjectError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True, "updated_at": edited["updated_at"], "stats": edited["stats"]}


@app.delete("/api/projects/{project_id}/transcript/edited")
def api_reset_transcript(project_id: str) -> dict[str, Any]:
    project = _project(project_id)
    T.reset_edits(project)
    return {"reset": True}


# ------------------------------------------------------------------ summary
@app.get("/api/projects/{project_id}/summary")
def api_get_summary(project_id: str) -> dict[str, Any]:
    project = _project(project_id)
    data = read_json(project.summary_final)
    if not data:
        raise HTTPException(status_code=404, detail="尚未生成纪要")
    return data


@app.put("/api/projects/{project_id}/summary")
def api_save_summary(project_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project = _project(project_id)
    data = read_json(project.summary_final)
    if not data:
        raise HTTPException(status_code=404, detail="尚未生成纪要")
    markdown = str(payload.get("markdown") or "").strip()
    if not markdown:
        raise HTTPException(status_code=400, detail="内容为空")
    data["markdown"] = markdown
    data["human_edited"] = True
    data["edited_at"] = now_iso()
    write_json(project.summary_final, data)
    from app.util import write_text

    write_text(project.summary_dir / "final.md", markdown + "\n")
    project.log("保存人工修订的纪要")
    return {"saved": True, "edited_at": data["edited_at"]}


@app.post("/api/projects/{project_id}/summary")
def api_generate_summary(project_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    project = _project(project_id)
    template = str(payload.get("template") or "interview")
    force = bool(payload.get("force"))

    def _run(job) -> dict[str, Any]:
        transcript, source = T.load_effective(project)
        names = T.effective_speaker_names(project, transcript)
        settings = load_settings()
        model = str(payload.get("model") or settings.get("ollama_model"))
        base_url = str(payload.get("ollama_url") or settings.get("ollama_url"))
        project.touch_stage("summary", "running")
        job.log(f"使用模板 {template}，模型 {model}，逐字稿来源：{'人工校对版' if source == 'edited' else '机器初稿'}")
        if not ollama_client.ping(base_url):
            raise SummaryError(f"无法连接本地 Ollama（{base_url}）。请先启动 `ollama serve`。")
        try:
            result = summarize(
                utterances=transcript["utterances"],
                names=names,
                meta=T.summary_source_meta(project, transcript, source),
                summary_dir=project.summary_dir,
                template=template,
                model=model,
                base_url=base_url,
                num_ctx=int(settings.get("ollama_num_ctx", 16384)),
                timeout_sec=float(settings.get("ollama_timeout_sec", 900)),
                max_chunk_chars=int(settings.get("summary_chunk_chars", 4500)),
                verify=bool(payload.get("verify", settings.get("verify_summary", True))),
                force=force,
                progress=lambda pct, stage, msg: job.set_progress(pct, stage, msg),
                is_cancelled=lambda: job.cancelled,
                log=lambda msg: project.log(msg),
            )
        except SummaryError:
            if job.cancelled:
                project.touch_stage("summary", "cancelled")
                return {"cancelled": True}
            raise
        except Exception:
            if job.cancelled:
                project.touch_stage("summary", "cancelled")
                return {"cancelled": True}
            raise
        project.touch_stage("summary", "done", template=template, model=model)
        return {
            "template": template,
            "model": model,
            "generated_at": result["generated_at"],
            "chars": len(result["markdown"]),
        }

    return _start_job("summary", project, _run)


# -------------------------------------------------------------------- ollama
@app.get("/api/ollama/models")
def api_ollama_models(url: str | None = None) -> dict[str, Any]:
    settings = load_settings()
    base = url or settings.get("ollama_url")
    try:
        models = ollama_client.list_models(base)
    except ollama_client.OllamaError as exc:
        return {"online": False, "error": str(exc), "models": []}
    return {"online": True, "models": models, "url": base, "version": ollama_client.version(base)}


# -------------------------------------------------------------------- export
@app.get("/api/projects/{project_id}/export/{fmt}")
def api_export(project_id: str, fmt: str, version: str = "effective", include_summary: bool = True):
    project = _project(project_id)
    try:
        utterances, names, source, payload = T.dump_for_export(project, version)
    except (T.TranscriptError, ProjectError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    meta = {
        "name": project.data.get("name"),
        "original_name": (project.data.get("audio") or {}).get("original_name"),
        "duration_sec": (project.data.get("audio") or {}).get("duration_sec"),
    }
    stamp = now_iso().replace(":", "").replace("Z", "")
    base = slugify(f"{project.data.get('name', project.id)}-{stamp}")
    out_dir = export_dir(project)
    summary_data = read_json(project.summary_final) if include_summary else None
    try:
        if fmt == "txt":
            path = out_dir / f"{base}.txt"
            path.write_text(F.to_txt(utterances, names, meta, source), encoding="utf-8")
            return FileResponse(path, filename=path.name, media_type="text/plain; charset=utf-8")
        if fmt == "md":
            path = out_dir / f"{base}.md"
            body = F.to_markdown(utterances, names, meta, source)
            if summary_data and summary_data.get("markdown"):
                body += "\n\n---\n\n# 结构化纪要\n\n" + summary_data["markdown"] + "\n"
            path.write_text(body, encoding="utf-8")
            return FileResponse(path, filename=path.name, media_type="text/markdown; charset=utf-8")
        if fmt == "srt":
            path = out_dir / f"{base}.srt"
            path.write_text(F.to_srt(utterances, names, fallback_end=meta.get("duration_sec")), encoding="utf-8-sig")
            return FileResponse(path, filename=path.name, media_type="application/x-subrip")
        if fmt == "json":
            path = out_dir / f"{base}.json"
            path.write_text(
                F.to_json_bundle(
                    project_id=project.id,
                    meta=project.data,
                    utterances=utterances,
                    names=names,
                    source=source,
                    summary=summary_data,
                ),
                encoding="utf-8",
            )
            return FileResponse(path, filename=path.name, media_type="application/json")
        if fmt == "docx":
            path = out_dir / f"{base}.docx"
            build_docx(
                path,
                title=str(project.data.get("name") or "访谈记录"),
                meta_lines=[
                    f"原始文件：{meta.get('original_name')}",
                    f"时长：{hhmmss(meta.get('duration_sec'))}",
                    f"逐字稿版本：{'人工校对版' if source == 'edited' else '机器初稿'}",
                ],
                utterances=utterances,
                names=names,
                summary_markdown=(summary_data or {}).get("markdown"),
                include_transcript=True,
                include_summary=bool(summary_data),
            )
            return FileResponse(
                path,
                filename=path.name,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"导出失败：{exc}") from exc
    raise HTTPException(status_code=400, detail=f"不支持的导出格式：{fmt}")


@app.get("/api/projects/{project_id}/summary.md")
def api_summary_md(project_id: str):
    project = _project(project_id)
    data = read_json(project.summary_final)
    if not data:
        raise HTTPException(status_code=404, detail="尚未生成纪要")
    path = project.summary_dir / "final.md"
    path.write_text(data["markdown"] + "\n", encoding="utf-8")
    return FileResponse(path, filename=f"{slugify(project.data.get('name', project.id))}-纪要.md", media_type="text/markdown; charset=utf-8")


@app.exception_handler(PipelineError)
def _pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
