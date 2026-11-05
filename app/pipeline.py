"""Pipeline orchestration: normalise → ASR → diarization → merge → draft transcript."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from app import audio as A
from app.config import load_settings
from app.jobs import Job
from app.projects import Project, get_project
from app.util import hhmmss, now_iso, read_json, write_json
from transcription import clean as C
from transcription import merge as M


class PipelineError(RuntimeError):
    pass


def _chunk_key(settings: dict[str, Any]) -> str:
    from transcription.asr import ASR_PROMPT_VERSION

    raw = json.dumps(
        {
            "model": settings.get("model"),
            "compute_type": settings.get("compute_type"),
            "language": settings.get("language"),
            "beam_size": settings.get("beam_size"),
            "align": settings.get("align"),
            "prompt_version": ASR_PROMPT_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def ensure_normalized(project: Project, job: Job) -> Path:
    wav = project.normalized_wav
    src = project.original_audio
    if not src.exists():
        raise PipelineError("原始音频文件丢失")
    if wav.exists() and wav.stat().st_size > 0 and wav.stat().st_mtime >= src.stat().st_mtime:
        return wav
    job.set_progress(1.0, "normalize", "正在转为 16kHz 单声道 WAV…")
    A.normalize(src, wav)
    job.log(f"标准化音频完成：{wav.name}（{wav.stat().st_size / 1e6:.1f} MB）")
    return wav


def _load_asr(project: Project, wav: Path, settings: dict[str, Any], job: Job) -> dict[str, Any]:
    """Transcribe chunk by chunk, caching each chunk result for resume."""
    from transcription.asr import AsrEngine, normalize_segments

    chunks_dir = project.asr_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    key = _chunk_key(settings)
    duration = float(project.data["audio"].get("duration_sec") or 0.0)

    manifest = read_json(project.asr_manifest, {}) or {}
    plan = manifest.get("chunks") if manifest.get("key") == key else None
    if not plan:
        job.set_progress(3.0, "normalize", "正在分析静音位置以决定切分点…")
        silences = A.detect_silences(wav)
        target = float(settings.get("chunk_minutes", 10.0)) * 60.0
        plan = [{"index": i, "start": a, "end": b} for i, (a, b) in enumerate(A.plan_chunks(duration, silences, target))]
        write_json(
            project.asr_manifest,
            {
                "key": key,
                "settings": settings,
                "duration_sec": duration,
                "created_at": now_iso(),
                "chunks": plan,
            },
        )
        job.log(f"音频切分为 {len(plan)} 段（每段约 {settings.get('chunk_minutes', 10)} 分钟，优先在静音处切割）")

    engine = AsrEngine(
        settings.get("model", "medium"),
        device="cpu",
        compute_type=settings.get("compute_type", "int8"),
        threads=int(settings.get("threads") or 0),
        beam_size=int(settings.get("beam_size") or 5),
        language=settings.get("language"),
    )
    job.log(f"WhisperX 模型已载入：{settings.get('model')} / {settings.get('compute_type')}（{engine.load_sec}s）")

    all_segments: list[dict[str, Any]] = []
    languages: list[str] = []
    language = settings.get("language") if settings.get("language") not in (None, "", "auto") else None
    batch_size = int(settings.get("batch_size") or 8)
    segments: list[dict[str, Any]] = []
    try:
        for idx, chunk in enumerate(plan):
            if job.cancelled:
                break
            cache = chunks_dir / f"{int(chunk['index']):03d}.json"
            cached = read_json(cache)
            if isinstance(cached, dict) and cached.get("key") == key and cached.get("segments") is not None:
                segments = cached["segments"]
                if cached.get("language"):
                    languages.append(str(cached["language"]))
                job.log(f"分段 {idx + 1}/{len(plan)} 使用已有结果（{hhmmss(chunk['start'])}）")
            else:
                tmp_wav = project.audio_dir / "chunks" / f"chunk_{int(chunk['index']):03d}.wav"
                t0 = time.perf_counter()
                A.slice_wav(wav, float(chunk["start"]), float(chunk["end"]), tmp_wav)
                base_pct = 5 + 58 * idx / max(1, len(plan))
                step_pct = 58 / max(1, len(plan))

                def _sub(pct: float, idx=idx, chunk=chunk, base_pct=base_pct, step_pct=step_pct) -> None:
                    job.set_progress(
                        base_pct + step_pct * min(max(pct, 0.0), 100.0) / 100.0,
                        "asr",
                        f"正在识别分段 {idx + 1}/{len(plan)}（{hhmmss(chunk['start'])} - {hhmmss(chunk['end'])}）{pct:.0f}%",
                    )

                _sub(0.0)
                result = engine.transcribe_chunk(
                    tmp_wav, language=language, batch_size=batch_size, progress=_sub
                )
                segs = normalize_segments(result, offset=float(chunk["start"]))
                detected = result.get("language") or language
                if detected:
                    languages.append(str(detected))
                if settings.get("align") and detected in ("zh", "ja", "en"):
                    try:
                        aligned = engine.align(
                            [{"start": s["start"] - float(chunk["start"]), "end": s["end"] - float(chunk["start"]), "text": s["text"]} for s in segs],
                            tmp_wav,
                            detected,
                        )
                        segs = normalize_segments(aligned, offset=float(chunk["start"]))
                        job.log(f"分段时间对齐完成（{detected}）")
                    except Exception as exc:  # alignment is optional
                        job.log(f"对齐失败，使用 Whisper 原生时间戳：{exc}")
                write_json(
                    cache,
                    {
                        "key": key,
                        "index": int(chunk["index"]),
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "language": detected,
                        "segments": segs,
                        "elapsed_sec": round(time.perf_counter() - t0, 1),
                        "created_at": now_iso(),
                    },
                )
                job.log(
                    f"分段 {idx + 1}/{len(plan)} 完成：{len(segs)} 句，用时 {round(time.perf_counter() - t0, 1)}s"
                )
                tmp_wav.unlink(missing_ok=True)
                segments = segs
            all_segments.extend(segments)
    finally:
        engine.unload()

    language_summary = max(set(languages), key=languages.count) if languages else (language or "auto")
    manifest.update(
        {
            "key": key,
            "settings": settings,
            "chunks": plan,
            "languages": languages,
            "detected_language": language_summary,
            "completed_chunks": len([1 for c in plan if (chunks_dir / f"{int(c['index']):03d}.json").exists()]),
            "updated_at": now_iso(),
            "segment_count": len(all_segments),
        }
    )
    write_json(project.asr_manifest, manifest)
    if job.cancelled:
        raise JobCancelled("转录已取消，已完成的分段已保存")
    return {"segments": all_segments, "language": language_summary, "chunks": len(plan)}


class JobCancelled(RuntimeError):
    pass


def _diarize(project: Project, wav: Path, settings: dict[str, Any], job: Job, start_pct: float, span_pct: float) -> dict[str, Any]:
    from transcription.diarization import DiarizationEngine, DiarizationError

    engine = DiarizationEngine(settings.get("diarization_model"))
    try:
        turns = engine.diarize(
            wav,
            num_speakers=settings.get("num_speakers"),
            min_speakers=settings.get("min_speakers"),
            max_speakers=settings.get("max_speakers"),
            progress=lambda pct, msg: job.set_progress(start_pct + span_pct * pct / 100.0, "diarize", msg),
        )
    finally:
        engine.unload()
    if job.cancelled:
        raise JobCancelled("说话人分离已取消")

    raw = {
        "schema_version": 1,
        "model": settings.get("diarization_model"),
        "created_at": now_iso(),
        "settings": {
            "num_speakers": settings.get("num_speakers"),
            "min_speakers": settings.get("min_speakers"),
            "max_speakers": settings.get("max_speakers"),
        },
        "speakers": sorted({t["speaker"] for t in turns}),
        "turns": turns,
    }
    write_json(project.diarization_raw, raw)
    job.log(f"说话人分离完成：{len(raw['speakers'])} 位说话人，{len(turns)} 个时间片段")
    return raw


def build_draft(project: Project, asr: dict[str, Any], diarization: dict[str, Any] | None) -> dict[str, Any]:
    turns = [(t["start"], t["end"], t["speaker"]) for t in (diarization or {}).get("turns", [])]
    cleaned_segments, clean_stats = C.clean_segments(asr["segments"])
    has_words = any(seg.get("words") for seg in cleaned_segments)
    segments = M.assign_speakers(cleaned_segments, turns)
    utterances = M.build_utterances(segments)
    if not utterances:
        raise PipelineError("没有识别到任何语音内容，请检查音频或语言设置")
    if clean_stats["filtered_total"]:
        project.log(
            f"已过滤 {clean_stats['filtered_total']} 段异常识别结果"
            f"（提示词复读 {clean_stats['dropped_echo']}、重复循环 {clean_stats['dropped_loop']}、"
            f"修剪 {clean_stats['trimmed']}）；原始结果仍在 asr/chunks 中"
        )

    speakers = M.speaker_stats(utterances)
    order = sorted(speakers)
    for i, spk in enumerate(order):
        if not project.speaker_names.get(spk):
            project.speaker_names[spk] = f"说话人 {chr(65 + i) if i < 26 else i + 1}"
    project.save()

    draft = {
        "schema_version": 1,
        "created_at": now_iso(),
        "source": "machine",
        "language": asr.get("language"),
        "audio": project.data.get("audio"),
        "asr": {
            "model": project.settings.get("model"),
            "compute_type": project.settings.get("compute_type"),
            "chunks": asr.get("chunks"),
            "aligned": has_words,
        },
        "diarization": {
            "enabled": bool(diarization),
            "model": (diarization or {}).get("model"),
            "speakers": (diarization or {}).get("speakers", []),
            "turn_count": len((diarization or {}).get("turns", [])),
        },
        "speakers": speakers,
        "speaker_names": dict(project.speaker_names),
        "utterances": utterances,
        "review": {
            "utterances": len(utterances),
            "uncertain_speaker": sum(1 for u in utterances if "speaker_uncertain" in u.get("flags", [])),
            "overlap": sum(1 for u in utterances if "overlap" in u.get("flags", [])),
            "low_confidence": sum(1 for u in utterances if "low_confidence" in u.get("flags", [])),
            "filtered_echo": clean_stats["dropped_echo"],
            "filtered_loop": clean_stats["dropped_loop"],
            "trimmed": clean_stats["trimmed"],
            "filtered_examples": clean_stats["examples"],
        },
    }
    write_json(project.draft_transcript, draft)
    return draft


# ------------------------------------------------------------------ job entry
def run_transcription(job: Job) -> dict[str, Any]:
    project = get_project(job.project_id)
    stored = load_settings()
    settings = {**stored, **{k: v for k, v in project.settings.items() if v not in (None, "")}}
    project.settings.update(settings)
    project.save()

    t_all = time.perf_counter()
    project.touch_stage("transcribe", "running")
    project.log("开始转录：" + json.dumps({k: settings.get(k) for k in ("model", "compute_type", "language", "beam_size", "num_speakers")}, ensure_ascii=False))

    wav = ensure_normalized(project, job)
    info = A.probe(wav)
    if info.duration_sec and not project.data["audio"].get("duration_sec"):
        project.data["audio"]["duration_sec"] = info.duration_sec
        project.save()

    t0 = time.perf_counter()
    asr = _load_asr(project, wav, settings, job)
    asr_sec = round(time.perf_counter() - t0, 1)
    project.record_timing("asr", asr_sec, segments=len(asr["segments"]), language=asr["language"])
    job.set_progress(66.0, "asr", f"语音识别完成：{len(asr['segments'])} 句，用时 {asr_sec}s")

    diarization: dict[str, Any] | None = None
    if settings.get("diarize_enabled"):
        t0 = time.perf_counter()
        try:
            diarization = _diarize(project, wav, settings, job, 66.0, 26.0)
            project.record_timing("diarize", round(time.perf_counter() - t0, 1), speakers=len(diarization.get("speakers", [])))
        except Exception as exc:
            project.log(f"说话人分离失败，逐字稿将使用单一说话人标签：{exc}")
            job.log(f"⚠ 说话人分离失败：{exc}")
            project.touch_stage("diarize", "failed", error=str(exc)[:300])
    else:
        job.log("已跳过说话人分离（设置中关闭）")

    job.set_progress(94.0, "merge", "正在合并时间轴与说话人标签…")
    t0 = time.perf_counter()
    draft = build_draft(project, asr, diarization)
    project.record_timing("merge", round(time.perf_counter() - t0, 1), utterances=len(draft["utterances"]))
    project.touch_stage("transcribe", "done")
    if diarization:
        project.touch_stage("diarize", "done")

    elapsed = round(time.perf_counter() - t_all, 1)
    project.log(f"转录流程完成：{len(draft['utterances'])} 段，总耗时 {elapsed}s")
    job.set_progress(100.0, "done", f"完成：{len(draft['utterances'])} 段逐字稿")
    return {
        "utterances": len(draft["utterances"]),
        "speakers": len(draft.get("speakers") or {}),
        "language": draft.get("language"),
        "elapsed_sec": elapsed,
        "asr_sec": asr_sec,
    }


def run_diarization_only(job: Job) -> dict[str, Any]:
    """Re-run only speaker diarization, reusing the cached ASR output."""
    project = get_project(job.project_id)
    stored = load_settings()
    settings = {**stored, **{k: v for k, v in project.settings.items() if v not in (None, "")}}
    key = _chunk_key(settings)
    manifest = read_json(project.asr_manifest, {}) or {}
    if manifest.get("key") != key:
        raise PipelineError("识别参数已变更，请先重新转录，或把参数改回原值后再重跑说话人分离")
    chunks_dir = project.asr_dir / "chunks"
    segments: list[dict[str, Any]] = []
    for chunk in manifest.get("chunks", []):
        cached = read_json(chunks_dir / f"{int(chunk['index']):03d}.json")
        if not isinstance(cached, dict):
            raise PipelineError("缺少 ASR 分段结果，请先完成语音识别")
        segments.extend(cached.get("segments") or [])
    if not segments:
        raise PipelineError("缺少 ASR 结果，请先完成语音识别")

    project.log("重新运行说话人分离（复用已有 ASR 结果）")
    wav = ensure_normalized(project, job)
    job.set_progress(5.0, "diarize", "正在加载说话人分离模型…")
    t0 = time.perf_counter()
    diarization = _diarize(project, wav, settings, job, 5.0, 80.0)
    project.record_timing("diarize", round(time.perf_counter() - t0, 1), speakers=len(diarization.get("speakers", [])))
    job.set_progress(90.0, "merge", "重新生成逐字稿…")
    asr = {"segments": segments, "language": manifest.get("detected_language"), "chunks": len(manifest.get("chunks", []))}
    draft = build_draft(project, asr, diarization)
    project.touch_stage("diarize", "done")
    job.set_progress(100.0, "done", "说话人分离已更新")
    return {"utterances": len(draft["utterances"]), "speakers": len(diarization.get("speakers", []))}


def run_rebuild_draft(job: Job) -> dict[str, Any]:
    """Re-merge ASR + diarization without touching either (after parameter tweaks)."""
    project = get_project(job.project_id)
    manifest = read_json(project.asr_manifest, {}) or {}
    chunks_dir = project.asr_dir / "chunks"
    segments: list[dict[str, Any]] = []
    for chunk in manifest.get("chunks", []):
        cached = read_json(chunks_dir / f"{int(chunk['index']):03d}.json")
        if isinstance(cached, dict):
            segments.extend(cached.get("segments") or [])
    if not segments:
        raise PipelineError("缺少 ASR 结果，请先完成语音识别")
    diarization = read_json(project.diarization_raw)
    job.set_progress(30.0, "merge", "重新合并逐字稿（并清理提示词复读/重复循环）…")
    t0 = time.perf_counter()
    draft = build_draft(project, {"segments": segments, "language": manifest.get("detected_language")}, diarization)
    project.record_timing("merge", round(time.perf_counter() - t0, 1), utterances=len(draft["utterances"]))
    project.log(f"重建逐字稿：{len(draft['utterances'])} 段（已过滤 {draft['review'].get('filtered_echo', 0) + draft['review'].get('filtered_loop', 0)} 段异常识别）")
    job.set_progress(100.0, "done", f"逐字稿已重建：{len(draft['utterances'])} 段")
    return {
        "utterances": len(draft["utterances"]),
        "filtered": draft["review"].get("filtered_echo", 0) + draft["review"].get("filtered_loop", 0),
    }


def run_align_only(job: Job) -> dict[str, Any]:
    """Add word-level timestamps to the cached ASR output, then rebuild the draft.

    Reuses `asr/chunks/*.json` (no re-transcription) and only runs wav2vec2 forced
    alignment, so speaker-change cuts land on real word boundaries.
    """
    from transcription.asr import ALIGN_MODELS, align_segments, normalize_segments

    project = get_project(job.project_id)
    stored = load_settings()
    settings = {**stored, **{k: v for k, v in project.settings.items() if v not in (None, "")}}
    manifest = read_json(project.asr_manifest, {}) or {}
    plan = manifest.get("chunks") or []
    if not plan:
        raise PipelineError("缺少 ASR 结果，请先完成语音识别")

    language = str(manifest.get("detected_language") or settings.get("language") or "zh")
    if language not in ALIGN_MODELS:
        raise PipelineError(f"语言 {language} 没有配置对齐模型（支持 zh / ja / en / ko）")

    wav = ensure_normalized(project, job)
    chunks_dir = project.asr_dir / "chunks"
    project.log(f"开始词级对齐（复用已有识别结果，语言 {language}）")

    t0 = time.perf_counter()
    aligned_chunks = 0
    segments: list[dict[str, Any]] = []
    for idx, chunk in enumerate(plan):
        if job.cancelled:
            raise JobCancelled("对齐已取消，已完成的分段已保存")
        cache = chunks_dir / f"{int(chunk['index']):03d}.json"
        cached = read_json(cache)
        if not isinstance(cached, dict) or not cached.get("segments"):
            raise PipelineError(f"缺少分段 {idx + 1} 的识别结果，请先完成语音识别")
        segs = cached["segments"]
        offset = float(chunk["start"])
        base_pct = 3 + 87 * idx / max(1, len(plan))
        if any(s.get("words") for s in segs):
            job.log(f"分段 {idx + 1}/{len(plan)} 已含词级时间戳，跳过")
        else:
            tmp_wav = project.audio_dir / "chunks" / f"chunk_{int(chunk['index']):03d}.wav"
            job.set_progress(base_pct, "align", f"正在对齐分段 {idx + 1}/{len(plan)}（{hhmmss(offset)}）…")
            A.slice_wav(wav, offset, float(chunk["end"]), tmp_wav)
            try:
                aligned = align_segments(
                    [
                        {
                            "start": float(s["start"]) - offset,
                            "end": float(s["end"]) - offset,
                            "text": s["text"],
                        }
                        for s in segs
                    ],
                    tmp_wav,
                    language,
                )
                segs = normalize_segments(aligned, offset=offset)
                cached.update({"segments": segs, "aligned": True, "aligned_at": now_iso()})
                write_json(cache, cached)
                aligned_chunks += 1
                job.log(f"分段 {idx + 1}/{len(plan)} 对齐完成：{sum(len(s['words']) for s in segs)} 个词级时间戳")
            finally:
                tmp_wav.unlink(missing_ok=True)
        segments.extend(segs)
        job.set_progress(base_pct + 87 / max(1, len(plan)), "align", f"分段 {idx + 1}/{len(plan)} 已完成")

    align_sec = round(time.perf_counter() - t0, 1)
    project.record_timing("align", align_sec, chunks=aligned_chunks, language=language)

    diarization = read_json(project.diarization_raw)
    job.set_progress(92.0, "merge", "按词级时间轴重新切分说话人…")
    t0 = time.perf_counter()
    draft = build_draft(
        project,
        {"segments": segments, "language": language, "chunks": len(plan)},
        diarization,
    )
    project.record_timing("merge", round(time.perf_counter() - t0, 1), utterances=len(draft["utterances"]))
    project.log(f"词级对齐完成：{len(draft['utterances'])} 段逐字稿（用时 {align_sec}s）")
    job.set_progress(100.0, "done", f"对齐完成：{len(draft['utterances'])} 段逐字稿")
    return {"utterances": len(draft["utterances"]), "aligned_chunks": aligned_chunks, "align_sec": align_sec}


def run_refine_speakers(job: Job) -> dict[str, Any]:
    """Second-pass speaker verification on the existing diarization result.

    Re-embeds every turn and re-clusters them globally, flipping the turns that sit
    confidently on the wrong side (pyannote occasionally swaps two speakers for a
    few seconds). No re-transcription, no re-diarization.
    """
    from transcription import recluster as RC

    project = get_project(job.project_id)
    raw = read_json(project.diarization_raw)
    if not raw or not raw.get("turns"):
        raise PipelineError("还没有说话人分离结果，请先转录或重跑分离")
    turns = raw["turns"]
    project.log(f"开始二次声纹校验：{len(turns)} 个 turn") 

    wav = ensure_normalized(project, job)
    t0 = time.perf_counter()
    refined, stats = RC.refine_turns(
        wav,
        turns,
        progress=lambda pct, msg: job.set_progress(3.0 + 0.85 * pct, "refine", msg),
    )
    RC.unload()
    elapsed = round(time.perf_counter() - t0, 1)

    backup = project.diarization_dir / "raw.before_refine.json"
    write_json(backup, raw)
    raw["turns"] = refined
    raw["refine"] = {**stats, "elapsed_sec": elapsed, "refined_at": now_iso()}
    write_json(project.diarization_raw, raw)
    project.record_timing("refine", elapsed, flipped=stats.get("flipped", 0), embedded=stats.get("embedded", 0))
    if stats.get("flipped"):
        project.log(
            f"二次声纹校验：{stats['flipped']} 个 turn 的归属被纠正（保守跳过 {stats.get('uncertain', 0)} 个），"
            f"原结果已备份到 diarization/raw.before_refine.json"
        )
    else:
        project.log("二次声纹校验：没有发现需要纠正的 turn") 

    job.set_progress(92.0, "merge", "按校验后的归属重新生成逐字稿…")
    segments: list[dict[str, Any]] = []
    manifest = read_json(project.asr_manifest, {}) or {}
    for chunk in manifest.get("chunks", []):
        cached = read_json(project.asr_dir / "chunks" / f"{int(chunk['index']):03d}.json")
        if isinstance(cached, dict):
            segments.extend(cached.get("segments") or [])
    if not segments:
        raise PipelineError("缺少 ASR 结果，请先完成语音识别")
    draft = build_draft(
        project,
        {"segments": segments, "language": manifest.get("detected_language")},
        raw,
    )
    project.record_timing("merge", 0.0, utterances=len(draft["utterances"]))
    job.set_progress(100.0, "done", f"声纹校验完成：{len(draft['utterances'])} 段逐字稿")
    return {
        "utterances": len(draft["utterances"]),
        "flipped": stats.get("flipped", 0),
        "elapsed_sec": elapsed,
    }
