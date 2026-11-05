"""Transcript access: machine draft vs human-corrected version."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.projects import Project, ProjectError
from app.util import now_iso, read_json, write_json
from transcription.merge import speaker_stats


class TranscriptError(RuntimeError):
    pass


def draft_hash(project: Project) -> str:
    if not project.draft_transcript.exists():
        return ""
    h = hashlib.sha256(project.draft_transcript.read_bytes()).hexdigest()
    return h[:32]


def load_draft(project: Project) -> dict[str, Any]:
    data = read_json(project.draft_transcript)
    if not isinstance(data, dict) or not data.get("utterances"):
        raise TranscriptError("尚未生成机器逐字稿")
    return data


def load_edited(project: Project) -> dict[str, Any] | None:
    data = read_json(project.edited_transcript)
    if isinstance(data, dict) and data.get("utterances"):
        return data
    return None


def load_effective(project: Project) -> tuple[dict[str, Any], str]:
    """Return (transcript, source) where source is 'edited' or 'draft'."""
    edited = load_edited(project)
    if edited:
        return edited, "edited"
    return load_draft(project), "draft"


def names_for(project: Project, transcript: dict[str, Any] | None = None) -> dict[str, str]:
    names = dict(project.speaker_names)
    if transcript:
        names.update({k: v for k, v in (transcript.get("speaker_names") or {}).items() if v})
    return names


def _validate_utterances(utterances: Any) -> list[dict[str, Any]]:
    if not isinstance(utterances, list):
        raise TranscriptError("utterances 必须是数组")
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for i, u in enumerate(utterances):
        if not isinstance(u, dict):
            raise TranscriptError(f"第 {i + 1} 条格式错误")
        uid = str(u.get("id") or f"u{i + 1:04d}")
        if uid in seen:
            raise TranscriptError(f"重复的段落 ID：{uid}")
        seen.add(uid)
        try:
            start = max(0.0, float(u.get("start") or 0.0))
            end = max(start, float(u.get("end") or start))
        except (TypeError, ValueError) as exc:
            raise TranscriptError(f"第 {i + 1} 条时间戳无效") from exc
        text = str(u.get("text") or "")
        speaker = str(u.get("speaker") or "SPEAKER_00")
        cleaned.append(
            {
                "id": uid,
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "text": text,
                "machine_text": str(u.get("machine_text") or u.get("text") or ""),
                "edited": bool(u.get("edited")) or str(u.get("machine_text") or text) != text,
                "flags": list(u.get("flags") or []),
                "speaker_confidence": u.get("speaker_confidence", 1.0),
            }
        )
    cleaned.sort(key=lambda x: (x["start"], x["id"]))
    return cleaned


def save_edits(
    project: Project,
    utterances: list[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    draft = load_draft(project)
    machine_by_id = {u["id"]: u for u in draft.get("utterances", [])}
    cleaned = _validate_utterances(utterances)
    for u in cleaned:
        src = machine_by_id.get(u["id"])
        if src is not None:
            u["machine_text"] = src.get("text", u["text"])
            u["edited"] = u["machine_text"] != u["text"] or src.get("speaker") != u["speaker"]
    names = dict(project.speaker_names)
    if speaker_names:
        for k, v in speaker_names.items():
            if k:
                names[str(k)] = str(v)[:40]
        project.speaker_names.clear()
        project.speaker_names.update(names)
        project.save()

    edited = {
        "schema_version": 1,
        "source": "human",
        "updated_at": now_iso(),
        "base_draft_sha256": draft_hash(project),
        "base_draft_created_at": draft.get("created_at"),
        "draft_stale": False,
        "speaker_names": names,
        "speakers": speaker_stats(cleaned),
        "utterances": cleaned,
        "stats": {
            "utterances": len(cleaned),
            "edited_utterances": sum(1 for u in cleaned if u.get("edited")),
            "chars": sum(len(u["text"]) for u in cleaned),
        },
    }
    write_json(project.edited_transcript, edited)
    project.log(f"保存人工校对逐字稿：{len(cleaned)} 段（修改 {edited['stats']['edited_utterances']} 段）")
    return edited


def reset_edits(project: Project) -> None:
    if project.edited_transcript.exists():
        project.edited_transcript.unlink()
        project.log("已放弃人工校对版本，回到机器初稿")


def transcript_payload(project: Project, version: str = "effective") -> dict[str, Any]:
    draft = load_draft(project)
    edited = load_edited(project)
    if version == "draft":
        payload, source = draft, "draft"
    elif version == "edited":
        if not edited:
            raise TranscriptError("没有人工校对版本")
        payload, source = edited, "edited"
    else:
        payload, source = (edited, "edited") if edited else (draft, "draft")

    current_hash = draft_hash(project)
    stale = bool(edited and edited.get("base_draft_sha256") and edited["base_draft_sha256"] != current_hash)
    names = names_for(project, payload)
    return {
        "project_id": project.id,
        "version": source,
        "has_edited": bool(edited),
        "has_draft": True,
        "draft_stale": stale,
        "updated_at": payload.get("updated_at") or payload.get("created_at"),
        "language": payload.get("language"),
        "audio": project.data.get("audio"),
        "asr": payload.get("asr"),
        "diarization": payload.get("diarization"),
        "speakers": payload.get("speakers"),
        "speaker_names": names,
        "utterances": payload.get("utterances", []),
        "review": payload.get("review") or {
            "uncertain_speaker": sum(1 for u in payload.get("utterances", []) if "speaker_uncertain" in u.get("flags", [])),
            "overlap": sum(1 for u in payload.get("utterances", []) if "overlap" in u.get("flags", [])),
            "low_confidence": sum(1 for u in payload.get("utterances", []) if "low_confidence" in u.get("flags", [])),
        },
        "summary_available": project.has_summary(),
    }


def effective_speaker_names(project: Project, transcript: dict[str, Any]) -> dict[str, str]:
    names = dict(transcript.get("speaker_names") or {})
    names.update({k: v for k, v in project.speaker_names.items() if v})
    # ensure every speaker has a label
    for spk in {u.get("speaker") for u in transcript.get("utterances", []) if u.get("speaker")}:
        if not names.get(spk):
            names[spk] = spk
    return names


def summary_source_meta(project: Project, transcript: dict[str, Any], source: str) -> dict[str, Any]:
    """Flat metadata block handed to the prompt builder."""
    audio = project.data.get("audio") or {}
    settings = project.settings or {}
    meta = dict(project.data)
    meta.update(
        {
            "transcript_source": source,
            "duration_sec": audio.get("duration_sec"),
            "original_name": audio.get("original_name"),
            "format": audio.get("format"),
            "language": settings.get("language") or "auto",
            "asr_model": settings.get("model"),
            "speaker_count": len(project.speaker_names or {}),
        }
    )
    return meta


def dump_for_export(project: Project, version: str = "effective") -> tuple[list[dict[str, Any]], dict[str, str], str, dict[str, Any]]:
    payload = transcript_payload(project, version)
    if not payload.get("utterances"):
        raise ProjectError("逐字稿为空")
    return payload["utterances"], payload["speaker_names"], payload["version"], payload


__all__ = [
    "TranscriptError",
    "draft_hash",
    "load_draft",
    "load_edited",
    "load_effective",
    "names_for",
    "save_edits",
    "reset_edits",
    "transcript_payload",
    "effective_speaker_names",
    "summary_source_meta",
    "dump_for_export",
    "json",
]
