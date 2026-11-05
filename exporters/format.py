"""Transcript / summary exporters: TXT, Markdown, SRT, DOCX, JSON."""
from __future__ import annotations

import json
from typing import Any, Sequence

from app.util import hhmmss, srt_time
from summarization.chunking import speaker_label


def _version_label(source: str) -> str:
    return {
        "edited": "人工校对版",
        "draft": "机器初稿（未校对）",
    }.get(source, source)


def _header_lines(meta: dict[str, Any], source: str, names: dict[str, str]) -> list[str]:
    lines = [
        f"项目：{meta.get('name', '未命名访谈')}",
        f"原始文件：{meta.get('original_name', '未知')}",
        f"时长：{hhmmss(meta.get('duration_sec'))}",
        f"版本：{_version_label(source)}",
    ]
    if names:
        lines.append("说话人：" + "；".join(f"{k} = {v}" for k, v in names.items()))
    return lines


def to_txt(
    utterances: Sequence[dict[str, Any]],
    names: dict[str, str],
    meta: dict[str, Any],
    source: str = "draft",
) -> str:
    out = ["=" * 60, "访谈逐字稿", "=" * 60]
    out += _header_lines(meta, source, names)
    out.append("=" * 60)
    out.append("")
    for u in utterances:
        stamp = hhmmss(u.get("start"))
        speaker = speaker_label(u.get("speaker", ""), names)
        out.append(f"[{stamp}] {speaker}：")
        out.append(str(u.get("text", "")).strip())
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_markdown(
    utterances: Sequence[dict[str, Any]],
    names: dict[str, str],
    meta: dict[str, Any],
    source: str = "draft",
) -> str:
    out = [f"# {meta.get('name', '未命名访谈')}", ""]
    for line in _header_lines(meta, source, names):
        out.append(f"- {line}")
    out.append("")
    out.append("---")
    out.append("")
    current_speaker = None
    for u in utterances:
        speaker = speaker_label(u.get("speaker", ""), names)
        stamp = hhmmss(u.get("start"))
        if speaker != current_speaker:
            out.append(f"**{speaker}**")
            current_speaker = speaker
        text = str(u.get("text", "")).strip().replace("\n", " ")
        out.append(f"> [{stamp}] {text}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_srt(
    utterances: Sequence[dict[str, Any]],
    names: dict[str, str],
    *,
    fallback_end: float | None = None,
) -> str:
    blocks: list[str] = []
    total = len(utterances)
    for i, u in enumerate(utterances):
        start = float(u.get("start") or 0.0)
        end = float(u.get("end") or start + 1.0)
        if end <= start:
            end = start + 0.8
        if i == total - 1 and fallback_end and end > fallback_end:
            end = min(end, fallback_end)
        speaker = speaker_label(u.get("speaker", ""), names)
        text = str(u.get("text", "")).strip()
        blocks.append(
            f"{i + 1}\n{srt_time(start)} --> {srt_time(end)}\n{speaker}: {text}\n"
        )
    return "\n".join(blocks)


def to_json_bundle(
    *,
    project_id: str,
    meta: dict[str, Any],
    utterances: Sequence[dict[str, Any]],
    names: dict[str, str],
    source: str,
    summary: dict[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": 1,
        "project_id": project_id,
        "project": meta,
        "transcript": {
            "source": source,
            "speaker_names": names,
            "utterances": list(utterances),
        },
        "summary": summary,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def split_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    """Parse '## heading' sections from the model output (for DOCX layout)."""
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in (markdown or "").splitlines():
        if line.startswith("## "):
            if current_title or any(ls.strip() for ls in current_lines):
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            continue
        else:
            current_lines.append(line)
    if current_title or any(ls.strip() for ls in current_lines):
        sections.append((current_title, current_lines))
    return [(t, "\n".join(ls).strip()) for t, ls in sections if t or any(ls.strip() for ls in ls)]
