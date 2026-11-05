"""Turn utterances into timestamped evidence text and long-text chunks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.util import hhmmss


def speaker_label(speaker: str, names: dict[str, str] | None) -> str:
    if names and names.get(speaker):
        return names[speaker]
    if speaker and speaker.upper().startswith("SPEAKER_"):
        suffix = speaker.split("_")[-1]
        return f"说话人 {suffix}"
    return speaker or "未知说话人"


def turn_line(utterance: dict[str, Any], names: dict[str, str] | None) -> str:
    return f"[{hhmmss(utterance.get('start'))}] {speaker_label(utterance.get('speaker', ''), names)}: {utterance.get('text', '').strip()}"


def format_evidence(utterances: Iterable[dict[str, Any]], names: dict[str, str] | None = None) -> str:
    return "\n".join(turn_line(u, names) for u in utterances if str(u.get("text", "")).strip())


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    text: str
    utterance_ids: list[str] = field(default_factory=list)
    first_line: str = ""
    last_line: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "utterance_ids": self.utterance_ids,
            "chars": len(self.text),
        }


def chunk_utterances(
    utterances: Sequence[dict[str, Any]],
    names: dict[str, str] | None = None,
    max_chars: int = 4500,
) -> list[Chunk]:
    """Split by time/speaker boundaries so each chunk stays under ~max_chars."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    ids: list[str] = []
    buf_chars = 0
    start: float | None = None
    end: float | None = None

    def flush() -> None:
        nonlocal buf, ids, buf_chars, start, end
        if not buf:
            return
        text = "\n".join(buf)
        chunks.append(
            Chunk(
                index=len(chunks),
                start=start or 0.0,
                end=end or 0.0,
                text=text,
                utterance_ids=ids,
                first_line=buf[0],
                last_line=buf[-1],
            )
        )
        # a one-line overlap keeps context across chunk borders
        buf = []
        ids = []
        buf_chars = 0
        start = None
        end = None

    for u in utterances:
        text = str(u.get("text", "")).strip()
        if not text:
            continue
        line = turn_line(u, names)
        line_len = len(line) + 1
        if buf and buf_chars + line_len > max_chars:
            last_speaker = buf[-1].split("] ", 1)[-1].split(":", 1)[0]
            cur_speaker = speaker_label(u.get("speaker", ""), names)
            # prefer breaking where the speaker changes; otherwise hard break
            if last_speaker == cur_speaker and buf_chars < max_chars * 1.6:
                pass
            else:
                flush()
        if start is None:
            start = float(u.get("start") or 0.0)
        end = float(u.get("end") or u.get("start") or 0.0)
        buf.append(line)
        ids.append(str(u.get("id")))
        buf_chars += line_len
    flush()
    return chunks


def total_chars(utterances: Sequence[dict[str, Any]]) -> int:
    return sum(len(str(u.get("text", ""))) for u in utterances)
