"""Measure the real cost of forced alignment (wav2vec2) vs Whisper-only timestamps.

Uses the cached ASR segments of an existing project (no re-transcription), runs
whisperx.align on the same WAV, and reports time / memory / word counts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROJECT = ROOT / "projects" / "20261103-113838-6baa9f"
CHUNK = PROJECT / "asr" / "chunks" / "000.json"
WAV = PROJECT / "audio" / "normalized_16k.wav"
LOG = ROOT / "logs" / "align_bench.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    data = json.loads(CHUNK.read_text(encoding="utf-8"))
    segs = data["segments"]
    lang = data.get("language") or "zh"
    log(f"样本 {WAV.name}，{len(segs)} 段，语言={lang}")
    log(f"Whisper 原生时间戳是否带 words 字段：{any(s.get('words') for s in segs)}")

    from transcription.asr import AsrEngine

    engine = AsrEngine("medium", compute_type="int8", threads=8, language=lang)
    base = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]
    t0 = time.perf_counter()
    try:
        aligned = engine.align(base, WAV, lang)
    except Exception as exc:
        log(f"对齐失败：{type(exc).__name__}: {exc}")
        return 1
    align_sec = time.perf_counter() - t0
    dur = max(float(s["end"]) for s in segs)
    log(f"对齐耗时 {align_sec:.1f}s（音频 {dur:.1f}s，约为 {align_sec / dur:.2f}× 实时）")

    n_words = sum(len(s.get("words") or []) for s in aligned.get("segments") or [])
    log(f"产出词级时间戳 {n_words} 个")
    for s in (aligned.get("segments") or [])[:2]:
        log(f"  段 [{s['start']:.2f}-{s['end']:.2f}] {s['text'][:30]}")
        for w in (s.get("words") or [])[:6]:
            log(f"    word [{w['start']:.2f}-{w['end']:.2f}] {w['word']!r} score={w.get('score')}")

    # how much do the segment boundaries move?
    moves = []
    for before, after in zip(segs, aligned.get("segments") or []):
        moves.append(abs(float(after["start"]) - float(before["start"])))
        moves.append(abs(float(after["end"]) - float(before["end"])))
    if moves:
        log(f"段边界平均位移 {sum(moves) / len(moves):.2f}s，最大 {max(moves):.2f}s")

    engine.unload()
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
