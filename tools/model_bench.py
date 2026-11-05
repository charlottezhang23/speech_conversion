"""Benchmark a Whisper model on the fixture audio (real run, writes logs/model_bench.txt)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / "logs" / "model_bench.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def dir_size(path: Path) -> float:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total / 1e9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--audio", default=str(ROOT / "tests" / "fixtures" / "audio" / "zh_interview_2speakers.wav"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    LOG.write_text("", encoding="utf-8")
    usage = shutil.disk_usage(ROOT)
    log(f"磁盘剩余 {usage.free / 1e9:.1f} GB；模型缓存 {dir_size(ROOT / 'models' / 'whisper'):.2f} GB")
    log(f"模型={args.model} 音频={Path(args.audio).name} threads={args.threads}")

    from app import audio as A
    from transcription.asr import AsrEngine, normalize_segments

    wav = Path(args.audio)
    info = A.probe(wav)
    log(f"音频时长 {info.duration_sec:.1f}s")

    t0 = time.perf_counter()
    log("开始加载模型（首次会下载权重，请耐心等待）…")
    engine = AsrEngine(args.model, compute_type="int8", threads=args.threads, language="zh")
    load_sec = time.perf_counter() - t0
    log(f"模型加载完成 {load_sec:.1f}s；缓存 {dir_size(ROOT / 'models' / 'whisper'):.2f} GB")

    t1 = time.perf_counter()
    result = engine.transcribe_chunk(wav, language="zh", batch_size=8)
    run_sec = time.perf_counter() - t1
    segs = normalize_segments(result)
    chars = sum(len(s["text"]) for s in segs)
    rtf = run_sec / max(1.0, info.duration_sec or 1)
    log(f"转写完成 {run_sec:.1f}s，{len(segs)} 段，{chars} 字")
    log(f"RTF={rtf:.2f}（按此外推：1 小时录音约 {3600 * rtf / 60:.0f} 分钟）")
    for s in segs[:3]:
        log(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text'][:60]}")

    engine.unload()
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
