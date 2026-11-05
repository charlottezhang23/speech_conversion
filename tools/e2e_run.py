"""End-to-end transcription run on the synthetic test audio (real WhisperX, CPU).

Usage:  python tools/e2e_run.py [audio_path] [--model medium] [--diarize]
Writes a readable report to logs/e2e.txt and prints a summary.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRIPT_LINES = [
    "你好，谢谢你今天接受采访。先请你简单介绍一下自己。",
    "好的。我今年二十九岁，在杭州做产品设计，家里养了两只猫。",
    "两只猫分别是什么品种？",
    "一只是英国短毛猫，另一只是挪威森林猫，都是三年前开始养的。",
    "平时养猫最让你困扰的事情是什么？",
    "最麻烦的是掉毛。挪威森林猫毛特别长，每天都要吸地，不然沙发上全是毛。",
    "那你会用什么工具来处理？",
    "我用过三款吸尘器，现在用的是一款手持的，但续航只有二十分钟，打扫一半就没电了。",
    "你买吸尘器的时候主要看什么？",
    "我最看重吸力和续航，价格的话两千块以内可以接受，超过三千我就要再想想。",
    "如果有一个新产品能解决续航问题，你会考虑吗？",
    "会考虑，但是我需要先看到真实的测评，不太相信广告。",
    "好的，最后一个问题，你平时会记录养猫的日常吗？",
    "偶尔会拍照发在社交平台上，但不会写很长的文字。",
]


def char_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", default=str(ROOT / "tests" / "fixtures" / "audio" / "zh_interview_2speakers.wav"))
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    from app import audio as A
    from app.jobs import Job
    from app.pipeline import run_transcription
    from app.projects import create_project, get_project
    from app.transcript import load_draft
    from app.util import hhmmss

    audio_path = Path(args.audio)
    info = A.probe(audio_path)
    lines = [f"音频：{audio_path.name}  时长 {hhmmss(info.duration_sec)}  采样率 {info.sample_rate}  声道 {info.channels}"]
    print(lines[-1])

    project = create_project(
        "E2E 合成中文访谈",
        audio_path,
        audio_path.name,
        duration_sec=info.duration_sec,
        size_bytes=info.size_bytes,
        fmt="wav",
    )
    project.settings.update(
        {
            "language": args.language,
            "model": args.model,
            "compute_type": "int8",
            "beam_size": 5,
            "batch_size": args.batch,
            "threads": args.threads,
            "diarize_enabled": bool(args.diarize),
            "chunk_minutes": 10,
            "min_speakers": 2,
            "max_speakers": 2,
        }
    )
    project.save()
    lines.append(f"项目 ID：{project.id}")

    job = Job(kind="transcribe", project_id=project.id)
    job._log_sink = project.log
    t0 = time.perf_counter()
    result = run_transcription(job)
    elapsed = time.perf_counter() - t0
    lines.append(f"流水线结果：{json.dumps(result, ensure_ascii=False)}")
    lines.append(f"总耗时：{elapsed:.1f}s（音频 {info.duration_sec or 0:.1f}s，RTF={elapsed / max(1.0, info.duration_sec or 1):.2f}）")

    draft = load_draft(project)
    lines.append(f"识别出 {len(draft['utterances'])} 段，语言={draft.get('language')}")
    lines.append("")
    lines.append("逐字稿：")
    for u in draft["utterances"]:
        lines.append(f"[{hhmmss(u['start'])} - {hhmmss(u['end'])}] {u['speaker']}: {u['text']}")

    joined = "".join(u["text"] for u in draft["utterances"])
    script = "".join(SCRIPT_LINES)
    sim = char_similarity(joined, script)
    lines.append("")
    lines.append(f"与脚本原文的字符相似度（仅用于验证流水线，不代表真实录音准确率）：{sim:.3f}")

    starts = [u["start"] for u in draft["utterances"]]
    lines.append(f"时间戳单调递增：{starts == sorted(starts)}；首段时间 {starts[0] if starts else None:.2f}s")

    report = ROOT / "logs" / "e2e.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-8:]))
    print(f"\n报告：{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
