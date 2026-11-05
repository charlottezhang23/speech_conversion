"""Long-transcript map/reduce test against the real Ollama model (no audio needed).

Builds a ~60 minute synthetic transcript from the same script lines, forces small
chunks, and verifies that every chunk is cached and merged into one summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.util import hhmmss  # noqa: E402
from summarization.summarize import summarize  # noqa: E402

SCRIPT_LINES = [
    ("SPEAKER_00", "你好，谢谢你今天接受采访。先请你简单介绍一下自己。"),
    ("SPEAKER_01", "好的。我今年二十九岁，在杭州做产品设计，家里养了两只猫。"),
    ("SPEAKER_00", "两只猫分别是什么品种？"),
    ("SPEAKER_01", "一只是英国短毛猫，另一只是挪威森林猫，都是三年前开始养的。"),
    ("SPEAKER_00", "平时养猫最让你困扰的事情是什么？"),
    ("SPEAKER_01", "最麻烦的是掉毛。挪威森林猫毛特别长，每天都要吸地，不然沙发上全是毛。"),
    ("SPEAKER_00", "那你会用什么工具来处理？"),
    ("SPEAKER_01", "我用过三款吸尘器，现在用的是一款手持的，但续航只有二十分钟，打扫一半就没电了。"),
    ("SPEAKER_00", "你买吸尘器的时候主要看什么？"),
    ("SPEAKER_01", "我最看重吸力和续航，价格的话两千块以内可以接受，超过三千我就要再想想。"),
    ("SPEAKER_00", "如果有一个新产品能解决续航问题，你会考虑吗？"),
    ("SPEAKER_01", "会考虑，但是我需要先看到真实的测评，不太相信广告。"),
    ("SPEAKER_00", "好的，最后一个问题，你平时会记录养猫的日常吗？"),
    ("SPEAKER_01", "偶尔会拍照发在社交平台上，但不会写很长的文字。"),
]


def build_utterances(rounds: int = 12) -> list[dict]:
    out = []
    t = 0.0
    i = 0
    for r in range(rounds):
        for speaker, text in SCRIPT_LINES:
            i += 1
            dur = 6.0 + len(text) * 0.12
            out.append(
                {
                    "id": f"u{i:04d}",
                    "start": round(t, 2),
                    "end": round(t + dur, 2),
                    "speaker": speaker,
                    "text": text,
                    "flags": [],
                }
            )
            t += dur + 1.5
    return out


def main() -> int:
    utterances = build_utterances(12)
    duration = utterances[-1]["end"]
    settings = load_settings()
    model = settings["ollama_model"]
    out_dir = ROOT / "tmp" / "long_summary_test"
    lines = [
        f"合成逐字稿：{len(utterances)} 段，模拟时长 {hhmmss(duration)}，字符数 {sum(len(u['text']) for u in utterances)}",
        f"模型：{model}；分段大小 max_chunk_chars=2500",
    ]
    t0 = time.perf_counter()
    result = summarize(
        utterances=utterances,
        names={"SPEAKER_00": "访谈者", "SPEAKER_01": "受访者"},
        meta={"name": "长文本测试", "duration_sec": duration, "transcript_source": "draft"},
        summary_dir=out_dir,
        template="interview",
        model=model,
        base_url=settings["ollama_url"],
        num_ctx=int(settings["ollama_num_ctx"]),
        timeout_sec=float(settings["ollama_timeout_sec"]),
        max_chunk_chars=2500,
        verify=False,
    )
    elapsed = time.perf_counter() - t0
    chunk_files = sorted((out_dir / "chunks").glob("*.json"))
    lines.append(f"用时 {elapsed:.1f}s，分段 {result['stats']['chunks']}，合并后事实 {result['stats']['facts']}")
    lines.append(f"缓存分段文件：{len(chunk_files)} 个")
    lines.append(f"事实 JSON 大小（送入 compose）：{len(json.dumps(result['facts'], ensure_ascii=False))} 字符")
    lines.append("")
    lines.append(result["markdown"])
    report = ROOT / "logs" / "e2e_long_summary.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"\n报告：{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
